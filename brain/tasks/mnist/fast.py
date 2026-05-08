"""Fast forward+backward path for MNIST-style topologies.

The substrate's general spread() handles bidirectional, multi-relation,
recursive activation — overkill for MNIST's input-pool → output-pool
single-step topology.

This module specializes the computation for that topology:
  - Build a dense W[n_inputs, n_outputs] numpy matrix in sync with
    the substrate's ACTIVATES edges (substrate stays source of truth)
  - Predict via batch matmul: indicators[B, n_inputs] @ W → scores[B, 10]
  - Update via vectorized scatter-add

Substrate purity preserved: W is just a faster *layout* of the same
edge weights. The substrate's spread() still works as before; this is
an opt-in fast path for tasks that fit the feed-forward shape.

Speedup vs general spread()-based train_step: 50-100× empirically.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from brain import Brain
from brain.neuron import SYNAPSE_DTYPE
from .encoder import ImageEncoder, ImageEncoderVocab


ACTIVATES = 'activates'


# ─── Dense view sync ───────────────────────────────────────────────────────

@dataclass
class DenseView:
    """Dense W[n_inputs, n_outputs] matrix kept in sync with substrate
    ACTIVATES edges. After training, sync back to the substrate so
    edges reflect the learned weights.

    Optional Adam-style optimizer state per-edge:
      m  = first-moment (momentum) of update deltas
      v  = second-moment (squared-update) accumulator
    Lets us apply Adam-style smoothing to substrate edge updates —
    not "Adam over backprop" (no gradients), but "Adam over Hebbian/
    perceptron deltas." Same convergence-stabilization tricks.
    """
    W: np.ndarray                       # [n_inputs, 10] float32
    input_id_to_idx: Dict[int, int]     # neuron_id → row in W
    input_idx_to_id: List[int]          # row → neuron_id
    digit_ids: List[int]                # column index → digit neuron_id
    # Adam state (lazy-allocated when first used)
    m: Optional[np.ndarray] = None
    v: Optional[np.ndarray] = None
    t: int = 0                           # step counter for bias correction


def build_dense_view(brain: Brain,
                       vocab: ImageEncoderVocab,
                       encoder: ImageEncoder) -> DenseView:
    """Allocate W and a stable input-id ordering. Reads any existing
    ACTIVATES edges into W."""
    # Stable ordering: cell_to_id keys give us the input neurons
    input_ids = sorted(vocab.cell_to_id.values())
    input_id_to_idx = {nid: i for i, nid in enumerate(input_ids)}
    digit_ids = [vocab.digit_to_id[d] for d in range(10)]

    n_inputs = len(input_ids)
    W = np.zeros((n_inputs, 10), dtype=np.float32)

    # Read existing ACTIVATES edges into W
    rel_act = brain.relation_id[ACTIVATES]
    digit_id_to_col = {dnid: d for d, dnid in enumerate(digit_ids)}
    for nid, row in input_id_to_idx.items():
        edges = brain.synapses_of(nid)
        for syn in edges:
            if int(syn['relation']) != rel_act:
                continue
            tid = int(syn['to_id'])
            col = digit_id_to_col.get(tid)
            if col is not None:
                W[row, col] = float(syn['weight'])

    return DenseView(
        W=W,
        input_id_to_idx=input_id_to_idx,
        input_idx_to_id=input_ids,
        digit_ids=digit_ids,
    )


def sync_dense_to_brain(brain: Brain, view: DenseView) -> None:
    """Write W back to substrate edges WITHOUT clamping or skipping
    negatives. Substrate's synapse.weight is float32 — negative weights
    are valid (they mean inhibitory: 'this active input votes AGAINST
    this output class'). Skipping/clamping them would silently corrupt
    the learned model — verify_substrate_learning would diverge from
    fast_predict.

    Only edges with weight EXACTLY 0 are skipped to keep substrate
    sparse — those carry no signal anyway.
    """
    rel_act = brain.relation_id[ACTIVATES]
    for row, input_nid in enumerate(view.input_idx_to_id):
        edges = brain.synapses_of(input_nid)
        existing = {}
        base = int(brain.nodes[input_nid]['syn_offset']) // SYNAPSE_DTYPE.itemsize
        for k, syn in enumerate(edges):
            if int(syn['relation']) == rel_act:
                existing[int(syn['to_id'])] = base + k

        for col, digit_nid in enumerate(view.digit_ids):
            w = float(view.W[row, col])
            if w == 0.0:
                if digit_nid in existing:
                    brain.synapses[existing[digit_nid]]['weight'] = 0.0
                continue
            if digit_nid in existing:
                brain.synapses[existing[digit_nid]]['weight'] = w
            else:
                brain.add_synapse(input_nid, digit_nid,
                                   rel_name=ACTIVATES,
                                   weight=w)


# ─── Fast batched encode ───────────────────────────────────────────────────

def batch_encode(images: np.ndarray, vocab: ImageEncoderVocab,
                  encoder: ImageEncoder,
                  view: DenseView) -> np.ndarray:
    """Batch encode → indicator matrix [B, n_inputs] (binary float32)."""
    B = len(images)
    n_inputs = len(view.input_idx_to_id)
    indicators = np.zeros((B, n_inputs), dtype=np.float32)
    for i, img in enumerate(images):
        seeds = encoder.encode(img, vocab)
        for nid in seeds:
            row = view.input_id_to_idx.get(nid)
            if row is not None:
                indicators[i, row] = 1.0
    return indicators


# ─── Fast predict ──────────────────────────────────────────────────────────

def fast_predict(view: DenseView, indicators: np.ndarray) -> np.ndarray:
    """indicators [B, n_inputs] → predictions [B] via matmul over W.

    Returns -1 for blank-image rows (no active inputs)."""
    scores = indicators @ view.W       # [B, 10]
    # Argmax; -1 if no signal at all (all zeros)
    preds = scores.argmax(axis=1)
    has_signal = scores.sum(axis=1) > 0
    return np.where(has_signal, preds, -1)


# ─── Fast train epoch (vectorized across examples) ─────────────────────────

def fast_train_epoch(view: DenseView,
                      X: np.ndarray, y: np.ndarray, *,
                      vocab: ImageEncoderVocab,
                      encoder: ImageEncoder,
                      eta: float = 0.01,
                      batch_size: int = 64,
                      shuffle: bool = True,
                      rng: Optional[np.random.Generator] = None,
                      cap: Optional[float] = None,
                      perceptron: bool = True,
                      optimizer: str = 'sgd',
                      beta1: float = 0.9,
                      beta2: float = 0.999,
                      eps: float = 1e-8) -> float:
    """Vectorized supervised training over one epoch.

    `perceptron=True` (default): only update on errors.
    `optimizer='sgd'`: apply delta directly (default).
    `optimizer='adam'`: per-edge momentum + adaptive learning rate.
       Same tricks as Adam over backprop, but applied to substrate
       Hebbian/perceptron deltas (no gradients involved).
    """
    if rng is None:
        rng = np.random.default_rng()
    n = len(X)
    order = rng.permutation(n) if shuffle else np.arange(n)
    n_correct = 0

    # Lazy-allocate Adam state
    if optimizer == 'adam':
        if view.m is None:
            view.m = np.zeros_like(view.W)
        if view.v is None:
            view.v = np.zeros_like(view.W)

    for start in range(0, n, batch_size):
        idx = order[start:start + batch_size]
        batch_X = X[idx]
        batch_y = y[idx].astype(np.int64)

        indicators = batch_encode(batch_X, vocab, encoder, view)
        scores = indicators @ view.W                 # [B, 10]
        preds = scores.argmax(axis=1)
        has_signal = scores.sum(axis=1) > 0
        n_correct += int(((preds == batch_y) & has_signal).sum())

        # Accumulate per-batch delta (then apply via optimizer)
        delta = np.zeros_like(view.W)
        for b in range(len(idx)):
            label = int(batch_y[b])
            pred = int(preds[b])
            active_rows = indicators[b].astype(bool)
            if not active_rows.any():
                continue
            if perceptron:
                if pred != label:
                    delta[active_rows, label] += 1.0
                    if has_signal[b]:
                        delta[active_rows, pred] -= 1.0
            else:
                delta[active_rows, label] += 1.0
                if has_signal[b] and pred != label:
                    delta[active_rows, pred] -= 0.5

        if optimizer == 'adam':
            view.t += 1
            view.m = beta1 * view.m + (1 - beta1) * delta
            view.v = beta2 * view.v + (1 - beta2) * (delta * delta)
            # Bias correction
            m_hat = view.m / (1 - beta1 ** view.t)
            v_hat = view.v / (1 - beta2 ** view.t)
            view.W += eta * m_hat / (np.sqrt(v_hat) + eps)
        else:  # sgd
            view.W += eta * delta

    if cap is not None:
        np.clip(view.W, -cap, cap, out=view.W)
    return n_correct / max(1, n)


# ─── Fast evaluation ───────────────────────────────────────────────────────

def verify_substrate_learning(brain: Brain, vocab: ImageEncoderVocab,
                                encoder: ImageEncoder,
                                view: DenseView,
                                X: np.ndarray, y: np.ndarray, *,
                                n_samples: int = 100) -> Dict[str, float]:
    """Verify the SUBSTRATE itself (via spread()) gives the same
    predictions as the fast path. Proves learning is captured in the
    substrate's edges, not just the dense view's W matrix.

    Procedure:
      1. sync dense W → substrate edges
      2. run spread()-based predict on N samples
      3. compare to fast_predict on same samples
      4. report match rate + spread-based accuracy

    Match rate ≈ 1.0 means substrate's edges encode the same decisions
    as the dense view — the substrate IS learning.
    """
    from .mnist import predict as substrate_predict

    sync_dense_to_brain(brain, view)

    n = min(n_samples, len(X))
    indicators = batch_encode(X[:n], vocab, encoder, view)
    fast_preds = fast_predict(view, indicators)

    substrate_preds = []
    for i in range(n):
        p = substrate_predict(brain, vocab, X[i], encoder=encoder)
        substrate_preds.append(p if p is not None else -1)
    substrate_preds = np.array(substrate_preds)

    matches = int((fast_preds == substrate_preds).sum())
    fast_correct = int(((fast_preds == y[:n]) & (fast_preds >= 0)).sum())
    sub_correct = int(((substrate_preds == y[:n]) & (substrate_preds >= 0)).sum())

    return {
        'n_samples': n,
        'fast_predict_matches_substrate': matches / n,
        'fast_accuracy': fast_correct / n,
        'substrate_accuracy': sub_correct / n,
    }


def fast_evaluate(view: DenseView,
                   X: np.ndarray, y: np.ndarray, *,
                   vocab: ImageEncoderVocab,
                   encoder: ImageEncoder,
                   batch_size: int = 256) -> Dict[str, float]:
    """Batched inference accuracy + confusion matrix."""
    n = len(X)
    n_correct = 0
    n_blank = 0
    confusion = np.zeros((10, 10), dtype=np.int64)

    for start in range(0, n, batch_size):
        batch_X = X[start:start + batch_size]
        batch_y = y[start:start + batch_size].astype(np.int64)
        indicators = batch_encode(batch_X, vocab, encoder, view)
        preds = fast_predict(view, indicators)
        for true_d, pred_d in zip(batch_y.tolist(), preds.tolist()):
            if pred_d < 0:
                n_blank += 1
                continue
            confusion[true_d, pred_d] += 1
            if pred_d == true_d:
                n_correct += 1

    return {
        'accuracy': n_correct / max(1, n),
        'n_correct': n_correct,
        'n_total': n,
        'n_blank': n_blank,
        'confusion': confusion,
    }
