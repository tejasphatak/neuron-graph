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
    edges reflect the learned weights."""
    W: np.ndarray                       # [n_inputs, 10] float32
    input_id_to_idx: Dict[int, int]     # neuron_id → row in W
    input_idx_to_id: List[int]          # row → neuron_id
    digit_ids: List[int]                # column index → digit neuron_id


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
    """Write W back to substrate edges. Call once at end of training
    so the substrate is up to date for spread()-based queries."""
    rel_act = brain.relation_id[ACTIVATES]
    for row, input_nid in enumerate(view.input_idx_to_id):
        # Existing edges from this input
        edges = brain.synapses_of(input_nid)
        existing = {}
        base = int(brain.nodes[input_nid]['syn_offset']) // SYNAPSE_DTYPE.itemsize
        for k, syn in enumerate(edges):
            if int(syn['relation']) == rel_act:
                existing[int(syn['to_id'])] = base + k

        for col, digit_nid in enumerate(view.digit_ids):
            w = float(view.W[row, col])
            if w <= 0:
                # Skip zero/negative — leave existing edge if any (decayed)
                if digit_nid in existing:
                    brain.synapses[existing[digit_nid]]['weight'] = 0.0
                continue
            if digit_nid in existing:
                brain.synapses[existing[digit_nid]]['weight'] = min(1.0, w)
            else:
                brain.add_synapse(input_nid, digit_nid,
                                   rel_name=ACTIVATES,
                                   weight=min(1.0, w))


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
                      perceptron: bool = True) -> float:
    """Vectorized supervised training over one epoch.

    `perceptron=True` (default): classical perceptron update — only
    update on errors. W[active, correct] += eta; W[active, wrong] -= eta.
    Stable on large corpora. ~92% on MNIST is achievable.

    `perceptron=False`: always update. Saturates fast on large corpora;
    OK only with low eta and a generous cap.

    `cap=None`: no weight clamp (lets the substrate's edge weights
    encode actual differences). Set to e.g. 5.0 if you want soft bound.
    """
    if rng is None:
        rng = np.random.default_rng()
    n = len(X)
    order = rng.permutation(n) if shuffle else np.arange(n)
    n_correct = 0

    for start in range(0, n, batch_size):
        idx = order[start:start + batch_size]
        batch_X = X[idx]
        batch_y = y[idx].astype(np.int64)

        indicators = batch_encode(batch_X, vocab, encoder, view)
        scores = indicators @ view.W                 # [B, 10]
        preds = scores.argmax(axis=1)
        has_signal = scores.sum(axis=1) > 0
        n_correct += int(((preds == batch_y) & has_signal).sum())

        for b in range(len(idx)):
            label = int(batch_y[b])
            pred = int(preds[b])
            active_rows = indicators[b].astype(bool)
            if not active_rows.any():
                continue
            if perceptron:
                # Only update on errors: classical perceptron rule
                if pred != label:
                    view.W[active_rows, label] += eta
                    if has_signal[b]:
                        view.W[active_rows, pred] -= eta
            else:
                # Always-update Hebbian
                view.W[active_rows, label] += eta
                if has_signal[b] and pred != label:
                    view.W[active_rows, pred] -= eta * 0.5

    if cap is not None:
        np.clip(view.W, -cap, cap, out=view.W)
    return n_correct / max(1, n)


# ─── Fast evaluation ───────────────────────────────────────────────────────

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
