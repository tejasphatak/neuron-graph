"""MNIST / image classification on the substrate.

Architecture:
  - INPUT neurons: grid_size × grid_size × n_levels  (via ImageEncoder)
  - OUTPUT neurons: 10 digit-class neurons
  - INPUT --activates--> OUTPUT  edges grown by supervised reward

Encoder (default 16×16 grid × 4 levels = 1024 input neurons) gives:
  - SCALE INVARIANCE: any image size → same fixed substrate dimensionality
  - INTENSITY GRADIENT: 4 levels capture mid-tones, not just on/off
  - SPARSITY: only one level fires per cell (one-hot per cell)

Inference:
  encode image → seed input neurons → spread() → argmax over digit neurons

Training (supervised reward):
  for each (image, label):
    predict via spread + argmax
    if correct: strengthen (active_input → label) edges
    if wrong:   weaken (active_input → predicted) AND
                strengthen (active_input → correct_label)

The substrate is essentially learning a sparse linear classifier in graph
form. The point: SAME primitives as TTT and LM, no architectural change.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from brain import Brain, WorkingMemory, spread
from brain.neuron import SYNAPSE_DTYPE

from .encoder import ImageEncoder, ImageEncoderVocab


ACTIVATES = 'activates'  # input → digit class

# Backward-compat alias
MnistVocab = ImageEncoderVocab


# ─── Brain construction ────────────────────────────────────────────────────

def build_mnist_brain(image_size: int = 28,
                       *, encoder: Optional[ImageEncoder] = None
                       ) -> Tuple[Brain, ImageEncoderVocab, ImageEncoder]:
    """Empty brain wired with input + digit class neurons.

    Args:
      image_size: kept for backward-compat; ignored when encoder is given
      encoder: ImageEncoder; defaults to 16×16 × 4 levels (scale-invariant)

    Returns (brain, vocab, encoder). The encoder is returned because
    inference needs the same one used at training (same grid_size etc.).
    """
    if encoder is None:
        encoder = ImageEncoder(grid_size=16, n_levels=4)

    brain = Brain()
    brain.relations = [(ACTIVATES, 1.0)]
    brain._rebuild_relation_index()

    vocab = encoder.build_vocab(brain)
    return brain, vocab, encoder


# ─── Encoding (delegated) ──────────────────────────────────────────────────

def encode_image(image: np.ndarray, vocab: ImageEncoderVocab,
                  *, encoder: Optional[ImageEncoder] = None,
                  threshold: float = 0.5) -> Dict[int, float]:
    """image → seed dict via the encoder.

    `threshold` is unused when an encoder is provided (kept for backward
    compat with the old binary-pixel API).
    """
    if encoder is None:
        # Legacy default — assumes vocab was built by an encoder we don't
        # have a handle to. Caller should pass the encoder explicitly.
        encoder = ImageEncoder(grid_size=16, n_levels=4)
    return encoder.encode(image, vocab)


# ─── Inference ─────────────────────────────────────────────────────────────

def predict(brain: Brain, vocab: ImageEncoderVocab, image: np.ndarray, *,
             encoder: Optional[ImageEncoder] = None,
             threshold: float = 0.5,
             wm_decay: float = 0.5,
             max_steps: int = 2) -> Optional[int]:
    """image → predicted digit (argmax over digit class activations).

    Returns None if no signal (blank image or untrained brain)."""
    seeds = encode_image(image, vocab, encoder=encoder, threshold=threshold)
    if not seeds:
        return None

    wm = WorkingMemory(decay=wm_decay, max_size=4096, floor=0.001)
    wm.absorb(seeds, gain=1.0)

    state = spread(brain, seeds=[],
                    working_memory=wm,
                    max_steps=max_steps,
                    sparsity=1.0)

    best_digit = None
    best_lvl = -1.0
    for nid, lvl in state.activation.items():
        if nid in vocab.id_to_digit and lvl > best_lvl:
            best_lvl = lvl
            best_digit = vocab.id_to_digit[nid]
    return best_digit


# ─── Edge mutation helper ──────────────────────────────────────────────────

def _strengthen(brain: Brain, from_id: int, to_id: int,
                  rel_name: str, delta: float, *, cap: float = 1.0) -> None:
    """Hebbian update: add or strengthen edge from→to. Capped at `cap`.
    Negative delta weakens (clamped at 0)."""
    rel_id = brain.relation_id[rel_name]
    edges = brain.synapses_of(from_id)
    for k, syn in enumerate(edges):
        if int(syn['to_id']) == to_id and int(syn['relation']) == rel_id:
            base = int(brain.nodes[from_id]['syn_offset']) // SYNAPSE_DTYPE.itemsize
            old = float(brain.synapses[base + k]['weight'])
            new = max(0.0, min(cap, old + delta))
            brain.synapses[base + k]['weight'] = new
            return
    if delta > 0:
        brain.add_synapse(from_id, to_id, rel_name=rel_name, weight=min(cap, delta))


# ─── Training ──────────────────────────────────────────────────────────────

def train_step(brain: Brain, vocab: ImageEncoderVocab,
                image: np.ndarray, label: int, *,
                encoder: Optional[ImageEncoder] = None,
                eta: float = 0.10,
                wrong_eta: float = 0.05,
                threshold: float = 0.5) -> Tuple[bool, Optional[int]]:
    """One supervised step: predict, compare, update edges."""
    pred = predict(brain, vocab, image, encoder=encoder, threshold=threshold)
    correct = (pred == label)

    seeds = encode_image(image, vocab, encoder=encoder, threshold=threshold)
    if not seeds:
        return False, pred

    label_nid = vocab.digit_to_id[label]
    pred_nid = vocab.digit_to_id[pred] if pred is not None else None

    if correct:
        for input_nid in seeds:
            _strengthen(brain, input_nid, label_nid, ACTIVATES, +eta)
    else:
        if pred_nid is not None and pred_nid != label_nid:
            for input_nid in seeds:
                _strengthen(brain, input_nid, pred_nid, ACTIVATES, -wrong_eta)
        for input_nid in seeds:
            _strengthen(brain, input_nid, label_nid, ACTIVATES, +eta)

    return correct, pred


def train_epoch(brain: Brain, vocab: ImageEncoderVocab,
                  X: np.ndarray, y: np.ndarray, *,
                  encoder: Optional[ImageEncoder] = None,
                  eta: float = 0.10,
                  wrong_eta: float = 0.05,
                  threshold: float = 0.5,
                  shuffle: bool = True,
                  rng: Optional[np.random.Generator] = None) -> float:
    """One pass over the training set. Returns running accuracy."""
    if rng is None:
        rng = np.random.default_rng()
    n = len(X)
    order = rng.permutation(n) if shuffle else np.arange(n)

    n_correct = 0
    for idx in order:
        correct, _ = train_step(brain, vocab, X[idx], int(y[idx]),
                                 encoder=encoder,
                                 eta=eta, wrong_eta=wrong_eta,
                                 threshold=threshold)
        n_correct += int(correct)
    return n_correct / max(1, n)


# ─── Evaluation ────────────────────────────────────────────────────────────

def evaluate(brain: Brain, vocab: ImageEncoderVocab,
              X: np.ndarray, y: np.ndarray, *,
              encoder: Optional[ImageEncoder] = None,
              threshold: float = 0.5) -> Dict[str, float]:
    """Run inference on (X, y); return accuracy + confusion stats."""
    n_correct = 0
    n_blank = 0
    confusion = np.zeros((10, 10), dtype=np.int64)
    for img, label in zip(X, y):
        pred = predict(brain, vocab, img, encoder=encoder, threshold=threshold)
        if pred is None:
            n_blank += 1
            continue
        confusion[int(label), pred] += 1
        if pred == int(label):
            n_correct += 1
    n = len(X)
    return {
        'accuracy': n_correct / max(1, n),
        'n_correct': n_correct,
        'n_total': n,
        'n_blank': n_blank,
        'confusion': confusion,
    }
