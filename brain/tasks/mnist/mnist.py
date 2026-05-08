"""MNIST classification on the substrate.

Architecture:
  - 784 PIXEL neurons (28×28 grid), each represents one image position
  - 10 DIGIT-CLASS neurons (digit_0 through digit_9)
  - PIXEL --activates--> DIGIT  edges grown by supervised reward signal

Encoding (binary thresholding):
  pixel intensity >= threshold → that pixel neuron is seeded at strength 1.0
  pixel intensity < threshold  → not seeded (off)

Inference:
  encode image → seed pixel neurons → spread() → argmax over digit neurons

Training (supervised reward):
  for each (image, label):
    predict
    if correct: strengthen (active_pixels → label) edges
    if wrong:   weaken those edges, strengthen (active_pixels → correct_label)

The substrate is essentially learning a sparse linear classifier in graph
form. The point is: SAME primitives as TTT and LM, no architectural change.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from brain import Brain, WorkingMemory, spread
from brain.neuron import SYNAPSE_DTYPE


ACTIVATES = 'activates'  # pixel → digit class


# ─── Vocabulary ────────────────────────────────────────────────────────────

@dataclass
class MnistVocab:
    """Maps pixel positions and digit classes to neuron ids."""
    pixel_to_id: Dict[Tuple[int, int], int] = field(default_factory=dict)
    id_to_pixel: Dict[int, Tuple[int, int]] = field(default_factory=dict)
    digit_to_id: Dict[int, int] = field(default_factory=dict)
    id_to_digit: Dict[int, int] = field(default_factory=dict)


# ─── Brain construction ────────────────────────────────────────────────────

def build_mnist_brain(image_size: int = 28) -> Tuple[Brain, MnistVocab]:
    """Empty brain wired with pixel + digit neurons.

    No edges between them yet — those grow from supervised training.
    """
    brain = Brain()
    brain.relations = [(ACTIVATES, 1.0)]
    brain._rebuild_relation_index()

    vocab = MnistVocab()
    # Pixel neurons (28×28 = 784)
    for r in range(image_size):
        for c in range(image_size):
            nid = brain.add_neuron(lemma=f'px:{r},{c}', decay=0.5)
            vocab.pixel_to_id[(r, c)] = nid
            vocab.id_to_pixel[nid] = (r, c)
    # Digit class neurons (0-9)
    for d in range(10):
        nid = brain.add_neuron(lemma=f'digit:{d}', decay=0.7)
        vocab.digit_to_id[d] = nid
        vocab.id_to_digit[nid] = d
    return brain, vocab


# ─── Encoding ──────────────────────────────────────────────────────────────

def encode_image(image: np.ndarray, vocab: MnistVocab,
                  *, threshold: float = 0.5) -> Dict[int, float]:
    """28×28 image (float [0,1]) → {pixel_neuron_id: 1.0} for active pixels.

    Binary thresholding for the simplest possible test. Could be extended
    to multi-bin or graded activation later.
    """
    seeds: Dict[int, float] = {}
    rows, cols = image.shape
    on = image >= threshold
    on_rows, on_cols = on.nonzero()
    for r, c in zip(on_rows.tolist(), on_cols.tolist()):
        seeds[vocab.pixel_to_id[(r, c)]] = 1.0
    return seeds


# ─── Inference ─────────────────────────────────────────────────────────────

def predict(brain: Brain, vocab: MnistVocab, image: np.ndarray, *,
             threshold: float = 0.5,
             wm_decay: float = 0.5,
             max_steps: int = 2) -> Optional[int]:
    """image → predicted digit (argmax over digit class activations).

    Returns None if no signal (blank image or untrained brain).
    """
    seeds = encode_image(image, vocab, threshold=threshold)
    if not seeds:
        return None

    wm = WorkingMemory(decay=wm_decay, max_size=2048, floor=0.001)
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


# ─── Edge mutation helpers ─────────────────────────────────────────────────

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

def train_step(brain: Brain, vocab: MnistVocab,
                image: np.ndarray, label: int, *,
                eta: float = 0.10,
                wrong_eta: float = 0.05,
                threshold: float = 0.5) -> Tuple[bool, Optional[int]]:
    """One supervised step:
      - predict the digit
      - if correct: strengthen (active_pixels → label) edges
      - if wrong: weaken (active_pixels → predicted) AND strengthen
        (active_pixels → correct_label)

    Returns (was_correct, predicted_digit).
    """
    pred = predict(brain, vocab, image, threshold=threshold)
    correct = (pred == label)

    seeds = encode_image(image, vocab, threshold=threshold)
    if not seeds:
        return False, pred

    label_nid = vocab.digit_to_id[label]
    pred_nid = vocab.digit_to_id[pred] if pred is not None else None

    if correct:
        # Reinforce active pixels → label
        for pixel_nid in seeds:
            _strengthen(brain, pixel_nid, label_nid, ACTIVATES, +eta)
    else:
        # Weaken active pixels → wrong predicted class
        if pred_nid is not None and pred_nid != label_nid:
            for pixel_nid in seeds:
                _strengthen(brain, pixel_nid, pred_nid, ACTIVATES, -wrong_eta)
        # Strengthen active pixels → correct label
        for pixel_nid in seeds:
            _strengthen(brain, pixel_nid, label_nid, ACTIVATES, +eta)

    return correct, pred


def train_epoch(brain: Brain, vocab: MnistVocab,
                  X: np.ndarray, y: np.ndarray, *,
                  eta: float = 0.10,
                  wrong_eta: float = 0.05,
                  threshold: float = 0.5,
                  shuffle: bool = True,
                  rng: Optional[np.random.Generator] = None) -> float:
    """One pass over the training set. Returns batch accuracy.
    Note: accuracy is measured on PRE-update predictions (running tally),
    which is more meaningful than post-hoc — it shows how the substrate
    is performing as it learns."""
    if rng is None:
        rng = np.random.default_rng()
    n = len(X)
    order = rng.permutation(n) if shuffle else np.arange(n)

    n_correct = 0
    for idx in order:
        correct, _ = train_step(brain, vocab, X[idx], int(y[idx]),
                                 eta=eta, wrong_eta=wrong_eta,
                                 threshold=threshold)
        n_correct += int(correct)
    return n_correct / max(1, n)


# ─── Evaluation ────────────────────────────────────────────────────────────

def evaluate(brain: Brain, vocab: MnistVocab,
              X: np.ndarray, y: np.ndarray, *,
              threshold: float = 0.5) -> Dict[str, float]:
    """Run inference on (X, y) and return accuracy + confusion stats."""
    n_correct = 0
    n_blank = 0
    confusion = np.zeros((10, 10), dtype=np.int64)
    for img, label in zip(X, y):
        pred = predict(brain, vocab, img, threshold=threshold)
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
