"""PPL #B — substrate-native spread()-based prediction.

The substrate's defining primitive is spread() with goal injection +
working memory. The LLM task currently bypasses spread() entirely:
W[ctx] @ ones is matrix multiplication.

This module rebuilds prediction using spread():
  1. Convert dense W → sparse substrate (top-K edges per row)
  2. Seed working memory with context tokens (positional decay)
  3. spread() with token-class neurons as goals
  4. Read out activation pattern over token neurons → predict

Hypothesis: activation flows to semantic neighbors via co-occurs
edges, smoothing predictions naturally. Should reduce PPL on
unseen contexts where W[ctx] is near-zero.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from brain import Brain, WorkingMemory, spread
from .llm import LLMView


PREDICTS = 'predicts'  # token → next-token (single relation)


def view_to_brain(view: LLMView, *,
                    top_k_per_row: int = 20,
                    min_weight: float = 0.01) -> Tuple[Brain, Dict[int, int]]:
    """Convert dense W matrix → sparse substrate Brain.

    For each from-token, keep only the top_k_per_row strongest edges
    (by absolute weight). Skip edges below min_weight to keep substrate
    truly sparse. Returns (brain, row_to_neuron_map).

    Memory: at V=2500, top_k=20 → ~50K edges = ~800 KB substrate
    (vs 25 MB dense W). And spread() actually works on it.
    """
    V = view.W.shape[0]
    brain = Brain()
    brain.relations = [(PREDICTS, 1.0)]
    brain._rebuild_relation_index()

    # One neuron per token-row
    row_to_nid = {}
    for row in range(V):
        nid = brain.add_neuron(lemma=f'tok:{row}', decay=0.5)
        row_to_nid[row] = nid

    # Top-K edges per row, magnitude-thresholded
    for from_row in range(V):
        row_weights = view.W[from_row]
        # Find indices of top-K by absolute value
        if top_k_per_row >= V:
            top_idx = np.where(np.abs(row_weights) >= min_weight)[0]
        else:
            # argpartition for top-k by magnitude
            abs_w = np.abs(row_weights)
            top_idx = np.argpartition(abs_w, -top_k_per_row)[-top_k_per_row:]
            top_idx = top_idx[abs_w[top_idx] >= min_weight]
        from_nid = row_to_nid[from_row]
        for to_row in top_idx:
            w = float(row_weights[to_row])
            if w == 0:
                continue
            brain.add_synapse(from_nid, row_to_nid[int(to_row)],
                                rel_name=PREDICTS, weight=w)

    return brain, row_to_nid


def predict_via_spread(brain: Brain, row_to_nid: Dict[int, int],
                         ctx_rows: List[int], ctx_weights: List[float],
                         V: int, *,
                         max_steps: int = 2,
                         wm_decay: float = 0.6) -> np.ndarray:
    """Run spread() from context-seeded WM, return activation scores
    over all V token neurons. Used for both predict and perplexity.

    Returns float32 array [V] of per-token activations from spread.
    """
    if not ctx_rows:
        return np.zeros(V, dtype=np.float32)

    wm = WorkingMemory(decay=wm_decay, max_size=64, floor=0.001)
    seeds = {}
    for r, w in zip(ctx_rows, ctx_weights):
        nid = row_to_nid.get(r)
        if nid is not None:
            seeds[nid] = float(w)
    if not seeds:
        return np.zeros(V, dtype=np.float32)
    wm.absorb(seeds, gain=1.0)

    state = spread(brain, seeds=[],
                    working_memory=wm,
                    max_steps=max_steps,
                    sparsity=1.0)

    # Project activation back to V-dim score vector
    scores = np.zeros(V, dtype=np.float32)
    nid_to_row = {n: r for r, n in row_to_nid.items()}
    for nid, lvl in state.activation.items():
        row = nid_to_row.get(nid)
        if row is not None:
            scores[row] = float(lvl)
    return scores


def perplexity_with_spread(view: LLMView, brain: Brain,
                              row_to_nid: Dict[int, int],
                              sequences: List[List[int]], *,
                              context_window: int = 4,
                              decay: float = 0.6,
                              max_steps: int = 2,
                              softmax_temperature: float = 1.0,
                              prob_floor: float = 1e-8) -> float:
    """PPL using substrate spread() for scoring instead of W @ ctx."""
    V = view.W.shape[0]
    decay_powers = [decay ** k for k in range(context_window)]
    log_loss = 0.0
    n_pred = 0
    floor_logp = math.log(prob_floor)

    for seq in sequences:
        rows = [view.tok_to_row.get(t, -1) for t in seq]
        for i in range(1, len(rows)):
            target_row = rows[i]
            if target_row < 0:
                continue
            ctx_rows = []
            ctx_w = []
            for k in range(context_window):
                j = i - 1 - k
                if j < 0:
                    break
                if rows[j] < 0:
                    continue
                ctx_rows.append(rows[j])
                ctx_w.append(decay_powers[k])
            if not ctx_rows:
                continue

            scores = predict_via_spread(brain, row_to_nid,
                                          ctx_rows, ctx_w, V,
                                          max_steps=max_steps)
            # Softmax + clip
            scores = scores / max(1e-6, softmax_temperature)
            scores -= scores.max()
            exp_s = np.exp(scores)
            denom = exp_s.sum()
            if denom <= 0:
                continue
            prob = exp_s[target_row] / denom
            target_logp = max(math.log(max(prob, 1e-30)), floor_logp)
            log_loss += -target_logp
            n_pred += 1

    if n_pred == 0:
        return float('inf')
    return math.exp(log_loss / n_pred)
