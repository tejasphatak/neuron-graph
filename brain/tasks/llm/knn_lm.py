"""PPL #F — kNN over substrate activation patterns.

The substrate is its own embedding model: for each context, the
score vector W[ctx] @ ones is already a V-dim continuous representation.
Inspired by Khandelwal et al 2020 "Generalization through Memorization:
Nearest Neighbor Language Models" (kNN-LM).

Approach:
  1. Train W as usual
  2. Sample N training (ctx, target) pairs into a datastore
  3. Each datastore entry: (score_vector_V_dim, target_token_id)
  4. At test, compute test score_vector
  5. Find K nearest by cosine similarity
  6. Predict = empirical distribution over their targets, weighted by sim
  7. Mix with #A unigram backoff for unseen contexts

Compute: 10K datastore × V scoring at test = 10K × 2500 = 25M float
ops per test prediction. Slow but tractable for small test sets.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np

from .llm import LLMView


def build_knn_datastore(view: LLMView,
                          sequences: List[List[int]], *,
                          n_samples: int = 10000,
                          context_window: int = 4,
                          decay: float = 0.6,
                          rng: Optional[np.random.Generator] = None,
                          ) -> Tuple[np.ndarray, np.ndarray]:
    """Sample n_samples (context, target) pairs from training sequences.

    For each sample: compute the score vector (sum over context of
    decay-weighted W rows). Store score_vector + target_row.

    Returns (datastore_X [N, V], datastore_y [N]).
    """
    if rng is None:
        rng = np.random.default_rng()
    decay_powers = np.array([decay ** k for k in range(context_window)],
                              dtype=np.float32)

    # Collect all eligible (ctx, target) positions across sequences
    positions: List[Tuple[int, int]] = []  # (seq_idx, token_position)
    for s_idx, seq in enumerate(sequences):
        for i in range(min(len(seq), 1000) - 1):  # cap per-seq for diversity
            positions.append((s_idx, i))
    if len(positions) > n_samples:
        sel = rng.choice(len(positions), size=n_samples, replace=False)
        positions = [positions[k] for k in sel]
    n = len(positions)

    V = view.W.shape[1]
    X = np.zeros((n, V), dtype=np.float32)
    y = np.zeros(n, dtype=np.int64)

    for k, (s_idx, i) in enumerate(positions):
        seq = sequences[s_idx]
        target_row = view.tok_to_row.get(seq[i + 1], -1)
        if target_row < 0:
            continue
        ctx_rows: List[int] = []
        ctx_w: List[float] = []
        for back in range(context_window):
            j = i - back
            if j < 0:
                break
            r = view.tok_to_row.get(seq[j])
            if r is None:
                continue
            ctx_rows.append(r)
            ctx_w.append(float(decay_powers[back]))
        if not ctx_rows:
            continue
        score = np.zeros(V, dtype=np.float32)
        for r, w in zip(ctx_rows, ctx_w):
            score += w * view.W[r]
        # Normalize for cosine similarity
        norm = np.linalg.norm(score)
        if norm > 0:
            score = score / norm
        X[k] = score
        y[k] = target_row

    # Drop empty rows
    keep = (X.sum(axis=1) != 0) | (y != 0)
    return X[keep], y[keep]


def perplexity_with_knn(view: LLMView,
                          test_sequences: List[List[int]],
                          datastore_X: np.ndarray,
                          datastore_y: np.ndarray,
                          unigram_log_probs: np.ndarray, *,
                          k_neighbors: int = 20,
                          alpha_knn: float = 0.5,
                          alpha_uni: float = 0.5,
                          context_window: int = 4,
                          decay: float = 0.6,
                          temperature: float = 5.0,
                          prob_floor: float = 1e-8) -> float:
    """PPL using kNN over datastore + unigram backoff.

      P(tok|ctx) ∝ alpha_knn · P_knn(tok|ctx) +
                   (1-alpha_knn) · [alpha_uni · P_unigram(tok) +
                                     (1-alpha_uni) · P_W(tok|ctx)]
    """
    decay_powers = np.array([decay ** k for k in range(context_window)],
                              dtype=np.float32)
    floor_logp = math.log(prob_floor)
    uni_logp = unigram_log_probs.astype(np.float32)
    V = view.W.shape[1]
    log_loss = 0.0
    n_pred = 0

    for seq in test_sequences:
        rows = [view.tok_to_row.get(t, -1) for t in seq]
        for i in range(1, len(rows)):
            target = rows[i]
            if target < 0:
                continue
            ctx_rows: List[int] = []
            ctx_w: List[float] = []
            for back in range(context_window):
                j = i - 1 - back
                if j < 0:
                    break
                if rows[j] < 0:
                    continue
                ctx_rows.append(rows[j])
                ctx_w.append(float(decay_powers[back]))
            if not ctx_rows:
                continue

            score = np.zeros(V, dtype=np.float32)
            for r, w in zip(ctx_rows, ctx_w):
                score += w * view.W[r]
            score_norm = np.linalg.norm(score)
            if score_norm == 0:
                # No signal — use unigram only
                target_logp = max(float(uni_logp[target]), floor_logp)
                log_loss += -target_logp
                n_pred += 1
                continue
            score_unit = score / score_norm

            # Cosine similarity to all datastore entries
            sims = datastore_X @ score_unit  # [N]
            # Top-K
            if len(sims) > k_neighbors:
                top_idx = np.argpartition(sims, -k_neighbors)[-k_neighbors:]
            else:
                top_idx = np.arange(len(sims))
            top_sims = sims[top_idx]
            top_targets = datastore_y[top_idx]

            # Build P_knn(tok | ctx) ∝ sum_k exp(sim_k / T) * δ(tok = y_k)
            # Softmax over similarities then bin into target distribution
            top_sims = top_sims / max(1e-6, temperature)
            top_sims -= top_sims.max()
            sim_weights = np.exp(top_sims)
            sim_weights /= sim_weights.sum() + 1e-30
            knn_probs = np.zeros(V, dtype=np.float32)
            for tgt, w in zip(top_targets, sim_weights):
                knn_probs[tgt] += w

            # P_W(tok | ctx) — softmax over score
            score -= score.max()
            exp_s = np.exp(score)
            ctx_probs = exp_s / (exp_s.sum() + 1e-30)

            # P_unigram(tok) — exp(unigram_log_probs)
            uni_probs = np.exp(uni_logp)

            # Mix
            log_knn = math.log(max(alpha_knn, 1e-30))
            log_other = math.log(max(1 - alpha_knn, 1e-30))
            log_alpha_uni = math.log(max(alpha_uni, 1e-30))
            log_one_m_uni = math.log(max(1 - alpha_uni, 1e-30))

            p_knn_target = max(float(knn_probs[target]), 1e-30)
            p_uni_target = max(float(uni_probs[target]), 1e-30)
            p_ctx_target = max(float(ctx_probs[target]), 1e-30)

            log_p_knn = math.log(p_knn_target)
            log_p_uni = math.log(p_uni_target)
            log_p_ctx = math.log(p_ctx_target)

            log_secondary = log_other + np.logaddexp(
                log_alpha_uni + log_p_uni,
                log_one_m_uni + log_p_ctx,
            )
            mixed = float(np.logaddexp(log_knn + log_p_knn, log_secondary))
            target_logp = max(mixed, floor_logp)
            log_loss += -target_logp
            n_pred += 1

    if n_pred == 0:
        return float('inf')
    return math.exp(log_loss / n_pred)
