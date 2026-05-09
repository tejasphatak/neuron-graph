"""PPL #1 — Working Memory integration for substrate-LLM.

Current architecture uses ctx_window=4 (last 4 tokens with decay).
Long-range coherence (10-100 tokens back) is invisible to it.

Substrate's WorkingMemory class (brain/working_memory.py) carries
sustained activation across calls — perfect for tracking long-range
context. This module wires it into LLM prediction:

  - At each test position i, accumulate W[seq[j]] across all prior
    tokens j with decay (positional-strength)
  - Prediction = W[ctx_window] + α · WM_long_range
  - WM_long_range captures information from positions beyond ctx_window

This is the substrate's NATIVE way to extend context without making
W bigger.
"""

from __future__ import annotations

import math
from typing import List, Optional

import numpy as np

from .llm import LLMView


def perplexity_with_wm(view: LLMView,
                         sequences: List[List[int]], *,
                         unigram_log_probs: Optional[np.ndarray] = None,
                         alpha_wm: float = 0.3,
                         alpha_uni: float = 0.5,
                         context_window: int = 4,
                         wm_decay: float = 0.85,
                         wm_max_window: int = 64,
                         decay: float = 0.6,
                         softmax_temperature: float = 1.0,
                         prob_floor: float = 1e-8) -> float:
    """PPL with working memory long-range context.

    For each position:
      ctx_score  = sum over last context_window tokens of decay^k · W[ctx]
                   (the existing local context model)
      wm_score   = sum over PRIOR tokens (up to wm_max_window back) of
                   wm_decay^pos · W[token]
                   (long-range substrate-native context)

    Final score = (1-α_wm)·ctx_score + α_wm·wm_score

    Optional unigram backoff: alpha_uni mixes in P_unigram on top.
    """
    decay_powers = np.array([decay ** k for k in range(context_window)],
                              dtype=np.float32)
    floor_logp = math.log(prob_floor)
    use_unigram = unigram_log_probs is not None
    log_loss = 0.0
    n_pred = 0
    V = view.W.shape[1]

    for seq in sequences:
        rows = [view.tok_to_row.get(t, -1) for t in seq]
        L = len(rows)
        for i in range(1, L):
            target = rows[i]
            if target < 0:
                continue

            # Local ctx_score (last 4 with decay)
            local_score = np.zeros(V, dtype=np.float32)
            local_n = 0
            for k in range(context_window):
                j = i - 1 - k
                if j < 0:
                    break
                if rows[j] < 0:
                    continue
                local_score += decay_powers[k] * view.W[rows[j]]
                local_n += 1
            if local_n == 0:
                # Fall back to unigram if available
                if use_unigram:
                    target_logp = max(float(unigram_log_probs[target]),
                                       floor_logp)
                    log_loss += -target_logp
                    n_pred += 1
                continue

            # Long-range WM score (prior tokens beyond ctx_window)
            wm_score = np.zeros(V, dtype=np.float32)
            wm_n = 0
            wm_start = max(0, i - 1 - wm_max_window)
            wm_end = i - 1 - context_window  # everything beyond ctx_window
            for j in range(wm_end, wm_start - 1, -1):
                if j < 0:
                    break
                if rows[j] < 0:
                    continue
                # positional strength: decay from current position
                offset = (i - 1) - j  # how many tokens back
                strength = wm_decay ** offset
                if strength < 1e-3:
                    break
                wm_score += strength * view.W[rows[j]]
                wm_n += 1

            # Mix
            if wm_n > 0:
                combined = (1 - alpha_wm) * local_score + alpha_wm * wm_score
            else:
                combined = local_score

            # Softmax with temperature
            combined = combined / max(1e-6, softmax_temperature)
            combined -= combined.max()
            exp_s = np.exp(combined)
            denom = exp_s.sum()
            if denom <= 0:
                continue
            ctx_logp_target = combined[target] - math.log(denom + 1e-30)

            if use_unigram and alpha_uni > 0:
                log_alpha = math.log(max(1 - alpha_uni, 1e-30))
                log_one_m = math.log(max(alpha_uni, 1e-30))
                mixed = float(np.logaddexp(
                    log_alpha + ctx_logp_target,
                    log_one_m + float(unigram_log_probs[target]),
                ))
            else:
                mixed = ctx_logp_target

            target_logp = max(mixed, floor_logp)
            log_loss += -target_logp
            n_pred += 1

    if n_pred == 0:
        return float('inf')
    return math.exp(log_loss / n_pred)
