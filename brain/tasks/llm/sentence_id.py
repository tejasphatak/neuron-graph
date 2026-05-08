"""Sentence-id binding for substrate-LLM.

Same mechanism that pushed brain/tasks/lm/rl.py from 67% → 89% on the
20-sentence corpus: each training document gets a unique "story-id"
neuron. The story-id activates alongside its tokens during training
and prediction, providing a per-document context anchor.

For TinyStories: each story gets one story-id neuron. RL grows two
classes of edges:
  prompt_token --co_occurs--> story_id  (so prompt → identify which story)
  story_id     --predicts--> next_token (so story-id → drive generation)

At inference, given a prompt, the substrate predicts which story-id
this prompt likely came from, then uses that as additional context.
Adds a soft retrieval / memory mechanism to the LLM.

Data layout reuses the existing dense LLMView. Story-ids are extra
neurons appended after vocab tokens. So vocab effectively becomes
[token_0, ..., token_V-1, story_0, ..., story_M-1] for M stories.

Memory: M × V floats per row (story → token edges) + V × M (token →
story_id) = ~2 × M × V × 4 bytes. At M=10K stories, V=4K: 320 MB.
Affordable for moderate M.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .llm import LLMView
from .tokenizer import WordTokenizer


@dataclass
class StoryIdView:
    """Augments an LLMView with story-id neurons.

    Two extra matrices alongside the main W[V, V]:
      U[V, M] — token → story-id co_occurs (prompt identifies story)
      D[M, V] — story-id → next-token predicts (story drives generation)

    Stored separately so they can be tuned independently. At prediction
    time we run W[ctx] + α·U^T_for_active[ctx] + β·D[predicted_sid].
    """
    n_stories: int
    U: np.ndarray  # [V, n_stories]
    D: np.ndarray  # [n_stories, V]
    # Adam state lazy
    U_m: Optional[np.ndarray] = None
    U_v: Optional[np.ndarray] = None
    D_m: Optional[np.ndarray] = None
    D_v: Optional[np.ndarray] = None
    t: int = 0

    @classmethod
    def build(cls, view: LLMView, n_stories: int) -> 'StoryIdView':
        V = view.W.shape[1]
        return cls(
            n_stories=n_stories,
            U=np.zeros((V, n_stories), dtype=np.float32),
            D=np.zeros((n_stories, V), dtype=np.float32),
        )


def predict_story_id(view: LLMView, sid_view: StoryIdView,
                       prompt_rows: List[int], *,
                       decay: float = 0.6,
                       context_window: int = 4) -> Optional[int]:
    """Given prompt token rows, find argmax story-id via U^T @ active_ctx."""
    if not prompt_rows:
        return None
    # Use last context_window tokens with positional decay
    decay_powers = np.array(
        [decay ** k for k in range(context_window)], dtype=np.float32)
    score = np.zeros(sid_view.n_stories, dtype=np.float32)
    L = len(prompt_rows)
    for k in range(min(context_window, L)):
        idx = L - 1 - k
        if idx < 0:
            break
        score += decay_powers[k] * sid_view.U[prompt_rows[idx]]
    if score.max() <= 0:
        return None
    return int(score.argmax())


def train_with_story_id(view: LLMView, sid_view: StoryIdView,
                          sequences: List[List[int]],
                          tokenizer: WordTokenizer, *,
                          context_window: int = 4,
                          eta: float = 0.05,
                          decay: float = 0.6,
                          weight_clip: Optional[float] = 5.0,
                          rng: Optional[np.random.Generator] = None) -> dict:
    """One epoch of training with story-id binding.

    For each (prompt, target) pair in sequence i:
      1. Score = W[ctx] + sid_view.D[i]   (use known story-id during training)
      2. Argmax → pred
      3. If wrong:
         - W[ctx, target] += w, W[ctx, pred] -= w  (token routing)
         - U[ctx, i] += w  (so prompt eventually identifies story i)
         - D[i, target] += w, D[i, pred] -= w  (story directly drives correct)
    """
    if rng is None:
        rng = np.random.default_rng()

    n = len(sequences)
    assert sid_view.n_stories >= n, (
        f'StoryIdView has {sid_view.n_stories} stories, '
        f'sequences has {n}'
    )

    decay_powers = np.array(
        [decay ** k for k in range(context_window)], dtype=np.float32)
    n_correct = 0
    n_pairs = 0
    delta_W = np.zeros_like(view.W)
    delta_U = np.zeros_like(sid_view.U)
    delta_D = np.zeros_like(sid_view.D)

    order = rng.permutation(n)
    for sid in order:
        seq = sequences[sid]
        rows = [view.tok_to_row.get(t, -1) for t in seq]
        L = len(rows)
        if L < 2:
            continue
        for i in range(L - 1):
            target_row = rows[i + 1]
            if target_row < 0:
                continue
            ctx_rows = []
            ctx_weights = []
            for k in range(context_window):
                j = i - k
                if j < 0:
                    break
                if rows[j] < 0:
                    continue
                ctx_rows.append(rows[j])
                ctx_weights.append(decay_powers[k])
            if not ctx_rows:
                continue

            # Score = W[ctx] + D[sid]
            scores = sid_view.D[sid].copy()
            for r, w in zip(ctx_rows, ctx_weights):
                scores += w * view.W[r]
            pred = int(scores.argmax())
            if scores.max() <= 0:
                pred = -1
            if pred == target_row:
                n_correct += 1
            else:
                for r, w in zip(ctx_rows, ctx_weights):
                    delta_W[r, target_row] += w
                    if pred >= 0:
                        delta_W[r, pred] -= w
                    delta_U[r, sid] += w
                delta_D[sid, target_row] += 1.0
                if pred >= 0:
                    delta_D[sid, pred] -= 1.0
            n_pairs += 1

    # Apply
    view.W += eta * delta_W
    sid_view.U += eta * delta_U
    sid_view.D += eta * delta_D
    if weight_clip is not None:
        np.clip(view.W, -weight_clip, weight_clip, out=view.W)
        np.clip(sid_view.U, -weight_clip, weight_clip, out=sid_view.U)
        np.clip(sid_view.D, -weight_clip, weight_clip, out=sid_view.D)

    return {
        'n_pairs': n_pairs,
        'n_correct': n_correct,
        'next_token_accuracy': n_correct / max(1, n_pairs),
    }


def perplexity_with_story_id(view: LLMView, sid_view: StoryIdView,
                                sequences: List[List[int]], *,
                                use_oracle_sid: bool = False,
                                context_window: int = 4,
                                decay: float = 0.6,
                                softmax_temperature: float = 1.0,
                                prob_floor: float = 1e-8) -> float:
    """PPL with story-id assist. If use_oracle_sid, uses the true
    sequence-index as story-id (training-mode sanity check).
    Otherwise predicts story-id from the prefix at each step."""
    import math
    log_loss = 0.0
    n_pred = 0
    decay_powers = np.array(
        [decay ** k for k in range(context_window)], dtype=np.float32)

    for seq_idx, seq in enumerate(sequences):
        rows = [view.tok_to_row.get(t, -1) for t in seq]
        L = len(rows)
        for i in range(1, L):
            target_row = rows[i]
            if target_row < 0:
                continue
            ctx_rows = []
            ctx_weights = []
            for k in range(context_window):
                j = i - 1 - k
                if j < 0:
                    break
                if rows[j] < 0:
                    continue
                ctx_rows.append(rows[j])
                ctx_weights.append(decay_powers[k])
            if not ctx_rows:
                continue

            # Pick story-id
            if use_oracle_sid and seq_idx < sid_view.n_stories:
                sid = seq_idx
            else:
                sid = predict_story_id(view, sid_view, ctx_rows,
                                          decay=decay,
                                          context_window=context_window)
                if sid is None:
                    sid = 0  # fallback

            scores = sid_view.D[sid].copy()
            for r, w in zip(ctx_rows, ctx_weights):
                scores += w * view.W[r]

            scores = scores / max(1e-6, softmax_temperature)
            scores -= scores.max()
            exp_scores = np.exp(scores)
            denom = exp_scores.sum()
            if denom <= 0:
                continue
            prob = exp_scores[target_row] / denom
            prob = max(prob, prob_floor)
            log_loss += -math.log(prob)
            n_pred += 1

    if n_pred == 0:
        return float('inf')
    return math.exp(log_loss / n_pred)
