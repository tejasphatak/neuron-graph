"""PPL #D — Two-stage prediction: P(POS_t+1 | ctx) × P(token | POS_t+1, ctx).

Predicts grammatical class (POS) first — low entropy, ~12 classes via
NLTK universal tagset. Then conditions token prediction on the predicted
class. The conditional has way less sparsity than direct token prediction.

Two W matrices:
  W_ctx_to_pos[V, P]    — context tokens → next-POS class
  W_ctx_pos_to_tok[V, P, V]  — too big; instead use:
  W_pos_to_tok[P, V]    — POS class → token (separate from context)
  Final score: W_ctx[ctx_tokens] @ unigram-prior + W_pos_to_tok[pred_pos]

Implementation strategy:
  1. Pre-tag all training tokens with NLTK
  2. Train W_ctx_to_pos: ctx → next-POS via perceptron + negsample
  3. Train W_pos_to_tok: pos → token frequency conditional on POS
  4. At inference: predict POS, then mix POS-conditional with W

Universal tagset has 12 POS:
  ADJ, ADP, ADV, CONJ, DET, NOUN, NUM, PRT, PRON, VERB, X, .
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .llm import LLMView
from .tokenizer import WordTokenizer


# Universal POS tags (NLTK universal tagset)
POS_TAGS = ['<UNK_POS>', 'ADJ', 'ADP', 'ADV', 'CONJ', 'DET',
            'NOUN', 'NUM', 'PRT', 'PRON', 'VERB', 'X', '.']
POS_TO_IDX = {tag: i for i, tag in enumerate(POS_TAGS)}
N_POS = len(POS_TAGS)


def pos_tag_token_sequences(token_sequences: List[List[int]],
                              tokenizer: WordTokenizer) -> List[List[int]]:
    """Convert each (token_id) sequence to (pos_idx) sequence using NLTK.

    Decode each sequence to words, run NLTK pos_tag, map to universal
    tagset, return aligned pos-idx list.
    """
    import nltk
    out = []
    for seq in token_sequences:
        words = []
        for tid in seq:
            tok = tokenizer.id_to_token.get(tid, tokenizer.UNK)
            # Skip special tokens, keep their POS slot as <UNK_POS>
            words.append(tok if tok not in (tokenizer.PAD, tokenizer.BOS, tokenizer.EOS) else '_')
        if not words:
            out.append([])
            continue
        tagged = nltk.pos_tag(words, tagset='universal')
        pos_seq = [POS_TO_IDX.get(t[1], 0) for t in tagged]
        out.append(pos_seq)
    return out


@dataclass
class POSStagedView:
    """Two-stage substrate:
      W_ctx_to_pos[V, N_POS]   — context token row → next-POS distribution
      W_pos_to_tok[N_POS, V]   — POS row → token-frequency given POS
    """
    V: int
    W_ctx_to_pos: np.ndarray
    W_pos_to_tok: np.ndarray   # log-prob of each token given POS

    @classmethod
    def build(cls, view: LLMView) -> 'POSStagedView':
        V = view.W.shape[0]
        return cls(
            V=V,
            W_ctx_to_pos=np.zeros((V, N_POS), dtype=np.float32),
            W_pos_to_tok=np.full((N_POS, V), -10.0, dtype=np.float32),
        )


def fit_pos_to_token(staged: POSStagedView,
                       token_sequences: List[List[int]],
                       pos_sequences: List[List[int]],
                       view: LLMView) -> None:
    """Build empirical P(token | POS) from training data."""
    counts = np.zeros((N_POS, staged.V), dtype=np.float64)
    for tseq, pseq in zip(token_sequences, pos_sequences):
        for tid, pid in zip(tseq, pseq):
            row = view.tok_to_row.get(tid)
            if row is not None and 0 <= pid < N_POS:
                counts[pid, row] += 1
    # Laplace-smooth and convert to log-probs
    for p in range(N_POS):
        total = counts[p].sum() + staged.V
        staged.W_pos_to_tok[p] = (np.log(counts[p] + 1) -
                                    np.log(total)).astype(np.float32)


def train_ctx_to_pos(staged: POSStagedView,
                       token_sequences: List[List[int]],
                       pos_sequences: List[List[int]],
                       view: LLMView, *,
                       context_window: int = 4,
                       eta: float = 0.05,
                       decay: float = 0.6,
                       weight_clip: float = 5.0,
                       epochs: int = 10,
                       rng: Optional[np.random.Generator] = None) -> None:
    """Train W_ctx_to_pos via perceptron: predict next-POS from
    context-window tokens."""
    if rng is None:
        rng = np.random.default_rng()
    decay_powers = [decay ** k for k in range(context_window)]

    for epoch in range(epochs):
        order = rng.permutation(len(token_sequences))
        for seq_idx in order:
            tseq = token_sequences[seq_idx]
            pseq = pos_sequences[seq_idx]
            L = min(len(tseq), len(pseq))
            for i in range(L - 1):
                target_pos = pseq[i + 1]
                if target_pos < 0 or target_pos >= N_POS:
                    continue
                ctx_rows = []
                ctx_w = []
                for k in range(context_window):
                    j = i - k
                    if j < 0:
                        break
                    row = view.tok_to_row.get(tseq[j])
                    if row is None:
                        continue
                    ctx_rows.append(row)
                    ctx_w.append(decay_powers[k])
                if not ctx_rows:
                    continue
                scores = np.zeros(N_POS, dtype=np.float32)
                for r, w in zip(ctx_rows, ctx_w):
                    scores += w * staged.W_ctx_to_pos[r]
                pred = int(scores.argmax())
                if pred != target_pos or scores.max() <= 0:
                    for r, w in zip(ctx_rows, ctx_w):
                        staged.W_ctx_to_pos[r, target_pos] += eta * w
                        if scores.max() > 0:
                            staged.W_ctx_to_pos[r, pred] -= eta * w
        np.clip(staged.W_ctx_to_pos, -weight_clip, weight_clip,
                 out=staged.W_ctx_to_pos)


def perplexity_pos_staged(view: LLMView, staged: POSStagedView,
                            token_sequences: List[List[int]], *,
                            unigram_log_probs: np.ndarray,
                            alpha_token: float = 0.5,
                            alpha_pos: float = 0.7,
                            context_window: int = 4,
                            decay: float = 0.6,
                            prob_floor: float = 1e-8) -> float:
    """PPL via two-stage:
       P(token | ctx) ∝ alpha_token · P_ctx_W(token) +
                         (1-alpha_token) · [
                            alpha_pos · sum_pos P(pos|ctx) · P(token|pos) +
                            (1-alpha_pos) · P_unigram(token)
                         ]
    Soft mixture: substrate context + POS-conditional + unigram backoff.
    """
    decay_powers = [decay ** k for k in range(context_window)]
    floor_logp = math.log(prob_floor)
    log_loss = 0.0
    n_pred = 0

    for seq in token_sequences:
        rows = [view.tok_to_row.get(t, -1) for t in seq]
        for i in range(1, len(rows)):
            target = rows[i]
            if target < 0:
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

            # Stage 1: P(pos | ctx) — softmax over POS
            pos_scores = np.zeros(N_POS, dtype=np.float32)
            for r, w in zip(ctx_rows, ctx_w):
                pos_scores += w * staged.W_ctx_to_pos[r]
            pos_scores -= pos_scores.max()
            pos_exp = np.exp(pos_scores)
            pos_probs = pos_exp / (pos_exp.sum() + 1e-30)

            # Stage 2: P(token | pos) — averaged across POS distribution
            tok_log_from_pos = -np.inf
            log_pos_probs = np.log(pos_probs + 1e-30)
            log_pos_tok = staged.W_pos_to_tok[:, target]
            tok_log_from_pos = np.logaddexp.reduce(log_pos_probs + log_pos_tok)

            # P(token) from W_ctx (the original substrate-LLM)
            scores = np.zeros(view.W.shape[1], dtype=np.float32)
            for r, w in zip(ctx_rows, ctx_w):
                scores += w * view.W[r]
            scores -= scores.max()
            ctx_logp_target = scores[target] - math.log(np.exp(scores).sum() + 1e-30)

            # Mix: alpha_token · ctx + (1-alpha_token) · [alpha_pos · pos + (1-alpha_pos) · uni]
            # In log-space:
            log_ctx = math.log(max(alpha_token, 1e-30)) + ctx_logp_target
            log_pos_branch = (math.log(max(alpha_pos, 1e-30)) + tok_log_from_pos)
            log_uni_branch = (math.log(max(1 - alpha_pos, 1e-30)) +
                              float(unigram_log_probs[target]))
            log_secondary = math.log(max(1 - alpha_token, 1e-30)) + np.logaddexp(
                log_pos_branch, log_uni_branch
            )
            mixed = float(np.logaddexp(log_ctx, log_secondary))
            target_logp = max(mixed, floor_logp)
            log_loss += -target_logp
            n_pred += 1

    if n_pred == 0:
        return float('inf')
    return math.exp(log_loss / n_pred)
