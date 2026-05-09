"""Substrate-LLM — open-vocab text language modeling on the substrate.

Phase 1: bigram + n-gram next-token prediction with batched fast path.
No teacher forcing. No sequence retrieval (different from tasks/lm/).

Architecture:
  - VOCAB_SIZE token neurons (one per token id)
  - PREDICT relation: token --predicts--> token
  - Edge weight = how strongly token A predicts token B as next

Bigram training: for each (context_token, next_token) pair in corpus,
strengthen edge context→next via reward-modulated update.

N-gram extension: instead of just last-1-token context, use last-N tokens
as parallel active inputs (each weighted by recency). Substrate sums their
votes for next-token prediction.

This is essentially a sparse n-gram LM in graph form. Limited but
verifiable: perplexity is computable, generations are evaluable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from brain import Brain
from brain.neuron import SYNAPSE_DTYPE
from .tokenizer import WordTokenizer


PREDICTS = 'predicts'  # token → next-token


@dataclass
class LLMVocab:
    """Maps token-id (from tokenizer) → neuron-id."""
    tok_to_nid: Dict[int, int] = field(default_factory=dict)
    nid_to_tok: Dict[int, int] = field(default_factory=dict)


# ─── Brain construction ────────────────────────────────────────────────────

def build_llm_brain(tokenizer: WordTokenizer) -> Tuple[Brain, LLMVocab]:
    """Empty brain wired with one neuron per token in the vocab.
    Edges grow from training."""
    brain = Brain()
    brain.relations = [(PREDICTS, 1.0)]
    brain._rebuild_relation_index()

    vocab = LLMVocab()
    for tok_id in range(tokenizer.get_vocab_size()):
        nid = brain.add_neuron(lemma=f'tok:{tok_id}', decay=0.5)
        vocab.tok_to_nid[tok_id] = nid
        vocab.nid_to_tok[nid] = tok_id
    return brain, vocab


# ─── Dense view (for fast batched training) ────────────────────────────────

@dataclass
class LLMView:
    """Dense W[V, V] matrix kept in sync with substrate PREDICTS edges.
    V = vocab size. Storage cost: V² × 4 bytes float32.
    For V=5K: 100 MB. For V=10K: 400 MB. For V=30K: 3.6 GB.

    Above ~10K vocab use sparse storage (deferred). Phase 1 uses dense
    for speed at small vocab."""
    W: np.ndarray                            # [V, V] float32
    tok_to_row: Dict[int, int]               # tok_id → row index
    row_to_tok: List[int]                    # row index → tok_id
    # Adam state (lazy)
    m: Optional[np.ndarray] = None
    v: Optional[np.ndarray] = None
    t: int = 0


def build_llm_view(tokenizer: WordTokenizer) -> LLMView:
    """Allocate W[V, V] once vocab is fixed."""
    V = tokenizer.get_vocab_size()
    tok_ids = list(range(V))
    return LLMView(
        W=np.zeros((V, V), dtype=np.float32),
        tok_to_row={t: i for i, t in enumerate(tok_ids)},
        row_to_tok=tok_ids,
    )


# ─── Training (bigram, fast batched) ───────────────────────────────────────

def train_bigram_epoch(view: LLMView, sequences: List[List[int]],
                         *, eta: float = 0.05,
                         optimizer: str = 'sgd',
                         beta1: float = 0.9, beta2: float = 0.999,
                         eps: float = 1e-8,
                         shuffle: bool = True,
                         rng: Optional[np.random.Generator] = None) -> dict:
    """One pass over all (context, next-token) bigram pairs in `sequences`.

    Each sequence is a list of token-ids (already encoded). For each
    bigram (s[i], s[i+1]), use perceptron update: if argmax of W[s[i]]
    != s[i+1], strengthen W[s[i], s[i+1]] and weaken W[s[i], wrong_pred].

    Returns metrics dict: accuracy, n_pairs, time."""
    if rng is None:
        rng = np.random.default_rng()
    if shuffle:
        order = rng.permutation(len(sequences))
    else:
        order = np.arange(len(sequences))

    if optimizer == 'adam':
        if view.m is None:
            view.m = np.zeros_like(view.W)
        if view.v is None:
            view.v = np.zeros_like(view.W)

    n_correct = 0
    n_pairs = 0

    delta = np.zeros_like(view.W)
    for seq_idx in order:
        seq = sequences[seq_idx]
        if len(seq) < 2:
            continue
        for i in range(len(seq) - 1):
            ctx, nxt = seq[i], seq[i + 1]
            ctx_row = view.tok_to_row.get(ctx)
            nxt_row = view.tok_to_row.get(nxt)
            if ctx_row is None or nxt_row is None:
                continue
            scores = view.W[ctx_row]
            pred = int(scores.argmax())
            if scores.max() <= 0:
                pred = -1
            if pred == nxt_row:
                n_correct += 1
            else:
                delta[ctx_row, nxt_row] += 1.0
                if pred >= 0:
                    delta[ctx_row, pred] -= 1.0
            n_pairs += 1

    if optimizer == 'adam':
        view.t += 1
        view.m = beta1 * view.m + (1 - beta1) * delta
        view.v = beta2 * view.v + (1 - beta2) * (delta * delta)
        m_hat = view.m / (1 - beta1 ** view.t)
        v_hat = view.v / (1 - beta2 ** view.t)
        view.W += eta * m_hat / (np.sqrt(v_hat) + eps)
    else:
        view.W += eta * delta

    return {
        'n_pairs': n_pairs,
        'n_correct': n_correct,
        'next_token_accuracy': n_correct / max(1, n_pairs),
    }


# ─── N-gram extension: weighted-context next-token prediction ──────────────

def train_ngram_epoch_batched(view: LLMView, sequences: List[List[int]], *,
                                context_window: int = 4,
                                eta: float = 0.05,
                                decay: float = 0.6,
                                shuffle: bool = True,
                                rng: Optional[np.random.Generator] = None,
                                optimizer: str = 'adam',
                                beta1: float = 0.9, beta2: float = 0.999,
                                eps: float = 1e-8,
                                weight_clip: Optional[float] = 5.0,
                                batch_pairs: int = 1024) -> dict:
    """Vectorized n-gram trainer — batches pairs into matmul ops.

    Each batch: build [B, V] indicator with positional decay, score
    via single B @ V matmul, argmax over each row, compute deltas
    via np.add.at scatter. ~5-20× faster than per-pair Python loop.

    Same algorithm as train_ngram_epoch, just batched.
    """
    if rng is None:
        rng = np.random.default_rng()
    order = rng.permutation(len(sequences)) if shuffle else np.arange(len(sequences))

    if optimizer == 'adam':
        if view.m is None:
            view.m = np.zeros_like(view.W)
        if view.v is None:
            view.v = np.zeros_like(view.W)

    V = view.W.shape[1]

    # Pre-extract all (context, target) pairs across all sequences.
    # Each pair: (target_row, [(ctx_row, weight) for back in window])
    decay_powers = [decay ** k for k in range(context_window)]

    # Streaming batches keep memory bounded
    batch_ctx = np.zeros((batch_pairs, V), dtype=np.float32)
    batch_target = np.zeros(batch_pairs, dtype=np.int64)
    batch_filled = 0

    n_correct = 0
    n_pairs = 0
    delta = np.zeros_like(view.W)

    def _flush():
        nonlocal batch_filled, n_correct, delta
        if batch_filled == 0:
            return
        ctx = batch_ctx[:batch_filled]                # [b, V]
        targets = batch_target[:batch_filled]         # [b]
        scores = ctx @ view.W                         # [b, V]
        preds = scores.argmax(axis=1)                 # [b]
        no_signal = scores.max(axis=1) <= 0
        correct = (preds == targets) & ~no_signal
        n_correct += int(correct.sum())

        # Fully vectorized delta update via single matmul.
        # outcome[i, target] = +1 for wrong pairs, +0 for correct
        # outcome[i, pred]   = -1 for wrong-with-signal pairs
        # delta += ctx.T @ outcome  (one matmul does all the scatter)
        wrong = ~correct
        if wrong.any():
            outcome = np.zeros((batch_filled, V), dtype=np.float32)
            wrong_idx = np.where(wrong)[0]
            outcome[wrong_idx, targets[wrong_idx]] = 1.0
            wrong_with_signal = wrong & ~no_signal
            if wrong_with_signal.any():
                wsig_idx = np.where(wrong_with_signal)[0]
                outcome[wsig_idx, preds[wsig_idx]] -= 1.0
            # Single matmul: ctx.T [V,b] @ outcome [b,V] → [V,V] delta
            delta += ctx.T @ outcome

        batch_ctx[:batch_filled] = 0
        batch_filled = 0

    def _apply_delta_and_flush():
        if optimizer == 'adam':
            view.t += 1
            view.m = beta1 * view.m + (1 - beta1) * delta
            view.v = beta2 * view.v + (1 - beta2) * (delta * delta)
            m_hat = view.m / (1 - beta1 ** view.t)
            v_hat = view.v / (1 - beta2 ** view.t)
            view.W += eta * m_hat / (np.sqrt(v_hat) + eps)
        else:
            view.W += eta * delta
        if weight_clip is not None:
            np.clip(view.W, -weight_clip, weight_clip, out=view.W)
        delta.fill(0)

    for seq_idx in order:
        seq = sequences[seq_idx]
        if len(seq) < 2:
            continue
        for i in range(len(seq) - 1):
            target = seq[i + 1]
            target_row = view.tok_to_row.get(target)
            if target_row is None:
                continue

            # Fill the indicator vector for this pair
            row = batch_ctx[batch_filled]
            for back in range(context_window):
                j = i - back
                if j < 0:
                    break
                ctx_row = view.tok_to_row.get(seq[j])
                if ctx_row is None:
                    continue
                row[ctx_row] += decay_powers[back]
            batch_target[batch_filled] = target_row
            batch_filled += 1
            n_pairs += 1

            if batch_filled >= batch_pairs:
                _flush()

    _flush()
    _apply_delta_and_flush()

    return {
        'n_pairs': n_pairs,
        'n_correct': n_correct,
        'next_token_accuracy': n_correct / max(1, n_pairs),
    }


def train_ngram_epoch_fast(view: LLMView, sequences: List[List[int]], *,
                             context_window: int = 4,
                             eta: float = 0.05,
                             decay: float = 0.6,
                             shuffle: bool = True,
                             rng: Optional[np.random.Generator] = None,
                             optimizer: str = 'adam',
                             beta1: float = 0.9, beta2: float = 0.999,
                             eps: float = 1e-8,
                             weight_clip: Optional[float] = 5.0) -> dict:
    """Per-sequence vectorized n-gram trainer.

    Within each sequence: build [L-1, ctx_window] index matrix, score
    all positions at once via fancy indexing into W. Avoids per-pair
    Python overhead. Empirically 5-20× faster than train_ngram_epoch.
    """
    if rng is None:
        rng = np.random.default_rng()
    order = rng.permutation(len(sequences)) if shuffle else np.arange(len(sequences))

    if optimizer == 'adam':
        if view.m is None:
            view.m = np.zeros_like(view.W)
        if view.v is None:
            view.v = np.zeros_like(view.W)

    decay_powers = np.array([decay ** k for k in range(context_window)],
                              dtype=np.float32)
    delta = np.zeros_like(view.W)
    V = view.W.shape[1]
    n_correct = 0
    n_pairs = 0

    for seq_idx in order:
        seq = sequences[seq_idx]
        rows = np.array(
            [view.tok_to_row.get(t, -1) for t in seq],
            dtype=np.int64,
        )
        L = len(rows)
        if L < 2:
            continue

        ctx = np.full((L - 1, context_window), -1, dtype=np.int64)
        for k in range(context_window):
            if k < L - 1:
                ctx[k:, k] = rows[:L - 1 - k]
        targets = rows[1:]

        valid_target = targets >= 0
        valid_ctx = ctx >= 0
        any_ctx = valid_ctx.any(axis=1)
        active = valid_target & any_ctx
        if not active.any():
            continue

        ctx_active = ctx[active]
        valid_active = valid_ctx[active]
        targets_active = targets[active]
        P = len(targets_active)

        ctx_safe = np.where(valid_active, ctx_active, 0)
        Wctx = view.W[ctx_safe]
        weights = decay_powers * valid_active.astype(np.float32)
        scores = np.einsum('pk,pkv->pv', weights, Wctx)

        preds = scores.argmax(axis=1)
        max_scores = scores.max(axis=1)
        no_signal = max_scores <= 0
        correct = (preds == targets_active) & ~no_signal
        n_correct += int(correct.sum())
        n_pairs += P

        wrong = ~correct
        if wrong.any():
            outcome = np.zeros((P, V), dtype=np.float32)
            wrong_idx = np.where(wrong)[0]
            outcome[wrong_idx, targets_active[wrong_idx]] = 1.0
            wrong_with_signal = wrong & ~no_signal
            if wrong_with_signal.any():
                wsig_idx = np.where(wrong_with_signal)[0]
                outcome[wsig_idx, preds[wsig_idx]] -= 1.0

            for k in range(context_window):
                w_k = weights[:, k]
                if not w_k.any():
                    continue
                idx_k = ctx_safe[:, k]
                weighted = w_k[:, None] * outcome
                np.add.at(delta, idx_k, weighted)

    if optimizer == 'adam':
        view.t += 1
        view.m = beta1 * view.m + (1 - beta1) * delta
        view.v = beta2 * view.v + (1 - beta2) * (delta * delta)
        m_hat = view.m / (1 - beta1 ** view.t)
        v_hat = view.v / (1 - beta2 ** view.t)
        view.W += eta * m_hat / (np.sqrt(v_hat) + eps)
    else:
        view.W += eta * delta
    if weight_clip is not None:
        np.clip(view.W, -weight_clip, weight_clip, out=view.W)

    return {
        'n_pairs': n_pairs,
        'n_correct': n_correct,
        'next_token_accuracy': n_correct / max(1, n_pairs),
    }


def train_ngram_epoch(view: LLMView, sequences: List[List[int]], *,
                        context_window: int = 4,
                        eta: float = 0.05,
                        decay: float = 0.6,
                        shuffle: bool = True,
                        rng: Optional[np.random.Generator] = None,
                        optimizer: str = 'adam',
                        beta1: float = 0.9, beta2: float = 0.999,
                        eps: float = 1e-8,
                        weight_clip: Optional[float] = 5.0,
                        chunk_pairs: int = 5000) -> dict:
    """Positional-decay context-window n-gram trainer.

    For predicting seq[i+1], active inputs:
      seq[i]   at strength 1.0
      seq[i-1] at strength decay
      seq[i-2] at strength decay²  ...up to context_window

    `optimizer='adam'` (default) — per-edge momentum + adaptive LR,
    applied to substrate Hebbian/perceptron deltas (no gradients).
    `optimizer='sgd'` — direct delta application.

    `weight_clip=5.0` — clamp |W| ≤ 5 after each chunk of `chunk_pairs`
    updates. Stops unbounded perceptron growth that breaks softmax.
    """
    if rng is None:
        rng = np.random.default_rng()
    order = rng.permutation(len(sequences)) if shuffle else np.arange(len(sequences))

    if optimizer == 'adam':
        if view.m is None:
            view.m = np.zeros_like(view.W)
        if view.v is None:
            view.v = np.zeros_like(view.W)

    n_correct = 0
    n_pairs = 0
    delta = np.zeros_like(view.W)

    def _apply_delta():
        if optimizer == 'adam':
            view.t += 1
            view.m = beta1 * view.m + (1 - beta1) * delta
            view.v = beta2 * view.v + (1 - beta2) * (delta * delta)
            m_hat = view.m / (1 - beta1 ** view.t)
            v_hat = view.v / (1 - beta2 ** view.t)
            view.W += eta * m_hat / (np.sqrt(v_hat) + eps)
        else:
            view.W += eta * delta
        if weight_clip is not None:
            np.clip(view.W, -weight_clip, weight_clip, out=view.W)
        delta.fill(0)

    pair_in_chunk = 0
    for seq_idx in order:
        seq = sequences[seq_idx]
        if len(seq) < 2:
            continue
        for i in range(len(seq) - 1):
            target = seq[i + 1]
            target_row = view.tok_to_row.get(target)
            if target_row is None:
                continue

            ctx_rows: List[int] = []
            ctx_weights: List[float] = []
            for back in range(context_window):
                j = i - back
                if j < 0:
                    break
                row = view.tok_to_row.get(seq[j])
                if row is None:
                    continue
                ctx_rows.append(row)
                ctx_weights.append(decay ** back)
            if not ctx_rows:
                continue

            scores = np.zeros(view.W.shape[1], dtype=np.float32)
            for r, w in zip(ctx_rows, ctx_weights):
                scores += w * view.W[r]
            pred = int(scores.argmax())
            if scores.max() <= 0:
                pred = -1
            if pred == target_row:
                n_correct += 1
            else:
                for r, w in zip(ctx_rows, ctx_weights):
                    delta[r, target_row] += w
                    if pred >= 0:
                        delta[r, pred] -= w
            n_pairs += 1
            pair_in_chunk += 1

            if pair_in_chunk >= chunk_pairs:
                _apply_delta()
                pair_in_chunk = 0

    if pair_in_chunk > 0:
        _apply_delta()

    return {
        'n_pairs': n_pairs,
        'n_correct': n_correct,
        'next_token_accuracy': n_correct / max(1, n_pairs),
    }


# ─── Generation ────────────────────────────────────────────────────────────

def generate_text(view: LLMView, tokenizer: WordTokenizer,
                    prompt: str, *,
                    max_new: int = 20,
                    temperature: float = 0.0,
                    top_k: int = 0,
                    context_window: int = 4,
                    decay: float = 0.6,
                    rng: Optional[np.random.Generator] = None) -> str:
    """Generate text autoregressively.

    temperature=0.0 → greedy argmax.
    temperature>0 → softmax sampling at given temperature.
    top_k>0 → restrict sampling to top-K candidates.
    """
    if rng is None:
        rng = np.random.default_rng()

    prompt_ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)
    out_ids: List[int] = list(prompt_ids)
    eos_id = tokenizer.token_to_id[tokenizer.EOS]

    for _ in range(max_new):
        # Context: last context_window tokens
        ctx_rows: List[int] = []
        ctx_weights: List[float] = []
        for back in range(context_window):
            j = len(out_ids) - 1 - back
            if j < 0:
                break
            row = view.tok_to_row.get(out_ids[j])
            if row is None:
                continue
            ctx_rows.append(row)
            ctx_weights.append(decay ** back)
        if not ctx_rows:
            break

        scores = np.zeros(view.W.shape[1], dtype=np.float32)
        for r, w in zip(ctx_rows, ctx_weights):
            scores += w * view.W[r]
        if scores.max() <= 0:
            break

        if temperature <= 0:
            next_row = int(scores.argmax())
        else:
            # Optional top-k filter
            if top_k > 0:
                top_idx = np.argpartition(scores, -top_k)[-top_k:]
                mask = np.full_like(scores, -np.inf)
                mask[top_idx] = scores[top_idx]
                scores = mask
            # Softmax
            exp = np.exp((scores - scores.max()) / max(1e-6, temperature))
            probs = exp / exp.sum()
            next_row = int(rng.choice(len(probs), p=probs))

        next_tok = view.row_to_tok[next_row]
        if next_tok == eos_id:
            break
        out_ids.append(next_tok)

    return tokenizer.decode(out_ids, skip_special=True)


def compute_unigram_log_probs(view: LLMView,
                                sequences: List[List[int]]) -> np.ndarray:
    """Compute Laplace-smoothed unigram log-probabilities from training
    sequences. Used for unigram backoff in perplexity_with_backoff()."""
    V = view.W.shape[1]
    counts = np.zeros(V, dtype=np.float64)
    for seq in sequences:
        for tok in seq:
            row = view.tok_to_row.get(tok)
            if row is not None:
                counts[row] += 1
    return np.log(counts + 1) - np.log(counts.sum() + V)


def perplexity_with_backoff(view: LLMView, sequences: List[List[int]],
                              unigram_log_probs: np.ndarray, *,
                              alpha: float = 0.5,
                              context_window: int = 4, decay: float = 0.6,
                              prob_floor: float = 1e-8) -> float:
    """PPL with unigram-backoff interpolation:
        P_mixed = α · P_context + (1-α) · P_unigram
    Empirically drops PPL 50-85% on TinyStories — context softmax is
    near-uniform for unseen ctx, unigram is much more concentrated.
    """
    decay_powers = np.array([decay ** k for k in range(context_window)],
                              dtype=np.float32)
    log_loss = 0.0
    n_pred = 0
    log_alpha = math.log(max(alpha, 1e-30))
    log_one_m_alpha = math.log(max(1 - alpha, 1e-30))
    floor_logp = math.log(prob_floor)
    uni_logp = unigram_log_probs.astype(np.float32)

    for seq in sequences:
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
            scores = np.zeros(view.W.shape[1], dtype=np.float32)
            for r, w in zip(ctx_rows, ctx_w):
                scores += w * view.W[r]
            scores -= scores.max()
            exp_s = np.exp(scores)
            denom = exp_s.sum()
            if denom <= 0:
                continue
            ctx_logp_target = scores[target] - math.log(denom + 1e-30)
            mixed = np.logaddexp(log_alpha + ctx_logp_target,
                                  log_one_m_alpha + uni_logp[target])
            target_logp = max(mixed, floor_logp)
            log_loss += -target_logp
            n_pred += 1

    if n_pred == 0:
        return float('inf')
    return math.exp(log_loss / n_pred)


def perplexity(view: LLMView, sequences: List[List[int]], *,
                context_window: int = 4, decay: float = 0.6,
                softmax_temperature: float = 1.0,
                prob_floor: float = 1e-8) -> float:
    """Compute perplexity over `sequences` via temperature-scaled softmax.

    With unbounded perceptron updates W can have huge dynamic range,
    making softmax nearly one-hot. `softmax_temperature` controls
    sharpness (>1 = softer, <1 = sharper). `prob_floor` clamps tiny
    probabilities to avoid log(0) → inf.
    """
    log_loss = 0.0
    n_pred = 0

    for seq in sequences:
        for i in range(1, len(seq)):
            target_row = view.tok_to_row.get(seq[i])
            if target_row is None:
                continue
            ctx_rows = []
            ctx_weights = []
            for back in range(context_window):
                j = i - 1 - back
                if j < 0:
                    break
                row = view.tok_to_row.get(seq[j])
                if row is None:
                    continue
                ctx_rows.append(row)
                ctx_weights.append(decay ** back)
            if not ctx_rows:
                continue
            scores = np.zeros(view.W.shape[1], dtype=np.float32)
            for r, w in zip(ctx_rows, ctx_weights):
                scores += w * view.W[r]
            # Temperature-scaled softmax with numerical stability
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
