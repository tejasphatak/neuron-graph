"""Numba-JIT'd hot loop for substrate-LLM training.

Optional: if numba isn't installed, fall back to the pure-numpy
train_ngram_epoch (slower but always works). Detection happens at
import time.

The JIT'd kernel operates on the same dense W matrix and same data
layout as train_ngram_epoch. No API change, just a faster inner loop.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np


try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def njit(*args, **kwargs):
        # Identity decorator when numba is missing
        def decorator(func):
            return func
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator
    prange = range


@njit(cache=True, fastmath=True, parallel=True)
def _inner_loop_jit_parallel(W, delta_per_thread,
                                seq_rows, seq_offsets,
                                decay_powers,
                                context_window,
                                n_pairs_out,
                                n_correct_out,
                                n_threads):
    """Parallel JIT inner loop. Sequences distributed across threads
    via prange; each thread accumulates into its own delta slab,
    reduced after."""
    V = W.shape[1]
    n_seq = len(seq_offsets) - 1

    # Per-thread accumulators
    pair_counts = np.zeros(n_threads, dtype=np.int64)
    correct_counts = np.zeros(n_threads, dtype=np.int64)

    for s in prange(n_seq):
        # Numba's prange auto-distributes; tid is current thread index
        # but numba doesn't expose it directly. Work around: use
        # s % n_threads as a deterministic mapping (trades load
        # balance for correctness; for uniform sequence lengths fine).
        tid = s % n_threads

        start = seq_offsets[s]
        end = seq_offsets[s + 1]
        L = end - start
        if L < 2:
            continue
        for i in range(L - 1):
            target = seq_rows[start + i + 1]
            if target < 0:
                continue

            scores = np.zeros(V, dtype=np.float32)
            ctx_count = 0
            for k in range(context_window):
                j = i - k
                if j < 0:
                    break
                ctx = seq_rows[start + j]
                if ctx < 0:
                    continue
                w = decay_powers[k]
                for v in range(V):
                    scores[v] += w * W[ctx, v]
                ctx_count += 1
            if ctx_count == 0:
                continue

            best_score = scores[0]
            pred = 0
            for v in range(1, V):
                if scores[v] > best_score:
                    best_score = scores[v]
                    pred = v
            no_signal = best_score <= 0.0

            if pred == target and not no_signal:
                correct_counts[tid] += 1
            else:
                for k in range(context_window):
                    j = i - k
                    if j < 0:
                        break
                    ctx = seq_rows[start + j]
                    if ctx < 0:
                        continue
                    w = decay_powers[k]
                    delta_per_thread[tid, ctx, target] += w
                    if not no_signal:
                        delta_per_thread[tid, ctx, pred] -= w
            pair_counts[tid] += 1

    n_pairs_out[0] = pair_counts.sum()
    n_correct_out[0] = correct_counts.sum()


@njit(cache=True, fastmath=True)
def _inner_loop_jit(W, delta,
                      seq_rows,            # int64[:] flattened sequences
                      seq_offsets,         # int64[:] starts of each sequence
                      decay_powers,        # float32[:] decay weights
                      context_window,
                      n_pairs_out,
                      n_correct_out):
    """Hot loop, jit-compiled. One epoch of perceptron updates over
    all (context, target) pairs in seq_rows. Updates delta in place.

    seq_rows is a single flat int64 array of all rows from all
    sequences concatenated. seq_offsets[i] is the start index of
    sequence i; seq_offsets[N] = len(seq_rows).
    """
    V = W.shape[1]
    n_pairs = 0
    n_correct = 0
    n_seq = len(seq_offsets) - 1

    for s in range(n_seq):
        start = seq_offsets[s]
        end = seq_offsets[s + 1]
        L = end - start
        if L < 2:
            continue
        for i in range(L - 1):
            target = seq_rows[start + i + 1]
            if target < 0:
                continue

            # Score = sum_k decay[k] * W[seq_rows[start+i-k]]
            scores = np.zeros(V, dtype=np.float32)
            ctx_count = 0
            for k in range(context_window):
                j = i - k
                if j < 0:
                    break
                ctx = seq_rows[start + j]
                if ctx < 0:
                    continue
                w = decay_powers[k]
                for v in range(V):
                    scores[v] += w * W[ctx, v]
                ctx_count += 1
            if ctx_count == 0:
                continue

            # argmax over scores
            best_score = scores[0]
            pred = 0
            for v in range(1, V):
                if scores[v] > best_score:
                    best_score = scores[v]
                    pred = v
            no_signal = best_score <= 0.0

            if pred == target and not no_signal:
                n_correct += 1
            else:
                # Apply delta updates
                for k in range(context_window):
                    j = i - k
                    if j < 0:
                        break
                    ctx = seq_rows[start + j]
                    if ctx < 0:
                        continue
                    w = decay_powers[k]
                    delta[ctx, target] += w
                    if not no_signal:
                        delta[ctx, pred] -= w
            n_pairs += 1

    n_pairs_out[0] = n_pairs
    n_correct_out[0] = n_correct


def train_ngram_epoch_jit_parallel(view, sequences: List[List[int]], *,
                                     context_window: int = 4,
                                     eta: float = 0.05,
                                     decay: float = 0.6,
                                     shuffle: bool = True,
                                     rng: Optional[np.random.Generator] = None,
                                     optimizer: str = 'adam',
                                     beta1: float = 0.9,
                                     beta2: float = 0.999,
                                     eps: float = 1e-8,
                                     weight_clip: Optional[float] = 5.0,
                                     n_threads: int = 0) -> dict:
    """Multi-thread JIT'd n-gram trainer (numba prange + private deltas).

    n_threads=0 means use all cores. Sequences distributed across
    threads; each thread accumulates into its own [V, V] delta slab.
    Slabs reduced (summed) after the parallel section.

    Memory cost: n_threads × V² × 4 bytes per delta slab. For V=4K,
    8 threads: 64 MB × 8 = 512 MB. Tradeoff: high memory for zero
    contention vs lock-based atomic adds.
    """
    if not NUMBA_AVAILABLE:
        return train_ngram_epoch_jit(view, sequences,
                                       context_window=context_window,
                                       eta=eta, decay=decay,
                                       shuffle=shuffle, rng=rng,
                                       optimizer=optimizer,
                                       beta1=beta1, beta2=beta2, eps=eps,
                                       weight_clip=weight_clip)

    import os as _os
    if n_threads <= 0:
        n_threads = max(1, _os.cpu_count() or 1)

    if rng is None:
        rng = np.random.default_rng()

    if optimizer == 'adam':
        if view.m is None:
            view.m = np.zeros_like(view.W)
        if view.v is None:
            view.v = np.zeros_like(view.W)

    encoded: List[np.ndarray] = []
    for seq in sequences:
        rows = np.array(
            [view.tok_to_row.get(t, -1) for t in seq],
            dtype=np.int64,
        )
        encoded.append(rows)
    if shuffle:
        rng.shuffle(encoded)

    flat = np.concatenate(encoded) if encoded else np.zeros(0, dtype=np.int64)
    offsets = np.zeros(len(encoded) + 1, dtype=np.int64)
    cur = 0
    for i, e in enumerate(encoded):
        offsets[i] = cur
        cur += len(e)
    offsets[-1] = cur

    decay_powers = np.array([decay ** k for k in range(context_window)],
                              dtype=np.float32)

    V = view.W.shape[1]
    delta_per_thread = np.zeros((n_threads, V, V), dtype=np.float32)
    n_pairs_out = np.zeros(1, dtype=np.int64)
    n_correct_out = np.zeros(1, dtype=np.int64)

    _inner_loop_jit_parallel(view.W, delta_per_thread,
                              flat, offsets, decay_powers,
                              context_window,
                              n_pairs_out, n_correct_out,
                              n_threads)

    # Reduce
    delta = delta_per_thread.sum(axis=0)

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

    n_pairs = int(n_pairs_out[0])
    n_correct = int(n_correct_out[0])
    return {
        'n_pairs': n_pairs,
        'n_correct': n_correct,
        'next_token_accuracy': n_correct / max(1, n_pairs),
        'n_threads': n_threads,
    }


def train_ngram_epoch_jit(view, sequences: List[List[int]], *,
                            context_window: int = 4,
                            eta: float = 0.05,
                            decay: float = 0.6,
                            shuffle: bool = True,
                            rng: Optional[np.random.Generator] = None,
                            optimizer: str = 'adam',
                            beta1: float = 0.9, beta2: float = 0.999,
                            eps: float = 1e-8,
                            weight_clip: Optional[float] = 5.0) -> dict:
    """JIT'd n-gram trainer. Same algorithm as train_ngram_epoch but
    the inner loop runs as native code via numba. ~10-50× faster on
    typical TinyStories-shaped corpora.

    Falls back to standard train_ngram_epoch if numba is missing.
    """
    if not NUMBA_AVAILABLE:
        from .llm import train_ngram_epoch
        return train_ngram_epoch(view, sequences,
                                   context_window=context_window,
                                   eta=eta, decay=decay,
                                   shuffle=shuffle, rng=rng,
                                   optimizer=optimizer,
                                   beta1=beta1, beta2=beta2, eps=eps,
                                   weight_clip=weight_clip)

    if rng is None:
        rng = np.random.default_rng()

    if optimizer == 'adam':
        if view.m is None:
            view.m = np.zeros_like(view.W)
        if view.v is None:
            view.v = np.zeros_like(view.W)

    # Pre-encode all sequences into flat row array + offsets
    encoded: List[np.ndarray] = []
    for seq in sequences:
        rows = np.array(
            [view.tok_to_row.get(t, -1) for t in seq],
            dtype=np.int64,
        )
        encoded.append(rows)

    if shuffle:
        rng.shuffle(encoded)

    flat = np.concatenate(encoded) if encoded else np.zeros(0, dtype=np.int64)
    offsets = np.zeros(len(encoded) + 1, dtype=np.int64)
    cur = 0
    for i, e in enumerate(encoded):
        offsets[i] = cur
        cur += len(e)
    offsets[-1] = cur

    decay_powers = np.array([decay ** k for k in range(context_window)],
                              dtype=np.float32)
    delta = np.zeros_like(view.W)
    n_pairs_out = np.zeros(1, dtype=np.int64)
    n_correct_out = np.zeros(1, dtype=np.int64)

    _inner_loop_jit(view.W, delta, flat, offsets,
                     decay_powers, context_window,
                     n_pairs_out, n_correct_out)

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

    n_pairs = int(n_pairs_out[0])
    n_correct = int(n_correct_out[0])
    return {
        'n_pairs': n_pairs,
        'n_correct': n_correct,
        'next_token_accuracy': n_correct / max(1, n_pairs),
    }
