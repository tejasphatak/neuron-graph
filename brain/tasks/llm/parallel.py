"""Multiprocess parallel training for substrate-LLM.

Pattern: data-parallel SGD with delta-aggregation per epoch.
Workers each get a snapshot of W (via shared memory) + a chunk of
training sequences. Each worker computes its own delta from
perceptron updates, returns delta. Main process sums deltas and
applies via Adam/SGD.

This is ASYNC-style: workers see a stale W (snapshot at epoch start)
but updates aggregate correctly. Equivalent to large-batch SGD with
batch_size = chunk_size.

Speedup: ~4-8× on 8 cores for V≤10K (limited by W broadcast cost).
"""

from __future__ import annotations

import os
from multiprocessing import shared_memory, Pool
from typing import List, Optional

import numpy as np

from .llm import LLMView


_GLOBAL_SHM_NAME = None
_GLOBAL_W_SHAPE = None
_GLOBAL_W_DTYPE = None


def _worker_init(shm_name: str, shape: tuple, dtype):
    """Attach to shared-memory W in each worker process."""
    global _GLOBAL_SHM_NAME, _GLOBAL_W_SHAPE, _GLOBAL_W_DTYPE
    _GLOBAL_SHM_NAME = shm_name
    _GLOBAL_W_SHAPE = shape
    _GLOBAL_W_DTYPE = dtype


def _train_chunk(args):
    """Worker: compute delta on a chunk of pre-encoded sequences."""
    encoded_chunk, ctx_window, decay, weight_clip = args
    # Attach to shared W (read-only view)
    shm = shared_memory.SharedMemory(name=_GLOBAL_SHM_NAME)
    W = np.ndarray(_GLOBAL_W_SHAPE, dtype=_GLOBAL_W_DTYPE, buffer=shm.buf)

    decay_powers = [decay ** k for k in range(ctx_window)]
    delta = np.zeros(_GLOBAL_W_SHAPE, dtype=_GLOBAL_W_DTYPE)
    n_correct = 0
    n_pairs = 0
    V = _GLOBAL_W_SHAPE[1]

    for seq in encoded_chunk:
        L = len(seq)
        for i in range(L - 1):
            target_row = seq[i + 1]
            ctx_rows = []
            ctx_weights = []
            for back in range(ctx_window):
                j = i - back
                if j < 0:
                    break
                ctx_rows.append(seq[j])
                ctx_weights.append(decay_powers[back])

            scores = np.zeros(V, dtype=np.float32)
            for r, w in zip(ctx_rows, ctx_weights):
                scores += w * W[r]
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

    shm.close()
    return delta, n_correct, n_pairs


def _encode_sequences_to_rows(sequences: List[List[int]],
                                tok_to_row) -> List[List[int]]:
    """Convert tokenizer ids → W row indices, dropping unknowns."""
    out: List[List[int]] = []
    for seq in sequences:
        rows: List[int] = []
        for tok in seq:
            row = tok_to_row.get(tok)
            if row is not None:
                rows.append(row)
        if len(rows) >= 2:
            out.append(rows)
    return out


def train_ngram_epoch_parallel(view: LLMView,
                                 sequences: List[List[int]], *,
                                 n_workers: int = 0,
                                 context_window: int = 4,
                                 eta: float = 0.05,
                                 decay: float = 0.6,
                                 shuffle: bool = True,
                                 rng: Optional[np.random.Generator] = None,
                                 optimizer: str = 'adam',
                                 beta1: float = 0.9,
                                 beta2: float = 0.999,
                                 eps: float = 1e-8,
                                 weight_clip: Optional[float] = 5.0) -> dict:
    """Multiprocess data-parallel n-gram epoch.

    n_workers=0 means use all CPU cores (os.cpu_count()).
    """
    if n_workers <= 0:
        n_workers = max(1, os.cpu_count() or 1)
    if rng is None:
        rng = np.random.default_rng()

    # Pre-encode sequences once (avoid Python dict lookups in workers)
    encoded = _encode_sequences_to_rows(sequences, view.tok_to_row)
    if shuffle:
        rng.shuffle(encoded)

    # Set up shared memory for W
    shm = shared_memory.SharedMemory(create=True, size=view.W.nbytes)
    try:
        W_shared = np.ndarray(view.W.shape, dtype=view.W.dtype, buffer=shm.buf)
        W_shared[:] = view.W

        # Split chunks
        n = len(encoded)
        chunk_size = max(1, (n + n_workers - 1) // n_workers)
        chunks = [encoded[i:i + chunk_size] for i in range(0, n, chunk_size)]
        args_list = [(c, context_window, decay, weight_clip) for c in chunks]

        # Optimizer state lazy init
        if optimizer == 'adam':
            if view.m is None:
                view.m = np.zeros_like(view.W)
            if view.v is None:
                view.v = np.zeros_like(view.W)

        with Pool(n_workers,
                   initializer=_worker_init,
                   initargs=(shm.name, view.W.shape, view.W.dtype)) as pool:
            results = pool.map(_train_chunk, args_list)
    finally:
        shm.close()
        shm.unlink()

    # Aggregate
    total_delta = np.zeros_like(view.W)
    n_correct = 0
    n_pairs = 0
    for d, c, p in results:
        total_delta += d
        n_correct += c
        n_pairs += p

    # Apply
    if optimizer == 'adam':
        view.t += 1
        view.m = beta1 * view.m + (1 - beta1) * total_delta
        view.v = beta2 * view.v + (1 - beta2) * (total_delta * total_delta)
        m_hat = view.m / (1 - beta1 ** view.t)
        v_hat = view.v / (1 - beta2 ** view.t)
        view.W += eta * m_hat / (np.sqrt(v_hat) + eps)
    else:
        view.W += eta * total_delta

    if weight_clip is not None:
        np.clip(view.W, -weight_clip, weight_clip, out=view.W)

    return {
        'n_pairs': n_pairs,
        'n_correct': n_correct,
        'next_token_accuracy': n_correct / max(1, n_pairs),
        'n_workers': n_workers,
    }
