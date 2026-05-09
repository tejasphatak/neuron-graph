"""ctypes binding to the substrate-LLM C kernel.

Loads brain/tasks/llm/_inner_loop.so and exposes
train_ngram_epoch_c_negsample() with the same API as the numba JIT
trainer but no venv/numba dependency. Phase C from original
architecture spec.

If the .so isn't built, falls back to numba JIT trainer with a
warning. To build:

  cd brain/tasks/llm
  gcc -O3 -march=native -fPIC -shared -o _inner_loop.so _inner_loop.c
"""

from __future__ import annotations

import ctypes
import os
from typing import List, Optional

import numpy as np


_C_LIB = None
_C_AVAILABLE = False


def _load_lib():
    global _C_LIB, _C_AVAILABLE
    if _C_LIB is not None:
        return _C_LIB
    here = os.path.dirname(os.path.abspath(__file__))
    so_path = os.path.join(here, '_inner_loop.so')
    if not os.path.exists(so_path):
        return None
    lib = ctypes.CDLL(so_path)
    lib.inner_loop_negsample.restype = ctypes.c_int
    lib.inner_loop_negsample.argtypes = [
        ctypes.POINTER(ctypes.c_float),     # W
        ctypes.POINTER(ctypes.c_float),     # delta
        ctypes.c_int64,                     # V
        ctypes.POINTER(ctypes.c_int64),     # seq_rows
        ctypes.POINTER(ctypes.c_int64),     # seq_offsets
        ctypes.c_int64,                     # n_seq
        ctypes.POINTER(ctypes.c_float),     # decay_powers
        ctypes.c_int,                       # ctx_window
        ctypes.c_int,                       # neg_samples
        ctypes.POINTER(ctypes.c_int64),     # neg_targets
        ctypes.c_float,                     # neg_scale
        ctypes.POINTER(ctypes.c_int64),     # n_pairs_out
        ctypes.POINTER(ctypes.c_int64),     # n_correct_out
    ]
    _C_LIB = lib
    _C_AVAILABLE = True
    return lib


def is_c_available() -> bool:
    """Returns True if the compiled .so is loaded successfully."""
    _load_lib()
    return _C_AVAILABLE


def train_ngram_epoch_c_negsample(view, sequences: List[List[int]], *,
                                     context_window: int = 4,
                                     eta: float = 0.05,
                                     decay: float = 0.6,
                                     shuffle: bool = True,
                                     rng: Optional[np.random.Generator] = None,
                                     weight_clip: Optional[float] = 5.0,
                                     neg_samples: int = 3,
                                     neg_scale: float = 0.1) -> dict:
    """C-kernel n-gram trainer with negative sampling.

    Same API and algorithm as train_ngram_epoch_jit_negsample but
    routes the inner loop through the compiled .so. No venv/numba
    dependency. Cleaner single-binary deployment path.
    """
    lib = _load_lib()
    if lib is None:
        # Fallback to numba JIT
        from .jit import train_ngram_epoch_jit_negsample
        return train_ngram_epoch_jit_negsample(
            view, sequences, context_window=context_window, eta=eta,
            decay=decay, shuffle=shuffle, rng=rng, weight_clip=weight_clip,
            neg_samples=neg_samples, neg_scale=neg_scale,
        )

    if rng is None:
        rng = np.random.default_rng()

    # Pre-encode sequences to flat row arrays
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
    n_total_pairs = sum(max(0, len(e) - 1) for e in encoded)
    neg_targets = rng.integers(0, V, size=n_total_pairs * neg_samples + 100,
                                  dtype=np.int64)

    delta = np.zeros_like(view.W)
    n_pairs_out = np.zeros(1, dtype=np.int64)
    n_correct_out = np.zeros(1, dtype=np.int64)

    # Ensure all arrays are contiguous + correct dtype
    W = np.ascontiguousarray(view.W, dtype=np.float32)
    delta = np.ascontiguousarray(delta, dtype=np.float32)
    flat = np.ascontiguousarray(flat, dtype=np.int64)
    offsets = np.ascontiguousarray(offsets, dtype=np.int64)
    decay_powers = np.ascontiguousarray(decay_powers, dtype=np.float32)
    neg_targets = np.ascontiguousarray(neg_targets, dtype=np.int64)

    rc = lib.inner_loop_negsample(
        W.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        delta.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_int64(V),
        flat.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        offsets.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        ctypes.c_int64(len(offsets) - 1),
        decay_powers.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_int(context_window),
        ctypes.c_int(neg_samples),
        neg_targets.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        ctypes.c_float(neg_scale),
        n_pairs_out.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        n_correct_out.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
    )
    if rc != 0:
        raise RuntimeError(f'C kernel returned error code {rc}')

    # The W pointer was passed as data_as — view.W shares storage if it was
    # contiguous; if not, we need to copy back. Be safe.
    view.W[:] = W

    # Apply delta (ascontiguous made a copy if needed; SGD update on view.W)
    view.W += eta * delta
    if weight_clip is not None:
        np.clip(view.W, -weight_clip, weight_clip, out=view.W)

    return {
        'n_pairs': int(n_pairs_out[0]),
        'n_correct': int(n_correct_out[0]),
        'next_token_accuracy': int(n_correct_out[0]) / max(1, int(n_pairs_out[0])),
        'backend': 'c',
    }
