"""Smoke tests for the Forward-Write LM — fast, synthetic, no TinyStories.

Guards the three load-bearing claims:
  1. the reservoir has the echo-state property (state stays bounded / contractive),
  2. repeated exposure drives the train surprise DOWN (the write rule learns),
  3. on a learnable synthetic language, held-out PPL beats the unigram floor.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(_HERE))))
sys.path.insert(0, _REPO)

from brain.tasks.llm.forward_write import (
    Reservoir, ForwardWriteLM, unigram_perplexity)


def _markov_corpus(V=20, n_seqs=80, length=30, seed=0):
    """A learnable language: next token = (cur + 1) % V with high probability,
    so context genuinely predicts the next token (unlike i.i.d. noise)."""
    rng = np.random.default_rng(seed)
    seqs = []
    for _ in range(n_seqs):
        t = int(rng.integers(V))
        s = [t]
        for _ in range(length - 1):
            t = (t + 1) % V if rng.random() < 0.9 else int(rng.integers(V))
            s.append(t)
        seqs.append(s)
    return seqs, V


def test_reservoir_echo_state_bounded():
    res = Reservoir(vocab=20, D=64, spectral_radius=0.9, seed=0)
    h = res.reset()
    norms = []
    rng = np.random.default_rng(1)
    for _ in range(200):
        h = res.step(h, int(rng.integers(20)))
        norms.append(np.linalg.norm(h))
    # state must not blow up — echo-state property
    assert np.isfinite(norms[-1])
    assert max(norms) < 100.0


def test_repeated_exposure_lowers_surprise():
    seqs, V = _markov_corpus(seed=2)
    lm = ForwardWriteLM(Reservoir(vocab=V, D=64, seed=2))
    first = lm.train_epoch(seqs, eta=0.05)
    for _ in range(4):
        last = lm.train_epoch(seqs, eta=0.05)
    # the traveled paths get heavy: re-reading drops the surprise
    assert last < first


def test_beats_unigram_floor_on_learnable_language():
    train, V = _markov_corpus(n_seqs=120, seed=3)
    test, _ = _markov_corpus(n_seqs=40, seed=99)
    lm = ForwardWriteLM(Reservoir(vocab=V, D=96, seed=3))
    for _ in range(6):
        lm.train_epoch(train, eta=0.05)
    fw = lm.perplexity(test)
    uni = unigram_perplexity(train, test, V)
    # context-aware reservoir readout must beat the context-free unigram
    assert fw < uni
    assert np.isfinite(fw)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
