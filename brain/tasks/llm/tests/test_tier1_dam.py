"""Smoke tests for the Tier-1 degree-n associative readout.

Fast and deterministic — a hand-built W with clean bigram structure, no
TinyStories download. Guards the bridge against bit-rot and asserts the two
load-bearing claims:

  1. degree=1 reproduces the existing linear readout (llm.perplexity).
  2. on a cleanly-memorized bigram language, the degree-n cleanup pass does not
     wreck prediction — PPL stays finite and competitive with the baseline.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(_HERE))))
sys.path.insert(0, _REPO)

from brain.tasks.llm.llm import LLMView, perplexity
from brain.tasks.llm import tier1_dam_readout as t1


def _bigram_view(V: int, seed: int = 0):
    """A deterministic ring language: token i is followed by (i+1) % V.
    W encodes it as a strong diagonal-shifted coupling plus light noise."""
    rng = np.random.default_rng(seed)
    W = 0.05 * rng.standard_normal((V, V)).astype(np.float32)
    for i in range(V):
        W[i, (i + 1) % V] += 4.0
    view = LLMView(W=W, tok_to_row={t: t for t in range(V)},
                   row_to_tok=list(range(V)))
    seqs = [[(start + k) % V for k in range(12)] for start in range(V)]
    return view, seqs


def test_degree1_matches_linear_perplexity():
    view, seqs = _bigram_view(24)
    ppl_ref = perplexity(view, seqs, context_window=4, decay=0.6)
    ppl_t1 = t1.perplexity_dam(view, seqs, degree=1, context_window=4,
                               decay=0.6, standardize=False)
    assert ppl_t1 == pytest.approx(ppl_ref, rel=1e-5)


def test_degree_n_readout_runs_and_predicts():
    view, seqs = _bigram_view(24)
    res = t1.degree_sweep(view, seqs, degrees=(1, 2, 4),
                          context_window=4, standardize=True)
    for k, v in res.items():
        assert np.isfinite(v), f"{k} PPL not finite"
    # cleanup must not blow prediction apart on a cleanly memorized language
    assert res["deg4"] < 5.0 * res["deg1"]


def test_softmax_readout_is_attention():
    view, seqs = _bigram_view(16)
    ppl = t1.perplexity_dam(view, seqs, degree=2, interaction="softmax",
                            context_window=4, standardize=True)
    assert np.isfinite(ppl)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
