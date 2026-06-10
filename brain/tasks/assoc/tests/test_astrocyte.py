"""Astrocyte associative-memory tests.

The credibility anchor (same discipline as MNIST's "spread() == fast path"
100% match): the SUBSTRATE gather/scatter, walking real CSR synapse blocks,
must reproduce the DENSE numpy reference exactly at full connectivity. Then a
capacity-separation smoke test (degree-4 stores more than degree-2) and the
softmax==self-attention equivalence.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(_HERE))))
sys.path.insert(0, _REPO)

from brain.astrocyte import NeuronAstrocyteMemory, SubstrateAstrocyteMemory
from brain.neuron import NeuronType


# --------------------------------------------------------------------------- #
# 1. Substrate == dense reference at full connectivity
# --------------------------------------------------------------------------- #
def test_substrate_gather_scatter_matches_dense_poly():
    """One step of substrate gather/scatter == dense matmul, bit-for-sign."""
    rng = np.random.default_rng(0)
    N, K = 40, 12
    P = rng.choice([-1.0, 1.0], size=(K, N))

    dense = NeuronAstrocyteMemory(interaction="poly", degree=4,
                                  activation="sign").store(P)
    sub = SubstrateAstrocyteMemory(N, interaction="poly", degree=4,
                                   activation="sign").store(P, connectivity=1.0)

    x = rng.choice([-1.0, 1.0], size=N)
    # internal passes agree to float tolerance...
    od = dense._gather(dense._phi(x))
    osub = sub._gather(sub._phi(x))
    assert np.allclose(od, osub, atol=1e-6)
    # ...and the post-phi sign output is exactly equal
    assert np.array_equal(dense.step(x), sub.step(x))


def test_substrate_retrieval_matches_dense():
    """Full relaxation from a corrupted cue lands on the same fixed point."""
    rng = np.random.default_rng(1)
    N, K = 48, 8
    P = rng.choice([-1.0, 1.0], size=(K, N))
    dense = NeuronAstrocyteMemory(degree=4, activation="sign").store(P)
    sub = SubstrateAstrocyteMemory(N, degree=4, activation="sign").store(P)

    cue = P[3].copy()
    cue[rng.choice(N, size=5, replace=False)] *= -1
    assert np.array_equal(dense.retrieve(cue), sub.retrieve(cue))


def test_substrate_recalls_clean_pattern():
    """Sanity: an uncorrupted stored pattern is a fixed point and recalls itself."""
    rng = np.random.default_rng(2)
    N, K = 32, 5
    P = rng.choice([-1.0, 1.0], size=(K, N))
    sub = SubstrateAstrocyteMemory(N, degree=4, activation="sign").store(P)
    assert np.array_equal(sub.retrieve(P[2].copy()), P[2])


# --------------------------------------------------------------------------- #
# 2. The astrocytes are real substrate neurons with real edges
# --------------------------------------------------------------------------- #
def test_astrocytes_are_neurons_with_edges():
    N, K = 16, 6
    P = np.ones((K, N))
    sub = SubstrateAstrocyteMemory(N).store(P, connectivity=1.0)
    assert sub.brain.size == N + K
    for aid in sub._astro_ids:
        assert sub.brain.nodes[aid]["type"] == NeuronType.ASTROCYTE
        assert len(sub.brain.synapses_of(aid)) == N      # full connectivity


def test_connectivity_controls_edge_count():
    """r = K/N knob: fewer neighbors per astrocyte at lower connectivity."""
    N, K = 100, 4
    P = np.ones((K, N))
    sub = SubstrateAstrocyteMemory(N).store(P, connectivity=0.25,
                                            rng=np.random.default_rng(0))
    for aid in sub._astro_ids:
        assert len(sub.brain.synapses_of(aid)) == 25     # round(0.25 * 100)


# --------------------------------------------------------------------------- #
# 3. Capacity separation: degree-4 stores more than degree-2
# --------------------------------------------------------------------------- #
def _recall_rate(N, K, degree, trials, rng):
    ok = 0
    for _ in range(trials):
        P = rng.choice([-1.0, 1.0], size=(K, N))
        mem = SubstrateAstrocyteMemory(N, degree=degree,
                                       activation="sign").store(P)
        t = int(rng.integers(K))
        cue = P[t].copy()
        cue[rng.choice(N, size=max(1, N // 10), replace=False)] *= -1
        if np.array_equal(mem.retrieve(cue, steps=30), P[t]):
            ok += 1
    return ok / trials


def test_quartic_beats_pairwise_capacity():
    """At a load that breaks classic Hopfield, the quartic still recalls."""
    rng = np.random.default_rng(3)
    N = 48
    K = N          # K = N: well past the ~0.14 N pairwise limit
    r2 = _recall_rate(N, K, 2, trials=8, rng=rng)
    r4 = _recall_rate(N, K, 4, trials=8, rng=rng)
    assert r4 > r2
    assert r4 >= 0.9          # quartic still essentially perfect at K=N


# --------------------------------------------------------------------------- #
# 4. softmax interaction == transformer self-attention
# --------------------------------------------------------------------------- #
def test_softmax_regime_is_self_attention():
    rng = np.random.default_rng(0)
    n_tok, d = 12, 16
    X = rng.standard_normal((n_tok, d))
    Wk, Wv, Wq = (rng.standard_normal((d, d)) / np.sqrt(d) for _ in range(3))
    Kmat, V = X @ Wk, X @ Wv
    Q = rng.standard_normal((5, d)) @ Wq
    beta = 1.0 / np.sqrt(d)

    scores = Q @ Kmat.T * beta
    A = np.exp(scores - scores.max(1, keepdims=True))
    A = A / A.sum(1, keepdims=True)
    ref = A @ V

    maxerr = 0.0
    for qi in range(Q.shape[0]):
        mem = SubstrateAstrocyteMemory(d, interaction="softmax", beta=beta,
                                       activation="identity").store(Kmat)
        glio = mem._glio(mem._gather(Q[qi]))     # softmax attention weights
        out = glio @ V                            # value scatter
        maxerr = max(maxerr, float(np.abs(out - ref[qi]).max()))
    assert maxerr < 1e-6                          # exact self-attention


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
