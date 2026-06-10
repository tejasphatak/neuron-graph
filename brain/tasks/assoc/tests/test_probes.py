"""Smoke tests for the reasoning + compounding dip-test probes.

Small, fast configs. These guard the probes against bit-rot and assert the core
claim in each: the substrate's compositional mechanism separates from the
retrieval/feedforward foil on held-out items.
"""

from __future__ import annotations

import os
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(_HERE))))
sys.path.insert(0, _REPO)

from brain.tasks.assoc import reasoning_probe as rp
from brain.tasks.assoc import compounding_probe as cp


# --------------------------------------------------------------------------- #
# reasoning probe: substrate beats the 1-hop retrieval foil on held-out items
# --------------------------------------------------------------------------- #
def test_rung1_chaining_beats_foil_on_heldout():
    r = rp.rung1_chaining(n_chains=8, chain_len=4, trials_per=20, seed=0)
    # held-out = hops >= 2; substrate should clear the foil and chance
    sub2 = sum(r["substrate"][1:]) / len(r["substrate"][1:])
    base2 = sum(r["baseline"][1:]) / len(r["baseline"][1:])
    assert sub2 > base2 + 0.2         # separates from the 1-hop foil (the claim)
    assert sub2 > 0.4                  # lenient absolute floor for the tiny config


def test_rung2_binding_dereferences():
    r = rp.rung2_binding(n_vars=12, n_alias=12, trials=150, seed=1)
    assert r["substrate"] > r["baseline"] + 0.3
    # the foil returns a variable (wrong type) most of the time
    assert r["baseline_type_error_rate"] > 0.5


def test_rung3_systematic_generalization():
    r = rp.rung3_scan(seed=2)
    assert r["substrate"] > 0.9          # composes held-out combinations
    assert r["baseline"] < r["substrate"]


def test_rung4_fsm_length_generalization():
    r = rp.rung4_fsm(n_states=4, max_len=6, trials=60, seed=3)
    # substrate (looped) holds on long strings where whole-string retrieval cliffs
    long_sub = r["substrate"][-1]
    long_base = r["baseline"][-1]
    assert long_sub > 0.9
    assert long_sub > long_base + 0.2


# --------------------------------------------------------------------------- #
# compounding probe: associative cleanup separates from feedforward
# --------------------------------------------------------------------------- #
def test_compounding_cleanup_beats_feedforward():
    r = cp.length_sweep(M=20, d=64, lengths=(1, 4, 16), trials=30, seed=0)
    # at the longest chain, cleanup ON should dominate feedforward OFF
    assert r["on"][-1] > r["off"][-1] + 0.3
    assert r["on"][-1] > 0.8
    # and feedforward should actually compound downward
    assert r["off"][-1] < r["off"][0]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
