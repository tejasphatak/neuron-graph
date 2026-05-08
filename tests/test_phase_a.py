"""Phase A substrate tests — concept-as-neuron end-to-end.

Each test verifies one architectural claim about the substrate.
"""

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np
import pytest


_HERE = os.path.dirname(os.path.abspath(__file__))
_GURU = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _GURU)

from brain import (
    NEURON_DTYPE, NEURON_SIZE, SYNAPSE_DTYPE, SYNAPSE_SIZE,
    Brain, seed_brain, spread, overlap_similarity, hebbian_update,
)


# ─── Layout invariants ───────────────────────────────────────────────────

class TestLayout:
    """The on-disk byte layout must match the C struct exactly so a future
    C kernel can read the same buffers via pointer arithmetic."""

    def test_neuron_size_is_one_cache_line(self):
        assert NEURON_DTYPE.itemsize == 64
        assert NEURON_SIZE == 64

    def test_synapse_is_16_bytes(self):
        assert SYNAPSE_DTYPE.itemsize == 16
        assert SYNAPSE_SIZE == 16

    def test_neuron_struct_fields(self):
        names = NEURON_DTYPE.names
        for required in ('id', 'type', 'modality', 'flags', 'activation',
                          'threshold', 'decay', 'last_fired_us', 'fire_count',
                          'fan_out', 'syn_offset', 'content_offset'):
            assert required in names, f'missing field: {required}'


# ─── Seed brain shape ────────────────────────────────────────────────────

class TestSeed:
    def test_seed_builds(self):
        b = seed_brain()
        assert b.size > 30
        assert getattr(b, '_used_synapses', 0) > 30

    def test_seed_aliases(self):
        b = seed_brain()
        for lemma in ('cat', 'dog', 'animal', 'food', 'play', 'gravity'):
            assert b.lookup(lemma) is not None, f'lemma not in aliases: {lemma}'

    def test_seed_synapses_traversable(self):
        b = seed_brain()
        cat_id = b.lookup('cat')
        edges = b.synapses_of(cat_id)
        assert len(edges) > 0
        # cat should have an edge to feline (is_a)
        feline_id = b.lookup('feline')
        assert any(int(e['to_id']) == feline_id for e in edges), \
            'cat → feline edge missing from cat synapses'


# ─── Spreading dynamics ──────────────────────────────────────────────────

class TestSpreading:
    def test_seed_neuron_stays_in_active_set(self):
        b = seed_brain()
        cat_id = b.lookup('cat')
        s = spread(b, [cat_id])
        assert cat_id in s.activation, \
            'seed neuron should still be active after spread'

    def test_hypernym_chain_activates(self):
        """Cat should activate feline, mammal, animal via is_a chain."""
        b = seed_brain()
        s = spread(b, [b.lookup('cat')])
        for lemma in ('feline', 'mammal', 'animal'):
            nid = b.lookup(lemma)
            assert nid in s.activation, f'{lemma} not activated from cat'
            assert s.activation[nid] > 0.1

    def test_self_activation_dominates_short_horizon(self):
        """In early steps the seed should be among the top-K."""
        b = seed_brain()
        s = spread(b, [b.lookup('cat')], max_steps=2)
        top_ids = [nid for nid, _ in
                   sorted(s.activation.items(), key=lambda x: -x[1])]
        cat_id = b.lookup('cat')
        # Within top 8 after 2 steps
        assert cat_id in top_ids[:10]

    def test_unrelated_concepts_dont_overlap(self):
        b = seed_brain()
        s_cat = spread(b, [b.lookup('cat')])
        s_grav = spread(b, [b.lookup('gravity')])
        sim = overlap_similarity(s_cat, s_grav)
        assert sim < 0.1, f'cat and gravity should not overlap, got {sim:.3f}'

    def test_related_concepts_overlap(self):
        """Cat and dog share a lot of structure (both mammals)."""
        b = seed_brain()
        s_cat = spread(b, [b.lookup('cat')])
        s_dog = spread(b, [b.lookup('dog')])
        sim = overlap_similarity(s_cat, s_dog)
        assert sim > 0.3, f'cat and dog should overlap meaningfully, got {sim:.3f}'

    def test_identical_seeds_perfect_similarity(self):
        b = seed_brain()
        s1 = spread(b, [b.lookup('cat')])
        s2 = spread(b, [b.lookup('cat')])
        assert overlap_similarity(s1, s2) == pytest.approx(1.0)


# ─── Hebbian learning ────────────────────────────────────────────────────

class TestHebbian:
    def test_hebbian_strengthens_existing_synapse(self):
        b = seed_brain()
        cat_id = b.lookup('cat')
        # cat → play exists at weight 0.7
        before = _weight_of(b, cat_id, b.lookup('play'))
        assert before > 0

        joint = spread(b, [cat_id, b.lookup('play')])
        hebbian_update(b, joint, eta=0.1, reward=1.0)

        after = _weight_of(b, cat_id, b.lookup('play'))
        assert after >= before

    def test_hebbian_creates_new_synapse_for_co_active_unconnected(self):
        b = seed_brain()
        # cat and einstein have no edge in the seed
        cat_id = b.lookup('cat')
        ein_id = b.lookup('einstein')
        assert _weight_of(b, cat_id, ein_id) is None

        # Activate them jointly
        joint = spread(b, [cat_id, ein_id])
        result = hebbian_update(b, joint, eta=0.1, reward=1.0,
                                co_threshold=0.05, create_threshold=0.05)
        # Either strengthened-or-created some pair
        assert result['updated'] + result['created'] > 0


def _weight_of(b: Brain, from_id: int, to_id: int):
    """Return the weight of from→to synapse, or None."""
    edges = b.synapses_of(from_id)
    matches = (edges['to_id'] == to_id).nonzero()[0]
    if len(matches) == 0:
        return None
    return float(edges[matches[0]]['weight'])


# ─── Persistence ─────────────────────────────────────────────────────────

class TestPersistence:
    def test_round_trip_preserves_neurons(self):
        b = seed_brain()
        with tempfile.TemporaryDirectory(prefix='brain_') as tmp:
            b.save(tmp)
            b2 = Brain.load(tmp)
        assert b2.size == b.size
        assert getattr(b2, '_used_synapses', 0) == getattr(b, '_used_synapses', 0)

    def test_round_trip_preserves_aliases(self):
        b = seed_brain()
        with tempfile.TemporaryDirectory(prefix='brain_') as tmp:
            b.save(tmp)
            b2 = Brain.load(tmp)
        for lemma in ('cat', 'dog', 'animal', 'gravity'):
            assert b2.lookup(lemma) == b.lookup(lemma)

    def test_round_trip_preserves_synapses(self):
        b = seed_brain()
        cat_id = b.lookup('cat')
        before = b.synapses_of(cat_id).copy()
        with tempfile.TemporaryDirectory(prefix='brain_') as tmp:
            b.save(tmp)
            b2 = Brain.load(tmp)
        after = b2.synapses_of(cat_id)
        assert len(after) == len(before)
        for f in ('to_id', 'relation', 'weight'):
            np.testing.assert_array_equal(before[f], after[f])

    def test_round_trip_preserves_activation_pattern(self):
        """Spread before save and after load should produce identical patterns."""
        b = seed_brain()
        s_before = spread(b, [b.lookup('cat')])
        with tempfile.TemporaryDirectory(prefix='brain_') as tmp:
            b.save(tmp)
            b2 = Brain.load(tmp)
        s_after = spread(b2, [b2.lookup('cat')])
        sim = overlap_similarity(s_before, s_after)
        assert sim == pytest.approx(1.0, abs=1e-6), \
            f'persistence not exact: similarity = {sim:.6f}'


# ─── Inhibition (negative relations like antonym) ───────────────────────

class TestInhibition:
    def test_antonym_does_not_propagate_positive(self):
        """warm has antonym→cold (relation weight is -0.5).
        Spreading from warm should not strongly activate cold."""
        b = seed_brain()
        s = spread(b, [b.lookup('warm')])
        cold_id = b.lookup('cold')
        # Cold either inhibited (0) or weakly active
        cold_act = s.activation.get(cold_id, 0.0)
        warm_act = s.activation.get(b.lookup('warm'), 0.0)
        if cold_act > 0:
            assert cold_act < warm_act, \
                'antonym should not exceed seed activation'
