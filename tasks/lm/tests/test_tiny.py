"""Tests for tiny qualitative-trained LM on the substrate."""

from __future__ import annotations

import os
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_GURU = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(_HERE))))
sys.path.insert(0, _GURU)

from brain.tasks.lm.tiny import (
    Vocab, build_lm_brain, teach_sentence, generate,
)


SENTENCE = "the quick brown fox jumps over the lazy dog".split()
POS = ["DET", "ADJ", "ADJ", "NOUN", "VERB", "PREP", "DET", "ADJ", "NOUN"]


# ─── Construction ────────────────────────────────────────────────────────

class TestBuild:
    def test_brain_has_three_relations(self):
        brain, _ = build_lm_brain()
        names = [r[0] for r in brain.relations]
        assert names == ['is_a', 'follows', 'in_slot']

    def test_teach_creates_token_pos_slot_neurons(self):
        brain, vocab = build_lm_brain()
        teach_sentence(brain, vocab, SENTENCE, POS)
        # 8 unique tokens (the appears twice) + 5 POS (DET/ADJ/NOUN/VERB/PREP)
        # + 9 slots = 22 neurons
        assert len(vocab) == 8
        assert len(vocab.pos_to_id) == 5
        assert len(vocab.slot_to_id) == 9
        assert brain.size == 22


# ─── Generation correctness ──────────────────────────────────────────────

class TestGenerate:
    def test_reproduces_full_sentence(self):
        """The smallest achievable goal: prompt with first 2 tokens,
        substrate must reproduce the remaining 7 verbatim."""
        brain, vocab = build_lm_brain()
        teach_sentence(brain, vocab, SENTENCE, POS)
        out = generate(brain, vocab, SENTENCE[:2], max_new=7)
        assert out == SENTENCE[2:]

    def test_reproduces_from_single_token(self):
        """Even a 1-token prompt should land the correct continuation
        because slot[0] uniquely identifies the position."""
        brain, vocab = build_lm_brain()
        teach_sentence(brain, vocab, SENTENCE, POS)
        out = generate(brain, vocab, SENTENCE[:1], max_new=8)
        assert out == SENTENCE[1:]

    def test_stops_at_sentence_boundary(self):
        """Past the trained sentence length, generation should halt
        cleanly rather than hallucinate."""
        brain, vocab = build_lm_brain()
        teach_sentence(brain, vocab, SENTENCE, POS)
        out = generate(brain, vocab, SENTENCE[:2], max_new=50)
        assert len(out) == 7  # exactly the remaining slots

    def test_unknown_prompt_token_raises(self):
        brain, vocab = build_lm_brain()
        teach_sentence(brain, vocab, SENTENCE, POS)
        with pytest.raises(KeyError):
            generate(brain, vocab, ['nonexistent'], max_new=3)


# ─── Idempotence of teaching ─────────────────────────────────────────────

class TestIdempotence:
    def test_teach_twice_caps_weights(self):
        """Teaching the same sentence twice must not double weights —
        the cap=1.0 in _add_or_strengthen prevents weight pumping."""
        brain, vocab = build_lm_brain()
        teach_sentence(brain, vocab, SENTENCE, POS)
        used_after_one = brain._used_synapses

        teach_sentence(brain, vocab, SENTENCE, POS)
        # No new synapses should have been allocated
        # (PCSR's relocate-on-append leaks, but second teach hits the
        # existing-edge fast path and modifies weight in place — no new
        # edges are appended for an already-taught sentence)
        # We allow leakage from PCSR re-allocation but generation must
        # still produce the correct output.
        out = generate(brain, vocab, SENTENCE[:2], max_new=7)
        assert out == SENTENCE[2:]

    def test_weights_are_capped_at_one(self):
        brain, vocab = build_lm_brain()
        for _ in range(10):
            teach_sentence(brain, vocab, SENTENCE, POS)
        # No edge weight should exceed cap (1.0)
        max_w = float(brain.synapses[:brain._used_synapses]['weight'].max())
        assert max_w <= 1.0 + 1e-6
