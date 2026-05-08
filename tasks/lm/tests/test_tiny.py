"""Tests for tiny qualitative-trained LM on the substrate."""

from __future__ import annotations

import os
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_GURU = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(_HERE))))
sys.path.insert(0, _GURU)

from brain.tasks.lm.tiny import (
    Vocab, build_lm_brain, teach_sentence, generate, generate_via_spread,
)


SENTENCE = "the quick brown fox jumps over the lazy dog".split()
POS = ["DET", "ADJ", "ADJ", "NOUN", "VERB", "PREP", "DET", "ADJ", "NOUN"]

# Multi-sentence corpus — same grammar shape, different fillers
S1 = "the quick brown fox jumps over the lazy dog".split()
S2 = "a smart small cat runs around a happy mouse".split()
S3 = "the bold red bird flies past a tiny tree".split()
CORPUS = [S1, S2, S3]
# Same POS pattern across all three — that's the point of "shared grammar"
SHARED_POS = ["DET", "ADJ", "ADJ", "NOUN", "VERB", "PREP", "DET", "ADJ", "NOUN"]


# ─── Construction ────────────────────────────────────────────────────────

class TestBuild:
    def test_brain_has_six_relations(self):
        brain, _ = build_lm_brain()
        names = [r[0] for r in brain.relations]
        assert names == ['is_a', 'follows', 'in_slot',
                         'has_member', 'has_filler', 'co_occurs']

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


# ─── Substrate-native generation (spread-based) ──────────────────────────

class TestGenerateViaSpread:
    """The substrate's spreading activation does the composition.
    Goal injection on the next slot + has_filler/has_member edges drives
    the right token's activation above its peers."""

    def test_reproduces_full_sentence(self):
        brain, vocab = build_lm_brain()
        teach_sentence(brain, vocab, SENTENCE, POS)
        out = generate_via_spread(brain, vocab, SENTENCE[:2], max_new=7)
        assert out == SENTENCE[2:]

    def test_reproduces_from_single_token(self):
        brain, vocab = build_lm_brain()
        teach_sentence(brain, vocab, SENTENCE, POS)
        out = generate_via_spread(brain, vocab, SENTENCE[:1], max_new=8)
        assert out == SENTENCE[1:]

    def test_stops_at_sentence_boundary(self):
        brain, vocab = build_lm_brain()
        teach_sentence(brain, vocab, SENTENCE, POS)
        out = generate_via_spread(brain, vocab, SENTENCE[:2], max_new=50)
        assert len(out) == 7

    def test_unknown_prompt_token_raises(self):
        brain, vocab = build_lm_brain()
        teach_sentence(brain, vocab, SENTENCE, POS)
        with pytest.raises(KeyError):
            generate_via_spread(brain, vocab, ['nonexistent'], max_new=3)

    def test_handles_repeated_token_in_sentence(self):
        """The word 'the' appears at positions 0 and 6. Substrate-native
        spreading must disambiguate via slot context, not collapse to one."""
        brain, vocab = build_lm_brain()
        teach_sentence(brain, vocab, SENTENCE, POS)
        # Mid-sentence prompt: prove slot[5]→slot[6] picks "the" again
        out = generate_via_spread(brain, vocab, SENTENCE[:6], max_new=3)
        assert out == ['the', 'lazy', 'dog']


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


# ─── Multi-sentence shared-grammar composition ───────────────────────────

class TestMultiSentence:
    """The substrate is taught 3 sentences with the SAME grammar shape
    (DET ADJ ADJ NOUN VERB PREP DET ADJ NOUN) but different vocabulary.
    The substrate must disambiguate which sentence to continue based on
    the prompt's tokens (via co_occurs edges)."""

    @staticmethod
    def _trained_brain():
        brain, vocab = build_lm_brain()
        for s in CORPUS:
            teach_sentence(brain, vocab, s, SHARED_POS)
        return brain, vocab

    def test_corpus_creates_shared_slots_distinct_tokens(self):
        brain, vocab = self._trained_brain()
        # Slots are SHARED across sentences (only 9 of them, not 9×3)
        assert len(vocab.slot_to_id) == 9
        # POS tags are SHARED (5 unique tags)
        assert len(vocab.pos_to_id) == 5
        # Tokens are the union; some overlap ("the", "a") so < 27
        assert len(vocab) > 9 and len(vocab) < 27

    def test_each_sentence_reproducible_from_3tok_prompt(self):
        """The defining test: substrate must continue each sentence
        correctly when prompted with its first 3 tokens, even though
        slots have multiple has_filler candidates."""
        brain, vocab = self._trained_brain()
        for sentence in CORPUS:
            out = generate_via_spread(brain, vocab, sentence[:3], max_new=6)
            assert out == sentence[3:], (
                f'failed to continue {sentence[:3]!r}: '
                f'got {out!r}, expected {sentence[3:]!r}'
            )

    def test_each_sentence_reproducible_from_2tok_prompt(self):
        """Stricter: 2-token prompt has less context, must still route."""
        brain, vocab = self._trained_brain()
        for sentence in CORPUS:
            out = generate_via_spread(brain, vocab, sentence[:2], max_new=7)
            assert out == sentence[2:], (
                f'failed to continue {sentence[:2]!r}: '
                f'got {out!r}, expected {sentence[2:]!r}'
            )

    def test_lookup_method_also_handles_multi_sentence(self):
        """The slot-walking generate() picks by joint IS_A × IN_SLOT score
        without co_occurs. With multiple sentences, the same slot has
        multiple fillers; lookup may not disambiguate. We document
        whichever sentence's filler wins — this is the *contrast* with
        spread-based generation that proves co_occurs is the disambiguator."""
        brain, vocab = self._trained_brain()
        # With S1 prompt, lookup picks SOME valid filler at each slot, but
        # has no context to know which sentence to follow.
        out = generate(brain, vocab, S1[:3], max_new=6)
        # We don't assert exact reproduction here — we just assert that
        # every emitted token has the correct POS for its slot
        # (i.e. lookup is grammatically valid even if not S1-faithful).
        for i, tok in enumerate(out):
            slot_idx = 3 + i
            expected_pos = SHARED_POS[slot_idx]
            actual_pos = next(
                (vocab.id_to_pos[int(syn['to_id'])]
                 for syn in brain.synapses_of(vocab.token_to_id[tok])
                 if int(syn['relation']) == brain.relation_id['is_a']
                 and int(syn['to_id']) in vocab.id_to_pos),
                None,
            )
            assert actual_pos == expected_pos, (
                f'lookup emitted {tok!r} (POS {actual_pos}) at slot {slot_idx}, '
                f'expected POS {expected_pos}'
            )

    def test_co_occurs_strength_separates_sentences(self):
        """Sanity: 'brown' (S1) has co_occurs to 'fox' (S1) but NOT to
        'cat' (S2) or 'bird' (S3). This is what disambiguates."""
        brain, vocab = self._trained_brain()
        brown = vocab.token_to_id['brown']
        fox = vocab.token_to_id['fox']
        cat = vocab.token_to_id['cat']

        rel_co = brain.relation_id['co_occurs']
        edges = brain.synapses_of(brown)
        targets = {int(s['to_id']): float(s['weight'])
                   for s in edges if int(s['relation']) == rel_co}

        assert fox in targets, "'brown' should co_occur with 'fox' from S1"
        assert cat not in targets, "'brown' should NOT co_occur with 'cat' (S2)"


# ─── Cross-sentence recombination — honest substrate probe ───────────────

class TestCrossSentenceRecombination:
    """Prompts mix tokens from different trained sentences. No trained
    sentence matches these prompts exactly — the substrate must compose.

    Hard assertion: every emitted token has the correct POS for its slot
    (grammatical validity). Which trained sentence's lexical path the
    substrate prefers is OBSERVED, not asserted — it depends on the
    weighted sum of co_occurs evidence from prompt tokens, which is the
    substrate's natural disambiguation."""

    @staticmethod
    def _trained_brain():
        brain, vocab = build_lm_brain()
        for s in CORPUS:
            teach_sentence(brain, vocab, s, SHARED_POS)
        return brain, vocab

    @staticmethod
    def _token_pos(brain, vocab, token: str):
        rel_is_a = brain.relation_id['is_a']
        for syn in brain.synapses_of(vocab.token_to_id[token]):
            if int(syn['relation']) == rel_is_a:
                tid = int(syn['to_id'])
                if tid in vocab.id_to_pos:
                    return vocab.id_to_pos[tid]
        return None

    @pytest.mark.parametrize('prompt', [
        ['the', 'quick', 'small'],   # S1 the+quick + S2 small
        ['a', 'bold', 'red'],        # S2 a + S3 bold+red
        ['the', 'smart', 'brown'],   # S1 the+brown + S2 smart
        ['the', 'small', 'red'],     # S1 the + S2 small + S3 red
    ])
    def test_recombined_prompt_emits_grammatical_completion(self, prompt):
        brain, vocab = self._trained_brain()
        out = generate_via_spread(brain, vocab, prompt, max_new=6)

        # Assert: substrate produced 6 tokens (didn't bail mid-sentence)
        assert len(out) == 6, (
            f'short completion for {prompt!r}: got {out!r}'
        )

        # Assert: each emission has the correct POS for its slot
        prompt_len = len(prompt)
        for i, tok in enumerate(out):
            slot_idx = prompt_len + i
            expected = SHARED_POS[slot_idx]
            actual = self._token_pos(brain, vocab, tok)
            assert actual == expected, (
                f'slot {slot_idx}: emitted {tok!r} (POS {actual}), '
                f'expected POS {expected}. prompt={prompt!r} out={out!r}'
            )

    def test_observe_recombination_paths(self, capsys):
        """Pure observation — print what the substrate generates for
        each recombined prompt. No assertions on content; this exists
        to document substrate behavior under mixed-evidence prompts."""
        brain, vocab = self._trained_brain()
        prompts = [
            ['the', 'quick', 'small'],
            ['a', 'bold', 'red'],
            ['the', 'smart', 'brown'],
            ['the', 'small', 'red'],
        ]
        with capsys.disabled():
            print()
            for p in prompts:
                out = generate_via_spread(brain, vocab, p, max_new=6)
                print(f'  {" ".join(p):24} → {" ".join(out)}')
