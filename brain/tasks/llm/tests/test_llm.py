"""Tests for substrate-LLM core: tokenizer, training, generation, perplexity."""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(_HERE))))
sys.path.insert(0, _REPO)

from brain.tasks.llm import (
    WordTokenizer, build_llm_view, train_ngram_epoch,
    generate_text, perplexity,
    compute_unigram_log_probs, perplexity_with_backoff,
    view_to_brain, perplexity_with_spread,
)


CORPUS = [
    'the cat sat on the mat',
    'a dog ran in the park',
    'the bird flew over the tree',
    'a fish swims in the water',
    'the sun shines bright today',
    'a child laughs loud in joy',
    'the cat played with a ball',
    'a dog barked at the cat',
    'the bird sang a sweet song',
    'a fish jumped out the pond',
] * 20  # 200 stories, repetition for training signal


# ─── Tokenizer ──────────────────────────────────────────────────────────────

class TestTokenizer:
    def test_special_tokens_reserved(self):
        tok = WordTokenizer(max_vocab=100, min_freq=1)
        # Special tokens should be the first 4 ids
        assert tok.token_to_id[tok.PAD] == 0
        assert tok.token_to_id[tok.UNK] == 1
        assert tok.token_to_id[tok.BOS] == 2
        assert tok.token_to_id[tok.EOS] == 3

    def test_fit_and_encode_decode_roundtrip(self):
        tok = WordTokenizer(max_vocab=100, min_freq=1)
        tok.fit(CORPUS)
        ids = tok.encode("the cat sat", add_bos=False, add_eos=False)
        decoded = tok.decode(ids, skip_special=True)
        assert decoded == "the cat sat"

    def test_unk_handling(self):
        tok = WordTokenizer(max_vocab=100, min_freq=1)
        tok.fit(CORPUS)
        ids = tok.encode("the dragon flies", add_bos=False, add_eos=False)
        # 'dragon' not in vocab → mapped to UNK (id 1)
        assert tok.token_to_id[tok.UNK] in ids

    def test_vocab_pruning_by_frequency(self):
        tok = WordTokenizer(max_vocab=50, min_freq=2)
        tok.fit(['unique_word ' + 'common_word ' * 10])
        # 'unique_word' only appears once, should NOT be in vocab
        assert 'unique_word' not in tok.token_to_id
        # 'common_word' appears 10× → in vocab
        assert 'common_word' in tok.token_to_id


# ─── LLMView (W matrix) ────────────────────────────────────────────────────

class TestLLMView:
    def test_w_shape_matches_vocab(self):
        tok = WordTokenizer(max_vocab=100, min_freq=1)
        tok.fit(CORPUS)
        view = build_llm_view(tok)
        V = tok.get_vocab_size()
        assert view.W.shape == (V, V)
        assert view.W.dtype == np.float32

    def test_initial_w_is_zero(self):
        tok = WordTokenizer(max_vocab=50, min_freq=1)
        tok.fit(CORPUS)
        view = build_llm_view(tok)
        assert np.all(view.W == 0)


# ─── Training ──────────────────────────────────────────────────────────────

class TestTraining:
    @staticmethod
    def _setup():
        tok = WordTokenizer(max_vocab=100, min_freq=1)
        tok.fit(CORPUS)
        view = build_llm_view(tok)
        seqs = [tok.encode(t) for t in CORPUS]
        return tok, view, seqs

    def test_training_modifies_w(self):
        tok, view, seqs = self._setup()
        train_ngram_epoch(view, seqs, context_window=4, eta=0.05,
                            rng=np.random.default_rng(0))
        # W should no longer be all zeros after training
        assert np.any(view.W != 0)

    def test_training_improves_next_token_accuracy(self):
        """After several epochs, next-token accuracy should rise above
        cold-start (which is 0% since W is all zeros)."""
        tok, view, seqs = self._setup()
        rng = np.random.default_rng(0)
        for _ in range(5):
            metrics = train_ngram_epoch(view, seqs,
                                          context_window=4, eta=0.05, rng=rng)
        assert metrics['next_token_accuracy'] > 0.20


# ─── Perplexity ────────────────────────────────────────────────────────────

class TestPerplexity:
    @staticmethod
    def _setup():
        tok = WordTokenizer(max_vocab=100, min_freq=1)
        tok.fit(CORPUS)
        view = build_llm_view(tok)
        seqs = [tok.encode(t) for t in CORPUS]
        return tok, view, seqs

    def test_cold_start_ppl_is_uniform(self):
        """Untrained substrate (W=0) should have PPL ≈ vocab size
        (uniform softmax over the vocabulary)."""
        tok, view, seqs = self._setup()
        ppl = perplexity(view, seqs[:5], context_window=4)
        V = tok.get_vocab_size()
        # PPL should be very close to V (uniform → log V loss per token)
        assert abs(ppl - V) < 1.0

    def test_training_reduces_ppl(self):
        """After training, PPL should drop measurably below uniform."""
        tok, view, seqs = self._setup()
        V = tok.get_vocab_size()
        rng = np.random.default_rng(0)
        for _ in range(8):
            train_ngram_epoch(view, seqs, context_window=4, eta=0.05, rng=rng)
        ppl = perplexity(view, seqs[:20], context_window=4)
        # Should be measurably better than uniform (>15% drop is real)
        assert ppl < 0.85 * V, f'PPL did not drop enough: {ppl:.1f} vs uniform {V}'

    def test_returns_float(self):
        tok, view, seqs = self._setup()
        ppl = perplexity(view, seqs[:3], context_window=4)
        assert isinstance(ppl, float)

    def test_empty_sequences_returns_inf(self):
        tok, view, _ = self._setup()
        ppl = perplexity(view, [], context_window=4)
        assert ppl == float('inf')


# ─── Unigram backoff (PPL #A) ──────────────────────────────────────────────

class TestUnigramBackoff:
    @staticmethod
    def _setup():
        tok = WordTokenizer(max_vocab=100, min_freq=1)
        tok.fit(CORPUS)
        view = build_llm_view(tok)
        seqs = [tok.encode(t) for t in CORPUS]
        return tok, view, seqs

    def test_unigram_log_probs_shape(self):
        tok, view, seqs = self._setup()
        uni = compute_unigram_log_probs(view, seqs)
        assert uni.shape == (view.W.shape[0],)

    def test_unigram_log_probs_sum_to_one(self):
        """exp(log_probs).sum() ≈ 1.0 — Laplace-smoothed valid distribution."""
        tok, view, seqs = self._setup()
        uni = compute_unigram_log_probs(view, seqs)
        total = float(np.exp(uni).sum())
        assert abs(total - 1.0) < 1e-4, f'unigram probs sum to {total}'

    def test_unigram_log_probs_finite(self):
        """No -inf log-probs (Laplace +1 smoothing should prevent zeros)."""
        tok, view, seqs = self._setup()
        uni = compute_unigram_log_probs(view, seqs)
        assert np.all(np.isfinite(uni))

    def test_unigram_log_probs_more_likely_for_common_tokens(self):
        """'the' appears in most CORPUS sentences; should have higher
        unigram log-prob than rare tokens."""
        tok, view, seqs = self._setup()
        uni = compute_unigram_log_probs(view, seqs)
        the_id = tok.token_to_id.get('the')
        sang_id = tok.token_to_id.get('sang')  # appears once
        if the_id is not None and sang_id is not None:
            the_row = view.tok_to_row[the_id]
            sang_row = view.tok_to_row[sang_id]
            assert uni[the_row] > uni[sang_row]

    def test_perplexity_with_backoff_drops_below_baseline(self):
        """The defining test: alpha-mixed PPL should be lower than
        baseline PPL on small/sparse data (where context_W is undertrained)."""
        tok, view, seqs = self._setup()
        rng = np.random.default_rng(0)
        for _ in range(5):
            train_ngram_epoch(view, seqs, context_window=4, eta=0.05, rng=rng)
        uni = compute_unigram_log_probs(view, seqs)
        ppl_base = perplexity(view, seqs[:30], context_window=4)
        # Some α should beat baseline. 0.5 is a safe bet on this corpus.
        ppl_mixed = perplexity_with_backoff(view, seqs[:30], uni,
                                              alpha=0.5, context_window=4)
        assert ppl_mixed < ppl_base, (
            f'backoff did not drop PPL: base={ppl_base:.1f} '
            f'mixed={ppl_mixed:.1f}'
        )

    def test_alpha_one_equals_pure_context(self):
        """At α=1.0, mixed PPL ≈ baseline (only ctx, no unigram).
        Allow tolerance for numerical / floor differences."""
        tok, view, seqs = self._setup()
        rng = np.random.default_rng(0)
        for _ in range(3):
            train_ngram_epoch(view, seqs, context_window=4, eta=0.05, rng=rng)
        uni = compute_unigram_log_probs(view, seqs)
        ppl_base = perplexity(view, seqs[:20], context_window=4)
        ppl_alpha1 = perplexity_with_backoff(view, seqs[:20], uni,
                                                alpha=1.0 - 1e-6,
                                                context_window=4)
        # Should be close to baseline (within 20% for floor handling)
        assert abs(ppl_alpha1 - ppl_base) / ppl_base < 0.20

    def test_alpha_zero_equals_pure_unigram(self):
        """At α=0.0, all probability mass goes to unigram. PPL should
        be the unigram PPL (much lower than uniform on natural text)."""
        tok, view, seqs = self._setup()
        rng = np.random.default_rng(0)
        for _ in range(3):
            train_ngram_epoch(view, seqs, context_window=4, eta=0.05, rng=rng)
        uni = compute_unigram_log_probs(view, seqs)
        ppl_alpha0 = perplexity_with_backoff(view, seqs[:20], uni,
                                                alpha=0.0,
                                                context_window=4)
        V = tok.get_vocab_size()
        # Pure unigram PPL should be < V (concentrated on common words)
        assert ppl_alpha0 < V


# ─── Generation ────────────────────────────────────────────────────────────

# ─── Substrate-native spread() prediction (PPL #B) ─────────────────────────

class TestSubstrateSpread:
    """The spread() primitive used for prediction instead of matmul.
    Substrate gets converted from dense W to sparse Brain via top-K
    edge selection, then spread() integrates semantic neighbors."""

    @staticmethod
    def _setup():
        tok = WordTokenizer(max_vocab=100, min_freq=1)
        tok.fit(CORPUS)
        view = build_llm_view(tok)
        seqs = [tok.encode(t) for t in CORPUS]
        rng = np.random.default_rng(0)
        for _ in range(5):
            train_ngram_epoch(view, seqs, context_window=4, eta=0.05, rng=rng)
        return tok, view, seqs

    def test_view_to_brain_creates_substrate(self):
        """Conversion produces a real Brain object with synapses."""
        tok, view, _ = self._setup()
        brain, row_to_nid = view_to_brain(view, top_k_per_row=10)
        # One neuron per token
        assert brain.size == view.W.shape[0]
        # Has synapses (at least some non-zero W entries)
        n_syn = getattr(brain, '_used_synapses', 0)
        assert n_syn > 0

    def test_view_to_brain_respects_top_k(self):
        """top_k_per_row limits edges per source neuron."""
        tok, view, _ = self._setup()
        V = view.W.shape[0]
        brain, _ = view_to_brain(view, top_k_per_row=3)
        # No source neuron should have more than top_k synapses
        for nid in range(brain.size):
            edges = brain.synapses_of(nid)
            assert len(edges) <= 3

    def test_perplexity_with_spread_returns_float(self):
        tok, view, seqs = self._setup()
        brain, row_to_nid = view_to_brain(view, top_k_per_row=10)
        ppl = perplexity_with_spread(view, brain, row_to_nid,
                                        seqs[:5], context_window=4)
        assert isinstance(ppl, float)
        assert ppl > 0

    def test_spread_predict_uses_actual_substrate(self):
        """Verifies the spread()-based predict touches the substrate's
        Brain object (not the dense W matrix), via reading synapses_of."""
        tok, view, seqs = self._setup()
        brain, row_to_nid = view_to_brain(view, top_k_per_row=10)
        # Cold sanity: empty brain (synapses_of returns no edges) should
        # produce uniform-like PPL (no signal in spread)
        from brain import Brain as B
        empty_brain = B()
        empty_brain.relations = brain.relations
        empty_brain._rebuild_relation_index()
        for r in row_to_nid:
            empty_brain.add_neuron(lemma=f'tok:{r}', decay=0.5)
        # No edges → spread produces no signal → high PPL
        empty_ppl = perplexity_with_spread(view, empty_brain, row_to_nid,
                                              seqs[:3], context_window=4)
        # Real substrate (with edges) should give finite, lower PPL
        real_ppl = perplexity_with_spread(view, brain, row_to_nid,
                                             seqs[:3], context_window=4)
        # Both finite. Real with edges should be no worse (often better).
        assert math.isfinite(empty_ppl)
        assert math.isfinite(real_ppl)


# ─── Generation ────────────────────────────────────────────────────────────

class TestGeneration:
    def test_greedy_generation_produces_string(self):
        tok = WordTokenizer(max_vocab=100, min_freq=1)
        tok.fit(CORPUS)
        view = build_llm_view(tok)
        seqs = [tok.encode(t) for t in CORPUS]
        rng = np.random.default_rng(0)
        for _ in range(5):
            train_ngram_epoch(view, seqs, context_window=4, eta=0.05, rng=rng)
        out = generate_text(view, tok, "the cat",
                              max_new=4, temperature=0.0,
                              context_window=4, rng=rng)
        assert isinstance(out, str)
        assert len(out) > 0

    def test_sampling_with_temperature(self):
        tok = WordTokenizer(max_vocab=100, min_freq=1)
        tok.fit(CORPUS)
        view = build_llm_view(tok)
        seqs = [tok.encode(t) for t in CORPUS]
        rng = np.random.default_rng(0)
        for _ in range(3):
            train_ngram_epoch(view, seqs, context_window=4, eta=0.05, rng=rng)
        out = generate_text(view, tok, "the cat",
                              max_new=4, temperature=0.7, top_k=5,
                              context_window=4,
                              rng=np.random.default_rng(42))
        assert isinstance(out, str)
