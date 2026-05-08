"""Tests for RL self-correction in the substrate-LM."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_GURU = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(_HERE))))
sys.path.insert(0, _GURU)

from brain.tasks.lm import build_lm_brain
from brain.tasks.lm.rl import (
    teach_minimal, trace_generate, reward_trajectory,
    apply_correction, train_rl, train_rl_curriculum,
    predict_sentence_id, GenerationStep, Trajectory,
)


CORPUS = [
    'the quick brown fox jumps over the lazy dog'.split(),
    'a smart small cat runs around a happy mouse'.split(),
    'the bold red bird flies past a tiny tree'.split(),
]
POS = ['DET', 'ADJ', 'ADJ', 'NOUN', 'VERB', 'PREP', 'DET', 'ADJ', 'NOUN']


def _seeded_brain():
    brain, vocab = build_lm_brain()
    for s in CORPUS:
        teach_minimal(brain, vocab, s, POS)
    return brain, vocab


# ─── teach_minimal preconditions ─────────────────────────────────────────

class TestTeachMinimal:
    def test_only_three_relation_types_used(self):
        """teach_minimal adds NO co_occurs, slot, or has_filler edges —
        the substrate must DISCOVER co_occurs via RL."""
        brain, vocab = _seeded_brain()
        from brain.tasks.lm.tiny import IS_A, FOLLOWS, HAS_MEMBER, CO_OCCURS, IN_SLOT, HAS_FILLER

        used = brain._used_synapses
        rel_to_count = {}
        for syn in brain.synapses[:used]:
            rel_id = int(syn['relation'])
            rel_name = brain.relations[rel_id][0]
            rel_to_count[rel_name] = rel_to_count.get(rel_name, 0) + 1

        # Only IS_A, HAS_MEMBER, FOLLOWS should be present
        assert IS_A in rel_to_count
        assert HAS_MEMBER in rel_to_count
        assert FOLLOWS in rel_to_count
        assert CO_OCCURS not in rel_to_count, \
            'teach_minimal must NOT add co_occurs — that is what RL learns'
        assert IN_SLOT not in rel_to_count, \
            'teach_minimal must NOT add in_slot — no slot anchoring'

    def test_no_slot_neurons_created(self):
        brain, vocab = _seeded_brain()
        assert len(vocab.slot_to_id) == 0, \
            'teach_minimal must not create slot neurons'


# ─── Trajectory tracing ──────────────────────────────────────────────────

class TestTraceGenerate:
    def test_returns_trajectory_with_step_per_emission(self):
        brain, vocab = _seeded_brain()
        traj = trace_generate(brain, vocab, CORPUS[0][:3],
                               max_new=4, pos_sequence=POS,
                               rng=np.random.default_rng(0),
                               explore_eps=0.0)
        # max_new=4 → up to 4 steps
        assert len(traj.steps) <= 4
        assert len(traj.output_tokens) == len(traj.steps)
        # Each step must have a valid emitted_id and non-empty activation
        for step in traj.steps:
            assert step.emitted_id in vocab.id_to_token
            assert len(step.activation) > 0

    def test_pos_sequence_overrides_substrate_prediction(self):
        """When pos_sequence is given, every emission must match the
        forced POS, regardless of what the substrate's POS-FOLLOWS edges
        would otherwise predict."""
        brain, vocab = _seeded_brain()
        # Force NOUN at every position (even when ungrammatical)
        forced = ['DET', 'NOUN', 'NOUN', 'NOUN', 'NOUN', 'NOUN', 'NOUN', 'NOUN', 'NOUN']
        traj = trace_generate(brain, vocab, ['the'],
                               max_new=4, pos_sequence=forced,
                               rng=np.random.default_rng(0),
                               explore_eps=0.0)
        from brain.tasks.lm.tiny import IS_A
        rel_is_a = brain.relation_id[IS_A]
        for tok in traj.output_tokens:
            tid = vocab.token_to_id[tok]
            pos_match = any(
                int(syn['to_id']) in vocab.id_to_pos
                and vocab.id_to_pos[int(syn['to_id'])] == 'NOUN'
                and int(syn['relation']) == rel_is_a
                for syn in brain.synapses_of(tid)
            )
            assert pos_match, f'{tok!r} forced to NOUN but is not a NOUN'


# ─── Reward + credit assignment ──────────────────────────────────────────

class TestReward:
    def test_correct_emission_gets_positive_reward(self):
        brain, vocab = _seeded_brain()
        # Hand-build a trajectory with one correct emission
        fox_id = vocab.token_to_id['fox']
        traj = Trajectory(
            prompt_ids=[],
            steps=[GenerationStep(emitted_id=fox_id)],
            output_tokens=['fox'],
        )
        rewards = reward_trajectory(traj, ['fox'])
        assert rewards == [1.0]

    def test_wrong_emission_gets_negative_reward(self):
        brain, vocab = _seeded_brain()
        cat_id = vocab.token_to_id['cat']
        traj = Trajectory(
            prompt_ids=[],
            steps=[GenerationStep(emitted_id=cat_id)],
            output_tokens=['cat'],
        )
        rewards = reward_trajectory(traj, ['fox'])
        assert rewards[0] < 0


class TestApplyCorrection:
    def test_positive_reward_grows_co_occurs_edges(self):
        """When RL rewards an emission, edges from recent context tokens
        to the emitted token must be created or strengthened."""
        brain, vocab = _seeded_brain()
        the_id = vocab.token_to_id['the']
        fox_id = vocab.token_to_id['fox']

        # Initially no co_occurs edge from 'the' to 'fox' — teach_minimal
        # didn't add one
        from brain.tasks.lm.tiny import CO_OCCURS
        rel_co = brain.relation_id[CO_OCCURS]

        def has_co_occurs(from_id, to_id):
            for syn in brain.synapses_of(from_id):
                if int(syn['to_id']) == to_id and int(syn['relation']) == rel_co:
                    return float(syn['weight'])
            return None

        assert has_co_occurs(the_id, fox_id) is None

        # Build a fake trajectory: emitted 'fox' with 'the' in recent context
        traj = Trajectory(
            prompt_ids=[the_id],
            steps=[GenerationStep(
                emitted_id=fox_id,
                activation={the_id: 0.6, fox_id: 1.4},
                next_pos_nid=vocab.pos_to_id['NOUN'],
                seed_ids=[the_id],
            )],
            output_tokens=['fox'],
        )
        stats = apply_correction(brain, vocab, traj, [1.0], eta=0.5)
        assert stats['created'] >= 1 or stats['updated'] >= 1
        # Now an edge should exist
        assert has_co_occurs(the_id, fox_id) is not None


# ─── End-to-end RL training ──────────────────────────────────────────────

class TestRLTraining:
    def test_rl_improves_accuracy_over_cold_start(self):
        """The defining test: starting from teach_minimal (no co_occurs
        seeds), RL training must measurably improve generation accuracy
        from cold-start to post-training."""
        brain, vocab = _seeded_brain()
        training_pairs = [(s[:3], s[3:], POS) for s in CORPUS]

        # Cold-start accuracy (greedy, no exploration)
        rng = np.random.default_rng(0)
        cold_correct = 0
        cold_total = 0
        for prompt, target, _ in training_pairs:
            traj = trace_generate(brain, vocab, prompt,
                                   max_new=len(target), rng=rng,
                                   explore_eps=0.0, pos_sequence=POS)
            cold_correct += sum(1 for a, b in zip(traj.output_tokens, target) if a == b)
            cold_total += len(target)
        cold_acc = cold_correct / cold_total

        # Train
        train_rl(brain, vocab, training_pairs, epochs=120,
                  eta=0.18, explore_eps_start=0.45, explore_eps_end=0.02,
                  seed=0)

        # Post-train accuracy
        post_correct = 0
        for prompt, target, _ in training_pairs:
            traj = trace_generate(brain, vocab, prompt,
                                   max_new=len(target), rng=rng,
                                   explore_eps=0.0, pos_sequence=POS)
            post_correct += sum(1 for a, b in zip(traj.output_tokens, target) if a == b)
        post_acc = post_correct / cold_total

        # RL must do meaningfully better than cold-start. Substrate is
        # self-correcting via reward — accuracy must climb.
        assert post_acc > cold_acc + 0.20, \
            f'RL training did not improve enough: cold={cold_acc:.1%} post={post_acc:.1%}'
        # And we should land in honest-progress territory (>= 60%)
        assert post_acc >= 0.60, \
            f'post-training accuracy too low: {post_acc:.1%}'

    def test_rl_grows_synapses_during_training(self):
        """The substrate's plasticity is real — synapse count must grow
        from initial seed during RL training."""
        brain, vocab = _seeded_brain()
        initial_synapses = brain._used_synapses

        training_pairs = [(s[:3], s[3:], POS) for s in CORPUS]
        train_rl(brain, vocab, training_pairs, epochs=50,
                  eta=0.15, explore_eps_start=0.40, explore_eps_end=0.05,
                  seed=0)

        final_synapses = brain._used_synapses
        # Substrate should have grown new edges from reward
        assert final_synapses > initial_synapses, \
            f'no edges grown: started {initial_synapses}, ended {final_synapses}'


# ─── Sentence-ID binding ─────────────────────────────────────────────────

class TestSentenceIdBinding:
    """Sentence-id neurons are per-sentence context anchors. Each
    training pair gets a unique neuron seeded as a goal during emission;
    RL grows (sentence_i → token) edges that route sentence-specifically.
    At inference, predict_sentence_id picks the right anchor from the
    prompt's co-activation pattern."""

    def test_sentence_neurons_created_when_enabled(self):
        brain, vocab = _seeded_brain()
        training_pairs = [(s[:3], s[3:], POS) for s in CORPUS]
        # Without use_sentence_id, no sentence neurons should exist
        assert len(vocab.sentence_to_id) == 0

        train_rl(brain, vocab, training_pairs, epochs=5,
                  eta=0.18, seed=0, use_sentence_id=True)
        # One sentence neuron per training pair
        assert len(vocab.sentence_to_id) == len(training_pairs)

    def test_predict_sentence_id_returns_known_sentence(self):
        """After training with sentence-id binding, predict_sentence_id
        should at least return SOME sentence neuron, not None."""
        brain, vocab = _seeded_brain()
        training_pairs = [(s[:3], s[3:], POS) for s in CORPUS]
        train_rl(brain, vocab, training_pairs, epochs=20,
                  eta=0.18, seed=0, use_sentence_id=True)

        # First sentence's prompt
        pred = predict_sentence_id(brain, vocab, CORPUS[0][:3])
        assert pred is not None
        assert pred in vocab.id_to_sentence

    def test_predict_sentence_id_returns_none_without_training(self):
        """Without sentence-id training, no sentence neurons exist —
        prediction should return None."""
        brain, vocab = _seeded_brain()
        pred = predict_sentence_id(brain, vocab, CORPUS[0][:3])
        assert pred is None


# ─── Curriculum learning ─────────────────────────────────────────────────

class TestCurriculum:
    """Curriculum learning: train on subsets first, then expand. The
    persistent replay buffer carries successful trajectories across
    stages so earlier sentences aren't forgotten when new ones are
    introduced."""

    def test_curriculum_stages_history(self):
        """History should be tagged with stage and stage_size."""
        brain, vocab = _seeded_brain()
        training_pairs = [(s[:3], s[3:], POS) for s in CORPUS]

        history = train_rl_curriculum(
            brain, vocab, training_pairs,
            stages=2, epochs_per_stage=10, seed=0,
        )
        # 2 stages × 10 epochs = 20 history entries
        assert len(history) == 20
        # Each entry has stage and stage_size
        for h in history:
            assert 'stage' in h
            assert 'stage_size' in h
        # Stage sizes should be monotonically non-decreasing
        sizes = [h['stage_size'] for h in history]
        assert all(sizes[i] <= sizes[i+1] for i in range(len(sizes)-1))
        # Final stage covers all training pairs
        assert history[-1]['stage_size'] == len(training_pairs)

    def test_curriculum_persistent_buffer_carries_across_stages(self):
        """Across stages, replay buffer keeps growing — earlier
        successful trajectories stay around to be replayed."""
        brain, vocab = _seeded_brain()
        training_pairs = [(s[:3], s[3:], POS) for s in CORPUS]

        history = train_rl_curriculum(
            brain, vocab, training_pairs,
            stages=2, epochs_per_stage=20,
            replay_min_reward=0.0,
            seed=0,
        )
        # In stage 2, buffer_size at end should be > buffer_size at start
        # of stage 1 (i.e. carried over)
        stage1_end = max(i for i, h in enumerate(history) if h['stage'] == 0)
        stage2_start_buf = history[stage1_end + 1]['buffer_size']
        stage1_end_buf = history[stage1_end]['buffer_size']

        # Persistent buffer: stage 2 starts where stage 1 left off
        assert stage2_start_buf >= stage1_end_buf - 1, (
            f'buffer not persisted: stage1_end={stage1_end_buf}, '
            f'stage2_start={stage2_start_buf}'
        )


# ─── Positional embeddings (step neurons) ────────────────────────────────

class TestPositionalEmbeddings:
    """Step neurons are the substrate's positional embeddings — discrete
    neurons (not continuous vectors) seeded as goals during generation
    so their co_occurs edges to position-correct tokens fire throughout
    the spread cycle."""

    def test_step_neurons_created_on_demand(self):
        """trace_generate must allocate step neurons up to max_new."""
        brain, vocab = _seeded_brain()
        # Initially no step neurons
        assert len(vocab.step_to_id) == 0

        rng = np.random.default_rng(0)
        trace_generate(brain, vocab, CORPUS[0][:3], max_new=6,
                        rng=rng, pos_sequence=POS, explore_eps=0.0)
        # 6 step neurons should have been created
        assert len(vocab.step_to_id) == 6
        # Each step has its own neuron
        assert set(vocab.step_to_id.keys()) == {0, 1, 2, 3, 4, 5}

    def test_step_neurons_grow_co_occurs_via_rl(self):
        """After RL training, step_i should have co_occurs edges to the
        tokens emitted at position i — that's the substrate learning a
        positional embedding from reward signal."""
        brain, vocab = _seeded_brain()
        from brain.tasks.lm.tiny import CO_OCCURS
        rel_co = brain.relation_id[CO_OCCURS]

        training_pairs = [(s[:3], s[3:], POS) for s in CORPUS]
        train_rl(brain, vocab, training_pairs, epochs=50,
                  eta=0.18, explore_eps_start=0.40, explore_eps_end=0.05,
                  seed=0)

        # step_0 (first emission after prompt = sentence position 3)
        # should have co_occurs to NOUNs (fox, cat, bird) — the tokens
        # that get emitted at that position across episodes
        step0_id = vocab.step_to_id[0]
        co_targets = set()
        for syn in brain.synapses_of(step0_id):
            if int(syn['relation']) == rel_co:
                tid = int(syn['to_id'])
                if tid in vocab.id_to_token:
                    co_targets.add(vocab.id_to_token[tid])

        # Substrate has GROWN at least one position-correct edge
        assert len(co_targets) >= 1, \
            f'step_0 has no co_occurs edges to tokens after RL: {co_targets}'
        # And the targets should be drawn from the position-3 NOUNs
        position_3_nouns = {'fox', 'cat', 'bird'}
        assert co_targets & position_3_nouns, \
            f'step_0 connected to wrong tokens: {co_targets} (expected ⊆ {position_3_nouns})'
