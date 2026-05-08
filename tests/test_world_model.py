"""Tests for the substrate-native world model (forward edges)."""

from __future__ import annotations

import os
import random
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_GURU = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _GURU)

from brain.play_ttt import (
    EMPTY, X, O, build_brain, encode_state,
    legal_moves, winner,
)
from brain.world_model import (
    ensure_predict_relation, observe_transition, predict_next,
    train_world_model, evaluate, collect_random_transitions,
    PREDICT_RELATION,
)


class TestPredictRelation:
    def test_relation_registered_on_brain(self):
        brain, _ = build_brain()
        assert PREDICT_RELATION not in brain.relation_id
        ensure_predict_relation(brain)
        assert PREDICT_RELATION in brain.relation_id

    def test_idempotent(self):
        brain, _ = build_brain()
        ensure_predict_relation(brain)
        rel_id = brain.relation_id[PREDICT_RELATION]
        ensure_predict_relation(brain)
        assert brain.relation_id[PREDICT_RELATION] == rel_id


class TestObserveTransition:
    def test_creates_edges(self):
        brain, neurons = build_brain()
        ensure_predict_relation(brain)
        before = getattr(brain, '_used_synapses', 0)
        board_pre = [EMPTY] * 9
        board_post = list(board_pre)
        board_post[4] = X
        observe_transition(brain, neurons, board_pre, 4, board_post)
        after = getattr(brain, '_used_synapses', 0)
        assert after > before

    def test_no_unchanged_means_no_stability(self):
        """Sanity: a transition where every cell changes still produces edges."""
        brain, neurons = build_brain()
        board_pre = [EMPTY] * 9
        board_post = [X] * 9   # impossible in TTT but valid as a probe
        # Should not crash; produces edges for all 9 changed cells
        result = observe_transition(brain, neurons, board_pre, 0, board_post)
        assert len(result['changed_cells']) == 9


class TestPredictNext:
    def test_returns_full_distribution(self):
        brain, neurons = build_brain()
        # Train on a single transition
        board_pre = [EMPTY] * 9
        board_post = list(board_pre)
        board_post[4] = X
        observe_transition(brain, neurons, board_pre, 4, board_post)

        pred = predict_next(brain, neurons, board_pre, 4)
        assert len(pred.cell_distributions) == 9
        assert len(pred.predicted_board) == 9
        # Each cell distribution has all three possible values
        for dist in pred.cell_distributions:
            assert set(dist.keys()) == {EMPTY, X, O}

    def test_single_transition_predicts_itself(self):
        """After observing exactly one transition, predicting the same
        (board, action) should give the same post-state for the changed cell."""
        brain, neurons = build_brain()
        board_pre = [EMPTY] * 9
        board_post = list(board_pre)
        board_post[4] = X
        # Observe several times to cement the signal
        for _ in range(5):
            observe_transition(brain, neurons, board_pre, 4, board_post)

        pred = predict_next(brain, neurons, board_pre, 4)
        # The played cell should be predicted as X
        assert pred.predicted_board[4] == X


class TestTrainedWorldModel:
    @pytest.fixture(scope='class')
    def trained(self):
        return train_world_model(n_games=200, rng_seed=11)

    def test_compact_edge_set(self, trained):
        brain, _, transitions = trained
        # Edges should be much fewer than the # of training transitions —
        # the rule is causal, so most observations strengthen existing edges
        # rather than create new ones.
        rel_id = brain.relation_id[PREDICT_RELATION]
        used = getattr(brain, '_used_synapses', 0)
        n_predict_edges = sum(
            1 for s in brain.synapses[:used]
            if int(s['relation']) == rel_id
        )
        # 99 distinct (action, post-cell) + (local-pre, post-cell) +
        # (stability) tuples — substrate stays compact regardless of training set
        assert n_predict_edges < 200, \
            f'edge count {n_predict_edges} is suspiciously high'

    def test_high_cell_accuracy(self, trained):
        brain, neurons, _ = trained
        rng = random.Random(99)
        test_set = collect_random_transitions(50, rng)
        stats = evaluate(brain, neurons, test_set)
        assert stats.cell_accuracy > 0.85, \
            f'cell accuracy {stats.cell_accuracy:.3f} below 0.85'

    def test_full_board_better_than_chance(self, trained):
        brain, neurons, _ = trained
        rng = random.Random(99)
        test_set = collect_random_transitions(50, rng)
        stats = evaluate(brain, neurons, test_set)
        # 33% cell-accuracy chance → 0.33^9 ≈ 0.000005 full-board chance
        # Anything above 10% full-board is substantially above chance
        assert stats.full_board_accuracy > 0.10, \
            f'full-board accuracy {stats.full_board_accuracy:.3f} no better than chance'
