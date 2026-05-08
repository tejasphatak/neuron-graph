"""Tests for the substrate-learned value head."""

from __future__ import annotations

import os
import random
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_GURU = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _GURU)

from brain.play_ttt import EMPTY, X, O, build_brain, legal_moves
from brain.value_head import (
    ensure_value_relation, ensure_outcome_neurons,
    credit_value_trajectory, substrate_value,
    substrate_eval_with_lookahead, train_value_head_from_games,
    VALUE_RELATION,
)


class TestSetup:
    def test_value_relation_registered(self):
        brain, _ = build_brain()
        assert VALUE_RELATION not in brain.relation_id
        ensure_value_relation(brain)
        assert VALUE_RELATION in brain.relation_id

    def test_outcome_neurons_idempotent(self):
        brain, _ = build_brain()
        o1 = ensure_outcome_neurons(brain)
        o2 = ensure_outcome_neurons(brain)
        # Same neuron IDs (idempotent)
        assert o1.win == o2.win
        assert o1.lose == o2.lose
        assert o1.draw == o2.draw
        # Three distinct neurons
        assert len({o1.win, o1.lose, o1.draw}) == 3


class TestValueQueries:
    @pytest.fixture(scope='class')
    def trained(self):
        brain, neurons = build_brain()
        stats = train_value_head_from_games(brain, neurons,
                                              n_games=500, rng_seed=11)
        return brain, neurons, stats['outcomes']

    def test_terminal_win_returns_one(self, trained):
        brain, neurons, outcomes = trained
        # X wins on top row
        board = [X, X, X, EMPTY, O, O, EMPTY, EMPTY, EMPTY]
        assert substrate_value(brain, neurons, outcomes, board, X) == 1.0

    def test_terminal_loss_returns_minus_one(self, trained):
        brain, neurons, outcomes = trained
        board = [X, X, X, EMPTY, O, O, EMPTY, EMPTY, EMPTY]
        assert substrate_value(brain, neurons, outcomes, board, O) == -1.0

    def test_full_draw_returns_zero(self, trained):
        brain, neurons, outcomes = trained
        board = [X, O, X, X, O, O, O, X, X]
        assert substrate_value(brain, neurons, outcomes, board, X) == 0.0

    def test_value_in_range(self, trained):
        brain, neurons, outcomes = trained
        board = [X, EMPTY, EMPTY, EMPTY, O, EMPTY, EMPTY, EMPTY, EMPTY]
        v = substrate_value(brain, neurons, outcomes, board, X)
        assert -1.0 <= v <= 1.0


class TestLookaheadCorrection:
    """The lookahead wrapper catches cases the raw value misses (terminal-
    adjacent states where opponent wins next turn)."""

    @pytest.fixture(scope='class')
    def trained(self):
        brain, neurons = build_brain()
        stats = train_value_head_from_games(brain, neurons,
                                              n_games=500, rng_seed=11)
        return brain, neurons, stats['outcomes']

    def test_opponent_winning_threat_gets_negative_score(self, trained):
        brain, neurons, outcomes = trained
        # O has two on top row, can win at cell 2; X to move
        board = [O, O, EMPTY, X, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY]
        v = substrate_eval_with_lookahead(brain, neurons, outcomes, board, X)
        assert v < 0, f'expected negative score for opp-near-win, got {v:.3f}'

    def test_my_winning_move_gets_positive_score(self, trained):
        brain, neurons, outcomes = trained
        # X has two on top row, can win at cell 2
        board = [X, X, EMPTY, EMPTY, O, EMPTY, EMPTY, EMPTY, EMPTY]
        v = substrate_eval_with_lookahead(brain, neurons, outcomes, board, X)
        assert v > 0.5, f'expected high score for my-near-win, got {v:.3f}'


class TestTraining:
    def test_value_edges_grow_with_training(self):
        brain, neurons = build_brain()
        rel_id = ensure_value_relation(brain)
        ensure_outcome_neurons(brain)

        before = sum(1 for s in brain.synapses[:getattr(brain, '_used_synapses', 0)]
                     if int(s['relation']) == rel_id)

        train_value_head_from_games(brain, neurons,
                                       n_games=100, rng_seed=42)

        after = sum(1 for s in brain.synapses[:getattr(brain, '_used_synapses', 0)]
                    if int(s['relation']) == rel_id)
        assert after > before

    def test_compact_value_graph(self):
        """81 max edges (27 cells × 3 outcomes); training shouldn't blow up."""
        brain, neurons = build_brain()
        train_value_head_from_games(brain, neurons,
                                       n_games=2000, rng_seed=42)
        rel_id = brain.relation_id[VALUE_RELATION]
        n_value = sum(1 for s in brain.synapses[:getattr(brain, '_used_synapses', 0)]
                       if int(s['relation']) == rel_id)
        assert n_value <= 200, f'value graph blew up to {n_value} edges'
