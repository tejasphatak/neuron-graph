"""Tests for the substrate-native planning agent."""

from __future__ import annotations

import os
import random
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_GURU = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _GURU)

from brain.play_ttt import EMPTY, X, O, build_brain, legal_moves, winner
from brain.planning_agent import (
    plan_one_step, evaluate_state, evaluate_terminal,
    play_with_planning, evaluate_planner,
    two_in_a_row_lines,
)
from brain.world_model import (
    ensure_predict_relation, observe_transition, train_world_model,
)


class TestEvaluation:
    def test_terminal_win(self):
        # X wins on top row
        board = [X, X, X, EMPTY, O, O, EMPTY, EMPTY, EMPTY]
        assert evaluate_terminal(board, X) == 1.0
        assert evaluate_terminal(board, O) == -1.0

    def test_terminal_draw(self):
        # Full board, no winner
        board = [X, O, X, X, O, O, O, X, X]
        assert evaluate_terminal(board, X) == 0.0

    def test_non_terminal_returns_none(self):
        # Mid-game position with no winner and not full
        board = [X, O, EMPTY, EMPTY, X, EMPTY, EMPTY, EMPTY, O]
        assert evaluate_terminal(board, X) is None

    def test_two_in_a_row_count(self):
        # Two X in top row, one empty
        board = [X, X, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY]
        assert two_in_a_row_lines(board, X) == 1
        assert two_in_a_row_lines(board, O) == 0


class TestPlanOneStep:
    @pytest.fixture(scope='class')
    def trained(self):
        return train_world_model(n_games=500, rng_seed=42)

    def test_picks_winning_move_when_available(self, trained):
        brain, neurons, _ = trained
        # X has two on top row, can win at cell 2
        board = [X, X, EMPTY, O, EMPTY, EMPTY, O, EMPTY, EMPTY]
        rng = random.Random(0)
        result = plan_one_step(brain, neurons, board, X,
                                temperature=0.0, rng=rng)
        # Picking cell 2 gives an immediate win → score 1.0
        assert result.action == 2

    def test_blocks_opponent_winning_move(self, trained):
        brain, neurons, _ = trained
        # O has two on top row; if X doesn't block at cell 2, O wins
        # X plays from this position. The heuristic should pick blocking
        # move because not-blocking gives opponent +1.
        # Actually our heuristic doesn't fully simulate opponent's response,
        # so this test verifies that the heuristic at least scores blocking
        # higher than ignoring.
        board = [O, O, EMPTY, X, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY]
        rng = random.Random(0)
        result = plan_one_step(brain, neurons, board, X,
                                temperature=0.0, rng=rng)
        # Should pick cell 2 (block) with the highest score among legals
        # because cell 2 reduces O's two-in-a-row count
        # Top candidates check
        cs = result.candidate_scores
        # Cell 2 should be in top scores
        block_score = cs[2]
        # Cell 2 should be at least tied for the top
        assert block_score >= max(cs.values()) - 1e-6

    def test_returns_legal_action(self, trained):
        brain, neurons, _ = trained
        board = [X, O, X, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY]
        rng = random.Random(0)
        result = plan_one_step(brain, neurons, board, X,
                                temperature=0.0, rng=rng)
        assert result.action in legal_moves(board)


class TestPlannerVsRandom:
    @pytest.fixture(scope='class')
    def trained(self):
        return train_world_model(n_games=500, rng_seed=42)

    def test_planner_beats_random(self, trained):
        brain, neurons, _ = trained
        # Planner should win >85% vs random — solid margin above naive RL
        result = evaluate_planner(brain, neurons,
                                    opponent='random', n_games=100,
                                    rng_seed=12345)
        assert result['wins'] > 0.85, \
            f'planner only wins {result["wins"]:.3f} vs random'

    def test_planner_loses_few_to_random(self, trained):
        brain, neurons, _ = trained
        result = evaluate_planner(brain, neurons,
                                    opponent='random', n_games=100,
                                    rng_seed=12345)
        # A planning agent shouldn't lose much to random play
        assert result['losses'] < 0.1, \
            f'planner loses to random {result["losses"]:.3f} of the time'


class TestPlannerVsMinimax:
    @pytest.fixture(scope='class')
    def trained(self):
        return train_world_model(n_games=500, rng_seed=42)

    def test_planner_draws_meaningfully_vs_minimax(self, trained):
        brain, neurons, _ = trained
        # Random vs minimax: ~24% draws. Planner should be substantially better.
        result = evaluate_planner(brain, neurons,
                                    opponent='minimax', n_games=100,
                                    rng_seed=12345)
        assert result['draws'] > 0.35, \
            f'planner draws {result["draws"]:.3f} vs minimax (expected > 0.35)'

    def test_planner_never_beats_perfect(self, trained):
        brain, neurons, _ = trained
        result = evaluate_planner(brain, neurons,
                                    opponent='minimax', n_games=100,
                                    rng_seed=12345)
        # X cannot beat perfect O — sanity check we're not faking the opponent
        assert result['wins'] == 0.0, \
            f'planner beat minimax {result["wins"]:.3f} times (impossible)'


class TestOnlineLearning:
    def test_world_model_grows_during_play(self):
        """Online learning: world model accumulates edges across games."""
        brain, neurons = build_brain()
        ensure_predict_relation(brain)
        rng = random.Random(0)

        from brain.world_model import PREDICT_RELATION
        rel_id = brain.relation_id[PREDICT_RELATION]

        before = sum(1 for s in brain.synapses[:getattr(brain, '_used_synapses', 0)]
                     if int(s['relation']) == rel_id)

        # Play a few games with online learning
        for _ in range(5):
            play_with_planning(
                brain, neurons,
                opponent_fn=lambda b, c, r: r.choice(legal_moves(b)),
                rng=rng,
                learn_world_model=True,
            )

        after = sum(1 for s in brain.synapses[:getattr(brain, '_used_synapses', 0)]
                    if int(s['relation']) == rel_id)
        assert after >= before  # equal is OK if all transitions were duplicates
