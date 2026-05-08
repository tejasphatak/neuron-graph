"""Tic-Tac-Toe — substrate's first proving ground.

Public API:
    game            — board, win check, opponents, neuron layout
    bandit          — 2-arm bandit RL test (substrate's simplest agent)
    world_model     — `predicts` relation, observe_transition, predict_next
    value_head      — `value` relation, outcome neurons, substrate_value
    planner         — one-step lookahead via world model
    curriculum      — substrate vs minimax (perfect teacher)
    shared_brain    — color-symmetric self-play (limits documented)
    mini_seed       — 50-neuron mini-brain (for sanity demos / probes)
"""

from .game import (
    EMPTY, X, O, WIN_LINES, winner, is_full, legal_moves, render_board,
    build_brain, encode_state, agent_pick_action, credit_trajectory,
    TTTNeurons, GameOutcome,
    play_one_game, train, baseline_random_vs_random,
    play_self_play_game, train_self_play,
)

__all__ = [
    'EMPTY', 'X', 'O', 'WIN_LINES',
    'winner', 'is_full', 'legal_moves', 'render_board',
    'build_brain', 'encode_state', 'agent_pick_action', 'credit_trajectory',
    'TTTNeurons', 'GameOutcome',
    'play_one_game', 'train', 'baseline_random_vs_random',
    'play_self_play_game', 'train_self_play',
]
