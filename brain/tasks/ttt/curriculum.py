"""Curriculum: substrate vs minimax (perfect player), then self-play.

Two phases:
  Phase 1 — Substrate plays X against MINIMAX (perfect O). Best-case for X
            vs perfect O is a DRAW (TTT is solved as drawn). Substrate
            should learn to draw consistently — losing if it doesn't,
            drawing if it does.

  Phase 2 — Clone the trained substrate into two copies. Both already
            know how to play. Pit them against each other (each updates
            its own weights). With both starting from a competent
            baseline, self-play should converge near drawing equilibrium,
            and slight divergences are real exploration.

Why curriculum? Random-opponent training (already done) lets X exploit
random's mistakes — substrate learns "win against bad players" rather
than "play well." Minimax forces actual competence: substrate cannot
win, can only learn to draw. After that signal, self-play between two
trained copies has a meaningful starting point.
"""

from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from brain.neuron import SYNAPSE_DTYPE
from .game import (
    EMPTY, X, O, WIN_LINES,
    winner, is_full, legal_moves, render_board,
    build_brain, encode_state, agent_pick_action, credit_trajectory,
    TTTNeurons, GameOutcome,
)
from brain.store import Brain
from brain.spread import spread


# ─── Minimax (perfect player) ─────────────────────────────────────────────

_MINIMAX_CACHE: Dict[Tuple[Tuple[int, ...], int], Tuple[List[int], int]] = {}


def _minimax(board: List[int], player: int) -> Tuple[List[int], int]:
    """Return (list of optimal moves, score from X's perspective).
    Memoized. Returns ALL optimal moves when there are ties — the caller
    can randomize among them so the teacher isn't deterministic.

    Score: +1 = X wins, 0 = draw, -1 = O wins.
    """
    key = (tuple(board), player)
    if key in _MINIMAX_CACHE:
        return _MINIMAX_CACHE[key]

    w = winner(board)
    if w == X:
        result = ([], 1)
    elif w == O:
        result = ([], -1)
    elif is_full(board):
        result = ([], 0)
    else:
        legal = legal_moves(board)
        best_score = -2 if player == X else 2
        best_moves: List[int] = []
        for mv in legal:
            board[mv] = player
            other = O if player == X else X
            _, s = _minimax(board, other)
            board[mv] = EMPTY
            if player == X:
                if s > best_score:
                    best_score = s
                    best_moves = [mv]
                elif s == best_score:
                    best_moves.append(mv)
            else:
                if s < best_score:
                    best_score = s
                    best_moves = [mv]
                elif s == best_score:
                    best_moves.append(mv)
        result = (best_moves, best_score)

    _MINIMAX_CACHE[key] = result
    return result


def minimax_pick(board: List[int], player: int,
                  rng: random.Random) -> int:
    """Pick a uniformly-random optimal move (tie-break randomized)."""
    moves, _ = _minimax(list(board), player)
    return rng.choice(moves) if moves else -1


# ─── Generalized game runner ──────────────────────────────────────────────

OpponentFn = Callable[[List[int], int, random.Random], int]


def _random_pick(board: List[int], player: int, rng: random.Random) -> int:
    legal = legal_moves(board)
    return rng.choice(legal) if legal else -1


def play_substrate_vs_opponent(
    brain: Brain, neurons: TTTNeurons,
    opponent_fn: OpponentFn,
    *, substrate_color: int = X,
    temperature: float, rng: random.Random,
) -> GameOutcome:
    """Substrate plays one color, opponent_fn plays the other.
    Trajectory accumulates only the substrate's moves."""
    board = [EMPTY] * 9
    trajectory: List[Tuple[List[int], int]] = []

    for turn in range(9):
        whose_turn = X if turn % 2 == 0 else O
        if whose_turn == substrate_color:
            action = agent_pick_action(brain, neurons, board,
                                         temperature=temperature, rng=rng)
            if action < 0:
                break
            trajectory.append((list(board), action))
            board[action] = whose_turn
        else:
            action = opponent_fn(board, whose_turn, rng)
            if action < 0:
                break
            board[action] = whose_turn

        w = winner(board)
        if w != EMPTY or is_full(board):
            return GameOutcome(result=w, moves=turn + 1, trajectory=trajectory)
    return GameOutcome(result=winner(board), moves=9, trajectory=trajectory)


# ─── Phase 1: Train vs minimax ────────────────────────────────────────────

def train_vs_minimax(
    n_games: int = 3000, *, eval_every: int = 250,
    rng_seed: int = 7777,
) -> Dict:
    """Substrate plays X against minimax-O. Best possible result for X is draw.
    Substrate succeeds when its loss-rate drops to ~0 and draw-rate rises."""
    rng = random.Random(rng_seed)
    brain, neurons = build_brain()

    outcomes: List[int] = []
    eval_records: List[Dict] = []

    def opponent(board, player, r):
        return minimax_pick(board, player, r)

    for g in range(n_games):
        progress = g / max(1, n_games - 1)
        temp = 1.5 * (1.0 - progress) + 0.05

        out = play_substrate_vs_opponent(
            brain, neurons, opponent,
            substrate_color=X, temperature=temp, rng=rng,
        )
        outcomes.append(out.result)

        if out.result == X:
            r_signal = 1.0       # impossible vs perfect O, but reward if it happens
        elif out.result == EMPTY:
            r_signal = 1.0       # draw is the BEST achievable vs minimax → reward as win
        else:
            r_signal = 0.0       # loss

        credit_trajectory(brain, neurons, out.trajectory, r_signal,
                          eta=0.15, gamma=0.85)

        if (g + 1) % eval_every == 0:
            recent = outcomes[-eval_every:]
            wr = sum(1 for r in recent if r == X) / len(recent)
            lr = sum(1 for r in recent if r == O) / len(recent)
            dr = sum(1 for r in recent if r == EMPTY) / len(recent)
            eval_records.append({'after_games': g + 1, 'wins': wr,
                                  'losses': lr, 'draws': dr, 'temp': temp})

    return {'brain': brain, 'neurons': neurons,
            'outcomes': outcomes, 'eval': eval_records}


def evaluate_vs_minimax(brain: Brain, neurons: TTTNeurons,
                         n_games: int = 200, rng_seed: int = 9999) -> Dict:
    """Pure exploitation against minimax. Temperature 0.05.
    Goal: as close to 100% draws as possible (X cannot beat perfect O)."""
    rng = random.Random(rng_seed)
    outcomes = []
    for _ in range(n_games):
        out = play_substrate_vs_opponent(
            brain, neurons,
            lambda b, p, r: minimax_pick(b, p, r),
            substrate_color=X, temperature=0.05, rng=rng,
        )
        outcomes.append(out.result)
    return {
        'wins': sum(1 for r in outcomes if r == X) / len(outcomes),
        'losses': sum(1 for r in outcomes if r == O) / len(outcomes),
        'draws': sum(1 for r in outcomes if r == EMPTY) / len(outcomes),
    }


# ─── Phase 2: Clone substrate, self-play ──────────────────────────────────

def clone_brain(b: Brain) -> Brain:
    """Deep copy of a Brain. Numpy arrays copied; aliases dict copied."""
    new = Brain()
    new.nodes = b.nodes.copy()
    new.synapses = b.synapses.copy()
    new.syn_offsets = b.syn_offsets.copy()
    new.content_offsets = b.content_offsets.copy()
    new.content_blobs = list(b.content_blobs)
    new.aliases = dict(b.aliases)
    new.relations = list(b.relations)
    new.next_id = b.next_id
    new._used_synapses = getattr(b, '_used_synapses', 0)
    new._rebuild_relation_index()
    return new


def selfplay_from_clones(
    trained_brain: Brain, trained_neurons: TTTNeurons,
    *, n_games: int = 2000, eval_every: int = 250, rng_seed: int = 31415,
) -> Dict:
    """Clone trained brain into X-side and O-side copies. They self-play."""
    rng = random.Random(rng_seed)
    brain_x = clone_brain(trained_brain)
    brain_o = clone_brain(trained_brain)
    neurons_x = trained_neurons   # neuron-id mapping is the same
    neurons_o = trained_neurons

    outcomes: List[int] = []
    eval_records: List[Dict] = []

    for g in range(n_games):
        progress = g / max(1, n_games - 1)
        temp = 0.5 * (1.0 - progress) + 0.05  # already trained — explore less

        # Play one game with each brain on its color
        from .game import play_self_play_game
        out = play_self_play_game(brain_x, neurons_x,
                                    brain_o, neurons_o,
                                    temperature=temp, rng=rng)
        outcomes.append(out.result)

        if out.result == X:
            r_x, r_o = 1.0, 0.0
        elif out.result == O:
            r_x, r_o = 0.0, 1.0
        else:
            r_x, r_o = 0.5, 0.5  # draw — neutral

        credit_trajectory(brain_x, neurons_x, out.traj_x, r_x,
                          eta=0.10, gamma=0.85)
        credit_trajectory(brain_o, neurons_o, out.traj_o, r_o,
                          eta=0.10, gamma=0.85)

        if (g + 1) % eval_every == 0:
            recent = outcomes[-eval_every:]
            wr = sum(1 for r in recent if r == X) / len(recent)
            lr = sum(1 for r in recent if r == O) / len(recent)
            dr = sum(1 for r in recent if r == EMPTY) / len(recent)
            eval_records.append({'after_games': g + 1, 'x_wins': wr,
                                  'o_wins': lr, 'draws': dr})

    return {'brain_x': brain_x, 'brain_o': brain_o,
            'outcomes': outcomes, 'eval': eval_records}


# ─── Main ─────────────────────────────────────────────────────────────────

def main():
    print('=' * 70)
    print('  CURRICULUM — substrate learns from minimax, then self-improves')
    print('=' * 70)

    # Sanity baseline
    print('\n  Sanity: random-X vs minimax-O over 200 games')
    rng = random.Random(0)
    base_outcomes = []
    for _ in range(200):
        board = [EMPTY] * 9
        for turn in range(9):
            who = X if turn % 2 == 0 else O
            if who == X:
                legal = legal_moves(board)
                if not legal: break
                board[rng.choice(legal)] = X
            else:
                mv = minimax_pick(board, O, rng)
                if mv < 0: break
                board[mv] = O
            if winner(board) != EMPTY or is_full(board):
                break
        base_outcomes.append(winner(board))
    base_w = sum(1 for r in base_outcomes if r == X) / 200
    base_l = sum(1 for r in base_outcomes if r == O) / 200
    base_d = sum(1 for r in base_outcomes if r == EMPTY) / 200
    print(f'    random-X: wins={base_w:.3f} losses={base_l:.3f} draws={base_d:.3f}')
    print('  (Random X cannot beat perfect O. Best random gets is occasional draws.)')

    # ─── Phase 1 ───────────────────────────────────────────────────────
    print('\n' + '─' * 70)
    print('  Phase 1: Substrate (X) trains against MINIMAX (O), 3000 games')
    print('─' * 70)
    p1 = train_vs_minimax(n_games=3000, eval_every=250)

    print(f"\n  {'after games':<12}  {'X wins':<7}  {'X losses':<9}  {'draws':<6}  temp")
    for rec in p1['eval']:
        print(f"  {rec['after_games']:>10}    "
              f"{rec['wins']:.3f}    "
              f"{rec['losses']:.3f}     "
              f"{rec['draws']:.3f}   "
              f"{rec['temp']:.2f}")

    # Pure-exploitation evaluation against minimax
    print('\n  Evaluation: trained substrate vs minimax over 200 deterministic games')
    eval1 = evaluate_vs_minimax(p1['brain'], p1['neurons'], n_games=200)
    print(f'    wins={eval1["wins"]:.3f}  losses={eval1["losses"]:.3f}  draws={eval1["draws"]:.3f}')
    print('  Target: losses → 0, draws → 1.0  (X cannot beat perfect O)')

    # ─── Phase 2 ───────────────────────────────────────────────────────
    print('\n' + '─' * 70)
    print('  Phase 2: Clone the trained substrate, two copies self-play 2000 games')
    print('─' * 70)
    p2 = selfplay_from_clones(p1['brain'], p1['neurons'],
                                n_games=2000, eval_every=250)

    print(f"\n  {'after games':<12}  {'X wins':<7}  {'O wins':<7}  draws")
    for rec in p2['eval']:
        print(f"  {rec['after_games']:>10}    "
              f"{rec['x_wins']:.3f}    "
              f"{rec['o_wins']:.3f}    "
              f"{rec['draws']:.3f}")

    early_dr = p2['eval'][0]['draws']
    late_dr = p2['eval'][-1]['draws']
    print(f'\n  Draw-rate change: {early_dr:.3f} → {late_dr:.3f}  '
          f'(delta {late_dr - early_dr:+.3f})')
    print('  Hypothesis: trained-vs-trained should converge near all-draws,')
    print('  unlike random-vs-random self-play which diverged toward X-dominance.')

    print('\n  ─── ASSESSMENT ────────────────────────────────────────────────')
    p1_loss_drop = p1['eval'][0]['losses'] - p1['eval'][-1]['losses']
    if eval1['losses'] < base_l - 0.2 and eval1['draws'] > base_d + 0.2:
        print('  Phase 1 PASS: substrate learned to lose less / draw more vs minimax')
    else:
        print('  Phase 1 PARTIAL: substrate improved but did not fully reach optimum')
    print(f'         loss-rate vs minimax: random {base_l:.3f} → trained {eval1["losses"]:.3f}')
    print(f'         draw-rate vs minimax: random {base_d:.3f} → trained {eval1["draws"]:.3f}')


if __name__ == '__main__':
    main()
