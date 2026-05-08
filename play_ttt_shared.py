"""Shared-brain self-play with color-symmetric state encoding.

ONE substrate plays BOTH sides. The trick: the brain never sees "X" or "O" —
it sees "me" and "opponent." When playing X, X-marks encode as "me" and
O-marks as "opp". When playing O, O-marks encode as "me" and X-marks as
"opp". Since TTT is symmetric under color swap, the same knowledge serves
both sides.

Layout (36 neurons total):
    27 state neurons:  cell_{i}_me, cell_{i}_opp, cell_{i}_empty   for i in 0..8
     9 action neurons: act_{i}                                       for i in 0..8

Per game: substrate plays both moves alternately. Both trajectories
(X-side and O-side) credit the SAME brain. Twice the training signal
per game; no asymmetric divergence between two independent learners.

Hypothesis to test: shared-brain self-play converges toward all-draws
(TTT theoretical optimum), unlike the previous two-brain experiments
which showed positive-feedback divergence.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from .play_ttt import (
    EMPTY, X, O, winner, is_full, legal_moves, render_board,
)
from .play_ttt_curriculum import minimax_pick, evaluate_vs_minimax
from .neuron import NeuronType, SYNAPSE_DTYPE
from .store import Brain
from .spread import spread


# ─── Neuron layout for shared brain ───────────────────────────────────────

ME, OPP = 1, 2     # internal labels (independent of X/O)


@dataclass
class SharedNeurons:
    state_ids: Dict[Tuple[int, int], int]    # (cell, ME|OPP|EMPTY) → neuron_id
    action_ids: Dict[int, int]                # cell → neuron_id


def build_shared_brain() -> Tuple[Brain, SharedNeurons]:
    """Empty brain with color-relative neurons."""
    b = Brain()
    state_ids: Dict[Tuple[int, int], int] = {}
    for cell in range(9):
        for value in (EMPTY, ME, OPP):
            sym = {EMPTY: 'e', ME: 'me', OPP: 'opp'}[value]
            nid = b.add_neuron(lemma=f'c{cell}_{sym}', type=NeuronType.CONCEPT)
            state_ids[(cell, value)] = nid
    action_ids = {cell: b.add_neuron(lemma=f'a{cell}', type=NeuronType.RULE)
                  for cell in range(9)}
    return b, SharedNeurons(state_ids=state_ids, action_ids=action_ids)


def encode_state_for_player(board: List[int], player: int,
                              n: SharedNeurons) -> List[int]:
    """Color-flipped encoding: player's marks → ME, opponent's → OPP.
    The substrate never sees X or O — only "me vs opp."""
    opponent = O if player == X else X
    seeds = []
    for cell, value in enumerate(board):
        if value == player:
            seeds.append(n.state_ids[(cell, ME)])
        elif value == opponent:
            seeds.append(n.state_ids[(cell, OPP)])
        else:
            seeds.append(n.state_ids[(cell, EMPTY)])
    return seeds


# ─── Shared agent pick + credit ───────────────────────────────────────────

def shared_pick_action(brain: Brain, n: SharedNeurons, board: List[int],
                        player: int, *, temperature: float,
                        rng: random.Random) -> int:
    seeds = encode_state_for_player(board, player, n)
    s = spread(brain, seeds, max_steps=3, sparsity=0.1)
    legal = legal_moves(board)
    if not legal:
        return -1
    scores = [s.activation.get(n.action_ids[c], 0.0) for c in legal]
    if temperature <= 0:
        return legal[max(range(len(legal)), key=lambda i: scores[i])]
    scaled = [v / temperature for v in scores]
    m = max(scaled)
    exps = [math.exp(v - m) for v in scaled]
    total = sum(exps) or 1.0
    probs = [e / total for e in exps]
    r = rng.random()
    cum = 0.0
    for i, p in enumerate(probs):
        cum += p
        if r <= cum:
            return legal[i]
    return legal[-1]


def adjust_weight(brain: Brain, from_id: int, to_id: int, delta: float) -> None:
    edges = brain.synapses_of(from_id)
    matches = (edges['to_id'] == to_id).nonzero()[0]
    if len(matches) > 0:
        base = int(brain.nodes[from_id]['syn_offset']) // SYNAPSE_DTYPE.itemsize
        idx = base + int(matches[0])
        old = float(brain.synapses[idx]['weight'])
        brain.synapses[idx]['weight'] = max(0.0, min(1.0, old + delta))
    else:
        initial = max(0.0, min(1.0, 0.5 + delta))
        brain.add_synapse(from_id, to_id, rel_name='co_occurs', weight=initial)


def credit_player_trajectory(brain: Brain, n: SharedNeurons,
                              trajectory: List[Tuple[List[int], int]],
                              player: int, reward: float, *,
                              eta: float = 0.15,
                              gamma: float = 0.85) -> None:
    """Credit one player's perspective. Each trajectory entry is encoded
    color-relative-to-player so the same brain is updated for both sides."""
    horizon = len(trajectory)
    for i, (board, action) in enumerate(trajectory):
        distance_from_end = horizon - 1 - i
        scaled = (2 * reward - 1) * (gamma ** distance_from_end)
        delta = eta * scaled
        action_id = n.action_ids[action]
        for state_neuron in encode_state_for_player(board, player, n):
            adjust_weight(brain, state_neuron, action_id, delta)


# ─── Self-play with one shared brain ──────────────────────────────────────

@dataclass
class SharedGame:
    result: int
    moves: int
    traj_x: List[Tuple[List[int], int]] = field(default_factory=list)
    traj_o: List[Tuple[List[int], int]] = field(default_factory=list)


def play_one_shared(brain: Brain, n: SharedNeurons, *,
                     temperature: float, rng: random.Random) -> SharedGame:
    board = [EMPTY] * 9
    traj_x, traj_o = [], []
    for turn in range(9):
        player = X if turn % 2 == 0 else O
        action = shared_pick_action(brain, n, board, player,
                                      temperature=temperature, rng=rng)
        if action < 0:
            break
        if player == X:
            traj_x.append((list(board), action))
        else:
            traj_o.append((list(board), action))
        board[action] = player
        w = winner(board)
        if w != EMPTY or is_full(board):
            return SharedGame(result=w, moves=turn + 1,
                                traj_x=traj_x, traj_o=traj_o)
    return SharedGame(result=winner(board), moves=9,
                        traj_x=traj_x, traj_o=traj_o)


def train_shared_self_play(n_games: int = 4000, *, eval_every: int = 400,
                             rng_seed: int = 8888) -> Dict:
    rng = random.Random(rng_seed)
    brain, neurons = build_shared_brain()

    outcomes: List[int] = []
    eval_records: List[Dict] = []

    for g in range(n_games):
        progress = g / max(1, n_games - 1)
        temp = 1.5 * (1.0 - progress) + 0.05

        out = play_one_shared(brain, neurons,
                                 temperature=temp, rng=rng)
        outcomes.append(out.result)

        # Reward each trajectory from its OWN player's perspective.
        # The same brain learns both sides simultaneously.
        if out.result == X:
            r_x, r_o = 1.0, 0.0
        elif out.result == O:
            r_x, r_o = 0.0, 1.0
        else:
            r_x, r_o = 0.6, 0.6  # draw — slight positive (better than losing)

        credit_player_trajectory(brain, neurons, out.traj_x, X, r_x,
                                  eta=0.12, gamma=0.85)
        credit_player_trajectory(brain, neurons, out.traj_o, O, r_o,
                                  eta=0.12, gamma=0.85)

        if (g + 1) % eval_every == 0:
            recent = outcomes[-eval_every:]
            wr = sum(1 for r in recent if r == X) / len(recent)
            lr = sum(1 for r in recent if r == O) / len(recent)
            dr = sum(1 for r in recent if r == EMPTY) / len(recent)
            eval_records.append({'after_games': g + 1, 'x_wins': wr,
                                  'o_wins': lr, 'draws': dr, 'temp': temp})

    return {'brain': brain, 'neurons': neurons,
            'outcomes': outcomes, 'eval': eval_records}


# ─── Eval the shared brain vs minimax (both sides) ────────────────────────

def eval_shared_vs_minimax_both_sides(brain: Brain, neurons: SharedNeurons,
                                        n_each: int = 100,
                                        rng_seed: int = 333) -> Dict:
    """Test the trained shared brain BOTH as X (vs minimax-O) and as O
    (vs minimax-X). Convergence test: brain should draw most games on
    both sides if it has captured TTT optimal play."""
    rng = random.Random(rng_seed)
    results = {'as_X': [], 'as_O': []}

    # As X
    for _ in range(n_each):
        board = [EMPTY] * 9
        for turn in range(9):
            who = X if turn % 2 == 0 else O
            if who == X:
                action = shared_pick_action(brain, neurons, board, X,
                                              temperature=0.05, rng=rng)
            else:
                action = minimax_pick(board, O, rng)
            if action < 0:
                break
            board[action] = who
            if winner(board) != EMPTY or is_full(board):
                break
        results['as_X'].append(winner(board))

    # As O
    for _ in range(n_each):
        board = [EMPTY] * 9
        for turn in range(9):
            who = X if turn % 2 == 0 else O
            if who == O:
                action = shared_pick_action(brain, neurons, board, O,
                                              temperature=0.05, rng=rng)
            else:
                action = minimax_pick(board, X, rng)
            if action < 0:
                break
            board[action] = who
            if winner(board) != EMPTY or is_full(board):
                break
        results['as_O'].append(winner(board))

    def stats(out_list, brain_color):
        wins = sum(1 for r in out_list if r == brain_color)
        losses = sum(1 for r in out_list if r != EMPTY and r != brain_color)
        draws = sum(1 for r in out_list if r == EMPTY)
        n = len(out_list)
        return wins / n, losses / n, draws / n

    x_w, x_l, x_d = stats(results['as_X'], X)
    o_w, o_l, o_d = stats(results['as_O'], O)
    return {
        'as_X': {'wins': x_w, 'losses': x_l, 'draws': x_d},
        'as_O': {'wins': o_w, 'losses': o_l, 'draws': o_d},
    }


# ─── Main ─────────────────────────────────────────────────────────────────

def main():
    print('=' * 70)
    print('  SHARED-BRAIN SELF-PLAY — color-symmetric encoding')
    print('=' * 70)
    print('  ONE substrate plays both sides. State encodes me/opp (not X/O).')
    print('  Both trajectories per game credit the SAME brain.')
    print('  Hypothesis: draw rate rises as the brain converges.')
    print()

    result = train_shared_self_play(n_games=4000, eval_every=400)

    print(f"  {'after games':<12}  {'X wins':<7}  {'O wins':<7}  {'draws':<6}  temp")
    for rec in result['eval']:
        print(f"  {rec['after_games']:>10}    "
              f"{rec['x_wins']:.3f}    "
              f"{rec['o_wins']:.3f}    "
              f"{rec['draws']:.3f}   "
              f"{rec['temp']:.2f}")

    early = result['eval'][0]
    late = result['eval'][-1]
    delta_draws = late['draws'] - early['draws']
    print('\n  ─── SHARED-BRAIN ASSESSMENT ─────────────────────────────────')
    print(f'  early (first 400):  X={early["x_wins"]:.3f}  O={early["o_wins"]:.3f}  draws={early["draws"]:.3f}')
    print(f'  late  (last 400):   X={late["x_wins"]:.3f}  O={late["o_wins"]:.3f}  draws={late["draws"]:.3f}')
    print(f'  draw-rate change:   {delta_draws:+.3f}  '
          f'({"converging toward draws" if delta_draws > 0 else "diverging"})')

    # Compare to two-brain version (recall: that was 0.10 → 0.03, delta -0.07)
    print('\n  Comparison vs two-independent-brain self-play:')
    print(f'    two-brain (earlier): draws 0.100 → 0.030  (Δ -0.070)')
    print(f'    shared-brain (this): draws {early["draws"]:.3f} → {late["draws"]:.3f}  '
          f'(Δ {delta_draws:+.3f})')

    # Vs minimax both sides
    print('\n  Evaluation: trained shared brain vs minimax, BOTH as X and as O')
    print('  (100 games each; deterministic exploitation; expect mostly draws)')
    ev = eval_shared_vs_minimax_both_sides(result['brain'], result['neurons'], n_each=100)
    print(f'  As X:  wins={ev["as_X"]["wins"]:.3f}  '
          f'losses={ev["as_X"]["losses"]:.3f}  '
          f'draws={ev["as_X"]["draws"]:.3f}')
    print(f'  As O:  wins={ev["as_O"]["wins"]:.3f}  '
          f'losses={ev["as_O"]["losses"]:.3f}  '
          f'draws={ev["as_O"]["draws"]:.3f}')

    # Compare to Phase 1 of curriculum (X vs minimax with two-brain layout)
    print('\n  Phase-1-curriculum (different layout) eval was: '
          'wins=0.000 losses=0.340 draws=0.660')


if __name__ == '__main__':
    main()
