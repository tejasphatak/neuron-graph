"""Substrate plays Tic-Tac-Toe.

Why this is interesting (more than the bandit):
- State is a CELL ASSEMBLY, not a single neuron. The board is encoded as
  9 simultaneously-active input neurons (one per cell, indicating its
  current value). This is exactly Tejas's "concept = bunch of neurons":
  the *concept* of the current board state is the population pattern.
- Compositional learning: substrate must associate a STATE PATTERN
  (multi-neuron) with an ACTION (single neuron), not just a single
  context with an action.
- Temporal credit assignment: reward arrives only at game end. Earlier
  moves must be credited via Monte Carlo (apply terminal reward to every
  (state, action) pair in the trajectory).

Setup:
- 3 input neurons per cell × 9 cells = 27 state neurons
    cell_0_X, cell_0_O, cell_0_empty, cell_1_X, ..., cell_8_empty
- 9 action neurons: act_0 ... act_8 (one per cell)
- No initial synapses. Substrate grows them through play.

Agent (substrate as X):
1. encode current board → activate 9 of 27 state neurons
2. spread → read activation of all action neurons
3. mask out illegal moves; softmax-pick from legal actions
4. play move; opponent (random) plays
5. on terminal: apply Monte Carlo reward (+1 win / 0 draw / −1 loss) to
   every (state-cell-neuron, chosen-action) pair in the trajectory,
   discounted by distance from terminal
6. repeat for N games

Success criterion: win-rate vs random opponent climbs above ~0.6 (random
plays poorly; an actual learner should win >60% by exploiting random's
mistakes). Theoretical max for X-vs-random is around 0.85–0.95.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .store import Brain
from .neuron import NeuronType, SYNAPSE_DTYPE
from .spread import spread


# ─── TTT environment ──────────────────────────────────────────────────────

EMPTY, X, O = 0, 1, 2

WIN_LINES = [
    (0, 1, 2), (3, 4, 5), (6, 7, 8),  # rows
    (0, 3, 6), (1, 4, 7), (2, 5, 8),  # cols
    (0, 4, 8), (2, 4, 6),              # diagonals
]


def winner(board: List[int]) -> int:
    """Return X, O, or 0 (no winner yet)."""
    for a, b, c in WIN_LINES:
        if board[a] != EMPTY and board[a] == board[b] == board[c]:
            return board[a]
    return EMPTY


def is_full(board: List[int]) -> bool:
    return all(c != EMPTY for c in board)


def legal_moves(board: List[int]) -> List[int]:
    return [i for i, v in enumerate(board) if v == EMPTY]


def render_board(board: List[int]) -> str:
    syms = {EMPTY: '.', X: 'X', O: 'O'}
    rows = []
    for r in range(3):
        rows.append(' '.join(syms[board[r * 3 + c]] for c in range(3)))
    return '\n'.join(rows)


# ─── Neuron layout ────────────────────────────────────────────────────────

@dataclass
class TTTNeurons:
    state_ids: Dict[Tuple[int, int], int]   # (cell, value) → neuron_id
    action_ids: Dict[int, int]               # cell → neuron_id


def build_brain() -> Tuple[Brain, TTTNeurons]:
    """Empty brain. 27 state neurons + 9 action neurons. No synapses yet."""
    b = Brain()
    state_ids: Dict[Tuple[int, int], int] = {}
    for cell in range(9):
        for value in (EMPTY, X, O):
            sym = {EMPTY: 'e', X: 'x', O: 'o'}[value]
            nid = b.add_neuron(lemma=f'cell{cell}_{sym}', type=NeuronType.CONCEPT)
            state_ids[(cell, value)] = nid

    action_ids: Dict[int, int] = {}
    for cell in range(9):
        nid = b.add_neuron(lemma=f'act{cell}', type=NeuronType.RULE)
        action_ids[cell] = nid

    return b, TTTNeurons(state_ids=state_ids, action_ids=action_ids)


def encode_state(board: List[int], n: TTTNeurons) -> List[int]:
    """Return list of state-neuron IDs that should fire for this board."""
    return [n.state_ids[(cell, value)] for cell, value in enumerate(board)]


# ─── Direct synapse update (RL-style targeted credit assignment) ──────────

def adjust_weight(brain: Brain, from_id: int, to_id: int, delta: float,
                   relation: str = 'co_occurs') -> None:
    """Read–modify–write a single synapse weight. If absent, create at 0.5+delta."""
    edges = brain.synapses_of(from_id)
    matches = (edges['to_id'] == to_id).nonzero()[0]
    if len(matches) > 0:
        base = int(brain.nodes[from_id]['syn_offset']) // SYNAPSE_DTYPE.itemsize
        abs_idx = base + int(matches[0])
        old = float(brain.synapses[abs_idx]['weight'])
        new = max(0.0, min(1.0, old + delta))
        brain.synapses[abs_idx]['weight'] = new
    else:
        initial = max(0.0, min(1.0, 0.5 + delta))
        brain.add_synapse(from_id, to_id, rel_name=relation, weight=initial)


# ─── Agent ────────────────────────────────────────────────────────────────

def agent_pick_action(brain: Brain, n: TTTNeurons, board: List[int],
                       *, temperature: float, rng: random.Random) -> int:
    """Spread from the current board state and pick a legal action."""
    seeds = encode_state(board, n)
    s = spread(brain, seeds, max_steps=3, sparsity=0.1)

    legal = legal_moves(board)
    if not legal:
        return -1

    # Read activation for each legal action
    scores = [s.activation.get(n.action_ids[c], 0.0) for c in legal]

    # Softmax over legal actions
    if temperature <= 0:
        return legal[max(range(len(legal)), key=lambda i: scores[i])]
    scaled = [s / temperature for s in scores]
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


def credit_trajectory(brain: Brain, n: TTTNeurons,
                       trajectory: List[Tuple[List[int], int]],
                       reward: float, *, eta: float = 0.1,
                       gamma: float = 0.9) -> None:
    """Monte Carlo update: terminal reward → every (state, action) pair
    in the trajectory, discounted by distance from end.

    For each pair, every active state-neuron in that step strengthens
    its synapse to the chosen action neuron by
        eta * (2*reward − 1) * gamma^(distance_from_end)
    """
    horizon = len(trajectory)
    for i, (board, action) in enumerate(trajectory):
        distance_from_end = horizon - 1 - i
        scaled = (2 * reward - 1) * (gamma ** distance_from_end)
        delta = eta * scaled
        action_id = n.action_ids[action]
        for state_neuron in encode_state(board, n):
            adjust_weight(brain, state_neuron, action_id, delta)


# ─── Game loop ────────────────────────────────────────────────────────────

@dataclass
class GameOutcome:
    result: int       # X (1) win, O (2) win, draw (0)
    moves: int
    trajectory: List[Tuple[List[int], int]] = field(default_factory=list)


def play_one_game(brain: Brain, n: TTTNeurons, *, temperature: float,
                   rng: random.Random) -> GameOutcome:
    """Substrate plays X. Random opponent plays O. X moves first."""
    board = [EMPTY] * 9
    trajectory: List[Tuple[List[int], int]] = []

    for turn in range(9):
        if turn % 2 == 0:  # X (substrate)
            action = agent_pick_action(brain, n, board,
                                        temperature=temperature, rng=rng)
            if action < 0:
                break
            trajectory.append((list(board), action))
            board[action] = X
        else:                # O (random)
            legal = legal_moves(board)
            if not legal:
                break
            board[rng.choice(legal)] = O

        w = winner(board)
        if w != EMPTY or is_full(board):
            return GameOutcome(result=w, moves=turn + 1, trajectory=trajectory)

    return GameOutcome(result=winner(board), moves=9, trajectory=trajectory)


# ─── Self-play (two substrate brains) ─────────────────────────────────────

@dataclass
class SelfPlayOutcome:
    result: int
    moves: int
    traj_x: List[Tuple[List[int], int]] = field(default_factory=list)
    traj_o: List[Tuple[List[int], int]] = field(default_factory=list)


def play_self_play_game(brain_x: Brain, neurons_x: TTTNeurons,
                         brain_o: Brain, neurons_o: TTTNeurons,
                         *, temperature: float,
                         rng: random.Random) -> SelfPlayOutcome:
    """X-substrate vs O-substrate. Both pick moves the same way; trajectories
    accumulated separately so each can be credit-assigned to its own brain.
    """
    board = [EMPTY] * 9
    traj_x: List[Tuple[List[int], int]] = []
    traj_o: List[Tuple[List[int], int]] = []

    for turn in range(9):
        if turn % 2 == 0:  # X
            action = agent_pick_action(brain_x, neurons_x, board,
                                         temperature=temperature, rng=rng)
            if action < 0: break
            traj_x.append((list(board), action))
            board[action] = X
        else:               # O
            action = agent_pick_action(brain_o, neurons_o, board,
                                         temperature=temperature, rng=rng)
            if action < 0: break
            traj_o.append((list(board), action))
            board[action] = O

        w = winner(board)
        if w != EMPTY or is_full(board):
            return SelfPlayOutcome(result=w, moves=turn + 1,
                                    traj_x=traj_x, traj_o=traj_o)
    return SelfPlayOutcome(result=winner(board), moves=9,
                            traj_x=traj_x, traj_o=traj_o)


def train_self_play(n_games: int = 4000, *, eval_every: int = 250,
                     rng_seed: int = 4321) -> Dict:
    """Two substrate brains learn from each other. Both start empty."""
    rng = random.Random(rng_seed)
    brain_x, neurons_x = build_brain()
    brain_o, neurons_o = build_brain()

    outcomes: List[int] = []
    eval_records: List[Dict] = []

    for g in range(n_games):
        progress = g / max(1, n_games - 1)
        temp = 1.5 * (1.0 - progress) + 0.05

        out = play_self_play_game(brain_x, neurons_x,
                                   brain_o, neurons_o,
                                   temperature=temp, rng=rng)
        outcomes.append(out.result)

        # Reward: each brain gets +1 if it won, 0 if lost, 0.5 if drew
        if out.result == X:
            r_x, r_o = 1.0, 0.0
        elif out.result == O:
            r_x, r_o = 0.0, 1.0
        else:
            r_x, r_o = 0.5, 0.5

        credit_trajectory(brain_x, neurons_x, out.traj_x, r_x,
                          eta=0.15, gamma=0.85)
        credit_trajectory(brain_o, neurons_o, out.traj_o, r_o,
                          eta=0.15, gamma=0.85)

        if (g + 1) % eval_every == 0:
            recent = outcomes[-eval_every:]
            wr = sum(1 for r in recent if r == X) / len(recent)
            lr = sum(1 for r in recent if r == O) / len(recent)
            dr = sum(1 for r in recent if r == EMPTY) / len(recent)
            eval_records.append({'after_games': g + 1, 'x_wins': wr,
                                  'o_wins': lr, 'draws': dr, 'temp': temp})

    return {'brain_x': brain_x, 'neurons_x': neurons_x,
            'brain_o': brain_o, 'neurons_o': neurons_o,
            'outcomes': outcomes, 'eval': eval_records}


# ─── Train + evaluate ─────────────────────────────────────────────────────

def train(n_games: int = 1000, *, eval_every: int = 100,
           rng_seed: int = 1234) -> Dict:
    rng = random.Random(rng_seed)
    brain, neurons = build_brain()

    outcomes: List[int] = []
    eval_records: List[Dict] = []

    for g in range(n_games):
        # Anneal temperature
        progress = g / max(1, n_games - 1)
        temp = 1.5 * (1.0 - progress) + 0.05

        outcome = play_one_game(brain, neurons,
                                 temperature=temp, rng=rng)
        outcomes.append(outcome.result)

        # Reward: +1 X wins, -1 O wins, 0 draw
        if outcome.result == X:
            reward = 1.0
        elif outcome.result == O:
            reward = 0.0
        else:
            reward = 0.5  # draw (better than losing, worse than winning)

        credit_trajectory(brain, neurons, outcome.trajectory, reward,
                          eta=0.15, gamma=0.85)

        if (g + 1) % eval_every == 0:
            recent = outcomes[-eval_every:]
            wr = sum(1 for r in recent if r == X) / len(recent)
            lr = sum(1 for r in recent if r == O) / len(recent)
            dr = sum(1 for r in recent if r == EMPTY) / len(recent)
            eval_records.append({'after_games': g + 1, 'wins': wr,
                                  'losses': lr, 'draws': dr, 'temp': temp})

    return {
        'brain': brain, 'neurons': neurons, 'outcomes': outcomes,
        'eval': eval_records,
    }


def baseline_random_vs_random(n_games: int = 500, rng_seed: int = 99) -> Dict:
    """Sanity baseline: random X vs random O. Confirms environment + counts."""
    rng = random.Random(rng_seed)
    outcomes = []
    for _ in range(n_games):
        board = [EMPTY] * 9
        for turn in range(9):
            legal = legal_moves(board)
            if not legal: break
            mark = X if turn % 2 == 0 else O
            board[rng.choice(legal)] = mark
            w = winner(board)
            if w != EMPTY:
                outcomes.append(w)
                break
        else:
            outcomes.append(winner(board))  # 0 if draw
    wr = sum(1 for r in outcomes if r == X) / len(outcomes)
    lr = sum(1 for r in outcomes if r == O) / len(outcomes)
    dr = sum(1 for r in outcomes if r == EMPTY) / len(outcomes)
    return {'wins': wr, 'losses': lr, 'draws': dr}


# ─── Main ─────────────────────────────────────────────────────────────────

def main():
    print('=' * 70)
    print('  SUBSTRATE PLAYS TIC-TAC-TOE — substrate (X) vs random (O)')
    print('=' * 70)

    print('\n  Baseline: random X vs random O over 500 games')
    base = baseline_random_vs_random(500)
    print(f'    X wins: {base["wins"]:.3f}   '
          f'O wins: {base["losses"]:.3f}   '
          f'draws: {base["draws"]:.3f}')
    print('  (X has first-move advantage; random X vs random O wins ~58%)')

    print('\n  Training substrate-X for 2000 games against random-O ...\n')
    result = train(n_games=2000, eval_every=200)

    print(f"  {'after games':<12}  {'wins':<6}  {'losses':<7}  {'draws':<6}  temp")
    for rec in result['eval']:
        print(f"  {rec['after_games']:>10}    "
              f"{rec['wins']:.3f}   "
              f"{rec['losses']:.3f}    "
              f"{rec['draws']:.3f}   "
              f"{rec['temp']:.2f}")

    final_wr = result['eval'][-1]['wins']
    final_lr = result['eval'][-1]['losses']
    delta_vs_random = final_wr - base['wins']

    print('\n  ─── ASSESSMENT ───────────────────────────────────────────────')
    print(f'  baseline (random X vs random O):  X wins {base["wins"]:.3f}')
    print(f'  trained substrate over last 200:  X wins {final_wr:.3f}')
    print(f'  delta:                            {delta_vs_random:+.3f}')

    if final_wr > base['wins'] + 0.05 and final_wr > 0.6:
        print('\n  PASS — substrate learned to play TTT better than random.')
        print('         Compositional state (cell assembly) + Monte Carlo credit')
        print('         assignment over multi-step trajectories: substrate handles it.')
    else:
        print('\n  FAIL — no measurable improvement over random play.')

    # Show a sample game from late in training
    print('\n  ─── SAMPLE GAME (after training, vs random) ──────────────────')
    rng = random.Random(7)
    sample = play_one_game(result['brain'], result['neurons'],
                           temperature=0.05, rng=rng)
    print(f"  Result: {'X wins' if sample.result == X else ('O wins' if sample.result == O else 'draw')}, "
          f"{sample.moves} moves")
    board = [EMPTY] * 9
    for i, (b_state, action) in enumerate(sample.trajectory):
        board = list(b_state)
        board[action] = X
        print(f'\n  X move {i+1} (cell {action}):')
        print('    ' + render_board(board).replace('\n', '\n    '))

    # ─── SELF-PLAY ──────────────────────────────────────────────────────
    print('\n' + '=' * 70)
    print('  SELF-PLAY — two empty substrate brains learn against each other')
    print('=' * 70)
    print('  As both brains learn, X first-move advantage should erode and')
    print('  draw rate should rise toward TTT theoretical optimum (always draw).')
    print()

    sp = train_self_play(n_games=4000, eval_every=400)
    print(f"  {'after games':<12}  {'X wins':<7}  {'O wins':<7}  {'draws':<6}  temp")
    for rec in sp['eval']:
        print(f"  {rec['after_games']:>10}    "
              f"{rec['x_wins']:.3f}    "
              f"{rec['o_wins']:.3f}    "
              f"{rec['draws']:.3f}   "
              f"{rec['temp']:.2f}")

    final = sp['eval'][-1]
    early = sp['eval'][0]
    print('\n  ─── SELF-PLAY ASSESSMENT ─────────────────────────────────────')
    print(f'  early  (first {sp["eval"][0]["after_games"]}):  X={early["x_wins"]:.3f}  '
          f'O={early["o_wins"]:.3f}  draws={early["draws"]:.3f}')
    print(f'  late  (last 400):     X={final["x_wins"]:.3f}  '
          f'O={final["o_wins"]:.3f}  draws={final["draws"]:.3f}')
    print(f'  draw-rate change:     {final["draws"] - early["draws"]:+.3f}')
    print(f'  X-dominance change:   {(final["x_wins"] - final["o_wins"]) - (early["x_wins"] - early["o_wins"]):+.3f}')

    # Sample late self-play game
    print('\n  ─── SAMPLE SELF-PLAY GAME (late training) ──────────────────────')
    rng2 = random.Random(13)
    spg = play_self_play_game(sp['brain_x'], sp['neurons_x'],
                                sp['brain_o'], sp['neurons_o'],
                                temperature=0.05, rng=rng2)
    result_label = 'X wins' if spg.result == X else ('O wins' if spg.result == O else 'draw')
    print(f'  Result: {result_label}, {spg.moves} moves')
    # Reconstruct full sequence
    board = [EMPTY] * 9
    move_count = 0
    tx_iter = iter(spg.traj_x)
    to_iter = iter(spg.traj_o)
    for turn in range(spg.moves):
        if turn % 2 == 0:
            try:
                _, action = next(tx_iter)
                board[action] = X
                move_count += 1
                print(f'\n  Turn {turn+1} (X plays {action}):')
            except StopIteration:
                break
        else:
            try:
                _, action = next(to_iter)
                board[action] = O
                move_count += 1
                print(f'\n  Turn {turn+1} (O plays {action}):')
            except StopIteration:
                break
        print('    ' + render_board(board).replace('\n', '\n    '))


if __name__ == '__main__':
    main()
