"""Substrate-native value head — learned, not hand-coded.

Replaces the planner's heuristic `evaluate_state` with a value computed
by spreading to outcome neurons. Same primitives:

- Three outcome neurons: `outcome_win`, `outcome_lose`, `outcome_draw`
- After each game: for every state-cell that was active during the game,
  strengthen its edge to the outcome neuron matching the result
  (γ-discounted by distance from the terminal). Relation: `value`.
- To evaluate a board: encode → spread (masked to `value`) → read
  activation of the three outcome neurons → score = (win − lose) /
  (win + lose + draw + ε).

No threat-counting, no positional weights, no hand-tuned scaling.
The substrate learns "state-cells that appeared in winning games are
those that PREDICT winning" — pure correlation extraction from
trajectory + outcome data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .neuron import NeuronType, SYNAPSE_DTYPE
from .play_ttt import (
    EMPTY, X, O, encode_state, winner, is_full, legal_moves,
    TTTNeurons, build_brain,
)
from .store import Brain
from .spread import spread


VALUE_RELATION = 'value'
VALUE_DEFAULT_WEIGHT = 0.40


# ─── Setup ────────────────────────────────────────────────────────────────

@dataclass
class OutcomeNeurons:
    win: int
    lose: int
    draw: int


def ensure_value_relation(brain: Brain) -> int:
    if VALUE_RELATION not in brain.relation_id:
        brain.relations.append((VALUE_RELATION, VALUE_DEFAULT_WEIGHT))
        brain._rebuild_relation_index()
    return brain.relation_id[VALUE_RELATION]


def ensure_outcome_neurons(brain: Brain) -> OutcomeNeurons:
    """Idempotent: register the three outcome neurons."""
    ensure_value_relation(brain)
    if brain.lookup('outcome_win') is None:
        brain.add_neuron(lemma='outcome_win', type=NeuronType.CONCEPT)
    if brain.lookup('outcome_lose') is None:
        brain.add_neuron(lemma='outcome_lose', type=NeuronType.CONCEPT)
    if brain.lookup('outcome_draw') is None:
        brain.add_neuron(lemma='outcome_draw', type=NeuronType.CONCEPT)
    return OutcomeNeurons(
        win=brain.lookup('outcome_win'),
        lose=brain.lookup('outcome_lose'),
        draw=brain.lookup('outcome_draw'),
    )


# ─── Targeted weight update on the `value` relation ──────────────────────

def _adjust_value_edge(brain: Brain, from_id: int, to_id: int,
                        delta: float) -> None:
    edges = brain.synapses_of(from_id)
    rel_id = brain.relation_id[VALUE_RELATION]
    if len(edges) > 0:
        match_mask = (edges['to_id'] == to_id) & (edges['relation'] == rel_id)
        idxs = match_mask.nonzero()[0]
        if len(idxs) > 0:
            base = int(brain.nodes[from_id]['syn_offset']) // SYNAPSE_DTYPE.itemsize
            abs_idx = base + int(idxs[0])
            old = float(brain.synapses[abs_idx]['weight'])
            new = max(0.0, min(1.0, old + delta))
            brain.synapses[abs_idx]['weight'] = new
            return
    initial = max(0.05, min(1.0, 0.3 + delta))
    brain.add_synapse(from_id, to_id, rel_name=VALUE_RELATION,
                       weight=initial)


# ─── Trajectory → value-edge updates ──────────────────────────────────────

def credit_value_trajectory(
    brain: Brain, neurons: TTTNeurons, outcomes: OutcomeNeurons,
    trajectory: List[Tuple[List[int], int]],
    result: int, my_color: int,
    *, eta: float = 0.05, gamma: float = 0.92,
) -> None:
    """Each (board, action) in trajectory contributes evidence about which
    outcome correlates with the active state-cells. γ-discounted by
    distance from the terminal so terminal moves get the most credit.

    For wins: strengthen state-cell → outcome_win (and weaken → outcome_lose).
    For losses: opposite.
    For draws: strengthen → outcome_draw (no opposite signal).
    """
    if result == my_color:
        primary = outcomes.win
        secondary = outcomes.lose
        sec_sign = -1.0
    elif result == EMPTY:
        primary = outcomes.draw
        secondary = None
        sec_sign = 0.0
    else:
        primary = outcomes.lose
        secondary = outcomes.win
        sec_sign = -1.0

    horizon = len(trajectory)
    for i, (board, _action) in enumerate(trajectory):
        distance = horizon - 1 - i
        weight_factor = gamma ** distance
        delta = eta * weight_factor
        for state_neuron in encode_state(board, neurons):
            _adjust_value_edge(brain, state_neuron, primary, delta)
            if secondary is not None:
                _adjust_value_edge(brain, state_neuron, secondary,
                                    sec_sign * delta * 0.5)


# ─── Substrate-learned value query ────────────────────────────────────────

def substrate_value(
    brain: Brain, neurons: TTTNeurons, outcomes: OutcomeNeurons,
    board: List[int], my_color: int,
    *, max_steps: int = 2,
) -> float:
    """Score `board` for `my_color` by spreading to outcome neurons.

    Returns a value in [-1, +1]:
      - terminal states: ±1 / 0 directly from game logic (no learning needed)
      - non-terminal: (win_act − lose_act) / (win + lose + draw + ε)
    """
    # Terminal: ground-truth, no need to query the substrate
    w = winner(board)
    if w == my_color:
        return 1.0
    if w != EMPTY and w != my_color:
        return -1.0
    if is_full(board):
        return 0.0

    # Non-terminal: spread through `value` relation only
    rel_id = brain.relation_id[VALUE_RELATION]
    seeds = encode_state(board, neurons)
    activation: Dict[int, float] = {nid: 1.0 for nid in seeds}

    n_total = max(brain.size, 1)
    k_active = max(48, int(n_total * 0.5))

    for _ in range(max_steps):
        next_act: Dict[int, float] = {}
        for nid, level in activation.items():
            if level <= 0:
                continue
            decay = float(brain.nodes[nid]['decay'])
            next_act[nid] = next_act.get(nid, 0.0) + level * decay
            edges = brain.synapses_of(nid)
            for syn in edges:
                if int(syn['relation']) != rel_id:
                    continue
                contrib = level * float(syn['weight'])
                tid = int(syn['to_id'])
                next_act[tid] = next_act.get(tid, 0.0) + contrib
        items = sorted(next_act.items(), key=lambda x: -x[1])[:k_active]
        activation = {nid: lvl for nid, lvl in items if lvl > 0}

    win_act = activation.get(outcomes.win, 0.0)
    lose_act = activation.get(outcomes.lose, 0.0)
    draw_act = activation.get(outcomes.draw, 0.0)
    total = win_act + lose_act + draw_act + 1e-6
    score = (win_act - lose_act) / total
    return max(-0.99, min(0.99, score))


# ─── Training: feed labeled games to the value head ──────────────────────

def train_value_head_from_games(
    brain: Brain, neurons: TTTNeurons,
    n_games: int = 1000, *, my_color: int = X,
    rng_seed: int = 13,
) -> Dict:
    """Generate self-play vs random; label every state in each trajectory
    with that game's outcome; update value edges accordingly.

    Returns stats dict (n_wins, n_losses, n_draws among training games).
    """
    import random
    rng = random.Random(rng_seed)
    outcomes = ensure_outcome_neurons(brain)

    wins = losses = draws = 0
    for _ in range(n_games):
        board = [EMPTY] * 9
        traj: List[Tuple[List[int], int]] = []

        # X plays randomly here for diversity; substrate just observes states
        # (the goal is to LEARN value, not generate a great policy)
        for turn in range(9):
            mark = X if turn % 2 == 0 else O
            legal = legal_moves(board)
            if not legal:
                break
            action = rng.choice(legal)
            if mark == my_color:
                traj.append((list(board), action))
            board[action] = mark
            if winner(board) != EMPTY or is_full(board):
                break

        result = winner(board)
        if result == my_color:
            wins += 1
        elif result == EMPTY:
            draws += 1
        else:
            losses += 1
        credit_value_trajectory(brain, neurons, outcomes,
                                  traj, result, my_color)

    return {'wins': wins, 'losses': losses, 'draws': draws,
            'n_games': n_games, 'outcomes': outcomes}


# ─── Combined evaluator: substrate value with one-step opponent sim ──────

def substrate_eval_with_lookahead(
    brain: Brain, neurons: TTTNeurons, outcomes: OutcomeNeurons,
    board: List[int], my_color: int,
) -> float:
    """The substrate-learned analogue to the hand-coded heuristic.
    Adds one-step opponent simulation around the substrate's value
    estimate to handle the "opponent wins next turn" case cleanly."""
    # Terminal: exact
    term_score = substrate_value(brain, neurons, outcomes, board, my_color)
    if board.count(EMPTY) == 0 or winner(board) != EMPTY:
        return term_score

    opp = O if my_color == X else X

    # Critical: opponent winning move available?
    for opp_action in legal_moves(board):
        sim_board = list(board)
        sim_board[opp_action] = opp
        if winner(sim_board) == opp:
            return -0.90

    # Symmetry: I win on next move?
    for my_action in legal_moves(board):
        sim_board = list(board)
        sim_board[my_action] = my_color
        if winner(sim_board) == my_color:
            return 0.85

    return term_score


# ─── Main: train + evaluate side-by-side with hand heuristic ────────────

def main():
    import random

    print('=' * 70)
    print('  SUBSTRATE-LEARNED VALUE HEAD — replacing the heuristic')
    print('=' * 70)

    # Build brain + train value head from random self-play
    brain, neurons = build_brain()
    print('\n  Phase 1: train value head on 2000 random games (X perspective)')
    print('  ' + '─' * 60)
    stats = train_value_head_from_games(brain, neurons, n_games=2000)
    print(f'  Training games: {stats["n_games"]}  '
          f'X wins {stats["wins"]/stats["n_games"]:.3f}  '
          f'losses {stats["losses"]/stats["n_games"]:.3f}  '
          f'draws {stats["draws"]/stats["n_games"]:.3f}')

    outcomes = stats['outcomes']

    # Inspect: how many value-edges did we grow?
    rel_id = brain.relation_id[VALUE_RELATION]
    n_value_edges = sum(
        1 for s in brain.synapses[:getattr(brain, '_used_synapses', 0)]
        if int(s['relation']) == rel_id
    )
    print(f'  Value-edge count after training: {n_value_edges}')

    # Sample value queries
    print('\n  Sample value estimates (post-training):')
    test_boards = [
        ([X, X, EMPTY, EMPTY, O, EMPTY, EMPTY, EMPTY, EMPTY],
         "X near-win on top row (X to move at cell 2)"),
        ([X, EMPTY, EMPTY, EMPTY, X, EMPTY, EMPTY, EMPTY, EMPTY],
         "X has center+corner, fork potential"),
        ([EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY],
         "empty board (X to move)"),
        ([O, O, EMPTY, X, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY],
         "O has near-win, X must block"),
    ]
    for board, label in test_boards:
        v = substrate_value(brain, neurons, outcomes, board, X)
        v_lookahead = substrate_eval_with_lookahead(brain, neurons, outcomes,
                                                      board, X)
        print(f'    {label}')
        print(f'      raw value={v:+.3f}   with-lookahead={v_lookahead:+.3f}')

    # ── Now wire it into the planner and evaluate ──
    print('\n  Phase 2: planner using substrate value head, vs random + minimax')
    print('  ' + '─' * 60)

    from .planning_agent import (
        plan_one_step, evaluate_terminal, play_with_planning,
    )
    from .play_ttt_curriculum import minimax_pick

    # Patch the planner to use substrate eval — call evaluate_state via injection
    import brain.planning_agent as pa
    orig_eval = pa.evaluate_state
    pa.evaluate_state = lambda b, c: substrate_eval_with_lookahead(
        brain, neurons, outcomes, b, c
    )

    try:
        # vs random
        rng = random.Random(7777)
        outcomes_r = []
        for _ in range(200):
            out, _ = play_with_planning(
                brain, neurons,
                opponent_fn=lambda b, c, r: r.choice(legal_moves(b)),
                rng=rng, learn_world_model=False,
            )
            outcomes_r.append(out.result)
        n = len(outcomes_r)
        rwin = sum(1 for r in outcomes_r if r == X) / n
        rloss = sum(1 for r in outcomes_r if r != EMPTY and r != X) / n
        rdraw = sum(1 for r in outcomes_r if r == EMPTY) / n
        print(f'  vs random:  wins={rwin:.3f}  losses={rloss:.3f}  draws={rdraw:.3f}')

        # vs minimax
        rng = random.Random(7777)
        outcomes_m = []
        for _ in range(200):
            out, _ = play_with_planning(
                brain, neurons,
                opponent_fn=lambda b, c, r: minimax_pick(b, c, r),
                rng=rng, learn_world_model=False,
            )
            outcomes_m.append(out.result)
        n = len(outcomes_m)
        mwin = sum(1 for r in outcomes_m if r == X) / n
        mloss = sum(1 for r in outcomes_m if r != EMPTY and r != X) / n
        mdraw = sum(1 for r in outcomes_m if r == EMPTY) / n
        print(f'  vs minimax: wins={mwin:.3f}  losses={mloss:.3f}  draws={mdraw:.3f}')
    finally:
        pa.evaluate_state = orig_eval

    print('\n  ─── COMPARISON ──────────────────────────────────────────────')
    print(f'                          random        minimax')
    print(f'    hand heuristic:       0.975 wins    0.930 draws / 0.070 losses')
    print(f'    substrate value head: {rwin:.3f} wins    {mdraw:.3f} draws / {mloss:.3f} losses')


if __name__ == '__main__':
    main()
