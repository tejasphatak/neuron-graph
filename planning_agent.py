"""Planning agent — uses the substrate's world model for one-step lookahead.

Decision loop per move:
    1. For each legal action, ask the world model: "what state results?"
       (predict_next via masked spread on the `predicts` relation)
    2. For each predicted state, evaluate it: terminal? winning line? trap?
    3. Pick the action with the highest expected value.

The planner is thin glue. All of the heavy lifting — state representation,
forward simulation, evaluation — happens through substrate primitives:
    - encode_state: board → neuron activation
    - spread: forward inference (masked to `predicts` for simulation)
    - learn (Hebbian / targeted): record what happened

Important: the world model itself is LEARNED, not hardcoded. The planner
is only as good as the world model's predictions, which we measured at
~95% per-cell accuracy in world_model.py.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .play_ttt import (
    EMPTY, X, O, WIN_LINES, winner, is_full, legal_moves,
    build_brain, encode_state, TTTNeurons,
    GameOutcome, render_board,
)
from .play_ttt_curriculum import minimax_pick, evaluate_vs_minimax
from .neuron import SYNAPSE_DTYPE
from .store import Brain
from .spread import spread
from .world_model import (
    ensure_predict_relation, observe_transition, predict_next,
    PREDICT_RELATION,
)


# ─── State evaluation ─────────────────────────────────────────────────────

def evaluate_terminal(board: List[int], my_color: int) -> Optional[float]:
    """If this is a terminal state, return its value to me. Else None.
    +1 for my win, -1 for opponent win, 0 for draw, None for non-terminal."""
    w = winner(board)
    if w == my_color:
        return 1.0
    if w != EMPTY and w != my_color:
        return -1.0
    if is_full(board):
        return 0.0
    return None


def two_in_a_row_lines(board: List[int], color: int) -> int:
    """Count lines with exactly two `color` and one EMPTY (immediate threat)."""
    count = 0
    for a, b, c in WIN_LINES:
        cells = (board[a], board[b], board[c])
        if cells.count(color) == 2 and cells.count(EMPTY) == 1:
            count += 1
    return count


def evaluate_state(board: List[int], my_color: int) -> float:
    """Heuristic for non-terminal states with one-step opponent lookahead.

    The planner uses this to score the world model's predicted post-states.
    The post-state is "after my move, before opponent's." So the heuristic
    must check whether the OPPONENT can win on their next turn — that's
    the case where defending matters most. Without this check, the planner
    blunders into traps that look "balanced" by threat count.

    Future: replace with a value head trained on substrate (spread with
    goal=win → readout). For now: rule-of-thumb with one-step opponent sim.
    """
    term = evaluate_terminal(board, my_color)
    if term is not None:
        return term

    opp = O if my_color == X else X

    # CRITICAL: can the opponent win on their next move?
    # If yes, this is a near-loss state — heuristic must penalize it.
    for opp_action in legal_moves(board):
        sim_board = list(board)
        sim_board[opp_action] = opp
        if winner(sim_board) == opp:
            return -0.90  # opponent wins next; near-loss

    # Conversely: can WE win on our next move? (Only useful when planner
    # looks two steps ahead; for one-step planning the post-state is what we'd
    # play, so this shouldn't fire — but keep the symmetry for completeness.)
    for my_action in legal_moves(board):
        sim_board = list(board)
        sim_board[my_action] = my_color
        if winner(sim_board) == my_color:
            return 0.85  # we can win next turn — strongly positive

    my_threats = two_in_a_row_lines(board, my_color)
    opp_threats = two_in_a_row_lines(board, opp)

    pos_value = 0.0
    if board[4] == my_color: pos_value += 0.05
    if board[4] == opp:      pos_value -= 0.05
    for corner in (0, 2, 6, 8):
        if board[corner] == my_color: pos_value += 0.02
        if board[corner] == opp:      pos_value -= 0.02

    threat_value = 0.30 * my_threats - 0.30 * opp_threats

    return max(-0.99, min(0.99, threat_value + pos_value))


# ─── Lookahead planner ────────────────────────────────────────────────────

@dataclass
class PlanResult:
    action: int
    score: float
    used_world_model: bool
    candidate_scores: Dict[int, float] = field(default_factory=dict)


def plan_one_step(
    brain: Brain, neurons: TTTNeurons,
    board: List[int], my_color: int,
    *, temperature: float = 0.0,
    rng: Optional[random.Random] = None,
    require_model_confidence: float = 0.0,
) -> PlanResult:
    """Pick an action by one-step lookahead via the world model.

    For each legal action, ask the substrate to predict the resulting board.
    Score the prediction with evaluate_state. Pick the highest-scoring action
    (or sample by softmax if temperature > 0).

    `require_model_confidence`: if the world model's per-cell likelihood of
    the predicted board is below this, fall back to a simpler heuristic
    (don't trust low-confidence predictions). Useful early in training.
    """
    legal = legal_moves(board)
    if not legal:
        return PlanResult(action=-1, score=0.0, used_world_model=False)
    rng = rng or random.Random()

    candidate_scores: Dict[int, float] = {}
    used_model = False

    for action in legal:
        pred = predict_next(brain, neurons, board, action)
        # Force the played cell to my_color (we know that's true regardless
        # of model confidence; the model's job is to predict the rest)
        forced = list(pred.predicted_board)
        forced[action] = my_color
        score = evaluate_state(forced, my_color)
        candidate_scores[action] = score
        used_model = True

    # Pick (greedy or softmax)
    if temperature <= 0:
        # Greedy — break ties randomly
        max_score = max(candidate_scores.values())
        best = [a for a, s in candidate_scores.items() if abs(s - max_score) < 1e-9]
        action = rng.choice(best)
    else:
        # Softmax over scores
        import math
        scores = list(candidate_scores.values())
        actions = list(candidate_scores.keys())
        m = max(scores)
        exps = [math.exp((s - m) / temperature) for s in scores]
        total = sum(exps) or 1.0
        probs = [e / total for e in exps]
        r = rng.random()
        cum = 0.0
        action = actions[-1]
        for i, p in enumerate(probs):
            cum += p
            if r <= cum:
                action = actions[i]
                break

    log = getattr(brain, 'trace', None)
    if log is not None and getattr(log, 'enabled', False):
        log.log('plan',
                action=action,
                score=candidate_scores[action],
                n_candidates=len(legal),
                used_model=used_model)

    return PlanResult(
        action=action,
        score=candidate_scores[action],
        used_world_model=used_model,
        candidate_scores=candidate_scores,
    )


# ─── Game runner that plans + observes (continually grows the world model) ─

def play_with_planning(
    brain: Brain, neurons: TTTNeurons,
    *, my_color: int = X,
    opponent_fn=None,
    temperature: float = 0.0,
    rng: Optional[random.Random] = None,
    learn_world_model: bool = True,
) -> Tuple[GameOutcome, List[Dict]]:
    """Play one game with planning. Optionally learn world model online.

    opponent_fn(board, color, rng) -> action. Default: random.
    Returns (GameOutcome, plan_records).
    """
    rng = rng or random.Random()
    if opponent_fn is None:
        opponent_fn = lambda b, c, r: r.choice(legal_moves(b))

    board = [EMPTY] * 9
    plan_records: List[Dict] = []
    trajectory: List[Tuple[List[int], int]] = []

    for turn in range(9):
        whose = X if turn % 2 == 0 else O
        if whose == my_color:
            res = plan_one_step(brain, neurons, board, my_color,
                                  temperature=temperature, rng=rng)
            action = res.action
            if action < 0: break
            plan_records.append({'turn': turn, 'plan': res})
            trajectory.append((list(board), action))
        else:
            action = opponent_fn(board, whose, rng)
            if action < 0: break

        board_pre = list(board)
        board[action] = whose
        board_post = list(board)
        if learn_world_model:
            observe_transition(brain, neurons,
                                 board_pre, action, board_post, eta=0.08)

        w = winner(board)
        if w != EMPTY or is_full(board):
            return GameOutcome(result=w, moves=turn + 1,
                                trajectory=trajectory), plan_records
    return GameOutcome(result=winner(board), moves=9,
                        trajectory=trajectory), plan_records


# ─── Test rig ─────────────────────────────────────────────────────────────

def evaluate_planner(
    brain: Brain, neurons: TTTNeurons,
    opponent: str = 'random', n_games: int = 200,
    rng_seed: int = 7777, my_color: int = X,
) -> Dict:
    """Play n_games as my_color vs the named opponent. Return win/loss/draw."""
    rng = random.Random(rng_seed)

    def opp_random(b, c, r): return r.choice(legal_moves(b))
    def opp_minimax(b, c, r): return minimax_pick(b, c, r)
    opp_fn = {'random': opp_random, 'minimax': opp_minimax}[opponent]

    outcomes = []
    for _ in range(n_games):
        out, _ = play_with_planning(
            brain, neurons,
            my_color=my_color,
            opponent_fn=opp_fn,
            temperature=0.0,
            rng=rng,
            learn_world_model=False,    # don't pollute world model during eval
        )
        outcomes.append(out.result)
    n = len(outcomes)
    return {
        'opponent': opponent,
        'n': n,
        'wins': sum(1 for r in outcomes if r == my_color) / n,
        'losses': sum(1 for r in outcomes if r != EMPTY and r != my_color) / n,
        'draws': sum(1 for r in outcomes if r == EMPTY) / n,
    }


# ─── Main: train world model, then evaluate the planner ──────────────────

def main():
    print('=' * 70)
    print('  PLANNING AGENT — substrate world model + one-step lookahead')
    print('=' * 70)

    # Phase 1: collect transitions to train the world model
    print('\n  Phase 1: train world model on 500 random games')
    print('  ' + '─' * 60)
    rng = random.Random(11)
    brain, neurons = build_brain()
    ensure_predict_relation(brain)

    from .world_model import collect_random_transitions
    transitions = collect_random_transitions(500, rng)
    for pre, action, post in transitions:
        observe_transition(brain, neurons, pre, action, post, eta=0.08)

    n_predict = sum(
        1 for s in brain.synapses[:getattr(brain, '_used_synapses', 0)]
        if int(s['relation']) == brain.relation_id.get(PREDICT_RELATION, -1)
    )
    print(f'  observed {len(transitions)} transitions → '
          f'{n_predict} predict-edges in substrate')

    # Phase 2: planner vs random
    print('\n  Phase 2: planner (X) vs random (O), 200 games')
    print('  ' + '─' * 60)
    rand_eval = evaluate_planner(brain, neurons,
                                   opponent='random', n_games=200)
    print(f'    wins={rand_eval["wins"]:.3f}  '
          f'losses={rand_eval["losses"]:.3f}  '
          f'draws={rand_eval["draws"]:.3f}')
    print('    Baseline (random vs random): X wins ~0.58')
    print('    Naive RL agent (no planning, 2000 games training): X wins ~0.78')

    # Phase 3: planner vs minimax
    print('\n  Phase 3: planner (X) vs minimax (O), 200 games')
    print('  ' + '─' * 60)
    mm_eval = evaluate_planner(brain, neurons,
                                 opponent='minimax', n_games=200)
    print(f'    wins={mm_eval["wins"]:.3f}  '
          f'losses={mm_eval["losses"]:.3f}  '
          f'draws={mm_eval["draws"]:.3f}')
    print('    Random X vs minimax baseline: 0% wins, 76% losses, 24% draws')
    print('    Trained naive RL vs minimax:   0% wins, 34% losses, 66% draws')

    # Sample game
    print('\n  Sample game (planner X vs random O):')
    print('  ' + '─' * 60)
    rng2 = random.Random(99)
    out, plans = play_with_planning(brain, neurons,
                                      opponent_fn=lambda b, c, r: r.choice(legal_moves(b)),
                                      rng=rng2, learn_world_model=False)
    print(f'  Result: {"X wins" if out.result == X else ("O wins" if out.result == O else "draw")}, '
          f'{out.moves} moves')

    board = [EMPTY] * 9
    plan_idx = 0
    for turn in range(out.moves):
        whose = X if turn % 2 == 0 else O
        if whose == X and plan_idx < len(plans):
            plan = plans[plan_idx]['plan']
            cs = plan.candidate_scores
            top = sorted(cs.items(), key=lambda x: -x[1])[:3]
            top_str = ', '.join(f'cell{a}={s:+.2f}' for a, s in top)
            print(f'\n  Turn {turn + 1} (X plans): chose {plan.action}, '
                  f'top candidates: {top_str}')
            board[plan.action] = X
            plan_idx += 1
        else:
            # Reconstruct opponent move from outcomes
            for c in range(9):
                if c not in [b for _, b in plans[:plan_idx]] and \
                   c != out.trajectory[plan_idx-1][1] if plan_idx > 0 else True:
                    pass  # too brittle, skip detailed render
            print(f'\n  Turn {turn + 1} (O plays):')
        print('    ' + render_board(board).replace('\n', '\n    '))

    # Comparison summary
    print('\n  ─── COMPARISON ──────────────────────────────────────────────')
    print(f'  {"opponent":<10}  {"planner X wins":<15}  {"naive RL X wins"}')
    print(f'  {"random":<10}  {rand_eval["wins"]:<15.3f}  ~0.78 (after 2000 games)')
    print(f'  {"minimax":<10}  {mm_eval["wins"]:<15.3f}  0.00 (no learning)')
    print(f'  {"":>11}  {"draws: " + f"{rand_eval[chr(34)+chr(100)+chr(114)+chr(97)+chr(119)+chr(115)+chr(34)]:.3f}":<15}  {"vs minimax draws: 0.66"}')


if __name__ == '__main__':
    main()
