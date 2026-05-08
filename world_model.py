"""Forward dynamics — substrate-native world model.

The substrate already has neurons + spreading + plasticity. To predict
"if I take action A in state S, what state S' will follow?", we add a
new RELATION TYPE rather than new machinery: a directed edge

    (pre-state-neuron OR action-neuron) ─predicts─> (post-state-neuron)

means: "when this pre-cell or this action fires, this post-cell tends
to fire next." The relation is learned by observing transitions and
strengthening edges from co-occurrence.

To PREDICT next state from (board, action):
    seeds = state-neurons-from(board) + [action-neuron]
    spread through ONLY the `predicts` relation (mask others)
    readout: post-state-cells that activate above threshold

This is identical machinery to spread() but constrained to one relation.
The same primitives — graph storage, spreading activation, weight update
— do double duty as world model.

What's new in this file:
    1. The 'predicts' relation registered in the brain's relation table
    2. observe_transition(brain, neurons, board_pre, action, board_post)
       — build/strengthen edges from a witnessed transition
    3. predict_next(brain, neurons, board, action) — spread forward and
       return per-cell predicted distribution
    4. measure_accuracy(brain, neurons, transitions) — fraction of
       per-cell predictions that match the actual post-state

State encoding here is the SAME 27-neuron-per-board layout used in
play_ttt: 3 values × 9 cells. Reused so the world model and the agent
share a single substrate.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .neuron import SYNAPSE_DTYPE
from .play_ttt import (
    EMPTY, X, O, encode_state, build_brain, TTTNeurons,
    legal_moves, winner, is_full,
)
from .store import Brain
from .spread import ActivationState


PREDICT_RELATION = 'predicts'
PREDICT_DEFAULT_WEIGHT = 0.40


# ─── Setup: register the predicts relation on a brain ────────────────────

def ensure_predict_relation(brain: Brain) -> int:
    """Register 'predicts' in the brain's relation table if not present.
    Returns its relation_id."""
    if PREDICT_RELATION not in brain.relation_id:
        brain.relations.append((PREDICT_RELATION, PREDICT_DEFAULT_WEIGHT))
        brain._rebuild_relation_index()
    return brain.relation_id[PREDICT_RELATION]


# ─── Targeted edge update on the predicts relation ───────────────────────

def _adjust_or_create(brain: Brain, from_id: int, to_id: int, delta: float,
                      *, relation: str = PREDICT_RELATION) -> None:
    """Look up (from_id, to_id) restricted to the given relation; nudge
    its weight by delta or create the edge with a starting value."""
    edges = brain.synapses_of(from_id)
    rel_id = brain.relation_id[relation]
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
    brain.add_synapse(from_id, to_id, rel_name=relation, weight=initial)


# ─── Observation: build edges from witnessed transitions ─────────────────

def observe_transition(
    brain: Brain, neurons: TTTNeurons,
    board_pre: List[int], action: int, board_post: List[int],
    *, eta: float = 0.08,
) -> Dict:
    """Strengthen forward edges from this single transition.

    Three kinds of edges are reinforced (each carries a different kind of signal):

      1. ACTION → changed-post-cell  (strong)
         The action causes that specific cell to take its new value.

      2. unchanged-pre-cell → same post-cell  (medium, stability prior)
         Cells that didn't change tend to stay the same across transitions.
         Without this, the model has no signal for "this stays put" and
         predicts garbage for unchanged cells.

      3. The local-pre-cell at the played position → changed-post-cell (weak)
         Captures "after cell_3_empty + action_3, cell_3_X follows."

    No global pre→post edges. They were too noisy: every cell on the board
    ended up implicated in every transition, leading the model to predict
    X everywhere whenever an X-action fired.

    Returns a small stats dict for tracing.
    """
    ensure_predict_relation(brain)
    action_id = neurons.action_ids[action]

    changed_cells = [i for i in range(9) if board_pre[i] != board_post[i]]
    unchanged_cells = [i for i in range(9) if board_pre[i] == board_post[i]]

    n_pairs = 0
    # 1. Action → changed cells (strong)
    # 3. Local pre-cell at action position → changed cell (weak)
    for cell in changed_cells:
        post_value = board_post[cell]
        post_neuron = neurons.state_ids[(cell, post_value)]
        _adjust_or_create(brain, action_id, post_neuron, eta * 1.0)
        n_pairs += 1

        # Local pre-state at this same cell (typically EMPTY before the move)
        local_pre_value = board_pre[cell]
        local_pre_neuron = neurons.state_ids[(cell, local_pre_value)]
        _adjust_or_create(brain, local_pre_neuron, post_neuron, eta * 0.4)
        n_pairs += 1

    # 2. Stability: for every unchanged cell, the pre-cell predicts itself
    for cell in unchanged_cells:
        val = board_pre[cell]
        same_neuron = neurons.state_ids[(cell, val)]
        _adjust_or_create(brain, same_neuron, same_neuron, eta * 0.6)
        n_pairs += 1

    log = getattr(brain, 'trace', None)
    if log is not None and getattr(log, 'enabled', False):
        log.log('observe_transition',
                action=action, n_changed=len(changed_cells),
                n_edges_touched=n_pairs)
    return {'changed_cells': changed_cells, 'edges_touched': n_pairs}


# ─── Prediction: spread forward through `predicts` only ──────────────────

@dataclass
class Prediction:
    """Predicted post-state per cell as a value distribution."""
    cell_distributions: List[Dict[int, float]]   # 9 dicts, value → score
    # Convenience: the most-likely value per cell
    predicted_board: List[int]

    def likelihood(self, true_board: List[int]) -> float:
        """Mean per-cell likelihood the true value got."""
        total = 0.0
        for i in range(9):
            d = self.cell_distributions[i]
            tot = sum(d.values()) or 1.0
            total += d.get(true_board[i], 0.0) / tot
        return total / 9


def predict_next(
    brain: Brain, neurons: TTTNeurons,
    board: List[int], action: int,
    *, max_steps: int = 2, sparsity: float = 0.5,
) -> Prediction:
    """Forward-spread through the `predicts` relation only.

    Mask all non-`predicts` relation contributions by zeroing their
    weight in a copy of the relation_weight vector (locally, just for
    this call). Then run a short spread; readout activation of each
    cell-value neuron.
    """
    ensure_predict_relation(brain)
    rel_id = brain.relation_id[PREDICT_RELATION]

    # Local relation-weight vector with everything except 'predicts' zeroed
    masked = np.zeros_like(brain.relation_weight)
    masked[rel_id] = brain.relation_weight[rel_id]

    # Manually run spread with masked weights — small inline implementation
    # so we don't perturb the global relation_weight.
    seeds = encode_state(board, neurons) + [neurons.action_ids[action]]
    activation: Dict[int, float] = {nid: 1.0 for nid in seeds}

    n_total = max(brain.size, 1)
    k_active = max(48, int(n_total * sparsity))   # generous; we want full board

    for _ in range(max_steps):
        next_act: Dict[int, float] = {}
        for nid, level in activation.items():
            if level <= 0: continue
            decay = float(brain.nodes[nid]['decay'])
            next_act[nid] = next_act.get(nid, 0.0) + level * decay
            edges = brain.synapses_of(nid)
            for syn in edges:
                if int(syn['relation']) != rel_id:
                    continue
                contrib = level * float(syn['weight']) * float(masked[rel_id])
                tid = int(syn['to_id'])
                next_act[tid] = next_act.get(tid, 0.0) + contrib
        # sparsify
        items = sorted(next_act.items(), key=lambda x: -x[1])[:k_active]
        activation = {nid: lvl for nid, lvl in items if lvl > 0}

    # Readout: per-cell distribution over {EMPTY, X, O}
    cell_dists: List[Dict[int, float]] = []
    predicted_board: List[int] = []
    for cell in range(9):
        dist = {}
        for value in (EMPTY, X, O):
            nid = neurons.state_ids[(cell, value)]
            dist[value] = activation.get(nid, 0.0)
        cell_dists.append(dist)
        # Most-activated value at this cell — fallback to current value
        # (which was a seed) if the predict-spread couldn't decide
        best_val = max(dist, key=dist.get) if max(dist.values()) > 0 else board[cell]
        predicted_board.append(best_val)

    return Prediction(cell_distributions=cell_dists,
                       predicted_board=predicted_board)


# ─── Training: observe many random / scripted games ──────────────────────

def collect_random_transitions(n_games: int, rng) -> List[Tuple]:
    """Roll out n_games of random vs random TTT, yielding all transitions.
    Each transition is (board_pre, action, board_post)."""
    transitions = []
    for _ in range(n_games):
        board = [EMPTY] * 9
        for turn in range(9):
            mark = X if turn % 2 == 0 else O
            legal = legal_moves(board)
            if not legal: break
            action = rng.choice(legal)
            board_pre = list(board)
            board[action] = mark
            board_post = list(board)
            transitions.append((board_pre, action, board_post))
            if winner(board) != EMPTY or is_full(board):
                break
    return transitions


def train_world_model(n_games: int = 500, rng_seed: int = 42) -> Tuple[Brain, TTTNeurons, List[Tuple]]:
    """Build a brain, observe random-game transitions, return everything."""
    import random
    rng = random.Random(rng_seed)
    brain, neurons = build_brain()
    transitions = collect_random_transitions(n_games, rng)
    for pre, action, post in transitions:
        observe_transition(brain, neurons, pre, action, post, eta=0.08)
    return brain, neurons, transitions


# ─── Evaluation ──────────────────────────────────────────────────────────

@dataclass
class WorldModelStats:
    n_evaluated: int
    cell_accuracy: float       # fraction of all 9*N cells correctly predicted
    full_board_accuracy: float # fraction of transitions where ALL 9 cells right
    mean_likelihood: float     # mean per-cell probability assigned to the true value


def evaluate(brain: Brain, neurons: TTTNeurons,
              transitions: List[Tuple]) -> WorldModelStats:
    """For each transition, ask the brain to predict and score it."""
    total_correct = 0
    total_cells = 0
    full_correct = 0
    total_likelihood = 0.0

    for pre, action, post in transitions:
        pred = predict_next(brain, neurons, pre, action)
        right = sum(1 for i in range(9) if pred.predicted_board[i] == post[i])
        total_correct += right
        total_cells += 9
        if right == 9:
            full_correct += 1
        total_likelihood += pred.likelihood(post)

    n = len(transitions) or 1
    return WorldModelStats(
        n_evaluated=len(transitions),
        cell_accuracy=total_correct / total_cells if total_cells else 0,
        full_board_accuracy=full_correct / n,
        mean_likelihood=total_likelihood / n,
    )


# ─── Main demo ───────────────────────────────────────────────────────────

def main():
    import random

    print('=' * 70)
    print('  WORLD MODEL — substrate-native forward dynamics')
    print('=' * 70)

    print('\n  Training: observe N random games, build `predicts` edges')
    print('  ' + '─' * 60)

    for n_train in (50, 200, 500, 1000):
        brain, neurons, transitions = train_world_model(n_games=n_train,
                                                          rng_seed=42)
        # Evaluate on a held-out test set
        rng = random.Random(99)
        test = collect_random_transitions(100, rng)
        stats = evaluate(brain, neurons, test)

        n_predict_edges = sum(
            1 for s in brain.synapses[:getattr(brain, '_used_synapses', 0)]
            if int(s['relation']) == brain.relation_id.get(PREDICT_RELATION, -1)
        )
        print(f'  trained on {n_train:>4} games '
              f'({len(transitions):>4} transitions, {n_predict_edges:>4} predict-edges) '
              f'→ cell-acc {stats.cell_accuracy:.3f}  '
              f'full-board {stats.full_board_accuracy:.3f}  '
              f'likelihood {stats.mean_likelihood:.3f}')

    # ─── Inspect a sample prediction ─────────────────────────────────
    print('\n  Sample prediction (after training on 1000 games):')
    print('  ' + '─' * 60)
    brain, neurons, _ = train_world_model(n_games=1000, rng_seed=42)
    rng = random.Random(7)

    # A reasonable midgame board
    board = [EMPTY, X, EMPTY, EMPTY, X, O, EMPTY, EMPTY, O]
    legal = legal_moves(board)
    action = rng.choice(legal)

    actual_board = list(board)
    actual_board[action] = X
    pred = predict_next(brain, neurons, board, action)

    def render(b):
        syms = {EMPTY: '.', X: 'X', O: 'O'}
        return '\n      '.join(' '.join(syms[b[r * 3 + c]] for c in range(3))
                                 for r in range(3))

    print(f'\n  Pre-state board:')
    print('      ' + render(board))
    print(f'\n  X plays cell {action}.  Actual post-state:')
    print('      ' + render(actual_board))
    print(f'\n  Substrate prediction:')
    print('      ' + render(pred.predicted_board))
    correct = sum(1 for i in range(9)
                  if pred.predicted_board[i] == actual_board[i])
    print(f'\n  Cells correctly predicted: {correct}/9')
    print(f'  Mean per-cell likelihood:   {pred.likelihood(actual_board):.3f}')


if __name__ == '__main__':
    main()
