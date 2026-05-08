"""Activation spreading — the only "thinking" primitive.

Single-threaded reference implementation. Phase C will add a C kernel
that takes the same numpy structured-array buffers via ctypes and runs
the spreading step in parallel with OpenMP. The buffer layout here
matches the C struct byte-for-byte (see neuron.NEURON_SIZE = 64 etc.),
so the C migration is pointer-passing only.

Sparsity (~2% of neurons active at once) is enforced via top-K pruning
each step — biologically inspired (HTM uses ~40 active out of 2048 ~ 2%).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


import numpy as np

from .store import Brain
from .working_memory import WorkingMemory


@dataclass
class ActivationState:
    """Result of a think() call. Activation map + read-out helpers."""
    activation: Dict[int, float]    # neuron_id → activation level (only > 0)
    steps_run: int
    converged: bool

    def top_k(self, brain: Brain, k: int = 10) -> List[Tuple[str, float, int]]:
        """Most-activated neurons as (label, activation, id) tuples, descending."""
        items = sorted(self.activation.items(), key=lambda x: -x[1])[:k]
        return [(brain.label(nid), lvl, nid) for nid, lvl in items]


def spread(
    brain: Brain,
    seeds: List[int],
    *,
    max_steps: int = 8,
    sparsity: float = 0.02,
    convergence_eps: float = 1e-3,
    initial_strength: float = 1.0,
    working_memory: Optional[WorkingMemory] = None,
    goals: Optional[List[int]] = None,
    goal_strength: float = 1.0,
    groups: Optional[Dict[int, int]] = None,
    group_top_k: Optional[int] = None,
) -> ActivationState:
    """Run the activation cycle on `brain` from the given seed neuron ids.

    Args:
        brain: the Brain instance
        seeds: list of neuron ids to start firing at strength 1.0
        max_steps: spreading-step bound
        sparsity: fraction of neurons kept active each step (top-K)
        convergence_eps: stop when L1 difference of activation < this
        initial_strength: starting activation for each seed
        working_memory: optional WorkingMemory whose current contents
            seed the spread (in addition to `seeds`) and which absorbs
            the final activation. Lets context carry across calls.
        goals: optional list of neuron ids kept clamped to `goal_strength`
            throughout the spread — top-down attention bias.
        goal_strength: clamp value for goal neurons (default 1.0)
        groups: optional {neuron_id: group_id} partition. Used together
            with `group_top_k` for sparsity-aware spread: within each
            group, only the top-K activations survive each step. Lets a
            task declare functional partitions (e.g. POS classes for LM,
            cell positions for TTT, pixel regions for vision) so spread
            doesn't fan out activation across all members of a noisy
            class. Modality-agnostic — substrate provides the mechanism,
            the task picks the partition.
        group_top_k: number of top-activated members to keep per group
            each step. None means no group-level sparsity (default).

    Returns:
        ActivationState with the final activation map.
    """
    relation_w = brain.relation_weight  # numpy float32 array, indexable by relation id

    activation: Dict[int, float] = {nid: float(initial_strength) for nid in seeds}
    # Working-memory carry-over: existing sustained pattern joins the seeds
    if working_memory is not None:
        for nid, lvl in working_memory.seeds().items():
            activation[nid] = max(activation.get(nid, 0.0), lvl)
    # Goal injection: clamp these neurons high from the start
    goal_set = set(goals) if goals else set()
    for gid in goal_set:
        activation[gid] = max(activation.get(gid, 0.0), goal_strength)
    if not activation:
        return ActivationState(activation={}, steps_run=0, converged=True)

    # Sparsity floor: HTM-inspired ~2% target for large brains, but always
    # keep at least 16 active so small brains (Phase A) don't collapse to one.
    n_total = max(brain.size, 1)
    k_active = max(16, int(n_total * sparsity))

    converged = False
    step = 0
    prev_activation: Dict[int, float] = {}

    for step in range(1, max_steps + 1):
        next_act: Dict[int, float] = {}

        # SPREAD: each active neuron fires to its synapse targets
        for nid, level in activation.items():
            if level <= 0:
                continue

            # Self-decay carry-over so the seed's own signal persists
            decay = float(brain.nodes[nid]['decay'])
            next_act[nid] = next_act.get(nid, 0.0) + level * decay

            # Walk synapses (zero-copy slice)
            edges = brain.synapses_of(nid)
            for syn in edges:
                rel_id = int(syn['relation'])
                rel_weight = float(relation_w[rel_id]) if rel_id < len(relation_w) else 0.0
                contribution = level * float(syn['weight']) * rel_weight
                if contribution == 0:
                    continue
                tid = int(syn['to_id'])
                next_act[tid] = next_act.get(tid, 0.0) + contribution

        # CLAMP goals back to high — they don't decay
        for gid in goal_set:
            next_act[gid] = max(next_act.get(gid, 0.0), goal_strength)

        # SPARSIFY: keep top-K active. Negative values (inhibition) drop too.
        # GOALS are protected: always preserved regardless of rank.
        if next_act:
            items = sorted(next_act.items(), key=lambda x: -x[1])[:k_active]
            next_act = {nid: lvl for nid, lvl in items if lvl > 0}
            for gid in goal_set:
                next_act[gid] = goal_strength

            # GROUP-LEVEL SPARSITY: within each declared group, keep only
            # top group_top_k members. Prevents the "firehose" where a
            # large class (many POS-NOUNs, many pixels of same row, etc.)
            # all pump activation equally and drown sentence-specific or
            # location-specific signals. Goals stay protected.
            if groups is not None and group_top_k is not None and group_top_k > 0:
                grouped: Dict[int, List[Tuple[int, float]]] = {}
                ungrouped: Dict[int, float] = {}
                for nid, lvl in next_act.items():
                    g = groups.get(nid)
                    if g is None:
                        ungrouped[nid] = lvl
                    else:
                        grouped.setdefault(g, []).append((nid, lvl))
                pruned: Dict[int, float] = dict(ungrouped)
                for g, members in grouped.items():
                    members.sort(key=lambda x: -x[1])
                    for nid, lvl in members[:group_top_k]:
                        pruned[nid] = lvl
                # Re-protect goals (some might have been pruned out of group)
                for gid in goal_set:
                    pruned[gid] = goal_strength
                next_act = pruned

        # CONVERGE: L1 difference of activation maps
        diff = _l1_diff(prev_activation, next_act)
        prev_activation = activation
        activation = next_act

        if diff < convergence_eps:
            converged = True
            break

    # Tracing
    log = getattr(brain, 'trace', None)
    if log is not None and getattr(log, 'enabled', False):
        log.log('spread', n_seeds=len(seeds), max_steps=max_steps,
                steps_run=step, converged=converged,
                n_active=len(activation), used_wm=working_memory is not None,
                n_goals=len(goal_set))

    # Working memory absorbs the result
    if working_memory is not None:
        working_memory.tick()
        working_memory.absorb(activation, gain=0.5)

    return ActivationState(activation=activation, steps_run=step, converged=converged)


def overlap_similarity(state_a: ActivationState, state_b: ActivationState) -> float:
    """Weighted Jaccard of two activation patterns. Range [0, 1].

    Numerator: sum over neurons present in both of min(level_a, level_b).
    Denominator: sum over neurons in either of max(level_a, level_b).
    Reduces to standard Jaccard if both are 0/1 indicators.
    """
    a, b = state_a.activation, state_b.activation
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    keys = set(a) | set(b)
    num = sum(min(a.get(k, 0.0), b.get(k, 0.0)) for k in keys)
    den = sum(max(a.get(k, 0.0), b.get(k, 0.0)) for k in keys)
    return num / den if den > 0 else 0.0


def _l1_diff(a: Dict[int, float], b: Dict[int, float]) -> float:
    keys = set(a) | set(b)
    return sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in keys)
