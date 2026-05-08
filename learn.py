"""Hebbian learning — local, online, no backprop.

After every query (success OR failure), pairs of co-active neurons get
their synapse weight nudged. New synapses are created on the fly when
two strongly-active neurons have no edge between them yet — this is
how the brain GROWS structure from experience.

No gradient computation. Each update touches two neurons and one weight.
"""

from __future__ import annotations

from typing import Dict, Tuple

from .store import Brain
from .spread import ActivationState


def hebbian_update(
    brain: Brain,
    state: ActivationState,
    *,
    eta: float = 0.05,
    reward: float = 1.0,
    co_threshold: float = 0.2,
    create_threshold: float = 0.4,
    max_pairs: int = 200,
    grow_relation: str = 'co_occurs',
) -> Dict[str, int]:
    """Strengthen synapses between co-active neurons in `state`.

    Args:
        brain: the Brain to mutate
        state: result of a recent think() call
        eta: learning rate
        reward: +1 for confirmed-good answer, -1 for confirmed-wrong, 0 = unknown
        co_threshold: minimum activation for a neuron to count as "co-active"
        create_threshold: minimum joint activation to BUILD a new synapse
                          (above just nudging existing weights)
        max_pairs: bound on pairs visited per call (cheap insurance)
        grow_relation: relation name to use when creating new synapses

    Returns:
        {'updated': N, 'created': M} counts.
    """
    co_active = [(nid, lvl) for nid, lvl in state.activation.items()
                 if lvl >= co_threshold]
    co_active.sort(key=lambda x: -x[1])
    co_active = co_active[:max_pairs]

    updated = 0
    created = 0

    for i, (a_id, a_lvl) in enumerate(co_active):
        for b_id, b_lvl in co_active[i + 1:]:
            if a_id == b_id:
                continue
            joint = a_lvl * b_lvl

            # Try to find an existing synapse a -> b
            existing = _find_synapse(brain, a_id, b_id)
            if existing is not None:
                # Nudge weight (clamped)
                old = float(brain.synapses[existing]['weight'])
                new = max(0.0, min(1.0, old + eta * joint * reward))
                brain.synapses[existing]['weight'] = new
                updated += 1
            elif joint >= create_threshold and reward > 0:
                # Grow a new edge — this is the substrate's plasticity
                brain.add_synapse(a_id, b_id, rel_name=grow_relation,
                                  weight=min(1.0, eta * joint * reward * 5))
                created += 1

    return {'updated': updated, 'created': created}


def _find_synapse(brain: Brain, from_id: int, to_id: int):
    """Linear scan over a neuron's synapses; return index in brain.synapses or None."""
    edges = brain.synapses_of(from_id)
    if len(edges) == 0:
        return None
    matches = (edges['to_id'] == to_id).nonzero()[0]
    if len(matches) == 0:
        return None
    # Convert local edge index to absolute index in brain.synapses
    syn_offset_bytes = int(brain.nodes[from_id]['syn_offset'])
    from .neuron import SYNAPSE_DTYPE
    base = syn_offset_bytes // SYNAPSE_DTYPE.itemsize
    return base + int(matches[0])


def decay_all(brain: Brain, rate: float = 0.999) -> None:
    """Global synaptic decay — call infrequently (per session, per epoch)."""
    used = getattr(brain, '_used_synapses', 0)
    if used > 0:
        brain.synapses[:used]['weight'] *= rate
