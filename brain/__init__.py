"""Concept-as-Neuron substrate — domain-agnostic core.

This package is the REUSABLE substrate. Identity-bearing neurons,
typed weighted edges, spreading activation, Hebbian + targeted
plasticity, working memory, goal injection, modulatory layer, replay
buffer, trace logging. No domain knowledge — those live under
`brain.tasks.<task_name>`.

Public API:
    Brain            — mutable in-RAM brain (store)
    spread()         — activation cycle (with optional WM + goals)
    overlap_similarity — Jaccard overlap between two activation patterns
    hebbian_update   — Hebbian co-activation strengthening
    decay_all        — global synaptic decay
    WorkingMemory    — sustained activation across calls
    Modulator        — global plasticity scalar (dopamine analog)
    ReplayBuffer     — episode ring + consolidate()
    TraceLog         — append-only event log

Domain demonstrations (TTT, bandit, world model, value head, planner,
all the empirical proofs) live under `brain.tasks.ttt`.
"""

from .neuron import (
    NEURON_DTYPE, SYNAPSE_DTYPE, NEURON_SIZE, SYNAPSE_SIZE,
    NeuronType, Modality, Flag,
)
from .store import Brain
from .spread import spread, overlap_similarity, ActivationState
from .learn import hebbian_update, decay_all
from .working_memory import WorkingMemory
from .trace import TraceLog
from .modulator import Modulator
from .replay import ReplayBuffer, Episode, consolidate
from .astrocyte import NeuronAstrocyteMemory, SubstrateAstrocyteMemory

__all__ = [
    # Layout / types
    'NEURON_DTYPE', 'SYNAPSE_DTYPE', 'NEURON_SIZE', 'SYNAPSE_SIZE',
    'NeuronType', 'Modality', 'Flag',
    # Substrate core
    'Brain', 'spread', 'overlap_similarity', 'ActivationState',
    'hebbian_update', 'decay_all',
    # Working primitives
    'WorkingMemory', 'TraceLog', 'Modulator',
    'ReplayBuffer', 'Episode', 'consolidate',
    # Associative memory (neuron-astrocyte, bipartite gather/scatter)
    'NeuronAstrocyteMemory', 'SubstrateAstrocyteMemory',
]
