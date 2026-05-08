"""Concept-as-Neuron substrate for Guru.

Phase A — small in-RAM brain with spreading activation + Hebbian learning.
End-to-end demo: seed → spread → readout → teach → re-spread.

Public API:
    Brain                  — mutable in-RAM brain (store.py)
    spread(brain, seeds)   — activation cycle (spread.py)
    overlap_similarity     — similarity from activation patterns (spread.py)
    hebbian_update         — Hebbian learning step (learn.py)
    seed_brain()           — build the Phase A 50-neuron mini-brain (seed.py)
"""

from .neuron import (
    NEURON_DTYPE, SYNAPSE_DTYPE, NEURON_SIZE, SYNAPSE_SIZE,
    NeuronType, Modality, Flag,
)
from .store import Brain
from .spread import spread, overlap_similarity, ActivationState
from .learn import hebbian_update, decay_all
from .seed import seed_brain
from .working_memory import WorkingMemory
from .trace import TraceLog
from .modulator import Modulator
from .replay import ReplayBuffer, Episode, consolidate

__all__ = [
    # Substrate core
    'NEURON_DTYPE', 'SYNAPSE_DTYPE', 'NEURON_SIZE', 'SYNAPSE_SIZE',
    'NeuronType', 'Modality', 'Flag',
    'Brain', 'spread', 'overlap_similarity', 'ActivationState',
    'hebbian_update', 'decay_all', 'seed_brain',
    # New substrate primitives
    'WorkingMemory', 'TraceLog', 'Modulator',
    'ReplayBuffer', 'Episode', 'consolidate',
]
