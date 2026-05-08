"""Neuron + Synapse data structures — cache-line-aligned via numpy.

A neuron is a 64-byte fixed-size struct (one x86_64 cache line). A synapse
is 16 bytes (four per cache line). Layouts match the binary file format
in store.py — this module provides the in-RAM working representation.

Numpy structured dtype + memoryview gives us:
- Cache-friendly contiguous arrays
- Zero-copy mmap'd backing (later phases)
- Per-field access without per-record Python overhead
"""

from __future__ import annotations

import numpy as np


NEURON_SIZE = 64
SYNAPSE_SIZE = 16


# ─── Type / modality / flag enums ─────────────────────────────────────────

class NeuronType:
    TEXT = 0          # word / synset
    IMAGE = 1
    AUDIO = 2
    EPISODE = 3       # a memory of a sequence of co-activations
    RULE = 4          # condition→action neuron
    CONCEPT = 5       # abstract concept with no sensory content
    ASSEMBLY = 6      # named cell-assembly summary

    NAMES = {0: 'text', 1: 'image', 2: 'audio', 3: 'episode',
             4: 'rule', 5: 'concept', 6: 'assembly'}


class Modality:
    TEXT = 0
    VISUAL = 1
    AUDIO = 2
    SYMBOLIC = 3
    TEMPORAL = 4

    NAMES = {0: 'text', 1: 'visual', 2: 'audio', 3: 'symbolic', 4: 'temporal'}


class Flag:
    ACTIVE = 1 << 0
    DIRTY = 1 << 1
    PINNED = 1 << 2
    DELETED = 1 << 3


# ─── numpy structured dtypes ──────────────────────────────────────────────

NEURON_DTYPE = np.dtype([
    ('id',             np.uint64),     # 8 — globally unique
    ('type',           np.uint8),      # 1
    ('modality',       np.uint8),      # 1
    ('flags',          np.uint16),     # 2
    ('activation',     np.float32),    # 4 — transient, 0 at rest
    ('threshold',      np.float32),    # 4
    ('decay',          np.float32),    # 4
    ('last_fired_us',  np.uint64),     # 8
    ('fire_count',     np.uint32),     # 4
    ('fan_out',        np.uint32),     # 4
    ('syn_offset',     np.uint64),     # 8 — byte offset into synapses.bin
    ('content_offset', np.uint64),     # 8 — byte offset into content.bin (0 = none)
    ('reserved',       np.uint64),     # 8
], align=False)
assert NEURON_DTYPE.itemsize == NEURON_SIZE, \
    f"NEURON_DTYPE is {NEURON_DTYPE.itemsize} B, expected {NEURON_SIZE}"


SYNAPSE_DTYPE = np.dtype([
    ('to_id',     np.uint64),    # 8
    ('relation',  np.uint16),    # 2 — index into relation table
    ('flags',     np.uint16),    # 2
    ('weight',    np.float32),   # 4 — [0, 1], Hebbian-updated
], align=False)
assert SYNAPSE_DTYPE.itemsize == SYNAPSE_SIZE, \
    f"SYNAPSE_DTYPE is {SYNAPSE_DTYPE.itemsize} B, expected {SYNAPSE_SIZE}"


# ─── Constructors ─────────────────────────────────────────────────────────

def make_neuron(*, id, type=NeuronType.TEXT, modality=Modality.TEXT,
                threshold=0.1, decay=0.85, fan_out=0, syn_offset=0,
                content_offset=0):
    """Build a single neuron record (numpy 0-d array)."""
    n = np.zeros(1, dtype=NEURON_DTYPE)
    n[0]['id'] = id
    n[0]['type'] = type
    n[0]['modality'] = modality
    n[0]['flags'] = 0
    n[0]['activation'] = 0.0
    n[0]['threshold'] = threshold
    n[0]['decay'] = decay
    n[0]['last_fired_us'] = 0
    n[0]['fire_count'] = 0
    n[0]['fan_out'] = fan_out
    n[0]['syn_offset'] = syn_offset
    n[0]['content_offset'] = content_offset
    n[0]['reserved'] = 0
    return n[0]


def make_synapse(*, to_id, relation=0, weight=0.5, flags=0):
    s = np.zeros(1, dtype=SYNAPSE_DTYPE)
    s[0]['to_id'] = to_id
    s[0]['relation'] = relation
    s[0]['flags'] = flags
    s[0]['weight'] = weight
    return s[0]
