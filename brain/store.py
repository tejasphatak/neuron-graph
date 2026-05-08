"""Brain storage — in-RAM working set + persistence to binary files.

Files (in `data/brain/`):
  nodes.bin          array of NEURON_DTYPE, indexed by id
  syn_offsets.bin    uint64[]; syn_offsets[i] = element offset where
                     neuron i's synapses start in synapses.bin
                     (use 0xFFFFFFFFFFFFFFFF as "no synapses" sentinel)
  synapses.bin       array of SYNAPSE_DTYPE, concatenated per-neuron
  content.bin        variable-size blobs (text definitions, image features)
  content_index.bin  uint64[]; element offset into content.bin (0 = no content)
  meta/header.json   schema version, n_neurons, n_synapses, last_id
  meta/relations.json  relation_id -> (name, default_weight)
  meta/aliases.json  lemma -> neuron_id   (the index for parse-time lookup)

For Phase A everything lives in RAM (numpy arrays). For Phase C we'll
flip the same arrays to np.memmap with no API change — that's why the
dtype matches the eventual binary file format byte-for-byte.

Cache-line layout, fixed 64 B per neuron, 16 B per synapse — the C
kernel (Phase C+) will read the same bytes via pointer arithmetic.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .neuron import (
    NEURON_DTYPE, SYNAPSE_DTYPE, NeuronType, Modality, Flag,
    make_neuron, make_synapse,
)


SENTINEL_NO_SYN = np.uint64(0xFFFFFFFFFFFFFFFF)


# ─── Default relation table ───────────────────────────────────────────────
# Stored in DB / on-disk in real config (no hardcoding for production).
# Phase A keeps these inline as a starting schema; teach() can extend.

DEFAULT_RELATIONS: List[Tuple[str, float]] = [
    ('synonym',     0.95),
    ('is_a',        0.70),   # asymmetric: hyponym → hypernym
    ('inverse_is_a', 0.55),  # hypernym → hyponym (weaker — too many descendants)
    ('part_of',     0.60),
    ('has_part',    0.45),
    ('related_to',  0.30),
    ('antonym',    -0.50),   # inhibitory
    ('co_occurs',   0.25),
    ('caused_by',   0.40),
    ('causes',      0.40),
    ('used_for',    0.35),
    ('located_in',  0.30),
    ('similar_to',  0.50),
    ('depicts',     0.40),   # image neuron → text concept it depicts
]


# ─── Brain ────────────────────────────────────────────────────────────────

@dataclass
class Brain:
    """Mutable in-RAM brain. Persistence is a separate save() / load()."""

    # Backing arrays (numpy → migrates to mmap in Phase C)
    nodes: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=NEURON_DTYPE))
    synapses: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=SYNAPSE_DTYPE))
    syn_offsets: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.uint64))

    # Variable-length content blobs (text definitions etc.)
    content_blobs: List[bytes] = field(default_factory=list)
    content_offsets: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.uint64))

    # Index for parse-time lookup: lemma → neuron_id
    aliases: Dict[str, int] = field(default_factory=dict)

    # Relation table (id ↔ name + default weight)
    relations: List[Tuple[str, float]] = field(default_factory=lambda: list(DEFAULT_RELATIONS))
    relation_id: Dict[str, int] = field(init=False)
    relation_weight: np.ndarray = field(init=False)

    next_id: int = 0

    # Optional substrate primitives — created on first access if needed
    trace: Any = field(default=None, repr=False)
    modulator: Any = field(default=None, repr=False)

    def __post_init__(self):
        self._rebuild_relation_index()
        # Lazy-import to avoid circular dep
        if self.trace is None:
            from .trace import TraceLog
            self.trace = TraceLog()
        if self.modulator is None:
            from .modulator import Modulator
            self.modulator = Modulator()

    def _rebuild_relation_index(self):
        self.relation_id = {name: i for i, (name, _) in enumerate(self.relations)}
        self.relation_weight = np.array(
            [w for _, w in self.relations], dtype=np.float32
        )

    # ─── Capacity management ─────────────────────────────────────────────

    def _ensure_node_capacity(self, n: int):
        """Make sure self.nodes has room for `n` total neurons."""
        cur = len(self.nodes)
        if n <= cur:
            return
        # Grow geometrically; double or to n, whichever larger
        new_cap = max(n, cur * 2 if cur else 64)
        new = np.zeros(new_cap, dtype=NEURON_DTYPE)
        if cur:
            new[:cur] = self.nodes
        self.nodes = new

        new_off = np.full(new_cap, SENTINEL_NO_SYN, dtype=np.uint64)
        if cur:
            new_off[:cur] = self.syn_offsets
        self.syn_offsets = new_off

        new_co = np.zeros(new_cap, dtype=np.uint64)
        if cur:
            new_co[:cur] = self.content_offsets
        self.content_offsets = new_co

    def _ensure_synapse_capacity(self, n: int):
        cur = len(self.synapses)
        if n <= cur:
            return
        new_cap = max(n, cur * 2 if cur else 256)
        new = np.zeros(new_cap, dtype=SYNAPSE_DTYPE)
        if cur:
            new[:cur] = self.synapses
        self.synapses = new

    # ─── Adding neurons ──────────────────────────────────────────────────

    def add_neuron(self, *, lemma: Optional[str] = None,
                   type=NeuronType.TEXT, modality=Modality.TEXT,
                   threshold=0.1, decay=0.85,
                   content: Optional[bytes] = None) -> int:
        """Allocate a new neuron, return its id. Optional `lemma` registers
        an alias for parse-time lookup."""
        nid = self.next_id
        self.next_id += 1
        self._ensure_node_capacity(nid + 1)

        self.nodes[nid] = make_neuron(
            id=nid, type=type, modality=modality,
            threshold=threshold, decay=decay,
        )
        if content is not None:
            ofs = len(b''.join(self.content_blobs)) + len(self.content_blobs)
            # Simple append: track blobs separately, lazily packed on save()
            self.content_blobs.append(content)
            self.content_offsets[nid] = len(self.content_blobs)  # 1-indexed; 0 = none
        if lemma:
            self.aliases[lemma.lower()] = nid
        return nid

    # ─── Adding synapses (PCSR-style: one neuron's edges contiguous) ─────

    def _allocate_synapse_block(self, count: int) -> int:
        """Append space for `count` synapses; return start index."""
        start = self._used_synapses if hasattr(self, '_used_synapses') else 0
        self._used_synapses = start + count
        self._ensure_synapse_capacity(self._used_synapses)
        return start

    def set_synapses(self, neuron_id: int,
                     edges: List[Tuple[int, str, float]]) -> None:
        """Set ALL outgoing synapses for `neuron_id` from a list of
        (to_id, relation_name, weight). Replaces any existing block.

        Uses append-only allocation in Phase A (the old block is
        leaked — Phase B compaction reclaims). For small graphs this
        is fine and keeps the code minimal.
        """
        n = len(edges)
        if n == 0:
            self.syn_offsets[neuron_id] = SENTINEL_NO_SYN
            self.nodes[neuron_id]['fan_out'] = 0
            self.nodes[neuron_id]['syn_offset'] = 0
            return

        start = self._allocate_synapse_block(n)
        for i, (to_id, rel_name, weight) in enumerate(edges):
            if rel_name not in self.relation_id:
                self._add_relation(rel_name, default_weight=0.3)
            self.synapses[start + i] = make_synapse(
                to_id=to_id, relation=self.relation_id[rel_name], weight=weight,
            )
        self.syn_offsets[neuron_id] = start
        self.nodes[neuron_id]['fan_out'] = n
        self.nodes[neuron_id]['syn_offset'] = start * SYNAPSE_DTYPE.itemsize

    def add_synapse(self, from_id: int, to_id: int,
                    rel_name: str = 'related_to', weight: float = 0.5) -> None:
        """Append one synapse to from_id's edge list (PCSR append)."""
        if rel_name not in self.relation_id:
            self._add_relation(rel_name, default_weight=0.3)

        cur_count = int(self.nodes[from_id]['fan_out'])
        cur_start_byte = int(self.nodes[from_id]['syn_offset'])
        cur_start = cur_start_byte // SYNAPSE_DTYPE.itemsize

        # Phase A: relocate the whole block to the end on every append.
        # Cheap for small graphs; Phase B's PCSR pre-allocates gaps.
        new_start = self._allocate_synapse_block(cur_count + 1)
        if cur_count:
            self.synapses[new_start:new_start + cur_count] = \
                self.synapses[cur_start:cur_start + cur_count]
        self.synapses[new_start + cur_count] = make_synapse(
            to_id=to_id, relation=self.relation_id[rel_name], weight=weight,
        )
        self.nodes[from_id]['fan_out'] = cur_count + 1
        self.nodes[from_id]['syn_offset'] = new_start * SYNAPSE_DTYPE.itemsize
        self.syn_offsets[from_id] = new_start

    # ─── Lookup ──────────────────────────────────────────────────────────

    def lookup(self, lemma: str) -> Optional[int]:
        return self.aliases.get(lemma.lower())

    def synapses_of(self, neuron_id: int) -> np.ndarray:
        """View into the synapse array for one neuron (zero-copy slice)."""
        n = int(self.nodes[neuron_id]['fan_out'])
        if n == 0:
            return self.synapses[:0]
        ofs = int(self.nodes[neuron_id]['syn_offset']) // SYNAPSE_DTYPE.itemsize
        return self.synapses[ofs:ofs + n]

    def neuron(self, neuron_id: int):
        return self.nodes[neuron_id]

    def label(self, neuron_id: int) -> str:
        """Reverse-lookup: best human-readable name for a neuron id."""
        for lemma, nid in self.aliases.items():
            if nid == neuron_id:
                return lemma
        return f'#{neuron_id}'

    @property
    def size(self) -> int:
        return self.next_id

    def _add_relation(self, name: str, default_weight: float = 0.3):
        new_id = len(self.relations)
        self.relations.append((name, default_weight))
        self._rebuild_relation_index()

    # ─── Persistence ─────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Write all backing arrays + metadata to `path/` (a directory)."""
        os.makedirs(path, exist_ok=True)
        os.makedirs(os.path.join(path, 'meta'), exist_ok=True)

        n = self.next_id
        # Trim arrays to actual size
        self.nodes[:n].tofile(os.path.join(path, 'nodes.bin'))
        self.syn_offsets[:n].tofile(os.path.join(path, 'syn_offsets.bin'))
        used = getattr(self, '_used_synapses', 0)
        self.synapses[:used].tofile(os.path.join(path, 'synapses.bin'))
        self.content_offsets[:n].tofile(os.path.join(path, 'content_index.bin'))

        # Pack variable-length content blobs with lengths
        with open(os.path.join(path, 'content.bin'), 'wb') as f:
            for blob in self.content_blobs:
                f.write(len(blob).to_bytes(4, 'little'))
                f.write(blob)

        with open(os.path.join(path, 'meta', 'header.json'), 'w') as f:
            json.dump({
                'version': 1,
                'n_neurons': n,
                'n_synapses': used,
                'next_id': self.next_id,
            }, f, indent=2)
        with open(os.path.join(path, 'meta', 'relations.json'), 'w') as f:
            json.dump(self.relations, f, indent=2)
        with open(os.path.join(path, 'meta', 'aliases.json'), 'w') as f:
            json.dump(self.aliases, f, indent=2)

    @classmethod
    def load(cls, path: str) -> 'Brain':
        with open(os.path.join(path, 'meta', 'header.json')) as f:
            header = json.load(f)
        with open(os.path.join(path, 'meta', 'relations.json')) as f:
            rels = [tuple(x) for x in json.load(f)]
        with open(os.path.join(path, 'meta', 'aliases.json')) as f:
            aliases = json.load(f)

        n = header['n_neurons']
        used = header['n_synapses']
        nodes = np.fromfile(os.path.join(path, 'nodes.bin'), dtype=NEURON_DTYPE, count=n)
        syn_offsets = np.fromfile(os.path.join(path, 'syn_offsets.bin'),
                                  dtype=np.uint64, count=n)
        synapses = np.fromfile(os.path.join(path, 'synapses.bin'),
                               dtype=SYNAPSE_DTYPE, count=used)
        content_offsets = np.fromfile(os.path.join(path, 'content_index.bin'),
                                       dtype=np.uint64, count=n)

        # Read content blobs
        content_blobs: List[bytes] = []
        try:
            with open(os.path.join(path, 'content.bin'), 'rb') as f:
                while True:
                    raw = f.read(4)
                    if not raw:
                        break
                    length = int.from_bytes(raw, 'little')
                    content_blobs.append(f.read(length))
        except FileNotFoundError:
            pass

        b = cls(
            nodes=nodes, synapses=synapses, syn_offsets=syn_offsets,
            content_blobs=content_blobs, content_offsets=content_offsets,
            aliases=aliases, relations=rels,
            next_id=header['next_id'],
        )
        b._used_synapses = used
        return b
