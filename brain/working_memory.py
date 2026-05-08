"""Working memory — sustained activation across calls.

A spread() call dissolves its activation pattern when it returns. Brains
don't work that way: a sustained pattern of "what I'm currently thinking
about" lives in working memory for seconds-to-minutes, decaying slowly,
biasing the next inference.

WorkingMemory holds an activation dict that:
  - absorbs the result of each spread() (merges in)
  - decays per tick() at a configurable rate
  - bounds itself to a max size (oldest weakest evicted)
  - exposes seeds() for the next spread to start from

This unblocks goal-directed behavior, multi-step reasoning, and
context carry-over between turns.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class WorkingMemory:
    """Sustained activation buffer with slow decay."""
    decay: float = 0.7        # per-tick multiplicative decay
    max_size: int = 64        # cap; smallest activations evicted
    floor: float = 0.01       # below this, drop the entry
    activation: Dict[int, float] = field(default_factory=dict)

    def absorb(self, new_activation: Dict[int, float],
               *, gain: float = 0.5) -> None:
        """Merge a fresh activation pattern into working memory.
        `gain` controls how much of the new pattern is kept."""
        for nid, lvl in new_activation.items():
            self.activation[nid] = max(self.activation.get(nid, 0.0),
                                         lvl * gain)
        self._cap()

    def tick(self) -> None:
        """Apply one decay step. Drops below-floor entries."""
        for nid in list(self.activation.keys()):
            self.activation[nid] *= self.decay
            if self.activation[nid] < self.floor:
                del self.activation[nid]

    def seeds(self) -> Dict[int, float]:
        """Current activation, suitable for seeding the next spread."""
        return dict(self.activation)

    def clear(self) -> None:
        self.activation.clear()

    def _cap(self) -> None:
        if len(self.activation) <= self.max_size:
            return
        sorted_items = sorted(self.activation.items(), key=lambda x: -x[1])
        self.activation = dict(sorted_items[:self.max_size])

    def snapshot(self) -> List[tuple]:
        """For tracing — a serializable view."""
        return sorted(self.activation.items(), key=lambda x: -x[1])
