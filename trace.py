"""Trace log — every substrate event recorded for inspection.

Every spread, every weight update, every goal injection, every replay
appends an entry. The log is append-only, JSON-serializable, dumpable
to disk. The point: at any time, "why did the substrate do that?" must
be answerable by reading the log.

Events recorded:
    'spread'   {seeds, max_steps, sparsity, n_active_final, converged}
    'update'   {from_id, to_id, relation, old_weight, new_weight, reason}
    'goal_set' {neuron_ids, strength}
    'replay'   {n_trajectories, eta_scale}
    'modulator' {old, new, reason}
    'wm_tick'  {n_active_before, n_active_after}
    'teach'    {lemma, edges_created}

The log is held on the Brain instance (brain.trace_log). Bounded by
default to avoid runaway memory; older entries roll off.
"""

from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, Iterable, List, Optional


@dataclass
class TraceLog:
    """Append-only ring of substrate events."""
    capacity: int = 10000
    entries: Deque[Dict[str, Any]] = field(
        default_factory=lambda: deque(maxlen=10000)
    )
    enabled: bool = True

    def __post_init__(self):
        # honor capacity if user changed it
        if self.entries.maxlen != self.capacity:
            self.entries = deque(self.entries, maxlen=self.capacity)

    def log(self, event_type: str, **payload) -> None:
        if not self.enabled:
            return
        self.entries.append({
            't': time.time(),
            'event': event_type,
            **payload,
        })

    def tail(self, n: int = 20) -> List[Dict[str, Any]]:
        return list(self.entries)[-n:]

    def filter(self, event_type: str) -> List[Dict[str, Any]]:
        return [e for e in self.entries if e['event'] == event_type]

    def dump(self, path: str) -> None:
        """Write the full log as a JSONL file (one event per line)."""
        with open(path, 'w', encoding='utf-8') as f:
            for entry in self.entries:
                f.write(json.dumps(entry, default=str) + '\n')

    def summary(self) -> Dict[str, int]:
        """Count by event type."""
        out: Dict[str, int] = {}
        for e in self.entries:
            out[e['event']] = out.get(e['event'], 0) + 1
        return out

    def clear(self) -> None:
        self.entries.clear()
