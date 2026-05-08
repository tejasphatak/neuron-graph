"""Replay buffer — offline re-experience for sample-efficient learning.

Brains consolidate during sleep: hippocampus replays recent episodes
to cortex, strengthening the patterns that mattered. Without this,
each experience is one-shot and rare events are forgotten.

ReplayBuffer holds a ring of recent (trajectory, reward) tuples.
consolidate() samples from the buffer and re-applies credit assignment
at reduced learning rate. Should run periodically, not every step.
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, List, Tuple


@dataclass
class Episode:
    """A single recorded trajectory + its reward + metadata."""
    trajectory: List[Tuple[Any, int]]    # (state, action) pairs
    reward: float
    metadata: dict = field(default_factory=dict)


@dataclass
class ReplayBuffer:
    capacity: int = 1000
    episodes: Deque[Episode] = field(default_factory=lambda: deque(maxlen=1000))

    def __post_init__(self):
        if self.episodes.maxlen != self.capacity:
            self.episodes = deque(self.episodes, maxlen=self.capacity)

    def record(self, trajectory: List[Tuple[Any, int]],
                reward: float, **meta) -> None:
        self.episodes.append(Episode(
            trajectory=list(trajectory),
            reward=reward,
            metadata=dict(meta),
        ))

    def sample(self, n: int, *, prefer_recent: bool = False,
                rng: random.Random = None) -> List[Episode]:
        if not self.episodes:
            return []
        rng = rng or random.Random()
        if prefer_recent:
            # Bias toward recent episodes (last quarter): exponential weighting
            # Sample without replacement, weights by recency
            ep_list = list(self.episodes)
            n_eps = len(ep_list)
            weights = [1.5 ** (i / n_eps) for i in range(n_eps)]
            return rng.choices(ep_list, weights=weights, k=min(n, n_eps))
        return rng.sample(list(self.episodes), k=min(n, len(self.episodes)))

    def stats(self) -> dict:
        if not self.episodes:
            return {'n': 0, 'avg_reward': 0.0, 'avg_horizon': 0.0}
        rewards = [e.reward for e in self.episodes]
        horizons = [len(e.trajectory) for e in self.episodes]
        return {
            'n': len(self.episodes),
            'avg_reward': sum(rewards) / len(rewards),
            'avg_horizon': sum(horizons) / len(horizons),
            'min_reward': min(rewards),
            'max_reward': max(rewards),
        }

    def __len__(self) -> int:
        return len(self.episodes)


def consolidate(buffer: ReplayBuffer, n_samples: int,
                 credit_fn: Callable[[Episode], None],
                 *, prefer_recent: bool = False,
                 rng: random.Random = None) -> dict:
    """Sample N episodes from the buffer; apply `credit_fn` to each.

    `credit_fn(episode)` should call the brain's credit_trajectory with
    appropriately-scaled eta (typically reduced for replay vs first-time).

    Returns a stats dict.
    """
    samples = buffer.sample(n_samples, prefer_recent=prefer_recent, rng=rng)
    for ep in samples:
        credit_fn(ep)
    return {'n_replayed': len(samples), 'buffer_size': len(buffer)}
