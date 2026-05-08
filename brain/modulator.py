"""Modulatory layer — global scalar that biases learning rate.

Biological inspiration: dopamine. After a positively-rewarded outcome,
the brain temporarily increases plasticity for related synapses. After
a negatively-rewarded one, plasticity drops. This is a coarse global
modulation, not per-synapse.

Implementation: a single float on the Brain that scales eta in all
weight updates. Decays back to neutral over time. Updated by the agent
after each reward signal.

eta_effective = eta_base * (1 + modulator)

Bounded so that catastrophically bad updates can't multiply errors.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Modulator:
    """Global plasticity modulator. Biological analog: dopamine level."""
    value: float = 0.0          # current modulator (0 = neutral)
    decay: float = 0.95         # per-update decay back to 0
    min_value: float = -0.5
    max_value: float = 1.0

    def adjust(self, reward_signal: float, *, scale: float = 0.3) -> float:
        """Adjust modulator based on a reward signal in [-1, +1].
        Returns the new value."""
        # Decay first (fades old signal)
        self.value *= self.decay
        # Add fresh contribution
        self.value += reward_signal * scale
        # Clamp
        self.value = max(self.min_value, min(self.max_value, self.value))
        return self.value

    def effective_eta(self, base_eta: float) -> float:
        """Scale a base learning rate by current modulator."""
        return base_eta * max(0.0, 1.0 + self.value)

    def reset(self) -> None:
        self.value = 0.0
