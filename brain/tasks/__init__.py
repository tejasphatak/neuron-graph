"""Domain applications of the substrate.

Each subpackage demonstrates a different task built on top of `brain.*`
substrate primitives. The substrate itself has no domain knowledge —
neurons + edges + spreading + plasticity. Tasks bring:
  - state encoding (input → neuron activation)
  - action / output decoding
  - reward / supervision signal
  - task-specific tests + demos

Current tasks:
    brain.tasks.ttt — Tic-Tac-Toe: bandit, naive RL agent, world model,
                       planning agent, learned value head. The proving
                       ground for the substrate's RL stack.

Planned next:
    brain.tasks.mnist — image classification (modality polymorphism test)
"""

__all__ = ['ttt']
