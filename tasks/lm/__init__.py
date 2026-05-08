"""Language modeling on the substrate — does spreading activation +
`follows` edges + working memory produce coherent autoregressive
next-token prediction?

Smallest test: train on one 9-word sentence, prompt with 2 tokens,
verify substrate reproduces the remaining 7. See `tiny.py`.
"""

from .tiny import (
    Vocab, build_lm_brain, teach_sentence,
    generate, generate_via_spread, generate_open_vocab,
)
from .rl import (
    teach_minimal, trace_generate, reward_trajectory,
    apply_correction, train_rl, train_rl_curriculum,
    GenerationStep, Trajectory,
)

__all__ = [
    'Vocab', 'build_lm_brain', 'teach_sentence',
    'generate', 'generate_via_spread', 'generate_open_vocab',
    'teach_minimal', 'trace_generate', 'reward_trajectory',
    'apply_correction', 'train_rl', 'train_rl_curriculum',
    'GenerationStep', 'Trajectory',
]
