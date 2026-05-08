"""Language modeling on the substrate — does spreading activation +
`follows` edges + working memory produce coherent autoregressive
next-token prediction?

Smallest test: train on one 9-word sentence, prompt with 2 tokens,
verify substrate reproduces the remaining 7. See `tiny.py`.
"""

from .tiny import (
    Vocab, build_lm_brain, teach_sentence, generate,
)

__all__ = ['Vocab', 'build_lm_brain', 'teach_sentence', 'generate']
