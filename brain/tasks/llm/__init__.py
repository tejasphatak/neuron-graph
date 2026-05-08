"""Substrate-LLM — open-vocabulary text language modeling.

Different from brain/tasks/lm/ (sentence retrieval). This task does
n-gram-style next-token prediction with no teacher forcing, on real
text corpora. The path to substrate-as-LLM.

Pipeline:
  text → tokenize → context window → next-token prediction
  Each token = one neuron (substrate vocabulary)
  Each (context_token → next_token) pair = ACTIVATES edge
  Reward: +1 if predicted token = actual next token, -1 otherwise
  Training: vectorized batch update over (context, target) pairs

Local-trainable on CPU. Parallelizable across cores.
"""

from .tokenizer import WordTokenizer
from .llm import (
    LLMVocab, build_llm_brain, build_llm_view, LLMView,
    train_bigram_epoch, train_ngram_epoch, train_ngram_epoch_batched,
    generate_text, perplexity,
)

__all__ = [
    'WordTokenizer',
    'LLMVocab', 'build_llm_brain', 'build_llm_view', 'LLMView',
    'train_bigram_epoch', 'train_ngram_epoch', 'train_ngram_epoch_batched',
    'generate_text', 'perplexity',
]
