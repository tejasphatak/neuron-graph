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
    train_bigram_epoch, train_ngram_epoch,
    train_ngram_epoch_batched, train_ngram_epoch_fast,
    generate_text, perplexity,
    compute_unigram_log_probs, perplexity_with_backoff,
)
from .jit import (
    train_ngram_epoch_jit, train_ngram_epoch_jit_negsample, NUMBA_AVAILABLE,
)
from .substrate_predict import (
    view_to_brain, perplexity_with_spread, perplexity_with_spread_and_backoff,
)
from .knn_lm import build_knn_datastore, perplexity_with_knn
from .eval_tasks import (
    cloze_benchmark, sentence_completion_eval, random_baseline_cloze,
)

__all__ = [
    'WordTokenizer',
    'LLMVocab', 'build_llm_brain', 'build_llm_view', 'LLMView',
    'train_bigram_epoch', 'train_ngram_epoch',
    'train_ngram_epoch_batched', 'train_ngram_epoch_fast',
    'train_ngram_epoch_jit', 'train_ngram_epoch_jit_negsample',
    'NUMBA_AVAILABLE',
    'generate_text', 'perplexity',
    'compute_unigram_log_probs', 'perplexity_with_backoff',
    'view_to_brain', 'perplexity_with_spread',
    'perplexity_with_spread_and_backoff',
    'build_knn_datastore', 'perplexity_with_knn',
    'cloze_benchmark', 'sentence_completion_eval', 'random_baseline_cloze',
]
