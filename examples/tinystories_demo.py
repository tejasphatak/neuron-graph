"""End-to-end substrate-LLM demo on TinyStories.

Trains the converged best architecture, prints PPL + cloze + generations.
Reproduces the headline numbers in the README.

Run:
  cd neuron-graph
  PYTHONPATH=. python3 examples/tinystories_demo.py

Adjust knobs via env:
  TRAIN_N=1000   number of stories (smaller = faster smoke test)
  EPOCHS=10
  VOCAB=2500
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import load_dataset

from brain.tasks.llm import (
    WordTokenizer, build_llm_view, generate_text,
    compute_unigram_log_probs, perplexity_with_backoff,
    build_knn_datastore, perplexity_with_knn,
    cloze_benchmark, random_baseline_cloze,
)
from brain.tasks.llm.jit import train_ngram_epoch_jit_negsample


def main():
    train_n = int(os.environ.get('TRAIN_N', '5000'))
    epochs = int(os.environ.get('EPOCHS', '10'))
    max_vocab = int(os.environ.get('VOCAB', '2500'))

    print(f'Loading TinyStories ({train_n} stories)...')
    ds = load_dataset('roneneldan/TinyStories',
                      split=f'train[:{train_n + 200}]')
    texts = list(ds['text'])[:train_n + 200]
    train_texts = texts[:train_n]
    test_texts = texts[train_n:train_n + 200]

    tokenizer = WordTokenizer(max_vocab=max_vocab, min_freq=2)
    tokenizer.fit(train_texts)
    V = tokenizer.get_vocab_size()
    print(f'Vocab: {V}')

    train_seqs = [tokenizer.encode(t) for t in train_texts]
    test_seqs = [tokenizer.encode(t) for t in test_texts]

    view = build_llm_view(tokenizer)

    print(f'\n=== TRAINING ({epochs} epochs, #C negsample) ===')
    rng = np.random.default_rng(0)
    t0 = time.time()
    for ep in range(epochs):
        train_ngram_epoch_jit_negsample(
            view, train_seqs, context_window=4, eta=0.05,
            neg_samples=3, neg_scale=0.1, rng=rng,
        )
    print(f'Trained in {time.time() - t0:.0f}s')

    # PPL with #A unigram backoff + #F kNN datastore (the converged best)
    print(f'\n=== PERPLEXITY ===')
    uni = compute_unigram_log_probs(view, train_seqs)
    ppl_baseline = perplexity_with_backoff(view, test_seqs, uni, alpha=1.0,
                                             context_window=4)
    ppl_a = perplexity_with_backoff(view, test_seqs, uni, alpha=0.5,
                                       context_window=4)
    ds_X, ds_y = build_knn_datastore(view, train_seqs, n_samples=10000,
                                        context_window=4, decay=0.6,
                                        rng=np.random.default_rng(1))
    ppl_acf = perplexity_with_knn(view, test_seqs[:50], ds_X, ds_y, uni,
                                     k_neighbors=50, alpha_knn=0.3,
                                     alpha_uni=0.5, temperature=5.0,
                                     context_window=4)
    print(f'  Baseline (W only):       PPL = {ppl_baseline:.1f}')
    print(f'  + #A unigram backoff:    PPL = {ppl_a:.1f}')
    print(f'  + #C+#F (full stack):    PPL = {ppl_acf:.1f}')
    print(f'  Uniform vocab baseline:  PPL = {V}')

    # Cloze benchmark
    print(f'\n=== CLOZE BENCHMARK (the real utility test) ===')
    cz = cloze_benchmark(view, tokenizer, test_seqs, n_samples=500,
                          context_window=4, rng=np.random.default_rng(42))
    rand = random_baseline_cloze(V)
    print(f'  Top-1 acc:  {cz["top1_acc"]:.2%}   (random: {rand["top1_acc"]:.3%})')
    print(f'  Top-5 acc:  {cz["top5_acc"]:.2%}   (random: {rand["top5_acc"]:.3%})')
    print(f'  Top-10 acc: {cz["top10_acc"]:.2%}   (random: {rand["top10_acc"]:.3%})')
    print(f'  Median rank of true target: {cz["median_rank"]:.0f} of {V}')

    # Generations
    print(f'\n=== GENERATIONS (T=0.7, top_p=0.9, rep_penalty=1.3) ===')
    for prompt in ['once upon a time', 'the little girl', 'tom and his',
                    'one sunny day']:
        rng2 = np.random.default_rng(42)
        out = generate_text(view, tokenizer, prompt, max_new=20,
                              temperature=0.7, top_p=0.9,
                              context_window=4, repetition_penalty=1.3,
                              no_repeat_ngram=3, rng=rng2)
        print(f'  {prompt!r:24} → {out!r}')

    print(f'\nDone. Substrate uses ~{view.W.nbytes / 1024 / 1024:.0f} MB.')
    print(f'Repo: github.com/tejasphatak/neuron-graph')


if __name__ == '__main__':
    main()
