"""TinyStories experiment for substrate-LLM Phase 1.

Anchor: Eldan & Li 2023 — 10M-param transformer reaches coherent English
on this dataset in ~2 GPU-hours. We aim for substrate-equivalent quality,
training on CPU only.

Run:
  PYTHONPATH=. python3 brain/tasks/llm/experiment.py

Env vars:
  TRAIN_N=1000   — number of stories to train on
  TEST_N=200     — held-out for perplexity
  EPOCHS=5
  ETA=0.05       — learning rate
  CONTEXT=4      — context window
  VOCAB=2000     — max vocab size (tokenizer pruning)
"""

from __future__ import annotations

import os
import time

import numpy as np

from brain.tasks.llm.tokenizer import WordTokenizer
from brain.tasks.llm.llm import (
    build_llm_view, train_ngram_epoch, generate_text, perplexity,
)


def load_tinystories(*, train_n: int = 1000, test_n: int = 200,
                       seed: int = 0):
    """Load TinyStories via HuggingFace datasets.
    First call downloads ~1-2 GB. Cached after."""
    from datasets import load_dataset
    print(f'Loading TinyStories (train_n={train_n}, test_n={test_n})...')
    ds = load_dataset('roneneldan/TinyStories',
                       split=f'train[:{train_n + test_n}]')
    texts = list(ds['text'])
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(texts))
    train_texts = [texts[i] for i in idx[:train_n]]
    test_texts = [texts[i] for i in idx[train_n:train_n + test_n]]
    return train_texts, test_texts


def main():
    train_n = int(os.environ.get('TRAIN_N', '1000'))
    test_n = int(os.environ.get('TEST_N', '200'))
    epochs = int(os.environ.get('EPOCHS', '5'))
    eta = float(os.environ.get('ETA', '0.05'))
    context_window = int(os.environ.get('CONTEXT', '4'))
    max_vocab = int(os.environ.get('VOCAB', '2000'))

    train_texts, test_texts = load_tinystories(train_n=train_n, test_n=test_n)
    print(f'Train texts: {len(train_texts)}, Test: {len(test_texts)}')
    print(f'Sample: {train_texts[0][:120]!r}')

    # Tokenizer
    print(f'\nFitting tokenizer (max_vocab={max_vocab})...')
    t0 = time.time()
    tokenizer = WordTokenizer(max_vocab=max_vocab, min_freq=2)
    tokenizer.fit(train_texts)
    print(f'  Vocab: {tokenizer.get_vocab_size()} tokens '
          f'(time: {time.time()-t0:.1f}s)')

    # Substrate
    view = build_llm_view(tokenizer)
    V = tokenizer.get_vocab_size()
    print(f'  Substrate: W[{V}, {V}] = {view.W.nbytes / 1024 / 1024:.1f} MB')

    # Encode
    print(f'\nEncoding training set...')
    t0 = time.time()
    train_seqs = [tokenizer.encode(t) for t in train_texts]
    test_seqs = [tokenizer.encode(t) for t in test_texts]
    n_train_tokens = sum(len(s) for s in train_seqs)
    print(f'  Train tokens: {n_train_tokens:,}  '
          f'time: {time.time()-t0:.1f}s')

    # Pre-train PPL (random W)
    print(f'\nCold-start PPL (W=0):')
    pre_ppl = perplexity(view, test_seqs[:50], context_window=context_window)
    print(f'  PPL: {pre_ppl:.2f} (uniform = {V})')

    # Train
    print(f'\n=== Training ({epochs} epochs, eta={eta}, ctx={context_window}) ===')
    rng = np.random.default_rng(0)
    best_ppl = pre_ppl
    for ep in range(epochs):
        t0 = time.time()
        m = train_ngram_epoch(view, train_seqs,
                                context_window=context_window,
                                eta=eta, rng=rng)
        elapsed = time.time() - t0
        ppl = perplexity(view, test_seqs[:50], context_window=context_window)
        if ppl < best_ppl:
            best_ppl = ppl
        print(f'  ep {ep+1}/{epochs}  '
              f'next_tok_acc={m["next_token_accuracy"]:.1%}  '
              f'pairs={m["n_pairs"]:,}  '
              f'PPL={ppl:.2f}  '
              f'(best {best_ppl:.2f})  '
              f'time={elapsed:.1f}s '
              f'({m["n_pairs"]/elapsed:,.0f} tok/s)')

    # Final PPL on full test set
    final_ppl = perplexity(view, test_seqs, context_window=context_window)
    print(f'\nFinal test PPL (full {len(test_seqs)} stories): {final_ppl:.2f}')

    # Sample generations
    print(f'\n=== Sample generations (greedy, ctx={context_window}) ===')
    test_prompts = [
        'once upon a time',
        'the little girl',
        'the boy went',
    ]
    for prompt in test_prompts:
        out = generate_text(view, tokenizer, prompt,
                             max_new=20, temperature=0.0,
                             context_window=context_window, rng=rng)
        print(f'  "{prompt}"')
        print(f'  → "{out}"')

    # Sample with temperature
    print(f'\n=== Sample generations (T=0.7, top_k=10) ===')
    for prompt in test_prompts[:2]:
        out = generate_text(view, tokenizer, prompt,
                             max_new=20, temperature=0.7, top_k=10,
                             context_window=context_window,
                             rng=np.random.default_rng(0))
        print(f'  "{prompt}"')
        print(f'  → "{out}"')


if __name__ == '__main__':
    main()
