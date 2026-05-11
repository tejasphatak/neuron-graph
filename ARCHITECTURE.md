# Substrate-LLM Architecture

A CPU-native language model built on a single trained transition matrix `W[V, V]`.
No layers, no embeddings, no attention, no backprop. Words are neurons, edges
are trained via perceptron rule.

## Architecture overview

```
  TEXT (corpus during training | prompt during inference)
    │
    ▼
  ┌─────────────────────────────────────────────┐
  │ TOKENIZER (BPE V=15K)                       │
  │   text → integer IDs  e.g. [42, 17, 89, 5]  │
  └────────────────┬────────────────────────────┘
                   │
                   ▼
  ┌─────────────────────────────────────────────────────────┐
  │ SUBSTRATE  W[V, V]                                      │
  │                                                          │
  │     ┌──────────────────────────────────────────┐        │
  │     │  W[i, j] = strength of edge i → j        │        │
  │     │                                          │        │
  │     │  Words are NEURONS.                      │        │
  │     │  Edges are TRAINED via perceptron rule.  │        │
  │     │                                          │        │
  │     │  No embeddings.  No layers.  No matmul   │        │
  │     │  beyond one row sum. No softmax in train.│        │
  │     │                                          │        │
  │     │  225 M parameters at V=15K.              │        │
  │     │  858 MB on disk/RAM.                     │        │
  │     └──────────────────────────────────────────┘        │
  │                                                          │
  └────────────────┬────────────────────────────────────────┘
                   │
                   ▼
  ┌─────────────────────────────────────────────────────────┐
  │ AUGMENTATIONS (optional, plug-in at inference)          │
  │                                                          │
  │   #A  Unigram backoff   — mix in unigram log-probs      │
  │   #C  Negative sampling — train-time pressure           │
  │   #F  kNN-LM datastore  — retrieve similar past contexts│
  │   Hi  Phrase compounds  — PMI-discovered subword units  │
  └─────────────────────────────────────────────────────────┘
```

## Training — one `(context, target)` pair

```
  sequence = [..., t_{i-3}, t_{i-2}, t_{i-1}, t_i, t_{i+1}, ...]
              ───────────────────────────────  ┬─┐
                     context window (4)         │
                                          target = t_{i+1}

  1. PREDICT
     scores = Σ_k=0..3   decay^k · W[ t_{i-k} ]
     pred   = argmax(scores)

  2. COMPARE
     correct?  → no update
     wrong?    → perceptron update:

  3. UPDATE  (for each context token, weighted by decay^k)
     W[ t_{i-k}, target ]  +=  η · decay^k     ← strengthen correct
     W[ t_{i-k}, pred   ]  -=  η · decay^k     ← weaken wrong

  4. NEGATIVE SAMPLE  (#C, optional)
     pick K random non-target tokens
     W[ t_{i-k}, neg_t ]  -=  η · decay^k · α  ← push down

  → no gradients, no backprop, no autodiff.
  → entirely local edge updates.
```

## Inference — generate one token

```
  prompt = "the cat sat on"
                │
                ▼
  tokenize:    ctx_ids = [42, 17, 89, 5]
                │
                ▼  (loop until max_len or EOS)
  ┌───────────────────────────────────────────────────────┐
  │ scores = 1.00·W[5]                  ← most-recent token│
  │        + 0.60·W[89]                                    │
  │        + 0.36·W[17]                                    │
  │        + 0.22·W[42]                  ← oldest of 4 ctx │
  │                                                         │
  │ score[v] += #A unigram_logp[v]      (optional)        │
  │ score[v] += #F kNN_logp[v | ctx]    (optional)        │
  │                                                         │
  │ next_tok = argmax(score)            ← or top-p sample │
  │                                                         │
  │ ctx_ids.append(next_tok); shift left if > window      │
  └───────────────────────────────────────────────────────┘
                │
                ▼
  output: "the cat sat on the mat" → repeat
```

## Comparison slice: Substrate vs Transformer

|                   | **Substrate-LLM**        | **Transformer (GPT-2 small)** |
|-------------------|--------------------------|-------------------------------|
| Mechanism         | single W matrix          | 12 layers, multi-head attn    |
| Parameters        | 225M (V=15K)             | 124M                          |
| Memory            | 858 MB                   | ~500 MB                       |
| Training          | perceptron rule          | backprop + SGD                |
|                   | (CPU, no gradient)       | (GPU, gradient descent)       |
| Inference ops     | ~3 K per token           | ~1 TFLOP per token            |
| Context window    | 4                        | 1024                          |
| WT-103 valid PPL  | 593 (our best)           | ~37.5                         |
| Compute ratio     | ~10⁷× less inference     | baseline                      |
| Inspectable       | yes (each edge labeled)  | no (opaque vectors)           |
| Online learning   | yes (one edge at a time) | no (full retrain)             |
| Phone-deployable  | yes                      | no                            |

## Empirical findings (this research)

What works (additive gains):
- **#A unigram backoff** — 5-10% PPL drop
- **#C negative sampling** during training — small but consistent
- **#F kNN-LM at inference** — ~38% PPL drop (the killer feature)
- **Hierarchical phrase compounds** (PMI-discovered) — ~9% PPL/V improvement

What doesn't work:
- **Multi-W training** (sum of W_next + W_skip + W_back) — naive average HURT PPL.
  Without learned gating between objectives, can't compose cleanly.
- **Naive row-cosine as attention** — paradigmatic ≠ syntagmatic; using
  semantic similarity as context-reweighting degrades next-token prediction.
- **Increasing context_window past 4** — far-back terms add noise without
  learned position-dependent weights.

## What this proves

The architecture **works** — it learns, generates English-like text, develops
emergent semantics (`king − man + woman ≈ queen`), and scales monotonically
with data. It does *not* match transformer PPL because of three structural
limits:

1. Single-layer bigram-style W vs transformer's 12-layer learned attention.
2. Context window = 4 vs 1024.
3. Trained data scale (31M tokens here vs GPT-2's 10B).

The 16× PPL gap against GPT-2 small is real and not closable with smart
coding alone — it requires either real attention layers (which is just
building a transformer) or a fundamentally new mechanism for long-range
modeling.

## What the substrate *is* good for

- Phone-class inference (~10⁷ less compute than transformer LLMs)
- Inspectable predictions (every output token traces to specific edges)
- Online learning without retraining (perceptron update per pair)
- Emergent semantic similarity (row-cosine gives word2vec-class embeddings free)
- Dual-purpose readout (generator + sentence encoder + similarity engine in one W)

The win is **compute and inspectability**, not raw PPL.
