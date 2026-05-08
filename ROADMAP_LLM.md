# Substrate-LLM Roadmap (verified, no assumptions)

## Reference points (from public technical reports and benchmarks)

| Model | Params | Train tokens | Train time | Source |
|---|---|---|---|---|
| **TinyStories** (smallest coherent) | <10M | ~1-2B synthetic | **~2 hrs on RTX 3070Ti** | [Eldan & Li 2023, arxiv 2305.07759](https://arxiv.org/abs/2305.07759) |
| **llama2.c** (Karpathy) | 15-44M | small | CPU inference: 100-150 tok/s | [Karpathy/llama2.c](https://github.com/karpathy/llama2.c) |
| **Qwen 2.5 0.5B** | 500M | ~18T multilingual | proprietary cluster | [Qwen 2.5 Tech Report, arxiv 2412.15115](https://arxiv.org/abs/2412.15115) |
| **Qwen3 0.6B** (smallest 2025) | 600M | ~18T+ | proprietary cluster | [Qwen3 release 2025](https://qwenlm.github.io/) |
| **nanoGPT speedrun** | various | 3.15B | 4 hrs on RTX 4090 (218K tok/s) | [Tyler Romero worklog](https://www.tylerromero.com/posts/nanogpt-speedrun-worklog/) |

## What this means for the substrate

The smallest *useful* transformer LMs are TinyStories-class (~10M params,
~1-2B tokens, generates coherent simple English). Qwen-class small LMs
are ~500M-600M params trained on 18T tokens.

**Our substrate today: 478 KB, 20 sentences trained.** Roughly 333× smaller
than TinyStories' smallest model and 17,000× smaller than Qwen 0.5B.

### The math, with verified anchors

**TinyStories-equivalent substrate:**
- Vocab ~5K-10K BPE tokens
- ~1M-10M sparse edges (active bigrams + n-grams)
- ~50-200 MB on disk (still 5-20× smaller than TinyStories transformer)
- Train data: ~1-2B tokens (their published number)
- CPU time estimate: **dense transformer 2 GPU-hr → substrate edges per-token cost is ~10-50× cheaper than backprop step** → roughly **4-20 CPU hours** single-thread, **30 min - 2 hrs on 8-core parallel**

**Qwen-class (~0.5B-equivalent):**
- Vocab 30K-50K BPE
- ~100M-500M sparse edges
- ~5-20 GB on disk (vs Qwen's 1 GB dense — substrate is bigger because each
  edge is identity-bearing, but trains cheaper per-update)
- Train data: 1-2T tokens (we likely need less than Qwen's 18T because we
  don't need to compress everything into 500M dense weights — substrate
  has explicit edges)
- CPU time: **1-2T tokens ÷ 100K tok/s × parallel** → **~1-2 weeks on 16-core
  CPU**, or 4-8 weeks on 4-core. Local-trainable.

These are honest order-of-magnitude estimates extrapolated from anchor
points, not conjured timelines.

## Phased plan with verifiable milestones

### Phase 1 — TinyStories smoke test (TODAY, hours not days)

**Anchor:** TinyStories paper trains a 10M-param transformer to coherent
English in 2 GPU-hours. We aim for substrate-equivalent quality.

**Components:**
- BPE tokenizer (`tokenizers` lib confirmed available in env)
- TinyStories loader (`datasets` lib confirmed available)
- Bigram-substrate trainer using `fast_train_epoch` pattern from MNIST
- Open-vocab generator using context-window spread

**Milestones (verifiable):**
- **M1.1** — train on 1K stories, generation produces tokens (not crashes). Today.
- **M1.2** — train on 100K stories, perplexity < 20. Today/tomorrow.
- **M1.3** — train on 500K stories (~50M tokens), perplexity < 8, 50-token
  generations rated grammatical by human. Days.

**Compute estimate:** Phase 1 fits comfortably in ≤1 day on 8-core CPU.

### Phase 2 — Multi-core parallel training (1-3 days)

Python multiprocessing with shared-memory numpy W matrix. Each worker
trains a chunk; updates merge via scatter-add (mostly non-conflicting
because each worker's batch touches different active edges).

**Anchor:** numpy with multiprocessing on 16-core CPU typically gets
4-12× scaling on data-parallel tasks (limited by shared-memory contention).

**Milestone M2.1:** 4-8× speedup on TinyStories training vs single-thread.

### Phase 3 — Wikipedia subset (1-2 weeks)

**Anchor:** Wikipedia is ~3B words, simple-Wikipedia subset is ~200M.
At 100K tok/s × 8 cores parallel = ~30 minutes per pass through
simple-Wikipedia.

**Milestones:**
- **M3.1** — train on simple-Wikipedia, factual QA on 100 hand-curated
  questions ≥30% (random ~0%)
- **M3.2** — TriviaQA-easy subset, ≥20% accuracy

### Phase 4 — Phase C performance: mmap brain (1-2 weeks)

Required only if substrate exceeds RAM. For TinyStories: NO. For
Wikipedia subset: probably NO (200MB substrate fits). For full Wikipedia
or web-crawl: YES.

Architecture spec already in the original substrate plan: `mmap` +
`madvise(MADV_RANDOM)` + huge pages. Implementation: 1-2 weeks.

### Phase 5 — Web-scale corpus (2-4 weeks)

**Anchor:** Common Crawl subset (1-2T tokens) is what Qwen-class models
need. Substrate likely needs less because edges are explicit.

**Milestone M5.1:** HellaSwag accuracy ≥50% (random 25%, GPT-2 ~35%).

### Phase 6 — Instruction tuning (2-4 weeks)

**Anchor:** Alpaca/ShareGPT style. Substrate's reward-modulated update
maps cleanly to instruction-following reward.

**Milestone M6.1:** Held-out instruction-following test, ≥60% human-rated.

## Total realistic timeline

**Substrate-LLM that does coherent simple-English generation: 1-2 weeks**
(TinyStories quality + 8-core parallel training).

**Qwen-0.5B-equivalent on simple benchmarks: 2-3 months** (Phases 1-5
complete). Specifically: Wikipedia QA + HellaSwag at small-LM level.

**Local-trainable:** YES throughout. No GPU at any stage. Storage budget
$20-100/month, compute $50-200/month if cloud, $0 if your own machine.

## What we WON'T match without more architecture work

- Transformer in-context learning fluency (latent-space arithmetic)
- Code completion at GPT level
- 100B+ param emergent capabilities

## What we WILL match or beat

- Inference speed on CPU (substrate is sparse, transformers are dense)
- Inspectability (every emission has a spreading-activation trace)
- Online learning (Hebbian + reward, no fine-tuning loop)
- Storage footprint at small/medium scale (sparse edges < dense weights)

## Sources for the verified anchors

- [Eldan & Li 2023 — TinyStories (arxiv 2305.07759)](https://arxiv.org/abs/2305.07759)
- [Qwen 2.5 Technical Report (arxiv 2412.15115)](https://arxiv.org/abs/2412.15115)
- [TinyStories CS224N project, Stanford](https://web.stanford.edu/class/cs224n/final-reports/256911763.pdf)
- [Karpathy llama2.c CPU benchmarks](https://github.com/karpathy/llama2.c)
- [NanoGPT speedrun worklog (Tyler Romero)](https://www.tylerromero.com/posts/nanogpt-speedrun-worklog/)
- [Qwen 3 release blog](https://qwenlm.github.io/)
