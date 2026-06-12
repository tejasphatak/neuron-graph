# Forward-Write LM — design sketch

**Learning by repeated exposure, forward-only, no backprop anywhere.** Each
forward pass nudges the weights toward reproducing what it just saw; repetition
makes the traveled paths heavy; the astrocyte memory is the store; QKV/softmax
(already proven ≡ attention) is the read. The open half is the **write**.

This is the Tier-2 destination reached from the *forward-only repetition* angle
instead of the cleanup angle — and it is shaped directly by today's Tier-1
negative result.

---

## The principle that keeps it honest

> A system that learns to *reproduce its input* is a tape recorder — PPL→1 on what
> it saw, useless on unseen text. The Tier-1 result proved associative cleanup on
> the *token distribution* destroys generalization. **So generalization cannot
> live in the storage. It must live in the representation.**

Therefore the design splits cleanly:

| component | job | trained? |
|---|---|---|
| **Reservoir** (fixed random recurrent net) | turn context → distributed generalizing features | **NO — fixed random** |
| **Readout** (associative memory) | features → next-token distribution | **YES — forward-only, repetition** |
| **Softmax** | retrieval = attention | (proven, no params) |

Nothing uses backprop. The reservoir is never trained (reservoir computing /
echo-state principle); the readout learns by a *local* rule.

---

## Architecture

```
token x_t ──embed(fixed random E)──► u_t ∈ R^d
                                       │
        h_{t-1} ──────────────────────┤   reservoir state (distributed context)
                                       ▼
   h_t = tanh( W_res · h_{t-1} + W_in · u_t )      W_res, W_in FIXED RANDOM (spectral radius < 1)
                                       │            h_t ∈ R^D  carries UNBOUNDED context
                                       ▼
   scores = W_out · h_t                            W_out (V×D) = the ASSOCIATIVE MEMORY
   p(next) = softmax(scores)                       ← the read we already proved = attention
```

`h_t` is the whole point: a fixed random recurrent map gives a distributed,
nonlinear, *unbounded-history* representation of the context — for free, no
training. Two different contexts that *mean* something similar land near each
other in `h`-space, so the readout generalizes to contexts it never saw. This is
exactly what the bag-of-4-tokens context in the current substrate-LLM cannot do.

---

## The write — surprise-gated Hebbian, once per forward pass

The readout `W_out` is the only thing that learns. The rule is the **delta rule**
(Widrow–Hoff / LMS) — local, single-layer, no backprop:

```
each step t, having emitted p_t and seen the true next token y_t:
    e_t   = onehot(y_t) − p_t            # surprise (prediction error), V-dim
    W_out += η · e_t · h_tᵀ              # Hebbian outer product: strengthen feature→token
```

- **Local & forward-only:** `e_t` is just (target − output); no gradient flows
  backward through the reservoir (it's fixed, and the readout is one layer so the
  "gradient" is the error itself). This is biologically the three-factor rule:
  pre = `h_t`, post = token unit, modulator = surprise `e_t`.
- **Repetition is the engine.** First pass over a passage → large `e` → big write.
  Re-read it → `p` already closer → smaller `e` → smaller write. Paths get heavy;
  surprise decays toward zero. Convergence = "it has learned this text."
- **Equivalent to the astrocyte write:** each row of `W_out` is a stored
  astrocyte pattern (token μ's feature-island); the outer-product update *is*
  appending/strengthening that astrocyte. Softmax retrieval = degree-n=softmax
  DAM = attention. The whole object is the neuron-astrocyte memory with a
  reservoir front-end.

**Why the readout stays degree-1/softmax, not high-degree:** Tier-1 showed
high-degree cleanup collapses the token mixture and *raises* PPL. So we keep the
output linear+softmax (= attention) and push all the representational richness
into the reservoir. The Tier-1 negative is not a setback here — it tells us
exactly where *not* to put the nonlinearity.

---

## Generation

Greedy/temperature sample from `softmax(W_out · h_t)`, feed the sampled token back
in as `x_{t+1}`, roll the reservoir forward. Optionally keep writing during
generation (continual adaptation) or freeze `W_out` (pure inference).

---

## The falsifiable test (cheap, same harness as Tier-1)

Train by repeated exposure (a few epochs of forward passes with the delta write)
on TinyStories; measure **held-out** PPL. Baselines, all forward-only:

| model | context model | learns via | what it isolates |
|---|---|---|---|
| unigram | none | counts | floor |
| substrate-LLM (existing) | bag-of-4 tokens, W (V×V) | perceptron (already forward-only) | the current substrate |
| **Forward-Write (this)** | **reservoir h_t (unbounded)** | **delta rule** | does distributed recurrent context help? |

**The claim under test:** the reservoir's unbounded distributed context lets the
forward-only readout generalize better than the bag-of-N substrate-LLM → **lower
held-out PPL.** If held-out PPL beats the bag-of-N model → distributed forward-only
representation is the lever. If it only matches/loses → random features aren't
rich enough for language, and the representation must be *learned* (which costs
backprop, breaking the pure-forward dream — itself a clean finding).

---

## Honest risks (where this dies)

1. **Random features may be too weak for language.** Reservoirs shine on
   low-dimensional dynamical signals; high-vocab token prediction may need
   *structured* (learned) features. This is the real risk and the most likely
   failure mode. Mitigation knobs: reservoir size D, spectral radius, input
   scaling, leaky integration — all cheap to sweep, none involve training.
2. **The readout ceiling is a linear model on fixed features.** The delta rule
   converges to the least-squares readout — so the ceiling is "best linear
   readout on random recurrent features." That can beat n-grams (unbounded
   context) but will sit far below a trained transformer. The honest target is
   *beat the bag-of-N substrate-LLM*, not GPT.
3. **Catastrophic interference** in `W_out` across passages (one corpus
   overwriting another). The capacity law (degree, D) governs how much it holds
   before interference — directly the basin/width result, now on a real readout.

---

## Implementation plan (~1 afternoon, pure numpy, no GPU)

```
brain/tasks/llm/forward_write.py
    class Reservoir:           # fixed random ESN — h_t = tanh(W_res h + W_in E[x])
        step(x_t) -> h_t
        reset()
    class ForwardWriteLM:
        reservoir, W_out (V×D)
        predict(h) -> softmax(W_out h)
        observe(h, y, eta)     # delta-rule write: W_out += eta * (onehot(y)-p) h^T
        train_epoch(seqs)      # roll reservoir, predict, write — forward only
        perplexity(seqs)       # held-out, write OFF
    main()                     # TinyStories, sweep D / spectral radius, vs baselines
tests/test_forward_write.py
    - reservoir is contractive (||h|| bounded; echo-state property)
    - repeated exposure on one passage drives its train-PPL down monotonically
    - held-out PPL finite; beats unigram floor
```

Reuses the Tier-1 harness (`_build_trained_view`'s loader, tokenizer, TinyStories
cache) and the existing `perplexity` softmax machinery. The astrocyte tie-in:
`W_out` rows are astrocyte patterns; `observe()` is the one-shot Hebbian store;
softmax read = attention. Same object, reservoir front-end, learned by repetition.

---

## Result (v1, TinyStories, 600 train / 100 test stories, V=1500, D=400, 4 epochs)

Held-out PPL, every model at its **own best temperature** (the Tier-1 control —
softmax sharpness must not confound the comparison), identical eval pipeline:

```
unigram floor            194.31
substrate-LLM bag-of-4   340.72   @T=0.25     forward-only perceptron, bag-of-4 tokens
Forward-Write (ours)     124.42   @T=2.0      forward-only delta rule, reservoir context
```

**Forward-Write beats both the floor and a temperature-matched bag-of-4.** No
backprop anywhere: fixed random reservoir + delta-rule writes + repeated exposure.
Generalization comes from the reservoir representation, not the store — exactly as
the Tier-1 negative predicted it had to. Train surprise is flat across epochs
(4.74 → 3.98): a delta rule converges in ~one pass.

**Honest scope (do not oversell):**
1. Small scale. The bag-of-4 here is data-starved — it is *worse than unigram*
   (sparse 4-gram stats over 600 stories). With 1000 stories + score
   standardization, the same bag-of-4 reached **131** in the Tier-1 study, so
   against a *maximally-tuned* bag-of-4, Forward-Write (124) is roughly
   *competitive*, not 63% ahead. Robust claim: **beats the floor and a like-for-like
   bag-of-4**; the gap to a best-case bag-of-4 is small and not yet measured under
   identical conditions.
2. PPL 124 is not a good LM (TinyStories transformers reach single-to-low-double
   digits). The win is *among forward-only, no-backprop methods* — it does not
   challenge backprop-trained models.
3. The readout ceiling is a linear model on fixed random features (the delta rule
   converges to least-squares on the reservoir). The reservoir-too-weak risk is
   real; the next lever is a richer / partially-structured reservoir, or a larger
   D, both still backprop-free.

**Next, to make the headline bulletproof:** Forward-Write vs the *best-case*
bag-of-4 (1000+ stories, standardized eval, best-temp) under one identical harness;
sweep D and spectral radius; confirm the reservoir advantage survives more data.

## One-line summary

**Echo-state reservoir (fixed, generalizing features) + delta-rule associative
readout (forward-only, repetition-driven) = the neuron-astrocyte memory turned
into a language model with no backprop.** Read is proven (attention); write is the
delta rule; the Tier-1 negative tells us to keep the output linear and put the
richness in the reservoir. Falsified or confirmed by one held-out-PPL run.
