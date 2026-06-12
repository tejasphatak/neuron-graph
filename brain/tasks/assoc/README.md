# Associative memory — neuron-astrocyte, on the substrate

Bipartite, no-backprop Dense Associative Memory, after Kozachkov, Slotine &
Krotov, *"Neuron-Astrocyte Associative Memory"* (PNAS 2025; arXiv:2311.08135),
wired onto the neuron-graph substrate.

The K stored patterns **are** the astrocytes. Each astrocyte μ is a real neuron
whose outgoing CSR synapse block is its synaptic island — one edge
`astrocyte μ → neuron i` carrying weight `ξ_iᵘ`. Retrieval is two passes over
that bipartite graph, both implemented as sparse edge multiply-accumulate (no
matmul on the critical path — the same discipline as `spread()`):

```
overlap_μ = Σ_i  ξ_iᵘ · φ(x_i)     # GATHER  (neuron → astrocyte calcium)
glio_μ    = F′(overlap_μ)           # gliotransmitter nonlinearity
x_new_i   = Σ_μ ξ_iᵘ · glio_μ       # SCATTER (astrocyte → neuron)
```

The N⁴ process-coupling tensor `T_ijkl` is never materialized; no gradient is
used to store memories (writing one appends an astrocyte neuron).

- `interaction="poly", degree=n` → `F′(z)=z^(n-1)`. n=2 classic Hopfield (~N),
  n=4 the paper's quartic (~N³).
- `interaction="softmax"` → the layer reduces to transformer self-attention.

See `brain/astrocyte.py` for `NeuronAstrocyteMemory` (dense reference) and
`SubstrateAstrocyteMemory` (the substrate wiring).

## Results (`python -m brain.tasks.assoc.experiments`, verified)

**Substrate ≡ dense reference.** Substrate gather/scatter over real CSR synapse
blocks reproduces the dense numpy reference exactly at full connectivity
(`test_astrocyte.py`: sign output bit-equal; softmax max abs error < 1e-6).

**Capacity scaling** — empirical `K_max` vs `N`, recall ≥ 90% from a
10%-corrupted cue, run through the substrate:

| regime | log-log slope | K_max/N |
|---|---|---|
| degree-2 (classic Hopfield) | **0.92** (theory ~1) | flat ~0.12 |
| degree-4 (paper's quartic)  | **2.34** (theory ~3) | **climbs 1.7 → 10.2** (N: 16 → 64) |

The pairwise model stores a constant number of memories per unit; the
astrocyte-mediated quartic stores a growing number — the paper's central claim,
reproduced on the substrate. The measured exponent sits below the asymptotic 3
because this is a strict basin-of-attraction measurement at finite N.

**Attention equivalence** — softmax interaction vs `softmax(QKᵀ/√d)V`:
mean cosine = 1.000000, max abs error ≈ 3e-8. Self-attention, with zero learned
process-to-process weights, over real substrate edges.

**Density sweep — the open experiment the paper points at but does not run.**
`r = K/N` (process-to-process connectivity) is here a genuine substrate knob:
the neighbor-set size of each astrocyte. Recall fidelity (fraction of bits
recovered) over an `(r, K)` grid, N=64, degree-4:

```
 r = K/N  edges | K=2   K=8   K=32  K=128
    1.00     64 | 1.00  1.00  1.00  1.00
    0.90     58 | 0.96  0.97  0.96  0.98
    0.75     48 | 0.89  0.89  0.91  0.92
    0.50     32 | 0.78  0.77  0.81  0.88
    0.25     16 | 0.65  0.66  0.72  0.84
```

The non-obvious finding: in the sparse regime fidelity is **flat or rising in K
but falls sharply in r**. The bottleneck is *coordinate coverage* (island size),
not pattern interference — a coordinate outside a target's island gets no drive
and never recovers, while *more* astrocytes cover *more* coordinates. Realized
capacity is set by island size, not pattern count.

## Reasoning probes — can the substrate compose, not just retrieve?

Retrieval alone can only match context, never compose over it. These probes test
the capabilities multi-step reasoning decomposes into. Each has a **held-out
combinatorial split** (so memorization scores at chance) and a **1-hop retrieval
foil** (structurally unable to answer held-out combinations). The
substrate−foil gap on held-out items is the signal.

**`reasoning_probe.py`** — four rungs (`python -m brain.tasks.assoc.reasoning_probe`):

| rung | capability | substrate | foil | chance |
|---|---|---|---|---|
| 1 chaining | transitive inference (multi-hop spread, with interference) | **0.79** | 0.18 | 0.20 |
| 2 binding | alias→var→value dereference | **1.00** | 0.24 | 0.25 |
| 3 generalization | held-out verb×direction combinations | **1.00** | 0.00 | 0.12 |
| 4 algorithmic | FSM state-tracking, length generalization | **1.00** | 0.39 | 0.25 |

The foil is pinned at chance on every held-out split; the substrate composes.
Verdict: the substrate **has the compositional primitives** retrieval lacks.
Caveat: rungs 2–4 hit 1.00 because those synthetics are noise-free — they prove
the mechanism exists, not that it is robust. That is what the next probe stresses.

**`compounding_probe.py`** — the hard-mode gate
(`python -m brain.tasks.assoc.compounding_probe`). A distributed-state FSM run in
vector space through a learned linear transition `W_x`, whose crosstalk is the
per-step error. Without cleanup, errors compound; with degree-4 DAM `retrieve()`
each step, the attractor corrects the drift.

```
COMPOUNDING (M=24, d=64), accuracy vs chain length:
    len N            1     2     4     8    16    32
    OFF (feedfwd)  1.00  0.98  0.75  0.55  0.15  0.07   <- tracks p^N (p=0.92)
    pred p^N       0.92  0.84  0.71  0.51  0.26  0.07
    ON  (cleanup)  1.00  1.00  1.00  1.00  1.00  1.00   <- compounding defeated

LOAD (N=16), accuracy vs #states M:
    M              16    24    32    48    64    96
    OFF          0.95  0.27  0.35  0.02  0.05  0.05
    ON           1.00  1.00  0.92  0.92  0.85  0.17   <- basin holds to M~48-64
```

**The result:** feedforward error compounds as `p^N` (a chain right 2% of the
time at M=48); associative cleanup defeats it (92% at M=48), flat to length 32
within the basin. **Reasoning-chain length is bounded by per-step drift vs basin
size, not by depth itself.** Past the load ceiling (~M=48–64 at d=64) single-step
crosstalk exceeds the basin and even cleanup compounds. The concrete lever for
longer reasoning is enlarging the basin (degree / d / encoding).

**`basin_scaling.py`** — what governs the basin / load ceiling
(`python -m brain.tasks.assoc.basin_scaling`). Measures the ceiling (largest load
M whose length-16 chain still recalls ≥90% with cleanup) vs the two knobs:

```
Ceiling vs DEGREE (d=64):     n=2 -> M=16   n=4 -> M=48   n=6 -> M=48
Ceiling vs DIMENSION (deg=4): d=32 -> M=16  d=64 -> M=48  d=128 -> M=128
                              ceiling/d:    0.50          0.75          1.00
```

Degree lifts the ceiling 2→4 (classic Hopfield's basins are too small for long
chains) but **saturates** past 4 — beyond degree-4 the limit is the transition
crosstalk, not the DAM. **Dimension is the dominant, super-linear lever**:
`ceiling/d` rises 0.50→0.75→1.00, and at d=128 the chain holds M=d states *and
beyond*. The substrate buys longer reasoning through **width (representation
dimension), not depth** — it self-corrects, so you enlarge the basin rather than
stack layers.

## The LLM bridge — Tier 1: does the capacity law touch real language?

Everything above is on synthetic patterns. The bridge test: the existing
substrate-LLM (`brain/tasks/llm/llm.py`) readout is **degree-1 and linear** —
`scores = Σ_k decay^k · W[ctx_row_k]`. That sum *is* the astrocyte gather at
degree 1, with no nonlinearity and no scatter. Tier 1 lifts it to the full
degree-n DAM pass over the **same trained W** (each vocabulary token is an
astrocyte; its synaptic island is its W row), zero retraining:

```
overlap_μ = Σ_i W[μ,i]·c_i ;   glio_μ = sign·|overlap_μ|^(n-1) ;   c_clean = Σ_μ W[μ,i]·glio_μ
```

Hypothesis: if `capacity ~ N^(n-1)` transfers, held-out PPL should fall as
degree rises. **It does not.** `brain/tasks/llm/tier1_dam_readout.py`, TinyStories,
V=1500, 5 epochs, **best-temperature** per readout (the fair control — each
degree's power nonlinearity peaks the score distribution differently, so a fixed
softmax temperature confounds shape with sharpness):

```
  readout       best PPL   @temp     vs deg1
  deg1 (linear)   131.66   0.25     —          <- best
  deg4            140.71   0.25    +6.9%
  deg3            178.26   0.5    +35.4%
  deg2            267.91   2.0   +103.5%
  softmax         424.24   0.5   +222.2%
```

(At a *fixed* temperature=1.0, degree-3 spuriously "wins" −34%; the temperature
sweep kills that — deg1's optimum is interior at 0.25, not a grid edge.)

**The finding (a clean negative):** associative cleanup and language modeling
want opposite things. Cleanup assumes the cue is a *corrupted single pattern* and
collapses it to the nearest attractor; a next-token distribution is genuinely a
*mixture* of continuations, and the readout cue `c` is a blend, not a corrupted
memory. Higher degree = sharper collapse = more of the mixture destroyed — hence
monotonic degradation. **Capacity (autoassociative cleanup) does not transfer to
the token-distribution readout.** It can only help where there *is* a single
correct latent state to clean toward — i.e. a reasoning *chain* (the compounding
probe), not a token *distribution*. That sharpens the Tier-2 hypothesis: cleanup
belongs on a distributed latent scratchpad between reasoning steps, never on the
output distribution. See `brain/tasks/llm/tier1_dam_readout.py`.

## Files

- `experiments.py` — capacity sweep, attention equivalence, density sweep.
- `reasoning_probe.py` — four-rung composition dip test vs a retrieval foil.
- `compounding_probe.py` — error-compounding vs associative-cleanup gate.
- `basin_scaling.py` — load ceiling vs interaction degree and dimension d.
- `tests/test_astrocyte.py` — substrate ≡ dense equivalence + capacity + attention.
- `tests/test_probes.py` — smoke tests that both probes run and separate from the foil.
- `../llm/tier1_dam_readout.py` — degree-n associative readout on the substrate-LLM
  (the Tier-1 bridge above); `../llm/tests/test_tier1_dam.py` guards it.
