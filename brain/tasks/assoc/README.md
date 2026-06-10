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

## Files

- `experiments.py` — capacity sweep, attention equivalence, density sweep.
- `tests/test_astrocyte.py` — substrate ≡ dense equivalence + capacity + attention.
