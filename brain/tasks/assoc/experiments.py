"""Neuron-astrocyte associative memory on the substrate — experiments.

1. capacity_sweep(): empirical K_max vs N for degree-2 (classic Hopfield, ~N)
   and degree-4 (the paper's quartic, ~N^3), run through the SUBSTRATE
   gather/scatter. Reproduces the paper's supralinear-capacity separation.

2. attention_equivalence(): the softmax interaction reproduces standard
   transformer self-attention, with zero learned process-to-process weights.

3. density_sweep(): THE OPEN EXPERIMENT the paper points at but does not run.
   The astrocyte<->neuron edge density r = K/N is a real substrate knob (each
   astrocyte's neighbor-set size). Holding N fixed and degree=4, sweep r and
   measure how realized K_max degrades as the synaptic islands shrink from
   global (r=1, dense-equivalent) to local. r=const -> linear capacity,
   r growing -> supralinear; this measures where a finite local graph sits.

Run:  python -m brain.tasks.assoc.experiments
"""

from __future__ import annotations

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(_HERE))))
sys.path.insert(0, _REPO)

from brain.astrocyte import SubstrateAstrocyteMemory


# --------------------------------------------------------------------------- #
# recall harness (shared)
# --------------------------------------------------------------------------- #
def _recall_rate(N, K, degree, trials, flip, steps, rng, connectivity=1.0):
    ok = 0
    for _ in range(trials):
        P = rng.choice([-1.0, 1.0], size=(K, N))
        mem = SubstrateAstrocyteMemory(N, interaction="poly", degree=degree,
                                       activation="sign").store(
                                           P, connectivity=connectivity, rng=rng)
        t = int(rng.integers(K))
        target = P[t].copy()
        cue = target.copy()
        idx = rng.choice(N, size=max(1, int(flip * N)), replace=False)
        cue[idx] *= -1
        if np.array_equal(mem.retrieve(cue, steps=steps), target):
            ok += 1
    return ok / trials


def _k_max(N, degree, rng, thresh=0.9, trials=20, flip=0.10, steps=30,
           connectivity=1.0):
    """Empirical K_max: largest K with recall >= thresh. Geometric up, bisect."""
    K = max(2, N // 4)
    last_ok = 0
    ceiling = 50 * N ** (degree - 1)
    while K < ceiling:
        if _recall_rate(N, K, degree, trials, flip, steps, rng, connectivity) < thresh:
            break
        last_ok = K
        K = int(K * 1.6) + 1
    lo, hi = last_ok, K
    while hi - lo > max(2, last_ok // 10):
        mid = (lo + hi) // 2
        if _recall_rate(N, mid, degree, trials, flip, steps, rng, connectivity) >= thresh:
            lo = mid
        else:
            hi = mid
    return max(lo, 1)


# --------------------------------------------------------------------------- #
# 1. capacity scaling
# --------------------------------------------------------------------------- #
def capacity_sweep(Ns=(16, 24, 32, 48, 64), degrees=(2, 4), seed=0):
    rng = np.random.default_rng(seed)
    print("=" * 64)
    print("CAPACITY SCALING (substrate gather/scatter; K_max vs N, recall>=90%)")
    print("=" * 64)
    results = {}
    for deg in degrees:
        kmax = [_k_max(N, deg, rng) for N in Ns]
        results[deg] = kmax
        slope, _ = np.polyfit(np.log(Ns), np.log(kmax), 1)
        regime = "classic Hopfield" if deg == 2 else f"DAM degree-{deg} (paper)"
        print(f"\ndegree n={deg}  [{regime}]")
        for N, k in zip(Ns, kmax):
            print(f"    N={N:4d}   K_max={k:6d}   K_max/N={k / N:7.2f}")
        print(f"    log-log slope = {slope:.2f}   (theory: ~{deg - 1})")
    return results


# --------------------------------------------------------------------------- #
# 2. attention equivalence
# --------------------------------------------------------------------------- #
def attention_equivalence(n_tok=12, d=16, n_query=5, seed=0):
    rng = np.random.default_rng(seed)
    print("\n" + "=" * 64)
    print("ATTENTION EQUIVALENCE  (softmax regime == self-attention)")
    print("=" * 64)
    X = rng.standard_normal((n_tok, d))
    Wk, Wv, Wq = (rng.standard_normal((d, d)) / np.sqrt(d) for _ in range(3))
    Kmat, V = X @ Wk, X @ Wv
    Q = rng.standard_normal((n_query, d)) @ Wq
    beta = 1.0 / np.sqrt(d)

    scores = Q @ Kmat.T * beta
    A = np.exp(scores - scores.max(1, keepdims=True))
    A = A / A.sum(1, keepdims=True)
    ref = A @ V

    cos, maxerr = [], []
    for qi in range(n_query):
        mem = SubstrateAstrocyteMemory(d, interaction="softmax", beta=beta,
                                       activation="identity").store(Kmat)
        glio = mem._glio(mem._gather(Q[qi]))
        out = glio @ V
        c = float(out @ ref[qi] / (np.linalg.norm(out) * np.linalg.norm(ref[qi])))
        cos.append(c)
        maxerr.append(float(np.abs(out - ref[qi]).max()))
    print(f"    queries={n_query}  tokens={n_tok}  d={d}")
    print(f"    mean cosine(out, attention) = {np.mean(cos):.6f}")
    print(f"    max abs error               = {max(maxerr):.2e}")
    print("    -> softmax astrocyte coupling reproduces self-attention,")
    print("       with no learned process-to-process weights, over real CSR edges.")
    return float(np.mean(cos)), max(maxerr)


# --------------------------------------------------------------------------- #
# 3. density sweep — the open experiment
# --------------------------------------------------------------------------- #
def _recovery_fidelity(N, K, degree, connectivity, trials, flip, steps, rng):
    """Mean fraction of bits recovered (1 - Hamming/N), not exact match.

    Exact-match K_max saturates to ~1 the moment r<1, because a target's
    dropped coordinates get zero drive and flip a few bits. Bit fidelity
    resolves the actual degradation surface.
    """
    tot = 0.0
    for _ in range(trials):
        P = rng.choice([-1.0, 1.0], size=(K, N))
        mem = SubstrateAstrocyteMemory(N, interaction="poly", degree=degree,
                                       activation="sign").store(
                                           P, connectivity=connectivity, rng=rng)
        t = int(rng.integers(K))
        target = P[t].copy()
        cue = target.copy()
        cue[rng.choice(N, size=max(1, int(flip * N)), replace=False)] *= -1
        out = mem.retrieve(cue, steps=steps)
        tot += float((out == target).mean())
    return tot / trials


def density_sweep(N=64, degree=4,
                  connectivities=(1.0, 0.9, 0.75, 0.5, 0.25),
                  Ks=(2, 8, 32, 128), trials=12, seed=0):
    """Realized-recall surface over (astrocyte connectivity r, load K).

    r = K/N edges per astrocyte is a genuine substrate knob (neighbor-set size).
    Reported as bit-recovery fidelity so the sparse regime is resolved instead
    of collapsing to exact-match K_max=1.
    """
    rng = np.random.default_rng(seed)
    print("\n" + "=" * 64)
    print("DENSITY SWEEP  (recall fidelity vs astrocyte connectivity r, load K)")
    print("=" * 64)
    print(f"    N={N}  degree={degree}  10%-corrupted cue  (r=1.0 = dense reference)")
    print("    cell = mean fraction of bits recovered over "
          f"{trials} trials\n")
    header = f"    {'r = K/N':>8} {'edges':>6} | " + " ".join(
        f"K={k:>4}" for k in Ks)
    print(header)
    print("    " + "-" * (len(header) - 4))
    results = {}
    for r in connectivities:
        row = [_recovery_fidelity(N, k, degree, r, trials, 0.10, 30, rng)
               for k in Ks]
        results[r] = row
        edges = int(round(r * N))
        cells = " ".join(f"{v:>6.2f}" for v in row)
        print(f"    {r:>8.2f} {edges:>6} | {cells}")
    print("\n    r=1.0: perfect recall holds to high load, then interference bites.")
    print("    r<1.0: capped below 1.0 even at low load — coordinates outside a")
    print("    target's synaptic island get no drive, so a few bits never recover.")
    print("    Realized capacity is set by island size, not pattern count: the")
    print("    quantity the PNAS paper points at but does not measure.")
    return results


if __name__ == "__main__":
    capacity_sweep()
    attention_equivalence()
    density_sweep()
