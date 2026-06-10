"""Basin-scaling experiment — the lever for longer reasoning.

compounding_probe established that reasoning-chain length is bounded by per-step
drift vs BASIN SIZE, not by depth itself: associative cleanup defeats error
compounding up to a load ceiling, then breaks. This experiment measures how that
ceiling moves with the two knobs that should govern basin size:

  - interaction DEGREE n. Dense Associative Memory capacity ~ N^(n-1)
    (Krotov & Hopfield 2016). Higher degree -> sharper, better-separated
    attractors -> larger basins -> higher ceiling.
  - dimension d. Per-step crosstalk of the learned transition scales ~ sqrt(M/d),
    and DAM capacity grows with d, so more dimensions both shrink the drift and
    enlarge the basin -> higher ceiling.

For each setting we run the compounding chain (cleanup ON) at a fixed long length
over a grid of loads M, and read off the ceiling = the largest M whose length-N
chain still lands correctly >= 90% of the time. The two sweeps test whether
degree and d are real levers and roughly how the ceiling scales.

Uses the dense NeuronAstrocyteMemory for the cleanup engine — proven sign-for-sign
equivalent to the substrate (tests/test_astrocyte.py) and far faster for a sweep.

Run:  python -m brain.tasks.assoc.basin_scaling
"""

from __future__ import annotations

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(_HERE))))
sys.path.insert(0, _REPO)

from brain.astrocyte import NeuronAstrocyteMemory
from brain.tasks.assoc.compounding_probe import _make_fsm, _accuracy


def _ceiling(degree, d, Ms, *, N, trials, n_inputs, seed, thresh=0.9):
    """For (degree, d): cleanup-ON accuracy across loads Ms at chain length N.
    Returns (accuracy_row, ceiling_M). Ceiling = largest M with acc >= thresh."""
    rng = np.random.default_rng(seed)
    row, ceiling = [], 0
    for M in Ms:
        states, table, W = _make_fsm(M, d, n_inputs, rng)
        mem = NeuronAstrocyteMemory(interaction="poly", degree=degree,
                                    activation="sign").store(states)
        acc = _accuracy(states, table, W, mem, N=N, cleanup=True,
                        trials=trials, n_inputs=n_inputs, rng=rng)
        row.append(acc)
        if acc >= thresh:
            ceiling = M
    return row, ceiling


def degree_sweep(d=64, degrees=(2, 4, 6), Ms=(8, 16, 24, 32, 48, 64, 96, 128),
                 N=16, trials=40, n_inputs=2, seed=0):
    print("=" * 74)
    print(f"DEGREE SWEEP  (load ceiling vs interaction degree; d={d}, chain N={N})")
    print("=" * 74)
    print(f"    cleanup-ON accuracy across load M; ceiling = largest M with acc>=0.90\n")
    print(f"    {'degree':>8} | " + " ".join(f"M={m:>3}" for m in Ms) + "  | ceiling")
    print("    " + "-" * (11 + 7 * len(Ms) + 11))
    ceilings = {}
    for deg in degrees:
        row, ceil = _ceiling(deg, d, Ms, N=N, trials=trials, n_inputs=n_inputs,
                             seed=seed)
        ceilings[deg] = ceil
        cells = " ".join(f"{v:>5.2f}" for v in row)
        print(f"    {deg:>8} | {cells}  | M={ceil}")
    print("\n    Higher degree -> sharper attractors -> larger basins -> the chain")
    print("    tolerates more states before single-step crosstalk breaks cleanup.")
    return ceilings


def dim_sweep(degree=4, ds=(32, 64, 128), fracs=(0.25, 0.5, 0.75, 1.0, 1.5, 2.0),
              N=16, trials=40, n_inputs=2, seed=1):
    print("\n" + "=" * 74)
    print(f"DIMENSION SWEEP  (load ceiling vs d; degree={degree}, chain N={N})")
    print("=" * 74)
    print("    loads scaled as M = frac * d; ceiling reported in absolute M\n")
    print(f"    {'d':>6} | " + " ".join(f"{f:>4}d" for f in fracs) + "  | ceiling  ceiling/d")
    print("    " + "-" * (9 + 7 * len(fracs) + 20))
    ceilings = {}
    for d in ds:
        Ms = [max(2, int(round(f * d))) for f in fracs]
        row, ceil = _ceiling(degree, d, Ms, N=N, trials=trials, n_inputs=n_inputs,
                             seed=seed)
        ceilings[d] = ceil
        cells = " ".join(f"{v:>5.2f}" for v in row)
        print(f"    {d:>6} | {cells}  | M={ceil:<5} {ceil / d:>6.2f}")
    print("\n    More dimensions shrink per-step crosstalk (~sqrt(M/d)) and enlarge")
    print("    capacity: the ceiling rises with d — longer/heavier chains stay on")
    print("    the manifold. ceiling/d trending up = super-linear headroom in d.")
    return ceilings


def main():
    dc = degree_sweep()
    sc = dim_sweep()
    print("\n" + "=" * 74)
    print("VERDICT")
    print("=" * 74)
    degs = sorted(dc)
    print(f"    Ceiling vs degree (d=64): " +
          ", ".join(f"n={k}->M={dc[k]}" for k in degs))
    ds = sorted(sc)
    print(f"    Ceiling vs dimension (deg=4): " +
          ", ".join(f"d={k}->M={sc[k]}" for k in ds))
    deg_rises = dc[degs[-1]] > dc[degs[0]]
    dim_rises = sc[ds[-1]] > sc[ds[0]]
    if deg_rises and dim_rises:
        print("\n    PASS: both degree and dimension raise the load ceiling. Basin")
        print("    size is the real lever — to reason over longer/heavier chains,")
        print("    spend capacity on the substrate (higher degree, more dims), not")
        print("    on depth. This is the concrete scaling knob the substrate offers.")
    else:
        print("\n    MIXED: a lever did not move the ceiling as predicted —")
        print(f"    degree raises ceiling: {deg_rises}; dimension raises ceiling: {dim_rises}.")


if __name__ == "__main__":
    main()
