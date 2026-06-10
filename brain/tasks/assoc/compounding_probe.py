"""Hard-mode dip test — does associative cleanup defeat error compounding?

The four-rung reasoning_probe showed the substrate HAS the compositional
primitives, but three rungs scored 1.00 because per-step inference was a clean,
noise-free lookup. The real wall for long reasoning is COMPOUNDING: if each step
is imperfect, errors accumulate across the chain unless something corrects them
mid-stream. A feedforward generator has no corrector. An associative memory
does: every stored pattern is an attractor with a basin, so a drifted
intermediate state snaps back to the correct one (Hopfield cleanup) BEFORE it
derails the next step.

Setup — a distributed-representation FSM run in VECTOR SPACE (no free re-snap to
clean symbols). M states are random bipolar vectors in R^d. For each input x the
transition is a LEARNED linear associator W_x = (1/d) Σ_i s[T(i,x)] s[i]^T — the
Hebbian map that sends each state vector to its successor. Applying W_x is an
imperfect "computation": the signal term recovers the target, but crosstalk from
the other M-1 stored maps injects per-step error. Carried forward WITHOUT
cleanup, that error accumulates and the trajectory drifts off the state manifold.

  cleanup OFF : v <- sign(W_x v).        Crosstalk compounds -> drift -> wrong.
  cleanup ON  : v <- retrieve(sign(W_x v)) through the degree-4 astrocyte DAM.
                The attractor removes the per-step drift -> chain stays on path.

Two sweeps:
  1. accuracy vs chain length N (fixed load): OFF decays as the drift compounds,
     ON stays flat -> compounding, and its defeat, made visible.
  2. accuracy vs load M (fixed long N): more states -> more crosstalk per step ->
     OFF collapses early; ON holds until the per-step drift exceeds the basin.
     The crossover is the load ceiling for long chains.

Run:  python -m brain.tasks.assoc.compounding_probe
"""

from __future__ import annotations

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(_HERE))))
sys.path.insert(0, _REPO)

from brain.astrocyte import SubstrateAstrocyteMemory


def _make_fsm(M, d, n_inputs, rng):
    """M distributed bipolar states, a transition table, and one learned linear
    associator W_x per input (the imperfect per-step 'compute' operator)."""
    states = rng.choice([-1.0, 1.0], size=(M, d))
    table = rng.integers(0, M, size=(M, n_inputs))            # T[state, input]
    W = []
    for x in range(n_inputs):
        targets = states[table[:, x]]                         # (M, d)
        W.append((targets.T @ states) / d)                   # (d, d) Hebbian map
    return states, table, W


def _decode(states, v):
    return int((states @ v).argmax())


def _run_chain(states, table, W, mem, *, start, inputs_seq, cleanup):
    """Run the chain in vector space; True if decoded final == ground-truth final."""
    truth = start
    v = states[start].copy()
    for x in inputs_seq:
        truth = int(table[truth, x])              # ground-truth path (exact)
        v = np.sign(W[x] @ v)                     # imperfect learned transition
        v[v == 0] = 1.0
        if cleanup:
            v = mem.retrieve(v, steps=5)          # attractor cleanup before carry
    return _decode(states, v) == truth


def _accuracy(states, table, W, mem, *, N, cleanup, trials, n_inputs, rng):
    ok = 0
    for _ in range(trials):
        start = int(rng.integers(states.shape[0]))
        seq = rng.integers(0, n_inputs, size=N)
        ok += _run_chain(states, table, W, mem, start=start, inputs_seq=seq,
                         cleanup=cleanup)
    return ok / trials


def length_sweep(M=24, d=64, n_inputs=2, lengths=(1, 2, 4, 8, 16, 32),
                 trials=60, seed=0):
    rng = np.random.default_rng(seed)
    states, table, W = _make_fsm(M, d, n_inputs, rng)
    mem = SubstrateAstrocyteMemory(d, interaction="poly", degree=4,
                                   activation="sign").store(states)
    print("=" * 70)
    print("COMPOUNDING SWEEP  (accuracy vs chain length)")
    print("=" * 70)
    print(f"    M={M} states  d={d}  bipolar  learned W_x transition  "
          f"degree-4 cleanup\n")
    off = [_accuracy(states, table, W, mem, N=N, cleanup=False, trials=trials,
                     n_inputs=n_inputs, rng=rng) for N in lengths]
    on = [_accuracy(states, table, W, mem, N=N, cleanup=True, trials=trials,
                    n_inputs=n_inputs, rng=rng) for N in lengths]
    # effective per-step success from the longest measured OFF point, so the
    # p^N reference reflects multi-step compounding (off[0] at N=1 is ~1.0 and
    # would render a flat, useless line).
    ref_i = max((i for i, v in enumerate(off) if v > 0.02), default=len(off) - 1)
    p = off[ref_i] ** (1.0 / lengths[ref_i]) if off[ref_i] > 0 else 0.0
    pred = [p ** N for N in lengths]
    print(f"    {'len N':>16} " + " ".join(f"{N:>6}" for N in lengths))
    print(f"    {'OFF (feedfwd)':>16} " + " ".join(f"{v:>6.2f}" for v in off))
    print(f"    {'  pred p^N':>16} " + " ".join(f"{v:>6.2f}" for v in pred))
    print(f"    {'ON  (cleanup)':>16} " + " ".join(f"{v:>6.2f}" for v in on))
    print(f"\n    effective per-step p = {p:.2f}.  OFF tracks p^N (pure compounding);")
    print("    ON stays high — cleanup corrects each step's drift before it derails.")
    return {"lengths": list(lengths), "off": off, "on": on, "pred": pred, "p": p}


def load_sweep(d=64, n_inputs=2, N=16, Ms=(16, 24, 32, 48, 64, 96),
               trials=60, seed=1):
    rng = np.random.default_rng(seed)
    print("\n" + "=" * 70)
    print(f"LOAD SWEEP  (accuracy vs #states M; fixed chain length N={N})")
    print("=" * 70)
    off, on = [], []
    for M in Ms:
        states, table, W = _make_fsm(M, d, n_inputs, rng)
        mem = SubstrateAstrocyteMemory(d, interaction="poly", degree=4,
                                       activation="sign").store(states)
        off.append(_accuracy(states, table, W, mem, N=N, cleanup=False,
                             trials=trials, n_inputs=n_inputs, rng=rng))
        on.append(_accuracy(states, table, W, mem, N=N, cleanup=True,
                            trials=trials, n_inputs=n_inputs, rng=rng))
    print(f"    {'M states':>16} " + " ".join(f"{m:>6}" for m in Ms))
    print(f"    {'OFF (feedfwd)':>16} " + " ".join(f"{v:>6.2f}" for v in off))
    print(f"    {'ON  (cleanup)':>16} " + " ".join(f"{v:>6.2f}" for v in on))
    ceiling = max((m for m, v in zip(Ms, on) if v >= 0.9), default=0)
    print(f"\n    long-chain load ceiling ~ M={ceiling} (ON still >=0.90 at N={N}).")
    print("    Beyond it, single-step crosstalk exceeds the basin and even cleanup")
    print("    can't keep a long chain on the manifold.")
    return {"Ms": list(Ms), "off": off, "on": on, "ceiling": ceiling}


def main():
    r1 = length_sweep()
    r2 = load_sweep()
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    N_long = r1["lengths"][-1]
    on_long, off_long = r1["on"][-1], r1["off"][-1]
    print(f"    Length sweep (M=24, within basin), at length {N_long}:")
    print(f"        feedforward = {off_long:.2f}  (drift compounds toward chance)")
    print(f"        cleanup     = {on_long:.2f}")
    # the headline gap: the load where cleanup separates most from feedforward
    gaps = [(on - off, m, on, off)
            for m, on, off in zip(r2["Ms"], r2["on"], r2["off"])]
    gap, gm, gon, goff = max(gaps)
    print(f"    Load sweep (N=16): biggest separation at M={gm} — "
          f"feedforward {goff:.2f} vs cleanup {gon:.2f}  (gap {gap:.2f})")
    print(f"    Long-chain load ceiling ~ M={r2['ceiling']} states (d=64).")
    if gap > 0.5 and r2["ceiling"] >= 24:
        print("\n    PASS: associative cleanup defeats error compounding WITHIN the")
        print("    basin — a chain feedforward gets right "
              f"{goff*100:.0f}% of the time becomes {gon*100:.0f}%.")
        print("    Reasoning-chain length is bounded by per-step drift vs basin")
        print("    size, NOT by depth itself. Past the load ceiling, single-step")
        print("    crosstalk exceeds the basin and even cleanup compounds.")
        print("    Next lever: enlarge the basin (degree / d / encoding) to raise")
        print("    the ceiling — that is the concrete knob for longer reasoning.")
    else:
        print("\n    WEAK: cleanup did not clearly separate from feedforward —")
        print("    per-step drift may already exceed the basin at this d/M.")


if __name__ == "__main__":
    main()
