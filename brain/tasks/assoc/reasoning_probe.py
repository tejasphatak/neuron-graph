"""Reasoning dip test — can the substrate COMPOSE, not just retrieve?

The astrocyte work proved retrieval (attention) + in-context copy. Retrieval
alone can only MATCH context, never compose over it (see context_sweep in the
research notes). Multi-step reasoning needs composition + iteration. This module
probes four atomic capabilities reasoning decomposes into, each as a cheap
synthetic with:

  - a HELD-OUT combinatorial split (unseen combinations of seen parts), so a
    memorizer scores at chance and any above-chance signal is real composition;
  - a 1-hop RETRIEVAL BASELINE (direct co-occurrence lookup) as the foil — it
    is structurally unable to answer held-out combinations;
  - a degradation curve where applicable (accuracy vs #hops / length).

The substrate−baseline gap on held-out items is the entire result.

  Rung 1  chaining        transitive inference over learned edges (multi-hop spread)
  Rung 2  binding         variable indirection: alias -> var -> value (deref)
  Rung 3  generalization  SCAN-mini: held-out verb x direction combinations
  Rung 4  algorithmic     FSM state tracking: per-step loop generalizes to length

Edges are LEARNED from exposure (only directly co-presented pairs are ever
connected); spread()/the external loop must compose the rest. The substrate uses
a dedicated 'assoc' relation (weight 1.0) so the curves reflect real spread
dynamics (branching, noise, interference), not an arbitrary attenuation constant.

Run:  python -m brain.tasks.assoc.reasoning_probe
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional, Sequence

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(_HERE))))
sys.path.insert(0, _REPO)

from brain import Brain, spread
from brain.neuron import NeuronType


ASSOC = "assoc"


# --------------------------------------------------------------------------- #
# shared substrate helpers
# --------------------------------------------------------------------------- #
class Graph:
    """Thin wrapper: named nodes + learned directed edges on a real Brain."""

    def __init__(self):
        self.brain = Brain()
        self.brain._add_relation(ASSOC, 1.0)
        self.id: Dict[str, int] = {}
        self._edges: Dict[int, Dict[int, float]] = {}     # for the retrieval foil

    def node(self, name: str, *, type=NeuronType.CONCEPT) -> int:
        if name not in self.id:
            self.id[name] = self.brain.add_neuron(type=type, decay=0.5)
        return self.id[name]

    def expose(self, src: str, dst: str, weight: float = 0.9) -> None:
        """Learn ONE directed association from a co-presentation. Only directly
        presented pairs ever get an edge — multi-hop is left for spread()."""
        a, b = self.node(src), self.node(dst)
        self.brain.add_synapse(a, b, rel_name=ASSOC, weight=weight)
        self._edges.setdefault(a, {})[b] = max(self._edges.get(a, {}).get(b, 0.0),
                                               weight)

    # ---- substrate read-out: multi-hop spread -------------------------------
    def spread_pick(self, query: str, candidates: Sequence[str], *,
                    steps: int, rng: np.random.Generator) -> str:
        q = self.id[query]
        state = spread(self.brain, seeds=[q], max_steps=steps,
                       convergence_eps=1e-4)
        act = state.activation
        return _argmax_named(candidates, lambda c: act.get(self.id[c], 0.0), rng)

    # ---- the foil: 1-hop direct co-occurrence -------------------------------
    def retrieval_pick(self, query: str, candidates: Sequence[str], *,
                       rng: np.random.Generator) -> str:
        q = self.id[query]
        edges = self._edges.get(q, {})
        return _argmax_named(candidates, lambda c: edges.get(self.id[c], 0.0), rng)


def _argmax_named(candidates, score_fn, rng) -> str:
    """Argmax with random tie-break (so no-signal cases score at chance)."""
    scored = [(c, score_fn(c)) for c in candidates]
    best = max(s for _, s in scored)
    winners = [c for c, s in scored if s == best]
    return winners[int(rng.integers(len(winners)))]


# --------------------------------------------------------------------------- #
# Rung 1 — chaining (transitive inference over multi-hop spread)
# --------------------------------------------------------------------------- #
def rung1_chaining(n_chains=12, chain_len=5, n_distract=4, cross_links=2,
                   cross_w=0.6, trials_per=40, seed=0):
    """Learn adjacent links A->B->C->...; query A, retrieve the node d hops down
    a chain that was NEVER co-presented with A. Foil: 1-hop co-occurrence.

    Interference is real: every node also gets `cross_links` strong-ish edges to
    OTHER chains, so distractors are reachable and the substrate must rely on the
    direct chain path (0.9^d) out-competing cross-link leakage. The hop curve
    then degrades as the path signal shrinks toward the interference floor."""
    rng = np.random.default_rng(seed)
    g = Graph()
    chains = [[f"c{ci}_{j}" for j in range(chain_len)] for ci in range(n_chains)]
    all_nodes = [n for ch in chains for n in ch]
    for n in all_nodes:
        g.node(n)
    # learn ONLY adjacent links (strong) ...
    for ch in chains:
        for j in range(chain_len - 1):
            g.expose(ch[j], ch[j + 1], weight=0.9)
    # ... plus cross-chain interference edges (the substrate must out-compete these)
    chain_of = {n: ci for ci, ch in enumerate(chains) for n in ch}
    for n in all_nodes:
        for _ in range(cross_links):
            other = all_nodes[int(rng.integers(len(all_nodes)))]
            if chain_of[other] != chain_of[n]:
                g.expose(n, other, weight=cross_w)

    sub = np.zeros(chain_len)   # accuracy by hop distance d=1..chain_len-1
    base = np.zeros(chain_len)
    cnt = np.zeros(chain_len)
    for _ in range(trials_per):
        for ch in chains:
            start = ch[0]
            for d in range(1, chain_len):
                target = ch[d]
                others = [n for oc in chains if oc is not ch for n in oc]
                distract = [others[i] for i in
                            rng.choice(len(others), size=n_distract, replace=False)]
                cands = [target] + distract
                rng.shuffle(cands)
                sub[d] += (g.spread_pick(start, cands, steps=d + 2, rng=rng) == target)
                base[d] += (g.retrieval_pick(start, cands, rng=rng) == target)
                cnt[d] += 1
    return {"hop": list(range(1, chain_len)),
            "substrate": (sub[1:] / cnt[1:]).tolist(),
            "baseline": (base[1:] / cnt[1:]).tolist(),
            "chance": 1.0 / (n_distract + 1)}


# --------------------------------------------------------------------------- #
# Rung 2 — variable binding / indirection (alias -> var -> value)
# --------------------------------------------------------------------------- #
def rung2_binding(n_vars=20, n_alias=20, trials=400, seed=1):
    """Direct bindings var_i = val_i (var->val). Aliases alias_k = var_i
    (alias->var). Query an alias, want the VALUE (two hops). The foil returns the
    variable (wrong TYPE) — it never saw alias co-occur with a value."""
    rng = np.random.default_rng(seed)
    g = Graph()
    vals = [f"val{i}" for i in range(n_vars)]
    vars_ = [f"var{i}" for i in range(n_vars)]
    for v in vals + vars_:
        g.node(v)
    for i in range(n_vars):
        g.expose(vars_[i], vals[i], weight=0.9)          # var -> value
    aliases = []
    for k in range(n_alias):
        tgt = int(rng.integers(n_vars))
        a = f"alias{k}"
        g.node(a)
        g.expose(a, vars_[tgt], weight=0.9)              # alias -> var (only!)
        aliases.append((a, tgt))

    sub_hit = base_hit = base_typeerr = 0
    for _ in range(trials):
        a, tgt = aliases[int(rng.integers(len(aliases)))]
        # candidates are VALUES (the question is "which value")
        distract = [vals[i] for i in rng.choice(n_vars, size=4, replace=False)
                    if i != tgt][:3]
        cands = [vals[tgt]] + distract
        rng.shuffle(cands)
        sub_hit += (g.spread_pick(a, cands, steps=3, rng=rng) == vals[tgt])
        base_hit += (g.retrieval_pick(a, cands, rng=rng) == vals[tgt])
        # foil's unrestricted top-1 is a VARIABLE, not a value (type error)
        top = g.retrieval_pick(a, vals + vars_, rng=rng)
        base_typeerr += top.startswith("var")
    return {"substrate": sub_hit / trials, "baseline": base_hit / trials,
            "baseline_type_error_rate": base_typeerr / trials,
            "chance": 1.0 / 4}


# --------------------------------------------------------------------------- #
# Rung 3 — systematic generalization (SCAN-mini): held-out combinations
# --------------------------------------------------------------------------- #
def rung3_scan(seed=2):
    """verbs x directions -> (action, turn). Train on all combos EXCEPT held-out
    ones; every verb and every direction still appears somewhere. Foil keys on
    the whole command string -> a held-out command matches nothing."""
    rng = np.random.default_rng(seed)
    verbs = ["jump", "walk", "look", "run"]
    dirs = ["left", "right"]
    held_out = {("jump", "left"), ("run", "right"), ("look", "left")}
    all_cmds = [(v, d) for v in verbs for d in dirs]
    train = [c for c in all_cmds if c not in held_out]

    # substrate: slot-factored. learn verb->ACT and dir->TURN from TRAIN only.
    g = Graph()
    for v in verbs:
        g.node(v); g.node(f"ACT_{v}")
    for d in dirs:
        g.node(d); g.node(f"TURN_{d}")
    for (v, d) in train:
        g.expose(v, f"ACT_{v}", weight=0.9)
        g.expose(d, f"TURN_{d}", weight=0.9)

    # foil: whole-command retrieval. key = "v d" -> (ACT_v, TURN_d), TRAIN only.
    seen_cmd = {f"{v} {d}": (f"ACT_{v}", f"TURN_{d}") for (v, d) in train}

    def substrate_predict(v, d):
        act_c = [f"ACT_{x}" for x in verbs]
        turn_c = [f"TURN_{x}" for x in dirs]
        st = spread(g.brain, seeds=[g.id[v], g.id[d]], max_steps=2,
                    convergence_eps=1e-4).activation
        a = _argmax_named(act_c, lambda c: st.get(g.id[c], 0.0), rng)
        t = _argmax_named(turn_c, lambda c: st.get(g.id[c], 0.0), rng)
        return a, t

    def foil_predict(v, d):
        key = f"{v} {d}"
        if key in seen_cmd:
            return seen_cmd[key]
        # nearest seen command sharing the most tokens (else random seen)
        best, bk = -1, None
        for k in seen_cmd:
            share = len(set(k.split()) & {v, d})
            if share > best:
                best, bk = share, k
        return seen_cmd[bk]

    def score(predict):
        ok = 0
        for (v, d) in held_out:
            ok += (predict(v, d) == (f"ACT_{v}", f"TURN_{d}"))
        return ok / len(held_out)

    return {"held_out": sorted(f"{v} {d}" for v, d in held_out),
            "substrate": score(substrate_predict),
            "baseline": score(foil_predict), "chance": 1.0 / (4 * 2)}


# --------------------------------------------------------------------------- #
# Rung 4 — algorithmic state tracking (FSM): per-step loop -> length generalize
# --------------------------------------------------------------------------- #
def rung4_fsm(n_states=4, max_len=10, trials=200, seed=3):
    """Learn the FSM transition table (state,input)->state'. Compute the final
    state of a length-L input string. Substrate = external loop applying one
    learned transition per step (chain-of-thought scaffold). Foil = whole-string
    retrieval, trained on short strings only -> cliffs on long held-out strings."""
    rng = np.random.default_rng(seed)
    inputs = ["a", "b"]
    table = {(s, x): int(rng.integers(n_states))
             for s in range(n_states) for x in inputs}

    # substrate: a config node (s,x) -> next-state node. ALL transitions learned.
    g = Graph()
    for s in range(n_states):
        g.node(f"S{s}")
    for (s, x), s2 in table.items():
        g.node(f"cfg_{s}_{x}")
        g.expose(f"cfg_{s}_{x}", f"S{s2}", weight=0.9)

    def substrate_run(s0, string):
        cur = s0
        for x in string:
            st = spread(g.brain, seeds=[g.id[f"cfg_{cur}_{x}"]], max_steps=1,
                        convergence_eps=1e-4).activation
            cand = [f"S{s}" for s in range(n_states)]
            nxt = _argmax_named(cand, lambda c: st.get(g.id[c], 0.0), rng)
            cur = int(nxt[1:])
        return cur

    def truth(s0, string):
        cur = s0
        for x in string:
            cur = table[(cur, x)]
        return cur

    # foil: whole-string key (s0, string) -> final, learned on TRAIN lengths 1-2
    seen = {}
    for _ in range(300):
        s0 = int(rng.integers(n_states))
        L = int(rng.integers(1, 3))
        string = "".join(inputs[int(rng.integers(2))] for _ in range(L))
        seen[(s0, string)] = truth(s0, string)

    def foil_run(s0, string):
        if (s0, string) in seen:
            return seen[(s0, string)]
        # nearest seen string by shared prefix length, else random state
        best, bv = -1, None
        for (ss, st), fv in seen.items():
            if ss != s0:
                continue
            p = 0
            while p < min(len(st), len(string)) and st[p] == string[p]:
                p += 1
            if p > best:
                best, bv = p, fv
        return bv if bv is not None else int(rng.integers(n_states))

    lengths = list(range(1, max_len + 1))
    sub = np.zeros(len(lengths)); base = np.zeros(len(lengths))
    for li, L in enumerate(lengths):
        for _ in range(trials):
            s0 = int(rng.integers(n_states))
            string = "".join(inputs[int(rng.integers(2))] for _ in range(L))
            t = truth(s0, string)
            sub[li] += (substrate_run(s0, string) == t)
            base[li] += (foil_run(s0, string) == t)
    return {"length": lengths, "substrate": (sub / trials).tolist(),
            "baseline": (base / trials).tolist(), "chance": 1.0 / n_states}


# --------------------------------------------------------------------------- #
# runner
# --------------------------------------------------------------------------- #
def _curve(xs, name, sub, base):
    print(f"    {name:>8} " + " ".join(f"{x:>5}" for x in xs))
    print(f"    {'substr':>8} " + " ".join(f"{v:>5.2f}" for v in sub))
    print(f"    {'foil':>8} " + " ".join(f"{v:>5.2f}" for v in base))


def main():
    print("=" * 70)
    print("REASONING DIP TEST  (substrate composition vs 1-hop retrieval foil)")
    print("=" * 70)
    verdicts = []

    print("\n[Rung 1] CHAINING — transitive inference, accuracy by hop distance")
    r1 = rung1_chaining()
    _curve(r1["hop"], "hop d", r1["substrate"], r1["baseline"])
    print(f"    chance = {r1['chance']:.2f}.  held-out = d>=2.")
    # judge on d>=2 (the held-out, composition-only regime)
    s2 = float(np.mean(r1["substrate"][1:])); b2 = float(np.mean(r1["baseline"][1:]))
    verdicts.append(("1 chaining", s2, b2, r1["chance"]))

    print("\n[Rung 2] BINDING — alias -> var -> value (2-step dereference)")
    r2 = rung2_binding()
    print(f"    substrate={r2['substrate']:.2f}  foil={r2['baseline']:.2f}  "
          f"chance={r2['chance']:.2f}")
    print(f"    foil top-1 is a VARIABLE (wrong type) on "
          f"{r2['baseline_type_error_rate']*100:.0f}% of alias queries")
    verdicts.append(("2 binding", r2["substrate"], r2["baseline"], r2["chance"]))

    print("\n[Rung 3] GENERALIZATION — held-out verb x direction combinations")
    r3 = rung3_scan()
    print(f"    held out: {r3['held_out']}")
    print(f"    substrate={r3['substrate']:.2f}  foil={r3['baseline']:.2f}  "
          f"chance={r3['chance']:.2f}")
    verdicts.append(("3 generaliz", r3["substrate"], r3["baseline"], r3["chance"]))

    print("\n[Rung 4] ALGORITHMIC — FSM final-state accuracy by string length")
    r4 = rung4_fsm()
    _curve(r4["length"], "len L", r4["substrate"], r4["baseline"])
    print(f"    chance = {r4['chance']:.2f}.  foil trained on L<=2 only.")
    # judge on long held-out strings (L>=3)
    li = [i for i, L in enumerate(r4["length"]) if L >= 3]
    s4 = float(np.mean([r4["substrate"][i] for i in li]))
    b4 = float(np.mean([r4["baseline"][i] for i in li]))
    verdicts.append(("4 algorithmic", s4, b4, r4["chance"]))

    print("\n" + "=" * 70)
    print("VERDICT  (held-out items only; PASS = substrate beats foil by >0.20)")
    print("=" * 70)
    print(f"    {'rung':>14}  {'substrate':>9}  {'foil':>6}  {'chance':>6}  verdict")
    for name, s, b, ch in verdicts:
        v = "PASS" if (s - b) > 0.20 and s > 0.5 else "weak/FAIL"
        print(f"    {name:>14}  {s:>9.2f}  {b:>6.2f}  {ch:>6.2f}  {v}")
    print("\n    The gap (substrate - foil) on held-out items is the signal:")
    print("    it isolates COMPOSITION from memorization. A foil at chance with")
    print("    the substrate well above it = the capability is real, not recall.")


if __name__ == "__main__":
    main()
