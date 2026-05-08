# neuron-graph

A CPU-native, identity-bearing neuron substrate. No matmul, no backprop, no GPU.

Each neuron is a 64-byte cache-line struct in a numpy array. Edges are CSR-laid-out
synapses. Spreading activation + Hebbian + reward-modulated plasticity drives learning.
The graph **self-organizes from reward** — same apparatus solves Tic-Tac-Toe (100%
draws vs minimax) and language modeling (87% real-inference accuracy on 20 sentences).

## Headline results

| Task | Result |
|---|---|
| TTT vs minimax (substrate-learned value head) | 100% draws |
| TTT next-state world model | 95% accuracy |
| LM 20-sentence corpus, real inference | **87%** (11/20 sentences perfect) |
| LM sentence-id prediction from prompt | 95% (19/20) |
| Speed (after vectorized spread) | 70 ms / training epoch |

The LM number is the interesting one: starting from only POS class membership and grammar
shape (no co_occurs taught), RL grows the routing graph via reward — 28% cold-start →
89% with curriculum + sentence-id binding. Substrate-native retrieval baked into the graph.

## Quickstart

```bash
git clone https://github.com/tejasphatak/neuron-graph.git
cd neuron-graph

# Run all tests (~110 tests)
python3 -m pytest -q

# TTT planning agent vs minimax
python3 -m tasks.ttt.demo_plus

# Smallest LM generation test (qualitative teach + spread)
python3 tasks/lm/tiny.py

# 20-sentence RL scaling experiment
python3 tasks/lm/scaling_experiment.py
```

## Architecture

```
neuron.py     64-byte cache-line struct (numpy structured array)
store.py      Brain dataclass: nodes, synapses, syn_offsets, aliases, relations
spread.py     Activation cycle (the only "thinking" primitive)
              Goal injection, working memory, group-aware sparsity
learn.py      Hebbian co-activation update
modulator.py  Global plasticity scalar (dopamine analog)
replay.py     Episode buffer + consolidate (offline re-experience)
trace.py      Per-event log (every spread/update inspectable)
working_memory.py  Sustained activation with positional decay

tasks/ttt/    Tic-Tac-Toe — RL games (proven domain)
              game, world_model, planner, value_head, curriculum
tasks/lm/     Language modeling
              tiny.py        qualitative teach + 3 generators
              rl.py          teach_minimal, train_rl, train_rl_curriculum
                              predict_sentence_id, btsp_credit
              scaling_experiment.py  20-sentence corpus
```

## Design rules

1. **No matmul.** Sparse activation + graph traversal, not dense weights.
2. **No backprop.** Local Hebbian + reward-modulated plasticity.
3. **Identity-bearing neurons.** Each neuron is a concept, position, sentence-id, etc.
   Not a tensor slot. Carries semantic meaning.
4. **Inspectable.** Every emission has a traceable spreading path.
5. **Modality-agnostic substrate.** Tasks bring encoders + reward; the substrate is generic.

## What's proven and what isn't

**Proven:**
- RL self-correction grows the routing graph from reward (28% → 89% on LM)
- Same substrate handles RL games (TTT 100% draws) and sequence modeling (LM 87%)
- CPU inference on commodity hardware
- Curriculum + replay + sentence-id binding compound multiplicatively at scale

**Not yet validated:**
- Scaling beyond 20 sentences (open question whether sentence-id mechanism keeps working)
- Modality polymorphism on vision (MNIST is the obvious next test)
- Open-ended generation without teacher-forced POS sequence
- Phase C: mmap + multi-core spread for billion-neuron substrate

## Pointers

- [`tasks/ttt/PROBE_RESULTS.md`](tasks/ttt/PROBE_RESULTS.md) — full TTT empirical findings
- Commit log — every commit message documents what was tested and what was learned
  (including negative results: BTSP credit propagation didn't fit per-step LM rewards)

## License

MIT — research code. Use it, fork it, build on it.
