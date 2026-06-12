"""Bulletproofing sweep for the Forward-Write LM.

Settles the open question from FORWARD_WRITE_SKETCH.md: does the reservoir's
advantage survive (a) more data and (b) a *maximally-tuned* bag-of-4, and how does
it scale with reservoir size D and spectral radius rho? One identical harness:
same stories, same tokenizer, same temperature-swept eval for every model.

  - bag-of-4 best case  = the substrate-LLM degree-1 readout with score
    standardization + best temperature (exactly the Tier-1 calibration that
    reached PPL ~131) — the strongest forward-only token-context baseline.
  - Forward-Write        = reservoir + delta rule, batched BLAS rollout, best
    temperature, swept over D and rho.

Run:  PYTHONPATH=. python3 -m brain.tasks.llm.forward_write_sweep
Env:  TRAIN_N TEST_N EPOCHS VOCAB ETA SEED

CAVEAT — this sweep trains via `train_epoch_batched` (BLAS) for speed, whose
large-batch dynamics differ from the online delta rule and UNDERSTATE Forward-
Write quality (the online mechanism reaches PPL 119 @1000 stories; this batched
sweep reports ~250). Use it to compare reservoir configs cheaply and to compute
the exact baselines; for any quality CLAIM, trust the online `train_epoch`.
Bottom line, validated online: bag-of-4 best-case 108 < Forward-Write 119 < unigram
194 — a *fixed random* reservoir loses ~10% to tuned token-context. See
FORWARD_WRITE_SKETCH.md.
"""

from __future__ import annotations

import os
import time

import numpy as np

from brain.tasks.llm.tokenizer import WordTokenizer
from brain.tasks.llm.experiment import load_tinystories
from brain.tasks.llm.llm import build_llm_view, train_ngram_epoch, perplexity
from brain.tasks.llm.tier1_dam_readout import perplexity_dam
from brain.tasks.llm.forward_write import (
    Reservoir, ForwardWriteLM, unigram_perplexity)

try:
    from brain.tasks.llm.jit import train_ngram_epoch_jit_parallel, NUMBA_AVAILABLE
except Exception:
    NUMBA_AVAILABLE = False

TEMPS = (0.1, 0.25, 0.5, 1.0, 2.0, 4.0)


def bag_of_4_best_case(view, train_seqs, test_seqs, epochs, eta, seed):
    """Train the bag-of-4 perceptron, eval at its very best: standardized scores
    + best temperature (the Tier-1 setup that reached ~131)."""
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        if NUMBA_AVAILABLE:
            train_ngram_epoch_jit_parallel(view, train_seqs, context_window=4,
                                           eta=eta, rng=rng)
        else:
            train_ngram_epoch(view, train_seqs, context_window=4, eta=eta, rng=rng)
    curve = {t: perplexity_dam(view, test_seqs, degree=1, standardize=True,
                               softmax_temperature=t, context_window=4)
             for t in TEMPS}
    bt = min(curve, key=curve.get)
    return curve[bt], bt


def forward_write_config(V, train_seqs, test_seqs, *, D, rho, leak, eta,
                         epochs, seed):
    res = Reservoir(vocab=V, D=D, spectral_radius=rho, leak=leak, seed=seed)
    lm = ForwardWriteLM(res)
    for _ in range(epochs):
        lm.train_epoch_batched(train_seqs, eta=eta)
    ppl, bt, _ = lm.best_temp_perplexity(test_seqs, TEMPS, batched=True)
    return ppl, bt


def main():
    train_n = int(os.environ.get('TRAIN_N', '1000'))
    test_n = int(os.environ.get('TEST_N', '150'))
    epochs = int(os.environ.get('EPOCHS', '4'))
    max_vocab = int(os.environ.get('VOCAB', '1500'))
    eta = float(os.environ.get('ETA', '0.05'))
    seed = int(os.environ.get('SEED', '0'))

    print("=== Forward-Write bulletproofing sweep ===")
    print(f"train_n={train_n} test_n={test_n} epochs={epochs} vocab={max_vocab}\n")

    train_texts, test_texts = load_tinystories(train_n=train_n, test_n=test_n,
                                               seed=seed)
    tok = WordTokenizer(max_vocab=max_vocab, min_freq=2)
    tok.fit(train_texts)
    V = tok.get_vocab_size()
    train_seqs = [tok.encode(t) for t in train_texts]
    test_seqs = [tok.encode(t) for t in test_texts]
    print(f"vocab={V}  train_tok={sum(len(s) for s in train_seqs):,}  "
          f"uniform PPL={V}\n")

    # --- floors / best-case baseline ---
    uni = unigram_perplexity(train_seqs, test_seqs, V)
    print(f"unigram floor:               {uni:.2f}")
    t0 = time.time()
    bag, bag_t = bag_of_4_best_case(build_llm_view(tok), train_seqs, test_seqs,
                                    epochs, eta, seed)
    print(f"bag-of-4 BEST CASE:          {bag:.2f} @T={bag_t}  "
          f"(standardized + best-temp, {time.time()-t0:.0f}s)\n")

    # --- Forward-Write sweeps ---
    print("Forward-Write sweep (batched):")
    rows = []
    # D sweep at rho=0.9
    for D in (200, 400, 800):
        t0 = time.time()
        ppl, bt = forward_write_config(V, train_seqs, test_seqs, D=D, rho=0.9,
                                       leak=0.3, eta=eta, epochs=epochs, seed=seed)
        rows.append((f"D={D:<4} rho=0.9", ppl, bt))
        print(f"  D={D:<4} rho=0.9   PPL={ppl:7.2f} @T={bt}   ({time.time()-t0:.0f}s)")
    # rho sweep at D=400
    for rho in (0.7, 1.1):
        t0 = time.time()
        ppl, bt = forward_write_config(V, train_seqs, test_seqs, D=400, rho=rho,
                                       leak=0.3, eta=eta, epochs=epochs, seed=seed)
        rows.append((f"D=400  rho={rho}", ppl, bt))
        print(f"  D=400  rho={rho}   PPL={ppl:7.2f} @T={bt}   ({time.time()-t0:.0f}s)")

    best_name, best_ppl, _ = min(rows, key=lambda r: r[1])
    print(f"\n=== summary (held-out PPL, lower = better) ===")
    print(f"  unigram floor        {uni:8.2f}")
    print(f"  bag-of-4 best case   {bag:8.2f}")
    print(f"  Forward-Write best   {best_ppl:8.2f}   ({best_name})")
    if best_ppl < bag and best_ppl < uni:
        print(f"\n  VERDICT: reservoir survives more data + a maximally-tuned "
              f"bag-of-4 — beats it by {(1-best_ppl/bag)*100:.0f}%. The reservoir "
              f"representation is a real forward-only lever.")
    elif best_ppl < uni:
        print(f"\n  VERDICT: beats the floor but the best-case bag-of-4 "
              f"({bag:.1f}) is competitive/better — the earlier win was partly the "
              f"data-starved baseline. Reservoir generalizes but does not dominate.")
    else:
        print(f"\n  VERDICT: does not beat the floor at scale — reservoir too weak.")


if __name__ == '__main__':
    main()
