"""Forward-Write LM — learning a language model by repeated exposure, no backprop.

Echo-state **reservoir** (fixed random recurrent net) turns the context into a
distributed, unbounded-history feature vector h_t — for free, no training. A
single **delta-rule** readout W_out (the associative memory) learns feature →
next-token by a local, forward-only update, once per forward pass:

    p_t   = softmax(W_out · h_t)              # READ  (= attention, proven elsewhere)
    e_t   = onehot(y_t) − p_t                 # surprise / neuromodulator
    W_out += η · e_t · h_tᵀ                   # WRITE (Hebbian outer product, no backprop)

Repetition is the engine: each re-read of a passage lowers the surprise, so the
write shrinks and the traveled paths get heavy — convergence = "it learned this."
Generalization lives in the reservoir (similar contexts → nearby h), NOT in the
store — the lesson of the Tier-1 negative, which also says keep the readout
linear+softmax and never high-degree. See FORWARD_WRITE_SKETCH.md.

Run:  PYTHONPATH=. python3 -m brain.tasks.llm.forward_write
Env:  TRAIN_N TEST_N EPOCHS VOCAB D RHO LEAK ETA SEED
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import List, Sequence

import numpy as np


# --------------------------------------------------------------------------- #
# Echo-state reservoir — fixed random, never trained
# --------------------------------------------------------------------------- #
@dataclass
class Reservoir:
    vocab: int
    d_in: int = 64
    D: int = 400
    spectral_radius: float = 0.9
    leak: float = 0.3
    in_scale: float = 1.0
    seed: int = 0
    # filled in __post_init__
    E: np.ndarray = field(default=None, repr=False)
    W_in: np.ndarray = field(default=None, repr=False)
    W_res: np.ndarray = field(default=None, repr=False)

    def __post_init__(self):
        rng = np.random.default_rng(self.seed)
        # fixed random token embedding
        self.E = rng.standard_normal((self.vocab, self.d_in)).astype(np.float32)
        self.W_in = (self.in_scale *
                     rng.uniform(-1, 1, (self.D, self.d_in)).astype(np.float32))
        # random recurrent matrix, rescaled to the target spectral radius
        W = rng.standard_normal((self.D, self.D)).astype(np.float32)
        eig = np.max(np.abs(np.linalg.eigvals(W)))
        self.W_res = (self.spectral_radius / eig * W).astype(np.float32)

    def reset(self) -> np.ndarray:
        return np.zeros(self.D, dtype=np.float32)

    def step(self, h: np.ndarray, token: int) -> np.ndarray:
        u = self.E[token]
        pre = self.W_res @ h + self.W_in @ u
        return (1.0 - self.leak) * h + self.leak * np.tanh(pre)


# --------------------------------------------------------------------------- #
# Forward-Write language model — delta-rule readout over reservoir features
# --------------------------------------------------------------------------- #
class ForwardWriteLM:
    def __init__(self, reservoir: Reservoir):
        self.res = reservoir
        self.V = reservoir.vocab
        self.W_out = np.zeros((self.V, reservoir.D), dtype=np.float32)

    def _probs(self, h: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        s = (self.W_out @ h) / max(1e-6, temperature)
        s -= s.max()
        e = np.exp(s)
        return e / e.sum()

    def train_epoch(self, sequences: Sequence[List[int]], *, eta: float) -> float:
        """One pass of repeated exposure. Rolls the reservoir, predicts, and
        applies the local delta write. Returns mean train surprise (−log p)."""
        loss, n = 0.0, 0
        for seq in sequences:
            seq = [t for t in seq if 0 <= t < self.V]
            if len(seq) < 2:
                continue
            h = self.res.reset()
            for i in range(len(seq) - 1):
                h = self.res.step(h, seq[i])
                y = seq[i + 1]
                p = self._probs(h)
                loss += -math.log(max(p[y], 1e-12))
                n += 1
                e = -p
                e[y] += 1.0                       # onehot(y) − p
                self.W_out += eta * np.outer(e, h)
        return loss / max(n, 1)

    def perplexity(self, sequences: Sequence[List[int]], *,
                   temperature: float = 1.0, prob_floor: float = 1e-8) -> float:
        """Held-out PPL with the write OFF (W_out frozen)."""
        log_loss, n = 0.0, 0
        for seq in sequences:
            seq = [t for t in seq if 0 <= t < self.V]
            if len(seq) < 2:
                continue
            h = self.res.reset()
            for i in range(len(seq) - 1):
                h = self.res.step(h, seq[i])
                p = self._probs(h, temperature)
                log_loss += -math.log(max(float(p[seq[i + 1]]), prob_floor))
                n += 1
        return math.exp(log_loss / n) if n else float('inf')

    def best_temp_perplexity(self, sequences, temps=(0.25, 0.5, 1.0, 2.0, 4.0)):
        """Fair calibrated PPL: minimum over a temperature sweep (the same
        control Tier-1 used — softmax sharpness must not confound the model
        comparison)."""
        curve = {t: self.perplexity(sequences, temperature=t) for t in temps}
        bt = min(curve, key=curve.get)
        return curve[bt], bt, curve


# --------------------------------------------------------------------------- #
# Baselines
# --------------------------------------------------------------------------- #
def unigram_perplexity(train_seqs, test_seqs, V, *, prob_floor=1e-8) -> float:
    counts = np.ones(V, dtype=np.float64)        # Laplace
    for seq in train_seqs:
        for t in seq:
            if 0 <= t < V:
                counts[t] += 1
    logp = np.log(counts / counts.sum())
    log_loss, n = 0.0, 0
    for seq in test_seqs:
        for i in range(1, len(seq)):
            t = seq[i]
            if 0 <= t < V:
                log_loss += -max(float(logp[t]), math.log(prob_floor))
                n += 1
    return math.exp(log_loss / n) if n else float('inf')


# --------------------------------------------------------------------------- #
# Harness
# --------------------------------------------------------------------------- #
def main():
    train_n = int(os.environ.get('TRAIN_N', '600'))
    test_n = int(os.environ.get('TEST_N', '100'))
    epochs = int(os.environ.get('EPOCHS', '4'))
    max_vocab = int(os.environ.get('VOCAB', '1500'))
    D = int(os.environ.get('D', '400'))
    rho = float(os.environ.get('RHO', '0.9'))
    leak = float(os.environ.get('LEAK', '0.3'))
    eta = float(os.environ.get('ETA', '0.05'))
    seed = int(os.environ.get('SEED', '0'))

    from brain.tasks.llm.tokenizer import WordTokenizer
    from brain.tasks.llm.experiment import load_tinystories
    # bag-of-N substrate-LLM baseline (also forward-only: perceptron)
    from brain.tasks.llm.llm import build_llm_view, train_ngram_epoch, perplexity
    try:
        from brain.tasks.llm.jit import train_ngram_epoch_jit_parallel, NUMBA_AVAILABLE
    except Exception:
        NUMBA_AVAILABLE = False

    print("=== Forward-Write LM — learning by repeated exposure, no backprop ===")
    print(f"train_n={train_n} test_n={test_n} epochs={epochs} vocab={max_vocab} "
          f"D={D} rho={rho} leak={leak} eta={eta}")

    train_texts, test_texts = load_tinystories(train_n=train_n, test_n=test_n,
                                               seed=seed)
    tok = WordTokenizer(max_vocab=max_vocab, min_freq=2)
    tok.fit(train_texts)
    V = tok.get_vocab_size()
    train_seqs = [tok.encode(t) for t in train_texts]
    test_seqs = [tok.encode(t) for t in test_texts]
    n_tok = sum(len(s) for s in train_seqs)
    print(f"  vocab={V}  train_tok={n_tok:,}  uniform PPL={V}\n")

    # --- baselines ---
    uni = unigram_perplexity(train_seqs, test_seqs, V)
    print(f"  unigram floor PPL:        {uni:.2f}")

    view = build_llm_view(tok)
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        if NUMBA_AVAILABLE:
            train_ngram_epoch_jit_parallel(view, train_seqs, context_window=4,
                                           eta=eta, rng=rng)
        else:
            train_ngram_epoch(view, train_seqs, context_window=4, eta=eta, rng=rng)
    # best-temperature: the perceptron W is wildly uncalibrated (huge dynamic
    # range), so a fixed temperature confounds the comparison — same control as
    # Tier-1. Sweep both models, report each at its own best temperature.
    temps = (0.25, 0.5, 1.0, 2.0, 4.0)
    bag_curve = {t: perplexity(view, test_seqs, context_window=4,
                               softmax_temperature=t) for t in temps}
    bag_t = min(bag_curve, key=bag_curve.get)
    bagn = bag_curve[bag_t]
    print(f"  substrate-LLM bag-of-4:   {bagn:.2f} @T={bag_t}  "
          f"(forward-only perceptron, best-temp)\n")

    # --- forward-write ---
    res = Reservoir(vocab=V, D=D, spectral_radius=rho, leak=leak, seed=seed)
    lm = ForwardWriteLM(res)
    print(f"  Forward-Write (reservoir D={D}), repeated exposure:")
    for ep in range(epochs):
        tr = lm.train_epoch(train_seqs, eta=eta)
        ppl = lm.perplexity(test_seqs)
        print(f"    ep {ep+1}/{epochs}  train surprise={tr:.3f}  held-out PPL@T1={ppl:.2f}")
    fw, fw_t, _ = lm.best_temp_perplexity(test_seqs, temps)

    print(f"\n=== held-out PPL, each model at its best temperature (lower = better) ===")
    print(f"  unigram floor          {uni:8.2f}")
    print(f"  substrate-LLM bag-of-4 {bagn:8.2f}  @T={bag_t}")
    print(f"  Forward-Write (ours)   {fw:8.2f}  @T={fw_t}")
    if fw < bagn and fw < uni:
        print(f"\n  verdict: reservoir's unbounded context BEATS bag-of-4 "
              f"({(1-fw/bagn)*100:.0f}% lower) AND unigram — distributed "
              f"forward-only representation is the lever")
    elif fw < uni:
        print(f"\n  verdict: beats unigram floor but NOT a calibrated bag-of-4 — "
              f"reservoir generalizes, but token context still wins")
    else:
        print(f"\n  verdict: does not beat unigram — random reservoir features "
              f"too weak for language; representation must be learned")


if __name__ == '__main__':
    main()
