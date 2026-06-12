"""Tier-1 LLM bridge — degree-n associative readout on the substrate-LLM.

The dip test for "can the neuron-astrocyte capacity law touch real language?"

The existing substrate-LLM readout is **degree-1 and linear**:

    c[V] = Σ_k  decay^k · W[ctx_row_k]          # llm.perplexity / generate_text

That sum *is* the astrocyte gather at degree 1, with no nonlinearity and no
scatter. Tier-1 lifts it to the full degree-n Dense Associative Memory pass
over the **same trained W**, with zero retraining:

    every vocabulary token μ is an astrocyte; its synaptic island is its W row.
    overlap_μ = Σ_i W[μ,i] · c_i        # GATHER  (cue → astrocyte calcium)
    glio_μ    = sign(overlap_μ)·|overlap_μ|^(degree-1)   # F'(z)=z^(n-1)
    c_clean_i = Σ_μ W[μ,i] · glio_μ      # SCATTER (astrocyte → token)

One synchronous cleanup pass pulls the noisy linear readout c toward the nearest
stored continuation profile(s). The hypothesis under test: if the capacity law
proven on synthetics (capacity ~N^(n-1), basins grow with degree) transfers to
language statistics, **test perplexity should fall as degree rises 1 → 2 → 4**,
with no change to W. If PPL does not move, the synthetic result does not transfer
— a finding either way.

This reuses ``NeuronAstrocyteMemory`` (the dense reference, proven bit-equal to
the substrate CSR wiring in ``tests/test_astrocyte.py``) with ``activation=
"identity"`` — the readout cue is a real-valued score vector, not a bipolar
pattern, so the sign() activation must be off.

Run:
  PYTHONPATH=. python3 -m brain.tasks.llm.tier1_dam_readout
Env vars mirror experiment.py (TRAIN_N, TEST_N, EPOCHS, ETA, CONTEXT, VOCAB).
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence

import numpy as np

from brain.astrocyte import NeuronAstrocyteMemory
from .llm import LLMView


# --------------------------------------------------------------------------- #
# The degree-n associative readout
# --------------------------------------------------------------------------- #
class DAMReadout:
    """Wraps a trained ``LLMView`` with a degree-n associative cleanup pass.

    Patterns are the rows of ``view.W`` (one stored continuation profile per
    vocabulary token). Built once, reused across every prediction.
    """

    def __init__(self, view: LLMView, *, interaction: str = "poly",
                 degree: int = 4, beta: float = 1.0):
        self.view = view
        self.degree = degree
        self.interaction = interaction
        # identity activation: the cue is a score vector, not a ±1 pattern.
        self.mem = NeuronAstrocyteMemory(interaction=interaction, degree=degree,
                                         beta=beta, activation="identity")
        self.mem.store(view.W)            # K=V astrocytes, island = W row (V wide)

    def clean(self, c: np.ndarray) -> np.ndarray:
        """One gather → F' → scatter pass over the readout cue ``c`` (V,)."""
        return self.mem.step(c)


# --------------------------------------------------------------------------- #
# Perplexity under the degree-n readout (degree=1 reproduces llm.perplexity)
# --------------------------------------------------------------------------- #
def perplexity_dam(view: LLMView, sequences: Sequence[List[int]], *,
                   degree: int = 1, interaction: str = "poly", beta: float = 1.0,
                   context_window: int = 4, decay: float = 0.6,
                   residual: float = 0.0, standardize: bool = True,
                   softmax_temperature: float = 1.0, prob_floor: float = 1e-8,
                   max_preds: Optional[int] = None,
                   readout: Optional[DAMReadout] = None) -> float:
    """Perplexity using the degree-n associative readout.

    degree=1 → no cleanup pass: ``scores = c`` (identical to ``llm.perplexity``,
    the linear baseline). degree>=2 → ``scores = c + residual·c  cleaned`` where
    the clean term is one DAM pass.

    standardize z-scores the score vector before the temperature softmax so the
    *shape* change from the nonlinearity is compared at matched sharpness, not
    confounded by the raw magnitude blow-up of z^(degree-1).
    """
    V = view.W.shape[1]
    decay_powers = [decay ** k for k in range(context_window)]
    # A cleanup pass is needed unless this is the pure degree-1 linear baseline.
    need_readout = (interaction == "softmax") or (interaction == "poly" and degree >= 2)
    if need_readout and readout is None:
        readout = DAMReadout(view, interaction=interaction, degree=degree,
                             beta=beta)

    log_loss = 0.0
    n_pred = 0
    for seq in sequences:
        rows = [view.tok_to_row.get(t, -1) for t in seq]
        for i in range(1, len(rows)):
            target = rows[i]
            if target < 0:
                continue
            # degree-1 linear readout cue c = Σ_k decay^k W[ctx_k]
            c = np.zeros(V, dtype=np.float32)
            any_ctx = False
            for k in range(context_window):
                j = i - 1 - k
                if j < 0:
                    break
                if rows[j] < 0:
                    continue
                c += decay_powers[k] * view.W[rows[j]]
                any_ctx = True
            if not any_ctx:
                continue

            if readout is not None:
                clean = readout.clean(c).astype(np.float32)
                scores = clean if residual == 0.0 else (residual * c + clean)
            else:
                scores = c

            if standardize:
                mu = float(scores.mean())
                sd = float(scores.std())
                if sd > 1e-12:
                    scores = (scores - mu) / sd
            scores = scores / max(1e-6, softmax_temperature)
            scores = scores - scores.max()
            exp_s = np.exp(scores)
            denom = exp_s.sum()
            if denom <= 0:
                continue
            prob = max(float(exp_s[target] / denom), prob_floor)
            log_loss += -math.log(prob)
            n_pred += 1
            if max_preds is not None and n_pred >= max_preds:
                return math.exp(log_loss / n_pred)

    if n_pred == 0:
        return float('inf')
    return math.exp(log_loss / n_pred)


def degree_sweep(view: LLMView, sequences: Sequence[List[int]], *,
                 degrees: Sequence[int] = (1, 2, 3, 4),
                 context_window: int = 4, decay: float = 0.6,
                 softmax_temperature: float = 1.0, standardize: bool = True,
                 max_preds: Optional[int] = None) -> dict:
    """PPL at each degree over the same trained W at a single temperature.
    degree 1 is the linear baseline; ``+softmax`` (attention readout) appended."""
    out = {}
    for d in degrees:
        out[f"deg{d}"] = perplexity_dam(
            view, sequences, degree=d, interaction="poly",
            context_window=context_window, decay=decay,
            softmax_temperature=softmax_temperature, standardize=standardize,
            max_preds=max_preds)
    out["softmax"] = perplexity_dam(
        view, sequences, degree=2, interaction="softmax",
        context_window=context_window, decay=decay,
        softmax_temperature=softmax_temperature, standardize=standardize,
        max_preds=max_preds)
    return out


def degree_temp_sweep(view: LLMView, sequences: Sequence[List[int]], *,
                      degrees: Sequence[int] = (1, 2, 3, 4),
                      temps: Sequence[float] = (0.25, 0.5, 1.0, 2.0, 4.0),
                      context_window: int = 4, decay: float = 0.6,
                      standardize: bool = True,
                      max_preds: Optional[int] = None) -> dict:
    """Best-temperature PPL per degree — the fair degree comparison.

    Each degree's power nonlinearity yields a differently-peaked score
    distribution, so a single temperature confounds the nonlinearity's *shape*
    with its *sharpness*. Here each readout is built once and swept over
    ``temps``; we report the minimum PPL (best temperature) per degree. This
    isolates whether a sharper associative nonlinearity genuinely models
    language better, independent of softmax calibration.

    Returns {label: {"ppl": best_ppl, "temp": best_temp, "curve": {t: ppl}}}.
    """
    def sweep_one(label, readout):
        curve = {}
        for t in temps:
            curve[t] = perplexity_dam(
                view, sequences, context_window=context_window, decay=decay,
                softmax_temperature=t, standardize=standardize,
                max_preds=max_preds, readout=readout,
                # degree/interaction only matter when readout is None (deg1)
                degree=1 if readout is None else 2,
                residual=0.0)
        best_t = min(curve, key=curve.get)
        return {"ppl": curve[best_t], "temp": best_t, "curve": curve}

    out = {}
    out["deg1"] = sweep_one("deg1", None)                 # linear baseline
    for d in degrees:
        if d < 2:
            continue
        ro = DAMReadout(view, interaction="poly", degree=d)
        out[f"deg{d}"] = sweep_one(f"deg{d}", ro)
    out["softmax"] = sweep_one(
        "softmax", DAMReadout(view, interaction="softmax", degree=2))
    return out


# --------------------------------------------------------------------------- #
# Harness: train one W, swap the readout, report the table
# --------------------------------------------------------------------------- #
def _build_trained_view(*, train_n: int, test_n: int, epochs: int, eta: float,
                        context_window: int, max_vocab: int, seed: int = 0):
    """Mirror experiment.py: load TinyStories, fit tokenizer, train W."""
    import time
    from brain.tasks.llm.tokenizer import WordTokenizer
    from brain.tasks.llm.llm import build_llm_view, train_ngram_epoch
    from brain.tasks.llm.experiment import load_tinystories
    try:
        from brain.tasks.llm.jit import (
            train_ngram_epoch_jit_parallel, NUMBA_AVAILABLE)
    except Exception:
        NUMBA_AVAILABLE = False

    train_texts, test_texts = load_tinystories(train_n=train_n, test_n=test_n,
                                               seed=seed)
    tokenizer = WordTokenizer(max_vocab=max_vocab, min_freq=2)
    tokenizer.fit(train_texts)
    V = tokenizer.get_vocab_size()
    view = build_llm_view(tokenizer)
    train_seqs = [tokenizer.encode(t) for t in train_texts]
    test_seqs = [tokenizer.encode(t) for t in test_texts]

    rng = np.random.default_rng(0)
    print(f"  vocab={V}  W={view.W.nbytes/1e6:.1f}MB  "
          f"train_tok={sum(len(s) for s in train_seqs):,}")
    for ep in range(epochs):
        t0 = time.time()
        if NUMBA_AVAILABLE:
            m = train_ngram_epoch_jit_parallel(
                view, train_seqs, context_window=context_window, eta=eta, rng=rng)
        else:
            m = train_ngram_epoch(
                view, train_seqs, context_window=context_window, eta=eta, rng=rng)
        print(f"  ep {ep+1}/{epochs}  acc={m['next_token_accuracy']:.1%}  "
              f"{time.time()-t0:.1f}s")
    return view, test_seqs, V


def main():
    import os
    train_n = int(os.environ.get('TRAIN_N', '1000'))
    test_n = int(os.environ.get('TEST_N', '100'))
    epochs = int(os.environ.get('EPOCHS', '5'))
    eta = float(os.environ.get('ETA', '0.05'))
    context_window = int(os.environ.get('CONTEXT', '4'))
    max_vocab = int(os.environ.get('VOCAB', '2000'))
    temp = float(os.environ.get('TEMP', '1.0'))
    max_preds = int(os.environ.get('MAX_PREDS', '4000'))

    print(f"=== Tier-1 degree-n associative readout (TinyStories) ===")
    print(f"train_n={train_n} test_n={test_n} epochs={epochs} "
          f"ctx={context_window} vocab={max_vocab}")
    view, test_seqs, V = _build_trained_view(
        train_n=train_n, test_n=test_n, epochs=epochs, eta=eta,
        context_window=context_window, max_vocab=max_vocab)

    temps = (0.1, 0.15, 0.25, 0.5, 1.0, 2.0, 4.0)
    print(f"\nEvaluating readouts on {len(test_seqs)} held-out stories "
          f"(<= {max_preds} preds, uniform PPL = {V})")
    print(f"best-temperature sweep over temps={temps} (fair degree comparison)...")
    res = degree_temp_sweep(view, test_seqs, degrees=(2, 3, 4), temps=temps,
                            context_window=context_window, standardize=True,
                            max_preds=max_preds)

    print(f"\n  readout       best PPL   @temp     vs deg1")
    print(f"  ------------- --------   -----    -------")
    base = res["deg1"]["ppl"]
    for k in ("deg1", "deg2", "deg3", "deg4", "softmax"):
        r = res[k]
        tag = "  (linear baseline)" if k == "deg1" else \
              f"  {(r['ppl']/base - 1)*100:+.1f}%"
        print(f"  {k:<13} {r['ppl']:8.2f}   {r['temp']:>4}{tag}")
    best = min(((res[k]['ppl'], k) for k in res))
    verdict = ("degree helps — associative cleanup beats the linear readout"
               if best[1] != "deg1"
               else "degree does NOT help on language")
    print(f"\n  best: {best[1]} @ PPL {best[0]:.2f}  ({verdict})")


if __name__ == '__main__':
    main()
