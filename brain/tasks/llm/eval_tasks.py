"""Task-based evaluation suite for substrate-LLM.

PPL is a training-time signal, not the bottom line. These functions
test whether the substrate-LLM can do USEFUL things:

1. cloze_test(view, tokenizer, sentence, mask_position)
   Given "the cat sat on the ___ .", does substrate predict 'mat'?
   Top-1 and top-K accuracy reported.

2. sentence_completion_eval(view, tokenizer, prompts, references)
   Given prompt prefixes, generate continuations. Measure overlap
   with held-out references (precision, recall, F1 over tokens).

3. cloze_benchmark(view, tokenizer, test_sequences, n_samples)
   Run cloze on N random positions across test sequences.
   Returns top-1 / top-5 / top-10 accuracy + average rank of true target.

Designed to give a real "is this thing useful?" answer beyond PPL.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from .llm import LLMView
from .tokenizer import WordTokenizer


def cloze_score(view: LLMView,
                  context_rows: List[int],
                  context_weights: List[float],
                  target_row: int) -> Tuple[int, int]:
    """Score how well substrate predicts target_row given context.

    Returns (rank_of_target, top1_correct).
    rank_of_target: 0 = target was top-1, 1 = top-2, ...
    """
    V = view.W.shape[1]
    scores = np.zeros(V, dtype=np.float32)
    for r, w in zip(context_rows, context_weights):
        scores += w * view.W[r]
    sorted_idx = np.argsort(-scores)
    rank = int(np.where(sorted_idx == target_row)[0][0])
    top1 = int(sorted_idx[0] == target_row)
    return rank, top1


def cloze_benchmark(view: LLMView, tokenizer: WordTokenizer,
                      test_sequences: List[List[int]], *,
                      n_samples: int = 500,
                      context_window: int = 4,
                      decay: float = 0.6,
                      min_position: int = 5,
                      rng: Optional[np.random.Generator] = None) -> Dict[str, float]:
    """Run cloze test: pick N random (sequence, position) pairs from
    test_sequences, score how well substrate predicts the actual token
    given preceding context.

    Returns:
      top1_acc, top5_acc, top10_acc — fraction of cloze tests where
        true target was in top-K
      mean_rank — average rank of true target (0-indexed; lower = better)
      median_rank — median rank
      n_evaluated — number of cloze positions actually scored
    """
    if rng is None:
        rng = np.random.default_rng()
    decay_powers = [decay ** k for k in range(context_window)]

    # Collect eligible positions (need at least min_position tokens of context)
    eligible: List[Tuple[int, int]] = []
    for s_idx, seq in enumerate(test_sequences):
        for i in range(min_position, len(seq)):
            eligible.append((s_idx, i))
    if len(eligible) > n_samples:
        sel_idx = rng.choice(len(eligible), size=n_samples, replace=False)
        eligible = [eligible[k] for k in sel_idx]

    ranks: List[int] = []
    top1 = 0
    top5 = 0
    top10 = 0
    n_evaluated = 0

    for s_idx, i in eligible:
        seq = test_sequences[s_idx]
        target_row = view.tok_to_row.get(seq[i], -1)
        if target_row < 0:
            continue
        ctx_rows: List[int] = []
        ctx_w: List[float] = []
        for back in range(context_window):
            j = i - 1 - back
            if j < 0:
                break
            r = view.tok_to_row.get(seq[j])
            if r is None:
                continue
            ctx_rows.append(r)
            ctx_w.append(decay_powers[back])
        if not ctx_rows:
            continue
        rank, t1 = cloze_score(view, ctx_rows, ctx_w, target_row)
        ranks.append(rank)
        top1 += t1
        if rank < 5:
            top5 += 1
        if rank < 10:
            top10 += 1
        n_evaluated += 1

    if n_evaluated == 0:
        return {'n_evaluated': 0}

    return {
        'top1_acc': top1 / n_evaluated,
        'top5_acc': top5 / n_evaluated,
        'top10_acc': top10 / n_evaluated,
        'mean_rank': float(np.mean(ranks)),
        'median_rank': float(np.median(ranks)),
        'n_evaluated': n_evaluated,
    }


def sentence_completion_overlap(generated: List[str],
                                  reference: List[str]) -> Dict[str, float]:
    """Token-level precision/recall/F1 between generated and reference."""
    gen_tokens = generated
    ref_tokens = reference
    if not gen_tokens or not ref_tokens:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    # Bag-of-tokens overlap (ignores order)
    from collections import Counter
    gen_c = Counter(gen_tokens)
    ref_c = Counter(ref_tokens)
    overlap = sum((gen_c & ref_c).values())
    p = overlap / max(1, len(gen_tokens))
    r = overlap / max(1, len(ref_tokens))
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return {'precision': p, 'recall': r, 'f1': f1}


def sentence_completion_eval(view: LLMView, tokenizer: WordTokenizer,
                                test_sequences: List[List[int]], *,
                                n_samples: int = 50,
                                prompt_tokens: int = 5,
                                ref_tokens: int = 15,
                                generate_kwargs: Optional[Dict] = None,
                                rng: Optional[np.random.Generator] = None
                                ) -> Dict[str, float]:
    """For each test sequence, take first prompt_tokens as prompt,
    generate ref_tokens continuation, compare to actual continuation.
    """
    from .llm import generate_text
    if generate_kwargs is None:
        generate_kwargs = {
            'temperature': 0.7,
            'top_p': 0.9,
            'repetition_penalty': 1.3,
            'no_repeat_ngram': 3,
        }
    if rng is None:
        rng = np.random.default_rng()

    eligible = [s for s in test_sequences
                if len(s) >= prompt_tokens + ref_tokens]
    if len(eligible) > n_samples:
        sel_idx = rng.choice(len(eligible), size=n_samples, replace=False)
        eligible = [eligible[k] for k in sel_idx]

    scores = []
    n_evaluated = 0
    for seq in eligible:
        prompt_ids = seq[:prompt_tokens]
        ref_ids = seq[prompt_tokens:prompt_tokens + ref_tokens]
        prompt_text = tokenizer.decode(prompt_ids, skip_special=True)
        gen_text = generate_text(view, tokenizer, prompt_text,
                                   max_new=ref_tokens,
                                   rng=rng, **generate_kwargs)
        # Tokenize both for comparison
        gen_tokens = gen_text.split()
        ref_text = tokenizer.decode(ref_ids, skip_special=True)
        ref_tokens_list = ref_text.split()
        sc = sentence_completion_overlap(gen_tokens, ref_tokens_list)
        scores.append(sc)
        n_evaluated += 1

    if n_evaluated == 0:
        return {'n_evaluated': 0}

    return {
        'precision': float(np.mean([s['precision'] for s in scores])),
        'recall': float(np.mean([s['recall'] for s in scores])),
        'f1': float(np.mean([s['f1'] for s in scores])),
        'n_evaluated': n_evaluated,
    }


def random_baseline_cloze(V: int, n_samples: int = 500) -> Dict[str, float]:
    """Theoretical random-baseline scores for cloze test.
    Random picks → 1/V chance for top-1, 5/V for top-5, etc."""
    return {
        'top1_acc': 1.0 / V,
        'top5_acc': 5.0 / V,
        'top10_acc': 10.0 / V,
        'mean_rank': V / 2.0,
        'median_rank': V / 2.0,
        'n_evaluated': n_samples,
    }
