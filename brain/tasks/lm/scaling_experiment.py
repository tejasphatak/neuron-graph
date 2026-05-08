"""Scaling experiment: does the substrate's RL plateau dissolve with
a larger corpus? Hypothesis: 20 sentences with diverse vocabulary
should give RL much more disambiguating signal than 3 sentences did.

Run:  PYTHONPATH=. python3 brain/tasks/lm/scaling_experiment.py
"""

from __future__ import annotations

import time

import numpy as np

from brain.tasks.lm import build_lm_brain
from brain.tasks.lm.rl import teach_minimal, trace_generate, train_rl


# 20 sentences, all shape: DET ADJ ADJ NOUN VERB PREP DET ADJ NOUN
CORPUS_20 = [
    'the quick brown fox jumps over the lazy dog',
    'a smart small cat runs around a happy mouse',
    'the bold red bird flies past a tiny tree',
    'a big gray wolf chases through a dark forest',
    'the slow blue turtle swims under a bright fish',
    'a fierce young lion hunts behind the small deer',
    'the tired old owl watches over a quiet field',
    'a wild brown horse runs across a green meadow',
    'the soft white sheep sleeps near the tall fence',
    'a silly black cat plays beside a warm fire',
    'the hungry green frog jumps into a clear pond',
    'a bright orange fish swims past the rocky reef',
    'the tiny gray mouse hides under a wooden floor',
    'a strong brave eagle flies above the rocky peak',
    'the cold black snake slithers between the dry leaves',
    'a swift sleek deer leaps over the fallen log',
    'the calm pink rabbit naps inside a cozy burrow',
    'a lazy fat pig rolls in the muddy pen',
    'the friendly red squirrel climbs up a tall oak',
    'a scared little duck waddles across the busy road',
]

POS = ['DET', 'ADJ', 'ADJ', 'NOUN', 'VERB', 'PREP', 'DET', 'ADJ', 'NOUN']


def _vocab_stats(vocab):
    """How many unique tokens of each POS?"""
    return {
        'tokens': len(vocab.token_to_id),
        'pos_classes': len(vocab.pos_to_id),
        'step_neurons': len(vocab.step_to_id),
    }


def evaluate(brain, vocab, sentences, *, rng_seed=0):
    """Greedy generation accuracy across the entire corpus."""
    rng = np.random.default_rng(rng_seed)
    correct = 0
    total = 0
    perfect_sentences = 0
    results = []
    for s in sentences:
        toks = s.split()
        traj = trace_generate(brain, vocab, toks[:3], max_new=6,
                                rng=rng, explore_eps=0.0,
                                pos_sequence=POS)
        target = toks[3:]
        c = sum(1 for a, b in zip(traj.output_tokens, target) if a == b)
        correct += c
        total += len(target)
        if traj.output_tokens == target:
            perfect_sentences += 1
        results.append((toks[:3], target, traj.output_tokens, c))
    return {
        'accuracy': correct / total,
        'token_correct': correct,
        'token_total': total,
        'perfect_sentences': perfect_sentences,
        'total_sentences': len(sentences),
        'results': results,
    }


def main():
    print(f'=== Scaling experiment: {len(CORPUS_20)} sentences ===\n')

    brain, vocab = build_lm_brain()
    for s in CORPUS_20:
        teach_minimal(brain, vocab, s.split(), POS)

    print('Initial state (after teach_minimal, no co_occurs taught):')
    stats = _vocab_stats(vocab)
    print(f'  vocab: {stats["tokens"]} tokens, {stats["pos_classes"]} POS classes')
    print(f'  brain: {brain.size} neurons, {brain._used_synapses} synapses')

    # Cold-start evaluation
    cold = evaluate(brain, vocab, CORPUS_20)
    print(f'\nCold-start accuracy (greedy): {cold["accuracy"]:.1%} '
          f'({cold["token_correct"]}/{cold["token_total"]} tokens, '
          f'{cold["perfect_sentences"]}/{cold["total_sentences"]} sentences perfect)')

    # Train
    training_pairs = [(s.split()[:3], s.split()[3:], POS) for s in CORPUS_20]

    print(f'\n=== RL training ({len(training_pairs)} pairs, 200 epochs) ===')
    t0 = time.time()
    history = train_rl(brain, vocab, training_pairs,
                        epochs=200,
                        eta=0.18,
                        explore_eps_start=0.40,
                        explore_eps_end=0.02,
                        seed=0,
                        verbose=False)
    elapsed = time.time() - t0
    print(f'Training time: {elapsed:.1f}s ({elapsed/200*1000:.0f} ms/epoch)')

    print('\nLearning curve:')
    for h in history[::20] + [history[-1]]:
        print(f'  ep {h["epoch"]:3d}  acc={h["accuracy"]:.1%}  '
              f'r̄={h["mean_reward"]:+.2f}  ε={h["eps"]:.2f}  '
              f'upd={h["updated"]:5d}  new={h["created"]:4d}')

    best = max(history, key=lambda h: h['accuracy'])
    print(f'\nBest epoch: {best["epoch"]}, accuracy: {best["accuracy"]:.1%}')

    # Final brain
    print(f'\nFinal brain: {brain.size} neurons, {brain._used_synapses} synapses')
    print(f'Synapses grown by RL: {brain._used_synapses - 143 * len(CORPUS_20) // 3}'
          f' (very rough)')

    # Final evaluation
    final = evaluate(brain, vocab, CORPUS_20)
    print(f'\nFinal greedy accuracy: {final["accuracy"]:.1%} '
          f'({final["token_correct"]}/{final["token_total"]} tokens, '
          f'{final["perfect_sentences"]}/{final["total_sentences"]} sentences perfect)')

    # Show a few sample results
    print('\nSample generations (showing first 5):')
    for prompt, target, generated, n_correct in final['results'][:5]:
        ok = 'OK' if generated == target else 'XX'
        print(f'  {ok} {" ".join(prompt):24} -> {" ".join(generated):42} ({n_correct}/6)')

    # Compare to 3-sentence baseline
    print('\n=== Plateau comparison ===')
    print(f'  3-sentence corpus best : ~83%')
    print(f'  20-sentence corpus     : {best["accuracy"]:.1%} best, {final["accuracy"]:.1%} final')

    return final, history


if __name__ == '__main__':
    main()
