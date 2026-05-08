"""Bandit v2 — uses replay + modulator. Measure: does it converge faster?

A/B: same task (left pays 80%, right pays 20%) with two configs:
  - V1: targeted update only (Phase A baseline)
  - V2: targeted update + modulator + replay consolidation every K trials

If V2 reaches a target left-rate (say 0.7) in fewer trials, the new
substrate primitives are pulling weight.
"""

from __future__ import annotations

import math
import random
from typing import Dict, List, Tuple

from .learn_bandit import (
    Bandit, build_agent_brain, softmax_choice, _edge_weight,
    TrialRecord,
)
from .replay import ReplayBuffer, consolidate, Episode
from .neuron import SYNAPSE_DTYPE
from .spread import spread


def _targeted(brain, from_id, to_id, *, eta, reward):
    delta = eta * (2 * reward - 1)
    edges = brain.synapses_of(from_id)
    matches = (edges['to_id'] == to_id).nonzero()[0]
    if len(matches) > 0:
        base = int(brain.nodes[from_id]['syn_offset']) // SYNAPSE_DTYPE.itemsize
        idx = base + int(matches[0])
        old = float(brain.synapses[idx]['weight'])
        brain.synapses[idx]['weight'] = max(0.0, min(1.0, old + delta))
    else:
        from .store import Brain
        initial = max(0.0, min(1.0, 0.5 + delta))
        brain.add_synapse(from_id, to_id, rel_name='co_occurs', weight=initial)


def run_bandit_v2(
    *, n_trials: int = 200, eta: float = 0.15,
    initial_temperature: float = 2.0, final_temperature: float = 0.1,
    rng_seed: int = 42,
    use_replay: bool = True, replay_every: int = 25, replay_n: int = 8,
    use_modulator: bool = True,
) -> Tuple[List[TrialRecord], Dict]:
    rng = random.Random(rng_seed)
    brain, ids = build_agent_brain()
    bandit = Bandit(p_left=0.8, p_right=0.2)
    buf = ReplayBuffer(capacity=200) if use_replay else None
    history: List[TrialRecord] = []

    for t in range(n_trials):
        # Spread
        s = spread(brain, [ids['context']], max_steps=4)
        a_left = s.activation.get(ids['left'], 0.0)
        a_right = s.activation.get(ids['right'], 0.0)

        # Anneal temp
        progress = t / max(1, n_trials - 1)
        temp = initial_temperature + progress * (final_temperature - initial_temperature)
        idx = softmax_choice([a_left, a_right], temp, rng)
        action_name = ['left', 'right'][idx]
        action_id = ids[action_name]

        reward = bandit.pull(action_name)

        # Effective eta = base * modulator
        eff_eta = brain.modulator.effective_eta(eta) if use_modulator else eta
        _targeted(brain, ids['context'], action_id,
                   eta=eff_eta, reward=reward)

        # Update modulator from this reward
        if use_modulator:
            brain.modulator.adjust(reward_signal=2 * reward - 1, scale=0.2)

        # Record into replay buffer
        if buf is not None:
            buf.record(trajectory=[('ctx', action_id)], reward=reward)

        # Consolidate periodically — replay sampled past episodes at lower eta
        if buf is not None and (t + 1) % replay_every == 0 and len(buf) >= replay_n:
            def credit_replay(ep: Episode):
                _targeted(brain, ids['context'],
                           ep.trajectory[0][1],
                           eta=eff_eta * 0.4,    # reduced eta for replay
                           reward=ep.reward)
            consolidate(buf, n_samples=replay_n, credit_fn=credit_replay,
                         prefer_recent=False, rng=rng)

        history.append(TrialRecord(
            trial=t, action=action_name, reward=reward,
            act_left=a_left, act_right=a_right,
            weight_context_left=_edge_weight(brain, ids['context'], ids['left']),
            weight_context_right=_edge_weight(brain, ids['context'], ids['right']),
        ))

    return history, {
        'final_left_rate': sum(1 for h in history[-50:] if h.action == 'left') / 50,
        'final_win_rate': sum(h.reward for h in history[-50:]) / 50,
    }


def trials_to_target(history: List[TrialRecord], target_left_rate: float = 0.7,
                      window: int = 25) -> int:
    """Earliest trial index where rolling left-choice rate exceeds target."""
    for i in range(window, len(history)):
        recent = history[i - window:i]
        rate = sum(1 for h in recent if h.action == 'left') / len(recent)
        if rate >= target_left_rate:
            return i
    return -1   # never reached


def main():
    print('=' * 70)
    print('  BANDIT v2 — A/B: Phase A baseline vs Phase A++ with replay/modulator')
    print('=' * 70)
    print('  Task: left pays 80%, right 20%. Target: left-rate ≥ 0.7 (rolling 25)')
    print()

    # Run multiple seeds to reduce variance
    results_a = {'baseline': [], 'replay_modulator': []}
    n_seeds = 5
    n_trials = 200
    target = 0.7

    print(f'  Running {n_seeds} seeds × {n_trials} trials per config ...\n')

    for seed in range(n_seeds):
        # Baseline: no replay, no modulator
        h_base, _ = run_bandit_v2(n_trials=n_trials, rng_seed=seed,
                                    use_replay=False, use_modulator=False)
        # V2: with replay and modulator
        h_v2, _ = run_bandit_v2(n_trials=n_trials, rng_seed=seed,
                                  use_replay=True, use_modulator=True)

        ttt_base = trials_to_target(h_base, target)
        ttt_v2 = trials_to_target(h_v2, target)
        results_a['baseline'].append(ttt_base)
        results_a['replay_modulator'].append(ttt_v2)

        final_base = sum(1 for h in h_base[-50:] if h.action == 'left') / 50
        final_v2 = sum(1 for h in h_v2[-50:] if h.action == 'left') / 50

        print(f'  seed {seed}:  '
              f'baseline trials_to_target={ttt_base:>3} (final left={final_base:.2f}) | '
              f'v2 trials_to_target={ttt_v2:>3} (final left={final_v2:.2f})')

    def mean(xs): return sum(xs) / len(xs) if xs else -1
    base_mean = mean([x for x in results_a['baseline'] if x >= 0])
    v2_mean = mean([x for x in results_a['replay_modulator'] if x >= 0])

    print()
    print(f'  baseline:           reached target in {results_a["baseline"]} '
          f'(mean {base_mean:.1f})')
    print(f'  replay+modulator:   reached target in {results_a["replay_modulator"]} '
          f'(mean {v2_mean:.1f})')

    if base_mean > 0 and v2_mean > 0:
        delta = base_mean - v2_mean
        print(f'\n  Δ trials: {delta:+.1f}  '
              f'({"v2 is faster" if delta > 0 else "no improvement / slower"})')


if __name__ == '__main__':
    main()
