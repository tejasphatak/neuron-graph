"""Substrate-as-agent test: a simple two-arm bandit.

The point: prove the substrate can ACQUIRE behavior from experience.
No new architecture — just the existing spread + Hebbian primitives,
driven by a reward signal from the environment.

Setup:
    Three new neurons added to a fresh brain:
        'context'  — always present at trial start (the "situation")
        'left'     — action 1
        'right'    — action 2
    No initial synapses between them. The brain knows nothing about
    which action is good.

Environment:
    Reward function: choosing 'left' pays +1.0 with probability p_left;
    'right' pays +1.0 with probability p_right. By default p_left=0.8,
    p_right=0.2 — so 'left' is the correct answer.

Agent loop, per trial:
    1. Spread from {context} through the brain
    2. Read activation of 'left' and 'right'; pick one stochastically
       (softmax over activation; epsilon-greedy or pure activation-based)
    3. Sample the environment's reward
    4. Hebbian update: strengthen co-active (context, chosen_action)
       proportional to the reward
    5. Repeat

Success criterion (proven, not asserted):
    After N trials, win-rate over the last K trials should be
    substantially higher than chance (0.5) — approaching p_left if
    the agent has learned to prefer 'left'.

Failure modes worth distinguishing:
    a) Hebbian doesn't change synapses → no learning curve, win-rate ~0.5
    b) Hebbian strengthens but spreading doesn't carry the signal →
       weights change but action choice doesn't
    c) Both pieces work → win-rate climbs over trials
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from .store import Brain
from .neuron import NeuronType
from .spread import spread, ActivationState
from .learn import hebbian_update


# ─── Environment ──────────────────────────────────────────────────────────

@dataclass
class Bandit:
    """Two-arm bandit. left pays p_left of the time; right pays p_right."""
    p_left: float = 0.8
    p_right: float = 0.2
    rng: random.Random = None

    def __post_init__(self):
        self.rng = self.rng or random.Random(0xC0FFEE)

    def pull(self, action: str) -> float:
        if action == 'left':
            return 1.0 if self.rng.random() < self.p_left else 0.0
        if action == 'right':
            return 1.0 if self.rng.random() < self.p_right else 0.0
        raise ValueError(f'unknown action: {action!r}')


# ─── Build the agent's brain ──────────────────────────────────────────────

def build_agent_brain() -> Tuple[Brain, dict]:
    """Tiny brain: just context, left, right. No prior knowledge."""
    b = Brain()
    ids = {
        'context': b.add_neuron(lemma='context', type=NeuronType.CONCEPT),
        'left':    b.add_neuron(lemma='left',    type=NeuronType.RULE),
        'right':   b.add_neuron(lemma='right',   type=NeuronType.RULE),
    }
    return b, ids


# ─── Action selection ─────────────────────────────────────────────────────

def softmax_choice(activations: List[float], temperature: float,
                    rng: random.Random) -> int:
    """Sample an index proportional to softmax(activation / temperature).
    Lower temperature → more greedy. Higher → more exploratory."""
    if temperature <= 0:
        # Greedy
        return int(np.argmax(activations))
    scaled = [a / temperature for a in activations]
    m = max(scaled)
    exps = [math.exp(s - m) for s in scaled]
    total = sum(exps)
    if total == 0:
        return rng.randrange(len(activations))
    probs = [e / total for e in exps]
    r = rng.random()
    cum = 0.0
    for i, p in enumerate(probs):
        cum += p
        if r <= cum:
            return i
    return len(activations) - 1


# ─── Trial loop ───────────────────────────────────────────────────────────

@dataclass
class TrialRecord:
    trial: int
    action: str
    reward: float
    act_left: float
    act_right: float
    weight_context_left: float
    weight_context_right: float


def run_trials(
    brain: Brain,
    ids: dict,
    bandit: Bandit,
    *,
    n_trials: int = 200,
    eta: float = 0.1,
    initial_temperature: float = 1.0,
    final_temperature: float = 0.05,
    rng_seed: int = 42,
    update_policy: str = 'targeted',
) -> List[TrialRecord]:
    """Run `n_trials` of pull-action-reward-update. Return per-trial records.

    update_policy:
      'hebbian' — generic hebbian_update on joint co-activation. Symmetric;
                  strengthens all co-active pairs. Found to saturate quickly
                  with only 3 neurons.
      'targeted' — directly nudge weight(context, chosen_action) by
                   eta * (2*reward - 1). Reward-modulated TD-style update.
                   Treats reward=0 as negative signal so unhelpful actions
                   shrink, not just stagnate.

    Temperature decays linearly from initial → final over the run.
    """
    rng = random.Random(rng_seed)
    history: List[TrialRecord] = []

    for t in range(n_trials):
        # Spread from context
        s = spread(brain, [ids['context']], max_steps=4)
        a_left = s.activation.get(ids['left'], 0.0)
        a_right = s.activation.get(ids['right'], 0.0)

        # Anneal temperature
        progress = t / max(1, n_trials - 1)
        temperature = initial_temperature + progress * (
            final_temperature - initial_temperature
        )
        idx = softmax_choice([a_left, a_right], temperature, rng)
        action = ['left', 'right'][idx]

        # Pull the bandit
        reward = bandit.pull(action)

        # Update synapses based on reward
        if update_policy == 'targeted':
            _targeted_update(brain, ids['context'], ids[action],
                             eta=eta, reward=reward)
        elif update_policy == 'hebbian':
            joint = spread(brain, [ids['context'], ids[action]], max_steps=2)
            scaled_reward = (2 * reward - 1)
            hebbian_update(
                brain, joint,
                eta=eta, reward=scaled_reward,
                co_threshold=0.05, create_threshold=0.05,
            )
        else:
            raise ValueError(f'unknown update_policy: {update_policy!r}')

        # Record current weights for analysis
        w_left = _edge_weight(brain, ids['context'], ids['left'])
        w_right = _edge_weight(brain, ids['context'], ids['right'])

        history.append(TrialRecord(
            trial=t, action=action, reward=reward,
            act_left=a_left, act_right=a_right,
            weight_context_left=w_left,
            weight_context_right=w_right,
        ))

    return history


def _targeted_update(brain: Brain, from_id: int, to_id: int, *,
                      eta: float, reward: float) -> None:
    """Directly modify the (from→to) synapse weight by eta*(2*reward−1).
    Creates the synapse if it doesn't exist. Substrate primitive: just
    finds the synapse offset and writes to it."""
    delta = eta * (2 * reward - 1)
    edges = brain.synapses_of(from_id)
    matches = (edges['to_id'] == to_id).nonzero()[0]
    if len(matches) > 0:
        # Modify in place — convert local index to absolute
        from .neuron import SYNAPSE_DTYPE
        base = int(brain.nodes[from_id]['syn_offset']) // SYNAPSE_DTYPE.itemsize
        abs_idx = base + int(matches[0])
        old = float(brain.synapses[abs_idx]['weight'])
        new = max(0.0, min(1.0, old + delta))
        brain.synapses[abs_idx]['weight'] = new
    else:
        # Create with the delta as initial weight (clamped)
        initial = max(0.0, min(1.0, 0.5 + delta))
        brain.add_synapse(from_id, to_id, rel_name='co_occurs', weight=initial)


def _edge_weight(brain: Brain, from_id: int, to_id: int) -> float:
    edges = brain.synapses_of(from_id)
    if len(edges) == 0:
        return 0.0
    matches = (edges['to_id'] == to_id).nonzero()[0]
    if len(matches) == 0:
        return 0.0
    return float(edges[matches[0]]['weight'])


# ─── Analysis ─────────────────────────────────────────────────────────────

def windowed_winrate(history: List[TrialRecord], window: int = 20) -> List[float]:
    rates = []
    for i in range(len(history)):
        lo = max(0, i - window + 1)
        rewards = [h.reward for h in history[lo:i + 1]]
        rates.append(sum(rewards) / len(rewards) if rewards else 0.0)
    return rates


def left_choice_rate(history: List[TrialRecord], window: int = 20) -> List[float]:
    rates = []
    for i in range(len(history)):
        lo = max(0, i - window + 1)
        chosen = [1 if h.action == 'left' else 0 for h in history[lo:i + 1]]
        rates.append(sum(chosen) / len(chosen) if chosen else 0.0)
    return rates


# ─── Demo ─────────────────────────────────────────────────────────────────

def main():
    print('=' * 70)
    print('  SUBSTRATE-AS-AGENT — two-arm bandit learning test')
    print('=' * 70)
    print('  Setup: empty brain (context, left, right; NO prior synapses).')
    print('  Reward: left pays 80%, right pays 20%. left is correct.')
    print('  Question: does the substrate, via Hebbian-only updates from')
    print('  reward signals, learn to prefer left over time?')
    print('=' * 70)

    # Run BOTH policies for an honest comparison
    print('\n  --- Policy A: generic hebbian_update (symmetric) ---')
    brain_h, ids_h = build_agent_brain()
    bandit_h = Bandit(p_left=0.8, p_right=0.2)
    history_h = run_trials(
        brain_h, ids_h, bandit_h,
        n_trials=200, eta=0.15,
        initial_temperature=2.0, final_temperature=0.1,
        update_policy='hebbian',
    )
    final_win_h = sum(h.reward for h in history_h[-50:]) / 50
    final_left_h = sum(1 for h in history_h[-50:] if h.action == 'left') / 50
    print(f'  Final win-rate: {final_win_h:.3f}, '
          f'left-choice: {final_left_h:.3f}, '
          f'w(ctx→left)={history_h[-1].weight_context_left:.3f}, '
          f'w(ctx→right)={history_h[-1].weight_context_right:.3f}')

    print('\n  --- Policy B: targeted reward-modulated update ---')
    brain, ids = build_agent_brain()
    bandit = Bandit(p_left=0.8, p_right=0.2)
    n_trials = 200
    history = run_trials(
        brain, ids, bandit,
        n_trials=n_trials,
        eta=0.15,
        initial_temperature=2.0,
        final_temperature=0.1,
        update_policy='targeted',
    )

    # Show learning curve in chunks
    win_rates = windowed_winrate(history, window=25)
    left_rates = left_choice_rate(history, window=25)

    print('\n  Trial-block summary (rolling 25-trial windows):')
    print(f"  {'block':<10}  {'win-rate':<10}  {'left-rate':<10}  {'w(ctx→left)':<12}  {'w(ctx→right)'}")
    for chunk in (24, 49, 99, 149, 199):
        if chunk >= n_trials:
            continue
        h = history[chunk]
        print(f"  trial {chunk:>3}  "
              f"{win_rates[chunk]:>8.3f}    "
              f"{left_rates[chunk]:>8.3f}    "
              f"{h.weight_context_left:>10.3f}    "
              f"{h.weight_context_right:>10.3f}")

    # ASCII sparkline of win-rate
    print('\n  Win-rate over trials (rolling 25, ascii):')
    _print_sparkline(win_rates, label='   win  ', width=60)
    _print_sparkline(left_rates, label='   left ', width=60)

    # Final assessment
    final_win = sum(h.reward for h in history[-50:]) / 50
    final_left = sum(1 for h in history[-50:] if h.action == 'left') / 50
    initial_win = sum(h.reward for h in history[:50]) / 50

    print('\n  ─── FINAL ASSESSMENT ─────────────────────────────────────────')
    print(f'  initial 50-trial win-rate:  {initial_win:.3f}  (chance = 0.5)')
    print(f'  final 50-trial win-rate:    {final_win:.3f}  (target ≈ 0.8)')
    print(f'  final left-choice rate:     {final_left:.3f}  (target high)')
    print(f'  weight ctx→left:   {history[-1].weight_context_left:.3f}')
    print(f'  weight ctx→right:  {history[-1].weight_context_right:.3f}')

    delta = final_win - initial_win
    print(f'  win-rate improvement:       {delta:+.3f}')

    # Substrate learning is real if BOTH:
    #   - left-choice rate > 0.7 (substrate clearly prefers correct action)
    #   - win-rate > 0.65 (above chance by margin)
    if final_left > 0.7 and final_win > 0.65:
        print('\n  PASS — substrate acquired behavior from reward signals.')
        print(f'         left preference {final_left:.2f}, win-rate {final_win:.2f} '
              f'(theoretical max ≈ {bandit.p_left:.2f}).')
    else:
        print('\n  FAIL — substrate did not measurably learn.')


def _print_sparkline(values: List[float], label: str, width: int = 60) -> None:
    n = len(values)
    if n == 0:
        print(f'  {label}|')
        return
    step = max(1, n // width)
    samples = [values[i] for i in range(0, n, step)][:width]
    chars = ' ▁▂▃▄▅▆▇█'
    line = ''.join(chars[min(8, int(v * 8))] for v in samples)
    print(f'  {label}|{line}|  (0.0 → 1.0)')


if __name__ == '__main__':
    main()
