"""End-to-end demo: all Phase A++ primitives in use, fully traceable.

Walks through:
1. Working memory carrying context across calls
2. Goal injection biasing spreading top-down
3. Modulator scaling learning rate by recent reward
4. Replay buffer + consolidate for offline re-learning
5. Trace log of every event, dumped to disk for inspection
"""

from __future__ import annotations

import os
import tempfile

from brain import (
    seed_brain, spread, hebbian_update,
    WorkingMemory, ReplayBuffer, consolidate, Episode,
)


def show(label: str, brain, state, k: int = 6) -> None:
    print(f'\n  {label}')
    for name, lvl, _ in state.top_k(brain, k=k):
        print(f'    {name:14}  {lvl:6.3f}')


def main():
    print('=' * 70)
    print('  Phase A++ INTEGRATION DEMO')
    print('  Working memory · goal neurons · modulator · replay · trace')
    print('=' * 70)

    brain = seed_brain()
    print(f'\n  Substrate: {brain.size} neurons, '
          f'{getattr(brain, "_used_synapses", 0)} synapses')

    # ─── 1. Working memory ───────────────────────────────────────────
    print('\n  [1] Working memory — context carries across two calls')
    print('  ' + '─' * 60)
    wm = WorkingMemory(decay=0.6)

    s_cat = spread(brain, [brain.lookup('cat')], working_memory=wm)
    show('First spread (cat) populates WM:', brain, s_cat, k=5)
    print(f'    WM holds {len(wm.activation)} neurons after first call')

    # Second call: NO seeds, but WM has carry-over
    s_followup = spread(brain, [], working_memory=wm)
    show('Second spread (empty seeds, WM carry-over):', brain, s_followup, k=5)
    print(f'    {len(s_followup.activation)} active from WM alone — '
          'context persistence demonstrated')

    # ─── 2. Goal neurons ─────────────────────────────────────────────
    print('\n  [2] Goal injection — top-down bias toward chosen concept')
    print('  ' + '─' * 60)
    physics_id = brain.lookup('physics')
    s_with_goal = spread(brain, [brain.lookup('cat')],
                          goals=[physics_id], goal_strength=1.0)
    print(f'    seed=cat, goal=physics  →  '
          f'physics activation: {s_with_goal.activation.get(physics_id):.3f}')
    s_without_goal = spread(brain, [brain.lookup('cat')])
    print(f'    seed=cat, no goal      →  '
          f'physics activation: {s_without_goal.activation.get(physics_id, 0):.3f}')

    # ─── 3. Modulator ────────────────────────────────────────────────
    print('\n  [3] Modulator — global plasticity scaling by recent reward')
    print('  ' + '─' * 60)
    base_eta = 0.1
    print(f'    base eta = {base_eta}')
    print(f'    initial modulator = {brain.modulator.value:.3f}, '
          f'effective eta = {brain.modulator.effective_eta(base_eta):.3f}')
    brain.modulator.adjust(reward_signal=1.0, scale=0.4)
    print(f'    after good reward: modulator = {brain.modulator.value:.3f}, '
          f'effective eta = {brain.modulator.effective_eta(base_eta):.3f}')
    brain.modulator.adjust(reward_signal=-1.0, scale=0.4)
    brain.modulator.adjust(reward_signal=-1.0, scale=0.4)
    print(f'    after bad rewards: modulator = {brain.modulator.value:.3f}, '
          f'effective eta = {brain.modulator.effective_eta(base_eta):.3f}')

    # ─── 4. Replay buffer ────────────────────────────────────────────
    print('\n  [4] Replay buffer — record episodes, sample for re-learning')
    print('  ' + '─' * 60)
    buf = ReplayBuffer(capacity=50)
    for i in range(20):
        # Fake episodes
        buf.record(trajectory=[(f'state_{i}', i % 9)], reward=0.7 if i % 3 else 0.2)
    print(f'    Buffer stats: {buf.stats()}')

    consolidations = []
    consolidate(buf, n_samples=5,
                 credit_fn=lambda ep: consolidations.append(ep.reward))
    print(f'    Consolidated 5 sampled episodes (rewards: '
          f'{[round(r, 2) for r in consolidations]})')

    # ─── 5. Trace log ────────────────────────────────────────────────
    print('\n  [5] Trace log — every substrate event recorded')
    print('  ' + '─' * 60)
    print(f'    Total events logged this session: {len(brain.trace.entries)}')
    print(f'    Event counts: {brain.trace.summary()}')
    print(f'    Last 3 events:')
    for e in brain.trace.tail(3):
        kept = {k: v for k, v in e.items() if k not in ('t',)}
        print(f'      {kept}')

    # Dump to disk
    with tempfile.TemporaryDirectory(prefix='brain_trace_') as tmp:
        path = os.path.join(tmp, 'trace.jsonl')
        brain.trace.dump(path)
        size = os.path.getsize(path)
        with open(path) as f:
            n_lines = sum(1 for _ in f)
        print(f'    Dumped to disk: {n_lines} lines, {size} bytes — '
              f'every event inspectable')

    # ─── Summary ─────────────────────────────────────────────────────
    print('\n' + '=' * 70)
    print('  All five primitives wired in. Substrate is more brain-shaped.')
    print('=' * 70)
    print('\n  What this enables next:')
    print('   - Multi-turn reasoning (working memory)')
    print('   - Goal-directed behavior (goal neurons + WM)')
    print('   - Sample-efficient RL (replay + modulator)')
    print('   - Inspectability for any substrate behavior (trace)')
    print()


if __name__ == '__main__':
    main()
