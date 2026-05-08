"""End-to-end demo: load mini-brain, spread, teach, re-spread.

Run:
    python3 -m brain.demo
"""

from __future__ import annotations

from brain import seed_brain, spread, hebbian_update, overlap_similarity


def show(brain, state, label: str, k: int = 8) -> None:
    print(f'\n  {label}  (steps={state.steps_run}, converged={state.converged})')
    for name, lvl, nid in state.top_k(brain, k=k):
        bar = '█' * int(lvl * 30)
        print(f'    {name:14}  {lvl:5.3f}  {bar}')


def main():
    print('=' * 70)
    print('  CONCEPT-AS-NEURON SUBSTRATE — Phase A demo')
    print('=' * 70)

    brain = seed_brain()
    print(f'\nSeeded: {brain.size} neurons, '
          f'{getattr(brain, "_used_synapses", 0)} synapses')

    # ── Demo 1: feeding "cat" lights up animal-related neurons ─────────
    cat_id = brain.lookup('cat')
    s_cat = spread(brain, [cat_id])
    show(brain, s_cat, 'spread from "cat":')

    # ── Demo 2: feeding "dog" lights up overlapping but distinct set ──
    dog_id = brain.lookup('dog')
    s_dog = spread(brain, [dog_id])
    show(brain, s_dog, 'spread from "dog":')

    sim = overlap_similarity(s_cat, s_dog)
    print(f'\n  similarity(cat, dog) = {sim:.3f}  '
          f'(both mammals, much shared activation)')

    # ── Demo 3: feeding two seeds simulates a sentence ────────────────
    eat_id = brain.lookup('eat')
    food_id = brain.lookup('food')
    s_eat = spread(brain, [eat_id, food_id])
    show(brain, s_eat, 'spread from {"eat", "food"}:')

    # ── Demo 4: similarity of unrelated concepts is low ───────────────
    s_grav = spread(brain, [brain.lookup('gravity')])
    sim_unrelated = overlap_similarity(s_cat, s_grav)
    print(f'\n  similarity(cat, gravity) = {sim_unrelated:.3f}  '
          f'(different domains, little shared activation)')

    # ── Demo 5: TEACH — Hebbian creates new synapses ──────────────────
    print('\n' + '─' * 70)
    print('  TEACHING: feeding "cat play" jointly to strengthen the link')
    print('─' * 70)

    play_id = brain.lookup('play')
    before_state = spread(brain, [cat_id])
    before_play = before_state.activation.get(play_id, 0.0)
    print(f'\n  before teaching: activation(play) when seed="cat" = {before_play:.3f}')

    # Teach by jointly activating cat + play with reward
    joint = spread(brain, [cat_id, play_id])
    update_stats = hebbian_update(brain, joint, eta=0.1, reward=1.0,
                                   co_threshold=0.15, create_threshold=0.3)
    print(f'  Hebbian update: {update_stats["updated"]} synapses strengthened, '
          f'{update_stats["created"]} new synapses created')

    after_state = spread(brain, [cat_id])
    after_play = after_state.activation.get(play_id, 0.0)
    print(f'  after teaching:  activation(play) when seed="cat" = {after_play:.3f}')
    delta = after_play - before_play
    print(f'  delta: {delta:+.3f}  '
          f'({"learned" if delta > 0.01 else "no measurable change"})')

    # ── Demo 6: persistence round-trip ────────────────────────────────
    print('\n' + '─' * 70)
    print('  PERSISTENCE: save → load → re-spread, expect same answers')
    print('─' * 70)

    import tempfile, os
    with tempfile.TemporaryDirectory(prefix='brain_') as tmp:
        brain.save(tmp)
        from brain.store import Brain
        reloaded = Brain.load(tmp)
        s_cat2 = spread(reloaded, [reloaded.lookup('cat')])
        sim_round_trip = overlap_similarity(s_cat, s_cat2)
        print(f'\n  similarity(spread before save, spread after load) = '
              f'{sim_round_trip:.3f}')
        # Wait — we mutated the brain via teach(), so this round-trip is
        # post-teach vs pre-teach. Use the after_state instead.
        sim_post = overlap_similarity(after_state, s_cat2)
        print(f'  similarity(post-teach in-RAM, post-teach reloaded) = '
              f'{sim_post:.3f}  (should be 1.000 if persistence is exact)')

    print('\n' + '=' * 70)
    print('  Phase A demo complete')
    print('=' * 70)


if __name__ == '__main__':
    main()
