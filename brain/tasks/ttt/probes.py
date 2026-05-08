"""Empirical probes — try to break the substrate, not confirm it.

Each probe asks a sharp question: does X actually work, measurably?
Output is a table of (probe, expected, observed, pass/fail).
No hand-waving — print the numbers.
"""

from __future__ import annotations

from . import seed_brain, spread, hebbian_update, overlap_similarity


def probe_transitive_activation():
    """Cat→feline (1 hop) and cat→mammal (2 hops) are hand-coded.
    Cat→vertebrate (3 hops via is_a chain) and cat→animal (4 hops) are NOT
    direct edges in the seed. If the substrate is doing real chained
    activation, those should still light up.
    """
    b = seed_brain()
    s = spread(b, [b.lookup('cat')])

    direct_edges = ('feline',)        # 1 hop, hand-coded
    chain_2 = ('mammal',)              # 2 hops via feline
    chain_3 = ('vertebrate',)          # 3 hops via mammal — NOT a direct edge
    chain_4 = ('animal',)              # 4 hops — NOT a direct edge
    chain_5 = ('living',)              # 5 hops — NOT a direct edge
    distant_unrelated = ('gravity', 'einstein', 'force')

    rows = []
    for label, lemmas in [
        ('1-hop direct',  direct_edges),
        ('2-hop chain',   chain_2),
        ('3-hop chain (emergent)', chain_3),
        ('4-hop chain (emergent)', chain_4),
        ('5-hop chain (emergent)', chain_5),
        ('distant unrelated',     distant_unrelated),
    ]:
        for lemma in lemmas:
            nid = b.lookup(lemma)
            level = s.activation.get(nid, 0.0)
            rows.append((label, lemma, level))

    return rows


def probe_hebbian_actually_helps():
    """Does Hebbian update measurably increase recall of co-active concepts?

    Setup: probe activation of `play` when seeded with `cat` BEFORE and
    AFTER teaching the joint pattern {cat, play}. Without good learning,
    the after value won't be higher than the before value.
    """
    b = seed_brain()
    cat_id = b.lookup('cat')
    play_id = b.lookup('play')

    before = spread(b, [cat_id])
    before_play = before.activation.get(play_id, 0.0)

    # Teach
    joint = spread(b, [cat_id, play_id])
    update = hebbian_update(b, joint, eta=0.2, reward=1.0,
                            co_threshold=0.05, create_threshold=0.05)

    after = spread(b, [cat_id])
    after_play = after.activation.get(play_id, 0.0)

    return {
        'before_activation_of_play_when_cat_seeded': before_play,
        'after_activation_of_play_when_cat_seeded': after_play,
        'delta': after_play - before_play,
        'hebbian_updated': update['updated'],
        'hebbian_created': update['created'],
        'pass': after_play > before_play,
    }


def probe_emergent_similarity():
    """Pairs not connected by a direct edge but semantically related.

    Cat and bird share vertebrate hypernym (3 hops apart through cat→feline
    →mammal→vertebrate←bird). They should be similar, but less so than
    cat and dog (which share more: mammal, fur, etc.).
    No direct cat↔bird edge in the seed.
    """
    b = seed_brain()

    pairs = [
        ('cat', 'cat',    'identical (sanity)'),
        ('cat', 'dog',    'both mammals, share much'),
        ('cat', 'feline', 'direct hypernym'),
        ('cat', 'bird',   'both vertebrates only — emergent'),
        ('cat', 'fish',   'both vertebrates only — emergent'),
        ('warm', 'cold',  'antonyms (should be LOW)'),
        ('cat', 'gravity','unrelated domains'),
        ('einstein', 'physics', 'related domain'),
        ('einstein', 'cat', 'unrelated, different ontology'),
    ]
    rows = []
    for a, b_l, note in pairs:
        sa = spread(b, [b.lookup(a)])
        sb = spread(b, [b.lookup(b_l)])
        sim = overlap_similarity(sa, sb)
        rows.append((a, b_l, sim, note))
    return rows


def probe_recall_completion():
    """Hopfield-style partial recall: given parts of cat's signature
    (fur, paw, whisker), does cat dominate the resulting activation?

    Cat is the only neuron that has all three of fur, paw, whisker as
    has_part edges. So spreading from {fur, paw, whisker} should
    converge with cat highly activated via the inverse part_of links.
    """
    b = seed_brain()
    seeds = [b.lookup('fur'), b.lookup('paw'), b.lookup('whisker')]
    s = spread(b, seeds)

    cat_id = b.lookup('cat')
    cat_lvl = s.activation.get(cat_id, 0.0)
    # Compare to dog (which has fur+paw but no whisker — should be lower)
    dog_id = b.lookup('dog')
    dog_lvl = s.activation.get(dog_id, 0.0)

    return {
        'seeds': ['fur', 'paw', 'whisker'],
        'cat_activation': cat_lvl,
        'dog_activation': dog_lvl,
        'pass': cat_lvl > dog_lvl,
    }


def probe_antonym_inhibition():
    """warm has antonym→cold (relation weight = -0.50 inhibitory).
    Spread from warm should NOT highly activate cold. If antonym
    isn't actually inhibiting, both warm and cold will spread together.
    """
    b = seed_brain()
    s_warm = spread(b, [b.lookup('warm')])
    cold_lvl = s_warm.activation.get(b.lookup('cold'), 0.0)
    warm_lvl = s_warm.activation.get(b.lookup('warm'), 0.0)
    return {
        'warm_activation': warm_lvl,
        'cold_activation': cold_lvl,
        'ratio_cold_to_warm': cold_lvl / warm_lvl if warm_lvl > 0 else 0,
        'pass': cold_lvl < warm_lvl * 0.3,  # cold should be < 30% of warm
    }


def main():
    print('=' * 70)
    print('  EMPIRICAL PROBES — what does the substrate actually do?')
    print('=' * 70)

    print('\n[Probe 1] Transitive activation through the is_a chain:')
    print('  (3-hop and beyond are NOT direct edges; if they activate,')
    print('   the substrate is doing real chained inference)\n')
    rows = probe_transitive_activation()
    print(f"  {'category':<25}  {'lemma':<14}  {'activation'}")
    for cat, lemma, level in rows:
        marker = '✓' if level > 0.05 else '·'
        print(f"  {cat:<25}  {lemma:<14}  {level:6.3f}  {marker}")

    print('\n[Probe 2] Hebbian update measurably improves recall:')
    r = probe_hebbian_actually_helps()
    print(f"  before activation(play | cat): {r['before_activation_of_play_when_cat_seeded']:.4f}")
    print(f"  after  activation(play | cat): {r['after_activation_of_play_when_cat_seeded']:.4f}")
    print(f"  delta: {r['delta']:+.4f}")
    print(f"  Hebbian: {r['hebbian_updated']} strengthened, {r['hebbian_created']} created")
    print(f"  pass: {r['pass']}")

    print('\n[Probe 3] Emergent similarity (concepts not directly connected):')
    rows = probe_emergent_similarity()
    print(f"  {'a':<10}  {'b':<10}  {'sim':<7}  note")
    for a, b_l, sim, note in rows:
        print(f"  {a:<10}  {b_l:<10}  {sim:5.3f}    {note}")

    print('\n[Probe 4] Recall completion (Hopfield-style: parts → whole):')
    r = probe_recall_completion()
    print(f"  seeds: {r['seeds']}")
    print(f"  cat activation: {r['cat_activation']:.3f}")
    print(f"  dog activation: {r['dog_activation']:.3f}")
    print(f"  pass: cat > dog? {r['pass']}")

    print('\n[Probe 5] Antonym inhibition:')
    r = probe_antonym_inhibition()
    print(f"  warm activation: {r['warm_activation']:.3f}")
    print(f"  cold activation: {r['cold_activation']:.3f}")
    print(f"  ratio: {r['ratio_cold_to_warm']:.3f}")
    print(f"  pass: cold < 30% of warm? {r['pass']}")

    print('\n' + '=' * 70)
    print('  REPORT — pass means the architectural claim holds empirically.')
    print('  fail means we built the thing but it does not do the thing.')
    print('=' * 70)


if __name__ == '__main__':
    main()
