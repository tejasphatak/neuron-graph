"""Phase A seed — a hand-curated 50-100 neuron mini-brain to validate
the substrate end-to-end.

Goal: small enough to fit in one screen of inspection, rich enough that
spreading produces visibly-correct patterns. Animals + actions + a few
abstract concepts so we can demo cross-domain spread.

Phase B replaces this with a WordNet importer.
"""

from __future__ import annotations

from typing import List, Tuple

from brain.store import Brain
from brain.neuron import NeuronType


# (lemma, type, optional definition)
SEED_NEURONS = [
    # Animals (concrete entities)
    ('cat',         NeuronType.TEXT, 'a small domesticated feline'),
    ('dog',         NeuronType.TEXT, 'a domesticated canine'),
    ('feline',      NeuronType.TEXT, 'a member of the cat family'),
    ('canine',      NeuronType.TEXT, 'a member of the dog family'),
    ('mammal',      NeuronType.TEXT, 'a warm-blooded vertebrate'),
    ('animal',      NeuronType.TEXT, 'a living organism that is not a plant'),
    ('vertebrate',  NeuronType.TEXT, 'an animal with a backbone'),
    ('pet',         NeuronType.TEXT, 'a domesticated animal kept for companionship'),
    ('bird',        NeuronType.TEXT, 'a feathered vertebrate'),
    ('fish',        NeuronType.TEXT, 'an aquatic vertebrate'),

    # Body parts / attributes
    ('tail',        NeuronType.TEXT, 'an appendage at the rear'),
    ('paw',         NeuronType.TEXT, 'an animal foot with claws'),
    ('whisker',     NeuronType.TEXT, 'a sensory hair'),
    ('fur',         NeuronType.TEXT, 'thick body hair on a mammal'),
    ('feather',     NeuronType.TEXT, 'a body covering on a bird'),

    # Actions (verbs)
    ('see',         NeuronType.TEXT, 'to perceive with the eyes'),
    ('hear',        NeuronType.TEXT, 'to perceive with the ears'),
    ('eat',         NeuronType.TEXT, 'to consume food'),
    ('drink',       NeuronType.TEXT, 'to consume liquid'),
    ('run',         NeuronType.TEXT, 'to move fast on legs'),
    ('walk',        NeuronType.TEXT, 'to move on legs at a normal pace'),
    ('sleep',       NeuronType.TEXT, 'to rest unconsciously'),
    ('play',        NeuronType.TEXT, 'to engage in recreational activity'),
    ('hunt',        NeuronType.TEXT, 'to pursue and capture prey'),
    ('bark',        NeuronType.TEXT, 'a dog\'s vocalization'),
    ('meow',        NeuronType.TEXT, 'a cat\'s vocalization'),
    ('chase',       NeuronType.TEXT, 'to pursue'),

    # Food
    ('food',        NeuronType.TEXT, 'something eaten for nourishment'),
    ('water',       NeuronType.TEXT, 'a clear liquid essential for life'),
    ('fish_food',   NeuronType.TEXT, 'fish as food'),
    ('meat',        NeuronType.TEXT, 'animal flesh as food'),
    ('milk',        NeuronType.TEXT, 'a liquid produced by mammals'),

    # Abstract / general
    ('thing',       NeuronType.TEXT, 'an entity'),
    ('living',      NeuronType.TEXT, 'alive'),
    ('warm',        NeuronType.TEXT, 'having heat'),
    ('cold',        NeuronType.TEXT, 'lacking heat'),
    ('big',         NeuronType.TEXT, 'large in size'),
    ('large',       NeuronType.TEXT, 'big in size'),
    ('small',       NeuronType.TEXT, 'little in size'),
    ('happy',       NeuronType.TEXT, 'feeling joy'),
    ('glad',        NeuronType.TEXT, 'pleased'),
    ('quiet',       NeuronType.TEXT, 'making little noise'),
    ('loud',        NeuronType.TEXT, 'making much noise'),

    # Domain anchor for "what is gravity?" question hijacking demo
    ('gravity',     NeuronType.TEXT, 'the force of attraction toward earth'),
    ('einstein',    NeuronType.TEXT, 'physicist who developed relativity'),
    ('relativity',  NeuronType.TEXT, 'einstein\'s theory of space and time'),
    ('physics',     NeuronType.TEXT, 'the science of matter and energy'),
    ('force',       NeuronType.TEXT, 'an influence causing motion'),
    ('person',      NeuronType.TEXT, 'a human being'),
    ('scientist',   NeuronType.TEXT, 'a person doing science'),
]


# (from_lemma, relation, to_lemma, weight)
SEED_EDGES: List[Tuple[str, str, str, float]] = [
    # Synonyms — symmetric
    ('big',     'synonym',     'large',    1.0),
    ('large',   'synonym',     'big',      1.0),
    ('happy',   'synonym',     'glad',     1.0),
    ('glad',    'synonym',     'happy',    1.0),

    # is_a chain (hyponym → hypernym)
    ('cat',         'is_a',     'feline',     1.0),
    ('dog',         'is_a',     'canine',     1.0),
    ('feline',      'is_a',     'mammal',     1.0),
    ('canine',      'is_a',     'mammal',     1.0),
    ('mammal',      'is_a',     'vertebrate', 1.0),
    ('bird',        'is_a',     'vertebrate', 1.0),
    ('fish',        'is_a',     'vertebrate', 1.0),
    ('vertebrate',  'is_a',     'animal',     1.0),
    ('animal',      'is_a',     'living',     1.0),
    ('animal',      'is_a',     'thing',      0.6),
    ('cat',         'is_a',     'pet',        0.8),
    ('dog',         'is_a',     'pet',        0.8),
    ('einstein',    'is_a',     'scientist',  1.0),
    ('einstein',    'is_a',     'person',     0.9),
    ('scientist',   'is_a',     'person',     1.0),

    # Inverse is_a — weaker (we don't want every hypernym to recall every descendant)
    ('feline',      'inverse_is_a',  'cat',     0.8),
    ('canine',      'inverse_is_a',  'dog',     0.8),
    ('mammal',      'inverse_is_a',  'feline',  0.6),
    ('mammal',      'inverse_is_a',  'canine',  0.6),

    # has_part / part_of
    ('cat',         'has_part',     'tail',      1.0),
    ('cat',         'has_part',     'paw',       1.0),
    ('cat',         'has_part',     'whisker',   1.0),
    ('cat',         'has_part',     'fur',       1.0),
    ('dog',         'has_part',     'tail',      1.0),
    ('dog',         'has_part',     'paw',       1.0),
    ('dog',         'has_part',     'fur',       1.0),
    ('mammal',      'has_part',     'fur',       0.9),
    ('bird',        'has_part',     'feather',   1.0),
    ('tail',        'part_of',      'animal',    0.9),
    ('paw',         'part_of',      'animal',    0.9),

    # Action / disposition relations (related_to as the catch-all)
    ('cat',     'related_to',   'meow',     1.0),
    ('dog',     'related_to',   'bark',     1.0),
    ('cat',     'related_to',   'hunt',     0.8),
    ('cat',     'related_to',   'play',     0.7),
    ('dog',     'related_to',   'play',     0.8),
    ('dog',     'related_to',   'chase',    0.8),
    ('cat',     'related_to',   'chase',    0.6),
    ('cat',     'related_to',   'sleep',    0.7),
    ('dog',     'related_to',   'run',      0.8),
    ('cat',     'related_to',   'milk',     0.6),
    ('cat',     'related_to',   'fish_food',0.7),
    ('mammal',  'related_to',   'milk',     0.9),

    # Eating / drinking
    ('eat',     'related_to',   'food',     1.0),
    ('drink',   'related_to',   'water',    1.0),
    ('animal',  'related_to',   'eat',      0.9),
    ('animal',  'related_to',   'drink',    0.9),
    ('food',    'related_to',   'meat',     0.7),
    ('food',    'related_to',   'fish_food',0.7),
    ('food',    'related_to',   'milk',     0.6),

    # Physics domain
    ('gravity',     'is_a',         'force',        1.0),
    ('gravity',     'related_to',   'physics',      0.9),
    ('einstein',    'related_to',   'relativity',   1.0),
    ('einstein',    'related_to',   'physics',      0.9),
    ('relativity',  'related_to',   'physics',      1.0),
    ('relativity',  'related_to',   'gravity',      0.5),  # general relativity

    # Antonyms (inhibitory — weight is positive, relation has negative coefficient)
    ('warm',        'antonym',      'cold',         1.0),
    ('cold',        'antonym',      'warm',         1.0),
    ('big',         'antonym',      'small',        1.0),
    ('small',       'antonym',      'big',          1.0),
    ('quiet',       'antonym',      'loud',         1.0),
    ('loud',        'antonym',      'quiet',        1.0),
]


def seed_brain() -> Brain:
    """Build the Phase A mini-brain. Returns a Brain ready for spreading."""
    b = Brain()

    # Allocate neurons (record their assigned IDs)
    for lemma, ntype, definition in SEED_NEURONS:
        b.add_neuron(
            lemma=lemma, type=ntype,
            content=definition.encode('utf-8') if definition else None,
        )

    # Lay out edges: collect by from-id, then set in one pass per neuron
    edges_by_from: dict[int, list[tuple[int, str, float]]] = {}
    for from_lemma, rel, to_lemma, weight in SEED_EDGES:
        from_id = b.lookup(from_lemma)
        to_id = b.lookup(to_lemma)
        if from_id is None or to_id is None:
            raise KeyError(f'edge references unknown lemma: '
                           f'{from_lemma!r} or {to_lemma!r}')
        edges_by_from.setdefault(from_id, []).append((to_id, rel, weight))

    for from_id, edges in edges_by_from.items():
        b.set_synapses(from_id, edges)

    return b


if __name__ == '__main__':
    b = seed_brain()
    print(f'Seeded brain: {b.size} neurons, '
          f'{getattr(b, "_used_synapses", 0)} synapses, '
          f'{len(b.relations)} relation types')
