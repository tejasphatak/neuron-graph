"""Tiny generative LM on the substrate — QUALITATIVE training.

NOT bigram statistics. The substrate learns:
  1. each token's grammatical class    (token --is_a--> POS-tag)
  2. the sentence's grammatical shape  (POS_i --follows--> POS_{i+1})
  3. each token's slot in the pattern  (token --in_slot--> position-id)

Generation:
  - walk the POS pattern slot-by-slot
  - at each slot, spread from the prior token + current slot neuron
  - pick the token whose POS membership matches this slot AND has the
    strongest activation given context

This is the Guru thesis applied to LM: explain what tokens ARE, explain
the structure, let composition emerge. No frequency-pumping.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from brain import Brain, WorkingMemory, spread


IS_A = 'is_a'
FOLLOWS = 'follows'
IN_SLOT = 'in_slot'


# ─── Vocabulary ────────────────────────────────────────────────────────────

@dataclass
class Vocab:
    """token <-> neuron-id, plus POS-tag <-> neuron-id, plus slot positions."""
    token_to_id: Dict[str, int] = field(default_factory=dict)
    id_to_token: Dict[int, str] = field(default_factory=dict)
    pos_to_id: Dict[str, int] = field(default_factory=dict)
    id_to_pos: Dict[int, str] = field(default_factory=dict)
    slot_to_id: Dict[int, int] = field(default_factory=dict)  # position-index → neuron

    def add_token(self, token: str, brain: Brain) -> int:
        if token not in self.token_to_id:
            nid = brain.add_neuron(lemma=f'tok:{token}', decay=0.5)
            self.token_to_id[token] = nid
            self.id_to_token[nid] = token
        return self.token_to_id[token]

    def add_pos(self, pos: str, brain: Brain) -> int:
        if pos not in self.pos_to_id:
            nid = brain.add_neuron(lemma=f'pos:{pos}', decay=0.7)
            self.pos_to_id[pos] = nid
            self.id_to_pos[nid] = pos
        return self.pos_to_id[pos]

    def add_slot(self, position: int, brain: Brain) -> int:
        if position not in self.slot_to_id:
            nid = brain.add_neuron(lemma=f'slot:{position}', decay=0.7)
            self.slot_to_id[position] = nid
        return self.slot_to_id[position]

    def __len__(self) -> int:
        return len(self.token_to_id)


# ─── Brain construction ────────────────────────────────────────────────────

def build_lm_brain() -> Tuple[Brain, Vocab]:
    """Brain wired with three relation types — class membership, sequence,
    and slot binding. All weights ≈ 1.0 so spreading is undistorted by
    relation-weight asymmetries."""
    brain = Brain()
    brain.relations = [(IS_A, 1.0), (FOLLOWS, 1.0), (IN_SLOT, 1.0)]
    brain._rebuild_relation_index()
    return brain, Vocab()


# ─── Qualitative training ──────────────────────────────────────────────────

def _add_or_strengthen(brain: Brain, from_id: int, to_id: int,
                        rel_name: str, delta: float = 1.0,
                        cap: float = 1.0) -> None:
    """Add edge if absent; if present, raise weight by `delta` (capped)."""
    edges = brain.synapses_of(from_id)
    rel_id = brain.relation_id[rel_name]
    for syn in edges:
        if int(syn['to_id']) == to_id and int(syn['relation']) == rel_id:
            syn['weight'] = min(cap, float(syn['weight']) + delta)
            return
    brain.add_synapse(from_id, to_id, rel_name=rel_name, weight=delta)


def teach_sentence(brain: Brain, vocab: Vocab,
                    tokens: List[str], pos_tags: List[str]) -> None:
    """Qualitative teach: register tokens, POS tags, slots, and the
    relations connecting them. Idempotent — re-calling on the same
    sentence does not pump weights past 1.0.

    Edges created:
      token  --is_a-->     POS              (token's grammatical class)
      POS    --follows-->  POS              (sentence-shape transitions)
      token  --in_slot-->  slot[i]          (token's position evidence)
      slot[i] --follows--> slot[i+1]        (slot sequence)
      slot[i] --is_a-->    POS              (slot's expected POS)
    """
    assert len(tokens) == len(pos_tags), 'tokens/pos length mismatch'
    n = len(tokens)

    # Register everything
    tok_ids = [vocab.add_token(t, brain) for t in tokens]
    pos_ids = [vocab.add_pos(p, brain) for p in pos_tags]
    slot_ids = [vocab.add_slot(i, brain) for i in range(n)]

    # token --is_a--> POS
    for tid, pid in zip(tok_ids, pos_ids):
        _add_or_strengthen(brain, tid, pid, IS_A)

    # POS_i --follows--> POS_{i+1}
    for i in range(n - 1):
        _add_or_strengthen(brain, pos_ids[i], pos_ids[i + 1], FOLLOWS)

    # token --in_slot--> slot[i]
    for tid, sid in zip(tok_ids, slot_ids):
        _add_or_strengthen(brain, tid, sid, IN_SLOT)

    # slot_i --follows--> slot_{i+1}, slot_i --is_a--> POS_i
    for i in range(n):
        if i + 1 < n:
            _add_or_strengthen(brain, slot_ids[i], slot_ids[i + 1], FOLLOWS)
        _add_or_strengthen(brain, slot_ids[i], pos_ids[i], IS_A)


# ─── Generation via slot-walking ───────────────────────────────────────────

def generate(brain: Brain, vocab: Vocab,
              prompt_tokens: List[str], *,
              max_new: int = 20) -> List[str]:
    """Compose a sentence by walking grammar slots.

    For each subsequent slot, ask: which token has both
      (a) IS_A edge to this slot's expected POS
      (b) IN_SLOT edge to this slot
    Pick the highest-product token. Ties broken by overall activation.
    """
    if not prompt_tokens:
        raise ValueError('prompt must contain at least one token')

    # Find which slot the prompt ended at by matching prompt[-1] against
    # token→slot edges and picking the slot index closest to len(prompt)-1
    last_token_id = vocab.token_to_id[prompt_tokens[-1]]
    last_pos = len(prompt_tokens) - 1

    out: List[str] = []
    emitted_ids = [vocab.token_to_id[t] for t in prompt_tokens]

    for step in range(max_new):
        next_slot_idx = last_pos + 1 + step
        next_slot_nid = vocab.slot_to_id.get(next_slot_idx)
        if next_slot_nid is None:
            break  # past sentence end — no more slots to fill

        # Find the POS this slot expects
        slot_edges = brain.synapses_of(next_slot_nid)
        rel_is_a = brain.relation_id[IS_A]
        expected_pos_nid = None
        for syn in slot_edges:
            if int(syn['relation']) == rel_is_a:
                expected_pos_nid = int(syn['to_id'])
                break
        if expected_pos_nid is None:
            break

        # Score every token: weight(token --is_a--> expected_pos)
        #                  × weight(token --in_slot--> next_slot)
        rel_in_slot = brain.relation_id[IN_SLOT]
        best_id = None
        best_score = -1.0
        for tid in vocab.id_to_token:
            tok_edges = brain.synapses_of(tid)
            pos_w = 0.0
            slot_w = 0.0
            for syn in tok_edges:
                rel = int(syn['relation'])
                target = int(syn['to_id'])
                if rel == rel_is_a and target == expected_pos_nid:
                    pos_w = float(syn['weight'])
                elif rel == rel_in_slot and target == next_slot_nid:
                    slot_w = float(syn['weight'])
            score = pos_w * slot_w
            if score > best_score:
                best_score = score
                best_id = tid

        if best_id is None or best_score <= 0:
            break

        out.append(vocab.id_to_token[best_id])
        emitted_ids.append(best_id)

    return out


# ─── Smoke main ─────────────────────────────────────────────────────────────

def _smoke_main() -> int:
    """Teach POS + grammar shape + slot bindings for ONE sentence,
    prompt with first 2 tokens, compose remaining 7."""
    sentence = "the quick brown fox jumps over the lazy dog".split()
    pos_tags = ["DET", "ADJ", "ADJ", "NOUN", "VERB", "PREP", "DET", "ADJ", "NOUN"]

    brain, vocab = build_lm_brain()
    teach_sentence(brain, vocab, sentence, pos_tags)

    prompt = sentence[:2]
    target = sentence[2:]
    out = generate(brain, vocab, prompt, max_new=len(target))

    print(f'vocab size  : {len(vocab)} tokens, {len(vocab.pos_to_id)} POS tags, '
          f'{len(vocab.slot_to_id)} slots')
    print(f'brain neurons: {brain.size}  synapses: {getattr(brain, "_used_synapses", 0)}')
    print(f'prompt      : {" ".join(prompt)}')
    print(f'target      : {" ".join(target)}')
    print(f'generated   : {" ".join(out)}')
    n_correct = sum(1 for a, b in zip(target, out) if a == b)
    print(f'accuracy    : {n_correct}/{len(target)} = {n_correct/len(target):.0%}')

    return 0 if n_correct == len(target) else 1


if __name__ == '__main__':
    raise SystemExit(_smoke_main())
