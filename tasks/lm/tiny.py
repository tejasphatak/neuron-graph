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
HAS_MEMBER = 'has_member'   # inverse of IS_A: POS → token
HAS_FILLER = 'has_filler'   # inverse of IN_SLOT: slot → token
CO_OCCURS = 'co_occurs'     # token ↔ token within the same sentence


# ─── Vocabulary ────────────────────────────────────────────────────────────

@dataclass
class Vocab:
    """token <-> neuron-id, plus POS-tag <-> neuron-id, plus slot positions,
    plus step-position neurons (substrate's positional embeddings),
    plus sentence-id neurons (per-sentence context anchors)."""
    token_to_id: Dict[str, int] = field(default_factory=dict)
    id_to_token: Dict[int, str] = field(default_factory=dict)
    pos_to_id: Dict[str, int] = field(default_factory=dict)
    id_to_pos: Dict[int, str] = field(default_factory=dict)
    slot_to_id: Dict[int, int] = field(default_factory=dict)  # position-index → neuron
    step_to_id: Dict[int, int] = field(default_factory=dict)  # step-relative-to-prompt → neuron
    id_to_step: Dict[int, int] = field(default_factory=dict)
    sentence_to_id: Dict[int, int] = field(default_factory=dict)  # sentence-index → neuron
    id_to_sentence: Dict[int, int] = field(default_factory=dict)

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

    def add_sentence(self, idx: int, brain: Brain) -> int:
        """Sentence-id neuron — context anchor for one training pair.

        Each training sentence gets a unique neuron. Seeded into WM during
        both training and inference so RL grows
          sentence_i --co_occurs--> token  edges
        that route SENTENCE-conditionally — breaks the same-position-same-POS
        ambiguity at scale (where 'fox' and 'cat' both compete at NOUN slot 3
        from prompts that look similar).

        At inference, the sentence neuron is identified from the prompt
        (which sentence does this prompt come from?). For shared-prompt
        cases, a sentence "search" via spread can pick the most-likely
        sentence-id from the prompt context.
        """
        if idx not in self.sentence_to_id:
            nid = brain.add_neuron(lemma=f'sent:{idx}', decay=0.7)
            self.sentence_to_id[idx] = nid
            self.id_to_sentence[nid] = idx
        return self.sentence_to_id[idx]

    def add_step(self, position: int, brain: Brain) -> int:
        """Position-step neuron — substrate's positional embedding.
        Counts emissions relative to end of prompt (step_0 is the first
        token to be emitted after the prompt). RL learns
          step_i --co_occurs--> token  edges that route position-conditionally.
        """
        if position not in self.step_to_id:
            nid = brain.add_neuron(lemma=f'step:{position}', decay=0.6)
            self.step_to_id[position] = nid
            self.id_to_step[nid] = position
        return self.step_to_id[position]

    def __len__(self) -> int:
        return len(self.token_to_id)


# ─── Brain construction ────────────────────────────────────────────────────

def build_lm_brain() -> Tuple[Brain, Vocab]:
    """Brain wired with five relation types:
      is_a / has_member  — token ↔ POS (forward strong, inverse weaker)
      in_slot / has_filler — token ↔ slot (forward strong, inverse strong)
      follows            — sequence transitions

    Asymmetric inverse weights matter: a POS has many members so
    spreading from POS shouldn't swamp the network — has_member is
    weaker than has_filler.
    """
    brain = Brain()
    brain.relations = [
        (IS_A,       1.0),
        (FOLLOWS,    0.8),
        (IN_SLOT,    1.0),
        (HAS_MEMBER, 0.3),
        (HAS_FILLER, 0.7),
        (CO_OCCURS,  0.4),   # context disambiguator across multiple sentences
    ]
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

    # token <--is_a/has_member--> POS
    for tid, pid in zip(tok_ids, pos_ids):
        _add_or_strengthen(brain, tid, pid, IS_A)
        _add_or_strengthen(brain, pid, tid, HAS_MEMBER)

    # POS_i --follows--> POS_{i+1}
    for i in range(n - 1):
        _add_or_strengthen(brain, pos_ids[i], pos_ids[i + 1], FOLLOWS)

    # token <--in_slot/has_filler--> slot[i]
    for tid, sid in zip(tok_ids, slot_ids):
        _add_or_strengthen(brain, tid, sid, IN_SLOT)
        _add_or_strengthen(brain, sid, tid, HAS_FILLER)

    # slot_i --follows--> slot_{i+1}, slot_i --is_a--> POS_i
    for i in range(n):
        if i + 1 < n:
            _add_or_strengthen(brain, slot_ids[i], slot_ids[i + 1], FOLLOWS)
        _add_or_strengthen(brain, slot_ids[i], pos_ids[i], IS_A)

    # Co-occurrence: every distinct token-pair in this sentence gets a
    # (distance-decayed) bidirectional co_occurs edge. This is what
    # lets the substrate disambiguate across sentences sharing grammar:
    # "brown" co-occurs with "fox" (S1) but not "cat" (S2), so a prompt
    # containing "brown" pulls "fox" via co_occurs even though both
    # tokens have equally-strong has_filler edges from slot[3].
    for i in range(n):
        for j in range(n):
            if i == j or tok_ids[i] == tok_ids[j]:
                continue
            distance = abs(i - j)
            delta = 1.0 / distance
            _add_or_strengthen(brain, tok_ids[i], tok_ids[j],
                                CO_OCCURS, delta=delta)


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


# ─── Substrate-native generation (via spread) ──────────────────────────────

def generate_via_spread(brain: Brain, vocab: Vocab,
                         prompt_tokens: List[str], *,
                         max_new: int = 20,
                         goal_strength: float = 2.0,
                         max_steps: int = 2,
                         wm_decay: float = 0.6) -> List[str]:
    """Substrate-native generation: at each step, inject the next slot
    as a goal, run spread() over the WM-seeded prompt context, read out
    the highest-activated token-neuron.

    The substrate's spreading does the work — slot/POS/token co-activation
    via has_filler and has_member edges naturally surfaces the right token.
    No direct edge lookup; pure activation pattern read-out.

    Position is tracked externally because the substrate has no native
    "what slot am I at" — that's a property of the *generation loop*, not
    of the substrate itself. The substrate provides the activation field;
    the loop chooses where to point it.
    """
    if not prompt_tokens:
        raise ValueError('prompt must contain at least one token')

    emitted_ids: List[int] = []
    for t in prompt_tokens:
        nid = vocab.token_to_id.get(t)
        if nid is None:
            raise KeyError(f'Token not in vocab: {t!r}')
        emitted_ids.append(nid)

    out: List[str] = []
    current_slot_idx = len(emitted_ids) - 1  # last filled slot

    rel_is_a = brain.relation_id[IS_A]

    for _ in range(max_new):
        next_slot_idx = current_slot_idx + 1
        next_slot_nid = vocab.slot_to_id.get(next_slot_idx)
        if next_slot_nid is None:
            break

        # The slot's expected POS is itself a substrate fact — slot --is_a--> POS.
        # Read it out so the candidate filter respects structural truth.
        expected_pos_nid = None
        for syn in brain.synapses_of(next_slot_nid):
            if int(syn['relation']) == rel_is_a:
                expected_pos_nid = int(syn['to_id'])
                break

        # Working memory: last few emitted tokens with positional decay.
        # Most recent has strength 1.0; older entries decay back.
        # Note: we deliberately do NOT seed the current slot — the
        # substrate is told only "where to go" (goal=next_slot), not
        # "where you are." Otherwise current_slot's has_filler back-edge
        # to the prompt token creates a loop that re-elects the prompt.
        wm = WorkingMemory(decay=wm_decay, max_size=64, floor=0.05)
        seeds = {nid: wm_decay ** offset
                  for offset, nid in enumerate(reversed(emitted_ids))}
        wm.absorb(seeds, gain=1.0)

        state = spread(brain, seeds=[],
                        working_memory=wm,
                        goals=[next_slot_nid],
                        goal_strength=goal_strength,
                        max_steps=max_steps,
                        sparsity=1.0)

        # Read-out: highest-activated TOKEN whose IS_A edge points to the
        # slot's expected POS. The POS filter respects the substrate's
        # own structural fact (slot --is_a--> POS); without it, raw
        # activation noise from prompt-token carry-over can elect a
        # wrong-POS token. has_filler from goal-clamped slot still
        # drives the *correct* candidate's activation above its peers.
        best_id = None
        best_score = -1.0
        for nid, lvl in state.activation.items():
            if nid not in vocab.id_to_token:
                continue
            if expected_pos_nid is not None:
                token_pos_match = any(
                    int(syn['to_id']) == expected_pos_nid
                    and int(syn['relation']) == rel_is_a
                    for syn in brain.synapses_of(nid)
                )
                if not token_pos_match:
                    continue
            if lvl > best_score:
                best_score = lvl
                best_id = nid

        if best_id is None or best_score <= 0:
            break

        out.append(vocab.id_to_token[best_id])
        emitted_ids.append(best_id)
        current_slot_idx = next_slot_idx

    return out


# ─── Open-vocabulary generation (no slots) ─────────────────────────────────

def generate_open_vocab(brain, vocab, prompt_tokens, *,
                         max_new: int = 20,
                         wm_decay: float = 0.6,
                         goal_strength: float = 2.0,
                         max_steps: int = 2,
                         antiloop_window: int = 2) -> List[str]:
    """Open-vocabulary generation — NO slot anchoring. No fixed positions.

    The substrate's POS-FOLLOWS edges (POS_i --follows--> POS_{i+1}) drive
    transitions between grammatical classes. The substrate's co_occurs
    edges drive token selection within a class. Slots play no role.

    Per-step pipeline:
      1. read last emitted token's POS (via its IS_A edge)
      2. find next-POS via that POS's strongest FOLLOWS edge
      3. spread with WM seeded by recent tokens, goal=next_pos
      4. read out highest-activated token whose IS_A points to next_pos
         (excluding the antiloop window of recent emissions)

    Sentence length is bounded only by `max_new` — there is no sentence
    boundary signal in this corpus. Adding a START/END token would be
    the next refinement; for now `max_new` is the stop condition.

    Pure-spread version (no POS-FOLLOWS, no goal) was tried first and
    looped within a single POS class — has_member edges keep activation
    circulating among same-class tokens with no transition signal. POS-
    FOLLOWS is a substrate-stored fact about grammar, not a crutch.
    """
    if not prompt_tokens:
        raise ValueError('prompt must contain at least one token')

    rel_is_a = brain.relation_id[IS_A]
    rel_follows = brain.relation_id[FOLLOWS]

    emitted_ids: List[int] = []
    for t in prompt_tokens:
        nid = vocab.token_to_id.get(t)
        if nid is None:
            raise KeyError(f'Token not in vocab: {t!r}')
        emitted_ids.append(nid)

    out: List[str] = []

    for _ in range(max_new):
        # 1. Last emitted token's POS — substrate fact
        last_id = emitted_ids[-1]
        cur_pos_nid = None
        for syn in brain.synapses_of(last_id):
            if int(syn['relation']) == rel_is_a:
                tid = int(syn['to_id'])
                if tid in vocab.id_to_pos:
                    cur_pos_nid = tid
                    break
        if cur_pos_nid is None:
            break

        # 2. Next POS via the strongest POS-FOLLOWS edge
        next_pos_nid = None
        best_w = -1.0
        for syn in brain.synapses_of(cur_pos_nid):
            if int(syn['relation']) == rel_follows:
                w = float(syn['weight'])
                if w > best_w:
                    best_w = w
                    next_pos_nid = int(syn['to_id'])
        if next_pos_nid is None:
            break  # current POS has no successor — sentence end

        # 3. Spread from WM-seeded prompt with next-POS as goal
        wm = WorkingMemory(decay=wm_decay, max_size=64, floor=0.05)
        seeds = {nid: wm_decay ** offset
                  for offset, nid in enumerate(reversed(emitted_ids))}
        wm.absorb(seeds, gain=1.0)

        state = spread(brain, seeds=[],
                        working_memory=wm,
                        goals=[next_pos_nid],
                        goal_strength=goal_strength,
                        max_steps=max_steps,
                        sparsity=1.0)

        # 4. Highest-activated token whose POS matches the goal
        recent = set(emitted_ids[-antiloop_window:]) if antiloop_window else set()
        best_id = None
        best_score = -1.0
        for nid, lvl in state.activation.items():
            if nid not in vocab.id_to_token:
                continue
            if nid in recent:
                continue
            token_pos_match = any(
                int(syn['to_id']) == next_pos_nid
                and int(syn['relation']) == rel_is_a
                for syn in brain.synapses_of(nid)
            )
            if not token_pos_match:
                continue
            if lvl > best_score:
                best_score = lvl
                best_id = nid

        if best_id is None or best_score <= 0:
            break

        out.append(vocab.id_to_token[best_id])
        emitted_ids.append(best_id)

    return out


# ─── Smoke main ─────────────────────────────────────────────────────────────

def _smoke_main() -> int:
    """Teach one sentence; compare both generation strategies."""
    sentence = "the quick brown fox jumps over the lazy dog".split()
    pos_tags = ["DET", "ADJ", "ADJ", "NOUN", "VERB", "PREP", "DET", "ADJ", "NOUN"]

    brain, vocab = build_lm_brain()
    teach_sentence(brain, vocab, sentence, pos_tags)

    prompt = sentence[:2]
    target = sentence[2:]
    out_lookup = generate(brain, vocab, prompt, max_new=len(target))
    out_spread = generate_via_spread(brain, vocab, prompt, max_new=len(target))

    print(f'vocab size  : {len(vocab)} tokens, {len(vocab.pos_to_id)} POS tags, '
          f'{len(vocab.slot_to_id)} slots')
    print(f'brain neurons: {brain.size}  synapses: {getattr(brain, "_used_synapses", 0)}')
    print(f'prompt              : {" ".join(prompt)}')
    print(f'target              : {" ".join(target)}')
    print(f'generated (lookup)  : {" ".join(out_lookup)}')
    print(f'generated (spread)  : {" ".join(out_spread)}')

    n_lookup = sum(1 for a, b in zip(target, out_lookup) if a == b)
    n_spread = sum(1 for a, b in zip(target, out_spread) if a == b)
    print(f'accuracy (lookup)   : {n_lookup}/{len(target)} = {n_lookup/len(target):.0%}')
    print(f'accuracy (spread)   : {n_spread}/{len(target)} = {n_spread/len(target):.0%}')

    return 0 if (n_lookup == len(target) and n_spread == len(target)) else 1


if __name__ == '__main__':
    raise SystemExit(_smoke_main())
