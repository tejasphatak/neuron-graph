"""RL self-correction for the substrate-LM.

The substrate learns the right token-to-token routing by REWARD, not by
being told. Same pattern as brain/tasks/ttt/value_head.py:

  generate → compare to target → credit-assign per step → modulate edges

Initial knowledge given by `teach_minimal` is intentionally THIN:
  - each token's IS_A → POS  (token's grammatical class)
  - POS-FOLLOWS chain        (sentence-shape transitions)

That's all. No slots. No co_occurs from teaching. The substrate has to
DISCOVER which tokens follow which via the RL reward signal.

After enough episodes, generation should converge from random-within-POS
to the target sentences — proving the substrate self-organizes its
token-routing graph from reward alone.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from brain import Brain, WorkingMemory, spread, ReplayBuffer, Episode

from .tiny import (
    Vocab, build_lm_brain,
    IS_A, FOLLOWS, HAS_MEMBER, CO_OCCURS,
    _add_or_strengthen,
)


# ─── Minimal seed teaching (no co_occurs, no slots) ────────────────────────

def teach_minimal(brain: Brain, vocab: Vocab,
                   tokens: List[str], pos_tags: List[str]) -> None:
    """Seed only the POS class membership and the POS-FOLLOWS chain.

    NOT taught: co_occurs between tokens, slot bindings. Those are the
    things RL must DISCOVER from reward signal.
    """
    assert len(tokens) == len(pos_tags)
    n = len(tokens)

    tok_ids = [vocab.add_token(t, brain) for t in tokens]
    pos_ids = [vocab.add_pos(p, brain) for p in pos_tags]

    # token <--is_a / has_member--> POS
    for tid, pid in zip(tok_ids, pos_ids):
        _add_or_strengthen(brain, tid, pid, IS_A)
        _add_or_strengthen(brain, pid, tid, HAS_MEMBER)

    # POS-FOLLOWS chain (grammar shape)
    for i in range(n - 1):
        _add_or_strengthen(brain, pos_ids[i], pos_ids[i + 1], FOLLOWS)


# ─── Traced generation (records per-step state for credit assignment) ──────

@dataclass
class GenerationStep:
    """One token emission, with the substrate state that produced it.
    Credit assignment uses `activation` to find which neurons co-fired."""
    emitted_id: int
    activation: Dict[int, float] = field(default_factory=dict)
    next_pos_nid: int = -1
    seed_ids: List[int] = field(default_factory=list)


@dataclass
class Trajectory:
    """Full generation episode."""
    prompt_ids: List[int]
    steps: List[GenerationStep]
    output_tokens: List[str]


def _build_pos_groups(brain: Brain, vocab: Vocab) -> Dict[int, int]:
    """Build a {token_neuron_id: pos_neuron_id} group map.

    For sparsity-aware spread: tokens that share a POS form a group;
    `spread()` will only keep the top-K most-activated tokens per POS
    class each step, instead of firehose-pumping every member of every
    POS class via has_member edges.
    """
    rel_is_a = brain.relation_id[IS_A]
    groups: Dict[int, int] = {}
    for tid in vocab.id_to_token:
        for syn in brain.synapses_of(tid):
            if int(syn['relation']) == rel_is_a:
                pid = int(syn['to_id'])
                if pid in vocab.id_to_pos:
                    groups[tid] = pid
                    break
    return groups


def trace_generate(brain: Brain, vocab: Vocab,
                    prompt_tokens: List[str], *,
                    max_new: int = 20,
                    wm_decay: float = 0.6,
                    goal_strength: float = 2.0,
                    max_steps: int = 2,
                    antiloop_window: int = 2,
                    rng: Optional[np.random.Generator] = None,
                    explore_eps: float = 0.1,
                    pos_sequence: Optional[List[str]] = None,
                    group_top_k: Optional[int] = None) -> Trajectory:
    """Like generate_open_vocab but records per-step substrate state.

    `explore_eps` adds ε-greedy exploration: with probability `explore_eps`
    pick a random valid candidate instead of the argmax. This lets RL
    escape local minima during training.

    `pos_sequence`, if given, OVERRIDES the substrate's POS-FOLLOWS
    prediction at each step — used during training to teacher-force the
    grammar shape. Lets RL focus on learning *which* token within a POS
    class, separately from *which POS comes next*. (The latter is
    ambiguous at cap-saturated FOLLOWS weights — a separate problem.)
    """
    if rng is None:
        rng = np.random.default_rng()

    rel_is_a = brain.relation_id[IS_A]
    rel_follows = brain.relation_id[FOLLOWS]

    emitted_ids: List[int] = []
    for t in prompt_tokens:
        nid = vocab.token_to_id.get(t)
        if nid is None:
            raise KeyError(f'Token not in vocab: {t!r}')
        emitted_ids.append(nid)

    prompt_len = len(emitted_ids)
    steps: List[GenerationStep] = []
    out_tokens: List[str] = []

    # Pre-allocate step neurons we expect to use this episode.
    # step_i = the i-th emission AFTER the prompt (relative position).
    for k in range(max_new):
        vocab.add_step(k, brain)

    # Build the POS-class group map ONCE per generate call. Substrate
    # uses it for sparsity-aware spread — within each POS class, only
    # the top-K most-activated tokens survive each spread step.
    pos_groups = _build_pos_groups(brain, vocab) if group_top_k else None

    for step_idx in range(max_new):
        # Determine next POS:
        #  - if pos_sequence is given (training), look it up by absolute index
        #  - otherwise read last token's POS, then walk POS-FOLLOWS
        if pos_sequence is not None:
            abs_idx = (len(emitted_ids) - len(prompt_tokens)) + len(prompt_tokens)
            # = len(emitted_ids), i.e., the position of the token we're about to emit
            target_pos_idx = len(emitted_ids)
            if target_pos_idx >= len(pos_sequence):
                break
            target_pos_name = pos_sequence[target_pos_idx]
            next_pos_nid = vocab.pos_to_id.get(target_pos_name)
            if next_pos_nid is None:
                break
        else:
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

            next_pos_nid = None
            best_w = -1.0
            for syn in brain.synapses_of(cur_pos_nid):
                if int(syn['relation']) == rel_follows:
                    w = float(syn['weight'])
                    if w > best_w:
                        best_w = w
                        next_pos_nid = int(syn['to_id'])
            if next_pos_nid is None:
                break

        # Spread from WM-seeded prompt with positional embedding.
        # Token seeds carry recent-context (positional decay) in WM.
        # The step-position neuron is a GOAL (clamped throughout spread)
        # so its co_occurs edges to position-correct tokens fire at full
        # strength, providing the disambiguator that separates same-POS
        # tokens at different sentence positions.
        wm = WorkingMemory(decay=wm_decay, max_size=64, floor=0.05)
        seed_dict = {nid: wm_decay ** offset
                      for offset, nid in enumerate(reversed(emitted_ids))}
        wm.absorb(seed_dict, gain=1.0)

        cur_step_nid = vocab.step_to_id[step_idx]
        state = spread(brain, seeds=[],
                        working_memory=wm,
                        goals=[next_pos_nid, cur_step_nid],
                        goal_strength=goal_strength,
                        max_steps=max_steps,
                        sparsity=1.0,
                        groups=pos_groups,
                        group_top_k=group_top_k)

        # Candidate set: tokens whose IS_A points to next_pos_nid,
        # excluding the antiloop window
        recent = set(emitted_ids[-antiloop_window:]) if antiloop_window else set()
        candidates: List[Tuple[int, float]] = []
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
            if lvl > 0:
                candidates.append((nid, lvl))

        # If no scored candidates, fall back to any token of the right POS
        # (otherwise RL never gets a chance to grow co_occurs from a cold start)
        if not candidates:
            for nid in vocab.id_to_token:
                if nid in recent:
                    continue
                token_pos_match = any(
                    int(syn['to_id']) == next_pos_nid
                    and int(syn['relation']) == rel_is_a
                    for syn in brain.synapses_of(nid)
                )
                if token_pos_match:
                    candidates.append((nid, 1e-3))

        if not candidates:
            break

        # ε-greedy pick
        if rng.random() < explore_eps:
            best_id = candidates[int(rng.integers(0, len(candidates)))][0]
        else:
            best_id = max(candidates, key=lambda x: x[1])[0]

        # Store seed_ids in CHRONOLOGICAL order (oldest first, most-
        # recent last). apply_correction's `seed_ids[-context_window:]`
        # then correctly takes the most-recent N tokens. The step neuron
        # is appended at the end to also be credited.
        seed_list = list(emitted_ids)  # already chronological
        seed_list.append(cur_step_nid)
        steps.append(GenerationStep(
            emitted_id=best_id,
            activation=dict(state.activation),
            next_pos_nid=next_pos_nid,
            seed_ids=seed_list,
        ))
        out_tokens.append(vocab.id_to_token[best_id])
        emitted_ids.append(best_id)

    return Trajectory(prompt_ids=emitted_ids[:prompt_len],
                       steps=steps, output_tokens=out_tokens)


# ─── Reward + credit assignment ────────────────────────────────────────────

def reward_trajectory(traj: Trajectory, target: List[str],
                       *, correct_r: float = 1.0,
                       wrong_r: float = -0.5) -> List[float]:
    """Per-step reward by direct token match. Mismatch beyond `len(target)`
    counts as wrong (penalty for over-generation)."""
    rewards: List[float] = []
    for i, step_token in enumerate(traj.output_tokens):
        if i < len(target) and step_token == target[i]:
            rewards.append(correct_r)
        else:
            rewards.append(wrong_r)
    return rewards


def apply_correction(brain: Brain, vocab: Vocab,
                      traj: Trajectory, rewards: List[float],
                      *, eta: float = 0.10,
                      co_threshold: float = 0.15,
                      create_threshold: float = 0.30,
                      context_window: int = 3,
                      grow_relation: str = CO_OCCURS) -> Dict[str, int]:
    """Reward-modulated Hebbian update with LOCAL credit assignment.

    For each generation step, credit (or blame) is assigned only to the
    last `context_window` tokens before the emission — NOT the full
    history. Locality matters: without it, late-sentence tokens like
    'dog' get strengthened by the same prompt tokens that should drive
    early-sentence 'fox', so the substrate can't tell slot[3] from
    slot[8] given identical prompts.

    Credit weight = (positional decay of seed in WM) × emitted activation.
    The seed's WM strength encodes its relevance; post-spread activation
    is too diluted by has_member amplification to distinguish well.

    Reward > 0 reinforces the path that produced this token; reward < 0
    weakens it. Edges grown through CO_OCCURS — the same relation that
    disambiguated cross-sentence routing in the qualitative-teach
    experiments. Substrate GROWS them from experience here.
    """
    rel_co = brain.relation_id[grow_relation]
    updated = 0
    created = 0

    for step, r in zip(traj.steps, rewards):
        if r == 0 or not step.activation:
            continue
        emitted = step.emitted_id
        emitted_lvl = step.activation.get(emitted, 0.0)
        if emitted_lvl <= 0:
            continue

        # Local context: last N TOKEN seeds + step neuron at full strength.
        # The step neuron is critical — it carries position info that lets
        # 'fox' (step_0) and 'dog' (step_5) differentiate even when their
        # token contexts overlap.
        all_seeds = step.seed_ids
        token_seeds = [s for s in all_seeds if s in vocab.id_to_token]
        step_seeds = [s for s in all_seeds if s in vocab.id_to_step]

        recent_tokens = token_seeds[-context_window:] if context_window else token_seeds
        wm_decay_assumed = 0.6  # matches trace_generate default
        seed_strength: Dict[int, float] = {
            nid: wm_decay_assumed ** offset
            for offset, nid in enumerate(reversed(recent_tokens))
        }
        # Step neurons get full credit weight (they were seeded at 1.0)
        for sid in step_seeds:
            seed_strength[sid] = 1.0

        for nid, base_strength in seed_strength.items():
            if nid == emitted:
                continue
            # Allow tokens AND step neurons (substrate's positional embeddings)
            if nid not in vocab.id_to_token and nid not in vocab.id_to_step:
                continue
            if base_strength < co_threshold:
                continue
            joint = base_strength * emitted_lvl
            delta = eta * joint * r

            # Find existing edge nid → emitted with co_occurs relation
            edges = brain.synapses_of(nid)
            existing_idx = None
            for k, syn in enumerate(edges):
                if int(syn['to_id']) == emitted and int(syn['relation']) == rel_co:
                    existing_idx = k
                    break
            if existing_idx is not None:
                # Mutate weight in place
                from brain.neuron import SYNAPSE_DTYPE
                base = int(brain.nodes[nid]['syn_offset']) // SYNAPSE_DTYPE.itemsize
                old = float(brain.synapses[base + existing_idx]['weight'])
                new = max(0.0, min(1.0, old + delta))
                brain.synapses[base + existing_idx]['weight'] = new
                updated += 1
            elif r > 0 and joint >= create_threshold:
                # Grow a new edge — substrate plasticity earned via reward
                w = min(1.0, max(0.05, eta * joint * r * 5))
                brain.add_synapse(nid, emitted, rel_name=grow_relation, weight=w)
                # Symmetric (co_occurs is bidirectional in spirit)
                brain.add_synapse(emitted, nid, rel_name=grow_relation, weight=w)
                created += 2

    return {'updated': updated, 'created': created}


# ─── RL training loop ──────────────────────────────────────────────────────

def train_rl(brain: Brain, vocab: Vocab,
              training_pairs: List[Tuple[List[str], List[str], List[str]]],
              *, epochs: int = 50,
              eta: float = 0.10,
              explore_eps_start: float = 0.30,
              explore_eps_end: float = 0.02,
              seed: int = 0,
              verbose: bool = False,
              replay_capacity: int = 200,
              replay_per_step: int = 2,
              replay_eta_scale: float = 0.5,
              replay_min_reward: float = 0.5,
              group_top_k: Optional[int] = None,
              replay_buffer: Optional[ReplayBuffer] = None) -> List[Dict[str, float]]:
    """Each tuple = (prompt_tokens, target_continuation, full_pos_sequence).

    Replay buffer integration (NEW, scaled-corpus optimization):
      - Successful trajectories (mean reward ≥ replay_min_reward) are
        stored in a ReplayBuffer of capacity `replay_capacity`
      - After each fresh episode, `replay_per_step` past trajectories
        are sampled from the buffer and apply_correction is re-run on
        them at `replay_eta_scale × eta` learning rate
      - This fights catastrophic forgetting: training on sentence 20
        no longer fully erases weights for sentences 1-3
      - Same recipe that fixed TTT's value-head plateau

    Per epoch:
      for each (prompt, target, pos):
        generate fresh trajectory
        score → rewards → apply_correction
        if successful: record to replay buffer
        replay K past trajectories at reduced eta

    Returns per-epoch stats including replay activity.
    """
    rng = np.random.default_rng(seed)
    py_rng = random.Random(seed)
    history: List[Dict[str, float]] = []

    if replay_buffer is None:
        replay_buffer = ReplayBuffer(capacity=replay_capacity)

    for epoch in range(epochs):
        frac = epoch / max(1, epochs - 1)
        eps = explore_eps_start * (1 - frac) + explore_eps_end * frac

        total_correct = 0
        total_steps = 0
        total_reward = 0.0
        ep_updated = 0
        ep_created = 0
        ep_replays = 0

        for prompt, target, pos_seq in training_pairs:
            traj = trace_generate(brain, vocab, prompt,
                                   max_new=len(target),
                                   rng=rng, explore_eps=eps,
                                   pos_sequence=pos_seq,
                                   group_top_k=group_top_k)
            rewards = reward_trajectory(traj, target)
            mean_r = sum(rewards) / max(1, len(rewards))
            stats = apply_correction(brain, vocab, traj, rewards, eta=eta)

            total_correct += sum(1 for a, b in zip(traj.output_tokens, target) if a == b)
            total_steps += len(target)
            total_reward += sum(rewards)
            ep_updated += stats['updated']
            ep_created += stats['created']

            # Record successful trajectories for replay
            if mean_r >= replay_min_reward:
                replay_buffer.record(
                    trajectory=[(s, s.emitted_id) for s in traj.steps],
                    reward=mean_r,
                    full_traj=traj,
                    rewards=list(rewards),
                )

            # Replay K past trajectories at reduced learning rate
            if len(replay_buffer) >= 2 and replay_per_step > 0:
                samples = replay_buffer.sample(replay_per_step, rng=py_rng)
                for ep in samples:
                    past_traj = ep.metadata['full_traj']
                    past_rewards = ep.metadata['rewards']
                    rs = apply_correction(brain, vocab, past_traj, past_rewards,
                                          eta=eta * replay_eta_scale)
                    ep_replays += 1
                    ep_updated += rs['updated']
                    ep_created += rs['created']

        accuracy = total_correct / max(1, total_steps)
        history.append({
            'epoch': epoch,
            'accuracy': accuracy,
            'mean_reward': total_reward / max(1, total_steps),
            'eps': eps,
            'updated': ep_updated,
            'created': ep_created,
            'replays': ep_replays,
            'buffer_size': len(replay_buffer),
        })
        if verbose:
            print(f'  ep {epoch:3d}  acc={accuracy:.2%}  '
                  f'r̄={total_reward/max(1,total_steps):+.3f}  '
                  f'ε={eps:.3f}  upd={ep_updated:4d}  new={ep_created:3d}  '
                  f'rpl={ep_replays}')

    return history


# ─── Curriculum learning ──────────────────────────────────────────────────

def train_rl_curriculum(brain: Brain, vocab: Vocab,
                          training_pairs: List[Tuple[List[str], List[str], List[str]]],
                          *,
                          stages: int = 4,
                          epochs_per_stage: int = 100,
                          eta: float = 0.18,
                          explore_eps_start: float = 0.40,
                          explore_eps_end: float = 0.02,
                          seed: int = 0,
                          verbose: bool = False,
                          replay_capacity: int = 300,
                          replay_per_step: int = 2,
                          replay_eta_scale: float = 0.5,
                          replay_min_reward: float = 0.0,
                          group_top_k: Optional[int] = None
                          ) -> List[Dict[str, float]]:
    """Progressive subset curriculum.

    Stage k trains on training_pairs[0 : ceil(N * (k+1)/stages)] for
    epochs_per_stage epochs. The replay buffer PERSISTS across stages
    so successful trajectories from earlier sentences keep getting
    re-applied even as new sentences are introduced — fights
    catastrophic interference when expanding the training set.

    Rationale: at small scale (3 sentences) substrate hits 83%; at
    20-sentence scale plain RL plateaus at ~40% because all sentences
    fight for the same neuron weights at once. Curriculum lets the
    substrate master smaller subsets first, then adapt those weights
    when more diversity is introduced.
    """
    import math

    n = len(training_pairs)
    if n == 0:
        return []

    # Persistent replay buffer carries successful trajectories
    # across all curriculum stages
    persistent_buffer = ReplayBuffer(capacity=replay_capacity)

    history: List[Dict[str, float]] = []

    for stage in range(stages):
        stage_size = max(1, math.ceil(n * (stage + 1) / stages))
        stage_pairs = training_pairs[:stage_size]
        stage_seed = seed + stage * 1000  # different stream per stage

        if verbose:
            print(f'  --- Stage {stage+1}/{stages}: {stage_size} sentences ---')

        stage_history = train_rl(
            brain, vocab, stage_pairs,
            epochs=epochs_per_stage,
            eta=eta,
            explore_eps_start=explore_eps_start,
            explore_eps_end=explore_eps_end,
            seed=stage_seed,
            verbose=False,
            replay_per_step=replay_per_step,
            replay_eta_scale=replay_eta_scale,
            replay_min_reward=replay_min_reward,
            group_top_k=group_top_k,
            replay_buffer=persistent_buffer,
        )
        for h in stage_history:
            h['stage'] = stage
            h['stage_size'] = stage_size
        history.extend(stage_history)
        if verbose:
            best_in_stage = max(stage_history, key=lambda h: h['accuracy'])
            print(f'    stage {stage+1} best: {best_in_stage["accuracy"]:.1%}')

    return history
