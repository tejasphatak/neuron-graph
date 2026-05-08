"""Tests for the Phase A++ substrate primitives:
working memory, trace log, modulator, replay buffer, goal injection."""

from __future__ import annotations

import os
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_GURU = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _GURU)

from brain import (
    Brain, seed_brain, spread, overlap_similarity,
    WorkingMemory, TraceLog, Modulator, ReplayBuffer, Episode, consolidate,
)


# ─── Working memory ──────────────────────────────────────────────────────

class TestWorkingMemory:
    def test_absorb_and_seeds(self):
        wm = WorkingMemory(decay=0.8)
        wm.absorb({1: 1.0, 2: 0.5}, gain=1.0)
        assert wm.seeds() == {1: 1.0, 2: 0.5}

    def test_tick_decays(self):
        wm = WorkingMemory(decay=0.5, floor=0.01)
        wm.absorb({1: 1.0}, gain=1.0)
        wm.tick()
        assert wm.activation[1] == pytest.approx(0.5)
        wm.tick()
        assert wm.activation[1] == pytest.approx(0.25)

    def test_floor_drops_below_threshold(self):
        wm = WorkingMemory(decay=0.1, floor=0.5)
        wm.absorb({1: 0.8}, gain=1.0)
        wm.tick()
        # 0.8 * 0.1 = 0.08, below floor 0.5 → dropped
        assert 1 not in wm.activation

    def test_max_size_caps(self):
        wm = WorkingMemory(max_size=3)
        wm.absorb({1: 0.9, 2: 0.7, 3: 0.5, 4: 0.3, 5: 0.1}, gain=1.0)
        assert len(wm.activation) == 3
        # Top 3 by activation kept
        assert set(wm.activation.keys()) == {1, 2, 3}

    def test_spread_with_wm_carries_context(self):
        """A second spread sees the residual activation from the first."""
        b = seed_brain()
        wm = WorkingMemory(decay=0.6)
        s1 = spread(b, [b.lookup('cat')], working_memory=wm)
        # WM should now hold cat-related activation
        assert len(wm.activation) > 0
        # A fresh spread with empty seeds should still produce activation
        # because WM carries it
        s2 = spread(b, [], working_memory=wm)
        assert len(s2.activation) > 0


# ─── Goal injection in spread ────────────────────────────────────────────

class TestGoalInjection:
    def test_goal_neuron_stays_active(self):
        """A goal neuron should remain in the top-K throughout."""
        b = seed_brain()
        cat = b.lookup('cat')
        # Pick something cat doesn't connect to
        physics = b.lookup('physics')
        s = spread(b, [cat], goals=[physics], goal_strength=1.0)
        # Physics should be present despite no path from cat
        assert physics in s.activation
        assert s.activation[physics] >= 1.0

    def test_no_goal_no_unrelated_activation(self):
        """Without a goal, unrelated neurons shouldn't activate."""
        b = seed_brain()
        s = spread(b, [b.lookup('cat')])
        physics = b.lookup('physics')
        assert s.activation.get(physics, 0.0) == 0.0


# ─── Trace log ───────────────────────────────────────────────────────────

class TestTraceLog:
    def test_spread_logs_event(self):
        b = seed_brain()
        b.trace.clear()
        spread(b, [b.lookup('cat')])
        events = b.trace.tail()
        assert any(e['event'] == 'spread' for e in events)

    def test_summary_counts(self):
        b = seed_brain()
        b.trace.clear()
        for _ in range(5):
            spread(b, [b.lookup('cat')])
        s = b.trace.summary()
        assert s.get('spread', 0) == 5

    def test_dump_roundtrip(self, tmp_path):
        b = seed_brain()
        b.trace.clear()
        spread(b, [b.lookup('cat')])
        path = str(tmp_path / 'trace.jsonl')
        b.trace.dump(path)
        with open(path) as f:
            lines = f.read().strip().splitlines()
        assert len(lines) >= 1
        import json
        first = json.loads(lines[0])
        assert 'event' in first


# ─── Modulator ───────────────────────────────────────────────────────────

class TestModulator:
    def test_neutral_does_not_scale(self):
        m = Modulator(value=0.0)
        assert m.effective_eta(0.1) == pytest.approx(0.1)

    def test_positive_scales_up(self):
        m = Modulator(value=0.5)
        assert m.effective_eta(0.1) == pytest.approx(0.15)

    def test_negative_clamps_at_zero(self):
        # value clamped to min_value = -0.5; effective_eta = 0.1 * (1 + -0.5) = 0.05
        m = Modulator(value=-0.5)
        assert m.effective_eta(0.1) == pytest.approx(0.05)

    def test_adjust_decays_old_value(self):
        m = Modulator(value=0.4, decay=0.5)
        # decay 0.4 → 0.2, then add 0.5 * 0.3 = 0.15 → 0.35
        new = m.adjust(0.5, scale=0.3)
        assert new == pytest.approx(0.35, abs=0.001)


# ─── Replay buffer ───────────────────────────────────────────────────────

class TestReplayBuffer:
    def test_records_and_caps(self):
        rb = ReplayBuffer(capacity=3)
        for i in range(5):
            rb.record([(i, i)], reward=float(i))
        assert len(rb) == 3
        # Oldest two evicted; latest three kept
        rewards = sorted(e.reward for e in rb.episodes)
        assert rewards == [2.0, 3.0, 4.0]

    def test_sample_returns_n_or_fewer(self):
        rb = ReplayBuffer()
        for i in range(10):
            rb.record([(i, i)], reward=float(i))
        s = rb.sample(5)
        assert len(s) == 5

    def test_consolidate_calls_credit_fn_n_times(self):
        rb = ReplayBuffer()
        for i in range(5):
            rb.record([(i, i)], reward=1.0)
        called = []
        consolidate(rb, n_samples=3, credit_fn=lambda ep: called.append(ep))
        assert len(called) == 3

    def test_stats_are_sane(self):
        rb = ReplayBuffer()
        rb.record([(1, 1), (2, 2)], reward=0.7)
        rb.record([(3, 3)], reward=0.3)
        st = rb.stats()
        assert st['n'] == 2
        assert st['avg_reward'] == pytest.approx(0.5)
        assert st['avg_horizon'] == pytest.approx(1.5)


# ─── Brain has trace + modulator attached ────────────────────────────────

class TestBrainAttachments:
    def test_brain_has_trace(self):
        b = Brain()
        assert b.trace is not None
        assert hasattr(b.trace, 'log')

    def test_brain_has_modulator(self):
        b = Brain()
        assert b.modulator is not None
        assert b.modulator.effective_eta(0.1) == pytest.approx(0.1)
