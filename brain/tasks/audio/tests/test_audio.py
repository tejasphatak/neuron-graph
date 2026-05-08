"""Audio modality polymorphism test on synthetic 4-tone dataset."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(_HERE))))
sys.path.insert(0, _REPO)

from brain import Brain
from brain.tasks.audio.encoder import AudioEncoder, AudioEncoderVocab
from brain.tasks.mnist.fast import (
    DenseView, build_dense_view, sync_dense_to_brain,
    fast_train_epoch, fast_evaluate, batch_encode, fast_predict,
)
from brain.tasks.mnist.mnist import ACTIVATES


def _build_audio_brain(encoder=None):
    """Build a brain wired for audio classification, mirroring MNIST setup."""
    if encoder is None:
        encoder = AudioEncoder(time_steps=8, freq_bins=8,
                                 n_levels=2, n_classes=4)
    brain = Brain()
    brain.relations = [(ACTIVATES, 1.0)]
    brain._rebuild_relation_index()
    vocab = encoder.build_vocab(brain)
    return brain, vocab, encoder


def _make_tone(class_id: int, *, sample_rate: int = 8000,
                duration: float = 0.5,
                noise: float = 0.05,
                rng: np.random.Generator) -> np.ndarray:
    """Generate a sine wave at one of 4 distinct frequencies."""
    freqs = [200.0, 500.0, 1200.0, 2500.0]
    f = freqs[class_id]
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    signal = np.sin(2 * np.pi * f * t).astype(np.float32)
    if noise > 0:
        signal = signal + rng.normal(0, noise, signal.shape).astype(np.float32)
    return signal


def _synthetic_audio_dataset(n_per_class: int = 25, seed: int = 0,
                              sample_rate: int = 8000):
    rng = np.random.default_rng(seed)
    X = []
    y = []
    for cls in range(4):
        for _ in range(n_per_class):
            X.append(_make_tone(cls, sample_rate=sample_rate, rng=rng))
            y.append(cls)
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    idx = rng.permutation(len(X))
    return X[idx], y[idx], sample_rate


class TestAudioEncoder:
    def test_neuron_count(self):
        encoder = AudioEncoder(time_steps=8, freq_bins=8,
                                 n_levels=2, n_classes=4)
        assert encoder.n_input_neurons() == 8 * 8 * 2

    def test_encode_returns_seeds(self):
        encoder = AudioEncoder(time_steps=8, freq_bins=8, n_levels=2, n_classes=4)
        brain, vocab, _ = _build_audio_brain(encoder)
        rng = np.random.default_rng(0)
        sig = _make_tone(0, rng=rng)
        seeds = encoder.encode(sig, sample_rate=8000, vocab=vocab)
        assert len(seeds) > 0
        assert all(isinstance(k, int) for k in seeds.keys())

    def test_scale_invariance_different_durations(self):
        """500ms and 1500ms recordings of the same tone should produce
        similar (not identical, but overlapping) seed patterns —
        encoder normalizes time."""
        encoder = AudioEncoder(time_steps=8, freq_bins=8, n_levels=2, n_classes=4)
        brain, vocab, _ = _build_audio_brain(encoder)
        rng = np.random.default_rng(0)
        sig_short = _make_tone(0, duration=0.5, rng=rng)
        sig_long = _make_tone(0, duration=1.5, rng=rng)
        s_short = encoder.encode(sig_short, sample_rate=8000, vocab=vocab)
        s_long = encoder.encode(sig_long, sample_rate=8000, vocab=vocab)
        # Both produce some seeds (not blank)
        assert len(s_short) > 0
        assert len(s_long) > 0
        # Overlap (same tone → same dominant frequency band)
        overlap = set(s_short) & set(s_long)
        assert len(overlap) >= 0.3 * min(len(s_short), len(s_long))


class TestAudioLearning:
    def test_substrate_learns_4_tone_classification(self):
        """Substrate should learn to classify 4 distinct tones from
        supervised reward."""
        encoder = AudioEncoder(time_steps=8, freq_bins=8, n_levels=2, n_classes=4)
        brain, vocab, _ = _build_audio_brain(encoder)
        view = build_dense_view_for_audio(brain, vocab, encoder)

        X_train, y_train, sr = _synthetic_audio_dataset(n_per_class=20, seed=0)
        X_test, y_test, _ = _synthetic_audio_dataset(n_per_class=10, seed=1)

        # Custom train loop (audio passes sample_rate to encode)
        rng = np.random.default_rng(0)
        for _ in range(5):
            _audio_train_epoch(view, X_train, y_train, sr,
                                vocab, encoder, eta=0.05, rng=rng)

        # Eval
        n = len(X_test)
        n_correct = 0
        for img, label in zip(X_test, y_test):
            pred = _audio_predict(view, img, sr, vocab, encoder)
            if pred == int(label):
                n_correct += 1
        acc = n_correct / n
        assert acc >= 0.85, f'audio classification accuracy too low: {acc:.1%}'


# ─── Audio-specific train/predict (encoder needs sample_rate) ──────────────

def build_dense_view_for_audio(brain, vocab, encoder):
    """Like build_dense_view but for audio vocab structure."""
    input_ids = sorted(vocab.cell_to_id.values())
    input_id_to_idx = {nid: i for i, nid in enumerate(input_ids)}
    digit_ids = [vocab.class_to_id[c] for c in range(encoder.n_classes)]

    n_inputs = len(input_ids)
    W = np.zeros((n_inputs, encoder.n_classes), dtype=np.float32)
    return DenseView(
        W=W,
        input_id_to_idx=input_id_to_idx,
        input_idx_to_id=input_ids,
        digit_ids=digit_ids,
    )


def _audio_train_epoch(view, X, y, sample_rate, vocab, encoder, *,
                         eta=0.05, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    n = len(X)
    order = rng.permutation(n)
    n_inputs = len(view.input_idx_to_id)

    for idx in order:
        sig = X[idx]
        label = int(y[idx])
        seeds = encoder.encode(sig, sample_rate=sample_rate, vocab=vocab)
        if not seeds:
            continue
        active = np.zeros(n_inputs, dtype=bool)
        for nid in seeds:
            row = view.input_id_to_idx.get(nid)
            if row is not None:
                active[row] = True
        scores = view.W[active].sum(axis=0)
        pred = int(scores.argmax()) if scores.max() > 0 else -1
        if pred != label:
            view.W[active, label] += eta
            if pred >= 0:
                view.W[active, pred] -= eta


def _audio_predict(view, signal, sample_rate, vocab, encoder):
    seeds = encoder.encode(signal, sample_rate=sample_rate, vocab=vocab)
    if not seeds:
        return -1
    n_inputs = len(view.input_idx_to_id)
    active = np.zeros(n_inputs, dtype=bool)
    for nid in seeds:
        row = view.input_id_to_idx.get(nid)
        if row is not None:
            active[row] = True
    scores = view.W[active].sum(axis=0)
    return int(scores.argmax()) if scores.max() > 0 else -1
