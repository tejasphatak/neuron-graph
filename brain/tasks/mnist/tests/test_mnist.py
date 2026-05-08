"""MNIST smoke tests on synthetic data — no network access required.

Tests both the substrate's classification pipeline AND the scale-
invariant encoder. A 4-class synthetic dataset (4 distinct templates
with noise) verifies the full train→predict loop works.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(_HERE))))
sys.path.insert(0, _REPO)

from brain.tasks.mnist.mnist import (
    build_mnist_brain,
    encode_image, predict, train_epoch, evaluate,
)
from brain.tasks.mnist.encoder import ImageEncoder, ImageEncoderVocab


def _make_template(class_id: int, *, noise: float = 0.0,
                    rng: np.random.Generator = None,
                    size: int = 28) -> np.ndarray:
    """4 visually-distinct templates, one per class. Configurable size."""
    img = np.zeros((size, size), dtype=np.float32)
    s_low = max(2, size // 14)
    s_high = max(s_low + 4, size - size // 14)
    s_mid_lo = max(4, size // 7)
    s_mid_hi = size - max(4, size // 7)
    if class_id == 0:    # top stripe
        img[s_low:s_low + 4, s_mid_lo:s_mid_hi] = 1.0
    elif class_id == 1:  # bottom stripe
        img[s_high - 4:s_high, s_mid_lo:s_mid_hi] = 1.0
    elif class_id == 2:  # left stripe
        img[s_mid_lo:s_mid_hi, s_low:s_low + 4] = 1.0
    elif class_id == 3:  # right stripe
        img[s_mid_lo:s_mid_hi, s_high - 4:s_high] = 1.0
    if noise > 0 and rng is not None:
        img = np.clip(img + rng.uniform(-noise, noise, img.shape), 0.0, 1.0)
    return img


def _synthetic_dataset(n_per_class: int = 25, noise: float = 0.1,
                        size: int = 28, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = []
    y = []
    for cls in range(4):
        for _ in range(n_per_class):
            X.append(_make_template(cls, noise=noise, rng=rng, size=size))
            y.append(cls)
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    idx = rng.permutation(len(X))
    return X[idx], y[idx]


class TestBuild:
    def test_brain_has_input_and_digit_neurons(self):
        brain, vocab, encoder = build_mnist_brain()
        # Default 16×16 × 4 levels = 1024 input + 10 digit = 1034 neurons
        expected = encoder.n_input_neurons() + 10
        assert brain.size == expected

    def test_initial_brain_has_no_synapses(self):
        brain, _, _ = build_mnist_brain()
        assert getattr(brain, '_used_synapses', 0) == 0

    def test_custom_encoder(self):
        encoder = ImageEncoder(grid_size=8, n_levels=2)
        brain, vocab, _ = build_mnist_brain(encoder=encoder)
        # 8×8 × 2 = 128 input + 10 digit = 138 neurons
        assert brain.size == 138


class TestEncoder:
    def test_blank_image_produces_no_seeds(self):
        _, vocab, encoder = build_mnist_brain()
        seeds = encode_image(np.zeros((28, 28), dtype=np.float32),
                              vocab, encoder=encoder)
        assert len(seeds) == 0

    def test_full_image_produces_one_seed_per_cell(self):
        """One-hot per cell: each grid cell fires exactly one level neuron."""
        _, vocab, encoder = build_mnist_brain()
        seeds = encode_image(np.ones((28, 28), dtype=np.float32),
                              vocab, encoder=encoder)
        # 16×16 = 256 cells, each fires one level → 256 seeds
        assert len(seeds) == encoder.grid_size * encoder.grid_size

    def test_scale_invariance_same_neuron_count(self):
        """Different image sizes should produce same fixed neuron pattern."""
        _, vocab, encoder = build_mnist_brain()
        # Same logical content (full white) at 3 different sizes
        s28 = encode_image(np.ones((28, 28), dtype=np.float32),
                            vocab, encoder=encoder)
        s64 = encode_image(np.ones((64, 64), dtype=np.float32),
                            vocab, encoder=encoder)
        s100 = encode_image(np.ones((100, 100), dtype=np.float32),
                             vocab, encoder=encoder)
        # All same length AND all activating same neurons (same content)
        assert len(s28) == len(s64) == len(s100)
        assert set(s28.keys()) == set(s64.keys()) == set(s100.keys())


class TestSyntheticTraining:
    """Substrate should learn 4-class synthetic templates from reward."""

    def test_substrate_learns_from_synthetic_data(self):
        X_train, y_train = _synthetic_dataset(n_per_class=25, seed=0)
        X_test, y_test = _synthetic_dataset(n_per_class=10, seed=1)

        brain, vocab, encoder = build_mnist_brain()
        cold = evaluate(brain, vocab, X_test, y_test, encoder=encoder)
        rng = np.random.default_rng(0)
        for _ in range(3):
            train_epoch(brain, vocab, X_train, y_train,
                        encoder=encoder, eta=0.15, rng=rng)
        post = evaluate(brain, vocab, X_test, y_test, encoder=encoder)

        assert post['accuracy'] > cold['accuracy'] + 0.30, (
            f'substrate did not learn: cold={cold["accuracy"]:.1%} '
            f'post={post["accuracy"]:.1%}'
        )
        assert post['accuracy'] >= 0.85, (
            f'post-training accuracy too low: {post["accuracy"]:.1%}'
        )

    def test_substrate_learns_at_different_input_size(self):
        """Same code should work on 64×64 inputs without changes — the
        encoder normalizes to internal grid."""
        X_train, y_train = _synthetic_dataset(n_per_class=25, size=64, seed=0)
        X_test, y_test = _synthetic_dataset(n_per_class=10, size=64, seed=1)

        brain, vocab, encoder = build_mnist_brain()
        rng = np.random.default_rng(0)
        for _ in range(3):
            train_epoch(brain, vocab, X_train, y_train,
                        encoder=encoder, eta=0.15, rng=rng)
        post = evaluate(brain, vocab, X_test, y_test, encoder=encoder)
        assert post['accuracy'] >= 0.80, (
            f'failed at 64×64: {post["accuracy"]:.1%}'
        )

    def test_synapses_grow_during_training(self):
        X_train, y_train = _synthetic_dataset(n_per_class=15, seed=0)
        brain, vocab, encoder = build_mnist_brain()
        initial = getattr(brain, '_used_synapses', 0)
        train_epoch(brain, vocab, X_train, y_train,
                    encoder=encoder, eta=0.1,
                    rng=np.random.default_rng(0))
        assert getattr(brain, '_used_synapses', 0) > initial


class TestPredict:
    def test_blank_image_predicts_none(self):
        brain, vocab, encoder = build_mnist_brain()
        pred = predict(brain, vocab,
                        np.zeros((28, 28), dtype=np.float32),
                        encoder=encoder)
        assert pred is None

    def test_untrained_predict_is_none(self):
        brain, vocab, encoder = build_mnist_brain()
        img = np.ones((28, 28), dtype=np.float32)
        pred = predict(brain, vocab, img, encoder=encoder)
        assert pred is None
