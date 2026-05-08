"""MNIST smoke tests on synthetic data — no network access required.

Tests the substrate's classification pipeline without depending on
sklearn/internet. A 4-class synthetic dataset (4 distinct 28×28
templates with noise) verifies the full train→predict loop works.
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
    MnistVocab, build_mnist_brain,
    encode_image, predict, train_epoch, evaluate,
)


def _make_template(class_id: int, *, noise: float = 0.0,
                    rng: np.random.Generator = None) -> np.ndarray:
    """4 visually-distinct templates, one per class."""
    img = np.zeros((28, 28), dtype=np.float32)
    if class_id == 0:    # top stripe
        img[2:6, 4:24] = 1.0
    elif class_id == 1:  # bottom stripe
        img[22:26, 4:24] = 1.0
    elif class_id == 2:  # left stripe
        img[4:24, 2:6] = 1.0
    elif class_id == 3:  # right stripe
        img[4:24, 22:26] = 1.0
    if noise > 0 and rng is not None:
        img = np.clip(img + rng.uniform(-noise, noise, img.shape), 0.0, 1.0)
    return img


def _synthetic_dataset(n_per_class: int = 25, noise: float = 0.1,
                        seed: int = 0):
    rng = np.random.default_rng(seed)
    X = []
    y = []
    for cls in range(4):
        for _ in range(n_per_class):
            X.append(_make_template(cls, noise=noise, rng=rng))
            y.append(cls)
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    idx = rng.permutation(len(X))
    return X[idx], y[idx]


class TestBuild:
    def test_brain_has_pixel_and_digit_neurons(self):
        brain, vocab = build_mnist_brain(image_size=28)
        # 28×28 pixels = 784, plus 10 digit class neurons
        assert brain.size == 784 + 10
        assert len(vocab.pixel_to_id) == 784
        assert len(vocab.digit_to_id) == 10

    def test_initial_brain_has_no_synapses(self):
        """teach_minimal-style: brain starts with NO edges; RL grows them."""
        brain, _ = build_mnist_brain()
        assert getattr(brain, '_used_synapses', 0) == 0


class TestEncoding:
    def test_blank_image_produces_no_seeds(self):
        _, vocab = build_mnist_brain()
        seeds = encode_image(np.zeros((28, 28), dtype=np.float32), vocab)
        assert len(seeds) == 0

    def test_full_image_produces_all_pixel_seeds(self):
        _, vocab = build_mnist_brain()
        seeds = encode_image(np.ones((28, 28), dtype=np.float32), vocab)
        assert len(seeds) == 784


class TestSyntheticTraining:
    """The defining test: substrate should learn 4-class synthetic
    templates from supervised reward."""

    def test_substrate_learns_from_synthetic_data(self):
        X_train, y_train = _synthetic_dataset(n_per_class=25, seed=0)
        X_test, y_test = _synthetic_dataset(n_per_class=10, seed=1)

        brain, vocab = build_mnist_brain()
        # Cold-start eval (untrained = ~25% on 4 balanced classes)
        cold = evaluate(brain, vocab, X_test, y_test)
        # Train for a few epochs
        rng = np.random.default_rng(0)
        for _ in range(3):
            train_epoch(brain, vocab, X_train, y_train, eta=0.15, rng=rng)
        post = evaluate(brain, vocab, X_test, y_test)

        # Substrate must significantly beat cold-start
        assert post['accuracy'] > cold['accuracy'] + 0.30, (
            f'substrate did not learn: cold={cold["accuracy"]:.1%} '
            f'post={post["accuracy"]:.1%}'
        )
        # On 4 visually-distinct classes, should reach high accuracy
        assert post['accuracy'] >= 0.90, (
            f'post-training accuracy too low: {post["accuracy"]:.1%}'
        )

    def test_synapses_grow_during_training(self):
        X_train, y_train = _synthetic_dataset(n_per_class=15, seed=0)
        brain, vocab = build_mnist_brain()
        initial = getattr(brain, '_used_synapses', 0)
        train_epoch(brain, vocab, X_train, y_train, eta=0.1,
                     rng=np.random.default_rng(0))
        assert getattr(brain, '_used_synapses', 0) > initial


class TestPredict:
    def test_blank_image_predicts_none(self):
        brain, vocab = build_mnist_brain()
        pred = predict(brain, vocab, np.zeros((28, 28), dtype=np.float32))
        assert pred is None

    def test_untrained_predict_is_arbitrary_but_valid(self):
        """An untrained brain has no edges → no digit activation → None."""
        brain, vocab = build_mnist_brain()
        # Simple all-on image
        img = np.ones((28, 28), dtype=np.float32)
        pred = predict(brain, vocab, img)
        # Untrained: no pixel→digit edges, so digit neurons don't activate
        assert pred is None
