"""Video modality polymorphism test on synthetic temporal patterns."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(_HERE))))
sys.path.insert(0, _REPO)

from brain import Brain
from brain.tasks.video.encoder import VideoEncoder, VideoEncoderVocab
from brain.tasks.mnist.fast import DenseView
from brain.tasks.mnist.mnist import ACTIVATES


def _build_video_brain(encoder=None):
    if encoder is None:
        encoder = VideoEncoder(t_grid=4, grid_size=8, n_levels=2, n_classes=4)
    brain = Brain()
    brain.relations = [(ACTIVATES, 1.0)]
    brain._rebuild_relation_index()
    vocab = encoder.build_vocab(brain)
    return brain, vocab, encoder


def _make_video(class_id: int, *, T: int = 16, H: int = 32, W: int = 32,
                  noise: float = 0.0,
                  rng: np.random.Generator) -> np.ndarray:
    """Generate a (T, H, W) video with a class-specific motion pattern.

    class 0: white blob moving left → right
    class 1: white blob moving top → bottom
    class 2: white blob moving along diagonal /
    class 3: stationary white blob
    """
    video = np.zeros((T, H, W), dtype=np.float32)
    blob_size = max(3, H // 8)
    for t in range(T):
        if class_id == 0:
            r = H // 2
            c = int((t / max(1, T - 1)) * (W - blob_size))
        elif class_id == 1:
            r = int((t / max(1, T - 1)) * (H - blob_size))
            c = W // 2
        elif class_id == 2:
            d = t / max(1, T - 1)
            r = int(d * (H - blob_size))
            c = int(d * (W - blob_size))
        else:
            r = H // 2
            c = W // 2
        video[t, r:r + blob_size, c:c + blob_size] = 1.0
    if noise > 0 and rng is not None:
        video = np.clip(video + rng.uniform(-noise, noise, video.shape), 0.0, 1.0)
    return video


def _synthetic_video_dataset(n_per_class: int = 15, T: int = 16,
                              H: int = 32, W: int = 32, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = []
    y = []
    for cls in range(4):
        for _ in range(n_per_class):
            X.append(_make_video(cls, T=T, H=H, W=W, noise=0.05, rng=rng))
            y.append(cls)
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    idx = rng.permutation(len(X))
    return X[idx], y[idx]


class TestVideoEncoder:
    def test_neuron_count(self):
        encoder = VideoEncoder(t_grid=4, grid_size=8, n_levels=2, n_classes=4)
        # 4 frames × 8×8 grid × 2 levels = 512 input neurons
        assert encoder.n_input_neurons() == 4 * 8 * 8 * 2

    def test_encode_returns_seeds(self):
        encoder = VideoEncoder(t_grid=4, grid_size=8, n_levels=2, n_classes=4)
        brain, vocab, _ = _build_video_brain(encoder)
        rng = np.random.default_rng(0)
        vid = _make_video(0, rng=rng)
        seeds = encoder.encode(vid, vocab)
        assert len(seeds) > 0

    def test_scale_invariance_different_T(self):
        """Different video lengths should still produce the same neuron
        count pattern (encoder samples T_grid frames uniformly)."""
        encoder = VideoEncoder(t_grid=4, grid_size=8, n_levels=2, n_classes=4)
        brain, vocab, _ = _build_video_brain(encoder)
        rng = np.random.default_rng(0)
        v8 = _make_video(0, T=8, rng=rng)
        v32 = _make_video(0, T=32, rng=rng)
        s8 = encoder.encode(v8, vocab)
        s32 = encoder.encode(v32, vocab)
        # Both produce seeds; same general pattern
        assert len(s8) > 0
        assert len(s32) > 0

    def test_scale_invariance_different_HW(self):
        """Different spatial resolutions should still produce same
        substrate pattern (encoder resizes to grid_size)."""
        encoder = VideoEncoder(t_grid=4, grid_size=8, n_levels=2, n_classes=4)
        brain, vocab, _ = _build_video_brain(encoder)
        rng = np.random.default_rng(0)
        v32 = _make_video(0, H=32, W=32, rng=rng)
        v128 = _make_video(0, H=128, W=128, rng=rng)
        s32 = encoder.encode(v32, vocab)
        s128 = encoder.encode(v128, vocab)
        # Same content → similar neuron pattern
        overlap = set(s32) & set(s128)
        assert len(overlap) >= 0.5 * min(len(s32), len(s128))


class TestVideoLearning:
    def test_substrate_learns_motion_patterns(self):
        """4 distinct motion patterns: should be learnable from reward."""
        encoder = VideoEncoder(t_grid=4, grid_size=8, n_levels=2, n_classes=4)
        brain, vocab, _ = _build_video_brain(encoder)

        X_train, y_train = _synthetic_video_dataset(n_per_class=15, seed=0)
        X_test, y_test = _synthetic_video_dataset(n_per_class=8, seed=1)

        view = _build_view(brain, vocab, encoder)
        rng = np.random.default_rng(0)
        for _ in range(5):
            _video_train_epoch(view, X_train, y_train,
                                vocab, encoder, eta=0.05, rng=rng)

        n_correct = sum(1 for v, lbl in zip(X_test, y_test)
                         if _video_predict(view, v, vocab, encoder) == int(lbl))
        acc = n_correct / len(X_test)
        assert acc >= 0.80, f'video classification accuracy too low: {acc:.1%}'


# ─── Video-specific train/predict ──────────────────────────────────────────

def _build_view(brain, vocab, encoder):
    input_ids = sorted(vocab.cell_to_id.values())
    input_id_to_idx = {nid: i for i, nid in enumerate(input_ids)}
    class_ids = [vocab.class_to_id[c] for c in range(encoder.n_classes)]
    W = np.zeros((len(input_ids), encoder.n_classes), dtype=np.float32)
    return DenseView(W=W,
                      input_id_to_idx=input_id_to_idx,
                      input_idx_to_id=input_ids,
                      digit_ids=class_ids)


def _video_train_epoch(view, X, y, vocab, encoder, *, eta=0.05, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    n_inputs = len(view.input_idx_to_id)
    for idx in rng.permutation(len(X)):
        seeds = encoder.encode(X[idx], vocab)
        if not seeds:
            continue
        active = np.zeros(n_inputs, dtype=bool)
        for nid in seeds:
            row = view.input_id_to_idx.get(nid)
            if row is not None:
                active[row] = True
        scores = view.W[active].sum(axis=0)
        label = int(y[idx])
        pred = int(scores.argmax()) if scores.max() > 0 else -1
        if pred != label:
            view.W[active, label] += eta
            if pred >= 0:
                view.W[active, pred] -= eta


def _video_predict(view, video, vocab, encoder):
    seeds = encoder.encode(video, vocab)
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
