"""Scale-invariant video encoder for the substrate.

Any video clip (any T, any H × W) → fixed-size substrate input. The
substrate doesn't see "frames at original resolution and frame rate"
— it sees "what intensity-level fired at each (time-chunk, row, col)."

Pipeline:
  T × H × W (or T × H × W × C) array
    → sample T_grid frames uniformly across time
    → convert each to grayscale if needed
    → resize each frame to grid_size × grid_size (PIL bilinear)
    → bin intensity into n_levels per pixel
    → emit one neuron per (frame_idx, row, col, level) above min level

Total neurons: T_grid × grid_size² × n_levels
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
from PIL import Image


@dataclass
class VideoEncoderVocab:
    """Maps (frame, row, col, level) → neuron-id, plus class neurons."""
    cell_to_id: Dict[Tuple[int, int, int, int], int] = field(default_factory=dict)
    id_to_cell: Dict[int, Tuple[int, int, int, int]] = field(default_factory=dict)
    class_to_id: Dict[int, int] = field(default_factory=dict)
    id_to_class: Dict[int, int] = field(default_factory=dict)


@dataclass
class VideoEncoder:
    t_grid: int = 8           # time chunks
    grid_size: int = 8        # spatial grid (per frame)
    n_levels: int = 2         # binary by default to keep neuron count low
    min_level_to_fire: int = 1
    n_classes: int = 10

    def n_input_neurons(self) -> int:
        return self.t_grid * self.grid_size * self.grid_size * self.n_levels

    def build_vocab(self, brain) -> VideoEncoderVocab:
        vocab = VideoEncoderVocab()
        for ti in range(self.t_grid):
            for r in range(self.grid_size):
                for c in range(self.grid_size):
                    for level in range(self.n_levels):
                        nid = brain.add_neuron(
                            lemma=f'vid:{ti},{r},{c},L{level}',
                            decay=0.5,
                        )
                        vocab.cell_to_id[(ti, r, c, level)] = nid
                        vocab.id_to_cell[nid] = (ti, r, c, level)
        for k in range(self.n_classes):
            nid = brain.add_neuron(lemma=f'vid_class:{k}', decay=0.7)
            vocab.class_to_id[k] = nid
            vocab.id_to_class[nid] = k
        return vocab

    def _to_grayscale(self, frame: np.ndarray) -> np.ndarray:
        """T×H×W or T×H×W×C → 2D grayscale [0,1]."""
        if frame.ndim == 3 and frame.shape[2] in (3, 4):
            # RGB or RGBA → luminance
            f = frame[..., :3].astype(np.float32)
            gray = 0.299 * f[..., 0] + 0.587 * f[..., 1] + 0.114 * f[..., 2]
            if frame.dtype == np.uint8:
                gray = gray / 255.0
        else:
            gray = frame.astype(np.float32)
            if frame.dtype == np.uint8 or gray.max() > 1.0:
                gray = gray / 255.0
        return np.clip(gray, 0.0, 1.0)

    def encode(self, video: np.ndarray,
                vocab: VideoEncoderVocab) -> Dict[int, float]:
        """video: array shape (T, H, W) or (T, H, W, C). Any T, H, W.
        Returns seed dict."""
        if video.ndim == 3:
            T = video.shape[0]
        elif video.ndim == 4:
            T = video.shape[0]
        else:
            raise ValueError(f'expected 3D or 4D video, got {video.shape}')

        # Sample T_grid frame indices uniformly across the video
        if T <= self.t_grid:
            sample_idx = list(range(T))
            # Pad by repeating last frame
            while len(sample_idx) < self.t_grid:
                sample_idx.append(T - 1)
        else:
            sample_idx = np.linspace(0, T - 1, self.t_grid).astype(int).tolist()

        seeds: Dict[int, float] = {}
        for ti, src_t in enumerate(sample_idx):
            frame = video[src_t]
            gray = self._to_grayscale(frame)

            # Resize to grid_size × grid_size
            if gray.shape != (self.grid_size, self.grid_size):
                pil = Image.fromarray((gray * 255).astype(np.uint8))
                pil = pil.resize((self.grid_size, self.grid_size), Image.BILINEAR)
                gray = np.asarray(pil, dtype=np.float32) / 255.0

            # Bin
            levels = (gray * self.n_levels).astype(np.int32)
            levels = np.clip(levels, 0, self.n_levels - 1)

            for r in range(self.grid_size):
                for c in range(self.grid_size):
                    level = int(levels[r, c])
                    if level >= self.min_level_to_fire:
                        nid = vocab.cell_to_id[(ti, r, c, level)]
                        seeds[nid] = 1.0
        return seeds
