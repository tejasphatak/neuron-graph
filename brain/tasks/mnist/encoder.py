"""Scale-invariant image encoder for the substrate.

Any image (any size, any aspect ratio) → fixed-size grid of substrate
neurons. The substrate doesn't see "pixels at original resolution" —
it sees "what intensity-level fired at each grid cell." That decouples
the substrate's input dimensionality from the input image's size.

Encoding:
  resize to grid_size × grid_size     (PIL bilinear)
  bin intensity into n_levels buckets (0 = empty, n_levels-1 = full)
  emit one neuron per (row, col, level) — only one level fires per cell

Total neurons: grid_size² × n_levels
  16 × 16 × 4 = 1024  (default — richer than 784 binary, scale-invariant)
   8 ×  8 × 4 =  256  (faster, less detail)
  32 × 32 × 4 = 4096  (more detail, slower)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
from PIL import Image


@dataclass
class ImageEncoderVocab:
    """Maps (grid-row, grid-col, intensity-level) → neuron-id, plus
    digit class neurons (output side)."""
    cell_to_id: Dict[Tuple[int, int, int], int] = field(default_factory=dict)
    id_to_cell: Dict[int, Tuple[int, int, int]] = field(default_factory=dict)
    digit_to_id: Dict[int, int] = field(default_factory=dict)
    id_to_digit: Dict[int, int] = field(default_factory=dict)


@dataclass
class ImageEncoder:
    """Scale-invariant: any input image → grid_size × grid_size × n_levels
    fixed activation pattern.

    `min_level_to_fire` skips firing the bottom level(s) — level 0 is
    'empty cell' which happens for all background pixels, so emitting it
    would just flood the substrate with background noise. Default 1 means
    only cells with non-zero intensity fire."""
    grid_size: int = 16
    n_levels: int = 4
    min_level_to_fire: int = 1

    def n_input_neurons(self) -> int:
        return self.grid_size * self.grid_size * self.n_levels

    def build_vocab(self, brain) -> ImageEncoderVocab:
        """Allocate input + class neurons in `brain`. Returns vocab."""
        vocab = ImageEncoderVocab()
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                for level in range(self.n_levels):
                    nid = brain.add_neuron(
                        lemma=f'cell:{r},{c},L{level}',
                        decay=0.5,
                    )
                    vocab.cell_to_id[(r, c, level)] = nid
                    vocab.id_to_cell[nid] = (r, c, level)
        for d in range(10):
            nid = brain.add_neuron(lemma=f'digit:{d}', decay=0.7)
            vocab.digit_to_id[d] = nid
            vocab.id_to_digit[nid] = d
        return vocab

    def encode(self, image: np.ndarray,
                vocab: ImageEncoderVocab) -> Dict[int, float]:
        """image (any size, float [0,1] or uint8) → seed dict.

        Steps:
          1. Normalize to float [0,1]
          2. Resize to grid_size × grid_size via PIL bilinear
          3. Bin intensity into n_levels (0 = empty, n_levels-1 = bright)
          4. Activate (row, col, level) neurons where level >= min_level_to_fire
        """
        # Normalize input to [0, 1] float32
        if image.dtype != np.float32 and image.dtype != np.float64:
            img = image.astype(np.float32) / 255.0
        else:
            img = image.astype(np.float32)
        img = np.clip(img, 0.0, 1.0)

        # Resize to internal grid
        if img.shape != (self.grid_size, self.grid_size):
            pil = Image.fromarray((img * 255).astype(np.uint8))
            pil = pil.resize((self.grid_size, self.grid_size), Image.BILINEAR)
            img = np.asarray(pil, dtype=np.float32) / 255.0

        # Bin into n_levels
        levels = (img * self.n_levels).astype(np.int32)
        levels = np.clip(levels, 0, self.n_levels - 1)

        # Activate neurons where level >= threshold
        seeds: Dict[int, float] = {}
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                level = int(levels[r, c])
                if level >= self.min_level_to_fire:
                    nid = vocab.cell_to_id[(r, c, level)]
                    seeds[nid] = 1.0
        return seeds
