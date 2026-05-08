"""Video classification on the substrate — modality polymorphism test #5.

Same pattern: any-size input → fixed-grid encoder → substrate.

Video encoding pipeline:
  T × H × W array (any T, any H, any W)
    → sample T_grid frames uniformly across video
    → each frame → grayscale → resize to grid_size × grid_size
    → bin intensity into n_levels per pixel
    → emit one neuron per (frame_idx, row, col, level)

Total input neurons: T_grid × grid_size² × n_levels
  Default 8 × 8 × 8 × 2 = 1024 (matches Image/Audio encoders)

This adds the time dimension — substrate sees temporal patterns
not just static frames.
"""

from .encoder import VideoEncoder, VideoEncoderVocab

__all__ = ['VideoEncoder', 'VideoEncoderVocab']
