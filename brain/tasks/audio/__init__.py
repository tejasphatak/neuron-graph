"""Audio classification on the substrate — modality polymorphism test #4.

Same pattern as the MNIST task: any-size input → fixed-grid encoder
→ substrate neurons → supervised reward training.

Audio encoding pipeline:
  1D signal (any length, any sample rate)
    → spectrogram (variable-time × fixed-freq-bins via scipy)
    → resize to fixed grid (time_steps × freq_bins)
    → bin intensity into n_levels
    → emit one neuron per (time, freq, level)

Total input neurons: time_steps × freq_bins × n_levels
  Default 16 × 16 × 4 = 1024 (matches ImageEncoder default)
"""

from .encoder import AudioEncoder, AudioEncoderVocab

__all__ = ['AudioEncoder', 'AudioEncoderVocab']
