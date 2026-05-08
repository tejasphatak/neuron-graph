"""Scale-invariant audio encoder for the substrate.

Any 1D audio signal (any length, any sample rate) → fixed-size
substrate input. The substrate doesn't see "samples at 16 kHz" —
it sees "what frequency-band fired at each time-window."

Pipeline:
  1D signal → spectrogram (scipy.signal.spectrogram)
            → log-magnitude in dB
            → resize to fixed (time_steps × freq_bins) grid (PIL bilinear)
            → bin into n_levels per cell
            → emit one neuron per (time, freq, level) above min level

Total neurons: time_steps × freq_bins × n_levels
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
from PIL import Image
from scipy.signal import spectrogram


@dataclass
class AudioEncoderVocab:
    """Maps (time, freq, level) → neuron-id, plus class output neurons."""
    cell_to_id: Dict[Tuple[int, int, int], int] = field(default_factory=dict)
    id_to_cell: Dict[int, Tuple[int, int, int]] = field(default_factory=dict)
    class_to_id: Dict[int, int] = field(default_factory=dict)
    id_to_class: Dict[int, int] = field(default_factory=dict)


@dataclass
class AudioEncoder:
    time_steps: int = 16
    freq_bins: int = 16
    n_levels: int = 4
    min_level_to_fire: int = 1
    n_classes: int = 10
    # Spectrogram parameters
    nperseg: int = 256
    noverlap: int = 128

    def n_input_neurons(self) -> int:
        return self.time_steps * self.freq_bins * self.n_levels

    def build_vocab(self, brain) -> AudioEncoderVocab:
        vocab = AudioEncoderVocab()
        for t in range(self.time_steps):
            for f in range(self.freq_bins):
                for level in range(self.n_levels):
                    nid = brain.add_neuron(
                        lemma=f'audio:{t},{f},L{level}',
                        decay=0.5,
                    )
                    vocab.cell_to_id[(t, f, level)] = nid
                    vocab.id_to_cell[nid] = (t, f, level)
        for c in range(self.n_classes):
            nid = brain.add_neuron(lemma=f'audio_class:{c}', decay=0.7)
            vocab.class_to_id[c] = nid
            vocab.id_to_class[nid] = c
        return vocab

    def _signal_to_spectrogram_image(self, signal: np.ndarray,
                                       sample_rate: int) -> np.ndarray:
        """1D signal → 2D log-magnitude spectrogram (variable shape)."""
        sig = np.asarray(signal, dtype=np.float32).flatten()
        if len(sig) < self.nperseg:
            # Pad short signals
            sig = np.pad(sig, (0, self.nperseg - len(sig)))
        f, t, Sxx = spectrogram(sig, fs=sample_rate,
                                 nperseg=self.nperseg,
                                 noverlap=self.noverlap)
        # Log magnitude (clamp to avoid log(0))
        Sxx = np.maximum(Sxx, 1e-10)
        log_Sxx = np.log10(Sxx)
        # Normalize to [0, 1] (per-clip min-max)
        lo, hi = log_Sxx.min(), log_Sxx.max()
        if hi > lo:
            log_Sxx = (log_Sxx - lo) / (hi - lo)
        else:
            log_Sxx = np.zeros_like(log_Sxx)
        return log_Sxx  # shape: (n_freq, n_time)

    def encode(self, signal: np.ndarray, sample_rate: int,
                vocab: AudioEncoderVocab) -> Dict[int, float]:
        """signal (1D, any length) → seed dict."""
        spec = self._signal_to_spectrogram_image(signal, sample_rate)

        # Resize to fixed (freq_bins × time_steps) via PIL
        if spec.shape != (self.freq_bins, self.time_steps):
            pil = Image.fromarray((spec * 255).astype(np.uint8))
            pil = pil.resize((self.time_steps, self.freq_bins), Image.BILINEAR)
            spec = np.asarray(pil, dtype=np.float32) / 255.0

        # Bin into n_levels
        levels = (spec * self.n_levels).astype(np.int32)
        levels = np.clip(levels, 0, self.n_levels - 1)

        seeds: Dict[int, float] = {}
        for f in range(self.freq_bins):
            for t in range(self.time_steps):
                level = int(levels[f, t])
                if level >= self.min_level_to_fire:
                    nid = vocab.cell_to_id[(t, f, level)]
                    seeds[nid] = 1.0
        return seeds
