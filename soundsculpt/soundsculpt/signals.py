"""Synthetic test signals and the high/low probe pair."""
from __future__ import annotations

import numpy as np


def _t(n: int, sr: int) -> np.ndarray:
    return np.arange(n) / sr


def impulse(n: int, sr: int, at_sec: float = 0.5, amp: float = 0.9
            ) -> np.ndarray:
    x = np.zeros(n)
    x[min(n - 1, int(round(at_sec * sr)))] = amp
    return x


def boundary_impulses(n: int, sr: int, margin: int = 4) -> np.ndarray:
    """Impulses at the very edges, to exercise extension/padding behaviour."""
    x = np.zeros(n)
    x[margin] = 0.9
    x[n - 1 - margin] = 0.9
    return x


def sine(n: int, sr: int, freq: float, amp: float = 0.5) -> np.ndarray:
    return amp * np.sin(2 * np.pi * freq * _t(n, sr))


def chirp(n: int, sr: int, f0: float = 200.0, f1: float = 8000.0,
          amp: float = 0.5) -> np.ndarray:
    """Linear frequency sweep f0 -> f1 across the whole buffer."""
    t = _t(n, sr)
    phase = 2 * np.pi * (f0 * t + 0.5 * (f1 - f0) / t[-1] * t ** 2)
    return amp * np.sin(phase)


def band_limited_noise(n: int, sr: int, f_lo: float, f_hi: float,
                       amp: float = 0.5, seed: int = 0) -> np.ndarray:
    """Noise whose energy is restricted (FFT-masked) to [f_lo, f_hi)."""
    rng = np.random.default_rng(seed)
    white = rng.standard_normal(n)
    spec = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    spec[(freqs < f_lo) | (freqs >= f_hi)] = 0.0
    x = np.fft.irfft(spec, n)
    scale = np.max(np.abs(x)) or 1.0
    return amp * x / scale


def probe_pair(n: int, sr: int, low_hz: float, high_hz: float,
               amp: float = 0.4) -> tuple[np.ndarray, float, float]:
    """Two steady tones used to measure realised gain after re-opening.

    The tones are summed so a single file exercises both bands at once.
    """
    x = sine(n, sr, low_hz, amp) + sine(n, sr, high_hz, amp)
    return x, low_hz, high_hz


def demo_mix(n: int, sr: int) -> np.ndarray:
    """Chirp + a narrow noise band + a mid impulse, for end-to-end demos."""
    x = 0.35 * chirp(n, sr, 150.0, 6000.0, amp=1.0)
    x += 0.30 * band_limited_noise(n, sr, 2400.0, 3000.0, amp=1.0, seed=3)
    x += impulse(n, sr, at_sec=0.25, amp=0.5)
    x += 0.2 * sine(n, sr, 330.0)
    return np.clip(x, -1.0, 1.0)
