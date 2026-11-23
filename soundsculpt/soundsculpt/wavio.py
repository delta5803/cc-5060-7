"""Minimal mono PCM WAV I/O using only the standard library.

Signals inside the package are float64 numpy arrays, nominally in [-1, 1].
"""
from __future__ import annotations

import wave

import numpy as np


def read_wav(path: str) -> tuple[np.ndarray, int]:
    with wave.open(path, "rb") as wf:
        nch = wf.getnchannels()
        sw = wf.getsampwidth()
        sr = wf.getframerate()
        n = wf.getnframes()
        raw = wf.readframes(n)

    if sw == 2:
        data = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
    elif sw == 1:
        data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float64) - 128.0) / 128.0
    elif sw == 4:
        data = np.frombuffer(raw, dtype="<i4").astype(np.float64) / 2147483648.0
    elif sw == 3:
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        i32 = (b[:, 0].astype(np.int32)
               | (b[:, 1].astype(np.int32) << 8)
               | (b[:, 2].astype(np.int32) << 16))
        i32 = np.where(i32 & 0x800000, i32 - 0x1000000, i32)
        data = i32.astype(np.float64) / 8388608.0
    else:
        raise ValueError(f"unsupported sample width {sw} bytes")

    if nch > 1:
        data = data.reshape(-1, nch).mean(axis=1)
    return np.ascontiguousarray(data), sr


def write_wav(path: str, signal: np.ndarray, sr: int) -> None:
    x = np.asarray(signal, dtype=np.float64)
    x = np.clip(x, -1.0, 1.0)
    pcm = np.round(x * 32767.0).astype("<i2")
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())
