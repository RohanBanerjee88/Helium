"""Private audio I/O helpers used across the metrics package."""
import wave
from pathlib import Path

import numpy as np


def load_wav(path: Path) -> tuple[np.ndarray, int]:
    """Load a WAV file as a float32 array normalised to [-1, 1].

    Returns (audio, sample_rate).  Multi-channel files are mixed to mono.
    Supports 8-, 16-, and 32-bit integer PCM (the only formats Python's
    stdlib wave module handles).
    """
    with wave.open(str(path)) as wf:
        sr = wf.getframerate()
        n_frames = wf.getnframes()
        n_ch = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        raw = wf.readframes(n_frames)

    if sampwidth == 1:
        samples = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        samples = samples / 128.0 - 1.0
    elif sampwidth == 2:
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
        samples /= 32768.0
    elif sampwidth == 4:
        samples = np.frombuffer(raw, dtype=np.int32).astype(np.float32)
        samples /= 2147483648.0
    else:
        raise ValueError(f"Unsupported sample width: {sampwidth} bytes in {path}")

    if n_ch > 1:
        samples = samples.reshape(-1, n_ch).mean(axis=1)

    return samples, sr


def write_wav(path: Path, audio: np.ndarray, sr: int) -> None:
    """Write a float32 mono array as a 16-bit PCM WAV file."""
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())
