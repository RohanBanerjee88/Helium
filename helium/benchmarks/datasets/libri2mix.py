"""Libri2Mix benchmark dataset interface.

Expected directory layout (wav16k / min / test):

    <root>/
      wav16k/
        min/
          test/
            mix_clean/    ← mixture WAVs (name = <spk1_id>_<spk2_id>.wav)
            s1/           ← speaker-1 reference WAVs (same names)
            s2/           ← speaker-2 reference WAVs (same names)
          dev/
          train-100/
          train-360/
      wav8k/
        ...               ← same structure at 8 kHz

Dataset homepage: https://github.com/JorisCos/LibriMix
"""
from pathlib import Path
from typing import Iterator, Optional

from .base import BenchmarkDataset, BenchmarkSample

_SPLITS = {"train-100", "train-360", "dev", "test"}
_MIX_MODES = {"mix_clean", "mix_both", "mix_single"}
_SR_DIRS = {8000: "wav8k", 16000: "wav16k"}


class Libri2MixDataset(BenchmarkDataset):
    """Libri2Mix two-speaker speech separation benchmark."""

    def __init__(
        self,
        root: Path,
        sample_rate: int = 16000,
        length_mode: str = "min",
        mix_mode: str = "mix_clean",
    ) -> None:
        self._root = Path(root)
        self.sample_rate = sample_rate
        self.length_mode = length_mode
        self.mix_mode = mix_mode

        if sample_rate not in _SR_DIRS:
            raise ValueError(f"sample_rate must be one of {set(_SR_DIRS)}, got {sample_rate}")
        if mix_mode not in _MIX_MODES:
            raise ValueError(f"mix_mode must be one of {_MIX_MODES}, got {mix_mode!r}")

    @property
    def name(self) -> str:
        return "libri2mix"

    @property
    def root(self) -> Path:
        return self._root

    def _split_dir(self, split: str) -> Path:
        return self._root / _SR_DIRS[self.sample_rate] / self.length_mode / split

    def validate(self) -> bool:
        if not self._root.exists():
            return False
        return (self._split_dir("test") / self.mix_mode).exists()

    def iter_samples(
        self, split: str = "test", n: Optional[int] = None
    ) -> Iterator[BenchmarkSample]:
        if split not in _SPLITS:
            raise ValueError(f"split must be one of {_SPLITS}, got {split!r}")

        split_dir = self._split_dir(split)
        mix_dir = split_dir / self.mix_mode
        s1_dir = split_dir / "s1"
        s2_dir = split_dir / "s2"

        if not mix_dir.exists():
            raise FileNotFoundError(
                f"Libri2Mix directory not found: {mix_dir}\n"
                f"Download from https://github.com/JorisCos/LibriMix"
            )

        mix_files = sorted(mix_dir.glob("*.wav"))
        if n is not None:
            mix_files = mix_files[:n]

        for mix_path in mix_files:
            s1_path = s1_dir / mix_path.name
            s2_path = s2_dir / mix_path.name
            if not s1_path.exists() or not s2_path.exists():
                continue
            yield BenchmarkSample(
                sample_id=mix_path.stem,
                mixture=mix_path,
                sources=[s1_path, s2_path],
                sample_rate=self.sample_rate,
                metadata={"split": split, "mix_mode": self.mix_mode},
            )
