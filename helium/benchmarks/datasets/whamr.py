"""WHAMR! benchmark dataset interface.

WHAMR! is a noisy + reverberant extension of WSJ0-2mix.

Expected directory layout (wav8k / min / test):

    <root>/
      wav8k/
        min/
          tt/             ← test split
            mix_both/     ← noisy+reverberant mixture
            s1_anechoic/  ← clean anechoic source 1
            s2_anechoic/  ← clean anechoic source 2
          cv/             ← validation split
          tr/             ← training split
      wav16k/
        ...               ← same structure at 16 kHz

Dataset homepage: https://wham.whisper.ai
"""
from pathlib import Path
from typing import Iterator, Optional

from .base import BenchmarkDataset, BenchmarkSample

# WHAMR! uses short split codes internally
_SPLIT_MAP = {"train": "tr", "val": "cv", "test": "tt"}
_SR_DIRS = {8000: "wav8k", 16000: "wav16k"}
_MIX_MODES = {"mix_both", "mix_clean", "mix_single"}


class WHAMRDataset(BenchmarkDataset):
    """WHAMR! noisy+reverberant two-speaker separation benchmark."""

    def __init__(
        self,
        root: Path,
        sample_rate: int = 8000,
        length_mode: str = "min",
        mix_mode: str = "mix_both",
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
        return "whamr"

    @property
    def root(self) -> Path:
        return self._root

    def _split_dir(self, split: str) -> Path:
        short = _SPLIT_MAP.get(split, split)
        return self._root / _SR_DIRS[self.sample_rate] / self.length_mode / short

    def validate(self) -> bool:
        if not self._root.exists():
            return False
        return (self._split_dir("test") / self.mix_mode).exists()

    def iter_samples(
        self, split: str = "test", n: Optional[int] = None
    ) -> Iterator[BenchmarkSample]:
        split_dir = self._split_dir(split)
        mix_dir = split_dir / self.mix_mode
        s1_dir = split_dir / "s1_anechoic"
        s2_dir = split_dir / "s2_anechoic"

        if not mix_dir.exists():
            raise FileNotFoundError(
                f"WHAMR! directory not found: {mix_dir}\n"
                f"Download from https://wham.whisper.ai"
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
