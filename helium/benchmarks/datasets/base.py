"""Abstract base class for all Helium benchmark datasets."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, List, Optional


@dataclass
class BenchmarkSample:
    """One evaluation sample: mixture + ground-truth reference sources."""

    sample_id: str
    mixture: Path
    sources: List[Path]       # reference clean sources, one per speaker
    sample_rate: int
    noise: Optional[Path] = None   # optional noise-only signal
    metadata: dict = field(default_factory=dict)


class BenchmarkDataset(ABC):
    """Contract every benchmark dataset must satisfy."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier used in output paths, e.g. 'libri2mix'."""

    @property
    @abstractmethod
    def root(self) -> Path:
        """Filesystem root of the dataset."""

    @abstractmethod
    def iter_samples(
        self, split: str = "test", n: Optional[int] = None
    ) -> Iterator[BenchmarkSample]:
        """Yield samples from the given split, optionally limited to n."""

    def validate(self) -> bool:
        """Return True if the dataset root exists and looks structurally valid."""
        return self.root.exists()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(root={self.root})"
