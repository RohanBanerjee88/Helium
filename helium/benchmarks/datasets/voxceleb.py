"""VoxCeleb dataset interface for speaker similarity evaluation.

VoxCeleb is a speaker recognition dataset, not a separation benchmark.
It is used here to build positive/negative speaker pairs for evaluating
voice-conversion identity preservation.

Expected directory layout:

    <root>/
      wav/
        <speaker_id>/
          <session_id>/
            <utterance_id>.wav

Dataset homepage: https://www.robots.ox.ac.uk/~vgg/data/voxceleb/
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional


@dataclass
class VoxCelebUtterance:
    path: Path
    speaker_id: str
    session_id: str
    utterance_id: str


@dataclass
class VoxCelebSpeakerPair:
    """A pair of utterances used for speaker similarity scoring."""

    utterance_a: VoxCelebUtterance
    utterance_b: VoxCelebUtterance
    same_speaker: bool


class VoxCelebDataset:
    """VoxCeleb speaker utterance dataset."""

    def __init__(self, root: Path) -> None:
        self._root = Path(root)

    @property
    def name(self) -> str:
        return "voxceleb"

    @property
    def root(self) -> Path:
        return self._root

    def validate(self) -> bool:
        wav_dir = self._root / "wav"
        if not wav_dir.exists():
            return False
        return any(wav_dir.iterdir())

    def list_speakers(self) -> List[str]:
        wav_dir = self._root / "wav"
        if not wav_dir.exists():
            return []
        return sorted(d.name for d in wav_dir.iterdir() if d.is_dir())

    def iter_utterances(
        self, speaker_id: Optional[str] = None
    ) -> Iterator[VoxCelebUtterance]:
        """Yield utterances, optionally filtered to a single speaker."""
        wav_dir = self._root / "wav"
        if not wav_dir.exists():
            raise FileNotFoundError(
                f"VoxCeleb wav/ directory not found: {wav_dir}\n"
                f"Download from https://www.robots.ox.ac.uk/~vgg/data/voxceleb/"
            )

        spk_dirs = (
            [wav_dir / speaker_id]
            if speaker_id
            else sorted(d for d in wav_dir.iterdir() if d.is_dir())
        )

        for spk_dir in spk_dirs:
            if not spk_dir.is_dir():
                continue
            for sess_dir in sorted(d for d in spk_dir.iterdir() if d.is_dir()):
                for utt_path in sorted(sess_dir.glob("*.wav")):
                    yield VoxCelebUtterance(
                        path=utt_path,
                        speaker_id=spk_dir.name,
                        session_id=sess_dir.name,
                        utterance_id=utt_path.stem,
                    )

    def iter_speaker_pairs(
        self,
        n_positive: int = 100,
        n_negative: int = 100,
        utterances_per_speaker: int = 2,
    ) -> Iterator[VoxCelebSpeakerPair]:
        """Yield positive (same) and negative (different) speaker pairs."""
        speakers = self.list_speakers()
        if len(speakers) < 2:
            return

        # Collect a few utterances per speaker
        spk_utts: Dict[str, List[VoxCelebUtterance]] = {}
        for spk in speakers:
            utts = list(self.iter_utterances(spk))
            if len(utts) >= 2:
                spk_utts[spk] = utts[:utterances_per_speaker]

        if len(spk_utts) < 2:
            return

        spk_list = list(spk_utts.keys())

        # Positive pairs: two utterances from the same speaker
        pos_count = 0
        for spk in spk_list:
            if pos_count >= n_positive:
                break
            utts = spk_utts[spk]
            yield VoxCelebSpeakerPair(
                utterance_a=utts[0], utterance_b=utts[1], same_speaker=True
            )
            pos_count += 1

        # Negative pairs: utterances from different speakers
        neg_count = 0
        for i in range(len(spk_list)):
            for j in range(i + 1, len(spk_list)):
                if neg_count >= n_negative:
                    return
                yield VoxCelebSpeakerPair(
                    utterance_a=spk_utts[spk_list[i]][0],
                    utterance_b=spk_utts[spk_list[j]][0],
                    same_speaker=False,
                )
                neg_count += 1
