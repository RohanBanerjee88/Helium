"""
Stage: extract

Extract per-speaker clean audio by splicing exclusive diarization turns
directly from the original mixture.

Why this beats separation for voice conversion
──────────────────────────────────────────────
SepFormer always produces exactly 2 streams.  With ≥3 speakers, each
stream is a blended group, not a clean isolation.  Even for 2 speakers,
SI-SDR tops out at ~16 dB in lab conditions — audibly cleaner but not
reference quality.

During *exclusive* turns (non-overlapping speech), the mixture literally
IS that speaker alone.  Splicing those windows gives near-perfect single-
speaker audio with no model, no latency, and no speaker-count limit.

That clean audio is the reference voice conversion needs.

Artifact layout per audio file (stem = filename without extension):
  extraction/
    <stem>_SPEAKER_00.wav      concatenated exclusive turns, SPEAKER_00
    <stem>_SPEAKER_01.wav      concatenated exclusive turns, SPEAKER_01
    ...one file per speaker...
    manifest.json              per-file results + total duration per speaker
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ...metrics._audio import load_wav, write_wav


# ── public entry point ────────────────────────────────────────────────────────

def run(
    audio_dir: Path,
    artifacts_dir: Path,
    audio_files: List[str],
    target_speakers: int,
) -> Dict[str, Any]:
    ext_dir = artifacts_dir / "extraction"
    ext_dir.mkdir(parents=True, exist_ok=True)

    diar_dir = artifacts_dir / "diarization"

    file_results: List[Dict[str, Any]] = []
    all_artifacts: List[str] = []

    for audio_file in audio_files:
        stem = Path(audio_file).stem
        audio_path = audio_dir / audio_file
        try:
            result = _extract_file(audio_path, diar_dir, ext_dir, artifacts_dir, stem)
        except Exception as exc:
            result = {
                "file": audio_file,
                "stem": stem,
                "status": "failed",
                "error": str(exc),
                "artifacts": [],
            }
        file_results.append(result)
        all_artifacts.extend(result.get("artifacts", []))

    manifest = {
        "stage": "extract",
        "files": file_results,
    }
    manifest_path = ext_dir / "manifest.json"
    with open(manifest_path, "w") as fh:
        json.dump(manifest, fh, indent=2)
    all_artifacts.append(str(manifest_path.relative_to(artifacts_dir)))

    completed = [f for f in file_results if f.get("status") == "completed"]
    total_speakers = sum(len(f.get("speakers", {})) for f in completed)

    return {
        "files_extracted": len(completed),
        "total_speakers": total_speakers,
        "artifacts": all_artifacts,
    }


# ── per-file extraction ───────────────────────────────────────────────────────

def _extract_file(
    audio_path: Path,
    diar_dir: Path,
    ext_dir: Path,
    artifacts_dir: Path,
    stem: str,
) -> Dict[str, Any]:
    turns_by_speaker = _load_exclusive_turns(diar_dir, stem)

    if not turns_by_speaker:
        return {
            "file": audio_path.name,
            "stem": stem,
            "status": "skipped",
            "reason": (
                f"no exclusive diarization turns found for '{stem}' — "
                "run the diarize stage first"
            ),
            "artifacts": [],
        }

    audio, sr = load_wav(audio_path)

    speaker_stats: Dict[str, Any] = {}
    artifacts: List[str] = []

    for speaker, windows in sorted(turns_by_speaker.items()):
        chunks = _splice_turns(audio, sr, windows)
        if chunks.size == 0:
            continue

        out_path = ext_dir / f"{stem}_{speaker}.wav"
        write_wav(out_path, chunks, sr)
        artifacts.append(str(out_path.relative_to(artifacts_dir)))

        speaker_stats[speaker] = {
            "duration_s": round(chunks.size / sr, 3),
            "turn_count": len(windows),
            "file": out_path.name,
        }

    if not speaker_stats:
        return {
            "file": audio_path.name,
            "stem": stem,
            "status": "failed",
            "reason": "exclusive turns loaded but all produced empty audio",
            "artifacts": [],
        }

    return {
        "file": audio_path.name,
        "stem": stem,
        "status": "completed",
        "sample_rate": sr,
        "speakers": speaker_stats,
        "artifacts": artifacts,
    }


def _splice_turns(
    audio: np.ndarray,
    sr: int,
    windows: List[Tuple[float, float]],
    fade_ms: int = 10,
) -> np.ndarray:
    """Splice and concatenate exclusive turn segments with short fade-in/out.

    Fading removes the click artifact that comes from hard-cutting a waveform
    mid-cycle.  10 ms is inaudible at normal playback speed.
    """
    fade_len = int(sr * fade_ms / 1000)
    chunks: List[np.ndarray] = []

    for start, end in windows:
        sf = int(start * sr)
        ef = min(int(end * sr), len(audio))
        if ef <= sf:
            continue
        chunk = audio[sf:ef].copy()
        if fade_len > 0 and len(chunk) >= 2 * fade_len:
            chunk[:fade_len] *= np.linspace(0.0, 1.0, fade_len, dtype=np.float32)
            chunk[-fade_len:] *= np.linspace(1.0, 0.0, fade_len, dtype=np.float32)
        chunks.append(chunk)

    return np.concatenate(chunks) if chunks else np.array([], dtype=np.float32)


# ── diarization loader ────────────────────────────────────────────────────────

def _load_exclusive_turns(
    diar_dir: Path,
    stem: str,
) -> Optional[Dict[str, List[Tuple[float, float]]]]:
    """Load per-speaker exclusive turn intervals from diarization artifacts.

    Only uses the exclusive turns file (non-overlapping speech).  Falls back
    to all turns if the exclusive file is absent, but this is less reliable
    for voice conversion since overlapping regions are noisier.
    """
    for fname in (f"{stem}_exclusive.json", f"{stem}.json"):
        path = diar_dir / fname
        if not path.exists():
            continue
        try:
            with open(path) as fh:
                data = json.load(fh)
            by_speaker: Dict[str, List[Tuple[float, float]]] = {}
            for turn in data.get("turns", []):
                spk = turn["speaker"]
                by_speaker.setdefault(spk, []).append(
                    (float(turn["start"]), float(turn["end"]))
                )
            return by_speaker or None
        except Exception:
            continue
    return None
