"""
Stage: diarize

Runs pyannote.audio speaker diarization on each audio file in the job.
Writes per-file artifacts so downstream stages (separate, evaluate) can
consume whichever files they need independently.

Prerequisites (all three required):
  1. pip install pyannote.audio
  2. Accept model conditions at
     https://hf.co/pyannote/speaker-diarization-community-1
  3. Set HELIUM_HF_TOKEN=hf_your_token in .env or environment

If any prerequisite is missing the stage returns status="placeholder" so
the pipeline remains honest about what actually ran.

Artifact layout per audio file (stem = filename without extension):
  diarization/
    <stem>.rttm              standard RTTM (consumed by evaluate DER)
    <stem>.json              turn list with start/end/duration/speaker
    <stem>_exclusive.json    turns with no speaker overlap (optional;
                             written when pyannote exposes overlap info)
    manifest.json            aggregate summary across all processed files
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from ...config import settings


# ── public entry point ────────────────────────────────────────────────────────

def run(
    audio_dir: Path,
    artifacts_dir: Path,
    audio_files: List[str],
    target_speakers: int,
) -> Dict[str, Any]:
    diar_dir = artifacts_dir / "diarization"
    diar_dir.mkdir(parents=True, exist_ok=True)

    # --- Prerequisite checks (fast, before any model download) ---
    if not settings.hf_token:
        return _placeholder(
            diar_dir, artifacts_dir, audio_files,
            reason=(
                "HELIUM_HF_TOKEN is not set. "
                "Create a token at https://hf.co/settings/tokens (read scope), "
                "accept model conditions at "
                "https://hf.co/pyannote/speaker-diarization-community-1, "
                "then set HELIUM_HF_TOKEN=hf_your_token in .env."
            ),
        )

    try:
        from pyannote.audio import Pipeline as _PyannotePipeline  # noqa: F401
    except ImportError:
        return _placeholder(
            diar_dir, artifacts_dir, audio_files,
            reason="pyannote.audio is not installed.  Run: pip install pyannote.audio",
        )

    # --- Load pipeline (downloads checkpoint on first run, then cached) ---
    try:
        pipeline = _load_pipeline(settings.diarization_backend, settings.hf_token)
    except Exception as exc:
        return _placeholder(
            diar_dir, artifacts_dir, audio_files,
            reason=f"Failed to load {settings.diarization_backend!r}: {exc}",
        )

    # --- Process each file ---
    all_artifacts: List[str] = []
    file_results: List[Dict[str, Any]] = []

    for audio_file in audio_files:
        audio_path = audio_dir / audio_file
        stem = Path(audio_file).stem
        try:
            result = _diarize_file(
                pipeline, audio_path, diar_dir, artifacts_dir, stem, target_speakers
            )
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

    # --- Write aggregate manifest ---
    manifest = {
        "stage": "diarize",
        "backend": settings.diarization_backend,
        "target_speakers": target_speakers,
        "files": file_results,
    }
    manifest_path = diar_dir / "manifest.json"
    with open(manifest_path, "w") as fh:
        json.dump(manifest, fh, indent=2)
    all_artifacts.append(str(manifest_path.relative_to(artifacts_dir)))

    completed = [f for f in file_results if f.get("status") == "completed"]
    return {
        "files_diarized": len(completed),
        "total_segments": sum(f.get("segment_count", 0) for f in completed),
        "total_speakers_found": (
            max((f.get("speaker_count", 0) for f in completed), default=0)
        ),
        "artifacts": all_artifacts,
    }


# ── per-file logic ────────────────────────────────────────────────────────────

def _diarize_file(
    pipeline,
    audio_path: Path,
    diar_dir: Path,
    artifacts_dir: Path,
    stem: str,
    target_speakers: int,
) -> Dict[str, Any]:
    """Run diarization on one file and write all per-file artifacts."""

    # Run pyannote — pass num_speakers when we have a firm expectation
    run_kwargs: Dict[str, Any] = {}
    if target_speakers > 0:
        run_kwargs["num_speakers"] = target_speakers

    output = pipeline(str(audio_path), **run_kwargs)

    # pyannote.audio 4.x returns a DiarizeOutput dataclass;
    # older versions returned an Annotation directly.
    if hasattr(output, "speaker_diarization"):
        diarization = output.speaker_diarization
        exclusive_annotation = output.exclusive_speaker_diarization
    else:
        diarization = output
        exclusive_annotation = None

    # --- Collect all turns ---
    turns = [
        {
            "start": round(turn.start, 3),
            "end": round(turn.end, 3),
            "duration": round(turn.duration, 3),
            "speaker": speaker,
        }
        for turn, _, speaker in diarization.itertracks(yield_label=True)
    ]
    speakers = sorted({t["speaker"] for t in turns})

    # --- RTTM ---
    rttm_path = diar_dir / f"{stem}.rttm"
    with open(rttm_path, "w") as fh:
        diarization.write_rttm(fh)

    # --- JSON turns ---
    json_path = diar_dir / f"{stem}.json"
    with open(json_path, "w") as fh:
        json.dump(
            {
                "file": stem,
                "speaker_count": len(speakers),
                "speakers": speakers,
                "turns": turns,
            },
            fh,
            indent=2,
        )

    artifacts = [
        str(rttm_path.relative_to(artifacts_dir)),
        str(json_path.relative_to(artifacts_dir)),
    ]

    # --- Exclusive turns (non-overlapping segments) ---
    exclusive = _exclusive_turns(diarization, turns, exclusive_annotation)
    if exclusive is not None:
        excl_path = diar_dir / f"{stem}_exclusive.json"
        with open(excl_path, "w") as fh:
            json.dump(
                {
                    "file": stem,
                    "note": "segments where only one speaker is active",
                    "turns": exclusive,
                },
                fh,
                indent=2,
            )
        artifacts.append(str(excl_path.relative_to(artifacts_dir)))

    total_speech = round(sum(t["duration"] for t in turns), 3)
    return {
        "file": audio_path.name,
        "stem": stem,
        "status": "completed",
        "segment_count": len(turns),
        "speaker_count": len(speakers),
        "speakers": speakers,
        "total_speech_duration_s": total_speech,
        "artifacts": artifacts,
    }


def _exclusive_turns(
    diarization, turns: List[Dict], exclusive_annotation=None
) -> Optional[List[Dict]]:
    """Return turns with no speaker overlap, or None if not computable."""
    # pyannote 4.x provides exclusive_speaker_diarization directly
    if exclusive_annotation is not None:
        try:
            return [
                {
                    "start": round(turn.start, 3),
                    "end": round(turn.end, 3),
                    "duration": round(turn.duration, 3),
                    "speaker": speaker,
                }
                for turn, _, speaker in exclusive_annotation.itertracks(yield_label=True)
            ]
        except Exception:
            pass

    # Fallback for older pyannote versions: compute via get_overlap()
    try:
        overlap_timeline = diarization.get_overlap()
    except AttributeError:
        return None

    if not overlap_timeline:
        return turns

    overlap_intervals = [(seg.start, seg.end) for seg in overlap_timeline]

    def _has_overlap(start: float, end: float) -> bool:
        return any(ov_s < end and ov_e > start for ov_s, ov_e in overlap_intervals)

    return [t for t in turns if not _has_overlap(t["start"], t["end"])]


# ── pipeline loader ───────────────────────────────────────────────────────────

def _load_pipeline(model_id: str, hf_token: str):
    """Load a pyannote.audio Pipeline, handling auth-token API differences."""
    from pyannote.audio import Pipeline

    # pyannote.audio 3.x uses use_auth_token; newer huggingface_hub uses token.
    # Try both to stay compatible across versions.
    try:
        return Pipeline.from_pretrained(model_id, use_auth_token=hf_token)
    except TypeError:
        return Pipeline.from_pretrained(model_id, token=hf_token)


# ── placeholder helpers ───────────────────────────────────────────────────────

def _placeholder(
    diar_dir: Path,
    artifacts_dir: Path,
    audio_files: List[str],
    reason: str,
) -> Dict[str, Any]:
    """Write an honest placeholder plan and return the placeholder status dict."""
    plan = {
        "stage": "diarize",
        "status": "placeholder",
        "backend": settings.diarization_backend,
        "reason": reason,
        "input_files": audio_files,
        "expected_outputs_per_file": [
            "<stem>.rttm",
            "<stem>.json",
            "<stem>_exclusive.json",
        ],
        "expected_manifest": "manifest.json",
    }
    plan_path = diar_dir / "diarization_plan.json"
    with open(plan_path, "w") as fh:
        json.dump(plan, fh, indent=2)

    return {
        "status": "placeholder",
        "note": reason,
        "artifacts": [str(plan_path.relative_to(artifacts_dir))],
    }
