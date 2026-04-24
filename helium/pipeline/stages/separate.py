"""
Stage: separate

Speech source separation using SepFormer (speechbrain).

Prerequisites:
  pip install speechbrain torchaudio

If speechbrain is not installed, the stage returns status="placeholder" so
the pipeline stays honest.  No other API keys are required.

How speaker assignment works
─────────────────────────────
SepFormer always produces exactly 2 separated streams.  On its own it has
no idea which stream corresponds to which speaker identity from diarization.
We recover that mapping by:

  1. Loading the exclusive speaker turns written by the diarization stage
     (<stem>_exclusive.json).  These are segments where only one speaker
     is active, so each turn is a clean reference window for that identity.

  2. Measuring the RMS energy of each SepFormer stream inside every
     exclusive turn.  The stream that "lights up" during SPEAKER_00's
     windows is SPEAKER_00's separated track.

  3. Running an optimal assignment (scipy.optimize.linear_sum_assignment if
     available, greedy otherwise) so the mapping is globally consistent.

If diarization artifacts are absent the stage still writes the raw streams as
speaker_0.wav / speaker_1.wav and marks diarization_guided=false in the
manifest.

Artifact layout per audio file (stem = filename without extension):
  separation/
    <stem>_speaker_0.wav    assigned separated track (speaker 0)
    <stem>_speaker_1.wav    assigned separated track (speaker 1)
    <stem>_raw_0.wav        raw SepFormer output stream 0 (before assignment)
    <stem>_raw_1.wav        raw SepFormer output stream 1
    manifest.json           per-file results including assignment details
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ...config import settings
from ...metrics._audio import write_wav


# ── public entry point ────────────────────────────────────────────────────────

def run(
    audio_dir: Path,
    artifacts_dir: Path,
    audio_files: List[str],
    target_speakers: int,
) -> Dict[str, Any]:
    sep_dir = artifacts_dir / "separation"
    sep_dir.mkdir(parents=True, exist_ok=True)

    # --- Load model (catches missing deps before any expensive work) ---
    load_result = _load_model(settings.separation_backend)
    if isinstance(load_result, dict):
        return _placeholder(sep_dir, artifacts_dir, audio_files, load_result["reason"])
    model, model_sr = load_result

    diar_dir = artifacts_dir / "diarization"
    all_artifacts: List[str] = []
    file_results: List[Dict[str, Any]] = []

    for audio_file in audio_files:
        stem = Path(audio_file).stem
        audio_path = audio_dir / audio_file
        try:
            result = _separate_file(
                model, model_sr, audio_path, sep_dir, artifacts_dir, stem, diar_dir
            )
            if target_speakers > 2:
                result["warning"] = (
                    f"SepFormer is designed for 2-speaker mixtures; "
                    f"target_speakers={target_speakers}.  Two streams were produced."
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
        "stage": "separate",
        "backend": settings.separation_backend,
        "target_speakers": target_speakers,
        "model_sample_rate": model_sr,
        "files": file_results,
    }
    manifest_path = sep_dir / "manifest.json"
    with open(manifest_path, "w") as fh:
        json.dump(manifest, fh, indent=2)
    all_artifacts.append(str(manifest_path.relative_to(artifacts_dir)))

    completed = [f for f in file_results if f.get("status") == "completed"]
    return {
        "files_separated": len(completed),
        "diarization_guided": any(f.get("diarization_guided", False) for f in completed),
        "artifacts": all_artifacts,
    }


# ── model loading ─────────────────────────────────────────────────────────────

def _load_model(model_id: str):
    """Load SepFormer.  Returns (model, sample_rate) or {"reason": str}."""
    try:
        import torch  # noqa: F401
    except ImportError:
        return {"reason": "torch is not installed.  Run: pip install torch torchaudio speechbrain"}

    # speechbrain 1.x moved inference classes; try both import paths
    SepformerClass = None
    for module_path, class_name in (
        ("speechbrain.inference.separation", "SepformerSeparation"),
        ("speechbrain.pretrained", "SepformerSeparation"),
    ):
        try:
            import importlib
            mod = importlib.import_module(module_path)
            SepformerClass = getattr(mod, class_name)
            break
        except (ImportError, AttributeError):
            continue

    if SepformerClass is None:
        return {
            "reason": (
                "speechbrain is not installed or its SepformerSeparation class "
                "was not found.  Run: pip install speechbrain"
            )
        }

    savedir = f"pretrained_models/{model_id.replace('/', '_')}"
    try:
        model = SepformerClass.from_hparams(source=model_id, savedir=savedir)
        model_sr: int = getattr(model.hparams, "sample_rate", 8000)
        return model, model_sr
    except Exception as exc:
        return {"reason": f"Failed to load {model_id!r}: {exc}"}


# ── per-file separation ───────────────────────────────────────────────────────

def _separate_file(
    model,
    model_sr: int,
    audio_path: Path,
    sep_dir: Path,
    artifacts_dir: Path,
    stem: str,
    diar_dir: Path,
) -> Dict[str, Any]:
    """Run SepFormer on one file and write all per-file artifacts."""
    est_sources = model.separate_file(path=str(audio_path))
    # Normalise to (time, n_speakers) regardless of batch dimension
    if est_sources.dim() == 3:          # (batch=1, time, n_speakers)
        est_sources = est_sources.squeeze(0)
    n_streams = est_sources.shape[-1]   # always 2 for SepFormer

    raw_streams: List[np.ndarray] = [
        est_sources[:, i].detach().cpu().numpy() for i in range(n_streams)
    ]

    # Write raw streams (before speaker assignment)
    raw_paths: List[str] = []
    for i, stream in enumerate(raw_streams):
        path = sep_dir / f"{stem}_raw_{i}.wav"
        _write_stream(path, stream, model_sr)
        raw_paths.append(str(path.relative_to(artifacts_dir)))

    # Diarization-guided speaker assignment
    assignment = _guided_assignment(raw_streams, diar_dir, stem, model_sr)
    diarization_guided = assignment is not None
    if assignment is None:
        # Fall back to identity mapping
        assignment = {f"speaker_{i}": i for i in range(n_streams)}

    # Write assigned speaker files
    assigned_paths: List[str] = []
    for spk_key, stream_idx in assignment.items():
        fname = spk_key.lower().replace(" ", "_")
        path = sep_dir / f"{stem}_{fname}.wav"
        _write_stream(path, raw_streams[stream_idx], model_sr)
        assigned_paths.append(str(path.relative_to(artifacts_dir)))

    return {
        "file": audio_path.name,
        "stem": stem,
        "status": "completed",
        "n_streams": n_streams,
        "sample_rate": model_sr,
        "speaker_assignment": assignment,
        "diarization_guided": diarization_guided,
        "artifacts": raw_paths + assigned_paths,
    }


def _write_stream(path: Path, audio: np.ndarray, sr: int) -> None:
    """Peak-normalise to −0.5 dBFS then write as 16-bit WAV."""
    peak = float(np.abs(audio).max())
    if peak > 1e-8:
        audio = audio / peak * 0.944  # −0.5 dBFS headroom
    write_wav(path, audio, sr)


# ── diarization-guided assignment ─────────────────────────────────────────────

def _guided_assignment(
    raw_streams: List[np.ndarray],
    diar_dir: Path,
    stem: str,
    model_sr: int,
) -> Optional[Dict[str, int]]:
    """Map speaker labels to stream indices via exclusive-turn energy matching.

    Algorithm:
      1. Load exclusive turns per speaker (no-overlap windows are cleanest).
      2. For each speaker, sum the squared energy of every raw stream over
         that speaker's exclusive windows → energy[speaker][stream].
      3. Solve the optimal assignment that maximises total matched energy.
         Falls back to greedy when scipy is not installed.

    Returns None if diarization artifacts are unavailable.
    """
    n_streams = len(raw_streams)
    turns_by_speaker = _load_diar_turns(diar_dir, stem)
    if not turns_by_speaker:
        return None

    # Limit to the top n_streams speakers by exclusive speech time
    spk_time = {
        spk: sum(e - s for s, e in windows)
        for spk, windows in turns_by_speaker.items()
    }
    top_spks = sorted(spk_time, key=spk_time.get, reverse=True)[:n_streams]
    if not top_spks:
        return None

    min_len = min(len(s) for s in raw_streams)
    energy = np.zeros((len(top_spks), n_streams), dtype=np.float64)

    for si, spk in enumerate(top_spks):
        for start, end in turns_by_speaker[spk]:
            sf = int(start * model_sr)
            ef = min(int(end * model_sr), min_len)
            if ef <= sf:
                continue
            for ti, stream in enumerate(raw_streams):
                chunk = stream[sf:ef]
                energy[si, ti] += float(np.dot(chunk, chunk))

    try:
        from scipy.optimize import linear_sum_assignment
        row_ind, col_ind = linear_sum_assignment(-energy)
        return {top_spks[row_ind[k]]: int(col_ind[k]) for k in range(len(row_ind))}
    except ImportError:
        used: set = set()
        result: Dict[str, int] = {}
        for si, spk in enumerate(top_spks):
            available = [j for j in range(n_streams) if j not in used]
            if not available:
                break
            best = max(available, key=lambda j: energy[si, j])
            result[spk] = best
            used.add(best)
        return result or None


def _load_diar_turns(
    diar_dir: Path, stem: str
) -> Optional[Dict[str, List[Tuple[float, float]]]]:
    """Load per-speaker turn intervals from diarization artifacts.

    Prefers exclusive turns (non-overlapping) for cleaner energy windows;
    falls back to all turns if the exclusive file was not written.
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


# ── placeholder ───────────────────────────────────────────────────────────────

def _placeholder(
    sep_dir: Path,
    artifacts_dir: Path,
    audio_files: List[str],
    reason: str,
) -> Dict[str, Any]:
    plan = {
        "stage": "separate",
        "status": "placeholder",
        "backend": settings.separation_backend,
        "reason": reason,
        "input_files": audio_files,
        "expected_outputs_per_file": [
            "<stem>_speaker_0.wav",
            "<stem>_speaker_1.wav",
            "<stem>_raw_0.wav",
            "<stem>_raw_1.wav",
        ],
    }
    plan_path = sep_dir / "separation_plan.json"
    with open(plan_path, "w") as fh:
        json.dump(plan, fh, indent=2)
    return {
        "status": "placeholder",
        "note": reason,
        "artifacts": [str(plan_path.relative_to(artifacts_dir))],
    }
