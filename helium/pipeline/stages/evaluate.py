"""Evaluate stage: load prior-stage manifests and compute available metrics.

Metric availability depends on which upstream stages have run (not just
scaffolded as placeholders).  Each sub-section reports its own status so
the result is always honest about what was actually computed.

Outputs written to artifacts/evaluation/
  metrics.json           — full metric values + availability notes
  experiment_summary.json — compact top-level view, suitable for quick diffing
  metrics.csv            — append-mode row for aggregating across experiments
"""
import csv
import json
import wave
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ...config import settings


# ── audio helpers ────────────────────────────────────────────────────────────

def _load_wav_f32(path: Path) -> tuple[np.ndarray, int]:
    """Load a WAV file as a float32 array in [-1, 1].  Mixes to mono."""
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
        raise ValueError(f"Unsupported sample width {sampwidth} in {path}")

    if n_ch > 1:
        samples = samples.reshape(-1, n_ch).mean(axis=1)
    return samples, sr


def _load_json(path: Path) -> Optional[Dict]:
    if path.exists():
        with open(path) as fh:
            return json.load(fh)
    return None


# ── per-module metric helpers ─────────────────────────────────────────────────

def _separation_metrics(artifacts_dir: Path) -> Dict[str, Any]:
    """Compute separation metrics from artifacts/separation/ if available."""
    sep_dir = artifacts_dir / "separation"
    unavail: Dict[str, Any] = {"status": "unavailable", "reason": "", "values": {}}

    if not sep_dir.exists():
        return {**unavail, "reason": "separation stage has not run"}

    plan = _load_json(sep_dir / "separation_plan.json")
    if plan and plan.get("status") == "placeholder":
        return {**unavail, "reason": "separation stage ran in placeholder mode — no audio outputs yet"}

    spk_files = sorted(sep_dir.glob("speaker_*.wav"))
    if len(spk_files) < 2:
        return {
            **unavail,
            "reason": (
                f"expected ≥2 speaker_*.wav files in {sep_dir.relative_to(artifacts_dir)}, "
                f"found {len(spk_files)}"
            ),
        }

    try:
        src_a, _ = _load_wav_f32(spk_files[0])
        src_b, _ = _load_wav_f32(spk_files[1])
    except Exception as exc:
        return {**unavail, "reason": f"could not load separation outputs: {exc}"}

    from ...metrics.separation import blind_coherence, pit_si_sdr, si_sdri

    values: Dict[str, Any] = {
        "blind_source_coherence": round(blind_coherence(src_a, src_b), 4),
    }

    # If reference sources and mixture are present, compute SI-SDR and SI-SDRi
    ref_files = sorted(sep_dir.glob("ref_speaker_*.wav"))
    mix_file = sep_dir / "mixture.wav"

    if len(ref_files) >= 2 and mix_file.exists():
        try:
            ref_a, _ = _load_wav_f32(ref_files[0])
            ref_b, _ = _load_wav_f32(ref_files[1])
            mix, _ = _load_wav_f32(mix_file)

            min_len = min(len(src_a), len(src_b), len(ref_a), len(ref_b), len(mix))
            sa, sb = src_a[:min_len], src_b[:min_len]
            ra, rb = ref_a[:min_len], ref_b[:min_len]
            m = mix[:min_len]

            pi_si_sdr, best_perm = pit_si_sdr([ra, rb], [sa, sb])
            values["si_sdr_db"] = round(pi_si_sdr, 2)

            ests_ordered = [sa, sb]
            si_sdri_vals = [
                si_sdri([ra, rb][i], ests_ordered[best_perm[i]], m) for i in range(2)
            ]
            values["si_sdri_db"] = round(sum(si_sdri_vals) / 2, 2)
        except Exception as exc:
            values["reference_metric_error"] = str(exc)
    else:
        values["note"] = (
            "SI-SDR and SI-SDRi require reference sources (ref_speaker_*.wav) "
            "and mixture.wav in the separation directory — available in benchmark runs."
        )

    return {"status": "computed", "values": values}


def _diarization_metrics(artifacts_dir: Path) -> Dict[str, Any]:
    """Report diarization outputs; compute DER if a reference RTTM exists."""
    diar_dir = artifacts_dir / "diarization"
    unavail: Dict[str, Any] = {"status": "unavailable", "reason": "", "values": {}}

    if not diar_dir.exists():
        return {**unavail, "reason": "diarization stage has not run"}

    plan = _load_json(diar_dir / "diarization_plan.json")
    if plan and plan.get("status") == "placeholder":
        return {**unavail, "reason": "diarization stage ran in placeholder mode — no RTTM outputs yet"}

    rttm_path = diar_dir / "speaker_turns.rttm"
    if not rttm_path.exists():
        return {**unavail, "reason": "speaker_turns.rttm not found in diarization/"}

    segments: List[Dict] = []
    with open(rttm_path) as fh:
        for line in fh:
            parts = line.strip().split()
            if parts and parts[0] == "SPEAKER" and len(parts) >= 8:
                segments.append({
                    "onset": float(parts[3]),
                    "duration": float(parts[4]),
                    "speaker": parts[7],
                })

    speakers = {s["speaker"] for s in segments}
    total_speech = sum(s["duration"] for s in segments)

    values: Dict[str, Any] = {
        "segment_count": len(segments),
        "speaker_count": len(speakers),
        "total_speech_duration_s": round(total_speech, 2),
    }

    ref_rttm = diar_dir / "reference_speaker_turns.rttm"
    if ref_rttm.exists():
        try:
            from ...metrics.diarization import der as compute_der
            der_vals = compute_der(str(ref_rttm), str(rttm_path))
            values.update({f"DER_{k}": v for k, v in der_vals.items()})
        except Exception as exc:
            values["der_error"] = str(exc)
    else:
        values["DER"] = "reference RTTM not available (needed for DER computation)"

    return {"status": "computed", "values": values}


def _speaker_sim_metrics(artifacts_dir: Path) -> Dict[str, Any]:
    """Report speaker-similarity availability from conversion outputs."""
    conv_dir = artifacts_dir / "conversion"
    unavail: Dict[str, Any] = {"status": "unavailable", "reason": "", "values": {}}

    if not conv_dir.exists():
        return {**unavail, "reason": "conversion stage has not run"}

    plan = _load_json(conv_dir / "conversion_plan.json")
    if plan and plan.get("status") == "placeholder":
        return {
            **unavail,
            "reason": "conversion stage ran in placeholder mode — no converted audio yet",
        }

    return {
        **unavail,
        "reason": (
            "speaker similarity requires a speaker embedding model "
            "(resemblyzer or speechbrain/spkrec-ecapa-voxceleb); "
            "install helium[research] and wire the backend in helium/metrics/speaker_sim.py"
        ),
    }


# ── CSV helper ────────────────────────────────────────────────────────────────

def _append_csv_row(csv_path: Path, row: Dict[str, Any]) -> None:
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# ── public stage entry point ──────────────────────────────────────────────────

def run(artifacts_dir: Path, audio_files: List[str], target_speakers: int) -> Dict[str, Any]:
    evaluation_dir = artifacts_dir / "evaluation"
    evaluation_dir.mkdir(parents=True, exist_ok=True)

    val_manifest = _load_json(artifacts_dir / "manifests" / "validation_report.json")

    sep = _separation_metrics(artifacts_dir)
    diar = _diarization_metrics(artifacts_dir)
    spk_sim = _speaker_sim_metrics(artifacts_dir)

    content_pres = {
        "status": "unavailable",
        "reason": (
            "WER requires a speech recognition backend "
            "(faster-whisper or openai-whisper); "
            "wire it in helium/metrics/content.py"
        ),
    }

    metrics_doc = {
        "stage": "evaluate",
        "target_speakers": target_speakers,
        "input_files": audio_files,
        "baselines": {
            "diarization": settings.diarization_backend,
            "separation": settings.separation_backend,
            "conversion": settings.conversion_backend,
        },
        "separation": sep,
        "diarization": diar,
        "speaker_similarity": spk_sim,
        "content_preservation": content_pres,
        "research_question": (
            "Can diarization-guided source separation plus style-preserving voice conversion "
            "outperform plain separation-plus-conversion baselines in noisy two-speaker mixtures?"
        ),
    }

    metrics_path = evaluation_dir / "metrics.json"
    with open(metrics_path, "w") as fh:
        json.dump(metrics_doc, fh, indent=2)

    # Compact summary for quick diffing across experiments
    computed_groups = [
        k for k, v in {
            "separation": sep,
            "diarization": diar,
            "speaker_similarity": spk_sim,
        }.items()
        if v.get("status") == "computed"
    ]
    unavailable_groups = [
        k for k, v in {
            "separation": sep,
            "diarization": diar,
            "speaker_similarity": spk_sim,
            "content_preservation": content_pres,
        }.items()
        if v.get("status") != "computed"
    ]

    summary = {
        "artifacts_dir": str(artifacts_dir),
        "target_speakers": target_speakers,
        "input_clip_count": len(audio_files),
        "validation_clip_count": (
            len(val_manifest.get("clips", [])) if val_manifest else None
        ),
        "metrics_computed": computed_groups,
        "metrics_unavailable": unavailable_groups,
        "separation_values": sep.get("values", {}),
        "diarization_values": diar.get("values", {}),
    }

    summary_path = evaluation_dir / "experiment_summary.json"
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)

    # CSV row — append so multiple runs accumulate in the same file
    csv_row: Dict[str, Any] = {
        "artifacts_dir": str(artifacts_dir),
        "target_speakers": target_speakers,
        "input_clips": len(audio_files),
        "si_sdr_db": sep.get("values", {}).get("si_sdr_db", ""),
        "si_sdri_db": sep.get("values", {}).get("si_sdri_db", ""),
        "blind_source_coherence": sep.get("values", {}).get("blind_source_coherence", ""),
        "DER_der": diar.get("values", {}).get("DER_der", ""),
        "segment_count": diar.get("values", {}).get("segment_count", ""),
        "speaker_count": diar.get("values", {}).get("speaker_count", ""),
    }
    csv_path = evaluation_dir / "metrics.csv"
    _append_csv_row(csv_path, csv_row)

    artifacts = [
        str(metrics_path.relative_to(artifacts_dir)),
        str(summary_path.relative_to(artifacts_dir)),
        str(csv_path.relative_to(artifacts_dir)),
    ]

    return {
        "metrics_computed": len(computed_groups),
        "metrics_planned": ["SI-SDR", "SI-SDRi", "DER", "speaker_similarity", "WER"],
        "dataset_count": 3,
        "artifacts": artifacts,
    }
