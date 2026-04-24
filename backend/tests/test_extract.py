"""Tests for helium/pipeline/stages/extract.py."""
import json
import wave
from pathlib import Path

import numpy as np
import pytest


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_wav(path: Path, samples: np.ndarray, sr: int = 8000) -> None:
    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


def _sine(freq: float, duration: float = 1.0, sr: int = 8000) -> np.ndarray:
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _write_excl_json(path: Path, turns: list) -> None:
    data = {"file": path.stem.replace("_exclusive", ""), "turns": turns}
    with open(path, "w") as fh:
        json.dump(data, fh)


# ── _load_exclusive_turns ─────────────────────────────────────────────────────

class TestLoadExclusiveTurns:
    def test_loads_exclusive_json(self, tmp_path):
        from helium.pipeline.stages.extract import _load_exclusive_turns

        excl = tmp_path / "clip_000_exclusive.json"
        _write_excl_json(excl, [
            {"speaker": "SPEAKER_00", "start": 0.0, "end": 1.0},
            {"speaker": "SPEAKER_01", "start": 2.0, "end": 3.0},
        ])

        result = _load_exclusive_turns(tmp_path, "clip_000")
        assert result is not None
        assert "SPEAKER_00" in result
        assert "SPEAKER_01" in result
        assert result["SPEAKER_00"] == [(0.0, 1.0)]

    def test_falls_back_to_all_turns(self, tmp_path):
        from helium.pipeline.stages.extract import _load_exclusive_turns

        path = tmp_path / "clip_000.json"
        _write_excl_json(path, [
            {"speaker": "SPEAKER_00", "start": 0.5, "end": 1.5},
        ])

        result = _load_exclusive_turns(tmp_path, "clip_000")
        assert result is not None
        assert "SPEAKER_00" in result

    def test_prefers_exclusive_over_all_turns(self, tmp_path):
        from helium.pipeline.stages.extract import _load_exclusive_turns

        excl = tmp_path / "clip_000_exclusive.json"
        _write_excl_json(excl, [
            {"speaker": "EXCLUSIVE_SPK", "start": 0.0, "end": 1.0},
        ])
        all_turns = tmp_path / "clip_000.json"
        _write_excl_json(all_turns, [
            {"speaker": "ALL_SPK", "start": 0.0, "end": 1.0},
        ])

        result = _load_exclusive_turns(tmp_path, "clip_000")
        assert "EXCLUSIVE_SPK" in result
        assert "ALL_SPK" not in result

    def test_returns_none_when_no_files(self, tmp_path):
        from helium.pipeline.stages.extract import _load_exclusive_turns
        assert _load_exclusive_turns(tmp_path, "missing") is None


# ── _splice_turns ─────────────────────────────────────────────────────────────

class TestSpliceTurns:
    def test_extracts_correct_samples(self, tmp_path):
        from helium.pipeline.stages.extract import _splice_turns

        sr = 8000
        audio = np.arange(sr * 4, dtype=np.float32) / (sr * 4)  # ramp 0→1

        # Take [1.0, 2.0) and [3.0, 4.0) → 2 * sr samples total
        result = _splice_turns(audio, sr, [(1.0, 2.0), (3.0, 4.0)])
        assert result.size == 2 * sr

    def test_empty_windows_return_empty(self):
        from helium.pipeline.stages.extract import _splice_turns

        audio = np.zeros(8000, dtype=np.float32)
        result = _splice_turns(audio, 8000, [])
        assert result.size == 0

    def test_out_of_bounds_window_clamped(self):
        from helium.pipeline.stages.extract import _splice_turns

        sr = 8000
        audio = np.ones(sr, dtype=np.float32)
        # Window extends past end of audio
        result = _splice_turns(audio, sr, [(0.5, 5.0)])
        assert result.size == sr // 2  # only the available samples

    def test_fade_applied_at_edges(self):
        from helium.pipeline.stages.extract import _splice_turns

        sr = 8000
        audio = np.ones(sr * 2, dtype=np.float32)
        fade_ms = 10
        fade_len = int(sr * fade_ms / 1000)  # 80 samples

        result = _splice_turns(audio, sr, [(0.0, 1.0)], fade_ms=fade_ms)
        # First sample of fade-in must be 0 (or very close)
        assert result[0] < 0.01
        # Last sample of fade-out must be 0 (or very close)
        assert result[-1] < 0.01
        # Middle should be ~1.0 (no fade applied there)
        mid = len(result) // 2
        assert result[mid] > 0.99


# ── run() integration ─────────────────────────────────────────────────────────

class TestExtractRun:
    def _make_diar_dir(self, tmp_path: Path) -> Path:
        diar_dir = tmp_path / "artifacts" / "diarization"
        diar_dir.mkdir(parents=True)
        _write_excl_json(diar_dir / "clip_000_exclusive.json", [
            {"speaker": "SPEAKER_00", "start": 0.0, "end": 0.5},
            {"speaker": "SPEAKER_01", "start": 0.6, "end": 1.0},
        ])
        return diar_dir

    def test_produces_per_speaker_wavs(self, tmp_path):
        from helium.pipeline.stages.extract import run

        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()
        _make_wav(audio_dir / "clip_000.wav", _sine(440.0, duration=1.0))

        artifacts_dir = tmp_path / "artifacts"
        artifacts_dir.mkdir(exist_ok=True)
        self._make_diar_dir(tmp_path)

        result = run(audio_dir, artifacts_dir, ["clip_000.wav"], target_speakers=2)

        assert result["files_extracted"] == 1
        assert result["total_speakers"] == 2

        ext_dir = artifacts_dir / "extraction"
        wavs = sorted(ext_dir.glob("*.wav"))
        assert len(wavs) == 2
        names = {w.name for w in wavs}
        assert "clip_000_SPEAKER_00.wav" in names
        assert "clip_000_SPEAKER_01.wav" in names

    def test_writes_manifest(self, tmp_path):
        from helium.pipeline.stages.extract import run

        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()
        _make_wav(audio_dir / "clip_000.wav", _sine(440.0, duration=1.0))

        artifacts_dir = tmp_path / "artifacts"
        artifacts_dir.mkdir(exist_ok=True)
        self._make_diar_dir(tmp_path)

        run(audio_dir, artifacts_dir, ["clip_000.wav"], target_speakers=2)

        manifest_path = artifacts_dir / "extraction" / "manifest.json"
        assert manifest_path.exists()
        manifest = json.loads(manifest_path.read_text())
        assert manifest["stage"] == "extract"
        files = manifest["files"]
        assert len(files) == 1
        assert files[0]["status"] == "completed"

    def test_skipped_when_no_diarization(self, tmp_path):
        from helium.pipeline.stages.extract import run

        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()
        _make_wav(audio_dir / "clip_000.wav", _sine(440.0, duration=1.0))

        artifacts_dir = tmp_path / "artifacts"
        artifacts_dir.mkdir(exist_ok=True)
        # No diarization dir at all

        result = run(audio_dir, artifacts_dir, ["clip_000.wav"], target_speakers=2)

        assert result["files_extracted"] == 0
        manifest = json.loads(
            (artifacts_dir / "extraction" / "manifest.json").read_text()
        )
        assert manifest["files"][0]["status"] == "skipped"

    def test_extracted_wav_has_correct_duration(self, tmp_path):
        from helium.pipeline.stages.extract import run

        sr = 8000
        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()
        _make_wav(audio_dir / "clip_000.wav", _sine(440.0, duration=2.0, sr=sr), sr=sr)

        artifacts_dir = tmp_path / "artifacts"
        artifacts_dir.mkdir(exist_ok=True)

        diar_dir = artifacts_dir / "diarization"
        diar_dir.mkdir()
        _write_excl_json(diar_dir / "clip_000_exclusive.json", [
            # SPEAKER_00 gets exactly 0.5s
            {"speaker": "SPEAKER_00", "start": 0.0, "end": 0.5},
        ])

        run(audio_dir, artifacts_dir, ["clip_000.wav"], target_speakers=2)

        out_wav = artifacts_dir / "extraction" / "clip_000_SPEAKER_00.wav"
        assert out_wav.exists()
        with wave.open(str(out_wav)) as wf:
            actual_frames = wf.getnframes()
            actual_sr = wf.getframerate()

        expected_frames = int(0.5 * sr)
        # Allow ±1 frame for rounding
        assert abs(actual_frames - expected_frames) <= 1
