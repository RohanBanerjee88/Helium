"""Tests for helium/pipeline/stages/separate.py.

speechbrain is not required — we test:
  - placeholder path (no speechbrain installed)
  - _load_diar_turns  (pure JSON parsing)
  - _guided_assignment (pure numpy energy matching)
  - _write_stream via the audio helper
  - evaluate stage reading the new separation artifacts
"""
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


def _sine(freq: float, duration: float = 1.0, sr: int = 8000, amp: float = 0.5) -> np.ndarray:
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _write_diar_json(path: Path, turns: list) -> None:
    data = {"file": path.stem.replace("_exclusive", ""), "turns": turns}
    with open(path, "w") as fh:
        json.dump(data, fh)


# ── placeholder path ──────────────────────────────────────────────────────────

class TestSeparatePlaceholder:
    def test_no_speechbrain_returns_placeholder(self, tmp_path):
        """When speechbrain is absent the stage must return placeholder, not raise."""
        from helium.pipeline.stages.separate import run

        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()
        _make_wav(audio_dir / "clip_000.wav", _sine(440.0))

        artifacts_dir = tmp_path / "artifacts"
        artifacts_dir.mkdir()

        result = run(audio_dir, artifacts_dir, ["clip_000.wav"], target_speakers=2)

        assert result["status"] == "placeholder"
        assert "artifacts" in result
        assert len(result["artifacts"]) >= 1

        # Plan file must be honest about reason
        plan_path = artifacts_dir / "separation" / "separation_plan.json"
        assert plan_path.exists()
        plan = json.loads(plan_path.read_text())
        assert plan["status"] == "placeholder"
        assert "reason" in plan
        assert plan["reason"]  # non-empty string

    def test_placeholder_does_not_write_wav(self, tmp_path):
        from helium.pipeline.stages.separate import run

        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()
        _make_wav(audio_dir / "clip_000.wav", _sine(440.0))
        artifacts_dir = tmp_path / "artifacts"
        artifacts_dir.mkdir()

        run(audio_dir, artifacts_dir, ["clip_000.wav"], target_speakers=2)

        sep_dir = artifacts_dir / "separation"
        wav_files = list(sep_dir.glob("*.wav"))
        assert len(wav_files) == 0, "placeholder must not write any audio"


# ── diarization turn loading ──────────────────────────────────────────────────

class TestLoadDiarTurns:
    def test_loads_exclusive_json_first(self, tmp_path):
        from helium.pipeline.stages.separate import _load_diar_turns

        excl = tmp_path / "clip_000_exclusive.json"
        _write_diar_json(excl, [
            {"speaker": "SPEAKER_00", "start": 0.0, "end": 2.0, "duration": 2.0},
            {"speaker": "SPEAKER_01", "start": 3.0, "end": 5.0, "duration": 2.0},
        ])
        # Also write all-turns JSON — should not be picked if exclusive exists
        all_turns = tmp_path / "clip_000.json"
        _write_diar_json(all_turns, [
            {"speaker": "DIFFERENT", "start": 0.0, "end": 1.0, "duration": 1.0},
        ])

        result = _load_diar_turns(tmp_path, "clip_000")
        assert result is not None
        assert "SPEAKER_00" in result
        assert "DIFFERENT" not in result  # exclusive JSON was preferred

    def test_falls_back_to_all_turns(self, tmp_path):
        from helium.pipeline.stages.separate import _load_diar_turns

        all_turns = tmp_path / "clip_000.json"
        _write_diar_json(all_turns, [
            {"speaker": "SPEAKER_00", "start": 0.0, "end": 1.5, "duration": 1.5},
        ])

        result = _load_diar_turns(tmp_path, "clip_000")
        assert result is not None
        assert "SPEAKER_00" in result

    def test_returns_none_when_no_files(self, tmp_path):
        from helium.pipeline.stages.separate import _load_diar_turns
        assert _load_diar_turns(tmp_path, "nonexistent") is None

    def test_groups_turns_by_speaker(self, tmp_path):
        from helium.pipeline.stages.separate import _load_diar_turns

        path = tmp_path / "clip_000.json"
        _write_diar_json(path, [
            {"speaker": "A", "start": 0.0, "end": 1.0, "duration": 1.0},
            {"speaker": "A", "start": 2.0, "end": 3.0, "duration": 1.0},
            {"speaker": "B", "start": 1.0, "end": 2.0, "duration": 1.0},
        ])

        result = _load_diar_turns(tmp_path, "clip_000")
        assert result is not None
        assert len(result["A"]) == 2
        assert len(result["B"]) == 1


# ── guided assignment ─────────────────────────────────────────────────────────

class TestGuidedAssignment:
    def _make_streams_and_turns(self, sr: int = 8000):
        """Create two clean sine waves + exclusive turns that should unmix them."""
        duration = 4.0
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        # stream_0 is mainly 440 Hz, stream_1 is mainly 880 Hz
        stream_0 = (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
        stream_1 = (0.5 * np.sin(2 * np.pi * 880.0 * t)).astype(np.float32)

        turns_by_speaker = {
            "SPEAKER_00": [(0.0, 1.0), (2.0, 3.0)],  # matches stream_0
            "SPEAKER_01": [(1.0, 2.0), (3.0, 4.0)],  # matches stream_1
        }
        return [stream_0, stream_1], turns_by_speaker

    def test_correct_assignment(self):
        from helium.pipeline.stages.separate import _guided_assignment

        # Build fake diar dir so _guided_assignment can load it via _load_diar_turns
        import tempfile, json
        with tempfile.TemporaryDirectory() as tmpdir:
            diar_dir = Path(tmpdir)
            streams, turns_by_speaker = self._make_streams_and_turns(sr=8000)

            # Flatten turns_by_speaker into the JSON format
            turns_flat = [
                {"speaker": spk, "start": s, "end": e, "duration": e - s}
                for spk, windows in turns_by_speaker.items()
                for s, e in windows
            ]
            (diar_dir / "clip_000_exclusive.json").write_text(
                json.dumps({"file": "clip_000", "turns": turns_flat})
            )

            assignment = _guided_assignment(streams, diar_dir, "clip_000", 8000)

        assert assignment is not None
        assert set(assignment.keys()) == {"SPEAKER_00", "SPEAKER_01"}
        # Both speakers must be assigned to distinct streams
        assert assignment["SPEAKER_00"] != assignment["SPEAKER_01"]
        assert set(assignment.values()) == {0, 1}

    def test_no_diar_returns_none(self, tmp_path):
        from helium.pipeline.stages.separate import _guided_assignment

        stream = _sine(440.0, duration=2.0)
        result = _guided_assignment([stream, stream], tmp_path, "missing", 8000)
        assert result is None

    def test_more_speakers_than_streams_picks_top_two(self):
        """When diarization found 4 speakers, pick top-2 by speech time."""
        from helium.pipeline.stages.separate import _guided_assignment

        import tempfile, json
        with tempfile.TemporaryDirectory() as tmpdir:
            diar_dir = Path(tmpdir)
            streams = [_sine(440.0, duration=8.0), _sine(880.0, duration=8.0)]
            sr = 8000

            # 4 speakers with varying amounts of speech
            turns = [
                {"speaker": "SPK_A", "start": 0.0, "end": 4.0, "duration": 4.0},
                {"speaker": "SPK_B", "start": 4.0, "end": 8.0, "duration": 4.0},
                {"speaker": "SPK_C", "start": 0.0, "end": 0.5, "duration": 0.5},
                {"speaker": "SPK_D", "start": 1.0, "end": 1.2, "duration": 0.2},
            ]
            (diar_dir / "clip_000.json").write_text(
                json.dumps({"file": "clip_000", "turns": turns})
            )

            assignment = _guided_assignment(streams, diar_dir, "clip_000", sr)

        assert assignment is not None
        # Must use exactly 2 speakers (top by speech time)
        assert len(assignment) == 2
        assert "SPK_A" in assignment
        assert "SPK_B" in assignment


# ── evaluate stage with real separation artifacts ────────────────────────────

class TestEvaluateSeparationMetrics:
    def _write_manifest(self, sep_dir: Path, diarization_guided: bool) -> None:
        manifest = {
            "stage": "separate",
            "backend": "speechbrain/sepformer-whamr",
            "target_speakers": 2,
            "model_sample_rate": 8000,
            "files": [
                {
                    "file": "clip_000.wav",
                    "stem": "clip_000",
                    "status": "completed",
                    "diarization_guided": diarization_guided,
                }
            ],
        }
        (sep_dir / "manifest.json").write_text(json.dumps(manifest))

    def test_computes_blind_coherence_from_assigned_files(self, tmp_path):
        from helium.pipeline.stages import evaluate

        artifacts_dir = tmp_path / "artifacts"
        sep_dir = artifacts_dir / "separation"
        sep_dir.mkdir(parents=True)
        self._write_manifest(sep_dir, diarization_guided=True)

        _make_wav(sep_dir / "clip_000_speaker_00.wav", _sine(440.0))
        _make_wav(sep_dir / "clip_000_speaker_01.wav", _sine(880.0))

        result = evaluate.run(artifacts_dir, ["clip_000.wav"], target_speakers=2)
        assert result["metrics_computed"] >= 1

        import json as _json
        doc = _json.loads((artifacts_dir / "evaluation" / "metrics.json").read_text())
        assert doc["separation"]["status"] == "computed"
        assert "blind_source_coherence" in doc["separation"]["values"]
        assert doc["separation"]["values"]["diarization_guided"] is True

    def test_falls_back_to_raw_streams_when_no_assigned(self, tmp_path):
        from helium.pipeline.stages import evaluate

        artifacts_dir = tmp_path / "artifacts"
        sep_dir = artifacts_dir / "separation"
        sep_dir.mkdir(parents=True)
        self._write_manifest(sep_dir, diarization_guided=False)

        # Write only raw streams (no assigned _speaker_ files)
        _make_wav(sep_dir / "clip_000_raw_0.wav", _sine(440.0))
        _make_wav(sep_dir / "clip_000_raw_1.wav", _sine(880.0))

        result = evaluate.run(artifacts_dir, ["clip_000.wav"], target_speakers=2)
        import json as _json
        doc = _json.loads((artifacts_dir / "evaluation" / "metrics.json").read_text())
        assert doc["separation"]["status"] == "computed"
