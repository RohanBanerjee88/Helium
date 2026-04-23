"""Unit tests for helium/metrics/.

All tests use pure numpy — no model backends required.
"""
import math
import struct
import tempfile
import wave
from pathlib import Path

import numpy as np
import pytest

from helium.metrics.separation import (
    blind_coherence,
    pit_si_sdr,
    sdr,
    sdri,
    si_sdr,
    si_sdri,
)
from helium.metrics.diarization import Segment, der, load_rttm
from helium.metrics.content import cer, wer


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_wav(path: Path, samples: np.ndarray, sr: int = 16000) -> None:
    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


def _sine(freq: float, duration: float = 1.0, sr: int = 16000, amp: float = 0.5) -> np.ndarray:
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _write_rttm(path: Path, segments: list[tuple]) -> None:
    """Write a minimal RTTM file.  Each tuple: (onset, duration, speaker)."""
    with open(path, "w") as fh:
        for onset, dur, spk in segments:
            fh.write(f"SPEAKER file 1 {onset:.3f} {dur:.3f} <NA> <NA> {spk} <NA> <NA>\n")


# ── separation metrics ────────────────────────────────────────────────────────

class TestSiSdr:
    def test_perfect_reconstruction(self):
        ref = _sine(440.0)
        # Perfect estimate → SI-SDR should be very high
        score = si_sdr(ref, ref.copy())
        assert score > 50.0

    def test_zero_signal_clipped(self):
        ref = _sine(440.0)
        noise = np.random.default_rng(0).normal(0, 1, len(ref)).astype(np.float32)
        # Very noisy estimate → low SI-SDR
        score = si_sdr(ref, noise)
        assert score < 0.0

    def test_scale_invariant(self):
        ref = _sine(440.0)
        noise = np.random.default_rng(0).normal(0, 0.1, len(ref)).astype(np.float32)
        est = ref + noise
        # SI-SDR is scale-invariant: multiplying the estimate by any non-zero scalar
        # leaves the score unchanged (valid when noise >> eps floor).
        score_1 = si_sdr(ref, est * 2.0)
        score_2 = si_sdr(ref, est * 0.1)
        assert abs(score_1 - score_2) < 0.1

    def test_different_lengths_handled(self):
        ref = _sine(440.0, duration=1.0)
        est = _sine(440.0, duration=0.8)
        # Should not raise; shorter length is used
        score = si_sdr(ref, est)
        assert math.isfinite(score)


class TestSdr:
    def test_perfect_reconstruction(self):
        ref = _sine(440.0)
        assert sdr(ref, ref.copy()) > 50.0

    def test_symmetry_not_required(self):
        ref = _sine(440.0)
        noisy = ref + 0.1 * np.random.default_rng(1).normal(size=len(ref)).astype(np.float32)
        fwd = sdr(ref, noisy)
        rev = sdr(noisy, ref)
        # sdr(a, b) ≠ sdr(b, a) in general — just check both are finite
        assert math.isfinite(fwd) and math.isfinite(rev)


class TestSdri:
    def test_good_separation_positive_sdri(self):
        ref_a = _sine(440.0)
        ref_b = _sine(880.0)
        mixture = ref_a + ref_b
        # Perfect recovery of ref_a → sdri should be positive
        assert sdri(ref_a, ref_a.copy(), mixture) > 0.0

    def test_si_sdri_matches_components(self):
        ref = _sine(440.0)
        est = ref + 0.01 * np.random.default_rng(2).normal(size=len(ref)).astype(np.float32)
        mix = ref + _sine(880.0)
        val = si_sdri(ref, est, mix)
        assert val == pytest.approx(si_sdr(ref, est) - si_sdr(ref, mix), abs=1e-4)


class TestPitSiSdr:
    def test_best_permutation_found(self):
        ref_a = _sine(440.0)
        ref_b = _sine(880.0)
        # estimates in reversed order
        score, perm = pit_si_sdr([ref_a, ref_b], [ref_b.copy(), ref_a.copy()])
        assert score > 30.0
        assert perm == [1, 0]  # second estimate matches first reference

    def test_mismatched_length_raises(self):
        with pytest.raises(ValueError):
            pit_si_sdr([_sine(440.0)], [_sine(440.0), _sine(880.0)])


class TestBlindCoherence:
    def test_same_signal_returns_one(self):
        sig = _sine(440.0)
        assert abs(blind_coherence(sig, sig.copy())) == pytest.approx(1.0, abs=1e-4)

    def test_orthogonal_signals_near_zero(self):
        # Cosine and sine at same frequency are orthogonal over a full period
        t = np.linspace(0, 1.0, 16000, endpoint=False)
        a = np.cos(2 * np.pi * 440.0 * t).astype(np.float32)
        b = np.sin(2 * np.pi * 440.0 * t).astype(np.float32)
        assert abs(blind_coherence(a, b)) < 0.05

    def test_silent_source_returns_zero(self):
        sig = _sine(440.0)
        silent = np.zeros(len(sig), dtype=np.float32)
        assert blind_coherence(sig, silent) == 0.0


# ── diarization metrics ───────────────────────────────────────────────────────

class TestLoadRttm:
    def test_parses_segments(self, tmp_path):
        rttm = tmp_path / "ref.rttm"
        _write_rttm(rttm, [(0.0, 2.5, "A"), (3.0, 1.5, "B")])
        segs = load_rttm(rttm)
        assert len(segs) == 2
        assert segs[0].speaker == "A"
        assert segs[0].duration == pytest.approx(2.5)
        assert segs[1].end == pytest.approx(4.5)

    def test_ignores_non_speaker_lines(self, tmp_path):
        rttm = tmp_path / "ref.rttm"
        rttm.write_text("FILEINFO file 1 0 5.0 <NA>\n")
        assert load_rttm(rttm) == []


class TestDer:
    def test_perfect_hypothesis(self, tmp_path):
        segs = [(0.0, 2.0, "A"), (2.5, 2.0, "B")]
        ref = tmp_path / "ref.rttm"
        hyp = tmp_path / "hyp.rttm"
        _write_rttm(ref, segs)
        _write_rttm(hyp, segs)
        result = der(ref, hyp, collar=0.0)
        # Perfect overlap — DER should be 0
        assert result["der"] == pytest.approx(0.0, abs=0.01)

    def test_complete_miss(self, tmp_path):
        ref = tmp_path / "ref.rttm"
        hyp = tmp_path / "hyp.rttm"
        _write_rttm(ref, [(0.0, 2.0, "A")])
        _write_rttm(hyp, [])  # hypothesis says nothing
        result = der(ref, hyp, collar=0.0)
        assert result["miss_rate"] == pytest.approx(1.0, abs=0.01)

    def test_empty_reference(self, tmp_path):
        ref = tmp_path / "ref.rttm"
        hyp = tmp_path / "hyp.rttm"
        _write_rttm(ref, [])
        _write_rttm(hyp, [(0.0, 1.0, "A")])
        result = der(ref, hyp, collar=0.0)
        assert result["der"] == pytest.approx(0.0)

    def test_confusion(self, tmp_path):
        ref = tmp_path / "ref.rttm"
        hyp = tmp_path / "hyp.rttm"
        # Reference: A speaks 0–2, B speaks 3–5
        _write_rttm(ref, [(0.0, 2.0, "A"), (3.0, 2.0, "B")])
        # Hypothesis: labels are swapped
        _write_rttm(hyp, [(0.0, 2.0, "B"), (3.0, 2.0, "A")])
        result = der(ref, hyp, collar=0.0)
        # Optimal mapping corrects the swap → DER should be ~0
        assert result["der"] < 0.05


# ── content metrics ───────────────────────────────────────────────────────────

class TestWer:
    def test_identical(self):
        assert wer("hello world", "hello world") == pytest.approx(0.0)

    def test_one_substitution(self):
        assert wer("hello world", "hello earth") == pytest.approx(0.5)

    def test_empty_reference(self):
        assert wer("", "") == pytest.approx(0.0)

    def test_empty_hypothesis_full_miss(self):
        assert wer("hello world", "") == pytest.approx(1.0)

    def test_case_insensitive(self):
        assert wer("Hello World", "hello world") == pytest.approx(0.0)


class TestCer:
    def test_identical(self):
        assert cer("abc", "abc") == pytest.approx(0.0)

    def test_one_char_substitution(self):
        assert cer("abc", "axc") == pytest.approx(1 / 3)

    def test_empty(self):
        assert cer("", "") == pytest.approx(0.0)


# ── evaluate stage integration ────────────────────────────────────────────────

class TestEvaluateStage:
    def test_runs_with_placeholder_upstreams(self, tmp_path):
        """evaluate should succeed even when separation/diarization are placeholders."""
        import json
        from helium.pipeline.stages import evaluate

        artifacts_dir = tmp_path / "artifacts"
        (artifacts_dir / "manifests").mkdir(parents=True)
        (artifacts_dir / "separation").mkdir()
        (artifacts_dir / "diarization").mkdir()

        # Write placeholder separation plan
        (artifacts_dir / "separation" / "separation_plan.json").write_text(
            json.dumps({"stage": "separate", "status": "placeholder"})
        )
        # Write placeholder diarization plan
        (artifacts_dir / "diarization" / "diarization_plan.json").write_text(
            json.dumps({"stage": "diarize", "status": "placeholder"})
        )

        result = evaluate.run(artifacts_dir, ["clip_000.wav"], target_speakers=2)

        assert isinstance(result, dict)
        assert "artifacts" in result
        assert result["metrics_computed"] == 0
        assert len(result["artifacts"]) == 3

        # metrics.json should exist and be valid JSON
        metrics_path = artifacts_dir / "evaluation" / "metrics.json"
        assert metrics_path.exists()
        doc = json.loads(metrics_path.read_text())
        assert doc["separation"]["status"] == "unavailable"
        assert doc["diarization"]["status"] == "unavailable"

    def test_writes_experiment_summary(self, tmp_path):
        import json
        from helium.pipeline.stages import evaluate

        artifacts_dir = tmp_path / "artifacts"
        artifacts_dir.mkdir()

        evaluate.run(artifacts_dir, ["clip_000.wav", "clip_001.wav"], target_speakers=2)

        summary_path = artifacts_dir / "evaluation" / "experiment_summary.json"
        assert summary_path.exists()
        summary = json.loads(summary_path.read_text())
        assert summary["target_speakers"] == 2
        assert summary["input_clip_count"] == 2
        assert "metrics_computed" in summary
        assert "metrics_unavailable" in summary

    def test_writes_csv_row(self, tmp_path):
        from helium.pipeline.stages import evaluate

        artifacts_dir = tmp_path / "artifacts"
        artifacts_dir.mkdir()

        evaluate.run(artifacts_dir, ["clip_000.wav"], target_speakers=2)

        csv_path = artifacts_dir / "evaluation" / "metrics.csv"
        assert csv_path.exists()
        lines = csv_path.read_text().splitlines()
        assert len(lines) == 2  # header + one data row

    def test_appends_csv_on_second_run(self, tmp_path):
        from helium.pipeline.stages import evaluate

        artifacts_dir = tmp_path / "artifacts"
        artifacts_dir.mkdir()

        evaluate.run(artifacts_dir, ["clip_000.wav"], target_speakers=2)
        evaluate.run(artifacts_dir, ["clip_000.wav"], target_speakers=2)

        csv_path = artifacts_dir / "evaluation" / "metrics.csv"
        lines = csv_path.read_text().splitlines()
        assert len(lines) == 3  # header + two data rows

    def test_computes_separation_metrics_when_real_outputs_exist(self, tmp_path):
        """evaluate should compute blind_coherence when real speaker WAVs are present."""
        import json
        from helium.pipeline.stages import evaluate

        artifacts_dir = tmp_path / "artifacts"
        sep_dir = artifacts_dir / "separation"
        sep_dir.mkdir(parents=True)

        # Write a real (non-placeholder) plan + two speaker WAVs
        (sep_dir / "separation_plan.json").write_text(
            json.dumps({"stage": "separate", "status": "completed"})
        )
        _make_wav(sep_dir / "speaker_0.wav", _sine(440.0))
        _make_wav(sep_dir / "speaker_1.wav", _sine(880.0))

        result = evaluate.run(artifacts_dir, ["mix.wav"], target_speakers=2)
        assert result["metrics_computed"] >= 1

        doc = json.loads((artifacts_dir / "evaluation" / "metrics.json").read_text())
        assert doc["separation"]["status"] == "computed"
        assert "blind_source_coherence" in doc["separation"]["values"]
