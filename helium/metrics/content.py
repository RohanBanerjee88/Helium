"""Content preservation metrics: WER and CER.

wer() and cer() are pure Python — no model dependency.
compute_wer_from_audio() tries available ASR backends (faster-whisper,
then openai-whisper) and is honest about availability.
"""
from pathlib import Path
from typing import Any, Dict, Union


def wer(reference: str, hypothesis: str) -> float:
    """Word Error Rate via Levenshtein edit distance on word sequences.

    Returns a fraction in [0, ∞).  0 = perfect match, 1 = 100% error rate.
    Values > 1 are possible when insertions exceed reference length.
    """
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()

    if not ref_words:
        return 0.0 if not hyp_words else float("inf")

    # Standard Levenshtein on word sequences
    d = list(range(len(hyp_words) + 1))
    for i, rw in enumerate(ref_words):
        prev = d[:]
        d[0] = i + 1
        for j, hw in enumerate(hyp_words):
            d[j + 1] = min(
                prev[j] + (0 if rw == hw else 1),
                d[j] + 1,
                prev[j + 1] + 1,
            )
    return d[len(hyp_words)] / len(ref_words)


def cer(reference: str, hypothesis: str) -> float:
    """Character Error Rate via Levenshtein edit distance on characters."""
    ref_chars = list(reference.lower().replace(" ", ""))
    hyp_chars = list(hypothesis.lower().replace(" ", ""))

    if not ref_chars:
        return 0.0 if not hyp_chars else float("inf")

    d = list(range(len(hyp_chars) + 1))
    for i, rc in enumerate(ref_chars):
        prev = d[:]
        d[0] = i + 1
        for j, hc in enumerate(hyp_chars):
            d[j + 1] = min(
                prev[j] + (0 if rc == hc else 1),
                d[j] + 1,
                prev[j + 1] + 1,
            )
    return d[len(hyp_chars)] / len(ref_chars)


def compute_wer_from_audio(
    audio_path: Union[str, Path],
    reference_text: str,
    backend: str = "auto",
) -> Dict[str, Any]:
    """Transcribe audio and compute WER against reference_text.

    Tries faster-whisper then openai-whisper.  Returns a dict with:
      - status: "computed" | "unavailable" | "error"
      - wer: float (only when status == "computed")
      - transcript: str (only when status == "computed")
      - backend: str (only when status == "computed")
      - reason: str (only when status != "computed")
    """
    backends = ["faster-whisper", "whisper"] if backend == "auto" else [backend]

    for name in backends:
        if name == "faster-whisper":
            result = _try_faster_whisper(Path(audio_path), reference_text)
        elif name == "whisper":
            result = _try_whisper(Path(audio_path), reference_text)
        else:
            result = {"status": "unavailable", "reason": f"unknown backend '{name}'"}

        if result["status"] == "computed":
            return result

    return {
        "status": "unavailable",
        "reason": (
            "No ASR backend available. "
            "Install faster-whisper (`pip install faster-whisper`) "
            "or openai-whisper (`pip install openai-whisper`) to enable WER."
        ),
    }


def _try_faster_whisper(audio_path: Path, reference_text: str) -> Dict[str, Any]:
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        return {"status": "unavailable", "reason": "faster-whisper not installed"}

    try:
        model = WhisperModel("tiny", device="cpu", compute_type="int8")
        segments, _ = model.transcribe(str(audio_path))
        hypothesis = " ".join(seg.text.strip() for seg in segments)
        return {
            "status": "computed",
            "wer": round(wer(reference_text, hypothesis), 4),
            "transcript": hypothesis,
            "backend": "faster-whisper",
        }
    except Exception as exc:
        return {"status": "error", "reason": str(exc)}


def _try_whisper(audio_path: Path, reference_text: str) -> Dict[str, Any]:
    try:
        import whisper
    except ImportError:
        return {"status": "unavailable", "reason": "openai-whisper not installed"}

    try:
        model = whisper.load_model("tiny")
        result = model.transcribe(str(audio_path))
        hypothesis = result["text"].strip()
        return {
            "status": "computed",
            "wer": round(wer(reference_text, hypothesis), 4),
            "transcript": hypothesis,
            "backend": "openai-whisper",
        }
    except Exception as exc:
        return {"status": "error", "reason": str(exc)}
