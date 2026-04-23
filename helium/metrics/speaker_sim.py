"""Speaker similarity scoring via embedding cosine distance.

The core cosine_similarity function is pure numpy.
compute_speaker_similarity tries available embedding backends in order
(resemblyzer → speechbrain) and returns a structured result dict that
is honest about availability.
"""
from pathlib import Path
from typing import Any, Dict, Union

import numpy as np


def cosine_similarity(emb_a: np.ndarray, emb_b: np.ndarray) -> float:
    """Cosine similarity between two embedding vectors (pure numpy)."""
    a = emb_a.flatten().astype(np.float64)
    b = emb_b.flatten().astype(np.float64)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-8 or norm_b < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def compute_speaker_similarity(
    audio_a: Union[str, Path],
    audio_b: Union[str, Path],
    backend: str = "auto",
) -> Dict[str, Any]:
    """Compute speaker similarity between two audio files.

    Tries available embedding backends in order (resemblyzer, then
    speechbrain).  Returns a dict with:
      - status: "computed" | "unavailable" | "error"
      - cosine_similarity: float (only when status == "computed")
      - backend: str (only when status == "computed")
      - reason: str (only when status != "computed")
    """
    backends = ["resemblyzer", "speechbrain"] if backend == "auto" else [backend]

    for name in backends:
        if name == "resemblyzer":
            result = _try_resemblyzer(Path(audio_a), Path(audio_b))
        elif name == "speechbrain":
            result = _try_speechbrain(Path(audio_a), Path(audio_b))
        else:
            result = {"status": "unavailable", "reason": f"unknown backend '{name}'"}

        if result["status"] == "computed":
            return result

    return {
        "status": "unavailable",
        "reason": (
            "No speaker embedding backend available. "
            "Install resemblyzer (`pip install resemblyzer`) "
            "or speechbrain (`pip install speechbrain`) to enable this metric."
        ),
    }


def _try_resemblyzer(audio_a: Path, audio_b: Path) -> Dict[str, Any]:
    try:
        from resemblyzer import VoiceEncoder, preprocess_wav
    except ImportError:
        return {"status": "unavailable", "reason": "resemblyzer not installed"}

    try:
        encoder = VoiceEncoder()
        wav_a = preprocess_wav(str(audio_a))
        wav_b = preprocess_wav(str(audio_b))
        emb_a = encoder.embed_utterance(wav_a)
        emb_b = encoder.embed_utterance(wav_b)
        sim = cosine_similarity(emb_a, emb_b)
        return {"status": "computed", "cosine_similarity": round(sim, 4), "backend": "resemblyzer"}
    except Exception as exc:
        return {"status": "error", "reason": str(exc)}


def _try_speechbrain(audio_a: Path, audio_b: Path) -> Dict[str, Any]:
    try:
        import speechbrain  # noqa: F401
    except ImportError:
        return {"status": "unavailable", "reason": "speechbrain not installed"}

    try:
        from speechbrain.pretrained import EncoderClassifier

        clf = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")
        emb_a = clf.encode_batch(clf.load_audio(str(audio_a))).squeeze().numpy()
        emb_b = clf.encode_batch(clf.load_audio(str(audio_b))).squeeze().numpy()
        sim = cosine_similarity(emb_a, emb_b)
        return {"status": "computed", "cosine_similarity": round(sim, 4), "backend": "speechbrain"}
    except Exception as exc:
        return {"status": "error", "reason": str(exc)}
