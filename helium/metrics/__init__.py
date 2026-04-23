"""Helium metrics: objective evaluation functions for speech research.

Pure-numpy functions (no model dependency):
  separation  — si_sdr, sdr, sdri, si_sdri, pit_si_sdr, blind_coherence
  diarization — der, load_rttm, Segment
  content     — wer, cer

Functions that probe optional backends at call time:
  speaker_sim — compute_speaker_similarity (resemblyzer / speechbrain)
  content     — compute_wer_from_audio (faster-whisper / openai-whisper)
"""

from .separation import blind_coherence, pit_si_sdr, sdr, sdri, si_sdr, si_sdri
from .diarization import Segment, der, load_rttm
from .speaker_sim import compute_speaker_similarity, cosine_similarity
from .content import cer, compute_wer_from_audio, wer

__all__ = [
    # separation
    "si_sdr",
    "sdr",
    "sdri",
    "si_sdri",
    "pit_si_sdr",
    "blind_coherence",
    # diarization
    "der",
    "load_rttm",
    "Segment",
    # speaker similarity
    "cosine_similarity",
    "compute_speaker_similarity",
    # content preservation
    "wer",
    "cer",
    "compute_wer_from_audio",
]
