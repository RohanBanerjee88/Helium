"""
Compute speech research metrics on existing audio files.

Metrics available:
  Separation:    SI-SDR, SDR, SI-SDRi, SDRi, PIT-SI-SDR, blind source coherence
  Diarization:   DER (requires reference and hypothesis RTTM files)
  Speaker sim:   cosine similarity (requires resemblyzer or speechbrain)
  Content (WER): word error rate (requires faster-whisper or openai-whisper)

Usage examples:

  # SI-SDR between reference and estimate
  python scripts/compute_metrics.py separation \
      --reference s1_ref.wav s2_ref.wav \
      --estimate  s1_est.wav s2_est.wav \
      --mixture   mixture.wav

  # Diarization error rate
  python scripts/compute_metrics.py diarization \
      --reference ref_turns.rttm \
      --hypothesis hyp_turns.rttm

  # Blind source coherence (no reference required)
  python scripts/compute_metrics.py coherence \
      --sources s1_est.wav s2_est.wav

  # Speaker similarity
  python scripts/compute_metrics.py speaker-sim \
      --audio-a speaker_a.wav \
      --audio-b speaker_b.wav

  # Word error rate
  python scripts/compute_metrics.py wer \
      --audio transcribed.wav \
      --reference-text "the quick brown fox"
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from helium.metrics._audio import load_wav
from helium.metrics import separation as sep_metrics
from helium.metrics import diarization as diar_metrics
from helium.metrics import speaker_sim as sim_metrics
from helium.metrics import content as content_metrics


def cmd_separation(args):
    refs = [load_wav(p)[0] for p in args.reference]
    ests = [load_wav(p)[0] for p in args.estimate]

    # Truncate to shortest
    min_len = min(*(len(r) for r in refs), *(len(e) for e in ests))
    refs = [r[:min_len] for r in refs]
    ests = [e[:min_len] for e in ests]

    results = {}

    if len(refs) == 2 and len(ests) == 2:
        pit, perm = sep_metrics.pit_si_sdr(refs, ests)
        results["pit_si_sdr_db"] = round(pit, 3)
        results["best_permutation"] = perm

        for i, (ref, est) in enumerate(zip(refs, [ests[p] for p in perm])):
            results[f"si_sdr_s{i+1}_db"] = round(sep_metrics.si_sdr(ref, est), 3)
            results[f"sdr_s{i+1}_db"] = round(sep_metrics.sdr(ref, est), 3)

            if args.mixture:
                mix, _ = load_wav(args.mixture)
                mix = mix[:min_len]
                results[f"si_sdri_s{i+1}_db"] = round(sep_metrics.si_sdri(ref, est, mix), 3)
                results[f"sdri_s{i+1}_db"] = round(sep_metrics.sdri(ref, est, mix), 3)
    else:
        for i, (ref, est) in enumerate(zip(refs, ests)):
            results[f"si_sdr_s{i+1}_db"] = round(sep_metrics.si_sdr(ref, est), 3)
            results[f"sdr_s{i+1}_db"] = round(sep_metrics.sdr(ref, est), 3)

    print(json.dumps(results, indent=2))


def cmd_diarization(args):
    der = diar_metrics.der(args.reference, args.hypothesis)
    print(json.dumps({"der": round(der, 4)}, indent=2))


def cmd_coherence(args):
    sources = [load_wav(p)[0] for p in args.sources]
    if len(sources) < 2:
        print("Error: need at least 2 source files", file=sys.stderr)
        sys.exit(1)
    min_len = min(len(s) for s in sources)
    sources = [s[:min_len] for s in sources]
    coherence = sep_metrics.blind_coherence(sources[0], sources[1])
    print(json.dumps({"blind_source_coherence": round(coherence, 4)}, indent=2))


def cmd_speaker_sim(args):
    sim = sim_metrics.compute_speaker_similarity(args.audio_a, args.audio_b)
    print(json.dumps({"cosine_similarity": round(sim, 4) if sim is not None else None}, indent=2))


def cmd_wer(args):
    wer = content_metrics.compute_wer_from_audio(args.audio, args.reference_text)
    print(json.dumps({"wer": round(wer, 4) if wer is not None else None}, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Compute speech research metrics on audio files")
    sub = parser.add_subparsers(dest="command", required=True)

    # separation
    p_sep = sub.add_parser("separation", help="Separation metrics (SI-SDR, SDR, PIT-SI-SDR, improvements)")
    p_sep.add_argument("--reference", nargs="+", required=True, type=Path, help="Reference source WAV files")
    p_sep.add_argument("--estimate",  nargs="+", required=True, type=Path, help="Estimated source WAV files")
    p_sep.add_argument("--mixture",   type=Path, default=None, help="Mixture WAV (required for SDRi/SI-SDRi)")

    # diarization
    p_diar = sub.add_parser("diarization", help="Diarization error rate from RTTM files")
    p_diar.add_argument("--reference",   required=True, type=Path, help="Reference RTTM file")
    p_diar.add_argument("--hypothesis",  required=True, type=Path, help="Hypothesis RTTM file")

    # coherence
    p_coh = sub.add_parser("coherence", help="Blind source coherence (no reference required)")
    p_coh.add_argument("--sources", nargs="+", required=True, type=Path, help="Estimated source WAV files")

    # speaker-sim
    p_sim = sub.add_parser("speaker-sim", help="Speaker similarity (requires resemblyzer or speechbrain)")
    p_sim.add_argument("--audio-a", required=True, type=Path, help="First speaker WAV")
    p_sim.add_argument("--audio-b", required=True, type=Path, help="Second speaker WAV")

    # wer
    p_wer = sub.add_parser("wer", help="Word error rate (requires faster-whisper or openai-whisper)")
    p_wer.add_argument("--audio",          required=True, type=Path, help="Audio file to transcribe")
    p_wer.add_argument("--reference-text", required=True, help="Ground-truth transcript")

    args = parser.parse_args()

    dispatch = {
        "separation":  cmd_separation,
        "diarization": cmd_diarization,
        "coherence":   cmd_coherence,
        "speaker-sim": cmd_speaker_sim,
        "wer":         cmd_wer,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
