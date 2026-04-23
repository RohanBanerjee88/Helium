"""Diarization Error Rate and RTTM utilities.

DER = (missed_speech + false_alarm + speaker_confusion) / total_reference_speech

Speaker assignment uses optimal linear-sum matching when scipy is available,
falling back to greedy otherwise.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Union

import numpy as np


@dataclass
class Segment:
    onset: float
    duration: float
    speaker: str

    @property
    def end(self) -> float:
        return self.onset + self.duration


def load_rttm(path: Union[str, Path]) -> List[Segment]:
    """Parse an RTTM file into a list of Segments sorted by onset."""
    segments: List[Segment] = []
    with open(path) as fh:
        for line in fh:
            parts = line.strip().split()
            if len(parts) < 8 or parts[0] != "SPEAKER":
                continue
            segments.append(Segment(
                onset=float(parts[3]),
                duration=float(parts[4]),
                speaker=parts[7],
            ))
    return sorted(segments, key=lambda s: s.onset)


def der(
    reference_rttm: Union[str, Path],
    hypothesis_rttm: Union[str, Path],
    collar: float = 0.25,
) -> Dict[str, float]:
    """Compute Diarization Error Rate at 10 ms frame resolution.

    Args:
        reference_rttm: ground-truth RTTM file path
        hypothesis_rttm: system-output RTTM file path
        collar: seconds excluded around each reference segment boundary
                (standard collar = 0.25 s)
    Returns:
        dict with keys: der, miss_rate, false_alarm_rate, confusion_rate
        All values are fractions (0–1), not percentages.
    """
    _zero = {"der": 0.0, "miss_rate": 0.0, "false_alarm_rate": 0.0, "confusion_rate": 0.0}
    resolution = 0.01  # 10 ms frames

    ref_segs = load_rttm(reference_rttm)
    hyp_segs = load_rttm(hypothesis_rttm)

    if not ref_segs:
        return _zero

    total_end = max(s.end for s in ref_segs)
    n = int(np.ceil(total_end / resolution)) + 2
    collar_frames = max(1, int(collar / resolution))

    ref_ids: Dict[str, int] = {}
    hyp_ids: Dict[str, int] = {}
    ref_arr = np.full(n, -1, dtype=np.int32)
    hyp_arr = np.full(n, -1, dtype=np.int32)

    for seg in ref_segs:
        sid = ref_ids.setdefault(seg.speaker, len(ref_ids))
        s, e = int(seg.onset / resolution), int(seg.end / resolution)
        ref_arr[s: min(n, e)] = sid

    for seg in hyp_segs:
        sid = hyp_ids.setdefault(seg.speaker, len(hyp_ids))
        s, e = int(seg.onset / resolution), int(seg.end / resolution)
        hyp_arr[s: min(n, e)] = sid

    # Build collar mask (True = included in scoring)
    collar_mask = np.ones(n, dtype=bool)
    for seg in ref_segs:
        for boundary in (seg.onset, seg.end):
            b = int(boundary / resolution)
            collar_mask[max(0, b - collar_frames): min(n, b + collar_frames + 1)] = False

    ref_speech = collar_mask & (ref_arr >= 0)
    total_ref = int(ref_speech.sum())
    if total_ref == 0:
        return _zero

    miss = int((ref_speech & (hyp_arr < 0)).sum())
    fa = int((collar_mask & (hyp_arr >= 0) & (ref_arr < 0)).sum())

    # Speaker confusion: both active frames with different speakers after optimal mapping
    both = ref_speech & (hyp_arr >= 0)
    n_ref = len(ref_ids)
    n_hyp = len(hyp_ids)
    confusion = 0

    if n_ref > 0 and n_hyp > 0 and both.sum() > 0:
        conf_mat = np.zeros((n_ref, n_hyp), dtype=np.int64)
        for f in np.where(both)[0]:
            conf_mat[ref_arr[f], hyp_arr[f]] += 1

        try:
            from scipy.optimize import linear_sum_assignment
            row_ind, col_ind = linear_sum_assignment(-conf_mat)
            matched = int(conf_mat[row_ind, col_ind].sum())
        except ImportError:
            # Greedy assignment: pick max-overlap pair, mark both used
            matched = 0
            used_h: set = set()
            for r in range(n_ref):
                best_h, best_v = -1, -1
                for h in range(n_hyp):
                    if h not in used_h and conf_mat[r, h] > best_v:
                        best_v, best_h = int(conf_mat[r, h]), h
                if best_h >= 0:
                    matched += best_v
                    used_h.add(best_h)

        confusion = int(both.sum()) - matched

    der_val = (miss + fa + confusion) / total_ref
    return {
        "der": round(float(der_val), 4),
        "miss_rate": round(float(miss / total_ref), 4),
        "false_alarm_rate": round(float(fa / total_ref), 4),
        "confusion_rate": round(float(confusion / total_ref), 4),
    }
