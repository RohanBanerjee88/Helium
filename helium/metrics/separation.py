"""Objective speech separation metrics.

All functions accept numpy float arrays.  No model backends required.
"""
from itertools import permutations
from typing import List, Tuple

import numpy as np


def _align(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = min(len(a), len(b))
    return a[:n], b[:n]


def si_sdr(reference: np.ndarray, estimate: np.ndarray, eps: float = 1e-8) -> float:
    """Scale-Invariant Signal-to-Distortion Ratio in dB.

    Higher = better.  Permutation-invariant wrapper: see pit_si_sdr.
    """
    ref, est = _align(
        reference.flatten().astype(np.float64),
        estimate.flatten().astype(np.float64),
    )
    ref -= ref.mean()
    est -= est.mean()
    alpha = np.dot(ref, est) / (np.dot(ref, ref) + eps)
    proj = alpha * ref
    noise = est - proj
    return float(10.0 * np.log10((np.dot(proj, proj) + eps) / (np.dot(noise, noise) + eps)))


def sdr(reference: np.ndarray, estimate: np.ndarray, eps: float = 1e-8) -> float:
    """Signal-to-Distortion Ratio in dB."""
    ref, est = _align(
        reference.flatten().astype(np.float64),
        estimate.flatten().astype(np.float64),
    )
    error = ref - est
    return float(10.0 * np.log10((np.dot(ref, ref) + eps) / (np.dot(error, error) + eps)))


def sdri(reference: np.ndarray, estimate: np.ndarray, mixture: np.ndarray) -> float:
    """SDR improvement over the unprocessed mixture."""
    return sdr(reference, estimate) - sdr(reference, mixture)


def si_sdri(reference: np.ndarray, estimate: np.ndarray, mixture: np.ndarray) -> float:
    """SI-SDR improvement over the unprocessed mixture."""
    return si_sdr(reference, estimate) - si_sdr(reference, mixture)


def pit_si_sdr(
    references: List[np.ndarray],
    estimates: List[np.ndarray],
) -> Tuple[float, List[int]]:
    """Permutation-Invariant SI-SDR.

    Exhaustively tries all speaker assignments and returns the best mean
    SI-SDR together with the winning permutation index list.

    Args:
        references: list of reference signals, one per speaker
        estimates:  list of estimated signals, one per speaker
    Returns:
        (mean_si_sdr_db, best_perm) where best_perm[i] is the estimate
        index assigned to reference i.
    """
    n = len(references)
    if n != len(estimates):
        raise ValueError("references and estimates must have the same length")

    best_score = float("-inf")
    best_perm: List[int] = list(range(n))

    for perm in permutations(range(n)):
        score = sum(si_sdr(references[i], estimates[perm[i]]) for i in range(n)) / n
        if score > best_score:
            best_score = score
            best_perm = list(perm)

    return best_score, best_perm


def blind_coherence(source_a: np.ndarray, source_b: np.ndarray) -> float:
    """Pearson correlation between two estimated sources.

    A proxy for separation quality that requires no reference signals.
    |r| close to 0 = sources are well-separated; |r| close to 1 = leakage.
    """
    a, b = _align(
        source_a.flatten().astype(np.float64),
        source_b.flatten().astype(np.float64),
    )
    std_a, std_b = a.std(), b.std()
    if std_a < 1e-8 or std_b < 1e-8:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])
