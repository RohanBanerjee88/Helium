"""Benchmark runner: evaluate separation quality on standard datasets.

Writes all outputs to disk so results are reproducible and inspectable
without re-running the evaluation.

Output layout
─────────────
  <output_dir>/
    <dataset_name>/
      <split>/
        samples/
          <sample_id>/
            metrics.json          ← per-sample metric values
            estimated_s1.wav      ← (optional) estimated source 1
            estimated_s2.wav      ← (optional) estimated source 2
            mixture.wav           ← (optional) copy of the mixture
        results.csv               ← one row per sample
        summary.json              ← aggregate statistics
"""
import csv
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from ..metrics._audio import load_wav, write_wav
from ..metrics.separation import blind_coherence, pit_si_sdr, si_sdri
from .datasets.base import BenchmarkDataset, BenchmarkSample

# Type alias: (mixture_path, sample_rate) → list of estimated source arrays
SeparationFn = Callable[[Path, int], List[np.ndarray]]


def evaluate_sample(
    sample: BenchmarkSample,
    estimates: List[np.ndarray],
) -> Dict[str, Any]:
    """Compute objective metrics for one sample.

    Args:
        sample:    ground-truth sample with mixture and source paths
        estimates: estimated source signals, one array per speaker
    Returns:
        dict with sample_id plus all computed metric values
    """
    refs = [load_wav(src)[0] for src in sample.sources]
    mix, _ = load_wav(sample.mixture)

    # Truncate everything to the shortest length
    min_len = min(len(mix), *(len(r) for r in refs), *(len(e) for e in estimates))
    mix = mix[:min_len]
    refs = [r[:min_len] for r in refs]
    ests = [e[:min_len] for e in estimates]

    row: Dict[str, Any] = {"sample_id": sample.sample_id}

    # Permutation-Invariant SI-SDR
    pi_si_sdr, best_perm = pit_si_sdr(refs, ests)
    row["si_sdr_db"] = round(pi_si_sdr, 3)
    row["best_permutation"] = best_perm

    # SI-SDRi (improvement over unprocessed mixture)
    si_sdri_vals = [
        si_sdri(refs[i], ests[best_perm[i]], mix) for i in range(len(refs))
    ]
    row["si_sdri_db"] = round(sum(si_sdri_vals) / len(si_sdri_vals), 3)

    # Blind source coherence (no reference required)
    if len(ests) >= 2:
        row["blind_source_coherence"] = round(blind_coherence(ests[0], ests[1]), 4)

    return row


def run_benchmark(
    dataset: BenchmarkDataset,
    output_dir: Path,
    separation_fn: Optional[SeparationFn] = None,
    split: str = "test",
    n_samples: Optional[int] = None,
    save_estimates: bool = False,
) -> Dict[str, Any]:
    """Run evaluation on a benchmark dataset and write all results to disk.

    Args:
        dataset:        benchmark dataset instance (must pass validate())
        output_dir:     root directory for all outputs
        separation_fn:  callable (mixture_path, sr) → [est_s1, est_s2, ...]
                        If None, looks for pre-computed estimates in
                        output_dir/<dataset>/<split>/samples/<id>/estimated_s*.wav
        split:          dataset split ('test', 'dev', 'train', …)
        n_samples:      cap the number of samples evaluated
        save_estimates: write estimated WAVs and mixture copies alongside metrics

    Returns:
        summary dict (also written to summary.json)
    """
    if not dataset.validate():
        raise FileNotFoundError(
            f"{dataset.name} dataset not found or incomplete at {dataset.root}.\n"
            "Download the dataset and pass the correct --data-dir."
        )

    run_dir = Path(output_dir) / dataset.name / split
    samples_dir = run_dir / "samples"
    run_dir.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(exist_ok=True)

    results: List[Dict[str, Any]] = []
    csv_path = run_dir / "results.csv"
    csv_fields: Optional[List[str]] = None

    for sample in dataset.iter_samples(split=split, n=n_samples):
        sample_dir = samples_dir / sample.sample_id
        sample_dir.mkdir(exist_ok=True)

        estimates = _get_estimates(sample, sample_dir, separation_fn)
        if estimates is None:
            continue

        row = evaluate_sample(sample, estimates)

        if save_estimates:
            for idx, est in enumerate(estimates):
                write_wav(sample_dir / f"estimated_s{idx + 1}.wav", est, sample.sample_rate)
            shutil.copy2(sample.mixture, sample_dir / "mixture.wav")

        with open(sample_dir / "metrics.json", "w") as fh:
            json.dump(row, fh, indent=2)

        results.append(row)

        # Stream result to CSV so partial runs are readable
        if csv_fields is None:
            csv_fields = list(row.keys())
            with open(csv_path, "w", newline="") as fh:
                csv.DictWriter(fh, fieldnames=csv_fields).writeheader()
        with open(csv_path, "a", newline="") as fh:
            csv.DictWriter(fh, fieldnames=csv_fields).writerow(row)

    summary = _aggregate(results, dataset, split, n_samples)
    with open(run_dir / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    return summary


def _get_estimates(
    sample: BenchmarkSample,
    sample_dir: Path,
    separation_fn: Optional[SeparationFn],
) -> Optional[List[np.ndarray]]:
    if separation_fn is not None:
        return separation_fn(sample.mixture, sample.sample_rate)

    est_files = sorted(sample_dir.glob("estimated_s*.wav"))
    if not est_files:
        return None
    return [load_wav(ef)[0] for ef in est_files]


def _aggregate(
    results: List[Dict[str, Any]],
    dataset: BenchmarkDataset,
    split: str,
    n_samples: Optional[int],
) -> Dict[str, Any]:
    if not results:
        return {
            "dataset": dataset.name,
            "split": split,
            "n_evaluated": 0,
            "error": "no samples had pre-computed estimates; pass --save-estimates or supply a separation_fn",
        }

    numeric_keys = [k for k in results[0] if isinstance(results[0].get(k), (int, float))]
    agg: Dict[str, Any] = {}
    for key in numeric_keys:
        vals = [r[key] for r in results if isinstance(r.get(key), (int, float))]
        if vals:
            agg[key] = {
                "mean": round(sum(vals) / len(vals), 3),
                "min": round(min(vals), 3),
                "max": round(max(vals), 3),
            }

    return {
        "dataset": dataset.name,
        "split": split,
        "n_evaluated": len(results),
        "n_requested": n_samples,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "metrics": agg,
    }
