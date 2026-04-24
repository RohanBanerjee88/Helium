"""
Run a benchmark evaluation on a standard speech separation dataset.

Supported datasets: libri2mix, whamr, voxceleb

Usage:
    python scripts/run_benchmark.py --dataset libri2mix --data-dir /data/Libri2Mix --output ./benchmark_results
    python scripts/run_benchmark.py --dataset whamr --data-dir /data/whamr --split tt --n-samples 100
    python scripts/run_benchmark.py --dataset voxceleb --data-dir /data/VoxCeleb

To evaluate a custom model, implement a separation function and pass it to run_benchmark():

    def my_model(mixture_path, sample_rate):
        # load model, run inference
        return [estimate_s1_array, estimate_s2_array]

    from helium.benchmarks.runner import run_benchmark
    from helium.benchmarks.datasets.libri2mix import Libri2Mix
    run_benchmark(Libri2Mix(...), output_dir, separation_fn=my_model)

Without a separation_fn, the script looks for pre-computed estimated_s*.wav files
in the output sample directories (useful for evaluating model outputs saved offline).
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helium.benchmarks.datasets.libri2mix import Libri2Mix
from helium.benchmarks.datasets.voxceleb import VoxCeleb
from helium.benchmarks.datasets.whamr import WHAMR
from helium.benchmarks.runner import run_benchmark

DATASETS = {
    "libri2mix": Libri2Mix,
    "whamr": WHAMR,
    "voxceleb": VoxCeleb,
}


def main():
    parser = argparse.ArgumentParser(description="Benchmark speech separation on standard datasets")
    parser.add_argument("--dataset", required=True, choices=list(DATASETS), help="Dataset to evaluate on")
    parser.add_argument("--data-dir", required=True, type=Path, help="Root directory of the dataset")
    parser.add_argument("--output", type=Path, default=Path("./benchmark_results"), help="Results output directory")
    parser.add_argument("--split", default="test", help="Dataset split: test / dev / train (default: test)")
    parser.add_argument("--n-samples", type=int, default=None, help="Limit number of samples evaluated")
    parser.add_argument("--save-estimates", action="store_true", help="Write estimated WAVs alongside metrics")
    parser.add_argument("--sample-rate", type=int, default=8000, choices=[8000, 16000], help="Sample rate (default: 8000)")
    args = parser.parse_args()

    dataset_cls = DATASETS[args.dataset]

    # Instantiate dataset — each class accepts root and sample_rate at minimum
    try:
        dataset = dataset_cls(root=args.data_dir, sample_rate=args.sample_rate)
    except TypeError:
        dataset = dataset_cls(root=args.data_dir)

    print(f"Dataset:  {args.dataset}")
    print(f"Data dir: {args.data_dir}")
    print(f"Split:    {args.split}")
    print(f"Output:   {args.output}")
    if args.n_samples:
        print(f"Samples:  {args.n_samples}")
    print()

    # To plug in your model:
    #   1. Define a separation_fn(mixture_path, sr) -> [s1_array, s2_array]
    #   2. Pass it as separation_fn=your_fn below
    separation_fn = None  # replace with your model callable

    try:
        summary = run_benchmark(
            dataset=dataset,
            output_dir=args.output,
            separation_fn=separation_fn,
            split=args.split,
            n_samples=args.n_samples,
            save_estimates=args.save_estimates,
        )
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    print("Summary:")
    print(json.dumps(summary, indent=2))

    summary_path = args.output / args.dataset / args.split / "summary.json"
    print(f"\nFull results at: {summary_path}")


if __name__ == "__main__":
    main()
