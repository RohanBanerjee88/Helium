"""CLI commands for reproducible benchmark evaluation.

Registered as the `helium benchmark` sub-command group.
"""
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from .datasets import Libri2MixDataset, VoxCelebDataset, WHAMRDataset
from .runner import run_benchmark

app = typer.Typer(help="Run reproducible benchmark evaluations on standard datasets.")
console = Console()

_DATASETS = {
    "libri2mix": "two-speaker separation (clean)",
    "whamr": "two-speaker separation (noisy + reverberant)",
    "voxceleb": "speaker identity / similarity",
}


def _build_dataset(name: str, data_dir: Path, sample_rate: int):
    """Instantiate the appropriate dataset class."""
    if name == "libri2mix":
        return Libri2MixDataset(data_dir, sample_rate=sample_rate)
    if name == "whamr":
        return WHAMRDataset(data_dir)
    if name == "voxceleb":
        return VoxCelebDataset(data_dir)
    raise ValueError(f"Unknown dataset: {name!r}")


@app.command("run")
def benchmark_run(
    dataset: str = typer.Argument(
        ...,
        help="Dataset name: libri2mix, whamr, or voxceleb.",
    ),
    data_dir: Path = typer.Option(
        ...,
        "--data-dir",
        help="Root directory of the dataset on disk.",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
    ),
    output_dir: Path = typer.Option(
        Path("benchmark_results"),
        "--output",
        "-o",
        help="Directory to write all benchmark outputs.",
    ),
    split: str = typer.Option("test", "--split", "-s", help="Dataset split to evaluate."),
    n_samples: Optional[int] = typer.Option(
        None, "--n-samples", "-n", help="Limit the number of samples evaluated."
    ),
    save_estimates: bool = typer.Option(
        False,
        "--save-estimates",
        help="Copy estimated source WAVs and mixtures alongside per-sample metrics.",
    ),
    sample_rate: int = typer.Option(
        16000, "--sample-rate", help="Sample rate for Libri2Mix (8000 or 16000)."
    ),
) -> None:
    """Evaluate separation quality on a standard benchmark dataset.

    Results are written to <output>/  <dataset>/<split>/ as:
      samples/<id>/metrics.json   per-sample metric values
      results.csv                 all samples in one CSV
      summary.json                aggregate statistics

    Pass --save-estimates to also save estimated source WAVs for inspection.

    If no separation model is wired, place pre-computed estimated_s1.wav /
    estimated_s2.wav files in the sample directories and re-run — the runner
    will pick them up automatically.
    """
    name = dataset.lower()
    if name not in _DATASETS:
        console.print(
            f"[red]Unknown dataset '{dataset}'.[/red] "
            f"Choose from: {', '.join(_DATASETS)}"
        )
        raise typer.Exit(1)

    console.print(f"\n[bold]Helium Benchmark[/bold]  {name} / {split}\n")
    console.print(f"  data_dir   : {data_dir}")
    console.print(f"  output_dir : {output_dir}")
    if n_samples:
        console.print(f"  n_samples  : {n_samples}")
    console.print()

    ds = _build_dataset(name, data_dir, sample_rate)
    if not ds.validate():
        console.print(
            f"[red]Dataset root not found or incomplete: {data_dir}[/red]\n"
            f"  Download {name} and point --data-dir at its root directory."
        )
        raise typer.Exit(1)

    with console.status(f"Evaluating {name}…", spinner="dots"):
        try:
            summary = run_benchmark(
                dataset=ds,
                output_dir=output_dir,
                split=split,
                n_samples=n_samples,
                save_estimates=save_estimates,
            )
        except FileNotFoundError as exc:
            console.print(f"[red]Error:[/red] {exc}")
            raise typer.Exit(1)

    n_eval = summary.get("n_evaluated", 0)
    console.print(f"  [green]Done.[/green]  Evaluated {n_eval} sample(s).\n")

    metrics = summary.get("metrics", {})
    if metrics:
        table = Table(box=None, show_header=True, header_style="bold", padding=(0, 2))
        table.add_column("METRIC")
        table.add_column("MEAN", justify="right")
        table.add_column("MIN", justify="right")
        table.add_column("MAX", justify="right")
        for metric, stats in metrics.items():
            table.add_row(metric, str(stats["mean"]), str(stats["min"]), str(stats["max"]))
        console.print(table)
        console.print()
    elif n_eval == 0:
        console.print(
            "  [dim]No pre-computed estimates found.[/dim]\n"
            "  Place estimated_s1.wav / estimated_s2.wav in each sample directory\n"
            "  and re-run, or wire a separation model to produce them.\n"
        )

    run_dir = output_dir / name / split
    console.print(f"  Results: [bold]{run_dir}[/bold]\n")


@app.command("list")
def benchmark_list() -> None:
    """List available benchmark datasets and their intended use."""
    table = Table(box=None, show_header=True, header_style="bold", padding=(0, 2))
    table.add_column("DATASET")
    table.add_column("TASK")
    table.add_column("HOMEPAGE")

    rows = [
        ("libri2mix", "two-speaker separation (clean)", "https://github.com/JorisCos/LibriMix"),
        ("whamr", "separation (noisy + reverberant)", "https://wham.whisper.ai"),
        ("voxceleb", "speaker identity / similarity", "https://www.robots.ox.ac.uk/~vgg/data/voxceleb/"),
    ]
    for row in rows:
        table.add_row(*row)

    console.print()
    console.print(table)
    console.print()
