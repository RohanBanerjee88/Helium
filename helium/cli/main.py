"""
Helium CLI — local photo-to-3D reconstruction.

Commands:
  helium run <photos>        Run reconstruction on a directory of photos
  helium jobs list           List all past jobs
  helium jobs show <id>      Show full detail for one job
  helium jobs clean <id>     Delete a job and its files
"""

import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from rich.table import Table

from helium import __version__
from helium.config import settings
from helium.models.job import Job, JobStatus, StageStatus
from helium.pipeline.stages import dense, export, features, matching, sfm, validate
from helium.storage.local import storage

app = typer.Typer(
    name="helium",
    help="Local-first photo-to-3D reconstruction.",
    add_completion=False,
)
jobs_app = typer.Typer(help="Manage past reconstruction jobs.")
app.add_typer(jobs_app, name="jobs")

console = Console()

_ALLOWED_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

_STAGE_ICON = {
    StageStatus.COMPLETED: "[green]✓[/green]",
    StageStatus.SKIPPED:   "[dim]↷[/dim]",
    StageStatus.FAILED:    "[red]✗[/red]",
    StageStatus.RUNNING:   "[yellow]…[/yellow]",
    StageStatus.PENDING:   "[dim]·[/dim]",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _collect_images(photos_dir: Path) -> List[Path]:
    paths = sorted(
        p for p in photos_dir.iterdir()
        if p.is_file() and p.suffix.lower() in _ALLOWED_SUFFIXES
    )
    return paths


def _copy_images(src_paths: List[Path], images_dir: Path) -> List[str]:
    saved = []
    for idx, src in enumerate(src_paths):
        dest_name = f"img_{idx:03d}{src.suffix.lower()}"
        shutil.copy2(src, images_dir / dest_name)
        saved.append(dest_name)
    return saved


def _summarize(stage_name: str, result: dict) -> str:
    """One-line human-readable summary of a stage result dict."""
    if stage_name == "validate":
        n = result.get("validated", 0)
        imgs = result.get("images", [])
        avg_mb = sum(i.get("size_mb", 0) for i in imgs) / n if n else 0
        return f"{n} images OK · avg {avg_mb:.1f} MB"

    if stage_name == "features":
        imgs = result.get("images", [])
        total_kp = sum(i.get("keypoints", 0) for i in imgs)
        avg_kp = total_kp // len(imgs) if imgs else 0
        return f"avg {avg_kp:,} keypoints/image"

    if stage_name == "matching":
        pairs = result.get("pairs_processed", 0)
        avg = result.get("avg_good_per_pair", 0)
        return f"{pairs} pairs · avg {avg:.0f} good matches"

    if stage_name == "sfm":
        n_reg = result.get("num_images_registered", "?")
        n_tot = result.get("num_images_total", "?")
        pts = result.get("num_points3d", 0)
        return f"{n_reg}/{n_tot} images · {pts:,} 3D points"

    return ""


def _run_stage_cli(
    job: Job,
    stage_name: str,
    fn,
    *args,
) -> Optional[dict]:
    """
    Run one pipeline stage, print a status line, and return the result dict.
    Returns None if the stage was skipped (placeholder) or failed.
    """
    t0 = time.monotonic()
    try:
        result = fn(*args)
        elapsed = time.monotonic() - t0

        if isinstance(result, dict) and result.get("status") == "placeholder":
            note = result.get("note", "not yet implemented")
            # Trim the note to fit on one line
            short = note.split(".")[0] if "." in note else note
            console.print(f"  [dim]↷  {stage_name:<12}skipped — {short}[/dim]")
            job.stages[stage_name].status = StageStatus.SKIPPED
            job.stages[stage_name].message = note
        else:
            summary = _summarize(stage_name, result) if isinstance(result, dict) else ""
            console.print(
                f"  [green]✓[/green]  {stage_name:<12}[dim]{elapsed:.1f}s[/dim]  {summary}"
            )
            job.stages[stage_name].status = StageStatus.COMPLETED
            job.stages[stage_name].message = str(result)
            if isinstance(result, dict) and "artifacts" in result:
                job.stages[stage_name].artifacts = list(result["artifacts"])

        storage.save_job(job)
        return result

    except Exception as exc:
        elapsed = time.monotonic() - t0
        console.print(
            f"  [red]✗[/red]  {stage_name:<12}[dim]{elapsed:.1f}s[/dim]  [red]{exc}[/red]"
        )
        job.stages[stage_name].status = StageStatus.FAILED
        job.stages[stage_name].message = str(exc)
        job.status = JobStatus.FAILED
        job.error = f"Stage '{stage_name}' failed: {exc}"
        storage.save_job(job)
        return None


# ---------------------------------------------------------------------------
# helium run
# ---------------------------------------------------------------------------

@app.command()
def run(
    photos: Path = typer.Argument(
        ...,
        help="Directory containing photos of the object.",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
    ),
    min_images: int = typer.Option(
        settings.min_images,
        "--min-images",
        help="Minimum number of images required.",
    ),
    max_images: int = typer.Option(
        settings.max_images,
        "--max-images",
        help="Maximum number of images allowed.",
    ),
) -> None:
    """
    Run photo-to-3D reconstruction on a directory of photos.

    Output is written to the Helium data directory
    (default: ~/.helium/jobs/<job-id>/).
    Override with HELIUM_DATA_DIR env var.
    """
    console.print(f"\n[bold]Helium[/bold] [dim]v{__version__}[/dim]\n")

    # --- Collect images ---
    image_paths = _collect_images(photos)

    if len(image_paths) < min_images:
        console.print(
            f"[red]Error:[/red] found {len(image_paths)} image(s) in [bold]{photos}[/bold], "
            f"need at least {min_images}."
        )
        raise typer.Exit(1)

    if len(image_paths) > max_images:
        console.print(
            f"[red]Error:[/red] found {len(image_paths)} images, "
            f"maximum is {max_images}. Remove some photos and try again."
        )
        raise typer.Exit(1)

    # --- Create job ---
    job = Job()
    storage.create_job_dirs(job.id)
    job.images = _copy_images(image_paths, storage.images_dir(job.id))
    job.image_count = len(job.images)
    job.status = JobStatus.RUNNING
    storage.save_job(job)

    console.print(
        f"  [dim]{job.image_count} photos[/dim]  ·  "
        f"[dim]job {job.id[:8]}…[/dim]\n"
    )

    # --- Run pipeline stages ---
    images_dir = storage.images_dir(job.id)
    artifacts_dir = storage.artifacts_dir(job.id)
    wall_start = time.monotonic()

    stages = [
        ("validate", validate.run,  images_dir, job.images),
        ("features", features.run,  images_dir, artifacts_dir, job.images),
        ("matching", matching.run,  images_dir, artifacts_dir, job.images),
        ("sfm",      sfm.run,       images_dir, artifacts_dir, job.images),
        ("dense",    dense.run,     artifacts_dir, job.images),
        ("export",   export.run,    artifacts_dir),
    ]

    for stage_name, fn, *args in stages:
        result = _run_stage_cli(job, stage_name, fn, *args)
        if job.status == JobStatus.FAILED:
            # Mark remaining stages as skipped
            stage_names = list(job.stages.keys())
            idx = stage_names.index(stage_name) + 1
            for name in stage_names[idx:]:
                job.stages[name].status = StageStatus.SKIPPED
            storage.save_job(job)
            break

        # Flag real reconstruction when sfm produced real output
        if stage_name == "sfm" and result and "artifacts" in result:
            job.real_reconstruction = True
            for rel in result["artifacts"]:
                name = Path(rel).name
                if name == "sparse.ply":
                    job.artifacts.sparse_ply = rel
                elif name == "cameras.json":
                    job.artifacts.cameras_json = rel
            storage.save_job(job)

    # --- Final status ---
    elapsed = time.monotonic() - wall_start

    if job.status != JobStatus.FAILED:
        job.status = JobStatus.COMPLETED
        storage.save_job(job)

    output_dir = storage.job_dir(job.id)
    console.print()

    if job.status == JobStatus.COMPLETED:
        console.print(f"  [green]Done[/green] in {elapsed:.1f}s")
    else:
        console.print(f"  [red]Failed[/red] after {elapsed:.1f}s  — {job.error}")

    console.print(f"  Output: [bold]{output_dir}[/bold]")
    console.print(
        f"\n  [dim]helium jobs show {job.id[:8]} — full detail[/dim]\n"
    )


# ---------------------------------------------------------------------------
# helium jobs list
# ---------------------------------------------------------------------------

@jobs_app.command("list")
def jobs_list() -> None:
    """List all reconstruction jobs, newest first."""
    all_jobs = []
    for job_id in storage.list_job_ids():
        job = storage.load_job(job_id)
        if job is not None:
            all_jobs.append(job)

    all_jobs.sort(key=lambda j: j.created_at, reverse=True)

    if not all_jobs:
        console.print("[dim]No jobs found. Run [bold]helium run <photos>[/bold] to start one.[/dim]")
        return

    table = Table(show_header=True, header_style="bold", box=None, padding=(0, 2))
    table.add_column("ID", style="dim", no_wrap=True)
    table.add_column("STATUS")
    table.add_column("IMAGES", justify="right")
    table.add_column("REAL RECON", justify="center")
    table.add_column("CREATED")

    _status_style = {
        JobStatus.COMPLETED: "green",
        JobStatus.FAILED:    "red",
        JobStatus.RUNNING:   "yellow",
        JobStatus.PENDING:   "dim",
    }

    for job in all_jobs:
        style = _status_style.get(job.status, "")
        age = _human_age(job.created_at)
        real = "✓" if job.real_reconstruction else "—"
        table.add_row(
            job.id[:12] + "…",
            f"[{style}]{job.status.value}[/{style}]",
            str(job.image_count),
            real,
            age,
        )

    console.print()
    console.print(table)
    console.print()


# ---------------------------------------------------------------------------
# helium jobs show <id>
# ---------------------------------------------------------------------------

@jobs_app.command("show")
def jobs_show(
    job_id: str = typer.Argument(..., help="Job ID or prefix."),
) -> None:
    """Show full status and per-stage detail for a job."""
    job = _find_job(job_id)

    _status_color = {
        JobStatus.COMPLETED: "green",
        JobStatus.FAILED:    "red",
        JobStatus.RUNNING:   "yellow",
        JobStatus.PENDING:   "dim",
    }
    color = _status_color.get(job.status, "")

    console.print()
    console.print(f"  [bold]Job[/bold]     {job.id}")
    console.print(f"  [bold]Status[/bold]  [{color}]{job.status.value}[/{color}]")
    console.print(f"  [bold]Images[/bold]  {job.image_count}")
    console.print(f"  [bold]Created[/bold] {job.created_at.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    if job.real_reconstruction:
        console.print("  [bold]SfM[/bold]     [green]real reconstruction[/green]")
    if job.error:
        console.print(f"  [bold]Error[/bold]   [red]{job.error}[/red]")

    console.print()
    console.print("  [bold]Stages[/bold]")
    for name, stage in job.stages.items():
        icon = _STAGE_ICON.get(stage.status, "·")
        duration = ""
        if stage.started_at and stage.completed_at:
            secs = (stage.completed_at - stage.started_at).total_seconds()
            duration = f"[dim]{secs:.1f}s[/dim]  "
        artifact_note = f"[dim]{len(stage.artifacts)} artifact(s)[/dim]" if stage.artifacts else ""
        console.print(f"    {icon}  {name:<12}{duration}{artifact_note}")

    console.print(f"\n  [bold]Output[/bold]  {storage.job_dir(job.id)}\n")


# ---------------------------------------------------------------------------
# helium jobs clean <id>
# ---------------------------------------------------------------------------

@jobs_app.command("clean")
def jobs_clean(
    job_id: str = typer.Argument(..., help="Job ID or prefix."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt."),
) -> None:
    """Delete a job and all its files."""
    job = _find_job(job_id)

    if not yes:
        typer.confirm(
            f"Delete job {job.id} and all its files?",
            abort=True,
        )

    storage.delete_job(job.id)
    console.print(f"[green]Deleted[/green] job {job.id[:12]}…")


# ---------------------------------------------------------------------------
# Internal utilities
# ---------------------------------------------------------------------------

def _find_job(prefix: str) -> Job:
    """Find a job by full ID or unambiguous prefix. Exits with error if not found."""
    # Exact match first
    job = storage.load_job(prefix)
    if job is not None:
        return job

    # Prefix match
    matches = [
        jid for jid in storage.list_job_ids()
        if jid.startswith(prefix)
    ]
    if len(matches) == 1:
        job = storage.load_job(matches[0])
        if job is not None:
            return job

    if len(matches) > 1:
        console.print(f"[red]Ambiguous prefix '{prefix}' matches {len(matches)} jobs. Use more characters.[/red]")
    else:
        console.print(f"[red]Job '{prefix}' not found.[/red]")
    raise typer.Exit(1)


def _human_age(dt: datetime) -> str:
    now = datetime.now(timezone.utc)
    secs = int((now - dt).total_seconds())
    if secs < 60:
        return f"{secs}s ago"
    if secs < 3600:
        return f"{secs // 60}m ago"
    if secs < 86400:
        return f"{secs // 3600}h ago"
    return f"{secs // 86400}d ago"
