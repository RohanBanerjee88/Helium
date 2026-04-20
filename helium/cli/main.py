"""
Helium CLI — local photo-to-3D reconstruction.

Commands:
  helium run <photos>          Run reconstruction on a directory of photos
  helium config                Show current configuration
  helium jobs list             List all past jobs
  helium jobs show <id>        Show full detail for one job
  helium jobs open <id>        Open job output directory in file manager
  helium jobs clean <id>       Delete a job and its files
"""

import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
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
    no_args_is_help=True,
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
# Version flag
# ---------------------------------------------------------------------------

def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"helium {__version__}")
        raise typer.Exit()


@app.callback()
def _main_callback(
    version: Optional[bool] = typer.Option(
        None,
        "--version", "-V",
        callback=_version_callback,
        is_eager=True,
        help="Show version and exit.",
    ),
) -> None:
    pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _collect_images(photos_dir: Path) -> List[Path]:
    return sorted(
        p for p in photos_dir.iterdir()
        if p.is_file() and p.suffix.lower() in _ALLOWED_SUFFIXES
    )


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


def _print_verbose(stage_name: str, result: dict) -> None:
    """Print a per-item detail table for stages that produce per-item data."""
    if stage_name == "validate":
        imgs = result.get("images", [])
        if not imgs:
            return
        t = Table(box=None, padding=(0, 3), show_header=True, header_style="dim")
        t.add_column("FILE", style="dim")
        t.add_column("W", justify="right")
        t.add_column("H", justify="right")
        t.add_column("SIZE", justify="right")
        for img in imgs:
            t.add_row(
                img["filename"],
                str(img["width"]),
                str(img["height"]),
                f"{img['size_mb']:.2f} MB",
            )
        console.print(t)

    elif stage_name == "features":
        imgs = result.get("images", [])
        if not imgs:
            return
        t = Table(box=None, padding=(0, 3), show_header=True, header_style="dim")
        t.add_column("FILE", style="dim")
        t.add_column("KEYPOINTS", justify="right")
        for img in imgs:
            kp = img["keypoints"]
            bar = "█" * min(20, kp // 100)
            t.add_row(img["filename"], f"{kp:,}  [dim]{bar}[/dim]")
        console.print(t)


def _make_progress_bar(description: str, total: int) -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn(f"  [dim]{description:<12}[/dim]"),
        BarColumn(bar_width=24),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    )


def _run_stage(
    job: Job,
    stage_name: str,
    fn,
    *args,
    verbose: bool = False,
    progress_total: Optional[int] = None,
) -> Optional[dict]:
    """
    Run one pipeline stage.

    If progress_total is set, fn must accept a progress_callback keyword arg
    and a rich progress bar is shown. Otherwise a spinner is used.
    Returns the result dict, or None if the stage failed.
    """
    t0 = time.monotonic()
    caught_exc: Optional[Exception] = None
    result: Optional[dict] = None

    if progress_total is not None:
        with _make_progress_bar(stage_name, progress_total) as prog:
            task = prog.add_task("", total=progress_total)

            def _cb(*_args):
                prog.advance(task)

            try:
                result = fn(*args, progress_callback=_cb)
            except Exception as exc:
                caught_exc = exc
    else:
        with console.status(f"  [dim]{stage_name}[/dim]", spinner="dots"):
            try:
                result = fn(*args)
            except Exception as exc:
                caught_exc = exc

    elapsed = time.monotonic() - t0

    if caught_exc is not None:
        console.print(
            f"  [red]✗[/red]  {stage_name:<12}[dim]{elapsed:.1f}s[/dim]  [red]{caught_exc}[/red]"
        )
        job.stages[stage_name].status = StageStatus.FAILED
        job.stages[stage_name].message = str(caught_exc)
        job.status = JobStatus.FAILED
        job.error = f"Stage '{stage_name}' failed: {caught_exc}"
        storage.save_job(job)
        return None

    if isinstance(result, dict) and result.get("status") == "placeholder":
        note = result.get("note", "not yet implemented")
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
        if verbose and isinstance(result, dict):
            _print_verbose(stage_name, result)

    storage.save_job(job)
    return result


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
    output: Optional[Path] = typer.Option(
        None,
        "--output", "-o",
        help="Copy artifacts to this directory when done.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose", "-v",
        help="Show per-image/per-pair detail after each completed stage.",
    ),
) -> None:
    """
    Run photo-to-3D reconstruction on a directory of photos.

    Output is saved under the Helium data directory
    (default: ~/.helium/jobs/<job-id>/).
    Override the data root with HELIUM_DATA_DIR, or use --output to copy
    finished artifacts to a specific path.
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

    # --- Run pipeline ---
    images_dir = storage.images_dir(job.id)
    artifacts_dir = storage.artifacts_dir(job.id)
    wall_start = time.monotonic()

    n = len(job.images)
    n_pairs = n * (n - 1) // 2

    stages = [
        ("validate", validate.run,  None,     images_dir, job.images),
        ("features", features.run,  n,        images_dir, artifacts_dir, job.images),
        ("matching", matching.run,  n_pairs,  images_dir, artifacts_dir, job.images),
        ("sfm",      sfm.run,       None,     images_dir, artifacts_dir, job.images),
        ("dense",    dense.run,     None,     artifacts_dir, job.images),
        ("export",   export.run,    None,     artifacts_dir),
    ]

    for stage_name, fn, prog_total, *args in stages:
        result = _run_stage(
            job, stage_name, fn, *args,
            verbose=verbose,
            progress_total=prog_total,
        )

        if job.status == JobStatus.FAILED:
            stage_names = list(job.stages.keys())
            idx = stage_names.index(stage_name) + 1
            for name in stage_names[idx:]:
                job.stages[name].status = StageStatus.SKIPPED
            storage.save_job(job)
            break

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

    job_dir = storage.job_dir(job.id)
    console.print()

    if job.status == JobStatus.COMPLETED:
        console.print(f"  [green]Done[/green] in {elapsed:.1f}s")
    else:
        console.print(f"  [red]Failed[/red] after {elapsed:.1f}s  — {job.error}")

    console.print(f"  Output: [bold]{job_dir}[/bold]")

    if output is not None and job.status == JobStatus.COMPLETED:
        artifacts_src = storage.artifacts_dir(job.id)
        output.mkdir(parents=True, exist_ok=True)
        if any(artifacts_src.iterdir()):
            shutil.copytree(str(artifacts_src), str(output), dirs_exist_ok=True)
            console.print(f"  Artifacts copied to: [bold]{output}[/bold]")
        else:
            console.print("  [dim]No artifacts to copy (all pipeline stages were placeholders).[/dim]")

    console.print(
        f"\n  [dim]helium jobs show {job.id[:8]} — full detail[/dim]\n"
    )


# ---------------------------------------------------------------------------
# helium config
# ---------------------------------------------------------------------------

@app.command("config")
def config_cmd() -> None:
    """Show current Helium configuration and the environment variables that control each setting."""
    rows = [
        ("data_dir",           str(settings.data_dir),          "HELIUM_DATA_DIR"),
        ("min_images",         str(settings.min_images),         "HELIUM_MIN_IMAGES"),
        ("max_images",         str(settings.max_images),         "HELIUM_MAX_IMAGES"),
        ("max_image_size_mb",  str(settings.max_image_size_mb),  "HELIUM_MAX_IMAGE_SIZE_MB"),
    ]

    t = Table(box=None, show_header=True, header_style="bold", padding=(0, 2))
    t.add_column("SETTING")
    t.add_column("VALUE")
    t.add_column("ENV VAR", style="dim")
    t.add_column("SOURCE", style="dim")

    for name, value, env_var in rows:
        from_env = env_var in os.environ
        source = "env" if from_env else "default"
        value_styled = f"[bold]{value}[/bold]" if from_env else value
        t.add_row(name, value_styled, env_var, source)

    console.print()
    console.print(t)
    console.print()


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
        console.print(
            "[dim]No jobs found. Run [bold]helium run <photos>[/bold] to start one.[/dim]"
        )
        return

    t = Table(show_header=True, header_style="bold", box=None, padding=(0, 2))
    t.add_column("ID", style="dim", no_wrap=True)
    t.add_column("STATUS")
    t.add_column("IMAGES", justify="right")
    t.add_column("REAL RECON", justify="center")
    t.add_column("CREATED")

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
        t.add_row(
            job.id[:12] + "…",
            f"[{style}]{job.status.value}[/{style}]",
            str(job.image_count),
            real,
            age,
        )

    console.print()
    console.print(t)
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
        artifact_note = (
            f"[dim]{len(stage.artifacts)} artifact(s)[/dim]" if stage.artifacts else ""
        )
        console.print(f"    {icon}  {name:<12}{duration}{artifact_note}")

    console.print(f"\n  [bold]Output[/bold]  {storage.job_dir(job.id)}\n")


# ---------------------------------------------------------------------------
# helium jobs open <id>
# ---------------------------------------------------------------------------

@jobs_app.command("open")
def jobs_open(
    job_id: str = typer.Argument(..., help="Job ID or prefix."),
) -> None:
    """Open the job output directory in the system file manager."""
    job = _find_job(job_id)
    job_dir = storage.job_dir(job.id)

    if sys.platform == "darwin":
        subprocess.run(["open", str(job_dir)], check=False)
    elif sys.platform == "win32":
        os.startfile(str(job_dir))  # type: ignore[attr-defined]
    else:
        subprocess.run(["xdg-open", str(job_dir)], check=False)

    console.print(f"  Opened [bold]{job_dir}[/bold]")


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
    job = storage.load_job(prefix)
    if job is not None:
        return job

    matches = [jid for jid in storage.list_job_ids() if jid.startswith(prefix)]
    if len(matches) == 1:
        job = storage.load_job(matches[0])
        if job is not None:
            return job

    if len(matches) > 1:
        console.print(
            f"[red]Ambiguous prefix '{prefix}' matches {len(matches)} jobs. "
            "Use more characters.[/red]"
        )
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
