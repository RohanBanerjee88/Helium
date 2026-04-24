"""
Helium CLI — local-first speech research orchestration.

Commands:
  helium run <audio-dir>       Run the audio pipeline on a directory of WAV files
  helium config                Show current configuration
  helium jobs list             List past jobs
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
from rich.table import Table

from helium import __version__
from helium.benchmarks.cli import app as _benchmark_app
from helium.config import settings
from helium.models.job import Job, JobStatus, StageStatus
from helium.pipeline.stages import convert, diarize, evaluate, export, extract, separate, validate
from helium.storage.local import storage

app = typer.Typer(
    name="helium",
    help="Local-first speech research orchestration.",
    add_completion=False,
    no_args_is_help=True,
)
jobs_app = typer.Typer(help="Manage past Helium jobs.")
app.add_typer(jobs_app, name="jobs")
app.add_typer(_benchmark_app, name="benchmark")

console = Console()

_ALLOWED_SUFFIXES = {".wav", ".wave"}
_ARTIFACT_FIELD_BY_STAGE = {
    "validate": "validation_report",
    "diarize": "diarization_output",
    "separate": "separation_output",
    "extract": "extraction_output",
    "convert": "conversion_output",
    "evaluate": "evaluation_report",
    "export": "export_manifest",
}
_STAGE_ICON = {
    StageStatus.COMPLETED: "[green]✓[/green]",
    StageStatus.SKIPPED: "[dim]↷[/dim]",
    StageStatus.FAILED: "[red]✗[/red]",
    StageStatus.RUNNING: "[yellow]…[/yellow]",
    StageStatus.PENDING: "[dim]·[/dim]",
}


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"helium {__version__}")
        raise typer.Exit()


@app.callback()
def _main_callback(
    version: Optional[bool] = typer.Option(
        None,
        "--version",
        "-V",
        callback=_version_callback,
        is_eager=True,
        help="Show version and exit.",
    ),
) -> None:
    pass


def _collect_audio(audio_dir: Path) -> List[Path]:
    return sorted(
        p for p in audio_dir.iterdir()
        if p.is_file() and p.suffix.lower() in _ALLOWED_SUFFIXES
    )


def _copy_audio(src_paths: List[Path], audio_dir: Path) -> List[str]:
    saved = []
    for idx, src in enumerate(src_paths):
        dest_name = f"clip_{idx:03d}{src.suffix.lower()}"
        shutil.copy2(src, audio_dir / dest_name)
        saved.append(dest_name)
    return saved


def _sync_job_artifacts(job: Job) -> None:
    for stage_name, field_name in _ARTIFACT_FIELD_BY_STAGE.items():
        stage = job.stages.get(stage_name)
        setattr(job.artifacts, field_name, stage.artifacts[0] if stage and stage.artifacts else None)

    job.model_outputs_ready = any(
        job.stages[name].status == StageStatus.COMPLETED and bool(job.stages[name].artifacts)
        for name in ("diarize", "separate", "convert")
    )


def _summarize(stage_name: str, result: dict) -> str:
    if stage_name == "validate":
        return (
            f"{result.get('validated', 0)} clip(s) · "
            f"{result.get('total_duration_seconds', 0):.1f}s"
        )
    if stage_name == "evaluate":
        computed = result.get("metrics_computed", 0)
        planned = len(result.get("metrics_planned", []))
        return f"{computed}/{planned} metric(s) computed"
    if stage_name == "export":
        return f"{result.get('artifact_count', 0)} artifact(s)"
    return ""


def _run_stage(job: Job, stage_name: str, fn, *args) -> Optional[dict]:
    started = time.monotonic()
    stage = job.stages[stage_name]
    stage.status = StageStatus.RUNNING
    stage.started_at = datetime.now(timezone.utc)
    storage.save_job(job)

    with console.status(f"  [dim]{stage_name}[/dim]", spinner="dots"):
        try:
            result = fn(*args)
        except Exception as exc:
            stage.status = StageStatus.FAILED
            stage.completed_at = datetime.now(timezone.utc)
            stage.message = str(exc)
            job.status = JobStatus.FAILED
            job.error = f"Stage '{stage_name}' failed: {exc}"
            storage.save_job(job)
            console.print(
                f"  [red]✗[/red]  {stage_name:<10}[dim]{time.monotonic() - started:.1f}s[/dim]  "
                f"[red]{exc}[/red]"
            )
            return None

    elapsed = time.monotonic() - started
    if isinstance(result, dict) and result.get("status") == "placeholder":
        stage.status = StageStatus.SKIPPED
        stage.message = result.get("note", "Not yet implemented.")
        if "artifacts" in result:
            stage.artifacts = list(result["artifacts"])
        short = stage.message.split(".")[0]
        console.print(f"  [dim]↷  {stage_name:<10}{elapsed:.1f}s  {short}[/dim]")
    else:
        stage.status = StageStatus.COMPLETED
        stage.message = str(result)
        if isinstance(result, dict) and "artifacts" in result:
            stage.artifacts = list(result["artifacts"])
        console.print(
            f"  [green]✓[/green]  {stage_name:<10}[dim]{elapsed:.1f}s[/dim]  "
            f"{_summarize(stage_name, result) if isinstance(result, dict) else ''}"
        )

    stage.completed_at = datetime.now(timezone.utc)
    _sync_job_artifacts(job)
    storage.save_job(job)
    return result if isinstance(result, dict) else None


@app.command()
def run(
    audio: Path = typer.Argument(
        ...,
        help="Directory containing WAV files.",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
    ),
    min_audio_files: int = typer.Option(
        settings.min_audio_files,
        "--min-audio-files",
        help="Minimum number of audio files required.",
    ),
    max_audio_files: int = typer.Option(
        settings.max_audio_files,
        "--max-audio-files",
        help="Maximum number of audio files allowed.",
    ),
    target_speakers: int = typer.Option(
        settings.target_speakers,
        "--target-speakers",
        help="Expected number of speakers in the mixture.",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Copy artifacts to this directory when done.",
    ),
) -> None:
    """
    Run the speech research pipeline on a directory of WAV files.
    """
    console.print(f"\n[bold]Helium[/bold] [dim]v{__version__}[/dim]\n")

    audio_paths = _collect_audio(audio)
    if len(audio_paths) < min_audio_files:
        console.print(
            f"[red]Error:[/red] found {len(audio_paths)} audio file(s) in [bold]{audio}[/bold], "
            f"need at least {min_audio_files}."
        )
        raise typer.Exit(1)
    if len(audio_paths) > max_audio_files:
        console.print(
            f"[red]Error:[/red] found {len(audio_paths)} audio file(s), "
            f"maximum is {max_audio_files}."
        )
        raise typer.Exit(1)

    job = Job(target_speakers=target_speakers)
    storage.create_job_dirs(job.id)
    job.audio_files = _copy_audio(audio_paths, storage.audio_dir(job.id))
    job.audio_count = len(job.audio_files)
    job.status = JobStatus.RUNNING
    storage.save_job(job)

    console.print(
        f"  [dim]{job.audio_count} clip(s)[/dim]  ·  "
        f"[dim]{target_speakers} target speaker(s)[/dim]  ·  "
        f"[dim]job {job.id[:8]}…[/dim]\n"
    )

    audio_dir = storage.audio_dir(job.id)
    artifacts_dir = storage.artifacts_dir(job.id)
    stages = [
        ("validate", validate.run, audio_dir, artifacts_dir, job.audio_files),
        ("diarize", diarize.run, audio_dir, artifacts_dir, job.audio_files, job.target_speakers),
        ("separate", separate.run, audio_dir, artifacts_dir, job.audio_files, job.target_speakers),
        ("extract", extract.run, audio_dir, artifacts_dir, job.audio_files, job.target_speakers),
        ("convert", convert.run, audio_dir, artifacts_dir, job.audio_files, job.target_speakers),
        ("evaluate", evaluate.run, artifacts_dir, job.audio_files, job.target_speakers),
        ("export", export.run, artifacts_dir, job.id),
    ]

    started = time.monotonic()
    for stage_name, fn, *args in stages:
        _run_stage(job, stage_name, fn, *args)
        if job.status == JobStatus.FAILED:
            for name, stage in job.stages.items():
                if stage.status == StageStatus.PENDING:
                    stage.status = StageStatus.SKIPPED
            storage.save_job(job)
            break

    if job.status != JobStatus.FAILED:
        job.status = JobStatus.COMPLETED
        _sync_job_artifacts(job)
        storage.save_job(job)

    console.print()
    elapsed = time.monotonic() - started
    if job.status == JobStatus.COMPLETED:
        console.print(f"  [green]Done[/green] in {elapsed:.1f}s")
    else:
        console.print(f"  [red]Failed[/red] after {elapsed:.1f}s  — {job.error}")

    job_dir = storage.job_dir(job.id)
    console.print(f"  Output: [bold]{job_dir}[/bold]")

    if output is not None and job.status == JobStatus.COMPLETED:
        output.mkdir(parents=True, exist_ok=True)
        shutil.copytree(str(storage.artifacts_dir(job.id)), str(output), dirs_exist_ok=True)
        console.print(f"  Artifacts copied to: [bold]{output}[/bold]")

    console.print(f"\n  [dim]helium jobs show {job.id[:8]} — full detail[/dim]\n")


@app.command("shell")
def shell_cmd() -> None:
    from helium.cli.shell import run_shell

    run_shell()


@app.command("explode", hidden=True)
def explode_cmd() -> None:
    from helium.cli.shell import run_shell

    run_shell()


@app.command("config")
def config_cmd() -> None:
    rows = [
        ("data_dir", str(settings.data_dir), "HELIUM_DATA_DIR"),
        ("min_audio_files", str(settings.min_audio_files), "HELIUM_MIN_AUDIO_FILES"),
        ("max_audio_files", str(settings.max_audio_files), "HELIUM_MAX_AUDIO_FILES"),
        ("max_audio_size_mb", str(settings.max_audio_size_mb), "HELIUM_MAX_AUDIO_SIZE_MB"),
        ("target_speakers", str(settings.target_speakers), "HELIUM_TARGET_SPEAKERS"),
        ("diarization_backend", settings.diarization_backend, "HELIUM_DIARIZATION_BACKEND"),
        ("separation_backend", settings.separation_backend, "HELIUM_SEPARATION_BACKEND"),
        ("conversion_backend", settings.conversion_backend, "HELIUM_CONVERSION_BACKEND"),
    ]

    table = Table(box=None, show_header=True, header_style="bold", padding=(0, 2))
    table.add_column("SETTING")
    table.add_column("VALUE")
    table.add_column("ENV VAR", style="dim")
    table.add_column("SOURCE", style="dim")

    for name, value, env_var in rows:
        from_env = env_var in os.environ
        table.add_row(name, f"[bold]{value}[/bold]" if from_env else value, env_var, "env" if from_env else "default")

    console.print()
    console.print(table)
    console.print()


@jobs_app.command("list")
def jobs_list() -> None:
    jobs = []
    for job_id in storage.list_job_ids():
        job = storage.load_job(job_id)
        if job is not None:
            jobs.append(job)

    jobs.sort(key=lambda item: item.created_at, reverse=True)
    if not jobs:
        console.print("[dim]No jobs found. Run [bold]helium run <audio-dir>[/bold] to start one.[/dim]")
        return

    table = Table(show_header=True, header_style="bold", box=None, padding=(0, 2))
    table.add_column("ID", style="dim", no_wrap=True)
    table.add_column("STATUS")
    table.add_column("AUDIO", justify="right")
    table.add_column("MODEL OUT", justify="center")
    table.add_column("CREATED")

    status_style = {
        JobStatus.COMPLETED: "green",
        JobStatus.FAILED: "red",
        JobStatus.RUNNING: "yellow",
        JobStatus.PENDING: "dim",
    }

    for job in jobs:
        table.add_row(
            job.id[:12] + "…",
            f"[{status_style.get(job.status, '')}]{job.status.value}[/{status_style.get(job.status, '')}]",
            str(job.audio_count),
            "✓" if job.model_outputs_ready else "—",
            _human_age(job.created_at),
        )

    console.print()
    console.print(table)
    console.print()


@jobs_app.command("show")
def jobs_show(job_id: str = typer.Argument(..., help="Job ID or prefix.")) -> None:
    job = _find_job(job_id)
    status_color = {
        JobStatus.COMPLETED: "green",
        JobStatus.FAILED: "red",
        JobStatus.RUNNING: "yellow",
        JobStatus.PENDING: "dim",
    }.get(job.status, "dim")

    console.print()
    console.print(f"  [bold]Job[/bold]       {job.id}")
    console.print(f"  [bold]Status[/bold]    [{status_color}]{job.status.value}[/{status_color}]")
    console.print(f"  [bold]Audio[/bold]     {job.audio_count}")
    console.print(f"  [bold]Speakers[/bold]  {job.target_speakers}")
    console.print(f"  [bold]Created[/bold]   {job.created_at.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    console.print(f"  [bold]Models[/bold]    {'ready' if job.model_outputs_ready else 'scaffold only'}")
    if job.error:
        console.print(f"  [bold]Error[/bold]     [red]{job.error}[/red]")

    console.print()
    console.print("  [bold]Stages[/bold]")
    for name, stage in job.stages.items():
        icon = _STAGE_ICON.get(stage.status, "·")
        duration = ""
        if stage.started_at and stage.completed_at:
            duration = f"[dim]{(stage.completed_at - stage.started_at).total_seconds():.1f}s[/dim]  "
        artifact_note = f"[dim]{len(stage.artifacts)} artifact(s)[/dim]" if stage.artifacts else ""
        console.print(f"    {icon}  {name:<10}{duration}{artifact_note}")

    console.print(f"\n  [bold]Output[/bold]    {storage.job_dir(job.id)}\n")


@jobs_app.command("open")
def jobs_open(job_id: str = typer.Argument(..., help="Job ID or prefix.")) -> None:
    job = _find_job(job_id)
    job_dir = storage.job_dir(job.id)

    if sys.platform == "darwin":
        subprocess.run(["open", str(job_dir)], check=False)
    elif sys.platform == "win32":
        os.startfile(str(job_dir))  # type: ignore[attr-defined]
    else:
        subprocess.run(["xdg-open", str(job_dir)], check=False)

    console.print(f"  Opened [bold]{job_dir}[/bold]")


@jobs_app.command("clean")
def jobs_clean(
    job_id: str = typer.Argument(..., help="Job ID or prefix."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt."),
) -> None:
    job = _find_job(job_id)
    if not yes:
        typer.confirm(f"Delete job {job.id} and all its files?", abort=True)
    storage.delete_job(job.id)
    console.print(f"[green]Deleted[/green] job {job.id[:12]}…")


def _find_job(prefix: str) -> Job:
    job = storage.load_job(prefix)
    if job is not None:
        return job

    matches = [job_id for job_id in storage.list_job_ids() if job_id.startswith(prefix)]
    if len(matches) == 1:
        match = storage.load_job(matches[0])
        if match is not None:
            return match

    if len(matches) > 1:
        console.print(f"[red]Ambiguous prefix '{prefix}' matches {len(matches)} jobs.[/red]")
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
