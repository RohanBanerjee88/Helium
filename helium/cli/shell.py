"""
Helium interactive shell.

A persistent REPL session for local speech research runs. It tab-completes
commands and job IDs, keeps history across sessions, and dispatches to the
same functions used by the one-shot CLI commands.

Launch with:  helium shell
"""

import shlex
from pathlib import Path
from typing import Optional

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.history import FileHistory
from prompt_toolkit.styles import Style
from rich.console import Console
from rich.table import Table

from helium import __version__
from helium.config import settings
from helium.storage.local import storage

console = Console()

_COMMANDS: dict[str, str] = {
    "run":    "<audio-dir> [--min-audio-files N] [--max-audio-files N] [--target-speakers N] [--output <dir>]",
    "jobs":   "list all speech research jobs",
    "show":   "<job-id>  — full stage detail",
    "open":   "<job-id>  — open job dir in file manager",
    "clean":  "<job-id> [-y]  — delete a job and its files",
    "config": "show current settings and env vars",
    "help":   "show this help",
    "exit":   "exit the shell  (also: quit · Ctrl+D)",
}

_STYLE = Style.from_dict({
    "prompt.name":  "#7c3aed bold",
    "prompt.arrow": "#6b7280",
})

_PROMPT = FormattedText([
    ("class:prompt.name",  "helium"),
    ("class:prompt.arrow", "> "),
])


# ---------------------------------------------------------------------------
# Tab completer
# ---------------------------------------------------------------------------

class _Completer(Completer):
    _JOB_CMDS = {"show", "open", "clean"}

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        parts = text.split()

        # First word → complete command names
        if not parts or (len(parts) == 1 and not text.endswith(" ")):
            word = parts[0] if parts else ""
            for cmd in _COMMANDS:
                if cmd.startswith(word):
                    yield Completion(cmd, start_position=-len(word))
            return

        cmd = parts[0].lower()

        # Second word for job commands → complete job IDs with metadata
        if cmd in self._JOB_CMDS and (
            len(parts) == 1 or (len(parts) == 2 and not text.endswith(" "))
        ):
            partial = parts[1] if len(parts) == 2 else ""
            for job_id in storage.list_job_ids():
                if not job_id.startswith(partial):
                    continue
                job = storage.load_job(job_id)
                meta = (
                    f"{job.status.value} · {job.audio_count} clip(s)" if job else ""
                )
                yield Completion(
                    job_id[:12],
                    start_position=-len(partial),
                    display_meta=meta,
                )


# ---------------------------------------------------------------------------
# Command dispatch
# ---------------------------------------------------------------------------

def _call(fn) -> None:
    """Call a CLI function, swallowing SystemExit (typer.Exit) and KeyboardInterrupt."""
    try:
        fn()
    except SystemExit:
        pass
    except KeyboardInterrupt:
        console.print("\n[dim]interrupted[/dim]")


def _dispatch_run(rest: list[str]) -> None:
    from helium.cli.main import run as cmd_run

    audio_dir: Optional[Path] = None
    min_audio_files: int = settings.min_audio_files
    max_audio_files: int = settings.max_audio_files
    target_speakers: int = settings.target_speakers
    output: Optional[Path] = None

    i = 0
    while i < len(rest):
        a = rest[i]
        if a == "--min-audio-files" and i + 1 < len(rest):
            try:
                min_audio_files = int(rest[i + 1])
            except ValueError:
                console.print("[red]Error:[/red] --min-audio-files requires an integer")
                return
            i += 2
        elif a == "--max-audio-files" and i + 1 < len(rest):
            try:
                max_audio_files = int(rest[i + 1])
            except ValueError:
                console.print("[red]Error:[/red] --max-audio-files requires an integer")
                return
            i += 2
        elif a == "--target-speakers" and i + 1 < len(rest):
            try:
                target_speakers = int(rest[i + 1])
            except ValueError:
                console.print("[red]Error:[/red] --target-speakers requires an integer")
                return
            i += 2
        elif a in ("--output", "-o") and i + 1 < len(rest):
            output = Path(rest[i + 1])
            i += 2
        elif not a.startswith("-"):
            audio_dir = Path(a)
            i += 1
        else:
            console.print(f"[dim]Unknown option skipped: {a}[/dim]")
            i += 1

    if audio_dir is None:
        console.print(
            "[dim]Usage: run <audio-dir> [--min-audio-files N] [--max-audio-files N] "
            "[--target-speakers N] [--output <dir>][/dim]"
        )
        return

    if not audio_dir.exists():
        console.print(f"[red]Error:[/red] path not found: {audio_dir}")
        return

    if not audio_dir.is_dir():
        console.print(f"[red]Error:[/red] not a directory: {audio_dir}")
        return

    _call(lambda: cmd_run(
        audio=audio_dir,
        min_audio_files=min_audio_files,
        max_audio_files=max_audio_files,
        target_speakers=target_speakers,
        output=output,
    ))


def _dispatch(line: str) -> bool:
    """Parse and execute one shell command. Returns False to signal exit."""
    from helium.cli import main as cli

    try:
        parts = shlex.split(line)
    except ValueError as exc:
        console.print(f"[red]Parse error:[/red] {exc}")
        return True

    if not parts:
        return True

    cmd, *rest = parts
    cmd = cmd.lower()

    if cmd in ("exit", "quit", "q"):
        return False

    elif cmd == "help":
        _print_help()

    elif cmd == "config":
        _call(cli.config_cmd)

    elif cmd in ("jobs", "list"):
        _call(cli.jobs_list)

    elif cmd == "show":
        if not rest:
            console.print("[dim]Usage: show <job-id>[/dim]")
        else:
            job_id = rest[0]
            _call(lambda: cli.jobs_show(job_id))

    elif cmd == "open":
        if not rest:
            console.print("[dim]Usage: open <job-id>[/dim]")
        else:
            job_id = rest[0]
            _call(lambda: cli.jobs_open(job_id))

    elif cmd == "clean":
        if not rest:
            console.print("[dim]Usage: clean <job-id> [-y][/dim]")
        else:
            yes = "-y" in rest or "--yes" in rest
            job_id = next((a for a in rest if not a.startswith("-")), None)
            if job_id:
                _call(lambda: cli.jobs_clean(job_id, yes=yes))

    elif cmd == "run":
        _dispatch_run(rest)

    else:
        console.print(
            f"[red]Unknown command:[/red] {cmd!r}  —  "
            "type [bold]help[/bold] to see available commands"
        )

    return True


def _print_help() -> None:
    t = Table(box=None, padding=(0, 3), show_header=False)
    t.add_column("CMD", style="bold", no_wrap=True)
    t.add_column("USAGE", style="dim")
    for cmd, desc in _COMMANDS.items():
        t.add_row(cmd, desc)
    console.print()
    console.print(t)
    console.print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_shell() -> None:
    """Start the interactive Helium shell."""
    data_dir = Path(settings.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    session: PromptSession = PromptSession(
        history=FileHistory(str(data_dir / "shell_history")),
        completer=_Completer(),
        complete_while_typing=True,
        style=_STYLE,
    )

    n_jobs = len(list(storage.list_job_ids()))
    jobs_note = f"{n_jobs} job(s) on disk" if n_jobs else "no jobs yet"

    console.print(
        f"\n  [bold]⬡  Helium[/bold] [dim]v{__version__}[/dim]"
        f"  ·  [dim]{settings.data_dir}[/dim]"
        f"  ·  [dim]{jobs_note}[/dim]"
    )
    console.print(
        "  [dim]tab-completes commands and job IDs  ·  "
        "Ctrl+D to exit[/dim]\n"
    )

    while True:
        try:
            line = session.prompt(_PROMPT, style=_STYLE)
        except KeyboardInterrupt:
            continue  # Ctrl+C clears the current line, keeps the session
        except EOFError:
            break  # Ctrl+D

        line = line.strip()
        if not line:
            continue

        if not _dispatch(line):
            break

    console.print("\n  [dim]bye[/dim]\n")
