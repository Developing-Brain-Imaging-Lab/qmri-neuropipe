"""Shared local-parallel execution for pipeline subject/session pairs."""

from __future__ import annotations

from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager, nullcontext
import inspect
import logging
import multiprocessing
import os
import sys
import threading
from typing import Any, Callable, Iterable, Iterator, Optional

from qmri_neuropipe.core.config import PipelineConfig

try:
    from rich import box
    from rich.console import Console
    from rich.layout import Layout
    from rich.live import Live
    from rich.panel import Panel
    from rich.progress import (
        BarColumn,
        Progress,
        SpinnerColumn,
        TextColumn,
        TimeRemainingColumn,
    )
    from rich.text import Text
except ImportError:  # Rich remains optional for programmatic pipeline use.
    Console = None
    Live = None


Task = tuple[str, Optional[str]]
Result = dict[str, Any]


def _rebuild_worker_config(config_dict: dict[str, Any]) -> PipelineConfig:
    """Rebuild a validated config and prevent nested parallel execution."""
    config = PipelineConfig.from_dict(config_dict)
    config.set("jobs", 1)
    return config


def _construct_pipeline(pipeline_cls: type, config: PipelineConfig, logger):
    """Pass an external logger only when the pipeline constructor accepts it."""
    parameters = inspect.signature(pipeline_cls).parameters.values()
    accepts_logger = any(
        parameter.name == "logger"
        or parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters
    )
    if accepts_logger:
        return pipeline_cls(config, logger=logger)
    return pipeline_cls(config)


def _worker_logger(job_id: int, config: PipelineConfig) -> logging.Logger:
    logger = logging.getLogger(f"qmri-neuropipe.worker.{job_id}")
    logger.setLevel(getattr(logging, config.log_level, logging.INFO))
    logger.handlers.clear()
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("[%(name)s] %(levelname)s: %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def _flush_stream(stream) -> None:
    try:
        stream.flush()
    except (AttributeError, OSError, ValueError):
        pass


def _restore_standard_streams(
    stdout_fd: int,
    stderr_fd: int,
    original_stdout,
    original_stderr,
) -> None:
    """Restore process FDs and Python stream objects after worker capture."""
    _flush_stream(sys.stdout)
    _flush_stream(sys.stderr)
    try:
        os.dup2(stdout_fd, 1)
        os.dup2(stderr_fd, 2)
    finally:
        os.close(stdout_fd)
        os.close(stderr_fd)
        sys.stdout = original_stdout
        sys.stderr = original_stderr


@contextmanager
def redirect_worker_output(
    log_queue,
    job_id: int,
) -> Iterator[None]:
    """Capture worker stdout/stderr and restore both reliably after each task."""
    if log_queue is None:
        yield
        return

    original_stdout = sys.stdout
    original_stderr = sys.stderr
    stdout_fd = os.dup(1)
    stderr_fd = os.dup(2)
    read_fd, write_fd = os.pipe()

    def forward_output() -> None:
        try:
            with os.fdopen(read_fd, "r", errors="replace") as stream:
                for line in stream:
                    if line.strip():
                        log_queue.put(("log", job_id, line.strip()))
        except (OSError, ValueError):
            return

    forwarder = threading.Thread(target=forward_output, daemon=True)
    forwarder.start()
    try:
        _flush_stream(original_stdout)
        _flush_stream(original_stderr)
        os.dup2(write_fd, 1)
        os.dup2(write_fd, 2)
        os.close(write_fd)
        write_fd = -1
        sys.stdout = os.fdopen(1, "w", buffering=1, closefd=False)
        sys.stderr = os.fdopen(2, "w", buffering=1, closefd=False)

        if Console is not None:
            from qmri_neuropipe.core import ui

            ui.console = Console(
                force_terminal=True,
                color_system="truecolor",
                soft_wrap=True,
                legacy_windows=False,
            )
        yield
    finally:
        if write_fd >= 0:
            os.close(write_fd)
        _restore_standard_streams(
            stdout_fd,
            stderr_fd,
            original_stdout,
            original_stderr,
        )
        forwarder.join(timeout=1)


def run_pipeline_worker(
    pipeline_cls: type,
    config_dict: dict[str, Any],
    subject: str,
    session: Optional[str],
    log_queue=None,
    slot_queue=None,
) -> Result:
    """Process one subject/session pair in a pool worker."""
    job_id = slot_queue.get() if slot_queue is not None else 0
    previous_worker_marker = os.environ.get("QMRI_PARALLEL_WORKER")
    os.environ["QMRI_PARALLEL_WORKER"] = "1"
    try:
        with redirect_worker_output(log_queue, job_id):
            config = _rebuild_worker_config(config_dict)
            gpu_ids = config.get("gpu_ids") or []
            if gpu_ids:
                os.environ["CUDA_VISIBLE_DEVICES"] = str(
                    gpu_ids[job_id % len(gpu_ids)]
                )

            logger = _worker_logger(job_id, config)
            if log_queue is not None:
                log_queue.put(
                    (
                        "info",
                        job_id,
                        (f"{subject} (ses-{session or 'N/A'})", "running"),
                    )
                )

            pipeline = _construct_pipeline(pipeline_cls, config, logger)
            if hasattr(pipeline, "_process_subject_with_subject_log"):
                pipeline._process_subject_with_subject_log(subject, session)
                result = {
                    "n_success": 1,
                    "n_failed": 0,
                    "n_skipped": 0,
                }
            else:
                result = dict(pipeline.run(pairs=[(subject, session)]))

            if log_queue is not None:
                status = "complete" if result.get("n_failed", 0) == 0 else "failed"
                log_queue.put(
                    ("info", job_id, (f"{subject} (Done)", status))
                )
            result.update({"subject": subject, "session": session})
            if result.get("n_failed", 0) and "error" not in result:
                failures = result.get("failures") or []
                result["error"] = (
                    failures[0].get("error", "Pipeline reported failure")
                    if failures
                    else "Pipeline reported failure"
                )
            return result
    except Exception as exc:
        if log_queue is not None:
            log_queue.put(
                ("info", job_id, (f"{subject} (Error)", "failed"))
            )
        return {
            "n_success": 0,
            "n_failed": 1,
            "n_skipped": 0,
            "error": str(exc),
            "subject": subject,
            "session": session,
        }
    finally:
        if previous_worker_marker is None:
            os.environ.pop("QMRI_PARALLEL_WORKER", None)
        else:
            os.environ["QMRI_PARALLEL_WORKER"] = previous_worker_marker
        if slot_queue is not None:
            slot_queue.put(job_id)


class ParallelUIState:
    """Rich presentation state for the shared parallel runner."""

    _COLORS = (
        "cyan",
        "magenta",
        "yellow",
        "green",
        "blue",
        "red",
        "bright_blue",
        "bright_magenta",
        "bright_cyan",
        "bright_yellow",
        "bright_green",
        "bright_red",
    )

    def __init__(self, jobs: int, total: int, console) -> None:
        self.jobs = jobs
        self.buffers = {index: deque(maxlen=10) for index in range(jobs)}
        self.job_info = {index: "[dim]Idle[/dim]" for index in range(jobs)}
        self.job_status = {index: "idle" for index in range(jobs)}
        self.lock = threading.Lock()
        self.tasks_done = 0
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=None),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=console,
            expand=True,
        )
        self.overall_task = self.progress.add_task(
            "[bold green]Total Progress",
            total=total,
        )

    def update_event(self, event) -> None:
        if event == "STOP":
            return
        event_type, job_id, payload = event
        with self.lock:
            if event_type == "log":
                self.buffers[job_id].append(payload)
            elif event_type == "info":
                if isinstance(payload, tuple):
                    self.job_info[job_id], self.job_status[job_id] = payload
                else:
                    self.job_info[job_id] = payload

    def advance(self) -> None:
        with self.lock:
            self.tasks_done += 1
            self.progress.update(self.overall_task, completed=self.tasks_done)

    def layout(self):
        status_icons = {
            "idle": "💤 [dim]Idle[/dim]",
            "running": "🔄 [bold cyan]Running[/bold cyan]",
            "complete": "✅ [bold green]Done[/bold green]",
            "failed": "❌ [bold red]ERROR[/bold red]",
        }
        with self.lock:
            columns = 2 if self.jobs > 1 else 1
            rows = (self.jobs + columns - 1) // columns
            row_layouts = []
            for row_index in range(rows):
                row = Layout(name=f"row_{row_index}")
                cells = []
                for column_index in range(columns):
                    job_id = row_index * columns + column_index
                    if job_id >= self.jobs:
                        cells.append(Layout())
                        continue
                    color = self._COLORS[job_id % len(self._COLORS)]
                    icon = status_icons.get(self.job_status[job_id], "•")
                    title = (
                        f"[{color}]Worker {job_id + 1}[/{color}] {icon} "
                        f"[bold]{self.job_info[job_id]}[/bold]"
                    )
                    content = Text("\n").join(
                        Text.from_ansi(line) for line in self.buffers[job_id]
                    )
                    cells.append(
                        Layout(
                            Panel(
                                content,
                                title=title,
                                border_style=color,
                                box=box.ROUNDED,
                                padding=(0, 1),
                            )
                        )
                    )
                row.split_row(*cells)
                row_layouts.append(row)

            body = Layout()
            body.split_column(*row_layouts)
            layout = Layout()
            layout.split_column(
                Layout(
                    Panel(
                        "[bold white on blue] qmri-neuropipe [/bold white on blue] "
                        "[blue]Parallel Processing Monitor[/blue]",
                        box=box.MINIMAL,
                    ),
                    size=3,
                ),
                Layout(body, name="main"),
                Layout(Panel(self.progress, box=box.MINIMAL), size=3),
            )
            return layout


def _failure(subject: str, session: Optional[str], error: Exception | str) -> Result:
    return {
        "n_success": 0,
        "n_failed": 1,
        "n_skipped": 0,
        "subject": subject,
        "session": session,
        "error": str(error),
    }


def _aggregate_results(results: Iterable[Result], n_skipped: int) -> Result:
    summary: Result = {
        "n_success": 0,
        "n_failed": 0,
        "n_skipped": n_skipped,
        "failures": [],
    }
    for result in results:
        summary["n_success"] += result.get("n_success", 0)
        summary["n_failed"] += result.get("n_failed", 0)
        summary["n_skipped"] += result.get("n_skipped", 0)
        if result.get("n_failed", 0):
            summary["failures"].append(
                {
                    "subject": result.get("subject"),
                    "session": result.get("session"),
                    "error": result.get("error", "Unknown error"),
                }
            )
    return summary


def _partition_tasks(
    tasks: Iterable[Task],
    should_skip: Optional[Callable[[str, Optional[str]], bool]],
    logger: logging.Logger,
) -> tuple[list[Task], int, list[Result]]:
    runnable = []
    skipped = 0
    failures = []
    for subject, session in tasks:
        try:
            if should_skip is not None and should_skip(subject, session):
                logger.info(
                    "Skipping %s %s (outputs already exist)",
                    subject,
                    session,
                )
                skipped += 1
            else:
                runnable.append((subject, session))
        except Exception as exc:
            logger.error("Skip check failed for %s %s: %s", subject, session, exc)
            failures.append(_failure(subject, session, exc))
    return runnable, skipped, failures


def _monitor_events(log_queue, state: ParallelUIState, live) -> None:
    while True:
        event = log_queue.get()
        if event == "STOP":
            return
        state.update_event(event)
        if live.is_started:
            live.update(state.layout())


def run_parallel(
    pipeline_cls: type,
    config: PipelineConfig,
    tasks: Iterable[Task],
    jobs: int,
    *,
    should_skip: Optional[Callable[[str, Optional[str]], bool]] = None,
    logger: Optional[logging.Logger] = None,
    show_ui: Optional[bool] = None,
) -> Result:
    """Run all non-cached pairs through the shared process-pool worker."""
    logger = logger or logging.getLogger("qmri-neuropipe.parallel")
    runnable, n_skipped, results = _partition_tasks(
        list(tasks),
        should_skip,
        logger,
    )
    if not runnable:
        return _aggregate_results(results, n_skipped)

    config_dict = config.to_dict()
    if show_ui is None:
        try:
            from qmri_neuropipe.core.ui import console

            show_ui = bool(Live is not None and console.is_terminal)
        except ImportError:
            show_ui = False

    needs_manager = bool(show_ui or config.get("gpu_ids"))
    manager_context = (
        multiprocessing.Manager() if needs_manager else nullcontext(None)
    )
    with manager_context as manager:
        slot_queue = manager.Queue() if manager is not None else None
        if slot_queue is not None:
            for job_id in range(jobs):
                slot_queue.put(job_id)
        log_queue = manager.Queue() if show_ui else None

        state = None
        live_context = None
        monitor = None
        if show_ui:
            from qmri_neuropipe.core.ui import console

            state = ParallelUIState(jobs, len(runnable), console)
            live_context = Live(
                state.layout(),
                console=console,
                refresh_per_second=4,
                screen=True,
            )

        with ProcessPoolExecutor(max_workers=jobs) as executor:
            futures = {
                executor.submit(
                    run_pipeline_worker,
                    pipeline_cls,
                    config_dict,
                    subject,
                    session,
                    log_queue,
                    slot_queue,
                ): (subject, session)
                for subject, session in runnable
            }

            if live_context is not None:
                with live_context as live:
                    monitor = threading.Thread(
                        target=_monitor_events,
                        args=(log_queue, state, live),
                        daemon=True,
                    )
                    monitor.start()
                    for future in as_completed(futures):
                        subject, session = futures[future]
                        try:
                            results.append(future.result())
                        except Exception as exc:
                            results.append(_failure(subject, session, exc))
                        state.advance()
                        live.update(state.layout())
            else:
                for future in as_completed(futures):
                    subject, session = futures[future]
                    try:
                        results.append(future.result())
                    except Exception as exc:
                        results.append(_failure(subject, session, exc))

        if log_queue is not None:
            log_queue.put("STOP")
        if monitor is not None:
            monitor.join(timeout=1)

    return _aggregate_results(results, n_skipped)
