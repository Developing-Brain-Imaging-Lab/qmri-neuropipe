"""
CLI for qmri-neuropipe with robust config/CLI argument merging.

This module provides:
- Config file loading (JSON/YAML)
- Command line argument parsing
- Intelligent merging (CLI overrides config)
- Required argument validation
- Clear error messages
"""

from __future__ import annotations
import sys
import warnings
from pathlib import Path
from typing import List, Optional
import typer
from rich.table import Table
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich import box
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
from rich.console import Group
from rich.text import Text
from collections import deque
import threading
import multiprocessing

# Global Warning Silencing
# 1. Suppress resource_tracker warnings (benign semaphore "leaks" at shutdown in some envs)
warnings.filterwarnings("ignore", category=UserWarning, module="multiprocessing.resource_tracker")
warnings.filterwarnings("ignore", message=".*resource_tracker: There appear to be.*leaked semaphore.*", category=UserWarning)

# 2. Suppress solution inaccuracy warnings (e.g. from CVXPY used in MAPMRI)
warnings.filterwarnings("ignore", message=".*Solution may be inaccurate.*", category=UserWarning)
# 3. Suppress DIPY DKI overflow warnings
warnings.filterwarnings("ignore", message=".*overflow encountered in exp.*", category=RuntimeWarning)
warnings.filterwarnings("ignore", message=".*invalid value encountered in log.*", category=RuntimeWarning)


from qmri_neuropipe.core import PipelineConfig
from qmri_neuropipe.core.exceptions import ConfigurationError
from qmri_neuropipe.core.ui import console
from qmri_neuropipe.io import DataLoader


app = typer.Typer(add_completion=False, help="qmri-neuropipe: Robust BIDS MRI processing pipeline")




def _print_args(config: PipelineConfig, **extra_kwargs):
    """
    Display pipeline arguments in a formatted table.
    
    Args:
        config: Pipeline configuration object
        **extra_kwargs: Additional arguments to display
    """
    table = Table(title="qmri-neuropipe Configuration")
    table.add_column("Parameter", style="bold")
    table.add_column("Value", style="cyan")
    
    # Core paths
    table.add_row("bids_dir", str(config.bids_dir) if config.bids_dir else "Not set")
    table.add_row("output_dir", str(config.output_dir) if config.output_dir else "Not set")
    table.add_row("work_dir", str(config.work_dir) if config.work_dir else "Not set")
    
    # Subject/session selection
    if config.participant_label:
        table.add_row("participant_label", str(config.participant_label))
    if config.session_label:
        table.add_row("session_label", str(config.session_label))
    
    # Computational resources
    table.add_row("n_cpus", str(config.n_cpus))
    table.add_row("memory_gb", str(config.memory_gb))
    table.add_row("use_gpu", str(config.use_gpu))
    
    # Execution control
    table.add_row("skip_existing", str(config.skip_existing))
    table.add_row("stop_on_error", str(config.stop_on_error))
    
    # Logging
    table.add_row("log_level", config.log_level)
    table.add_row("verbose", str(config.verbose))
    table.add_row("debug", str(config.debug))
    
    # Extra arguments
    for k, v in extra_kwargs.items():
        table.add_row(k, str(v))
    
    console.print(table)


def merge_cli_and_config(
    config_file: Optional[Path],
    cli_args: dict
) -> PipelineConfig:
    """
    Merge configuration from file and CLI arguments.
    
    Priority (highest to lowest):
    1. CLI arguments (explicitly provided by user)
    2. Config file values
    3. Default values
    
    Args:
        config_file: Path to config file (if provided)
        cli_args: Dictionary of CLI arguments (only non-None values)
    
    Returns:
        Merged PipelineConfig object
    
    Raises:
        ConfigurationError: If config file is invalid or required fields missing
    """
    # Start with config file if provided
    if config_file:
        try:
            console.print(f"[blue]Loading configuration from:[/blue] {config_file}")
            config = PipelineConfig.from_file(config_file)
        except FileNotFoundError:
            raise ConfigurationError(
                f"Configuration file not found: {config_file}",
                details="Please check the path and try again."
            )
        except Exception as e:
            raise ConfigurationError(
                f"Failed to load configuration file: {config_file}",
                details=str(e)
            )
    else:
        # Start with empty config
        config = PipelineConfig()
    
    # Override with CLI arguments (only if explicitly provided)
    # We only merge non-None values to distinguish between "not provided" and "provided as None"
    for key, value in cli_args.items():
        if value is not None:
            # Handle list conversion for participant/session labels
            if key in ['participant_label', 'session_label'] and isinstance(value, str):
                value = [value]
            
            config.set(key, value)
    
    return config


def validate_required_arguments(config: PipelineConfig) -> None:
    """
    Validate that all required arguments are provided.
    
    Args:
        config: Pipeline configuration to validate
    
    Raises:
        ConfigurationError: If required arguments are missing
    """
    required_fields = {
        'bids_dir': 'Input BIDS dataset directory',
        'output_dir': 'Output directory for derivatives',
    }
    
    missing = []
    for field, description in required_fields.items():
        value = getattr(config, field, None)
        if value is None:
            missing.append(f"  --{field.replace('_', '-')} : {description}")
    
    if missing:
        error_msg = (
            "Missing required arguments. Please provide them via config file or command line:\n\n"
            + "\n".join(missing) +
            "\n\nExamples:\n"
            "  1. Via command line:\n"
            "     qmri-neuropipe --bids-dir /data/bids --output-dir /data/derivatives\n\n"
            "  2. Via config file:\n"
            "     qmri-neuropipe --config config.yaml\n\n"
            "  3. Mixed (CLI overrides config):\n"
            "     qmri-neuropipe --config config.yaml --n-cpus 16"
        )
        raise ConfigurationError(error_msg)
    
    # Validate paths exist
    if not config.bids_dir.exists():
        raise ConfigurationError(
            f"BIDS directory does not exist: {config.bids_dir}",
            details="Please check the path and ensure it's accessible."
        )
    
    if not config.bids_dir.is_dir():
        raise ConfigurationError(
            f"BIDS directory path is not a directory: {config.bids_dir}"
        )


@app.command()
def main(
    # Core paths (can be provided via CLI or config)
    bids_dir: Optional[Path] = typer.Option(
        None, 
        "--bids-dir",
        help="Path to BIDS dataset directory",
        exists=False,  # We'll validate manually for better error messages
        file_okay=False, 
        dir_okay=True, 
        readable=True
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output-dir",
        help="Path to output directory for derivatives",
        file_okay=False, 
        dir_okay=True,
        writable=True
    ),
    work_dir: Optional[Path] = typer.Option(
        None, 
        "--work-dir",
        help="Path to working directory for temporary files"
    ),
    
    # Config file
    config_file: Optional[Path] = typer.Option(
        None, 
        "--config", "-c",
        help="Path to YAML/JSON configuration file",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True
    ),
    
    # Pipeline selection
    pipeline: Optional[str] = typer.Option(
        None, 
        "--pipeline",
        help="Pipeline to run (e.g., dmri, fmri, anat)"
    ),
    level: Optional[str] = typer.Option(
        None,
        "--level",
        help="BIDS App analysis level: participant, group (default: participant)"
    ),
    
    # Subject/session selection
    participant_label: Optional[str] = typer.Option(
        None, 
        "--participant-label", "-p",
        help="Participant ID(s) to process (e.g., sub-01 or 01)"
    ),
    session_label: Optional[str] = typer.Option(
        None, 
        "--session-label", "-s",
        help="Session ID(s) to process (e.g., ses-01 or 01)"
    ),
    
    # Computational resources
    n_cpus: Optional[int] = typer.Option(
        None, 
        "--n-cpus",
        help="Number of CPUs to use"
    ),
    memory_gb: Optional[float] = typer.Option(
        None,
        "--memory-gb",
        help="Memory limit in GB"
    ),
    use_gpu: Optional[bool] = typer.Option(
        None,
        "--use-gpu/--no-gpu",
        help="Enable GPU acceleration (if available)"
    ),
    omp_nthreads: Optional[int] = typer.Option(
        None,
        "--omp-nthreads",
        help="Number of OpenMP threads"
    ),
    gpu_ids: Optional[str] = typer.Option(
        None,
        "--gpu-ids",
        help="GPU IDs to use (e.g. '0' or '0,1')"
    ),
    
    # Execution control
    skip_existing: Optional[bool] = typer.Option(
        None,
        "--skip-existing/--no-skip-existing",
        help="Skip already processed subjects"
    ),
    stop_on_error: Optional[bool] = typer.Option(
        None,
        "--stop-on-error/--continue-on-error",
        help="Stop pipeline on first error"
    ),
    skip_bids_validation: bool = typer.Option(
        False, 
        "--skip-bids-validation",
        help="Skip BIDS dataset validation"
    ),
    
    # Logging
    log_level: Optional[str] = typer.Option(
        None,
        "--log-level",
        help="Logging level (DEBUG, INFO, WARNING, ERROR)"
    ),
    verbose: bool = typer.Option(
        False, 
        "--verbose", "-v",
        help="Enable verbose output"
    ),
    debug: Optional[bool] = typer.Option(
        None,
        "--debug",
        help="Enable debug mode"
    ),
    
    # Other options
    dry_run: bool = typer.Option(
        False, 
        "--dry-run",
        help="Perform a dry run without executing the pipeline"
    ),
    jobs: int = typer.Option(
        1,
        "--jobs", "-j",
        help="Number of parallel jobs (subjects) to run locally"
    ),
    outlier_indices: Optional[str] = typer.Option(
        None,
        "--outlier-indices",
        help="Comma-separated list of volume indices to remove (e.g., '0,1,2')"
    ),
    subjects_file: Optional[Path] = typer.Option(
        None,
        "--subjects-file",
        help="Path to text file containing 'subject,session' list (for HTCondor submission)",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True
    ),
    submit: bool = typer.Option(
        False,
        "--submit",
        help="Generate HTCondor submit file (saves to default location 'job.sub')"
    ),
    submit_file: Optional[Path] = typer.Option(
        None,
        "--submit-file",
        help="Generate HTCondor submit file at a specific path. Implies --submit.",
        file_okay=True,
        dir_okay=False,
        writable=True
    ),
):
    """
    qmri-neuropipe: Robust neuroimaging pipeline for BIDS datasets.
    
    This command processes MRI data using configurable pipelines. Configuration
    can be provided via a YAML/JSON file and/or command line arguments.
    Command line arguments take precedence over config file values.
    
    REQUIRED ARGUMENTS (via CLI or config file):
    
        --bids-dir: Path to BIDS dataset directory
        --output-dir: Path to output directory
    
    EXAMPLES:
    
        # Using config file only:
        qmri-neuropipe --config config.yaml
        
        # Using CLI arguments only:
        qmri-neuropipe --bids-dir /data/bids --output-dir /data/derivatives
        
        # Mixed (CLI overrides config):
        qmri-neuropipe --config config.yaml --n-cpus 16 --participant-label sub-01
        
        # Process specific subjects:
        qmri-neuropipe --config config.yaml -p sub-01 -s ses-01
        
        # Dry run to check configuration:
        qmri-neuropipe --config config.yaml --dry-run --verbose
    """
    
    try:
        # Parse gpu_ids
        gpu_ids_list = None
        if gpu_ids is not None:
            try:
                gpu_ids_list = [int(x.strip()) for x in gpu_ids.split(',')]
            except ValueError:
                console.print(f"[bold red]Error:[/bold red] Invalid gpu_ids format: {gpu_ids}. Expected comma-separated integers.")
                raise typer.Exit(code=1)

        # Parse outlier_indices
        outlier_indices_list = None
        if outlier_indices is not None:
            try:
                outlier_indices_list = [int(x.strip()) for x in outlier_indices.split(',')]
            except ValueError:
                console.print(f"[bold red]Error:[/bold red] Invalid outlier-indices format: {outlier_indices}. Expected comma-separated integers.")
                raise typer.Exit(code=1)

        # Collect CLI arguments (only non-None values)
        # Note: pipeline and level are NOT part of PipelineConfig dataclass
        # They control which workflow runs, not how it runs
        cli_args = {
            'bids_dir': bids_dir,
            'output_dir': output_dir,
            'work_dir': work_dir,
            'participant_label': participant_label,
            'session_label': session_label,
            'subjects_file': subjects_file,
            'n_cpus': n_cpus,
            'memory_gb': memory_gb,
            'use_gpu': use_gpu,
            'gpu_ids': gpu_ids_list,
            'omp_nthreads': omp_nthreads,
            'skip_existing': skip_existing,
            'stop_on_error': stop_on_error,
            'log_level': log_level,
            'debug': debug,
            'jobs': jobs,
            'submit': submit,
            # Pass outlier indices to the nested config path
            'dmri.preprocessing.outliers.manual_indices': outlier_indices_list,
            'dmri.preprocessing.outliers.method': 'manual' if outlier_indices_list else None,
            'dmri.preprocessing.outliers.enabled': True if outlier_indices_list else None,
            # If submit_file is provided, pass it as 'submit' value (path string)
            # If submit is True but no file, pass True (or "DEFAULT")
            # We'll normalize this logic here
        }
        
        # Normalize 'submit' argument for config
        if submit_file:
             cli_args['submit'] = str(submit_file)
        elif submit:
             cli_args['submit'] = True
        else:
             cli_args['submit'] = None # Explicitly set to None if neither provided (will be removed below)

        # Remove None values to distinguish "not provided" from "explicitly None"
        cli_args = {k: v for k, v in cli_args.items() if v is not None}
        
        # Merge config file and CLI arguments
        config = merge_cli_and_config(config_file, cli_args)
        
        # Handle pipeline and level (can come from config file or CLI)
        # Priority: CLI > Config File > Defaults
        # These are stored in config.config_data if they were in the config file
        pipeline_name = pipeline
        analysis_level = level
        
        # Check config.config_data for pipeline/level if not provided via CLI
        # This avoids reloading the config file
        if pipeline_name is None:
            pipeline_name = config.config_data.get('pipeline')
        
        if analysis_level is None:
            analysis_level = config.config_data.get('level')
        

        # Set defaults if still not specified
        if pipeline_name is None:
            # Smart default: If relaxometry explicitly configured and dmri not, use relaxometry
            if config.config_data.get('relaxometry') and not config.config_data.get('dmri'):
                pipeline_name = 'relaxometry'
            else:
                pipeline_name = 'dmri'

        if analysis_level is None:
            analysis_level = 'participant'
        
        # Validate required arguments
        validate_required_arguments(config)
        
        # Create output directories
        config.output_dir.mkdir(parents=True, exist_ok=True)
        # derivatives/qmri-neuropipe no longer created automatically
        # Outputs go directly to config.output_dir
        
        # Set work_dir if not specified
        if config.work_dir is None:
            config.work_dir = config.output_dir / "work"
        config.work_dir.mkdir(parents=True, exist_ok=True)
        
        
        # --- Configure Environment for Threading (ANTs, OpenMP, etc.) ---
        # This MUST be done before importing pipeline modules that might initialize libraries (like ANTs)
        import os
        
        # Determine effective CPU count
        effective_cpus = config.n_cpus
        if not effective_cpus:
            # Fallback or default
            effective_cpus = 1
            
        str_cpus = str(effective_cpus)
        
        # ANTs / ITK
        os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str_cpus
        
        # OpenMP (used by many C modules)
        # If omp_nthreads is explicitly set via CLI, use it. Otherwise default to n_cpus or system default.
        if omp_nthreads:
            os.environ["OMP_NUM_THREADS"] = str(omp_nthreads)
        else:
            # It is often safer to set OMP_NUM_THREADS to n_cpus to avoid oversubscription 
            # if the underlying libraries default to all cores.
            os.environ["OMP_NUM_THREADS"] = str_cpus
            
        # MKL / BLAS
        os.environ["MKL_NUM_THREADS"] = str_cpus
        os.environ["OPENBLAS_NUM_THREADS"] = str_cpus
        os.environ["VECLIB_MAXIMUM_THREADS"] = str_cpus
        os.environ["NUMEXPR_NUM_THREADS"] = str_cpus
        
        if config.verbose or verbose:
             console.print(f"[blue]Setting threading environment variables:[/blue] ITK={str_cpus}, OMP={os.environ['OMP_NUM_THREADS']}")

        # Print configuration if verbose
        if config.verbose or verbose or dry_run:
            console.print("\n[bold green]Configuration validated successfully![/bold green]\n")
            _print_args(
                config, 
                pipeline=pipeline_name,
                level=analysis_level,
                dry_run=dry_run,
                skip_bids_validation=skip_bids_validation,
                omp_nthreads=omp_nthreads
            )
        
        # Dry run - just validate and print config
        if dry_run:
            console.print("\n[bold yellow]Dry run complete. No processing performed.[/bold yellow]")
            return
        
        # Load data
        console.print(f"\n[blue]Loading BIDS dataset from:[/blue] {config.bids_dir}")
        loader = DataLoader(config.bids_dir)
        
        # Determine subjects/sessions to process
        if config.participant_label:
            console.print(f"Using participant label(s): {config.participant_label}")
            data = loader.load_multiple_subjects(subjects=config.participant_label, sessions=config.session_label)
        elif config.subjects_file:
            console.print(f"Using subjects from file: {config.subjects_file}")
            data = loader.load_from_subjects_file(config.subjects_file)
        else:
            console.print("No subject selection provided. Discovering all subjects...")
            data = loader.load_multiple_subjects()
        
        tasks = list(data.keys())
        console.print(f"Total tasks to process: {len(tasks)}")

        # If parallel execution requested
        # If parallel execution requested
        if jobs > 1 and not submit:
             console.print(f"\n[bold blue]Running locally in PARALLEL mode with {jobs} workers.[/bold blue]")
             
             import concurrent.futures
             from qmri_neuropipe.core.ui import console as main_console
             
             # Serialize Config
             config_dict = config.to_dict()
             
             # Initialize Manager for inter-process communication
             manager = multiprocessing.Manager()
             log_queue = manager.Queue()
             slot_queue = manager.Queue()
             for i in range(jobs):
                  slot_queue.put(i)
             
             # UI State tracking
             class UIState:
                  def __init__(self, n_jobs, total):
                       self.n_jobs = n_jobs
                       self.buffers = {i: deque(maxlen=10) for i in range(n_jobs)}
                       self.job_info = {i: "[dim]Idle[/dim]" for i in range(n_jobs)}
                       self.job_status = {i: "idle" for i in range(n_jobs)}
                       # Distinct vibrant colors for workers
                       self.colors = ["cyan", "magenta", "yellow", "green", "blue", "red", "bright_blue", "bright_magenta"]
                       self.progress = Progress(
                            SpinnerColumn(),
                            TextColumn("[progress.description]{task.description}"),
                            BarColumn(bar_width=None),
                            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                            TimeRemainingColumn(),
                            console=main_console,
                            expand=True
                       )
                       self.overall_task = self.progress.add_task("[bold green]Total Progress", total=total)
                       self.lock = threading.Lock()
                       self.tasks_done = 0

                  def update_log(self, j_id, msg):
                       with self.lock:
                            # Strip common redundant prefixes if needed, but keep it clean
                            self.buffers[j_id].append(msg)

                  def update_info(self, j_id, info, status="running"):
                       with self.lock:
                            self.job_info[j_id] = info
                            self.job_status[j_id] = status

                  def advance(self):
                       with self.lock:
                            self.tasks_done += 1
                            self.progress.update(self.overall_task, completed=self.tasks_done)

                  def get_layout(self):
                       with self.lock:
                            status_icons = {
                                 "idle": "💤 [dim]Idle[/dim]",
                                 "running": "🔄 [bold cyan]Running[/bold cyan]",
                                 "complete": "✅ [bold green]Done[/bold green]",
                                 "failed": "❌ [bold red]ERROR[/bold red]"
                            }
                            
                            # Build the main body grid using Layout
                            cols = 2 if self.n_jobs > 1 else 1
                            rows = (self.n_jobs + cols - 1) // cols
                            
                            body_layout = Layout()
                            row_layouts = []
                            for r in range(rows):
                                 row_l = Layout(name=f"row_{r}")
                                 col_layouts = []
                                 for c in range(cols):
                                      idx = r * cols + c
                                      if idx < self.n_jobs:
                                           color = self.colors[idx % len(self.colors)]
                                           icon = status_icons.get(self.job_status[idx], "•")
                                           title = f"[{color}]Worker {idx+1}[/{color}] {icon} [bold]{self.job_info[idx]}[/bold]"
                                           
                                           # Convert buffer lines from ANSI to Rich Text objects to preserve colors
                                           rich_buffer = []
                                           for line in self.buffers[idx]:
                                                rich_buffer.append(Text.from_ansi(line))
                                           
                                           content = Text("\n").join(rich_buffer)
                                           col_layouts.append(Layout(Panel(content, title=title, border_style=color, box=box.ROUNDED, padding=(0, 1))))
                                      else:
                                           col_layouts.append(Layout())
                                 row_l.split_row(*col_layouts)
                                 row_layouts.append(row_l)
                            
                            body_layout.split_column(*row_layouts)

                            layout = Layout()
                            layout.split_column(
                                 Layout(Panel("[bold white on blue] qmri-neuropipe [/bold white on blue] [blue]Parallel Processing Monitor[/blue]", box=box.MINIMAL), size=3),
                                 Layout(body_layout, name="main"),
                                 Layout(Panel(self.progress, box=box.MINIMAL), size=3)
                            )
                            return layout

             ui_state = UIState(jobs, len(tasks))
             
             def log_monitor(live_obj):
                  while True:
                       try:
                            item = log_queue.get()
                            if item == "STOP": break
                            msg_type, j_id, msg = item
                            if msg_type == "log":
                                 ui_state.update_log(j_id, msg)
                            elif msg_type == "info":
                                 # Expecting msg to be (subject_info, status)
                                 if isinstance(msg, tuple):
                                      ui_state.update_info(j_id, msg[0], msg[1])
                                 else:
                                      ui_state.update_info(j_id, msg)
                            
                            # Trigger UI refresh
                            if live_obj and live_obj.is_started:
                                 try:
                                       live_obj.update(ui_state.get_layout())
                                 except: pass
                       except: break

             results = []
             # Use screen=True for maximum stability (own terminal buffer)
             with Live(ui_state.get_layout(), console=main_console, refresh_per_second=4, screen=True) as live:
                  # Start monitor with live access
                  monitor_thread = threading.Thread(target=log_monitor, args=(live,), daemon=True)
                  monitor_thread.start()

                  with concurrent.futures.ProcessPoolExecutor(max_workers=jobs) as executor:
                       futures = {}
                       for i, (sub, ses) in enumerate(tasks):
                            g_id = None
                            if config.gpu_ids:
                                 g_id = config.gpu_ids[i % len(config.gpu_ids)]
                            
                            f = executor.submit(_run_parallel_worker, sub, ses, config_dict, g_id, pipeline_name, log_queue, slot_queue)
                            futures[f] = (sub, ses)

                       for f in concurrent.futures.as_completed(futures):
                            sub_id, ses_id = futures[f]
                            try:
                                 res = f.result()
                                 results.append(res or {'n_failed': 1})
                            except Exception as e:
                                 results.append({'n_failed': 1, 'error': str(e)})
                            ui_state.advance()
                            live.update(ui_state.get_layout())

             log_queue.put("STOP")
             monitor_thread.join(timeout=1)

             # Aggregate Stats and collect failures
             stats = {'n_success': 0, 'n_failed': 0, 'n_skipped': 0}
             failures = []
             for r in results:
                  stats['n_success'] += r.get('n_success', 0)
                  stats['n_failed'] += r.get('n_failed', 0)
                  stats['n_skipped'] += r.get('n_skipped', 0)
                  if r.get('n_failed', 0) > 0:
                       failures.append(r)
        
        else:
            # Create and run pipeline
            console.print(f"\n[blue]Initializing {pipeline_name} pipeline...[/blue]")
            if pipeline_name == 'dmri':
                from .workflows.pipelines.dmri import DMRIPipeline
                pipeline_obj = DMRIPipeline(config)
            elif pipeline_name == 'anat':
                from .workflows.pipelines.anat import AnatPipeline
                pipeline_obj = AnatPipeline(config)
            elif pipeline_name == 'relaxometry':
                from .workflows.pipelines.relaxometry import RelaxometryPipeline
                pipeline_obj = RelaxometryPipeline(config)
            else:
                raise ConfigurationError(
                    f"Unsupported pipeline: {pipeline_name}",
                    details=f"Available pipelines: dmri, anat, relaxometry"
                )
            
            # Run pipeline
            console.print("\n[bold green]Starting pipeline execution...[/bold green]\n")
            stats = pipeline_obj.run(pairs=tasks)
            # Single processing doesn't handle failures list yet, but we'll focus on parallel for now
            failures = []
        

        if stats:
            if stats.get('n_failed', 0) > 0:
                 console.print(f"\n[bold red]Pipeline completed with errors![/bold red]")
                 
                 # Print detailed summary of failed subjects
                 if failures:
                      console.print("\n[bold red]Failed Subjects Summary:[/bold red]")
                      for f in failures:
                           sub_info = f"sub-{f.get('subject')}"
                           if f.get('session'): sub_info += f" ses-{f.get('session')}"
                           err = f.get('error', 'Unknown error')
                           console.print(f"  ❌ [bold]{sub_info}[/bold]: {err}")
                 
                 console.print(f"\n[bold yellow]Totals:[/bold yellow] Success: {stats['n_success']}, Failed: {stats['n_failed']}, Skipped: {stats['n_skipped']}")
                 raise typer.Exit(code=1)
            
            console.print("\n[bold green]Pipeline completed successfully![/bold green]")
            console.print(f"Success: {stats['n_success']}, Failed: {stats['n_failed']}, Skipped: {stats['n_skipped']}\n")
        
    except ConfigurationError as e:
        console.print(f"\n[bold red]Configuration Error:[/bold red] {e.message}")
        if e.details:
            console.print(f"[red]Details:[/red] {e.details}")
        console.print("\n[yellow]Use --help for usage information.[/yellow]\n")
        raise typer.Exit(code=1)
    
    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {str(e)}")
        if config.debug if 'config' in locals() else debug:
            console.print_exception()
        raise typer.Exit(code=1)


def _run_parallel_worker(
    subject: str,
    session: Optional[str],
    config_dict: dict,
    gpu_id: Optional[int],
    pipeline_name: str,
    log_queue: Optional[Any] = None,
    slot_queue: Optional[Any] = None
) -> dict:
    """
    Worker function for parallel pipeline execution.
    """
    import os
    import sys
    import threading
    import logging
    from pathlib import Path
    
    # 0. Acquire Slot and Identify Job
    job_id = 0
    if slot_queue:
         job_id = slot_queue.get()

    # Communication flag for parallel-aware modules
    os.environ["QMRI_PARALLEL_WORKER"] = "1"

    # 1. Robust OS-level Redirection (Capture all stdout/stderr, including C-extensions and subprocesses)
    if log_queue:
        pipe_out, pipe_in = os.pipe()
        
        def forwarder():
            try:
                # Use os.fdopen to read lines from the pipe
                with os.fdopen(pipe_out, 'r', errors='replace') as f:
                    for line in f:
                        if line.strip():
                            log_queue.put(("log", job_id, line.strip()))
            except Exception:
                pass

        threading.Thread(target=forwarder, daemon=True).start()

        # Redirect FD 1 and 2 (stdout/stderr) to pipe
        os.dup2(pipe_in, 1)
        os.dup2(pipe_in, 2)
        # Original pipe_in is no longer needed
        os.close(pipe_in)
        
        # Force Python to flush and use the new FDs
        sys.stdout = os.fdopen(1, 'w', buffering=1)
        sys.stderr = os.fdopen(2, 'w', buffering=1)

    try:
        from qmri_neuropipe.core import PipelineConfig, ui
        from rich.console import Console

        # 2. Isolate GPU Environment
        if gpu_id is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            
        # 3. Reconstruct Config
        valid_keys = set(PipelineConfig.__annotations__.keys())
        if 'config_data' in valid_keys:
            valid_keys.remove('config_data')
            
        standard_args = {k: v for k, v in config_dict.items() if k in valid_keys}
        
        try:
            config = PipelineConfig(**standard_args, config_data=config_dict)
        except Exception:
            config = PipelineConfig()
            for k, v in standard_args.items():
                if hasattr(config, k):
                    setattr(config, k, v)
            config.config_data = config_dict
            
        if config.bids_dir: config.bids_dir = Path(config.bids_dir)
        if config.output_dir: config.output_dir = Path(config.output_dir)
        if config.work_dir: config.work_dir = Path(config.work_dir)
        if config.subjects_file: config.subjects_file = Path(config.subjects_file)

        config.set('jobs', 1)
        if 'jobs' in config.config_data:
            config.config_data['jobs'] = 1

        # 4. Setup Worker UI/Logging (Now that redirection is active)
        if log_queue:
             # Force TRUE terminal features for child processes so they generate ANSI colors/styles
             ui.console = Console(force_terminal=True, color_system="truecolor", soft_wrap=True, legacy_windows=False)

             log_queue.put(("info", job_id, (f"{subject} (ses-{session if session else 'N/A'})", "running")))
             ui.console.print(f"🚀 [bold cyan]Starting pipeline for {subject}[/bold cyan]")
             
             # Configure logging to use the redirected stderr (FD 2)
             # We use basicConfig to catch EVERYTHING on the root logger.
             logging.basicConfig(level=logging.INFO, force=True, handlers=[
                 logging.StreamHandler(sys.stderr)
             ])
             # CRITICAL: Disable propagation for the main library logger so it doesn't 
             # double-report to the root logger we just set up.
             logging.getLogger("qmri-neuropipe").propagate = False
             logging.getLogger().setLevel(logging.INFO)
             

        # 4. Initialize Pipeline
        if pipeline_name == 'dmri':
            from qmri_neuropipe.workflows.pipelines.dmri import DMRIPipeline
            pipeline_obj = DMRIPipeline(config)
        elif pipeline_name == 'anat':
            from qmri_neuropipe.workflows.pipelines.anat import AnatPipeline
            pipeline_obj = AnatPipeline(config)
        elif pipeline_name == 'relaxometry':
            from qmri_neuropipe.workflows.pipelines.relaxometry import RelaxometryPipeline
            pipeline_obj = RelaxometryPipeline(config)
        else:
             return {'n_success': 0, 'n_failed': 1, 'n_skipped': 0, 'error': f"Unknown pipeline {pipeline_name}"}
        
        # 5. Run (Single Subject mode)
        stats = pipeline_obj.run(
            subjects=[subject], 
            sessions=[session] if session else None
        )
        
        if log_queue:
             status = "complete" if stats.get('n_success', 0) > 0 else "failed"
             log_queue.put(("info", job_id, (f"{subject} (Done)", status)))
             ui.console.print(f"✨ [bold green]Finished {subject}[/bold green]")
             
        stats.update({'subject': subject, 'session': session})
        return stats
        
    except Exception as e:
        if log_queue:
             log_queue.put(("info", job_id, (f"{subject} (Error)", "failed")))
             ui.console.print(f"❌ [bold red]FATAL ERROR: {str(e)}[/bold red]")
        return {
            'n_success': 0, 
            'n_failed': 1, 
            'n_skipped': 0, 
            'error': str(e),
            'subject': subject,
            'session': session
        }
    finally:
         if slot_queue:
              # Reset info after a short delay might be nice, but leaving it as (Done) is better
              slot_queue.put(job_id)

if __name__ == "__main__":
    app()
