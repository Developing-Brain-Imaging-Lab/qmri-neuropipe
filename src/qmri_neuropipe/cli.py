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
from pathlib import Path
from typing import List, Optional
import typer
from rich.console import Console
from rich.table import Table

from qmri_neuropipe.core import PipelineConfig
from qmri_neuropipe.core.exceptions import ConfigurationError
from qmri_neuropipe.io import DataLoader


app = typer.Typer(add_completion=False, help="qmri-neuropipe: Robust BIDS MRI processing pipeline")
console = Console()


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
        help="Generate HTCondor submit files instead of running locally"
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
            'verbose': verbose,
            'debug': debug,
            'jobs': jobs,
            'submit': submit,
        }
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
        
        # Create and run pipeline
        console.print(f"\n[blue]Initializing {pipeline_name} pipeline...[/blue]")
        if pipeline_name == 'dmri':
            from .workflows.pipelines.dmri import DMRIPipeline
            pipeline_obj = DMRIPipeline(config)
        elif pipeline_name == 'anat':
            from .workflows.pipelines.anat import AnatPipeline
            pipeline_obj = AnatPipeline(config)
        else:
            raise ConfigurationError(
                f"Unsupported pipeline: {pipeline_name}",
                details=f"Available pipelines: dmri, anat"
            )
        
        # Run pipeline
        console.print("\n[bold green]Starting pipeline execution...[/bold green]\n")
        stats = pipeline_obj.run(subjects=config.participant_label,
                                 sessions=config.session_label)
        
        if stats and stats.get('n_failed', 0) > 0:
             console.print(f"\n[bold red]Pipeline completed with errors![/bold red]")
             console.print(f"Success: {stats['n_success']}, Failed: {stats['n_failed']}, Skipped: {stats['n_skipped']}")
             raise typer.Exit(code=1)
        
        console.print("\n[bold green]Pipeline completed successfully![/bold green]\n")
        
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


if __name__ == "__main__":
    app()