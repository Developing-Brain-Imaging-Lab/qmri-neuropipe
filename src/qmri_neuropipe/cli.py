
from __future__ import annotations
import json
from pathlib import Path
from typing import List, Optional
import typer
from rich.console import Console
from rich.table import Table
from .core.logging import init_logging, log
from .core.deriv import RunLedger
from .utils.bids import select_participants_sessions
from .workflows.dmri_dti import build_dmri_dti_workflow

app = typer.Typer(add_completion=False, help="qmri-neuropipe (BIDS App skeleton)")

def _print_args(bids_dir, output_dir, level, **kwargs):
    console = Console(); table = Table(title="qmri-neuropipe arguments")
    table.add_column("arg"); table.add_column("value", style="cyan")
    for k, v in [("bids_dir", bids_dir), ("output_dir", output_dir), ("level", level), *kwargs.items()]:
        table.add_row(k, str(v))
    console.print(table)

@app.command()
def main(
    bids_dir: Path = typer.Argument(..., exists=True, file_okay=False, dir_okay=True, readable=True),
    output_dir: Path = typer.Argument(..., file_okay=False, dir_okay=True, writable=True),
    level: str = typer.Argument("participant", help="BIDS App level: participant (only)"),
    participant_label: List[str] = typer.Option(None, "--participant-label", "-p"),
    session_label: List[str] = typer.Option(None, "--session-label", "-s"),
    pipeline: str = typer.Option("dmri-dti", help="Pipeline to run (e.g., dmri-dti)"),
    work_dir: Optional[Path] = typer.Option(None, "--work-dir"),
    n_cpus: int = typer.Option(1, "--n-cpus"),
    omp_nthreads: int = typer.Option(1, "--omp-nthreads"),
    skip_bids_validation: bool = typer.Option(False, "--skip-bids-validation"),
    config: Optional[Path] = typer.Option(None, "--config", help="JSON/YAML advanced config"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "derivatives" / "qmri-neuropipe").mkdir(parents=True, exist_ok=True)
    (output_dir / "work").mkdir(parents=True, exist_ok=True)
    if work_dir is None: work_dir = output_dir / "work"
    init_logging(output_dir, verbose=verbose)
    _print_args(bids_dir, output_dir, level, pipeline=pipeline, n_cpus=n_cpus, omp_nthreads=omp_nthreads, work_dir=str(work_dir), dry_run=dry_run)

    items = select_participants_sessions(bids_dir=bids_dir, participants=participant_label, sessions=session_label, skip_validation=skip_bids_validation)
    if not items:
        typer.echo("No participants/sessions selected. Exiting."); raise typer.Exit(1)

    cfg = {}
    if config:
        cpath = Path(config)
        if cpath.suffix.lower() in {".yaml",".yml"}:
            import yaml; cfg = yaml.safe_load(cpath.read_text())
        else:
            cfg = json.loads(cpath.read_text())

    ledger = RunLedger(base=output_dir / "derivatives" / "qmri-neuropipe" / "logs"); ledger.start_run()
    try:
        for sub, ses in items:
            log.info({"event": "begin_participant", "sub": sub, "ses": ses, "pipeline": pipeline})
            if pipeline == "dmri-dti":
                wf = build_dmri_dti_workflow(bids_dir=bids_dir, output_dir=output_dir, work_dir=work_dir, sub=sub, ses=ses, n_cpus=n_cpus, omp_nthreads=omp_nthreads, config=cfg, dry_run=dry_run)
            else:
                raise RuntimeError(f"Unsupported pipeline: {pipeline}")
            if not dry_run:
                from pydra import Submitter
                with Submitter(plugin="cf", n_procs=n_cpus) as subm:
                    subm(wf); _ = wf.result()
            ledger.mark_subject(sub=sub, ses=ses, status="ok")
    except Exception as e:
        log.exception("pipeline_failed"); ledger.mark_subject(sub=sub, ses=ses, status="error", error=str(e)); raise
    finally:
        ledger.finish_run()

if __name__ == "__main__":
    app()
