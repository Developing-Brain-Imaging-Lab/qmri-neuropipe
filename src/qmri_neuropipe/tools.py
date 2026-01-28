"""
CLI Tools for direct function access (e.g. Model Fitting).

This module exposes internal library functions as standalone CLI commands.
"""

import typer
from pathlib import Path
from typing import Optional, List
from qmri_neuropipe.core.ui import console

# Import interfaces (lazy import inside commands to avoid heavy loading if not needed?)
# Actually, top level imports are fine for CLI usually, but robust imports are better.

app = typer.Typer(help="Standalone tools for modeling and processing.")
# console = Console()

def _setup_threading(nthreads: int):
    """Result threading environment variables."""
    import os
    os.environ["OMP_NUM_THREADS"] = str(nthreads)
    os.environ["MKL_NUM_THREADS"] = str(nthreads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(nthreads)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(nthreads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(nthreads)
    os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(nthreads)


def _parse_metric_list(metrics: List[str]) -> List[str]:
    """Support both multiple --metrics flags and comma-separated strings."""
    final = []
    for m in metrics:
        if "," in m:
            final.extend([x.strip() for x in m.split(",") if x.strip()])
        else:
            final.append(m)
    return final


@app.command("fit-dti")
def fit_dti_cli(
    input: Path = typer.Option(..., "--input", "-i", help="Input DWI NIfTI file", exists=True),
    output_dir: Path = typer.Option(..., "--output-dir", "-o", help="Output directory"),
    bval: Path = typer.Option(..., "--bval", help="Path to bval file", exists=True),
    bvec: Path = typer.Option(..., "--bvec", help="Path to bvec file", exists=True),
    mask: Optional[Path] = typer.Option(None, "--mask", "-m", help="Path to brain mask", exists=True),
    method: str = typer.Option("WLLS", "--method", help="Fitting method (WLLS, OLS, NLLS, RESTORE)"),
    nthreads: int = typer.Option(1, "--nthreads", "-n", help="Number of threads for fitting"),
    smoothing: Optional[float] = typer.Option(None, help="Gaussian smoothing FWHM (mm)."),
    metrics: List[str] = typer.Option(["fa", "md", "ad", "rd", "color_fa", "tensor"], help="Metrics to calculate."),
    grad_nonlin: Optional[Path] = typer.Option(None, help="Path to gradient nonlinearity tensor file for correction."),
):
    """
    Fit Diffusion Tensor Imaging (DTI) model.
    """
    _setup_threading(nthreads)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    console.print(f"[bold blue]Running DTI Fit ({method})[/bold blue]")
    console.print(f"  Input: {input}")
    console.print(f"  Output: {output_dir}")
    
    from qmri_neuropipe.interfaces.dipy import fit_dti
    
    kwargs = {}
    if smoothing:
        kwargs['smoothing_fwhm'] = smoothing
        
    try:
        fit_dti(
            in_file=input,
            out_dir=output_dir,
            bval_file=bval,
            bvec_file=bvec,
            mask_file=mask,
            fit_method=method,
            metrics=_parse_metric_list(metrics),
            nthreads=nthreads,
            grad_nonlin=grad_nonlin,
            **kwargs
        )
        console.print("[bold green]Success![/bold green]")
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        import traceback
        traceback.print_exc()
        raise typer.Exit(code=1)


@app.command("fit-dki")
def fit_dki_cli(
    input: Path = typer.Option(..., "--input", "-i", help="Input DWI NIfTI file", exists=True),
    output_dir: Path = typer.Option(..., "--output-dir", "-o", help="Output directory"),
    bval: Path = typer.Option(..., "--bval", help="Path to bval file", exists=True),
    bvec: Path = typer.Option(..., "--bvec", help="Path to bvec file", exists=True),
    mask: Optional[Path] = typer.Option(None, "--mask", "-m", help="Path to brain mask", exists=True),
    nthreads: int = typer.Option(1, "--nthreads", "-n", help="Number of threads"),
    smoothing: Optional[float] = typer.Option(None, "--smoothing", help="Sigma/FWHM for smoothing (optional)"),
    mean_signal: bool = typer.Option(False, "--mean-signal", help="Use Mean Signal DKI (MSDKI)."),
    metrics: List[str] = typer.Option(["mk", "ak", "rk", "fa", "md"], help="Metrics to calculate."),
    grad_nonlin: Optional[Path] = typer.Option(None, help="Path to gradient nonlinearity tensor file for correction."),
):
    """
    Fit Diffusion Kurtosis Imaging (DKI) model.
    """
    _setup_threading(nthreads)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    console.print(f"[bold blue]Running DKI Fit[/bold blue]")
    if mean_signal:
        console.print("  Using Mean Signal DKI (MSDKI)")
        
    from qmri_neuropipe.interfaces.dipy import fit_dki
    
    kwargs = {}
    if smoothing:
        kwargs['smoothing_fwhm'] = smoothing
    
    # Pass mean_signal to fit_dki (it handles it via kwargs)
    kwargs['mean_signal'] = mean_signal

    try:
        fit_dki(
            in_file=input,
            out_dir=output_dir,
            bval_file=bval,
            bvec_file=bvec,
            mask_file=mask,
            metrics=_parse_metric_list(metrics),
            nthreads=nthreads,
            grad_nonlin=grad_nonlin,
            **kwargs
        )
        console.print("[bold green]Success![/bold green]")
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(code=1)


@app.command("fit-noddi")
def fit_noddi_cli(
    input: Path = typer.Option(..., "--input", "-i", help="Input DWI NIfTI file", exists=True),
    output_dir: Path = typer.Option(..., "--output-dir", "-o", help="Output directory"),
    bval: Path = typer.Option(..., "--bval", help="Path to bval file", exists=True),
    bvec: Path = typer.Option(..., "--bvec", help="Path to bvec file", exists=True),
    mask: Optional[Path] = typer.Option(None, "--mask", "-m", help="Path to brain mask", exists=True),
    backend: str = typer.Option("dmipy", "--backend", help="Backend implementation (dmipy or amico)"),
    nthreads: int = typer.Option(1, "--nthreads", "-n", help="Number of threads"),
    parallel_diff: float = typer.Option(1.7e-9, "--parallel-diff", help="Parallel diffusivity"),
    iso_diff: float = typer.Option(3.0e-9, "--iso-diff", help="Isotropic diffusivity"),
    # Dmipy specific defaults
    solver: str = typer.Option("brute2fine", "--solver", help="Optimization solver (e.g. brute2fine)"),
    distribution: str = typer.Option("Watson", "--distribution", help="Distribution type (Watson, Bingham)"),
    model_type: str = typer.Option("standard", "--model-type", help="Model structure (standard, smt)"),
):
    """
    Fit NODDI model.
    """
    _setup_threading(nthreads)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    console.print(f"[bold blue]Running NODDI Fit ({backend})[/bold blue]")
    
    try:
        if backend.lower() == 'dmipy':
            from qmri_neuropipe.interfaces.dmipy import fit_noddi
            fit_noddi(
                input,
                output_dir,
                bval_file=bval,
                bvec_file=bvec,
                mask_file=mask,
                nthreads=nthreads,
                parallel_diffusivity=parallel_diff,
                iso_diffusivity=iso_diff,
                solver=solver,
                distribution=distribution,
                model_type=model_type
            )
        elif backend.lower() == 'amico':
            from qmri_neuropipe.interfaces.amico import fit_noddi
            # AMICO binding might have different signature, let's check
            fit_noddi(
                input,
                output_dir,
                bval_file=bval,
                bvec_file=bvec,
                mask_file=mask,
                nthreads=nthreads,
                dPar=parallel_diff,
                dIso=iso_diff
            )
        else:
            raise ValueError(f"Unknown backend: {backend}")
            
        console.print("[bold green]Success![/bold green]")
        
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        import traceback
        traceback.print_exc()
        raise typer.Exit(code=1)


@app.command("fit-mapmri")
def fit_mapmri_cli(
    input: Path = typer.Option(..., "--input", "-i", help="Input DWI NIfTI file", exists=True),
    output_dir: Path = typer.Option(..., "--output-dir", "-o", help="Output directory"),
    bval: Path = typer.Option(..., "--bval", help="Path to bval file", exists=True),
    bvec: Path = typer.Option(..., "--bvec", help="Path to bvec file", exists=True),
    mask: Optional[Path] = typer.Option(None, "--mask", "-m", help="Path to brain mask", exists=True),
    nthreads: int = typer.Option(1, "--nthreads", "-n", help="Number of threads"),
    laplacian: bool = typer.Option(True, "--laplacian/--no-laplacian", help="Use Laplacian regularization"),
    positivity: bool = typer.Option(True, help="Enforce positivity constraint."),
    metrics: List[str] = typer.Option(["rtop", "rtap", "rtpp", "qiv", "msd"], help="Metrics to calculate."),
    grad_nonlin: Optional[Path] = typer.Option(None, help="Path to gradient nonlinearity tensor file for correction."),
):
    """
    Fit MAP-MRI model.
    """
    _setup_threading(nthreads)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    console.print(f"[bold blue]Running MAP-MRI Fit[/bold blue]")
    
    from qmri_neuropipe.interfaces.dipy import fit_mapmri
    
    try:
        fit_mapmri(
            in_file=input,
            out_dir=output_dir,
            bval_file=bval,
            bvec_file=bvec,
            mask_file=mask,
            laplacian=laplacian,
            positivity=positivity,
            metrics=_parse_metric_list(metrics),
            nthreads=nthreads,
            grad_nonlin=grad_nonlin,
        )
        console.print("[bold green]Success![/bold green]")
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(code=1)


@app.command("run-relaxometry")
def run_relaxometry_cli(
    output_dir: Path = typer.Option(..., "--output-dir", "-o", help="Output directory"),
    spgr: List[Path] = typer.Option(..., "--spgr", help="SPGR Input(s) (4D or multiple 3D)."),
    ssfp: List[Path] = typer.Option([], "--ssfp", help="SSFP Input(s)."),
    irspgr: List[Path] = typer.Option([], "--irspgr", help="IR-SPGR Input(s) for HIFI."),
    b1: List[Path] = typer.Option([], "--b1", help="B1 Map/AFI Inputs."),
    t1w: Optional[Path] = typer.Option(None, "--t1w", help="Structural T1w for coregistration."),
    nthreads: int = typer.Option(1, "--nthreads", "-n", help="Number of threads"),
    config_file: Optional[Path] = typer.Option(None, "--config", "-c", help="Configuration YAML file."),
):
    """
    Run Relaxometry Pipeline (DESPOT1/2/HIFI/mcDESPOT).
    Includes Preprocessing, B1 Mapping, Fitting, and Post-processing.
    """
    _setup_threading(nthreads)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    console.print(f"[bold blue]Running Relaxometry Pipeline[/bold blue]")
    
    from qmri_neuropipe.workflows.pipelines.relaxometry import RelaxometryWorkflow
    from qmri_neuropipe.core import PipelineConfig
    from qmri_neuropipe.core.types import ImageFile
    import logging
    
    # Load Config
    cfg = PipelineConfig()
    if config_file:
        cfg.load_yaml(config_file)
        
    # Setup Logger
    logger = logging.getLogger("relaxometry")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(ch)
        
    # Prepare Context input
    context = {'relax_files': []}
    
    # helper
    def _add_files(paths, acq_tag, suffix=""):
        for p in paths:
             ents = {'acq': acq_tag, 'suffix': suffix} # Minimal entities
             context['relax_files'].append(ImageFile(img=p, entities=ents, bids_dir=p.parent))
             
    _add_files(spgr, 'spgr')
    _add_files(ssfp, 'ssfp')
    _add_files(irspgr, 'irspgr')
    _add_files(b1, 'afi', 'b1')
    
    if t1w:
         context['t1w_file'] = ImageFile(img=t1w, entities={'suffix': 'T1w'})
    
    workflow = RelaxometryWorkflow(config=cfg, logger=logger, provenance={})
    
    try:
        workflow.run(output_dir=output_dir, context=context, final_output_dir=output_dir)
        console.print("[bold green]Relaxometry Pipeline Completed![/bold green]")
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        import traceback
        traceback.print_exc()
        raise typer.Exit(code=1)

@app.command("create-gnl-map")
def create_gnl_map_cli(
    input: Path = typer.Option(..., "--input", "-i", help="Input NIfTI file (processed/final grid).", exists=True),
    coeffs: Path = typer.Option(..., "--coeffs", "-c", help="Gradient nonlinearity coefficients file (.dat).", exists=True),
    output: Path = typer.Option(..., "--output", "-o", help="Output path for the .nii.gz tensor map."),
    initial_image: Optional[Path] = typer.Option(None, "--initial-image", help="Optional native space image (for resampled inputs).", exists=True),
    nthreads: int = typer.Option(1, "--nthreads", "-n", help="Number of threads."),
    force: bool = typer.Option(False, "--force", help="Force overwrite existing output."),
):
    """
    Generate Gradient Nonlinearity (GNL) tensor map using TORTOISE.
    """
    _setup_threading(nthreads)
    
    console.print(f"[bold blue]Creating GNL Tensor Map[/bold blue]")
    console.print(f"  Input: {input}")
    console.print(f"  Coeffs: {coeffs}")
    console.print(f"  Output: {output}")
    
    from qmri_neuropipe.lib.dmri.grad_nonlin import create_gnl_map
    from qmri_neuropipe.core.types import ImageFile
    
    # Wrap paths in ImageFile for compatibility with the library function
    # Note: bval/bvec are not strictly required for b0 extraction if it's already a b0 image, 
    # but the tool tries to be smart.
    input_obj = ImageFile(img=input, entities={})
    native_obj = ImageFile(img=initial_image, entities={}) if initial_image else None
    
    try:
        create_gnl_map(
            input_image=input_obj,
            output_path=output,
            grad_coeffs=coeffs,
            native_reference=native_obj,
            nthreads=nthreads,
            force=force
        )
        console.print("[bold green]Success![/bold green]")
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        import traceback
        traceback.print_exc()
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
