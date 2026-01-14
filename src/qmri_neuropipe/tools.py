"""
CLI Tools for direct function access (e.g. Model Fitting).

This module exposes internal library functions as standalone CLI commands.
"""

import typer
from pathlib import Path
from typing import Optional, List
from rich.console import Console

# Import interfaces (lazy import inside commands to avoid heavy loading if not needed?)
# Actually, top level imports are fine for CLI usually, but robust imports are better.

app = typer.Typer(help="Standalone tools for modeling and processing.")
console = Console()

def _setup_threading(nthreads: int):
    """Result threading environment variables."""
    import os
    os.environ["OMP_NUM_THREADS"] = str(nthreads)
    os.environ["MKL_NUM_THREADS"] = str(nthreads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(nthreads)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(nthreads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(nthreads)
    os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(nthreads)


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
            metrics=metrics,
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
            metrics=metrics,
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
            metrics=metrics,
            nthreads=nthreads,
            grad_nonlin=grad_nonlin,
        )
        console.print("[bold green]Success![/bold green]")
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(code=1)

if __name__ == "__main__":
    app()
