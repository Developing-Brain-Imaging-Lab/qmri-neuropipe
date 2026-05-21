"""
CLI Tools for direct function access (e.g. Model Fitting).

This module exposes internal library functions as standalone CLI commands.
"""

import logging
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


def _require_gnl_backend(model_name: str, backend: str, grad_nonlin: Optional[Path], supported_backend: str) -> None:
    """Fail fast when a CLI backend cannot apply a requested GNL correction."""
    if grad_nonlin and backend.lower() != supported_backend.lower():
        raise ValueError(
            f"{model_name} gradient nonlinearity correction is only supported with the "
            f"'{supported_backend}' backend, not '{backend}'."
        )


@app.command("enrich-gnl-metadata")
def enrich_gnl_metadata_cli(
    dicom_dir: Path = typer.Option(..., "--dicom-dir", help="Directory containing source GE DICOMs.", exists=True),
    search_root: Optional[Path] = typer.Option(None, "--search-root", help="Root directory under which to find existing *_dwi.json sidecars.", exists=True),
    json_paths: List[Path] = typer.Option([], "--json", help="Specific DWI JSON sidecar(s) or directories containing them."),
    strict: bool = typer.Option(False, "--strict", help="Fail if any target sidecar cannot be matched to a DICOM series."),
    fix_phase_encoding: bool = typer.Option(False, "--fix-phase-encoding", help="Rewrite AP/PA DWI PhaseEncodingDirection to BIDS j/j-."),
    phase_from: str = typer.Option("dir", "--phase-from", help="Preferred source for phase sign: dir, existing, or polarity."),
):
    """
    Enrich existing DWI JSON sidecars with GE gradient nonlinearity metadata from DICOM headers.
    """
    if not search_root and not json_paths:
        console.print("[bold red]Error:[/bold red] provide either --search-root or at least one --json target.")
        raise typer.Exit(code=2)
    if phase_from not in {"dir", "existing", "polarity"}:
        console.print("[bold red]Error:[/bold red] --phase-from must be one of: dir, existing, polarity.")
        raise typer.Exit(code=2)

    try:
        from qmri_neuropipe.lib.common.gnl_metadata import enrich_existing_dwi_sidecars_with_ge_gnl
        logger = logging.getLogger("qmri_tools.gnl_metadata")

        result = enrich_existing_dwi_sidecars_with_ge_gnl(
            dicom_dir=dicom_dir,
            search_root=search_root,
            json_paths=json_paths or None,
            logger=logger,
            strict=strict,
            fix_phase_encoding=fix_phase_encoding,
            phase_from=phase_from,
        )
        console.print(
            "[bold green]Success![/bold green] "
            f"updated={result['updated']}, unchanged={result['unchanged']}, unmatched={len(result['unmatched'])}"
        )
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        import traceback
        traceback.print_exc()
        raise typer.Exit(code=1)


@app.command("patch-ge-dwi-sidecars")
def patch_ge_dwi_sidecars_cli(
    search_root: Optional[Path] = typer.Option(None, "--search-root", help="Root directory under which to find *_dwi.json sidecars.", exists=True),
    json_paths: List[Path] = typer.Option([], "--json", help="Specific DWI JSON sidecar(s) or directories containing them."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print summary without modifying files."),
    no_backup: bool = typer.Option(False, "--no-backup", help="Do not create .bak backups before overwriting JSON files."),
    no_fix_phase_encoding: bool = typer.Option(False, "--no-fix-phase-encoding", help="Do not rewrite PhaseEncodingDirection."),
    phase_from: str = typer.Option("dir", "--phase-from", help="Preferred source for phase sign: dir, existing, or polarity."),
    no_update_isocenter_offset: bool = typer.Option(False, "--no-update-isocenter-offset", help="Only update the PDB center, not IsocenterOffsetScannerRASmm."),
    strict: bool = typer.Option(False, "--strict", help="Fail immediately on the first sidecar error."),
):
    """
    Patch existing GE DWI sidecars from stored RepresentativeDicom/RepresentativeNifti paths.
    """
    if not search_root and not json_paths:
        console.print("[bold red]Error:[/bold red] provide either --search-root or at least one --json target.")
        raise typer.Exit(code=2)
    if phase_from not in {"dir", "existing", "polarity"}:
        console.print("[bold red]Error:[/bold red] --phase-from must be one of: dir, existing, polarity.")
        raise typer.Exit(code=2)

    try:
        from qmri_neuropipe.lib.common.gnl_metadata import patch_existing_dwi_sidecars_from_representatives
        logger = logging.getLogger("qmri_tools.gnl_metadata")

        result = patch_existing_dwi_sidecars_from_representatives(
            search_root=search_root,
            json_paths=json_paths or None,
            logger=logger,
            dry_run=dry_run,
            backup=not no_backup,
            fix_phase_encoding=not no_fix_phase_encoding,
            phase_from=phase_from,
            update_isocenter_offset=not no_update_isocenter_offset,
            strict=strict,
        )
        status = "Dry run complete." if dry_run else "Success!"
        console.print(
            f"[bold green]{status}[/bold green] "
            f"updated={result['updated']}, unchanged={result['unchanged']}, failed={len(result['failed'])}"
        )
        if dry_run:
            for item in result["details"]:
                marker = "would update" if item["changed"] else "unchanged"
                console.print(
                    f"  [{marker}] {item['json']} "
                    f"PhaseEncodingDirection={item['phase']}, "
                    f"PDBCenter={item['pdb_center']}, "
                    f"IsocenterOffset={item['isocenter_offset']}"
                )
        for failure in result["failed"]:
            console.print(f"[yellow]Failed:[/yellow] {failure['json']} - {failure['error']}")
        if result["failed"]:
            raise typer.Exit(code=1)
    except typer.Exit:
        raise
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        import traceback
        traceback.print_exc()
        raise typer.Exit(code=1)


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
    grad_nonlin: Optional[Path] = typer.Option(None, help="Path to gradient nonlinearity tensor file for correction."),
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
                model_type=model_type,
                grad_nonlin=grad_nonlin,
            )
        elif backend.lower() == 'amico':
            _require_gnl_backend("NODDI", backend, grad_nonlin, "dmipy")
            from qmri_neuropipe.interfaces.amico import fit_noddi
            fit_noddi(
                input,
                output_dir,
                bval_file=bval,
                bvec_file=bvec,
                mask_file=mask,
                n_cpus=nthreads,
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


@app.command("fit-sandi")
def fit_sandi_cli(
    input: Path = typer.Option(..., "--input", "-i", help="Input DWI NIfTI file", exists=True),
    output_dir: Path = typer.Option(..., "--output-dir", "-o", help="Output directory"),
    bval: Path = typer.Option(..., "--bval", help="Path to bval file", exists=True),
    bvec: Path = typer.Option(..., "--bvec", help="Path to bvec file", exists=True),
    mask: Optional[Path] = typer.Option(None, "--mask", "-m", help="Path to brain mask", exists=True),
    backend: str = typer.Option("dmipy", "--backend", help="Backend implementation (dmipy or amico)"),
    nthreads: int = typer.Option(1, "--nthreads", "-n", help="Number of threads"),
    parallel_diff: float = typer.Option(1.7e-9, "--parallel-diff", help="Parallel diffusivity"),
    iso_diff: float = typer.Option(3.0e-9, "--iso-diff", help="Isotropic diffusivity"),
    solver: str = typer.Option("brute2fine", "--solver", help="Optimization solver (e.g. brute2fine)"),
    delta: Optional[Path] = typer.Option(None, "--delta", help="Path to small-delta timing file.", exists=True),
    big_delta: Optional[Path] = typer.Option(None, "--big-delta", help="Path to big-Delta timing file.", exists=True),
    grad_nonlin: Optional[Path] = typer.Option(None, help="Path to gradient nonlinearity tensor file for correction."),
):
    """
    Fit SANDI model.
    """
    _setup_threading(nthreads)
    output_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"[bold blue]Running SANDI Fit ({backend})[/bold blue]")

    try:
        if backend.lower() == 'dmipy':
            from qmri_neuropipe.interfaces.dmipy import fit_sandi

            fit_sandi(
                input,
                output_dir,
                bval_file=bval,
                bvec_file=bvec,
                delta_file=delta,
                Delta_file=big_delta,
                mask_file=mask,
                nthreads=nthreads,
                parallel_diffusivity=parallel_diff,
                iso_diffusivity=iso_diff,
                solver=solver,
                grad_nonlin=grad_nonlin,
            )
        elif backend.lower() == 'amico':
            _require_gnl_backend("SANDI", backend, grad_nonlin, "dmipy")
            from qmri_neuropipe.interfaces.amico import fit_sandi

            fit_sandi(
                input,
                output_dir,
                bval_file=bval,
                bvec_file=bvec,
                mask_file=mask,
                n_cpus=nthreads,
                delta_file=delta,
                Delta_file=big_delta,
            )
        else:
            raise ValueError(f"Unknown backend: {backend}")

        console.print("[bold green]Success![/bold green]")
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        import traceback
        traceback.print_exc()
        raise typer.Exit(code=1)


@app.command("fit-microglia")
def fit_microglia_cli(
    input: Path = typer.Option(..., "--input", "-i", help="Input DWI NIfTI file", exists=True),
    output_dir: Path = typer.Option(..., "--output-dir", "-o", help="Output directory"),
    bval: Path = typer.Option(..., "--bval", help="Path to bval file", exists=True),
    bvec: Path = typer.Option(..., "--bvec", help="Path to bvec file", exists=True),
    mask: Optional[Path] = typer.Option(None, "--mask", "-m", help="Path to brain mask", exists=True),
    nthreads: int = typer.Option(1, "--nthreads", "-n", help="Number of threads"),
    parallel_diff: float = typer.Option(1.7e-9, "--parallel-diff", help="Parallel diffusivity"),
    iso_diff: float = typer.Option(3.0e-9, "--iso-diff", help="Isotropic diffusivity"),
    small_diameter: float = typer.Option(4e-6, "--small-diameter", help="Small-sphere diameter in meters."),
    large_diameter: float = typer.Option(8e-6, "--large-diameter", help="Large-sphere diameter in meters."),
    solver: str = typer.Option("brute2fine", "--solver", help="Optimization solver (e.g. brute2fine)"),
    delta: Optional[Path] = typer.Option(None, "--delta", help="Path to small-delta timing file.", exists=True),
    big_delta: Optional[Path] = typer.Option(None, "--big-delta", help="Path to big-Delta timing file.", exists=True),
    grad_nonlin: Optional[Path] = typer.Option(None, help="Path to gradient nonlinearity tensor file for correction."),
):
    """
    Fit the 4-compartment Microglia model.
    """
    _setup_threading(nthreads)
    output_dir.mkdir(parents=True, exist_ok=True)

    console.print("[bold blue]Running Microglia Fit[/bold blue]")

    from qmri_neuropipe.interfaces.dmipy_microglia import fit_microglia

    try:
        fit_microglia(
            input,
            output_dir,
            bval_file=bval,
            bvec_file=bvec,
            delta_file=delta,
            Delta_file=big_delta,
            mask_file=mask,
            nthreads=nthreads,
            parallel_diffusivity=parallel_diff,
            iso_diffusivity=iso_diff,
            small_diameter=small_diameter,
            large_diameter=large_diameter,
            solver=solver,
            grad_nonlin=grad_nonlin,
        )
        console.print("[bold green]Success![/bold green]")
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        import traceback
        traceback.print_exc()
        raise typer.Exit(code=1)


@app.command("fit-nexi")
def fit_nexi_cli(
    input: Path = typer.Option(..., "--input", "-i", help="Input DWI NIfTI file", exists=True),
    output_dir: Path = typer.Option(..., "--output-dir", "-o", help="Output directory"),
    bval: Path = typer.Option(..., "--bval", help="Path to bval file", exists=True),
    td_file: Path = typer.Option(..., "--td-file", help="Path to diffusion time file.", exists=True),
    lowb_noisemap: Path = typer.Option(..., "--lowb-noisemap", help="Path to low-b noise map.", exists=True),
    mask: Optional[Path] = typer.Option(None, "--mask", "-m", help="Path to brain mask", exists=True),
    debug: bool = typer.Option(False, "--debug", help="Enable NEXI debug mode."),
):
    """
    Fit NEXI model.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    console.print("[bold blue]Running NEXI Fit[/bold blue]")

    from qmri_neuropipe.interfaces.nexi import fit_nexi

    try:
        fit_nexi(
            input,
            output_dir,
            bval_file=bval,
            td_file=td_file,
            lowb_noisemap_file=lowb_noisemap,
            mask_file=mask,
            debug=debug,
        )
        console.print("[bold green]Success![/bold green]")
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        import traceback
        traceback.print_exc()
        raise typer.Exit(code=1)


@app.command("fit-fwe-dti")
def fit_fwe_dti_cli(
    input: Path = typer.Option(..., "--input", "-i", help="Input DWI NIfTI file", exists=True),
    output_dir: Path = typer.Option(..., "--output-dir", "-o", help="Output directory"),
    bval: Path = typer.Option(..., "--bval", help="Path to bval file", exists=True),
    bvec: Path = typer.Option(..., "--bvec", help="Path to bvec file", exists=True),
    mask: Optional[Path] = typer.Option(None, "--mask", "-m", help="Path to brain mask", exists=True),
    method: str = typer.Option("NLLS", "--method", help="Fitting method (WLS or NLLS)."),
    nthreads: int = typer.Option(1, "--nthreads", "-n", help="Number of threads"),
    metrics: List[str] = typer.Option(["fa", "md", "ad", "rd", "f"], help="Metrics to calculate."),
    grad_nonlin: Optional[Path] = typer.Option(None, help="Path to gradient nonlinearity tensor file for correction."),
):
    """
    Fit free-water elimination DTI model.
    """
    _setup_threading(nthreads)
    output_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"[bold blue]Running FWE-DTI Fit ({method})[/bold blue]")

    from qmri_neuropipe.interfaces.dipy import fit_fwe_dti

    try:
        fit_fwe_dti(
            in_file=input,
            out_dir=output_dir,
            bval_file=bval,
            bvec_file=bvec,
            mask_file=mask,
            fit_method=method,
            metrics=_parse_metric_list(metrics),
            nthreads=nthreads,
            grad_nonlin=grad_nonlin,
        )
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


@app.command("fit-csd")
def fit_csd_cli(
    input: Path = typer.Option(..., "--input", "-i", help="Input DWI NIfTI file", exists=True),
    output_dir: Path = typer.Option(..., "--output-dir", "-o", help="Output directory"),
    bval: Path = typer.Option(..., "--bval", help="Path to bval file", exists=True),
    bvec: Path = typer.Option(..., "--bvec", help="Path to bvec file", exists=True),
    mask: Optional[Path] = typer.Option(None, "--mask", "-m", help="Path to brain mask", exists=True),
    method: str = typer.Option("msmt_csd", "--method", help="FOD algorithm (msmt_csd or csd)."),
    response_algorithm: str = typer.Option("dhollander", "--response-algorithm", help="Response estimation algorithm."),
    lmax: Optional[str] = typer.Option(None, "--lmax", help="Optional lmax (single value or comma-separated list)."),
    nthreads: int = typer.Option(1, "--nthreads", "-n", help="Number of threads"),
    force: bool = typer.Option(False, "--force", help="Overwrite existing outputs."),
):
    """
    Fit Constrained Spherical Deconvolution (CSD) using MRtrix3.
    """
    _setup_threading(nthreads)
    output_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"[bold blue]Running CSD Fit ({method})[/bold blue]")
    console.print(f"  Input: {input}")
    console.print(f"  Output: {output_dir}")

    from qmri_neuropipe.core.types import DWIFile
    from qmri_neuropipe.interfaces.mrtrix import dwi2response, dwi2fod
    from qmri_neuropipe.io.bids import build_bids_name, get_entities_from_path
    import json

    try:
        dwi = DWIFile(img=input, entities=get_entities_from_path(input), bval=bval, bvec=bvec)
        ent_base = get_entities_from_path(input)
        ent_base.pop("desc", None)
        ent_base["model"] = "CSD"

        response_out = output_dir / "response"
        response_out.mkdir(parents=True, exist_ok=True)
        responses = dwi2response(
            dwi,
            response_out,
            in_bvec=bvec,
            in_bval=bval,
            mask_file=mask,
            algorithm=response_algorithm,
            nthreads=nthreads,
            force=force,
        )

        lmax_value: Optional[object] = None
        if lmax:
            lmax_value = [int(x.strip()) for x in lmax.split(",")] if "," in lmax else int(lmax)

        fods = dwi2fod(
            dwi,
            responses,
            output_dir,
            in_bvec=bvec,
            in_bval=bval,
            mask_file=mask,
            algorithm=method,
            lmax=lmax_value,
            nthreads=nthreads,
            force=force,
        )

        renamed_outputs: list[Path] = []
        for key, path in fods.items():
            suffix = f"{key}FOD"
            if key == "fod":
                suffix = "FOD"

            new_name = build_bids_name({**ent_base, "suffix": suffix})
            new_path = output_dir / new_name

            if path.exists() and path != new_path:
                if new_path.exists() and force:
                    new_path.unlink()
                path.rename(new_path)
            elif path.exists():
                new_path = path

            if new_path.exists():
                sidecar = {
                    "ModelName": "Constrained Spherical Deconvolution",
                    "FittingSoftware": "MRtrix3",
                    "InputData": input.name,
                    "Algorithm": method,
                    "ResponseAlgorithm": response_algorithm,
                }
                with open(str(new_path).replace(".nii.gz", ".json"), "w") as f:
                    json.dump(sidecar, f, indent=4)
                renamed_outputs.append(new_path)

        if not renamed_outputs:
            raise RuntimeError("CSD fitting completed without producing any FOD outputs.")

        console.print("[bold green]Success![/bold green]")
        for out_path in renamed_outputs:
            console.print(f"  {out_path.name}")
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        import traceback
        traceback.print_exc()
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

@app.command("create-gnl-tensor")
@app.command("create-gnl-map")
def create_gnl_map_cli(
    input: Path = typer.Option(..., "--input", "-i", help="Input NIfTI file (processed/final grid).", exists=True),
    coeffs: Path = typer.Option(..., "--coeffs", "-c", help="Gradient nonlinearity coefficients file (.dat).", exists=True),
    output: Path = typer.Option(..., "--output", "-o", help="Output path for the .nii.gz tensor map."),
    initial_image: Optional[Path] = typer.Option(None, "--initial-image", help="Optional native space image (for resampled inputs).", exists=True),
    method: str = typer.Option("tortoise", "--method", help="GNL backend: tortoise, native_ge, native, or latest_native."),
    bval: Optional[Path] = typer.Option(None, "--bval", help="Path to bval file (required if input is 4D).", exists=True),
    bvec: Optional[Path] = typer.Option(None, "--bvec", help="Path to bvec file (required if input is 4D).", exists=True),
    initial_bval: Optional[Path] = typer.Option(None, "--initial-bval", help="Path to bval file for initial image (if different).", exists=True),
    initial_bvec: Optional[Path] = typer.Option(None, "--initial-bvec", help="Path to bvec file for initial image (if different).", exists=True),
    nthreads: int = typer.Option(1, "--nthreads", "-n", help="Number of threads."),
    force: bool = typer.Option(False, "--force", help="Force overwrite existing output."),
):
    """
    Generate a Gradient Nonlinearity (GNL) tensor map using TORTOISE or the native GE implementation.
    """
    _setup_threading(nthreads)
    
    console.print(f"[bold blue]Creating GNL Tensor Map[/bold blue]")
    console.print(f"  Input: {input}")
    console.print(f"  Coeffs: {coeffs}")
    console.print(f"  Output: {output}")
    console.print(f"  Method: {method}")
    
    from qmri_neuropipe.lib.dmri.grad_nonlin import create_gnl_map
    from qmri_neuropipe.core.types import ImageFile
    from qmri_neuropipe.io.bids import _sidecar
    
    # Wrap paths in ImageFile for compatibility with the library function
    input_json = _sidecar(input, ".json")
    input_obj = ImageFile(img=input, entities={}, json=input_json if input_json.exists() else None)
    if bval: input_obj.bval = bval
    if bvec: input_obj.bvec = bvec
    
    native_obj = None
    if initial_image:
        native_json = _sidecar(initial_image, ".json")
        native_obj = ImageFile(img=initial_image, entities={}, json=native_json if native_json.exists() else None)
        # Use specific initial grads if provided, otherwise fallback to main grads
        native_obj.bval = initial_bval if initial_bval else bval
        native_obj.bvec = initial_bvec if initial_bvec else bvec
    
    try:
        create_gnl_map(
            input_image=input_obj,
            output_path=output,
            grad_coeffs=coeffs,
            native_reference=native_obj,
            method=method,
            nthreads=nthreads,
            force=force
        )
        console.print("[bold green]Success![/bold green]")
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        import traceback
        traceback.print_exc()
        raise typer.Exit(code=1)


@app.command("tracker-init")
def tracker_init_cli(
    output: Path = typer.Option(..., "--output", "-o", help="Path to create the tracker Excel file."),
):
    """
    Initialize a new tracker Excel file from the template.
    """
    from qmri_neuropipe.lib.common.tracker import NeuroimagingTracker
    try:
        NeuroimagingTracker.create_empty_tracker(output)
        console.print(f"[bold green]Created tracker at {output}[/bold green]")
    except Exception as e:
        console.print(f"[bold red]Error creating tracker:[/bold red] {e}")
        raise typer.Exit(code=1)


@app.command("tracker-dashboard")
def tracker_dashboard_cli(
    tracker: Optional[Path] = typer.Option(None, "--tracker", "-t", help="Path to tracker Excel file."),
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to pipeline config file (to read tracker_file from)."),
    port: int = typer.Option(8501, "--port", "-p", help="Port to run Streamlit on."),
):
    """
    Launch the Streamlit tracker dashboard.
    """
    import subprocess
    import os
    import yaml
    
    app_path = Path(__file__).parent / "tracker" / "app.py"
    
    if not app_path.exists():
         console.print(f"[bold red]Error:[/bold red] Dashboard app not found at {app_path}")
         raise typer.Exit(code=1)

    # 1. Resolve tracker path
    final_tracker = tracker
    if not final_tracker:
        # Try finding config
        config_path = config
        if not config_path:
             # Look for default preproc.yaml in cwd
             if Path("preproc.yaml").exists():
                 config_path = Path("preproc.yaml")
        
        if config_path and config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    cfg_data = yaml.safe_load(f)
                    t_file = cfg_data.get('tracker_file')
                    if t_file:
                        final_tracker = Path(t_file)
            except Exception as e:
                console.print(f"[yellow]Warning: Failed to read tracker from config {config_path}: {e}[/yellow]")

    if final_tracker:
        os.environ["TRACKER_PATH"] = str(final_tracker.absolute())
        console.print(f"[bold green]Auto-loading tracker: {final_tracker}[/bold green]")
    
    cmd = ["streamlit", "run", str(app_path), "--server.port", str(port)]
    
    # Check if streamlit is installed
    try:
        import streamlit
    except ImportError:
        console.print("[bold yellow]Streamlit not found.[/bold yellow] Please install it with: pip install streamlit plotly")
        raise typer.Exit(code=1)

    console.print(f"[bold blue]Launching Dashboard on port {port}...[/bold blue]")
    console.print(f"[dim]Press Ctrl+C to stop[/dim]")
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        console.print("\n[yellow]Dashboard stopped.[/yellow]")

@app.command("viewer")
def viewer_cli(
    images: List[Path] = typer.Argument(None, help="Generic NIfTI images to load into the viewer"),
    t1: Optional[Path] = typer.Option(None, "--t1", help="T1 map for synthetic generation"),
    t2: Optional[Path] = typer.Option(None, "--t2", help="T2 map for synthetic generation"),
    m0: Optional[Path] = typer.Option(None, "--m0", help="M0 (PD) map for synthetic generation"),
    adc: Optional[Path] = typer.Option(None, "--adc", help="ADC or MD map for synthetic diffusion")
):
    """
    Launch the interactive Plotly Dash NIfTI Image Viewer & Synthetic MRI Generator.
    Requires Dash and PyWebView: pip install -e '.[viewer]'
    """
    from qmri_neuropipe.lib.plotly_viewer import launch_viewer
    
    try:
        launch_viewer(images=images, t1_path=t1, t2_path=t2, m0_path=m0, adc_path=adc)
    except ImportError as e:
        console.print(f"[bold red]Import Error:[/bold red] {e}")
        raise typer.Exit(code=1)
    except Exception as e:
        console.print(f"[bold red]Viewer Error:[/bold red] {e}")
        import traceback
        traceback.print_exc()
        raise typer.Exit(code=1)

if __name__ == "__main__":
    app()
