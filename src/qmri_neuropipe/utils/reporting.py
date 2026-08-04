"""
Reporting utilities for diffusion MRI workflows.

This module contains helper functions for generating reports,
creating visualizations, and exporting metrics.
"""

from pathlib import Path
from typing import Optional, Dict

from qmri_neuropipe.core.utils import get_nifti_stem
from qmri_neuropipe.io.bids import build_bids_name


def report_preprocessing_step(
    reporter,
    step,
    dwi,
    prev_img,
    current_arg,
    target_img,
    figures_dir: Path,
    step_kwargs: Optional[Dict] = None
):
    """
    Report results for a preprocessing step.
    
    Parameters
    ----------
    reporter : ReportGenerator
        Report generator instance
    step : BaseProcessingStep
        The processing step that was executed
    dwi : DWIFile
        Input DWI file
    prev_img : ImageFile
        Previous image (for comparison)
    current_arg : dict or ImageFile
        Current processing result
    target_img : Path, optional
        Target image for coregistration
    figures_dir : Path
        Directory to save figures
    step_kwargs : dict, optional
        Additional keyword arguments passed to the step
    """
    from qmri_neuropipe.lib.reporting.viz import create_ortho_view, plot_comparison
    from qmri_neuropipe.lib.dmri.reorient import DMRIReorientStep
    from qmri_neuropipe.lib.common.denoise import DenoisingStep
    from qmri_neuropipe.lib.common.gibbs import GibbsUnringingStep
    from qmri_neuropipe.lib.dmri.eddy import EddyCorrectionStep
    from qmri_neuropipe.lib.dmri.qc import EddyQuadStep
    from qmri_neuropipe.lib.dmri.topup import TopupStep
    from qmri_neuropipe.lib.dmri.apply_topup import ApplyTopupStep
    from qmri_neuropipe.lib.dmri.drbuddi import NativeDrbuddiStep
    from qmri_neuropipe.lib.common.bias import BiasCorrectionStep
    from qmri_neuropipe.lib.dmri.grad_check import GradientCheckStep
    from qmri_neuropipe.lib.common.registration import CoregistrationStep
    from qmri_neuropipe.lib.dmri.outliers import OutlierRemovalStep
    from qmri_neuropipe.lib.common.mask import BrainMaskingStep
    from qmri_neuropipe.lib.common.resample import ResampleStep
    
    curr_img_obj = current_arg.get("current_image") if isinstance(current_arg, dict) else current_arg
    step_name = step.__class__.__name__
    
    stem = get_nifti_stem(dwi.img) if dwi else "global"
    output_dir = figures_dir.parent
    
    figures_list = []
    details = {}
    tables = []
    
    # Handle different step types
    if isinstance(step, DMRIReorientStep):
        details = {"Method": step.method, "Status": "Completed (Global)"}
        
    elif isinstance(step, DenoisingStep):
        details = {"Method": step.method, "Stem": stem}
        if step.method in ['mppca', 'nlmeans']:
            details["Patch Radius"] = str(step.patch_radius)
            if step.method == 'mppca':
                details["Block Radius"] = str(step.block_radius)
                details["PCA Method"] = step.pca_method
                
        fig_out = figures_dir / f"denoise_comp_{stem}.png"
        if prev_img and curr_img_obj:
            create_ortho_view(curr_img_obj.img, fig_out, title="Denoised DWI (b0)")
            figures_list.append({
                "path": str(fig_out),
                "title": "Denoising",
                "caption": "Denoised Image"
            })
            
    elif isinstance(step, GibbsUnringingStep):
        details = {"Method": step.method, "Stem": stem}
        fig_out = figures_dir / f"gibbs_comp_{stem}.png"
        if curr_img_obj:
            create_ortho_view(curr_img_obj.img, fig_out, title="Gibbs Corrected")
            figures_list.append({
                "path": str(fig_out),
                "title": "Gibbs Unringing",
                "caption": "Gibbs Corrected Image"
            })
            
    elif isinstance(step, EddyCorrectionStep):
        details = {"Method": step.method, "Stem": stem}
        
    elif isinstance(step, EddyQuadStep):
        details = {"Status": "QC Generated"}
        
    elif isinstance(step, TopupStep):
        details = {"Method": "Topup (FSL)"}

    elif isinstance(step, ApplyTopupStep):
        details = {"Method": f"ApplyTopup (FSL, {step.method})"}

    elif isinstance(step, NativeDrbuddiStep):
        details = {
            "Method": "Native DRBUDDI",
            "Transform": step.transform_type,
            "Interpolator": step.interpolator,
            "Symmetric Pairwise": str(step.symmetric_pairwise),
            "PE Axis Constraint": f"{step.pe_axis_constraint:.2f}",
        }

        if curr_img_obj:
            drbuddi_dir = curr_img_obj.img.parent
            before_mean = drbuddi_dir / "drbuddi_before_meanb0.nii.gz"
            after_mean = drbuddi_dir / "drbuddi_after_meanb0.nii.gz"
            if before_mean.exists():
                fig_out = figures_dir / f"drbuddi_before_{stem}.png"
                create_ortho_view(before_mean, fig_out, title="Native DRBUDDI Mean b0 Before")
                figures_list.append({
                    "path": str(fig_out),
                    "title": "Native DRBUDDI Before",
                    "caption": "Mean b0 before native reverse-PE refinement",
                })
            if after_mean.exists():
                fig_out = figures_dir / f"drbuddi_after_{stem}.png"
                create_ortho_view(after_mean, fig_out, title="Native DRBUDDI Mean b0 After")
                figures_list.append({
                    "path": str(fig_out),
                    "title": "Native DRBUDDI After",
                    "caption": "Mean b0 after native reverse-PE refinement",
                })

        qc_rows = []
        for group in current_arg.get("drbuddi_qc", []) if isinstance(current_arg, dict) else []:
            axis = str(group.get("axis", "")).upper()
            qc_rows.append({
                "Axis": axis,
                "Pos Series": str(group.get("positive_series", "")),
                "Neg Series": str(group.get("negative_series", "")),
                "Mean Residual Before": f"{group.get('residual_before_mean', 0.0):.4f}",
                "Mean Residual After": f"{group.get('residual_after_mean', 0.0):.4f}",
                "Improvement %": f"{group.get('improvement_percent', 0.0):.2f}",
            })

            residual_before_path = group.get("residual_before_path")
            if residual_before_path and Path(residual_before_path).exists():
                fig_out = figures_dir / f"drbuddi_residual_before_axis-{axis}_{stem}.png"
                create_ortho_view(Path(residual_before_path), fig_out, title=f"Residual Before ({axis})")
                figures_list.append({
                    "path": str(fig_out),
                    "title": f"Residual Before {axis}",
                    "caption": "Absolute blip-up vs blip-down mean b0 difference before refinement",
                })

            residual_after_path = group.get("residual_after_path")
            if residual_after_path and Path(residual_after_path).exists():
                fig_out = figures_dir / f"drbuddi_residual_after_axis-{axis}_{stem}.png"
                create_ortho_view(Path(residual_after_path), fig_out, title=f"Residual After ({axis})")
                figures_list.append({
                    "path": str(fig_out),
                    "title": f"Residual After {axis}",
                    "caption": "Absolute blip-up vs blip-down mean b0 difference after refinement",
                })

        if qc_rows:
            tables.append({
                "title": "Native DRBUDDI Residual Summary",
                "data": qc_rows,
            })
        
    elif isinstance(step, BiasCorrectionStep):
        details = {"Method": step.method, "Stem": stem}
        fig_out = figures_dir / f"bias_corrected_{stem}.png"
        if curr_img_obj:
            create_ortho_view(curr_img_obj.img, fig_out, title="Bias Corrected")
            figures_list.append({
                "path": str(fig_out),
                "title": "Bias Correction",
                "caption": "Bias Corrected Image"
            })

    elif isinstance(step, GradientCheckStep):
        details = {"Status": "Verified"}
        
    elif isinstance(step, CoregistrationStep):
        details = {"Method": step.method, "Stem": stem}
        if step_kwargs and "options" in step_kwargs:
            opts = step_kwargs["options"]
            if step.method == 'ants':
                details["Transform"] = opts.get('transform_type', 'Rigid')
                details["Interp"] = opts.get('interpolation', 'linear')
            elif step.method == 'fsl':
                details["Cost"] = opts.get('cost', 'normmi')
                details["DOF"] = str(opts.get('dof', 6))

        fig_out = figures_dir / f"coreg_check_{stem}.png"
        if target_img and curr_img_obj:
            plot_comparison(
                target_img,
                curr_img_obj.img,
                fig_out,
                title="Coregistration Check (T1w vs DWI)"
            )
            figures_list.append({
                "path": str(fig_out),
                "title": "Coregistration Quality",
                "caption": "Overlay of aligned DWI (red) on T1w (gray)"
            })
            
    elif isinstance(step, OutlierRemovalStep):
        details = {"Method": step.method, "Stem": stem}
        details["Threshold"] = str(getattr(step, 'threshold', 'N/A'))
        
    elif isinstance(step, BrainMaskingStep):
        details = {"Method": step.method, "Stem": stem}
        fig_out = figures_dir / f"brain_mask_{stem}.png"
        if curr_img_obj:
            create_ortho_view(
                curr_img_obj.img,
                fig_out,
                title="Brain Masked"
            )
            figures_list.append({
                "path": str(fig_out),
                "title": "Brain Masking",
                "caption": "Masked Image"
            })
        
    elif isinstance(step, ResampleStep):
        details = {
            "Resolution": str(getattr(step, 'resolution', 'Target')),
            "Stem": stem
        }

    # Generic catch-all for other steps
    if not details and not figures_list:
        details = {"Status": "Completed"}

    # Add the step to report
    reporter.add_dmri_step(
        step_name,
        details,
        figures=figures_list,
        tables=tables,
        commands=getattr(step, "last_commands", []),
    )
    
    # Check for metrics to report
    if isinstance(current_arg, dict) and ('qc_metrics' in current_arg or 'outlier_stats' in current_arg):
        report_metrics_summary(reporter, dwi, current_arg)


def report_metrics_summary(reporter, dwi, current_arg: Dict):
    """
    Report QC metrics and outlier statistics.
    
    Parameters
    ----------
    reporter : ReportGenerator
        Report generator instance
    dwi : DWIFile
        DWI file being processed
    current_arg : dict
        Processing results containing metrics
    """
    outlier_stats = current_arg.get("outlier_stats")
    tables = []
    
    if outlier_stats:
        main_rows = [
            {"Metric": "Total Volumes", "Value": str(outlier_stats["total_volumes"])},
            {"Metric": "Removed Volumes", "Value": str(outlier_stats["removed_volumes"])},
            {"Metric": "Percent Removed", "Value": f"{outlier_stats['percent_removed']:.2f}%"}
        ]
        tables.append({
            "title": f"Outlier Stats: {dwi.img.name}",
            "data": main_rows
        })
        
        if outlier_stats.get("bvalue_stats"):
            breakdown_rows = []
            for b_stat in outlier_stats["bvalue_stats"]:
                breakdown_rows.append({
                    "B-value": str(b_stat["b_value"]),
                    "Total": str(b_stat["total"]),
                    "Removed": str(b_stat["removed"]),
                    "% Removed": f"{b_stat['percent']:.2f}%"
                })
            tables.append({
                "title": "Outlier Breakdown",
                "data": breakdown_rows
            })

    qc_metrics = current_arg.get("qc_metrics")
    if qc_metrics:
        if "motion" in qc_metrics:
            mo_rows = [
                {"Metric": k, "Value": v}
                for k, v in qc_metrics["motion"].items()
            ]
            tables.append({
                "title": "QC: Motion",
                "data": mo_rows
            })
        if "cnr" in qc_metrics:
            tables.append({
                "title": "QC: CNR",
                "data": qc_metrics["cnr"]
            })
        if "outliers_breakdown" in qc_metrics:
            tables.append({
                "title": "QC: Outliers Breakdown",
                "data": qc_metrics["outliers_breakdown"]
            })
        if "outliers_summary" in qc_metrics:
            out_rows = [
                {"Metric": k, "Value": v}
                for k, v in qc_metrics["outliers_summary"].items()
            ]
            tables.append({
                "title": "QC: Outliers Summary",
                "data": out_rows
            })
        elif "motion" not in qc_metrics:  # Legacy
            qc_rows = [
                {"Metric": k, "Value": v}
                for k, v in qc_metrics.items()
            ]
            tables.append({
                "title": "QC Summary (Eddy Quad)",
                "data": qc_rows
            })
    
    if tables:
        reporter.add_dmri_step("Quality Control", {}, tables=tables)


def report_modeling_step(
    reporter,
    step,
    dwi,
    output_dir: Path,
    report_output_dir: Optional[Path] = None
):
    """
    Report results for a modeling step.
    
    Parameters
    ----------
    reporter : ReportGenerator
        Report generator instance
    step : BaseProcessingStep
        The modeling step that was executed
    dwi : DWIFile
        DWI file that was processed
    output_dir : Path
        Directory where figures should be physically created (staging dir)
    report_output_dir : Path, optional
        Directory to use for the path in the report (final output dir)
        If None, uses output_dir
    """
    from qmri_neuropipe.lib.reporting.viz import create_metric_grid
    from qmri_neuropipe.lib.dmri.fitting import (
        DTIFittingStep,
        DKIFittingStep,
        NODDIFittingStep,
        SANDIFittingStep,
        MAPMRIFittingStep
    )
    
    step_name = step.__class__.__name__
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True, parents=True)
    
    report_base = report_output_dir if report_output_dir else output_dir
    stem = get_nifti_stem(dwi.img)
    
    ents = dwi.entities.copy()
    for k in ['desc', 'suffix']:
        if k in ents:
            del ents[k]
            
    metric_map = {}
    base_model_dir = output_dir
    
    # Define metrics based on step type
    if isinstance(step, DTIFittingStep):
        ents['model'] = 'DTI'
        base_model_dir = output_dir / "DTI"
        metric_map = {
            "FA": {"suffix": "FA", "title": "Fractional Anisotropy (DTI)"},
            "MD": {"suffix": "MD", "title": "Mean Diffusivity (DTI)"},
            "RD": {"suffix": "RD", "title": "Radial Diffusivity (DTI)"},
            "AD": {"suffix": "AD", "title": "Axial Diffusivity (DTI)"}
        }
    elif isinstance(step, DKIFittingStep):
        ents['model'] = 'DKI'
        base_model_dir = output_dir / "DKI"
        metric_map = {
            "MK": {"suffix": "MK", "title": "Mean Kurtosis (DKI)"},
            "RK": {"suffix": "RK", "title": "Radial Kurtosis (DKI)"},
            "AK": {"suffix": "AK", "title": "Axial Kurtosis (DKI)"}
        }
    elif isinstance(step, NODDIFittingStep):
        ents['model'] = 'NODDI'
        base_model_dir = output_dir / "NODDI"
        metric_map = {
            "ODI": {"suffix": "ODI", "title": "Orientation Dispersion Index (NODDI)"},
            "ICVF": {"suffix": "ICVF", "title": "Intra-Cellular Volume Fraction (NODDI)"}
        }
    elif isinstance(step, SANDIFittingStep):
        ents['model'] = 'SANDI'
        base_model_dir = output_dir / "sandi"
        metric_map = {
            "fsoma": {"suffix": "fsoma", "title": "Soma Fraction (SANDI)"}
        }
    elif isinstance(step, MAPMRIFittingStep):
        base_model_dir = output_dir / "mapmri"
        metric_map = {
            "RTOP": {"suffix": "rtop", "title": "Return To Origin Probability (MAPMRI)"},
            "NG": {"suffix": "ng", "title": "Non-Gaussivity (MAPMRI)"}
        }
    else:
        return
        
    # Find and plot metrics
    metrics_to_plot = []
    found_any = False
    
    for key, info in metric_map.items():
        suffix = info['suffix']
        
        # Robust file finding
        fpath = find_metric_file(base_model_dir, key, suffix, ents)
        
        if fpath and fpath.exists():
            found_any = True
            metrics_to_plot.append({
                "path": fpath,
                "title": key,
                "desc": info['title']
            })

    if found_any:
        grid_name = f"{step_name}_grid_{stem}.png"
        grid_out = figures_dir / grid_name
        grid_report = report_base / "figures" / grid_name if report_output_dir else grid_out
        
        create_metric_grid(
            metrics_to_plot,
            grid_out,
            title=f"{ents.get('model', 'dMRI')} Metrics"
        )
        
        details = {
            "Model": ents.get('model', 'Unknown'),
            "Metrics Found": ", ".join([m['title'] for m in metrics_to_plot])
        }
        
        figures = [{
            "path": str(grid_report), 
            "title": f"{ents.get('model')} Metrics Grid", 
            "caption": "Rows: Metrics, Cols: Sagittal / Coronal / Axial"
        }]
        
        reporter.add_dmri_step(
            f"Modeling: {ents.get('model', step_name)}",
            details,
            figures=figures
        )


def find_metric_file(
    base_dir: Path,
    key: str,
    suffix: str,
    entities: Dict
) -> Optional[Path]:
    """
    Find a metric file in a directory with robust matching.
    
    Parameters
    ----------
    base_dir : Path
        Directory to search
    key : str
        Metric key (e.g., 'FA', 'MD')
    suffix : str
        File suffix to match
    entities : dict
        BIDS entities for the file
        
    Returns
    -------
    Path or None
        Path to the found file, or None if not found
    """
    # Search for candidates
    candidates = list(base_dir.glob(f"*{suffix}*"))
    
    sub_id = entities.get("sub")
    ses_id = entities.get("ses")
    
    valid_candidates = []
    for c in candidates:
        if not c.name.endswith((".nii.gz", ".nii")):
            continue
            
        stem_lower = c.name.lower().split(".")[0]
        
        # Skip non-scalar outputs
        if any(x in stem_lower for x in ['tensor', 'color_fa', 'evals', 'evecs']):
            continue
            
        # Strict suffix match
        target_suffix = f"_{key.lower()}"
        if not stem_lower.endswith(target_suffix):
            continue
            
        # Match subject/session
        if sub_id and f"sub-{sub_id}" not in c.name:
            continue
        if ses_id and f"ses-{ses_id}" not in c.name:
            continue
        
        valid_candidates.append(c)
        
    if valid_candidates:
        # Prefer file with model label
        best_cand = valid_candidates[0]
        model_label = entities.get("model", "")
        if model_label:
            for vc in valid_candidates:
                if f"model-{model_label}" in vc.name:
                    best_cand = vc
                    break
        return best_cand
    else:
        # Try strict BIDS naming
        fname = build_bids_name(entities, suffix=suffix)
        fpath_strict = base_dir / fname
        if fpath_strict.exists():
            return fpath_strict
    
    return None
