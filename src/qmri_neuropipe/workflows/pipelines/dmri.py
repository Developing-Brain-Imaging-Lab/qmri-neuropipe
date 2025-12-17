# my_dmri_pipeline.py
from pathlib import Path
from qmri_neuropipe.core import (
    BasePipeline, BaseWorkflow, PipelineConfig,
)
from qmri_neuropipe.core.types import ImageFile, DWIFile
from qmri_neuropipe.io.bids import _load_json_field
from qmri_neuropipe.io.anat.bids import bids_find_t1w, bids_find_t2w
from qmri_neuropipe.io.dmri.bids import bids_find_dwi, find_reversed_phase_groups
from qmri_neuropipe.io.fmap.bids import bids_find_fmap
from qmri_neuropipe.lib.common.denoise import DenoisingStep
from qmri_neuropipe.lib.common.gibbs import GibbsUnringingStep
from qmri_neuropipe.lib.dmri.eddy import EddyCorrectionStep
from qmri_neuropipe.lib.dmri.synb0 import Synb0EstimationStep
from qmri_neuropipe.lib.dmri.topup import TopupStep
from qmri_neuropipe.lib.common.bias import BiasCorrectionStep
from qmri_neuropipe.lib.dmri.grad_nonlin import TortoiseGradNonlinCorrectStep
from qmri_neuropipe.lib.common.registration import CoregistrationStep
from qmri_neuropipe.lib.dmri.grad_check import GradientCheckStep

from qmri_neuropipe.lib.common.mask import BrainMaskingStep
from .anat import AnatPreprocessingWorkflow
from qmri_neuropipe.lib.common.mask import BrainMaskingStep
from .anat import AnatPreprocessingWorkflow
from qmri_neuropipe.lib.common.resample import ResampleStep
from qmri_neuropipe.interfaces.mriqc import run_mriqc
from qmri_neuropipe.lib.reporting.report import ReportGenerator
from qmri_neuropipe.lib.reporting.viz import create_ortho_view, plot_comparison
from qmri_neuropipe.lib.dmri.outliers import OutlierRemovalStep
from qmri_neuropipe.lib.dmri.qc import EddyQuadStep
import time
try:
    from rich.table import Table
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
    console = Console()
except ImportError:
    console = None
    Progress = None


# Define a simple preprocessing workflow
class PreprocessingWorkflow(BaseWorkflow):
    """
    Preprocessing workflow for DWI:
    - Takes a context dict (DWI files, topup groups, etc.)
    - Applies configurable steps in sequence.
    """

    def _initialize_steps(self):
        self.steps = []

    def build_pipeline(self, context: dict):
        dwi_files: list[DWIFile] = context.get("dwi_files", [])
        topup_groups = context.get("topup_groups", [])
        
        # 1. Audit inputs
        # self._audit_inputs(dwi_files, topup_groups) # Removed at user request
        self.logger.info(f"Auditing inputs: {len(dwi_files)} DWI files, {len(topup_groups)} topup groups.")

        dmri_cfg = self.config.get('dmri', {}).get('preprocessing', {})
        
        # 0. Resampling (New Step)
        res_cfg = dmri_cfg.get('resample', {})
        if res_cfg.get('enabled', False):
             self.logger.info("Adding ResampleStep (per-image)...")
             self.add_step(ResampleStep(self.config, self.logger, self.provenance, resolution=res_cfg.get('resolution')))
        
        # 0. Gradient Check (Initial Data Verification)
        # New config key: dmri.preprocessing.grad_check.enabled
        # from qmri_neuropipe.lib.dmri.grad_check import GradientCheckStep # Moved to top-level imports
        grad_check_cfg = dmri_cfg.get('grad_check', {})
        if grad_check_cfg.get('enabled', False):
            self.logger.info("Adding GradientCheckStep (initial verification)...")
            # Note: This is an "initial" step but it operates per-image.
            # However, PreprocessingWorkflow separates logic into Global vs Per-Image.
            # GradientCheckStep is Per-Image. We add it to self.steps.
            # It should run BEFORE other per-image steps.
            self.add_step(GradientCheckStep(self.config, self.logger, self.provenance))



        # 1. Distortion Correction Strategy (unified)
        # New config key: dmri.preprocessing.distcorr
        #   method: "synb0" | "topup" | "none"
        #   fallback: bool (optional, fallback to synb0 when topup requested but no reverse PE)
        distcorr_cfg = dmri_cfg.get('distcorr', {})
        dist_method = distcorr_cfg.get('method', 'none')
        fallback = distcorr_cfg.get('fallback', False)

        has_reverse_pe = len(topup_groups) > 0
        t1w_files = context.get("t1w_files", [])

        # Apply selected distortion correction method
        if dist_method == 'synb0':
            if t1w_files:
                self.logger.info("Adding Synb0EstimationStep (synthetic reverse‑PE b0)...")
                self.add_step(Synb0EstimationStep(self.config, self.logger, self.provenance))
                self.logger.info("Adding TopupStep (using synthetic b0)")
                self.add_step(TopupStep(self.config, self.logger, self.provenance))
                context['do_topup'] = True
            else:
                self.logger.warning("Synb0 requested but no T1w files found. Skipping Synb0/Topup.")
        elif dist_method == 'topup':
            if has_reverse_pe:
                self.logger.info("Adding TopupStep (native reverse‑PE data found)")
                self.add_step(TopupStep(self.config, self.logger, self.provenance))
                context['do_topup'] = True
            elif fallback:
                if t1w_files:
                    self.logger.info("Fallback: Adding Synb0EstimationStep (synthetic reverse‑PE b0)...")
                    self.add_step(Synb0EstimationStep(self.config, self.logger, self.provenance))
                    self.logger.info("Adding TopupStep (using synthetic b0)")
                    self.add_step(TopupStep(self.config, self.logger, self.provenance))
                    context['do_topup'] = True
                else:
                    self.logger.warning("Fallback to Synb0 requested but no T1w files found. Skipping distortion correction.")
            else:
                self.logger.warning("Topup requested but no reverse‑PE data found. Skipping distortion correction.")
        elif dist_method == 'none':
            self.logger.info("Distortion correction disabled via distcorr.method='none'.")
        else:
            self.logger.warning(f"Unknown distcorr method '{dist_method}'. No distortion correction step added.")

        
        # 2. Denoising
        denoise_cfg = dmri_cfg.get('denoising', {})
        if denoise_cfg.get('enabled', True):
            method = denoise_cfg.get('method', 'mrtrix')
            self.logger.info(f"Adding DenoisingStep (method={method})")
            self.add_step(DenoisingStep(
                config=self.config, 
                logger=self.logger, 
                provenance=self.provenance,
                method=method,
                patch_radius=denoise_cfg.get('patch_radius', 2),
                block_radius=denoise_cfg.get('block_radius', 5)
            ))
            
        # 3. Gibbs Unringing
        degibbs_cfg = dmri_cfg.get('degibbs', {})
        if degibbs_cfg.get('enabled', True):
            method = degibbs_cfg.get('method', 'mrtrix')
            self.logger.info(f"Adding GibbsUnringingStep (method={method})")
            self.add_step(GibbsUnringingStep(
                config=self.config, 
                logger=self.logger, 
                provenance=self.provenance,
                method=method
            ))
            
        # 4. Eddy Current Correction
        eddy_cfg = dmri_cfg.get('eddy', {})
        run_eddy = eddy_cfg.get('enabled', True)
        if run_eddy:
             method = eddy_cfg.get('method', 'eddy')
             self.logger.info(f"Adding EddyCorrectionStep (method={method})")
             self.add_step(EddyCorrectionStep(
                config=self.config, 
                logger=self.logger, 
                provenance=self.provenance,
                method=method
            ))

        # 4.6 Eddy QC (Quad) - Automatic if eddy is run
        # User requested: if eddy is run, then eddy_quad should also be run
        if run_eddy:
             # Ensure method is compatible (fsl eddy)
             eddy_method = eddy_cfg.get('method', 'eddy')
             if eddy_method == 'eddy':
                 self.logger.info("Adding EddyQuadStep (Automatic with Eddy)")
                 self.add_step(EddyQuadStep(
                     config=self.config,
                     logger=self.logger,
                     provenance=self.provenance
                 ))

        # 4.5 Outlier Removal
        outlier_cfg = dmri_cfg.get('outliers', {})
        if outlier_cfg.get('enabled', False):
              method = outlier_cfg.get('method', 'manual')
              self.logger.info(f"Adding OutlierRemovalStep (method={method})")
              self.add_step(OutlierRemovalStep(
                  config=self.config,
                  logger=self.logger,
                  provenance=self.provenance,
                  method=method,
                  threshold=outlier_cfg.get('threshold', 0.05),
                  manual_indices=outlier_cfg.get('manual_indices')
              ))

        # 5. Bias Field Correction
        bias_cfg = dmri_cfg.get('bias_correction', {})
        # Check explicit enabled flag in 'dmri.preprocessing.bias_correction' OR top-level legacy 'do_bias_correction' (default False)
        do_bias = bias_cfg.get('enabled') or self.config.get("do_bias_correction", False)
        
        if do_bias and bias_cfg.get('enabled', True) != False: # Handle explicit 'false' if mixed
             # If bias_cfg exists, use it. If not, fallback to top-level.
             # Careful: if bias_cfg['enabled'] is False, we should skip.
             if bias_cfg.get('enabled') is False:
                 do_bias = False
             
        if do_bias:
            method = bias_cfg.get('method') or self.config.get("bias_method", "ants")
            self.logger.info(f"Adding BiasCorrectionStep (method={method})")
            self.add_step(BiasCorrectionStep(
                config=self.config, 
                logger=self.logger, 
                provenance=self.provenance,
                method=method
            ))

        # 6. Gradient Nonlinearity Correction (Tortoise)
        gnl_cfg = dmri_cfg.get('grad_nonlin', {})
        if gnl_cfg.get('enabled', False):
             self.logger.info(f"Adding TortoiseGradNonlinCorrectStep")
             self.add_step(TortoiseGradNonlinCorrectStep(
                 config=self.config,
                 logger=self.logger,
                 provenance=self.provenance
             ))

        # 7. Coregistration (to T1w)
        coreg_cfg = dmri_cfg.get('coregistration', {})
        do_coreg = coreg_cfg.get('enabled') or self.config.get("do_coregistration", False)
        
        if coreg_cfg.get('enabled') is False:
             do_coreg = False
             
        if do_coreg:
            method = coreg_cfg.get('method') or self.config.get("coreg_method", "ants")
            self.logger.info(f"Adding CoregistrationStep (method={method})")
            self.add_step(CoregistrationStep(
                config=self.config, 
                logger=self.logger, 
                provenance=self.provenance,
                method=method
            ))

        # 8. Final Brain Masking
        mask_cfg = dmri_cfg.get('brain_masking', {})
        do_masking = mask_cfg.get('enabled') or self.config.get("do_brain_masking", False)

        if do_masking:
            method = mask_cfg.get('method') or self.config.get("masking_method", "mrtrix")
            self.logger.info(f"Adding BrainMaskingStep (method={method})")
            self.add_step(BrainMaskingStep(
                config=self.config,
                logger=self.logger,
                provenance=self.provenance,
                method=method
            ))


    def _audit_inputs(self, dwi_files: list[DWIFile], topup_groups: list):
        """Log detailed input summary using Rich if available."""
        if not console:
            return

        table = Table(title="DWI Input Audit")
        table.add_column("File", style="cyan")
        table.add_column("Ents", style="magenta")
        table.add_column("PE Dir", style="green")
        
        for d in dwi_files:
            pe = _load_json_field(d.json, "PhaseEncodingDirection") or "N/A"
            ents = ",".join(f"{k}={v}" for k, v in d.entities.items() if k not in ['sub','ses'])
            table.add_row(d.img.name, ents, pe)
            
        console.print(table)
        
        if topup_groups:
            console.print(f"[bold green]Topup Groups:[/bold green] {len(topup_groups)}")



    def _update_json_history(self, json_path: Path, steps: list):
        """Update JSON sidecar with processing history (Helper)."""
        import json
        history_msg = "Pipeline Steps: " + ", ".join([s.__class__.__name__ for s in steps])
        
        data = {}
        if json_path.exists():
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
            except Exception:
                pass
        
        prev_history = data.get("History", "")
        if prev_history:
             data["History"] = prev_history + "; " + history_msg
        else:
             data["History"] = history_msg
             
        data["ProcessingSteps"] = [s.__class__.__name__ for s in steps]

        with open(json_path, 'w') as f:
            json.dump(data, f, indent=4)


    def run(self, output_dir: Path, context: dict, reporter=None) -> dict:
        """
        Execute the workflow on a list of DWI files.
        """
        dwi_files: list[DWIFile] = context.get("dwi_files", [])
        self.logger.info(f"Starting PreprocessingWorkflow for {len(dwi_files)} files.")
        
        # Prepare context
        context = dict(context) 
        
        # Reporting: Inputs
        if reporter and dwi_files:
            reporter.add_section("Diffusion Inputs", f"DWI Files: {len(dwi_files)}")
            dwi = dwi_files[0]
            # Plot b0 or first volume
            p = output_dir / "report_input_dwi_b0.png"
            # Extract first volume if 4D? ants.image_read reads 4D. 
            # create_ortho_view handles 3D. 
            # ants.plot might handle 4D by plotting mean?
            # Let's try to pass the file path, our viz tool reads it.
            # If 4D, we might want to extract b0?
            # For now, let's rely on basic plot behavior or just catch error.
            try:
                 # TODO: Explicitly valid extraction for 4D?
                 # Assuming viz tool handles or we trust it.
                 create_ortho_view(dwi.img, p, title="Input DWI (first volume)")
                 reporter.add_figure("Input DWI", p, caption=dwi.img.name)
            except Exception as e:
                 self.logger.warning(f"Failed to plot input DWI: {e}")

        # Separate Global vs Per-Image steps
        global_steps = []
        per_image_steps = []
        
        for step in self.steps:
            if isinstance(step, (Synb0EstimationStep, TopupStep, GradientCheckStep)):
                 global_steps.append(step)
            else:
                 per_image_steps.append(step)
        
        # Prepare for execution with granular progress
        total_steps = len(global_steps) + (len(dwi_files) * len(per_image_steps))
        
        # Helper to execute the core loop logic
        def _execute_processing(progress_ctx=None, task_id=None):
            nonlocal context
            nonlocal dwi_files # Also used and reassigned
            
            # Run Global Steps
            for step in global_steps:
                 if progress_ctx:
                     progress_ctx.update(task_id, description=f"[cyan]Global Step: {step.__class__.__name__}")
                 
                 self.logger.info(f"Executing Global Step: {step.__class__.__name__}...")
                 
                 # Capture old dwi_files
                 old_dwis = context.get("dwi_files", [])
                 
                 # Pass T1w context implicitly via context dict
                 # We need to update the outer context variable, but it's local scope.
                 # Dict is mutable, so updating keys works. Reassigning 'context' var doesn't update outer scope.
                 # Actually context is passed as arg, so it's a reference. But the line context = step.run(...) reapplies it.
                 # Let's be careful.
                 new_ctx = step.run(context, output_dir=output_dir)
                 
                 # Merge updates back to original context dict reference if possible, or update nonlocal
                 # Usually step returns a modified copy or same dict.
                 if new_ctx is not context:
                     context.update(new_ctx)
                 
                 # Check if dwi_files changed
                 new_dwis = context.get("dwi_files", [])
                 if new_dwis is not old_dwis or (new_dwis and old_dwis and new_dwis[0].img != old_dwis[0].img):
                     self.logger.info("DWI files updated by global step. Refreshing Topup Groups...")
                     from qmri_neuropipe.io.dmri.bids import find_reversed_phase_groups
                     context["topup_groups"] = find_reversed_phase_groups(new_dwis)
                 
                 # Save intermediate output
                 if self.config.get("save_intermediates", False):
                     self._save_global_intermediate(step, output_dir, context)

                 if progress_ctx:
                     progress_ctx.advance(task_id)

            # Run Per-Image Steps
            processed_dwis = []
            processed_masks = []
            topup_map = context.get("topup_map", {})
            
            dwi_files = context.get("dwi_files", []) # Refresh in case global steps changed them

            for dwi_idx, dwi in enumerate(dwi_files, 1):
                if not progress_ctx and not console:
                     self.logger.info(f"Processing image: {dwi.img.name}")
                
                # Per-image context
                ctx = dict(context)
                ctx["current_image"] = dwi
                if dwi.img in topup_map:
                    ctx["topup_base"] = topup_map[dwi.img]
                
                t1w_files = ctx.get("t1w_files", [])
                target_img = t1w_files[0].img if t1w_files else None
                
                current_arg = ctx
                step_metrics = [] # Local for report

                # Report figures directory (created once per image if needed, or already exists)
                figures_dir = output_dir / "figures"
                if reporter: figures_dir.mkdir(exist_ok=True, parents=True)

                for step in per_image_steps:
                    step_name = step.__class__.__name__
                    if progress_ctx:
                         progress_ctx.update(task_id, description=f"[cyan]File {dwi_idx}/{len(dwi_files)}: {step_name}")
                    
                    # --- Step Execution Logic (Simplified for brevity but preserving functional logic) ---
                    step_kwargs = {}
                    if isinstance(step, CoregistrationStep):
                        if target_img:
                            step_kwargs["target"] = target_img
                            coreg_cfg = self.config.get('dmri', {}).get('preprocessing', {}).get('coregistration', {})
                            step_kwargs["options"] = coreg_cfg
                        else:
                            self.logger.warning(f"Skipping CoregistrationStep for {dwi.img.name} - No structural target found.")
                            step_metrics.append({"Step": step_name, "Status": "Skipped", "Duration": "0s"})
                            if progress_ctx: progress_ctx.advance(task_id)
                            continue
                            
                    if isinstance(step, BrainMaskingStep):
                            step_kwargs["return_mask"] = True
                    
                    prev_img_obj = current_arg.get("current_image") if isinstance(current_arg, dict) else current_arg
                    
                    try:
                        st = time.time()
                        current_arg = step.run(current_arg, output_dir=output_dir, **step_kwargs)
                        dur = time.time() - st
                        
                        step_metrics.append({"Step": step_name, "Status": "Completed", "Duration": f"{dur:.2f}s"})
                        
                        # Reporting & plotting logic preserved but condensed call
                        if reporter: self._report_step(reporter, step, dwi, prev_img_obj, current_arg, target_img, figures_dir)
                        
                        # Save intermediates
                        if self.config.get("save_intermediates", False):
                            self._save_image_intermediate(output_dir, context, current_arg)

                    except Exception as e:
                        if self.config.get("stop_on_error", False):
                            raise e
                        self.logger.error(f"Step {step_name} failed: {e}")
                        step_metrics.append({"Step": step_name, "Status": "Failed", "Duration": "0s"})
                        break

                    if progress_ctx:
                         progress_ctx.advance(task_id)

                # Post-image processing (Reporting Summary)
                if reporter:
                    reporter.add_summary_table("Execution Summary", step_metrics)
                    self._report_metrics_summary(reporter, dwi, current_arg)

                # Collect results
                result_ctx = current_arg
                if isinstance(result_ctx, dict):
                    processed_dwis.append(result_ctx.get("current_image", dwi))
                    processed_masks.append(result_ctx.get("current_mask"))
                else:
                    processed_dwis.append(result_ctx)
                    processed_masks.append(None)
            
            context["preprocessed_dwis"] = processed_dwis
            context["preprocessed_masks"] = processed_masks
            context["current_image"] = processed_dwis[-1] if processed_dwis else context.get("current_image")
            return context

        # Execution Wrapper with Rich
        if Progress:
             with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeRemainingColumn(),
                transient=True 
             ) as progress:
                 task = progress.add_task(f"[cyan]Starting Preprocessing...", total=total_steps)
                 return _execute_processing(progress_ctx=progress, task_id=task)
        else:
             return _execute_processing()

    def _save_global_intermediate(self, step, output_dir, context):
         import shutil
         # Logic from original run()
         # ... (preserving logic, simplified for injection)
         sub = context.get("subject")
         ses = context.get("session")
         base_out = self.config.output_dir
         final_dwi_dir = base_out / f"sub-{sub}"
         if ses: final_dwi_dir /= f"ses-{ses}"
         final_dwi_dir /= "dwi"
         dest_inter = final_dwi_dir / "intermediate"
         dest_inter.mkdir(parents=True, exist_ok=True)
         
         if isinstance(step, TopupStep):
              src = output_dir / "topup"
              if src.exists(): shutil.copytree(src, dest_inter / "topup", dirs_exist_ok=True)
         elif isinstance(step, Synb0EstimationStep):
              src = output_dir / "synb0"
              if src.exists(): shutil.copytree(src, dest_inter / "synb0", dirs_exist_ok=True)

    def _save_image_intermediate(self, output_dir, context, current_arg):
         import shutil
         sub = context.get("subject")
         ses = context.get("session")
         base_out = self.config.output_dir
         final_dwi_dir = base_out / f"sub-{sub}"
         if ses: final_dwi_dir /= f"ses-{ses}"
         final_dwi_dir /= "dwi"
         inter_dir = final_dwi_dir / "intermediate"
         inter_dir.mkdir(parents=True, exist_ok=True)
         
         curr_img = current_arg.get("current_image") if isinstance(current_arg, dict) else None
         if curr_img and hasattr(curr_img, "img") and curr_img.img.exists():
             src_dir = curr_img.img.parent
             if src_dir != output_dir and output_dir in src_dir.parents:
                  step_folder_name = src_dir.name
                  target_step_dir = inter_dir / step_folder_name
                  shutil.copytree(src_dir, target_step_dir, dirs_exist_ok=True)
                  self.logger.info(f"Saved intermediate directory: {target_step_dir}")

    def _report_step(self, reporter, step, dwi, prev_img, current_arg, target_img, figures_dir):
         # Extract reporting logic
         curr_img_obj = current_arg.get("current_image") if isinstance(current_arg, dict) else current_arg
         if isinstance(step, DenoisingStep):
             fig_out = figures_dir / f"denoise_comp_{dwi.img.stem}.png"
             if prev_img and curr_img_obj:
                create_ortho_view(curr_img_obj.img, fig_out, title="Denoised DWI (b0)")
                reporter.add_figure("Denoising", fig_out, caption="Denoised Image (Orthogonal View)")
         elif isinstance(step, GibbsUnringingStep):
             fig_out = figures_dir / f"gibbs_comp_{dwi.img.stem}.png"
             if curr_img_obj:
                 create_ortho_view(curr_img_obj.img, fig_out, title="Gibbs Corrected")
                 reporter.add_figure("Gibbs Unringing", fig_out, caption="Gibbs Corrected Image")
         elif isinstance(step, CoregistrationStep):
             fig_out = figures_dir / f"coreg_check_{dwi.img.stem}.png"
             if target_img and curr_img_obj:
                 plot_comparison(target_img, curr_img_obj.img, fig_out, title="Coregistration Check (T1w vs DWI)")
                 reporter.add_figure("Coregistration Quality", fig_out, caption="Overlay of aligned DWI (red/overlay) on T1w (gray)")

    def _report_metrics_summary(self, reporter, dwi, current_arg):
         # Extract summary reporting logic
         outlier_stats = current_arg.get("outlier_stats")
         if outlier_stats:
             main_rows = [
                 {"Metric": "Total Volumes", "Value": str(outlier_stats["total_volumes"])},
                 {"Metric": "Removed Volumes", "Value": str(outlier_stats["removed_volumes"])},
                 {"Metric": "Percent Removed", "Value": f"{outlier_stats['percent_removed']:.2f}%"}
             ]
             reporter.add_summary_table(f"Outlier Removal Summary: {dwi.img.name}", main_rows)
             
             if outlier_stats.get("bvalue_stats"):
                 breakdown_rows = []
                 for b_stat in outlier_stats["bvalue_stats"]:
                     breakdown_rows.append({
                         "B-value": str(b_stat["b_value"]),
                         "Total": str(b_stat["total"]),
                         "Removed": str(b_stat["removed"]),
                         "% Removed": f"{b_stat['percent']:.2f}%"
                     })
                 reporter.add_summary_table(f"Outlier Removal Breakdown: {dwi.img.name}", breakdown_rows)

         qc_metrics = current_arg.get("qc_metrics")
         if qc_metrics:
             if "motion" in qc_metrics:
                 mo_rows = [{"Metric": k, "Value": v} for k, v in qc_metrics["motion"].items()]
                 reporter.add_summary_table(f"QC: Motion Statistics ({dwi.img.name})", mo_rows)
             if "cnr" in qc_metrics:
                 reporter.add_summary_table(f"QC: Contrast/SNR per Shell ({dwi.img.name})", qc_metrics["cnr"])
             if "outliers_breakdown" in qc_metrics:
                 reporter.add_summary_table(f"QC: Outliers Breakdown ({dwi.img.name})", qc_metrics["outliers_breakdown"])
             if "outliers_summary" in qc_metrics:
                 out_rows = [{"Metric": k, "Value": v} for k, v in qc_metrics["outliers_summary"].items()]
                 reporter.add_summary_table(f"QC: Outliers Summary ({dwi.img.name})", out_rows)
             elif "motion" not in qc_metrics: # Legacy
                 qc_rows = [{"Metric": k, "Value": v} for k, v in qc_metrics.items()]
                 reporter.add_summary_table(f"QC Summary (Eddy Quad): {dwi.img.name}", qc_rows)
    

# Define complete pipeline
class DMRIPipeline(BasePipeline):
    """Diffusion MRI Processing pipeline."""
    
    @property
    def name(self):
        return 'dmri-pipeline'
    
    @property
    def version(self):
        return '1.0.0'
    
    def _initialize_pipeline(self):
        self.preprocessing = PreprocessingWorkflow(self.config, self.logger, self.provenance)
        # Initialize Anat Workflow if configured (we assume it might be used)
        self.anat_preprocessing = AnatPreprocessingWorkflow(self.config, self.logger, self.provenance)
    
    def process_subject(self, subject: str, session: str | None = None):
        ses = f"ses-{session}" if session else ""
        subj_dir = (Path(self.config.get('bids_dir')) / f'sub-{subject}' / ses)
        anat_dir = (Path(subj_dir / 'anat'))
        dwi_dir  = (Path(subj_dir / 'dwi'))
        fmap_dir = (Path(subj_dir / 'fmap'))

        # Prepare Work Directory (for intermediate steps)
        work_root = Path(self.config.get('work_dir'))
        if session:
            subj_work_dir = work_root / f'sub-{subject}' / f'ses-{session}' / 'dwi'
            anat_work_dir = work_root / f'sub-{subject}' / f'ses-{session}' / 'anat'
        else:
            subj_work_dir = work_root / f'sub-{subject}' / 'dwi'
            anat_work_dir = work_root / f'sub-{subject}' / 'anat'
        
        subj_work_dir.mkdir(parents=True, exist_ok=True)
        anat_work_dir.mkdir(parents=True, exist_ok=True)

        # Prepare Final Output Directory
        output_dir = self._get_output_dir(subject, session) / 'dwi'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 0. Run MRIQC if enabled (Before any processing)
        qc_cfg = self.config.get("qc", {}).get("mriqc", {})
        if qc_cfg.get("enabled"):
             # Determine modalities (Default for DMRIPipeline is dwi)
             # However, allow T1w/T2w if user requested them, as pipeline has access to them
             pipeline_mods = {'dwi', 'T1w', 'T2w'}
             cfg_mods = qc_cfg.get("modalities")
             
             if cfg_mods:
                 target_mods = list(pipeline_mods.intersection(set(cfg_mods)))
             else:
                 target_mods = list(pipeline_mods)
             
             if target_mods:
                 self.logger.info(f"Running MRIQC for sub-{subject} {ses} (modalities={target_mods})")
                 mriqc_out = self.config.output_dir.parent / "mriqc"
                 try:
                     run_mriqc(
                         bids_dir=self.config.bids_dir,
                         output_dir=mriqc_out,
                         participant_label=subject,
                         session_id=session,
                         n_procs=self.config.n_cpus,
                         modalities=target_mods,
                         verbose_reports=self.config.verbose
                     )
                 except Exception as e:
                     self.logger.warning(f"MRIQC failed: {e}. continuing pipeline...")
                     if self.config.stop_on_error:
                          raise e
             else:
                 self.logger.info(f"Skipping MRIQC for dmri pipeline (Computed modalities empty. Config: {cfg_mods})")

        # 1. Build the Preprocessing Context 
        t1w_files:  list[ImageFile] = bids_find_t1w(anat_dir)
        t2w_files:  list[ImageFile] = bids_find_t2w(anat_dir)
        dwi_files:  list[DWIFile]   = bids_find_dwi(dwi_dir)
        
        if not dwi_files:
            self.logger.warning(f"No DWI files found for sub-{subject} {ses}. Skipping.")
            return

        # 1.5. Run Anatomical Pipeline if enabled
        # Check config: anat.preprocessing.enabled? Or just simple check if anat block exists?
        # User said "if anat and dmri pipelines are enabled".
        # We can check self.config.get("anat", {}).get("preprocessing", {}).get("enabled", False)
        # Or implicitly if 'anat' key exists and is dict
        # Config structure from user request: anat: preprocessing: ...
        
        anat_cfg = self.config.get("anat", {}).get("preprocessing", {})
        # Simplistic check: if we have some config steps enabled? 
        # Or just assume if config has 'anat', we run it?
        # Safe bet: Check if 'anat' is in config.
        # But CLI might default values.
        # Let's rely on explicit "enabled" if present, or existence of block.
        # To be safe against empty block, assume we run if 'anat' key exists in generic config_data?
        # But merge_cli_and_config creates a config object. 
        # BasePipeline stores `config` object which has `config_data` dict.
        
        run_anat = False
        if self.config.get("anat") and anat_cfg:
             # run it
             run_anat = True
        
        # Initialize Reporter
        # Reporting output usually in dwi dir or top level?
        # User requested generic reporting.
        # Let's put it in the diffusion output directory or a separate 'report' dir?
        # Standard: 'sub-XX/ses-YY/dwi/report.html'.
        # Let's use the 'dwi' output dir for simplicity or better, a dedicated sibling dir if desired.
        # But 'output_dir' variable here points to '.../dwi'.
        # User request: save one directory up (output_dir.parent).
        report_title = f"Diffusion Pipeline Report: sub-{subject} {ses}"
        reporter = ReportGenerator(output_dir.parent, title=report_title)

        if run_anat:
             self.logger.info("="*60)
             self.logger.info("  RUNNING ANATOMICAL PIPELINE  ")
             self.logger.info("="*60)
             self.logger.info("Anatomical Pipeline enabled (via config). Running AnatPreprocessing...")
             
             anat_context = {
                 "subject": subject,
                 "session": session,
                 "t1w_files": t1w_files,
                 "t2w_files": t2w_files
             }
             
             # Run Anat Workflow
             # Run Anat Workflow
             # Writes to anat_work_dir
             anat_final_dir = self._get_output_dir(subject, session) / 'anat'
             anat_results = self.anat_preprocessing.run(anat_work_dir, anat_context, final_output_dir=anat_final_dir, reporter=reporter)
             
             # Check for preprocessed outputs
             pre_t1 = anat_results.get("preprocessed_t1w")
             pre_t2 = anat_results.get("preprocessed_t2w_coreg") or anat_results.get("preprocessed_t2w")
             
             if pre_t1:
                  self.logger.info(f"Using preprocessed T1w as structural reference.")
                  t1w_files = [pre_t1] # Replace lists
             
             if pre_t2:
                  self.logger.info(f"Using preprocessed T2w as additional reference.")
                  t2w_files = [pre_t2]


        # 2. Check if dMRI processing is enabled
        dmri_cfg = self.config.get("dmri")
        if not dmri_cfg:
             self.logger.info("dMRI processing not enabled or missing in config. Skipping diffusion steps.")
             
             # Generate report even if only anat ran
             reporter.generate()
             try:
                 reporter.generate_pdf()
             except Exception:
                 pass
             return

        # Group by reversed phase encoding for TOPUP/EDDY
        self.logger.info("="*60)
        self.logger.info("  RUNNING DIFFUSION PIPELINE   ")
        self.logger.info("="*60)
        topup_groups = find_reversed_phase_groups(dwi_files)
        self.logger.info(
            f"Found {len(dwi_files)} DWI files and {len(topup_groups)} topup group(s) "
            f"for sub-{subject} {ses}."
        )

        # 3a. COPY RAW DATA TO WORK DIRECTORY
        raw_work_dir = subj_work_dir / "rawdata"
        raw_work_dir.mkdir(parents=True, exist_ok=True)
        
        import shutil
        copied_dwi_files = []
        for d in dwi_files:
            # Construct destination path
            dest_img = raw_work_dir / d.img.name
            if not dest_img.exists():
                shutil.copy(d.img, dest_img)
            
            # Copy sidecars
            dest_bval = raw_work_dir / d.bval.name if d.bval else None
            if d.bval and not dest_bval.exists():
                shutil.copy(d.bval, dest_bval)
                
            dest_bvec = raw_work_dir / d.bvec.name if d.bvec else None
            if d.bvec and not dest_bvec.exists():
                shutil.copy(d.bvec, dest_bvec)
                
            dest_json = raw_work_dir / d.json.name if d.json else None
            if d.json and not dest_json.exists():
                shutil.copy(d.json, dest_json)
            
            # Create new DWIFile pointing to work dir
            new_dwi = DWIFile(
                img=dest_img,
                json=dest_json,
                bval=dest_bval,
                bvec=dest_bvec,
                entities=d.entities
            )
            copied_dwi_files.append(new_dwi)
            
        dwi_files = copied_dwi_files
        
        # 3b. Re-Group by reversed phase encoding (using local copies)
        # We need to re-run this because file paths changed
        topup_groups = find_reversed_phase_groups(dwi_files)

        # 4. Build preprocessing context
        context = {
            "subject": subject,
            "session": session,
            "current_image": dwi_files[0],
            "dwi_files": dwi_files,
            "topup_groups": topup_groups,
            "t1w_files": t1w_files,
        }

        # 5. Build the preprocessing workflow/pipeline
        self.preprocessing.build_pipeline(context)

        # 6. Run preprocessing workflow (writing to WORK DIR)
        preprocessed_context = self.preprocessing.run(subj_work_dir, context, reporter=reporter)
        
        # Generate Report
        reporter.generate()
        try:
            reporter.generate_pdf()
        except Exception:
            pass

        # 7. Extract and Copy Final Outputs to Output Dir
        preprocessed_dwis = preprocessed_context.get("preprocessed_dwis", dwi_files)
        preprocessed_masks = preprocessed_context.get("preprocessed_masks", [])
        
        # Ensure masks list matches dwis if empty
        if not preprocessed_masks:
            preprocessed_masks = [None] * len(preprocessed_dwis)
        
        from qmri_neuropipe.io.bids import build_bids_name

        for d, mask in zip(preprocessed_dwis, preprocessed_masks):
            # Force desc='preproc'
            final_entities = dict(d.entities)
            final_entities['desc'] = 'preproc'
            final_name = build_bids_name(final_entities)
            
            if not final_name.endswith(".nii.gz") and not final_name.endswith(".nii"):
                 final_name += ".nii.gz"
                 
            dest_path = output_dir / final_name
            
            if not dest_path.exists() or not self.config.skip_existing:
                 self.logger.info(f"Saving Final Preprocessed DWI: {d.img.name} -> {dest_path}")
                 shutil.copy(d.img, dest_path)
                 
                 # Copy sidecars (bvec, bval, json) with matching basename
                 final_base = dest_path.with_suffix("").with_suffix("") # remove .nii.gz if present
                 
                 if d.bvec and d.bvec.exists():
                     shutil.copy(d.bvec, final_base.with_suffix(".bvec"))
                 
                 if d.bval and d.bval.exists():
                     shutil.copy(d.bval, final_base.with_suffix(".bval"))
                     
                 if d.json and d.json.exists():
                     shutil.copy(d.json, final_base.with_suffix(".json"))
            
            # 2. Save Mask (as sidecar to dwi PREPROC)
            if mask:
                 mask_name = build_bids_name(final_entities, suffix='mask')
                 if not mask_name.endswith(".nii.gz") and not mask_name.endswith(".nii"):
                      mask_name += ".nii.gz"
                 
                 mask_dest = output_dir / mask_name
                 if not mask_dest.exists() or not self.config.skip_existing:
                      self.logger.info(f"Saving Final Brain Mask: {mask.img.name} -> {mask_dest}")
                      shutil.copy(mask.img, mask_dest)


        self.logger.info(
            f"Preprocessing complete for sub-{subject} {ses}. Results in {output_dir}"
        )

# Run the pipeline
if __name__ == '__main__':
    # Create configuration
    config = PipelineConfig(
        bids_dir='/data/my_study',
        output_dir='/data/derivatives/simple-pipeline',
        n_cpus=8,
        skip_existing=True
    )
    
    # Create and run pipeline
    def run(self, subjects: list[str] = None, sessions: list[str] = None):
         """Run pipeline with progress bars."""
         if subjects is None:
             subjects = self._get_all_subjects()
         
         # Use tqdm for subjects list
         if not subjects:
             self.logger.warning("No subjects found.")
             return

         pbar = self.get_progress_bar(subjects, desc="Processing Subjects")
         
         # Overriding run is cleaner but BasePipeline.run logic is complex (skipping, error handling).
         # Ideally BasePipeline.run should support progress bars.
         # Since we can't easily modify BasePipeline.run to use pbar without overriding, 
         # we will rely on BasePipeline.run calling self.process_subject, and simple logging.
         # But wait, BasePipeline.run iterates 'loader.load_multiple_subjects'.
         # We can't inject tqdm there easily unless we override run entirely.
         
         # Let's call super().run() but if we want per-subject progress bar, we have to do it there.
         # For now, let's just let BasePipeline.run do its job, which I updated to use rich logging 
         # but NOT progress bars for the subject loop itself (I only added get_progress_bar helper).
         # To add a progress bar for the subject loop, I would need to copy-paste BasePipeline.run here or modify BasePipeline.
         
         super().run(subjects=subjects, sessions=sessions)

# Run the pipeline
if __name__ == '__main__':
    # Create configuration
    config = PipelineConfig(
        bids_dir='/data/my_study',
        output_dir='/data/derivatives/simple-pipeline',
        n_cpus=8,
        skip_existing=True,
        log_level='INFO'
    )
    
    # Create and run pipeline
    pipeline = DMRIPipeline(config)
    pipeline.run()  # Processes all subjects automatically!
