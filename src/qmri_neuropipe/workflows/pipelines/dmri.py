from pathlib import Path
from typing import Optional
from qmri_neuropipe.core import (
    BasePipeline, BaseWorkflow, PipelineConfig,
)
from qmri_neuropipe.lib.dmri.normalization import NormalizationStep
from qmri_neuropipe.lib.common.segmentation import SegmentationStep
from qmri_neuropipe.core.types import ImageFile, DWIFile
from qmri_neuropipe.io.bids import _load_json_field, build_bids_name
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
from qmri_neuropipe.lib.dmri.reorient import DMRIReorientStep

from qmri_neuropipe.lib.common.mask import BrainMaskingStep
from .anat import AnatPreprocessingWorkflow
from qmri_neuropipe.lib.common.resample import ResampleStep
from qmri_neuropipe.interfaces.mriqc import run_mriqc
from qmri_neuropipe.lib.reporting.report import ReportGenerator
from qmri_neuropipe.lib.dmri.outliers import OutlierRemovalStep
from qmri_neuropipe.lib.dmri.qc import EddyQuadStep
from ...lib.dmri.fitting import DTIFittingStep, DKIFittingStep, NODDIFittingStep, SANDIFittingStep, MAPMRIFittingStep, CSDFittingStep
import time
# Rich and Viz imports moved to local scope




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
        self.logger.info(f"Auditing inputs: {len(dwi_files)} DWI files, {len(topup_groups)} topup groups.")

        dmri_cfg = self.config.get('dmri', {}).get('preprocessing', {})
        
        # -1. Reorientation (Standard Axes + Bvec Rotation)
        # New config key: dmri.preprocessing.reorient.enabled
        reor_cfg = dmri_cfg.get('reorient', {})
        if reor_cfg.get('enabled', False):
             self.logger.info("Adding DMRIReorientStep (reorient to standard RAS + bvec rotation)...")
             self.add_step(DMRIReorientStep(self.config, self.logger, self.provenance))

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
            reporter.set_dmri_input_summary(f"DWI Files: {len(dwi_files)}")
            dwi = dwi_files[0]
            # Plot b0 or first volume
            p = output_dir / "report_input_dwi_b0.png"
            try:
                 from qmri_neuropipe.lib.reporting.viz import create_ortho_view
                 create_ortho_view(dwi.img, p, title="Input DWI (first volume)")
                 reporter.add_dmri_input_figure(p, caption=dwi.img.name)
            except Exception as e:
                 self.logger.warning(f"Failed to plot input DWI: {e}")

        # Separate Global vs Per-Image steps
        global_steps = []
        per_image_steps = []
        
        for step in self.steps:
            if isinstance(step, (Synb0EstimationStep, TopupStep, GradientCheckStep, DMRIReorientStep)):
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

                 if reporter:
                      figures_dir = output_dir / "figures"
                      figures_dir.mkdir(exist_ok=True, parents=True) # Ensure exists
                      # Report global step (dwi=None). Pass context as current_arg.
                      self._report_step(reporter, step, None, None, context, None, figures_dir)
                 
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
                            # Flatten options: top-level + nested 'options' key
                            flat_opts = dict(coreg_cfg)
                            if "options" in flat_opts:
                                sub_opts = flat_opts.pop("options")
                                if isinstance(sub_opts, dict):
                                    flat_opts.update(sub_opts)
                            step_kwargs["options"] = flat_opts
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
                        
                        # Check if skipped but reporting needed (if output exists)
                        # Reporting & plotting logic via _report_step
                        if reporter: self._report_step(reporter, step, dwi, prev_img_obj, current_arg, target_img, figures_dir, step_kwargs)            
                        
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
                    # Add Final Preprocessed Output Paths
                    final_dwi = current_arg.get("current_image") if isinstance(current_arg, dict) else current_arg
                    dmri_outputs = []
                    
                    if final_dwi and hasattr(final_dwi, 'img') and final_dwi.img.exists():
                         dmri_outputs.append({"key": "Output DWI", "path": str(final_dwi.img)})
                         if final_dwi.bval and final_dwi.bval.exists():
                              dmri_outputs.append({"key": "Output Bval", "path": str(final_dwi.bval)})
                         if final_dwi.bvec and final_dwi.bvec.exists():
                              dmri_outputs.append({"key": "Output Bvec", "path": str(final_dwi.bvec)})
                         
                         reporter.set_dmri_outputs(dmri_outputs)
                    
                    # Generic summary table
                    reporter.add_dmri_summary("Execution Summary", step_metrics)
                    # Detailed metric summary
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
        try:
            from rich.console import Console
            from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
            console = Console()
        except ImportError:
            console = None
            Progress = None
            
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

    def _report_step(self, reporter, step, dwi, prev_img, current_arg, target_img, figures_dir, step_kwargs=None):
         # Extract reporting logic
         curr_img_obj = current_arg.get("current_image") if isinstance(current_arg, dict) else current_arg
         try:
             from qmri_neuropipe.lib.reporting.viz import create_ortho_view, plot_comparison
             
             figures_list = []
             details = {}
             step_name = step.__class__.__name__

             # Helper to check if processing occurred
             # If curr_img_obj is None but step skipped, might need to find output file? 
             # Actually `current_arg` usually holds the output even if restored or passed through.
             # If step was skipped because output exists, the pipeline runner usually sets `current_arg` 
             # to the existing output. So `curr_img_obj` should be valid if step succeeded (run or skip).
             
             if not dwi and isinstance(dwi, (str, Path)): 
                  # Handle case where dwi matches file path if refactoring
                  pass
                  
             stem = dwi.img.stem if dwi else "global"
             output_dir = figures_dir.parent # Derived for use in step-specific logic
              
             if not curr_img_obj and not isinstance(step, (TopupStep, GradientCheckStep, EddyQuadStep, DMRIReorientStep)):
                 self.logger.debug(f"No image object for reporting step {step_name} (might be None or failed).")
                 return

             if isinstance(step, DMRIReorientStep):
                 details = {"Method": step.method, "Status": "Completed (Global)"}
                 # Optional: Add figure for first image if accessible?
                 # Since it modifies dwi_files in context, we could grab one?
                 # For now, just details.
                 
             if isinstance(step, DenoisingStep):
                 details = {"Method": step.method}
                 if step.method in ['mppca', 'nlmeans']:
                     details["Patch Radius"] = str(step.patch_radius)
                     if step.method == 'mppca':
                         details["Block Radius"] = str(step.block_radius)
                         
                 fig_out = figures_dir / f"denoise_comp_{stem}.png"
                 if prev_img and curr_img_obj:
                    create_ortho_view(curr_img_obj.img, fig_out, title="Denoised DWI (b0)")
                    figures_list.append({"path": str(fig_out), "title": "Denoising", "caption": "Denoised Image"})
                    
             elif isinstance(step, GibbsUnringingStep):
                 details = {"Method": step.method}
                 fig_out = figures_dir / f"gibbs_comp_{stem}.png"
                 if curr_img_obj:
                     create_ortho_view(curr_img_obj.img, fig_out, title="Gibbs Corrected")
                     figures_list.append({"path": str(fig_out), "title": "Gibbs Unringing", "caption": "Gibbs Corrected Image"})
                     
             elif isinstance(step, EddyCorrectionStep):
                  details = {"Method": step.method}
                  # Eddy often produces parameter files. Can we report them?
                  # For now, just confirming it ran.
                  # Optionally plot first volume / b0?
                  # fig_out = figures_dir / f"eddy_res_{dwi.img.stem}.png"
                  # if curr_img_obj:
                  #    create_ortho_view(curr_img_obj.img, fig_out, title="Eddy Corrected")
                  #    figures_list.append({"path": str(fig_out), "title": "Eddy Correction", "caption": "Corrected Image"})
                  pass
                  
             elif isinstance(step, EddyQuadStep):
                  # Eddy QC produces a PDF or directory of QC. 
                  # We should link to it or extract motion metrics.
                  import json
                  qc_dir = output_dir / "eddy_quad" # Default output dir structure?
                  # Check output dir of step.
                  # Ideally we parse qc.json if available.
                  # For now, just adding a note.
                  details = {"Status": "QC Generated"}
                  
             elif isinstance(step, TopupStep):
                  details = {"Method": "Topup (FSL)"}
                  # Fieldmap result?
                  
             elif isinstance(step, BiasCorrectionStep):
                  details = {"Method": step.method}
                  fig_out = figures_dir / f"bias_corrected_{stem}.png"
                  if curr_img_obj:
                      create_ortho_view(curr_img_obj.img, fig_out, title="Bias Corrected")
                      figures_list.append({"path": str(fig_out), "title": "Bias Correction", "caption": "Bias Corrected Image"})

             elif isinstance(step, GradientCheckStep):
                  details = {"Status": "Verified"}
                  # Maybe validation details? 
                  
             elif isinstance(step, CoregistrationStep):
                  details = {"Method": step.method}
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
                      plot_comparison(target_img, curr_img_obj.img, fig_out, title="Coregistration Check (T1w vs DWI)")
                      figures_list.append({"path": str(fig_out), "title": "Coregistration Quality", "caption": "Overlay of aligned DWI (red) on T1w (gray)"})
                     
             elif isinstance(step, OutlierRemovalStep):
                  details = {"Method": step.method}
                  details["Threshold"] = str(getattr(step, 'threshold', 'N/A'))
                  
             elif isinstance(step, ResampleStep):
                  details = {"Resolution": str(getattr(step, 'resolution', 'Target'))}

             # Generic catch-all for other steps
             if not details and not figures_list:
                 details = {"Status": "Completed"}

             # Add the step to report
             reporter.add_dmri_step(step_name, details, figures=figures_list)
                     
         except ImportError:
             self.logger.warning("Could not import qmri_neuropipe.lib.reporting.viz. Skipping plotting for step report.")


    def _report_metrics_summary(self, reporter, dwi, current_arg):
         # Extract summary reporting logic
         outlier_stats = current_arg.get("outlier_stats")
         tables = []
         if outlier_stats:
             main_rows = [
                 {"Metric": "Total Volumes", "Value": str(outlier_stats["total_volumes"])},
                 {"Metric": "Removed Volumes", "Value": str(outlier_stats["removed_volumes"])},
                 {"Metric": "Percent Removed", "Value": f"{outlier_stats['percent_removed']:.2f}%"}
             ]
             tables.append({"title": f"Outlier Stats: {dwi.img.name}", "data": main_rows})
             
             if outlier_stats.get("bvalue_stats"):
                 breakdown_rows = []
                 for b_stat in outlier_stats["bvalue_stats"]:
                     breakdown_rows.append({
                         "B-value": str(b_stat["b_value"]),
                         "Total": str(b_stat["total"]),
                         "Removed": str(b_stat["removed"]),
                         "% Removed": f"{b_stat['percent']:.2f}%"
                     })
                 tables.append({"title": "Outlier Breakdown", "data": breakdown_rows})

         qc_metrics = current_arg.get("qc_metrics")
         if qc_metrics:
             if "motion" in qc_metrics:
                 mo_rows = [{"Metric": k, "Value": v} for k, v in qc_metrics["motion"].items()]
                 tables.append({"title": "QC: Motion", "data": mo_rows})
             if "cnr" in qc_metrics:
                 tables.append({"title": "QC: CNR", "data": qc_metrics["cnr"]})
             if "outliers_breakdown" in qc_metrics:
                 tables.append({"title": "QC: Outliers Breakdown", "data": qc_metrics["outliers_breakdown"]})
             if "outliers_summary" in qc_metrics:
                 out_rows = [{"Metric": k, "Value": v} for k, v in qc_metrics["outliers_summary"].items()]
                 tables.append({"title": "QC: Outliers Summary", "data": out_rows})
             elif "motion" not in qc_metrics: # Legacy
                 qc_rows = [{"Metric": k, "Value": v} for k, v in qc_metrics.items()]
                 tables.append({"title": "QC Summary (Eddy Quad)", "data": qc_rows})
         
         if tables:
              reporter.add_dmri_step("Quality Control", {}, tables=tables)
    

class ModelingWorkflow(BaseWorkflow):
    """
    Workflow for diffusion model fitting.
    Executes configured model fitting steps on preprocessed data.
    """

    def _initialize_steps(self):
        self.steps = []
        
    def build_pipeline(self, context: dict):
        modeling_cfg = self.config.get('dmri', {}).get('modeling', {})
        
        # 1. DTI
        dti_cfg = modeling_cfg.get('dti', {}) or modeling_cfg.get('tensor', {}) # Support 'tensor' for backward compatibility/legacy
        if dti_cfg.get('enabled', False):
            method = dti_cfg.get('method', 'dipy')
            self.logger.info(f"Adding DTIFittingStep (method={method})")
            self.add_step(DTIFittingStep(
                config=self.config,
                logger=self.logger,
                provenance=self.provenance,
                method=method,
                n_cpus=self.config.n_cpus,
                **dti_cfg.get('parameters', {}) or dti_cfg.get('options', {}) # support parameters or options
            ))
            
        # 2. DKI
        dki_cfg = modeling_cfg.get('dki', {})
        if dki_cfg.get('enabled', False):
            method = dki_cfg.get('method', 'dipy')
            self.logger.info(f"Adding DKIFittingStep (method={method})")
            self.add_step(DKIFittingStep(
                config=self.config,
                logger=self.logger,
                provenance=self.provenance,
                method=method,
                n_cpus=self.config.n_cpus,
                **dki_cfg.get('parameters', {})
            ))

        # 2.5 Constrained Spherical Deconvolution (CSD)
        csd_cfg = modeling_cfg.get('csd', {})
        if csd_cfg.get('enabled', False):
            method = csd_cfg.get('method', 'msmt_csd')
            self.logger.info(f"Adding CSDFittingStep (method={method})")
            
            # Combine options from top-level and parameters/options dict if present
            # ensuring parameters/options take precedence if both exist
            csd_kwargs = dict(csd_cfg)
            if 'parameters' in csd_cfg: csd_kwargs.update(csd_cfg['parameters'])
            if 'options' in csd_cfg: csd_kwargs.update(csd_cfg['options'])
            
            # Remove keys handled explicitly to avoid duplicates if passed as kwargs
            csd_kwargs.pop('enabled', None)
            csd_kwargs.pop('method', None)
            
            self.add_step(CSDFittingStep(
                config=self.config,
                logger=self.logger,
                provenance=self.provenance,
                method=method,
                n_cpus=self.config.n_cpus,
                **csd_kwargs
            ))

        # 3. NODDI
        noddi_cfg = modeling_cfg.get('noddi', {})
        if noddi_cfg.get('enabled', False):
            method = noddi_cfg.get('method', 'dmipy')
            self.logger.info(f"Adding NODDIFittingStep (method={method})")
            self.add_step(NODDIFittingStep(
                config=self.config,
                logger=self.logger,
                provenance=self.provenance,
                method=method,
                n_cpus=self.config.n_cpus,
                **noddi_cfg.get('parameters', {})
            ))
            
        # 4. SANDI
        sandi_cfg = modeling_cfg.get('sandi', {})
        if sandi_cfg.get('enabled', False):
            method = sandi_cfg.get('method', 'amico')
            self.logger.info(f"Adding SANDIFittingStep (method={method})")
            self.add_step(SANDIFittingStep(
                config=self.config,
                logger=self.logger,
                provenance=self.provenance,
                method=method,
                n_cpus=self.config.n_cpus,
                **sandi_cfg.get('parameters', {})
            ))
            
        # 5. MAPMRI
        map_cfg = modeling_cfg.get('mapmri', {})
        if map_cfg.get('enabled', False):
            method = map_cfg.get('method', 'dipy')
            self.logger.info(f"Adding MAPMRIFittingStep (method={method})")
            self.add_step(MAPMRIFittingStep(
                config=self.config,
                logger=self.logger,
                provenance=self.provenance,
            method=method,
                n_cpus=self.config.n_cpus,
                **map_cfg.get('parameters', {})
            ))
            

            
    
    def _report_modeling_step(self, reporter, step, dwi, output_dir, report_output_dir=None):
        """
        Report results for a modeling step.
        
        Parameters
        ----------
        output_dir : Path
            Directory where figures should be physically created (e.g. staging dir).
        report_output_dir : Path, optional
            Directory to use for the path in the report (e.g. final output dir).
            If None, uses output_dir.
        """
        from qmri_neuropipe.lib.reporting.viz import create_ortho_view

        step_name = step.__class__.__name__
        figures_dir = output_dir / "figures"
        figures_dir.mkdir(exist_ok=True, parents=True) # Create in physical location
        
        report_base = report_output_dir if report_output_dir else output_dir
        
        stem = dwi.img.stem # Fix for multi_replace replacement
        
        ents = dwi.entities.copy()
        for k in ['desc', 'suffix']:
             if k in ents: del ents[k]
             
        metric_map = {}
        base_model_dir = output_dir # Will adjust per model
        
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
             ents['model'] = 'MAPMRI'
             base_model_dir = output_dir / "mapmri"
             metric_map = {
                 "RTOP": {"suffix": "rtop", "title": "Return To Origin Probability (MAPMRI)"}
             }

        else:
             return
             
        # Generate items
        # Prepare aggregation
        metrics_to_plot = []
        found_any = False
        
        for key, info in metric_map.items():
            suffix = info['suffix']
            
            # Robust file finding
            fpath = None
            candidates = list(base_model_dir.glob(f"*{suffix}*"))
            
            sub_id = ents.get("sub")
            ses_id = ents.get("ses")
            
            valid_candidates = []
            for c in candidates:
                if not c.name.endswith(".nii.gz") and not c.name.endswith(".nii"): continue
                stem_lower = c.name.lower().split(".")[0]
                if not stem_lower.endswith(f"_{key.lower()}"): continue

                if sub_id and f"sub-{sub_id}" not in c.name: continue
                if ses_id and f"ses-{ses_id}" not in c.name: continue
                
                valid_candidates.append(c)
                
            if valid_candidates:
                best_cand = valid_candidates[0]
                model_label = ents.get("model", "")
                if model_label:
                    for vc in valid_candidates:
                        if f"model-{model_label}" in vc.name:
                            best_cand = vc
                            break
                fpath = best_cand
            else:
                 fname = build_bids_name(ents, suffix=suffix)
                 fpath_strict = base_model_dir / fname
                 if fpath_strict.exists():
                     fpath = fpath_strict
            
            if fpath and fpath.exists():
                found_any = True
                metrics_to_plot.append({
                    "path": fpath,
                    "title": key, # Use short key for grid row label
                    "desc": info['title']
                })
            else:
                self.logger.debug(f"Report: Metric file for {key} not found (Suffix: {suffix}). Base dir: {base_model_dir}")

        if found_any:
            from qmri_neuropipe.lib.reporting.viz import create_metric_grid
            
            grid_name = f"{step_name}_grid_{stem}.png"
            grid_out = figures_dir / grid_name # Physical location (staging)
            
            # Logical report location
            grid_report = grid_out
            if report_output_dir:
                 grid_report = report_base / "figures" / grid_name
                 
            create_metric_grid(metrics_to_plot, grid_out, title=f"{ents.get('model', 'dMRI')} Metrics")
            
            details = {"Model": ents.get('model', 'Unknown'), "Metrics Found": ", ".join([m['title'] for m in metrics_to_plot])}
            
            figures = [{
                "path": str(grid_report), 
                "title": f"{ents.get('model')} Metrics Grid", 
                "caption": "Rows: Metrics, Cols: Sagittal / Coronal / Axial"
            }]
            
            reporter.add_dmri_step(f"Modeling: {ents.get('model', step_name)}", details, figures=figures)
        else:
            self.logger.warning(f"No metrics found for modeling step {step_name} in {base_model_dir}")
    
    def run(self, work_dir: Path, context: dict, reporter=None, final_output_dir: Optional[Path] = None) -> dict:
        """
        Run the modeling workflow.
        
        Parameters
        ----------
        work_dir : Path
            Working directory for intermediate files.
        context : dict
            Context containing 'preprocessed_dwis', 'preprocessed_masks', etc.
        reporter : ReportGenerator, optional
            Reporter instance.
        final_output_dir : Path, optional
            If provided, model outputs will be written directly to this directory 
            (within subfolders like 'dti', 'dki') instead of the work_dir.
        """
        import rich
        from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
        
        self.logger.info("Starting Modeling Workflow...")
        
        # Determine effective output directory
        # If final_output_dir is provided, we write there. 
        # Otherwise we write to work_dir/modeling
        # Determine staging and final directories
        staging_dir = work_dir / "modeling"
        staging_dir.mkdir(parents=True, exist_ok=True)
        
        # If final_output_dir is provided, we use it for checking existence and final destination
        final_dest = final_output_dir if final_output_dir else staging_dir
        if final_output_dir:
            final_output_dir.mkdir(parents=True, exist_ok=True)

        preprocessed_dwis = context.get('preprocessed_dwis', [])
        preprocessed_masks = context.get('preprocessed_masks', [])
        
        # Pad masks if needed
        if not preprocessed_masks:
            preprocessed_masks = [None] * len(preprocessed_dwis)
        elif len(preprocessed_masks) < len(preprocessed_dwis):
             preprocessed_masks.extend([None] * (len(preprocessed_dwis) - len(preprocessed_masks)))
             
        # Setup Progress Bar
        total_steps = len(preprocessed_dwis) * len(self.steps)
        
        import shutil
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=rich.get_console() 
        ) as progress:
            
            task_id = progress.add_task("Fitting models...", total=total_steps)
            
            for i, (dwi, mask) in enumerate(zip(preprocessed_dwis, preprocessed_masks)):
                img_name = dwi.img.name
                
                # Update context for current image
                context['current_image'] = dwi
                
                for step in self.steps:
                    step_name = step.__class__.__name__
                    
                    # Check skip status using FINAL destination
                    skipping = hasattr(step, 'should_skip') and step.should_skip(context, final_dest)
                    
                    if skipping:
                        progress.update(task_id, description=f"Skipping {step_name} (Exists)")
                        progress.advance(task_id)
                        # Report skipped step
                        if reporter:
                             try:
                                 # Use staging_dir for reporting to avoid creating figures in final_dest
                                 self._report_modeling_step(reporter, step, dwi, output_dir=staging_dir)
                             except Exception as e:
                                 self.logger.warning(f"Reporting failed for skipped step {step_name}: {e}")
                        continue
                        
                    progress.update(task_id, description=f"Processing {img_name} - {step_name}")
                    
                    try:
                        # Run step writing to STAGING directory
                        step.run(context, output_dir=staging_dir, mask=mask)
                        
                        # Copy results to final destination immediately (excluding figures)
                        if final_output_dir:
                            shutil.copytree(staging_dir, final_output_dir, dirs_exist_ok=True, ignore=shutil.ignore_patterns("figures"))
                            
                        progress.advance(task_id)
                        
                    except Exception as e:
                        if self.config.stop_on_error: raise e
                        self.logger.error(f"Modeling step {step_name} failed: {e}")

                    # Reporting (Using final destination where files should now exist)
                    if reporter:
                         try:
                             # Report using staging dir (where figures are)
                             self._report_modeling_step(reporter, step, dwi, output_dir=staging_dir)
                         except Exception as e:
                             self.logger.warning(f"Reporting failed for step {step_name}: {e}")


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
                 task = progress.add_task(f"[cyan]Starting Modeling...", total=total_steps)
                 return _execute_modeling(progress_ctx=progress, task_id=task)
        else:
             return _execute_modeling()
    


# Define Normalization Workflow
class NormalizationWorkflow(BaseWorkflow):
    """
    Workflow for spatial normalization.
    """
    
    def _initialize_steps(self):
        self.steps = []
        
    def build_pipeline(self, context: dict):
        # Config is now at dmri.normalization level, not dmri.modeling.normalization
        norm_cfg = self.config.get('dmri', {}).get('normalization', {})
        
        if norm_cfg.get('enabled', False):
            self.logger.info("Adding NormalizationStep (separate workflow)")
            norm_kwargs = dict(norm_cfg)
            if 'parameters' in norm_cfg: norm_kwargs.update(norm_cfg['parameters'])
            if 'options' in norm_cfg: norm_kwargs.update(norm_cfg['options'])
            
            norm_kwargs.pop('enabled', None)
            
            self.add_step(NormalizationStep(
                config=self.config,
                logger=self.logger,
                provenance=self.provenance,
                **norm_kwargs 
            ))

    def _report_normalization_step(self, reporter, step, dwi, output_dir, report_output_dir=None):
         # Logic from ModelingWorkflow._report_modeling_step adapted for Normalization
         if not reporter: return 
         
         step_name = step.__class__.__name__
         figures_dir = output_dir / "figures"
         figures_dir.mkdir(exist_ok=True, parents=True)
         
         report_base = report_output_dir if report_output_dir else output_dir
         stem = dwi.img.stem
         
         ents = dwi.entities.copy()
         ents['model'] = 'Normalization'
         
         # Normalization outputs are in output_dir/space-{SpaceName}
         # NormalizationWorkflow usually runs with output_dir = work_dir/normalization
         # Base directory for metrics is output_dir/space-{SpaceName}
         base_model_dir = output_dir / f"space-{step.space_name}"
         
         metric_map = {
             "FA": {"suffix": "FA", "title": f"Fractional Anisotropy (Normalized {step.space_name})"},
             "MD": {"suffix": "MD", "title": f"Mean Diffusivity (Normalized {step.space_name})"},
             "ODI": {"suffix": "ODI", "title": f"NODDI ODI (Normalized {step.space_name})"}
         }
         
         metrics_to_plot = []
         found_any = False
         
         for key, info in metric_map.items():
            suffix = info['suffix']
            candidates = list(base_model_dir.glob(f"*{suffix}*"))
            valid_candidates = []
            for c in candidates:
                if not c.name.endswith(".nii.gz") and not c.name.endswith(".nii"): continue
                stem_lower = c.name.lower()
                # Check suffix match (e.g. _FA.nii.gz)
                if not (stem_lower.endswith(f"_{key.lower()}.nii.gz") or stem_lower.endswith(f"_{key.lower()}.nii")): continue
                valid_candidates.append(c)
                
            if valid_candidates:
                metrics_to_plot.append({
                    "path": valid_candidates[0], # Take first match
                    "title": key,
                    "desc": info['title']
                })
                found_any = True
                
         if found_any:
            from qmri_neuropipe.lib.reporting.viz import create_metric_grid
            grid_name = f"normalization_grid_{stem}.png"
            grid_out = figures_dir / grid_name
            
            grid_report = report_base / "figures" / grid_name
            
            create_metric_grid(metrics_to_plot, grid_out, title=f"Normalized Metrics ({step.space_name})")
            
            details = {"Space": step.space_name, "Metrics": ", ".join([m['title'] for m in metrics_to_plot])}
            figures = [{
                "path": str(grid_report), 
                "title": f"Normalization ({step.space_name})", 
                "caption": f"Normalized metrics in {step.space_name} space"
            }]
            reporter.add_dmri_step("Normalization", details, figures=figures)

    def run(self, work_dir: Path, context: dict, reporter=None, final_output_dir: Optional[Path] = None) -> dict:
        self.logger.info("Starting Normalization Workflow...")
        
        # Staging: work_dir/normalization
        staging_dir = work_dir / "normalization"
        staging_dir.mkdir(parents=True, exist_ok=True)
        
        final_dest = final_output_dir if final_output_dir else staging_dir
        if final_output_dir: final_output_dir.mkdir(parents=True, exist_ok=True)

        dwis = context.get('preprocessed_dwis', [])
        
        import shutil
        total_steps = len(dwis) * len(self.steps)
        
        # Simple execution loop
        for dwi in dwis:
            context['current_image'] = dwi
            for step in self.steps:
                # NormalizationStep needs modeling_results in context
                # Those should be there from ModelingWorkflow
                
                try:
                    step.run(context, output_dir=staging_dir)
                    
                    # Copy to final (excluding figures for now or not?)
                    # NormalizationStep.run already copies to internal output_dir structure
                    # But we want to copy the whole staging structure to final_output_dir/normalization if needed.
                    # Actually NormalizationStep puts things in output_dir/space-MNI...
                    # If final_output_dir is passed, step.run might use it?
                    # The interface says step.run(context, output_dir=staging_dir).
                    # Then we copy staging_dir contents to final_output_dir.
                    
                    if final_output_dir:
                         shutil.copytree(staging_dir, final_output_dir, dirs_exist_ok=True, ignore=shutil.ignore_patterns("figures"))
                    
                    if reporter:
                        self._report_normalization_step(reporter, step, dwi, staging_dir)
                        
                except Exception as e:
                    self.logger.error(f"Normalization failed: {e}")
                    if self.config.stop_on_error: raise e
                    
        return context


# Define Segmentation Workflow
class SegmentationWorkflow(BaseWorkflow):
    """
    Workflow for ROI segmentation and statistics.
    """
    
    def _initialize_steps(self):
        self.steps = []
        
    def build_pipeline(self, context: dict):
        seg_cfg = self.config.get('dmri', {}).get('segmentation', {})
        
        if seg_cfg.get('enabled', False):
            self.logger.info("Adding SegmentationStep")
            seg_kwargs = dict(seg_cfg)
            seg_kwargs.pop('enabled', None)
            
            self.add_step(SegmentationStep(
                config=self.config,
                logger=self.logger,
                provenance=self.provenance,
                **seg_kwargs 
            ))

    def run(self, work_dir: Path, context: dict, reporter=None, final_output_dir: Optional[Path] = None) -> dict:
        self.logger.info("Starting Segmentation Workflow...")
        
        # Staging: work_dir/stats
        staging_dir = work_dir / "stats"
        staging_dir.mkdir(parents=True, exist_ok=True)
        
        final_dest = final_output_dir if final_output_dir else staging_dir
        if final_output_dir: final_output_dir.mkdir(parents=True, exist_ok=True)

        for step in self.steps:
            try:
                step.run(context, output_dir=staging_dir)
                
                # Copy results to final output dir
                import shutil
                if final_output_dir:
                     for f in staging_dir.glob("*.tsv"):
                         dest = final_output_dir / f.name
                         shutil.copy(f, dest)
                
                # Report
                if reporter and 'segmentation_stats' in context:
                    # Try to add a summary table
                    stats_files = context['segmentation_stats']
                    if stats_files:
                        import pandas as pd
                        try:
                            df = pd.read_csv(stats_files[-1], sep='\t')
                            # Summary: Mean across all ROIs for each metric?
                            # Or just list top ROIs?
                            # Let's show first 10 rows for verify
                            preview = df.head(10).to_dict(orient='records')
                            # Simplify keys for report table
                            # data = [{"ROI": r["roi_name"], "Metric": r["metric"], "Mean": f"{r['mean']:.4f}"} for r in preview]
                            # reporter.add_dmri_step("Segmentation Statistics", {}, tables=[{"title": "ROI Stats Preview", "data": data}])
                            
                            # Better: Aggregate summary
                            summary = df.groupby('metric')['mean'].mean().reset_index()
                            summary_rows = [{"Metric": r['metric'], "Global Mean": f"{r['mean']:.4f}"} for _, r in summary.iterrows()]
                             
                            reporter.add_dmri_step("Segmentation Statistics", {"Status": "Completed", "File": str(stats_files[-1])}, 
                                                   tables=[{"title": "Global Mean per Metric", "data": summary_rows}])
                        except Exception as e:
                            self.logger.warning(f"Failed to report stats: {e}")

            except Exception as e:
                self.logger.error(f"Segmentation failed: {e}")
                if self.config.stop_on_error: raise e
                    
        return context

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
        self.modeling = ModelingWorkflow(self.config, self.logger, self.provenance)
        self.normalization = NormalizationWorkflow(self.config, self.logger, self.provenance)
        self.segmentation = SegmentationWorkflow(self.config, self.logger, self.provenance) # New
        # Initialize Anat Workflow if configured (we assume it might be used)
        self.anat_preprocessing = AnatPreprocessingWorkflow(self.config, self.logger, self.provenance)
    
    def _should_skip(self, subject: str, session: Optional[str]) -> bool:
        """
        Override default skip logic.
        
        The default BasePipeline._should_skip checks if *any* output exists.
        For a multi-stage pipeline (Anat -> DWI -> Model), this is too aggressive,
        as presence of anatomical/preproc outputs would prevent modeling from running.
        
        We return False here to allow process_subject to run. The individual steps
        (especially modeling) implement their own granular checks for existing outputs.
        """
        return False

    def process_subject(self, subject: str, session: Optional[str]):
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
             except Exception as e:
                 self.logger.warning(f"PDF Generation failed: {e}")
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
        
        # Check if preprocessing is already done (outputs exist in FINAL output_dir)
        # We need to reconstruct the expected final filename for the first DWI
        # This is heuristics since we usually process multiple DWIs merged or single.
        # Assuming dwi_files[0] is representative.
        from qmri_neuropipe.io.bids import build_bids_name
        first_dwi = dwi_files[0]
        final_ents = dict(first_dwi.entities)
        final_ents['desc'] = 'preproc'
        expected_preproc_name = build_bids_name(final_ents, suffix='dwi')
        if not expected_preproc_name.endswith('.nii.gz'): expected_preproc_name += '.nii.gz'
        
        expected_preproc_path = output_dir / expected_preproc_name
        expected_mask_path = output_dir / expected_preproc_name.replace('_dwi.nii.gz', '_mask.nii.gz')

        if hasattr(self.config, 'debug') and self.config.debug:
            print(f"DEBUG: Checking preproc path: {expected_preproc_path}")
            print(f"DEBUG: Exists? {expected_preproc_path.exists()}")
            print(f"DEBUG: skip_existing? {self.config.skip_existing}")

        preprocessed_context = None
        
        if self.config.skip_existing and expected_preproc_path.exists():
            self.logger.info(f"Skipping preprocessing (Final output exists: {expected_preproc_path})")
            
            # Load existing results into context so modeling can use them
            # We need to construct DWIFile objects for them
            # DWIFile and ImageFile are imported globally
            
            # Load sidecars
            bval_path = expected_preproc_path.with_suffix("").with_suffix(".bval") # approx
            # Better check with_name replacement logic or check existence
            if not bval_path.exists(): bval_path = Path(str(expected_preproc_path).replace('.nii.gz', '.bval'))
            
            bvec_path = expected_preproc_path.with_suffix("").with_suffix(".bvec")
            if not bvec_path.exists(): bvec_path = Path(str(expected_preproc_path).replace('.nii.gz', '.bvec'))
            
            json_path = expected_preproc_path.with_suffix("").with_suffix(".json")
            if not json_path.exists(): json_path = Path(str(expected_preproc_path).replace('.nii.gz', '.json'))

            loaded_dwi = DWIFile(
                img=expected_preproc_path,
                bval=bval_path if bval_path.exists() else None,
                bvec=bvec_path if bvec_path.exists() else None,
                json=json_path if json_path.exists() else None,
                entities=final_ents
            )
            
            mask_ents = dict(final_ents)
            mask_ents['suffix'] = 'mask'
            loaded_mask = ImageFile(entities=mask_ents, img=expected_mask_path) if expected_mask_path.exists() else None
            
            # Reconstruct context
            preprocessed_context = dict(context)
            preprocessed_context['preprocessed_dwis'] = [loaded_dwi]
            preprocessed_context['preprocessed_masks'] = [loaded_mask]
            
            if reporter:
                # 0. Reporting: Inputs (Matches PreprocessingWorkflow.run logic)
                if dwi_files:
                    reporter.set_dmri_input_summary(f"DWI Files: {len(dwi_files)} (Pipeline Skipped - Output Exists)")
                    dwi_in = dwi_files[0]
                    # Use work dir for figures
                    p_in = subj_work_dir / "report_input_dwi_b0.png"
                    try:
                         from qmri_neuropipe.lib.reporting.viz import create_ortho_view
                         create_ortho_view(dwi_in.img, p_in, title="Input DWI (first volume)")
                         reporter.add_dmri_input_figure(p_in, caption=dwi_in.img.name)
                    except Exception:
                         if p_in.exists():
                              reporter.add_dmri_input_figure(p_in, caption=dwi_in.img.name)

                # 0.5 Execution Summary
                preproc_steps = self.preprocessing.steps
                step_list = []
                for s in preproc_steps:
                   step_list.append({
                       "Step": s.__class__.__name__, 
                       "Status": "Skipped (Exists)", 
                       "Duration": "N/A"
                   })
                
                # Add Final Output Paths to Summary
                dmri_outputs = []
                dmri_outputs.append({"key": "Output DWI", "path": str(expected_preproc_path)})
                
                if bval_path.exists():
                    dmri_outputs.append({"key": "Output Bval", "path": str(bval_path)})
                if bvec_path.exists():
                    dmri_outputs.append({"key": "Output Bvec", "path": str(bvec_path)})
                
                if dmri_outputs:
                    reporter.set_dmri_outputs(dmri_outputs)
                
                if step_list:
                    reporter.add_dmri_summary("Execution Summary (Skipped Run)", step_list)

                # 1. Verification Step (Preprocessed Output)
                # We can present this as a "Verification" step or generic step
                p = subj_work_dir / "report_preproc_dwi_b0.png"
                fig_list = []
                try:
                     from qmri_neuropipe.lib.reporting.viz import create_ortho_view
                     create_ortho_view(expected_preproc_path, p, title="Preprocessed DWI (Existing)")
                     fig_list.append({"path": str(p), "title": "Preprocessed DWI", "caption": "Existing Output (Recovered)"})
                except Exception:
                     if p.exists():
                          fig_list.append({"path": str(p), "title": "Preprocessed DWI", "caption": "Existing Output (Recovered)"})
                     
                reporter.add_dmri_step("Verification (Existing Output)", {"Status": "Verified", "Path": str(expected_preproc_path)}, figures=fig_list)
                
                # Recover metrics (Outliers dependencies, QC)
                self._recover_missed_metrics(reporter, context, subj_work_dir, final_ents, expected_preproc_path)
                
                # Recover figures (from intermediate work dir)
                self._recover_missed_figures(reporter, context, subj_work_dir)

            
        else:
            preprocessed_context = self.preprocessing.run(subj_work_dir, context, reporter=reporter)
        
        # 6b. Run Modeling Workflow (Using Preprocessed Data)
        # We need to make sure the preprocessed data (which might be in work dir) 
        # is accessible. Modeling steps will write to FINAL output dir directly?
        # Or work dir?
        # Plan says: "Handle saving of model outputs ... to derivatives/.../dwi"
        # Since modeling doesn't typically feed into further steps that modify the image geometry,
        # we can write directly to output_dir or write to work_dir and copy.
        # Writing to work_dir is safer for idempotency/cleanness, then copy.
        # Let's write to work_dir first.
        
        # Initialize modeling steps based on config
        self.modeling.build_pipeline(preprocessed_context)
        
        if self.modeling.steps:
            self.logger.info("="*60)
            self.logger.info("  RUNNING MODEL FITTING        ")
            self.logger.info("="*60)
            # Pass output_dir / "models" to ModelingWorkflow so it copies modeling/DTI -> models/DTI
            preprocessed_context = self.modeling.run(subj_work_dir, preprocessed_context, reporter=reporter, final_output_dir=output_dir / "models")
        else:
            self.logger.info("No modeling steps configured.")

        # 6c. Run Normalization Workflow
        # Initialize and run using modeling results (in context)
        self.normalization.build_pipeline(preprocessed_context)
        
        if self.normalization.steps:
            self.logger.info("="*60)
            self.logger.info("  RUNNING NORMALIZATION        ")
            self.logger.info("="*60)
            # Use work_dir for staging, write final results to output_dir/normalization
            norm_output_dir = output_dir / "normalization"
            preprocessed_context = self.normalization.run(subj_work_dir, preprocessed_context, reporter=reporter, final_output_dir=norm_output_dir)

        else:
            self.logger.info("No normalization steps configured.")

        # 6d. Run Segmentation Workflow
        # Initialize
        self.segmentation.build_pipeline(preprocessed_context)
        
        if self.segmentation.steps:
            self.logger.info("="*60)
            self.logger.info("  RUNNING SEGMENTATION         ")
            self.logger.info("="*60)
            
            # Use work_dir for staging, write final results to output_dir/stats
            stats_output_dir = output_dir / "stats"
            preprocessed_context = self.segmentation.run(subj_work_dir, preprocessed_context, reporter=reporter, final_output_dir=stats_output_dir)


        
        # Report generation moved to end to include final outputs


        # 7. Extract and Copy Final Outputs to Output Dir
        preprocessed_dwis = preprocessed_context.get("preprocessed_dwis", dwi_files)
        preprocessed_masks = preprocessed_context.get("preprocessed_masks", [])
        
        # Ensure masks list matches dwis if empty
        if not preprocessed_masks:
            preprocessed_masks = [None] * len(preprocessed_dwis)
        
        from qmri_neuropipe.io.bids import build_bids_name
        
        # Collect final outputs for report
        dmri_outputs = []

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
                
                # Add to Outputs (mask)
                dmri_outputs.append({"key": "Brain Mask", "path": str(mask_dest)})
            
            # Add Preproc Output
            dmri_outputs.append({"key": "Preprocessed DWI", "path": str(dest_path)})
            if d.bval: dmri_outputs.append({"key": "Bval", "path": str(dest_path.with_suffix("").with_suffix(".bval"))})
            if d.bvec: dmri_outputs.append({"key": "Bvec", "path": str(dest_path.with_suffix("").with_suffix(".bvec"))})

            # 3. Copy Model Outputs (if any)
            # Scan work_dir/modeling/dti, work_dir/modeling/dki, etc.
            # Folder names in fitting.py: DTI, DKI, NODDI, sandi, mapmri, CSD
            # We must match these names (case-sensitive) or just scan all folders in modeling?
            # Safest is to iterate known folders.
            
            modeling_work = subj_work_dir / "modeling"
            norm_work = subj_work_dir / "normalization" # Normalization is sibling to modeling usually?
            # NormalizationStep puts outputs into output_dir.parent / "normalization".
            # If modeling run() passed staging_dir = work_dir/modeling, output_dir.parent is work_dir.
            # So outputs are in work_dir/normalization
            
            model_folders = ['DTI', 'DKI', 'NODDI', 'sandi', 'mapmri', 'CSD']
            
            # Copy Normalization Results first or logic?
            # Copy Normalization Results (handled by NormalizationWorkflow, but check if manual needed if workflow skipped?)
            # NormalizationWorkflow handles copying to final_output_dir.
            # If we want to be safe, we can leave this or remove it.
            # Removing redundant copy as Workflow handles it.

            
            if modeling_work.exists():
                for model_folder in model_folders:
                    src_model_dir = modeling_work / model_folder
                    if src_model_dir.exists():
                         # Ensure destination subfolder exists (in output_dir/models)
                         dest_dir = output_dir / "models" / model_folder
                         dest_dir.mkdir(exist_ok=True, parents=True)
                         
                         for f in src_model_dir.glob("*.nii.gz"):
                             dest = dest_dir / f.name
                             if not dest.exists() or not self.config.skip_existing:
                                 self.logger.info(f"Saving {model_folder} Map: {f.name}")
                                 shutil.copy(f, dest)
                                 
                                 # Report
                                 # Key: DTI FA, NODDI ODI, etc.
                                 # parse suffix
                                 name_part = f.name.replace('.nii.gz', '')
                                 suffix = name_part.split('_')[-1]
                                 if 'model-' in f.name:
                                      # Try to get nicer key
                                      pass
                                 dmri_outputs.append({"key": f"{model_folder} {suffix}", "path": str(dest)})

                                 # Copy sidecar
                                 json_src = f.with_suffix("").with_suffix(".json") # .nii.gz -> .json
                                 if not json_src.exists(): json_src = Path(str(f).replace('.nii.gz', '.json'))
                                 
                                 if json_src.exists():
                                     dest_json = dest_dir / json_src.name
                                     shutil.copy(json_src, dest_json)
                                     
            # 4. Normalization Outputs (Scanning)
            norm_dir = output_dir / "normalization"
            if norm_dir.exists():
                 # Recursively find files or just space-* folders?
                 # Should be in space-MNI/
                 for f in norm_dir.rglob("*.nii.gz"):
                      # Key: Normalized FA, etc.
                      # Parse entities
                      name = f.name
                      # Try to extract suffix and space
                      space = "Standard"
                      if 'space-' in name:
                          space = name.split('space-')[1].split('_')[0]
                      
                      suffix = name.replace('.nii.gz', '').split('_')[-1]
                      
                      key_name = f"Normalized {suffix} ({space})"
                      if 'desc-warp' in name: key_name = f"Warp Field ({space})"
                      
                      dmri_outputs.append({"key": key_name, "path": str(f)})
            
            # 5. Add to Reporter and Generate
            if reporter:
                 if dmri_outputs:
                      reporter.set_dmri_outputs(dmri_outputs)
                 
                 reporter.generate()
                 try:
                     reporter.generate_pdf()
                 except Exception as e:
                     self.logger.warning(f"PDF Generation failed: {e}")




        self.logger.info(
            f"Preprocessing complete for sub-{subject} {ses}. Results in {output_dir}"
        )

    def _recover_missed_metrics(self, reporter, context, output_dir, final_ents, final_dwi_path):
        """
        Attempt to recover and report metrics (outliers, QC) when preprocessing was skipped.
        """
        import numpy as np
        import json
        
        # 1. QA/QC (Eddy Quad)
        # Check for qc/eddy_quad/qc.json
        # Location logic mirrors EddyQuadStep
        qc_json = output_dir / "qc" / "eddy_quad" / "qc.json"
        if qc_json.exists():
            try:
                with open(qc_json, 'r') as f:
                     metrics = json.load(f)
                
                # Extract Summary (Simplified)
                # Motion
                if 'qc_mot_abs' in metrics:
                     mo_rows = [
                         {"Metric": "Absolute Motion (mm)", "Value": f"{metrics.get('qc_mot_abs',0):.2f}"},
                         {"Metric": "Relative Motion (mm)", "Value": f"{metrics.get('qc_mot_rel',0):.2f}"}
                     ]
                     reporter.add_dmri_step("QC: Motion Statistics (Recovered)", {}, tables=[{"title": "Motion Stats", "data": mo_rows}])
                
                # Outliers
                if 'qc_outliers_tot' in metrics:
                     out_rows = [{"Metric": "Total Outliers (%)", "Value": f"{metrics.get('qc_outliers_tot',0):.2f}"}]
                     reporter.add_dmri_step("QC: Outliers (Recovered)", {}, tables=[{"title": "Outliers Summary", "data": out_rows}])
                     
            except Exception as e:
                self.logger.warning(f"Failed to recover QC metrics from {qc_json}: {e}")

        # 2. Outlier Removal Stats (Re-calculate diff)
        # Needs input bval and output bval
        dwi_files = context.get('dwi_files', [])
        if not dwi_files: return
        
        input_bval_path = dwi_files[0].bval
        
        # Output bval (loaded in skip logic? we can find it)
        output_bval_path = final_dwi_path.with_suffix("").with_suffix(".bval")
        if not output_bval_path.exists(): output_bval_path = Path(str(final_dwi_path).replace('.nii.gz', '.bval'))
        
        if input_bval_path and input_bval_path.exists() and output_bval_path.exists():
            try:
                in_bvals = np.loadtxt(input_bval_path)
                out_bvals = np.loadtxt(output_bval_path)
                
                # Handle 1D/2D
                n_in = in_bvals.size
                n_out = out_bvals.size
                
                if n_in > n_out:
                    removed = n_in - n_out
                    pct = (removed / n_in) * 100
                    
                    rows = [
                        {"Metric": "Total Volumes (Input)", "Value": str(n_in)},
                        {"Metric": "Total Volumes (Output)", "Value": str(n_out)},
                        {"Metric": "Removed Volumes", "Value": str(removed)},
                        {"Metric": "Percent Removed", "Value": f"{pct:.2f}%"}
                    ]
                    reporter.add_dmri_step("Outlier Removal Summary (Re-calculated)", {}, tables=[{"title": "Outlier Counts", "data": rows}])
            except Exception as e:
                self.logger.warning(f"Failed to recalculate outlier stats: {e}")

    def _recover_missed_figures(self, reporter, context, work_dir):
        """
        Recover and report existing figures from previous runs.
        """
        dwi_files = context.get('preprocessed_dwis', [])
        figures_dir = work_dir / "figures"
        
        if not figures_dir.exists():
            return
            
        for dwi in dwi_files:
            stem = dwi.img.stem
            
            # 1. Denoising
            denoise_fig = figures_dir / f"denoise_comp_{stem}.png"
            if denoise_fig.exists():
                fig_item = [{"path": str(denoise_fig), "title": "Denoising", "caption": "Denoised Image (Orthogonal View) [Recovered]"}]
                reporter.add_dmri_step("Denoising", {"Status": "Recovered"}, figures=fig_item)
                
            # 2. Gibbs
            gibbs_fig = figures_dir / f"gibbs_comp_{stem}.png"
            if gibbs_fig.exists():
                fig_item = [{"path": str(gibbs_fig), "title": "Gibbs Unringing", "caption": "Gibbs Corrected Image [Recovered]"}]
                reporter.add_dmri_step("Gibbs Unringing", {"Status": "Recovered"}, figures=fig_item)
                
            # 3. Coregistration
            coreg_fig = figures_dir / f"coreg_check_{stem}.png"
            if coreg_fig.exists():
                fig_item = [{"path": str(coreg_fig), "title": "Coregistration Quality", "caption": "Overlay of aligned DWI on T1w [Recovered]"}]
                reporter.add_dmri_step("Coregistration", {"Status": "Recovered"}, figures=fig_item)



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
