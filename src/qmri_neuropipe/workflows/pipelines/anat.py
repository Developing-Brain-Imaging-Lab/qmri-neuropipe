"""
Anatomical Processing Pipeline (anat_proc).

1) Resample/resize
2) Reorient to standard
3) Denoise
4) Gibbs correction
5) Bias correction
6) Coregistration (T1w <-> T2w)
7) Brain Mask
8) FreeSurfer Recon-all (optional)
9) Nonlinear Registration (to template)
"""

from pathlib import Path
from typing import Optional, List
import logging

from ...core import BasePipeline, BaseWorkflow, PipelineConfig
from ...core.types import ImageFile
from ...io.anat.bids import bids_find_t1w, bids_find_t2w
from ...io.bids import build_bids_name
import shutil
import json

# Steps
from ...lib.common.resample import ResampleStep
from ...lib.common.tracking import TrackingStep
from ...lib.common.reorient import ReorientStep
from ...lib.common.denoise import DenoisingStep
from ...lib.common.gibbs import GibbsUnringingStep
from ...lib.common.bias import BiasCorrectionStep
from ...lib.common.registration import CoregistrationStep, NonlinearRegistrationStep
from ...lib.common.mask import BrainMaskingStep
from ...lib.anat.recon import ReconAllStep, FreeSurferStatsStep
from ...lib.common.sharpen import SharpeningStep
from ...lib.common.segmentation import SegmentationStep
from ...interfaces.mriqc import run_mriqc
from ...interfaces import ants # For manual apply_transforms
from ...lib.reporting.report import ReportGenerator
import time
# Rich and Viz imports moved to local scope

class AnatPreprocessingWorkflow(BaseWorkflow):
    """
    Workflow for preprocessing T1w (and optionally T2w) images.
    """
    def _initialize_steps(self):
        self.modality = "Anatomical"
        # Configure Steps
        anat_cfg = self.config.get("anat", {}).get("preprocessing", {})
        
        # 1. Resample
        if anat_cfg.get("resample", {}).get("enabled"):
             # If resolution is 0 or missing, it might use config default?
             # For now, simplistic check
             self.add_step(ResampleStep(self.config, self.logger, self.provenance, resolution=anat_cfg.get("resample", {}).get("resolution")))

        # 2. Reorient
        # Always run by default or config? 
        # User said "reorient ... using fslreorient2std" implying it's a step.
        if anat_cfg.get("reorient", {}).get("enabled", False):
             self.add_step(ReorientStep(self.config, self.logger, self.provenance))

        # 3. Denoise
        den_cfg = anat_cfg.get("denoising", {})
        if den_cfg.get("enabled"):
             # Ensure method defaults if not present
             self.add_step(DenoisingStep(self.config, self.logger, self.provenance, method=den_cfg.get("method", "ants")))

        # 4. Gibbs
        gibbs_cfg = anat_cfg.get("degibbs") or anat_cfg.get("gibbs", {})
        if gibbs_cfg.get("enabled"):
             # For anatomical (3D), standard method usually 'mrtrix' or 'dipy'
             # Note: MRtrix mrdegibbs supports T1w/T2w
             self.add_step(GibbsUnringingStep(self.config, self.logger, self.provenance, method=gibbs_cfg.get("method", "mrtrix")))

        # 5. Bias
        bias_cfg = anat_cfg.get("bias_correction", {})
        if bias_cfg.get("enabled"):
             self.add_step(BiasCorrectionStep(self.config, self.logger, self.provenance, method=bias_cfg.get("method", "ants"))) # Default N4
        
        # 5b. Sharpening (Optional)
        sharp_cfg = anat_cfg.get("sharpen", {})
        if sharp_cfg.get("enabled"):
             # Retrieve options or method
             self.add_step(SharpeningStep(self.config, self.logger, self.provenance, method=sharp_cfg.get("method", "ants")))
        
        # 6. Brain Mask
        mask_cfg = anat_cfg.get("brain_masking", {})
        # If using FreeSurfer, we use its brain mask, so skip this step
        use_fs = anat_cfg.get("use_freesurfer", False)
        
        if mask_cfg.get("enabled", False) and not use_fs:
             self.add_step(BrainMaskingStep(self.config, self.logger, self.provenance, method=mask_cfg.get("method", "ants")))

        # 7. Recon-all
        recon_cfg = anat_cfg.get("recon_all", {})
        if recon_cfg.get("enabled") or use_fs:
             self.add_step(ReconAllStep(self.config, self.logger, self.provenance))
             # Always add stats extraction if FS is enabled
             self.add_step(FreeSurferStatsStep(self.config, self.logger, self.provenance))
             
        # 8. Nonlinear Registration
        norm_cfg = anat_cfg.get("normalization", {})
        if norm_cfg.get("enabled"):
             # Template should be in config
             self.add_step(NonlinearRegistrationStep(self.config, self.logger, self.provenance, template=norm_cfg.get("template")))

        # 9. Segmentation
        seg_cfg = self.config.get("anat", {}).get("segmentation", {})
        if seg_cfg.get("enabled"):
             self.add_step(SegmentationStep(
                 self.config, self.logger, self.provenance,
                 atlas_file=seg_cfg.get("atlas_file"),
                 atlas_labels=seg_cfg.get("atlas_labels"),
                 metrics=seg_cfg.get("metrics"),
                 atlas_threshold=seg_cfg.get("atlas_threshold")
             ))


    def run(self, output_dir: Path, context: dict, final_output_dir: Optional[Path] = None, reporter=None) -> dict:
        """
        Execute workflow.
        """
        self.logger.info(f"Starting AnatPreprocessingWorkflow")
        
        t1w_files = context.get('t1w_files', [])
        t2w_files = context.get('t2w_files', [])
        
        if not t1w_files:
             raise ValueError("No T1w image found.")

        step_metrics = []
        processed_masks = []
        figures_dir = output_dir / "figures"
        if reporter: figures_dir.mkdir(exist_ok=True, parents=True)

        # Reporter: Inputs
        if reporter:
             t1 = t1w_files[0]
             reporter.set_header_info(context.get('subject'), context.get('session'), 
                 bids_dir=str(self.config.bids_dir), work_dir=str(output_dir))
             
             p = output_dir / "report_input_t1w.png"
             try:
                 from ...lib.reporting.viz import create_ortho_view, plot_comparison
                 create_ortho_view(t1.img, p, title="Input T1w")
                 # New API:
                 reporter.add_anat_input(modality="T1w", path=t1.img, figure_path=p, caption=t1.img.name)
                 
                 if t2w_files:
                     t2 = t2w_files[0]
                     p2 = output_dir / "report_input_t2w.png"
                     create_ortho_view(t2.img, p2, title="Input T2w")
                     reporter.add_anat_input(modality="T2w", path=t2.img, figure_path=p2, caption=t2.img.name)
                     
             except Exception as e:
                 self.logger.warning(f"Failed to plot input T1w/T2w: {e}")

        # Pre-calc counts for progress bar
        # T1 steps: Steps that are NOT Coreg/BrainMask/Recon/NonlinReg?
        # In current logic: Coreg/BrainMask skipped in T1 loop. Recon/NonlinReg kept?
        # The T1 loop explicitly skips Coreg, BrainMask.
        # It DOES execute ReconAll, NonlinReg? 
        # Wait, the logic is:
        # for step in self.steps: if is Coreg or BrainMask -> continue.
        # else run.
        # So T1 loop runs: Resample, Reorient, Denoise, Gibbs, Bias, Sharpen, ReconAll, NonlinReg.
        
        use_fs = self.config.get("anat", {}).get("use_freesurfer", False) or \
                 self.config.get("anat", {}).get("preprocessing", {}).get("use_freesurfer", False)

        t1_runnable = []
        for s in self.steps:
            if isinstance(s, (CoregistrationStep, BrainMaskingStep, NonlinearRegistrationStep, SegmentationStep)): 
                 continue
            if use_fs:
                 # If using FS, skip standard preproc for T1w
                 if isinstance(s, (ResampleStep, ReorientStep, DenoisingStep, GibbsUnringingStep, BiasCorrectionStep, SharpeningStep)):
                      continue
            t1_runnable.append(s)
        
        # T2 loop runs: Resample...Bias. Not Recon, Nonlin, Mask, Coreg.
        t2_runnable = [s for s in self.steps if not isinstance(s, (ReconAllStep, NonlinearRegistrationStep, BrainMaskingStep, CoregistrationStep, SegmentationStep))] if t2w_files else []
        
        # Coregistration: Runs if enabled and T1w & T2w exist.
        coreg_cfg_run = self.config.get("anat", {}).get("preprocessing", {}).get("coregistration", {})
        do_coreg = (t1w_files and t2w_files and coreg_cfg_run.get("enabled", False))
        
        # Brain Masking: Runs if step exists in self.steps
        mask_step = next((s for s in self.steps if isinstance(s, BrainMaskingStep)), None)
        do_mask = (mask_step is not None)
        
        total_steps = len(t1_runnable) + len(t2_runnable)
        if do_coreg: total_steps += 1
        if do_mask: total_steps += 1
        
        def _execute_anat(progress_ctx=None, task_id=None):
            from ...io.bids import build_bids_name
            
            save_inter = self.config.get("save_intermediate", False)
            skip_existing = self.config.get("skip_existing", False)
            
            # Force Run Override
            force_run = self.config.get("anat", {}).get("force_run", False)
            if force_run:
                 self.logger.info("Anatomical force_run enabled: Ignoring existing outputs.")
                 skip_existing = False
            
            # Primary flow: Process T1w
            if t1w_files:
                 # Process T1w
                 current_t1 = t1w_files[0]
                 context["current_image"] = current_t1
                 
                 processed_t1 = current_t1
                 
                 for step in self.steps:
                      if isinstance(step, (CoregistrationStep, BrainMaskingStep, NonlinearRegistrationStep, SegmentationStep, FreeSurferStatsStep)):
                           # Handle Coregistration, Normalization, Segmentation, FS Stats etc separately
                           continue
                      
                      # If using FS, skip standard T1w preproc
                      if use_fs and isinstance(step, (ResampleStep, ReorientStep, DenoisingStep, GibbsUnringingStep, BiasCorrectionStep, SharpeningStep)):
                           tracker = self.config.tracker
                           if tracker and context.get('subject') and context.get('session'):
                                tracker_module = step.normalize_tracker_module(step.__class__.__name__)
                                study = context.get('study_name', self.config.get('study_name'))
                                tracker.update_status(context['subject'], context['session'], tracker_module, "completed (FreeSurfer)", study=study, modality="Anatomical")
                           continue
                      
                      step_name = step.__class__.__name__
                      if progress_ctx:
                          progress_ctx.update(task_id, description=f"[cyan]T1w: {step_name}")
                      else:
                          self.logger.info(f"Running T1w step: {step_name}")

                      # --- RECOVERY LOGIC ---
                      # Determine descriptor for this step
                      step_desc = None
                      if isinstance(step, ResampleStep): step_desc = 'resample'
                      elif isinstance(step, ReorientStep): step_desc = 'reorient'
                      elif isinstance(step, DenoisingStep): step_desc = 'denoise'
                      elif isinstance(step, GibbsUnringingStep): step_desc = 'gibbs'
                      elif isinstance(step, BiasCorrectionStep): step_desc = 'bias'
                      elif isinstance(step, SharpeningStep): step_desc = 'sharpen'
                      elif isinstance(step, ReconAllStep): pass # Saved differently usually?
                      elif isinstance(step, NonlinearRegistrationStep): step_desc = 'normalize'
                      
                      skipped = False
                      if final_output_dir and step_desc:
                          # Construct expected filename
                          ents = dict(processed_t1.entities)
                          ents['desc'] = step_desc
                          if 'suffix' not in ents: ents['suffix'] = 'T1w'
                          fname = build_bids_name(ents)
                          expected_path = final_output_dir / fname
                          
                          # Check for existence (standard or final preproc fallback)
                          if not expected_path.exists() and skip_existing and t1_runnable and step == t1_runnable[-1]:
                               # Last step: check for final desc-preproc if specific desc-sharpen missing
                               p_ents = dict(processed_t1.entities)
                               p_ents['desc'] = 'preproc'
                               if 'suffix' not in p_ents: p_ents['suffix'] = 'T1w'
                               p_fname = build_bids_name(p_ents)
                               p_dest = final_output_dir / p_fname
                               if p_dest.exists():
                                   # Redirect to final output
                                   expected_path = p_dest
                                   fname = p_fname # Update for logging purposes
                          
                          if expected_path.exists() and (skip_existing or save_intermediate):
                               self.logger.info(f"Skipping {step_name} (Found existing output: {fname})")
                               # Load existing image to proceed
                               import nibabel as nib
                               try:
                                   # Manually wrap
                                   img_obj = ImageFile(entities=ents, img=expected_path) # Fixed instantiation
                                   processed_t1 = img_obj
                                   skipped = True
                                   
                                   step_metrics.append({
                                      "Step": f"T1w_{step_name}",
                                      "Status": "Skipped (Found)",
                                      "Duration": "0s"
                                   })
                                   
                                   # Update tracker for skip
                                   tracker_module = step.normalize_tracker_module(step_name)
                                   tracker = self.config.tracker
                                   subject = context.get('subject')
                                   session = context.get('session')
                                   study = context.get('study_name', self.config.get('study_name'))
                                   
                                   if tracker and subject and session:
                                       tracker.update_status(subject, session, tracker_module, "completed (cached)", study=study, modality="Anatomical")
                                       tracker.save()
                               except Exception as e:
                                   self.logger.warning(f"Failed to load existing intermediate {fname}: {e}. Re-running.")
                      
                      if not skipped:
                          # ReconAllStep might depend on T1w specifically
                          prev_t1 = processed_t1
                          st = time.time()
                          # Pass force if step accepts it within kwargs (BaseProcessingStep does)
                          processed_t1 = step(processed_t1, output_dir=output_dir, force=force_run)
                          dur = time.time() - st
                          
                          step_metrics.append({
                              "Step": f"T1w_{step_name}",
                              "Status": "Completed",
                              "Duration": f"{dur:.2f}s"
                          })
                          
                          # If step returned context, extract image
                          if isinstance(processed_t1, dict):
                               context.update(processed_t1)
                               processed_t1 = context["current_image"]
                      
                      # --- INTERMEDIATE SAVING ---
                      if save_inter and not skipped and final_output_dir and step_desc:
                             ents = dict(processed_t1.entities)
                             ents['desc'] = step_desc
                             if 'suffix' not in ents: ents['suffix'] = 'T1w'
                             fname = build_bids_name(ents)
                             dest = final_output_dir / fname
                             
                             if not dest.exists(): # Don't overwrite if we just loaded it (though skipped ensures)
                                 dest.parent.mkdir(parents=True, exist_ok=True)
                                 shutil.copy(processed_t1.img, dest)
                                 # Sidecar?
                                 if processed_t1.json and processed_t1.json.exists():
                                     shutil.copy(processed_t1.json, dest.with_suffix("").with_suffix(".json"))

                      # Reporting: T1w
                      if reporter and not skipped: # Only plot newly computed? Or always plot? Always plot is better for complete report.
                           try:
                               from ...lib.reporting.viz import create_ortho_view
                               fig_list = []
                               
                               if isinstance(step, DenoisingStep):
                                    p = figures_dir / "t1w_denoised.png"
                                    # Plot even if skipped? plotting requires loading data. 
                                    # If skipped, processed_t1 is loaded.
                                    create_ortho_view(processed_t1.img, p, title="Denoised T1w")
                                    fig_list.append({"path": str(p), "title": "Denoising", "caption": "Denoised T1w"})
                                    
                               elif isinstance(step, GibbsUnringingStep):
                                    p = figures_dir / "t1w_gibbs.png"
                                    create_ortho_view(processed_t1.img, p, title="Gibbs Corrected T1w")
                                    fig_list.append({"path": str(p), "title": "Gibbs", "caption": "Gibbs Corrected T1w"})
                               
                               # Extract params (simplified assumption: step attributes)
                               details = {"Modality": "T1w"}
                               if hasattr(step, "method"): details["Method"] = step.method
                               if hasattr(step, "patch_radius"): details["Patch Radius"] = step.patch_radius
                               
                               reporter.add_anat_step(step_name, details, figures=fig_list)
                               
                           except ImportError:
                               pass
                           except Exception as e:
                               self.logger.warning(f"Failed to plot T1w report figure: {e}")
                                
                      if progress_ctx: progress_ctx.advance(task_id)

                 
                 context["preprocessed_t1w"] = processed_t1
                 self.logger.info(f"T1w processing complete: {processed_t1.img}")

            # Process T2w if exists
            if t2w_files:
                 current_t2 = t2w_files[0]
                 processed_t2 = current_t2
                 
                 for step in self.steps:
                      if isinstance(step, (ReconAllStep, NonlinearRegistrationStep, BrainMaskingStep, CoregistrationStep, SegmentationStep)):
                           continue
                      
                      step_name = step.__class__.__name__
                      if progress_ctx:
                          progress_ctx.update(task_id, description=f"[cyan]T2w: {step_name}")
                      else:
                          self.logger.info(f"Running T2w step: {step_name}")
                      
                      # --- RECOVERY T2w ---
                      step_desc = None
                      if isinstance(step, ResampleStep): step_desc = 'resample'
                      elif isinstance(step, ReorientStep): step_desc = 'reorient'
                      elif isinstance(step, DenoisingStep): step_desc = 'denoise'
                      elif isinstance(step, GibbsUnringingStep): step_desc = 'gibbs'
                      elif isinstance(step, BiasCorrectionStep): step_desc = 'bias'
                      elif isinstance(step, SharpeningStep): step_desc = 'sharpen'
                      
                      skipped = False
                      if final_output_dir and step_desc:
                          ents = dict(processed_t2.entities)
                          ents['desc'] = step_desc
                          if 'suffix' not in ents: ents['suffix'] = 'T2w'
                          fname = build_bids_name(ents)
                          expected_path = final_output_dir / fname
                          
                          if expected_path.exists() and (skip_existing or save_intermediate):
                               self.logger.info(f"Skipping {step_name} (Found existing output: {fname})")
                               try:
                                   img_obj = ImageFile(expected_path, self.config.bids_dir)
                                   processed_t2 = img_obj
                                   skipped = True
                                   step_metrics.append({"Step": f"T2w_{step_name}", "Status": "Skipped (Found)", "Duration": "0s"})
                                   
                                   # Update tracker for skip
                                   tracker_module = step.normalize_tracker_module(step_name)
                                   tracker = self.config.tracker
                                   subject = context.get('subject')
                                   session = context.get('session')
                                   study = context.get('study_name', self.config.get('study_name'))
                                   
                                   if tracker and subject and session:
                                       tracker.update_status(subject, session, tracker_module, "completed (cached)", study=study, modality="Anatomical")
                                       tracker.save()
                               except Exception:
                                   pass

                      if not skipped:
                          st = time.time()
                          st = time.time()
                          processed_t2 = step(processed_t2, output_dir=output_dir)
                          dur = time.time() - st
                          
                          step_metrics.append({
                              "Step": f"T2w_{step_name}",
                              "Status": "Completed",
                              "Duration": f"{dur:.2f}s"
                          })

                          if isinstance(processed_t2, dict):
                                processed_t2 = processed_t2.get("current_image", processed_t2)
                      
                      # --- SAVE INTERMEDIATE T2w ---
                      if save_inter and not skipped and final_output_dir and step_desc:
                             ents = dict(processed_t2.entities)
                             ents['desc'] = step_desc
                             if 'suffix' not in ents: ents['suffix'] = 'T2w'
                             fname = build_bids_name(ents)
                             dest = final_output_dir / fname
                             
                             if not dest.exists():
                                 dest.parent.mkdir(parents=True, exist_ok=True)
                                 shutil.copy(processed_t2.img, dest)
                                 if processed_t2.json and processed_t2.json.exists():
                                     shutil.copy(processed_t2.json, dest.with_suffix("").with_suffix(".json"))
                      
                      # Reporting: T2w
                      if reporter: # Plotting recovery?
                           try: 
                                from ...lib.reporting.viz import create_ortho_view
                                fig_list = []
                                if isinstance(step, DenoisingStep):
                                     p = figures_dir / "t2w_denoised.png"
                                     create_ortho_view(processed_t2.img, p, title="Denoised T2w")
                                     fig_list.append({"path": str(p), "title": "Denoising", "caption": "Denoised T2w"})
                                
                                details = {"Modality": "T2w"}
                                if hasattr(step, "method"): details["Method"] = step.method
                                
                                reporter.add_anat_step(step_name, details, figures=fig_list)
                           except Exception as e:
                                self.logger.warning(f"Failed to plot T2w report figure: {e}")
                                
                      if progress_ctx: progress_ctx.advance(task_id)

                 self.logger.info(f"T2w processing complete: {processed_t2.img}")
                 
                 # Coregister only if T1w exists and T1w processing successful
                 if t1w_files and "preprocessed_t1w" in context:
                      processed_t1 = context["preprocessed_t1w"]
                      
                      if do_coreg:
                           if progress_ctx: progress_ctx.update(task_id, description="[cyan]Coregistration")
                           
                           coreg_step = CoregistrationStep(self.config, self.logger, self.provenance, method=coreg_cfg_run.get("method", "fsl"))
                           coreg_step.modality = "Anatomical"
                           
                           ref_img = coreg_cfg_run.get("reference_image", "t1w").lower()
                           
                           # Flatten options logic (Top-level + Nested 'options')
                           coreg_options = dict(coreg_cfg_run)
                           if "options" in coreg_options:
                               sub_opts = coreg_options.pop("options")
                               if isinstance(sub_opts, dict):
                                   coreg_options.update(sub_opts)
                           
                           st = time.time()
                           if ref_img == 't2w':
                               # Reference is T2w. Moving is T1w.
                               self.logger.info("Coregistration: Reference=T2w. Registering T1w -> T2w.")
                               res_t1 = coreg_step(processed_t1, output_dir=output_dir, target=processed_t2.img, options=coreg_options)
                               if isinstance(res_t1, dict): 
                                    res_t1 = res_t1.get("current_image")
                               
                               # Update T1w in context to be the coregistered one
                               context["preprocessed_t1w"] = res_t1
                               # Plot: T1w aligned on T2w
                               if reporter:
                                    try:
                                        from ...lib.reporting.viz import plot_comparison
                                        p = figures_dir / "coreg_t1_on_t2.png"
                                        plot_comparison(processed_t2.img, res_t1.img, p, title="Coreg T1w -> T2w")
                                        # New API
                                        details = {"Method": coreg_step.method, "Reference": "T2w", "Moving": "T1w"}
                                        fig_item = [{"path": str(p), "title": "Coregistration", "caption": "T1w (overlay) on T2w"}]
                                        reporter.add_anat_step("Coregistration", details, figures=fig_item)
                                    except Exception as e:
                                        self.logger.warning(f"Failed to plot Coregistration report: {e}")

                           else:
                               # Reference is T1w (Default). Moving is T2w.
                               self.logger.info("Coregistration: Reference=T1w. Registering T2w -> T1w.")
                               res_t2 = coreg_step(processed_t2, output_dir=output_dir, target=processed_t1.img, options=coreg_options)
                               if isinstance(res_t2, dict):
                                    res_t2 = res_t2.get("current_image")
                               
                               context["preprocessed_t2w_coreg"] = res_t2
                               # Plot: T2w aligned on T1w
                               if reporter:
                                    try:
                                        from ...lib.reporting.viz import plot_comparison
                                        p = figures_dir / "coreg_t2_on_t1.png"
                                        plot_comparison(processed_t1.img, res_t2.img, p, title="Coreg T2w -> T1w")
                                        # New API
                                        details = {"Method": coreg_step.method, "Reference": "T1w", "Moving": "T2w"}
                                        fig_item = [{"path": str(p), "title": "Coregistration", "caption": "T2w (overlay) on T1w"}]
                                        reporter.add_anat_step("Coregistration", details, figures=fig_item)
                                    except Exception as e:
                                        self.logger.warning(f"Failed to plot Coregistration report: {e}")
                           
                           dur = time.time() - st
                           step_metrics.append({"Step": "Coregistration", "Status": "Completed", "Duration": f"{dur:.2f}s"})
                           
                           if progress_ctx: progress_ctx.advance(task_id)
                           
                           # SAVE INTERMEDIATE: Coregistration
                           if save_inter and final_output_dir:
                                img_to_save = context.get("preprocessed_t1w") # T1 aligned to T2
                                if ref_img == 't1w': img_to_save = context.get("preprocessed_t2w_coreg") # T2 aligned to T1
                                
                                if img_to_save:
                                     ents = dict(img_to_save.entities)
                                     ents['desc'] = 'coreg'
                                     if 'suffix' not in ents: ents['suffix'] = 'T1w' if ref_img == 't2w' else 'T2w'
                                     fname = build_bids_name(ents)
                                     dest = final_output_dir / fname
                                     if not dest.exists():
                                         dest.parent.mkdir(parents=True, exist_ok=True)
                                         shutil.copy(img_to_save.img, dest)
                                         if img_to_save.json and img_to_save.json.exists():
                                              shutil.copy(img_to_save.json, dest.with_suffix("").with_suffix(".json"))

                 else:
                      self.logger.info("Skipping Coregistration (No T1w present).")
                 
           
            # 7. Post-Processing Brain Masking
            if mask_step:
                if progress_ctx: progress_ctx.update(task_id, description="[cyan]Brain Masking")
                
                # Determine reference image
                ref_img_key = "preprocessed_t1w" # default assumption: T1w is structural reference
                coreg_cfg_local = self.config.get("anat", {}).get("preprocessing", {}).get("coregistration", {})
                ref_mode = coreg_cfg_local.get("reference_image", "t1w").lower()
                
                if ref_mode == 't2w':
                     ref_img_key = "preprocessed_t2w"
                
                # Fallback
                if ref_img_key == "preprocessed_t1w" and "preprocessed_t1w" not in context and "preprocessed_t2w" in context:
                    ref_img_key = "preprocessed_t2w"
                elif ref_img_key == "preprocessed_t2w" and "preprocessed_t2w" not in context and "preprocessed_t1w" in context:
                    ref_img_key = "preprocessed_t1w"
                
                target_img = context.get(ref_img_key)
                
                if target_img:
                    self.logger.info(f"Generating binary brain mask using reference: {ref_img_key} ({mask_step.method})")
                    
                    # Check for existing mask (SKIP LOGIC)
                    skipped_mask = False
                    if final_output_dir and skip_existing:
                         # Construct expected mask name. Usually desc-preproc_mask
                         # Matches save_results logic (desc='preproc', suffix='mask')
                         m_ents = dict(target_img.entities)
                         m_ents['desc'] = 'preproc'
                         m_ents['suffix'] = 'mask'
                         # Ensure no stale keys
                         for k in ['space', 'res']: m_ents.pop(k, None)
                         
                         m_fname = build_bids_name(m_ents)
                         m_dest = final_output_dir / m_fname
                         
                         if m_dest.exists():
                              self.logger.info(f"Skipping Brain Masking (Found existing mask: {m_fname})")
                              context["brain_mask"] = ImageFile(entities=m_ents, img=m_dest)
                              skipped_mask = True
                              step_metrics.append({"Step": "BrainMasking", "Status": "Skipped (Found)", "Duration": "0s"})
                              
                              # Update tracker for skip
                              tracker_module = mask_step.normalize_tracker_module(mask_step.__class__.__name__)
                              tracker = self.config.tracker
                              subject = context.get('subject')
                              session = context.get('session')
                              study = context.get('study_name', self.config.get('study_name'))
                              
                              if tracker and subject and session:
                                  tracker.update_status(subject, session, tracker_module, "completed (cached)", study=study, modality="Anatomical")
                                  tracker.save()
                    
                    if not skipped_mask:
                        st = time.time()
                    try:
                        brain_masked, mask = mask_step(target_img, output_dir=output_dir, return_mask=True)
                        processed_masks.append(mask)
                        dur = time.time() - st
                        step_metrics.append({"Step": "BrainMasking", "Status": "Completed", "Duration": f"{dur:.2f}s"})
                        
                        # Add mask to context
                        context["brain_mask"] = mask

                        if reporter:
                             p = figures_dir / "brain_mask_check.png"
                             # Overlay mask on target
                             try:
                                from ...lib.reporting.viz import plot_comparison
                                plot_comparison(target_img.img, mask.img, p, title="Brain Mask Check", overlay_alpha=0.3, overlay_cmap="autumn")
                                details = {"Method": mask_step.method, "Reference Image": ref_img_key}
                                fig_item = [{"path": str(p), "title": "Brain Mask", "caption": "Brain Mask (overlay) on Reference"}]
                                reporter.add_anat_step("BrainMasking", details, figures=fig_item)
                             except Exception as e:
                                 self.logger.warning(f"Failed to plot brain mask report: {e}")
                             
                    except Exception as e:
                         self.logger.warning(f"Brain Masking failed: {e}")
                         step_metrics.append({"Step": "BrainMasking", "Status": "Failed", "Duration": "0s"})
                    
                    # Fix indentation for else block of checked_mask if needed, but structure implies simple if not skipped...
                    pass 
                else:
                    self.logger.warning("No reference image found for brain masking.")
                
                if progress_ctx: progress_ctx.advance(task_id)

                # SAVE INTERMEDIATE: Brain Mask
                if save_inter and final_output_dir and "brain_mask" in context:
                     mask_to_save = context["brain_mask"]
                     ents = dict(mask_to_save.entities)
                     ents['desc'] = 'brain' # Raw brain mask from step
                     ents['suffix'] = 'mask'
                     fname = build_bids_name(ents)
                     dest = final_output_dir / fname
                     if not dest.exists():
                         dest.parent.mkdir(parents=True, exist_ok=True)
                         shutil.copy(mask_to_save.img, dest)

            # 8. Normalization (Nonlinear Registration)
            # Executed after Coreg/Masking to ensure T1/T2 aligned first, and Mask available
            norm_step = next((s for s in self.steps if isinstance(s, NonlinearRegistrationStep)), None)
            if norm_step:
                 if progress_ctx: progress_ctx.update(task_id, description=f"[cyan]Normalization ({norm_step.template})")
                 
                 # 1. Normalize T1w (Source)
                 t1_img = context.get("preprocessed_t1w")
                 # Mask?
                 t1_mask = context.get("brain_mask")
                 
                 if t1_img:
                      self.logger.info("Running Normalization on T1w...")
                      # Run step (updates context with 'template_transform')
                      norm_step(context, output_dir=output_dir) 
                      
                      # output is updated 'current_image' -> context['current_image']
                      # We should update preprocessed_t1w
                      norm_t1 = context.get("current_image")
                      context["preprocessed_t1w"] = norm_t1
                      
                      # 2. Apply to T2w
                      transform = context.get("template_transform")
                      t2_img = context.get("preprocessed_t2w")
                      if "preprocessed_t2w_coreg" in context: t2_img = context["preprocessed_t2w_coreg"]
                      
                      if t2_img and transform and transform.exists():
                           self.logger.info("Applying Normalization Warp to T2w...")
                           # Manual Apply
                           
                           ents = dict(t2_img.entities)
                           ents['space'] = 'Standard'
                           ents['desc'] = 'norm'
                           if 'suffix' not in ents: ents['suffix'] = 'T2w'
                           
                           norm_t2_path = output_dir / build_bids_name(ents)
                           
                           if not norm_t2_path.exists() or not self.config.skip_existing:
                                template = norm_step.template or self.config.get("anat", {}).get("preprocessing", {}).get("normalization", {}).get("template")
                                
                                if template:
                                     prefix = Path(transform)
                                     warp = prefix.with_suffix("").parent / (prefix.name + "1Warp.nii.gz")
                                     affine = prefix.with_suffix("").parent / (prefix.name + "0GenericAffine.mat")
                                     
                                     if warp.exists() and affine.exists():
                                          ants.apply_transforms(
                                            fixed_file=Path(template),
                                            moving_file=t2_img.img,
                                            out_file=norm_t2_path,
                                            transforms=[warp, affine],
                                            interpolator='linear' # T2w is structural
                                          )
                                          # Wrap result
                                          # Wrap result
                                          norm_t2_obj = ImageFile(norm_t2_path, self.config.bids_dir, entities=ents)
                                          context["preprocessed_t2w"] = norm_t2_obj # Update T2w to normalized
                                     else:
                                          self.logger.warning("Could not find warp/affine files for T2 normalization.")
                                else:
                                     self.logger.warning("No template reference found for T2 normalization apply.")
                           else:
                                self.logger.info(f"Skipping T2w Normalization (Exists): {norm_t2_path}")
                                context["preprocessed_t2w"] = ImageFile(norm_t2_path, self.config.bids_dir, entities=ents)
                                
                                # Update tracker for skip
                                tracker = self.config.tracker
                                if tracker and context.get('subject') and context.get('session'):
                                     # Use 'Coregistration' as normalization is mapped there
                                     study = context.get('study_name', self.config.get('study_name'))
                                     tracker.update_status(context['subject'], context['session'], "Coregistration", "completed (cached)", study=study, modality="Anatomical")
                                     tracker.save()

                 if progress_ctx: progress_ctx.advance(task_id)

                 # SAVE INTERMEDIATE: Normalization
                 if save_inter and final_output_dir:
                      # Save T1w Normalized
                      if "preprocessed_t1w" in context:
                           norm_t1 = context["preprocessed_t1w"]
                           # Check if it looks normalized (standard space)
                           if getattr(norm_t1, 'entities', {}).get('space') == 'Standard':
                                ents = dict(norm_t1.entities)
                                ents['desc'] = 'norm'
                                if 'suffix' not in ents: ents['suffix'] = 'T1w'
                                fname = build_bids_name(ents)
                                dest = final_output_dir / fname
                                if not dest.exists():
                                     dest.parent.mkdir(parents=True, exist_ok=True)
                                     shutil.copy(norm_t1.img, dest)
                                     if norm_t1.json and norm_t1.json.exists():
                                          shutil.copy(norm_t1.json, dest.with_suffix("").with_suffix(".json"))
                      
                      # Save T2w Normalized
                      if "preprocessed_t2w" in context:
                           norm_t2 = context["preprocessed_t2w"]
                           if getattr(norm_t2, 'entities', {}).get('space') == 'Standard':
                                ents = dict(norm_t2.entities)
                                ents['desc'] = 'norm'
                                if 'suffix' not in ents: ents['suffix'] = 'T2w'
                                fname = build_bids_name(ents)
                                dest = final_output_dir / fname
                                if not dest.exists():
                                     dest.parent.mkdir(parents=True, exist_ok=True)
                                     shutil.copy(norm_t2.img, dest)
                                     if norm_t2.json and norm_t2.json.exists():
                                         shutil.copy(norm_t2.json, dest.with_suffix("").with_suffix(".json"))

            # 8.5 FreeSurfer Stats
            fs_stats_step = next((s for s in self.steps if isinstance(s, FreeSurferStatsStep)), None)
            if fs_stats_step:
                 if progress_ctx: progress_ctx.update(task_id, description="[cyan]FreeSurfer Stats")
                 self.logger.info("Parsing FreeSurfer Stats...")
                 fs_stats_step(context, output_dir=output_dir)
                 if progress_ctx: progress_ctx.advance(task_id)
                 
            # 9. Segmentation
            # Runs on whatever is in context (Normalized T1/T2 if Norm ran, else Native/Coreg)
            seg_step = next((s for s in self.steps if isinstance(s, SegmentationStep)), None)
            if seg_step:
                 if progress_ctx: progress_ctx.update(task_id, description="[cyan]Segmentation")
                 self.logger.info("Running Segmentation...")
                 seg_step(context, output_dir=output_dir)
                 if progress_ctx: progress_ctx.advance(task_id)
                 
                 # SAVE INTERMEDIATE: Segmentation
                 if save_inter and final_output_dir:
                      # Segmentation usually outputs multiple files (prob maps, seg image).
                      # context["segmentation"] might be the dseg image.
                      if "segmentation" in context:
                           seg = context["segmentation"]
                           # It might be a dict or ImageFile
                           if hasattr(seg, 'img'):
                                ents = dict(seg.entities)
                                # maintain desc/suffix from step
                                fname = build_bids_name(ents)
                                dest = final_output_dir / fname
                                if not dest.exists():
                                     dest.parent.mkdir(parents=True, exist_ok=True)
                                     shutil.copy(seg.img, dest)
                                     if seg.json and seg.json.exists():
                                          shutil.copy(seg.json, dest.with_suffix("").with_suffix(".json"))
                      
                      # Also save 5tt if exists?
                      if "5tt" in context:
                           f5tt = context["5tt"]
                           if hasattr(f5tt, 'img'):
                                ents = dict(f5tt.entities)
                                fname = build_bids_name(ents)
                                dest = final_output_dir / fname
                                if not dest.exists():
                                     dest.parent.mkdir(parents=True, exist_ok=True)
                                     shutil.copy(f5tt.img, dest)

            # Add Output Paths to Summary
            if final_output_dir:
                 anat_outputs = {
                     "Final Preprocessed Images": [],
                     "Modeling Derivatives": [],
                     "Normalized Derivatives": [],
                     "Segmentation Outputs": []
                 }
                 
                 # T1w Path
                 pre_t1 = context.get("preprocessed_t1w")
                 if pre_t1:
                     ents = dict(pre_t1.entities)
                     ents['desc'] = 'preproc'
                     if 'suffix' not in ents: ents['suffix'] = 'T1w'
                     t1_path = final_output_dir / build_bids_name(ents)
                     anat_outputs["Final Preprocessed Images"].append({"key": "Preprocessed T1w", "path": str(t1_path)})
                 
                 # T2w Path
                 pre_t2 = context.get("preprocessed_t2w_coreg") or context.get("preprocessed_t2w")
                 if pre_t2:
                     ents = dict(pre_t2.entities)
                     ents['desc'] = 'preproc'
                     if 'suffix' not in ents: ents['suffix'] = 'T2w'
                     t2_path = final_output_dir / build_bids_name(ents)
                     anat_outputs["Final Preprocessed Images"].append({"key": "Preprocessed T2w", "path": str(t2_path)})
                     
                 # Mask Path
                 mask_img = context.get("brain_mask")
                 if mask_img:
                     ents = dict(mask_img.entities)
                     ents['desc'] = 'preproc'
                     ents['suffix'] = 'mask'
                     mask_path = final_output_dir / build_bids_name(ents)
                     anat_outputs["Segmentation Outputs"].append({"key": "Brain Mask", "path": str(mask_path)})
                 
                 # Scan for Intermediate Outputs (Normalized, Segmentation, Coreg)
                 # Normalized
                 for f in final_output_dir.glob("*desc-norm*.nii.gz"):
                      anat_outputs["Normalized Derivatives"].append({"key": f.name, "path": str(f)})
                 
                 # Segmentation
                 for f in final_output_dir.glob("*dseg*.nii.gz"):
                      anat_outputs["Segmentation Outputs"].append({"key": f.name, "path": str(f)})
                 for f in final_output_dir.glob("*probseg*.nii.gz"):
                      anat_outputs["Segmentation Outputs"].append({"key": f.name, "path": str(f)})
                 for f in final_output_dir.glob("*5tt*.nii.gz"):
                      anat_outputs["Segmentation Outputs"].append({"key": f.name, "path": str(f)})
                 
                 if reporter and anat_outputs:
                     reporter.set_anat_outputs(anat_outputs)

            if reporter:
                 reporter.add_anat_summary("Anatomical Execution Summary", step_metrics)

            # Save results if final_output_dir provided
            if final_output_dir:
                self.save_results(context, final_output_dir)

            # 6. Tracker Update
            try:
                 # Ensure study name is in context
                 context['study_name'] = self.config.get('study_name')
                 tracking = TrackingStep(self.config, self.logger)
                 tracking.run(context, final_output_dir or output_dir)
            except Exception as e:
                 self.logger.warning(f"Tracker update failed: {e}")

            return context
        
        # Execution Wrapper
        try:
            from qmri_neuropipe.core.ui import console
            from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
        except ImportError:
            console = None
            Progress = None
            
        if Progress and total_steps > 0:
             with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeRemainingColumn(),
                transient=True,
                console=console
             ) as progress:
                 task = progress.add_task(f"[cyan]Starting Anatomical Preprocessing...", total=total_steps)
                 return _execute_anat(progress_ctx=progress, task_id=task)
        else:
             return _execute_anat()

    def _update_json_history(self, json_path: Path, steps: list):
        """Update JSON sidecar with processing history."""
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

    def save_results(self, context, output_dir: Path):
         """Save final anatomical outputs to output_dir."""
         # Ensure output dir exists
         output_dir.mkdir(parents=True, exist_ok=True)
         
         # Copy T1w
         pre_t1 = context.get("preprocessed_t1w")
         if pre_t1 and hasattr(pre_t1, 'img') and pre_t1.img.exists():
               # Build name with desc='preproc'
               entities = dict(pre_t1.entities)
               entities['desc'] = 'preproc'
               # Ensure suffix is correct (T1w)
               if 'suffix' not in entities: entities['suffix'] = 'T1w'
               
               fname = build_bids_name(entities)
               dest = output_dir / fname
               
               if not dest.exists():
                    self.logger.info(f"Saving Final Anat T1w: {dest}")
                    shutil.copy(pre_t1.img, dest)
                    
                    # Save and update JSON
                    dest_json = dest.with_suffix("").with_suffix(".json")
                    if pre_t1.json and pre_t1.json.exists():
                            shutil.copy(pre_t1.json, dest_json)
                    
                    # Update JSON with steps
                    self._update_json_history(dest_json, self.steps)
               else:
                    self.logger.info(f"Final Anat T1w already exists, skipping copy: {dest}")
               
               # Update context to point to final preproc image
               context['preprocessed_t1w'] = ImageFile(entities=entities, img=dest)

         # Copy Brain Mask
         mask = context.get("brain_mask")
         if mask and hasattr(mask, 'img') and mask.img.exists():
              entities = dict(mask.entities)
              # Ensure correct naming. Usually aligns with reference.
              # Use desc='preproc'? or 'brain'?
              # User said "binary brain mask included... similar to diffusion".
              # Let's use suffix='mask'.
              entities['desc'] = 'preproc' # Matches the image it masks
              entities['suffix'] = 'mask'
              
              fname = build_bids_name(entities)
              dest = output_dir / fname
              
              if not dest.exists():
                  self.logger.info(f"Saving Final Brain Mask: {dest}")
                  shutil.copy(mask.img, dest)
                  # No sidecar usually for mask, or simple one?
                  dest_json = dest.with_suffix("").with_suffix(".json")
                  # if mask.json... often mask doesn't have rich json.
              else:
                   self.logger.info(f"Final Brain Mask already exists, skipping copy: {dest}")
              
              # Update context
              context['brain_mask'] = ImageFile(entities=entities, img=dest)

         # Copy T2w (processed)
         pre_t2 = context.get("preprocessed_t2w_coreg") or context.get("preprocessed_t2w")
         if pre_t2 and hasattr(pre_t2, 'img') and pre_t2.img.exists():
             entities = dict(pre_t2.entities)
             entities['desc'] = 'preproc'
             if 'suffix' not in entities: entities['suffix'] = 'T2w'
             
             fname = build_bids_name(entities)
             dest = output_dir / fname
             
             if not dest.exists():
                 self.logger.info(f"Saving Final Anat T2w: {dest}")
                 shutil.copy(pre_t2.img, dest)
                 
                 dest_json = dest.with_suffix("").with_suffix(".json")
                 if pre_t2.json and pre_t2.json.exists():
                     shutil.copy(pre_t2.json, dest_json)
                 
                 self._update_json_history(dest_json, self.steps) # referencing self.steps of workflow
             else:
                 self.logger.info(f"Final Anat T2w already exists, skipping copy: {dest}")
             
             # Update context to point to final preproc image (preferring coreg if present)
             key = "preprocessed_t2w_coreg" if "preprocessed_t2w_coreg" in context else "preprocessed_t2w"
             context[key] = ImageFile(entities=entities, img=dest)


class AnatPipeline(BasePipeline):
    """
    Top-level Anatomical Pipeline.
    Finds T1w/T2w files and runs AnatPreprocessingWorkflow.
    """
    
    def __init__(self, config: PipelineConfig):
        super().__init__(config)
        # self.preprocessing initialized in _initialize_pipeline

    @property
    def name(self) -> str:
        return "anat-pipeline"

    @property
    def version(self) -> str:
        return "1.0.0"

    def _initialize_pipeline(self) -> None:
        """Initialize workflows."""
        self.preprocessing = AnatPreprocessingWorkflow(self.config, self.logger, self.provenance)

    def _get_work_dir(self, subject: str, session: str | None = None) -> Path:
        """Get working directory for subject/session."""
        work_root = Path(self.config.get('work_dir'))
        if session:
            return work_root / f'sub-{subject}' / f'ses-{session}' / 'anat'
        return work_root / f'sub-{subject}' / 'anat'

    def _get_output_dir(self, subject: str, session: str | None = None) -> Path:
        """Get output directory for subject/session."""
        out_root = Path(self.config.get('output_dir'))
        if session:
            return out_root / f'sub-{subject}' / f'ses-{session}'
        return out_root / f'sub-{subject}'

    def process_subject(self, subject: str, session: str | None = None):
        """Process a single subject/session."""
        ses = f"ses-{session}" if session else ""
        
        # QC: MRIQC
        qc_cfg = self.config.get("qc", {}).get("mriqc", {})
        if qc_cfg.get("enabled"):
              # Determine modalities to run
              # Default for AnatPipeline is structural
              pipeline_mods = {'T1w', 'T2w'}
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
                       self.logger.warning(f"MRIQC failed: {e}")
                       if self.config.stop_on_error: raise e
              else:
                   self.logger.info(f"Skipping MRIQC for sub-{subject} (Computed modalities empty. Config: {cfg_mods})")
        
        # Find inputs
        subj_dir = self.config.bids_dir / f"sub-{subject}"
        if session:
            subj_dir = subj_dir / f"ses-{session}"
            
        anat_dir = subj_dir / "anat"
        
        t1w = bids_find_t1w(anat_dir)
        t2w = bids_find_t2w(anat_dir)
        
        if not t1w and not t2w:
             self.logger.warning(f"No T1w or T2w found for sub-{subject}. Skipping.")
             return
        
        subj_work_dir = self._get_work_dir(subject, session)
        
        context = {
             "subject": subject,
             "session": session,
             "t1w_files": t1w,
             "t2w_files": t2w
        }
        
        # Initialize Reporter
        final_anat_dir = self._get_output_dir(subject, session) / "anat"
        report_title = f"QMRI-Neuropipe Report: sub-{subject} {ses}"
        
        # Save report one level up? Or in anat dir?
        # User requested reports one dir up for dMRI pipeline.
        # Let's apply consistent logic: session root.
        final_anat_dir.mkdir(parents=True, exist_ok=True)
        reporter = ReportGenerator(final_anat_dir.parent, title=report_title)

        # Participant Summary
        part_summ = f"Participant: sub-{subject}"
        if session: part_summ += f", Session: {session}"
        reporter.set_participant_summary(part_summ, details={
            "Subject": subject,
            "Session": session or "N/A",
            "BIDS Path": str(self.config.bids_dir),
            "Output Path": str(self.config.output_dir)
        })
        
        try:
             # run returns results dict
             results = self.preprocessing.run(subj_work_dir, context, final_output_dir=final_anat_dir, reporter=reporter)
             
             # Generate Report
             reporter.generate()
             try:
                 reporter.generate_pdf()
             except Exception:
                 pass
             
             # Final Overall Status Update
             tracker = self.config.tracker
             if tracker and subject and session:
                  study = self.config.get('study_name')
                  tracker.update_status(subject, session, "Overall_Status", "Complete", study=study, modality="Anatomical")
                  tracker.save()
                  
        except Exception as e:
             self.logger.error(f"Error processing sub-{subject}: {e}")
             if self.config.stop_on_error: raise e
