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

# Steps
from ...lib.common.resample import ResampleStep
from ...lib.common.reorient import ReorientStep
from ...lib.common.denoise import DenoisingStep
from ...lib.common.gibbs import GibbsUnringingStep
from ...lib.common.bias import BiasCorrectionStep
from ...lib.common.registration import CoregistrationStep, NonlinearRegistrationStep
from ...lib.common.mask import BrainMaskingStep
from ...lib.anat.recon import ReconAllStep
from ...lib.common.mask import BrainMaskingStep
from ...lib.anat.recon import ReconAllStep
from ...lib.common.sharpen import SharpeningStep
from ...interfaces.mriqc import run_mriqc
from ...lib.reporting.report import ReportGenerator
from ...lib.reporting.viz import create_ortho_view, plot_comparison
import time
try:
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
    console = Console()
except ImportError:
    console = None
    Progress = None

class AnatPreprocessingWorkflow(BaseWorkflow):
    """
    Workflow for preprocessing T1w (and optionally T2w) images.
    """
    def _initialize_steps(self):
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
        if anat_cfg.get("reorient", {}).get("enabled", True): # Default True?
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
        if mask_cfg.get("enabled", True):
             self.add_step(BrainMaskingStep(self.config, self.logger, self.provenance, method=mask_cfg.get("method", "ants")))

        # 7. Recon-all
        recon_cfg = anat_cfg.get("recon_all", {})
        if recon_cfg.get("enabled"):
             self.add_step(ReconAllStep(self.config, self.logger, self.provenance))
             
        # 8. Nonlinear Registration
        norm_cfg = anat_cfg.get("normalization", {})
        if norm_cfg.get("enabled"):
             # Template should be in config
             self.add_step(NonlinearRegistrationStep(self.config, self.logger, self.provenance, template=norm_cfg.get("template")))


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
             reporter.add_section("Inputs", f"Subject: {context.get('subject')}, Session: {context.get('session')}")
             p = output_dir / "report_input_t1w.png"
             try:
                 create_ortho_view(t1.img, p, title="Input T1w")
                 reporter.add_figure("Input T1w", p, caption=t1.img.name)
             except Exception as e:
                 self.logger.warning(f"Failed to plot input T1w: {e}")

        # Pre-calc counts for progress bar
        # T1 steps: Steps that are NOT Coreg/BrainMask/Recon/NonlinReg?
        # In current logic: Coreg/BrainMask skipped in T1 loop. Recon/NonlinReg kept?
        # The T1 loop explicitly skips Coreg, BrainMask.
        # It DOES execute ReconAll, NonlinReg? 
        # Wait, the logic is:
        # for step in self.steps: if is Coreg or BrainMask -> continue.
        # else run.
        # So T1 loop runs: Resample, Reorient, Denoise, Gibbs, Bias, Sharpen, ReconAll, NonlinReg.
        
        t1_runnable = [s for s in self.steps if not isinstance(s, (CoregistrationStep, BrainMaskingStep))]
        
        # T2 loop runs: Resample...Bias. Not Recon, Nonlin, Mask, Coreg.
        t2_runnable = [s for s in self.steps if not isinstance(s, (ReconAllStep, NonlinearRegistrationStep, BrainMaskingStep, CoregistrationStep))] if t2w_files else []
        
        # Coregistration: Runs if enabled and T1w & T2w exist.
        coreg_cfg_run = self.config.get("anat", {}).get("preprocessing", {}).get("coregistration", {})
        do_coreg = (t1w_files and t2w_files and coreg_cfg_run.get("enabled", True))
        
        # Brain Masking: Runs if step exists in self.steps
        mask_step = next((s for s in self.steps if isinstance(s, BrainMaskingStep)), None)
        do_mask = (mask_step is not None)
        
        total_steps = len(t1_runnable) + len(t2_runnable)
        if do_coreg: total_steps += 1
        if do_mask: total_steps += 1
        
        def _execute_anat(progress_ctx=None, task_id=None):
            # Primary flow: Process T1w
            if t1w_files:
                 # Process T1w
                 current_t1 = t1w_files[0]
                 context["current_image"] = current_t1
                 
                 processed_t1 = current_t1
                 
                 for step in self.steps:
                      if isinstance(step, (CoregistrationStep, BrainMaskingStep)):
                           # Handle Coregistration and Brain Masking separately
                           continue
                      
                      step_name = step.__class__.__name__
                      if progress_ctx:
                          progress_ctx.update(task_id, description=f"[cyan]T1w: {step_name}")
                      else:
                          self.logger.info(f"Running T1w step: {step_name}")

                      # ReconAllStep might depend on T1w specifically
                      prev_t1 = processed_t1
                      st = time.time()
                      processed_t1 = step.run(processed_t1, output_dir=output_dir)
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
                      
                      # Reporting: T1w
                      if reporter:
                           if isinstance(step, DenoisingStep):
                                p = figures_dir / "t1w_denoised.png"
                                create_ortho_view(processed_t1.img, p, title="Denoised T1w")
                                reporter.add_figure("T1w Denoising", p, caption="Denoised T1w")
                           elif isinstance(step, GibbsUnringingStep):
                                p = figures_dir / "t1w_gibbs.png"
                                create_ortho_view(processed_t1.img, p, title="Gibbs Corrected T1w")
                                reporter.add_figure("T1w Gibbs", p, caption="Gibbs Corrected T1w")
                                
                      if progress_ctx: progress_ctx.advance(task_id)

                 
                 context["preprocessed_t1w"] = processed_t1
                 self.logger.info(f"T1w processing complete: {processed_t1.img}")

            # Process T2w if exists
            if t2w_files:
                 current_t2 = t2w_files[0]
                 processed_t2 = current_t2
                 
                 for step in self.steps:
                      if isinstance(step, (ReconAllStep, NonlinearRegistrationStep, BrainMaskingStep, CoregistrationStep)):
                           continue
                      
                      step_name = step.__class__.__name__
                      if progress_ctx:
                          progress_ctx.update(task_id, description=f"[cyan]T2w: {step_name}")
                      else:
                          self.logger.info(f"Running T2w step: {step_name}")
                      
                      st = time.time()
                      processed_t2 = step.run(processed_t2, output_dir=output_dir)
                      dur = time.time() - st
                      
                      step_metrics.append({
                          "Step": f"T2w_{step_name}",
                          "Status": "Completed",
                          "Duration": f"{dur:.2f}s"
                      })

                      if isinstance(processed_t2, dict):
                            processed_t2 = processed_t2.get("current_image", processed_t2)
                      
                      # Reporting: T2w
                      if reporter:
                           if isinstance(step, DenoisingStep):
                                p = figures_dir / "t2w_denoised.png"
                                create_ortho_view(processed_t2.img, p, title="Denoised T2w")
                                reporter.add_figure("T2w Denoising", p, caption="Denoised T2w")
                                
                      if progress_ctx: progress_ctx.advance(task_id)

                 context["preprocessed_t2w"] = processed_t2
                 self.logger.info(f"T2w processing complete: {processed_t2.img}")
                 
                 # Coregister only if T1w exists and T1w processing successful
                 if t1w_files and "preprocessed_t1w" in context:
                      processed_t1 = context["preprocessed_t1w"]
                      
                      if do_coreg:
                           if progress_ctx: progress_ctx.update(task_id, description="[cyan]Coregistration")
                           
                           coreg_step = CoregistrationStep(self.config, self.logger, self.provenance, method=coreg_cfg_run.get("method", "fsl"))
                           
                           ref_img = coreg_cfg_run.get("reference_image", "t1w").lower()
                           coreg_options = coreg_cfg_run.get("options", {})
                           
                           st = time.time()
                           if ref_img == 't2w':
                               # Reference is T2w. Moving is T1w.
                               self.logger.info("Coregistration: Reference=T2w. Registering T1w -> T2w.")
                               res_t1 = coreg_step.run(processed_t1, output_dir=output_dir, target=processed_t2.img, options=coreg_options)
                               if isinstance(res_t1, dict): 
                                    res_t1 = res_t1.get("current_image")
                               
                               # Update T1w in context to be the coregistered one
                               context["preprocessed_t1w"] = res_t1
                               
                               # Plot: T1w aligned on T2w
                               if reporter:
                                    p = figures_dir / "coreg_t1_on_t2.png"
                                    plot_comparison(processed_t2.img, res_t1.img, p, title="Coreg T1w -> T2w")
                                    reporter.add_figure("Coregistration (T1->T2)", p, caption="T1w (overlay) aligned to T2w")

                           else:
                               # Reference is T1w (Default). Moving is T2w.
                               self.logger.info("Coregistration: Reference=T1w. Registering T2w -> T1w.")
                               res_t2 = coreg_step.run(processed_t2, output_dir=output_dir, target=processed_t1.img, options=coreg_options)
                               if isinstance(res_t2, dict):
                                    res_t2 = res_t2.get("current_image")
                               
                               context["preprocessed_t2w_coreg"] = res_t2
                               
                               # Plot: T2w aligned on T1w
                               if reporter:
                                    p = figures_dir / "coreg_t2_on_t1.png"
                                    plot_comparison(processed_t1.img, res_t2.img, p, title="Coreg T2w -> T1w")
                                    reporter.add_figure("Coregistration (T2->T1)", p, caption="T2w (overlay) aligned to T1w")
                           
                           dur = time.time() - st
                           step_metrics.append({"Step": "Coregistration", "Status": "Completed", "Duration": f"{dur:.2f}s"})
                           
                           if progress_ctx: progress_ctx.advance(task_id)

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
                    
                    st = time.time()
                    try:
                        brain_masked, mask = mask_step.run(target_img, output_dir=output_dir, return_mask=True)
                        processed_masks.append(mask)
                        dur = time.time() - st
                        step_metrics.append({"Step": "BrainMasking", "Status": "Completed", "Duration": f"{dur:.2f}s"})
                        
                        # Add mask to context
                        context["brain_mask"] = mask

                        if reporter:
                             p = figures_dir / "brain_mask_check.png"
                             # Overlay mask on target
                             plot_comparison(target_img.img, mask.img, p, title="Brain Mask Check", overlay_alpha=0.4)
                             reporter.add_figure("Brain Masking", p, caption="Brain Mask (overlay) on Reference")
                             
                    except Exception as e:
                         self.logger.warning(f"Brain Masking failed: {e}")
                         step_metrics.append({"Step": "BrainMasking", "Status": "Failed", "Duration": "0s"})
                else:
                    self.logger.warning("No reference image found for brain masking.")
                
                if progress_ctx: progress_ctx.advance(task_id)

            if reporter:
                 reporter.add_summary_table("Anatomical Execution Summary", step_metrics)

            # Save results if final_output_dir provided
            if final_output_dir:
                self.save_results(context, final_output_dir)

            return context
        
        # Execution Wrapper
        if Progress and total_steps > 0:
             with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeRemainingColumn(),
                transient=True 
             ) as progress:
                 task = progress.add_task(f"[cyan]Starting Anatomical Preprocessing...", total=total_steps)
                 return _execute_anat(progress_ctx=progress, task_id=task)
        else:
             return _execute_anat()

    def _update_json_history(self, json_path: Path, steps: list):
        """Update JSON sidecar with processing history."""
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

    def save_results(self, context, output_dir: Path):
         """Save final anatomical outputs to output_dir."""
         import shutil
         # Ensure output dir exists
         output_dir.mkdir(parents=True, exist_ok=True)
         
         # Copy T1w
         pre_t1 = context.get("preprocessed_t1w")
         if pre_t1 and hasattr(pre_t1, 'img') and pre_t1.img.exists():
               # Build name with desc='preproc'
               from ...io.bids import build_bids_name
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
        report_title = f"Anatomical Pipeline Report: sub-{subject} {ses}"
        
        # Save report one level up? Or in anat dir?
        # User requested reports one dir up for dMRI pipeline.
        # Let's apply consistent logic: session root.
        final_anat_dir.mkdir(parents=True, exist_ok=True)
        reporter = ReportGenerator(final_anat_dir.parent, title=report_title)
        
        try:
             # run returns results dict
             results = self.preprocessing.run(subj_work_dir, context, final_output_dir=final_anat_dir, reporter=reporter)
             
             # Generate Report
             reporter.generate()
             try:
                 reporter.generate_pdf()
             except Exception:
                 pass
             
        except Exception as e:
             self.logger.error(f"Error processing sub-{subject}: {e}")
             if self.config.stop_on_error: raise e
