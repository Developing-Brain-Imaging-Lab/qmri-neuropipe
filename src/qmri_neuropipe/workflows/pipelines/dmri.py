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
from qmri_neuropipe.lib.dmri.merge import MergeStep

from qmri_neuropipe.lib.common.mask import BrainMaskingStep
from .anat import AnatPreprocessingWorkflow
from qmri_neuropipe.lib.common.resample import ResampleStep
from qmri_neuropipe.interfaces.mriqc import run_mriqc
from qmri_neuropipe.lib.reporting.report import ReportGenerator
from qmri_neuropipe.lib.dmri.outliers import OutlierRemovalStep
from qmri_neuropipe.lib.dmri.qc import EddyQuadStep
from qmri_neuropipe.lib.dmri.motion import NiiFreezeStep
from ...lib.dmri.fitting import DTIFittingStep, DKIFittingStep, NODDIFittingStep, SANDIFittingStep, MAPMRIFittingStep, CSDFittingStep, FWDTIFittingStep
from ...lib.dmri.tractography import TractSegStep, PyAFQStep
from ...lib.dmri.analysis import AtlasRegistrationStep, StatsExtractionStep
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
        self.steps = []  # Reset steps
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

        
        # 1.5 Merging (Critical for Topup/Eddy)
        # Should happen after Topup (to calculate field on raw) but before Denoise? 
        # User requested: After Topup, so Denoise/Gibbs/Eddy run on merged.
        # Check config or default to enabled if multiple input files?
        # Let's assume enabled unless disabled.
        merge_cfg = dmri_cfg.get('merging', {})
        do_merge = merge_cfg.get('enabled', True) # Defaulting to True for now if multiple files?
        # Logic: If >1 file, and we are doing Topup, we likely want to merge.
        if len(dwi_files) > 1 and context.get('do_topup', False):
             do_merge = True
             
        if do_merge and len(dwi_files) > 1:
             self.logger.info("Adding MergeStep (concatenating DWI files for unified processing)...")
             self.add_step(MergeStep(self.config, self.logger, self.provenance))

        # 2. Denoising
        denoise_cfg = dmri_cfg.get('denoising', {})
        if denoise_cfg.get('enabled', True):
            method = denoise_cfg.get('method', 'mrtrix')
            params = denoise_cfg.get('parameters', {})
            
            self.logger.info(f"Adding DenoisingStep (method={method})")
            self.add_step(DenoisingStep(
                config=self.config, 
                logger=self.logger, 
                provenance=self.provenance,
                method=method,
                patch_radius=params.get('patch_radius') or denoise_cfg.get('patch_radius', 2),
                block_radius=params.get('block_radius') or denoise_cfg.get('block_radius', 5),
                pca_method=params.get('pca_method') or denoise_cfg.get('pca_method', 'eig')
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
            
        # 4. Motion / Eddy Current Correction Strategy
        # New config priority: dmri.preprocessing.motion_correction
        # Fallback to legacy: dmri.preprocessing.eddy
        
        motion_cfg = dmri_cfg.get('motion_correction', {})
        legacy_eddy_cfg = dmri_cfg.get('eddy', {})
        
        # Determine method: 'eddy', 'niifreeze', 'none'
        motion_method = motion_cfg.get('method')
        
        # Backward compatibility: if not set, check legacy eddy enabled
        if not motion_method:
             if legacy_eddy_cfg.get('enabled', True):
                 motion_method = 'eddy'
             else:
                 motion_method = 'none'

        run_eddy = False 
        
        if motion_method == 'eddy':
             run_eddy = True
             method = legacy_eddy_cfg.get('method', 'eddy')
             self.logger.info(f"Adding EddyCorrectionStep (method={method})")
             self.add_step(EddyCorrectionStep(
                config=self.config, 
                logger=self.logger, 
                provenance=self.provenance,
                method=method
            ))
            
             # 4.6 Eddy QC (Quad) - Automatic if eddy is run (FSL only)
             if method == 'eddy':
                  self.logger.info("Adding EddyQuadStep (Automatic with Eddy)")
                  self.add_step(EddyQuadStep(
                      config=self.config,
                      logger=self.logger,
                      provenance=self.provenance
                  ))
                  
        elif motion_method == 'niifreeze':
             self.logger.info("Adding NiiFreezeStep (motion correction)")
             self.add_step(NiiFreezeStep(
                 config=self.config
                 # Config might need merge or passed directly? 
                 # BaseProcessingStep uses self.config. 
                 # We can pass specific options via config dict if needed, 
                 # but usually config object has everything.
             ))
             
        elif motion_method == 'none':
             self.logger.info("Motion correction disabled.")
        else:
             self.logger.warning(f"Unknown motion correction method '{motion_method}'. Skipping.")

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

        
        # 6. Coregistration (to T1w)
        coreg_cfg = dmri_cfg.get('coregistration', {})
        do_coreg = coreg_cfg.get('enabled') or self.config.get("do_coregistration", False)
        
        if coreg_cfg.get('enabled') is False:
             do_coreg = False
        
        # We track if coregistration changed the resolution/grid
        coreg_resampled = False
        native_dwi_ref = None # To store native reference for GNL if needed
        
        if do_coreg:
            method = coreg_cfg.get('method') or self.config.get("coreg_method", "ants")
            
            # Check output resolution
            coreg_opts = coreg_cfg.get("options", {})
            out_res = coreg_opts.get("output_resolution", coreg_cfg.get("output_resolution", "anatomical")).lower()
            
            # If resolution is 'anatomical', the output dwi is upsampled/resampled.
            # If 'native' or 'dwi', it stays in native grid (structural is downsampled).
            
            if out_res == "anatomical":
                coreg_resampled = True
            
            self.logger.info(f"Adding CoregistrationStep (method={method})")
            # We add logic to capture the native image before coreg replaces it in context?
            # Actually, CoregStep replaces 'current_image' in context with the registered one.
            # So if we want the native one for GNL, we must ensure it's available.
            # But wait, step instantiation is just setup. Execution is later.
            # We need to handle this data flow during EXECUTION (inside _execute_processing or steps).
            # But the Steps are classes. 
            
            # The steps modify 'context'.
            # CoregistrationStep updates 'current_image'.
            # GNL needs 'current_image' (which is now registered) AND potentially 'native_image'.
            # So CoregistrationStep should probably save 'native_image' in context if it changes resolution?
            # OR we ensure GNL step knows how to find it.
            
            self.add_step(CoregistrationStep(
                config=self.config, 
                logger=self.logger, 
                provenance=self.provenance,
                method=method
            ))

        # 7. Gradient Nonlinearity Correction (Tortoise)
        # MOVED after Coregistration per user request.
        gnl_cfg = dmri_cfg.get('grad_nonlin', {})
        if gnl_cfg.get('enabled', False):
             self.logger.info(f"Adding TortoiseGradNonlinCorrectStep")
             self.add_step(TortoiseGradNonlinCorrectStep(
                 config=self.config,
                 logger=self.logger,
                 provenance=self.provenance,
                 is_resampled=coreg_resampled # Pass flag? 
                 # Wait, 'coreg_resampled' determined at init time is based on config.
                 # This is correct since pipeline structure is static.
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
        Execute the workflow on a list of DWI files (Unified Global/Per-Image).
        """
        dwi_files: list[DWIFile] = context.get("dwi_files", [])
        self.logger.info(f"Starting PreprocessingWorkflow for {len(dwi_files)} files.")
        
        # Prepare context (copy)
        context = dict(context)
        
        # Reporting: Inputs
        if reporter and dwi_files:
            reporter.set_dmri_input_summary(f"DWI Files: {len(dwi_files)}")
            dwi = dwi_files[0]
            p = output_dir / "report_input_dwi_b0.png"
            try:
                 from qmri_neuropipe.lib.reporting.viz import create_ortho_view
                 create_ortho_view(dwi.img, p, title="Input DWI (first volume)")
                 reporter.add_dmri_input_figure(p, caption=dwi.img.name)
            except Exception as e:
                 self.logger.warning(f"Failed to plot input DWI: {e}")

        # Execution Wrapper with Rich
        try:
            from rich.console import Console
            from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
            console = Console()
        except ImportError:
            console = None
            Progress = None
            
        # Helper for execution
        def _execute_processing(progress_ctx=None, task_id=None):
            nonlocal context
            nonlocal dwi_files
            
            # Identify Global Steps Types
            GLOBAL_STEPS = (Synb0EstimationStep, TopupStep, GradientCheckStep, DMRIReorientStep, MergeStep)
            
            current_dwis = context.get("dwi_files", [])
            current_masks = context.get("masks", [None] * len(current_dwis))
            # If masks missing or short, pad
            if len(current_masks) < len(current_dwis):
                current_masks.extend([None] * (len(current_dwis) - len(current_masks)))

            for step_idx, step in enumerate(self.steps):
                step_name = step.__class__.__name__
                
                if isinstance(step, GLOBAL_STEPS):
                    # --- Global Step ---
                    if progress_ctx:
                         progress_ctx.update(task_id, description=f"[cyan]Global: {step_name}")
                    
                    self.logger.info(f"Executing Global Step: {step_name}...")
                    
                    # Update context with current state of lists
                    context['dwi_files'] = current_dwis
                    
                    old_dwis = list(current_dwis)
                    
                    # Run Global Step
                    force_run = self.config.get("dmri", {}).get("force_run", False)
                    new_ctx = step.run(context, output_dir=output_dir, force=force_run)
                    
                    if new_ctx is not context:
                        context.update(new_ctx)
                        
                    # Update current pointers
                    current_dwis = context.get("dwi_files", [])

                    # If dwi changed, refresh topup groups
                    if current_dwis != old_dwis:
                        from qmri_neuropipe.io.dmri.bids import find_reversed_phase_groups
                        context["topup_groups"] = find_reversed_phase_groups(current_dwis)
                        # Reset masks if sizes mismatch
                        if len(current_masks) != len(current_dwis):
                             current_masks = [None] * len(current_dwis)
                    
                     # Save intermediate
                    save_inter = self.config.get("save_intermediates", False)
                    if save_inter:
                         self._save_global_intermediate(step, output_dir, context)
                         
                    # Report
                    if reporter:
                         figures_dir = output_dir / "figures"
                         figures_dir.mkdir(exist_ok=True, parents=True)
                         self._report_step(reporter, step, None, None, context, None, figures_dir)
                         
                    if progress_ctx: progress_ctx.advance(task_id)

                else:
                    # --- Per-Image Step ---
                    new_dwis = []
                    new_masks = []
                    topup_map = context.get("topup_map", {})
                    
                    for i, (dwi, mask) in enumerate(zip(current_dwis, current_masks)):
                         if progress_ctx:
                              progress_ctx.update(task_id, description=f"[cyan]Step: {step_name} ({i+1}/{len(current_dwis)})")
                         
                         # Prepare per-image context
                         img_ctx = dict(context)
                         img_ctx['current_image'] = dwi
                         img_ctx['current_mask'] = mask
                         if dwi.img in topup_map:
                             img_ctx["topup_base"] = topup_map[dwi.img]

                         # T1w target logic
                         t1w_files = img_ctx.get("t1w_files", [])
                         target_img = t1w_files[0].img if t1w_files else None
                         
                         # Step kwargs
                         step_kwargs = {}
                         if isinstance(step, CoregistrationStep):
                             if target_img:
                                 step_kwargs["target"] = target_img
                                 coreg_cfg = self.config.get('dmri', {}).get('preprocessing', {}).get('coregistration', {})
                                 flat_opts = dict(coreg_cfg)
                                 if "options" in flat_opts: flat_opts.update(flat_opts.pop("options"))
                                 step_kwargs["options"] = flat_opts
                             else:
                                 # Skip
                                 new_dwis.append(dwi)
                                 new_masks.append(mask)
                                 continue
                         
                         if isinstance(step, BrainMaskingStep):
                             step_kwargs["return_mask"] = True
                             
                             # Optimization: Use structural mask if coregistered to T1w space
                             coreg_cfg = self.config.get('dmri', {}).get('preprocessing', {}).get('coregistration', {})
                             coreg_enabled = coreg_cfg.get('enabled', False) or self.config.get("do_coregistration", False)
                             
                             # Check output resolution (default anatomical)
                             coreg_opts = coreg_cfg.get("options", {}) if isinstance(coreg_cfg.get("options"), dict) else {}
                             out_res = coreg_opts.get("output_resolution", coreg_cfg.get("output_resolution", "anatomical")).lower()
                             
                             if coreg_enabled and out_res == "anatomical" and t1w_files:
                                 # We are in T1w space. Try to find the T1w mask.
                                 # t1w_files[0] should be the structural reference.
                                 t1_ref = t1w_files[0]
                                 if hasattr(t1_ref, 'img'):
                                      t1_path = t1_ref.img
                                      # Patterns to check:
                                      # 1. _desc-brain_mask.nii.gz (Anat workflow typical)
                                      # 2. _desc-preproc_mask.nii.gz (Another variant)
                                      # 3. _mask.nii.gz (Generic)
                                      
                                      # Assuming t1_path is ..._desc-preproc_T1w.nii.gz
                                      # Try replacing desc-preproc_T1w with desc-brain_mask
                                      
                                      parent = t1_path.parent
                                      # We use entities if available for robust search
                                      if hasattr(t1_ref, 'entities'):
                                           m_ents = t1_ref.entities.copy()
                                           m_ents['suffix'] = 'mask'
                                           
                                           # Variant A: desc-brain
                                           m_ents['desc'] = 'brain'
                                           from qmri_neuropipe.io.bids import build_bids_name
                                           c_path = parent / build_bids_name(m_ents)
                                           
                                           # Variant B: desc-preproc
                                           m_ents['desc'] = 'preproc'
                                           c_path_2 = parent / build_bids_name(m_ents)
                                           
                                           if c_path.exists():
                                               step_kwargs["structural_mask"] = c_path
                                               self.logger.info(f"optimization: Finding structural mask to avoid re-computation: {c_path}")
                                           elif c_path_2.exists():
                                               step_kwargs["structural_mask"] = c_path_2
                                               self.logger.info(f"optimization: Finding structural mask to avoid re-computation: {c_path_2}")
                                 
                             
                         # Run Step
                         try:
                             from time import time as now
                             st = now()
                             force_run = self.config.get("dmri", {}).get("force_run", False)
                             result = step.run(img_ctx, output_dir=output_dir, force=force_run, **step_kwargs)
                             dur = now() - st
                             
                             # Extract output
                             out_dwi = result.get("current_image") if isinstance(result, dict) else result
                             out_mask = result.get("current_mask") if isinstance(result, dict) else None
                             
                             new_dwis.append(out_dwi)
                             new_masks.append(out_mask if out_mask is not None else mask)
                             
                             # Accumulate QC Metrics for Export
                             # We create a record for this image
                             if "qc_metrics" in result or "outlier_stats" in result:
                                  record = {
                                      "subject": context.get("subject"),
                                      "session": context.get("session"),
                                      "file_name": out_dwi.img.name if hasattr(out_dwi, 'img') else "unknown"
                                  }
                                  if "qc_metrics" in result:
                                      record.update(result["qc_metrics"])
                                  if "outlier_stats" in result:
                                      record["outliers"] = result["outlier_stats"]
                                  
                                  # Add to global list in context (requires list init outside loop)
                                  if "all_qc_metrics" not in context:
                                       context["all_qc_metrics"] = []
                                  
                                  # Avoid duplication if we loop multiple steps? 
                                  # We should probably upsert based on filename?
                                  # Or just append separate records for separate steps?
                                  # Ideally we want one row per subject/image with all metrics.
                                  # Since we loop steps, we build up the record?
                                  # `new_dwis` replaces `current_dwis` each step.
                                  # This structure makes it hard to aggregate ACROSS steps for the same image easily 
                                  # without a persistent object tracking metadata.
                                  
                                  # Alternative: Use a sidecar dictionary in context mapping filename -> metrics
                                  if "qc_registry" not in context: context["qc_registry"] = {}
                                  
                                  img_key = out_dwi.img.name
                                  if img_key not in context["qc_registry"]:
                                       context["qc_registry"][img_key] = record
                                  else:
                                       # Update existing
                                       record.pop("subject", None)
                                       record.pop("session", None)
                                       record.pop("file_name", None)
                                       context["qc_registry"][img_key].update(record)
                                  
                                  # Update the list view for final export
                                  context["all_qc_metrics"] = list(context["qc_registry"].values())
                             
                             # Collect GNL Map if present (for per-image step)
                             if isinstance(result, dict) and "gnl_map" in result:
                                 if "gnl_maps" not in context:
                                     context["gnl_maps"] = []
                                 gnl_path = result["gnl_map"]
                                 if gnl_path not in context["gnl_maps"]:
                                     context["gnl_maps"].append(gnl_path)
                             
                             # Reporting
                             if reporter:
                                 figures_dir = output_dir / "figures"
                                 figures_dir.mkdir(exist_ok=True, parents=True)
                                 self._report_step(reporter, step, dwi, dwi, result, target_img, figures_dir, step_kwargs)
                                 # Add duration to summary
                                 reporter.add_dmri_summary("Execution Summary", [{"Step": step_name, "Status": "Completed", "Duration": f"{dur:.2f}s"}])
                             
                             # Save intermediate
                             if self.config.get("save_intermediates", False):
                                 self._save_image_intermediate(output_dir, context, result)
                                 
                         except Exception as e:
                             if self.config.get("stop_on_error", False): raise e
                             self.logger.error(f"Step {step_name} failed on {dwi.img.name}: {e}")
                             new_dwis.append(dwi)
                             new_masks.append(mask)

                    # Update lists for next step
                    current_dwis = new_dwis
                    current_masks = new_masks
                    
                    if progress_ctx: progress_ctx.advance(task_id)

            # Finalize
            context["preprocessed_dwis"] = current_dwis
            context["preprocessed_masks"] = current_masks
            if current_dwis:
                context["current_image"] = current_dwis[-1] # Fallback for single-image consumers
            return context

        # Run logic
        calc_total = 0
        GLOBAL_STEPS = (Synb0EstimationStep, TopupStep, GradientCheckStep, DMRIReorientStep)
        for s in self.steps:
             if isinstance(s, GLOBAL_STEPS): calc_total += 1
             else: calc_total += len(dwi_files) 
             
        if Progress:
             with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeRemainingColumn(),
                transient=True,
                console=console
             ) as progress:
                 task = progress.add_task(f"[cyan]Preprocessing...", total=calc_total)
                 context = _execute_processing(progress_ctx=progress, task_id=task)
        else:
             context = _execute_processing()
             
        # Save Final Outputs
        self._save_final_outputs(context)
        
        # Save QC Metrics CSV
        try:
             from qmri_neuropipe.lib.reporting.export import save_qc_metrics_csv
             
             # Collect metrics from context
             # Helper to extract metrics for all processed files
             qc_list = []
             preproc_dwis = context.get("preprocessed_dwis", [])
             
             # We might have stored metrics in context slightly differently per step overrides...
             # But steps modify the 'context' copy within _execute_processing, and we return the final context.
             # Actually, _execute_processing updates 'preprocessed_dwis' but doesn't explicitly store 
             # the per-image context with metrics in the main context list.
             # Wait, the `_execute_processing` has `new_dwis` and `new_masks` lists.
             # We lost the per-image metadata like 'qc_metrics' unless we attached it to the DWIFile object?
             # DWIFile dataclass has 'json' and 'entities'. It doesn't have a specific 'metrics' field.
             
             # We need to look at where we can grab these metrics.
             # `_report_step` accesses `current_arg` which had the metrics.
             # But `_execute_processing` loop discards `result` dictionary after reporting/saving intermediate. 
             # It only appends `out_dwi` to `new_dwis`.
             
             # FIX: We need to accumulate metrics in `_execute_processing`.
             # I will modify `_execute_processing` below to store metrics in context["all_qc_metrics"].
             
             all_qc = context.get("all_qc_metrics", [])
             if all_qc:
                 qc_csv = self.config.output_dir / "qc" / "group_qc_metrics.csv"
                 self.logger.info(f"Exporting QC metrics to {qc_csv}")
                 save_qc_metrics_csv(all_qc, qc_csv)
                 
        except ImportError:
             self.logger.warning("Could not import export module. QC CSV not saved.")
        except Exception as e:
             self.logger.warning(f"Failed to save QC CSV: {e}")

        return context

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
                  target_step_dir = inter_dir / src_dir.name
                  shutil.copytree(src_dir, target_step_dir, dirs_exist_ok=True)
                  shutil.copytree(src_dir, target_step_dir, dirs_exist_ok=True)
                  self.logger.info(f"Saved intermediate directory: {target_step_dir}")

    def _save_final_outputs(self, context: dict):
        """
        Save final preprocessed DWI files to output directory.
        Target: <output_dir>/sub-XX/[ses-YY]/dwi/sub-XX_[ses-YY]_desc-preproc_dwi.nii.gz
        """
        import shutil
        from qmri_neuropipe.io.bids import build_bids_name
        
        dwis = context.get("preprocessed_dwis", [])
        if not dwis:
            return

        self.logger.info(f"Saving {len(dwis)} preprocessed DWI files to final output directory.")

        base_out = self.config.output_dir
        
        for dwi in dwis:
            if not dwi.img.exists():
                self.logger.warning(f"Final DWI missing: {dwi.img}")
                continue

            ents = dwi.entities.copy()
            sub = ents.get("sub")
            ses = ents.get("ses")
            
            if not sub:
                 # Fallback
                 sub = context.get("subject", "unknown")
                 ents['sub'] = sub
            if ses:
                 ents['ses'] = ses
                 
            # Force valid suffix
            ents['suffix'] = 'dwi'
            ents['desc'] = 'preproc'
            # Remove processing-specific entities if any (like run-1_split-1) if we re-merged?
            # Assuming current entities reflect final state.
            
            # Construct Directory
            target_dir = base_out / f"sub-{sub}"
            if ses: target_dir /= f"ses-{ses}"
            target_dir /= "dwi"
            target_dir.mkdir(parents=True, exist_ok=True)
            
            # Construct Filenames
            fname = build_bids_name(ents)
            if not fname.endswith(".nii.gz"): fname += ".nii.gz"
            
            target_img = target_dir / fname
            target_bval = target_img.with_suffix("").with_suffix(".bval")
            target_bvec = target_img.with_suffix("").with_suffix(".bvec")
            target_json = target_img.with_suffix("").with_suffix(".json")
            
            # Copy Files
            self.logger.info(f"Saving final output: {target_img}")
            shutil.copy(dwi.img, target_img)
            
            if dwi.bval and dwi.bval.exists():
                shutil.copy(dwi.bval, target_bval)
            if dwi.bvec and dwi.bvec.exists():
                shutil.copy(dwi.bvec, target_bvec)
            if dwi.json and dwi.json.exists():
                shutil.copy(dwi.json, target_json)

        # Save GNL Maps if present (handling per-image saving)
        gnl_maps = context.get("gnl_maps", [])
        # Fallback for old single-map context
        if not gnl_maps and context.get("gnl_map"):
             gnl_maps = [context.get("gnl_map")]
             
        for gnl_map in gnl_maps:
            if gnl_map and isinstance(gnl_map, Path) and gnl_map.exists():
                 # We try to infer the target directory based on entities in the filename
                 # or fallback to the general session folder.
                 
                 # Re-parse entities from GNL map name
                 # It should have sub-XX_ses-YY_...
                 from qmri_neuropipe.io.bids import get_entities_from_path
                 g_ents = get_entities_from_path(gnl_map)
                 
                 sub_g = g_ents.get("sub") or context.get("subject", "unknown")
                 ses_g = g_ents.get("ses")
                 
                 t_dir = base_out / f"sub-{sub_g}"
                 if ses_g: t_dir /= f"ses-{ses_g}"
                 t_dir /= "dwi"
                 t_dir.mkdir(parents=True, exist_ok=True)
                 
                 target_gnl = t_dir / gnl_map.name
                 
                 # Avoid overwriting if existing? Should be fine to overwrite final output.
                 self.logger.info(f"Saving GNL Tensor Map: {target_gnl}")
                 shutil.copy(gnl_map, target_gnl)

    def recover_intermediates(self, work_dir: Path, output_dir: Path):
        """
        Recover intermediate data from the final output directory back to the working directory.
        This allows the pipeline to skip steps that were previously computed and saved.
        """
        import shutil
        
        # Define what intermediates we expect to recover
        # Step classes save to specific subfolders in intermediate/
        # e.g. topup/ -> intermediate/topup/
        # synb0/ -> intermediate/synb0/
        # eddy/ -> intermediate/eddy/ (not usually saved as intermediate, but could be)
        
        self.logger.info("Attempting to recover intermediate data...")
        
        # Construct paths
        # output_dir (final_dwi_dir) / intermediate
        intermediate_store = output_dir / "intermediate"
        
        if not intermediate_store.exists():
            self.logger.debug(f"No intermediate storage found at {intermediate_store}")
            return
            
        # 1. Topup
        if (intermediate_store / "topup").exists():
            target_work = work_dir / "topup"
            if not target_work.exists():
                self.logger.info(f"Recovering Topup results from {intermediate_store / 'topup'}")
                shutil.copytree(intermediate_store / "topup", target_work, dirs_exist_ok=True)
                
        # 2. Synb0
        if (intermediate_store / "synb0").exists():
            target_work = work_dir / "synb0"
            if not target_work.exists():
                self.logger.info(f"Recovering Synb0 results from {intermediate_store / 'synb0'}")
                shutil.copytree(intermediate_store / "synb0", target_work, dirs_exist_ok=True)

        # 3. Eddy? or Gradient Nonlinearity?
        # Add more as needed based on what _save_global_intermediate saves.


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
                         details["PCA Method"] = step.pca_method
                         
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
             
             # Check for metrics to report (e.g. from EddyQuadStep or OutlierRemoval)
             if isinstance(current_arg, dict) and ('qc_metrics' in current_arg or 'outlier_stats' in current_arg):
                 self._report_metrics_summary(reporter, dwi, current_arg)
                     
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
        # Reset steps to prevent accumulation across subjects
        self.steps = []
        
        modeling_cfg = self.config.get('dmri', {}).get('modeling', {})
        
        # 1. DTI
        # Dependency Check: Auto-enable CSD if TractSeg is requested
        # We must do this BEFORE processing CSD step.
        tract_cfg_chk = modeling_cfg.get('tractography', {})
        # Support fallback source for tract_cfg (dmri root)
        if not tract_cfg_chk:
             tract_cfg_chk = self.config.get('dmri', {}).get('tractography', {})
             
        if tract_cfg_chk.get('tractseg', {}).get('enabled', False):
             csd_chk = modeling_cfg.get('csd', {})
             if not csd_chk.get('enabled', False):
                 self.logger.info("TractSeg enabled: Auto-enabling CSD Fitting to ensure reuse/saving of FODs.")
                 modeling_cfg.setdefault('csd', {})['enabled'] = True
                 
        
        # 1. DTI
        dti_cfg = modeling_cfg.get('dti', {}) or modeling_cfg.get('tensor', {}) # Support 'tensor' for backward compatibility/legacy
        if dti_cfg.get('enabled', False):
            method = dti_cfg.get('method', 'dipy')
            self.logger.info(f"Adding DTIFittingStep (method={method})")
            # Prepare kwargs
            dti_kwargs = dict(dti_cfg)
            if 'parameters' in dti_kwargs: dti_kwargs.update(dti_kwargs.pop('parameters'))
            if 'options' in dti_kwargs: dti_kwargs.update(dti_kwargs.pop('options'))
            dti_kwargs.pop('enabled', None)
            dti_kwargs.pop('method', None)

            self.add_step(DTIFittingStep(
                config=self.config,
                logger=self.logger,
                provenance=self.provenance,
                method=method,
                n_cpus=self.config.n_cpus,
                **dti_kwargs
            ))
            
        # 2. DKI
        dki_cfg = modeling_cfg.get('dki', {})
        if dki_cfg.get('enabled', False):
            method = dki_cfg.get('method', 'dipy')
            self.logger.info(f"Adding DKIFittingStep (method={method})")
            # Prepare kwargs
            dki_kwargs = dict(dki_cfg)
            if 'parameters' in dki_kwargs: dki_kwargs.update(dki_kwargs.pop('parameters'))
            dki_kwargs.pop('enabled', None)
            dki_kwargs.pop('method', None)

            self.add_step(DKIFittingStep(
                config=self.config,
                logger=self.logger,
                provenance=self.provenance,
                method=method,
                n_cpus=self.config.n_cpus,
                **dki_kwargs
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
            # Prepare kwargs from config, prioritizing top-level keys but supporting 'parameters' too if used
            step_kwargs = noddi_cfg.copy()
            # Remove keys that are passed explicitly or not needed
            step_kwargs.pop('method', None) 
            step_kwargs.pop('enabled', None)
            
            # Merge nested 'parameters' if present (legacy support)
            if 'parameters' in step_kwargs:
                 nested_params = step_kwargs.pop('parameters')
                 if isinstance(nested_params, dict):
                     step_kwargs.update(nested_params)

            self.add_step(NODDIFittingStep(
                config=self.config,
                logger=self.logger,
                provenance=self.provenance,
                method=method,
                n_cpus=self.config.n_cpus,
                **step_kwargs
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
            # Prepare kwargs
            map_kwargs = dict(map_cfg)
            if 'parameters' in map_kwargs: map_kwargs.update(map_kwargs.pop('parameters'))
            map_kwargs.pop('enabled', None)
            map_kwargs.pop('method', None)

            self.add_step(MAPMRIFittingStep(
                config=self.config,
                logger=self.logger,
                provenance=self.provenance,
            method=method,
                n_cpus=self.config.n_cpus,
                **map_kwargs
            ))
            
            
        # 6. FWE-DTI (Free-Water Elimination DTI)
        fwe_dti_cfg = modeling_cfg.get('fwe_dti', {}) or modeling_cfg.get('fwdti', {})
        if fwe_dti_cfg.get('enabled', False):
            method = fwe_dti_cfg.get('method', 'dipy')
            self.logger.info(f"Adding FWDTIFittingStep (method={method})")
            
            # Combine options
            step_kwargs = fwe_dti_cfg.copy()
            step_kwargs.pop('method', None)
            step_kwargs.pop('enabled', None)
            if 'parameters' in step_kwargs:
                step_kwargs.update(step_kwargs.pop('parameters'))
            
            self.add_step(FWDTIFittingStep(
                config=self.config,
                logger=self.logger,
                provenance=self.provenance,
                method=method,
                n_cpus=self.config.n_cpus,
                **step_kwargs
            ))
            
            
        # 10. Tractography & Segmentation
        tract_cfg = modeling_cfg.get('tractography', {})
        # Note: In config, it might be under 'dmri.tractography' OR 'dmri.modeling.tractography'.
        # The user instruction was 'dmri.tractography'.
        # But here we are using 'modeling_cfg' which is 'dmri.modeling'.
        # We should check 'dmri.tractography' as well if strictly following user guide.
        # However, logically it makes sense in modeling workflow. 
        # Let's check 'dmri' root too if not found in modeling.
        
        dmri_cfg = self.config.get('dmri', {})
        if not tract_cfg:
             tract_cfg = dmri_cfg.get('tractography', {})
             
        # 10.1 TractSeg
        if tract_cfg.get('tractseg', {}).get('enabled', False):
             self.logger.info("Adding TractSegStep")
             self.add_step(TractSegStep(
                 config=self.config,
                 logger=self.logger,
                 provenance=self.provenance,
                 method='tractseg',
                 **tract_cfg.get('tractseg', {}).get('options', {})
             ))

        # 10.2 PyAFQ / BabyAFQ
        if tract_cfg.get('pyafq', {}).get('enabled', False):
             self.logger.info("Adding PyAFQStep")
             self.add_step(PyAFQStep(
                 config=self.config,
                 logger=self.logger,
                 provenance=self.provenance,
                 method='pyafq',
                 **tract_cfg.get('pyafq', {}).get('options', {})
             ))
             
        # 11. Analysis & Statistics
        analysis_cfg = dmri_cfg.get('analysis', {})
        # Also check inside modeling?
        if not analysis_cfg:
             analysis_cfg = modeling_cfg.get('analysis', {})
        
        # 11.1 Atlas Registration
        if analysis_cfg.get('atlases', {}):
             self.logger.info("Adding AtlasRegistrationStep")
             self.add_step(AtlasRegistrationStep(
                 config=self.config,
                 logger=self.logger,
                 provenance=self.provenance,
                 method='ants' # Default
             ))
             
        # 11.2 Statistics Extraction
        if analysis_cfg.get('full_stats', False):
             self.logger.info("Adding StatsExtractionStep")
             self.add_step(StatsExtractionStep(
                 config=self.config,
                 logger=self.logger,
                 provenance=self.provenance
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
                
                # Explicitly skip non-scalar 4D outputs that might match matching patterns accidentally
                if any(x in stem_lower for x in ['tensor', 'color_fa', 'evals', 'evecs']):
                    continue
                    
                # Strict Suffix Match: Ensure the filename ends with _<suffix> 
                # (e.g. _FA matches ..._FA.nii.gz, but not ..._somethingFA.nii.gz)
                target_suffix = f"_{key.lower()}"
                if not stem_lower.endswith(target_suffix): continue
                
                # Double check to ensure we haven't matched a substring of a longer word
                # e.g. "color_fa" ends with "_fa" is FALSE (it ends with "or_fa" -> "r_fa" -> "color_fa")
                # Wait, color_fa.nii.gz -> stem=color_fa. endswith("_fa")?
                # "color_fa"[-3:] is "_fa". So it DOES match. 
                # Ideally we want to ensure the part before the suffix is a meaningful separator or end of ID.
                # But since we explicitly excluded 'color_fa' above, strict endswith should be safe for 'fa', 'md', etc.

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
        self.steps = [] # Reset steps
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
        self.steps = [] # Reset steps
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
        t1w_files = self._find_anat_files(subject, session, 'T1w')
        t2w_files = self._find_anat_files(subject, session, 'T2w')
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
            
            # Ensure entities match the processing subject/session
            # This fixes issues where filenames on disk might differ from folder structure (e.g. missing letters in ID)
            current_entities = d.entities.copy()
            current_entities['sub'] = subject
            if session:
                current_entities['ses'] = session
            
            # Create new DWIFile pointing to work dir
            new_dwi = DWIFile(
                img=dest_img,
                json=dest_json,
                bval=dest_bval,
                bvec=dest_bvec,
                entities=current_entities
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
        
        # Attempt to recover intermediates BEFORE checking for final preproc skip
        # This helps if we resume a run that crashed halfway or if we are re-running with some steps enabled.
        # But if we skip the whole pipeline below, this recovery is moot but harmless.
        # However, calling it here allows steps to see files in work_dir immediately.
        
        dwi_final_dir = self._get_output_dir(subject, session) / 'dwi'
        if self.config.get("save_intermediates", False):
             self.preprocessing.recover_intermediates(subj_work_dir, dwi_final_dir)

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
        
        force_run = self.config.get("dmri", {}).get("force_run", False) or self.config.get("force", False)
        
        # Check if already processed (Skip unless forced)
        if expected_preproc_path.exists() and not force_run:
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
            
            # Recover GNL Map if it exists (for modeling)
            # We look for *desc-gnl_tensor* in the output directory
            gnl_candidates = list(output_dir.glob(f"sub-{subject}*_desc-gnl_tensor*.nii.gz"))
            if gnl_candidates:
                 # Pick the one matching session if present
                 match = gnl_candidates[0]
                 if session:
                      # prioritize one that has ses-{session}
                      ses_matches = [g for g in gnl_candidates if f"ses-{session}" in g.name]
                      if ses_matches: match = ses_matches[0]
                 
                 self.logger.info(f"Recovered GNL Tensor Map for modeling: {match.name}")
                 preprocessed_context['gnl_map'] = match
            
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
            
            # --- Save Final Preprocessed Output to Main Output Directory ---
            # The user explicitly requested to save "preproc_dwi" files in the output directory
            # before modeling/normalization begins.
            
            import shutil
            preproc_dwis = preprocessed_context.get("preprocessed_dwis", [])
            preproc_masks = preprocessed_context.get("preprocessed_masks", [])
            
            # We assume output structure: output_dir / sub-X / ses-Y / dwi
            # dwi_final_dir is already defined as: self._get_output_dir(subject, session) / 'dwi'
            # But let's verify
            dwi_final_dir.mkdir(parents=True, exist_ok=True)
            
            updated_dwis = []
            updated_masks = []
            
            for i, (pdwi, pmask) in enumerate(zip(preproc_dwis, preproc_masks)):
                if not pdwi.img.exists():
                    self.logger.warning(f"Preprocessed DWI does not exist: {pdwi.img}")
                    updated_dwis.append(pdwi)
                    updated_masks.append(pmask)
                    continue
                    
                # Construct new filename with _desc-preproc
                # Use original entities but override desc
                ents = pdwi.entities.copy()
                ents['desc'] = 'preproc'
                
                new_name_dwi = build_bids_name(ents, suffix='dwi')
                if not new_name_dwi.endswith('.nii.gz'): new_name_dwi += '.nii.gz'
                
                dest_dwi = dwi_final_dir / new_name_dwi
                
                self.logger.info(f"Saving final preprocessed DWI: {dest_dwi}")
                shutil.copy(pdwi.img, dest_dwi)
                
                # Copy sidecars
                new_bval_path = None
                if pdwi.bval and pdwi.bval.exists():
                    dest_path_str = str(dest_dwi)
                    for ext in ['.nii.gz', '.nii']:
                        if dest_path_str.endswith(ext):
                            dest_path_str = dest_path_str[:-len(ext)]
                            break
                    dest_bval = Path(dest_path_str + ".bval")
                    shutil.copy(pdwi.bval, dest_bval)
                    new_bval_path = dest_bval

                new_bvec_path = None
                if pdwi.bvec and pdwi.bvec.exists():
                     dest_path_str = str(dest_dwi)
                     for ext in ['.nii.gz', '.nii']:
                        if dest_path_str.endswith(ext):
                            dest_path_str = dest_path_str[:-len(ext)]
                            break
                     dest_bvec = Path(dest_path_str + ".bvec")
                     shutil.copy(pdwi.bvec, dest_bvec)
                     new_bvec_path = dest_bvec

                new_json_path = None
                if pdwi.json and pdwi.json.exists():
                     dest_path_str = str(dest_dwi)
                     for ext in ['.nii.gz', '.nii']:
                        if dest_path_str.endswith(ext):
                            dest_path_str = dest_path_str[:-len(ext)]
                            break
                     dest_json = Path(dest_path_str + ".json")
                     shutil.copy(pdwi.json, dest_json)
                     new_json_path = dest_json
                     
                # Save MASK
                new_mask_obj = pmask
                if pmask and hasattr(pmask, 'img') and pmask.img.exists():
                     mask_ents = ents.copy()
                     mask_ents['suffix'] = 'mask' # Ensure suffix is mask
                     if 'desc' in mask_ents: 
                         # Keep preproc desc for mask? Usually yes: desc-preproc_mask
                         pass
                         
                     new_name_mask = build_bids_name(mask_ents)
                     if not new_name_mask.endswith('.nii.gz'): new_name_mask += '.nii.gz'
                     
                     dest_mask = dwi_final_dir / new_name_mask
                     self.logger.info(f"Saving final preprocessed Mask: {dest_mask}")
                     shutil.copy(pmask.img, dest_mask)
                     new_mask_obj = ImageFile(entities=mask_ents, img=dest_mask)

                # Update context object to point to the SAVED file in output dir
                # This ensures downstream steps use the official output file
                updated_dwi_obj = DWIFile(
                    img=dest_dwi,
                    bval=new_bval_path,
                    bvec=new_bvec_path,
                    json=new_json_path,
                    entities=ents
                )
                updated_dwis.append(updated_dwi_obj)
                updated_masks.append(new_mask_obj)
            
            # --- Save GNL Map if present ---
            gnl_map = preprocessed_context.get("gnl_map")
            if gnl_map and isinstance(gnl_map, Path) and gnl_map.exists():
                 dest_gnl = dwi_final_dir / gnl_map.name
                 self.logger.info(f"Saving final GNL Tensor Map: {dest_gnl}")
                 shutil.copy(gnl_map, dest_gnl)
            # -------------------------------
                
            # Update context with the relocated files
            preprocessed_context['preprocessed_dwis'] = updated_dwis
            preprocessed_context['preprocessed_masks'] = updated_masks
        
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
            # Pass output_dir / "models" or configured models_dir with BIDS structure
            if self.config.get('models_dir'):
                 # Use relative path from standard output root (sub-X/ses-Y/dwi)
                 rel_path = output_dir.relative_to(self.config.output_dir)
                 models_out = Path(self.config.get('models_dir')) / rel_path
            else:
                 models_out = output_dir / "models"
                 
            models_out.mkdir(parents=True, exist_ok=True)
            preprocessed_context = self.modeling.run(subj_work_dir, preprocessed_context, reporter=reporter, final_output_dir=models_out)
        else:
            self.logger.info("No modeling steps configured.")

        # 6c. Run Normalization Workflow
        # Initialize and run using modeling results (in context)
        self.normalization.build_pipeline(preprocessed_context)
        
        if self.normalization.steps:
            self.logger.info("="*60)
            self.logger.info("  RUNNING NORMALIZATION        ")
            self.logger.info("="*60)
            # Use work_dir for staging, write final results to output_dir/normalization or normalization_dir
            if self.config.get('normalization_dir'):
                 rel_path = output_dir.relative_to(self.config.output_dir)
                 norm_output_dir = Path(self.config.get('normalization_dir')) / rel_path
            else:
                 norm_output_dir = output_dir / "normalization"
                 
            norm_output_dir.mkdir(parents=True, exist_ok=True)
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
            
            # Use work_dir for staging, write final results to output_dir/stats or segmentation_dir
            if self.config.get('segmentation_dir'):
                 rel_path = output_dir.relative_to(self.config.output_dir)
                 stats_output_dir = Path(self.config.get('segmentation_dir')) / rel_path
            else:
                 stats_output_dir = output_dir / "stats"
                 
            stats_output_dir.mkdir(parents=True, exist_ok=True)
            preprocessed_context = self.segmentation.run(subj_work_dir, preprocessed_context, reporter=reporter, final_output_dir=stats_output_dir)


        
        # Report generation moved to end to include final outputs


        # 7. Extract and Copy Final Outputs to Output Dir
        preprocessed_dwis = preprocessed_context.get("preprocessed_dwis", dwi_files)
        preprocessed_masks = preprocessed_context.get("preprocessed_masks", [])
        
        # Ensure masks list matches dwis if empty
        if not preprocessed_masks:
            preprocessed_masks = [None] * len(preprocessed_dwis)
        
        from qmri_neuropipe.io.bids import build_bids_name
        # Collecting Outputs for Report
        dmri_outputs = {
            "Final Preprocessed Images": [],
            "Modeling Derivatives": [],
            "Normalized Derivatives": [],
            "Segmentation Outputs": []
        }
            
        # 1. Output Preprocessed DWI
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
                
                # Add to Outputs (mask) -> Segmentation Outputs
                dmri_outputs["Segmentation Outputs"].append({"key": "Brain Mask", "path": str(mask_dest)})
            
            # Add Preproc Output -> Final Preprocessed Images
            dmri_outputs["Final Preprocessed Images"].append({"key": "Preprocessed DWI", "path": str(dest_path)})
            if d.bval: dmri_outputs["Final Preprocessed Images"].append({"key": "Bval", "path": str(dest_path.with_suffix("").with_suffix(".bval"))})
            if d.bvec: dmri_outputs["Final Preprocessed Images"].append({"key": "Bvec", "path": str(dest_path.with_suffix("").with_suffix(".bvec"))})

            # 3. Copy Model Outputs (if any) AND Report
            # Scan output_dir/models for results (whether newly processed or existing)
            models_root = output_dir / "models"
            
            # Also check work_dir for new results to copy if they exist and aren't in output yet
            # (Though skipping logic usually handles this, let's ensure we report whatever is KEY)
            
            # First, ensure any NEW results in work_dir are copied (if not skipped)
            modeling_work = subj_work_dir / "modeling"
            model_folders = ['DTI', 'DKI', 'NODDI', 'sandi', 'mapmri', 'CSD']
            
            if modeling_work.exists():
                for model_folder in model_folders:
                    src_model_dir = modeling_work / model_folder
                    if src_model_dir.exists():
                         dest_dir = output_dir / "models" / model_folder
                         dest_dir.mkdir(exist_ok=True, parents=True)
                         for f in src_model_dir.glob("*.nii.gz"):
                             dest = dest_dir / f.name
                             if not dest.exists() or not self.config.skip_existing:
                                 self.logger.info(f"Saving {model_folder} Map: {f.name}")
                                 shutil.copy(f, dest)
                                 # Sidecar
                                 json_src = f.with_suffix("").with_suffix(".json")
                                 if not json_src.exists(): json_src = Path(str(f).replace('.nii.gz', '.json'))
                                 if json_src.exists():
                                     shutil.copy(json_src, dest_dir / json_src.name)

            # NOW SCAN output_dir/models for Report
            if models_root.exists():
                 for model_dir in models_root.iterdir():
                      if model_dir.is_dir():
                           model_name = model_dir.name # e.g. DTI
                           for f in model_dir.glob("*.nii.gz"):
                                # Report
                                name_part = f.name.replace('.nii.gz', '')
                                suffix = name_part.split('_')[-1]
                                key_name = f"{model_name} {suffix}"
                                dmri_outputs["Modeling Derivatives"].append({"key": key_name, "path": str(f)})
                                     
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
                      
                      dmri_outputs["Normalized Derivatives"].append({"key": key_name, "path": str(f)})
            
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

    def _find_anat_files(self, subject: str, session: str | None, modality: str = 'T1w') -> list[ImageFile]:
        """
        Find anatomical files using config overrides or standard BIDS search.
        """
        # 1. Custom Check
        # Check in order of preference:
        # 1. anat: input: ... (Nested Clean)
        # 2. anat: ... (Nested Direct)
        # 3. anat_input: ... (Top Level Legacy)
        
        anat_section = self.config.get('anat', {})
        anat_cfg = {}
        
        if anat_section.get('input'):
             anat_cfg = anat_section['input']
        elif any(k.endswith('_path') or k.endswith('_search_pattern') for k in anat_section):
             anat_cfg = anat_section
        else:
             anat_cfg = self.config.get('anat_input', {})
             
        if not anat_cfg: anat_cfg = {}
        
        path_key = f"{modality.lower()}_path"
        pattern_key = f"{modality.lower()}_search_pattern"
        
        custom_path = anat_cfg.get(path_key)
        custom_pattern = anat_cfg.get(pattern_key)
        
        results = []
        
        if custom_path:
             p = Path(custom_path)
             if p.exists():
                 # Create minimal ImageFile
                 # Entities might be fake but we need at least sub/ses
                 ent = {"sub": subject, "ses": session, "suffix": modality}
                 results.append(ImageFile(entities=ent, img=p, json=None))
                 self.logger.info(f"Using custom anatomical file: {p}")
        
        elif custom_pattern:
             # Glob with formatting
             p_str = custom_pattern.format(subject=subject, session=session if session else "")
             p = Path(p_str)
             
             matches = []
             if p.is_absolute():
                  matches = list(p.parent.glob(p.name))
             else:
                  # Relative to bids_dir? Or cwd? 
                  # Implementation Plan said "derivatives/..." 
                  # Let's assume relative to BIDS_DIR if not absolute.
                  root = self.config.bids_dir
                  matches = list(root.glob(p_str))
                  
             for m in matches:
                  ent = {"sub": subject, "ses": session, "suffix": modality}
                  # Try to find sidecar
                  json_path = m.with_suffix("").with_suffix(".json")
                  if not json_path.exists(): json_path = Path(str(m).replace('.nii.gz', '.json'))
                  
                  results.append(ImageFile(entities=ent, img=m, json=json_path if json_path.exists() else None))
             
             if matches:
                 self.logger.info(f"Found {len(matches)} custom anatomical files matching pattern.")

        if results:
             return results
             
        # 2. Fallback to Standard BIDS
        # Finding everything then filtering is inefficient if we just want one subject.
        # But 'bids_find_t1w(root)' does full scan.
        # Better: use bids_find with subject specific path if possible?
        # Standard method:
        from qmri_neuropipe.io.anat.bids import bids_find_t1w, bids_find_t2w
        
        # We need the 'root' used previously. In `process_subject`, 'anat_dir' was derived.
        # But here we can use self.config.bids_dir and filter.
        # Or construct the subject directory and search there?
        # BIDS structure: bids_dir / sub-X / ses-Y / anat
        
        search_dir = self.config.bids_dir / f"sub-{subject}"
        if session:
            search_dir = search_dir / f"ses-{session}"
        search_dir = search_dir / "anat"
        
        if not search_dir.exists():
             # Try without session? Or maybe it's not in 'anat'?
             # Some BIDS have different structure?
             # Fallback to recursively searching subject dir
             search_dir = self.config.bids_dir / f"sub-{subject}"
        
        if modality == 'T1w':
             return bids_find_t1w(search_dir)
        elif modality == 'T2w':
             return bids_find_t2w(search_dir)
        return []

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
