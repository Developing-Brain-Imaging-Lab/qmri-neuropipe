"""
Execution engine for workflow processing.

This module provides a generic execution framework for running pipeline steps
with progress tracking, error handling, and state management.
"""

from pathlib import Path
from typing import Optional, Dict, List, Any, Callable
import time
from ..lib.common.spatial_transforms import normalize_transform_chain, append_transform


class ExecutionEngine:
    """
    Generic execution engine for processing workflow steps.
    
    Handles:
    - Progress tracking with rich progress bars
    - Per-step and per-image execution
    - Error handling and recovery
    - Context state management
    """
    
    def __init__(self, config, logger):
        """
        Initialize the execution engine.
        
        Parameters
        ----------
        config : PipelineConfig
            Pipeline configuration
        logger : Logger
            Logger instance
        """
        self.config = config
        self.logger = logger
        
    def execute_steps(
        self,
        steps: List,
        context: Dict,
        output_dir: Path,
        reporter=None,
        progress_callback: Optional[Callable] = None
    ) -> Dict:
        """
        Execute a list of processing steps.
        
        Parameters
        ----------
        steps : list
            List of processing steps to execute
        context : dict
            Processing context with input data
        output_dir : Path
            Output directory for results
        reporter : ReportGenerator, optional
            Reporter for generating reports
        progress_callback : callable, optional
            Callback for progress updates (description, advance)
            
        Returns
        -------
        dict
            Updated context after processing
        """
        # Identify global vs per-image steps
        GLOBAL_STEPS = self._get_global_step_types()
        
        current_dwis = context.get("dwi_files", [])
        current_masks = context.get("masks", [None] * len(current_dwis))
        native_ref_map = context.setdefault("gnl_native_reference_map", {})
        transform_map = context.setdefault("gnl_transform_map", {})
        for dwi in current_dwis:
            if getattr(dwi, "img", None):
                native_ref_map.setdefault(dwi.img, dwi)
        
        # Ensure masks list matches length
        if len(current_masks) < len(current_dwis):
            current_masks.extend([None] * (len(current_dwis) - len(current_masks)))
        
        # Execute each step
        for step_idx, step in enumerate(steps):
            step_name = step.__class__.__name__
            
            if isinstance(step, GLOBAL_STEPS):
                # Execute as global step
                context = self._execute_global_step(
                    step,
                    context,
                    current_dwis,
                    output_dir,
                    reporter,
                    progress_callback
                )
                
                # Update current state
                current_dwis = context.get("dwi_files", current_dwis)
                
                # Update masks if needed
                if len(current_masks) != len(current_dwis):
                    current_masks = [None] * len(current_dwis)
                    
            else:
                # Execute as per-image step
                current_dwis, current_masks = self._execute_per_image_step(
                    step,
                    context,
                    current_dwis,
                    current_masks,
                    output_dir,
                    reporter,
                    progress_callback
                )
        
        # Finalize context
        context["preprocessed_dwis"] = current_dwis
        context["preprocessed_masks"] = current_masks
        if current_dwis:
            context["current_image"] = current_dwis[-1]
            
        return context
    
    def _get_global_step_types(self):
        """Get tuple of step types that are executed globally (not per-image)."""
        try:
            from qmri_neuropipe.lib.dmri.synb0 import Synb0EstimationStep
            from qmri_neuropipe.lib.dmri.topup import TopupStep
            from qmri_neuropipe.lib.dmri.grad_check import GradientCheckStep
            from qmri_neuropipe.lib.dmri.reorient import DMRIReorientStep
            from qmri_neuropipe.lib.dmri.merge import MergeStep
            
            return (Synb0EstimationStep, TopupStep, GradientCheckStep, DMRIReorientStep, MergeStep)
        except ImportError as e:
            self.logger.warning(f"Could not import step types: {e}")
            return tuple()
    
    def _execute_global_step(
        self,
        step,
        context: Dict,
        current_dwis: List,
        output_dir: Path,
        reporter=None,
        progress_callback: Optional[Callable] = None
    ) -> Dict:
        """Execute a global step (operates on all files at once)."""
        step_name = step.__class__.__name__
        
        if progress_callback:
            progress_callback(f"[cyan]Global: {step_name}", advance=False)
        
        self.logger.info(f"Executing Global Step: {step_name}...")
        
        # Update context with current state
        context['dwi_files'] = current_dwis
        old_dwis = list(current_dwis)
        
        # Run step
        force_run = self.config.get("dmri", {}).get("force_run", False)
        new_ctx = step(context, output_dir=output_dir, force=force_run)
        
        if new_ctx is not context:
            context.update(new_ctx)
        
        # Check if files changed
        current_dwis = context.get("dwi_files", [])
        if current_dwis != old_dwis:
            # Refresh topup groups if files changed
            from qmri_neuropipe.io.dmri.bids import find_reversed_phase_groups
            context["topup_groups"] = find_reversed_phase_groups(current_dwis)
            native_ref_map = context.setdefault("gnl_native_reference_map", {})
            transform_map = context.setdefault("gnl_transform_map", {})
            for old_dwi, new_dwi in zip(old_dwis, current_dwis):
                if getattr(new_dwi, "img", None):
                    native_ref_map[new_dwi.img] = native_ref_map.get(getattr(old_dwi, "img", None), old_dwi)
                    context.setdefault("gnl_map_by_image", {})[new_dwi.img] = context.setdefault(
                        "gnl_map_by_image", {}
                    ).get(getattr(old_dwi, "img", None))
                    prev_chain = transform_map.get(getattr(old_dwi, "img", None))
                    new_transform = getattr(new_dwi, "spatial_transform", None)
                    if new_transform is not None:
                        transform_map[new_dwi.img] = append_transform(prev_chain, new_transform)
                    elif getattr(old_dwi, "img", None) in transform_map:
                        transform_map[new_dwi.img] = normalize_transform_chain(prev_chain)
        
        # Save intermediates if configured
        if self.config.get("save_intermediates", False):
            self._save_global_intermediate(step, output_dir, context)
        
        # Report
        if reporter:
            figures_dir = output_dir / "figures"
            figures_dir.mkdir(exist_ok=True, parents=True)
            self._report_step(reporter, step, None, None, context, None, figures_dir)
        
        if progress_callback:
            progress_callback(None, advance=True)
        
        return context
    
    def _execute_per_image_step(
        self,
        step,
        context: Dict,
        current_dwis: List,
        current_masks: List,
        output_dir: Path,
        reporter=None,
        progress_callback: Optional[Callable] = None
    ) -> tuple:
        """Execute a per-image step (processes each file separately)."""
        step_name = step.__class__.__name__
        new_dwis = []
        new_masks = []
        topup_map = context.get("topup_map", {})
        
        for i, (dwi, mask) in enumerate(zip(current_dwis, current_masks)):
            if progress_callback:
                progress_callback(
                    f"[cyan]Step: {step_name} ({i+1}/{len(current_dwis)})",
                    advance=False
                )
            
            # Prepare per-image context
            img_ctx = dict(context)
            img_ctx['current_image'] = dwi
            img_ctx['current_mask'] = mask
            native_ref = context.get("gnl_native_reference_map", {}).get(dwi.img, dwi)
            img_ctx['native_dwi_for_gnl'] = native_ref
            img_ctx['gnl_spatial_transform'] = normalize_transform_chain(context.get("gnl_transform_map", {}).get(dwi.img))
            img_ctx['gnl_map'] = context.get("gnl_map_by_image", {}).get(dwi.img)
            
            if dwi.img in topup_map:
                img_ctx["topup_base"] = topup_map[dwi.img]
            
            # Prepare step kwargs
            step_kwargs = self._prepare_step_kwargs(step, img_ctx)
            
            # Skip if no required inputs
            if step_kwargs is None:
                new_dwis.append(dwi)
                new_masks.append(mask)
                continue
            
            # Run step
            try:
                start_time = time.time()
                force_run = self.config.get("dmri", {}).get("force_run", False)
                result = step(img_ctx, output_dir=output_dir, force=force_run, **step_kwargs)
                duration = time.time() - start_time
                
                # Extract outputs
                out_dwi = result.get("current_image") if isinstance(result, dict) else result
                out_mask = result.get("current_mask") if isinstance(result, dict) else None
                
                # Ensure mask is ImageFile if it's a Path
                if out_mask is not None and isinstance(out_mask, Path):
                    from qmri_neuropipe.core.types import ImageFile
                    out_mask = ImageFile(
                        img=out_mask,
                        entities=dict(out_dwi.entities, suffix="mask")
                    )

                # Preserve DWIFile metadata even if a step returns ImageFile
                from qmri_neuropipe.core.types import DWIFile, ImageFile
                if isinstance(out_dwi, ImageFile) and not isinstance(out_dwi, DWIFile) and isinstance(dwi, DWIFile):
                    result_bval = result.get("bval") if isinstance(result, dict) else None
                    result_bvec = result.get("bvec") if isinstance(result, dict) else None
                    result_json = result.get("json") if isinstance(result, dict) else None
                    out_dwi = DWIFile(
                        img=out_dwi.img,
                        entities=out_dwi.entities,
                        json=result_json or getattr(out_dwi, "json", None) or dwi.json,
                        bval=result_bval or dwi.bval,
                        bvec=result_bvec or dwi.bvec
                    )
                elif isinstance(out_dwi, DWIFile) and isinstance(dwi, DWIFile):
                    if out_dwi.bval is None and dwi.bval is not None:
                        out_dwi.bval = dwi.bval
                    if out_dwi.bvec is None and dwi.bvec is not None:
                        out_dwi.bvec = dwi.bvec
                    if getattr(out_dwi, "json", None) is None and dwi.json is not None:
                        out_dwi.json = dwi.json

                new_dwis.append(out_dwi)
                new_masks.append(out_mask if out_mask is not None else mask)
                if getattr(out_dwi, "img", None):
                    context.setdefault("gnl_native_reference_map", {})[out_dwi.img] = native_ref
                    prev_transform = context.setdefault("gnl_transform_map", {}).get(dwi.img)
                    new_transform = result.get("spatial_transform") if isinstance(result, dict) else getattr(out_dwi, "spatial_transform", None)
                    if new_transform is not None:
                        context["gnl_transform_map"][out_dwi.img] = append_transform(prev_transform, new_transform)
                    elif prev_transform is not None:
                        context["gnl_transform_map"][out_dwi.img] = normalize_transform_chain(prev_transform)

                    if isinstance(result, dict) and "gnl_map" in result:
                        context.setdefault("gnl_map_by_image", {})[out_dwi.img] = result["gnl_map"]
                    else:
                        context.setdefault("gnl_map_by_image", {})[out_dwi.img] = context.get(
                            "gnl_map_by_image", {}
                        ).get(dwi.img, context.get("gnl_map"))
                
                # Update QC metrics registry
                self._update_qc_metrics(context, result, out_dwi)
                
                # Collect GNL maps
                if isinstance(result, dict) and "gnl_map" in result:
                    if "gnl_maps" not in context:
                        context["gnl_maps"] = []
                    gnl_path = result["gnl_map"]
                    if gnl_path not in context["gnl_maps"]:
                        context["gnl_maps"].append(gnl_path)
                
                # Report
                if reporter:
                    figures_dir = output_dir / "figures"
                    figures_dir.mkdir(exist_ok=True, parents=True)
                    self._report_step(
                        reporter, step, dwi, dwi, result,
                        step_kwargs.get('target'), figures_dir, step_kwargs
                    )

                # Track step parameters for provenance JSON
                if isinstance(step_kwargs, dict):
                    detail = {
                        "step": step_name,
                        "parameters": {
                            k: v for k, v in step_kwargs.items()
                            if k not in {"target", "output_dir", "force"}
                        }
                    }
                    context.setdefault("processing_steps_detail", []).append(detail)
                
                # Save intermediates
                if self.config.get("save_intermediates", False):
                    self._save_image_intermediate(output_dir, context, result)
                
            except Exception as e:
                if self.config.get("stop_on_error", False):
                    raise e
                self.logger.error(f"Step {step_name} failed on {dwi.img.name}: {e}")
                new_dwis.append(dwi)
                new_masks.append(mask)
        
        if progress_callback:
            progress_callback(None, advance=True)
        
        return new_dwis, new_masks
    
    def _prepare_step_kwargs(self, step, context: Dict) -> Optional[Dict]:
        """
        Prepare keyword arguments for a step based on its type.
        
        Returns None if step should be skipped (missing required inputs).
        """
        from qmri_neuropipe.lib.common.registration import CoregistrationStep
        from qmri_neuropipe.lib.common.mask import BrainMaskingStep
        
        step_kwargs = {}
        
        if isinstance(step, CoregistrationStep):
            # Determine target image
            t1w_files = context.get("t1w_files", [])
            t2w_files = context.get("t2w_files", [])
            
            target_modality = self.config.get(
                'dmri.preprocessing.coregistration.reference_image',
                'T1w'
            )
            coreg_cfg = self.config.get('dmri.preprocessing.coregistration', {})
            
            target_img = None
            actual_modality = target_modality
            
            if target_modality == "T2w":
                target_img = t2w_files[0].img if t2w_files else None
                if not target_img and t1w_files:
                    self.logger.info(
                        "T2w reference requested but not found. "
                        "Falling back to T1w for coregistration."
                    )
                    target_img = t1w_files[0].img
                    actual_modality = "T1w"
            else:  # Default T1w
                target_img = t1w_files[0].img if t1w_files else None
                if not target_img and t2w_files:
                    self.logger.info(
                        "T1w reference requested but not found. "
                        "Falling back to T2w for coregistration."
                    )
                    target_img = t2w_files[0].img
                    actual_modality = "T2w"
            
            if not target_img:
                return None  # Skip this step
            
            step_kwargs["target"] = target_img
            flat_opts = dict(coreg_cfg)
            if "options" in flat_opts:
                flat_opts.update(flat_opts.pop("options"))
            step_kwargs["options"] = flat_opts
            step_kwargs["target_modality"] = actual_modality
            
        elif isinstance(step, BrainMaskingStep):
            step_kwargs["return_mask"] = True
            
            # Optimization: Use structural mask if coregistered
            coreg_cfg = self.config.get('dmri.preprocessing.coregistration', {})
            coreg_enabled = coreg_cfg.get('enabled')
            if coreg_enabled is None:
                coreg_enabled = self.config.get("do_coregistration", False)
            
            coreg_opts = coreg_cfg.get("options", {})
            out_res = coreg_opts.get(
                "output_resolution",
                coreg_cfg.get("output_resolution", "anatomical")
            ).lower()
            
            if coreg_enabled and out_res == "anatomical":
                step_kwargs["structural_mask"] = self._find_structural_mask(context)
        
        return step_kwargs
    
    def _find_structural_mask(self, context: Dict) -> Optional[Path]:
        """Find structural brain mask if available."""
        from qmri_neuropipe.io.bids import build_bids_name
        
        t1w_files = context.get("t1w_files", [])
        t2w_files = context.get("t2w_files", [])
        
        # Determine which modality was used for coregistration
        target_modality = self.config.get(
            'dmri.preprocessing.coregistration.reference_image',
            'T1w'
        )
        
        structural_files = t2w_files if target_modality == "T2w" else t1w_files
        
        if not structural_files:
            return None
        
        struct_ref = structural_files[0]
        if not hasattr(struct_ref, 'img'):
            return None
        
        struct_path = struct_ref.img
        parent = struct_path.parent
        
        if not hasattr(struct_ref, 'entities'):
            return None
        
        # Try finding mask with different descriptions
        m_ents = struct_ref.entities.copy()
        m_ents['suffix'] = 'mask'
        
        # Variant A: desc-brain
        m_ents['desc'] = 'brain'
        mask_path = parent / build_bids_name(m_ents)
        if mask_path.exists():
            self.logger.info(f"Found structural mask: {mask_path}")
            return mask_path
        
        # Variant B: desc-preproc
        m_ents['desc'] = 'preproc'
        mask_path = parent / build_bids_name(m_ents)
        if mask_path.exists():
            self.logger.info(f"Found structural mask: {mask_path}")
            return mask_path
        
        return None
    
    def _update_qc_metrics(self, context: Dict, result: Dict, out_dwi):
        """Update QC metrics registry from step results."""
        if not isinstance(result, dict):
            return
        
        if "qc_metrics" not in result and "outlier_stats" not in result:
            return
        
        # Initialize QC registry
        if "qc_registry" not in context:
            context["qc_registry"] = {}
        
        img_key = out_dwi.img.name
        
        # Create or update record
        if img_key not in context["qc_registry"]:
            context["qc_registry"][img_key] = {
                "subject": context.get("subject"),
                "session": context.get("session"),
                "file_name": img_key
            }
        
        # Add metrics
        if "qc_metrics" in result:
            context["qc_registry"][img_key].update(result["qc_metrics"])
        
        if "outlier_stats" in result:
            context["qc_registry"][img_key]["outliers"] = result["outlier_stats"]
        
        # Update list view
        context["all_qc_metrics"] = list(context["qc_registry"].values())
    
    def _save_global_intermediate(self, step, output_dir: Path, context: Dict):
        """Save global step intermediate results."""
        import shutil
        from qmri_neuropipe.lib.dmri.topup import TopupStep
        from qmri_neuropipe.lib.dmri.synb0 import Synb0EstimationStep
        
        sub = context.get("subject")
        ses = context.get("session")
        base_out = self.config.output_dir
        
        final_dwi_dir = base_out / f"sub-{sub}"
        if ses:
            final_dwi_dir /= f"ses-{ses}"
        final_dwi_dir /= "dwi"
        
        dest_inter = final_dwi_dir / "intermediate"
        dest_inter.mkdir(parents=True, exist_ok=True)
        
        if isinstance(step, TopupStep):
            src = output_dir / "topup"
            if src.exists():
                shutil.copytree(src, dest_inter / "topup", dirs_exist_ok=True)
        elif isinstance(step, Synb0EstimationStep):
            src = output_dir / "synb0"
            if src.exists():
                shutil.copytree(src, dest_inter / "synb0", dirs_exist_ok=True)
    
    def _save_image_intermediate(self, output_dir: Path, context: Dict, current_arg):
        """Save per-image step intermediate results."""
        import shutil
        
        sub = context.get("subject")
        ses = context.get("session")
        base_out = self.config.output_dir
        
        final_dwi_dir = base_out / f"sub-{sub}"
        if ses:
            final_dwi_dir /= f"ses-{ses}"
        final_dwi_dir /= "dwi"
        
        inter_dir = final_dwi_dir / "intermediate"
        inter_dir.mkdir(parents=True, exist_ok=True)
        
        curr_img = current_arg.get("current_image") if isinstance(current_arg, dict) else None
        if curr_img and hasattr(curr_img, "img") and curr_img.img.exists():
            src_dir = curr_img.img.parent
            if src_dir != output_dir and output_dir in src_dir.parents:
                target_step_dir = inter_dir / src_dir.name
                shutil.copytree(src_dir, target_step_dir, dirs_exist_ok=True)
                self.logger.info(f"Saved intermediate: {target_step_dir}")
    
    def _report_step(self, reporter, step, dwi, prev_img, current_arg, target_img, figures_dir, step_kwargs=None):
        """Report step results."""
        try:
            from .reporting import report_preprocessing_step
            report_preprocessing_step(
                reporter, step, dwi, prev_img,
                current_arg, target_img, figures_dir, step_kwargs
            )
        except Exception as e:
            self.logger.warning(f"Reporting failed for step: {e}")
