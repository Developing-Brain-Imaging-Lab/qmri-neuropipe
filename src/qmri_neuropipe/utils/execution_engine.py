"""
Execution engine for workflow processing.

This module provides a generic execution framework for running pipeline steps
with progress tracking, error handling, and state management.
"""

from pathlib import Path
from typing import Optional, Dict, List, Any, Callable
import logging
import time
from ..core.step_control import get_rerun_from_step, step_force_active
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
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        
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
        rerun_from_step = get_rerun_from_step(self.config, "dmri.preprocessing", "preprocessing")
        force_from_step_active = False
        for step_idx, step in enumerate(steps):
            step_name = step.__class__.__name__
            force_from_step_active = step_force_active(force_from_step_active, step, rerun_from_step)
            
            if isinstance(step, GLOBAL_STEPS):
                # Execute as global step
                context = self._execute_global_step(
                    step,
                    context,
                    current_dwis,
                    output_dir,
                    reporter,
                    progress_callback,
                    force_from_step=force_from_step_active,
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
                    progress_callback,
                    force_from_step=force_from_step_active,
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
            from qmri_neuropipe.lib.dmri.drbuddi import NativeDrbuddiStep
            from qmri_neuropipe.lib.dmri.tortoise_v4 import TortoiseV4CorrectionStep
            
            return (Synb0EstimationStep, TopupStep, GradientCheckStep, DMRIReorientStep, MergeStep, NativeDrbuddiStep, TortoiseV4CorrectionStep)
        except ImportError as e:
            self.logger.warning(f"Could not import step types: {e}")
            return tuple()

    def _build_image_mapping(self, old_dwis: List, new_dwis: List) -> Dict[Any, Any]:
        image_mapping: Dict[Any, Any] = {}

        for old_dwi, new_dwi in zip(old_dwis, new_dwis):
            old_img = getattr(old_dwi, "img", None)
            if old_img is None or getattr(new_dwi, "img", None) is None:
                continue
            image_mapping[old_img] = new_dwi
            image_mapping[str(old_img)] = new_dwi

        return image_mapping

    def _remap_group_images(self, items: List[Any], image_mapping: Dict[Any, Any]) -> List[Any]:
        remapped: List[Any] = []
        seen = set()

        for item in items or []:
            key = getattr(item, "img", item)
            mapped = image_mapping.get(key)
            if mapped is None:
                mapped = image_mapping.get(str(key))
            if mapped is None:
                mapped = item

            dedupe_key = getattr(mapped, "img", mapped)
            dedupe_key = str(dedupe_key)
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            remapped.append(mapped)

        return remapped

    def _remap_topup_groups(self, context: Dict, image_mapping: Dict[Any, Any]) -> None:
        topup_groups = context.get("topup_groups")
        if not topup_groups:
            return

        remapped_groups = []
        for group_item in topup_groups:
            if isinstance(group_item, dict):
                updated_group = dict(group_item)
                updated_group["inputs"] = self._remap_group_images(
                    group_item.get("inputs", []),
                    image_mapping,
                )
                updated_group["targets"] = self._remap_group_images(
                    group_item.get("targets", []),
                    image_mapping,
                )
                if updated_group.get("inputs"):
                    remapped_groups.append(updated_group)
            else:
                remapped_inputs = self._remap_group_images(list(group_item), image_mapping)
                if remapped_inputs:
                    remapped_groups.append(remapped_inputs)

        context["topup_groups"] = remapped_groups

    def _remap_topup_map(self, context: Dict, image_mapping: Dict[Any, Any]) -> None:
        topup_map = context.get("topup_map")
        if not topup_map:
            return

        remapped_map = dict(topup_map)
        for old_key, new_dwi in image_mapping.items():
            if isinstance(old_key, str):
                continue
            if old_key in topup_map:
                value = topup_map[old_key]
            elif str(old_key) in topup_map:
                value = topup_map[str(old_key)]
            else:
                continue

            new_img = getattr(new_dwi, "img", None)
            if new_img is None:
                continue
            remapped_map[new_img] = value
            remapped_map[str(new_img)] = value

        context["topup_map"] = remapped_map
    
    def _execute_global_step(
        self,
        step,
        context: Dict,
        current_dwis: List,
        output_dir: Path,
        reporter=None,
        progress_callback: Optional[Callable] = None,
        force_from_step: bool = False,
    ) -> Dict:
        """Execute a global step (operates on all files at once)."""
        step_name = step.__class__.__name__
        
        if progress_callback:
            progress_callback(f"[cyan]Global: {step_name}", advance=False)
        
        self.logger.info(f"Executing Global Step: {step_name}...")
        
        # Update context with current state
        context['dwi_files'] = current_dwis
        old_dwis = list(current_dwis)
        old_topup_groups = context.get("topup_groups", [])
        
        # Run step
        force_run = self.config.get("dmri", {}).get("force_run", False) or force_from_step
        if force_from_step:
            self.logger.info(f"Forcing {step_name} because rerun_from_step has been reached.")
        new_ctx = step(context, output_dir=output_dir, force=force_run)
        
        if new_ctx is not context:
            context.update(new_ctx)
        
        # Check if files changed
        current_dwis = context.get("dwi_files", [])
        if current_dwis != old_dwis:
            image_mapping = self._build_image_mapping(old_dwis, current_dwis)
            self._remap_topup_groups(context, image_mapping)
            self._remap_topup_map(context, image_mapping)

            if not context.get("topup_groups") and old_topup_groups:
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
        progress_callback: Optional[Callable] = None,
        force_from_step: bool = False,
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
            image_key = getattr(dwi, "img", None)
            image_key_str = str(image_key) if image_key is not None else None
            native_ref_map = context.get("gnl_native_reference_map", {})
            transform_map = context.get("gnl_transform_map", {})
            gnl_map_by_image = context.get("gnl_map_by_image", {})

            native_ref = (
                native_ref_map.get(image_key)
                or native_ref_map.get(image_key_str)
                or dwi
            )
            img_ctx['native_dwi_for_gnl'] = native_ref
            img_ctx['gnl_spatial_transform'] = normalize_transform_chain(
                transform_map.get(image_key) or transform_map.get(image_key_str)
            )
            img_ctx['gnl_map'] = (
                gnl_map_by_image.get(image_key)
                or gnl_map_by_image.get(image_key_str)
            )
            img_ctx['_execution_output_dir'] = output_dir
            
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
                self.logger.info(
                    "Input to %s: %s",
                    step_name,
                    getattr(dwi, "img", dwi),
                )
                force_run = self.config.get("dmri", {}).get("force_run", False) or force_from_step
                if force_from_step:
                    self.logger.info(f"Forcing {step_name} because rerun_from_step has been reached.")
                result = step(img_ctx, output_dir=output_dir, force=force_run, **step_kwargs)
                duration = time.time() - start_time

                if isinstance(result, dict):
                    for map_name in ("gnl_source_map", "gnl_native_reference_map", "gnl_transform_map"):
                        result_map = result.get(map_name)
                        if isinstance(result_map, dict):
                            context.setdefault(map_name, {}).update(result_map)
                
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
                prior_gnl_map = (
                    context.get("gnl_map_by_image", {}).get(image_key)
                    or context.get("gnl_map_by_image", {}).get(image_key_str)
                    or context.get("gnl_map")
                )
                if getattr(out_dwi, "img", None):
                    out_key = out_dwi.img
                    out_key_str = str(out_key)
                    context.setdefault("gnl_native_reference_map", {})[out_key] = native_ref
                    context["gnl_native_reference_map"][out_key_str] = native_ref
                    prev_transform = (
                        context.setdefault("gnl_transform_map", {}).get(image_key)
                        or context["gnl_transform_map"].get(image_key_str)
                    )
                    new_transform = result.get("spatial_transform") if isinstance(result, dict) else getattr(out_dwi, "spatial_transform", None)
                    if new_transform is not None:
                        chained_transform = append_transform(prev_transform, new_transform)
                        context["gnl_transform_map"][out_key] = chained_transform
                        context["gnl_transform_map"][out_key_str] = chained_transform
                    elif prev_transform is not None:
                        chained_transform = normalize_transform_chain(prev_transform)
                        context["gnl_transform_map"][out_key] = chained_transform
                        context["gnl_transform_map"][out_key_str] = chained_transform

                    if isinstance(result, dict) and "gnl_map" in result:
                        context.setdefault("gnl_map_by_image", {})[out_key] = result["gnl_map"]
                        context["gnl_map_by_image"][out_key_str] = result["gnl_map"]
                    else:
                        carried_gnl = (
                            context.get("gnl_map_by_image", {}).get(image_key)
                            or context.get("gnl_map_by_image", {}).get(image_key_str)
                            or context.get("gnl_map")
                        )
                        context.setdefault("gnl_map_by_image", {})[out_key] = carried_gnl
                        context["gnl_map_by_image"][out_key_str] = carried_gnl
                
                # Update QC metrics registry
                self._update_qc_metrics(context, result, out_dwi)
                
                # Collect GNL maps
                if isinstance(result, dict) and "gnl_map" in result:
                    if "gnl_maps" not in context:
                        context["gnl_maps"] = []
                    gnl_path = result["gnl_map"]
                    if prior_gnl_map is not None:
                        context["gnl_maps"] = [
                            existing for existing in context["gnl_maps"]
                            if existing != prior_gnl_map
                        ]
                    if gnl_path not in context["gnl_maps"]:
                        context["gnl_maps"].append(gnl_path)
                    context["gnl_map"] = gnl_path
                
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

        if new_dwis != current_dwis:
            image_mapping = self._build_image_mapping(current_dwis, new_dwis)
            self._remap_topup_groups(context, image_mapping)
            self._remap_topup_map(context, image_mapping)
        
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
            flat_opts = dict(coreg_cfg)
            if "options" in flat_opts and isinstance(flat_opts["options"], dict):
                nested_opts = dict(flat_opts.pop("options"))
                nested_opts.update(flat_opts)
                flat_opts = nested_opts

            supersynth_mode = str(target_modality).lower() in {
                "supersynth",
                "syntht1w",
                "synthetic_t1w",
                "supersynth_multivariate",
            }
            if supersynth_mode:
                prepared = self._prepare_dmri_supersynth_coregistration(
                    context,
                    flat_opts,
                    backend=step.method,
                    multivariate=(
                        str(target_modality).lower() == "supersynth_multivariate"
                        or str(flat_opts.get("supersynth_registration", "")).lower() == "multivariate"
                    ),
                )
                if prepared is None:
                    self.logger.warning(
                        "Skipping CoregistrationStep: reference_image=%s could not "
                        "be prepared because no usable anatomical reference was loaded "
                        "(T1w=%d, T2w=%d).",
                        target_modality,
                        len(t1w_files),
                        len(t2w_files),
                    )
                    return None
                target_img, actual_modality, flat_opts = prepared
            
            if target_img is not None:
                pass
            elif target_modality == "T2w":
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
                self.logger.warning(
                    "Skipping CoregistrationStep: reference_image=%s was requested, "
                    "but no usable anatomical reference was loaded (T1w=%d, T2w=%d).",
                    target_modality,
                    len(t1w_files),
                    len(t2w_files),
                )
                return None
            
            step_kwargs["target"] = target_img
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

    def _prepare_dmri_supersynth_coregistration(
        self,
        context: Dict,
        options: Dict,
        *,
        backend: str,
        multivariate: bool,
    ) -> Optional[tuple[Path, str, Dict]]:
        """Prepare synthetic fixed/moving contrasts for dMRI coregistration."""
        from qmri_neuropipe.core.utils import get_nifti_stem
        from qmri_neuropipe.lib.anat.super_synth import (
            ensure_matched_supersynth_registration_inputs,
            extract_mean_b0_for_supersynth,
        )

        dwi = context.get("current_image")
        output_dir = context.get("_execution_output_dir")
        if dwi is None or output_dir is None:
            self.logger.warning("SuperSynth dMRI coregistration is missing the current DWI or output directory.")
            return None

        preference = str(options.get("supersynth_input", "auto")).lower()
        t1w_files = context.get("t1w_files", [])
        t2w_files = context.get("t2w_files", [])
        if preference == "t2w":
            anatomical = t2w_files[0] if t2w_files else None
            modality = "T2w"
        elif preference == "t1w":
            anatomical = t1w_files[0] if t1w_files else None
            modality = "T1w"
        else:
            anatomical = t1w_files[0] if t1w_files else (t2w_files[0] if t2w_files else None)
            modality = "T1w" if t1w_files else "T2w"

        if anatomical is None:
            self.logger.warning("SuperSynth dMRI coregistration requires a T1w or T2w anatomical image.")
            return None

        force = bool(
            self.config.get("dmri", {}).get("force_run", False)
            or options.get("force", False)
        )
        stem = get_nifti_stem(Path(dwi.img))
        synth_root = Path(output_dir) / "coregistration" / "supersynth_dwi" / stem
        mean_b0 = extract_mean_b0_for_supersynth(
            dwi,
            synth_root / "mean_b0.nii.gz",
            self.logger,
            b0_threshold=float(options.get("supersynth_b0_threshold", 50.0)),
            force=force,
        )

        helper_kwargs = {
            "mode": options.get("supersynth_mode"),
            "device": options.get("supersynth_device"),
            "sharpen_synths": options.get("supersynth_sharpen_synths"),
            "force": force,
        }
        use_multivariate = multivariate and backend == "ants"
        if multivariate and not use_multivariate:
            self.logger.warning(
                f"SuperSynth multivariate registration is not supported by the "
                f"'{backend}' backend; using the synthetic T1w pair only."
            )
        required = (
            ("synth_t1w", "synth_t2w")
            if use_multivariate
            else ("synth_t1w",)
        )
        try:
            fixed_outputs, moving_outputs = (
                ensure_matched_supersynth_registration_inputs(
                    anatomical,
                    mean_b0,
                    synth_root,
                    self.config,
                    self.logger,
                    required_contrasts=required,
                    fixed_subdir="fixed_anatomical",
                    moving_subdir="moving_dwi",
                    **helper_kwargs,
                )
            )
        except Exception as exc:
            self.logger.warning(
                "SuperSynth dMRI coregistration could not prepare a matched "
                "synthetic pair: %s",
                exc,
            )
            return None

        prepared_options = dict(options)
        prepared_options.update({
            "registration_fixed": fixed_outputs["synth_t1w"],
            "registration_moving": moving_outputs["synth_t1w"],
            "application_fixed": Path(anatomical.img),
            "transform_type": prepared_options.get("transform_type", "Rigid"),
        })
        if use_multivariate:
            prepared_options.update({
                "registration_fixed_extras": [fixed_outputs["synth_t2w"]],
                "registration_moving_extras": [moving_outputs["synth_t2w"]],
            })

        context.setdefault("dmri_supersynth_coregistration", {})[str(dwi.img)] = {
            "mean_b0": mean_b0,
            "fixed_outputs": fixed_outputs,
            "moving_outputs": moving_outputs,
        }
        self.logger.info(
            "Using SuperSynth dMRI coregistration "
            f"with {backend} ({'T1w+T2w' if use_multivariate else 'T1w'} synthetic contrasts); "
            "the resulting transform will be applied to the original DWI."
        )
        return Path(anatomical.img), modality, prepared_options
    
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
        
        sub = context.get("subject")
        ses = context.get("session")
        base_out = self.config.output_dir
        
        final_dwi_dir = base_out / f"sub-{sub}"
        if ses:
            final_dwi_dir /= f"ses-{ses}"
        final_dwi_dir /= "dwi"
        
        dest_inter = final_dwi_dir / "intermediate"
        dest_inter.mkdir(parents=True, exist_ok=True)

        # Use the step's canonical directory mapping (for example,
        # Synb0EstimationStep -> synb0estimation) rather than maintaining a
        # second, incomplete list of aliases here.
        src = step.get_step_output_dir(output_dir)
        if src.exists() and any(src.iterdir()):
            shutil.copytree(src, dest_inter / src.name, dirs_exist_ok=True)
            self.logger.info(f"Saved intermediate: {dest_inter / src.name}")
    
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
