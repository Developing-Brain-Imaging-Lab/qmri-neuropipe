from pathlib import Path
from typing import Any
import json
import logging
import os
import shutil

import nibabel as nib

from ...core import BaseProcessingStep, ProcessingError
from ...core.utils import ensure_dir
from ..common.registration import prepare_registration_images, _ALL_SKULL_STRIP_OPTION_KEYS


def _is_reference_target(value: Any) -> bool:
    token = "".join(ch for ch in str(value or "").lower() if ch.isalnum())
    return token in {
        "spgrref",
        "reference",
        "ref",
        "currentimage",
        "sourceimage",
    }


class NormalizationStep(BaseProcessingStep):
    """
    Step to register diffusion metrics to a standard space template.
    Usually calculates transform based on a driving metric (e.g. FA) 
    and applies it to all other available metrics.
    """
    def _ensure_3d(self, img, is_driving=False):
        """
        Ensure ANTs image is 3D.
        If 4D with 1 volume, slice it.
        If 4D with >1 volume:
           - if is_driving: Error (Registration requires 3D scalar/volume)
           - else: Return as is, flag as 4D.
        Returns: (img, is_4d_series)
        """
        import ants
        if img.dimension == 4:
            if img.shape[3] == 1:
                # Squeeze using slice
                # ants.slice_image(image, axis, idx)
                # axis 3 is time
                return ants.slice_image(img, axis=3, idx=0), False
            else:
                if is_driving:
                    raise ProcessingError(f"Driving metric is 4D ({img.shape}). Registration requires a 3D volume.")
                return img, True
        return img, False

    def __init__(self, config, logger, provenance, template=None, driving_metric='FA', space_name=None, tool='ants', **kwargs):
        super().__init__(config, logger, provenance)
        self.template = Path(template) if template else None
        self.driving_metric = driving_metric
        self.space_name = space_name or "Standard"
        self.space_entity = kwargs.get('space_entity') or self.space_name
        tool_normalized = tool.lower()
        if tool_normalized == 'mri_synthmorph':
            tool_normalized = 'synthmorph'
        self.tool = tool_normalized
        self.save_transforms = kwargs.get('save_transforms', True)
        self.include_all_metrics = kwargs.get('include_all_metrics', True)
        self.kwargs = kwargs

    @staticmethod
    def _resolve_ants_interpolator(interpolator: str) -> str:
        interp = str(interpolator or "linear")
        mapping = {
            "nearest": "nearestNeighbor",
            "nearestneighbor": "nearestNeighbor",
            "label": "genericLabel",
            "genericlabel": "genericLabel",
            "multilabel": "multiLabel",
            "cubic": "bspline",
        }
        return mapping.get(interp.lower(), interp)

    @staticmethod
    def _is_4d_image(path: Path) -> bool:
        shape = nib.load(str(path)).shape
        return len(shape) >= 4 and shape[3] > 1

    @staticmethod
    def _json_safe(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {str(k): NormalizationStep._json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [NormalizationStep._json_safe(v) for v in value]
        return value

    def _synthmorph_transform_extension(
        self,
        model: str | None = None,
        requested: str | None = None,
    ) -> str:
        from ...interfaces.freesurfer import synthmorph_transform_extension

        try:
            return synthmorph_transform_extension(model, requested)
        except ValueError as exc:
            raise ProcessingError(str(exc)) from exc

    @staticmethod
    def _clean_transform_entities(ref_path: Path, space_entity: str) -> dict[str, Any]:
        from ...io.bids import get_entities_from_path

        ents = get_entities_from_path(ref_path)
        for key in ['acq', 'dir', 'run', 'echo', 'part', 'model', 'desc', 'space']:
            ents.pop(key, None)
        ents['space'] = space_entity
        ents['suffix'] = 'xfm'
        return ents

    def _robust_manifest_path(self, ref_path: Path, norm_out: Path) -> Path:
        from ...io.bids import build_bids_name

        ents = self._clean_transform_entities(ref_path, self.space_entity)
        ents['desc'] = 'robustiterative'
        return norm_out / build_bids_name(ents, extension='.json')

    def _predicted_normalized_outputs(self, metrics_to_norm: dict[str, dict[str, Path]], norm_out: Path) -> list[tuple[str, str, Path]]:
        from ...io.bids import build_bids_name, get_entities_from_path

        predicted_outputs = []
        for model_name, metrics in metrics_to_norm.items():
            for name, path in metrics.items():
                ents = get_entities_from_path(path)
                ents['space'] = self.space_entity
                if 'model' not in ents or not ents['model']:
                    ents['model'] = model_name
                predicted_outputs.append((model_name, name, norm_out / build_bids_name(ents)))
        return predicted_outputs

    def _populate_existing_normalized_results(
        self,
        context: dict,
        predicted_outputs: list[tuple[str, str, Path]],
        manifest_path: Path | None = None,
    ) -> dict:
        normalized_results = {}
        normalized_results_by_model = {}
        for model_name, name, existing_path in predicted_outputs:
            normalized_results_by_model.setdefault(model_name, {})[name] = existing_path
            prefixed_key = f"{model_name}_{name}"
            normalized_results[prefixed_key] = existing_path
            if name not in normalized_results:
                normalized_results[name] = existing_path
        context['normalized_results'] = normalized_results
        context['normalized_results_by_model'] = normalized_results_by_model
        if manifest_path:
            context['normalization_transform_manifest'] = manifest_path
        return context

    def _run_ants_registration_bundle(
        self,
        moving_file: Path,
        fixed_file: Path,
        out_prefix: Path,
        transform_type: str,
        interpolator: str,
        nthreads: int,
        registration_kwargs: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(nthreads)
        import ants

        registration_kwargs = dict(registration_kwargs or {})
        registration_kwargs.pop("interpolator", None)

        fixed_img = ants.image_read(str(fixed_file))
        if fixed_img.dimension == 4:
            fixed_img = ants.slice_image(fixed_img, axis=3, idx=0)

        moving_raw = ants.image_read(str(moving_file))
        moving_img, _ = self._ensure_3d(moving_raw, is_driving=True)

        reg = ants.registration(
            fixed=fixed_img,
            moving=moving_img,
            type_of_transform=transform_type,
            outprefix=str(out_prefix),
            **registration_kwargs,
        )

        warped_out = out_prefix.with_suffix("").parent / f"{out_prefix.name}Warped.nii.gz"
        ants.image_write(reg['warpedmovout'], str(warped_out))

        return {
            "warped": warped_out,
            "forward": [Path(p) for p in reg['fwdtransforms']],
            "inverse": [Path(p) for p in reg.get('invtransforms', [])],
            "transform_type": transform_type,
            "registration_kwargs": registration_kwargs,
        }

    def _apply_ants_transform_chain(
        self,
        moving_file: Path,
        fixed_file: Path,
        transforms: list[Path],
        out_file: Path,
        interpolator: str,
        nthreads: int,
    ) -> Path:
        from ...interfaces import ants as ants_iface

        apply_kwargs = {}
        if self._is_4d_image(moving_file):
            apply_kwargs["imagetype"] = 3

        ants_iface.apply_transforms(
            fixed_file=fixed_file,
            moving_file=moving_file,
            out_file=out_file,
            transforms=transforms,
            interpolator=self._resolve_ants_interpolator(interpolator),
            nthreads=nthreads,
            **apply_kwargs,
        )
        return out_file

    def _apply_robust_transform_manifest_to_image(
        self,
        moving_file: Path,
        out_file: Path,
        manifest: dict[str, Any],
        interpolator: str,
        nthreads: int,
    ) -> Path:
        return apply_robust_transform_manifest(
            manifest=manifest,
            moving_file=moving_file,
            out_file=out_file,
            interpolator=interpolator,
            nthreads=nthreads,
        )

    def _run_robust_iterative_normalization(
        self,
        context: dict,
        metrics_to_norm: dict[str, dict[str, Path]],
        ref_path: Path,
        norm_out: Path,
        nthreads: int,
        force: bool = False,
    ) -> dict:
        from ...interfaces import ants as ants_iface
        from ...interfaces.freesurfer import lta_to_itk, mri_synthmorph_register, mri_warp_convert_to_itk

        skip = self.config.get('skip_existing', False) and not force
        predicted_outputs = self._predicted_normalized_outputs(metrics_to_norm, norm_out)
        manifest_path = self._robust_manifest_path(ref_path, norm_out)

        if skip and predicted_outputs and manifest_path.exists() and all(path.exists() for _, _, path in predicted_outputs):
            in_mtime = ref_path.stat().st_mtime
            out_mtime = predicted_outputs[0][2].stat().st_mtime
            if in_mtime <= out_mtime:
                self.logger.info("Skipping robust iterative normalization (all outputs and transform manifest exist).")
                return self._populate_existing_normalized_results(context, predicted_outputs, manifest_path=manifest_path)

        robust_cfg = dict(self.kwargs.get('robust_iterative') or {})
        iterations = int(robust_cfg.get('iterations', 2))
        if iterations < 1:
            raise ProcessingError("robust_iterative.iterations must be >= 1")

        synth_cfg = dict(robust_cfg.get('synthmorph') or {})
        ants_cfg = dict(robust_cfg.get('ants') or {})
        synth_enabled = synth_cfg.get('enabled', True)
        ants_enabled = ants_cfg.get('enabled', True)
        if not synth_enabled and not ants_enabled:
            raise ProcessingError("robust_iterative requires at least one of synthmorph or ants to be enabled")

        work_dir = norm_out / "robust_iterative"
        work_dir.mkdir(parents=True, exist_ok=True)

        template = Path(self.template)
        current_ref = Path(ref_path)
        iterations_manifest: list[dict[str, Any]] = []
        final_driving = None
        cli_forward_chain: list[str] = []
        cli_inverse_chain: list[str] = []

        transform_ents = self._clean_transform_entities(ref_path, self.space_entity)

        self.logger.info(
            "Running robust iterative normalization with %d iteration(s): SynthMorph=%s, ANTs=%s",
            iterations,
            synth_enabled,
            ants_enabled,
        )

        for iteration in range(1, iterations + 1):
            iteration_record: dict[str, Any] = {"index": iteration}

            if synth_enabled:
                from ...io.bids import build_bids_name, get_entities_from_path

                synth_model = str(synth_cfg.get('model', 'joint'))
                synth_ext = self._synthmorph_transform_extension(
                    synth_model,
                    synth_cfg.get('transform_ext'),
                )
                synth_tx = work_dir / build_bids_name(
                    {**transform_ents, "desc": f"iter{iteration}synthmorph"},
                    extension=synth_ext,
                )
                synth_inv_tx = work_dir / build_bids_name(
                    {**transform_ents, "desc": f"iter{iteration}synthmorphInverse"},
                    extension=synth_ext,
                )
                synth_ents = get_entities_from_path(ref_path)
                synth_ents['space'] = self.space_entity
                synth_ents['desc'] = f"iter{iteration}synthmorph"
                synth_warped = work_dir / build_bids_name(synth_ents)

                mri_synthmorph_register(
                    moving=current_ref,
                    target=template,
                    transform_out=synth_tx,
                    output_image=synth_warped,
                    inverse_transform_out=synth_inv_tx,
                    model=synth_model,
                    extra_args=str(synth_cfg.get('register_args', '') or ''),
                    overwrite=True,
                )

                if synth_ext == '.lta':
                    synth_itk = work_dir / build_bids_name(
                        {**transform_ents, "desc": f"iter{iteration}synthmorphItk"},
                        extension='.txt',
                    )
                    synth_inv_itk = work_dir / build_bids_name(
                        {**transform_ents, "desc": f"iter{iteration}synthmorphInverseItk"},
                        extension='.txt',
                    )
                    lta_to_itk(synth_tx, synth_itk, src=current_ref, trg=template)
                    lta_to_itk(synth_inv_tx, synth_inv_itk, src=template, trg=current_ref)
                else:
                    synth_itk = work_dir / build_bids_name(
                        {**transform_ents, "desc": f"iter{iteration}synthmorphItk"},
                        extension='.nii.gz',
                    )
                    synth_inv_itk = work_dir / build_bids_name(
                        {**transform_ents, "desc": f"iter{iteration}synthmorphInverseItk"},
                        extension='.nii.gz',
                    )
                    mri_warp_convert_to_itk(synth_tx, synth_itk, src_geom=current_ref)
                    mri_warp_convert_to_itk(synth_inv_tx, synth_inv_itk, src_geom=template)

                current_ref = synth_warped
                iteration_record["synthmorph"] = {
                    "transform": str(synth_tx),
                    "inverse_transform": str(synth_inv_tx),
                    "itk_transform": str(synth_itk),
                    "itk_inverse_transform": str(synth_inv_itk),
                    "warped": str(synth_warped),
                    "model": synth_model,
                    "register_args": str(synth_cfg.get('register_args', '') or ''),
                    "apply_args": str(synth_cfg.get('apply_args', '') or ''),
                }

            if ants_enabled:
                ants_prefix = work_dir / f"{manifest_path.stem}_iter{iteration}_ants_"
                ants_bundle = self._run_ants_registration_bundle(
                    moving_file=current_ref,
                    fixed_file=template,
                    out_prefix=ants_prefix,
                    transform_type=str(ants_cfg.get('transform_type', self.kwargs.get('transform_type', 'SyN'))),
                    interpolator=str(ants_cfg.get('apply_interpolator', 'linear')),
                    nthreads=nthreads,
                    registration_kwargs=dict(ants_cfg.get('registration_kwargs') or {}),
                )
                current_ref = ants_bundle["warped"]
                iteration_record["ants"] = {
                    "warped": str(ants_bundle["warped"]),
                    "forward": [str(p) for p in ants_bundle["forward"]],
                    "inverse": [str(p) for p in ants_bundle["inverse"]],
                    "transform_type": ants_bundle["transform_type"],
                    "registration_kwargs": self._json_safe(ants_bundle["registration_kwargs"]),
                    "apply_interpolator": str(ants_cfg.get('apply_interpolator', 'linear')),
                }

            step_cli_forward = []
            if iteration_record.get("ants", {}).get("forward"):
                step_cli_forward.extend(iteration_record["ants"]["forward"])
            if iteration_record.get("synthmorph", {}).get("itk_transform"):
                step_cli_forward.append(iteration_record["synthmorph"]["itk_transform"])
            if step_cli_forward:
                cli_forward_chain = step_cli_forward + cli_forward_chain

            step_cli_inverse = []
            if iteration_record.get("synthmorph", {}).get("itk_inverse_transform"):
                step_cli_inverse.append(iteration_record["synthmorph"]["itk_inverse_transform"])
            if iteration_record.get("ants", {}).get("inverse"):
                step_cli_inverse.extend(iteration_record["ants"]["inverse"])
            if step_cli_inverse:
                cli_inverse_chain.extend(step_cli_inverse)

            iterations_manifest.append(iteration_record)
            final_driving = current_ref

        if final_driving is None:
            raise ProcessingError("Robust iterative normalization did not produce a final driving image")

        normalized_results = {}
        normalized_results_by_model = {}

        collapsed_field = work_dir / build_bids_name(
            {**transform_ents, "desc": "robustiterativeCollapsedWarp"},
            extension='.nii.gz',
        )
        composite_h5 = work_dir / build_bids_name(
            {**transform_ents, "desc": "robustiterativeComposite"},
            extension='.h5',
        )

        if cli_forward_chain:
            ants_iface.collapse_transforms(
                reference_file=template,
                transforms=[Path(p) for p in cli_forward_chain],
                out_file=collapsed_field,
                collapse='displacement',
                nthreads=nthreads,
            )
            ants_iface.collapse_transforms(
                reference_file=template,
                transforms=[Path(p) for p in cli_forward_chain],
                out_file=composite_h5,
                collapse='composite',
                nthreads=nthreads,
            )

        manifest = {
            "tool": "robust_iterative",
            "template": str(template),
            "space_name": self.space_name,
            "space_entity": self.space_entity,
            "driving_metric": self.driving_metric,
            "iterations": iterations_manifest,
            "forward_chain_cli_order": cli_forward_chain,
            "inverse_chain_cli_order": cli_inverse_chain,
            "collapsed_forward_displacement": str(collapsed_field) if collapsed_field.exists() else None,
            "collapsed_forward_composite": str(composite_h5) if composite_h5.exists() else None,
        }
        manifest_path.write_text(json.dumps(self._json_safe(manifest), indent=2) + "\n")

        for model_name, metric_name, out_path in predicted_outputs:
            src_path = metrics_to_norm[model_name][metric_name]
            if skip and out_path.exists():
                normalized_path = out_path
            else:
                normalized_path = self._apply_robust_transform_manifest_to_image(
                    moving_file=Path(src_path),
                    out_file=out_path,
                    manifest=manifest,
                    interpolator=str((robust_cfg.get('apply') or {}).get('default_scalar_interpolator', 'linear')),
                    nthreads=nthreads,
                )

            normalized_results_by_model.setdefault(model_name, {})[metric_name] = normalized_path
            prefixed_key = f"{model_name}_{metric_name}"
            normalized_results[prefixed_key] = normalized_path
            if metric_name not in normalized_results:
                normalized_results[metric_name] = normalized_path

        context['normalized_results'] = normalized_results
        context['normalized_results_by_model'] = normalized_results_by_model
        context['normalization_transform_manifest'] = manifest_path
        return context
    def run(self, context: dict | object, output_dir: Path, **kwargs) -> dict | object:
        # Context is likely the main dictionary from ModelingWorkflow
        if not isinstance(context, dict):
            # If passed an object, we can't really do normalization unless it's a specific file?
            # Normalization expects a set of maps.
            self.logger.warning("NormalizationStep expects a context dictionary with 'modeling_results'.")
            return context

        if not self.template or not self.template.exists():
            self.logger.warning(f"Normalization skipped: Template not found at {self.template}")
            return context

        modeling_results = context.get('modeling_results', {})
        if not modeling_results and not self.include_all_metrics:
            self.logger.warning("Normalization skipped: No modeling results found.")
            return context

        # Fetch nthreads
        nthreads = kwargs.get('nthreads', self.config.get('n_cpus', 1))
        os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(nthreads)
        
        # Build full metric list (including any extra outputs not tracked in context)
        from ...io.bids import get_entities_from_path
        metrics_to_norm = {}

        def _add_metric(model_name: str, metric_name: str, path: Path):
            if not path:
                return
            if not isinstance(path, Path):
                path = Path(path)
            metrics_to_norm.setdefault(model_name, {})
            if metric_name not in metrics_to_norm[model_name]:
                metrics_to_norm[model_name][metric_name] = path

        if modeling_results:
            for model_name, metrics in modeling_results.items():
                if isinstance(metrics, dict):
                    for name, path in metrics.items():
                        _add_metric(model_name, name, path)

        if self.include_all_metrics:
            candidate_dirs = set()
            for model_metrics in metrics_to_norm.values():
                for path in model_metrics.values():
                    if path:
                        candidate_dirs.add(path.parent)

            if not candidate_dirs:
                potential_bases = []
                if (output_dir / "modeling").exists():
                    potential_bases.append(output_dir / "modeling")
                potential_bases.append(output_dir)

                for base in potential_bases:
                    for model_dir in [
                        "DTI", "DKI", "NODDI", "SANDI", "MAPMRI", "CSD",
                        "FWE_DTI", "FWDTI", "dti", "dki", "noddi", "sandi",
                        "mapmri", "csd", "fwe_dti", "fwdti"
                    ]:
                        p = base / model_dir
                        if p.exists():
                            candidate_dirs.add(p)

            for model_dir in candidate_dirs:
                for p in model_dir.glob("*.nii.gz"):
                    ents = get_entities_from_path(p)
                    model_name = ents.get("model") or model_dir.name
                    metric_name = ents.get("suffix")
                    if not metric_name:
                        name_part = p.name.replace(".nii.gz", "")
                        metric_name = name_part.split("_")[-1]
                    _add_metric(model_name, metric_name, p)

        if not metrics_to_norm:
            self.logger.warning("Normalization skipped: No modeling results found.")
            return context

        # 1. Find driving metric
        ref_path = None
        
        # Flatten results to find driving metric
        flat_metrics = {}
        for model_metrics in metrics_to_norm.values():
            flat_metrics.update(model_metrics)
        
        # Look for partial match if exact match fails? e.g. 'FA' vs 'tensor_fa'
        # Or require exact match.
        if _is_reference_target(self.driving_metric):
            current_image = context.get('current_image')
            if hasattr(current_image, 'img'):
                ref_path = current_image.img
            elif current_image:
                ref_path = current_image
        elif self.driving_metric in flat_metrics:
            ref_path = flat_metrics[self.driving_metric]
        else:
            # Try caseless?
            for k, v in flat_metrics.items():
                if k.lower() == self.driving_metric.lower():
                    ref_path = v
                    break
        
        if not ref_path:
             self.logger.warning(f"Normalization skipped: Driving metric '{self.driving_metric}' not found in results.")
             return context

        # Ensure driving metric is included for warping even if it wasn't
        # listed in modeling_results (e.g., custom metrics lists).
        ref_ents = get_entities_from_path(ref_path)
        ref_model = ref_ents.get('model') or ('Reference' if _is_reference_target(self.driving_metric) else 'Unknown')
        ref_metric = ref_ents.get('suffix')
        if not ref_metric:
             ref_metric = 'SPGRref' if _is_reference_target(self.driving_metric) else Path(ref_path).name.replace(".nii.gz", "").split("_")[-1]
        _add_metric(ref_model, ref_metric, Path(ref_path))

        # Output dir
        norm_out = output_dir.parent / "normalization" / f"space-{self.space_name}"
        if output_dir.name == 'modeling': # heuristic fix if nested
             norm_out = output_dir.parent / "normalization"
             
        norm_out.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Normalizing metrics to space-{self.space_name} (Template: {self.template.name}) using {self.tool}")
        self.logger.info(f"Driving metric: {ref_path.name}")
        ref_for_reg = Path(ref_path)
        template_for_reg = self.template
        registration_inputs_stripped = False
        
        # Check for existing outputs (Skip Logic)
        force = bool(kwargs.get('force', False))
        skip = self.config.get('skip_existing', False) and not force
        # Predict outputs.
        from ...io.bids import build_bids_name, get_entities_from_path

        all_exist = True
        predicted_outputs = self._predicted_normalized_outputs(metrics_to_norm, norm_out)

        for _, _, out_path in predicted_outputs:
            if not out_path.exists():
                all_exist = False
                    
        # Also check transforms if saving (ANTs only)
        if self.save_transforms and self.tool == 'ants':
             # Driving metric entities
             d_ents = get_entities_from_path(ref_path)
             # Affine
             d_ents.update({'space': self.space_entity, 'suffix': 'xfm', 'desc': 'affine', 'extension': '.mat', 'model': None}) # .mat usually for ANTs affine
             # Simplified name: sub-X_ses-Y_space-MNI_desc-affine_xfm.mat
             # Remove other entities?
             for k in ['acq', 'dir', 'run', 'echo', 'part']: 
                 if k in d_ents: del d_ents[k]
                 
             affine_path = norm_out / build_bids_name(d_ents)
             
             d_ents['desc'] = 'warp'
             d_ents['extension'] = '.nii.gz'
             warp_path = norm_out / build_bids_name(d_ents)
             
             if not affine_path.exists() or not warp_path.exists():
                 all_exist = False
        elif self.tool == 'synthmorph':
             d_ents = get_entities_from_path(ref_path)
             for k in ['acq', 'dir', 'run', 'echo', 'part']:
                 if k in d_ents:
                     del d_ents[k]
             d_ents['space'] = self.space_entity
             d_ents['suffix'] = 'xfm'
             d_ents['desc'] = 'synthmorph'
             ext = self._synthmorph_transform_extension(
                 self.kwargs.get('synthmorph_model'),
                 self.kwargs.get('synthmorph_transform_ext'),
             )
             tx_path = norm_out / build_bids_name(d_ents, extension=ext)
             if not tx_path.exists():
                 all_exist = False
        elif self.tool == 'robust_iterative':
             if not self._robust_manifest_path(ref_path, norm_out).exists():
                 all_exist = False
        elif self.save_transforms and self.tool not in ['ants', 'synthmorph']:
             self.logger.warning(f"Save transforms not supported for tool '{self.tool}'.")
                 
        if skip and all_exist:
             # Check timestamps
             # ref_path is the driving metric
             in_mtime = ref_path.stat().st_mtime
             # out_mtime is the first predicted output
             out_mtime = predicted_outputs[0][2].stat().st_mtime
             
             if in_mtime > out_mtime:
                  self.logger.info(f"Normalization driving metric ({ref_path.name}) is newer than output. Re-running.")
             else:
                  self.logger.info("Skipping Normalization (All outputs exist and are up-to-date).")
                  # Populate context
             return self._populate_existing_normalized_results(
                 context,
                 predicted_outputs,
                 manifest_path=self._robust_manifest_path(ref_path, norm_out) if self.tool == 'robust_iterative' else None,
             )

        if self.tool == 'robust_iterative':
            return self._run_robust_iterative_normalization(
                context=context,
                metrics_to_norm=metrics_to_norm,
                ref_path=Path(ref_path),
                norm_out=norm_out,
                nthreads=nthreads,
                force=force,
            )
        
        # 2. Registration
        tx_forward = []
        tx_inverse = []
        synthmorph_tx = None

        if self.tool in {'ants', 'synthmorph'}:
            ref_for_reg, template_for_reg, registration_inputs_stripped = prepare_registration_images(
                self.config,
                self.logger,
                Path(ref_path),
                self.template,
                norm_out,
                self.kwargs,
                nthreads,
                force=force,
            )
        
        if self.tool == 'ants':
            try:
                import ants
                mov_raw = ants.image_read(str(ref_for_reg))
                mov, _ = self._ensure_3d(mov_raw, is_driving=True)
                
                fix = ants.image_read(str(template_for_reg))
                # Fix should also be 3D
                if fix.dimension == 4:
                    fix = ants.slice_image(fix, axis=3, idx=0)

                # Registration
                # Use SyN usually? 
                # type_of_transform: 'SyN', 'SyNRA', 'Rigid', 'Affine'
                tf_type = self.kwargs.get('transform_type', 'SyN')
                
                # Check for existing transform?
                # ...
                
                # ...
                
                registration_kwargs = {
                    k: v for k, v in self.kwargs.items()
                    if k not in {
                        'save_transforms', 'include_all_metrics', 'space_entity',
                        'synthmorph_args', 'synthmorph_moving_flag',
                        'synthmorph_target_flag', 'synthmorph_output_flag',
                        'synthmorph_model', 'synthmorph_transform_ext',
                        'synthmorph_register_args',
                        'synthmorph_apply_args', 'robust_iterative',
                        'transform_type',
                    } | _ALL_SKULL_STRIP_OPTION_KEYS
                }
                reg = ants.registration(fixed=fix, moving=mov, type_of_transform=tf_type, **registration_kwargs)
                tx_forward = reg['fwdtransforms']
                
                # Save transforms if requested
                if self.save_transforms:
                    # Copy transform files to output dir with BIDS names
                    import shutil
                    from ...io.bids import build_bids_name, get_entities_from_path
                    
                    d_ents = get_entities_from_path(ref_path)
                    # Clean ents for transform
                    for k in ['acq', 'dir', 'run', 'echo', 'part', 'model']: 
                        if k in d_ents: del d_ents[k]
                    
                    d_ents['space'] = self.space_entity
                    d_ents['suffix'] = 'xfm'
                    
                    # ANTs returns [Warp, Affine] usually for SyN
                    # Check what tx_forward contains. usually paths to tmp files.
                    # 0GenericAffine.mat, 1Warp.nii.gz
                    
                    for tf_file in tx_forward:
                        tf_path = Path(tf_file)
                        if tf_path.suffix == '.mat':
                            d_ents['desc'] = 'affine'
                            # Explicitly pass extension as the function kwarg overrides it or defaults to .nii.gz
                            out_name = build_bids_name(d_ents, extension='.mat')
                            shutil.copy(tf_path, norm_out / out_name)
                        elif 'Warp' in tf_path.name and tf_path.suffix == '.gz': 
                            d_ents['desc'] = 'warp'
                            out_name = build_bids_name(d_ents, extension='.nii.gz')
                            shutil.copy(tf_path, norm_out / out_name)
                
                # Save warped driving metric
                # reg['warpedmovout']
                from ...io.bids import build_bids_name, get_entities_from_path
                driving_ents = get_entities_from_path(ref_path)
                driving_ents['space'] = self.space_entity
                if not driving_ents.get('model'):
                    driving_ents['model'] = 'Unknown'
                driving_out = norm_out / build_bids_name(driving_ents)
                try:
                    driving_raw = ants.image_read(str(ref_path))
                    driving_mov, _ = self._ensure_3d(driving_raw, is_driving=True)
                    warped_driving = ants.apply_transforms(
                        fixed=fix,
                        moving=driving_mov,
                        transformlist=tx_forward,
                        imagetype=0
                    )
                    ants.image_write(warped_driving, str(driving_out))
                except Exception as e:
                    self.logger.warning(f"Failed to normalize driving metric: {e}")
                
            except ImportError:
                self.logger.error("ANTsPy not installed.")
                return context
        elif self.tool == 'synthmorph':
            from ...interfaces.freesurfer import mri_synthmorph_apply, mri_synthmorph_register
            d_ents = get_entities_from_path(ref_path)
            for k in ['acq', 'dir', 'run', 'echo', 'part']:
                if k in d_ents:
                    del d_ents[k]
            d_ents['space'] = self.space_entity
            d_ents['suffix'] = 'xfm'
            d_ents['desc'] = 'synthmorph'
            synthmorph_model = self.kwargs.get('synthmorph_model')
            ext = self._synthmorph_transform_extension(
                synthmorph_model,
                self.kwargs.get('synthmorph_transform_ext'),
            )
            synthmorph_tx = norm_out / build_bids_name(d_ents, extension=ext)

            # Compute driving metric output path so register can write it directly.
            driving_ents = get_entities_from_path(ref_path)
            driving_ents['space'] = self.space_entity
            if not driving_ents.get('model'):
                driving_ents['model'] = 'Unknown'
            if not driving_ents.get('suffix'):
                driving_name_part = Path(ref_path).name.replace(".nii.gz", "")
                driving_ents['suffix'] = driving_name_part.split("_")[-1]
            driving_out = norm_out / build_bids_name(driving_ents)

            try:
                mri_synthmorph_register(
                    moving=ref_for_reg,
                    target=template_for_reg,
                    transform_out=synthmorph_tx,
                    output_image=None if registration_inputs_stripped else driving_out,
                    model=synthmorph_model,
                    extra_args=self.kwargs.get('synthmorph_register_args', ''),
                    overwrite=not skip
                )
            except Exception as e:
                self.logger.warning(f"SynthMorph register failed: {e}")
                return context

            # Fallback: if register didn't produce the warped driving image, apply the transform.
            if registration_inputs_stripped or not driving_out.exists():
                try:
                    mri_synthmorph_apply(
                        moving=ref_path,
                        target=template_for_reg,
                        transform_in=synthmorph_tx,
                        out_file=driving_out,
                        extra_args=self.kwargs.get('synthmorph_apply_args', ''),
                        overwrite=not skip
                    )
                except Exception as e:
                    self.logger.warning(f"Failed to normalize driving metric: {e}")
        else:
            self.logger.warning(f"Normalization tool '{self.tool}' not implemented.")
            return context
             
        # 3. Apply to all metrics
        from ...io.bids import build_bids_name, get_entities_from_path
        normalized_results = {}
        normalized_results_by_model = {}
        
        for model_name, metrics in metrics_to_norm.items():
            for name, path in metrics.items():
                 ents = get_entities_from_path(path)
                 ents['space'] = self.space_entity
                 if 'model' not in ents or not ents['model']:
                      ents['model'] = model_name
                 
                 new_name = build_bids_name(ents)
                 out_path = norm_out / new_name
                 
                 if self.tool == 'ants':
                      try:
                          img_raw = ants.image_read(str(path))
                          img, is_4d = self._ensure_3d(img_raw, is_driving=False)
                          
                          imagetype = 3 if is_4d else 0
                          
                          warped = ants.apply_transforms(
                              fixed=fix, 
                              moving=img, 
                              transformlist=tx_forward,
                              imagetype=imagetype
                          )
                          ants.image_write(warped, str(out_path))
                      except Exception as e:
                          self.logger.warning(f"Failed to normalize {name}: {e}")
                          continue
                 elif self.tool == 'synthmorph':
                      from ...interfaces.freesurfer import mri_synthmorph_apply
                      if not synthmorph_tx or not Path(synthmorph_tx).exists():
                          self.logger.warning("SynthMorph transform missing; skipping apply.")
                          continue
                      try:
                          mri_synthmorph_apply(
                              moving=path,
                              target=template_for_reg,
                              transform_in=synthmorph_tx,
                              out_file=out_path,
                              extra_args=self.kwargs.get('synthmorph_apply_args', ''),
                              overwrite=not skip
                          )
                      except Exception as e:
                          self.logger.warning(f"Failed to normalize {name}: {e}")
                          continue
                 
                 normalized_results_by_model.setdefault(model_name, {})[name] = out_path
                 prefixed_key = f"{model_name}_{name}"
                 normalized_results[prefixed_key] = out_path
                 if name not in normalized_results:
                      normalized_results[name] = out_path
        
        context['normalized_results'] = normalized_results
        context['normalized_results_by_model'] = normalized_results_by_model
        return context


def apply_robust_transform_manifest(
    manifest: dict[str, Any] | Path,
    moving_file: Path,
    out_file: Path,
    interpolator: str = "linear",
    nthreads: int = 1,
) -> Path:
    from ...interfaces import ants as ants_iface
    from ...interfaces.freesurfer import mri_synthmorph_apply

    if not isinstance(manifest, dict):
        manifest = json.loads(Path(manifest).read_text())

    template = Path(manifest["template"])
    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    collapsed_forward = manifest.get("collapsed_forward_displacement") or manifest.get("collapsed_forward_composite")
    if collapsed_forward and Path(collapsed_forward).exists():
        from ...interfaces import ants as ants_iface

        apply_kwargs = {}
        if NormalizationStep._is_4d_image(Path(moving_file)):
            apply_kwargs["imagetype"] = 3
        ants_iface.apply_transforms(
            fixed_file=template,
            moving_file=moving_file,
            out_file=out_file,
            transforms=[Path(collapsed_forward)],
            interpolator=NormalizationStep._resolve_ants_interpolator(interpolator),
            nthreads=nthreads,
            **apply_kwargs,
        )
        return out_file

    current_path = Path(moving_file)
    temp_dir = out_file.parent / "robust_iterative_apply"
    temp_dir.mkdir(parents=True, exist_ok=True)

    for idx, iteration in enumerate(manifest.get("iterations", []), start=1):
        synth = iteration.get("synthmorph")
        if synth and synth.get("transform"):
            synth_out = temp_dir / f"{out_file.stem}_iter{idx}_synth.nii.gz"
            mri_synthmorph_apply(
                moving=current_path,
                target=template,
                transform_in=Path(synth["transform"]),
                out_file=synth_out,
                extra_args=str(synth.get("apply_args", "") or ""),
                overwrite=True,
            )
            current_path = synth_out

        ants_step = iteration.get("ants")
        if ants_step and ants_step.get("forward"):
            ants_out = out_file if idx == len(manifest.get("iterations", [])) else temp_dir / f"{out_file.stem}_iter{idx}_ants.nii.gz"
            apply_kwargs = {}
            if NormalizationStep._is_4d_image(current_path):
                apply_kwargs["imagetype"] = 3
            ants_iface.apply_transforms(
                fixed_file=template,
                moving_file=current_path,
                out_file=ants_out,
                transforms=[Path(p) for p in ants_step["forward"]],
                interpolator=NormalizationStep._resolve_ants_interpolator(interpolator),
                nthreads=nthreads,
                **apply_kwargs,
            )
            current_path = ants_out

    if current_path != out_file:
        shutil.copyfile(current_path, out_file)
    return out_file
