import json
from pathlib import Path
from typing import List, Optional

import nibabel as nib
import numpy as np

from ...core import BaseProcessingStep
from ...core.types import ImageFile
from ...core.utils import ensure_dir, get_nifti_stem
from ...io.bids import build_bids_name
from ...interfaces import ants
from ...utils.relax_params import _extract_bids_param
from ..common.json_metadata import copy_json_with_metadata
from ..common.registration import prepare_registration_images, _ALL_SKULL_STRIP_OPTION_KEYS

class RelaxometryMotionCorrectionStep(BaseProcessingStep):
    """
    Motion correction for relaxometry SPGR, SSFP, and IR-SPGR data.

    The workflow normally supplies a materialized shared SPGR reference. The
    highest-flip-angle fallback is retained for direct callers.
    """
    

    def __init__(self, config, logger, provenance, method="ants", options: dict = None):
        super().__init__(config, logger, provenance)
        self.method = method
        self.options = options or {}

    @staticmethod
    def normalize_tracker_module(step_name: str) -> str:
        return "Motion_Correction"

    @staticmethod
    def _preprocessed_entities(img: ImageFile, modality: Optional[str]) -> dict:
        """Return output entities without repeating a modality in ``desc``."""
        entities = dict(img.entities)
        acquisition = str(entities.get("acq", "") or "").strip().lower()
        acquisition_key = acquisition.replace("-", "").replace("_", "")

        if acquisition_key in {"spgr", "ssfp", "irspgr"}:
            entities["desc"] = "preproc"
            return entities

        fallback_label = str(modality or acquisition or "moco").upper()
        entities["desc"] = f"{fallback_label}preproc"
        return entities

    @staticmethod
    def _normalize_ants_transform(transform: str) -> str:
        """
        Normalize legacy ANTs shell shorthand to antspy transform names.
        """
        mapping = {
            "r": "Rigid",
            "rigid": "Rigid",
            "a": "Affine",
            "affine": "Affine",
            "s": "SyN",
            "syn": "SyN",
            "sr": "SyNRA",
            "synra": "SyNRA",
            "b": "SyN",
            "br": "SyNRA",
            "bo": "SyNOnly",
            "so": "SyNOnly",
            "t": "Translation",
            "translation": "Translation",
        }
        return mapping.get(str(transform).strip().lower(), str(transform))

    @staticmethod
    def _normalize_fsl_interpolator(interpolator: str) -> str:
        """Translate shared interpolation names to FLIRT's ``-interp`` values."""
        value = str(interpolator or "trilinear").strip()
        aliases = {
            "linear": "trilinear",
            "nearest": "nearestneighbour",
            "nearestneighbor": "nearestneighbour",
            "nearestneighbour": "nearestneighbour",
            "bspline": "spline",
            "cubic": "spline",
            "spline": "spline",
            "sinc": "sinc",
            "lanczos": "sinc",
            "lanczoswindowedsinc": "sinc",
        }
        return aliases.get(value.lower(), value)

    @staticmethod
    def _cleanup_ants_outputs(out_prefix: Path) -> None:
        candidates = list(out_prefix.parent.glob(f"{out_prefix.name}*"))
        for path in sorted(candidates, key=lambda item: len(item.parts), reverse=True):
            try:
                if path.is_dir():
                    import shutil
                    shutil.rmtree(path, ignore_errors=True)
                elif path.exists():
                    path.unlink()
            except Exception:
                pass

    def _ssfp_two_stage_config(self) -> dict:
        raw = self.options.get("ssfp_two_stage", {})
        if isinstance(raw, bool):
            return {"enabled": raw}
        return dict(raw or {})

    def _ssfp_two_stage_enabled(self, modality: Optional[str]) -> bool:
        return (
            str(modality or "").strip().upper() == "SSFP"
            and bool(self._ssfp_two_stage_config().get("enabled", False))
        )

    def _aligned_templates_config(self) -> dict:
        raw = self._ssfp_two_stage_config().get("aligned_templates", True)
        if isinstance(raw, bool):
            return {"enabled": raw}
        config = dict(raw or {})
        config.setdefault("enabled", True)
        return config

    def _aligned_templates_enabled(self) -> bool:
        return bool(self._aligned_templates_config().get("enabled", True))

    def _expected_ssfp_strategy(self) -> str:
        if self._aligned_templates_enabled():
            return "ssfp_two_stage_aligned_templates"
        return "ssfp_two_stage"

    def _stage_options(self, stage: str) -> dict:
        """Merge shared motion options with optional SSFP stage overrides."""
        options = {
            key: value
            for key, value in self.options.items()
            if key != "ssfp_two_stage"
        }
        stage_overrides = self._ssfp_two_stage_config().get(
            f"{stage}_options", {}
        )
        if isinstance(stage_overrides, dict):
            options.update(stage_overrides)
        else:
            stage_overrides = {}

        if stage == "cross" and self._aligned_templates_enabled():
            if not ({"transform_type", "type_of_transform"} & options.keys()):
                options["transform_type"] = "DenseRigid"
            options.setdefault("aff_metric", "mattes")
            # Cross-modality templates benefit from ANTs' center-of-mass
            # initialization even when the shared within-stage setting uses
            # Identity. An explicit cross-stage value still takes precedence.
            if "initial_transform" not in stage_overrides:
                options["initial_transform"] = None
        return options

    @staticmethod
    def _build_aligned_template(
        sources,
        output: Path,
        *,
        mode: str = "median",
        normalize: bool = True,
        index: int = 0,
    ) -> Path:
        """Create a normalized 3D registration template from aligned images."""
        if isinstance(sources, (str, Path, ImageFile)):
            sources = [sources]

        volumes = []
        template_affine = None
        template_header = None
        template_shape = None
        for source in sources:
            source_path = Path(source.img if isinstance(source, ImageFile) else source)
            nii = nib.load(str(source_path))
            if len(nii.shape) not in {3, 4}:
                raise ValueError(
                    f"Registration-template input must be 3D or 4D; got {nii.shape}."
                )
            data = np.asanyarray(nii.dataobj, dtype=np.float32)
            source_volumes = [data] if data.ndim == 3 else [
                data[..., volume_index] for volume_index in range(data.shape[3])
            ]
            if template_shape is None:
                template_shape = source_volumes[0].shape
                template_affine = nii.affine
                template_header = nii.header.copy()
            elif source_volumes[0].shape != template_shape or not np.allclose(
                nii.affine, template_affine, rtol=1e-5, atol=1e-5
            ):
                raise ValueError(
                    "Aligned registration-template inputs must share a voxel grid."
                )
            volumes.extend(source_volumes)

        if not volumes:
            raise ValueError("Cannot build a registration template without images.")

        mode = str(mode or "median").strip().lower()
        if mode == "index":
            if index < 0 or index >= len(volumes):
                raise ValueError(
                    f"Registration-template index {index} is out of range for "
                    f"{len(volumes)} volumes."
                )
            reference = volumes[index]
        else:
            normalized_volumes = []
            for volume in volumes:
                template_volume = volume.copy()
                if normalize:
                    valid = np.isfinite(template_volume) & (template_volume > 0)
                    scale = (
                        float(np.median(template_volume[valid]))
                        if np.any(valid)
                        else 0.0
                    )
                    if scale > 0:
                        template_volume /= scale
                normalized_volumes.append(template_volume)
            stacked = np.stack(normalized_volumes, axis=3)
            if mode == "median":
                reference = np.nanmedian(stacked, axis=3)
            elif mode == "mean":
                reference = np.nanmean(stacked, axis=3)
            else:
                raise ValueError(
                    "Registration-template mode must be 'median', 'mean', "
                    f"or 'index'; got {mode!r}."
                )

        reference = np.nan_to_num(reference, copy=False)
        template_header.set_data_shape(reference.shape)
        output.parent.mkdir(parents=True, exist_ok=True)
        nib.save(
            nib.Nifti1Image(reference, template_affine, template_header),
            str(output),
        )
        return output

    @staticmethod
    def _build_ssfp_reference(
        source: Path,
        output: Path,
        *,
        mode: str = "median",
        normalize: bool = True,
        index: int = 0,
    ) -> Path:
        """Create the initial 3D SSFP within-modality registration target."""
        nii = nib.load(str(source))
        if len(nii.shape) != 4 or int(nii.shape[3]) < 2:
            raise ValueError(
                "Two-stage SSFP motion correction requires a 4D SSFP series "
                f"with at least two volumes; got {nii.shape}."
            )
        return RelaxometryMotionCorrectionStep._build_aligned_template(
            [source],
            output,
            mode=mode,
            normalize=normalize,
            index=index,
        )

    def _estimate_ants_transforms(
        self,
        moving: Path,
        fixed: Path,
        out_prefix: Path,
        options: dict,
    ) -> list[Path]:
        """Estimate an ANTs transform while retaining it for composition."""
        nthreads = int(options.get("nthreads", options.get("threads", 4)))
        moving_for_reg, fixed_for_reg, _ = prepare_registration_images(
            self.config,
            self.logger,
            Path(moving),
            Path(fixed),
            out_prefix.parent,
            options,
            nthreads,
            force=True,
        )
        transform_type = self._normalize_ants_transform(
            options.get(
                "transform_type", options.get("type_of_transform", "Rigid")
            )
        )
        interpolator = options.get(
            "interpolation", options.get("interpolator", "linear")
        )
        registration_kwargs = {
            key: value
            for key, value in options.items()
            if key
            not in {
                "transform_type",
                "type_of_transform",
                "threads",
                "nthreads",
                "interpolation",
                "interpolator",
                "args",
                "extra_args",
            }
            | _ALL_SKULL_STRIP_OPTION_KEYS
        }
        _, transforms = ants.registration(
            fixed_file=fixed_for_reg,
            moving_file=moving_for_reg,
            out_prefix=out_prefix,
            transform_type=transform_type,
            interpolator=interpolator,
            nthreads=nthreads,
            **registration_kwargs,
        )
        return [Path(transform) for transform in transforms]

    def _run_two_stage_ssfp(
        self,
        volumes: List[Path],
        ssfp_reference: Path,
        spgr_reference: Path,
        split_dir: Path,
        cross_reference: Optional[Path] = None,
    ) -> List[Path]:
        """Compose volume-to-SSFP and SSFP-to-SPGR transforms per volume."""
        if self.method != "ants":
            raise ValueError(
                "Two-stage SSFP motion correction currently requires method: ants."
            )

        split_dir.mkdir(parents=True, exist_ok=True)
        within_options = self._stage_options("within")
        cross_options = self._stage_options("cross")
        cross_prefix = split_dir / "ssfp_to_spgr_ants_"
        cleanup_prefixes = [cross_prefix]
        try:
            within_transform_sets = []
            for index, volume in enumerate(volumes):
                within_prefix = split_dir / f"vol{index:04d}_to_ssfp_ants_"
                cleanup_prefixes.append(within_prefix)
                within_transforms = self._estimate_ants_transforms(
                    Path(volume),
                    ssfp_reference,
                    within_prefix,
                    within_options,
                )
                within_transform_sets.append(within_transforms)

            cross_moving = ssfp_reference
            if self._aligned_templates_enabled():
                aligned_volumes = []
                for index, (volume, within_transforms) in enumerate(
                    zip(volumes, within_transform_sets)
                ):
                    aligned_output = split_dir / f"vol{index:04d}_within_aligned.nii.gz"
                    ants.apply_transforms(
                        fixed_file=ssfp_reference,
                        moving_file=volume,
                        out_file=aligned_output,
                        transforms=within_transforms,
                        interpolator=within_options.get(
                            "interpolation",
                            within_options.get("interpolator", "linear"),
                        ),
                        nthreads=int(
                            within_options.get(
                                "nthreads", within_options.get("threads", 4)
                            )
                        ),
                    )
                    aligned_volumes.append(aligned_output)

                template_cfg = self._aligned_templates_config()
                cross_moving = split_dir / "ssfp_aligned_template.nii.gz"
                self._build_aligned_template(
                    aligned_volumes,
                    cross_moving,
                    mode=template_cfg.get("mode", "median"),
                    normalize=bool(template_cfg.get("normalize", True)),
                    index=int(template_cfg.get("index", 0)),
                )

            cross_fixed = Path(cross_reference or spgr_reference)
            cross_transforms = self._estimate_ants_transforms(
                cross_moving,
                cross_fixed,
                cross_prefix,
                cross_options,
            )

            corrected: List[Path] = []
            for index, (volume, within_transforms) in enumerate(
                zip(volumes, within_transform_sets)
            ):
                output = split_dir / f"vol{index:04d}_moco.nii.gz"
                # ANTs lists the later transform first: original volume ->
                # SSFP reference -> SPGR reference.
                composed_transforms = [*cross_transforms, *within_transforms]
                ants.apply_transforms(
                    fixed_file=spgr_reference,
                    moving_file=volume,
                    out_file=output,
                    transforms=composed_transforms,
                    interpolator=cross_options.get(
                        "interpolation",
                        cross_options.get("interpolator", "linear"),
                    ),
                    nthreads=int(
                        cross_options.get(
                            "nthreads", cross_options.get("threads", 4)
                        )
                    ),
                )
                corrected.append(output)
            return corrected
        finally:
            for prefix in cleanup_prefixes:
                self._cleanup_ants_outputs(prefix)
        
    def run(self, 
            images: List[ImageFile], 
            output_dir: Path, 
            force: bool = False,
            reference_image: Optional[ImageFile] = None,
            modality: Optional[str] = None,
            cross_reference_image: Optional[ImageFile] = None,
           ) -> List[ImageFile]:
           

        output_dir = ensure_dir(output_dir)
        processed_outputs = []

        # 1. Identify Reference (Max Flip Angle)
        if not reference_image:
             max_fa = -1.0
             ref_img_candidate = None
             for img in images:
                 fa = _extract_bids_param(img, "FlipAngle", 0.0)
                 if isinstance(fa, list): fa = max(fa) if fa else 0.0
                 if float(fa) > max_fa:
                     max_fa = float(fa)
                     ref_img_candidate = img
             
             if not ref_img_candidate:
                 ref_img_candidate = images[0]
                 
             reference_image = ref_img_candidate
             self.logger.info(f"Selected reference image (FA={max_fa}): {reference_image.img.name}")
             
        # Ensure Reference is 3D
        ref_path = Path(reference_image.img)
        try:
            ref_nii = nib.load(ref_path)
            if len(ref_nii.shape) == 4 and ref_nii.shape[3] > 1:
                 temp_ref = output_dir / "temp_ref.nii.gz"
                 ref_data = ref_nii.dataobj[..., 0]
                 nib.save(
                     nib.Nifti1Image(ref_data, ref_nii.affine, ref_nii.header.copy()),
                     temp_ref,
                 )
                 ref_path = temp_ref
        except Exception as e:
            self.logger.warning(f"Could not check dimensions of ref: {e}")

        cross_ref_path = (
            Path(cross_reference_image.img)
            if cross_reference_image is not None
            else ref_path
        )

        # 2. Process Inputs
        for img in images:
            # Check if 4D
            is_4d = False
            try:
                nii = nib.load(img.img)
                if len(nii.shape) == 4 and nii.shape[3] > 1:
                    is_4d = True
            except:
                pass
            
            # acq-SPGR/acq-SSFP already identifies the sequence, so the
            # canonical derivative is simply desc-preproc. Retain the legacy
            # modality-qualified fallback when no recognized acq is present.
            ents = self._preprocessed_entities(img, modality)
            
            out_name = build_bids_name(ents)
            out_path = output_dir / out_name
            out_json = out_path.with_suffix("").with_suffix(".json")
            
            # Check if exists and valid
            use_two_stage_ssfp = is_4d and self._ssfp_two_stage_enabled(modality)
            if out_path.exists() and not force:
                try: 
                    check = nib.load(out_path)
                    if is_4d and (len(check.shape) != 4 or check.shape[3] < 2):
                         self.logger.warning(f"Existing output {out_name} appears truncated. Re-running.")
                    else:
                         existing_metadata = {}
                         if out_json.exists():
                             existing_metadata = json.loads(out_json.read_text())
                         existing_strategy = existing_metadata.get(
                             "MotionCorrection", {}
                         ).get("strategy")
                         expected_strategy = (
                             self._expected_ssfp_strategy()
                             if use_two_stage_ssfp
                             else None
                         )
                         if existing_strategy != expected_strategy:
                             self.logger.info(
                                 "Motion-correction strategy changed for %s; re-running.",
                                 out_name,
                             )
                             raise ValueError("motion-correction strategy changed")
                         self.logger.info(f"Skipping Motion Correction (Exists): {out_name}")
                         if not out_json.exists():
                             copy_json_with_metadata(getattr(img, "json", None), out_json)
                         result_json = out_json if out_json.exists() else getattr(img, "json", None)
                         processed_outputs.append(ImageFile(img=out_path, entities=ents, json=result_json))
                         continue
                except:
                    pass 

            sequence_label = str(
                modality or img.entities.get("acq") or "Relaxometry"
            ).upper()
            self.logger.info(
                "Processing %s Motion Correction for: %s",
                sequence_label,
                img.img.name,
            )
            
            if is_4d:
                from ...interfaces.fsl import split, merge
                self.logger.info(f"  Input is 4D. Splitting and registering {nii.shape[3]} volumes...")
                
                split_dir = output_dir / f"temp_split_{img.img.stem}"
                split_dir.mkdir(exist_ok=True)
                split_prefix = split_dir / "vol"
                
                vols = split(img.img, split_prefix)
                
                if use_two_stage_ssfp:
                    two_stage_cfg = self._ssfp_two_stage_config()
                    ssfp_ref = split_dir / "ssfp_reference.nii.gz"
                    self._build_ssfp_reference(
                        Path(img.img),
                        ssfp_ref,
                        mode=two_stage_cfg.get("reference_mode", "median"),
                        normalize=bool(two_stage_cfg.get("normalize", True)),
                        index=int(two_stage_cfg.get("reference_index", 0)),
                    )
                    self.logger.info(
                        "  Running two-stage SSFP motion correction via %s",
                        ssfp_ref.name,
                    )
                    corrected_vols = self._run_two_stage_ssfp(
                        vols,
                        ssfp_ref,
                        ref_path,
                        split_dir,
                        cross_reference=cross_ref_path,
                    )
                else:
                    corrected_vols = []
                    for i, vol in enumerate(vols):
                        vol_out = split_dir / f"vol{i:04d}_moco.nii.gz"
                        self._register(vol, ref_path, vol_out)
                        corrected_vols.append(vol_out)
                    
                merge(corrected_vols, out_path, dimension='t')
                
                import shutil
                shutil.rmtree(split_dir)
                
            else:
                self._register(img.img, ref_path, out_path)

            copy_json_with_metadata(getattr(img, "json", None), out_json)
            if use_two_stage_ssfp:
                metadata = json.loads(out_json.read_text()) if out_json.exists() else {}
                two_stage_cfg = self._ssfp_two_stage_config()
                aligned_templates_cfg = self._aligned_templates_config()
                metadata["MotionCorrection"] = {
                    "strategy": self._expected_ssfp_strategy(),
                    "ssfp_reference_mode": two_stage_cfg.get(
                        "reference_mode", "median"
                    ),
                    "ssfp_reference_normalized": bool(
                        two_stage_cfg.get("normalize", True)
                    ),
                    "transform_application": "composed_single_resampling",
                }
                if self._aligned_templates_enabled():
                    metadata["MotionCorrection"].update(
                        {
                            "cross_modality_templates": "within_aligned",
                            "template_mode": aligned_templates_cfg.get(
                                "mode", "median"
                            ),
                            "template_normalized": bool(
                                aligned_templates_cfg.get("normalize", True)
                            ),
                        }
                    )
                with out_json.open("w") as f:
                    json.dump(metadata, f, indent=2)
                    f.write("\n")
            result_json = out_json if out_json.exists() else getattr(img, "json", None)
                
            processed_outputs.append(ImageFile(img=out_path, entities=ents, json=result_json))
            
        # Cleanup temp ref if it was created
        if ref_path.name == "temp_ref.nii.gz" and ref_path.exists():
            try:
                ref_path.unlink()
            except Exception as e:
                self.logger.warning(f"Failed to remove temp ref: {e}")
            
        return processed_outputs

    def _register(self, in_file, ref_file, out_file):
        """Helper to run registration."""
        nthreads = int(self.options.get('nthreads', self.options.get('threads', 4)))
        moving_for_reg, ref_for_reg, registration_inputs_stripped = prepare_registration_images(
            self.config,
            self.logger,
            Path(in_file),
            Path(ref_file),
            Path(out_file).parent,
            self.options,
            nthreads,
            force=True,
        )
        if self.method == 'ants':
             transform_type = self._normalize_ants_transform(
                 self.options.get('transform_type', self.options.get('type_of_transform', 'Rigid'))
             )
             interpolator = self.options.get('interpolation', self.options.get('interpolator', 'linear'))
             out_prefix = out_file.parent / f"{get_nifti_stem(out_file)}_ants_"
             registration_kwargs = {
                 k: v for k, v in self.options.items()
                 if k not in {
                     'transform_type', 'type_of_transform', 'threads', 'nthreads',
                     'interpolation', 'interpolator', 'args', 'extra_args',
                     'ssfp_two_stage'
                 } | _ALL_SKULL_STRIP_OPTION_KEYS
             }
             ignored_shell_args = self.options.get('args') or self.options.get('extra_args')
             if ignored_shell_args:
                 self.logger.warning(
                     "Ignoring relaxometry motion-correction ANTs shell arguments for antspy registration: %s",
                     ignored_shell_args,
                 )

             warped, transforms = ants.registration(
                 fixed_file=ref_for_reg,
                 moving_file=moving_for_reg,
                 out_prefix=out_prefix,
                 transform_type=transform_type,
                 interpolator=interpolator,
                 nthreads=nthreads,
                 **registration_kwargs,
             )
             if registration_inputs_stripped:
                 ants.apply_transforms(
                     fixed_file=ref_for_reg,
                     moving_file=in_file,
                     out_file=out_file,
                     transforms=transforms,
                     interpolator=interpolator,
                     nthreads=nthreads,
                 )
             else:
                 warped_path = Path(warped)
                 if warped_path != Path(out_file):
                     warped_path.replace(out_file)
             self._cleanup_ants_outputs(out_prefix)

        elif self.method == 'fsl':
             from ...interfaces.fsl import flirt
             mat_file = Path(out_file).parent / f"{get_nifti_stem(out_file)}.mat"
             flirt_out = Path(out_file)
             if registration_inputs_stripped:
                 flirt_out = Path(out_file).parent / f"{get_nifti_stem(out_file)}_registration_estimate.nii.gz"
             flirt_kwargs = {
                 'in_file': moving_for_reg, 'ref_file': ref_for_reg, 'out_file': flirt_out,
                 'omat': mat_file, 'dof': self.options.get('dof', 6)
             }
             if 'cost' in self.options:
                 flirt_kwargs['cost'] = self.options['cost']

             interpolator = self._normalize_fsl_interpolator(
                 self.options.get(
                     'interpolation', self.options.get('interpolator', 'trilinear')
                 )
             )
             extra_args = str(
                 self.options.get('extra_args', self.options.get('args', '')) or ''
             ).strip()
             if '-interp ' not in extra_args:
                 extra_args = f"{extra_args} -interp {interpolator}".strip()
             flirt_kwargs['extra_args'] = extra_args

             ants_only_options = {
                 'transform_type', 'type_of_transform', 'threads', 'nthreads',
                 'interpolation', 'interpolator', 'aff_metric', 'aff_sampling',
                 'aff_random_sampling_rate', 'aff_iterations',
                 'aff_shrink_factors', 'aff_smoothing_sigmas',
                 'initial_transform', 'smoothing_in_mm', 'random_seed',
                 'write_composite_transform', 'restrict_transformation',
                 'singleprecision', 'use_legacy_histogram_matching',
                 'mask', 'moving_mask', 'mask_all_stages', 'grad_step',
                 'flow_sigma', 'total_sigma', 'syn_metric', 'syn_sampling',
                 'reg_iterations', 'multivariate_extras',
                 'rotation_search',
             }
             extra_opts = {
                 k: v for k, v in self.options.items()
                 if k not in {
                     'dof', 'cost', 'extra_args', 'args', 'ssfp_two_stage'
                 } | ants_only_options | _ALL_SKULL_STRIP_OPTION_KEYS
             }
             if extra_opts:
                 flirt_kwargs['extra_opts'] = extra_opts
             flirt(**flirt_kwargs)
             if registration_inputs_stripped:
                 flirt(
                     in_file=in_file,
                     ref_file=ref_for_reg,
                     out_file=out_file,
                     extra_args=f"-applyxfm -init {mat_file} -interp {interpolator}",
                 )


# Compatibility alias for external imports. Instances report the new,
# modality-neutral class name in logs and provenance.
SPGRMotionCorrectionStep = RelaxometryMotionCorrectionStep
