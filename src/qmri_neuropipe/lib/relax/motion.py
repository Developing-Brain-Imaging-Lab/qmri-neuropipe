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
        return options

    @staticmethod
    def _build_ssfp_reference(
        source: Path,
        output: Path,
        *,
        mode: str = "median",
        normalize: bool = True,
        index: int = 0,
    ) -> Path:
        """Create a 3D SSFP reference without modifying modeling intensities."""
        nii = nib.load(str(source))
        if len(nii.shape) != 4 or int(nii.shape[3]) < 2:
            raise ValueError(
                "Two-stage SSFP motion correction requires a 4D SSFP series "
                f"with at least two volumes; got {nii.shape}."
            )

        data = np.asanyarray(nii.dataobj, dtype=np.float32)
        mode = str(mode or "median").strip().lower()
        if mode == "index":
            if index < 0 or index >= data.shape[3]:
                raise ValueError(
                    f"SSFP reference index {index} is out of range for "
                    f"{data.shape[3]} volumes."
                )
            reference = data[..., index]
        else:
            reference_data = data.copy()
            if normalize:
                for volume_index in range(reference_data.shape[3]):
                    volume = reference_data[..., volume_index]
                    valid = np.isfinite(volume) & (volume > 0)
                    scale = float(np.median(volume[valid])) if np.any(valid) else 0.0
                    if scale > 0:
                        reference_data[..., volume_index] = volume / scale
            if mode == "median":
                reference = np.nanmedian(reference_data, axis=3)
            elif mode == "mean":
                reference = np.nanmean(reference_data, axis=3)
            else:
                raise ValueError(
                    "SSFP two-stage reference_mode must be 'median', 'mean', "
                    f"or 'index'; got {mode!r}."
                )

        reference = np.nan_to_num(reference, copy=False)
        header = nii.header.copy()
        header.set_data_shape(reference.shape)
        output.parent.mkdir(parents=True, exist_ok=True)
        nib.save(nib.Nifti1Image(reference, nii.affine, header), str(output))
        return output

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
            cross_transforms = self._estimate_ants_transforms(
                ssfp_reference,
                spgr_reference,
                cross_prefix,
                cross_options,
            )
            corrected: List[Path] = []
            for index, volume in enumerate(volumes):
                within_prefix = split_dir / f"vol{index:04d}_to_ssfp_ants_"
                cleanup_prefixes.append(within_prefix)
                within_transforms = self._estimate_ants_transforms(
                    Path(volume),
                    ssfp_reference,
                    within_prefix,
                    within_options,
                )
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
            modality: Optional[str] = None
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
                         existing_two_stage = (
                             existing_metadata.get("MotionCorrection", {}).get("strategy")
                             == "ssfp_two_stage"
                         )
                         if use_two_stage_ssfp != existing_two_stage:
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
                metadata["MotionCorrection"] = {
                    "strategy": "ssfp_two_stage",
                    "ssfp_reference_mode": two_stage_cfg.get(
                        "reference_mode", "median"
                    ),
                    "ssfp_reference_normalized": bool(
                        two_stage_cfg.get("normalize", True)
                    ),
                    "transform_application": "composed_single_resampling",
                }
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

             extra_args = self.options.get('extra_args', self.options.get('args', ''))
             if extra_args:
                 flirt_kwargs['extra_args'] = extra_args

             extra_opts = {
                 k: v for k, v in self.options.items()
                 if k not in {'dof', 'cost', 'extra_args', 'args', 'ssfp_two_stage'} | _ALL_SKULL_STRIP_OPTION_KEYS
             }
             if extra_opts:
                 flirt_kwargs['extra_opts'] = extra_opts
             flirt(**flirt_kwargs)
             if registration_inputs_stripped:
                 flirt(
                     in_file=in_file,
                     ref_file=ref_for_reg,
                     out_file=out_file,
                     extra_args=f"-applyxfm -init {mat_file} -interp {self.options.get('interpolation', 'trilinear')}",
                 )


# Compatibility alias for external imports. Instances report the new,
# modality-neutral class name in logs and provenance.
SPGRMotionCorrectionStep = RelaxometryMotionCorrectionStep
