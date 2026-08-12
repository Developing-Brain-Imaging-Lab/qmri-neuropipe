from pathlib import Path
from typing import List, Optional
import nibabel as nib

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
            if out_path.exists() and not force:
                try: 
                    check = nib.load(out_path)
                    if is_4d and (len(check.shape) != 4 or check.shape[3] < 2):
                         self.logger.warning(f"Existing output {out_name} appears truncated. Re-running.")
                    else:
                         self.logger.info(f"Skipping Motion Correction (Exists): {out_name}")
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
                     'interpolation', 'interpolator', 'args', 'extra_args'
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
                 if k not in {'dof', 'cost', 'extra_args', 'args'} | _ALL_SKULL_STRIP_OPTION_KEYS
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
