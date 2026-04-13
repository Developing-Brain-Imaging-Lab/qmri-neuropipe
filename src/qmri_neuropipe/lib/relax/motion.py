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

class SPGRMotionCorrectionStep(BaseProcessingStep):
    """
    Motion Correction for VFA (SPGR/SSFP) data.
    Registers all volumes to the volume with the highest Flip Angle (highest SNR).
    """
    

    def __init__(self, config, logger, provenance, method="ants", options: dict = None):
        super().__init__(config, logger, provenance)
        self.method = method
        self.options = options or {}

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
            
            # Prepare Output Name
            ents = dict(img.entities)
            
            # Naming Logic: Force desc to {Modality}preproc
            # e.g. sub-01_acq-spgr_desc-SPGRpreproc_VFA.nii.gz
            if modality:
                acq_label = modality.upper()
            else:
                acq_label = ents.get('acq', '').upper()
                
            if not acq_label: acq_label = "Moco" # Fallback
            
            new_desc = f"{acq_label}preproc"
            ents['desc'] = new_desc
            
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

            self.logger.info(f"Processing Motion Correction for: {img.img.name}")
            
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
        if self.method == 'ants':
             nthreads = int(self.options.get('nthreads', self.options.get('threads', 4)))
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
                 }
             }
             ignored_shell_args = self.options.get('args') or self.options.get('extra_args')
             if ignored_shell_args:
                 self.logger.warning(
                     "Ignoring relaxometry motion-correction ANTs shell arguments for antspy registration: %s",
                     ignored_shell_args,
                 )

             warped, _ = ants.registration(
                 fixed_file=ref_file,
                 moving_file=in_file,
                 out_prefix=out_prefix,
                 transform_type=transform_type,
                 interpolator=interpolator,
                 nthreads=nthreads,
                 **registration_kwargs,
             )
             warped_path = Path(warped)
             if warped_path != Path(out_file):
                 warped_path.replace(out_file)

        elif self.method == 'fsl':
             from ...interfaces.fsl import flirt
             flirt_kwargs = {
                 'in_file': in_file, 'ref_file': ref_file, 'out_file': out_file,
                 'dof': self.options.get('dof', 6)
             }
             if 'cost' in self.options:
                 flirt_kwargs['cost'] = self.options['cost']

             extra_args = self.options.get('extra_args', self.options.get('args', ''))
             if extra_args:
                 flirt_kwargs['extra_args'] = extra_args

             extra_opts = {
                 k: v for k, v in self.options.items()
                 if k not in {'dof', 'cost', 'extra_args', 'args'}
             }
             if extra_opts:
                 flirt_kwargs['extra_opts'] = extra_opts
             flirt(**flirt_kwargs)
