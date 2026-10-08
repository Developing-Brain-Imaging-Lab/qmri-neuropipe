
from pathlib import Path
from typing import Optional, Dict
import nibabel as nib
import numpy as np
from nibabel.affines import voxel_sizes
from nibabel.processing import resample_from_to

from ...core import BaseProcessingStep
from ...core.types import ImageFile
from ...core.utils import ensure_dir
from ...io.bids import build_bids_name
from ...interfaces import ants, fsl # Assuming existence
from ...utils.relax_params import _extract_bids_param
from ..common.registration import prepare_registration_images, _ALL_SKULL_STRIP_OPTION_KEYS

class B1MappingStep(BaseProcessingStep):
    """
    Handles B1 Map preparation.
    1. If AFI: Registers AFI Reference to SPGR Reference, applies transform to B1 Map.
    2. If External: Registers/Resamples to SPGR Reference.
    """
    
    def __init__(
        self,
        config,
        logger,
        provenance,
        method="afi",
        smoothing_fwhm: float = 0.0,
        registration: Optional[Dict] = None,
    ):
        super().__init__(config, logger, provenance)
        self.method = method # 'afi', 'external', 'hifi'
        self.smoothing_fwhm = float(smoothing_fwhm if smoothing_fwhm is not None else 0.0)
        self.registration = dict(registration or {})

    def _registration_method(self) -> str:
        return str(self.registration.get("method", "fsl") or "fsl").lower()

    def _registration_interpolator(self) -> str:
        return str(
            self.registration.get(
                "interpolation",
                self.registration.get("interpolator", "linear"),
            )
        )

    def _fsl_interpolator(self) -> str:
        value = self._registration_interpolator().strip()
        aliases = {
            "linear": "trilinear",
            "nearest": "nearestneighbour",
            "nearestneighbor": "nearestneighbour",
            "nearestneighbour": "nearestneighbour",
            "bspline": "spline",
            "cubic": "spline",
            "lanczoswindowedsinc": "sinc",
        }
        return aliases.get(value.lower(), value)

    @staticmethod
    def _as_bool(value) -> bool:
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on", "enabled"}
        return bool(value)

    def _assume_aligned(self) -> bool:
        return self._as_bool(
            self.registration.get(
                "assume_aligned",
                self.registration.get("assume_registered", False),
            )
        )

    def _match_registration_resolution(self) -> bool:
        return self._as_bool(
            self.registration.get(
                "match_resolution",
                self.registration.get("match_registration_resolution", False),
            )
        )

    def _ants_transform_type(self) -> str:
        mapping = {
            "r": "Rigid",
            "rigid": "Rigid",
            "a": "Affine",
            "affine": "Affine",
            "translation": "Translation",
            "t": "Translation",
        }
        value = str(self.registration.get("transform_type", "Rigid") or "Rigid").strip()
        return mapping.get(value.lower(), value)

    def _ants_registration_kwargs(self) -> Dict:
        reserved = {
            "method",
            "transform_type",
            "interpolation",
            "interpolator",
            "dof",
            "cost",
            "assume_aligned",
            "assume_registered",
            "match_resolution",
            "match_registration_resolution",
            "resolution_matching_interpolation",
            "moving_volume",
            "fixed_volume",
            "reference_volume",
        } | _ALL_SKULL_STRIP_OPTION_KEYS
        return {k: v for k, v in self.registration.items() if k not in reserved}

    @staticmethod
    def _scipy_interpolation_order(value: str) -> int:
        value = str(value).strip().lower()
        if value in {"nearest", "nearestneighbor", "genericlabel", "label"}:
            return 0
        if value in {"cubic", "bspline", "spline"}:
            return 3
        return 1

    def _interpolation_order(self) -> int:
        return self._scipy_interpolation_order(self._registration_interpolator())

    def _resample_assuming_aligned(
        self,
        source: Path,
        reference: Path,
        output: Path,
    ) -> Path:
        """Resample by NIfTI world coordinates without estimating a transform."""
        source_img = nib.load(str(source))
        reference_img = nib.load(str(reference))
        if len(source_img.shape) != 3 or len(reference_img.shape) != 3:
            raise ValueError("Header-based B1 resampling requires 3D images.")
        resampled = resample_from_to(
            source_img,
            (reference_img.shape, reference_img.affine),
            order=self._interpolation_order(),
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        data = np.asanyarray(resampled.dataobj, dtype=np.float32)
        header = reference_img.header.copy()
        header.set_data_dtype(np.float32)
        nib.save(nib.Nifti1Image(data, reference_img.affine, header), str(output))
        return output

    def _fixed_resolution_proxy(
        self,
        moving: Path,
        fixed: Path,
        output_dir: Path,
    ) -> Path:
        """Downsample the fixed image to the moving voxel size for estimation."""
        if not self._match_registration_resolution():
            return Path(fixed)

        moving_img = nib.load(str(moving))
        fixed_img = nib.load(str(fixed))
        moving_zooms = np.asarray(voxel_sizes(moving_img.affine)[:3], dtype=float)
        fixed_zooms = np.asarray(voxel_sizes(fixed_img.affine)[:3], dtype=float)
        if np.allclose(moving_zooms, fixed_zooms, rtol=0.05, atol=0.05):
            return Path(fixed)

        fixed_shape = np.asarray(fixed_img.shape[:3], dtype=int)
        physical_extent = np.maximum(fixed_shape - 1, 1) * fixed_zooms
        proxy_shape = np.maximum(
            np.rint(physical_extent / moving_zooms).astype(int) + 1,
            1,
        )
        directions = fixed_img.affine[:3, :3] / fixed_zooms
        proxy_affine = fixed_img.affine.copy()
        proxy_affine[:3, :3] = directions * moving_zooms

        fixed_center_vox = (fixed_shape - 1) / 2.0
        proxy_center_vox = (proxy_shape - 1) / 2.0
        fixed_center_world = nib.affines.apply_affine(
            fixed_img.affine, fixed_center_vox
        )
        proxy_affine[:3, 3] = (
            fixed_center_world
            - proxy_affine[:3, :3] @ proxy_center_vox
        )

        proxy = resample_from_to(
            fixed_img,
            (tuple(int(value) for value in proxy_shape), proxy_affine),
            order=self._scipy_interpolation_order(
                self.registration.get("resolution_matching_interpolation", "linear")
            ),
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        output = output_dir / "b1_registration_fixed_resolution_matched.nii.gz"
        header = proxy.header.copy()
        header.set_data_dtype(np.float32)
        nib.save(
            nib.Nifti1Image(
                np.asanyarray(proxy.dataobj, dtype=np.float32),
                proxy_affine,
                header,
            ),
            str(output),
        )
        self.logger.info(
            "Matched B1 registration resolution: fixed %s mm -> %s mm",
            tuple(round(value, 3) for value in fixed_zooms),
            tuple(round(value, 3) for value in moving_zooms),
        )
        return output

    def _ensure_3d_registration_image(self, image_path: Path, output_dir: Path, label: str) -> Path:
        image_path = Path(image_path)
        try:
            img = nib.load(str(image_path))
        except Exception:
            return image_path

        if len(img.shape) < 4 or img.shape[3] <= 1:
            return image_path

        index = int(self.registration.get(f"{label}_volume", self.registration.get("reference_volume", 0)))
        if index < 0 or index >= img.shape[3]:
            raise ValueError(
                f"Requested {label}_volume={index} for ANTs B1 registration, "
                f"but {image_path.name} has {img.shape[3]} volumes."
            )

        out_path = output_dir / f"{image_path.stem}_{label}vol{index:04d}.nii.gz"
        if not out_path.exists():
            vol_data = np.asanyarray(img.dataobj[..., index])
            nib.save(nib.Nifti1Image(vol_data, img.affine, img.header.copy()), str(out_path))
        return out_path

    def _prepare_registration_estimation_images(
        self,
        moving: Path,
        reference: Path,
        output_dir: Path,
    ) -> tuple[Path, Path]:
        nthreads = int(self.registration.get("nthreads", self.registration.get("threads", self.config.get("n_cpus", 1))))
        moving_3d = self._ensure_3d_registration_image(moving, output_dir, "moving")
        reference_3d = self._ensure_3d_registration_image(reference, output_dir, "fixed")
        moving_for_reg, reference_for_reg, _ = prepare_registration_images(
            self.config,
            self.logger,
            moving_3d,
            reference_3d,
            output_dir,
            self.registration,
            nthreads,
            force=True,
        )
        reference_for_reg = self._fixed_resolution_proxy(
            moving_for_reg,
            reference_for_reg,
            output_dir,
        )
        return Path(moving_for_reg), Path(reference_for_reg)

    def _register_with_ants(self, moving: Path, reference: Path, output_dir: Path, prefix_name: str):
        nthreads = int(self.registration.get("nthreads", self.registration.get("threads", self.config.get("n_cpus", 1))))
        out_prefix = output_dir / prefix_name
        if not out_prefix.suffix:
            out_prefix = output_dir / f"{prefix_name}transform.nii.gz"
        moving_for_reg, reference_for_reg = self._prepare_registration_estimation_images(
            moving,
            reference,
            output_dir,
        )
        warped, transforms = ants.registration(
            fixed_file=reference_for_reg,
            moving_file=moving_for_reg,
            out_prefix=out_prefix,
            transform_type=self._ants_transform_type(),
            interpolator=self._registration_interpolator(),
            nthreads=nthreads,
            **self._ants_registration_kwargs(),
        )
        return warped, transforms, reference_for_reg
        
    def run(self, 
            b1_image: ImageFile, 
            reference_image: ImageFile, 
            output_dir: Path, 
            force: bool = False,
            b1_ref_image: Optional[ImageFile] = None
           ) -> ImageFile:
           
        output_dir = ensure_dir(output_dir)
        
        ents = dict(b1_image.entities)
        # Final Output Name: sub-XX_TB1map.nii.gz
        # Filter entities to minimal set (ensure sub/ses keys exist)
        minimal_ents = {}
        # Normalize to short keys for build_bids_name
        if 'subject' in ents: minimal_ents['sub'] = ents['subject']
        if 'sub' in ents: minimal_ents['sub'] = ents['sub']
        if 'session' in ents: minimal_ents['ses'] = ents['session']
        if 'ses' in ents: minimal_ents['ses'] = ents['ses']

        minimal_ents['suffix'] = 'TB1map' # Force suffix
        
        out_name = build_bids_name(minimal_ents)
        out_path = output_dir / out_name
        
        if out_path.exists() and not force:
             self.logger.info(f"Skipping B1 Alignment (Exists): {out_name}")
             return ImageFile(img=out_path, entities=ents)
        if out_path.exists() and force:
             out_path.unlink()
             
        self.logger.info(f"Aligning B1 Map ({self.method}) to {reference_image.img.name}")
        

        if self.method == 'afi':
            registration_method = self._registration_method()
            # Check if input is Map or Raw (2 volumes)
            import nibabel as nib
            img = nib.load(b1_image.img)
            
            # Identify if Raw AFI (4th dim exists and > 1)
            is_raw_afi = (len(img.shape) == 4 and img.shape[3] >= 2)
            
            b1_map_path = b1_image.img # Default if already map
            
            if is_raw_afi:
                 self.logger.info("Detected Raw AFI Input (4D). Aligning before computation...")
                 
                 # 1. Split 4D
                 from ...interfaces.fsl import split, merge
                 
                 # Temp dir for split
                 tmp_split = output_dir / "tmp_afi_split"
                 tmp_split.mkdir(exist_ok=True)
                 
                 split_base = tmp_split / "vol"
                 vols = split(b1_image.img, split_base)
                 
                 if len(vols) < 2:
                     raise ValueError("AFI input seems to be 4D but found less than 2 volumes.")
                     
                 # 2. Register Vol 0 (S1) to Reference, unless scanner-space
                 # headers are explicitly trusted.
                 vol0 = vols[0]
                 mat_file = output_dir / "afi_to_spgr.mat"
                 
                 if self._assume_aligned():
                      self.logger.info(
                          "AFI and SPGR are assumed aligned; skipping transform estimation."
                      )
                      transforms = None
                 elif registration_method == "ants":
                      _, transforms, ants_fixed = self._register_with_ants(
                          moving=vol0,
                          reference=reference_image.img,
                          output_dir=output_dir,
                          prefix_name="afi_to_spgr_ants_",
                      )
                 elif registration_method == "fsl":
                      moving_for_reg, reference_for_reg = (
                          self._prepare_registration_estimation_images(
                              vol0,
                              reference_image.img,
                              output_dir,
                          )
                      )
                      estimate_out = tmp_split / "vol0_aligned_ref.nii.gz"
                      if force:
                           mat_file.unlink(missing_ok=True)
                           estimate_out.unlink(missing_ok=True)
                      if not mat_file.exists() or force:
                           fsl.flirt(
                               in_file=moving_for_reg,
                               ref_file=reference_for_reg,
                               out_file=estimate_out,
                               omat=mat_file,
                               dof=int(self.registration.get("dof", 6)),
                               cost=str(self.registration.get("cost", "normmi")),
                           )
                      transforms = None
                 else:
                      raise ValueError(f"Unknown AFI registration method: {registration_method}")
                 
                 # 3. Apply Transform to ALL volumes independently
                 aligned_vols = []
                 for i, v in enumerate(vols):
                     out_v = tmp_split / f"vol{i}_aligned.nii.gz"
                     if self._assume_aligned():
                          self._resample_assuming_aligned(
                              v,
                              reference_image.img,
                              out_v,
                          )
                     elif registration_method == "ants":
                          ants.apply_transforms(
                              fixed_file=reference_image.img,
                              moving_file=v,
                              out_file=out_v,
                              transforms=transforms,
                              interpolator=self._registration_interpolator(),
                          )
                     else:
                          fsl.flirt(
                              in_file=v,
                              ref_file=reference_image.img,
                              out_file=out_v,
                              extra_args=(
                                  f"-applyxfm -init {mat_file} "
                                  f"-interp {self._fsl_interpolator()}"
                              ),
                          )
                     aligned_vols.append(out_v)
                     
                 # 4. Merge back to 4D
                 from ...core.utils import get_nifti_stem
                 aligned_afi_path = output_dir / f"{get_nifti_stem(b1_image.img)}_aligned.nii.gz"
                 merge(aligned_vols, aligned_afi_path, dimension='t')
                 
                 # Cleanup split
                 import shutil
                 shutil.rmtree(tmp_split)
                 
                 # Update B1 Image object to point to aligned data
                 aligned_afi_img = ImageFile(img=aligned_afi_path, entities=b1_image.entities, json=b1_image.json)
                 
                 # 5. Compute Map from Aligned Data
                 b1_map_path = self._compute_afi(aligned_afi_img, output_dir)
                 
                 # Move result to final if needed? _compute_afi returns path to generated map
                 # Ensure it goes to out_path
                 if b1_map_path != out_path:
                      import shutil
                      shutil.move(b1_map_path, out_path)
                      
            else:
                 # Input is already a map? Or separate files?
                 # Assuming Map if single volume or user pre-processed.
                 moving = b1_ref_image.img if b1_ref_image else b1_image.img
                 
                 # 1. Register B1 Ref -> SPGR Ref
                 mat_file = output_dir / "b1_to_spgr.mat"
                 
                 if self._assume_aligned():
                      self.logger.info(
                          "B1 and SPGR are assumed aligned; resampling by image headers."
                      )
                      self._resample_assuming_aligned(
                          b1_map_path,
                          reference_image.img,
                          out_path,
                      )
                 elif registration_method == "ants":
                      self.logger.info("Registering B1 reference to SPGR with ANTs")
                      _, transforms, ants_fixed = self._register_with_ants(
                          moving=moving,
                          reference=reference_image.img,
                          output_dir=output_dir,
                          prefix_name="b1_to_spgr_ants_",
                      )
                      self.logger.info("Applying ANTs transform to B1 Map")
                      ants.apply_transforms(
                          fixed_file=reference_image.img,
                          moving_file=b1_map_path,
                          out_file=out_path,
                          transforms=transforms,
                          interpolator=self._registration_interpolator(),
                      )
                 elif registration_method == "fsl":
                      # Calculate transform
                      moving_for_reg, reference_for_reg = (
                          self._prepare_registration_estimation_images(
                              Path(moving),
                              Path(reference_image.img),
                              output_dir,
                          )
                      )
                      estimate_out = output_dir / "b1_ref_aligned.nii.gz"
                      if force:
                           mat_file.unlink(missing_ok=True)
                           estimate_out.unlink(missing_ok=True)
                      fsl.flirt(
                          in_file=moving_for_reg,
                          ref_file=reference_for_reg,
                          out_file=estimate_out,
                          omat=mat_file,
                          dof=int(self.registration.get("dof", 6)),
                          cost=str(self.registration.get("cost", "normmi")),
                      )
                      
                      # Apply to B1 Map on the original SPGR output grid.
                      self.logger.info("Applying transform to B1 Map")
                      fsl.flirt(
                          in_file=b1_map_path,
                          ref_file=reference_image.img,
                          out_file=out_path,
                          extra_args=(
                              f"-applyxfm -init {mat_file} "
                              f"-interp {self._fsl_interpolator()}"
                          ),
                      )
                 else:
                      raise ValueError(f"Unknown B1 registration method: {registration_method}")
            
        elif self.method == 'external':
             # Just resample to match grid if needed, assuming already aligned? 
             # Or Register? 
             self.logger.info("Resampling External B1 to Reference Grid")
             from ...interfaces.fsl import resample_to_image
             resample_to_image(source_file=b1_image.img, reference_file=reference_image.img, out_file=out_path)
             
        elif self.method == 'hifi':
             pass # Handled by fitting binary directly usually? 
             
        return ImageFile(img=out_path, entities=ents)

    def _compute_afi(self, afi_image: ImageFile, output_dir: Path) -> Path:
        """
        Calculate B1 map from AFI (2-volume 4D) data.
        B1 = arccos( (r*n - 1) / (n - r) ) / nominal_FA   (Standard)
        Here using r = S2/S1 (Long/Short).
        Derived: val = (n - r) / (n*r - 1)
        """
        import numpy as np
        import scipy.ndimage as nd
        
        # Check for explicit TRRatio (n)
        n_ratio = _extract_bids_param(afi_image, "TRRatio")
        
        if n_ratio:
             self.logger.info(f"Using provided TRRatio from JSON: {n_ratio}")
             n_ratio = float(n_ratio)
        else:
            # Fallback to TR calculation
            tr = _extract_bids_param(afi_image, "RepetitionTime")
            if isinstance(tr, list) and len(tr) == 2:
                # TR1 is Short, TR2 is Long usually? 
                # Standards: TR1 = 20ms, TR2 = 100ms -> n=5.
                tr_sorted = sorted(float(value) for value in tr)
                n_ratio = float(tr_sorted[1] / tr_sorted[0])
            else:
                self.logger.warning("AFI: RepetitionTime not a list of 2 in JSON and no TRRatio found. Assuming n=5 for testing.")
                n_ratio = 5.0 # Fallback
        
        flip_angle_deg = _extract_bids_param(afi_image, "FlipAngle")
        # If list, take first? Usually same FA.
        if isinstance(flip_angle_deg, list): flip_angle_deg = flip_angle_deg[0]
        if not flip_angle_deg: 
             self.logger.warning("AFI: FlipAngle missing. Assuming 60 degrees.")
             flip_angle_deg = 60.0
        flip_angle_deg = float(flip_angle_deg)
        
        flip_angle_rad = np.deg2rad(flip_angle_deg)
        
        # Load Data
        img_nii = nib.load(afi_image.img)
        data = img_nii.get_fdata()
        
        # Check dim
        if data.shape[3] < 2:
             raise ValueError("AFI data must have at least 2 volumes.")
             
        s1 = data[..., 0]
        s2 = data[..., 1]
        
        # Smoothing (if requested)
        if self.smoothing_fwhm > 0:
            self.logger.info(f"Applying Gaussian Smoothing (FWHM={self.smoothing_fwhm}mm)")
            
            # Convert FWHM to Sigma (voxels)
            # sigma = FWHM / 2.355
            # sigma_vox = sigma / vox_size
            vox_sizes = img_nii.header.get_zooms()[:3]
            sigmas = [(self.smoothing_fwhm / 2.355) / v for v in vox_sizes]
            
            s1 = nd.gaussian_filter(s1, sigma=sigmas)
            s2 = nd.gaussian_filter(s2, sigma=sigmas)
            

        
        r = s2 / s1
        r[r>1] = 1

        arg = (r*n_ratio - 1)/(n_ratio-r)
        arg[arg>1] = 1
        arg[arg<-1] = -1 
        
        alpha_act = np.arccos(arg)
        b1_map = alpha_act / flip_angle_rad
        
        # Save
        # Save Intermediate: sub-XX_desc-preproc_TB1AFI
        # Use entities from input
        afi_ents = dict(afi_image.entities)
        # Keep sub/ses, add/overwrite desc/suffix
        afi_inter_ents = {}
        if 'subject' in afi_ents: afi_inter_ents['sub'] = afi_ents['subject']
        if 'sub' in afi_ents: afi_inter_ents['sub'] = afi_ents['sub']
        if 'session' in afi_ents: afi_inter_ents['ses'] = afi_ents['session']
        if 'ses' in afi_ents: afi_inter_ents['ses'] = afi_ents['ses']
        
        afi_inter_ents['desc'] = 'preproc'
        afi_inter_ents['suffix'] = 'TB1AFI'
        
        out_name = output_dir / build_bids_name(afi_inter_ents)
        nib.save(nib.Nifti1Image(b1_map, img_nii.affine, img_nii.header), out_name)
        
        return out_name
