from pathlib import Path
from typing import Optional, Any
import logging
import numpy as np
import nibabel as nib

from ...core import BaseProcessingStep, ValidationError, ProcessingError
from ...core.types import ImageLike, DWIFile, ImageFile
from ...core.run import run_cmd
from ...interfaces import tortoise
from ...interfaces import ants, c3d
from ...io.bids import build_bids_name
from ...io.bids import build_bids_name
from ...interfaces.mrtrix import dwiextract, mrcalc, mrmath
from ...core.utils import check_nifti_integrity, extract_image_path
from .grad_nonlin_native import create_native_ge_gnl_map
from .grad_nonlin_native import _rotation_from_fsl_affine, _reorient_tensor_components, _same_grid

def create_gnl_map(
    input_image: ImageLike,
    output_path: Path,
    grad_coeffs: Path,
    native_reference: Optional[ImageLike] = None,
    method: str = "tortoise",
    spatial_transform: Optional[dict] = None,
    nthreads: int = 1,
    force: bool = False,
    logger: Optional[logging.Logger] = None
) -> Path:
    """
    Standalone function to calculate the Gradient Nonlinearity tensor map.
    
    Args:
        input_image: The target image (defining the grid for GNL correction).
        output_path: Where to save the final .nii.gz tensor map.
        grad_coeffs: Path to the .dat coefficients file.
        native_reference: Optional original native space image (for resampled/coregistered inputs).
        nthreads: Number of threads to use.
        force: Force re-calculation even if output exists.
        logger: Logger instance.
        
    Returns:
        Path to the generated GNL tensor map file.
    """
    if logger is None:
        logger = logging.getLogger("GNL")

    if output_path.exists() and not force:
        if check_nifti_integrity(output_path):
            logger.info(f"Skipping GNL calculation (Output exists: {output_path.name})")
            return output_path
        else:
            logger.warning(f"GNL output corrupt: {output_path}. Re-calculating.")
            output_path.unlink()

    if not grad_coeffs or not grad_coeffs.exists():
        raise ProcessingError(f"Gradient nonlinearity coefficients file not found: {grad_coeffs}")

    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Calculating Gradient Nonlinearity Tensor Map (method={method})...")

    if method == "native_ge":
        return create_native_ge_gnl_map(
            input_image=input_image,
            output_path=output_path,
            grad_coeffs=grad_coeffs,
            native_reference=native_reference,
            spatial_transform=spatial_transform,
            nthreads=nthreads,
            force=force,
            logger=logger,
        )
    
    try:
        # Prepare 3D mean b0 images for initial and final spaces
        initial_image_path = None
        
        # 1. Native Space 3D Reference (Initial Image) - Only if native_reference provided
        if native_reference:
            native_img_path = extract_image_path(native_reference)
            native_b0 = output_dir / "native_b0_mean.nii.gz"
            
            if not native_b0.exists() or force:
                 # Extract b0s and mean
                 temp_b0s = output_dir / "native_b0s.mif"
                 
                 # Explicitly pass grads if available
                 nbvec = getattr(native_reference, 'bvec', None)
                 nbval = getattr(native_reference, 'bval', None)
                 
                 dwiextract(native_img_path, temp_b0s, bzero=True, in_bvec=nbvec, in_bval=nbval, force=True)
                 mrmath(temp_b0s, "mean", native_b0, axis=3, force=True)
                 temp_b0s.unlink(missing_ok=True)
            initial_image_path = native_b0

        # 2. Final Space 3D Reference (Final Image)
        # Input image (likely processed/coregistered)
        final_b0_path = output_dir / "final_b0_mean.nii.gz"
        if not final_b0_path.exists() or force:
             temp_b0s_final = output_dir / "final_b0_final.mif"
             
             # Explicitly pass grads if available
             fbvec = getattr(input_image, 'bvec', None)
             fbval = getattr(input_image, 'bval', None)
             
             dwiextract(extract_image_path(input_image), temp_b0s_final, bzero=True, in_bvec=fbvec, in_bval=fbval, force=True)
             mrmath(temp_b0s_final, "mean", final_b0_path, axis=3, force=True)
             temp_b0s_final.unlink(missing_ok=True)
        
        final_image_path = final_b0_path
        
        # Run GNL.
        tortoise.apply_grad_nonlin(
            initial_image=initial_image_path,
            final_image=final_image_path,
            grad_coeffs=grad_coeffs,
            nthreads=nthreads,
            force=force,
            cwd=output_dir
        )
        
        # TORTOISE typically outputs *graddev_c.nii
        candidates = list(output_dir.glob("*graddev_c.nii"))
        if not candidates:
            candidates = list(output_dir.glob("*graddev_c.nii.gz"))
        
        if candidates:
             actual_output = candidates[0]
             # Gzip if needed and rename to standardized proper output_path
             if actual_output.suffix == '.nii':
                 logger.info(f"Gzipping and renaming GNL output: {actual_output.name} -> {output_path.name}")
                 import gzip
                 import shutil
                 with open(actual_output, 'rb') as f_in:
                     with gzip.open(output_path, 'wb') as f_out:
                         shutil.copyfileobj(f_in, f_out)
                 actual_output.unlink() # remove .nii
             else:
                 logger.info(f"Renaming GNL output: {actual_output.name} -> {output_path.name}")
                 actual_output.rename(output_path)
        
    except Exception as e:
        raise ProcessingError(f"TORTOISE GNL calculation failed: {e}") from e
        
    if not output_path.exists():
         logger.warning(f"Expected GNL output {output_path} not found.")
         
    return output_path


class TortoiseGradNonlinCorrectStep(BaseProcessingStep):
    """
    Gradient Nonlinearity Correction using TORTOISE.
    """

    def __init__(
        self,
        config,
        logger: Optional[logging.Logger] = None,
        provenance = None,
        grad_coeffs: Optional[Path] = None,
        is_resampled: bool = False
    ):
        super().__init__(config, logger, provenance)
        self.grad_coeffs = grad_coeffs
        self.is_resampled = is_resampled
        self.method = (
            self.config.get('dmri', {})
            .get('preprocessing', {})
            .get('grad_nonlin', {})
            .get('method', 'tortoise')
        )
        self.logger.info("Initialized TortoiseGradNonlinCorrectStep")

    def validate_inputs(self, first_arg, **kwargs) -> None:
        pass

    def validate_outputs(self, result) -> None:
        pass

    def _extract_image(self, first_arg) -> ImageLike:
        if isinstance(first_arg, dict):
            img = first_arg.get("current_image")
            if img is None:
                raise ValidationError("TortoiseGradNonlinCorrectStep expects context['current_image'] to be set")
            return img
        return first_arg

    def run(self, first_arg, output_dir: Path, **kwargs) -> Any:
        
        is_context = isinstance(first_arg, dict)
        if is_context:
            context = dict(first_arg)
            input_img = self._extract_image(context)
            native_ref = context.get('native_dwi_for_gnl')
            spatial_transform = context.get('gnl_spatial_transform')
        else:
            context = None
            input_img = first_arg
            native_ref = kwargs.get('native_reference')
            spatial_transform = kwargs.get('spatial_transform')

        if self.is_resampled and not native_ref:
            self.logger.warning(
                "GNL Step is configured for resampled data but 'native_dwi_for_gnl' not found in context. Using input as native."
            )
            native_ref = input_img

        output_dir = Path(output_dir) / "grad_nonlin"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Output is a tensor map
        new_desc = "gnl_tensor" 
        output_map = output_dir / build_bids_name({**input_img.entities, "desc": new_desc})
        
        # Determine Gradient Cofficients
        coeffs = self.grad_coeffs
        if not coeffs: coeffs = kwargs.get('grad_coeffs')
        if not coeffs:
             dmri_cfg = self.config.get('dmri', {}).get('preprocessing', {}).get('grad_nonlin', {})
             c_path = dmri_cfg.get('coeff_file')
             if c_path: coeffs = Path(c_path)
        
        if not coeffs or not coeffs.exists():
            raise ProcessingError(f"Gradient nonlinearity coefficients file not found/provided.")

        # Call the standalone function
        nthreads = kwargs.get('nthreads', self.config.n_cpus)
        force = kwargs.get('force', False)
        
        # If not resampled, we don't pass native_ref to the command (as it implies identity)
        # But create_gnl_map handles the logic. If we want exactly the same behavior as before:
        passing_native = native_ref if self.is_resampled else None
        
        result_map = create_gnl_map(
            input_image=input_img,
            output_path=output_map,
            grad_coeffs=coeffs,
            native_reference=passing_native,
            method=self.method,
            spatial_transform=spatial_transform,
            nthreads=nthreads,
            force=force,
            logger=self.logger
        )

        # Store map in context
        if is_context:
            context["gnl_map"] = result_map
            if spatial_transform is not None:
                context["spatial_transform"] = spatial_transform
            return context
        else:
            return result_map


class AlignFinalGNLTensorStep(BaseProcessingStep):
    """
    Linearly map a GNL tensor from its creation/reference space to the final
    preprocessed DWI space using a rigid-only transform.
    """

    def __init__(self, config, logger: Optional[logging.Logger] = None, provenance = None):
        super().__init__(config, logger, provenance)
        self.logger.info("Initialized AlignFinalGNLTensorStep")

    def validate_inputs(self, first_arg, **kwargs) -> None:
        pass

    def validate_outputs(self, result) -> None:
        pass

    def _extract_image(self, first_arg) -> ImageLike:
        if isinstance(first_arg, dict):
            img = first_arg.get("current_image")
            if img is None:
                raise ValidationError(
                    "AlignFinalGNLTensorStep expects context['current_image'] to be set"
                )
            return img
        return first_arg

    def run(self, first_arg, output_dir: Path, **kwargs) -> Any:
        context = dict(first_arg) if isinstance(first_arg, dict) else None
        if context is None:
            raise ValidationError("AlignFinalGNLTensorStep must be called with context")

        dwi_image = self._extract_image(context)
        if not hasattr(dwi_image, "img"):
            return context

        final_dwi = Path(dwi_image.img)
        gnl_map = (
            context.get("gnl_map_by_image", {}).get(final_dwi)
            or context.get("gnl_map")
        )
        if not gnl_map:
            self.logger.info("No GNL map available for final-space alignment; skipping.")
            return context

        gnl_map = Path(gnl_map)
        if not gnl_map.exists():
            self.logger.warning(f"GNL map missing for final alignment: {gnl_map}")
            return context

        native_ref = context.get("gnl_native_reference_map", {}).get(final_dwi, dwi_image)
        native_ref_img = Path(getattr(native_ref, "img", native_ref))

        if not native_ref_img.exists():
            self.logger.warning(
                f"Native GNL reference image missing for {final_dwi.name}; skipping alignment."
            )
            return context

        try:
            if _same_grid(gnl_map, final_dwi):
                # Already in final voxel lattice. Keep as-is.
                context["gnl_map"] = gnl_map
                context.setdefault("gnl_map_by_image", {})[final_dwi] = gnl_map
                return context
        except Exception:
            # If affine/shape inspection fails, continue with an explicit mapping.
            self.logger.warning("Could not compare GNL and final DWI grids; forcing remap.")

        map_output_dir = Path(gnl_map).parent
        if str(map_output_dir) == ".":
            map_output_dir = output_dir / "grad_nonlin"
        map_output_dir.mkdir(parents=True, exist_ok=True)

        def _as_3d_reference(src_path: Path, tag: str) -> Path:
            img = nib.load(str(src_path))
            if img.ndim < 4:
                return src_path
            ref3d = map_output_dir / f"{src_path.stem}_{tag}_3d_ref.nii.gz"
            if ref3d.exists() and not kwargs.get("force", False):
                return ref3d
            data = img.get_fdata(dtype=np.float32)
            data3d = np.mean(data, axis=3).astype(np.float32)
            out_header = img.header.copy()
            out_header.set_data_shape(data3d.shape)
            nib.save(nib.Nifti1Image(data3d, img.affine, out_header), str(ref3d))
            return ref3d

        final_ref = _as_3d_reference(final_dwi, "final")
        native_ref_for_reg = _as_3d_reference(native_ref_img, "native")

        # Keep original filename and add explicit final-space suffix.
        if gnl_map.name.endswith(".nii.gz"):
            base = gnl_map.name[:-7]
            suffix = ".nii.gz"
        else:
            base = gnl_map.stem
            suffix = gnl_map.suffix
        mapped_map = map_output_dir / f"{base}_final{suffix}"

        try:
            nthreads = kwargs.get("nthreads", self.config.n_cpus)
            reg_prefix = map_output_dir / "gnl_native_to_final_"
            _, transform_list = ants.registration(
                fixed_file=final_ref,
                moving_file=native_ref_for_reg,
                out_prefix=reg_prefix,
                transform_type="Rigid",
                interpolator="linear",
                nthreads=nthreads,
            )

            ants.apply_transforms(
                fixed_file=final_dwi,
                moving_file=gnl_map,
                out_file=mapped_map,
                transforms=transform_list,
                interpolator="linear",
                nthreads=nthreads,
                imagetype=3,
            )
        except Exception as map_err:
            self.logger.warning(
                f"Failed to compute/apply final GNL alignment for {final_dwi.name}: {map_err}"
            )
            context["gnl_map"] = gnl_map
            context.setdefault("gnl_map_by_image", {})[final_dwi] = gnl_map
            return context

        affine_file = next(
            (Path(t) for t in transform_list if str(t).endswith(".mat")),
            None
        )
        if affine_file and affine_file.exists():
            fsl_affine = map_output_dir / (mapped_map.stem + "_fsl.mat")
            try:
                c3d.ants2fsl(final_dwi, native_ref_img, affine_file, fsl_affine)
                rotation = _rotation_from_fsl_affine(fsl_affine, native_ref_img, final_dwi)
                _reorient_tensor_components(mapped_map, rotation)
            except Exception as map_err:
                self.logger.warning(
                    f"Could not reorient final-space GNL tensor basis: {map_err}"
                )
        else:
            self.logger.warning(
                f"No affine found from registration; skipping tensor reorientation for {mapped_map.name}"
            )

        # Update context to guarantee modeling uses this final-space map.
        context["gnl_map"] = mapped_map
        context.setdefault("gnl_map_by_image", {})[final_dwi] = mapped_map
        return context
