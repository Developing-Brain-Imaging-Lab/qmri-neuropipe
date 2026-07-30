"""
Topup field estimation module for dMRI data.

Encapsulates FSL topup functionality as a processing step.
"""

from pathlib import Path
from typing import Optional, Dict, Any
import logging

from ...core import BaseProcessingStep, ProcessingError
from ...interfaces import fsl

class TopupStep(BaseProcessingStep):
    """
    Step to run FSL Topup for field map estimation.
    
    This step processes groups of reversed phase encoding images found in the context.
    It updates the context with the location of topup results ('topup_base') for
    subsequent steps like Eddy.
    """
    
    def __init__(
        self,
        config,
        logger: Optional[logging.Logger] = None,
        provenance = None,
    ):
        super().__init__(config, logger, provenance)
    
    def validate_inputs(self, first_arg, **kwargs) -> None:
        """
        Validate inputs for Topup.
        
        Detailed validation happens in run() because we need to check the context
        for topup groups.
        """
        pass

    def validate_outputs(self, result) -> None:
        """
        Validate Topup outputs.
        """
        pass

    def run(self, first_arg, output_dir: Path, **kwargs) -> Dict[str, Any]:
        """
        Run Topup on groups of reversed phase encoding images.
        
        Args:
            first_arg: Context dictionary containing 'topup_groups' and 'do_topup'.
            output_dir: Output directory for topup results.
            
        Returns:
            Updated context dictionary with 'topup_map' (path mapping).
        """
        context, _ = self.unpack_input(first_arg)
        if context is None:
            raise ProcessingError("TopupStep must run in pipeline context mode.")
        topup_groups = context.get("topup_groups", [])
        do_topup = context.get("do_topup", False)
        
        if not do_topup:
            self.logger.info("Topup logic skipped (do_topup=False or not configured).")
            return context

        # Retrieve topup config from distcorr section
        dmri_cfg = self.config.get('dmri', {}).get('preprocessing', {})
        distcorr_cfg = dmri_cfg.get('distcorr', {})
        topup_config = distcorr_cfg.get('config')
        
        if topup_config:
            self.logger.info(f"Using Topup config file: {topup_config}")

        if not topup_groups:
            self.logger.info("No reversed phase encoding groups found for Topup.")
            return context

        topup_dir = self.get_step_output_dir(output_dir)
        
        topup_map: Dict[Path, str] = context.get("topup_map", {})
        
        self.logger.info(f"Running Topup on {len(topup_groups)} groups...")
        
        for idx, group_item in enumerate(topup_groups):
            if isinstance(group_item, dict):
                 group = group_item.get('inputs', [])
                 targets = group_item.get('targets', [])
                 group_acqp = group_item.get('acqp')
                 group_index = group_item.get('index')
            else:
                 group = group_item
                 targets = []
                 group_acqp = None
                 group_index = None

            if not group:
                self.logger.warning(f"Topup group {idx} is empty, skipping.")
                continue
            
            try:
                # We use a base name for the topup output files
                base_name = f"topup_group{idx}"
                base = topup_dir / base_name
                
                # Define output paths
                fieldcoef = base.with_name(f"{base_name}_fieldcoef.nii.gz")
                movpar = base.with_name(f"{base_name}_movpar.txt")
                field_map = base.with_name(f"{base_name}_fmap.nii.gz")
                
                should_skip = False
                if fieldcoef.exists() and movpar.exists() and field_map.exists() and not kwargs.get('force', False):
                     # Check timestamps of inputs (peek at first entry)
                     from ...core.utils import extract_image_path
                     in_p = extract_image_path(group[0])
                     in_mtime = in_p.stat().st_mtime
                     out_mtime = fieldcoef.stat().st_mtime
                     
                     if in_mtime > out_mtime:
                          self.logger.info(f"Topup input ({in_p.name}) is newer than output. Re-running group {idx}.")
                     else:
                          self.logger.info(f"Skipping Topup group {idx} (outputs exist and are up-to-date): {base}")
                          should_skip = True
                
                if not should_skip:
                    # Optional Coregistration to first b0
                    if distcorr_cfg.get('coregister_inputs', False) and not (group_acqp or group_index):
                        self.logger.info(f"Coregistering Topup inputs using MCFLIRT...")
                        from ...core.utils import extract_image_path
                        from ...core.types import DWIFile
                        import shutil
                        import nibabel as nib
                        import numpy as np

                        # 0. Extract b0 volumes from input group
                        # The inputs might be full DWI series. We only want b=0 for Topup/MCFLIRT.
                        b0_group = []
                        for idx, img_obj in enumerate(group):
                            img_path = extract_image_path(img_obj)
                            bval_path = getattr(img_obj, 'bval', None)
                            
                            # If no bval, assume it's already a b0 or just an image
                            if not bval_path or not Path(bval_path).exists():
                                b0_group.append(img_obj)
                                continue
                                
                            try:
                                bvals = np.loadtxt(bval_path)
                                # Handle single value
                                if bvals.ndim == 0: bvals = np.array([bvals])
                                
                                # Default b0 threshold
                                b0_indices = np.where(bvals < 50)[0]
                                
                                if len(b0_indices) == 0:
                                    self.logger.warning(f"No b0 volumes found in {img_path.name}. Skipping.")
                                    continue
                                    
                                # If all volumes are b0, keep original
                                if len(b0_indices) == len(bvals):
                                    b0_group.append(img_obj)
                                    continue
                                
                                # Extract specific frames individualy to ensure flat 1-to-1 mapping
                                self.logger.debug(f"  Extracting {len(b0_indices)} b0 volumes from {img_path.name}")
                                nii = nib.load(str(img_path))
                                data = nii.get_fdata()
                                
                                orig_json = getattr(img_obj, 'json', None)
                                
                                for v_i, vol_idx in enumerate(b0_indices):
                                    # Extract single volume
                                    if data.ndim == 4:
                                        b0_vol = data[..., vol_idx]
                                    else:
                                        b0_vol = data
                                    
                                    # Save extracted b0
                                    b0_out = topup_dir / f"{base_name}_input{idx}_vol{vol_idx}.nii.gz"
                                    nib.save(nib.Nifti1Image(b0_vol, nii.affine, nii.header), str(b0_out))
                                    
                                    # Prepare metadata
                                    b0_json = None
                                    if orig_json and orig_json.exists():
                                        b0_json = b0_out.with_suffix('').with_suffix('.json')
                                        b0_json = b0_out.parent / (b0_out.name.split('.')[0] + '.json')
                                        shutil.copy(orig_json, b0_json)
                                        
                                    b0_bval = b0_out.parent / (b0_out.name.split('.')[0] + '.bval')
                                    b0_bvec = b0_out.parent / (b0_out.name.split('.')[0] + '.bvec')
                                    
                                    b0_bval.write_text("0\n")
                                    b0_bvec.write_text("0 0 0\n")
                                    
                                    b0_img = DWIFile(
                                        img=b0_out,
                                        bval=b0_bval,
                                        bvec=b0_bvec,
                                        json=b0_json,
                                        entities=getattr(img_obj, 'entities', {})
                                    )
                                    b0_group.append(b0_img)

                            except Exception as e:
                                self.logger.warning(f"Failed to extract b0s from {img_path}: {e}. Using full image.")
                                # If fallback, assume full image is what we want? 
                                # But this might break the flat list assumption if full image is 4D.
                                # Let's assume fallback is rare or user checks logs.
                                b0_group.append(img_obj)
                        
                        # Update group to use extracted b0s
                        group = b0_group

                        # 1. Merge all b0 inputs into 4D volume
                        merged_b0 = topup_dir / f"{base_name}_merged_b0.nii.gz"
                        self.logger.debug(f"  Merging b0s -> {merged_b0}")
                        fsl.merge(group, merged_b0, dimension='t')
                        
                        # 2. Run MCFLIRT on the formatted 4D volume
                        # (Registers to volume 0 by default, which matches our 'ref_img = group[0]' logic)
                        mcf_b0 = topup_dir / f"{base_name}_merged_b0_mcf.nii.gz"
                        self.logger.debug(f"  Running MCFLIRT -> {mcf_b0}")
                        fsl.mcflirt(merged_b0, mcf_b0, ref_vol=0, extra_args="-stages 3 -dof 6")
                        
                        # 3. Split the motion-corrected 4D volume back into 3D files
                        split_prefix = topup_dir / f"{base_name}_mcf_split_"
                        self.logger.debug(f"  Splitting corrected volume...")
                        split_files = fsl.split(mcf_b0, split_prefix, dimension='t')
                        
                        if len(split_files) != len(group):
                            raise RuntimeError(f"MCFLIRT split produced {len(split_files)} files, expected {len(group)}")
                            
                        # 4. Wrap split files as DWIFiles with original metadata
                        new_group = []
                        for i, (split_path, original_img) in enumerate(zip(split_files, group)):
                            # Identify original JSON from the source image
                            mov_json = getattr(original_img, 'json', None)
                            if not mov_json and hasattr(original_img, 'path'): 
                                candidate = Path(original_img.path).with_suffix('.json')
                                if candidate.exists(): mov_json = candidate
                            elif not mov_json:
                                orig_path = extract_image_path(original_img)
                                candidate = orig_path.with_suffix('').with_suffix('.json')
                                if candidate.exists(): mov_json = candidate

                            # Copy JSON to accompany the split file
                            if mov_json and mov_json.exists():
                                split_json = split_path.parent / (split_path.name.split('.')[0] + '.json')
                                shutil.copy(mov_json, split_json)
                            else:
                                split_json = None
                                self.logger.warning(f"Could not find JSON sidecar for input {i}. Topup might fail to detect PE direction.")

                            # Create dummy bval/bvec for the split 3D file
                            split_bval = split_path.parent / (split_path.name.split('.')[0] + '.bval')
                            split_bvec = split_path.parent / (split_path.name.split('.')[0] + '.bvec')
                            split_bval.write_text("0\n")
                            split_bvec.write_text("0 0 0\n")

                            # Copy entities
                            ents = getattr(original_img, 'entities', {})
                            
                            # Wrap
                            new_img_obj = DWIFile(img=split_path, bval=split_bval, bvec=split_bvec, json=split_json, entities=ents)
                            new_group.append(new_img_obj)
                            
                        # Use the new group for topup
                        group = new_group
                    elif distcorr_cfg.get('coregister_inputs', False):
                        self.logger.info(
                            "Skipping Topup input coregistration because this group uses merged acqp/index metadata."
                        )

                    nthreads = kwargs.get('nthreads', self.config.n_cpus)
                    fsl.topup(
                        group,
                        out_base=base,
                        field_output=True,
                        nthreads=nthreads,
                        config=topup_config,
                        acqp=Path(group_acqp) if group_acqp else None,
                        index=Path(group_index) if group_index else None,
                    )
                
                # Map each input image in the group to this topup result
                for d in group:
                    topup_map[d.img] = str(base)
                
                # Map targets
                for t in targets:
                    if hasattr(t, 'img'):
                        topup_map[t.img] = str(base)
                    else:
                        topup_map[t] = str(base)
                    
                self.logger.info(f"Topup successful for group {idx}.")
                
            except Exception as exc:
                self.logger.error(f"TOPUP failed for group {idx}: {exc}")
                raise ProcessingError(f"Topup failed for group {idx}: {exc}") from exc

        # Update context
        context["topup_map"] = topup_map
        
        # If only one group, set it as default topup_base
        if len(topup_groups) == 1:
            # Re-construct base path of the single group
            # (Loop logic above uses idx, but we know it's 0)
            base_name = f"topup_group0"
            base = topup_dir / base_name
            context["topup_base"] = str(base)
            self.logger.info(f"Setting default topup_base: {base}")
        
        return context
