"""
Topup field estimation module for dMRI data.

Encapsulates FSL topup functionality as a processing step.
"""

from pathlib import Path
from typing import Optional, List, Dict, Any
import logging

from ...core import BaseProcessingStep, ProcessingError
from ...core.types import DWIFile
from ...interfaces import fsl
from ...io.bids import build_bids_name

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
            else:
                 group = group_item
                 targets = []

            if len(group) < 2:
                self.logger.warning(f"Topup group {idx} has fewer than 2 images, skipping.")
                continue
            
            try:
                # We use a base name for the topup output files
                base_name = f"topup_group{idx}"
                base = topup_dir / base_name
                
                # Check if outputs exist
                fieldcoef = base.with_name(f"{base.name}_fieldcoef.nii.gz")
                movpar = base.with_name(f"{base.name}_movpar.txt")
                field_map = base.with_name(f"{base.name}_fmap.nii.gz")
                
                if fieldcoef.exists() and movpar.exists() and field_map.exists() and not kwargs.get('force', False):
                     self.logger.info(f"Skipping Topup group {idx} (outputs exist): {base}")
                else:
                    # Optional Coregistration to first b0
                    if distcorr_cfg.get('coregister_inputs', False):
                        self.logger.info(f"Coregistering Topup inputs to the first volume (FLIRT dof 6)...")
                        
                        # Identify reference
                        ref_img = group[0]
                        # We need paths
                        from ...core.utils import extract_image_path
                        from ...core.types import ImageFile
                        ref_path = extract_image_path(ref_img)
                        
                        new_group = [ref_img] # First one is reference, keep as is
                        
                        for i, mov_img in enumerate(group[1:], start=1):
                            mov_path = extract_image_path(mov_img)
                            # Define output for coregistered b0
                            # Maybe in a temp subdir or just alongside?
                            # Use base_name as prefix
                            coreg_out = topup_dir / f"{base_name}_input{i}_coreg.nii.gz"
                            
                            self.logger.debug(f"  Registering {mov_path.name} -> {ref_path.name}")
                            fsl.flirt(in_file=mov_path, ref_file=ref_path, out_file=coreg_out, dof=6)
                            
                            import shutil
                            
                            # Preserve metadata (JSON sidecar) for Topup
                            mov_json = getattr(mov_img, 'json', None)
                            if not mov_json and hasattr(mov_img, 'path'): 
                                # Try sidecar from path
                                candidate = Path(mov_img.path).with_suffix('.json')
                                if candidate.exists(): mov_json = candidate
                            elif not mov_json:
                                # Try path from mov_path
                                candidate = mov_path.with_suffix('').with_suffix('.json') # Handle .nii.gz
                                if candidate.exists(): mov_json = candidate

                            # Copy JSON if found
                            if mov_json and mov_json.exists():
                                coreg_json = coreg_out.with_suffix('').with_suffix('.json')
                                if coreg_json.name.endswith('.nii.json'): # Fix double suffix if any
                                     coreg_json = coreg_out.with_name(coreg_out.name.replace('.nii.gz', '.json').replace('.nii', '.json'))
                                
                                # Simple replace
                                coreg_json = coreg_out.parent / (coreg_out.name.split('.')[0] + '.json')
                                
                                shutil.copy(mov_json, coreg_json)
                                
                                # Wrap in ImageFile
                                # Entities? Try to copy
                                ents = getattr(mov_img, 'entities', {})
                                new_img_obj = ImageFile(coreg_out, entities=ents)
                                new_group.append(new_img_obj)
                            else:
                                self.logger.warning(f"Could not find JSON sidecar for {mov_path.name}. Topup might fail.")
                                new_group.append(ImageFile(coreg_out))
                            
                        # Use the new group for topup
                        group = new_group

                    nthreads = kwargs.get('nthreads', self.config.n_cpus)
                    fsl.topup(group, out_base=base, field_output=True, nthreads=nthreads, config=topup_config)
                
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
