"""
Merge step for dMRI data.

Merges multiple DWI files into a single 4D volume, concatenating bvals, bvecs,
and generating index/acqparams files for Topup/Eddy.
"""

from pathlib import Path
from typing import Optional, List, Dict, Any
import logging
import numpy as np
import nibabel as nib
import shutil

from ...core import BaseProcessingStep, ProcessingError, ValidationError
from ...core.types import DWIFile
from ...interfaces import fsl
from ...io.bids import build_bids_name, _load_json_field

class MergeStep(BaseProcessingStep):
    """
    Step to merge multiple DWI files into a single dataset.
    
    This is critical for Topup/Eddy processing where multiple phase-encoding
    directions need to be processed together.
    
    Outputs:
    - Merged NIfTI
    - Merged .bval, .bvec
    - index.txt (for eddy)
    - acqparams.txt (if not already managed by Topup)
    """
    
    def __init__(
        self,
        config,
        logger: Optional[logging.Logger] = None,
        provenance = None,
    ):
        super().__init__(config, logger, provenance)
    
    def run(self, first_arg, output_dir: Path, **kwargs) -> Dict[str, Any]:
        """
        Merge DWI files found in context.
        """
        context, _ = self.unpack_input(first_arg)
        if context is None:
            raise ProcessingError("MergeStep must run in pipeline context mode.")
            
        dwi_files: List[DWIFile] = context.get("dwi_files", [])
        
        if len(dwi_files) < 2:
            self.logger.info("Fewer than 2 DWI files. Skipping MergeStep.")
            return context

        merge_dir = self.get_step_output_dir(output_dir)
        
        # Determine output filename
        # Use entities of first file, but update description
        first_dwi = dwi_files[0]
        ents = first_dwi.entities.copy()
        ents['desc'] = 'merged'
        # Remove dir entity if present, as merged file contains multiple directions
        if 'dir' in ents:
            del ents['dir']
        # Remove session/run specific if they differ? 
        # Usually we keep first file's naming conv mostly
        
        out_name = build_bids_name(ents)
        out_nii = merge_dir / out_name
        out_bval = out_nii.with_suffix("").with_suffix(".bval")
        out_bvec = out_nii.with_suffix("").with_suffix(".bvec")
        out_index = merge_dir / "index.txt"
        
        # Check if outputs exist
        if out_nii.exists() and out_bval.exists() and out_bvec.exists() and out_index.exists() and not kwargs.get('force', False):
            self.logger.info(f"Skipping Merge (Outputs exist: {out_nii})")
            
            # Load result
            merged_dwi = DWIFile(
                img=out_nii,
                entities=ents,
                bval=out_bval,
                bvec=out_bvec,
                json=first_dwi.json # Approximate JSON from first file
            )
            
            # Update context
            context["dwi_files"] = [merged_dwi] # Replace list with single file
            context["merged_index"] = out_index
            
            # We also need to map acqp if it exists?
            # TopupStep generates/updates topup_map.
            # Eddy needs appropriate acqp and index.
            # MergeStep should ensure index.txt corresponds to the merged sequence.
            
            return context
            
        self.logger.info(f"Merging {len(dwi_files)} DWI files...")
        
        # 1. Merge NIfTI
        in_imgs = [d.img for d in dwi_files]
        fsl.merge(in_files=in_imgs, out_file=out_nii)
        
        # 2. Merge Bvals/Bvecs
        bvals = []
        bvecs = []
        
        # 3. Build Index
        # We need to assign an index ID to each volume.
        # This ID refers to a line in acqparams.txt
        # If TopupStep ran, it might have defined topup_groups or topup_map?
        # But we need a global index for eddy.
        # Assuming we have N input files.
        # Each input file corresponds to a specific acquisition parameter set (e.g. AP or PA).
        # We need to know which acqp line corresponds to each FILE.
        
        # Strategy:
        # Construct acqparams list.
        # For each file, identify its params (readout time, PE direction).
        # Add to acqparams list if unique.
        # Get its index (1-based).
        # Repeat that index for N volumes in that file.
        
        acq_params_list = []
        full_index = []
        
        for d in dwi_files:
            # Read bval/bvec
            try:
                bv = np.loadtxt(d.bval)
                # Ensure 1D
                if bv.ndim == 0: bv = np.array([bv])
                # If file has multiple vols
                n_vols = bv.size
                bvals.append(bv)
                
                bc = np.loadtxt(d.bvec)
                # Check shape (3, N) or (N, 3)
                if bc.shape[0] != 3: bc = bc.T
                bvecs.append(bc)
                
                # Identify acqp for this file
                # Use PhaseEncodingDirection and TotalReadoutTime from JSON
                pe_dir = _load_json_field(d.json, "PhaseEncodingDirection")
                trt = _load_json_field(d.json, "TotalReadoutTime")
                eff_echo = _load_json_field(d.json, "EffectiveEchoSpacing")
                
                # FSL format: x y z time
                # x,y,z derived from PE dir (i, i-, j, j-, k, k-)
                vec = [0, 0, 0]
                if pe_dir == 'i': vec = [1, 0, 0]
                elif pe_dir == 'i-': vec = [-1, 0, 0]
                elif pe_dir == 'j': vec = [0, 1, 0]
                elif pe_dir == 'j-': vec = [0, -1, 0]
                elif pe_dir == 'k': vec = [0, 0, 1]
                elif pe_dir == 'k-': vec = [0, 0, -1]
                else:
                    self.logger.warning(f"Unknown or missing PhaseEncodingDirection '{pe_dir}' for {d.img.name}. Defaulting to [0,1,0].")
                    vec = [0, 1, 0]
                
                # Readout time: TRT is best. If missing, use EES * (dim - 1). 
                ro_time = 0.05 # Default
                if trt:
                    ro_time = trt
                elif eff_echo:
                    # Calculate from EES. Need PE dimension size.
                    # Assuming PE axis matches vec.
                    try:
                        img_shape = nib.load(d.img).shape
                        dim_size = 0
                        if vec[0] != 0: dim_size = img_shape[0]
                        elif vec[1] != 0: dim_size = img_shape[1]
                        elif vec[2] != 0: dim_size = img_shape[2]
                        
                        if dim_size > 0:
                            ro_time = eff_echo * (dim_size - 1)
                            self.logger.info(f"Calculated TotalReadoutTime from EES: {ro_time:.6f} s (EES={eff_echo}, dim={dim_size})")
                    except Exception as e:
                        self.logger.warning(f"Could not calculate TotalReadoutTime from EES: {e}")
                else:
                    self.logger.warning(f"No TotalReadoutTime or EffectiveEchoSpacing found for {d.img.name}. Using default {ro_time}.")
                
                param_line = f"{int(vec[0])} {int(vec[1])} {int(vec[2])} {float(ro_time):.6f}"
                self.logger.info(f"Detected params for {d.img.name}: {param_line} (PE={pe_dir})")
                
                # Find or add to list
                if param_line not in acq_params_list:
                    acq_params_list.append(param_line)
                
                idx = acq_params_list.index(param_line) + 1 # 1-based
                
                # Add to index array
                full_index.extend([idx] * n_vols)
                
            except Exception as e:
                raise ProcessingError(f"Failed to parse bval/bvec/json for {d.img}: {e}")
                
        # Save merged bval/bvec
        final_bval = np.concatenate(bvals)
        final_bvec = np.concatenate(bvecs, axis=1) # (3, N_total)
        
        np.savetxt(out_bval, final_bval, fmt='%g')
        np.savetxt(out_bvec, final_bvec, fmt='%.6f')
        
        # Save index
        np.savetxt(out_index, np.array(full_index), fmt='%d', newline=' ') # space separated row? FSL usually accepts col or row.
        # Eddy usually expects row or column.
        
        # Save acqparams.txt
        out_acqp = merge_dir / "acqparams.txt"
        with open(out_acqp, 'w') as f:
            f.write('\n'.join(acq_params_list))
            
        self.logger.info(f"Merge complete. Output: {out_nii}")
        
        # Update context
        merged_dwi = DWIFile(
            img=out_nii,
            entities=ents,
            bval=out_bval,
            bvec=out_bvec,
            json=first_dwi.json
        )
        
        context["dwi_files"] = [merged_dwi]
        context["merged_index"] = out_index
        context["merged_acqp"] = out_acqp
        # We need to tell topup/eddy to use this acqp?
        # EddyStep should check context["merged_acqp"] and context["merged_index"]
        # And pass them to eddy.
        
        context["acqp"] = out_acqp
        context["index"] = out_index
        
        return context
