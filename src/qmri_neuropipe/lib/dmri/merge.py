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

from ...core import BaseProcessingStep, ProcessingError
from ...core.types import DWIFile
from ...interfaces import fsl
from ...io.bids import build_bids_name, _load_json_field
from ...io.dmri.bids import (
    infer_fsl_phase_encoding_direction,
    infer_phase_encoding_direction,
    fsl_phase_encoding_direction_to_vector,
)

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
        out_Delta = merge_dir / (out_name.replace(".nii.gz", ".bigdelta"))
        out_delta = merge_dir / (out_name.replace(".nii.gz", ".delta"))
        out_index = merge_dir / "index.txt"
        
        out_acqp = merge_dir / "acqparams.txt"
        
        # Check if outputs exist and are valid (>100 bytes)
        outputs_valid = (
            self.check_output_validity(out_nii) and 
            self.check_output_validity(out_bval, min_size=5) and 
            self.check_output_validity(out_bvec, min_size=5) and 
            self.check_output_validity(out_index, min_size=1) and 
            self.check_output_validity(out_acqp, min_size=5)
        )
        
        if outputs_valid and not kwargs.get('force', False):
            self.logger.info(f"Skipping Merge (Outputs exist: {out_nii})")
            
            # Load result
            merged_dwi = DWIFile(
                img=out_nii,
                entities=ents,
                bval=out_bval,
                bvec=out_bvec,
                json=first_dwi.json, # Approximate JSON from first file
                Delta=out_Delta if out_Delta.exists() else None,
                delta=out_delta if out_delta.exists() else None
            )
            
            # Update context
            context["dwi_files"] = [merged_dwi] # Replace list with single file
            context["merged_index"] = out_index
            context["merged_acqp"] = out_acqp
            merge_source_info = self._build_merge_source_info(dwi_files)
            context["merge_source_info"] = merge_source_info
            
            # Provide standard keys for downstream steps (Eddy)
            context["acqp"] = out_acqp
            context["index"] = out_index
            self._propagate_topup_mapping(context, dwi_files, merged_dwi)
            self._configure_merged_topup_groups(
                context,
                merged_dwi,
                out_acqp,
                out_index,
                merge_source_info,
            )
            
            return context
            
        self.logger.info(f"Merging {len(dwi_files)} DWI files...")
        
        # 1. Merge NIfTI
        in_imgs = [d.img for d in dwi_files]
        fsl.merge(in_files=in_imgs, out_file=out_nii)
        
        # 2. Merge Bvals/Bvecs/Deltas/deltas
        bvals = []
        bvecs = []
        Deltas = []
        deltas = []
        
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
                
                # Check for Delta/delta
                if d.Delta and d.Delta.exists():
                    D = np.loadtxt(d.Delta)
                    if D.ndim == 0: D = np.array([D])
                    if D.size == 1 and n_vols > 1:
                         # Broadcast scalar to all volumes
                         D = np.repeat(D, n_vols)
                    elif D.size != n_vols:
                         self.logger.warning(f"Delta file size {D.size} mismatch with bvals {n_vols} for {d.img.name}")
                    Deltas.append(D)
                    
                if d.delta and d.delta.exists():
                    d_s = np.loadtxt(d.delta)
                    if d_s.ndim == 0: d_s = np.array([d_s])
                    if d_s.size == 1 and n_vols > 1:
                         d_s = np.repeat(d_s, n_vols)
                    elif d_s.size != n_vols:
                         self.logger.warning(f"delta file size {d_s.size} mismatch with bvals {n_vols} for {d.img.name}")
                    deltas.append(d_s)
                
                # Identify acqp for this file
                # Use PhaseEncodingDirection and TotalReadoutTime from JSON
                pe_dir = infer_fsl_phase_encoding_direction(dwi=d, json_path=d.json, entities=d.entities)
                trt = _load_json_field(d.json, "TotalReadoutTime")
                eff_echo = _load_json_field(d.json, "EffectiveEchoSpacing")
                
                # FSL format: x y z time
                # x,y,z derived from PE dir (i, i-, j, j-, k, k-)
                if pe_dir is None:
                    self.logger.warning(f"Unknown or missing PhaseEncodingDirection '{pe_dir}' for {d.img.name}. Defaulting to [0,1,0].")
                    vec = [0, 1, 0]
                else:
                    try:
                        vec = fsl_phase_encoding_direction_to_vector(pe_dir).astype(int).tolist()
                    except ValueError as exc:
                        raise ProcessingError(str(exc)) from exc
                
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
                
        # Save merged bval/bvec/Delta/delta
        final_bval = np.concatenate(bvals)
        final_bvec = np.concatenate(bvecs, axis=1) # (3, N_total)
        
        np.savetxt(out_bval, final_bval, fmt='%g')
        np.savetxt(out_bvec, final_bvec, fmt='%.6f')
        
        if Deltas:
             if len(Deltas) == len(dwi_files): # Ensure all had it if any
                 final_Delta = np.concatenate(Deltas)
                 np.savetxt(out_Delta, final_Delta, fmt='%g')
             else:
                 self.logger.warning("Inconsistent Delta files across inputs. Merged Delta file not created.")
                 out_Delta = None # Prevent saving path to DWIFile
                 
        if deltas:
             if len(deltas) == len(dwi_files):
                 final_delta = np.concatenate(deltas)
                 np.savetxt(out_delta, final_delta, fmt='%g')
             else:
                 self.logger.warning("Inconsistent delta files across inputs. Merged delta file not created.")
                 out_delta = None
        
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
            json=first_dwi.json,
            Delta=out_Delta if out_Delta and out_Delta.exists() else None,
            delta=out_delta if out_delta and out_delta.exists() else None
        )
        
        context["dwi_files"] = [merged_dwi]
        context["merged_index"] = out_index
        context["merged_acqp"] = out_acqp
        merge_source_info = self._build_merge_source_info(dwi_files)
        context["merge_source_info"] = merge_source_info
        # We need to tell topup/eddy to use this acqp?
        # EddyStep should check context["merged_acqp"] and context["merged_index"]
        # And pass them to eddy.
        
        context["acqp"] = out_acqp
        context["index"] = out_index
        self._propagate_topup_mapping(context, dwi_files, merged_dwi)
        self._configure_merged_topup_groups(
            context,
            merged_dwi,
            out_acqp,
            out_index,
            merge_source_info,
        )
        
        return context

    def _build_merge_source_info(self, dwi_files: List[DWIFile]) -> list[dict[str, Any]]:
        source_info: list[dict[str, Any]] = []
        start = 0

        for dwi in dwi_files:
            if not dwi.bval or not Path(dwi.bval).exists():
                raise ProcessingError(f"MergeStep requires bval files for source bookkeeping: {dwi.img}")
            bvals = np.loadtxt(dwi.bval)
            bvals = np.atleast_1d(bvals)
            n_vols = int(bvals.size)
            stop = start + n_vols
            source_info.append(
                {
                    "img": str(dwi.img),
                    "json": str(dwi.json) if dwi.json else None,
                    "bval": str(dwi.bval) if dwi.bval else None,
                    "bvec": str(dwi.bvec) if dwi.bvec else None,
                    "entities": dict(getattr(dwi, "entities", {}) or {}),
                    "start": start,
                    "stop": stop,
                    "n_vols": n_vols,
                    "phase_encoding_direction": infer_phase_encoding_direction(dwi),
                }
            )
            start = stop

        return source_info

    def _propagate_topup_mapping(self, context: Dict[str, Any], source_dwis: List[DWIFile], merged_dwi: DWIFile) -> None:
        topup_map = context.get("topup_map", {})
        merged_base = None

        for dwi in source_dwis:
            key = getattr(dwi, "img", None)
            if key in topup_map:
                candidate = topup_map[key]
            elif key is not None and str(key) in topup_map:
                candidate = topup_map[str(key)]
            else:
                continue

            if merged_base is None:
                merged_base = candidate
            elif merged_base != candidate:
                merged_base = None
                break

        if merged_base is None:
            merged_base = context.get("topup_base")

        if merged_base:
            topup_map[merged_dwi.img] = merged_base
            topup_map[str(merged_dwi.img)] = merged_base
            context["topup_map"] = topup_map
            context["topup_base"] = merged_base

    def _configure_merged_topup_groups(
        self,
        context: Dict[str, Any],
        merged_dwi: DWIFile,
        acqp: Path,
        index: Path,
        merge_source_info: List[Dict[str, Any]],
    ) -> None:
        if not self._has_reverse_phase_sources(merge_source_info):
            return

        # After merge, Topup must consume the merged image plus merged acqp/index
        # so the field estimate stays on the same grid as downstream Eddy input.
        context["topup_groups"] = [{
            "inputs": [merged_dwi],
            "targets": [merged_dwi],
            "acqp": acqp,
            "index": index,
        }]

    def _has_reverse_phase_sources(self, merge_source_info: List[Dict[str, Any]]) -> bool:
        axis_signs: Dict[str, set[int]] = {}

        for source in merge_source_info:
            ped = source.get("phase_encoding_direction")
            if not ped:
                continue
            axis = str(ped)[0]
            sign = -1 if str(ped).endswith("-") else 1
            axis_signs.setdefault(axis, set()).add(sign)

        return any(len(signs) > 1 for signs in axis_signs.values())
