"""
dMRI Outlier Removal Step
"""
import glob
from pathlib import Path
from typing import Optional, List
import numpy as np
import nibabel as nib

from ...core.base import BaseProcessingStep
from ...core.types import DWIFile
from ...io.dmri.bids import build_bids_name
import json


def _resolve_volumes_file(
    volumes_file: str,
    entities: dict,
) -> Optional[Path]:
    """Resolve BIDS entity placeholders and globs for a manual index file."""
    template_values = {
        str(key): str(value)
        for key, value in (entities or {}).items()
        if value is not None
    }
    subject = template_values.get("sub") or template_values.get("subject")
    session = template_values.get("ses") or template_values.get("session")
    if subject is not None:
        template_values["sub"] = subject
        template_values["subject"] = subject
    if session is not None:
        template_values["ses"] = session
        template_values["session"] = session

    try:
        resolved_pattern = str(volumes_file).format_map(template_values)
    except KeyError as exc:
        available = ", ".join(sorted(template_values)) or "none"
        raise ValueError(
            f"Unknown or unavailable volumes_file placeholder {{{exc.args[0]}}}; "
            f"available entities: {available}."
        ) from exc

    if glob.has_magic(resolved_pattern):
        matches = sorted(
            Path(match)
            for match in glob.glob(resolved_pattern, recursive=True)
            if Path(match).is_file()
        )
    else:
        candidate = Path(resolved_pattern)
        matches = [candidate] if candidate.is_file() else []

    if len(matches) > 1:
        names = ", ".join(str(match) for match in matches)
        raise ValueError(
            "volumes_file pattern must resolve to exactly one file for each DWI; "
            f"{resolved_pattern!r} matched {len(matches)} files: {names}"
        )
    return matches[0] if matches else None


class OutlierRemovalStep(BaseProcessingStep):
    """
    Identifies and removes outlier volumes from DWI data.
    """
    
    def __init__(self, config, logger, provenance, method: str = "manual", threshold: float = 0.05, manual_indices: Optional[List[int]] = None, volumes_file: Optional[str] = None):
        super().__init__(config, logger, provenance)
        self.method = method
        self.threshold = threshold
        self.manual_indices = manual_indices
        self.volumes_file = volumes_file
        
    def run(self, context: dict, output_dir: Path, **kwargs) -> dict:
        """
        Run outlier removal.
        """
        dwi_files: List[DWIFile] = context.get("dwi_files", [])
        if not dwi_files:
             return context
             
        # Assumes working on 'current_image' or modifying the list?
        # Usually checking the processed data so far.
        # But 'dwi_files' in context might be the list of all inputs.
        # PreprocessingWorkflow handles context['current_image'].
        
        current_img: DWIFile = context.get("current_image")
        if not current_img:
             return context
        
        bad_indices = []
        
        # --- Check if output already exists ---
        # Use BIDS naming utilities
        from ...core.utils import get_nifti_stem
        stem = get_nifti_stem(current_img.img)
        
        # Determine output paths using entities
        out_ents = current_img.entities.copy()
        out_ents['desc'] = f"{out_ents.get('desc', '')}clean" if out_ents.get('desc') else 'clean'
        
        out_path = output_dir / build_bids_name(out_ents)
        
        # If output exists, we assume outlier removal was performed successfully previously.
        # Note: If previous run found NO outliers, out_path wouldn't exist, and we'd correctly fall through to check again.
        if out_path.exists() and not kwargs.get('force', False):
             try:
                 _ = nib.load(out_path)
             except Exception as e:
                 self.logger.warning(
                     f"Existing outlier-removed DWI is invalid ({out_path.name}): {e}. Removing and re-running."
                 )
                 try:
                     out_path.unlink(missing_ok=True)
                 except Exception:
                     pass
        if out_path.exists() and not kwargs.get('force', False):
             # Check timestamps
             in_mtime = current_img.img.stat().st_mtime
             out_mtime = out_path.stat().st_mtime
             
             if in_mtime > out_mtime:
                 self.logger.info(f"Outlier input ({current_img.img.name}) is newer than output. Re-running.")
                 self.logger.debug(f"Debug: Input mtime={in_mtime}, Output mtime={out_mtime}, Diff={in_mtime-out_mtime:.2f}s")
             else:
                 # Validate existing output before skipping
                 try:
                     _ = nib.load(out_path)
                 except Exception as e:
                     self.logger.warning(
                         f"Existing outlier-removed DWI is invalid ({out_path.name}): {e}. Re-running."
                     )
                 else:
                     self.logger.info(f"Skipping Outlier Removal (Output exists and up-to-date: {out_path.name})")
                 
                     # Construct output object using consistent BIDS naming
                     new_bval_path = None
                     new_bvec_path = None
                     new_Delta_path = None
                     new_delta_path = None
                     base_name = build_bids_name(out_ents)
                     
                     if current_img.bval:
                         bval_name = base_name.replace('.nii.gz', '.bval').replace('.nii', '.bval')
                         new_bval_path = output_dir / bval_name
                         # Fallback to original if cleaned version doesn't exist
                         if not new_bval_path.exists(): 
                             new_bval_path = current_img.bval
                     
                     if current_img.bvec:
                         bvec_name = base_name.replace('.nii.gz', '.bvec').replace('.nii', '.bvec')
                         new_bvec_path = output_dir / bvec_name
                         # Fallback to original if cleaned version doesn't exist
                         if not new_bvec_path.exists(): 
                             new_bvec_path = current_img.bvec

                     if getattr(current_img, "Delta", None):
                         Delta_name = base_name.replace('.nii.gz', '.bigdelta').replace('.nii', '.bigdelta')
                         candidate = output_dir / Delta_name
                         new_Delta_path = candidate if candidate.exists() else None

                     if getattr(current_img, "delta", None):
                         delta_name = base_name.replace('.nii.gz', '.delta').replace('.nii', '.delta')
                         candidate = output_dir / delta_name
                         new_delta_path = candidate if candidate.exists() else None
                     
                     new_dwi_file = DWIFile(
                          entities=current_img.entities,
                          img=out_path,
                          json=current_img.json,
                          bval=new_bval_path,
                          bvec=new_bvec_path,
                          Delta=new_Delta_path,
                          delta=new_delta_path,
                     )
                     
                     context["current_image"] = new_dwi_file
                     
                     # Attempt to load stats if possible? 
                     # For now, just return context to ensure downstream steps get the CLEAN image.
                     return context

        # 1. Identify Bad Indices
        if self.method == "manual":
             if self.manual_indices:
                  bad_indices = self.manual_indices
                  self.logger.info(f"Using manual outlier indices from config: {bad_indices}")
             elif self.volumes_file:
                  v_file = _resolve_volumes_file(
                      self.volumes_file,
                      current_img.entities,
                  )
                  if v_file is not None:
                       self.logger.info(f"Reading manual outlier indices from file: {v_file}")
                       try:
                            content = v_file.read_text().replace(",", " ").split()
                            bad_indices = [int(idx) for idx in content]
                       except Exception as e:
                            self.logger.error(f"Failed to read outlier volumes file {v_file}: {e}")
                  else:
                       self.logger.warning(
                           "No manual outlier file matched volumes_file pattern "
                           f"{self.volumes_file!r} for {current_img.img.name}."
                       )
                  
        elif self.method == "eddy_qc" or self.method == "threshold":
             # Look for eddy outlier report associated with the current image
             # The current image is likely the output of eddy (e.g. ..._desc-eddycorrected_dwi.nii.gz)
             # The outlier map should be ..._desc-eddycorrected_dwi.eddy_outlier_map
             
             # Expected path
             img_path = context.get('current_image').img
             # Remove extensions (.nii or .nii.gz)
             base_path = img_path.with_suffix("").with_suffix("") if img_path.name.endswith(".gz") else img_path.with_suffix("")
             
             outlier_map_path = base_path.with_suffix(".eddy_outlier_map")
             
             # Fallback: check if we just stripped only one suffix if it wasn't .gz?
             # Safe way:
             if not outlier_map_path.exists():
                  # Try appending to pure stem?
                  # If file is sub-01_desc-eddy_dwi.nii.gz, base is sub-01_desc-eddy_dwi
                  # map is sub-01_desc-eddy_dwi.eddy_outlier_map
                  pass
             
             if not outlier_map_path.exists():
                 self.logger.warning(f"Eddy outlier map not found at {outlier_map_path}. Skipping outlier removal.")
             else:
                 self.logger.info(f"Reading eddy outlier map: {outlier_map_path}")
                 try:
                     # Load map: rows=volumes, cols=slices
                     # 1 = outlier, 0 = valid
                     # Read as integer matrix
                     # Skip header row as it contains text
                     map_data = np.loadtxt(outlier_map_path, dtype=int, skiprows=1)
                     
                     # Check dimensions
                     # If only 1 volume, map_data might be 1D array of slices?
                     if map_data.ndim == 1:
                         # Single volume case
                         map_data = map_data.reshape(1, -1)
                     
                     n_vols, n_slices = map_data.shape
                     
                     for i in range(n_vols):
                         n_outliers = np.sum(map_data[i, :])
                         percent = n_outliers / n_slices
                         
                         if percent > self.threshold:
                             bad_indices.append(i)
                             self.logger.info(f"Volume {i}: {n_outliers}/{n_slices} slices are outliers ({percent:.2%}). Marked for removal.")
                             
                 except Exception as e:
                     self.logger.warning(f"Failed to parse eddy outlier map: {e}")
             
        elif self.method == "deep_learning":
             self.logger.warning("Deep learning outlier detection not implemented.")
        
        if not bad_indices:
             self.logger.info("No outliers identified to remove.")
             return context
             
        # 2. Remove Volumes
        self.logger.info(f"Removing {len(bad_indices)} outlier volumes: {bad_indices}")
        
        img = nib.load(current_img.img)
        data = img.get_fdata()
        
        # Determine keep indices
        n_vols = data.shape[-1]
        keep_indices = [i for i in range(n_vols) if i not in bad_indices]
        
        if len(keep_indices) == 0:
             raise ValueError("All volumes identified as outliers! Cannot proceed.")
             
        # Create cleaned image
        new_data = data[..., keep_indices]
        # Copy header to avoid modifying original
        new_header = img.header.copy()
        new_img = nib.Nifti1Image(new_data, img.affine.copy(), new_header)
        
        # Save new image using a temp file, then atomically replace
        tmp_name = f"{out_path.name}.tmp"
        if out_path.name.endswith(".nii.gz"):
            tmp_name = out_path.name.replace(".nii.gz", ".tmp.nii.gz")
        elif out_path.name.endswith(".nii"):
            tmp_name = out_path.name.replace(".nii", ".tmp.nii")
        tmp_path = out_path.with_name(tmp_name)
        
        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save with explicit flush to disk
        try:
            nib.save(new_img, tmp_path)
            # Force sync to disk before validation
            import os
            if hasattr(os, 'sync'):
                os.sync()
        except Exception as e:
            self.logger.error(f"Failed to save outlier-removed DWI to {tmp_path}: {e}")
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass
            context["current_image"] = current_img
            return context
        
        # Validate the saved file
        try:
            test_img = nib.load(tmp_path)
            test_data = test_img.get_fdata()
            assert test_data.shape == new_data.shape, f"Shape mismatch: {test_data.shape} != {new_data.shape}"
        except Exception as e:
            self.logger.warning(
                f"Outlier-removed DWI validation failed ({tmp_path.name}): {e}. Using original DWI instead."
            )
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass
            context["current_image"] = current_img
            return context
        
        # Atomic replace
        tmp_path.replace(out_path)
        self.logger.info(f"Saved outlier-removed DWI to: {out_path} (dims: {new_data.shape})")

        # Handle bvals/bvecs
        new_bval_path = None
        new_bvec_path = None
        new_Delta_path = None
        new_delta_path = None
        
        
        # Bvals
        if current_img.bval and current_img.bval.exists():
             try:
                 bvals = np.loadtxt(current_img.bval)
                 # bvals might be 1D or 2D row
                 if bvals.ndim == 1:
                      new_bvals = bvals[keep_indices]
                 else:
                      new_bvals = bvals[:, keep_indices] # Usually (N,)
                 
                 # Build the base BIDS name and replace image extension with .bval
                 base_name = build_bids_name(out_ents)
                 bval_name = base_name.replace('.nii.gz', '.bval').replace('.nii', '.bval')
                 new_bval_path = output_dir / bval_name
                 tmp_bval = output_dir / f"{bval_name}.tmp"
                 
                 # Save to temp file first
                 np.savetxt(tmp_bval, new_bvals.reshape(1, -1), fmt='%d') # FSL style: single row
                 
                 # Validate it can be read back
                 _ = np.loadtxt(tmp_bval)
                 
                 # Atomic replace
                 tmp_bval.replace(new_bval_path)
                 self.logger.info(f"Saved cleaned bvals to: {new_bval_path}")
             except Exception as e:
                 self.logger.error(f"Failed to save cleaned bvals: {e}")
                 if 'tmp_bval' in locals():
                     try:
                         tmp_bval.unlink(missing_ok=True)
                     except Exception:
                         pass
        
        # Bvecs
        if current_img.bvec and current_img.bvec.exists():
             try:
                 bvecs = np.loadtxt(current_img.bvec)
                 # bvecs is usually (3, N)
                 if bvecs.shape[0] == 3:
                      new_bvecs = bvecs[:, keep_indices]
                 else:
                      new_bvecs = bvecs[keep_indices, :].T # Reshape to 3xN logic?
                 
                 # Build the base BIDS name and replace image extension with .bvec
                 base_name = build_bids_name(out_ents)
                 bvec_name = base_name.replace('.nii.gz', '.bvec').replace('.nii', '.bvec')
                 new_bvec_path = output_dir / bvec_name
                 tmp_bvec = output_dir / f"{bvec_name}.tmp"
                 
                 # Save to temp file first
                 np.savetxt(tmp_bvec, new_bvecs, fmt='%.6f')
                 
                 # Validate it can be read back
                 _ = np.loadtxt(tmp_bvec)
                 
                 # Atomic replace
                 tmp_bvec.replace(new_bvec_path)
                 self.logger.info(f"Saved cleaned bvecs to: {new_bvec_path}")
             except Exception as e:
                 self.logger.error(f"Failed to save cleaned bvecs: {e}")
                 if 'tmp_bvec' in locals():
                     try:
                         tmp_bvec.unlink(missing_ok=True)
                     except Exception:
                         pass

        # Diffusion timings
        for attr, extension in (("Delta", ".bigdelta"), ("delta", ".delta")):
             timing_path = getattr(current_img, attr, None)
             if timing_path and timing_path.exists():
                 try:
                     timings = np.atleast_1d(np.loadtxt(timing_path))
                     if timings.size == 1 and n_vols > 1:
                         timings = np.repeat(timings, n_vols)
                     if timings.size != n_vols:
                         self.logger.warning(
                             f"{extension} file size {timings.size} does not match DWI volumes {n_vols}; not writing cleaned timing file."
                         )
                         continue
                     new_timings = timings[keep_indices]
                     base_name = build_bids_name(out_ents)
                     timing_name = base_name.replace('.nii.gz', extension).replace('.nii', extension)
                     new_timing_path = output_dir / timing_name
                     tmp_timing = output_dir / f"{timing_name}.tmp"
                     np.savetxt(tmp_timing, new_timings.reshape(1, -1), fmt='%.9g')
                     _ = np.loadtxt(tmp_timing)
                     tmp_timing.replace(new_timing_path)
                     if attr == "Delta":
                         new_Delta_path = new_timing_path
                     else:
                         new_delta_path = new_timing_path
                     self.logger.info(f"Saved cleaned {extension} timings to: {new_timing_path}")
                 except Exception as e:
                     self.logger.error(f"Failed to save cleaned {extension} timings: {e}")
                     if 'tmp_timing' in locals():
                         try:
                             tmp_timing.unlink(missing_ok=True)
                         except Exception:
                             pass

        # --- Calculate Statistics ---
        total_vols = n_vols
        num_removed = len(bad_indices)
        percent_removed = (num_removed / total_vols) * 100
        
        stats = {
            "total_volumes": total_vols,
            "removed_volumes": num_removed,
            "percent_removed": percent_removed,
            "bvalue_stats": []
        }
                
        # Detailed stats (reuse loaded bvals if possible, or reload)
        current_bvals = None
        if current_img.bval and current_img.bval.exists():
             try:
                 bvals = np.loadtxt(current_img.bval)
                 if bvals.ndim == 1:
                     current_bvals = bvals
                 else:
                     current_bvals = bvals.ravel()
                     
                 rounded_bvals = np.round(current_bvals, -2) # Round to nearest 100
                 unique_b = np.unique(rounded_bvals)
                 
                 for b in unique_b:
                     b_indices = np.where(rounded_bvals == b)[0]
                     total_b = len(b_indices)
                     removed_b = len(set(b_indices).intersection(set(bad_indices)))
                     pct_b = (removed_b / total_b * 100) if total_b > 0 else 0
                     
                     stats["bvalue_stats"].append({
                         "b_value": int(b),
                         "total": total_b,
                         "removed": removed_b,
                         "percent": pct_b
                     })
             except Exception as e:
                 self.logger.warning(f"Could not calculate b-value stats: {e}")
                 

        # Update context
        new_dwi_file = DWIFile(
             entities=current_img.entities,
             img=out_path,
             json=current_img.json,
             bval=new_bval_path,
             bvec=new_bvec_path,
             Delta=new_Delta_path,
             delta=new_delta_path,
        )
        
        context["current_image"] = new_dwi_file
        context["outlier_stats"] = stats
                
        # --- Save persistent stats JSON ---
        try:
             stats_ents = dict(current_img.entities)
             stats_ents['desc'] = 'outliers'
             stats_ents['suffix'] = 'stats'
             stats_json_name = build_bids_name(stats_ents).replace('.nii.gz', '').replace('.nii', '') + ".json"
             stats_json_path = output_dir / stats_json_name
             
             with open(stats_json_path, 'w') as f:
                  json.dump(stats, f, indent=4)
             self.logger.info(f"Saved outlier statistics to: {stats_json_path}")
        except Exception as e:
             self.logger.warning(f"Failed to save outlier stats JSON: {e}")


        return context
