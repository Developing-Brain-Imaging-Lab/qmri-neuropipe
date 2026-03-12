"""
dMRI QC Steps
"""
import json
from pathlib import Path
from ...core.base import BaseProcessingStep
from ...interfaces import fsl
from ...io.dmri.bids import build_acqp_index

class EddyQuadStep(BaseProcessingStep):
    """
    Run FSL eddy_quad for subject-level QC.
    """
    def run(self, context: dict, output_dir: Path, **kwargs) -> dict:
        dwi_files = context.get("dwi_files", [])
        if not dwi_files:
            return context
            
        current_img = context.get("current_image")
        if not current_img:
             return context

        # We need to find the eddy output basename.
        # This step should run AFTER eddy.
        # current_img.img points to ..._desc-eddycorrected_dwi.nii.gz
        
        img_path = current_img.img
        if "eddy" not in img_path.name:
             self.logger.warning("Current image does not appear to be eddy output. QC might fail.")
             
        # Strip extensions to get base
        base_path = img_path.with_suffix("").with_suffix("") if img_path.name.endswith(".gz") else img_path.with_suffix("")
        
        # Required inputs for quad
        # 1. eddy base (prefix of .eddy_rotated_bvecs, etc?)
        # Actually eddy_quad expects the prefix used in eddy --out call.
        # If pipeline renamed output to BIDS, we have to correspond.
        # 
        # Pipeline likely produced:
        #   sub-01_desc-eddycorrected_dwi.nii.gz
        #   sub-01_desc-eddycorrected_dwi.eddy_parameters
        #   sub-01_desc-eddycorrected_dwi.eddy_rotated_bvecs ...
        #
        # So base is indeed the stem.
        
        eddy_base = base_path
        
        # 2. bvals/bvecs (Must be the ones used for eddy?? or output?)
        # Quad docs: "bvals and bvecs files used in eddy"
        # Usually checking the *rotated* bvecs is good, but quad might want original?
        # Usage: eddy_quad <eddyOutputBase> ... -b <bvals> -g <bvecs>
        # It reads <eddyOutputBase>.eddy_parameters.
        # Let's provide the generated bvecs if available, or input.
        
        bvecs = current_img.bvec
        bvals = current_img.bval
        if not bvecs or not bvals:
             self.logger.warning("Missing bvecs/bvals for QC. Skipping.")
             return context
             
        # 3. Mask (nodif_mask)
        # Passed in context? Or use current mask?
        mask = context.get("current_mask")
        if not mask:
             # Try finding sidecar mask
             mask_name = base_path.name.replace("dwi", "mask") + ".nii.gz"
             mask = output_dir / mask_name
             if not mask.exists():
                 # Fallback to standard eddy mask name
                 mask_v2 = output_dir / "eddy_mask.nii.gz"
                 if mask_v2.exists():
                     mask = mask_v2
                 else:
                     self.logger.warning(f"No mask found for qc. Tried: {mask.name}, {mask_v2.name}")
                     return context
        else:
             # mask_obj might be Image or path
             if hasattr(mask, "img"): mask = mask.img
             
        # 4. Acqparams and Index
        # Should be in context
        acqp = context.get("acqp")
        index = context.get("index")
        
        if not acqp or not index:
             # Try building
             acqp, index = build_acqp_index(
                 current_img.json,
                 current_img.img,
                 support_dir=output_dir / "qc" / "eddy_support",
             )
             if not acqp or not index:
                 self.logger.warning("Missing acqp/index for QC.")
                 return context

        qc_out = output_dir / "qc" / "eddy_quad"
        qc_json = qc_out / "qc.json"

        # Check if already run
        if qc_json.exists() and not kwargs.get('force', False):
             self.logger.info("Skipping eddy_quad (Output exists)")
        else:
             # If directory exists but no json, incomplete run?
             # eddy_quad raises ValueError if dir exists.
             if qc_out.exists():
                 import shutil
                 self.logger.info(f"Removing incomplete QC directory: {qc_out}")
                 shutil.rmtree(qc_out)
             
             # qc_out.mkdir(parents=True, exist_ok=True) # REMOVED: eddy_quad requires dir to not exist
             # Ensure parent 'qc' dir exists
             qc_out.parent.mkdir(parents=True, exist_ok=True)
             
             try:
                 self.logger.info(f"Running eddy_quad on {eddy_base.name}")
                 fsl.eddy_quad(
                     eddy_base=eddy_base,
                     idx=index,
                     par=acqp,
                     mask=mask,
                     bvals=bvals,
                     bvecs=bvecs,
                     output_dir=qc_out,
                     verbose=True
                 )
             except Exception as e:
                 self.logger.warning(f"eddy_quad failed: {e}")
                 return context
            
        # Parse output
        qc_json = qc_out / "qc.json"
        if qc_json.exists():
            try:
                with open(qc_json) as f:
                    metrics = json.load(f)
                    
                # Extract key summaries
                # qc_json structure: {'qc_score': ..., 'qc_mot_abs': ..., 'qc_mot_rel': ..., 'qc_cnr_avg': ...}
                # Extract detailed summaries
                # 1. Motion
                motion_stats = {
                    "DWI_Motion_Abs_mm": metrics.get('qc_mot_abs', 0),
                    "DWI_Motion_Rel_mm": metrics.get('qc_mot_rel', 0),
                    "DWI_Motion_FD_Mean": metrics.get('qc_mot_rel', 0)
                }
                
                # 2. SNR/CNR
                cnr_stats = {}
                # B0 SNR
                if 'qc_s2s_b0_avg' in metrics:
                     cnr_stats["DWI_b0_SNR"] = metrics['qc_s2s_b0_avg']
                     cnr_stats["DWI_SNR"] = metrics['qc_s2s_b0_avg']
                
                # DWI CNR
                cnr_vals = metrics.get('qc_cnr_avg', [])
                for i, val in enumerate(cnr_vals):
                    cnr_stats[f"DWI_Shell_{i+1}_CNR"] = val
                    
                # 3. Outliers
                outlier_stats = {
                    "DWI_Outliers_Total_Pct": metrics.get('qc_outliers_tot', 0)
                }
                
                # Check for eddy_outlier_n_sqr if available to get raw counts
                # quad might not put raw counts in qc.json, but it might be in the output folder
                outlier_file = eddy_base.parent / (eddy_base.name + ".eddy_outlier_n_sqr")
                if outlier_file.exists():
                     try:
                         with open(outlier_file) as f:
                             # This file is a table of outliers per slice/volume
                             # For now, just count non-zeros? No, eddy_quad might have it.
                             pass
                     except: pass

                summary = {
                    **motion_stats,
                    **cnr_stats,
                    **outlier_stats
                }
                
                # Add overall QC score if present
                if 'qc_score' in metrics:
                     summary["DWI_Total_Score"] = metrics['qc_score']

                context["qc_metrics"] = summary

                # Save persistent summary for robust recovery (e.g. if re-running TrackingStep only)
                try:
                    summary_file = qc_out / "qc_summary.json"
                    with open(summary_file, 'w') as f:
                        json.dump(summary, f, indent=4)
                    self.logger.debug(f"Saved persistent QC summary to {summary_file}")
                except Exception as e_save:
                    self.logger.warning(f"Failed to save persistent QC summary: {e_save}")
            except Exception as e:
                 self.logger.warning(f"Failed to parse QC metrics: {e}")
            
        return context
