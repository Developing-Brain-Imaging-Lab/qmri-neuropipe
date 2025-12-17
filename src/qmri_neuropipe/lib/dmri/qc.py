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
             acqp, index = build_acqp_index(current_img.json, current_img.img)
             if not acqp or not index:
                 self.logger.warning("Missing acqp/index for QC.")
                 return context

        qc_out = output_dir / "qc" / "eddy_quad"
        qc_json = qc_out / "qc.json"

        # Check if already run
        if qc_json.exists():
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
                    "Absolute Motion (mm)": f"{metrics.get('qc_mot_abs', 0):.2f}",
                    "Relative Motion (mm)": f"{metrics.get('qc_mot_rel', 0):.2f}"
                }
                
                # 2. SNR/CNR
                # qc_cnr_avg is list of CNRs per b-shell (excluding b=0)
                # qc_s2s_b0_avg is scalar/list for b=0
                # b-values are in 'qc_bvals' ?? No, usually we infer or just list them.
                # qc.json often has 'bvals' key with actual values used.
                
                cnr_stats = []
                # B0 SNR
                if 'qc_s2s_b0_avg' in metrics:
                     cnr_stats.append({"Shell": "b=0 (SNR)", "Value": f"{metrics['qc_s2s_b0_avg']:.2f}"})
                
                # DWI CNR
                cnr_vals = metrics.get('qc_cnr_avg', [])
                for i, val in enumerate(cnr_vals):
                    cnr_stats.append({"Shell": f"Shell {i+1} (CNR)", "Value": f"{val:.2f}"})
                    
                # 3. Outliers Breakdown
                outlier_stats = {
                    "Total Outliers (%)": f"{metrics.get('qc_outliers_tot', 0):.2f}"
                }
                
                # Per-shell outliers
                outliers_b = metrics.get('qc_outliers_b', [])
                outlier_breakdown = []
                for i, val in enumerate(outliers_b):
                     outlier_breakdown.append({"Category": f"Shell {i+1}", "Outliers (%)": f"{val:.2f}"})
                     
                # Per-PE outliers
                outliers_pe = metrics.get('qc_outliers_pe', [])
                for i, val in enumerate(outliers_pe):
                     outlier_breakdown.append({"Category": f"PE Dir {i+1}", "Outliers (%)": f"{val:.2f}"})
                
                summary = {
                    "motion": motion_stats,
                    "cnr": cnr_stats,
                    "outliers_summary": outlier_stats,
                    "outliers_breakdown": outlier_breakdown
                }
                
                context["qc_metrics"] = summary
            except Exception as e:
                 self.logger.warning(f"Failed to parse QC metrics: {e}")
            
        return context
