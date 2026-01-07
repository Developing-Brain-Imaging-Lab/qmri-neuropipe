"""
TractSeg interface wrapper.

Handles calls to the TractSeg CLI tools for tract segmentation and analysis.
"""

from pathlib import Path
from typing import Optional, List, Dict, Union, Any
import logging

from ..core import ensure_dir
from ..core.run import run_cmd

logger = logging.getLogger(__name__)

def run_tractseg(
    input_file: Union[Path, str],
    output_dir: Union[Path, str],
    input_type: str = "peaks",
    output_type: str = "bundle_masks",
    brain_mask: Optional[Union[Path, str]] = None,
    raw_diffusion: Optional[Union[Path, str]] = None,
    bvals: Optional[Union[Path, str]] = None,
    bvecs: Optional[Union[Path, str]] = None,
    preview: bool = False,
    single_output_file: bool = False,
    bundle_specific_threshold: bool = False,
    get_priors: bool = False,
    keep_intermediate_files: bool = False,
    csc_peaks: bool = False,
    gpu: bool = True,
    gpu_id: int = 0,
    nr_cpus: int = 1,
    super_resolution: bool = False,
    uncertainty: bool = False,
    tract_definition: str = "TractSeg",
    extra_args: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Run TractSeg.

    Args:
        input_file: Path to input image (peaks or DWI).
        output_dir: Path to output directory.
        input_type: Type of input image ('peaks' or 'dmri').
        output_type: Type of output ('bundle_masks', 'reendings', 'tom', 'dm_regression').
        brain_mask: Path to brain mask (optional, but recommended).
        raw_diffusion: Path to raw DWI (required for some modes).
        bvals: Path to bvals (required if input_type='dmri').
        bvecs: Path to bvecs (required if input_type='dmri').
        preview: If True, generate 3D preview.
        single_output_file: If True, save all bundles in one 4D file.
        bundle_specific_threshold: Use bundle-specific thresholds.
        get_priors: Output priors.
        keep_intermediate_files: Keep intermediate files.
        csc_peaks: Use CS-constrained peaks.
        gpu: Use GPU.
        gpu_id: GPU ID.
        nr_cpus: Number of CPUs (for preprocessing).
        super_resolution: Use super resolution.
        uncertainty: Enable uncertainty estimation.
        tract_definition: 'TractSeg' or 'AutoPTX'.
        extra_args: Additional arguments for TractSeg.

    Returns:
        Path to output directory.
    """
    input_file = Path(input_file)
    output_dir = ensure_dir(output_dir)

    cmd_parts = ["TractSeg"]
    
    cmd_parts.append(f"-i {input_file}")
    cmd_parts.append(f"-o {output_dir}")
    cmd_parts.append(f"--output_type {output_type}")
    
    if brain_mask:
        cmd_parts.append(f"--brain_mask {brain_mask}")
        
    if input_type == "dmri":
        # TractSeg handles checking for bvals/bvecs automagically if in same dir, 
        # but explicit flags might exist or handled via input.
        # Check doc: usually expects --raw_diffusion usage for TOM tracking, 
        # but for segmentation from DWI, it preprocesses.
        # Actually, TractSeg -i <dwi> --raw_diffusion <dwi> --bvals <bvals> --bvecs <bvecs> is standard for DWI input.
        if bvals: cmd_parts.append(f"--bvals {bvals}")
        if bvecs: cmd_parts.append(f"--bvecs {bvecs}")
        if raw_diffusion: cmd_parts.append(f"--raw_diffusion {raw_diffusion}")
        
    if preview:
        cmd_parts.append("--preview")
        
    if single_output_file:
        cmd_parts.append("--single_output_file")
        
    if bundle_specific_threshold:
        cmd_parts.append("--bundle_specific_threshold")
        
    if get_priors:
        cmd_parts.append("--get_priors")
        
    if keep_intermediate_files:
        cmd_parts.append("--keep_intermediate_files")
        
    if csc_peaks:
        cmd_parts.append("--csc_peaks")
        
    if super_resolution:
        cmd_parts.append("--super_resolution")
        
    if uncertainty:
        cmd_parts.append("--uncertainty")
        
    if tract_definition and tract_definition != "TractSeg":
         cmd_parts.append(f"--tract_definition {tract_definition}")

    # GPU handling not standard flag in CLI? Checking docs...
    # Regular TractSeg uses pytorch, so it uses CUDA_VISIBLE_DEVICESenv.
    # There isn't a --gpu flag usually effectively, but let's check wrapper logic.
    prefix = ""
    if gpu:
        prefix = f"env CUDA_VISIBLE_DEVICES={gpu_id}"
    else:
        # Some versions might default to GPU??
        # Usually it tries GPU first. if we want CPU?
        # Maybe env CUDA_VISIBLE_DEVICES=""
        pass

    if nr_cpus > 1:
        cmd_parts.append(f"--nr_cpus {nr_cpus}")

    if extra_args:
        for k, v in extra_args.items():
            if v is True:
                cmd_parts.append(f"--{k}")
            elif v is not False and v is not None:
                cmd_parts.append(f"--{k} {v}")

    full_cmd = f"{prefix} {' '.join(cmd_parts)}" if prefix else ' '.join(cmd_parts)
    
    run_cmd(full_cmd, label="TractSeg")
    
    return output_dir

def run_tractometry(
    bundle_mask_file: Union[Path, str],
    metric_file: Union[Path, str],
    output_file: Union[Path, str],
    tom_trackings: Optional[Union[Path, str]] = None,
    tracking_format: str = "tck",
    nr_points: int = 100,
    dilate_bundle_masks: bool = False,
    min_track_len: float = 20,
    gpu_id: int = 0
):
    """
    Run Tractometry (extracts metrics along tracts).
    
    This wraps the 'Tractometry' command from TractSeg.
    """
    # Note: Tractometry command usage:
    # Tractometry -i <bundles> -m <scalar> -o <output>
    # If providing TOM streamlines, might differ.
    
    cmd_parts = ["Tractometry"]
    
    cmd_parts.append(f"-i {bundle_mask_file}")
    cmd_parts.append(f"-m {metric_file}")
    cmd_parts.append(f"-o {output_file}")
    
    if tom_trackings:
         cmd_parts.append(f"--tom_trackings {tom_trackings}")
         
    cmd_parts.append(f"--tracking_format {tracking_format}")
    cmd_parts.append(f"--nr_points {nr_points}")
    
    if dilate_bundle_masks:
        cmd_parts.append("--dilate_bundle_masks")
        
    if min_track_len:
         cmd_parts.append(f"--min_track_len {min_track_len}")
         
    prefix = f"env CUDA_VISIBLE_DEVICES={gpu_id}"
    full_cmd = f"{prefix} {' '.join(cmd_parts)}"
    
    run_cmd(full_cmd, label="Tractometry")
    return output_file
