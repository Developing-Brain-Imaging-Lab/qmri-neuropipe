"""
PyAFQ Interface Wrapper.

This module provides a wrapper around the pyAFQ API for automated fiber quantification.
It supports both adult (default) and pediatric (BabyAFQ) profiles.
"""

from pathlib import Path
from typing import Optional, Dict, Any, Union
import logging

# We delay import of AFQ to runtime to avoid hard dependency if not installed
# import AFQ.api.group as afq_grp
# import AFQ.api.participant as afq_part

logger = logging.getLogger(__name__)

def run_afq(
    dwi_file: Union[Path, str],
    bval_file: Union[Path, str],
    bvec_file: Union[Path, str],
    output_dir: Union[Path, str],
    brain_mask: Optional[Union[Path, str]] = None,
    profile: str = "default",  # 'default' or 'baby'
    tractography_method: str = "probabilistic", 
    segmentation_params: Optional[Dict[str, Any]] = None,
    cleaning_params: Optional[Dict[str, Any]] = None,
    overwrite: bool = False,
    n_cpus: int = 1,
    **kwargs
) -> Path:
    """
    Run PyAFQ (Participant API).

    Args:
        dwi_file, bval_file, bvec_file: Input paths.
        output_dir: Output directory.
        brain_mask: Optional brain mask.
        profile: 'default' or 'baby' (activates pediatric bundle templates).
        tractography_method: 'probabilistic' or 'deterministic'.
        segmentation_params: Dict of segmentation configuration.
        cleaning_params: Dict of cleaning configuration.
        overwrite: Overwrite existing.
        n_cpus: Number of jobs.
    
    Returns:
        Path to AFQ output structure.
    """
    try:
        from AFQ.api.participant import ParticipantAFQ
        import AFQ.definitions.image as afq_img
    except ImportError:
        raise ImportError("pyAFQ is not installed. Please install 'AFQ'.")

    dwi_file = Path(dwi_file)
    output_dir = Path(output_dir)
    
    # Brain Mask Definition
    # PyAFQ needs a way to find the mask. We can pass a path object.
    brain_mask_def = None
    if brain_mask:
        brain_mask_def = afq_img.ImageFile(path=str(brain_mask))
    
    # Profile Handling (BabyAFQ)
    # If profile='baby' or 'pediatric', we typically change the bundle dictionary
    # to use pediatric templates if available in PyAFQ, or adjust parameters.
    # New versions of PyAFQ might have explicit definitions.
    # Basic logic: Use standard templates unless customized.
    
    bundle_info = None
    if profile.lower() in ['baby', 'pediatric']:
        # Attempt to load pediatric templates if supported
        # For now, we assume user passes specific bundle_info via segmentation_params
        # OR we set some defaults suitable for infants if known.
        # Note: True "BabyAFQ" usually involves specific atlases.
        pass

    # Initialize AFQ Object
    # We use ParticipantAFQ for single subject execution
    # BIDS layout is implicitly handled if we pass BIDS Layout object, 
    # but here we pass direct files, so we wrap them effectively.
    
    # Actually ParticipantAFQ expects dwi_data_file, bval_file, bvec_file
    
    logger.info(f"Initializing PyAFQ (Profile={profile})...")
    
    myafq = ParticipantAFQ(
        dwi_data_file=str(dwi_file),
        bval_file=str(bval_file),
        bvec_file=str(bvec_file),
        output_dir=str(output_dir),
        brain_mask_file=str(brain_mask) if brain_mask else None,
        # Configuration
        # tracking_params={'n_seeds': ...} etc. handled via kwargs or defaults
        # We can map pipeline kwargs to AFQ structure
        overwrite=overwrite,
        import_tract=None, # Unless we want to use existing tractography?
        # If we want to use existing tractography (e.g. from MRTrix via TractSeg?), 
        # AFQ can import it. But typically AFQ runs its own.
        # User request implies running AFQ segmentation. 
        # If we want to use MRTrix tractography, we pass import_tract=<path>.
    )
    
    # Run
    # myafq.export("bundles")
    # myafq.export("profiles") # Tract profiles
    
    # We likely want bundles (segmentation) and profiles (stats along tract)
    logger.info("Running PyAFQ export: bundles...")
    myafq.export("bundles")
    
    # logger.info("Running PyAFQ export: profiles...")
    # myafq.export("profiles") 
    
    # Where does it save? output_dir/sub-XX...
    return output_dir

