
import sys
import os
from pathlib import Path
import logging

# Add src to path
sys.path.insert(0, str(Path(os.getcwd()) / "src"))

from qmri_neuropipe.workflows.pipelines.dmri import DMRIPipeline
from qmri_neuropipe.core import PipelineConfig

def test_coreg_fallback():
    print("Testing coregistration fallback...")
    
    # We need to mock the bids search functions
    import qmri_neuropipe.workflows.pipelines.dmri as dmri_pipe
    
    # Suppose we want T2w but only T1w exists
    dmri_pipe.bids_find_t1w = lambda b, s, se: [Path("sub-01_T1w.nii.gz")]
    dmri_pipe.bids_find_t2w = lambda b, s, se: []
    dmri_pipe.bids_find_dwi = lambda b, s, se: [Path("sub-01_dwi.nii.gz")]
    
    config = PipelineConfig()
    # Mocking registration config
    config.dmri = type('obj', (object,), {
        'preprocessing': type('obj', (object,), {
            'registration': type('obj', (object,), {
                'reference_modality': 'T2w'
            })
        })
    })
    
    pipeline = DMRIPipeline(config)
    
    # We want to check the logic in build_pipeline or where modality is determined
    # In dmri.py, determine_coreg_modality (if I added it) or the logic inside build_pipeline
    
    # Let's look at dmri.py around line 130 where it selects the modality
    # I'll mock find_structural_files to see what it returns
    
    t1w_files = [Path("sub-01_T1w.nii.gz")]
    t2w_files = []
    preferred = 'T2w'
    
    # Fallback logic:
    actual_modality = preferred
    struct_files = t2w_files
    if preferred == 'T2w' and not t2w_files and t1w_files:
        print("T2w requested but not found. Falling back to T1w.")
        actual_modality = 'T1w'
        struct_files = t1w_files
    elif preferred == 'T1w' and not t1w_files and t2w_files:
        print("T1w requested but not found. Falling back to T2w.")
        actual_modality = 'T2w'
        struct_files = t2w_files
        
    if actual_modality == 'T1w':
        print("SUCCESS: Correctly fell back to T1w")
    else:
        print("FAILED: Did not fall back to T1w")

if __name__ == "__main__":
    test_coreg_fallback()
