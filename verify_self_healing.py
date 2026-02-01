import sys
from pathlib import Path
import os
import shutil
import pandas as pd
from openpyxl import load_workbook

# Add src to path
sys.path.insert(0, str(Path.cwd() / "src"))

from qmri_neuropipe.lib.common.tracker import NeuroimagingTracker
from qmri_neuropipe.lib.common.tracking import TrackingStep
from qmri_neuropipe.core import PipelineConfig

def test_self_healing():
    # 1. Setup mock session
    test_dir = Path("test_self_healing")
    if test_dir.exists(): shutil.rmtree(test_dir)
    test_dir.mkdir()
    
    tracker_path = test_dir / "study_tracker.xlsx"
    bids_dir = test_dir / "bids"
    bids_dir.mkdir()
    
    # 2. Simulate Filesystem Derivatives (Completed Steps)
    sub = "SUB99"
    ses = "ses-01"
    sub_prefix = f"sub-{sub}_ses-{ses}"
    session_root = test_dir / "derivatives" / "qmri-neuropipe" / sub / ses
    session_root.mkdir(parents=True)
    
    # Session Folders
    dwi_path = session_root / "dwi"
    dwi_path.mkdir()
    anat_path = session_root / "anat"
    anat_path.mkdir()
    dti_path = session_root / "dti"
    dti_path.mkdir()
    
    # Pre-existing Files (DWI)
    (dwi_path / f"{sub_prefix}_desc-denoised_dwi.nii.gz").touch()
    (dwi_path / f"{sub_prefix}_desc-eddy_dwi.nii.gz").touch()
    
    # Pre-existing Files (Anatomical - T2w & Mask)
    (anat_path / f"{sub_prefix}_desc-denoised_T2w.nii.gz").touch()
    (anat_path / f"{sub_prefix}_desc-brain_mask.nii.gz").touch()
    
    # Pre-existing Model
    (dti_path / f"{sub_prefix}_model-dti_FA.nii.gz").touch()
    
    # Pre-existing FS
    fs_sub_dir = bids_dir / "derivatives" / "freesurfer" / sub_prefix
    (fs_sub_dir / "mri").mkdir(parents=True)
    (fs_sub_dir / "mri" / "brain.mgz").touch()
    
    # 3. Initialize Tracker
    tracker = NeuroimagingTracker.create_empty_tracker(tracker_path)
    
    # 4. Run TrackingStep with EMPTY context (simulating a skip)
    config = PipelineConfig()
    config.bids_dir = bids_dir
    config.tracker_file = tracker_path
    config.tracker = tracker
    
    tstep = TrackingStep(config)
    
    # Simulate a DWI re-run where only 1 step is in context
    context = {
        'subject': sub,
        'session': ses,
        'study_name': 'SelfHealingStudy',
        'reorienting_status': 'Complete' # Only this one is "new"
    }
    
    print(f"Running tracking for {sub_prefix} with partial context...")
    # Wrap in a dwi subfolder to trigger inference
    dwi_output = session_root / "dwi"
    tstep.run(context, dwi_output)
    
    # 5. Verify Results
    print(f"Verifying tracker at {tracker_path}...")
    
    with pd.ExcelFile(tracker_path) as xls:
        print(f"Sheets present: {xls.sheet_names}")
        assert 'Processing_Times' not in xls.sheet_names
        assert 'Errors_Notes' not in xls.sheet_names
        
        # Check Diffusion sheet (should have Eddy/Denoise complete via healing)
        df_dwi = pd.read_excel(xls, "Diffusion_Status")
        print("\n--- Diffusion Status (Self-Healed) ---")
        print(df_dwi.to_string(index=False))
        
        assert df_dwi['Denoising'].iloc[0] == 'Complete'
        assert df_dwi['Eddy_Correction'].iloc[0] == 'Complete'
        assert df_dwi['Reorienting'].iloc[0] == 'Complete'
        assert 'DTI' in str(df_dwi['Model_Fits'].iloc[0])

        # Check Anatomical Cross-Modality Recovery
        # Even though we ran DWI tracking, it should have picked up Anat files!
        df_anat = pd.read_excel(xls, "Anatomical_Status")
        print("\n--- Anatomical Status (Cross-Modality Recovery) ---")
        print(df_anat.to_string(index=False))
        assert df_anat['Denoising'].iloc[0] == 'Complete'  # From T2w detection
        assert df_anat['Brain_Masking'].iloc[0] == 'Complete' # From mask detection
        assert df_anat['Segmentation'].iloc[0] == 'Complete' # From FS detection
        
    # Run Tracking for Anat to test FS healing
    print(f"\nRunning tracking for Anat now...")
    context_anat = {
        'subject': sub,
        'session': ses,
        'study_name': 'SelfHealingStudy'
    }
    # Use the actual anat folder to trigger inference
    tstep.run(context_anat, anat_path)
    
    # Verify Anat Results (New Open)
    with pd.ExcelFile(tracker_path) as xls:
        df_anat = pd.read_excel(xls, "Anatomical_Status")
        print("\n--- Anatomical Status (Self-Healed) ---")
        print(df_anat.to_string(index=False))
        assert df_anat['Segmentation'].iloc[0] == 'Complete'
        assert df_anat['Segmentation_Method'].iloc[0] == 'FreeSurfer'

    print("\nSelf-Healing & Cleanup Verified Successfully!")

if __name__ == "__main__":
    test_self_healing()
