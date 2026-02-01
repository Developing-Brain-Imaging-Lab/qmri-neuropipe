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

def test_granular_tracking():
    # 1. Setup mock session
    test_dir = Path("test_granular_integration")
    if test_dir.exists(): shutil.rmtree(test_dir)
    test_dir.mkdir()
    
    tracker_path = test_dir / "study_tracker.xlsx"
    
    # 2. Initialize Tracker
    tracker = NeuroimagingTracker.create_empty_tracker(tracker_path)
    
    # 3. Simulate Granular Status Updates via Context
    # Modality: Diffusion
    context_dwi = {
        'subject': 'SUB01',
        'session': 'ses-01',
        'study_name': 'GranularStudy',
        'denoising_status': 'Complete',
        'eddy_status': 'Complete',
        'topup_status': 'N/A',
        'synb0_status': 'Complete',
        'models_fitted': ['DTI', 'DKI'],
        'roi_stats_files': {'JHU': 'fake.tsv', 'HarvardOxford': 'fake2.tsv'}
    }
    
    # Run tracking step logic (simulated)
    # Actually, let's just use a TrackingStep instance
    from qmri_neuropipe.core import PipelineConfig
    config = PipelineConfig()
    config.tracker_file = tracker_path
    config.tracker = tracker
    
    tstep = TrackingStep(config)
    # Simulate being in a dwi output directory
    output_dwi = test_dir / "dwi"
    output_dwi.mkdir(exist_ok=True)
    tstep.run(context_dwi, output_dwi)
    
    # Modality: Anatomical
    context_anat = {
        'subject': 'SUB01',
        'session': 'ses-01',
        'study_name': 'GranularStudy',
        'recon_all_status': 'Complete',
        'brain_masking_status': 'Complete',
        'roi_stats_files': {'Aseg': 'fake.tsv'}
    }
    output_anat = test_dir / "anat"
    output_anat.mkdir(exist_ok=True)
    tstep.run(context_anat, output_anat)
    
    # 4. Save and Verify
    tracker.save(force=True)
    
    print(f"Verifying granular tracker at {tracker_path}...")
    
    # Verify Columns
    with pd.ExcelFile(tracker_path) as xls:
        df_dwi = pd.read_excel(xls, "Diffusion_Status")
        print("\n--- Diffusion Status ---")
        print(df_dwi.columns.tolist())
        assert 'Eddy_Correction' in df_dwi.columns
        assert 'Model_Fits' in df_dwi.columns
        assert df_dwi['Eddy_Correction'].iloc[0] == 'Complete'
        assert 'DTI, DKI' in str(df_dwi['Model_Fits'].iloc[0])
        
        df_anat = pd.read_excel(xls, "Anatomical_Status")
        print("\n--- Anatomical Status ---")
        print(df_anat.columns.tolist())
        assert 'Segmentation' in df_anat.columns
        assert df_anat['Segmentation_Method'].iloc[0] == 'FreeSurfer'
        
    # Verify Data Validation (Dropdowns)
    wb = load_workbook(tracker_path)
    ws = wb['Diffusion_Status']
    print(f"\nData Validations in Diffusion_Status: {len(ws.data_validations.dataValidation)}")
    assert len(ws.data_validations.dataValidation) > 0
    
    dv = ws.data_validations.dataValidation[0]
    print(f"Validation Type: {dv.type}")
    print(f"Validation Formula: {dv.formula1}")
    assert "Complete" in dv.formula1
    assert "Manual Pass" in dv.formula1
    
    print("\nGranular Tracking & Excel Interactivity Verified Successfully!")

if __name__ == "__main__":
    test_granular_tracking()
