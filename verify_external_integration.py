import sys
from pathlib import Path
import os
import shutil
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path.cwd() / "src"))

from qmri_neuropipe.lib.common.tracker import NeuroimagingTracker
from qmri_neuropipe.lib.common.tracking import TrackingStep
from qmri_neuropipe.core import PipelineConfig

def test_external_integration():
    # 1. Setup mock session
    test_dir = Path("test_external_integration")
    if test_dir.exists(): shutil.rmtree(test_dir)
    test_dir.mkdir()
    
    tracker_path = test_dir / "study_tracker.xlsx"
    
    # 2. Initialize Tracker
    tracker = NeuroimagingTracker.create_empty_tracker(tracker_path)
    
    # 3. Simulate Status Updates (Primary and Modality Specific)
    # Subject 1: Diffusion Complete
    tracker.update_status("SUB01", "ses-01", "Overall_Pipeline", "Complete", "TestStudy")
    tracker.update_status("SUB01", "ses-01", "Preprocessing", "Complete", "TestStudy", modality="Diffusion")
    tracker.update_status("SUB01", "ses-01", "Analysis", "Complete", "TestStudy", modality="Diffusion")
    tracker.update_status("SUB01", "ses-01", "Overall", "Complete", "TestStudy", modality="Diffusion")
    
    # Subject 2: Anatomical In Progress
    tracker.update_status("SUB02", "ses-01", "Overall_Pipeline", "In Progress", "TestStudy")
    tracker.update_status("SUB02", "ses-01", "Preprocessing", "Complete", "TestStudy", modality="Anatomical")
    tracker.update_status("SUB02", "ses-01", "Analysis", "In Progress", "TestStudy", modality="Anatomical")
    tracker.update_status("SUB02", "ses-01", "Overall", "In Progress", "TestStudy", modality="Anatomical")
    
    # Subject 3: Failed
    tracker.update_status("SUB03", "ses-01", "Overall_Pipeline", "Failed", "TestStudy")
    tracker.update_status("SUB03", "ses-01", "Preprocessing", "Complete", "TestStudy", modality="Relaxometry")
    tracker.update_status("SUB03", "ses-01", "Analysis", "Error", "TestStudy", modality="Relaxometry")
    tracker.update_status("SUB03", "ses-01", "Overall", "Failed", "TestStudy", modality="Relaxometry")
    
    # 4. Save (triggers summary recalculation and styling)
    tracker.save(force=True)
    
    # 5. Verify Sheets
    print(f"Verifying tracker at {tracker_path}...")
    with pd.ExcelFile(tracker_path) as xls:
        print(f"Sheets: {xls.sheet_names}")
        
        # Check Summary
        df_sum = pd.read_excel(xls, "Summary")
        print("\n--- Summary ---")
        print(df_sum.to_string(index=False))
        
        assert "Total Subjects" in df_sum['Metric'].values
        assert df_sum[df_sum['Metric'] == 'Total Subjects']['Value'].values[0] == 3
        
        # Check Modality Status
        for mod in ["Anatomical", "Diffusion", "Relaxometry"]:
            sheet = f"{mod}_Status"
            df_mod = pd.read_excel(xls, sheet)
            print(f"\n--- {sheet} ---")
            print(df_mod.to_string(index=False))
            assert len(df_mod) >= 1
            
    print("\nTracker Integration Verified Backend Successfully!")
    print("Next step: Manual verification of Excel colors and Dashboard views.")

if __name__ == "__main__":
    test_external_integration()
