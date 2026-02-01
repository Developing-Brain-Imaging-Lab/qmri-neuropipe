import pandas as pd
from pathlib import Path
import shutil
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from qmri_neuropipe.lib.common.tracker import NeuroimagingTracker

def test_tracker_persistence():
    test_tracker = Path("test_study_tracker.xlsx")
    if test_tracker.exists():
        test_tracker.unlink()
    
    # 1. Initialize and add Subject A
    tracker = NeuroimagingTracker.create_empty_tracker(test_tracker)
    tracker.update_status("sub-A", "ses-01", "DWI_Preproc", "Completed")
    tracker.add_metrics("sub-A", "ses-01", {"QC_DWI_SNR": 15.5})
    tracker.save(force=True)
    print("Added Subject A")

    # 2. Simulate another subject being added (separate session/run)
    # We re-load the tracker to simulate a new pipeline run
    tracker2 = NeuroimagingTracker(test_tracker)
    tracker2.update_status("sub-B", "ses-01", "DWI_Preproc", "Completed")
    tracker2.add_metrics("sub-B", "ses-01", {"QC_DWI_SNR": 20.1})
    tracker2.save(force=True)
    print("Added Subject B")

    # 3. Simulate re-running Subject A (Upsert)
    tracker3 = NeuroimagingTracker(test_tracker)
    tracker3.add_metrics("sub-A", "ses-01", {"QC_DWI_SNR": 18.0}) # Updated SNR
    tracker3.save(force=True)
    print("Updated Subject A (SNR 15.5 -> 18.0)")

    # 4. Verify results
    final_tracker = NeuroimagingTracker(test_tracker)
    df_status = final_tracker._data['Processing_Status']
    df_qc = final_tracker._data['Quality_Metrics']

    print("\n--- Final Processing Status ---")
    print(df_status[['Subject_ID', 'Session', 'DWI_Preproc_Status']])
    
    print("\n--- Final Quality Metrics ---")
    print(df_qc[['Subject_ID', 'Session', 'QC_DWI_SNR']])

    # Assertions
    assert len(df_status) == 2, f"Expected 2 subjects, got {len(df_status)}"
    assert df_qc.loc[df_qc['Subject_ID'] == 'sub-A', 'QC_DWI_SNR'].values[0] == 18.0
    
    # Check Sorting
    subjects = df_status['Subject_ID'].tolist()
    assert subjects == sorted(subjects), f"Tracker not sorted correctly: {subjects}"
    print("\nVerification Successful: Multi-subject persistence, Upsert, and Sorting confirmed.")

    # Cleanup
    if test_tracker.exists():
        test_tracker.unlink()
    if Path(str(test_tracker) + ".bak").exists():
        Path(str(test_tracker) + ".bak").unlink()

if __name__ == "__main__":
    test_tracker_persistence()
