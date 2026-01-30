import pandas as pd
from pathlib import Path
from qmri_neuropipe.lib.common.tracker import NeuroimagingTracker
import multiprocessing as mp
import time
import os
import random

def update_tracker(subject_id, tracker_path):
    print(f"Process {os.getpid()} updating {subject_id}")
    tracker = NeuroimagingTracker(tracker_path)
    
    # Simulate some work
    time.sleep(random.uniform(0.1, 0.5))
    
    tracker.update_status(subject_id, "ses-1", "DTI", "completed")
    tracker.add_metrics(subject_id, "ses-1", {"SNR": random.uniform(20, 50)})
    
    # Simulate a crash in one of the processes during save (optional/manual)
    # In a real test, we would kill it, but for automated test we just call save
    tracker.save(force=True)
    print(f"Process {os.getpid()} finished {subject_id}")

def test_concurrency():
    tracker_path = Path("test_robust_tracker.xlsx")
    if tracker_path.exists(): tracker_path.unlink()
    
    # Initialize tracker
    NeuroimagingTracker.create_empty_tracker(tracker_path)
    
    subjects = [f"sub-{i:03d}" for i in range(1, 11)]
    
    with mp.Pool(processes=4) as pool:
        pool.starmap(update_tracker, [(sub, tracker_path) for sub in subjects])
    
    # Verify results
    tracker = NeuroimagingTracker(tracker_path)
    status_df = tracker._data['Processing_Status']
    print("\nFinal Processing Status:")
    print(status_df[['Subject_ID', 'DTI_Status']])
    
    assert len(status_df) == 10, f"Expected 10 subjects, found {len(status_df)}"
    assert (status_df['DTI_Status'] == 'completed').all(), "Not all subjects are completed"
    print("\nConcurrency test passed!")

def test_stale_lock():
    tracker_path = Path("test_stale_tracker.xlsx")
    if tracker_path.exists(): tracker_path.unlink()
    NeuroimagingTracker.create_empty_tracker(tracker_path)
    
    lock_path = tracker_path.with_suffix(".xlsx.lock")
    
    # Create a stale lock file with a non-existent PID
    with open(lock_path, 'w') as f:
        f.write("999999") # Assuming this PID doesn't exist
    
    print(f"\nCreated stale lock for PID 999999")
    
    tracker = NeuroimagingTracker(tracker_path)
    tracker.update_status("sub-stale", "ses-1", "DTI", "completed")
    
    # This should detect the stale lock and proceed
    start_time = time.time()
    tracker.save(force=True)
    end_time = time.time()
    
    print(f"Save completed in {end_time - start_time:.2f} seconds")
    assert not lock_path.exists(), "Lock file should have been removed"
    print("Stale lock test passed!")

if __name__ == "__main__":
    test_concurrency()
    test_stale_lock()
