import os
import sys
from pathlib import Path
import logging
import pandas as pd

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from qmri_neuropipe.core import BaseProcessingStep, PipelineConfig
from qmri_neuropipe.lib.common.tracker import NeuroimagingTracker
from qmri_neuropipe.lib.common.tracking import TrackingStep

# Set up logging
logging.basicConfig(level=logging.INFO)

class DummyStep(BaseProcessingStep):
    def run(self, context, output_dir, **kwargs):
        self.logger.info("Executing DummyStep")
        # Simulate some work
        context['dummy_status'] = 'completed'
        context['qc_metrics'] = {'SNR': 25.5, 'FD': 0.12}
        return context

def test_tracking():
    tracker_path = Path("test_tracker_run.xlsx")
    if tracker_path.exists(): tracker_path.unlink()
    
    # Init tracker
    NeuroimagingTracker.create_empty_tracker(tracker_path)
    
    # Create Config
    config = PipelineConfig(
        bids_dir=Path("."), # Not used here but required for validation
        output_dir=Path("./output"),
        tracker=NeuroimagingTracker(tracker_path),
        config_data={'tracker_file': str(tracker_path)}
    )
    
    # 1. Test BaseProcessingStep Hook
    context = {
        'subject': 'sub-01',
        'session': 'ses-01',
        'study_name': 'SIM_STUDY'
    }
    
    step = DummyStep(config)
    context = step(context, Path("./output"))
    
    # Check if status was updated
    df_status = pd.read_excel(tracker_path, sheet_name='Processing_Status')
    print("\nProcessing Status after DummyStep:")
    print(df_status)
    
    assert df_status.iloc[0]['dummy_Status'] == 'completed'
    
    # 2. Test TrackingStep explicitly (aggregation)
    tracking_step = TrackingStep(config)
    # Add some mock ROI stats files to context
    roi_stats = Path("roi_stats.tsv")
    with open(roi_stats, "w") as f:
        f.write("LabelID\tLabelName\tMean\tStd\tCount\n")
        f.write("1\tCortex\t1.5\t0.1\t100\n")
        f.write("2\tWM\t0.8\t0.05\t200\n")
        
    context['roi_stats_files'] = {'DTI': str(roi_stats)}
    
    tracking_step.run(context, Path("./output"))
    
    # Check DTI Metrics
    df_dti = pd.read_excel(tracker_path, sheet_name='DTI_Metrics')
    print("\nDTI Metrics after TrackingStep:")
    print(df_dti)
    
    assert df_dti.iloc[0]['ROI_Cortex_Mean'] == 1.5
    
    # Check QC Metrics
    df_qc = pd.read_excel(tracker_path, sheet_name='Quality_Metrics')
    print("\nQC Metrics after TrackingStep:")
    print(df_qc)
    assert df_qc.iloc[0]['SNR'] == 25.5

    print("\n✅ Verification Successful!")
    
    # Cleanup
    if roi_stats.exists(): roi_stats.unlink()
    # if tracker_path.exists(): tracker_path.unlink()

if __name__ == "__main__":
    test_tracking()
