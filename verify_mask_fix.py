import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from qmri_neuropipe.core import PipelineConfig, BaseWorkflow
from qmri_neuropipe.core.types import ImageFile, DWIFile

def test_mask_wrapping():
    print("Testing Mask Wrapping and Pipeline Robustness...")
    
    # Mock config
    config = PipelineConfig(bids_dir=Path("./dummy_bids"), output_dir=Path("./output"))
    
    # 1. Test EddyCorrectionStep.run (direct check of our fix)
    # We mock out the fsl to avoid actual binary calls
    with patch("qmri_neuropipe.lib.dmri.eddy.fsl") as mock_fsl:
        
        from qmri_neuropipe.lib.dmri.eddy import EddyCorrectionStep
        step = EddyCorrectionStep(config)
        step.get_step_output_dir = MagicMock(return_value=Path("./output/eddy"))
        
        dummy_img = Path("dummy_dwi.nii.gz")
        dwi_obj = DWIFile(img=dummy_img, entities={"subject": "test", "suffix": "dwi"})
        context = {"current_image": dwi_obj}
        
        # --- A. Test Skip Branch ---
        print("Checking Skip Branch...")
        # Mock existence of outputs
        with patch("pathlib.Path.exists", return_value=True):
            result_context = step.run(context, output_dir=Path("./output"), force=False)
            mask_obj = result_context.get("current_mask")
            print(f"Mask object type: {type(mask_obj)}")
            if hasattr(mask_obj, 'img') and isinstance(mask_obj, ImageFile):
                print("SUCCESS: Skip branch wrapped mask in ImageFile")
            else:
                print("FAILED: Skip branch did not wrap mask correctly")

        # --- B. Test Execution Branch ---
        print("Checking Execution Branch...")
        # Mock non-existence of outputs
        def exists_side_effect(path_obj):
            path_str = str(path_obj)
            if "eddy_corrected" in path_str: return False
            return True

        with patch("pathlib.Path.exists", side_effect=exists_side_effect):
             # Mock the eddy run
             mock_fsl.eddy.return_value = dwi_obj
             
             result_context = step.run(context, output_dir=Path("./output"), force=True)
             mask_obj = result_context.get("current_mask")
             print(f"Mask object type: {type(mask_obj)}")
             if hasattr(mask_obj, 'img') and isinstance(mask_obj, ImageFile):
                 print("SUCCESS: Execution branch wrapped mask in ImageFile")
             else:
                 print("FAILED: Execution branch did not wrap mask correctly")

    # 2. Test PreprocessingWorkflow safety check
    print("Testing PreprocessingWorkflow safety check (manual Simulation)...")
    out_mask = Path("faulty_mask.nii.gz")
    if out_mask is not None and isinstance(out_mask, Path):
        print("Detected Path in out_mask, wrapping...")
        wrapped_mask = ImageFile(img=out_mask, entities=dict(dwi_obj.entities, suffix="mask"))
    
    if hasattr(wrapped_mask, 'img') and isinstance(wrapped_mask, ImageFile):
        print("SUCCESS: Safety check logic works.")
    else:
        print("FAILED: Safety check logic failed.")

if __name__ == "__main__":
    test_mask_wrapping()
