
import subprocess
from pathlib import Path
import os
import sys

def run_debug():
    in_file = Path("/scratch/isla/qmri-neuropipe/sub-1024457/ses-01/anat/sharpening/sub-1024457_ses-01_acq-CUBE_desc-sharp_T2w.nii.gz")
    ref_file = Path("/scratch/isla/qmri-neuropipe/sub-1024457/ses-01/anat/sub-1024457_ses-01_desc-preproc_T1w.nii.gz")
    out_file = Path("/scratch/isla/qmri-neuropipe/sub-1024457/ses-01/anat/registration/sub-1024457_ses-01_acq-CUBE_desc-coreg_T2w.nii.gz")
    omat = Path("/scratch/isla/qmri-neuropipe/sub-1024457/ses-01/anat/registration/sub-1024457_ses-01_acq-CUBE_desc-coreg_transform.mat")
    wmseg = Path("/scratch/isla/qmri-neuropipe/sub-1024457/ses-01/anat/registration/sub-1024457_ses-01_desc-preproc_T1w.nii_wmseg.nii.gz")
    
    # Ensure parent dir
    out_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Clean up
    if out_file.exists(): out_file.unlink()
    if omat.exists(): omat.unlink()
    
    cmd = [
        "flirt",
        "-in", str(in_file),
        "-ref", str(ref_file),
        "-out", str(out_file),
        "-omat", str(omat),
        "-dof", "6",
        "-cost", "bbr",
        "-searchcost", "corratio",
        "-bins", "256",
        "-interp", "trilinear",
        "-usesqform",
        "-wmseg", str(wmseg)
    ]
    
    full_cmd = " ".join(cmd)
    print(f"Executing: {full_cmd}")
    
    res = subprocess.run(full_cmd, shell=True, capture_output=True, text=True)
    
    print(f"Return Code: {res.returncode}")
    print(f"STDOUT:\n{res.stdout}")
    print(f"STDERR:\n{res.stderr}")
    
    if out_file.exists():
        print(f"SUCCESS: Output file exists: {out_file}")
        print(f"Size: {out_file.stat().st_size}")
    else:
        print(f"FAILURE: Output file missing!")
        if out_file.parent.exists():
             print(f"Parent dir contents: {list(out_file.parent.iterdir())}")
        else:
             print("Parent dir missing!")

if __name__ == "__main__":
    run_debug()
