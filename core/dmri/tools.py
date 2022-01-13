import string, os, sys, subprocess, shutil, time, copy
from glob import glob

import numpy as np
import nibabel as nib
import nibabel.processing as nib_proc
import ants

from core.utils.io import Image, DWImage


def extract_b0s(input_dwi, output_b0, compute_mean=True):
    bvals   = np.loadtxt(input_dwi._get_bvals())
    ii      = np.where(bvals == 0)
    jj      = np.where(bvals != 0)
    
    dwi_data, affine, dwi_img = load_nifti(input_dwi._get_filename(),                                                  return_img=True)
    
    output_b0 = Image(file = output_b0)
    b0_data = dwi_data[:,:,:,np.asarray(ii).flatten()]
        
    if compute_mean:
        save_nifti(output_b0._get_filename(), np.mean(b0_data, 3), affine, dwi_img.header)
    else:
        save_nifti(output_b0._get_filename(), b0_data, affine, dwi_img.header)

    return output_b0

