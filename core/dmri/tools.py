import string, os, sys, subprocess, shutil, time, copy
from glob import glob

import numpy as np
import nibabel as nib
from dipy.io.image import load_nifti, save_nifti
from core.utils.io import Image, DWImage


def extract_b0s(input_dwi, output_b0, compute_mean=True):
    bvals   = np.loadtxt(input_dwi.bvals)
    ii      = np.where(bvals == 0)
    jj      = np.where(bvals != 0)
    
    dwi_data, affine, dwi_img = load_nifti(input_dwi.filename,
                                           return_img=True)
    
    b0_data = dwi_data[:,:,:,np.asarray(ii).flatten()]
        
    if compute_mean:
        save_nifti(output_b0.filename, np.mean(b0_data, 3), affine, dwi_img.header)
    else:
        save_nifti(output_b0.filename, b0_data, affine, dwi_img.header)

    return output_b0

def extract_dwis(input_dwi, output_dwi, compute_mean=True):
    bvals   = np.loadtxt(input_dwi.bvals)
    ii      = np.where(bvals != 0)
    jj      = np.where(bvals == 0)
    
    dwi_data, affine, dwi_img = load_nifti(input_dwi.filename,
                                           return_img=True)
    
    dwi_data = dwi_data[:,:,:,np.asarray(ii).flatten()]
        
    if compute_mean:
        save_nifti(output_dwi.filename, np.mean(dwi_data, 3), affine, dwi_img.header)
    else:
        save_nifti(output_dwi.filename, dwi_data, affine, dwi_img.header)

    return output_dwi
