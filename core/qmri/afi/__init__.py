import string, os, sys, subprocess, shutil, time, json
from glob import glob
import pydicom as dcm
import numpy as np
import nibabel as nib
from scipy.ndimage.filters import gaussian_filter

def compute_afi_b1map(input_img1, input_img2, output_img, theta=55, n=5, fwhm=6):

    #Smooth the images
    img1 = nib.load(input_img1)
    img2 = nib.load(input_img2)

    #First, smooth the data
    img1_smoothed = gaussian_filter(img1.get_fdata(), fwhm/2.35)
    img2_smoothed = gaussian_filter(img2.get_fdata(), fwhm/2.35)

    r = img2_smoothed / img1_smoothed
    r[r>1] = 1.00

    n=float(n)
    theta=float(theta)

    arg = (r*n-1.0)/(n-r)
    arg[arg>1]=1
    arg[arg<0]=1;

    b1 = np.degrees(np.arccos(arg))/theta

    b1map = nib.Nifti1Image(b1.astype(np.float32), img1.get_affine(), img1.header)
    b1map.set_sform(img1.get_sform())
    b1map.set_qform(img1.get_qform())
    nib.save(b1map , output_img)


def compute_afi_b1map(input_afi, input_json, output_img, n=5, fwhm=6):
    afi_img = nib.load(input_afi)
    img1_smoothed = gaussian_filter(afi_img.get_fdata()[:,:,:,0], fwhm/2.35)
    img2_smoothed = gaussian_filter(afi_img.get_fdata()[:,:,:,1], fwhm/2.35)

    r = img2_smoothed / img1_smoothed
    r[r>1] = 1.00

    n = float(n)

    with open(input_json, 'r+') as afi_file:
        data = json.load(afi_file)
        theta = float(data["FlipAngle"])


    arg = (r*n-1.0)/(n-r)
    arg[arg>1]=1
    arg[arg<0]=1;

    b1 = np.degrees(np.arccos(arg))/theta

    b1map = nib.Nifti1Image(b1.astype(np.float32), afi_img.get_affine(), afi_img.header)
    b1map.set_sform(afi_img.get_sform())
    b1map.set_qform(afi_img.get_qform())
    nib.save(b1map, output_img)


def register_afi_flirt(input_afi, input_b1, ref_img, output_b1, dof='6', cost='normcorr', searchrx='30', searchry='30', searchrz='30'):

    tmp_mat = os.path.split(output_b1._get_filename())[0] + '/tmp.mat'
    tmp_img = os.path.split(output_b1._get_filename())[0] + '/tmp.nii.gz'

    os.system('fslmaths ' + input_afi._get_filename() + ' -Tmean ' + tmp_img)

    flirt_cmd = 'flirt -in ' + tmp_img + ' -ref ' + ref_img._get_filename() + ' -omat ' + tmp_mat + ' -dof 6 -searchrx -'+searchrx+' ' +searchrx + ' -searchry -'+searchry+ ' '+searchry + ' -searchrz -'+searchrz+' '+ searchrz
    os.system(flirt_cmd)

    os.system('flirt -in ' + input_b1._get_filename() + ' -ref ' + ref_img._get_filename() + ' -out ' + output_b1._get_filename() + ' -applyxfm -init ' + tmp_mat)
    os.system('rm -rf ' + tmp_mat)
    os.system('rm -rf ' + tmp_img)
