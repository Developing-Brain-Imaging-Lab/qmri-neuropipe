import string, os, sys, subprocess, shutil, time, json
from glob import glob
import pydicom as dcm
import numpy as np
import nibabel as nib
from scipy.ndimage.filters import gaussian_filter

from core.utils.io import Image
import core.registration.linreg as coreg
import core.registration.apply_transform as transform

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


def compute_afi_b1map(afi, b1map, fwhm=6):

    theta=0.0
    n=0.0
    with open(afi.json, 'r+') as afi_file:
        data  = json.load(afi_file)
        theta = float(data["FlipAngle"])
        n     = float(data["TRRatio"])
    
    afi_img = nib.load(afi.filename)
    tr1_smoothed = gaussian_filter(afi_img.get_fdata()[:,:,:,0], fwhm/2.35)
    tr2_smoothed = gaussian_filter(afi_img.get_fdata()[:,:,:,1], fwhm/2.35)

    r = tr2_smoothed / tr1_smoothed
    r[r>1] = 1.00

    arg = (r*n-1.0)/(n-r)
    arg[arg>1]=1
    arg[arg<0]=1

    b1 = np.degrees(np.arccos(arg))/theta

    b1img = nib.Nifti1Image(b1.astype(np.float32), afi_img.affine, afi_img.header)
    b1img.set_sform(afi_img.get_sform())
    b1img.set_qform(afi_img.get_qform())
    
    nib.save(b1img, b1map.filename)


def coregister_afi(input_afi, ref_img, out_afi, dof='6', cost='normcorr', searchrx='30', searchry='30', searchrz='30'):

    tmp_in  = Image(filename = os.path.join(os.path.dirname(out_afi.filename),"tmp.nii.gz"))
    tmp_mat = os.path.join(os.path.dirname(out_afi.filename),"tmp.mat")

    os.system('fslmaths ' + input_afi.filename + ' -Tmean ' + tmp_in.filename)

    flirt_opts = "-cost " + cost + " -searchrx -"+searchrx+" "+searchrx + " -searchry -"+searchry+" "+searchry + " -searchrz -"+searchrz+" "+searchrz
    coreg.linreg(input                = tmp_in,
                 ref                  = ref_img,
                 out_mat              = tmp_mat,
                 dof                  = dof,
                 method               = "fsl", 
                 flirt_options        = flirt_opts)
    
    merge_cmd="fslmerge -t " + out_afi.filename
    for i in range(0,2):
        
        os.system('fslroi ' + input_afi.filename + " " + tmp_in.filename + " " + str(i) + " 1")
        tmp_out  = Image(filename = os.path.join(os.path.dirname(out_afi.filename),"tmp_"+str(i)+".nii.gz"))

        transform.apply_transform(input     = tmp_in,
                                  ref       = ref_img, 
                                  out       = tmp_out,
                                  transform = tmp_mat,
                                  method    = "fsl")
        
        merge_cmd += " " + tmp_out.filename
    
    os.system(merge_cmd)
        
    os.system('rm -rf ' + tmp_mat)
    os.system('rm -rf ' + tmp_in.filename)
