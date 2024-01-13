#!/usr/bin/env python
import os, subprocess, copy

import numpy as np
import nibabel as nib

from dipy.denoise.nlmeans import nlmeans
from dipy.denoise.noise_estimate import estimate_sigma
from dipy.io import read_bvals_bvecs
from dipy.core.gradients import gradient_table
from dipy.denoise.localpca import localpca
from dipy.denoise.localpca import mppca
from dipy.denoise.patch2self import patch2self
from dipy.denoise.pca_noise_estimate import pca_noise_estimate

from core.utils.io import Image, DWImage
import core.utils.tools as img_tools
from core.utils.mask import mask_image


def denoise_image(input_img, output_file, method='mrtrix', mask=None, noise_map=None, noise_model="Rician", nthreads=0, debug=False):

    output_img          = copy.deepcopy(input_img)
    output_img.filename = output_file
    output_dir, tmp     = os.path.split(output_img.filename)

    if mask==None:
        mask = Image(os.path.join(output_dir, 'temp_mask.nii.gz'))
        img_tools.calculate_mean_img(input_img, mask.filename)
        mask_image(input_img, mask, algo='bet', bet_options='-f 0.05', nthreads = nthreads)
    
    CMD=""
    if method=="mrtrix":
        CMD="dwidenoise " + input_img.filename + " " + output_img.filename + " -mask " + mask.filename \
           +" -nthreads " + str(nthreads) + " -quiet -force"
        
        if noise_map:
            CMD+=" -noise " + noise_map.filename

        if debug:
            print("Denoising image")
            print(CMD)
        
        print(CMD)
        subprocess.run([CMD], shell=True, stderr=subprocess.STDOUT)

    elif method=="ants":
        CMD="DenoiseImage -d 3 -i " + input_img.filename + " -n " + noise_model

        # if mask:
        #     CMD += " -x " + mask.filename
            
        if noise_map:
            CMD += " -o [" + output_img.filename + "," + noise_map + "]"
        else:
            CMD += " -o " + output_img.filename
            
        subprocess.run([CMD], shell=True, stderr=subprocess.STDOUT)

    elif method[0:4]=='dipy':
        img = nib.load(input_img.filename)
        data = img.get_fdata()
        mask_data = nib.load(mask.filename).get_fdata()
        sigma = 0
        
        if input_img.get_type() == "DWImage":
            sigma = pca_noise_estimate(data, gtab, correct_bias=True, smooth=2)
        else:
            sigma = estimate_sigma(data)

        if method=='dipy-nlmeans':
            denoised_arr = nlmeans(data,sigma=sigma, mask=mask_data, rician=True, patch_radius=2, block_radius=2)
        elif method=='dipy-localpca':
            denoised_arr = localpca(data, sigma, mask=mask_data, tau_factor=2.3, patch_radius=2, pca_method="svd")
        elif method=='dipy-mppca':
            denoised_arr = mppca(data, mask=mask_data, patch_radius=2, pca_method="svd")
        elif method=='dipy-patch2self':
            if input_img.get_type() != "DWImage":
                print("Input needs to be diffusion image to use dipy-patch2self")
                exit(-1)
            bvals, bvecs = read_bvals_bvecs(input_img.bvals, input_img.bvecs)
            denoised_arr = patch2self(data, bvals)

        denoised_img = nib.Nifti1Image(denoised_arr.astype(np.float32), img.affine, img.header)
        denoised_img.set_sform(img.get_sform())
        denoised_img.set_qform(img.get_qform())
        nib.save(denoised_img, output_img.filename)
        
    else:
        print('Invalid Denoising Method')
        exit(-1)

    if os.path.exists(os.path.join(output_dir, 'temp_mask.nii.gz')):
        os.remove(os.path.join(output_dir, 'temp_mask.nii.gz'))

    return output_img



if __name__ == '__main__':
   
   import argparse
   
   parser = argparse.ArgumentParser(description='QMRI-Neuropipe Denoising function')
   
   parser.add_argument('--input',
                       type=str,
                       help="Input image to be denoise",
                       default=None)
   
   parser.add_argument('--output',
                       type=str,
                       help="Denoised output image",
                       default=None)
   
   parser.add_argument('--bvals',
                       type=str,
                       help="B-values of DWI input",
                       default=None)
   
   parser.add_argument('--bvecs',
                       type=str,
                       help="B-bvectors of DWI input",
                       default=None)
   
   parser.add_argument('--mask',
                       type=str,
                       help="Output binary mask",
                       default=None)
   
   parser.add_argument('--noise_map',
                       type=str,
                       help="Output noise map",
                       default=None)
   
   parser.add_argument('--noise_model',
                       type=str,
                       help="Noise model for denoising",
                       choices=["Rician", "Gaussian"],
                       default="Rician")
   
   parser.add_argument('--method',
                       type=str,
                       help="Denoising algorithm",
                       choices=["mrtrix", "ants", "dipy-nlmeans", "dipy-localpca", "dipy-mppca", "dipy-patch2self"],
                       default="bet")
   
   parser.add_argument("--nthreads",
                       type=int,
                       help="Number of threads",
                       default=1)
   
   parser.add_argument("--debug",
                       type=bool,
                       help="Print debugging statements",
                       default=False)
   
   parser.add_argument("--logfile",
                       type=str,
                       help="Log file to print statements",
                       default=None)            
   
   args, unknown = parser.parse_known_args()

   if args.bvals and args.bvecs:
       input_img = DWImage(filename = args.input,
                           bvals    = args.bvals,
                           bvecs    = args.bvecs)
   else:
       input_img = Image(filename = args.input)

   denoise_image(input_img  = input_img,
                 output_file= args.output,
                 method     = args.method,
                 mask       = Image(filename = args.mask),
                 noise_map  = args.noise_map, 
                 noise_model= args.noise_model, 
                 nthreads   = args.nthreads,
                 debug      = args.debug)
   
