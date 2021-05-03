import string, os, sys, subprocess, shutil, time, copy
from glob import glob

import numpy as np
import nibabel as nib
import nibabel.processing as nib_proc

from dipy.denoise.nlmeans import nlmeans
from dipy.denoise.noise_estimate import estimate_sigma
from dipy.io import read_bvals_bvecs
from dipy.io.bvectxt import reorient_vectors
from dipy.core.gradients import gradient_table
from dipy.denoise.localpca import localpca
from dipy.denoise.localpca import mppca
from dipy.denoise.patch2self import patch2self
from dipy.denoise.pca_noise_estimate import pca_noise_estimate
from dipy.denoise.gibbs import gibbs_removal

from core.utils.io import Image, DWImage
import core.utils.tools as img_tools
from core.utils.mask import mask_image


def gibbs_ringing_correction(input_img, output_file, method='mrtrix', nthreads=0):

    output_img = copy.deepcopy(input_img)
    output_img._set_filename(output_file)

    if method=='mrtrix':
        subprocess.run(['mrdegibbs',
                        input_img._get_filename(),
                        output_img._get_filename(),
                        '-nthreads', str(nthreads),
                        '-quiet',
                        '-force'], stderr=subprocess.STDOUT)

    if method=='dipy':
        img = nib.load(input_img._get_filename())
        data = img.get_data()
        data_corrected = gibbs_removal(data, num_threads=nthreads)

        corrected_img = nib.Nifti1Image(data_corrected.astype(np.float32), img.get_affine(), img.header)
        corrected_img.set_sform(img.get_sform())
        corrected_img.set_qform(img.get_qform())
        nib.save(corrected_img, output_img._get_filename())

    return output_img

def denoise_image(input_img, output_file, method='mrtrix', mask_img=None, output_noise=None, nthreads=0):

    output_img = copy.deepcopy(input_img)
    output_img._set_filename(output_file)

    if mask_img==None:
        output_root, tmp    = os.path.split(output_file)
        mask_img            = Image(output_root + '/mask.nii.gz')
        mask_image(input_img, mask_img, method='bet', nthreads = nthreads)
    else:
        output_root, tmp = os.path.split(mask_img._get_filename())

    if method=='mrtrix':
        if output_noise:
            subprocess.run(['dwidenoise',
                            input_img._get_filename(),
                            output_img._get_filename(),
                            '-mask', mask_img._get_filename(),
                            '-noise', output_noise,
                            '-nthreads', str(nthreads),
                            '-quiet',
                            '-force'], stderr=subprocess.STDOUT)
        else:
            subprocess.run(['dwidenoise',
                            input_img._get_filename(),
                            output_img._get_filename(),
                            '-mask', mask_img._get_filename(),
                            '-nthreads', str(nthreads),
                            '-quiet'
                            '-force'], stderr=subprocess.STDOUT)

    elif method[0:4]=='dipy':
        img = nib.load(input_img._get_filename())
        data = img.get_data()
        mask = nib.load(mask_img._get_filename()).get_data()

        if method=='dipy-nlmeans':
            sigma = estimate_sigma(data)
            denoised_arr = nlmeans(data,sigma=sigma, mask=mask, rician=True, patch_radius=1, block_radius=1)
        elif method=='dipy-localpca':
            bvals, bvecs = read_bvals_bvecs(input_img._get_bvals(), input_img._get_bvecs())
            gtab = gradient_table(bvals, bvecs)
            sigma = pca_noise_estimate(data, gtab, correct_bias=True, smooth=3)
            denoised_arr = localpca(data, sigma, mask=mask, tau_factor=2.3, patch_radius=2)
        elif method=='dipy-mppca':
            denoised_arr = mppca(data, mask=mask, patch_radius=2)
        elif method=='dipy-patch2self':
            bvals, bvecs = read_bvals_bvecs(input_img._get_bvals(), input_img._get_bvecs())
            denoised_arr = patch2self(data, bvals)

        denoised_img = nib.Nifti1Image(denoised_arr.astype(np.float32), img.get_affine(), img.header)
        denoised_img.set_sform(img.get_sform())
        denoised_img.set_qform(img.get_qform())
        nib.save(denoised_img, output_img._get_filename())

    else:
        print('Invalid Denoising Method')
        exit()

    if os.path.exists(output_root + '/mask.nii.gz'):
        os.remove(output_root + '/mask.nii.gz')

    return output_img
