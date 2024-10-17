import os, copy
import nibabel as nib
import numpy as np

from dipy.io.image import load_nifti, save_nifti
from bids.layout import writing, parse_file_entities

from core.utils.io import Image, DWImage
import core.utils.mask as mask
import core.utils.biascorrect as bias_tools
from core.utils.denoise import denoise_image
from core.registration.linreg import linreg
from core.registration.nonlinreg import nonlinreg
from core.registration.apply_transform import apply_transform
from core.registration.convert_fsl2ants import convert_fsl2ants
from core.registration.create_composite_transform import create_composite_transform

import core.segmentation.segmentation as seg_tools


def register_to_anat(dwi_image, anat_image, working_dir, anat_mask=None, noresample=False, mask_method='hd-bet', reg_method = 'linear', linreg_method='fsl', dof=6, anat_modality='t1w', freesurfer_subjs_dir=None, nthreads=1, verbose=False):

    parsed_filename = parse_file_entities(dwi_image.filename)

    entities = {
    'extension': '.nii.gz',
    'subject': parsed_filename.get('subject'),
    'session': parsed_filename.get('session'),
    'suffix':  'dwi',
    'desc': 'CoregisteredToAnatomy'
    }

    id_patterns          = 'sub-{subject}[_ses-{session}]'
    id                   = writing.build_path(entities, id_patterns)

    working_dir = os.path.join(working_dir, "coregister-to-anatomy",)

    if not os.path.exists(working_dir):
        os.makedirs(working_dir)

    filename_patterns   = working_dir + '/sub-{subject}[_ses-{session}][_desc-{desc}]_{suffix}{extension}'

    coreg_file = writing.build_path(entities, filename_patterns)
    entities['extension'] = '.bvec'
    coreg_bvec = writing.build_path(entities, filename_patterns)
    entities['extension'] = '.bval'
    coreg_bval = writing.build_path(entities, filename_patterns)

    coreg_img = copy.deepcopy(dwi_image)
    coreg_img.filename = coreg_file
    coreg_img.bvecs = coreg_bvec
    coreg_img.bvals = coreg_bval

    if not coreg_img.exists():
        if verbose:
            print('Coregistering DWI to Anatomy')

        dwi_data, affine, dwi_img = load_nifti(dwi_image.filename, return_img=True)
        bvals    = np.loadtxt(dwi_image.bvals)
        ii       = np.where(bvals == 0)
        jj       = np.where(bvals != 0)
        
        mean_b0         = Image(filename = working_dir + '/mean_b0.nii.gz')
        mean_b0_data    = np.mean(dwi_data[:,:,:,np.asarray(ii).flatten()], 3)
        save_nifti(mean_b0.filename, mean_b0_data, affine, dwi_img.header)

        mean_dwi        = Image(filename = working_dir + '/mean_dwi.nii.gz')
        mean_dwi_data   = np.mean(dwi_data[:,:,:,np.asarray(jj).flatten()], 3)
        save_nifti(mean_dwi.filename, mean_dwi_data, affine, dwi_img.header)

        if linreg_method == "freesurfer-bbr":
            if verbose:
                print('Coregistering DWI to Anatomy using FreeSurfer BBR')
            
            # Setup BBR registration paths and variables
            bbrdwi2T1_dir = os.path.join(working_dir, 'bbrdwi2T1')
            os.makedirs(bbrdwi2T1_dir, exist_ok=True)

            b0toT1flirtmtx = os.path.join(bbrdwi2T1_dir, 'b0toT1flirt.mtx')
            fsl2antsAffine = os.path.join(bbrdwi2T1_dir, 'b0toT1flirtmtx_fsl2antsAffine.txt')
            b0toT1flirtmtx_mrtrixformat = os.path.join(bbrdwi2T1_dir, 'b0toT1flirtmtx_mrtrixformat.txt')
            nuT1 = os.path.join(freesurfer_subjs_dir, id, 'mri', 'orig_nu.mgz')
            T1nii = os.path.join(bbrdwi2T1_dir, 'orig_nu.nii.gz')

            # Convert T1 to NIfTI
            mri_convert_cmd = f"mri_convert --in_type mgz --out_type nii {nuT1} {T1nii}"
            os.system(mri_convert_cmd)
 
            linreg(input                = mean_b0, 
                   ref                  = T1nii,
                   out_mat              = b0toT1flirtmtx,
                   out                  = None,
                   method               = "bbr", 
                   freesurfer_subjs_dir = freesurfer_subjs_dir, 
                   debug                = verbose)

            convert_fsl2ants(mean_b0, T1nii, b0toT1flirtmtx, fsl2antsAffine)
            os.systemI(f"transformconvert {b0toT1flirtmtx} {mean_b0.filename} {T1nii} flirt_import {b0toT1flirtmtx_mrtrixformat} -force")

            if noresample:
                final_transform = b0toT1flirtmtx_mrtrixformat
            else:
                final_transform = fsl2antsAffine
            
        else:
            ref_img               = []
            mov_img               = []
            fsl_transform         = os.path.join(working_dir, "fsl.mat")
            ants_transform        = os.path.join(working_dir, "ants_")
            itk_transform         = os.path.join(working_dir, "itk_0GenericAffine.txt")
            nonlin_transform      = os.path.join(working_dir, "nonlinear_composite.nii.gz")
            final_transform       = ''

            mask_img   = Image(filename = os.path.join(working_dir, "mask.nii.gz"))
            dwi_masked = Image(filename = os.path.join(working_dir, "dwi_masked.nii.gz"))
            b0_masked  = Image(filename = os.path.join(working_dir, "b0_masked.nii.gz"))

            mask.mask_image(input       = mean_dwi,
                            mask        = mask_img,
                            mask_img    = dwi_masked,
                            algo        = mask_method,
                            bet_options = '-f 0.25')

            mask.apply_mask(input       = mean_b0,
                            mask        = mask_img,
                            output      = b0_masked)
                            
            #If structural T2w available, use it with the b=0
            if anat_modality == 't1w':
                mov_img.append(mean_dwi)
            elif anat_modality == 't2w':
                mov_img.append(mean_b0)
            else:
                print('Invalid anatomy contrast')
                exit()
    
            #Mask the Anatomical image and bias-correct
            anat_masked = Image(filename = os.pathj.join(working_dir, "anat_masked.nii.gz"))
            if not anat_mask:
                anat_mask = Image(file = os.path.join(working_dir, "anat_mask.nii.gz"))
                mask.mask_image(input   = anat_image,
                                mask    = anat_mask,
                                algo    = mask_method)
            
            mask.apply_mask(input       = anat_image,
                            mask        = anat_mask,
                            output      = anat_masked)
                    
        
            ref_img.append(anat_masked)
        
            #First, perform linear registration using FSL flirt
            tmp_coreg_img = Image(filename = os.path.join(working_dir, "dwi_coreg.nii.gz"))
            linreg(input      = mov_img,
                ref           = ref_img,
                out_mat       = fsl_transform,
                out           = [tmp_coreg_img],
                method        = 'fsl',
                dof           = dof,
                flirt_options =  '-searchrx -180 180 -searchry -180 180 -searchrz -180 180')

            if reg_method == 'bbr-fsl':
                #Create WM segmentation from structural image
                wmseg_img = seg_tools.create_wmseg(input_img     = ref_img[0],
                                                output_dir       = working_dir + '/wmseg/',
                                                nthreads         = nthreads )
                    
            
                #Next, re-run flirt, using bbr cost function and WM segmentation
                bbr_options = ' -cost bbr -wmseg ' + wmseg_img.filename \
                            + ' -schedule $FSLDIR/etc/flirtsch/bbr.sch -interp sinc -bbrtype global_abs -bbrslope 0.25 -finesearch 18 -init ' \
                            + fsl_transform

                linreg(input      = mov_img,
                    ref           = ref_img,
                    out_mat       = fsl_transform,
                    out           = [tmp_coreg_img],
                    method        = 'fsl',
                    dof           = dof,
                    flirt_options = bbr_options)
                                    
            #Convert to ITK format for warping
            convert_fsl2ants(input    = mov_img[0],
                             ref      = ref_img[0],
                             fsl_mat  = fsl_transform,
                             ants_mat = itk_transform )
                                    
            if reg_method == 'linear' or reg_method == 'fsl-bbr':
                final_transform = itk_transform
                
            elif reg_method == 'nonlinear':
                mov_img[0] = tmp_coreg_img
                nonlinreg(input           = mov_img,
                          ref             = ref_img,
                          mask            = anat_mask,
                          out_xfm_base    = ants_transform,
                          nthreads        = nthreads,
                          method          = 'ants',
                          ants_options    = '-j 1')

                #Create the final transform
                create_composite_transform(ref        = ref_img[0],
                                           out        = nonlin_transform,
                                           transforms = [ants_transform + "1Warp.nii.gz", ants_transform+"0GenericAffine.mat", itk_transform])

                final_transform = nonlin_transform

        
        #Apply the transformation
        apply_transform(input         = dwi_image,
                        ref           = ref_img[0],
                        out           = coreg_img,
                        transform     = final_transform,
                        noresample    = noresample,
                        nthreads      = nthreads,
                        method        = 'mrtrix',
                        ants_options  = '-e 3 -n BSpline[5]' )



    return coreg_img
