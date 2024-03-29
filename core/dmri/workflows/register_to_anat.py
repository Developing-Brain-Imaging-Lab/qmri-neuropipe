import os,sys, shutil, json, argparse, copy
import nibabel as nib
import numpy as np

from dipy.io.image import load_nifti, save_nifti
from bids.layout import writing, parse_file_entities

from core.utils.io import Image, DWImage
import core.utils.mask as mask
import core.utils.biascorrect as bias_tools
from core.dmri.utils.qc import rotate_bvecs
import core.registration.registration as reg_tools
import core.segmentation.segmentation as seg_tools


def register_to_anat(dwi_image, working_dir, anat_image=None, anat_mask=None, mask_method='hd-bet', reg_method = 'linear', linreg_method='FSL', dof=6, nonlinreg_method='ANTS', anat_modality='t1w', use_freesurfer=False, freesurfer_subjs_dir=None, wmseg_reference=None, wmseg=None, nthreads=1, verbose=False, debug=False):

    parsed_filename = parse_file_entities(dwi_image._get_filename())

    entities = {
    'extension': '.nii.gz',
    'subject': parsed_filename.get('subject'),
    'session': parsed_filename.get('session'),
    'suffix':  'dwi',
    'desc': 'CoregisteredToAnatomy'
    }

    working_dir += '/coregister-to-anatomy'

    if not os.path.exists(working_dir):
        os.makedirs(working_dir)

    filename_patterns   = working_dir + '/sub-{subject}[_ses-{session}][_desc-{desc}]_{suffix}{extension}'

    coreg_file = writing.build_path(entities, filename_patterns)
    entities['extension'] = '.bvec'
    coreg_bvec = writing.build_path(entities, filename_patterns)
    entities['extension'] = '.bval'
    coreg_bval = writing.build_path(entities, filename_patterns)

    coreg_img = copy.deepcopy(dwi_image)
    coreg_img._set_filename(coreg_file)
    coreg_img._set_bvecs(coreg_bvec)
    coreg_img._set_bvals(coreg_bval)

    if not coreg_img.exists():
        if verbose:
            print('Coregistering DWI to Anatomy')

        dwi_data, affine, dwi_img = load_nifti(dwi_image._get_filename(), return_img=True)
        bvals    = np.loadtxt(dwi_image._get_bvals())
        ii       = np.where(bvals == 0)
        jj       = np.where(bvals != 0)
        
        mean_b0         = Image(file = working_dir + '/mean_b0.nii.gz')
        mean_b0_data    = np.mean(dwi_data[:,:,:,np.asarray(ii).flatten()], 3)
        save_nifti(mean_b0._get_filename(), mean_b0_data, affine, dwi_img.header)

        mean_dwi        = Image(file = working_dir + '/mean_dwi.nii.gz')
        mean_dwi_data   = np.mean(dwi_data[:,:,:,np.asarray(jj).flatten()], 3)
        save_nifti(mean_dwi._get_filename(), mean_dwi_data, affine, dwi_img.header)

        ref_img               = []
        mov_img               = []
        fsl_transform         = working_dir + '/fsl.mat'
        ants_transform        = working_dir + '/ants_'
        itk_transform         = working_dir + '/itk_0GenericAffine.txt'
        nonlin_transform      = working_dir + '/nonlinear_composite.nii.gz'
        final_transform       = ''

        mask_img   = Image(file = working_dir + '/mask.nii.gz')
        dwi_masked = Image(file = working_dir + '/dwi_masked.nii.gz')
        b0_masked  = Image(file = working_dir + '/b0_masked.nii.gz')

        mask.mask_image(input_img       = mean_dwi,
                        output_mask     = mask_img,
                        output_img      = dwi_masked,
                        method          = mask_method,
                        bet_options     = '-f 0.25')

        mask.apply_mask(input_img       = mean_b0,
                        mask_img        = mask_img,
                        output_img      = b0_masked)
                        
        #If structural T2w available, use it with the b=0
        if anat_modality == 't1w':
            mov_img.append(dwi_masked)
        elif anat_modality == 't2w':
            mov_img.append(b0_masked)
        else:
            print('Invalid anatomy contrast')
            exit()
            
    
        #Mask the Anatomical image and bias-correct
        anat_masked = Image(file = working_dir + '/anat_masked.nii.gz')
        
        if not anat_mask:
            anat_mask = Image(file = working_dir + '/anat_mask.nii.gz')
            mask.mask_image(input_img       = anat_image,
                            output_mask     = anat_mask,
                            method          = mask_method)
        
        mask.apply_mask(input_img       = anat_image,
                        mask_img        = anat_mask,
                        output_img      = anat_masked)
                
        
        anat_biascorr = Image(file = working_dir + '/anat_biascorr.nii.gz')
        bias_tools.biasfield_correction(input_img = anat_masked,
                                        output_file = anat_biascorr._get_filename(),
                                        iterations=3)
    
        ref_img.append(anat_biascorr)
    
            
        #First, perform linear registration using FSL flirt
        tmp_coreg_img = Image(file = working_dir + '/dwi_coreg.nii.gz')
        reg_tools.linear_reg(input_img      = mov_img,
                             reference_img  = ref_img,
                             output_matrix  = fsl_transform,
                             output_file    = tmp_coreg_img._get_filename(),
                             method         = 'FSL',
                             dof            = dof,
                             flirt_options =  '-searchrx -180 180 -searchry -180 180 -searchrz -180 180 -interp sinc')

        if reg_method == 'bbr':
            #Create WM segmentation from structural image
            wmseg_img = seg_tools.create_wmseg(input_img        = ref_img[0],
                                               brain_mask       = anat_mask,
                                               output_dir       = working_dir + '/atropos/',
                                               modality         = anat_modality )
                
           
            #Next, re-run flirt, using bbr cost function and WM segmentation
            bbr_options = ' -cost bbr -wmseg ' + wmseg_img._get_filename() + ' -schedule $FSLDIR/etc/flirtsch/bbr.sch -interp sinc -bbrtype global_abs -bbrslope 0.25 -finesearch 18 -init ' + fsl_transform

            
            reg_tools.linear_reg(input_img      = mov_img,
                                 reference_img  = ref_img,
                                 output_matrix  = fsl_transform,
                                 output_file    = tmp_coreg_img._get_filename(),
                                 method         = 'FSL',
                                 dof            = dof,
                                 flirt_options =  bbr_options)
                                
                                
        #Convert to ITK format for warping
        reg_tools.convert_fsl2ants(mov_img  = mov_img[0],
                                   ref_img  = ref_img[0],
                                   fsl_mat  = fsl_transform,
                                   ants_mat = itk_transform )
                                   
        if reg_method == 'linear' or reg_method == 'bbr':
            final_transform = itk_transform
            

        elif reg_method == 'nonlinear':
        
            mov_img[0] = tmp_coreg_img

            reg_tools.nonlinear_reg(input_img       = mov_img,
                                    reference_img   = ref_img,
                                    reference_mask  = anat_mask,
                                    output_base     = ants_transform,
                                    nthreads        = nthreads,
                                    method          = nonlinreg_method,
                                    ants_options    = '-j 1')

        
            #Create the final transform
            reg_tools.create_composite_transform(reference_img  = ref_img[0],
                                                 output_file    = nonlin_transform,
                                                 transforms     = [ants_transform + '1Warp.nii.gz', ants_transform+'0GenericAffine.mat', itk_transform])

            final_transform = nonlin_transform
        
        
        #Apply the transformation
        reg_tools.apply_transform(input_img     = dwi_image,
                                  reference_img = ref_img[0],
                                  output_img    = coreg_img,
                                  matrix        = final_transform,
                                  nthreads      = nthreads,
                                  method        = 'MRTRIX',
                                  ants_options  = '-e 3 -n BSpline[5]' )



    return coreg_img
