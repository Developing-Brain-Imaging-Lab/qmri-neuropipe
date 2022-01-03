import os,sys, shutil, json, argparse, copy
import nibabel as nib
import numpy as np

from dipy.io.image import load_nifti, save_nifti
from bids.layout import writing, parse_file_entities

from core.utils.io import Image, DWImage
import core.utils.mask as mask
from core.dmri.utils.qc import rotate_bvecs
import core.registration.registration as reg_tools

def register_to_anat(dwi_image, working_dir, coreg_to_anat = True, T1_image=None, T2_image=None, linreg_method='FSL', nthreads=1, verbose=False):

    if coreg_to_anat:

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

        coreg_img = copy.deepcopy(dwi_image)
        coreg_img._set_filename(coreg_file)
        coreg_img._set_bvecs(coreg_bvec)

        if not coreg_img.exists():
            if verbose:
                print('Coregistering DWI to Anatomy Registration-Based Distortion Correction')
                
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
                
            ref_img                     = []
            mov_img                     = []
            rigid_fsl_transform         = working_dir + '/rigid_fsl.mat'
            rigid_itk_transform         = working_dir + '/rigid_0GenericAffine.mat'
            
            mask       = Image(file = working_dir + '/mask.nii/gz')
            dwi_masked = Image(file = working_dir + '/dwi_masked.nii.gz')
            b0_masked  = Image(file = working_dir + '/b0_masked.nii.gz')
            
            mask.mask_image(input_img       = mean_dwi,
                            output_mask     = mask,
                            output_image    = dwi_masked,
                            method          = 'BET')
                
            mask.mask_image(input_img       = mean_b0,
                            output_mask     = mask,
                            output_image    = b0_masked,
                            method          = 'BET')

            if T1_image != None:
                ref_img.append(T1_image)
                mov_img.append(dwi_masked)
                flirt_options = '-cost normmi '
            elif T2_image != None:
                ref_img.append(T2_image)
                mov_img.append(b0_masked)
                flirt_options = '-cost normcorr '
            else:
                print('No Anatomical Image!')
                exit()

            if linreg_method == 'FSL':

                reg_tools.linear_reg(input_img      = mov_img,
                                     reference_img  = ref_img,
                                     output_matrix  = rigid_fsl_transform,
                                     method         = 'FSL',
                                     flirt_options =  flirt_options+'-searchrx -180 180 -searchry -180 180 -searchrz -180 180')

                reg_tools.convert_fsl2ants(mov_img  = mov_img[0],
                                           ref_img  = ref_img[0],
                                           fsl_mat  = rigid_fsl_transform,
                                           ants_mat = rigid_itk_transform )

            elif linreg_method == 'ANTS':

                reg_tools.linear_reg(input_img      = mov_img,
                                     reference_img  = ref_img,
                                     output_matrix  = rigid_itk_transform,
                                     method         = 'ANTS',
                                     ants_options   =  '-j 1')
                                     
            reg_tools.apply_transform(input_img     = dwi_image,
                                      reference_img = ref_img[0],
                                      output_file   = coreg_img._get_filename(),
                                      matrix        = rigid_itk_transform,
                                      nthreads      = nthreads,
                                      method        = 'ANTS',
                                      ants_options  = '-e 3 -n BSpline[5]' )
                                      
            
            rotate_bvecs(input_bvecs    = dwi_image._get_bvecs(),
                         output_bvecs   = coreg_img._get_bvecs(),
                         transform      = rigid_itk_transform,
                         linreg_method  = linreg_method)


        return coreg_img

    else:
        return dwi_image
