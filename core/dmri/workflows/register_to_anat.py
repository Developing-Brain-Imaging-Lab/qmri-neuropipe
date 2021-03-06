import os,sys, shutil, json, argparse, copy
import nibabel as nib
import numpy as np

from dipy.io.image import load_nifti, save_nifti
from bids.layout import writing, parse_file_entities

from core.utils.io import Image, DWImage
import core.utils.mask as mask
from core.dmri.utils.qc import rotate_bvecs
import core.registration.registration as reg_tools

def register_to_anat(dwi_image, working_dir, coreg_to_anat = True, T1_image=None, T2_image=None, anat_mask=None, mask_method='hd-bet', reg_method = 'linear', linreg_method='FSL', dof=6, nonlinreg_method='ANTS', use_freesurfer=False, freesurfer_subjs_dir=None, nthreads=1, verbose=False):

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

            dwi_laplacian   = Image(file = working_dir + '/dwi_laplacian.nii.gz')
            b0_laplacian    = Image(file = working_dir + '/b0_laplacian.nii.gz')
            os.system('ImageMath 3 ' + dwi_laplacian._get_filename() + ' Laplacian ' + mean_dwi._get_filename())
            os.system('ImageMath 3 ' + b0_laplacian._get_filename() + ' Laplacian ' + mean_b0._get_filename())

            ref_img               = []
            mov_img               = []
            fsl_transform         = working_dir + '/fsl.mat'
            ants_transform        = working_dir + '/ants_'
            itk_transform         = working_dir + '/itk_0GenericAffine.txt'
            transform             = working_dir + '/ants_nonlinear_composite.nii.gz'

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

            if T1_image != None:
                
                #If Freesurfer, convert to NIFTI
                t1w=T1_image;
            
                ref_img.append(t1w)
                mov_img.append(dwi_masked)
                flirt_options = '-cost normmi '

                T1_laplacian = Image(file = working_dir + '/T1_laplacian.nii.gz')
                os.system('ImageMath 3 ' + T1_laplacian._get_filename() + ' Laplacian ' + T1_image._get_filename())

                ref_img.append(T1_laplacian)
                mov_img.append(dwi_laplacian)

                if T2_image != None:
                    ref_img.append(T2_image)
                    mov_img.append(b0_masked)
                    flirt_options = '-cost normcorr '

                    T2_laplacian = Image(file = working_dir + '/T2_laplacian.nii.gz')
                    os.system('ImageMath 3 ' + T2_laplacian._get_filename() + ' Laplacian ' + T2_image._get_filename())

                    ref_img.append(T2_laplacian)
                    mov_img.append(b0_laplacian)

            else:
                print('No Anatomical Image!')
                exit()

            if reg_method == 'linear':
                transform = itk_transform

                if linreg_method == 'FSL':

                    reg_tools.linear_reg(input_img      = mov_img,
                                         reference_img  = ref_img,
                                         output_matrix  = fsl_transform,
                                         method         = 'FSL',
                                         dof            = dof,
                                         flirt_options =  flirt_options+'-searchrx -180 180 -searchry -180 180 -searchrz -180 180')

                    reg_tools.convert_fsl2ants(mov_img  = mov_img[0],
                                               ref_img  = ref_img[0],
                                               fsl_mat  = fsl_transform,
                                               ants_mat = itk_transform )

                elif linreg_method == 'ANTS':
                    reg_tools.linear_reg(input_img      = mov_img,
                                         reference_img  = ref_img,
                                         output_matrix  = ants_transform,
                                         method         = 'ANTS',
                                         dof            = dof,
                                         ants_options   = '-x '+ anat_mask._get_filename())
                    os.system('ConvertTransformFile 3 ' +  ants_transform+'0GenericAffine.mat ' +  itk_transform)
                    
                elif linreg_method == 'BBR':
                    reg_tools.linear_reg(input_img      = mov_img,
                                         reference_img  = ref_img,
                                         output_matrix  = itk_transform,
                                         method         = 'BBR',
                                         dof            = dof,
                                         freesurfer_subjs_dir = freesurfer_subjs_dir)
                    

            elif reg_method == 'nonlinear':

                reg_tools.nonlinear_reg(input_img       = mov_img,
                                        reference_img   = ref_img,
                                        reference_mask  = anat_mask,
                                        output_base     = ants_transform,
                                        nthreads        = nthreads,
                                        method          = nonlinreg_method,
                                        ants_options    = '-j 1')


                reg_tools.create_composite_transform(reference_img  = ref_img[0],
                                                     output_file    = working_dir + '/ants_nonlinear_composite.nii.gz',
                                                     transforms     = [ants_transform + '1Warp.nii.gz', ants_transform+'0GenericAffine.mat'])

                #Convert the ants transform to ITK
                os.system('ConvertTransformFile 3 ' +  ants_transform+'0GenericAffine.mat ' +  itk_transform)

            reg_tools.apply_transform(input_img     = dwi_image,
                                      reference_img = ref_img[0],
                                      output_file   = coreg_img._get_filename(),
                                      matrix        = transform,
                                      nthreads      = nthreads,
                                      method        = 'ANTS',
                                      ants_options  = '-e 3 -n BSpline[5]' )

            if verbose:
                print('Rotating bvecs')

            rotate_bvecs(input_img      = dwi_image,
                         ref_img        = ref_img[0],
                         output_bvec    = coreg_img._get_bvecs(),
                         transform      = itk_transform,
                         nthreads       = nthreads)


        return coreg_img

    else:
        return dwi_image
