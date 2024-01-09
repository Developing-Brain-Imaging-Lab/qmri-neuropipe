import os, json, copy, glob

import numpy as np
import nibabel as nib

from bids.layout import writing, parse_file_entities
from dipy.nn.synb0 import Synb0

from core.utils.io import Image, DWImage
import core.utils.tools as img_tools
import core.utils.mask as mask_tools
import core.dmri.tools as dmri_tools
from core.registration.linreg import linreg
from core.registration.nonlinreg import nonlinreg
from core.registration.convert_fsl2ants import convert_fsl2ants
from core.registration.apply_transform import apply_transform
from core.registration.create_composite_transform import create_composite_transform

from core.segmentation.segmentation import create_wmseg


def topup_fsl(input_dwi, output_topup_base, config_file=None, field_output=False, verbose=False):

    #First, find the indices of the B0 images
    dwi_img   = nib.load(input_dwi.filename)
    aff       = dwi_img.affine
    dwi_data  = dwi_img.get_fdata()

    bvals     = np.loadtxt(input_dwi.bvals)
    index     = np.loadtxt(input_dwi.index)
    acqparams = np.loadtxt(input_dwi.acqparams)
    ii        = np.where(bvals == 0)

    b0_data      = dwi_data[:,:,:,np.asarray(ii).flatten()]
    b0_indices   = index[ii].astype(int)
    b0_acqparams = acqparams[b0_indices-1]

    indices,jj   = np.unique(b0_indices, return_index=True)
    topup_data   = np.zeros([b0_data.shape[0], b0_data.shape[1], b0_data.shape[2], len(indices)])

    for i in range(0,len(indices)):
        tmp_indices         = np.where(b0_indices == indices[i])
        tmp_data            = b0_data[:,:,:,np.asarray(tmp_indices).flatten()]
        topup_data[:,:,:,i] = np.mean(tmp_data, axis=3)

    topup_acqparams = b0_acqparams[jj]


    output_dir = os.path.dirname(output_topup_base)
    tmp_acqparams = output_dir + '/tmp.acqparams.txt'
    tmp_b0 = output_dir + '/tmp.B0.nii.gz'

    topup_imgs = nib.Nifti1Image(topup_data, aff, dwi_img.header)
    nib.save(topup_imgs, tmp_b0)
    np.savetxt(tmp_acqparams, topup_acqparams, fmt='%.8f')

    topup_command = 'topup --imain='+ tmp_b0 \
                  + ' --datain=' + tmp_acqparams \
                  + ' --out=' + output_topup_base

    if config_file != None:
        topup_command += ' --config='+config_file
    if field_output:
        topup_command += ' --fout='+output_topup_base+'_fmap.nii.gz'

    if verbose:
        print(topup_command)


    os.system(topup_command)
    os.remove(output_dir + '/tmp.acqparams.txt')
    os.remove(output_dir + '/tmp.B0.nii.gz')

def registration_method(input_dwi, working_dir, T1_image=None, T2_image=None, linreg_method='FSL', distortion_modality='t1w', resample_to_anat=False, nthreads=1, verbose=False):

    if not os.path.exists(working_dir):
        os.makedirs(working_dir)

    parsed_filename = parse_file_entities(input_dwi.filename)

    entities = {
    'extension': '.nii.gz',
    'subject': parsed_filename.get('subject'),
    'session': parsed_filename.get('session'),
    'suffix':  'dwi',
    'desc': 'DistortionCorrected'
    }

    filename_patterns  = working_dir + '/sub-{subject}[_ses-{session}][_desc-{desc}]_{suffix}{extension}'

    out_file = writing.build_path(entities, filename_patterns)
    entities['extension'] = '.bvec'
    out_bvec = writing.build_path(entities, filename_patterns)

    output_img = copy.deepcopy(input_dwi)
    output_img.filename = out_file
    output_img.bvecs    = out_bvec

    # if not output_img.exists():

    #     dwi_data, affine, dwi_img = load_nifti(input_dwi.filename, return_img=True)

    #     bvals   = np.loadtxt(input_dwi._get_bvals())
    #     ii      = np.where(bvals == 0)
    #     jj      = np.where(bvals != 0)

    #     mean_b0         = Image(file = working_dir + '/mean_b0.nii.gz')
    #     mean_b0_data    = np.mean(dwi_data[:,:,:,np.asarray(ii).flatten()], 3)
    #     save_nifti(mean_b0.filename, mean_b0_data, affine, dwi_img.header)

    #     mean_dwi        = Image(file = working_dir + '/mean_dwi.nii.gz')
    #     mean_dwi_data   = np.mean(dwi_data[:,:,:,np.asarray(jj).flatten()], 3)
    #     save_nifti(mean_dwi.filename, mean_dwi_data, affine, dwi_img.header)

    #     #Now bias correct the mean B0 and DWI
    #     mean_b0     = img_tools.biasfield_correction(input_img   = mean_b0,
    #                                                  output_file = mean_b0.filename)

    #     mean_dwi    = img_tools.biasfield_correction(input_img   = mean_dwi,
    #                                                  output_file = mean_dwi.filename)

    #     #Determine the Phase Encode Direction
    #     #Read the JSON file and get the
    #     with open(input_dwi._get_json()) as f:
    #         json_data = json.load(f)

    #     dwi_strides = subprocess.check_output(['mrinfo', '-strides',mean_dwi.filename]).decode('utf-8').strip().split(' ')
    #     dwi_strides = [abs(int(i)) for i in dwi_strides]

    #     ants_phase_encode_dir = ''
    #     pe_index = ''
    #     if json_data["PhaseEncodingDirection"] == 'i'  or json_data["PhaseEncodingDirection"] == 'i-':
    #         pe_index = dwi_strides.index(1)
    #     elif json_data["PhaseEncodingDirection"] == 'j'  or json_data["PhaseEncodingDirection"] == 'j-':
    #         pe_index = dwi_strides.index(2)
    #     elif json_data["PhaseEncodingDirection"] == 'k'  or json_data["PhaseEncodingDirection"] == 'k-':
    #         pe_index = dwi_strides.index(3)
    #     else:
    #         print('Incorrect Phase Encoding Direction - please check JSON file')
    #         exit()

    #     if pe_index == 0:
    #         ants_phase_encode_dir='1x0x0'
    #     elif pe_index == 1:
    #         ants_phase_encode_dir='0x1x0'
    #     elif pe_index == 2:
    #         ants_phase_encode_dir='0x0x1'
    #     else:
    #         print('Incorrect Phase Encoding Direction')
    #         exit()

    #     ref_img                     = []
    #     mov_img                     = []
    #     nonlin_mov_img              = []
    #     nonlin_ref_img              = []
    #     rigid_fsl_transform         = working_dir + '/rigid_fsl.mat'
    #     rigid_ants_transform        = working_dir + '/rigid_'
    #     rigid_itk_transform         = working_dir + '/rigid_0GenericAffine.txt'
    #     ants_nonlinear_cmd          = 'antsRegistration -d 3 -o ' + working_dir + '/tmp_ants_nonlinear_'
    #     ants_composite_transform    = working_dir + '/ants_composite.nii.gz'

    #     #Decide if data is going to be resampled to Anatomical Space
    #     if resample_to_anat:
    #         dwi_aligned = Image(file = working_dir + '/dwi_aligned.nii.gz')
    #         b0_aligned  = Image(file = working_dir + '/b0_aligned.nii.gz')

    #         if linreg_method == 'FSL':
    #             flirt_options=''

    #             if distortion_modality == 't1w' and T1_image != None:
    #                 ref_img.append(T1_image)
    #                 mov_img.append(mean_dwi)
    #                 nonlin_mov_img.append(dwi_aligned)
    #                 nonlin_ref_img.append(T1_image)
    #                 out_img = dwi_aligned
    #                 flirt_options = '-cost normmi '

    #             elif distortion_modality == 't2w' and T2_image != None:
    #                 ref_img.append(T2_image)
    #                 mov_img.append(mean_b0)
    #                 nonlin_mov_img.append(b0_aligned)
    #                 nonlin_ref_img.append(T2_image)
    #                 out_img = b0_aligned
    #                 flirt_options = '-cost normcorr '
    #             else:
    #                 print('No Structural Image provided!')
    #                 exit()


    #             linreg(input          = mov_img,
    #                    ref            = ref_img,
    #                    out            = output_img.filename,
    #                    out_mat        = rigid_fsl_transform,
    #                    method         = 'fsl',
    #                    dof            = 6,
    #                    flirt_options  =  flirt_options+'-searchrx -180 180 -searchry -180 180 -searchrz -180 180')

    #             apply_transform(input_img     = mean_dwi,
    #                                       reference_img = ref_img[0],
    #                                       output_img    = dwi_aligned,
    #                                       matrix        = rigid_fsl_transform,
    #                                       method        = 'FSL',
    #                                       flirt_options = '-interp sinc')

    #            apply_transform(input_img     = mean_b0,
    #                                       reference_img = ref_img[0],
    #                                       output_img    = b0_aligned,
    #                                       matrix        = rigid_fsl_transform,
    #                                       method        = 'FSL',
    #                                       flirt_options = '-interp sinc')

    #             convert_fsl2ants(mov_img  = mov_img[0],
    #                                        ref_img  = ref_img[0],
    #                                        fsl_mat  = rigid_fsl_transform,
    #                                        ants_mat = rigid_itk_transform )

    #         elif linreg_method == 'ANTS':

    #             if T1_image != None:
    #                 ref_img.append(T1_image)
    #                 mov_img.append(mean_dwi)
    #                 nonlin_mov_img.append(dwi_aligned)
    #                 nonlin_ref_img.append(T1_image)

    #                 #Also use the Laplacian
    #                 dwi_laplacian = Image(file = working_dir + '/dwi_laplacian.nii.gz')
    #                 t1_laplacian  = Image(file = working_dir + '/t1_laplacian.nii.gz')
    #                 os.system('ImageMath 3 ' + dwi_laplacian.filename + ' Laplacian ' + mean_dwi.filename)
    #                 os.system('ImageMath 3 ' + t1_laplacian.filename  + ' Laplacian ' + T1_image.filename)

    #                 ref_img.append(t1_laplacian)
    #                 mov_img.append(dwi_laplacian)
    #                 nonlin_mov_img.append(dwi_laplacian)
    #                 nonlin_ref_img.append(t1_laplacian)


    #             if T2_image != None:
    #                 ref_img.append(T2_image)
    #                 mov_img.append(mean_b0)
    #                 nonlin_mov_img.append(b0_aligned)
    #                 nonlin_ref_img.append(T2_image)

    #                 b0_laplacian = Image(file = working_dir + '/b0_laplacian.nii.gz')
    #                 t2_laplacian  = Image(file = working_dir + '/t2_laplacian.nii.gz')
    #                 os.system('ImageMath 3 ' + b0_laplacian.filename + ' Laplacian ' + mean_b0.filename)
    #                 os.system('ImageMath 3 ' + t2_laplacian.filename  + ' Laplacian ' + T2_image.filename)

    #                 ref_img.append(t2_laplacian)
    #                 mov_img.append(b0_laplacian)
    #                 nonlin_mov_img.append(b0_laplacian)
    #                 nonlin_ref_img.append(t2_laplacian)

    #             reg_tools.linear_reg(input_img      = mov_img,
    #                                  reference_img  = ref_img,
    #                                  output_file    = output_img.filename,
    #                                  output_matrix  = rigid_ants_transform,
    #                                  method         = 'ANTS',
    #                                  nthreads       = nthreads,
    #                                  dof            = 6,
    #                                  ants_options   =  '-j 1')
    #             #Convert the ants transform to ITK
    #             os.system('ConvertTransformFile 3 ' +  rigid_ants_transform+'0GenericAffine.mat ' +  rigid_itk_transform)

    #             reg_tools.apply_transform(input_img     = mean_dwi,
    #                                       reference_img = ref_img[0],
    #                                       output_img    = dwi_aligned,
    #                                       matrix        = rigid_itk_transform,
    #                                       method        = 'ANTS')

    #             reg_tools.apply_transform(input_img     = mean_b0,
    #                                       reference_img = ref_img[0],
    #                                       output_img    = b0_aligned,
    #                                       matrix        = rigid_itk_transform,
    #                                       method        = 'ANTS')

    #     else:
    #         t1_aligned = Image(file = working_dir + '/t1w_aligned.nii.gz')
    #         t2_aligned = Image(file = working_dir + '/t2w_aligned.nii.gz')

    #         if linreg_method == 'FSL':

    #             flirt_options = ''

    #             if distortion_modality == 't1w' and T1_image != None:
    #                 ref_img.append(mean_dwi)
    #                 mov_img.append(T1_image)
    #                 nonlin_mov_img.append(mean_dwi)
    #                 nonlin_ref_img.append(t1_aligned)
    #                 out_img = t1_aligned
    #                 flirt_options = '-cost normmi '

    #             elif distortion_modality == 't2w' and T2_image != None:
    #                 ref_img.append(mean_b0)
    #                 mov_img.append(T2_image)
    #                 nonlin_mov_img.append(mean_b0)
    #                 nonlin_ref_img.append(t2_aligned)
    #                 out_img = t2_aligned
    #                 flirt_options = '-cost normcorr '
    #             else:
    #                 print('No Structural Image provided!')
    #                 exit()

    #             reg_tools.linear_reg(input_img      = mov_img,
    #                                  reference_img  = ref_img,
    #                                  output_file    = output_img.filename,
    #                                  output_matrix  = rigid_fsl_transform,
    #                                  method         = 'FSL',
    #                                  dof            = 6,
    #                                  flirt_options  =  flirt_options+'-searchrx -180 180 -searchry -180 180 -searchrz -180 180')

    #             reg_tools.convert_fsl2ants(mov_img  = mov_img[0],
    #                                        ref_img  = ref_img[0],
    #                                        fsl_mat  = rigid_fsl_transform,
    #                                        ants_mat = rigid_itk_transform )

    #         elif linreg_method == 'ANTS':

    #             if T1_image != None:
    #                 ref_img.append(mean_dwi)
    #                 mov_img.append(T1_image)
    #                 nonlin_mov_img.append(mean_dwi)
    #                 nonlin_ref_img.append(t1_aligned)

    #                 #Also use the Laplacian
    #                 dwi_laplacian = Image(file = working_dir + '/dwi_laplacian.nii.gz')
    #                 t1_laplacian  = Image(file = working_dir + '/t1_laplacian.nii.gz')
    #                 os.system('ImageMath 3 ' + dwi_laplacian.filename + ' Laplacian ' + mean_dwi.filename)
    #                 os.system('ImageMath 3 ' + t1_laplacian.filename  + ' Laplacian ' + T1_image.filename)

    #                 mov_img.append(t1_laplacian)
    #                 ref_img.append(dwi_laplacian)
    #                 nonlin_ref_img.append(dwi_laplacian)
    #                 nonlin_mov_img.append(t1_laplacian)

    #             if T2_image != None:
    #                 ref_img.append(mean_b0)
    #                 mov_img.append(T2_image)
    #                 nonlin_mov_img.append(mean_b0)
    #                 nonlin_ref_img.append(t2_aligned)

    #                 b0_laplacian = Image(file = working_dir + '/b0_laplacian.nii.gz')
    #                 t2_laplacian  = Image(file = working_dir + '/t2_laplacian.nii.gz')
    #                 os.system('ImageMath 3 ' + b0_laplacian.filename + ' Laplacian ' + mean_b0.filename)
    #                 os.system('ImageMath 3 ' + t2_laplacian.filename  + ' Laplacian ' + T2_image.filename)

    #                 mov_img.append(t2_laplacian)
    #                 ref_img.append(b0_laplacian)
    #                 nonlin_ref_img.append(b0_laplacian)
    #                 nonlin_mov_img.append(t2_laplacian)

    #             reg_tools.linear_reg(input_img      = mov_img,
    #                                  reference_img  = ref_img,
    #                                  output_file    = output_img.filename,
    #                                  output_matrix  = rigid_ants_transform,
    #                                  nthreads       = nthreads,
    #                                  dof            = 6,
    #                                  method         = 'ANTS',
    #                                  ants_options   =  '-j 1')

    #             #Convert the ants transform to ITK
    #             os.system('ConvertTransformFile 3 ' +  rigid_ants_transform+'0GenericAffine.mat ' +  rigid_itk_transform)


    #         if T1_image != None:
    #             reg_tools.apply_transform(input_img     = T1_image,
    #                                       reference_img = ref_img[0],
    #                                       output_img    = t1_aligned,
    #                                       matrix        = rigid_itk_transform,
    #                                       method        = 'ANTS')

    #         if T2_image != None:
    #             reg_tools.apply_transform(input_img     = T2_image,
    #                                       reference_img = ref_img[0],
    #                                       output_img    = t2_aligned,
    #                                       matrix        = rigid_itk_transform,
    #                                       method        = 'ANTS')


    #     reg_tools.nonlinear_phase_encode_restricted_reg(input_img             = nonlin_mov_img,
    #                                                     reference_img         = nonlin_ref_img,
    #                                                     output_base           = working_dir + '/ants_nonlinear_',
    #                                                     ants_phase_encode_dir = ants_phase_encode_dir,
    #                                                     nthreads              = nthreads)

    #     if resample_to_anat:
    #         transform = working_dir + '/ants_nonlinear_composite.nii.gz'
    #         reg_tools.create_composite_transform(reference_img  = nonlin_ref_img[0],
    #                                              output_file    = working_dir + '/ants_nonlinear_composite.nii.gz',
    #                                              transforms     = [working_dir + '/ants_nonlinear_0Warp.nii.gz', rigid_itk_transform])
    #     else:
    #         transform = working_dir + '/ants_nonlinear_0Warp.nii.gz'

    #     reg_tools.apply_transform(input_img     = input_dwi,
    #                               reference_img = nonlin_ref_img[0],
    #                               output_img    = output_img,
    #                               matrix        = transform,
    #                               nthreads      = nthreads,
    #                               method        = 'ANTS',
    #                               ants_options  = ' -e 3 -n BSpline')

    #     if verbose:
    #         print('Rotating bvecs')


    #     rotate_bvecs(input_img      = input_dwi,
    #                  ref_img        = nonlin_ref_img[0],
    #                  output_bvec    = output_img._get_bvecs(),
    #                  transform      = rigid_itk_transform,
    #                  nthreads       = nthreads)


    return output_img


def epi_reg_fsl(input_dwi, input_bval, fieldmap, fieldmap_ref, struct_img, struct_brain, output_dwi, pedir, dwellTime, fm_ref_brain=''):

    dwi_img = nib.load(input_dwi)
    dwi_data = dwi_img.get_fdata()
    bvals = np.loadtxt(input_bval)
    ii = np.where(bvals == 0)

    output_dir = os.path.dirname(output_dwi)
    epi_ref = output_dir + '/tmp.epi.nii.gz'
    os.system('fslroi ' + input_dwi + ' ' + epi_ref + ' 0 1')

    #Align the structual image to the mean B0 to keep things in DWI spaces
    struct_img_aligned = output_dir + '/tmp.struct.nii.gz'
    struct_brain_aligned = output_dir + '/tmp.struct_brain.nii.gz'
    struct_img_mat = output_dir +'/tmp.struct_2_dwi.mat'
    os.system('flirt -in ' + struct_img + ' -ref ' + epi_ref + ' -out ' + struct_img_aligned + ' -omat ' + struct_img_mat + ' -searchrx -180 180 -searchrz -180 180 -searchry -180 180')
    os.system('flirt -in ' + struct_brain + ' -ref ' + epi_ref + ' -out ' + struct_brain_aligned + ' -applyxfm -init ' + struct_img_mat)

    bias_fieldmap_ref = output_dir + '/tmp.fm_ref.bias.nii.gz'
    os.system('N4BiasFieldCorrection -d 3 -i ' + fieldmap_ref + ' -o ' + bias_fieldmap_ref)
    if fm_ref_brain == '':
        struct2ref = output_dir + '/tmp.struct2ref.nii.gz'
        fm_ref_brain = output_dir + '/tmp.fm_ref_brain.nii.gz'
        fm_ref_omat = output_dir + '/tmp.fm_ref_brain.mat'
        os.system('flirt -in ' + struct_img + ' -ref ' +bias_fieldmap_ref + ' -omat ' + fm_ref_omat + ' -searchrx -180 180 -searchry -180 180 -searchrz -180 180')
        os.system('flirt -in ' + struct_brain + ' -ref ' +bias_fieldmap_ref + ' -out ' + struct2ref + ' -applyxfm -init ' + fm_ref_omat )
        os.system('fslmaths ' + struct2ref + ' -bin -fillh ' + struct2ref)
        os.system('fslmaths ' + fieldmap_ref + ' -mas ' + struct2ref + ' ' + fm_ref_brain)

    fm_rads = output_dir + '/tmp.fm.rads.nii.gz'
    os.system('fslmaths ' + fieldmap + ' -mul 6.28 ' + fm_rads)

    epi_reg_out = output_dir + '/tmp.epi_reg'
    os.system('epi_reg --epi=' + epi_ref + ' --t1=' + struct_img_aligned + ' --t1brain='+ struct_brain_aligned + ' --fmap=' + fm_rads + ' --fmapmag='+bias_fieldmap_ref+ ' --fmapmagbrain='+fm_ref_brain + ' --pedir=' + pedir + ' --echospacing='+dwellTime + ' --out='+epi_reg_out)

    #Apply warp to dwi series
    os.system('applywarp -i ' + input_dwi + ' -r ' + struct_img_aligned + ' -o ' + output_dwi + ' -w ' + epi_reg_out + '_warp.nii.gz --interp=spline --rel')
    os.system('rm -rf ' + output_dir + '/tmp*')

def fugue_fsl(dwi_image, fmap_image, fmap_ref_image, working_dir):

    if not os.path.exists(working_dir):
        os.mkdir(working_dir)

    fmap_ref_base=''
    if fmap_ref_image.filename.endswith('.nii'):
        fmap_ref_base = fmap_ref_image.filename[0:len(fmap_ref_image.filename)-4]
    else:
        fmap_ref_base = fmap_ref_image.filename[0:len(fmap_ref_image.filename)-7]

    fmap_base=''
    if fmap_image.filename.endswith('.nii'):
        fmap_base = fmap_image.filename[0:len(fmap_image.filename)-4]
    else:
        fmap_base = fmap_image.filename[0:len(fmap_image.filename)-7]
        

    parsed_filename = parse_file_entities(dwi_image.filename)
    entities = {
    'extension': '.nii.gz',
    'subject': parsed_filename.get('subject'),
    'session': parsed_filename.get('session'),
    'suffix':  'dwi',
    'desc': 'DistortionCorrected'
    }
    filename_patterns   = working_dir + '/sub-{subject}[_ses-{session}][_desc-{desc}]_{suffix}{extension}'

    out_file = writing.build_path(entities, filename_patterns)
    
    output_img = copy.deepcopy(dwi_image)
    output_img._set_filename(out_file)
  
        
    #Determine the Phase Encode Direction
    #Read the JSON file and get the
    with open(dwi_image.json) as f:
        json_data = json.load(f)

    dwell_time  = json_data["EffectiveEchoSpacing"]
    unwarpdir   = ''
    if json_data["PhaseEncodingDirection"] == 'i':
        unwarpdir = 'x-'
    elif json_data["PhaseEncodingDirection"] == 'i-':
        unwarpdir = 'x'
    elif json_data["PhaseEncodingDirection"] == 'j':
        unwarpdir = 'y-'
    elif json_data["PhaseEncodingDirection"] == 'j-':
        unwarpdir = 'y'
    elif json_data["PhaseEncodingDirection"] == 'k':
        unwarpdir = 'z-'
    elif json_data["PhaseEncodingDirection"] == 'k-':
        unwarpdir = 'z'
    else:
        print('Incorrect Phase Encoding Data')
        exit()
        
    #Skull-strip the reference
    mask_img=Image(filename = working_dir + '/mask.nii.gz')
    os.system('N4BiasFieldCorrection -d 3 -i ' + fmap_ref_image.filename + ' -o ' + mask_img.filename)
    os.system('bet ' + mask_img.filename + ' ' + mask_img.filename)
    os.system('fslmaths ' + mask_img.filename + ' -bin -fillh -dilM -dilM -ero -ero -bin ' + mask_img.filename)

    fm_ref_mask = Image(filename = working_dir + '/fmap_ref_mask.nii.gz')
    os.system('fslmaths ' + fmap_ref_image.filename + ' -mas ' + mask_img.filename + ' ' + fm_ref_mask.filename)


    #Now scale the field map and mask
    fmap_rads = Image(filename = working_dir + '/fmap_radians.nii.gz')
    os.system('fslmaths ' + fmap_image.filename + ' -mul 6.28 ' + fmap_rads.filename)
    os.system('fugue --loadfmap='+fmap_rads.filename+' --despike --smooth3=2 --savefmap='+fmap_rads.filename)

    #Warp the reference image
    input_fm_ref_warp = working_dir + '/fmap_warp.nii.gz'
    os.system('fugue -i ' + fm_ref_mask.filename + ' --unwarpdir='+str(unwarpdir) + ' --dwell='+str(dwell_time) + ' --loadfmap='+fmap_rads.filename + ' -w ' + input_fm_ref_warp)

    dwi_ref = working_dir + '/dwi_ref.nii.gz'
    bvals = np.loadtxt(dwi_image.bvals)
    ii = np.where(bvals != 0)

    dwi_img = nib.load(dwi_image.filename)
    aff = dwi_img.get_affine()
    sform = dwi_img.get_sform()
    qform = dwi_img.get_qform()
    dwi_data = dwi_img.get_fdata()

    dwi_mean = np.mean(dwi_data, axis=3)
    dwi_mean_img = nib.Nifti1Image(dwi_mean, aff, dwi_img.header)
    nib.save(dwi_mean_img, dwi_ref)
    os.system('N4BiasFieldCorrection -d 3 -i ' + dwi_ref + ' -o ' + dwi_ref)
    os.system('bet ' + dwi_ref + ' ' + dwi_ref)

    #Align warped reference to the dwi data
    fm_ref_warp_align = working_dir + '/fmap_warp-aligned.nii.gz'
    fm_ref_mat = working_dir + '/fmap2dwi.mat'
    os.system('flirt -in ' + input_fm_ref_warp + ' -ref ' + dwi_ref + ' -out ' + fm_ref_warp_align + ' -omat ' + fm_ref_mat + ' -dof 6 -cost normmi')

    #Apply this to the field map
    fm_rads_warp = working_dir + '/fmap_radians-warp.nii.gz'
    os.system('flirt -in ' + fmap_rads.filename + ' -ref ' + dwi_ref + ' -applyxfm -init ' + fm_ref_mat + ' -out ' + fm_rads_warp)

    #Now, undistort the image
    os.system('fugue -i ' + dwi_image.filename + ' --icorr --unwarpdir='+str(unwarpdir) + ' --dwell='+str(dwell_time) + ' --loadfmap='+fm_rads_warp+' -u ' + output_img.filename)
    
    return output_img


def prep_external_fieldmap(input_dwi, input_fm, input_fm_ref, dwellTime, unwarpdir, field_map_dir):

    if not os.path.exists(field_map_dir):
        os.mkdir(field_map_dir)

    #Skull-strip the reference
    if input_fm_ref.endswith('.nii'):
        input_fm_ref_base = input_fm_ref[0:len(input_fm_ref)-4]
    else:
        input_fm_ref_base = input_fm_ref[0:len(input_fm_ref)-7]

    fm_ref_mask = input_fm_ref_base + '.mask.nii.gz'

    os.system('bet ' + input_fm_ref + ' ' + fm_ref_mask)

    if input_fm.endswith('.nii'):
        input_fm_base = input_fm[0:len(input_fm)-4]
    else:
        input_fm_base = input_fm[0:len(input_fm)-7]

    fm_rads = input_fm_base + '.rads.nii.gz'

    #Now scale the field map and mask
    os.system('fslmaths ' + input_fm + ' -mul 6.28 -mas ' + fm_ref_mask + ' ' + fm_rads)

    input_fm_ref_warp = input_fm_ref_base + '.warp.nii.gz'
    #Warp the reference image
    os.system('fugue -i ' + fm_ref_mask + ' --unwarpdir='+unwarpdir + ' --dwell='+dwellTime + ' --loadfmap='+fm_rads + ' -w ' + input_fm_ref_warp)

    dwi_ref = field_map_dir + '/dwi_ref.nii.gz'
    os.system('fslroi ' + input_dwi + ' ' + dwi_ref + ' 0 1' )

    #Align warped reference to the dwi data
    fm_ref_warp_align = input_fm_ref_base + '.warp.aligned.nii.gz'
    fm_ref_mat = input_fm_ref_base + '_2_dwi.mat'
    os.system('flirt -in ' + input_fm_ref_warp + ' -ref ' + dwi_ref + ' -out ' + fm_ref_warp_align + ' -omat ' + fm_ref_mat)

    #Apply this to the field map
    fm_rads_warp = input_fm_base + '.rads.warp.nii.gz'
    os.system('flirt -in ' + fm_rads + ' -ref ' + dwi_ref + ' -applyxfm -init ' + fm_ref_mat + ' -out ' + fm_rads_warp)

    fm_hz_warp = input_fm_base + '.hz.warp.nii.gz'
    os.system('fslmaths ' + fm_rads_warp + ' -mul 0.1592 ' + fm_hz_warp)


def run_synb0_disco(dwi_img, t1w_img, topup_base, mask_method="mri_synthstrip", topup_config='b02b0.cnf', wmseg_img=None, nthreads=1, cleanup_files=True, verbose=True):

    working_dir = os.path.dirname(topup_base)

    if not os.path.exists(working_dir):
        os.makedirs(working_dir)

    parsed_filename = parse_file_entities(dwi_img.filename)

    entities = {
    'extension': '.nii.gz',
    'subject': parsed_filename.get('subject'),
    'session': parsed_filename.get('session'),
    'suffix':  'dwi',
    'desc': 'DistortionCorrected'
    }

    filename_patterns   = working_dir + '/sub-{subject}[_ses-{session}][_desc-{desc}]_{suffix}{extension}'

    out_file = writing.build_path(entities, filename_patterns)
    entities['extension'] = '.bvec'
    out_bvec = writing.build_path(entities, filename_patterns)

    output_img = copy.deepcopy(dwi_img)
    output_img.filename = out_file
    output_img.bvecs    = out_bvec

    #Extract the B0s from the DWI and compute mean                                     
    mean_b0 = Image(filename = working_dir + '/mean_b0.nii.gz')
    mean_b0 = dmri_tools.extract_b0s(input_dwi     = dwi_img,
                                     output_b0     = mean_b0,
                                     compute_mean  = True)

    #Skull strip T1w
    T1w_mask  = Image(filename = os.path.join(working_dir, "T1w_mask.nii.gz"))
    T1w_brain = Image(filename = os.path.join(working_dir, "T1w_brain.nii.gz"))
    mask_tools.mask_image(input    = t1w_img, 
                          mask     = T1w_mask,
                          mask_img = T1w_brain, 
                          algo     = mask_method)
    
    if wmseg_img == None:
        #Create WMseg
        wmseg_img = create_wmseg(input_img  = t1w_img,
                                 output_dir = os.path.join(working_dir), 
                                 nthreads   = nthreads)
    #Normalize the T1w image
    T1w_wm   = Image(filename = os.path.join(working_dir, "T1w_wm.nii.gz"))
    T1w_norm = Image(filename = os.path.join(working_dir, "T1w_norm.nii.gz"))
    os.system("fslmaths " + t1w_img.filename + " -mas " + wmseg_img.filename + "  " + T1w_wm.filename)

    WM     = nib.load(T1w_wm.filename).get_fdata()
    avg_wm = np.mean(WM[np.nonzero(WM)])
    os.system("fslmaths " + t1w_img.filename + " -div " + str(avg_wm) + " -mul 110 " + T1w_norm.filename)

    #Coregister the T1w to the DWI image
    T1w_2_dwi         = Image(filename = os.path.join(working_dir, "T1w_toDWI.nii.gz"))
    T1w_2_dwi_fslmat  = os.path.join(working_dir, "T1w_2_dwi.mat")
    T1w_2_dwi_antsmat = os.path.join(working_dir, "T1w_2_dwi.txt")

    linreg(input          = T1w_brain,
           ref            = mean_b0,
           out            = T1w_2_dwi,
           out_mat        = T1w_2_dwi_fslmat,
           method         = 'fsl',
           dof            = 6,
           flirt_options  = '-cost normmi -searchrx -180 180 -searchry -180 180 -searchrz -180 180')

    #CONVERT FSL TO ANTS
    convert_fsl2ants(input    = T1w_brain,
                     ref      = mean_b0,
                     fsl_mat  = T1w_2_dwi_fslmat,
                     ants_mat = T1w_2_dwi_antsmat)

    #REGISTER T1 to Atlas
    mni_atlas_img       = Image(filename = os.path.join(os.path.dirname(__file__), "data", "mni_icbm152_t1_tal_nlin_asym_09c_mask_2_5.nii.gz"))
    T1w_mni             = Image(filename = os.path.join(working_dir, "T1w_mni.nii.gz"))
    dwi_2_mni_fslmat    = os.path.join(working_dir, "dwi_2_mni.mat")
    dwi_2_mni_antsmat   = os.path.join(working_dir, "dwi_2_mni.txt")

    linreg(input          = T1w_2_dwi,
           ref            = mni_atlas_img,
           out            = T1w_mni,
           out_mat        = dwi_2_mni_fslmat,
           method         = 'fsl',
           dof            = 12,
           flirt_options  = '-searchrx -180 180 -searchry -180 180 -searchrz -180 180')
    
    convert_fsl2ants(input    = T1w_2_dwi,
                     ref      = mni_atlas_img,
                     fsl_mat  = dwi_2_mni_fslmat,
                     ants_mat = dwi_2_mni_antsmat)


    b0_in_mni         = Image(filename = os.path.join(working_dir, "b0_in_mni.nii.gz"))
    T1w_in_mni        = Image(filename = os.path.join(working_dir, "T1w_in_mni.nii.gz"))
    T1w_2_mni_antsmat = os.path.join(working_dir, "T1w_2_mni.txt")

    create_composite_transform(ref        = mni_atlas_img,
                               out        = T1w_2_mni_antsmat,
                               transforms = [dwi_2_mni_antsmat, T1w_2_dwi_antsmat], 
                               linear     = True,
                               inverse    = 0)
    #WARP B0 TO MNI
    apply_transform(input        = mean_b0,
                    ref          = mni_atlas_img,
                    out          = b0_in_mni,
                    transform    = dwi_2_mni_antsmat,
                    method       = "ants",
                    nthreads     = nthreads,
                    ants_options = "-n BSpline")

    #WARP T1w TO MNI
    apply_transform(input        = t1w_img,
                    ref          = mni_atlas_img,
                    out          = T1w_in_mni,
                    transform    = T1w_2_mni_antsmat,
                    method       = "ants",
                    nthreads     = nthreads,
                    ants_options = "-n BSpline")
        
    #USE Synb0 to predict the reverse encoded image
    if verbose:
        print('Creating synthetic undistorted b0 images')
    SyNb0       = Synb0(verbose)

    b0_img  = nib.load(b0_in_mni.filename)
    T1w_img = nib.load(T1w_in_mni.filename)
    rev_b0_data = SyNb0.predict(b0_img.get_fdata(), T1w_img.get_fdata())

    rev_b0_mni = Image(filename = os.path.join(working_dir, "b0_u_mni.nii.gz"))
    nib.save(nib.Nifti1Image(rev_b0_data.astype(b0_img.get_data_dtype()), b0_img.affine), rev_b0_mni.filename)
    
    #Inverse warp the image
    rev_b0 = Image(filename = os.path.join(working_dir, "b0_u.nii.gz"))
    os.system('antsApplyTransforms -d 3 -i ' + rev_b0_mni.filename + ' -r ' + mean_b0.filename + ' -t ['+dwi_2_mni_antsmat+',1] -o ' +  rev_b0.filename)

    
    #Smooth original b0 slightly
    b0_d = Image(filename = os.path.join(working_dir, "b0_d.nii.gz"))
    os.system('fslmaths ' + mean_b0.filename + ' -s 1.15 ' + b0_d.filename)

    #Merge and run topup
    all_b0s = Image(filename = os.path.join(working_dir, "b0s_all.nii.gz"))
    img_tools.merge_images(list_of_images = [b0_d,  rev_b0],
                           output_img    = all_b0s)


    #Create acqparams file for topup:
    acqparams = np.loadtxt(dwi_img.acqparams)
    syn_acqparams = np.copy(acqparams)
    syn_acqparams[3] = 0

    disco_acqparams = np.vstack((acqparams, syn_acqparams))
    disco_acqparams_path = working_dir + '/tmp_acqparams.txt'
    np.savetxt(disco_acqparams_path, disco_acqparams, fmt='%.8f')

    #Run TOPUP
    synb0_config = os.path.join(os.path.dirname(__file__), "data", "synb0.conf")
    topup_command = 'topup --imain='+ all_b0s.filename \
                  + ' --datain=' + disco_acqparams_path \
                  + ' --config=' + synb0_config \
                  + ' --out=' + topup_base \
                  + ' --fout=' + topup_base + '_fmap.nii.gz'

    if verbose:
        print(topup_command)
        
    os.system(topup_command)

    if cleanup_files:
        print("Cleanup all temporary files that were created")
