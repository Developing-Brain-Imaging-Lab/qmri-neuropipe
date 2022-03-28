import string, os, sys, subprocess, shutil, time, copy

#Neuroimaging Modules
import numpy as np
import nibabel as nib
import ants
from dipy.io.image import load_nifti, save_nifti
from dipy.segment.mask import median_otsu

from bids.layout import writing, parse_file_entities
from core.utils.io import Image, DWImage
import core.utils.tools as img_tools

def convert_fsl2ants(mov_img, ref_img, fsl_mat, ants_mat):
    os.system('c3d_affine_tool -ref ' + ref_img._get_filename() + ' -src ' +  mov_img._get_filename() + ' ' + fsl_mat + ' -fsl2ras -oitk ' + ants_mat)

def create_composite_transform(reference_img, output_file, transforms):

    cmd = 'antsApplyTransforms -d 3 -o [' + output_file + ',1] -r ' + reference_img._get_filename()

    for xfm in transforms:
        cmd += ' -t ' + xfm

    os.system(cmd)

def create_composite_linear_transform(reference_img, output_file, transforms):

    cmd = 'antsApplyTransforms -d 3 -o Linear[' + output_file + '] -r ' + reference_img._get_filename()

    for xfm in transforms:
        cmd += ' -t ' + xfm

    os.system(cmd)

def apply_transform(input_img, reference_img, output_file, matrix, nthreads=1, method='FSL', flirt_options=None, ants_options=None):

    cmd = ''

    output_img = copy.deepcopy(input_img)
    output_img._set_filename(output_file)

    if method == 'FSL':
        cmd = 'flirt -in ' + input_img._get_filename() \
              + ' -ref ' + reference_img._get_filename() \
              + ' -out ' + output_img._get_filename() \
              + ' -applyxfm -init ' + matrix

        if flirt_options != None:
            cmd += ' ' + flirt_options

        os.system(cmd)

    elif method == 'ANTS':
        os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = str(nthreads)
        cmd = 'antsApplyTransforms -d 3 -i ' + input_img._get_filename() \
              + ' -r ' + reference_img._get_filename() \
              + ' -o ' + output_img._get_filename() \
              + ' -t ' + matrix

        if ants_options != None:
            cmd += ' ' + ants_options

        os.system(cmd)


    elif method == 'MRTRIX':

        output_dir = os.path.dirname(output_file)

        mrtrix_img = output_dir + '/img.mif'
        os.system('mrconvert -fslgrad ' + input_img._get_bvecs() + ' ' + input_img._get_bvals() + ' ' + input_img._get_filename() + ' ' + mrtrix_img + ' -force -quiet -nthreads ' + str(nthreads))

        ident_warp = output_dir + '/identity_warp'
        mrtrix_warp = output_dir + '/mrtrix_warp'
        os.system('warpinit ' + mrtrix_img + ' ' + ident_warp+'[].nii -force -quiet')

        for i in range(0,3):
            os.system('antsApplyTransforms -d 3 -e 0 -i ' + ident_warp+str(i)+'.nii -o ' + mrtrix_warp+str(i)+'.nii -r ' + reference_img._get_filename() + ' -t ' + matrix)

        mrtrix_corr_warp = output_dir + 'mrtrix_warp_corrected.mif'
        os.system('warpcorrect ' + mrtrix_warp+'[].nii ' +  mrtrix_corr_warp + ' -force -quiet')

        warped_img = output_dir + '/img_warped.mif'
        os.system('mrtransform ' + mrtrix_img + ' -warp ' + mrtrix_corr_warp + ' ' + warped_img + ' -template ' + reference_img._get_filename() + ' -strides ' + reference_img._get_filename() + ' -force -quiet -nthreads ' + str(nthreads) + ' -interp sinc')
        os.system('mrconvert -force -quiet ' + warped_img + ' ' + output_file + ' -nthreads ' + str(nthreads))

        os.remove(mrtrix_img)
        os.system('rm -rf ' + ident_warp+'*')
        os.system('rm -rf ' + mrtrix_warp+'*')
        os.remove(mrtrix_corr_warp)
        os.remove(warped_img)

    #elif method == 'FreeSurfer':


def linear_reg(input_img, reference_img, output_matrix, output_file=None, dof=6, nthreads=1, method='FSL', flirt_options=None, ants_options=None, freesurfer_subjs_dir=None):

    cmd = ''

    if output_file != None:
        if type(input_img) is list:
            output_img = copy.deepcopy(input_img[0])
        else:
            output_img = copy.deepcopy(input_img)

        output_img._set_filename(output_file)

    if method == 'FSL':

        if type(input_img) is list:
            input_img       = input_img[0]
            reference_img   = reference_img[0]

        cmd = 'flirt -in ' + input_img._get_filename() \
            + ' -ref ' +  reference_img._get_filename() \
            + ' -omat ' + output_matrix \
            + ' -dof ' + str(dof)

        if output_file != None:
            cmd += ' -out ' + output_img._get_filename()
        if flirt_options != None:
            cmd += ' ' + flirt_options

    elif method == 'ANTS':

        os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = str(nthreads)
        cmd = 'antsRegistrationSyN.sh -d 3 -o ' + output_matrix + ' -n ' + str(nthreads)

        if dof == 6:
            cmd += ' -t r'
        else:
            cmd += ' -t a'

        if type(input_img) is list:
            for i in range(0,len(input_img)):
                cmd += ' -m ' + input_img[i]._get_filename() \
                    +  ' -f ' + reference_img[i]._get_filename()
        else:
                cmd += ' -m ' + input_img._get_filename() \
                    +  ' -f ' + reference_img._get_filename()

        cmd += ' ' + ants_options
        
    elif method == 'BBR':
    
        if type(input_img) is list:
            input_img       = input_img[0]
            reference_img   = reference_img[0]

        parsed_filename = parse_file_entities(input_img._get_filename())
        entities = {
        'subject': parsed_filename.get('subject'),
        'session': parsed_filename.get('session'),
        }
        subid_patterns   = 'sub-{subject}[_ses-{session}]'
        subid = writing.build_path(entities, subid_patterns)
    
        os.environ["SUBJECTS_DIR"] = freesurfer_subjs_dir
        output_dir = os.path.dirname(output_matrix)
        freesurfer_tmp_dir = output_dir + '/tmp/'
        if not os.path.exists(freesurfer_tmp_dir):
            os.makedirs(freesurfer_tmp_dir)

        ## run bbregister and output transform in fsl format
        b0toT1mat      = output_dir + '/b0toT1.mat'
        b0toT1lta      = output_dir + '/b0toT1.lta'
        b0toT1flirtmtx = output_dir + '/b0toT1flirt.mtx'
        
        os.system('bbregister --s '+subid + ' --mov ' + input_img._get_filename() + ' --reg ' + reference_img._get_filename() + ' --dti --init-fsl --lta ' + b0toT1lta + ' --fslmat ' + b0toT1flirtmtx + ' --tmp ' + freesurfer_tmp_dir)

        convert_fsl2ants(input_img, reference_img, b0toT1flirtmtx, output_matrix)
        
        


def nonlinear_reg(input_img, reference_img, reference_mask, output_base, nthreads=1, method='ANTS', ants_options=None):

    cmd = ''

    if method == 'ANTS' or method == 'ANTS-QUICK':

        os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = str(nthreads)

        if method == 'ANTS':
            cmd = 'antsRegistrationSyN.sh'
        else:
            cmd = 'antsRegistrationSyNQuick.sh'

        cmd += ' -d 3 -n ' + str(nthreads) + ' -o ' + output_base

        if type(input_img) is list:
            for i in range(0,len(input_img)):
                cmd += ' -m ' + input_img[i]._get_filename() \
                    +  ' -f ' + reference_img[i]._get_filename()
        else:
                cmd += ' -m ' + input_img._get_filename() \
                    +  ' -f ' + reference_img._get_filename()

        cmd += ' -x ' + reference_mask._get_filename()

        if ants_options != None:
            cmd += ' ' + ants_options


    os.system(cmd)



def nonlinear_phase_encode_restricted_reg(input_img, reference_img, output_base, ants_phase_encode_dir, nthreads=1):

    os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = str(nthreads)
    cmd = 'antsRegistration -d 3 -o ' + output_base \
        + '  --convergence [150x75x50x25,1e-4,5] -t SyN[0.5,3,0] -f 8x4x2x1 -s 4x2x1x0mm -z 1 -u 1 -g ' + ants_phase_encode_dir

    if type(input_img) is list:
        for i in range(0, len(input_img)):
            cmd += ' --metric CC['+reference_img[i]._get_filename()+','+input_img[i]._get_filename()+',.8,5,Regular,0.25]'


    os.system(cmd)
