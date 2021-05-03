import os,sys, shutil, json, argparse, copy
import nibabel as nib

from bids.layout import writing
from core.utils.io import Image, DWImage
import core.utils.dmri.qc as dmri_qc
import core.utils.tools as img_tools
import core.utils.mask as mask
import core.registration.registration as reg_tools
import core.workflows.dmri.distort_corr as distort_proc

def prep_anat_rawdata(bids_id, bids_rawdata_dir, bids_derivative_dir, nthreads=1, verbose=False):

    #Setup raw data paths
    bids_rawdata_anat_dir        = os.path.join(bids_rawdata_dir, 'anat/')
    bids_derivative_anat_dir     = os.path.join(bids_derivative_dir, 'anat/')

    if not os.path.exists(bids_derivative_anat_dir):
        os.makedirs(bids_derivative_anat_dir)

    raw_t1w = Image(file = bids_rawdata_anat_dir + bids_id + '_T1w.nii.gz',
                json = bids_rawdata_anat_dir + bids_id + '_T1w.json')

    raw_t2w = Image(file = bids_rawdata_anat_dir + bids_id + '_T2w.nii.gz',
                json = bids_rawdata_anat_dir + bids_id + '_T2w.json')
                
    t1w = Image(file = bids_derivative_anat_dir + bids_id + '_T1w.nii.gz',
                json = bids_rawdata_anat_dir + bids_id + '_T1w.json')

    t2w = Image(file = bids_derivative_anat_dir + bids_id + '_T2w.nii.gz',
                json = bids_rawdata_anat_dir + bids_id + '_T2w.json')

    if not raw_t1w.exists() and not raw_t2w.exists():
        print('WARNING: No anatomical images found')
        t1w = None
        t2w = None
    elif not raw_t1w.exists():
        t1w = None
        
        if not t2w.exists():
            if verbose:
                print('Reorienting T2w image to standard')
            t2w = img_tools.reorient_to_standard(raw_t2w, t2w._get_filename())
    elif not raw_t2w.exists():
        t2w = None
        
        if not t1w.exists():
            if verbose:
                print('Reorienting T1w image to standard')
            t1w = img_tools.reorient_to_standard(raw_t1w, t1w._get_filename())
    else:
        
        if not t1w.exists():
            if verbose:
                print('Reorienting T1w image to standard')
            t1w = img_tools.reorient_to_standard(raw_t1w, t1w._get_filename())
            
        if not t2w.exists():
            if verbose:
                print('Reorienting T2w image to standard')
            t2w = img_tools.reorient_to_standard(raw_t2w, t2w._get_filename())
        
        #Coregister the two images
        coreg_t2 = copy.deepcopy(t2w)
        coreg_t2._set_filename(bids_derivative_anat_dir + bids_id + '_space-individual-T1w_T2w.nii.gz')

        if not coreg_t2.exists():
            if verbose:
                print('Coregistering T1w and T2w images')
            reg_tools.linear_reg(input_img      = t2w,
                                 reference_img  = t1w,
                                 output_matrix  = bids_derivative_anat_dir + bids_id + '_space-individual-T1w_T2w.mat',
                                 output_file    = coreg_t2._get_filename(),
                                 method         = 'FSL',
                                 dof            = 6,
                                 flirt_options =  '-cost normmi -interp sinc -searchrx -180 180 -searchry -180 180 -searchrz -180 180')
        t2w = coreg_t2

    return t1w, t2w


def prep_dwi_rawdata(bids_id, bids_rawdata_dir, bids_derivative_dir, resample_resolution=None, remove_last_vol = False, topup_config=None, nthreads=1, verbose=False ):

    #Setup raw data paths
    bids_rawdata_dwi_dir        = bids_rawdata_dir + '/dwi/'
    bids_derivative_dwi_dir     = bids_derivative_dir + '/dwi/'

    #Define directories and image paths
    preprocess_dir              = bids_derivative_dwi_dir +'/preprocessed/rawdata/'

    if not os.path.exists(preprocess_dir):
        os.makedirs(preprocess_dir)


    raw_dwi = DWImage(file    = bids_rawdata_dwi_dir + bids_id + '_dwi.nii.gz',
                      bvals   = bids_rawdata_dwi_dir + bids_id + '_dwi.bval',
                      bvecs   = bids_rawdata_dwi_dir + bids_id + '_dwi.bvec',
                      json    = bids_rawdata_dwi_dir + bids_id + '_dwi.json')


    out_dwi = DWImage(file      = preprocess_dir + bids_id + '_dwi.nii.gz',
                      bvals     = preprocess_dir + bids_id + '_dwi.bval',
                      bvecs     = preprocess_dir + bids_id + '_dwi.bvec',
                      index     = preprocess_dir + bids_id + '_desc-Index_dwi.txt',
                      acqparams = preprocess_dir + bids_id + '_desc-Acqparams_dwi.txt',
                      json      = preprocess_dir + bids_id + '_dwi.json')

    topup_base = None

    #Check to see if TOPUP Style data exists and if so, create merged DWI input image
    if os.path.exists(bids_rawdata_dwi_dir + bids_id + '_desc-pepolar-0_dwi.nii.gz') and os.path.exists(bids_rawdata_dwi_dir + bids_id + '_desc-pepolar-1_dwi.nii.gz'):

        topup_base = preprocess_dir + '/topup/' + bids_id + '_desc-Topup'
        
        if not out_dwi.exists():
            if verbose:
                print('Merging DWIs with different phase encode directions')

            pepolar_0 = DWImage(file    = bids_rawdata_dwi_dir + bids_id + '_desc-pepolar-0_dwi.nii.gz',
                                bvals   = bids_rawdata_dwi_dir + bids_id + '_desc-pepolar-0_dwi.bval',
                                bvecs   = bids_rawdata_dwi_dir + bids_id + '_desc-pepolar-0_dwi.bvec',
                                json    = bids_rawdata_dwi_dir + bids_id + '_desc-pepolar-0_dwi.json')

            pepolar_1 = DWImage(file    = bids_rawdata_dwi_dir + bids_id + '_desc-pepolar-1_dwi.nii.gz',
                                bvals   = bids_rawdata_dwi_dir + bids_id + '_desc-pepolar-1_dwi.bval',
                                bvecs   = bids_rawdata_dwi_dir + bids_id + '_desc-pepolar-1_dwi.bvec',
                                json    = bids_rawdata_dwi_dir + bids_id + '_desc-pepolar-1_dwi.json')
                                
            raw_dwi = dmri_qc.merge_phase_encodes(DWI_pepolar0 = pepolar_0,
                                                  DWI_pepolar1 = pepolar_1,
                                                  output_base  = preprocess_dir + bids_id)
            
            shutil.copy2(pepolar_0._get_json(), out_dwi._get_json())
        
        
        if not os.path.exists(topup_base + '_fieldcoef.nii.gz'):
            distort_proc.perform_topup(dwi_image    = raw_dwi,
                                       topup_base   = topup_base,
                                       topup_config = topup_config,
                                       dist_corr    = 'Topup',
                                       verbose=verbose)
    else:
        shutil.copy2(raw_dwi._get_filename(), out_dwi._get_filename())
        shutil.copy2(raw_dwi._get_bvecs(), out_dwi._get_bvecs())
        shutil.copy2(raw_dwi._get_bvals(), out_dwi._get_bvals())
        shutil.copy2(raw_dwi._get_json(), out_dwi._get_json())
    
    
    #Ensure ISOTROPIC voxels prior to processing
    if verbose:
        print('Ensuring DWIs have isotropic voxels')

    out_dwi = img_tools.check_isotropic_voxels(input_img          = raw_dwi,
                                               output_file        = out_dwi._get_filename(),
                                               target_resolution  = resample_resolution)

    #Remove Last DWI volume before processing further
    if remove_last_vol:
        if verbose:
            print('Removing Last DWI in volume')

        out_dwi = img_tools.remove_end_img(input_img   = out_dwi,
                                           output_file = out_dwi._get_filename())

    #Check the Image Sizes to Ensure Proper Length:
    if verbose:
        print('Checking DWI Acquisition Size and Gradient Orientations')

    dmri_qc.check_bvals_bvecs(input_dwi   = raw_dwi,
                              output_base = preprocess_dir + bids_id)

    dmri_qc.check_gradient_directions(input_dwi   = raw_dwi,
                                      nthreads    = nthreads)

    index     = preprocess_dir + bids_id + '_desc-Index_dwi.txt'
    acqparams = preprocess_dir + bids_id + '_desc-Acqparams_dwi.txt'
    if not os.path.exists(index) or not os.path.exists(acqparams):
        index, acqparams = dmri_qc.create_index_acqparam_files(input_dwi   = raw_dwi,
                                                               output_base = preprocess_dir + bids_id)
    out_dwi._set_index(index)
    out_dwi._set_acqparams(acqparams)

    slspec = preprocess_dir + bids_id + '_desc-Slspec_dwi.txt'
    if not os.path.exists( slspec ):
        slspec = dmri_qc.create_slspec_file(input_dwi    = raw_dwi,
                                            output_base  = preprocess_dir + bids_id)
    out_dwi._set_slspec(slspec)

    return out_dwi, topup_base
