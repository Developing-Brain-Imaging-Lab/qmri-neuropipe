import os,sys, shutil, copy

from bids.layout import writing
from core.utils.io import Image, DWImage
import core.utils.tools as img_tools
import core.utils.mask as mask

import core.dmri.utils.qc as dmri_qc
import core.dmri.utils.distortion_correction as distcorr
import core.dmri.workflows.eddy_corr as eddy_proc
import core.dmri.workflows.distort_corr as distort_proc

import core.registration.registration as reg_tools
import core.segmentation.segmentation as seg_tools


def prep_anat_rawdata(bids_id, bids_rawdata_dir, bids_derivative_dir, bids_t1w_dir='anat', bids_t2w_dir='anat', t1w_reorient_img=None, t2w_reorient_img=None, t1w_type='t1w', nthreads=1, verbose=False):

    #Setup raw data paths
    bids_t1w_rawdata_dir         = os.path.join(bids_rawdata_dir, bids_t1w_dir,'')
    bids_t1w_derivative_dir      = os.path.join(bids_derivative_dir, bids_t1w_dir,'')

    bids_t2w_rawdata_dir         = os.path.join(bids_rawdata_dir, bids_t2w_dir,'')
    bids_t2w_derivative_dir      = os.path.join(bids_derivative_dir, bids_t2w_dir,'')

    #Setup Paths for T1w Images of different types
    raw_t1w=None
    t1w=None
    if t1w_type == 'mp2rage':
        raw_t1w._set_filename(file = bids_t1w_rawdata_dir + bids_id + '_inv-2_part-mag_MP2RAGE.nii.gz',
                              json = bids_t1w_rawdata_dir + bids_id + '_inv-2_MP2RAGE.json ')
        t1w = Image(file = bids_t1w_derivative_dir + bids_id + '_inv-2_part-mag_MP2RAGE.nii.gz',
                    json = bids_t1w_rawdata_dir + bids_id + '_inv-2_part-mag_MP2RAGE.nii.gz')        
    elif t1w_type == 'mpnrage':
        raw_t1w._set_filename(file = bids_t1w_rawdata_dir + bids_id + '_rec-MoCo_MPnRAGE.nii.gz',
                              json = bids_t1w_rawdata_dir + bids_id + '_MPnRAGE.json')
        t1w = Image(file = bids_t1w_derivative_dir + bids_id + '_rec-MoCo_MPnRAGE.nii.gz',
                    json = bids_t1w_rawdata_dir + bids_id + '_MPnRAGE.json')
    else:
        raw_t1w = Image(file = bids_t1w_rawdata_dir + bids_id + '_T1w.nii.gz',
                        json = bids_t1w_rawdata_dir + bids_id + '_T1w.json')

        t1w = Image(file = bids_t1w_derivative_dir + bids_id + '_T1w.nii.gz',
            json = bids_t1w_rawdata_dir + bids_id + '_T1w.json')

    #Setup Paths for T2w Images
    raw_t2w = Image(file = bids_t2w_rawdata_dir + bids_id + '_T2w.nii.gz',
                    json = bids_t2w_rawdata_dir + bids_id + '_T2w.json')
    t2w = Image(file = bids_t2w_derivative_dir + bids_id + '_T2w.nii.gz',
                json = bids_t2w_rawdata_dir + bids_id + '_T2w.json')

    #Check if the files exist, and if so, reorient them to standard orientation
    if not raw_t1w.exists() and not raw_t2w.exists():
        print('WARNING: No anatomical images found')
        t1w = None
        t2w = None
    elif not raw_t1w.exists():
        t1w = None
        if not os.path.exists(bids_t2w_derivative_dir):
            os.makedirs(bids_t2w_derivative_dir)
        if not t2w.exists():
            if verbose:
                print('Reorienting T2w image to standard')

            t2w = img_tools.reorient_to_standard(input_img      = raw_t2w,
                                                 output_file    = t2w._get_filename(),
                                                 reorient_img   = t2w_reorient_img)
    elif not raw_t2w.exists():
        t2w = None
        if not os.path.exists(bids_t1w_derivative_dir):
            os.makedirs(bids_t1w_derivative_dir)

        if not t1w.exists():
            if verbose:
                print('Reorienting T1w image to standard')
            t1w = img_tools.reorient_to_standard(input_img      = raw_t1w,
                                                 output_file    = t1w._get_filename(),
                                                 reorient_img   = t1w_reorient_img)
    else:
        if not os.path.exists(bids_t1w_derivative_dir):
            os.makedirs(bids_t1w_derivative_dir)

        if not os.path.exists(bids_t2w_derivative_dir):
            os.makedirs(bids_t2w_derivative_dir)

        if not t1w.exists():
            if verbose:
                print('Reorienting T1w image to standard')
            t1w = img_tools.reorient_to_standard(input_img      = raw_t1w,
                                                 output_file    = t1w._get_filename(),
                                                 reorient_img   = t1w_reorient_img)
        if not t2w.exists():
            if verbose:
                print('Reorienting T2w image to standard')
            t2w = img_tools.reorient_to_standard(input_img      = raw_t2w,
                                                 output_file    = t2w._get_filename(),
                                                 reorient_img   = t2w_reorient_img)

        #If both T1w and T2w images exist, coregister the two images using two-stage flirt and BBR ()
        coreg_t2 = copy.deepcopy(t2w)
        coreg_t2._set_filename(bids_t2w_derivative_dir + bids_id + '_space-individual-T1w_T2w.nii.gz')

        if not coreg_t2.exists():
            if verbose:
                print('Coregistering T1w and T2w images')

            #First, segment T1w image using FSL FAST
            seg_tools.fsl_fast(input_img    = t1w,
                               output_dir   = bids_t2w_derivative_dir + '/fast/',
                               fast_options = '-B -g -t 1 -n 4')

            WM_Seg = Image(bids_t2w_derivative_dir + '/fast/fast_seg_1.nii.gz')

            reg_tools.linear_reg(input_img      = t2w,
                                 reference_img  = t1w,
                                 output_matrix  = bids_t2w_derivative_dir + bids_id + '_space-individual-T1w_T2w.mat',
                                 method         = 'FSL',
                                 dof            = 6,
                                 flirt_options =  ' -cost normmi -interp sinc -searchrx -180 180 -searchry -180 180 -searchrz -180 180')

            bbr_options = ' -cost bbr -wmseg ' + WM_Seg._get_filename() + ' -schedule $FSLDIR/etc/flirtsch/bbr.sch -interp sinc -bbrtype global_abs -bbrslope 0.25 -finesearch 18 -init ' + bids_t2w_derivative_dir + bids_id + '_space-individual-T1w_T2w.mat'

            reg_tools.linear_reg(input_img     = t2w,
                                reference_img  = t1w,
                                output_matrix  = bids_t2w_derivative_dir + bids_id + '_space-individual-T1w_T2w.mat',
                                output_file    = coreg_t2._get_filename(),
                                method         = 'FSL',
                                dof            = 6,
                                flirt_options =  bbr_options)
            
            t2w = coreg_t2

    if os.path.exists(bids_t1w_derivative_dir + 'tmp_t1.nii.gz'):
        os.remove(bids_t1w_derivative_dir + 'tmp_t1.nii.gz')

    if os.path.exists(bids_t2w_derivative_dir + '/fast/'):
        shutil.rmtree(bids_t2w_derivative_dir + '/fast/')

    return t1w, t2w


def prep_dwi_rawdata(bids_id, bids_rawdata_dir, bids_derivative_dir, bids_dwi_dir='dwi', check_gradients=False, resample_resolution=None, remove_last_vol=False, distortion_correction='None', topup_config=None, outlier_detection=None, t1w_img=None, t1w_mask=None, nthreads=1, cmd_args=None, verbose=False):

    #Setup raw data paths
    bids_rawdata_dwi_dir        = os.path.join(bids_rawdata_dir, bids_dwi_dir,'')
    bids_derivative_dwi_dir     = os.path.join(bids_derivative_dir,bids_dwi_dir,'')

    #Define directories and image paths
    preprocess_dir              = bids_derivative_dwi_dir + '/preprocessed/rawdata/'

    if not os.path.exists(preprocess_dir):
        os.makedirs(preprocess_dir)

    dwi_img = DWImage(file      = preprocess_dir + bids_id + '_dwi.nii.gz',
                      bvals     = preprocess_dir + bids_id + '_dwi.bval',
                      bvecs     = preprocess_dir + bids_id + '_dwi.bvec',
                      index     = preprocess_dir + bids_id + '_desc-Index_dwi.txt',
                      acqparams = preprocess_dir + bids_id + '_desc-Acqparams_dwi.txt',
                      json      = preprocess_dir + bids_id + '_dwi.json')

    topup_base = preprocess_dir + '/topup/' + bids_id + '_desc-Topup'
    run_topup  = False

    #Check to see if TOPUP Style data exists and if so, create merged DWI input image
    if os.path.exists(bids_rawdata_dwi_dir + bids_id + '_desc-pepolar-0_dwi.nii.gz') and os.path.exists(bids_rawdata_dwi_dir + bids_id + '_desc-pepolar-1_dwi.nii.gz'):

        if distortion_correction == 'Topup' or distortion_correction == 'Topup-Separate':
            run_topup  = True

        if not dwi_img.exists():
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

            dwi_img = dmri_qc.merge_phase_encodes(DWI_pepolar0 = pepolar_0,
                                                  DWI_pepolar1 = pepolar_1,
                                                  output_base  = preprocess_dir + bids_id)

    else:
        shutil.copy2(bids_rawdata_dwi_dir + bids_id + '_dwi.nii.gz', dwi_img._get_filename())
        shutil.copy2(bids_rawdata_dwi_dir + bids_id + '_dwi.bvec', dwi_img._get_bvecs())
        shutil.copy2(bids_rawdata_dwi_dir + bids_id + '_dwi.bval', dwi_img._get_bvals())
        shutil.copy2(bids_rawdata_dwi_dir + bids_id + '_dwi.json', dwi_img._get_json())

    #Ensure ISOTROPIC voxels prior to processing
    if verbose:
        print('Ensuring DWIs have isotropic voxels')

    dwi_img = img_tools.check_isotropic_voxels(input_img          = dwi_img,
                                               output_file        = dwi_img._get_filename(),
                                               target_resolution  = resample_resolution)

    #Remove Last DWI volume before processing further
    if remove_last_vol:
        if verbose:
            print('Removing Last DWI in volume')

        dwi_img = img_tools.remove_end_img(input_img   = dwi_img,
                                           output_file = dwi_img._get_filename())

    #Check the Image Sizes to Ensure Proper Length:
    if verbose:
        print('Checking DWI Acquisition Size and Gradient Orientations')

    dmri_qc.check_bvals_bvecs(input_dwi   = dwi_img,
                              output_base = preprocess_dir + bids_id)


    if check_gradients:
        dmri_qc.check_gradient_directions(input_dwi   = dwi_img,
                                          nthreads    = nthreads)

    index     = preprocess_dir + bids_id + '_desc-Index_dwi.txt'
    acqparams = preprocess_dir + bids_id + '_desc-Acqparams_dwi.txt'
    if not os.path.exists(index) or not os.path.exists(acqparams):
        index, acqparams = dmri_qc.create_index_acqparam_files(input_dwi   = dwi_img,
                                                               output_base = preprocess_dir + bids_id)
    dwi_img._set_index(index)
    dwi_img._set_acqparams(acqparams)

    slspec = preprocess_dir + bids_id + '_desc-Slspec_dwi.txt'
    if not os.path.exists( slspec ):
        slspec = dmri_qc.create_slspec_file(input_dwi        = dwi_img,
                                            output_base      = preprocess_dir + bids_id)
    dwi_img._set_slspec(slspec)

    if outlier_detection == 'Manual':
        outlier_detection_dir = os.path.join(bids_derivative_dir, 'dwi/', 'preprocessed/', 'outlier-removed-images/')

        if verbose:
            print('Removing DWIs from manual selection')

        dwi_img = eddy_proc.perform_outlier_detection(dwi_image         = dwi_img,
                                                      working_dir       = outlier_detection_dir,
                                                      method            = outlier_detection,
                                                      manual_report_dir = bids_rawdata_dwi_dir,
                                                      verbose           = verbose )

    if run_topup or distortion_correction == 'Synb0-Disco':
        
        if not os.path.exists(topup_base + '_fieldcoef.nii.gz'):
    
            #First going to run eddy and motion-correction to ensure images are aligned prior to estimating fields. Data are only used
            #here and not for subsequent processing
            eddy_img = eddy_proc.perform_eddy(dwi_image                  = dwi_img,
                                              working_dir                = os.path.join(preprocess_dir, 'tmp-eddy-correction/'),
                                              topup_base                 = None,
                                              method                     = cmd_args.dwi_eddy_current_correction,
                                              gpu                        = cmd_args.gpu,
                                              cuda_device                = cmd_args.cuda_device,
                                              nthreads                   = cmd_args.nthreads,
                                              data_shelled               = cmd_args.dwi_data_shelled,
                                              repol                      = cmd_args.repol,
                                              estimate_move_by_suscept   = cmd_args.estimate_move_by_suscept,
                                              mporder                    = cmd_args.mporder,
                                              slspec                     = cmd_args.dwi_slspec,
                                              fsl_eddy_options           = cmd_args.dwi_eddy_options,
                                              verbose                    = cmd_args.verbose)
        
            topup_base = preprocess_dir + '/topup/' + bids_id + '_desc-Topup'

            if run_topup:
                distort_proc.perform_topup(dwi_image    = eddy_img,
                                           topup_base   = topup_base,
                                           topup_config = topup_config,
                                           dist_corr    = 'Topup',
                                           verbose=verbose)


            if distortion_correction == 'Synb0-Disco':
                #Run the Synb0 distortion correction'
                distcorr.run_synb0_disco(dwi_img        = eddy_img,
                                         t1w_img        = t1w_img,
                                         t1w_mask       = t1w_mask,
                                         topup_base     = topup_base,
                                         topup_config   = topup_config,
                                         nthreads   = 1)

    return dwi_img, topup_base
