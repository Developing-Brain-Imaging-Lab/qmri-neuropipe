import os, shutil

from core.utils.io import Image, DWImage
import core.utils.tools as img_tools
import core.utils.mask as mask
import core.utils.biascorrect as bias_tools

import core.dmri.utils.qc as dmri_qc
import core.dmri.utils.distortion_correction as distcorr
import core.dmri.workflows.eddy_corr as eddy_proc
import core.dmri.workflows.distort_corr as distort_proc
from core.dmri.workflows.dmri_reorient import dmri_reorient

def prep_dwi_rawdata(bids_id, bids_rawdata_dir, dwi_preproc_dir, check_gradients=False, reorient_dwi=False, dwi_reorient_template=None, resample_resolution=None, remove_last_vol=False, distortion_correction='None', topup_config=None, outlier_detection=None, t1w_img=None, t1w_mask=None, nthreads=1, cmd_args=None, verbose=False):

    #Setup raw data paths
    bids_rawdata_dwi_dir        = os.path.join(bids_rawdata_dir, "dwi/")

    #Define directories and image paths
    preprocess_dir              = os.path.join(dwi_preproc_dir, "rawdata/")

    if not os.path.exists(preprocess_dir):
        os.makedirs(preprocess_dir)

    dwi_img = DWImage(filename  = preprocess_dir + bids_id + '_dwi.nii.gz',
                      bvals     = preprocess_dir + bids_id + '_dwi.bval',
                      bvecs     = preprocess_dir + bids_id + '_dwi.bvec',
                      index     = preprocess_dir + bids_id + '_desc-Index_dwi.txt',
                      acqparams = preprocess_dir + bids_id + '_desc-Acqparams_dwi.txt',
                      json      = preprocess_dir + bids_id + '_dwi.json')

    topup_base = None
    run_topup  = False

    #Check to see if TOPUP Style data exists and if so, create merged DWI input image
    if os.path.exists(os.path.join(bids_rawdata_dwi_dir,bids_id+'_desc-pepolar-0_dwi.nii.gz')) and os.path.exists(os.path.join(bids_rawdata_dwi_dir,bids_id+'_desc-pepolar-1_dwi.nii.gz')):

        if distortion_correction == "Topup" or distortion_correction == "Topup-Separate":
            run_topup  = True

        if not dwi_img.exists():
            if verbose:
                print('Merging DWIs with different phase encode directions')

            pepolar_0 = DWImage(filename = bids_rawdata_dwi_dir + bids_id + '_desc-pepolar-0_dwi.nii.gz',
                                bvals    = bids_rawdata_dwi_dir + bids_id + '_desc-pepolar-0_dwi.bval',
                                bvecs    = bids_rawdata_dwi_dir + bids_id + '_desc-pepolar-0_dwi.bvec',
                                json     = bids_rawdata_dwi_dir + bids_id + '_desc-pepolar-0_dwi.json')

            pepolar_1 = DWImage(filename = bids_rawdata_dwi_dir + bids_id + '_desc-pepolar-1_dwi.nii.gz',
                                bvals    = bids_rawdata_dwi_dir + bids_id + '_desc-pepolar-1_dwi.bval',
                                bvecs    = bids_rawdata_dwi_dir + bids_id + '_desc-pepolar-1_dwi.bvec',
                                json     = bids_rawdata_dwi_dir + bids_id + '_desc-pepolar-1_dwi.json')

            dwi_img = dmri_qc.merge_phase_encodes(DWI_pepolar0 = pepolar_0,
                                                  DWI_pepolar1 = pepolar_1,
                                                  output_base  = preprocess_dir + bids_id)

    else:
        shutil.copy2(bids_rawdata_dwi_dir + bids_id + '_dwi.nii.gz', dwi_img.filename)
        shutil.copy2(bids_rawdata_dwi_dir + bids_id + '_dwi.bvec', dwi_img.bvecs)
        shutil.copy2(bids_rawdata_dwi_dir + bids_id + '_dwi.bval', dwi_img.bvals)
        shutil.copy2(bids_rawdata_dwi_dir + bids_id + '_dwi.json', dwi_img.json)

    #Ensure ISOTROPIC voxels prior to processing
    if verbose:
        print('Ensuring DWIs have isotropic voxels')

    dwi_img = img_tools.check_isotropic_voxels(input_img          = dwi_img,
                                               output_file        = dwi_img.filename,
                                               target_resolution  = resample_resolution)

    #Remove Last DWI volume before processing further
    if remove_last_vol:
        if verbose:
            print('Removing Last DWI in volume')

        dwi_img = img_tools.remove_end_img(input_img   = dwi_img,
                                           output_file = dwi_img.filename)

    #Check the Image Sizes to Ensure Proper Length:
    if verbose:
        print('Checking DWI Acquisition Size and Gradient Orientations')

    dmri_qc.check_bvals_bvecs(input_dwi   = dwi_img,
                              output_base = preprocess_dir + bids_id)

    if check_gradients:
        dmri_qc.check_gradient_directions(input_dwi   = dwi_img,
                                          nthreads    = nthreads)

    dwi_img.index     = preprocess_dir + bids_id + '_desc-Index_dwi.txt'
    dwi_img.acqparams = preprocess_dir + bids_id + '_desc-Acqparams_dwi.txt'
    
    if not os.path.exists(dwi_img.index) or not os.path.exists(dwi_img.acqparams):
        dwi_img.index, dwi_img.acqparams = dmri_qc.create_index_acqparam_files(input_dwi   = dwi_img,
                                                                               output_base = preprocess_dir + bids_id)

    dwi_img.slspec = preprocess_dir + bids_id + '_desc-Slspec_dwi.txt'
    if not os.path.exists( dwi_img.slspec ):
        dwi_img.slspec = dmri_qc.create_slspec_file(input_dwi        = dwi_img,
                                                    output_base      = preprocess_dir+bids_id)

    if outlier_detection == 'Manual':
        outlier_detection_dir = os.path.join(dwi_preproc_dir, 'outlier-removed-images/')

        if verbose:
            print('Removing DWIs from manual selection')

        dwi_img = eddy_proc.perform_outlier_detection(dwi_image         = dwi_img,
                                                      working_dir       = outlier_detection_dir,
                                                      method            = outlier_detection,
                                                      manual_report_dir = bids_rawdata_dwi_dir,
                                                      verbose           = verbose )
        
    if reorient_dwi:
        dmri_reorient(in_dwi  = dwi_img,
                      out_dwi = dwi_img,
                      ref_img = dwi_reorient_template)

    if run_topup or distortion_correction == 'Synb0-Disco':
        topup_base = os.path.join(preprocess_dir, "topup", bids_id+"_desc-Topup")
        
        if not os.path.exists(topup_base+'_fieldcoef.nii.gz'):
    
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
                                         nthreads       = nthreads)

    return dwi_img, topup_base
