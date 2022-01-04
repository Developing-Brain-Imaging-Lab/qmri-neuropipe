import os,sys, shutil, json, argparse, copy
import nibabel as nib
import numpy as np

from bids.layout import writing
from core.utils.io import Image, DWImage

import core.utils.workflows.prep_rawdata as raw_proc
import core.utils.workflows.denoise_degibbs as img_proc
import core.dmri.workflows.eddy_corr as eddy_proc
import core.dmri.workflows.distort_corr as distort_proc
import core.dmri.workflows.register_to_anat as coreg_proc

from core.anatomical.anat_proc import AnatomicalPrepPipeline

import core.dmri.utils.qc as dmri_qc
import core.utils.tools as img_tools
import core.utils.mask as mask
import core.utils.denoise as denoise

from core.dmri.models.dti import DTI_Model, FWEDTI_Model
from core.dmri.models.dki import DKI_Model
from core.dmri.models.csd import CSD_Model
from core.dmri.models.noddi import NODDI_Model

class DiffusionProcessingPipeline:

    def __init__(self, verbose=False):
        if verbose:
            print('Creating Diffusion Processing Pipeline')

    def run(self):

        parser = argparse.ArgumentParser()

        parser.add_argument('--bids_dir',
                            type=str,
                            help='BIDS Data Directory')

        parser.add_argument('--bids_rawdata_dir',
                            type=str, help='BIDS RAWDATA Directory',
                            default='rawdata')

        parser.add_argument('--bids_dwi_dir',
                            type=str, help='BIDS DWI RAWDATA Directory Basename',
                            default='dwi')

        parser.add_argument('--bids_t1w_dir',
                            type=str, help='BIDS T1w RAWDATA Directory Basename',
                            default='anat')

        parser.add_argument('--bids_t2w_dir',
                            type=str, help='BIDS T2w RAWDATA Directory Basename',
                            default='anat')

        parser.add_argument('--load_json',
                            type=str, help='Load settings from file in json format. Command line options are overriden by values in file.',
                            default=None)

        parser.add_argument('--subject',
                            type=str,
                            help='Subject ID')

        parser.add_argument('--session',
                            type=str,
                            help='Subject Timepoint',
                            default=None)

        parser.add_argument('--bids_pipeline_name',
                            type=str, help='BIDS PIPELINE Name',
                            default='wbic_dmri_pipeline')

        parser.add_argument('--nthreads',
                            type=int,
                            help='Number of Threads',
                            default=1)

        parser.add_argument('--gpu',
                            type=bool,
                            help='CUDA GPU Available',
                            default=False)

        parser.add_argument('--cuda_device',
                            type=int,
                            help='CUDA Device Number',
                            default=None)

        parser.add_argument('--anat_preproc_pipeline',
                            type=bool,
                            help='Preprocess the Anataomical Imaging Data',
                            default=False)

        parser.add_argument('--dwi_cleanup',
                            type=bool,
                            help='Clean up the Preprocessing Subdirectories',
                            default=False)

        parser.add_argument('--dwi_remove_last_vol',
                            type=bool,
                            help='Remove End DWI in 4d File',
                            default=False)

        parser.add_argument('--dwi_data_shelled',
                            type=bool,
                            help='Multiple Shell Diffusion Data',
                            default=False)

        parser.add_argument('--dwi_check_gradients',
                            type=bool,
                            help='Check DWI Gradient Directions',
                            default=False)

        parser.add_argument('--dwi_mask_method',
                            type=str,
                            help='Skull-stripping Algorithm',
                            choices=['bet', 'mrtrix', 'ants', 'antspynet'],
                            default='bet')

        parser.add_argument('--dwi_ants_mask_template',
                            type=str,
                            help='Image to use for registration based skull-stripping',
                            default=os.environ['FSLDIR']+'/data/standard/MNI152_T1_1mm.nii.gz')

        parser.add_argument('--dwi_ants_mask_template_mask',
                            type=str,
                            help='Brain mask to use for registration based skull-stripping',
                            default=os.environ['FSLDIR']+'/data/standard/MNI152_T1_1mm_brain_mask.nii.gz')

        parser.add_argument('--dwi_antspynet_modality',
                            type=str,
                            help='ANTsPyNet modality/network name',
                            default='t1')
                            
        parser.add_argument('--dwi_denoise_degibbs',
                            type=bool,
                            help='Perform Noise and Gibbs Ringing Correction',
                            default=True)

        parser.add_argument('--dwi_denoise_method',
                            type=str,
                            help='Method for Denoising DWIs',
                            choices=['mrtrix', 'dipy-nlmeans', 'dipy-localpca', 'dipy-mppca', 'dipy-patch2self'],
                            default='mrtrix')

        parser.add_argument('--dwi_gibbs_correction_method',
                            type=str,
                            help='Method for Gibbs Ringing Correction',
                            choices=['mrtrix', 'dipy'],
                            default='mrtrix')
                            
        parser.add_argument('--dwi_biasfield_correction',
                            type=bool,
                            help='Perform DWI Bias-Field Correction',
                            default=True)

        parser.add_argument('--dwi_biasfield_correction_method',
                            type=str,
                            help='Method for Gibbs Ringing Correction',
                            choices=['ants', 'fsl', 'N4'],
                            default='ants')

        parser.add_argument('--dwi_outlier_detection',
                            type=str,
                            help='Outlier Detection Method',
                            choices=[None, 'EDDY-QUAD', 'Threshold', 'Manual'],
                            default=None)

        parser.add_argument('--dwi_outlier_detection_threshold',
                            type=float,
                            help='Outlier Detection Method',
                            default=0.1)

        parser.add_argument('--dwi_dist_corr',
                            type=str,
                            help='Distortion Correction Flag',
                            choices=['Topup', 'Topup-Separate', 'Fieldmap', 'Anatomical-Coregistration'],
                            default=None)

        parser.add_argument('--dwi_distortion_linreg_method',
                            type=str,
                            help='Linear registration method to be used for registration based distortion correction',
                            choices=['FSL', 'ANTS'],
                            default='FSL')

        parser.add_argument('--dwi_topup_config',
                            type=str,
                            help='Configuration File for TOPUP',
                            default=None)

        parser.add_argument('--dwi_eddy_current_correction',
                            type=str,
                            help='Eddy current correction method',
                            choices=['eddy', 'eddy_correct', 'two-pass'],
                            default='eddy')

        parser.add_argument('--dwi_eddy_options',
                            type=str,
                            help='Additional eddy current correction options to pass to eddy',
                            default='')

        parser.add_argument('--dwi_slspec',
                            type=str,
                            help='Text file specifying slices/MB order in acquisition',
                            default=None)

        parser.add_argument('--repol',
                            type=bool,
                            help='EDDY Outlier Replacement',
                            choices=[0,1],
                            default=True)

        parser.add_argument('--estimate_move_by_suscept',
                            type=bool,
                            help='Correcting susceptibility-by-movement interactions with eddy',
                            default=False)

        parser.add_argument('--mporder',
                            type=int,
                            help='EDDY mporder',
                            default=0)

        parser.add_argument('--coregister_dwi_to_anat',
                            type = bool,
                            help = 'Coregister Diffusion MRI to Structural MRI',
                            default = False)
                            
        parser.add_argument('--coregister_dwi_to_anat_modality',
                            type=str,
                            help = 'Structural Image Modality to use for coregistration based distortion correction',
                            default = 't1w',
                            choices = ['t1w', 't2w'])

        parser.add_argument('--coregister_dwi_to_anat_method',
                            type = str,
                            help = 'Linear Registration for DWI to Anat',
                            default = 'linear')

        parser.add_argument('--coregister_dwi_to_anat_linear_method',
                            type = str,
                            help = 'Linear Registration for DWI to Anat',
                            default = 'FSL')

        parser.add_argument('--coregister_dwi_to_anat_nonlinear_method',
                            type = str,
                            help = 'Linear Registration for DWI to Anat',
                            default = 'ANTS')

        parser.add_argument('--dwi_resample_resolution',
                            type=int,
                            nargs='+',
                            help='Resampling Input Resolution',
                            default=None)

        parser.add_argument('--dti_fit_method',
                            type=str,
                            help='Fitting Algorithm for Diffusion Tensor Imaging Model',
                            choices=['dipy-OLS', 'dipy-WLS', 'dipy-NLLS', 'dipy-RESTORE', 'mrtrix', 'camino-RESTORE', 'camino-WLS', 'camino-NLLS', 'camino-OLS'],
                            default=None)

        parser.add_argument('--dti_bmax',
                            type=float,
                            help='Maximum B-value to use for DTI fitting',
                            default=None)

        parser.add_argument('--dti_full_output',
                            type=bool,
                            help='Output Additional DTI Parameters and Fit Residuals (more memory)',
                            default=False)

        parser.add_argument('--noddi_fit_method',
                            type=str,
                            help='Fitting Algorithm for Neurite Orietation Dispersion and Density Imaging Model',
                            choices=['amico', 'noddi-watson', 'noddi-bingham'],
                            default=None)

        parser.add_argument('--noddi_dpar',
                            type=float,
                            help='Parallel diffusivity value to use in the NODDI model fitting',
                            default=1.7e-9)

        parser.add_argument('--noddi_diso',
                            type=float,
                            help='Isotropic diffusivity value to use in the NODDI model fitting',
                            default=3e-9)

        parser.add_argument('--noddi_solver',
                            type=str,
                            help='DMIPY Optimization solver for NODDI model',
                            choices=['brute2fine', 'mix'],
                            default='brute2fine')

        parser.add_argument('--fwe_fit_method',
                            type=str,
                            help='Fitting Algorithm for Diffusion Tensor Imaging Model',
                            choices=['WLS', 'NLS'],
                            default=None)

        parser.add_argument('--dki_fit_method',
                            type=str,
                            help='Fitting Algorithm for Diffusion Kurtosis Imaging Model',
                            choices=['dipy-OLS', 'dipy-WLS'],
                            default=None)

        parser.add_argument('--csd_response_func_algo',
                            type=str,
                            help='Response Function Estimation Algorithm',
                            choices=['tournier', 'dhollander', 'tax', 'fa', 'manual', 'msmt_5tt'],
                            default='tournier')

        parser.add_argument('--csd_fod_algo',
                            type=str,
                            help='Fiber Orientation Dispersion Estimation Algorithm',
                            choices=['csd', 'msmt_csd'],
                            default=None)
                            
        parser.add_argument('--csd_response_function',
                            type=str,
                            help='CSD Response Function file',
                            default=None)
                    
        parser.add_argument('--csd_wm_response_function',
                            type=str,
                            help='CSD White Matter Response Function file',
                            default=None)
        
        parser.add_argument('--csd_gm_response_function',
                            type=str,
                            help='CSD Gray Matter Response Function file',
                            default=None)
                            
        parser.add_argument('--csd_csf_response_function',
                            type=str,
                            help='CSD CSF Response Function file',
                            default=None)

        parser.add_argument('--micro_dki',
                            type=bool,
                            help='Perform Microscopic Kurtosis modeling',
                            default=False)

        parser.add_argument('--setup_gbss',
                            type=bool,
                            help='Perform Initial steps for GBSS processing',
                            default=False)

        parser.add_argument('--verbose',
                            type=bool,
                            help='Print out information meassages and progress status',
                            default=False)
        
        args, unknown = parser.parse_known_args()
        
        if args.load_json:
            with open(args.load_json, 'rt') as f:
                t_args = argparse.Namespace()
                t_dict = vars(t_args)
                t_dict.update(json.load(f))
                args, unknown = parser.parse_known_args(namespace=t_args)


        #Setup the BIDS Directories and Paths
        entities = {
        'extension': '.nii.gz',
        'subject': args.subject,
        'session': args.session,
        'modality': 'dwi',
        'suffix': 'dwi'
        }

        id_patterns = 'sub-{subject}[_ses-{session}]'
        rawdata_patterns = args.bids_dir + '/'+ args.bids_rawdata_dir + '/sub-{subject}[/ses-{session}]/'
        derivative_patterns = args.bids_dir + '/derivatives/' + args.bids_pipeline_name + '/sub-{subject}[/ses-{session}]/'

        bids_id             = writing.build_path(entities, id_patterns)
        bids_rawdata_dir    = writing.build_path(entities, rawdata_patterns)
        bids_derivative_dir = writing.build_path(entities, derivative_patterns)

        preproc_dir = os.path.join(bids_derivative_dir, args.bids_dwi_dir, 'preprocessed/')

        #Create final processed DWI dataset
        final_base = os.path.join(preproc_dir, bids_id+'_desc-preproc')
        final_dwi = DWImage(file        = final_base + '_dwi.nii.gz',
                            bvecs       = final_base + '_dwi.bvec',
                            bvals       = final_base + '_dwi.bval',
                            acqparams   = final_base + '-Acqparams_dwi.txt',
                            index       = final_base + '-Index_dwi.txt',
                            slspec      = final_base + '-Slspec_dwi.txt')

        dwi_mask = Image(file=os.path.join(preproc_dir, bids_id+'_desc-brain_mask.nii.gz'))

        t1w=None
        t2w=None
        anat_mask=None

        #Setup the Anatomical Imaging Data if needed
        if (args.dwi_dist_corr == 'Anatomical-Coregistration' or args.coregister_dwi_to_anat):
            anat_pipeline = AnatomicalPrepPipeline()
            t1w, t2w, anat_mask = anat_pipeline.run()


        ##################################
        ### DWI PROCESSING STARTS HERE ###
        ##################################

        if not final_dwi.exists():

            #Setup the raw data and perform some basic checks on the data and associated files
            dwi_img, topup_base =  raw_proc.prep_dwi_rawdata(bids_id                = bids_id,
                                                             bids_rawdata_dir       = bids_rawdata_dir,
                                                             bids_derivative_dir    = bids_derivative_dir,
                                                             bids_dwi_dir           = args.bids_dwi_dir,
                                                             nthreads               = args.nthreads,
                                                             resample_resolution    = args.dwi_resample_resolution,
                                                             remove_last_vol        = args.dwi_remove_last_vol,
                                                             topup_config           = args.dwi_topup_config,
                                                             outlier_detection      = args.dwi_outlier_detection,
                                                             check_gradients        = args.dwi_check_gradients,
                                                             verbose                = args.verbose)

            if args.dwi_denoise_degibbs:
                dwi_img = img_proc.denoise_degibbs(img             = dwi_img,
                                                   working_dir     = os.path.join(preproc_dir, 'denoise-degibbs/'),
                                                   suffix          = 'dwi',
                                                   denoise_method  = args.dwi_denoise_method,
                                                   gibbs_method    = args.dwi_gibbs_correction_method,
                                                   nthreads        = args.nthreads,
                                                   verbose         = args.verbose)


            dwi_img = eddy_proc.perform_eddy(dwi_image                  = dwi_img,
                                             working_dir                = os.path.join(preproc_dir, 'eddy-correction/'),
                                             topup_base                 = topup_base,
                                             method                     = args.dwi_eddy_current_correction,
                                             gpu                        = args.gpu,
                                             cuda_device                = args.cuda_device,
                                             nthreads                   = args.nthreads,
                                             data_shelled               = args.dwi_data_shelled,
                                             repol                      = args.repol,
                                             estimate_move_by_suscept   = args.estimate_move_by_suscept,
                                             mporder                    = args.mporder,
                                             slspec                     = args.dwi_slspec,
                                             fsl_eddy_options           = args.dwi_eddy_options,
                                             verbose                    = args.verbose)

            if args.dwi_outlier_detection != 'Manual':
                dwi_img = eddy_proc.perform_outlier_detection(dwi_image         = dwi_img,
                                                              working_dir       = os.path.join(preproc_dir, 'outlier-removed-images/'),
                                                              method            = args.dwi_outlier_detection,
                                                              percent_threshold = args.dwi_outlier_detection_threshold,
                                                              verbose           = args.verbose)


            if args.dwi_dist_corr == 'Anatomical-Coregistration':
                dwi_img = distort_proc.perform_distortion_correction(dwi_image           = dwi_img,
                                                                     working_dir         = preproc_dir,
                                                                     t1w_image           = t1w,
                                                                     t2w_image           = t2w,
                                                                     distortion_method   = args.dwi_dist_corr,
                                                                     distortion_modality = args.coregister_dwi_to_anat_modality,
                                                                     linreg_method       = args.dwi_distortion_linreg_method,
                                                                     nthreads            = args.nthreads,
                                                                     verbose             = args.verbose)

            ###BIAS FIELD CORRECTION ###
            if args.dwi_biasfield_correction:
                dwi_img = img_proc.perform_bias_correction(img         = dwi_img,
                                                           working_dir = os.path.join(preproc_dir, 'biasfield-correction/'),
                                                           suffix      = 'dwi',
                                                           method      = args.dwi_biasfield_correction_method,
                                                           nthreads    = args.nthreads,
                                                           verbose     = args.verbose)


            if args.coregister_dwi_to_anat:
                dwi_img = coreg_proc.register_to_anat(dwi_image           = dwi_img,
                                                      working_dir         = os.path.join(bids_derivative_dir, args.bids_dwi_dir, 'preprocessed/'),
                                                      coreg_to_anat       = args.coregister_dwi_to_anat,
                                                      T1_image            = t1w,
                                                      T2_image            = t2w,
                                                      anat_mask           = anat_mask,
                                                      reg_method          = args.coregister_dwi_to_anat_method,
                                                      linreg_method       = args.coregister_dwi_to_anat_linear_method,
                                                      nonlinreg_method    = args.coregister_dwi_to_anat_nonlinear_method,
                                                      dof                 = 6,
                                                      nthreads            = args.nthreads,
                                                      verbose             = args.verbose)

                if args.verbose:
                    print('Copying Anatomical Mask')
                    shutil.copy2(anat_mask._get_filename(), dwi_mask._get_filename())
            else:

                if args.verbose:
                    print('Creating DWI Brain Mask')

                mask.mask_image(input_img            = dwi_img,
                                output_mask          = dwi_mask,
                                method               = args.dwi_mask_method,
                                nthreads             = args.nthreads,
                                ref_img              = args.dwi_ants_mask_template,
                                ref_mask             = args.dwi_ants_mask_template_mask,
                                antspynet_modality   = args.dwi_antspynet_modality)

            if not final_dwi.exists():
                if args.verbose:
                    print('Creating Preprocessed DWI')

                final_dwi.copy_image(dwi_img, datatype=np.float32)



        if args.dwi_cleanup:
            if args.verbose:
                print('Cleaning up DWI Preprocessing Files')

            dirs_to_cleanup = []
            dirs_to_cleanup.append('rawdata')
            dirs_to_cleanup.append('anatomical-distortion-correction')
            dirs_to_cleanup.append('fieldmap-distortion-correction')
            dirs_to_cleanup.append('biasfield-correction')
            dirs_to_cleanup.append('denoise-degibbs')
            dirs_to_cleanup.append('eddy-correction')
            dirs_to_cleanup.append('topup')
            dirs_to_cleanup.append('coregister-to-anatomy')

            files_to_cleanup = []
            files_to_cleanup.append(bids_id + '_desc-Acqparams_dwi.txt')
            files_to_cleanup.append(bids_id + '_desc-Slspec_dwi.txt')
            files_to_cleanup.append(bids_id + '_desc-Index_dwi.txt')

            outlier_files_to_cleanup = []
            outlier_files_to_cleanup.append(bids_id + '_desc-OutlierRemoved_dwi.bval')
            outlier_files_to_cleanup.append(bids_id + '_desc-OutlierRemoved_dwi.bvec')
            outlier_files_to_cleanup.append(bids_id + '_desc-OutlierRemoved_dwi.nii.gz')
            outlier_files_to_cleanup.append(bids_id + '_desc-OutlierRemoved-Index_dwi.txt')

            for dir in dirs_to_cleanup:
                if os.path.exists(os.path.join(bids_derivative_dir, args.bids_dwi_dir, 'preprocessed/', dir)):
                    shutil.rmtree(os.path.join(bids_derivative_dir, args.bids_dwi_dir, 'preprocessed/', dir))

            for file in files_to_cleanup:
                if os.path.exists(os.path.join(bids_derivative_dir, args.bids_dwi_dir, 'preprocessed/', file)):
                    os.remove(os.path.join(bids_derivative_dir, args.bids_dwi_dir, 'preprocessed/', file))

            for file in outlier_files_to_cleanup:
                if os.path.exists(os.path.join(bids_derivative_dir, args.bids_dwi_dir, 'preprocessed/outlier-removed-images/', file)):
                    os.remove(os.path.join(bids_derivative_dir, args.bids_dwi_dir, 'preprocessed/outlier-removed-images/', file))


        ############### PREPROCESSING OF DWI DATA FINISHED ####################

        models_dir = os.path.join(bids_derivative_dir, args.bids_dwi_dir, 'models/')

        ###DTI MODELING ###
        if args.dti_fit_method != None:
            if not os.path.exists(models_dir + 'DTI/' + bids_id + '_model-DTI_parameter-FA.nii.gz'):
                if args.verbose:
                    print('Fitting DTI model with ' + args.dti_fit_method + '...')

                dti_model = DTI_Model(dwi_img       = final_dwi,
                                      out_base      = models_dir + 'DTI/' + bids_id,
                                      fit_type      = args.dti_fit_method,
                                      mask          = dwi_mask,
                                      bmax          = args.dti_bmax,
                                      full_output   = args.dti_full_output)
                dti_model.fit()

        ####FWE MODELING ###
        if args.fwe_fit_method != None:
            if not os.path.exists( models_dir + 'FWE-DTI/' + bids_id + '_model-FWE-DTI_parameter-F.nii.gz' ):
                if args.verbose:
                    print('Fitting Free-Water Elimination DTI Model')

                fwedti_model = FWEDTI_Model(dwi_img   = final_dwi,
                                            out_base  = models_dir + 'FWE-DTI/' + bids_id,
                                            fit_type  = args.fwe_fit_method,
                                            mask      = dwi_mask)
                fwedti_model.fit()



        if args.noddi_fit_method != None:
            if not os.path.exists( models_dir + args.noddi_fit_method+'/' + bids_id + '_model-NODDI_parameter-ICVF.nii.gz'):
                if args.verbose:
                    print('Fitting '+args.noddi_fit_method+' model...')

                noddi_model = NODDI_Model(dwi_img               = final_dwi,
                                          out_base              = models_dir + args.noddi_fit_method+'/' + bids_id,
                                          fit_type              = args.noddi_fit_method,
                                          mask                  = dwi_mask,
                                          parallel_diffusivity  = args.noddi_dpar,
                                          iso_diffusivity       = args.noddi_diso,
                                          solver                = args.noddi_solver,
                                          nthreads              = args.nthreads,
                                          verbose               = args.verbose)
                noddi_model.fit()


        if args.dki_fit_method != None:
            if not os.path.exists( models_dir + 'DKI/' + bids_id + '_model-DKI_parameter-FA.nii.gz' ):
                if args.verbose:
                    print('Fitting Diffusion Kurtosis Model')

                dki_model = DKI_Model(dwi_img      = final_dwi,
                                         out_base  = models_dir + 'DKI/' + bids_id,
                                         fit_type  = args.dki_fit_method,
                                         mask      = dwi_mask)
                dki_model.fit()


        if args.csd_fod_algo != None:
            if not os.path.exists( models_dir + 'CSD/' + bids_id + '_model-CSD_parameter-FOD.nii.gz' ) and not os.path.exists( models_dir + 'CSD/' + bids_id + '_model-MSMT-5tt_parameter-WMfod.nii.gz' ):
                if args.verbose:
                    print('Fitting Constrained Spherical Deconvolution Model')

                csd_model = CSD_Model(dwi_img       = final_dwi,
                                      out_base      = models_dir + 'CSD/' + bids_id,
                                      response_algo = args.csd_response_func_algo,
                                      fod_algo      = args.csd_fod_algo,
                                      mask          = dwi_mask,
                                      struct_img    = t1w,
                                      response      = args.csd_response_function,
                                      wm_response   = args.csd_wm_response_function,
                                      gm_response   = args.csd_gm_response_function,
                                      csf_response  = args.csd_csf_response_function,
                                      nthreads      = args.nthreads)
                csd_model.fit()





## ###GBSS PSEUDO T1w ###
# if args.setup_gbss:
#     if not os.path.exists(bids_derivative_dwi_dir + '/GBSS/' + bids_id + '_desc-GBSS-Pseudo-T1w.nii.gz'):
#         if args.verbose:
#             print('Creating GBSS Pseudo T1-weighted Image')
#
#         if os.path.exists(bids_derivative_dwi_dir +'/DTI/' + bids_id + '_model-DTI_parameter-FA.nii.gz') and os.path.exists(bids_derivative_dwi_dir +'/NODDI-'+args.noddi_fit_method+'/' + bids_id + '_model-NODDI_parameter-ISO.nii.gz'):
#
#             diff_util.create_pseudoT1_img(fa_img        = bids_derivative_dwi_dir +'/DTI/' + bids_id + '_model-DTI_parameter-FA.nii.gz',
#                                           fiso_img      = bids_derivative_dwi_dir +'/NODDI-'+args.noddi_fit_method+'/' + bids_id + '_model-NODDI_parameter-ISO.nii.gz',
#                                           mask_img      = preprocess_dir + bids_id + '_desc-brain_mask.nii.gz',
#                                           pseudoT1_img  = bids_derivative_dwi_dir + '/GBSS/' + bids_id + '_desc-GBSS-Pseudo-T1w.nii.gz')
