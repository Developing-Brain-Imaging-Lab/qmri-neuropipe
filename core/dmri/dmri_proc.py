import os, shutil, json, argparse
import numpy as np

from bids.layout import writing
from core.utils.io import Image, DWImage

import core.dmri.workflows.prep_rawdata as dmri_rawprep
import core.utils.workflows.denoise_degibbs as img_proc

import core.dmri.workflows.eddy_corr as eddy_proc
import core.dmri.workflows.distort_corr as distort_proc
import core.dmri.utils.distortion_correction as distcorr
import core.dmri.workflows.register_to_anat as coreg_proc
import core.dmri.workflows.prep_grad_nonlin as grad_non_lin_prep


from core.anat.anat_proc import AnatomicalPrepPipeline

import core.dmri.utils.qc as dmri_qc
import core.utils.mask as mask

from core.dmri.models.dti import DTI_Model, FWEDTI_Model
from core.dmri.models.dki import DKI_Model
from core.dmri.models.csd import CSD_Model
from core.dmri.models.noddi import NODDI_Model, SMT_NODDI_Model

from core.dmri.workflows.dmri_to_standard import dmri_to_standard

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
        
        parser.add_argument('--use_freesurfer',
                            type=bool,
                            help='Use FreeSurfer processed data',
                            default=False)
        
        parser.add_argument('--freesurfer_subjects_dir',
                            type=str,
                            help="Freesurfer Subjects Directory",
                            default=None)

        parser.add_argument('--proc_json',
                            type=str, help='Processing json with pipeline options. Command line options are overriden by values in file.',
                            default=None)

        parser.add_argument('--subject',
                            type=str,
                            help='Subject ID')

        parser.add_argument('--session',
                            type=str,
                            help='Subject Timepoint',
                            default=None)

        parser.add_argument('--preproc_derivative_dir',
                            type=str, help='Preprocessing Derivative Output',
                            default='dmri-neuropipe-preproc')
        
        parser.add_argument('--models_derivative_dir',
                            type=str, help='BIDS PIPELINE Name',
                            default='dmri-neuropipe-models')

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

        parser.add_argument('--cleanup',
                            type=bool,
                            help='Clean up the Preprocessing Subdirectories',
                            default=False)

        parser.add_argument('--remove_last_vol',
                            type=bool,
                            help='Remove End DWI in 4d File',
                            default=False)

        parser.add_argument('--check_gradients',
                            type=bool,
                            help='Check DWI Gradient Directions',
                            default=False)
        
        parser.add_argument('--reorient',
                            type=bool,
                            help='Reorient the diffusion images',
                            default=False)
        
        parser.add_argument('--reorient_template',
                            type=str,
                            help='Template to use to reorient diffusion images',
                            default=None)

        parser.add_argument('--mask_method',
                            type=str,
                            help='Skull-stripping Algorithm',
                            choices=['bet', 'hd-bet', 'mrtrix', 'ants', 'antspynet'],
                            default='bet')

        parser.add_argument('--ants_mask_template',
                            type=str,
                            help='Image to use for registration based skull-stripping',
                            default=os.environ.get("FSLDIR")+'/data/standard/MNI152_T1_1mm.nii.gz')

        parser.add_argument('--ants_mask_template_mask',
                            type=str,
                            help='Brain mask to use for registration based skull-stripping',
                            default=os.environ.get("FSLDIR")+'/data/standard/MNI152_T1_1mm_brain_mask.nii.gz')

        parser.add_argument('--antspynet_modality',
                            type=str,
                            help='ANTsPyNet modality/network name',
                            default='t1')
                            
        parser.add_argument('--denoise_degibbs',
                            type=bool,
                            help='Perform Noise and Gibbs Ringing Correction',
                            default=True)

        parser.add_argument('--denoise_method',
                            type=str,
                            help='Method for Denoising DWIs',
                            choices=['mrtrix', 'dipy-nlmeans', 'dipy-localpca', 'dipy-mppca', 'dipy-patch2self'],
                            default='mrtrix')

        parser.add_argument('--gibbs_correction_method',
                            type=str,
                            help='Method for Gibbs Ringing Correction',
                            choices=['mrtrix', 'dipy'],
                            default='mrtrix')
                            
        parser.add_argument('--biasfield_correction',
                            type=bool,
                            help='Perform DWI Bias-Field Correction',
                            default=True)

        parser.add_argument('--biasfield_correction_method',
                            type=str,
                            help='Method for Gibbs Ringing Correction',
                            choices=["mrtrix-ants", "mrtrix-fsl", 'ants', 'fsl', 'N4'],
                            default='ants')

        parser.add_argument('--outlier_detection',
                            type=str,
                            help='Outlier Detection Method',
                            choices=[None, 'EDDY-QUAD', 'Threshold', 'Manual'],
                            default=None)

        parser.add_argument('--outlier_detection_threshold',
                            type=float,
                            help='Outlier Detection Method',
                            default=0.1)

        parser.add_argument('--dist_correction',
                            type=str,
                            help='Distortion Correction Flag',
                            choices=['Topup', 'Topup-Separate','Synb0-Disco', 'Fieldmap', 'Anatomical-Coregistration'],
                            default=None)

        parser.add_argument('--distortion_linreg_method',
                            type=str,
                            help='Linear registration method to be used for registration based distortion correction',
                            choices=['fsl', 'ants'],
                            default='fsl')

        parser.add_argument('--topup_config',
                            type=str,
                            help='Configuration File for TOPUP',
                            default=None)

        parser.add_argument('--eddy_current_correction',
                            type=str,
                            help='Eddy current correction method',
                            choices=['eddy', 'eddy_correct', 'two-pass', 'tortoise-diffprep'],
                            default='eddy')

        parser.add_argument('--fsl_eddy_options',
                            type=str,
                            help='Additional eddy current correction options to pass to eddy',
                            default='')
        
        parser.add_argument('--tortoise_diffprep_options',
                            type=str,
                            help='Additional eddy current correction options to pass to TORTOISE DIFFPREP',
                            default='')
        
        parser.add_argument('--gradnonlin_correction',
                            type=bool,
                            help='Do gradient non-linearity correction',
                            default=False)

        parser.add_argument('--gw_coils_dat',
                            type=str,
                            help='Path to scanner spherical harmonics coefficients file gw_coils.dat',
                            default=None)

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
                            default = 'fsl')

        parser.add_argument('--coregister_dwi_to_anat_nonlinear_method',
                            type = str,
                            help = 'Linear Registration for DWI to Anat',
                            default = 'ants')
        
        parser.add_argument('--noresample_dwi_to_anat',
                            type = bool,
                            help = 'Apply Only to Header Linear xform Diffusion MRI to Structural MRI',
                            default = False)

        parser.add_argument('--resample_resolution',
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
                            choices=['amico', 'noddi-watson', 'noddi-bingham', 'smt'],
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
        
        parser.add_argument('--dki_smooth_input',
                            type=bool,
                            help='Smooth input DWI data prior to DKI fitting',
                            default=True)
    
        parser.add_argument('--dki_smooth_fwhm',
                            type=float,
                            help='FWHM to smooth input DWI data prior to DKI fitting',
                            default=1.25)

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
                            
        parser.add_argument('--micro_dki',
                            type=bool,
                            help='Perform Microscopic Kurtosis modeling',
                            default=False)
        
        parser.add_argument('--dwi_to_standard',
                            type=bool,
                            help="Perform registration to standard space",
                            default=False)
        
        parser.add_argument('--standard_template_method',
                            type=str,
                            help="Standard template file",
                            choices=['fsl', 'ants'],
                            default='ants')
        
        parser.add_argument('--standard_template',
                            type=str,
                            help="Standard template file",
                            default=None)
        
        parser.add_argument('--standard_template_mask',
                            type=str,
                            help="Standard template file",
                            default=None)

        parser.add_argument('--setup_gbss',
                            type=bool,
                            help='Perform Initial steps for GBSS processing',
                            default=False)

        parser.add_argument('--verbose',
                            type=bool,
                            help='Print out information meassages and progress status',
                            default=False)
        
        parser.add_argument('--debug',
                    type=bool,
                    help='Print out debugging messages',
                    default=False)
        
        
        args, unknown = parser.parse_known_args()
        
        if args.proc_json:
            with open(args.proc_json, 'rt') as f:
                t_args = argparse.Namespace()
                t_dict = vars(t_args)
                test_json = json.load(f)
                t_dict.update(test_json)
                t_dict.update(test_json["dwi"])
                args, unknown = parser.parse_known_args(namespace=t_args)


        #Setup the BIDS Directories and Paths
        entities = {
        'extension': '.nii.gz',
        'subject': args.subject,
        'session': args.session
        }
         
        id_patterns          = "sub-{subject}[_ses-{session}]"
        rawdata_patterns     = os.path.join(args.bids_dir, args.bids_rawdata_dir, "sub-{subject}[/ses-{session}]",)
        output_patterns      = os.path.join(args.bids_dir, "derivatives", args.preproc_derivative_dir, "sub-{subject}[/ses-{session}]",)
        
        id              = writing.build_path(entities, id_patterns)
        rawdata_dir     = writing.build_path(entities, rawdata_patterns)
        output_dir      = writing.build_path(entities, output_patterns)
        
        anat_preproc_dir = os.path.join(output_dir, "anat/")
        dmri_preproc_dir = os.path.join(output_dir, "dwi/")
        
        dmri_img_pattern = os.path.join(dmri_preproc_dir, "sub-{subject}[_ses-{session}][_acq-{acq}][_dir-{dir}][_rec-{rec}][_desc-{desc}]_{modality}.nii.gz")
        txt_pattern      = os.path.join(dmri_preproc_dir, "sub-{subject}[_ses-{session}][_acq-{acq}][_dir-{dir}][_rec-{rec}][_desc-{desc}]_{modality}.txt")
        
        dmri_ent = entities.copy()
        dmri_ent['modality'] = 'dwi'
        dmri_ent['desc'] = 'preproc'
        
        acqp_ent   = dmri_ent.copy()
        index_ent  = dmri_ent.copy()
        slspec_ent = dmri_ent.copy()
        acqp_ent['desc']   = 'preproc-acqparams'
        index_ent['desc']  = 'preproc-index'
        slspec_ent['desc'] = 'preproc-slspec'
            
        #Final processed DWI dataset
        dmri_preproc = DWImage(filename    = writing.build_path(dmri_ent, dmri_img_pattern),
                               bvecs       = writing.build_path(dmri_ent, dmri_img_pattern.replace('.nii.gz', '.bvec')),
                               bvals       = writing.build_path(dmri_ent, dmri_img_pattern.replace('.nii.gz', '.bval')),
                               acqparams   = writing.build_path(acqp_ent, txt_pattern),
                               index       = writing.build_path(index_ent, txt_pattern),
                               slspec      = writing.build_path(slspec_ent, txt_pattern),
                               json        = writing.build_path(dmri_ent, dmri_img_pattern.replace('.nii.gz', '.json')))

        dmri_mask    = Image(filename = os.path.join(dmri_preproc_dir, id+'_desc-brain_mask.nii.gz'),
                             json     = os.path.join(dmri_preproc_dir, id+'_desc-brain_mask.json'))

        t1w=None
        t2w=None
        brain_mask=None
        
        anat_img  = None
        anat_mask = None
        
        fmap_image=None
        fmap_ref_image=None
        gradnonlin_image=None

        topup_base=None
        
        #Setup anatomical imaging data if its neded first
        #
        #OPTIONS:
        #   1. Use FreeSurfer processed data
        #   2. Use Anatomical Preprocessing Pipeline Output
        #
        #
        
        if args.dist_correction:
            if str.lower(args.dist_correction) == 'fieldmap':
                fmap_image      = Image(filename = os.path.join(rawdata_dir, 'fmap', id+'_fieldmap.nii.gz'))
                fmap_ref_image  = Image(filename = os.path.join(rawdata_dir, 'fmap', id+'_magnitude.nii.gz'))
        
        
        freesurfer_subjs_dir = None
        if args.use_freesurfer or args.coregister_dwi_to_anat or args.dist_correction == 'synb0' or args.dist_correction == 'anatomical-coregistration' or args.eddy_current_correction == 'tortoise-diffprep':
            
            coreg_dir = os.path.join(dmri_preproc_dir, 'coregister-to-anatomy',)
            if not os.path.exists(coreg_dir):
                os.makedirs(coreg_dir)
           
            if args.use_freesurfer:
                if args.freesurfer_subjects_dir:
                    freesurfer_subjs_dir = args.freesurfer_subjects_dir
                elif os.path.exists(os.path.join(args.bids_dir,'derivatives', 'freesurfer',)):
                    freesurfer_subjs_dir = os.path.join(args.bids_dir,'derivatives','freesurfer',)
                else:
                    print("Freesurfer Directory doesn't exist or was not specified")
                    exit(-1)
                    
                freesurfer_t1w  = os.path.join(freesurfer_subjs_dir, id, "mri", "orig_nu.mgz")
                freesurfer_mask = os.path.join(freesurfer_subjs_dir, id, "mri", "brainmask.mgz")
                
                anat_img  = Image(filename = os.path.join(coreg_dir, id+'_desc-Freesurfer_T1w.nii.gz'))
                anat_mask = Image(filename = os.path.join(coreg_dir, id+'_desc-Freesurfer_T1w_brain_mask.nii.gz'))

                #Convert to NIFTI
                os.system('mri_convert --in_type mgz --out_type nii -i ' + freesurfer_t1w + ' -o ' + anat_img.filename)
                os.system('mri_convert --in_type mgz --out_type nii -i ' + freesurfer_mask + ' -o ' + anat_mask.filename)
            
            else:
                anat_pipeline = AnatomicalPrepPipeline()
                t1w, t2w, anat_mask = anat_pipeline.run(proc_dir = args.preproc_derivative_dir)
                                
                if args.eddy_current_correction == 'tortoise-diffprep':
                    if t2w:
                        anat_img = t2w
                    elif t1w:
                        #If we have a T1w image only, create a synthetic T2w using T1w
                        from core.anat.workflows.compute_synthetic_t2w import compute_synthetic_t2w
                        if args.verbose or args.debug:
                            print('Creating Synthetic T2w Image')

                        anat_img   = compute_synthetic_t2w(input_t1w    = t1w,
                                                           output_dir   = coreg_dir,
                                                           cmd_args     = args,
                                                           syn_t2w      = id+"_desc-SyntheticFromT1w_T2w.nii.gz", 
                                                           t1w_mask     = brain_mask, 
                                                           debug        = args.debug)
                           
                elif args.coregister_dwi_to_anat_modality == 't1w' and t1w:
                    anat_img = t1w
                elif args.coregister_dwi_to_anat_modality == 't2w':
                    if t2w:
                        anat_img = t2w
                    elif t1w:
                        from core.anat.workflows.compute_synthetic_t2w import compute_synthetic_t2w
                        if args.verbose or args.debug:
                            print('Creating Synthetic T2w Image')

                        anat_img = compute_synthetic_t2w(input_t1w    = t1w,
                                                         output_dir   = coreg_dir,
                                                         cmd_args     = args,
                                                         syn_t2w      = id+"_desc-SyntheticFromT1w_T2w.nii.gz", 
                                                         t1w_mask     = anat_mask, 
                                                         debug        = args.debug)
                else:
                    print('No anatomical image!')
                    exit()
      

        ##################################
        ### DWI PROCESSING STARTS HERE ###
        ##################################
        if not dmri_preproc.exists():

            #Setup the raw data and perform some basic checks on the data and associated files
            dwi_img =  dmri_rawprep.prep_rawdata(bids_dir               = args.bids_dir, 
                                                 preproc_dir            = dmri_preproc_dir,
                                                 id                     = args.subject,
                                                 session                = args.session, 
                                                 bids_filter            = args.proc_json,
                                                 check_gradients        = args.check_gradients, 
                                                 reorient_dwi           = args.reorient,     
                                                 dwi_reorient_template  = Image(filename=args.reorient_template), 
                                                 resample_resolution    = args.resample_resolution, 
                                                 remove_last_vol        = args.remove_last_vol,
                                                 outlier_detection      = args.outlier_detection, 
                                                 nthreads               = args.nthreads,
                                                 verbose                = args.verbose) 
            
            #Calculate Topup/SynB0-DISCO field maps
            if args.dist_correction.lower()[0:5] == 'topup' or args.dist_correction.lower() == 'synb0-disco':
                topup_base = os.path.join(dmri_preproc_dir, "rawdata", "topup", id+"_desc-Topup")
            
                if not os.path.exists(f"{topup_base}_fieldcoef.nii.gz"):
                    #First going to run eddy_correct in order to perform an initial motion-correction to ensure images are aligned prior to estimating fields. Data are only used
                    #here and not for subsequent processing
                    eddy_img = eddy_proc.perform_eddy(dwi_image         = dwi_img,
                                                      working_dir       = os.path.join(dmri_preproc_dir, "rawdata", "tmp-eddy-correction/"),
                                                      method            ='eddy',
                                                      gpu               = args.gpu,
                                                      cuda_device       = args.cuda_device,
                                                      nthreads          = args.nthreads,
                                                      fsl_eddy_options  = " --data_is_shelled",
                                                      verbose           = args.verbose) 
                                                      
                    if args.dist_correction.lower()[0:5] == 'topup' :
                        distort_proc.perform_topup(dwi_image    = eddy_img,
                                                   topup_base   = topup_base,
                                                   topup_config = args.topup_config,
                                                   dist_corr    = 'Topup',
                                                   verbose      = args.verbose)

                    if args.dist_correction.lower() == 'synb0-disco':
                        #Run the Synb0 distortion correction'
                        distcorr.run_synb0_disco(dwi_img        = eddy_img,
                                                 t1w_img        = anat_img,
                                                 topup_base     = topup_base,
                                                 topup_config   = args.topup_config,
                                                 nthreads       = args.nthreads)
                


            if args.denoise_degibbs:
                dwi_img = img_proc.denoise_degibbs(input_img       = dwi_img,
                                                   working_dir     = os.path.join(dmri_preproc_dir, 'denoise-degibbs',),
                                                   suffix          = 'dwi',
                                                   denoise_method  = args.denoise_method,
                                                   gibbs_method    = args.gibbs_correction_method,
                                                   nthreads        = args.nthreads,
                                                   verbose         = args.verbose)
                
            dwi_img = eddy_proc.perform_eddy(dwi_image                  = dwi_img,
                                             working_dir                = os.path.join(dmri_preproc_dir, 'eddy-correction',),
                                             topup_base                 = topup_base,
                                             method                     = args.eddy_current_correction,
                                             gpu                        = args.gpu,
                                             cuda_device                = args.cuda_device,
                                             nthreads                   = args.nthreads,
                                             fsl_eddy_options           = args.fsl_eddy_options,
                                             tortoise_options           = args.tortoise_diffprep_options,
                                             struct_img                 = anat_img,
                                             verbose                    = args.verbose)                 

            if args.outlier_detection != None and args.outlier_detection != 'Manual':
                dwi_img = eddy_proc.perform_outlier_detection(dwi_image         = dwi_img,
                                                              working_dir       = os.path.join(dmri_preproc_dir, 'outlier-removed-images',),
                                                              method            = args.outlier_detection,
                                                              percent_threshold = args.outlier_detection_threshold,
                                                              verbose           = args.verbose)
 

            if args.dist_correction == 'Anatomical-Coregistration' or args.dist_correction == 'Fieldmap':
                dwi_img = distort_proc.perform_distortion_correction(dwi_image           = dwi_img,
                                                                     working_dir         = dmri_preproc_dir,
                                                                     t1w_image           = t1w,
                                                                     t2w_image           = t2w,
                                                                     fmap_image          = fmap_image,
                                                                     fmap_ref_image      = fmap_ref_image,
                                                                     distortion_method   = args.dist_correction,
                                                                     distortion_modality = args.coregister_dwi_to_anat_modality,
                                                                     linreg_method       = args.distortion_linreg_method,
                                                                     nthreads            = args.nthreads,
                                                                     verbose             = args.verbose)

            ###BIAS FIELD CORRECTION ###
            if args.biasfield_correction:
                dwi_img = img_proc.perform_biasfield_correction(input_img   = dwi_img,
                                                                working_dir = os.path.join(dmri_preproc_dir, 'biasfield-correction',),
                                                                suffix      = 'dwi',
                                                                method      = args.biasfield_correction_method,
                                                                nthreads    = args.nthreads,
                                                                verbose     = args.verbose)

            if args.coregister_dwi_to_anat:
                dwi_img = coreg_proc.register_to_anat(dwi_image            = dwi_img,
                                                      working_dir          = dmri_preproc_dir,
                                                      anat_image           = anat_img,
                                                      anat_mask            = anat_mask,
                                                      mask_method          = args.mask_method,
                                                      reg_method           = args.coregister_dwi_to_anat_method,
                                                      linreg_method        = args.coregister_dwi_to_anat_linear_method,
                                                      anat_modality        = args.coregister_dwi_to_anat_modality,
                                                      freesurfer_subjs_dir = freesurfer_subjs_dir,
                                                      noresample           = args.noresample_dwi_to_anat,
                                                      nthreads             = args.nthreads,
                                                      verbose              = args.verbose)

            
            #Create brain mask
            if args.coregister_dwi_to_anat and not args.noresample_dwi_to_anat:
                if args.verbose:
                    print('Copying Anatomical Mask')

                shutil.copy2(anat_mask.filename, dmri_mask.filename)
            
            else:
                if args.verbose:
                    print('Creating DWI Brain Mask')

                mask.mask_image(input                = dwi_img,
                                mask                 = dmri_mask,
                                algo                 = args.mask_method,
                                nthreads             = args.nthreads,
                                ref_img              = args.ants_mask_template,
                                ref_mask             = args.ants_mask_template_mask,
                                antspynet_modality   = args.antspynet_modality)
              
            #Create the preprocessed DWI file
            if args.verbose:
                print('Creating Preprocessed DWI')

            dmri_preproc.copy_image(dwi_img, datatype=np.float32)
            dmri_qc.check_gradient_directions(input_dwi   = dmri_preproc,
                                              nthreads    = args.nthreads)
                
            if args.gradnonlin_correction:
                if args.verbose:
                    print('Creating gradient deviation tensor map')

                grad_non_lin_prep.grad_dev_tensor(dwi_img                = dwi_img,
                                                  gw_coils               = args.gw_coils_dat,
                                                  coregister_dwi_to_anat = args.coregister_dwi_to_anat,
                                                  gpu                    = args.gpu,
                                                  working_dir            = dmri_preproc_dir)
                
                gradnonlin_image = Image(filename = os.path.join(dmri_preproc_dir, id+'_desc-GradNonLinTensor_dwi.nii.gz'))
                
             


        if args.cleanup:
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
            files_to_cleanup.append(id + '_desc-acqparams_dwi.txt')
            files_to_cleanup.append(id + '_desc-slspec_dwi.txt')
            files_to_cleanup.append(id + '_desc-index_dwi.txt')

            outlier_files_to_cleanup = []
            outlier_files_to_cleanup.append(id + '_desc-OutlierRemoved_dwi.bval')
            outlier_files_to_cleanup.append(id + '_desc-OutlierRemoved_dwi.bvec')
            outlier_files_to_cleanup.append(id + '_desc-OutlierRemoved_dwi.nii.gz')
            outlier_files_to_cleanup.append(id + '_desc-OutlierRemoved-Index_dwi.txt')

            for dir in dirs_to_cleanup:
                if os.path.exists(os.path.join(dmri_preproc_dir, dir,)):
                    shutil.rmtree(os.path.join(dmri_preproc_dir, dir,))

            for file in files_to_cleanup:
                if os.path.exists(os.path.join(dmri_preproc_dir, file)):
                    os.remove(os.path.join(dmri_preproc_dir, file))

            for file in outlier_files_to_cleanup:
                if os.path.exists(os.path.join(dmri_preproc_dir, 'outlier-removed-images', file)):
                    os.remove(os.path.join(dmri_preproc_dir, 'outlier-removed-images', file))
                    
        
        ##MASK THE PREPROCESSED DWI to save space
        mask.apply_mask(input   = dmri_preproc,
                        mask    = dmri_mask,
                        output  = dmri_preproc)
        

        if args.gradnonlin_correction:
            gradnonlin_image = Image(filename = os.path.join(dmri_preproc_dir, id+'_desc-GradNonLinTensor_dwi.nii.gz'))
    

        ############### PREPROCESSING OF DWI DATA FINISHED ####################
        
        subject_entities = {}
        subject_entities['subject'] = args.subject
        subject_entities['session'] = args.session
                
        dmri_model_patterns = os.path.join(args.bids_dir, "derivatives", args.models_derivative_dir, "sub-{subject}[/ses-{session}]","dwi",)
        dmri_models_dir     = writing.build_path(entities, dmri_model_patterns)
        
        ###DTI MODELING ###
        if args.dti_fit_method != None:
            
            FAmap_patterns = os.path.join(dmri_model_patterns, "sub-{subject}[_ses-{session}]_model-DTI_param-FA.nii.gz")
            
            if not os.path.exists(writing.build_path(entities, FAmap_patterns)):
                if args.verbose:
                    print("Fitting DTI model with " + args.dti_fit_method + "...")

                dti_model = DTI_Model(dwi_img       = dmri_preproc,
                                      sub_info      = subject_entities,
                                      out_dir       = dmri_models_dir,
                                      fit_type      = args.dti_fit_method,
                                      mask          = dmri_mask,
                                      grad_nonlin   = gradnonlin_image,
                                      bmax          = args.dti_bmax,
                                      full_output   = args.dti_full_output)
                dti_model.fit()

        ####FWE MODELING ###
        if args.fwe_fit_method != None:
            
            Fmap_patterns = os.path.join(dmri_model_patterns, "sub-{subject}[_ses-{session}]_model-FWE-DTI_param-F.nii.gz")
    
            if not os.path.exists(writing.build_path(entities, Fmap_patterns)):
                if args.verbose:
                    print('Fitting Free-Water Elimination DTI Model')

                fwedti_model = FWEDTI_Model(dwi_img     = dmri_preproc,
                                            sub_info    = subject_entities,
                                            out_dir     = dmri_models_dir,
                                            fit_type    = args.fwe_fit_method,
                                            mask        = dmri_mask,
                                            grad_nonlin = gradnonlin_image,
                                            nthreads    = args.nthreads)
                fwedti_model.fit()


        if args.noddi_fit_method != None:
            
            NODDImap_patterns = os.path.join(dmri_model_patterns, "sub-{subject}[_ses-{session}]_model-NODDI_param-ICVF.nii.gz")

            if not os.path.exists(writing.build_path(entities, NODDImap_patterns)):
                if args.verbose:
                    print('Fitting '+args.noddi_fit_method+' model...')

                noddi_model = None
                if args.noddi_fit_method.lower() == 'smt':
                    noddi_model = SMT_NODDI_Model(dwi_img               = dmri_preproc,
                                                  sub_info              = subject_entities,
                                                  out_dir               = dmri_models_dir,
                                                  mask                  = dmri_mask,
                                                  grad_nonlin           = gradnonlin_image,
                                                  parallel_diffusivity  = args.noddi_dpar,
                                                  iso_diffusivity       = args.noddi_diso,
                                                  solver                = args.noddi_solver,
                                                  nthreads              = args.nthreads,
                                                  verbose               = args.verbose)
                else:
                    noddi_model = NODDI_Model(dwi_img               = dmri_preproc,
                                              sub_info              = subject_entities,
                                              out_dir               = dmri_models_dir,
                                              fit_type              = args.noddi_fit_method,
                                              mask                  = dmri_mask,
                                              grad_nonlin           = gradnonlin_image,
                                              parallel_diffusivity  = args.noddi_dpar,
                                              iso_diffusivity       = args.noddi_diso,
                                              solver                = args.noddi_solver,
                                              nthreads              = args.nthreads,
                                              verbose               = args.verbose)
                noddi_model.fit()


        if args.dki_fit_method != None:
            DKImap_patterns = os.path.join(dmri_model_patterns, "sub-{subject}[_ses-{session}]_model-DKI_param-MK.nii.gz")
            if not os.path.exists( writing.build_path(entities, DKImap_patterns) ):
                if args.verbose:
                    print('Fitting Diffusion Kurtosis Model')

                dki_model = DKI_Model(dwi_img     = dmri_preproc,
                                      sub_info    = subject_entities,
                                      out_dir     = dmri_models_dir,
                                      fit_type    = args.dki_fit_method,
                                      mask        = dmri_mask,
                                      smooth_data = args.dki_smooth_input,
                                      fwhm        = args.dki_smooth_fwhm)
                dki_model.fit()


        if args.csd_fod_algo != None:
                        
            CSDmap_patterns = os.path.join(dmri_model_patterns, "sub-{subject}[_ses-{session}]_model-{model}_param-{param}.nii.gz")
            
            entities['model'] = 'CSD'
            entities['param'] = 'FOD'
            CSD_fod = writing.build_path(entities, CSDmap_patterns)
            
            entities['model'] = 'MSMT-5tt'
            entities['param'] = 'WMfod'
            CSD_msmt    = writing.build_path(entities, CSDmap_patterns)
            
            entities['model'] = 'DHOLLANDER'
            entities['param'] = 'WMfod'
            CSD_dhollander    = writing.build_path(entities, CSDmap_patterns)
            
            if not os.path.exists( CSD_fod ) and not os.path.exists( CSD_msmt ) and not os.path.exists( CSD_dhollander ):
                
                if args.verbose:
                    print('Fitting Constrained Spherical Deconvolution Model')

                csd_model = CSD_Model(dwi_img       = dmri_preproc,
                                      sub_info      = subject_entities,
                                      out_dir       = dmri_models_dir,
                                      response_algo = args.csd_response_func_algo,
                                      fod_algo      = args.csd_fod_algo,
                                      mask          = dmri_mask,
                                      struct_img    = anat_img,
                                      nthreads      = args.nthreads)
                csd_model.fit()


        if args.dwi_to_standard:
            if args.verbose:
                print("Running Registration to Standard Space")

            registration_dir = os.path.join(bids_derivative_dir, args.bids_dwi_dir, "registration/")
            normalized_dir   = os.path.join(bids_derivative_dir, args.bids_dwi_dir, "models-normalized/")

            dmri_to_standard(bids_id, 
                             dwi_models_dir         = models_dir, 
                             dwi_registration_dir   = registration_dir, 
                             dwi_normalized_dir     = normalized_dir, 
                             template               = Image(filename = args.dwi_standard_template),
                             template_mask          = Image(filename = args.dwi_standard_template_mask),
                             method                 = args.dwi_standard_template_method,
                             nthreads               = args.nthreads)

if __name__ == "__main__":
    dmriproc = DiffusionProcessingPipeline()
    dmriproc.run()
        

#          ###GBSS PSEUDO T1w ###
#         if args.setup_gbss:
#             if not os.path.exists(bids_derivative_dwi_dir + '/GBSS/' + bids_id + '_desc-GBSS-Pseudo-T1w.nii.gz'):
#                 if args.verbose:
#                     print('Creating GBSS Pseudo T1-weighted Image')
#
#                 if os.path.exists(models_dir + 'DTI/' + bids_id + '_model-DTI_parameter-FA.nii.gz') and os.path.exists(models_dir + args.noddi_fit_method+'/' + bids_id + '_model-NODDI_parameter-ISO.nii.gz'):
#
#                     diff_util.create_pseudoT1_img(fa_img        = models_dir + 'DTI/' + bids_id + '_model-DTI_parameter-FA.nii.gz',
#                                                   fiso_img      = models_dir + args.noddi_fit_method+'/' + bids_id + '_model-NODDI_parameter-ISO.nii.gz',
#                                                   mask_img      = dwi_mask,
#                                                   pseudoT1_img  = bids_derivative_dwi_dir + '/GBSS/' + bids_id + '_desc-GBSS-Pseudo-T1w.nii.gz')
