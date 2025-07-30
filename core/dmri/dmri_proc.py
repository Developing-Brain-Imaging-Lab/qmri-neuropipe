import os, shutil, json, argparse, copy
import numpy as np

from bids import BIDSLayout
from bids.layout import writing

from dipy.io.image import load_nifti, save_nifti

from core.utils.io import Image, DWImage
from core.utils.cmd import run_cmd

import core.dmri.utils.qc as dmri_qc
from core.dmri.utils.dmri_reorient import dmri_reorient
import core.utils.tools as img_tools
import core.utils.mask as mask

import core.utils.denoise as denoise
import core.utils.gibbs_correction as degibbs
import core.dmri.utils.eddy_correction as ecc
import core.utils.biascorrect as biascorr
import core.dmri.utils.distortion_correction as distcorr

from core.dmri.utils.prep_grad_nonlin import grad_dev_tensor

from core.registration.linreg import linreg
from core.registration.nonlinreg import nonlinreg
from core.registration.apply_transform import apply_transform
from core.registration.convert_fsl2ants import convert_fsl2ants
from core.registration.create_composite_transform import create_composite_transform

import core.segmentation.segmentation as seg_tools

from core.anat.anat_proc import AnatomicalProcessingPipeline



from core.dmri.models.dti import DTI_Model, FWEDTI_Model
from core.dmri.models.dki import DKI_Model
from core.dmri.models.csd import CSD_Model
from core.dmri.models.noddi import NODDI_Model, SMT_NODDI_Model

from core.dmri.workflows.dmri_to_standard import dmri_to_standard

def parse_cmdline():
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
                        choices=['bet', 'hd-bet', 'mri_synthstrip', 'mrtrix', 'ants', 'antspynet'],
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
    
    parser.add_argument('--biasfield_correction_iterations',
                        type=int,
                        help='Number of iterations for bias-field correction',
                        default=1)

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
    
    parser.add_argument('--constrain_noddi',
                        type=bool,
                        help='Constrain NODDI FISO parameter based on FWE-DTI Free Water Fraction',
                        default=False)

    parser.add_argument('--fwe_fit_method',
                        type=str,
                        help='Fitting Algorithm for Diffusion Tensor Imaging Model',
                        choices=['WLS', 'NLS'],
                        default=None)
    
    parser.add_argument('--fwe_bmax',
                        type=float,
                        help='Maximum B-value to use for FWE-DTI fitting',
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

    return args

class DiffusionProcessingPipeline:

    def __init__(self, verbose=False):
        if verbose:
            print('Creating Diffusion Processing Pipeline')

    def Initialize(self, args):
        self.opts     = args
        self.id       = args.subject
        self.ses      = args.session
        self.bids_dir = args.bids_dir
        
        #Setup the BIDS Directories and Paths
        self.entities = {
            'extension': '.nii.gz',
            'subject': args.subject,
            'session': args.session,
            'suffix': 'dwi',
            'modality': 'dwi',
            'desc': 'preproc'
        }

        self.bids_id     = writing.build_path(self.entities, "sub-{subject}[_ses-{session}]")
        self.rawdata_dir = writing.build_path(self.entities, os.path.join(self.bids_dir, self.opts.bids_rawdata_dir, "sub-{subject}[/ses-{session}]",))
        self.preproc_dir = writing.build_path(self.entities, os.path.join(self.bids_dir, "derivatives", self.opts.preproc_derivative_dir, "sub-{subject}[/ses-{session}]",))
        
        self.anat_preproc_dir = None  
        self.dmri_preproc_dir = os.path.join(self.preproc_dir, "dwi",)
        os.makedirs(self.dmri_preproc_dir, exist_ok=True)
 
        self.dmri_img_pattern = os.path.join(self.dmri_preproc_dir, "sub-{subject}[_ses-{session}][_acq-{acq}][_dir-{dir}][_desc-{desc}]_{modality}.nii.gz")
        txt_pattern      = os.path.join(self.dmri_preproc_dir, "sub-{subject}[_ses-{session}][_acq-{acq}][_dir-{dir}][_desc-{desc}]_{modality}.txt")
        
        acqp_ent   = self.entities.copy()
        index_ent  = self.entities.copy() 
        slspec_ent = self.entities.copy()
        log_ent    = self.entities.copy()
        acqp_ent['desc']   = 'preproc-acqparams'
        index_ent['desc']  = 'preproc-index'
        slspec_ent['desc'] = 'preproc-slspec'
        log_ent['desc']    = 'preproc-logfile'

        self.rawdata = {}
        self.rawdata['dwi-img'] = None
        self.rawdata['dwi-mask'] = None

        self.preproc = {}
        self.preproc['dwi-img'] = DWImage(filename    = writing.build_path(self.entities, self.dmri_img_pattern),
                                          bvecs       = writing.build_path(self.entities, self.dmri_img_pattern.replace('.nii.gz', '.bvec')),
                                          bvals       = writing.build_path(self.entities, self.dmri_img_pattern.replace('.nii.gz', '.bval')),
                                          acqparams   = writing.build_path(acqp_ent, txt_pattern),
                                          index       = writing.build_path(index_ent, txt_pattern),
                                          slspec      = writing.build_path(slspec_ent, txt_pattern),
                                          json = writing.build_path(self.entities, self.dmri_img_pattern.replace('.nii.gz', '.json')))
        
        self.preproc['dwi-mask'] = Image(filename = os.path.join(self.dmri_preproc_dir, f'{self.bids_id}_desc-brain_mask.nii.gz'))

        self.preproc['t1w-img']  = None
        self.preproc['t2w-img']  = None
        self.preproc['anat-img'] = None

        self.preproc['fmap-img'] = None
        self.preproc['fmap-ref'] = None

        self.preproc['gradnonlin-img'] = None
        self.preproc['topup_base'] = None

        self.preproc['log-file'] = writing.build_path(log_ent, txt_pattern)

        self.preproc['run_topup'] = False
        self.preproc['run_synb0'] = False

        if self.opts.dist_correction is not None:
            if self.opts.dist_correction.lower()[0:5] == 'topup':
                self.preproc['run_topup'] = True
                if self.opts.topup_config is None:
                    self.opts.topup_config = os.path.join(os.environ.get("FSLDIR"), "etc/flirtsch", "b02b0.cnf")          
                self.preproc['topup_base'] = os.path.join(self.dmri_preproc_dir, "rawdata", "topup", self.bids_id+"_desc-Topup")
            elif self.opts.dist_correction.lower() == 'synb0-disco':
                self.preproc['run_synb0'] = True
                if self.opts.topup_config is None:
                    self.opts.topup_config = os.path.join(os.environ.get("FSLDIR"), "etc/flirtsch", "b02b0.cnf")
                self.preproc['topup_base'] = os.path.join(self.dmri_preproc_dir, "rawdata", "topup", self.bids_id+"_desc-Topup")

    def AnatPrep(self):

        if (self.opts.use_freesurfer or 
            self.opts.coregister_dwi_to_anat or 
            self.opts.dist_correction == 'synb0' or 
            self.opts.dist_correction == 'anatomical-coregistration' or 
            self.opts.eddy_current_correction == 'tortoise-diffprep'):
        
            self.anat_preproc_dir = os.path.join(self.preproc_dir, "anat",)
            os.makedirs(self.anat_preproc_dir, exist_ok=True)

            if self.opts.use_freesurfer:
                if self.opts.freesurfer_subjects_dir:
                    freesurfer_subjs_dir = self.opts.freesurfer_subjects_dir
                elif os.path.exists(os.path.join(self.opts.bids_dir,'derivatives', 'freesurfer',)):
                    freesurfer_subjs_dir = os.path.join(self.opts.bids_dir,'derivatives','freesurfer',)
                elif os.path.exists(os.path.join(os.environ["SUBJECT_DIR"], self.bids_id, "mri", "orig_nu.mgz")):
                    freesurfer_subjs_dir = os.path.join(os.environ["SUBJECT_DIR"])
                else:
                    print("Freesurfer Directory doesn't exist or was not specified")
                    exit(-1)

                self.preproc['t1w-img']  = Image(filename = os.path.join(self.anat_preproc_dir, self.bids_id+'_desc-Freesurfer_T1w.nii.gz'))
                self.preproc['t1w-mask'] = Image(filename = os.path.join(self.anat_preproc_dir, self.bids_id+'_desc-Freesurfer_T1w_brain_mask.nii.gz'))
                
                freesurfer_t1w  = os.path.join(freesurfer_subjs_dir, self.bids_id, "mri", "orig_nu.mgz")
                freesurfer_mask = os.path.join(freesurfer_subjs_dir, self.bids_id, "mri", "brainmask.mgz")

                convert_t1w_cmd  = f"mri_convert --in_type mgz --out_type nii -i {freesurfer_t1w} -o  {self.preproc['t1w-img'].filename}"
                convert_mask_cmd = f"mri_convert --in_type mgz --out_type nii -i {freesurfer_mask} -o {self.preproc['t1w-mask'].filename}"
                run_cmd(convert_t1w_cmd)
                run_cmd(convert_mask_cmd)

                self.preproc['anat-img']  = self.preproc['t1w-img']
                self.preproc['anat-mask'] = self.preproc['t1w-mask']

            else:
                anat_pipeline = AnatomicalProcessingPipeline()
                anat_proc     = anat_pipeline.run()

                self.preproc['t1w-img']   = anat_proc['t1w-img']
                self.preproc['t1w-mask']  = anat_proc['t1w-mask']
                self.preproc['t2w-img']   = anat_proc['t2w-img']
                self.preproc['t2w-mask']  = anat_proc['t2w-mask']
                
                if self.opts.eddy_current_correction == 'tortoise-diffprep' or  self.opts.coregister_dwi_to_anat_modality.lower() == 't2w':
                    if anat_proc['t2w-img'] is not None:
                        self.preproc['anat-img']  = anat_proc['t2w-img']
                        self.preproc['anat-mask'] = anat_proc['t2w-mask']

                    elif anat_proc['t1w-img'] is not None:
                        #If we have a T1w image only, create a synthetic T2w using T1w
                        from core.anat.utils.compute_synthetic_t2w import compute_synthetic_t2w
                        
                        self.preproc['t2w-mask'] = anat_proc['t1w-mask']

                        if not os.path.exists(os.path.join(self.anat_preproc_dir, self.bids_id+"_desc-SyntheticFromT1w_T2w.nii.gz")):
                            if self.opts.verbose or self.opts.debug:
                                print('Creating Synthetic T2w Image')

                            self.preproc['t2w-img']   = compute_synthetic_t2w(input_t1w    = self.preproc['t1w-img'],
                                                                               output_dir   = self.anat_preproc_dir,
                                                                               cmd_args     = self.opts,
                                                                               syn_t2w      = self.bids_id+"_desc-SyntheticFromT1w_T2w.nii.gz", 
                                                                               t1w_mask     = self.preproc['t1w-mask'], 
                                                                               debug        = self.opts.debug)
                            
                        self.preproc['anat-img']  = self.preproc['t2w-img']
                        self.preproc['anat-mask'] = self.preproc['t2w-mask']
                                
                elif self.opts.coregister_dwi_to_anat_modality.lower() == 't1w' and self.preproc['t1w-img'] is not None:
                    self.preproc['anat-img']  = anat_proc['t1w-img']
                    self.preproc['anat-mask'] = anat_proc['t1w-mask']
                   
                else:
                    print('No anatomical image!')
        
        
    def RawDataPrep(self):

        layout    = BIDSLayout(self.bids_dir, validate=False)
        proc_dir  = os.path.join(self.dmri_preproc_dir, "rawdata/")
          
        os.makedirs(proc_dir, exist_ok=True)

        #Get the subject's diffusion data
        subj_data = layout.get(subject=self.id, 
                               session=self.ses, 
                               datatype='dwi', 
                               suffix='dwi', 
                               extension='nii.gz', 
                               return_type='filename')
        
        num_dwis  = len(subj_data)

        self.rawdata['dwi-img'] = DWImage(filename  = f"{proc_dir}/{self.bids_id}_dwi.nii.gz",
                                          bvals     = f"{proc_dir}/{self.bids_id}_dwi.bval",
                                          bvecs     = f"{proc_dir}/{self.bids_id}_dwi.bvec",
                                          index     = f"{proc_dir}/{self.bids_id}_desc-Index_dwi.txt",
                                          acqparams = f"{proc_dir}/{self.bids_id}_desc-Acqparams_dwi.txt",
                                          json      = f"{proc_dir}/{self.bids_id}_dwi.json")
        
        if num_dwis == 1:
            img     = layout.get(subject=self.id, session=self.ses, datatype='dwi', suffix='dwi', extension='nii.gz', return_type='filename')[0]
            bvals   = layout.get(subject=self.id, session=self.ses, datatype='dwi', suffix='dwi', extension='bval', return_type='filename')[0]
            bvecs   = layout.get(subject=self.id, session=self.ses, datatype='dwi', suffix='dwi', extension='bvec', return_type='filename')[0]
            sidecar = layout.get(subject=self.id, session=self.ses, datatype='dwi', suffix='dwi', extension='json', return_type='filename')[0]
            
            shutil.copy2(img, self.rawdata['dwi-img'].filename)
            shutil.copy2(bvals, self.rawdata['dwi-img'].bvals)
            shutil.copy2(bvecs, self.rawdata['dwi-img'].bvecs)
            shutil.copy2(sidecar, self.rawdata['dwi-img'].json)

            if self.opts.dist_correction.lower()[0:5] == 'topup':
                print("Only one DWI image found, using SynB0-Disco instead")
                self.run_topup = False
                self.run_synb0 = True

        else:      

            imgs_to_merge = []
            dwi_filter = json.load(open(self.opts.proc_json, 'r'))['dwi']
            
            if "rpe_direction" in dwi_filter:
                dwi_dirs = dwi_filter['rpe_direction']

                for rpe_dir in dwi_dirs:
                    
                    img     = layout.get(subject=self.id, session=self.ses, datatype='dwi', direction=rpe_dir, suffix='dwi', extension='nii.gz', return_type='filename')[0]
                    bvals   = layout.get(subject=self.id, session=self.ses, datatype='dwi', direction=rpe_dir, suffix='dwi', extension='bval', return_type='filename')[0]
                    bvecs   = layout.get(subject=self.id, session=self.ses, datatype='dwi', direction=rpe_dir, suffix='dwi', extension='bvec', return_type='filename')[0]
                    sidecar = layout.get(subject=self.id, session=self.ses, datatype='dwi', direction=rpe_dir, suffix='dwi', extension='json', return_type='filename')[0]

                    imgs_to_merge.append(DWImage(filename=img, bvals=bvals, bvecs=bvecs,json=sidecar))

            elif "description" in dwi_filter:          
                dwi_desc = dwi_filter['description']

                img     = layout.get(subject=self.id, session=self.ses, datatype='dwi', suffix='dwi', extension='nii.gz', return_type='filename')
                bvals   = layout.get(subject=self.id, session=self.ses, datatype='dwi', suffix='dwi', extension='bval', return_type='filename')
                bvecs   = layout.get(subject=self.id, session=self.ses, datatype='dwi', suffix='dwi', extension='bvec', return_type='filename')
                sidecar = layout.get(subject=self.id, session=self.ses, datatype='dwi', suffix='dwi', extension='json', return_type='filename')
                    
                for img_desc in dwi_desc:
                    for i in range(len(img)):    
                        if img_desc in img[i]:
                            if self.opts.verbose:
                                print(f"Found DWI image with description '{img_desc}'")
                            imgs_to_merge.append(DWImage(filename=img[i], bvals=bvals[i], bvecs=bvecs[i],json= sidecar[i]))
                            continue
            else:
                print("Please provide a valid BIDS filter")
                exit(-1)      

            self.rawdata['dwi-img'] = dmri_qc.merge_phase_encodes(DWI_pepolar0 = imgs_to_merge[0], 
                                                                  DWI_pepolar1 = imgs_to_merge[1], 
                                                                  output_base  = f"{proc_dir}/{self.bids_id}")
            if len(imgs_to_merge) > 2:
                for i in range(2, len(imgs_to_merge)):
                    self.rawdata['dwi-img'] = dmri_qc.merge_phase_encodes(DWI_pepolar0 = self.rawdata['dwi-img'], 
                                                                          DWI_pepolar1 = imgs_to_merge[i], 
                                                                          output_base  = f"{proc_dir}/{self.bids_id}")
            self.run_topup  = True
            self.run_synb0  = False

        
        #Ensure ISOTROPIC voxels prior to processing
        if self.opts.verbose:
            print('Ensuring DWIs have isotropic voxels')

        self.rawdata['dwi-img'] = img_tools.check_isotropic_voxels(input_img          = self.rawdata['dwi-img'],
                                                                   output_file        = self.rawdata['dwi-img'].filename,
                                                                   target_resolution  = self.opts.resample_resolution,
                                                                   debug              = self.opts.verbose)

        #Remove Last DWI volume before processing further
        if self.opts.remove_last_vol:
            if self.opts.verbose:
                print('Removing Last DWI in volume')

            self.rawdata['dwi-img'] = img_tools.remove_end_img(input_img   = self.rawdata['dwi-img'],
                                                               output_file = self.rawdata['dwi-img'].filename)

        #Check the Image Sizes to Ensure Proper Length:
        if self.opts.verbose:
            print('Checking DWI Acquisition Size and Gradient Orientations')

        dmri_qc.check_bvals_bvecs(input_dwi   = self.rawdata['dwi-img'],
                                  output_base = f"{proc_dir}/{self.bids_id}")

        if self.opts.check_gradients:
            dmri_qc.check_gradient_directions(input_dwi   = self.rawdata['dwi-img'],
                                              nthreads    = self.opts.nthreads)

        self.rawdata['dwi-img'].index     = f"{proc_dir}/{self.bids_id}_desc-Index_dwi.txt"
        self.rawdata['dwi-img'].acqparams = f"{proc_dir}/{self.bids_id}_desc-Acqparams_dwi.txt"
        
        if not os.path.exists(self.rawdata['dwi-img'].index) or not os.path.exists(self.rawdata['dwi-img'].acqparams):
            self.rawdata['dwi-img'].index, self.rawdata['dwi-img'].acqparams = dmri_qc.create_index_acqparam_files(input_dwi   = self.rawdata['dwi-img'],
                                                                                                                   input_json  = self.rawdata['dwi-img'].json,
                                                                                                                   output_base = f"{proc_dir}/{self.bids_id}")

        self.rawdata['dwi-img'].slspec = f"{proc_dir}/{self.bids_id}_desc-Slspec_dwi.txt"
        if not os.path.exists( self.rawdata['dwi-img'].slspec ):
            self.rawdata['dwi-img'].slspec = dmri_qc.create_slspec_file(input_dwi        = self.rawdata['dwi-img'],
                                                                        input_json       = self.rawdata['dwi-img'].json,
                                                                        output_base      = f"{proc_dir}/{self.bids_id}")

        if self.opts.outlier_detection.lower() == 'manual':
            outlier_detection_dir = os.path.join(self.dmri_preproc_dir, 'outlier-removed-images/')

            if self.opts.verbose:
                print('Removing DWIs from manual selection')

            self.rawdata['dwi-img'] =  dmri_qc.remove_outlier_imgs(input_dwi                = self.rawdata['dwi-img'],  
                                                                   output_base              = f"{outlier_detection_dir}/{self.bids_id}", 
                                                                   output_removed_imgs_dir  = outlier_detection_dir,
                                                                   method                   = self.opts.outlier_detection,    
                                                                   manual_report_dir        = f"{self.rawdata_dir}/dwi")
            
        if self.opts.reorient:
            dmri_reorient(in_dwi  = self.rawdata['dwi-img'],
                          out_dwi = self.rawdata['dwi-img'],
                          ref_img = self.opts.reorient_template)
    

    def EddyCurrentCorrection(self, DWI, working_dir, method='eddy', struct_img=None):

        os.makedirs(working_dir, exist_ok=True)
        
        entities = {
            'extension': '.nii.gz',
            'subject': self.id,
            'session': self.ses,
            'suffix':  'dwi',
            'desc': 'EddyCurrentCorrected'
        }

        filename_patterns   = os.path.join(working_dir, "sub-{subject}[_ses-{session}][_desc-{desc}]_{suffix}{extension}")
        outputbase_patterns = os.path.join(working_dir, "sub-{subject}[_ses-{session}]")
        output_base         = writing.build_path(entities, outputbase_patterns)

        eddy_file = writing.build_path(entities, filename_patterns)
        entities['extension'] = '.bvec'
        eddy_bvec = writing.build_path(entities, filename_patterns)

        ECC          = copy.deepcopy(DWI)
        ECC.filename = eddy_file
        ECC.bvecs    = eddy_bvec

        CMD = f"mrconvert -nthreads {self.opts.nthreads} -datatype int16 -force -quiet {DWI.filename} {DWI.filename}"
        run_cmd(CMD)    
        
        if not ECC.exists():

            if method == 'eddy':
                if self.opts.verbose:
                    print('Running FSL EDDY...')

                ECC = ecc.eddy_fsl(input_dwi        = DWI,
                                   output_base      = output_base,
                                   topup_base       = self.preproc['topup_base'],
                                   cuda             = self.opts.gpu,
                                   cuda_device      = self.opts.cuda_device,
                                   fsl_eddy_options = self.opts.fsl_eddy_options,
                                   nthreads         = self.opts.nthreads)

            elif method == 'eddy-correct':
                if self.opts.verbose:
                    print('Running FSL Eddy-Correct')

                ECC = ecc.eddy_correct_fsl(input_dwi   = DWI,
                                           output_base = output_base)


            elif method == 'two-pass':
                if self.opts.verbose:
                    print('Running a Two-stage Eddy/Motion correction with FSL EDDY and Eddy-Correct')

                print('Running EDDY')
                eddy_corr_img = ecc.eddy_fsl(input_dwi        = DWI,
                                             output_base      = output_base,
                                             topup_base       = self.preproc['topup_base'],
                                             cuda             = self.opts.gpu,
                                             cuda_device      = self.opts.cuda_device,
                                             fsl_eddy_options = self.opts.fsl_eddy_options,
                                             nthreads         = self.opts.nthreads,
                                             debug            = self.opts.verbose)
                
                                                    
                print('Running EDDY-CORRECT')
                ECC = ecc.eddy_correct_fsl(input_dwi   = eddy_corr_img,
                                           output_base = output_base)
        
            elif method == 'tortoise-diffprep':
                if self.opts.verbose:
                    print('Running TORTOISE DIFFPREP')

                ECC = ecc.diffprep_tortoise(input_dwi        = DWI,
                                            output_base      = output_base,
                                            tortoise_options = self.opts.tortoise_options,
                                            struct_img       = struct_img,
                                            nthreads         = self.opts.nthreads)
            
            else:
                print('Incorrect Eddy method, exiting')
                exit(-1)


        return ECC
    
    def CoregisterDWItoAnat(self, working_dir, DWI):

        os.makedirs(working_dir, exist_ok=True)

        self.entities['desc'] = 'CoregisteredToAnatomy'
        coreg_filename = writing.build_path(self.entities, self.dmri_img_pattern)
        coreg_bvec     = writing.build_path(self.entities, self.dmri_img_pattern.replace('.nii.gz', '.bvec'))    
        coreg_bval     = writing.build_path(self.entities, self.dmri_img_pattern.replace('.nii.gz', '.bval')) 
        
        coreg_img = copy.deepcopy(DWI)
        coreg_img.filename = coreg_filename
        coreg_img.bvecs = coreg_bvec
        coreg_img.bvals = coreg_bval

        if not coreg_img.exists():
            if self.opts.verbose:
                print('Coregistering DWI to Anatomy')

            dwi_data, affine, dwi_img = load_nifti(DWI.filename, return_img=True)
            bvals    = np.loadtxt(DWI.bvals)
            ii       = np.where(bvals == 0)
            jj       = np.where(bvals != 0)
            
            mean_b0         = Image(filename = os.path.join(working_dir, "mean_b0.nii.gz"))
            mean_b0_data    = np.mean(dwi_data[:,:,:,np.asarray(ii).flatten()], 3)
            save_nifti(mean_b0.filename, mean_b0_data, affine, dwi_img.header)

            mean_dwi        = Image(filename = os.path.join(working_dir, "mean_dwi.nii.gz"))
            mean_dwi_data   = np.mean(dwi_data[:,:,:,np.asarray(jj).flatten()], 3)
            save_nifti(mean_dwi.filename, mean_dwi_data, affine, dwi_img.header)

            ref_img = []

            if self.opts.coregister_dwi_to_anat_method.lower() == "freesurfer-bbr":
                if self.opts.verbose:
                    print('Coregistering DWI to Anatomy using FreeSurfer BBR')
                
                # Setup BBR registration paths and variables
                bbrdwi2T1_dir = os.path.join(working_dir, 'bbrdwi2T1')
                os.makedirs(bbrdwi2T1_dir, exist_ok=True)

                b0toT1flirtmtx = os.path.join(bbrdwi2T1_dir, 'b0toT1flirt.mtx')
                fsl2antsAffine = os.path.join(bbrdwi2T1_dir, 'b0toT1flirtmtx_fsl2antsAffine.txt')
                b0toT1flirtmtx_mrtrixformat = os.path.join(bbrdwi2T1_dir, 'b0toT1flirtmtx_mrtrixformat.txt')
                
                #If Freesurfer is used, we already have grabbed the Nu.mgz file and have converted to NIFTI
                ref_img.append(self.preproc['t1w-img'])

                linreg(input                = mean_b0, 
                       ref                  = ref_img,
                       out_mat              = b0toT1flirtmtx,
                       out                  = None,
                       method               = "bbr", 
                       freesurfer_subjs_dir = self.opts.freesurfer_subjects_dir, 
                       debug                = self.opts.verbose)
                    
                convert_fsl2ants(mean_b0, self.preproc['t1w-img'], b0toT1flirtmtx, fsl2antsAffine)
                CMD = f"transformconvert {b0toT1flirtmtx} {mean_b0.filename} {self.preproc['t1w-img'].filename} flirt_import {b0toT1flirtmtx_mrtrixformat} -force -quiet"
                run_cmd(CMD)

                if  self.opts.noresample_dwi_to_anat:
                    final_transform = b0toT1flirtmtx_mrtrixformat
                else:
                    final_transform = fsl2antsAffine
                
            else:
                ref_img               = []
                mov_img               = []
                fsl_transform         = os.path.join(working_dir, "fsl.mat")
                ants_transform        = os.path.join(working_dir, "ants_")
                itk_transform         = os.path.join(working_dir, "itk_0GenericAffine.txt")
                nonlin_transform      = os.path.join(working_dir, "nonlinear_composite.nii.gz")
                final_transform       = ''

                mask_img   = Image(filename = os.path.join(working_dir, "mask.nii.gz"))
                dwi_masked = Image(filename = os.path.join(working_dir, "dwi_masked.nii.gz"))
                b0_masked  = Image(filename = os.path.join(working_dir, "b0_masked.nii.gz"))

                mask.mask_image(input       = mean_dwi,
                                mask        = mask_img,
                                mask_img    = dwi_masked,
                                algo        = self.opts.mask_method,
                                bet_options = '-f 0.25')

                mask.apply_mask(input       = mean_b0,
                                mask        = mask_img,
                                output      = b0_masked)
                                
                #If structural T2w available, use it with the b=0
                if self.opts.coregister_dwi_to_anat_modality.lower() == 't1w':
                    mov_img.append(mean_dwi)
                elif self.opts.coregister_dwi_to_anat_modality.lower() == 't2w':
                    mov_img.append(mean_b0)
                else:
                    print('Invalid anatomy contrast')
                    exit()
        
                #Mask the Anatomical image and bias-correct
                anat_masked = Image(filename = os.path.join(working_dir, "anat_masked.nii.gz"))
                if not anat_masked.exists():

                    if self.preproc['anat-mask'].exists():

                        mask.apply_mask(input       = self.preproc['anat-img'],
                                        mask        = self.preproc['anat-mask'],
                                        output      = anat_masked)
                    else:
                        self.preproc['anat-mask'] = Image(filename = os.path.join(working_dir, "anat_mask.nii.gz"))
                        mask.mask_image(input    = self.preproc['anat-img'],
                                        mask     = self.preproc['anat-mask'],
                                        mask_img = anat_masked,
                                        algo     = self.opts.mask_method)

                ref_img.append(anat_masked)
            
                #First, perform linear registration using FSL flirt
                tmp_coreg_img     = Image(filename = os.path.join(working_dir, "dwi_coreg.nii.gz"))
                linreg(input         = mov_img,
                       ref           = ref_img,
                       out_mat       = fsl_transform,
                       out           = [tmp_coreg_img],
                       method        = 'fsl',
                       dof           = 12,
                       flirt_options =  '-searchrx -180 180 -searchry -180 180 -searchrz -180 180')

                if self.opts.coregister_dwi_to_anat_method.lower() == 'bbr-fsl':
                    #Create WM segmentation from structural image
                    wmseg_img = seg_tools.create_wmseg(input_img     = ref_img[0],
                                                       output_dir = os.path.join(working_dir,'wmseg/',),
                                                       nthreads  = self.opts.nthreads )
                        
                    #Next, re-run flirt, using bbr cost function and WM segmentation
                    bbr_options = ' -cost bbr -wmseg ' + wmseg_img.filename \
                                + ' -schedule $FSLDIR/etc/flirtsch/bbr.sch -interp sinc -bbrtype global_abs -bbrslope 0.25 -finesearch 18 -init ' \
                                + fsl_transform

                    linreg(input      = mov_img,
                           ref           = ref_img,
                           out_mat       = fsl_transform,
                           out           = [tmp_coreg_img],
                           method        = 'fsl',
                           dof           = 12,
                           flirt_options = bbr_options)

                
                if self.opts.noresample_dwi_to_anat:
                    CMD = f"transformconvert {fsl_transform} {mov_img[0].filename} {ref_img[0].filename} flirt_import {itk_transform} -force -quiet"
                    run_cmd(CMD)
                else:               
                    #Convert to ITK format for warping
                    convert_fsl2ants(input    = mov_img[0],
                                     ref      = ref_img[0],
                                     fsl_mat  = fsl_transform,
                                     ants_mat = itk_transform )

                                        
                if self.opts.coregister_dwi_to_anat_method.lower() == 'linear' or self.opts.coregister_dwi_to_anat_method.lowe() == 'fsl-bbr':
                    final_transform = itk_transform
                    
                elif self.opts.coregister_dwi_to_anat_method.lower() == 'nonlinear':
                    mov_img[0] = tmp_coreg_img
                    nonlinreg(input           = mov_img,
                              ref             = ref_img,
                              mask            = self.preproc['anat-mask'],
                              out_xfm_base    = ants_transform,
                              nthreads        = self.opts.nthreads,
                              method          = 'ants',
                              ants_options    = '-j 1')

                    #Create the final transform
                    create_composite_transform(ref        = ref_img[0],
                                               out        = nonlin_transform,
                                               transforms = [ants_transform + "1Warp.nii.gz", ants_transform+"0GenericAffine.mat", itk_transform])

                    final_transform = nonlin_transform

            
            #Apply the transformation
            apply_transform(input         = DWI,
                            ref           = ref_img[0],
                            out           = coreg_img,
                            transform     = final_transform,
                            noresample    = self.opts.noresample_dwi_to_anat,
                            nthreads      = self.opts.nthreads,
                            method        = 'mrtrix',
                            ants_options  = '-e 3 -n BSpline[5]')

        return coreg_img
    
    
    def Preprocessing(self): 
        
        DWI = self.rawdata['dwi-img']

        if not self.preproc['dwi-img'].exists():
        
            #Calculate Topup/SynB0-DISCO field maps
            if self.preproc['run_topup'] or self.preproc['run_synb0'] or self.opts.dist_correction != None:
                    
                if not os.path.exists(os.path.join(self.dmri_preproc_dir, "rawdata", "topup", self.bids_id+"_desc-Topup_fieldcoef.nii.gz")):
                    #First going to run eddy_correct in order to perform an initial motion-correction to ensure images are aligned prior to estimating fields. Data are only used
                    #here and not for subsequent processing
                    
                    self.preproc["topup_base"] = None
                    eddy_img = self.EddyCurrentCorrection(DWI         = DWI,
                                                          working_dir = os.path.join(self.dmri_preproc_dir, "rawdata", "tmp-eddy-correction",),
                                                          method      ='eddy') 
                    
                    self.preproc["topup_base"] = os.path.join(self.dmri_preproc_dir, "rawdata", "topup", self.bids_id+"_desc-Topup")

                    print(self.run_topup)
                    print(self.run_synb0)

                    print(self.run_topup.type())
                                                    
                    if self.preproc['run_topup'] or self.opts.dist_correction.lower()[0:5] == 'topup':

                        if self.preproc['run_topup']:
                            print("RUNNING TOPUP")
                            distcorr.topup_fsl(input_dwi            = eddy_img,
                                               output_topup_base    = self.preproc["topup_base"],
                                               config_file          = self.opts.topup_config,
                                               field_output         = True)

                        elif not self.preproc['run_topup'] and self.preproc['run_synb0']:
                            print('No reverse phase encoded images found for TOPUP, switching to Synb0-DISCO distortion correction')
                            self.opts.dist_correction = 'synb0-disco'
                            self.preproc['run_synb0'] = True

                    if self.preproc['run_synb0'] or self.opts.dist_correction.lower() == 'synb0-disco':
                        #Run the Synb0 distortion correction'
                        distcorr.run_synb0_disco(dwi_img        = eddy_img,
                                                 t1w_img        = self.preproc['t1w-img'],
                                                 mask_method    = self.opts.mask_method,
                                                 topup_base     = self.preproc["topup_base"],
                                                 topup_config   = self.opts.topup_config,
                                                 nthreads       = self.opts.nthreads)
                        
                        
            ##DENOISING ###
            if self.opts.denoise_method != None:

                self.entities['desc'] = 'denoised'
                denoised_filename = writing.build_path(self.entities, self.dmri_img_pattern)

                if not os.path.exists(denoised_filename): 
                    if self.opts.verbose:
                        print('Denoising Image')
                    
                    DWI = denoise.denoise_image(input_img     = DWI,
                                                output_file   = denoised_filename,
                                                method        = self.opts.denoise_method ,
                                                mask          = self.preproc['dwi-mask'],
                                                noise_map     = None,
                                                noise_model   = "Rician",
                                                nthreads      = self.opts.nthreads,
                                                debug         = self.opts.debug)
                else:
                    DWI.filename = denoised_filename
                        

            ##GIBBS RINGING CORRECTION ###    
            if self.opts.gibbs_correction_method != None:

                self.entities['desc'] = 'degibbs'
                degibbs_filename = writing.build_path(self.entities, self.dmri_img_pattern)

                if not os.path.exists(degibbs_filename):
                    if self.opts.verbose:
                        print('Gibbs Ringing Correction')

                    
                    DWI = degibbs.gibbs_ringing_correction(input_img      = DWI,
                                                           output_file    = degibbs_filename,
                                                           method         = self.opts.gibbs_correction_method,
                                                           nthreads       = self.opts.nthreads, 
                                                           debug          = self.opts.debug)
                else:
                    DWI.filename = degibbs_filename
                
            ##EDDY CURRENT CORRECTION ###
            DWI = self.EddyCurrentCorrection(DWI         = DWI, 
                                             working_dir = os.path.join(self.dmri_preproc_dir, 'eddy-correction',),
                                             method      = self.opts.eddy_current_correction,
                                             struct_img  = self.preproc['anat-img'])


            if self.opts.outlier_detection != None and self.opts.outlier_detection != 'Manual':
                
                working_dir = os.path.join(self.dmri_preproc_dir, 'outlier-removed-images',)
                os.makedirs(working_dir, exist_ok=True)

                if self.opts.verbose:
                    print('Removing Outliers from DWIs')

                DWI = dmri_qc.remove_outlier_imgs(input_dwi = DWI,
                                                  output_base             = os.path.join(working_dir, f"{self.bids_id}"),
                                                  method                  = self.opts.outlier_detection,
                                                  percent_threshold       = self.opts.outlier_detection_threshold,
                                                  output_removed_imgs_dir = working_dir)


            ###BIAS FIELD CORRECTION ###
            if self.opts.biasfield_correction:
                
                self.entities['desc'] = 'biascorrected'
                biascorr_filename = writing.build_path(self.entities, self.dmri_img_pattern)

                if not os.path.exists(biascorr_filename):

                    DWI = biascorr.biasfield_correction(input_img    = DWI,
                                                        output_file  = biascorr_filename,
                                                        mask_img     = self.preproc['dwi-mask'],
                                                        method       = self.opts.biasfield_correction_method,
                                                        nthreads     = self.opts.nthreads,
                                                        iterations   = self.opts.biasfield_correction_iterations,
                                                        debug        = self.opts.verbose)
                else:
                    DWI.filename = biascorr_filename

            
            

            # ### DISTORTION CORRECTION ###
            # if self.opts.dist_correction is not None:
            #     self.entities['desc'] = 'distcorr'
            #     distcorr_filename = writing.build_path(self.entities, self.dmri_img_pattern)

            #     distcorr_dir = os.path.join(self.dmri_preproc_dir, 'distortion-correction',)
            #     os.makedirs(distcorr_dir, exist_ok=True)

            #     if not os.path.exists(distcorr_filename):
            #         if self.opts.verbose:
            #             print('Distortion Correction')

            #         if self.opts.dist_correction.lower() == 'anatomical-coregistration':
            #             DWI = distcorr.registration_method(input_dwi           = DWI,
            #                                                working_dir         = distcorr_dir,
            #                                                distortion_modality = distortion_modality,
            #                                                T1_image            = self.preproc['t1w-img'],
            #                                                T2_image            = self.preproc['t2w-img'],
            #                                                linreg_method       = self.opts.distortion_linreg_method,
            #                                                resample_to_anat    = self.opts.noresample_dwi_to_anat,
            #                                                nthreads            = self.opts.nthreads,
            #                                                verbose             = self.opts.verbose)

            #         if distortion_method == 'fieldmap':
            #             DWI = distcorr.fugue_fsl(dwi_image         = DWI,
            #                                      fmap_image        = self.preproc['fmap-img'],
            #                                      fmap_ref_image    = self.preproc['fmap-ref'],
            #                                      working_dir       = distcorr_dir)
                                    


        
            ### Anat to DWI Coregistration ###
            if self.opts.coregister_dwi_to_anat:

                DWI = self.CoregisterDWItoAnat(DWI         = DWI,
                                               working_dir = os.path.join(self.dmri_preproc_dir, 'coregistered-to-anat',))

        

            ### Brain Mask ###
            if self.opts.coregister_dwi_to_anat and not self.opts.noresample_dwi_to_anat:
                if self.opts.verbose:
                    print('Copying Anatomical Mask')

                shutil.copy2(self.preproc["anat-mask"].filename, self.preproc["dwi-mask"].filename)

            else:
                if self.opts.verbose:
                    print('Creating DWI Brain Mask')

                mask.mask_image(input                = DWI,
                                mask                 = self.preproc["dwi-mask"],
                                algo                 = self.opts.mask_method,
                                nthreads             = self.opts.nthreads,
                                ref_img              = self.opts.ants_mask_template,
                                ref_mask             = self.opts.ants_mask_template_mask,
                                antspynet_modality   = self.opts.antspynet_modality)
            
        
            #Create the preprocessed DWI file
            if self.opts.verbose:
                print('Creating Preprocessed DWI')

            dmri_qc.check_gradient_directions(input_dwi   = DWI,
                                            nthreads    = self.opts.nthreads)
            
            self.preproc["dwi-img"].copy_image(DWI, datatype=np.float32)


            if self.opts.gradnonlin_correction:
                if self.opts.verbose:
                    print('Creating gradient deviation tensor map')

                self.preproc["gradnonlin-img"] = grad_dev_tensor(dwi_img                = self.preproc["dwi-img"],
                                                                 gw_coils               = self.opts.gw_coils_dat,
                                                                 coregister_dwi_to_anat = self.opts.coregister_dwi_to_anat,
                                                                 gpu                    = self.opts.gpu,
                                                                 working_dir            = self.dmri_preproc_dir)
                

        if self.opts.cleanup:
            if self.opts.verbose:
                print('Cleaning up DWI Preprocessing Files')

            dirs_to_cleanup = []
            dirs_to_cleanup.append('rawdata')
            dirs_to_cleanup.append('anatomical-distortion-correction')
            dirs_to_cleanup.append('fieldmap-distortion-correction')
            dirs_to_cleanup.append('biasfield-correction')
            dirs_to_cleanup.append('denoise-degibbs')
            dirs_to_cleanup.append('eddy-correction')
            dirs_to_cleanup.append('coregistered-to-anat')

            files_to_cleanup = []
            files_to_cleanup.append(self.bids_id + '_desc-acqparams_dwi.txt')
            files_to_cleanup.append(self.bids_id + '_desc-slspec_dwi.txt')
            files_to_cleanup.append(self.bids_id + '_desc-index_dwi.txt')
            files_to_cleanup.append(self.bids_id + '_desc-biascorrected_dwi.nii.gz')
            files_to_cleanup.append(self.bids_id + '_desc-denoised_dwi.nii.gz')
            files_to_cleanup.append(self.bids_id + '_desc-degibbs_dwi.nii.gz')
            files_to_cleanup.append(self.bids_id + '_desc-CoregisteredToAnatomy_dwi.nii.gz')
            files_to_cleanup.append(self.bids_id + '_desc-CoregisteredToAnatomy_dwi.bval')
            files_to_cleanup.append(self.bids_id + '_desc-CoregisteredToAnatomy_dwi.bvec')
            files_to_cleanup.append('img.mif')
            files_to_cleanup.append('img_warped.mif')


            outlier_files_to_cleanup = []
            outlier_files_to_cleanup.append(self.bids_id + '_desc-OutlierRemoved_dwi.bval')
            outlier_files_to_cleanup.append(self.bids_id + '_desc-OutlierRemoved_dwi.bvec')
            outlier_files_to_cleanup.append(self.bids_id + '_desc-OutlierRemoved_dwi.nii.gz')
            outlier_files_to_cleanup.append(self.bids_id + '_desc-OutlierRemoved-Index_dwi.txt')

            for dir in dirs_to_cleanup:
                if os.path.exists(os.path.join(self.dmri_preproc_dir, dir,)):
                    shutil.rmtree(os.path.join(self.dmri_preproc_dir, dir,))

            for file in files_to_cleanup:
                if os.path.exists(os.path.join(self.dmri_preproc_dir, file)):
                    os.remove(os.path.join(self.dmri_preproc_dir, file))

            for file in outlier_files_to_cleanup:
                if os.path.exists(os.path.join(self.dmri_preproc_dir, 'outlier-removed-images', file)):
                    os.remove(os.path.join(self.dmri_preproc_dir, 'outlier-removed-images', file))
                    
            

    def ModelFit(self):

        dmri_models_dir = os.path.join(self.bids_dir, "derivatives", self.opts.models_derivative_dir,)
        os.makedirs(dmri_models_dir, exist_ok=True) 
        
        ###DTI MODELING ###
        if self.opts.dti_fit_method != None:
            dti_dir        = os.path.join(dmri_models_dir, "sub-{subject}[/ses-{session}]", "dwi", "DTI",)
            FAmap_patterns = os.path.join(dti_dir, "sub-{subject}[_ses-{session}]_model-DTI_param-FA.nii.gz")
            
            if not os.path.exists(writing.build_path(self.entities, FAmap_patterns)):
                if self.opts.verbose:
                    print("Fitting DTI model with " + self.opts.dti_fit_method + "...")

                dti_model = DTI_Model(dwi_img       = self.preproc['dwi-img'],
                                      sub_info      = self.entities,
                                      out_dir       = writing.build_path(self.entities, dti_dir),
                                      fit_type      = self.opts.dti_fit_method,
                                      mask          = self.preproc['dwi-mask'],
                                      grad_nonlin   = self.preproc['gradnonlin-img'],
                                      bmax          = self.opts.dti_bmax,
                                      full_output   = self.opts.dti_full_output)
                dti_model.fit()

            ####FWE MODELING ###
        if self.opts.fwe_fit_method != None:
            fwedti_dir    = os.path.join(dmri_models_dir, "sub-{subject}[/ses-{session}]", "dwi", "FWE-DTI",)
            Fmap_patterns = os.path.join(fwedti_dir, "sub-{subject}[_ses-{session}]_model-FWE-DTI_param-F.nii.gz")
    
            if not os.path.exists(writing.build_path(self.entities, Fmap_patterns)):
                if self.opts.verbose:
                    print('Fitting Free-Water Elimination DTI Model')

                fwedti_model = FWEDTI_Model(dwi_img     = self.preproc['dwi-img'],
                                            sub_info    = self.entities,
                                            out_dir     = writing.build_path(self.entities, fwedti_dir),
                                            fit_type    = self.opts.fwe_fit_method,
                                            mask        = self.preproc['dwi-mask'],
                                            bmax        = self.opts.fwe_bmax,
                                            grad_nonlin = self.preproc['gradnonlin-img'],
                                            nthreads    = self.opts.nthreads)
                fwedti_model.fit()


        if self.opts.dki_fit_method != None:
            dki_dir    = os.path.join(dmri_models_dir, "sub-{subject}[/ses-{session}]", "dwi", "DKI",)
            DKImap_patterns = os.path.join(dki_dir, "sub-{subject}[_ses-{session}]_model-DKI_param-MK.nii.gz")

            if not os.path.exists( writing.build_path(self.entities, DKImap_patterns) ):
                if self.opts.verbose:
                    print('Fitting Diffusion Kurtosis Model')

                dki_model = DKI_Model(dwi_img   = self.preproc["dwi-img"],
                                        sub_info    = self.entities,
                                        out_dir     = writing.build_path(self.entities, dki_dir),
                                        fit_type    = self.opts.dki_fit_method,
                                        mask        = self.preproc["dwi-mask"],
                                        smooth_data = self.opts.dki_smooth_input,
                                        fwhm        = self.opts.dki_smooth_fwhm)
                dki_model.fit()

        if self.opts.noddi_fit_method != None:
            
            noddi_dir    = os.path.join(dmri_models_dir, "sub-{subject}[/ses-{session}]", "dwi", "NODDI",)
            NODDImap_patterns = os.path.join(noddi_dir, "sub-{subject}[_ses-{session}]_model-NODDI_param-ICVF.nii.gz")

            if not os.path.exists(writing.build_path(self.entities, NODDImap_patterns)):
                if self.opts.verbose:
                    print('Fitting '+self.opts.noddi_fit_method+' model...')

                noddi_model = None
                if self.opts.noddi_fit_method.lower() == 'smt':

                    fix_fiso = None
                    if self.opts.constrain_noddi and os.path.exists(writing.build_path(self.entities, os.path.join(dmri_models_dir, "sub-{subject}[/ses-{session}]", "dwi", "FWE-DTI", "sub-{subject}[_ses-{session}]_model-FWE-DTI_param-F.nii.gz"))):
                        fix_fiso = writing.build_path(self.entities, os.path.join(dmri_models_dir, "sub-{subject}[/ses-{session}]", "dwi", "FWE-DTI", "sub-{subject}[_ses-{session}]_model-FWE-DTI_param-F.nii.gz"))

                    noddi_model = SMT_NODDI_Model(dwi_img               = self.preproc["dwi-img"],
                                                    sub_info              = self.entities,
                                                    out_dir               = writing.build_path(self.entities, noddi_dir),
                                                    mask                  = self.preproc["dwi-mask"],
                                                    grad_nonlin           = self.preproc["gradnonlin-img"],
                                                    parallel_diffusivity  = self.opts.noddi_dpar,
                                                    iso_diffusivity       = self.opts.noddi_diso,
                                                    fix_fiso              = fix_fiso,
                                                    solver                = self.opts.noddi_solver,
                                                    threads               = self.opts.nthreads,
                                                    verbose               = self.opts.verbose)
                else:
                    noddi_model = NODDI_Model(dwi_img               = self.preproc["dwi-img"],
                                                sub_info              = self.entities,
                                                out_dir               = writing.build_path(self.entities, noddi_dir),
                                                fit_type              = self.opts.noddi_fit_method,
                                                mask                  = self.preproc["dwi-mask"],
                                                grad_nonlin           = self.preproc["gradnonlin-img"],
                                                parallel_diffusivity  = self.opts.noddi_dpar,
                                                iso_diffusivity       = self.opts.noddi_diso,
                                                solver                = self.opts.noddi_solver,
                                                nthreads              = self.opts.nthreads,
                                                verbose               = self.opts.verbose)
                noddi_model.fit()




    #     if args.csd_fod_algo != None:
                        
    #         CSDmap_patterns = os.path.join(dmri_model_patterns, "sub-{subject}[_ses-{session}]_model-{model}_param-{param}.nii.gz")
            
    #         entities['model'] = 'CSD'
    #         entities['param'] = 'FOD'
    #         CSD_fod = writing.build_path(entities, CSDmap_patterns)
            
    #         entities['model'] = 'MSMT-5tt'
    #         entities['param'] = 'WMfod'
    #         CSD_msmt    = writing.build_path(entities, CSDmap_patterns)
            
    #         entities['model'] = 'DHOLLANDER'
    #         entities['param'] = 'WMfod'
    #         CSD_dhollander    = writing.build_path(entities, CSDmap_patterns)
            
    #         if not os.path.exists( CSD_fod ) and not os.path.exists( CSD_msmt ) and not os.path.exists( CSD_dhollander ):
                
    #             if args.verbose:
    #                 print('Fitting Constrained Spherical Deconvolution Model')

    #             csd_model = CSD_Model(dwi_img       = dmri_preproc,
    #                                   sub_info      = subject_entities,
    #                                   out_dir       = dmri_models_dir,
    #                                   response_algo = args.csd_response_func_algo,
    #                                   fod_algo      = args.csd_fod_algo,
    #                                   mask          = dmri_mask,
    #                                   struct_img    = anat_img,
    #                                   nthreads      = args.nthreads)
    #             csd_model.fit()


    # def Register(self):
    #     if args.dwi_to_standard:
    #         if args.verbose:
    #             print("Running Registration to Standard Space")

    #         registration_dir = os.path.join(bids_derivative_dir, args.bids_dwi_dir, "registration/")
    #         normalized_dir   = os.path.join(bids_derivative_dir, args.bids_dwi_dir, "models-normalized/")

    #         dmri_to_standard(bids_id, 
    #                          dwi_models_dir         = models_dir, 
    #                          dwi_registration_dir   = registration_dir, 
    #                          dwi_normalized_dir     = normalized_dir, 
    #                          template               = Image(filename = args.dwi_standard_template),
    #                          template_mask          = Image(filename = args.dwi_standard_template_mask),
    #                          method                 = args.dwi_standard_template_method,
    #                          nthreads               = args.nthreads)


    def run(self):
        #OPTIONS:
        #   1. Use FreeSurfer processed data
        #   2. Use Anatomical Preprocessing Pipeline Output
        #
        #

        args = parse_cmdline()
        self.Initialize(args)

        #Gather Anatomical Data if needed
        self.AnatPrep()
        self.RawDataPrep()
    
        self.Preprocessing()
        self.ModelFit()
        


        
        # if args.dist_correction:
        #     if str.lower(args.dist_correction) == 'fieldmap':
        #         fmap_image      = Image(filename = os.path.join(rawdata_dir, 'fmap', id+'_fieldmap.nii.gz'))
        #         fmap_ref_image  = Image(filename = os.path.join(rawdata_dir, 'fmap', id+'_magnitude.nii.gz'))
        
        
  
      

      

            
            
            

            
            
                
             


        # if args.cleanup:
        #     if args.verbose:
        #         print('Cleaning up DWI Preprocessing Files')

        #     dirs_to_cleanup = []
        #     dirs_to_cleanup.append('rawdata')
        #     dirs_to_cleanup.append('anatomical-distortion-correction')
        #     dirs_to_cleanup.append('fieldmap-distortion-correction')
        #     dirs_to_cleanup.append('biasfield-correction')
        #     dirs_to_cleanup.append('denoise-degibbs')
        #     dirs_to_cleanup.append('eddy-correction')
        #     dirs_to_cleanup.append('topup')
        #     dirs_to_cleanup.append('coregister-to-anatomy')

        #     files_to_cleanup = []
        #     files_to_cleanup.append(id + '_desc-acqparams_dwi.txt')
        #     files_to_cleanup.append(id + '_desc-slspec_dwi.txt')
        #     files_to_cleanup.append(id + '_desc-index_dwi.txt')

        #     outlier_files_to_cleanup = []
        #     outlier_files_to_cleanup.append(id + '_desc-OutlierRemoved_dwi.bval')
        #     outlier_files_to_cleanup.append(id + '_desc-OutlierRemoved_dwi.bvec')
        #     outlier_files_to_cleanup.append(id + '_desc-OutlierRemoved_dwi.nii.gz')
        #     outlier_files_to_cleanup.append(id + '_desc-OutlierRemoved-Index_dwi.txt')

        #     for dir in dirs_to_cleanup:
        #         if os.path.exists(os.path.join(dmri_preproc_dir, dir,)):
        #             shutil.rmtree(os.path.join(dmri_preproc_dir, dir,))

        #     for file in files_to_cleanup:
        #         if os.path.exists(os.path.join(dmri_preproc_dir, file)):
        #             os.remove(os.path.join(dmri_preproc_dir, file))

        #     for file in outlier_files_to_cleanup:
        #         if os.path.exists(os.path.join(dmri_preproc_dir, 'outlier-removed-images', file)):
        #             os.remove(os.path.join(dmri_preproc_dir, 'outlier-removed-images', file))
                    
        
        # ##MASK THE PREPROCESSED DWI to save space
        # mask.apply_mask(input   = dmri_preproc,
        #                 mask    = dmri_mask,
        #                 output  = dmri_preproc)
        

        # if args.gradnonlin_correction:
        #     gradnonlin_image = Image(filename = os.path.join(dmri_preproc_dir, id+'_desc-GradNonLinTensor_dwi.nii.gz'))
    

        # ############### PREPROCESSING OF DWI DATA FINISHED ####################
        
        
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
