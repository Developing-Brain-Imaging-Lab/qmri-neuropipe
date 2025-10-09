import os, sys, shutil, copy, json, argparse, subprocess
import nibabel as nib
from bids.layout import writing

from core.utils.io import Image
from core.utils.cmd import run_cmd

import core.utils.mask as mask
import core.utils.tools as img_tools
import core.utils.denoise as denoise
import core.utils.gibbs_correction as degibbs
import core.utils.biascorrect as biascorrect
import core.utils.create_dataset_json as create_dataset_json

from core.registration.linreg import linreg
from core.registration.nonlinreg import nonlinreg

from core.segmentation.segmentation import create_wmseg


def parse_cmdline():
    parser = argparse.ArgumentParser()

    parser.add_argument('--proc_json',
                        type=str, 
                        help='Load settings from file in json format. Command line options are overriden by values in file.', 
                        default=None)

    parser.add_argument('--bids_dir',
                        type=str,
                        help='BIDS Data Directory')

    parser.add_argument('--bids_rawdata_dir',
                        type=str, help='BIDS RAWDATA Directory',
                        default='rawdata') 

    parser.add_argument('--preproc_derivative_dir',
                        type=str, help='Pipeline Derivative Directory',
                        default='qmri-neuropipe-preproc')

    parser.add_argument('--subject',
                        type=str,
                        help='Subject ID')

    parser.add_argument('--session',
                        type=str,
                        help='Subject Timepoint',
                        default=None)

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
                        default=0)
    
    parser.add_argument('--cleanup',
                        type=bool,
                        help='Cleanup Anatomical Image Files',
                        default=False)
    
    parser.add_argument('--is_mpnrage',
                        type=bool,
                        help='Is MPnRAGE Data',
                        default=False)
                        
    parser.add_argument('--infant_mode',
                        type=bool,
                        help='Infant Mode for Processing',
                        default=False)
                        
    parser.add_argument('--brain_size',
                        type=str,
                        help='Estimate of Brain size (used for robustfov)',
                        default="220")
    
    parser.add_argument('--t1w_type',
                        type=str,
                        help='Type of T1w Acquisition',
                        choices = ['t1w', 'mp2rage', 'mpnrage'],
                        default='t1w')
                        
    parser.add_argument('--sharpen_images',
                        type=bool,
                        help='Sharpen anatomical images using Laplacian sharpening filter',
                        default=False)
                
    parser.add_argument('--acpc_align',
                        type=bool,
                        help='Run ACPC Alignment',
                        default=False)

    parser.add_argument('--t1w_acpc_img',
                        type=str,
                        help='Image to use to reorient/correct header direction for T1w images',
                        default=None)

    parser.add_argument('--t2w_acpc_img',
                        type=str,
                        help='Image to use to reorient/correct header direction for T2w images',
                        default=None)

    parser.add_argument('--denoise_method',
                        type=str,
                        help='Method for Denoising Anatomical Images',
                        choices=["ants", "dipy-nlmeans"],
                        default="ants")

    parser.add_argument('--gibbs_correction_method',
                        type=str,
                        help='Method for Gibbs Ringing Correction',
                        choices=['mrtrix', 'dipy'],
                        default='mrtrix')

    parser.add_argument('--biasfield_correction_method',
                        type=str,
                        help='Method for Gibbs Ringing Correction',
                        choices=['mrtrix-ants', 'mrtrix-fsl', 'ants', 'fsl'],
                        default='ants')

    parser.add_argument('--mask_method',
                        type=str,
                        help='Skull-stripping Algorithm',
                        choices=['bet', 'hd-bet', 'mrtrix', 'ants', 'antspynet', 'mri_synthstrip'],
                        default='bet')
    parser.add_argument('--t1w_size',
                        type=int,
                        nargs='+',
                        help='Crop/Pad T1w to this image size',
                        default=None)
    
    parser.add_argument('--t2w_size',
                        type=int,
                        nargs='+',
                        help='Crop/Pad T2w to this image size',
                        default=None)

    parser.add_argument('--t1w_mask_template',
                        type=str,
                        help='Image to use for registration based skull-stripping for T1w',
                        default=os.environ['FSLDIR']+'/data/standard/MNI152_T1_1mm.nii.gz')

    parser.add_argument('--t1w_mask_template_mask',
                        type=str,
                        help='Brain mask to use for registration based skull-stripping',
                        default=os.environ['FSLDIR']+'/data/standard/MNI152_T1_1mm_brain_mask.nii.gz')
                        
    parser.add_argument('--t2w_mask_template',
                        type=str,
                        help='Image to use for registration based skull-stripping for T2w',
                        default=os.environ['FSLDIR']+'/data/standard/MNI152_T1_1mm.nii.gz')

    parser.add_argument('--t2w_mask_template_mask',
                        type=str,
                        help='Brain mask to use for registration based skull-stripping for T2w',
                        default=os.environ['FSLDIR']+'/data/standard/MNI152_T1_1mm_brain_mask.nii.gz')
    
    parser.add_argument('--mpnrage_derivatives_dir',
                        type=str,
                        help='Derivatives directory for MPnRAGE Processed data',
                        default="mpnrage-processed")
                        
    parser.add_argument('--wmseg',
                        type=str,
                        help='White matter segmentation file to use for BBR coregistration',
                        default=None)

    parser.add_argument('--antspynet_modality',
                        type=str,
                        help='ANTsPyNet modality/network name',
                        choices=['t1', 't2'],
                        default='t1')
    
    parser.add_argument('--resample_resolution',
                        type=int,
                        nargs='+',
                        help='Resampling Input Resolution',
                        default=None)
    
    parser.add_argument('--to_standard',
                        type=bool,
                        help="Perform registration to standard space",
                        default=False)
    
    parser.add_argument('--standard_space',
                        type=str,
                        help="Label for the Standarad space",
                        default=None)

    parser.add_argument('--standard_registration_dir',
                        type=str,
                        help="Registration directory",
                        default=None)
    
    parser.add_argument('--to_standard_method',
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

    parser.add_argument('--verbose',
                        type=bool,
                        help='Print out information meassages and progress status',
                        default=False)

    args, unknown = parser.parse_known_args()

    if args.proc_json:
        with open(args.proc_json, 'rt') as f:
            t_args = argparse.Namespace()
            t_dict = vars(t_args)
            test_json = json.load(f)
            t_dict.update(test_json)
            t_dict.update(test_json["anat"])
            args, unknown = parser.parse_known_args(namespace=t_args)

    return args



class AnatomicalProcessingPipeline:

    def __init__(self, verbose=False):
        if verbose:
            print('Creating Anatomical Preprocessing Pipeline')

    def Initialize(self, args):

        self.opts     = args
        self.id       = args.subject
        self.ses      = args.session
        self.bids_dir = args.bids_dir
        
        #Setup the BIDS Directories and Paths
        self.entities = {
            'extension': '.nii.gz',
            'subject': self.opts.subject,
            'session': self.opts.session
        }

        self.bids_id     = writing.build_path(self.entities, "sub-{subject}[_ses-{session}]")
        self.rawdata_dir = writing.build_path(self.entities, os.path.join(self.bids_dir, self.opts.bids_rawdata_dir, "sub-{subject}[/ses-{session}]",))
        self.preproc_dir = writing.build_path(self.entities, os.path.join(self.bids_dir, "derivatives", self.opts.preproc_derivative_dir, "sub-{subject}[/ses-{session}]", "anat",))

        self.preproc = {}
        self.preproc['t1w-img']  = None
        self.preproc['t1w-mask'] = None
        self.preproc['t2w-img']  = None
        self.preproc['t2w-mask'] = None 

        self.rawdata = {}
        self.rawdata['t1w-img']  = None
        self.rawdata['t2w-img']  = None
         
        #Create dataset_description.json
        if not os.path.exists(os.path.join(self.bids_dir, "derivatives", self.opts.preproc_derivative_dir, "dataset_description.json")):
            create_dataset_json.create_preproc_bids_dataset_description_json(path          = os.path.join(self.bids_dir, "derivatives",),
                                                                             bids_pipeline =  self.opts.preproc_derivative_dir)
            

    def ACPC_Align(input_img, template=None, suffix="T1w", brain_size="150"):

        output_dir = os.path.join(self.preproc_dir, "acpc-align")
        os.makedirs(output_dir, exist_ok)

        robustroi       = Image(filename=os.path.join(output_dir, f"{self.bids_id}_desc-robustfov_{suffix}.nii.gz"))
        roi2full_mat    = os.path.join(output_dir, f"{self.bids_id}_desc-roi2full_{suffix}.mat")
        full2roi_mat    = os.path.join(output_dir, f"{self.bids_id}_desc-full2roi_{suffix}.mat")
        sub2std_mat     = os.path.join(output_dir, f"{self.bids_id}_desc-sub2std_{suffix}.mat")
        acpc_mat        = os.path.join(output_dir, f"{self.bids_id}_desc-ACPC-Alignment_{suffix}.mat")
        acpc_aligned    = Image(filename=os.path.join(output_dir, f"{self.bids_id}_desc-ACPC-Aligned_{suffix}.nii.gz"))
        
        CMD = f"robustfov -i {input_img.filename} -m {roi2full_mat} -r {robustroi.filename} -b {brain_size}"
        run_cmd(CMD)         
               
        CMD = f"convert_xfm -omat {full2roi_mat} -inverse {roi2full_mat}"
        run_cmd(CMD) 
        
        #Register to the supplied template.
        if template is None:
            print("Error: No Template for ACPC-Alignment Supplied!", flush=True)
            exit(-1)
        
        CMD = f"flirt -ref {template.filename} -in {robustroi.filename} -omat {sub2std_mat}  -interp spline -searchrx -180 180 -searchry -180 180 -searchrz -180 180"
        run_cmd(CMD)

        CMD = f"convert_xfm -omat {acpc_mat} -concat {sub2std_mat} {full2roi_mat}"
        run_cmd(CMD)
        
        CMD = f"applywarp --rel --interp=spline -i {input_img.filename} -r {template.filename}  --premat={acpc_mat} -o {acpc_aligned.filename}"
        run_cmd(CMD)

        return acpc_aligned

    def Coregister(self, T1w, T2w, infant_mode, brain_size, nthreads):

        RefImage    = T1w
        InImage     = T2w
        FullImage   = T2w
        OutImage    = Image(filename = os.path.join(self.preproc_dir, f"{self.bids_id}_space-individual-T1w_T2w.nii.gz"))
        T1w_output  = T1w
        T2w_output  = OutImage

        if not OutImage.exists():
            print("Coregistering T1w and T2w image", flush=True)
        
            if infant_mode:
                RefImage    = T2w
                InImage     = T1w
                FullImage   = T1w
                OutImage    = Image(filename = os.path.join(self.preproc_dir, f"{self.bids_id}_space-individual-T2w_T1w.nii.gz"))
                T1w_output  = OutImage
                T2w_output  = T2w
                
            FullMat = os.path.join(self.preproc_dir, f"{self.bids_id}_desc-T2w-2-T1w.mat")
            InitMat = os.path.join(self.preproc_dir, f"{self.bids_id}_desc-Coreg.mat")
            BBRMat  = os.path.join(self.preproc_dir, f"{self.bids_id}_desc-BBR.mat")
            
            #Create FOV for RefImage
            RefImage_robustfov      = Image(filename = os.path.join(self.preproc_dir, "RefImage_robustfov.nii.gz"))
            RefImage_roi2full_mat   = os.path.join(self.preproc_dir, "RefImage_roi2full.mat")
            RefImage_full2roi_mat   = os.path.join(self.preproc_dir, "RefImage_full2roi.mat")
            CMD = f"robustfov -i {RefImage.filename} -m {RefImage_roi2full_mat} -r {RefImage_robustfov.filename} -b {str(brain_size)}"
            run_cmd(CMD)
            
            CMD = f"convert_xfm -omat {RefImage_full2roi_mat} -inverse {RefImage_roi2full_mat}"
            run_cmd(CMD)

            
            InImage_robustfov      = Image(filename = os.path.join(self.preproc_dir, "InImage_robustfov.nii.gz"))
            InImage_roi2full_mat   = os.path.join(self.preproc_dir, "InImage_roi2full.mat")
            InImage_full2roi_mat   = os.path.join(self.preproc_dir, "InImage_full2roi.mat")
            CMD = f"robustfov -i {InImage.filename} -m {InImage_roi2full_mat} -r {InImage_robustfov.filename} -b {str(brain_size)}"
            run_cmd(CMD)

            CMD = f"convert_xfm -omat {InImage_full2roi_mat} -inverse {InImage_roi2full_mat}"
            run_cmd(CMD)

            InImage_robustfov = biascorrect.biasfield_correction(input_img   = InImage_robustfov,
                                                                output_file = InImage_robustfov.filename, 
                                                                method      = "ants", 
                                                                nthreads    = self.opts.nthreads, 
                                                                iterations  = 1)

            RefImage_robustfov = biascorrect.biasfield_correction(input_img   = RefImage_robustfov,
                                                                output_file = RefImage_robustfov.filename, 
                                                                method      = "ants", 
                                                                nthreads    = self.opts.nthreads, 
                                                                iterations  = 1) 
            flirt_opts = "-init $FSLDIR/etc/flirtsch/ident.mat " \
                    + "-interp spline -cost normmi " \
                    + "-searchrx -10 10 -searchry -10 10 -searchrz -10 10 -finesearch 2 -coarsesearch 5"

            linreg(input   = InImage_robustfov,
                ref     = RefImage_robustfov,
                out_mat = InitMat,
                dof=6, 
                method="fsl", 
                flirt_options=flirt_opts,
                nthreads=1)
                    
            # #Create WMseg for BBR
            # WMsegImg = create_wmseg(input_img  = RefImage_robustfov, 
            #                         output_dir = os.path.join(self.preproc_dir, "wmseg",), 
            #                         nthreads   = nthreads)

            # bbr_opts = f"-init {InitMat} -interp spline " \
            #          + f"-wmseg {WMsegImg.filename} " \
            #          + f"-searchrx -5 5 -searchry -5 5  -searchrz -5 5"\
            #          + f"-cost bbr -bbrtype global_abs -bbrslope 0.5 -finesearch 10 "\
            #          + f"-schedule ${FSLDIR}/etc/flirtsch/bbr.sch"

            # linreg(input   = InImage_robustfov,
            #        ref     = RefImage_robustfov,
            #        out_mat = BBRMat,
            #        dof=6, 
            #        method="fsl", 
            #        flirt_options=flirt_opts,
            #        nthreads=1)

            CMD = f"convert_xfm -omat {FullMat} -concat {InitMat} {InImage_full2roi_mat}"
            run_cmd(CMD)
            
            CMD = f"convert_xfm -omat {FullMat} -concat {RefImage_roi2full_mat} {FullMat}"
            run_cmd(CMD)

            CMD = f"applywarp --rel --interp=spline -i {FullImage.filename} -r {RefImage.filename} --premat={FullMat} -o {OutImage.filename}"
            run_cmd(CMD)

            #Clean up files
            if InImage.exists():
                InImage.remove()

            if os.path.exists(InImage_robustfov.filename):
                os.remove(InImage_robustfov.filename)
            if os.path.exists(InImage_roi2full_mat):
                os.remove(InImage_roi2full_mat)
            if os.path.exists(InImage_full2roi_mat):
                os.remove(InImage_full2roi_mat)

            if os.path.exists(RefImage_robustfov.filename):
                os.remove(RefImage_robustfov.filename)
            if os.path.exists(RefImage_roi2full_mat):
                os.remove(RefImage_roi2full_mat)
            if os.path.exists(RefImage_full2roi_mat):
                os.remove(RefImage_full2roi_mat)
                
            if os.path.exists(InitMat):
                os.remove(InitMat)
            if os.path.exists(FullMat):
                os.remove(FullMat)

        return T1w_output, T2w_output

    
    def RawDataPrep(self):
        from bids import BIDSLayout

        layout   = BIDSLayout(self.bids_dir, validate=False)
        os.makedirs(self.preproc_dir, exist_ok=True)

        #Get T1w images is they exist
        subj_data = layout.get(subject=self.id, 
                               session=self.ses, 
                               datatype='anat', 
                               suffix='T1w', 
                               extension='nii.gz', 
                               return_type='filename')

        num_t1w  = len(subj_data)
    
        if num_t1w >= 1:
            self.rawdata['t1w-img'] = Image(filename = os.path.join(self.preproc_dir, os.path.basename(subj_data[0])),
                                            json     = os.path.join(self.preproc_dir, os.path.basename(subj_data[0].replace('.nii.gz', '.json'))))

            print(subj_data[0])
            print(self.rawdata['t1w-img'].filename )
            self.rawdata['t1w-img'].copy_image(img2copy=Image(filename = subj_data[0]))
            

            CMD="fslreorient2std {0} {1}".format(self.rawdata['t1w-img'].filename, self.rawdata['t1w-img'].filename)
            run_cmd(CMD)

            if self.opts.t1w_size is not None:
                if self.opts.verbose:
                    print("Cropping/Padding T1w to size: " + str(self.opts.t1w_size), flush=True)
                img_tools.crop_or_pad_image(input_img    = self.rawdata['t1w-img'],
                                            output_file  = self.rawdata['t1w-img'].filename,
                                            target_size  = self.opts.t1w_size,
                                            debug        = self.opts.verbose)
            
        elif self.opts.t1w_type.lower() == 'mpnrage':
            if self.opts.mpnrage_derivatives_dir == '':
                raise ValueError("If using MPnRAGE images, please provide the derivatives directory")
            
            mpnrage_dir = writing.build_path(self.entities, os.path.join(self.bids_dir, "derivatives", self.opts.mpnrage_derivatives_dir, "sub-{subject}[/ses-{session}]", "anat",))

            self.rawdata['t1w-img'] = Image(filename = os.path.join(mpnrage_dir, f"{self.bids_id}_acq-MPnRAGE_rec-MoCo_T1w.nii.gz"),
                                            json     = os.path.join(mpnrage_dir, f"{self.bids_id}_acq-MPnRAGE_rec-MoCo_T1w.json"))

        elif self.opts.t1w_type.lower() == 'mp2rage':
            self.rawdata['t1w-img'] = Image(filename = os.path.join(self.rawdata_dir, "anat", f"{self.bids_id}_inv-2_part-mag_MP2RAGE.nii.gz"),
                                            json     = os.path.join(self.rawdata_dir, "anat", f"{self.bids_id}_inv-2_MP2RAGE.json"))
        elif self.opts.t1w_type.lower() == 'spgr-vfa':

            print("VFA!")

            #SPGR-VFA
            spgr_vfa_img = Image(filename = os.path.join(self.rawdata_dir, "anat", f"{self.bids_id}_desc-SPGR_VFA.nii.gz"),
                                 json     = os.path.join(self.rawdata_dir, "anat", f"{self.bids_id}_desc-SPGR_VFA.json"))
            
            if spgr_vfa_img.exists():
                #Extract the T1w image from the SPGR-VFA
                #Create target image and coregister images to the target
                self.rawdata['t1w-img'] = Image(filename = os.path.join(self.preproc_dir, f"{self.bids_id}_desc-SPGR_T1w.nii.gz"),
                                                json     = os.path.join(self.preproc_dir, f"{self.bids_id}_desc-SPGR_T1w.json"))
                if self.opts.verbose:
                    print("Creating Target Image for DESPOT-VFA")

                spgr_img = nib.load(spgr_vfa_img.filename)
                num_spgr = spgr_img.shape[3]

                ref_img = nib.Nifti1Image(spgr_img.get_fdata()[:,:,:,num_spgr-1], spgr_img.affine)
                ref_img.to_filename(self.rawdata['t1w-img'].filename)
            
        else:   
            self.rawdata['t1w-img'] = None
            if self.opts.verbose:
                print("WARNING: No anatomical T1w image found")


        #Do the same thing for T2w scans
        subj_data = layout.get(subject=self.id, 
                               session=self.ses, 
                               datatype='anat', 
                               suffix='T2w', 
                               extension='nii.gz', 
                               return_type='filename')

        num_t2w   = len(subj_data)

        if num_t2w >= 1:
            self.rawdata['t2w-img'] = Image(filename = os.path.join(self.preproc_dir, os.path.basename(subj_data[0])),
                                            json     = os.path.join(self.preproc_dir, os.path.basename(subj_data[0].replace('.nii.gz', '.json'))))

            self.rawdata['t2w-img'].copy_image(img2copy=Image(filename = subj_data[0]))

            CMD="fslreorient2std {0} {1}".format(self.rawdata['t2w-img'].filename, self.rawdata['t2w-img'].filename)
            run_cmd(CMD)

            if self.opts.t2w_size is not None:
                if self.opts.verbose:
                    print("Cropping/Padding T2w to size: " + str(self.opts.t2w_size), flush=True)
                img_tools.crop_or_pad_image(input_img    = self.rawdata['t2w-img'],
                                            output_file  = self.rawdata['t2w-img'].filename,
                                            target_size  = self.opts.t2w_size,
                                            debug        = self.opts.verbose)
        else:
            self.rawdata['t2w-img'] = None
            if self.opts.verbose:
                print("WARNING: No anatomical T2w image found")

        if self.opts.resample_resolution:
            if self.rawdata['t1w-img'] is not None:
                self.rawdata['t1w-img'] = img_tools.check_isotropic_voxels(input_img          = self.rawdata['t1w-img'],
                                                                           output_file        = self.rawdata['t1w-img'].filename,
                                                                           target_resolution  = self.opts.resample_resolution,
                                                                           debug              = self.opts.verbose)
            
            if self.rawdata['t2w-img'] is not None:
                self.rawdata['t2w-img'] = img_tools.check_isotropic_voxels(input_img          = self.rawdata['t2w-img'],
                                                                           output_file        = self.rawdata['t2w-img'].filename,
                                                                           target_resolution  = self.opts.resample_resolution,
                                                                           debug              = self.opts.verbose)
    

    def Preprocess(self, input_img=None, suffix=None):

        Image_pattern    = os.path.join(self.preproc_dir, "sub-{subject}[_ses-{session}][_acq-{acq}][_rec-{rec}][_desc-{desc}]_{modality}.nii.gz")
        mpnrage_patterns = os.path.join(self.opts.bids_dir, "derivatives", self.opts.mpnrage_derivatives_dir, "sub-{subject}[/ses-{session}]", "anat",)

        preproc_ent = self.entities.copy()
        preproc_ent['modality'] = suffix
        preproc_ent['desc'] = 'preproc'

        Mask_ent = self.entities.copy()
        Mask_ent['modality'] = suffix
        Mask_ent['desc']     = 'brain-mask'
            
        if suffix == "T1w" and self.opts.t1w_type.lower() == "mpnrage":
            preproc_ent['acq'] = 'MPnRAGE'
            Mask_ent['acq'] = 'MPnRAGE'
        if suffix == "T1w" and self.opts.t1w_type.lower() == "mp2rage":
            preproc_ent['acq'] = 'MP2RAGE'
            Mask_ent['acq'] = 'MP2RAGE'
        
        preproc_img = Image(filename = writing.build_path(preproc_ent, Image_pattern),
                            json = writing.build_path(preproc_ent, Image_pattern.replace(".nii.gz", ".json")))
        
        preproc_mask = Image(filename = writing.build_path(Mask_ent, Image_pattern))
        
        if (input_img and not preproc_img.exists()):
          
            if self.opts.verbose:
                print("#######################################", flush=True)
                print("Running Anatomical Preparation Pipeline", flush=True)
                print(flush=True)

            denoise_ent         = preproc_ent.copy()
            noisemap_ent        = preproc_ent.copy()
            gibbs_ent           = preproc_ent.copy()
            bias_ent            = preproc_ent.copy()
                
            denoise_ent['desc'] = 'Denoised'
            noisemap_ent['desc']= 'NoiseMap'
            gibbs_ent['desc']   = 'GibbsRingingCorrected'
            bias_ent['desc']    = 'BiasFieldCorrected'
                            
            denoise_img     = Image(filename = writing.build_path(denoise_ent, Image_pattern))
            noisemap        = Image(filename = writing.build_path(noisemap_ent, Image_pattern))
            gibbs_img       = Image(filename = writing.build_path(gibbs_ent, Image_pattern))
            bias_img        = Image(filename = writing.build_path(bias_ent, Image_pattern))
            
            print("Working on " + preproc_ent['modality'] + " image", flush=True)
                
            if not os.path.exists(preproc_img.filename):
                if self.opts.verbose:
                    print("\tMasking image...", flush=True)

                mask.mask_image(input                = input_img,
                                mask                 = preproc_mask,
                                algo                 = self.opts.mask_method,
                                nthreads             = self.opts.nthreads,
                                ref_img              = self.opts.t1w_mask_template,
                                ref_mask             = self.opts.t1w_mask_template_mask,
                                antspynet_modality   = self.opts.antspynet_modality)
                    
                if not os.path.exists(denoise_img.filename):
                    if self.opts.verbose:
                        print("\tDenoising image...", flush = True)
                        
                    denoise_img = denoise.denoise_image(input_img     = input_img,
                                                        mask          = preproc_mask,
                                                        output_file   = denoise_img.filename,
                                                        method        = self.opts.denoise_method,
                                                        noise_map     = noisemap.filename,
                                                        nthreads      = self.opts.nthreads)
                    if self.opts.verbose:
                        print("\tDenoising Successful", flush = True)
                        print(flush = True)
                            
                    if not os.path.exists(gibbs_img.filename):
                        if self.opts.verbose:
                            print("\tCorrecting Gibbs Ringing...", flush = True)
                    
                        gibbs_img = degibbs.gibbs_ringing_correction(input_img    = denoise_img,
                                                                     output_file  = gibbs_img.filename,
                                                                     method       = self.opts.gibbs_correction_method,
                                                                     nthreads     = self.opts.nthreads)
                        if self.opts.verbose:
                            print("\tGibbs Ringing Correction Successful", flush = True)
                            print(flush = True)
                            
                    if not os.path.exists(bias_img.filename):
                        if self.opts.verbose:
                            print("\tCorrecting Bias Field...", flush = True)
                            
                        preproc_img = biascorrect.biasfield_correction(input_img   = gibbs_img,
                                                                       output_file = preproc_img.filename, 
                                                                       method      = "ants", 
                                                                       mask_img    = preproc_mask, 
                                                                       nthreads    = self.opts.nthreads, 
                                                                       iterations  = 1)
                        if self.opts.verbose:
                            print("\tBias Field Correction Successful", flush = True)

                        if(self.opts.sharpen_images):
                            if self.opts.verbose:
                                print("\tSharpening image contrast", flush=True)
                            CMD = f"ImageMath 3 {preproc_img.filename} Sharpen {preproc_img.filename}"
                            run_cmd(CMD)
                            
                            if self.opts.verbose:
                                print("\tSharpening Successful", flush = True)
                    
                if self.opts.cleanup:
                    if denoise_img.exists():
                        denoise_img.remove()
                    if gibbs_img.exists():
                        gibbs_img.remove()
                    if bias_img.exists():
                        bias_img.remove()
                    if noisemap.exists():
                        noisemap.remove()

        if not input_img.exists():
            preproc_img = None 
            preproc_mask = None  
                
        return preproc_img, preproc_mask
                    
            
    def run(self):
        # parse commandline
        args = parse_cmdline()

        self.Initialize(args)
        self.RawDataPrep()
          
        if self.rawdata['t1w-img'] is not None and self.rawdata['t2w-img'] is not None:
            if args.verbose:
                print("\tCoregistering T1w and T2w images")
                print(flush=True)

            self.rawdata['t1w-img'], self.rawdata['t2w-img'] = self.Coregister(T1w = self.rawdata['t1w-img'],
                                                                               T2w = self.rawdata['t2w-img'],
                                                                               infant_mode = self.opts.infant_mode,
                                                                               brain_size  = self.opts.brain_size,
                                                                               nthreads    = self.opts.nthreads)
            
            if args.verbose:
                print("Finished coregistering T1w and T2w images")
                print(flush=True)


        if self.rawdata['t1w-img'] is not None:
            self.preproc['t1w-img'], self.preproc['t1w-mask'] = self.Preprocess(input_img = self.rawdata['t1w-img'], suffix = "T1w")
            
        if self.rawdata['t2w-img'] is not None:
            self.preproc['t2w-img'], self.preproc['t2w-mask'] = self.Preprocess(input_img = self.rawdata['t2w-img'], suffix = "T2w")
        

        if self.preproc['t1w-img'] is not None and self.preproc['t2w-img'] is not None:   

            self.preproc['brain-mask'] = Image(filename = os.path.join(self.preproc_dir, f"{self.bids_id}_desc-brain-mask.nii.gz"))
            
            CMD = f"fslmaths {self.preproc['t1w-mask'].filename} -add {self.preproc['t2w-mask'].filename} -bin -fillh {self.preproc['brain-mask'].filename}"
            run_cmd(CMD)

            self.preproc['t1w-mask'].remove()
            self.preproc['t2w-mask'].remove()
            self.preproc['t1w-mask'] = self.preproc['brain-mask']
            self.preproc['t2w-mask'] = self.preproc['brain-mask']
            
            create_dataset_json.create_bids_sidecar_json(image = self.preproc['brain-mask'], 
                                                         data = {"Description": "Brain Mask",
                                                                "Sources": f"{self.preproc['t1w-img'].filename} and {self.preproc['t2w-img'].filename}",
                                                                "SkullStripped": True,
                                                                "SkllStrippingMethod": self.opts.mask_method})

        if self.opts.acpc_align:
            #Run ACPC Alignment (using the provided templates)
            self.preproc['t1w-img'] = ACPC_Align(input_img = self.preproc['t1w-img'], 
                                                 template  = Image(filename = self.opts.t1w_acpc_img), 
                                                 suffix    = "T1w", 
                                                 brain_size=self.opts.brain_size)

            self.preproc['t2w-img'] = ACPC_Align(input_img = self.preproc['t2w-img'], 
                                                 template  = Image(filename = self.opts.t2w_acpc_img), 
                                                 suffix    = "T2w", 
                                                 brain_size=self.opts.brain_size)


        if self.opts.to_standard:
            if self.opts.verbose:
                print("Running Registration to Standard Space")

            registration_patterns = os.path.join(self.bids_dir, "derivatives", self.opts.standard_registration_dir, "sub-{subject}[/ses-{session}]", "anat",)
            out_dir               = writing.build_path(self.entities, registration_patterns)
            os.makedirs(out_dir, exist_ok=True)

            if self.preproc['t1w-img'].exists():
                out_base = os.path.join(out_dir, self.bids_id+"_desc-ANTs_space-"+self.opts.standard_space+"_")

                if args.t1w_type.lower() == 'mpnrage':
                    out_base = os.path.join(out_dir, self.bids_id+"_acq-MPnRAGE_space-"+self.standard_space+"_desc-ANTsNonlin_")
            
                nonlinreg(input        = self.preproc['t1w-img'],
                          ref          = Image(filename=self.opts.standard_template), 
                          mask         = Image(filename=self.opts.standard_template_mask),
                          out_xfm      = out_base+"FwdTransform.nii.gz", 
                          out_xfm_base = out_base,
                          nthreads     = self.opts.nthreads, 
                          method       = self.opts.to_standard_method)



        if self.preproc['t1w-img'] is not None and self.preproc['t1w-img'].exists():
            create_dataset_json.create_bids_sidecar_json(image = self.preproc['t1w-img'], 
                                                         data  = {"Modality": "T1w",
                                                                "Description": "Preprocessed T1w Image",
                                                                "Sources": self.preproc['t1w-img'].filename,
                                                                "SkullStripped": True,
                                                                "SkullStrippingMethod": self.opts.mask_method,
                                                                "Denoised": True,
                                                                "DenoisingMethod": self.opts.denoise_method,
                                                                "GibbsCorrected": True,
                                                                "GibbsCorrectionMethod": self.opts.gibbs_correction_method,
                                                                "BiasCorrected": True,
                                                                "BiasCorrectionMethod": self.opts.biasfield_correction_method,
                                                                "Sharpened": self.opts.sharpen_images})            
        if self.preproc['t2w-img'] is not None and self.preproc['t2w-img'].exists():
            create_dataset_json.create_bids_sidecar_json(image = self.preproc['t2w-img'], 
                                                         data  = {"Modality": "T2w",
                                                                "Description": "Preprocessed T2w Image",
                                                                "Sources": self.preproc['t2w-img'].filename,
                                                                "SkullStripped": True,
                                                                "SkullStrippingMethod": self.opts.mask_method,
                                                                "Denoised": True,
                                                                "DenoisingMethod": self.opts.denoise_method,
                                                                "GibbsCorrected": True,
                                                                "GibbsCorrectionMethod": self.opts.gibbs_correction_method,
                                                                "BiasCorrected": True,
                                                                "BiasCorrectionMethod": self.opts.biasfield_correction_method,
                                                                "Sharpened": self.opts.sharpen_images})

        #Cleanup the files  
        if self.opts.cleanup:
            if self.opts.verbose:
                print("Cleaning up files", flush=True)
                
            # if T1w is not None and T1w.img.exists():
            #     T1w.img.remove()

            #     if T1w.mask is not None and T1w.mask.exists():
            #         T1w.mask.remove()

            # if T2w is not None and T2w.img.exists():
            #     T2w.img.remove()

            #     if T2w.mask is not None and T2w.mask.exists():
            #         T2w.mask.remove()

            if args.verbose:
                print("Finished cleaning up files", flush=True)
                print(flush=True)
            
      
        if args.verbose:
            print("Anatomical Processing Successful")
            print("")
            
                    
        return self.preproc
            
            
        
if __name__ == "__main__":
    anatproc = AnatomicalProcessingPipeline()
    anatproc.run()

    
        
        
        
        
        
        
        
        
        







