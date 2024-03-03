import os, sys, shutil, copy, json, argparse, subprocess

from bids.layout import writing
from core.utils.io import Image
import core.utils.mask as mask
import core.utils.denoise as denoise
import core.utils.gibbs_correction as degibbs
import core.utils.biascorrect as biascorrect
import core.utils.create_dataset_json as create_dataset_json

import core.anat.workflows.prep_rawdata as raw_proc
import core.anat.workflows.hcp_process as hcp

from core.registration.nonlinreg import nonlinreg


class AnatomicalPrepPipeline:

    def __init__(self, verbose=False):
        if verbose:
            print('Creating Anatomical Preprocessing Pipeline')

    def run(self):
        # parse commandline
        parser = argparse.ArgumentParser()

        parser.add_argument('--load_json',
                            type=str, 
                            help='Load settings from file in json format. Command line options are overriden by values in file.', 
                            default=None)

        parser.add_argument('--bids_dir',
                            type=str,
                            help='BIDS Data Directory')

        parser.add_argument('--bids_rawdata_dir',
                            type=str, help='BIDS RAWDATA Directory',
                            default='rawdata') 
    
        parser.add_argument('--pipeline_name',
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
                            default="150")
        
        parser.add_argument('--t1w_type',
                            type=str,
                            help='Type of T1w Acquisition',
                            choices = ['t1w', 'mp2rage', 'mpnrage'],
                            default='t1w')
                            
        parser.add_argument('--sharpen_images',
                            type=bool,
                            help='Sharpen anatomical images using Laplacian sharpening filter',
                            default=False)
                    
        parser.add_argument('--do_hcp_preproc',
                            type=bool,
                            help='Run HCP Preprocessing Steps',
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

        if args.load_json:
            with open(args.load_json, "rt") as f:
                t_args = argparse.Namespace()
                t_dict = vars(t_args)
                t_dict.update(json.load(f))
                args, unknown = parser.parse_known_args(namespace=t_args)

        #Setup the BIDS Directories and Paths
        entities = {
            'extension': ".nii.gz",
            'subject': args.subject,
            'session': args.session,
        }
                             
        id_patterns         = "sub-{subject}[_ses-{session}]"
        rawdata_patterns    = os.path.join(args.bids_dir, args.bids_rawdata_dir, "sub-{subject}[/ses-{session}]",)
        derivative_patterns = os.path.join(args.bids_dir, "derivatives", args.pipeline_name),
        output_patterns     = os.path.join(args.bids_dir, "derivatives", args.pipeline_name, "sub-{subject}[/ses-{session}]", "anat",)
        mpnrage_patterns    = os.path.join(args.bids_dir, "derivatives", args.mpnrage_derivatives_dir, "sub-{subject}[/ses-{session}]", "anat",)
        
        bids_id             = writing.build_path(entities, id_patterns)
        bids_rawdata_dir    = writing.build_path(entities, rawdata_patterns)
        bids_derivative_dir = writing.build_path(entities, derivative_patterns)
        bids_output_dir     = writing.build_path(entities, output_patterns)
        
        Image_pattern  = os.path.join(bids_output_dir, "sub-{subject}[_ses-{session}][_acq-{acq}][_rec-{rec}][_desc-{desc}]_{modality}.nii.gz")
                
        if not os.path.exists(bids_output_dir):
            os.makedirs(bids_output_dir)
        
        #Create dataset_description.json
        if not os.path.exists(os.path.join(bids_derivative_dir, "dataset_description.json")):
            create_dataset_json.create_preproc_bids_dataset_description_json(path          = bids_derivative_dir,
                                                                             bids_pipeline =  args.pipeline_name)

        logfile     = open(os.path.join(bids_output_dir, bids_id+"_desc-AnatProc.log"), 'w')
        sys.stdout  = logfile
        
        T1w, T2w = raw_proc.prep_anat_rawdata(id                        = bids_id,
                                              rawdata_dir               = bids_rawdata_dir,
                                              t1w_type                  = args.t1w_type,
                                              mpnrage_derivatives_dir   = writing.build_path(entities, mpnrage_patterns),
                                              verbose                   = args.verbose)           
        T1w_preproc_ent = entities.copy()
        T1w_preproc_ent['modality'] = 'T1w'
        T1w_preproc_ent['desc']     = 'preproc'
            
        if args.t1w_type == 'mpnrage' or args.t1w_type == 'MPnRAGE':
            T1w_preproc_ent['acq'] = 'MPnRAGE'
        if args.t1w_type == 'mp2rage' or args.t1w_type == 'MP2RAGE':
            T1w_preproc_ent['acq'] = 'MP2RAGE'
        
        T1w_preproc = Image(filename = writing.build_path(T1w_preproc_ent, Image_pattern),
                            json     = writing.build_path(T1w_preproc_ent, Image_pattern.replace(".nii.gz", ".json")))    
        
        
        T2w_preproc_ent = entities.copy()
        T2w_preproc_ent['modality'] = 'T2w'
        T2w_preproc_ent['desc']     = 'preproc'
                
        T2w_preproc = Image(filename = writing.build_path(T2w_preproc_ent, Image_pattern),
                            json     = writing.build_path(T2w_preproc_ent, Image_pattern.replace(".nii.gz", ".json")))  
        
        brain_mask = Image(filename = os.path.join(bids_output_dir, bids_id+"_desc-brain-mask.nii.gz"),
                           json     = os.path.join(bids_output_dir, bids_id+"_desc-brain-mask.json"))
        
        brain_mask_t1w=None
        brain_mask_t2w=None
        
        if (T1w and not T1w_preproc.exists()) or (T2w and not T2w_preproc.exists()):
          
            if args.verbose:
                print("#######################################", flush=True)
                print("Running Anatomical Preparation Pipeline", flush=True)
                print(flush=True)
                    
            if args.do_hcp_preproc:
                #Run ACPC Alignment (using the provided templates), brain extraction, and T1w-T2w coregistration (if both exists)
                T1w, T2w = hcp.acpc_align(output_dir      = bids_output_dir,
                                        id              = bids_id,
                                        T1w             = T1w,
                                        T2w             = T2w,
                                        T1w_template    = Image(filename = args.t1w_acpc_img),
                                        T2w_template    = Image(filename = args.t2w_acpc_img),
                                        BrainSize       = args.brain_size,
                                        logfile         = logfile)
            

            if T1w:
                img_ent                 = entities.copy()
                img_ent['modality']     = 'T1w'
                img_ent['desc']         = 'brain'
                
                if args.t1w_type == 'mpnrage' or args.t1w_type == 'MPnRAGE':
                    img_ent['acq'] = 'MPnRAGE'
                if args.t1w_type == 'mp2rage' or args.t1w_type == 'MP2RAGE':
                    img_ent['acq'] = 'MP2RAGE'
                
                mask_ent            = img_ent.copy()
                denoise_ent         = img_ent.copy()
                noisemap_ent        = img_ent.copy()
                gibbs_ent           = img_ent.copy()
                bias_ent            = img_ent.copy()
                
                mask_ent['desc']    = 'brain-mask'
                denoise_ent['desc'] = 'Denoised'
                noisemap_ent['desc']= 'NoiseMap'
                gibbs_ent['desc']   = 'GibbsRingingCorrected'
                bias_ent['desc']    = 'BiasFieldCorrected'
                             
                brain_mask_t1w  = Image(filename = writing.build_path(mask_ent, Image_pattern))
                denoise_img     = Image(filename = writing.build_path(denoise_ent, Image_pattern))
                noisemap        = Image(filename = writing.build_path(noisemap_ent, Image_pattern))
                gibbs_img       = Image(filename = writing.build_path(gibbs_ent, Image_pattern))
                bias_img        = Image(filename = writing.build_path(bias_ent, Image_pattern))
                
                print("Working on " + img_ent['modality'] + " image", flush=True)
                
                #First, create T1w brain mask
                if not os.path.exists(brain_mask.filename):
                    if args.verbose:
                        print("\tMasking image...", flush=True)
                        
                    mask.mask_image(input                = T1w,
                                    mask                 = brain_mask_t1w,
                                    algo                 = args.mask_method,
                                    nthreads             = args.nthreads,
                                    ref_img              = args.t1w_mask_template,
                                    ref_mask             = args.t1w_mask_template_mask,
                                    antspynet_modality   = args.antspynet_modality,
                                    logfile              = logfile)
                    
                if not os.path.exists(bias_img.filename):
                    if not os.path.exists(denoise_img.filename):
                        if args.verbose:
                            print("\tDenoising image...", flush = True)
                            
                        denoise_img = denoise.denoise_image(input_img     = T1w,
                                                            output_file   = denoise_img.filename,
                                                            method        = args.denoise_method,
                                                            noise_map     = noisemap.filename,
                                                            nthreads      = args.nthreads)
                        if args.verbose:
                            print("\tDenoising Successful", flush = True)
                            print(flush = True)
                            
                    if not os.path.exists(gibbs_img.filename):
                        if args.verbose:
                            print("\tCorrecting Gibbs Ringing...", flush = True)
                    
                        gibbs_img = degibbs.gibbs_ringing_correction(input_img    = denoise_img,
                                                                     output_file  = gibbs_img.filename,
                                                                     method       = args.gibbs_correction_method,
                                                                     nthreads     = args.nthreads)
                        if args.verbose:
                            print("\tGibbs Ringing Correction Successful", flush = True)
                            print(flush = True)
                            
                    if not os.path.exists(bias_img.filename):
                        if args.verbose:
                            print("\tCorrecting Bias Field...", flush = True)
                            
                        bias_img = biascorrect.biasfield_correction(input_img   = gibbs_img,
                                                                    output_file = bias_img.filename, 
                                                                    method      = "ants", 
                                                                    mask_img    = brain_mask_t1w, 
                                                                    nthreads    = args.nthreads, 
                                                                    iterations  = 1)
                        if args.verbose:
                            print("\tBias Field Correction Successful", flush = True)

                        if(args.sharpen_images):
                            if args.verbose:
                                print("\tSharpening image contrast", flush=True)
                            CMD = "ImageMath 3 " + bias_img.filename + " Sharpen " + bias_img.filename
                            subprocess.run([CMD], shell=True, stdout=logfile)
                            
                            if args.verbose:
                                print("\tSharpening Successful", flush = True)
                                
                T1w_proc = bias_img
                
                if args.cleanup:
                    if os.path.exists(denoise_img.filename):
                        os.remove(denoise_img.filename)
                    if os.path.exists(gibbs_img.filename):
                        os.remove(gibbs_img.filename)
                    if os.path.exists(noisemap.filename):
                        os.remove(noisemap.filename)
                    
            else:
                T1w_proc       = None
            
            if T2w:
                img_ent                 = entities.copy()
                img_ent['modality']     = 'T2w'
                img_ent['desc']         = 'brain'
                            
                mask_ent            = img_ent.copy()
                denoise_ent         = img_ent.copy()
                noisemap_ent        = img_ent.copy()
                gibbs_ent           = img_ent.copy()
                bias_ent            = img_ent.copy()
                
                mask_ent['desc']    = 'brain-mask'
                denoise_ent['desc'] = 'Denoised'
                noisemap_ent['desc']= 'NoiseMap'
                gibbs_ent['desc']   = 'GibbsRingingCorrected'
                bias_ent['desc']    = 'BiasFieldCorrected'
                            
                
                brain_mask_t2w  = Image(filename = writing.build_path(mask_ent, Image_pattern))
                denoise_img     = Image(filename = writing.build_path(denoise_ent, Image_pattern))
                noisemap        = Image(filename = writing.build_path(noisemap_ent, Image_pattern))
                gibbs_img       = Image(filename = writing.build_path(gibbs_ent, Image_pattern))
                bias_img        = Image(filename = writing.build_path(bias_ent, Image_pattern))
                
                print("Working on " + img_ent['modality'] + " image", flush=True)
                
                #First, create T2w brain mask
                if not os.path.exists(brain_mask_t2w.filename):
                    if args.verbose:
                        print("\tMasking image...", flush=True)
                        
                    mask.mask_image(input                = T2w,
                                    mask                 = brain_mask_t2w,
                                    algo                 = args.mask_method,
                                    nthreads             = args.nthreads,
                                    ref_img              = args.t1w_mask_template,
                                    ref_mask             = args.t1w_mask_template_mask,
                                    antspynet_modality   = args.antspynet_modality,
                                    logfile              = logfile)
                    
                if not os.path.exists(bias_img.filename):
                    if not os.path.exists(denoise_img.filename):
                        if args.verbose:
                            print("\tDenoising image...", flush = True)
                            
                        denoise_img = denoise.denoise_image(input_img     = T2w,
                                                            output_file   = denoise_img.filename,
                                                            method        = args.denoise_method,
                                                            noise_map     = noisemap.filename,
                                                            nthreads      = args.nthreads)
                        if args.verbose:
                            print("\tDenoising Successful", flush = True)
                            print(flush = True)
                            
                    if not os.path.exists(gibbs_img.filename):
                        if args.verbose:
                            print("\tCorrecting Gibbs Ringing...", flush = True)
                    
                        gibbs_img = degibbs.gibbs_ringing_correction(input_img    = denoise_img,
                                                                     output_file  = gibbs_img.filename,
                                                                     method       = args.gibbs_correction_method,
                                                                     nthreads     = args.nthreads)
                        if args.verbose:
                            print("\tGibbs Ringing Correction Successful", flush = True)
                            print(flush = True)
                            
                    if not os.path.exists(bias_img.filename):
                        if args.verbose:
                            print("\tCorrecting Bias Field...", flush = True)
                            
                        bias_img = biascorrect.biasfield_correction(input_img   = gibbs_img,
                                                                    output_file = bias_img.filename, 
                                                                    method      = "ants", 
                                                                    mask_img    = brain_mask_t2w, 
                                                                    nthreads    = args.nthreads, 
                                                                    iterations  = 1)
                        if args.verbose:
                            print("\tBias Field Correction Successful", flush = True)

                        if(args.sharpen_images):
                            if args.verbose:
                                print("\tSharpening image contrast", flush=True)
                            CMD = "ImageMath 3 " + bias_img.filename + " Sharpen " + bias_img.filename
                            subprocess.run([CMD], shell=True, stdout=logfile)
                            
                            if args.verbose:
                                print("\tSharpening Successful", flush = True)
                                
                T2w_proc       = bias_img
                
                if args.cleanup:
                    if os.path.exists(denoise_img.filename):
                        os.remove(denoise_img.filename)
                    if os.path.exists(gibbs_img.filename):
                        os.remove(gibbs_img.filename)
                    if os.path.exists(noisemap.filename):
                        os.remove(noisemap.filename)
                
            else:
                T2w_proc       = None
            
            
            #Coregister the images if both exist
            if T1w_proc and T2w_proc:
                if args.verbose:
                    print("\tCoregistering T1w and T2w images")
                    print(flush=True)

                T1w_coreg, T2w_coreg = hcp.coregister_images(output_dir       = bids_output_dir,
                                                            id                = bids_id,
                                                            T1w               = T1w_proc,
                                                            T2w               = T2w_proc,
                                                            infant_mode       = args.infant_mode,
                                                            brain_size        = args.brain_size,
                                                            nthreads          = args.nthreads,
                                                            logfile           = logfile)
                if args.verbose:
                    print("Finished coregistering T1w and T2w images")
                    print(flush=True)
                    
                #Update the mask
                img_to_mask = T1w_coreg
                ref_img     = args.t1w_mask_template
                ref_mask    = args.t1w_mask_template_mask
                
                if args.infant_mode:
                    args.antspynet_modality = "t2infant"
                    img_to_mask = T2w_coreg
                    ref_img     = args.t2w_mask_template
                    ref_mask    = args.t2w_mask_template_mask
                
                brain_mask = Image(filename = os.path.join(bids_output_dir, bids_id+"_desc-brain-mask.nii.gz"),
                                json     = os.path.join(bids_output_dir, bids_id+"_desc-brain-mask.json"))
                
                create_dataset_json.create_bids_sidecar_json(image = brain_mask, 
                                                            data = {"Description": "Brain Mask",
                                                                    "Sources": img_to_mask.filename,
                                                                    "SkullStripped": True,
                                                                    "SkllStrippingMethod": args.mask_method})
                if not os.path.exists(brain_mask.filename):
                    if args.verbose:
                        print("Updating Brain Mask", flush=True)
                
                    mask.mask_image(input                = img_to_mask,
                                    mask                 = brain_mask,
                                    algo                 = args.mask_method,
                                    nthreads             = args.nthreads,
                                    ref_img              = ref_img,
                                    ref_mask             = ref_mask,
                                    antspynet_modality   = args.antspynet_modality,
                                    logfile              = logfile)
                    if args.verbose:
                        print("Finished updating the brain mask", flush=True)
                        print(flush=True)

                T1w_proc      = T1w_coreg
                T2w_proc      = T2w_coreg
            
            if T1w_proc:
                
                T1w_preproc.copy_image(T1w_proc, datatype="float32")

                create_dataset_json.create_bids_sidecar_json(image = T1w_preproc, 
                                                             data  = {"Modality": "T1w",
                                                                    "Description": "Preprocessed T1w Image",
                                                                    "Sources": T1w.filename,
                                                                    "SkullStripped": True,
                                                                    "SkullStrippingMethod": args.mask_method,
                                                                    "Denoised": True,
                                                                    "DenoisingMethod": args.denoise_method,
                                                                    "GibbsCorrected": True,
                                                                    "GibbsCorrectionMethod": args.gibbs_correction_method,
                                                                    "BiasCorrected": True,
                                                                    "BiasCorrectionMethod": args.biasfield_correction_method,
                                                                    "Sharpened": args.sharpen_images})            
            if T2w_proc:
               
                T2w_preproc.copy_image(T2w_proc, datatype="float32")
                create_dataset_json.create_bids_sidecar_json(image = T2w_preproc, 
                                                             data  = {"Modality": "T2w",
                                                                    "Description": "Preprocessed T2w Image",
                                                                    "Sources": T2w.filename,
                                                                    "SkullStripped": True,
                                                                    "SkullStrippingMethod": args.mask_method,
                                                                    "Denoised": True,
                                                                    "DenoisingMethod": args.denoise_method,
                                                                    "GibbsCorrected": True,
                                                                    "GibbsCorrectionMethod": args.gibbs_correction_method,
                                                                    "BiasCorrected": True,
                                                                    "BiasCorrectionMethod": args.biasfield_correction_method,
                                                                    "Sharpened": args.sharpen_images})
                
            if not brain_mask.exists():   
                if T1w_proc and not T2w_proc:  
                    create_dataset_json.create_bids_sidecar_json(image = brain_mask, 
                                                                 data  = {"Description": "Brain Mask",
                                                                        "Sources": T1w.filename,
                                                                        "SkullStripped": True,
                                                                        "SkllStrippingMethod": args.mask_method})
                    brain_mask.copy_image(brain_mask_t1w, datatype="uint8")
                    
                    
                elif not T1w_proc and T2w_proc:
                    create_dataset_json.create_bids_sidecar_json(image = brain_mask, 
                                                                data = {"Description": "Brain Mask",
                                                                        "Sources": T2w.filename,
                                                                        "SkullStripped": True,
                                                                        "SkllStrippingMethod": args.mask_method})
                    
                    brain_mask.copy_image(brain_mask_t2w, datatype="uint8")

            
            #Cleanup the files  
            if args.cleanup:
                if args.verbose:
                    print("Cleaning up files", flush=True)
                    
                if T1w_proc:
                    T1w_proc.remove()
                if brain_mask_t1w:
                    brain_mask_t1w.remove()
                if brain_mask_t2w:
                    brain_mask_t2w.remove()
                if T2w_proc:
                    T2w_proc.remove()
                if os.path.exists(os.path.join(bids_output_dir, bids_id+"_desc-BiasFieldCorrected_T1w.nii.gz")):
                    os.remove(os.path.join(bids_output_dir, bids_id+"_desc-BiasFieldCorrected_T1w.nii.gz"))
                if os.path.exists(os.path.join(bids_output_dir, bids_id+"_desc-BiasFieldCorrected_T2w.nii.gz")):
                    os.remove(os.path.join(bids_output_dir, bids_id+"_desc-BiasFieldCorrected_T2w.nii.gz"))
    
                if args.verbose:
                    print("Finished cleaning up files", flush=True)
                    print(flush=True)

            
        if args.to_standard:
            if args.verbose:
                print("Running Registration to Standard Space")

            registration_patterns = os.path.join(args.bids_dir, "derivatives", args.standard_registration_dir, "sub-{subject}[/ses-{session}]", "anat",)
            out_dir               = writing.build_path(entities, registration_patterns)

            if not os.path.exists(out_dir):
                os.makedirs(out_dir)

            if T1w_preproc.exists():
                out_base = os.path.join(out_dir, bids_id+"_desc-ANTs_space-"+args.standard_space+"_")

                if args.t1w_type.lower() == 'mpnrage':
                    out_base = os.path.join(out_dir, bids_id+"_acq-MPnRAGE_space-"+args.standard_space+"_desc-ANTsNonlin_")
            
                nonlinreg(input          = T1w_preproc,
                            ref          = Image(filename=args.standard_template), 
                            mask         = Image(filename=args.standard_template_mask),
                            out_xfm      = out_base+"FwdTransform.nii.gz", 
                            out_xfm_base = out_base,
                            nthreads     = args.nthreads, 
                            method       = args.to_standard_method)
            
        
        if args.verbose:
            print("Anatomical Processing Successful")
            print("")
        
                
        return T1w_preproc, T2w_preproc, brain_mask
            
        
if __name__ == "__main__":
    anatproc = AnatomicalPrepPipeline()
    anatproc.run()
        
        
        
        
        
        
        
        
        







