import os,sys, shutil, json, argparse, copy, subprocess
import nibabel as nib

from bids.layout import writing
from core.utils.io import Image
import core.utils.mask as mask

import core.anatomical.workflows.prep_rawdata as raw_proc
import core.anatomical.workflows.preprocess as anat_proc
import core.anatomical.workflows.hcp_process as hcp

import core.utils.denoise as denoise

import core.registration.registration as reg_tools
import core.segmentation.segmentation as seg_tools
import core.anatomical.workflows.compute_synthetic as compute_synthetic


class AnatomicalPrepPipeline:

    def __init__(self, verbose=False):
        if verbose:
            print('Creating Anatomical Preprocessing Pipeline')

    def run(self):
        # parse commandline
        parser = argparse.ArgumentParser()

        parser.add_argument('--load_json',
                    type=str, help='Load settings from file in json format. Command line options are overriden by values in file.', default=None)


        parser.add_argument('--bids_dir',
                            type=str,
                            help='BIDS Data Directory')

        parser.add_argument('--bids_rawdata_dir',
                            type=str, help='BIDS RAWDATA Directory',
                            default='rawdata')

        parser.add_argument('--bids_t1w_dir',
                            type=str, help='BIDS T1w RAWDATA Directory Basename',
                            default='anat')

        parser.add_argument('--bids_t2w_dir',
                            type=str, help='BIDS T2w RAWDATA Directory Basename',
                            default='anat')

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
                            default=0)
                            
        parser.add_argument('--infant_mode',
                            type=bool,
                            help='Infant Mode for Processing',
                            default=False)
                            
        parser.add_argument('--brain_size',
                            type=str,
                            help='Estimate of Brain size (used for robustfov)',
                            default="150")
                            
        parser.add_argument('--sharpen_images',
                            type=bool,
                            help='Sharpen anatomical images using Laplacian sharpening filter',
                            default=False)
                    
        parser.add_argument('--anat_cleanup',
                            type=bool,
                            help='Cleanup Anatomical Image Files',
                            default=False)

        parser.add_argument('--anat_t1w_reorient_img',
                            type=str,
                            help='Image to use to reorient/correct header direction for T1w images',
                            default=None)

        parser.add_argument('--anat_t2w_reorient_img',
                            type=str,
                            help='Image to use to reorient/correct header direction for T2w images',
                            default=None)

        parser.add_argument('--anat_t1w_type',
                            type=str,
                            help='Type of T1w Acquisition',
                            choices = ['t1w', 'mp2rage', 'mpnrage'],
                            default='t1w')

        parser.add_argument('--anat_denoise_method',
                            type=str,
                            help='Method for Denoising DWIs',
                            choices=['dipy-nlmeans'],
                            default='dipy-nlmeans')

        parser.add_argument('--anat_gibbs_correction_method',
                            type=str,
                            help='Method for Gibbs Ringing Correction',
                            choices=['mrtrix', 'dipy'],
                            default='mrtrix')

        parser.add_argument('--anat_biasfield_correction_method',
                            type=str,
                            help='Method for Gibbs Ringing Correction',
                            choices=['mrtrix', 'fsl', 'N4'],
                            default='N4')

        parser.add_argument('--anat_mask_method',
                            type=str,
                            help='Skull-stripping Algorithm',
                            choices=['bet', 'hd-bet', 'mrtrix', 'ants', 'antspynet'],
                            default='bet')

        parser.add_argument('--anat_t1w_ants_mask_template',
                            type=str,
                            help='Image to use for registration based skull-stripping for T1w',
                            default=os.environ['FSLDIR']+'/data/standard/MNI152_T1_1mm.nii.gz')

        parser.add_argument('--anat_t1w_ants_mask_template_mask',
                            type=str,
                            help='Brain mask to use for registration based skull-stripping',
                            default=os.environ['FSLDIR']+'/data/standard/MNI152_T1_1mm_brain_mask.nii.gz')
                            
        parser.add_argument('--anat_t2w_ants_mask_template',
                            type=str,
                            help='Image to use for registration based skull-stripping for T2w',
                            default=os.environ['FSLDIR']+'/data/standard/MNI152_T1_1mm.nii.gz')

        parser.add_argument('--anat_t2w_ants_mask_template_mask',
                            type=str,
                            help='Brain mask to use for registration based skull-stripping for T2w',
                            default=os.environ['FSLDIR']+'/data/standard/MNI152_T1_1mm_brain_mask.nii.gz')
                            
        parser.add_argument('--anat_wmseg',
                            type=str,
                            help='White matter segmentation file to use for BBR coregistration',
                            default=None)

        parser.add_argument('--anat_antspynet_modality',
                            type=str,
                            help='ANTsPyNet modality/network name',
                            default='t1')

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

        id_patterns = "sub-{subject}[_ses-{session}]"
        rawdata_patterns = args.bids_dir + "/" + args.bids_rawdata_dir + "/sub-{subject}[/ses-{session}]/"
        derivative_patterns = args.bids_dir + "/derivatives/" + args.bids_pipeline_name + "/sub-{subject}[/ses-{session}]/"

        bids_id             = writing.build_path(entities, id_patterns)
        bids_rawdata_dir    = writing.build_path(entities, rawdata_patterns)
        bids_derivative_dir = writing.build_path(entities, derivative_patterns)
        bids_output_dir     = os.path.join(bids_derivative_dir, "anat")
        

            
        if not os.path.exists(bids_output_dir):
            os.makedirs(bids_output_dir)
            
        logfile     = open(os.path.join(bids_output_dir, "QMRI-NeuroPipe_AnatomicalProcessing_Log.txt"), 'w')
        sys.stdout  = logfile
        
        T1w, T2w = raw_proc.prep_anat_rawdata(bids_id              = bids_id,
                                              bids_rawdata_dir     = bids_rawdata_dir,
                                              bids_t1w_dir         = args.bids_t1w_dir,
                                              bids_t2w_dir         = args.bids_t2w_dir,
                                              t1w_type             = args.anat_t1w_type,
                                              verbose              = args.verbose)
        if args.verbose:
            print("#######################################", flush=True)
            print("Running Anatomical Preparation Pipeline", flush=True)
            print(flush=True)

        #First, run ACPC Alignment (using the provided templates), brain extraction, and T1w-T2w coregistration (if both exists)
        T1w_acpc, T2w_acpc = hcp.acpc_align(output_dir      = bids_output_dir,
                                            id              = bids_id,
                                            T1w             = T1w,
                                            T2w             = T2w,
                                            T1w_template    = Image(file = args.anat_t1w_reorient_img),
                                            T2w_template    = Image(file = args.anat_t2w_reorient_img),
                                            BrainSize       = args.brain_size,
                                            logfile         = logfile)
        
        T1w_brain      = Image(file = os.path.join(bids_output_dir, bids_id+"_desc-brain_T1w.nii.gz"))
        T1w_brain_mask = Image(file = os.path.join(bids_output_dir, bids_id+"_desc-brain-mask_T1w.nii.gz"))
        T2w_brain      = Image(file = os.path.join(bids_output_dir, bids_id+"_desc-brain_T2w.nii.gz"))
        T2w_brain_mask = Image(file = os.path.join(bids_output_dir, bids_id+"_desc-brain-mask_T2w.nii.gz"))
        
        brain_mask     = Image(file = os.path.join(bids_output_dir, bids_id+"_desc-brain-mask.nii.gz"))
        
        
        T1w_denoise    = Image(file = os.path.join(bids_output_dir, bids_id+"_desc-Denoised_T1w.nii.gz"))
        T1w_noise_map  = Image(file = os.path.join(bids_output_dir, bids_id+"_desc-NoiseMap_T1w.nii.gz"))
        T1w_gibbs      = Image(file = os.path.join(bids_output_dir, bids_id+"_desc-GibbsRingingCorrected_T1w.nii.gz"))
        T1w_bias       = Image(file = os.path.join(bids_output_dir, bids_id+"_desc-BiasFieldCorrected_T1w.nii.gz"))
        
        T2w_denoise    = Image(file = os.path.join(bids_output_dir, bids_id+"_desc-Denoised_T2w.nii.gz"))
        T2w_noise_map  = Image(file = os.path.join(bids_output_dir, bids_id+"_desc-NoiseMap_T2w.nii.gz"))
        T2w_gibbs      = Image(file = os.path.join(bids_output_dir, bids_id+"_desc-GibbsRingingCorrected_T2w.nii.gz"))
        T2w_bias       = Image(file = os.path.join(bids_output_dir, bids_id+"_desc-BiasFieldCorrected_T2w.nii.gz"))
        
    
        
        if T1w_acpc and not os.path.exists(T1w_bias._get_filename()):
            #Denoise, correct for Gibbs ringing, and BiasField correct
            T1w_robustroi       = Image(file=os.path.join(bids_output_dir, "T1w_robustroi.nii.gz"))
            T1w_robustroi_mask  = Image(file=os.path.join(bids_output_dir, "T1w_robustroi_mask.nii.gz"))
            T1w_roi2full_mat    = os.path.join(bids_output_dir, "skullstrip_roi2full.mat")

            CMD = "robustfov -i " + T1w_acpc._get_filename() \
                  + " -m " + T1w_roi2full_mat \
                  + " -r " + T1w_robustroi._get_filename() \
                  + " -b " + str(args.brain_size)
            subprocess.run([CMD], shell=True, stdout=logfile)
                  
            #Now run the mask
            if not os.path.exists(T1w_brain_mask._get_filename()):
                if args.verbose:
                    print("Masking T1w image...", flush=True)
                mask.mask_image(input_img            = T1w_robustroi,
                                output_mask          = T1w_robustroi_mask,
                                method               = args.anat_mask_method,
                                nthreads             = args.nthreads,
                                ref_img              = args.anat_t1w_ants_mask_template,
                                ref_mask             = args.anat_t1w_ants_mask_template_mask,
                                antspynet_modality   = args.anat_antspynet_modality,
                                logfile              = logfile)
                if args.verbose:
                    print("Successful T1w Masking", flush=True)
                    print(flush=True)
                    
                CMD = "applywarp --rel --interp=nn -i " + T1w_robustroi_mask._get_filename() \
                          + " -r " + T1w_acpc._get_filename() \
                          + " --premat=" + T1w_roi2full_mat \
                          + " -o " + T1w_brain_mask._get_filename()
                subprocess.run([CMD], shell=True, stdout=logfile)
                          
                #Create brain image
                if args.verbose:
                    print("Applying mask to T1w image...", flush=True)
                mask.apply_mask(input_img   = T1w_acpc,
                                mask_img    = T1w_brain_mask,
                                output_img  = T1w_brain)
                if args.verbose:
                    print("Finished applying mask to T1w image", flush=True)
                #Clean up the ROBUSTFOV files
                                
        else:
            T1w_brain       = None
            T1w_brain_mask  = None
            
            
        if T2w_acpc not os.path.exists(T2w_bias._get_filename()):
            T2w_robustroi       = Image(file=os.path.join(bids_output_dir, "T2w_robustroi.nii.gz"))
            T2w_robustroi_mask  = Image(file=os.path.join(bids_output_dir, "T2w_robustroi_mask.nii.gz"))
            T2w_roi2full_mat    = os.path.join(bids_output_dir, "skullstrip_roi2full.mat")

            CMD = "robustfov -i " + T2w_acpc._get_filename() \
                  + " -m " + T2w_roi2full_mat \
                  + " -r " + T2w_robustroi._get_filename() \
                  + " -b " + str(args.brain_size)
            subprocess.run([CMD], shell=True, stdout=logfile)
                  
            #Now run the mask
            if not os.path.exists(T2w_brain_mask._get_filename()):
                if args.verbose:
                    print("Masking T2w image...", flush=True)
                mask.mask_image(input_img            = T2w_robustroi,
                                output_mask          = T2w_robustroi_mask,
                                method               = args.anat_mask_method,
                                nthreads             = args.nthreads,
                                ref_img              = args.anat_t2w_ants_mask_template,
                                ref_mask             = args.anat_t2w_ants_mask_template_mask,
                                antspynet_modality   = args.anat_antspynet_modality,
                                logfile              = logfile)
                if args.verbose:
                    print("Successful T2w Masking", flush=True)
                    print(flush=True)
                
                #Convert back to full ROI
                CMD = "applywarp --rel --interp=nn -i " + T2w_robustroi_mask._get_filename() \
                          + " -r " + T2w_acpc._get_filename() \
                          + " --premat=" + T2w_roi2full_mat \
                          + " -o " + T2w_brain_mask._get_filename()
                subprocess.run([CMD], shell=True, stdout=logfile)
                          
                #Create brain image
                if args.verbose:
                    print("Applying mask to T2w image...", flush=True)
                mask.apply_mask(input_img   = T2w_acpc,
                                mask_img    = T2w_brain_mask,
                                output_img  = T2w_brain)
                if args.verbose:
                    print("Finished applying mask to T2w image", flush=True)
                                
        else:
            T2w_brain       = None
            T2w_brain_mask  = None
        
        
        #Coregister the images if both exist
        if (T1w_acpc and T2w_acpc) and not os.path.exists(T1w_bias._get_filename()):
            if args.verbose:
                print("Coregistering T1w and T2w images")
                print(flush=True)
            
            T1w_coreg, T2w_coreg = hcp.coregister_images(output_dir        = bids_output_dir,
                                                         id                = bids_id,
                                                         T1w               = T1w_acpc,
                                                         T2w               = T2w_acpc,
                                                         T1w_brain         = T1w_brain,
                                                         T2w_brain         = T2w_brain,
                                                         T1w_brain_mask    = T1w_brain_mask,
                                                         T2w_brain_mask    = T2w_brain_mask,
                                                         infant_mode       = args.infant_mode,
                                                         brain_size        = args.brain_size,
                                                         logfile           = logfile)
            if args.verbose:
                print("Finished coregistering T1w and T2w images")
                print(flush=True)
                
            #Update the mask
            img_to_mask = T1w_coreg
            ref_img     = args.anat_t1w_ants_mask_template
            ref_mask    = args.anat_t1w_ants_mask_template_mask
            
            if args.infant_mode:
                args.anat_antspynet_modality = 't2infant'
                img_to_mask = T2w_coreg
                ref_img     = args.anat_t2w_ants_mask_template
                ref_mask    = args.anat_t2w_ants_mask_template_mask
            
            if not os.path.exists(brain_mask._get_filename()):
                if args.verbose:
                    print("Updating Brain Mask", flush=True)
                mask.mask_image(input_img            = img_to_mask,
                                output_mask          = brain_mask,
                                method               = args.anat_mask_method,
                                nthreads             = args.nthreads,
                                ref_img              = ref_img,
                                ref_mask             = ref_mask,
                                antspynet_modality   = args.anat_antspynet_modality,
                                logfile              = logfile)
                if args.verbose:
                    print("Finished updating the brain mask", flush=True)
                    print(flush=True)

            T1w_acpc = T1w_coreg
            T2w_acpc = T2w_coreg
            T1w_brain_mask = brain_mask
            T2w_brain_mask = brain_mask
         
 
        if T1w_acpc and not os.path.exists(T1w_bias._get_filename()):
            if not os.path.exists(T1w_denoise._get_filename()):
                if args.verbose:
                    print("Denoising T1w...", flush = True)
                    
                T1w_denoise = denoise.denoise_image(input_img     = T1w_acpc,
                                                    output_file   = T1w_denoise._get_filename(),
                                                    method        = args.anat_denoise_method,
                                                    output_noise  = T1w_noise_map._get_filename(),
                                                    nthreads      = args.nthreads)
                if args.verbose:
                    print("Denoising Successful", flush = True)
                    print(flush = True)
                    
            if not os.path.exists(T1w_gibbs._get_filename()):
                if args.verbose:
                    print("Correcting T1w Gibbs Ringing...", flush = True)
            
                T1w_gibbs = denoise.gibbs_ringing_correction(input_img     = T1w_denoise,
                                                             output_file   = T1w_gibbs._get_filename(),
                                                             method        = "mrtrix",
                                                             nthreads      = args.nthreads)
                if args.verbose:
                    print("T1w Gibbs Ringing Correction Successful", flush = True)
                    print(flush = True)
                    
            if not os.path.exists(T1w_bias._get_filename()):
                if args.verbose:
                    print("Correcting T1w Bias Field...", flush = True)
                    
                CMD = "N4BiasFieldCorrection -d 3 -i " + T1w_gibbs._get_filename() + " -o " + T1w_bias._get_filename()
                subprocess.run([CMD], shell=True, stdout=logfile)

                if args.verbose:
                    print("T1w Bias Field Correction Successful", flush = True)

                if(args.sharpen_images):
                    if args.verbose:
                        print("Sharpening T1w image contrast", flush=True)
                    CMD = "ImageMath 3 " + T1w_bias._get_filename() + " Sharpen " + T1w_bias._get_filename()
                    subprocess.run([CMD], shell=True, stdout=logfile)
                    
                    if args.verbose:
                        print("T1w Sharpening Successful", flush = True)
        
        else:
            T1w_bias        = None
            T1w_brain_mask  = None
        
        
        if T2w_acpc and not os.path.exists(T2w_bias._get_filename():
            if not os.path.exists(T2w_denoise._get_filename()):
            
                if args.verbose:
                    print("Denoising T2w...", flush = True)
                    
                T2w_denoise = denoise.denoise_image(input_img     = T2w_acpc,
                                                    output_file   = T2w_denoise._get_filename(),
                                                    method        = args.anat_denoise_method,
                                                    output_noise  = T2w_noise_map._get_filename(),
                                                    nthreads      = args.nthreads)
  
                if args.verbose:
                    print("Denoising Successful", flush = True)
                    print(flush = True)
  
            if not os.path.exists(T2w_gibbs._get_filename()):
                if args.verbose:
                    print("Correcting T2w Gibbs Ringing...", flush = True)
                    
                T2w_gibbs = denoise.gibbs_ringing_correction(input_img     = T2w_denoise,
                                                             output_file   = T2w_gibbs._get_filename(),
                                                             method        = "mrtrix",
                                                             nthreads      = args.nthreads)
                    
                if args.verbose:
                    print("T2w Gibbs Ringing Correction Successful", flush = True)
                    print(flush = True)
                    
            if not os.path.exists(T2w_bias._get_filename()):
                if args.verbose:
                    print("Correcting T2w Bias Field...", flush = True)
                    
                CMD = "N4BiasFieldCorrection -d 3 -i " + T2w_gibbs._get_filename() + " -o " + T2w_bias._get_filename()
                subprocess.run([CMD], shell=True, stdout=logfile)

                if args.verbose:
                    print("T2w Bias Field Correction Successful", flush = True)

                if(args.sharpen_images):
                    if args.verbose:
                        print("Sharpening T2w image contrast", flush=True)
                    
                    CMD = "ImageMath 3 " + T2w_bias._get_filename() + " Sharpen " + T2w_bias._get_filename()
                    subprocess.run([CMD], shell=True, stdout=logfile)
                    
                    if args.verbose:
                        print("T2w Sharpening Successful", flush = True)
                        print(flush=True)
                    
        else:
            T2w_bias        = None
            T2w_brain_mask  = None
        

        
        if args.verbose:
            print("Anatomical Processing Successful")
            print("")
        
        
        return T1w_bias, T2w_bias, T1w_brain_mask, T2w_brain_mask
            
        
        
        
        
        
        
        
        
        
        
        







