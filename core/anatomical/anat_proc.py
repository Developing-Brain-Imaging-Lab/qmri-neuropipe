import os,sys, shutil, json, argparse, copy
import nibabel as nib

from bids.layout import writing
from core.utils.io import Image
import core.utils.mask as mask

import core.utils.workflows.prep_rawdata as raw_proc
import core.utils.workflows.denoise_degibbs as img_proc
import core.anatomical.workflows.preprocess as anat_proc

import core.registration.registration as reg_tools
import core.segmentation.segmentation as seg_tools


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
        }

        id_patterns = 'sub-{subject}[_ses-{session}]'
        rawdata_patterns = args.bids_dir + '/'+ args.bids_rawdata_dir + '/sub-{subject}[/ses-{session}]/'
        derivative_patterns = args.bids_dir + '/derivatives/' + args.bids_pipeline_name + '/sub-{subject}[/ses-{session}]/'

        bids_id             = writing.build_path(entities, id_patterns)
        bids_rawdata_dir    = writing.build_path(entities, rawdata_patterns)
        bids_derivative_dir = writing.build_path(entities, derivative_patterns)

        if not os.path.exists(bids_derivative_dir):
            os.makedirs(bids_derivative_dir)

        if args.verbose:
            print('Running Anatomical Preparation Pipeline')

        t1w, t2w = raw_proc.prep_anat_rawdata(bids_id              = bids_id,
                                              bids_rawdata_dir     = bids_rawdata_dir,
                                              bids_derivative_dir  = bids_derivative_dir,
                                              bids_t1w_dir         = args.bids_t1w_dir,
                                              bids_t2w_dir         = args.bids_t2w_dir,
                                              t1w_reorient_img     = args.anat_t1w_reorient_img,
                                              t2w_reorient_img     = args.anat_t2w_reorient_img,
                                              t1w_type             = args.anat_t1w_type,
                                              nthreads             = args.nthreads,
                                              verbose              = args.verbose)

        #Create Brain Mask
        brain_mask = Image(file=os.path.join(bids_derivative_dir, args.bids_t1w_dir, bids_id+'_desc-brain_mask.nii.gz'))
        t1w_brain_mask = Image(file=os.path.join(bids_derivative_dir, args.bids_t1w_dir, bids_id+'_desc-T1w-brain_mask.nii.gz'))
        t2w_brain_mask = Image(file=os.path.join(bids_derivative_dir, args.bids_t2w_dir, bids_id+'_desc-T2w-brain_mask.nii.gz'))
        
        if t1w == None and t2w == None:
            print('Anatomical Images do not exist!')
            exit()
        
        if t1w:
            if args.anat_antspynet_modality=='infant':
                args.anat_antspynet_modality = 't1infant'
            
            if not os.path.exists(t1w_brain_mask._get_filename()):
                mask.mask_image(input_img            = t1w,
                                output_mask          = t1w_brain_mask,
                                method               = args.anat_mask_method,
                                nthreads             = args.nthreads,
                                ref_img              = args.anat_t1w_ants_mask_template,
                                ref_mask             = args.anat_t1w_ants_mask_template_mask,
                                antspynet_modality   = args.anat_antspynet_modality)
                            
        if t2w:
            if args.anat_antspynet_modality=='infant':
                args.anat_antspynet_modality = 't2infant'
            
            if not os.path.exists(t2w_brain_mask._get_filename()):
                mask.mask_image(input_img            = t2w,
                                output_mask          = t2w_brain_mask,
                                method               = args.anat_mask_method,
                                nthreads             = args.nthreads,
                                ref_img              = args.anat_t2w_ants_mask_template,
                                ref_mask             = args.anat_t2w_ants_mask_template_mask,
                                antspynet_modality   = args.anat_antspynet_modality)

            
        #Denoise and Degibbs raw data
        biascorr_t1w = None
        biascorr_t2w = None
        if t1w != None:
            denoise_t1w = img_proc.denoise_degibbs(img             = t1w,
                                                   working_dir     = os.path.join(bids_derivative_dir,  args.bids_t1w_dir,''),
                                                   suffix          = 'T1w',
                                                   mask_img        = t1w_brain_mask,
                                                   denoise_method  = args.anat_denoise_method,
                                                   gibbs_method    = args.anat_gibbs_correction_method,
                                                   nthreads        = args.nthreads,
                                                   verbose         = args.verbose)

            biascorr_t1w = img_proc.perform_bias_correction(img         = denoise_t1w,
                                                            working_dir = os.path.join(bids_derivative_dir,  args.bids_t1w_dir,''),
                                                            suffix      = 'T1w',
                                                            mask_img    = t1w_brain_mask,
                                                            method      = args.anat_biasfield_correction_method,
                                                            nthreads    = args.nthreads,
                                                            verbose     = args.verbose)

        if t2w != None:
            denoise_t2w = img_proc.denoise_degibbs(img             = t2w,
                                                   working_dir     = os.path.join(bids_derivative_dir, args.bids_t2w_dir,''),
                                                   suffix          = 'T2w',
                                                   mask_img        = t2w_brain_mask,
                                                   denoise_method  = args.anat_denoise_method,
                                                   gibbs_method    = args.anat_gibbs_correction_method,
                                                   nthreads        = args.nthreads,
                                                   verbose         = args.verbose)

            biascorr_t2w = img_proc.perform_bias_correction(img         = denoise_t2w,
                                                            working_dir = os.path.join(bids_derivative_dir,  args.bids_t2w_dir,''),
                                                            suffix      = 'T2w',
                                                            mask_img    = t2w_brain_mask,
                                                            method      = args.anat_biasfield_correction_method,
                                                            nthreads    = args.nthreads,
                                                            verbose     = args.verbose)

        
        #Coregister the T1w and T2w if both exist
        if biascorr_t1w and biascorr_t2w:
            #First create skull-stripped images based on masks created
            t1w_masked = Image(file=os.path.join(bids_derivative_dir, args.bids_t1w_dir, bids_id+'_desc-T1w-brain.nii.gz'))
            t2w_masked = Image(file=os.path.join(bids_derivative_dir, args.bids_t2w_dir, bids_id+'_desc-T2w-brain.nii.gz'))
            
            mask.apply_mask(input_img   = biascorr_t1w,
                            mask_img    = t1w_brain_mask,
                            output_img  = t1w_masked)
                            
            mask.apply_mask(input_img   = biascorr_t2w,
                            mask_img    = t2w_brain_mask,
                            output_img  = t2w_masked)
                            
            
            #First, create wm segmentation from T1w image
            coreg_t2 = copy.deepcopy(biascorr_t2w)
            coreg_t2._set_filename(os.path.join(bids_derivative_dir, args.bids_t2w_dir, bids_id+'_space-individual-T1w_T2w.nii.gz'))
            
            
            wmseg_img = seg_tools.create_wmseg(input_img    = biascorr_t1w,
                                               output_dir   = os.path.join(bids_derivative_dir,  args.bids_t1w_dir,'wmseg'),
                                               brain_mask   = t1w_brain_mask)
                                   
            reg_tools.linear_reg(input_img      = biascorr_t2w,
                                 reference_img  = biascorr_t1w,
                                 output_matrix  = os.path.join(bids_derivative_dir, args.bids_t2w_dir, bids_id+'_space-individual-T1w_T2w.mat'),
                                 method         = 'FSL',
                                 dof            = 6,
                                 flirt_options =  '-interp sinc -searchrx -180 180 -searchry -180 180 -searchrz -180 180')

            bbr_options = ' -cost bbr -wmseg ' + wmseg_img._get_filename() + ' -schedule $FSLDIR/etc/flirtsch/bbr.sch -interp sinc -bbrtype global_abs -bbrslope 0.25 -finesearch 10 -init ' + os.path.join(bids_derivative_dir, args.bids_t2w_dir, bids_id+'_space-individual-T1w_T2w.mat')

            reg_tools.linear_reg(input_img      = biascorr_t2w,
                                 reference_img  = biascorr_t1w,
                                 output_file    = coreg_t2._get_filename(),
                                 output_matrix  = os.path.join(bids_derivative_dir, args.bids_t2w_dir, bids_id+'_space-individual-T1w_T2w.mat'),
                                 method         = 'FSL',
                                 dof            = 6,
                                 flirt_options =  bbr_options)
            
            #Apply registration to T2w
            reg_tools.apply_transform(input_img     = t2w,
                                      reference_img = biascorr_t1w
                                      output_img    = coreg_t2._get_filename(),
                                      matrix        = os.path.join(bids_derivative_dir, args.bids_t2w_dir, bids_id+'_space-individual-T1w_T2w.mat'),
                                      method        = 'FSL',
                                      flirt_options = '-interp sinc')
                                      
            os.rename(t1w_brain_mask._get_filename(), brain_mask._get_filename())
            
            denoise_t2w = img_proc.denoise_degibbs(img             = coreg_t2,
                                                   working_dir     = os.path.join(bids_derivative_dir, args.bids_t2w_dir,''),
                                                   suffix          = 'T2w',
                                                   mask_img        = brain_mask,
                                                   denoise_method  = args.anat_denoise_method,
                                                   gibbs_method    = args.anat_gibbs_correction_method,
                                                   nthreads        = args.nthreads,
                                                   verbose         = args.verbose)

            biascorr_t2w = img_proc.perform_bias_correction(img         = denoise_t2w,
                                                            working_dir = os.path.join(bids_derivative_dir,  args.bids_t2w_dir,''),
                                                            suffix      = 'T2w',
                                                            mask_img    = brain_mask,
                                                            method      = args.anat_biasfield_correction_method,
                                                            nthreads    = args.nthreads,
                                                            verbose     = args.verbose)

            os.remove(t1w_masked._get_filename())
            os.remove(t2w_masked._get_filename())
            os.remove(t2w_brain_mask._get_filename())
          
          
          
        if args.anat_t1w_type == 'mp2rage':
            anat_proc.prepocess_mp2rage(bids_id             = bids_id,
                                        bids_rawdata_dir    = bids_rawdata_dir,
                                        bids_derivative_dir = bids_derivative_dir,
                                        brain_mask          = brain_mask,
                                        reorient_img        = args.anat_t1w_reorient_img,
                                        cleanup_files       = args.anat_cleanup,
                                        nthreads            = args.nthreads,
                                        verbose             = args.verbose)


        if args.anat_t1w_type == 'mpnrage':
            anat_proc.prepocess_mpnrage(bids_id             = bids_id,
                                        bids_rawdata_dir    = bids_rawdata_dir,
                                        bids_derivative_dir = bids_derivative_dir,
                                        brain_mask          = brain_mask,
                                        reorient_img        = args.anat_t1w_reorient_img,
                                        nthreads            = args.nthreads,
                                        verbose             = args.verbose)


        if args.anat_cleanup:
            if os.path.exists(bids_derivative_dir+args.bids_t1w_dir+bids_id+'_desc-Denoised_T1w.nii.gz'):
                os.remove(bids_derivative_dir+args.bids_t1w_dir+bids_id+'_desc-Denoised_T1w.nii.gz')
            if os.path.exists(bids_derivative_dir+args.bids_t1w_dir+bids_id+'_desc-GibbsRinging_T1w.nii.gz'):
                os.remove(bids_derivative_dir+args.bids_t1w_dir+bids_id+'_desc-GibbsRinging_T1w.nii.gz')

            if os.path.exists(bids_derivative_dir+args.bids_t2w_dir+bids_id+'_desc-Denoised_T2w.nii.gz'):
                os.remove(bids_derivative_dir+args.bids_t2w_dir+bids_id+'_desc-Denoised_T2w.nii.gz')
            if os.path.exists(bids_derivative_dir+args.bids_t2w_dir+bids_id+'_desc-GibbsRinging_T2w.nii.gz'):
                os.remove(bids_derivative_dir+args.bids_t2w_dir+bids_id+'_desc-GibbsRinging_T2w.nii.gz')


        return biascorr_t1w, biascorr_t2w, brain_mask
