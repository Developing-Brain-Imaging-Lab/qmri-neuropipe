import os,sys, shutil, json, argparse, copy
import nibabel as nib

from bids.layout import writing
from core.utils.io import Image

import core.utils.mask as mask
import core.workflows.prep_rawdata as raw_proc
import core.workflows.denoise_degibbs as img_proc


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
                            choices=['BET', 'MRTRIX', 'ANTS', 'ANTSPYNET'],
                            default='BET')

        parser.add_argument('--anat_ants_mask_template',
                            type=str,
                            help='Image to use for registration based skull-stripping',
                            default=os.environ['FSLDIR']+'/data/standard/MNI152_T1_1mm.nii.gz')

        parser.add_argument('--anat_ants_mask_template_mask',
                            type=str,
                            help='Brain mask to use for registration based skull-stripping',
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

        if args.verbose:
            print('Running Anatomical Preparation Pipeline')

        t1w, t2w = raw_proc.prep_anat_rawdata(bids_id              = bids_id,
                                              bids_rawdata_dir     = bids_rawdata_dir,
                                              bids_derivative_dir  = bids_derivative_dir,
                                              nthreads             = args.nthreads,
                                              verbose              = args.verbose)


        #Create Brain Mask
        brain_mask = Image(file=os.path.join(bids_derivative_dir, 'anat/', bids_id+'_desc-brain_mask.nii.gz'))
        if t1w and t2w:
            if args.anat_antspynet_modality=='infant':
                args.anat_antspynet_modality = 't1t2infant'
                input_img = [t1w, t2w]
            else:
                input_img = t1w
        elif t1w:
            input_img = t1w

            if args.anat_antspynet_modality=='infant':
                args.anat_antspynet_modality = 't1infant'

        elif t2w:
            input_img = t2w

            if args.anat_antspynet_modality=='infant':
                args.anat_antspynet_modality = 't2infant'

        else:
            print('Anatomical Images do not exist!')
            exit()

        if not os.path.exists(brain_mask._get_filename()):
            mask.mask_image(input_img            = input_img,
                            output_mask          = brain_mask,
                            method               = args.anat_mask_method,
                            nthreads             = args.nthreads,
                            ref_img              = args.anat_ants_mask_template,
                            ref_mask             = args.anat_ants_mask_template_mask,
                            antspynet_modality   = args.anat_antspynet_modality)

        #Denoise and Degibbs raw data
        biascorr_t1w = None
        biascorr_t2w = None
        if t1w:
            t1w = img_proc.denoise_degibbs(img             = t1w,
                                           working_dir     = os.path.join(bids_derivative_dir, 'anat/'),
                                           suffix          = 'T1w',
                                           mask_img        = brain_mask,
                                           denoise_method  = args.anat_denoise_method,
                                           gibbs_method    = args.anat_gibbs_correction_method,
                                           nthreads        = args.nthreads,
                                           verbose         = args.verbose)

            biascorr_t1w = img_proc.perform_bias_correction(img         = t1w,
                                                            working_dir = os.path.join(bids_derivative_dir, 'anat/'),
                                                            suffix      = 'T1w',
                                                            mask_img    = brain_mask,
                                                            method      = args.anat_biasfield_correction_method,
                                                            nthreads    = args.nthreads,
                                                            verbose     = args.verbose)


        if t2w:
            t2w = img_proc.denoise_degibbs(img             = t2w,
                                           working_dir     = os.path.join(bids_derivative_dir, 'anat/'),
                                           suffix          = 'T2w',
                                           mask_img        = brain_mask,
                                           denoise_method  = args.anat_denoise_method,
                                           gibbs_method    = args.anat_gibbs_correction_method,
                                           nthreads        = args.nthreads,
                                           verbose         = args.verbose)

            biascorr_t2w = img_proc.perform_bias_correction(img         = t2w,
                                                            working_dir = os.path.join(bids_derivative_dir, 'anat/'),
                                                            suffix      = 'T2w',
                                                            mask_img    = brain_mask,
                                                            method      = args.anat_biasfield_correction_method,
                                                            nthreads    = args.nthreads,
                                                            verbose     = args.verbose)

        return biascorr_t1w, biascorr_t2w, brain_mask
