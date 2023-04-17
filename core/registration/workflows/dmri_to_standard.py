import string, os, sys, subprocess, shutil, time, copy, argparse, json

#Neuroimaging Modules
import numpy as np
import nibabel as nib
import ants

from bids.layout import writing, parse_file_entities
from core.utils.io import Image, DWImage
import core.utils.tools as img_tools


class DiffusionNormalizationPipeline:

    def __init__(self, verbose=False):
        if verbose:
            print('qmri-neuropipe Diffusion Normalization Pipeline')

    def run(self):

        parser = argparse.ArgumentParser()

        parser.add_argument('--bids_dir',
                            type=str,
                            help='BIDS Data Directory')
        
        parser.add_argument('--bids_pipeline_name',
                    type=str, help='BIDS PIPELINE Name',
                    default='qmri-neuropipe')
    
        parser.add_argument('--load_json',
                        type=str, 
                        help='Load settings from file in json format. Command line options are overriden by values in file.',
                        default=None)
        
        parser.add_argument('--nthreads',
                        type=int,
                        help='Number of Threads',
                        default=1)
        
        parser.add_argument('--subject',
                            type=str,
                            help='Subject ID')

        parser.add_argument('--session',
                            type=str,
                            help='Subject Timepoint',
                            default=None)
        
        parser.add_argument('--standard_template',
                            type=str,
                            help='Template to use for registration',
                            default=None)
    
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
        derivative_patterns = args.bids_dir + '/derivatives/' + args.bids_pipeline_name + '/sub-{subject}[/ses-{session}]/'

        bids_id             = writing.build_path(entities, id_patterns)
        bids_derivative_dir = writing.build_path(entities, derivative_patterns)

        models_dir          = os.path.join(bids_derivative_dir, "dwi", "models" )
        registration_dir    = os.path.join(bids_derivative_dir, "dwi", 'registration-to-standard/')
        normalization_dir   = os.path.join(bids_derivative_dir, "dwi", 'normalized-to-standard/')

        if not os.path.exists(registration_dir):
            os.makedirs(registration_dir)

        if not os.path.exists(normalization_dir):
            os.makedirs(normalization_dir)


        #Use ANTS to do the registration
        dti_dir = os.path.join(models_dir, "DTI")
        fa_map  = os.path.join(dti_dir, bids_id+"_model-DTI_parameter-FA.nii.gz")

        output_base = os.path.join(registration_dir, bids_id+"_desc-RegistrationToStandard_")

        if os.path.exists(fa_map):

            cmd = "antsRegistrationSyN.sh -d 3" \
                + " -m " + fa_map \
                + " -f " + args.standard_template \
                + " -o " + output_base \
                + " -n " + str(args.nthreads) \
                + " -e 0 -j 1"
            
            os.system(cmd)


