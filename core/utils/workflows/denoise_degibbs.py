import os,sys, shutil, json, argparse, copy
import nibabel as nib

from bids.layout import BIDSFile
from bids.layout import writing, parse_file_entities
from core.utils.io import Image, DWImage
import core.utils.denoise as denoise
import core.utils.tools as img_tools
import core.utils.biascorrect as biascorr

def denoise_degibbs(img, working_dir, suffix, denoise_method='mrtrix', gibbs_method='mrtrix', mask_img=None, nthreads=1, noise_map=True, verbose=False):

    parsed_filename = parse_file_entities(img._get_filename())

    entities = {
    'extension': '.nii.gz',
    'subject': parsed_filename.get('subject'),
    'session': parsed_filename.get('session'),
    'suffix':  suffix,
    'desc': 'Denoised'
    }

    noisemap_entities = {
    'extension': '.nii.gz',
    'subject': parsed_filename.get('subject'),
    'session': parsed_filename.get('session'),
    'suffix':  suffix,
    'desc': 'NoiseMap'
    }

    filename_patterns = working_dir + '/sub-{subject}[_ses-{session}][_desc-{desc}]_{suffix}{extension}'

    denoised_img = copy.deepcopy(img)
    denoised_img._set_filename(writing.build_path(entities, filename_patterns))

    if not os.path.exists(working_dir):
        os.makedirs(working_dir)

    if not denoised_img.exists():
        if verbose:
            print('Performing Noise Correction...')

        denoised_img = denoise.denoise_image(input_img     = img,
                                             output_file   = denoised_img._get_filename(),
                                             method        = denoise_method,
                                             output_noise  = writing.build_path(noisemap_entities, filename_patterns),
                                             mask_img      = mask_img,
                                             nthreads      = nthreads)

    entities['desc'] = 'GibbsRinging'

    degibbs_img = copy.deepcopy(denoised_img)
    degibbs_img._set_filename(writing.build_path(entities, filename_patterns))
    ###GIBBS RINGING CORRECTION ###
    if not degibbs_img.exists():
        if verbose:
            print('Performing Gibbs Ringing Correction...')
        degibbs_img = denoise.gibbs_ringing_correction(input_img      = denoised_img,
                                                       output_file    = degibbs_img._get_filename(),
                                                       method         = gibbs_method,
                                                       nthreads       = nthreads)

    return degibbs_img


def perform_bias_correction(img, working_dir, suffix, method='ants', mask_img=None, nthreads=1, verbose=False):

    if not os.path.exists(working_dir):
        os.makedirs(working_dir)

    parsed_filename = parse_file_entities(img._get_filename())

    entities = {
    'extension': '.nii.gz',
    'subject': parsed_filename.get('subject'),
    'session': parsed_filename.get('session'),
    'suffix':  suffix,
    'desc': 'BiasFieldCorrected'
    }

    filename_patterns = working_dir + '/sub-{subject}[_ses-{session}][_desc-{desc}]_{suffix}{extension}'

    biascorr_img = copy.deepcopy(img)
    biascorr_img._set_filename(writing.build_path(entities, filename_patterns))

    if not biascorr_img.exists():
        if verbose:
            print('Performing Bias-Field Correction...')

        biascorr_img = biascorr.biasfield_correction(input_img    = img,
                                                     output_file  = biascorr_img._get_filename(),
                                                     mask_img     = mask_img,
                                                     method       = method)
    return biascorr_img
