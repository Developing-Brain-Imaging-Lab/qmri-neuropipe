import os, copy

from bids.layout import writing, parse_file_entities

from core.utils.io import Image
import core.utils.denoise as denoise
import core.utils.gibbs_correction as degibbs
import core.utils.biascorrect as biascorr

def denoise_degibbs(input_img, working_dir, suffix, denoise_method="mrtrix", gibbs_method="mrtrix", mask_img=None, output_noise_map=True, noise_model="Rician", nthreads=1, verbose=False, debug=False):

    parsed_filename = parse_file_entities(input_img.filename)

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

    denoised_img = copy.deepcopy(input_img)
    denoised_img.filename = writing.build_path(entities, filename_patterns)

    noise_map = None
    if output_noise_map:
        noise_map = Image(filename = writing.build_path(noisemap_entities, filename_patterns))

    if not os.path.exists(working_dir):
        os.makedirs(working_dir)

    if not denoised_img.exists():
        if verbose or debug:
            print('Performing Noise Correction...')

        denoised_img = denoise.denoise_image(input_img     = input_img,
                                             output_file   = denoised_img.filename,
                                             method        = denoise_method,
                                             mask          = mask_img,
                                             noise_map     = noise_map,
                                             noise_model   = noise_model,
                                             nthreads      = nthreads,
                                             debug         = debug)
        

    entities['desc'] = 'GibbsRinging'
    degibbs_img = copy.deepcopy(denoised_img)
    degibbs_img.filename = writing.build_path(entities, filename_patterns)
    
    ###GIBBS RINGING CORRECTION ###
    if not degibbs_img.exists():
        if verbose or debug:
            print('Performing Gibbs Ringing Correction...')
        degibbs_img = degibbs.gibbs_ringing_correction(input_img      = denoised_img,
                                                       output_file    = degibbs_img.filename,
                                                       method         = gibbs_method,
                                                       nthreads       = nthreads, 
                                                       debug          = debug)
        
    return degibbs_img


def perform_biasfield_correction(input_img, working_dir, suffix, method="ants", mask_img=None, nthreads=1, iterations=1, verbose=False, debug=False):

    if not os.path.exists(working_dir):
        os.makedirs(working_dir)

    parsed_filename = parse_file_entities(input_img.filename)

    entities = {
    'extension': '.nii.gz',
    'subject': parsed_filename.get('subject'),
    'session': parsed_filename.get('session'),
    'suffix':  suffix,
    'desc': 'BiasFieldCorrected'
    }

    filename_patterns = working_dir + '/sub-{subject}[_ses-{session}][_desc-{desc}]_{suffix}{extension}'

    biascorr_img = copy.deepcopy(input_img)
    biascorr_img.filename = writing.build_path(entities, filename_patterns)

    if not biascorr_img.exists():
        if verbose or debug:
            print('Performing Bias-Field Correction...')

        biascorr_img = biascorr.biasfield_correction(input_img    = input_img,
                                                     output_file  = biascorr_img.filename,
                                                     mask_img     = mask_img,
                                                     method       = method,
                                                     nthreads     = nthreads,
                                                     iterations   = iterations,
                                                     debug        = debug)
        
    return biascorr_img
