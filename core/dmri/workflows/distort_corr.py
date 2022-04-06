import os, copy

from bids.layout import writing, parse_file_entities
import core.dmri.utils.distortion_correction as distcorr

def perform_topup(dwi_image, topup_base, topup_config, dist_corr, verbose=False):

    if dist_corr == 'Topup' or dist_corr == 'Topup-Separated':

        working_dir     = os.path.dirname(topup_base)
        parsed_filename = parse_file_entities(dwi_image._get_filename())

        if not os.path.exists(working_dir):
            os.makedirs(working_dir)

        entities = {
        'extension': '.nii.gz',
        'subject': parsed_filename.get('subject'),
        'session': parsed_filename.get('session'),
        'suffix':  'fmap',
        'desc': 'Topup'
        }

        filename_patterns = working_dir + '/sub-{subject}[_ses-{session}][_desc-{desc}]_{suffix}{extension}'
        fieldmap = writing.build_path(entities, filename_patterns)

        if not os.path.exists(topup_base + '_fieldcoef.nii.gz'):

            if verbose:
                print('Performing Topup Disortion Correction')

            if dist_corr == 'Topup':
                distcorr.topup_fsl(input_dwi            = dwi_image,
                                   output_topup_base    = topup_base,
                                   config_file          = topup_config,
                                   field_output         = True)

            elif dist_corr == 'Topup-Separated':
                print('Need to implement this')

            else:
                print('Incorrect Method')
                exit()

def perform_distortion_correction(dwi_image, working_dir, fmap_ref_image=None, fmap_image=None, t1w_image=None, t2w_image=None, fmap=None, distortion_method=None, distortion_modality='t1w', linreg_method='FSL', resample_to_anat=False, nthreads=1, verbose=False):

    if distortion_method != None:

        parsed_filename = parse_file_entities(dwi_image._get_filename())

        entities = {
        'extension': '.nii.gz',
        'subject': parsed_filename.get('subject'),
        'session': parsed_filename.get('session'),
        'suffix':  'dwi',
        'desc': 'DistortionCorrected'
        }

        if distortion_method == 'Anatomical-Coregistration':
            working_dir += '/anatomical-distortion-correction'
        elif distortion_method == 'Fieldmap':
            working_dir += '/fieldmap-distortion-correction'

        if not os.path.exists(working_dir):
            os.makedirs(working_dir)

        filename_patterns   = working_dir + '/sub-{subject}[_ses-{session}][_desc-{desc}]_{suffix}{extension}'

        distcorr_file = writing.build_path(entities, filename_patterns)
        entities['extension'] = '.bvec'
        distcorr_bvec = writing.build_path(entities, filename_patterns)

        distcorr_img = copy.deepcopy(dwi_image)
        distcorr_img._set_filename(distcorr_file)
        distcorr_img._set_bvecs(distcorr_bvec)

        if not distcorr_img.exists():
            if distortion_method == 'Anatomical-Coregistration':
                if verbose:
                    print('Performing Registration-Based Distortion Correction')

                distcorr_img = distcorr.registration_method(input_dwi           = dwi_image,
                                                            working_dir         = working_dir,
                                                            distortion_modality = distortion_modality,
                                                            T1_image            = t1w_image,
                                                            T2_image            = t2w_image,
                                                            linreg_method       = linreg_method,
                                                            resample_to_anat    = resample_to_anat,
                                                            nthreads            = nthreads,
                                                            verbose             = verbose)

            if distortion_method == 'Fieldmap':
                if verbose:
                    print('Performing Fieldmap Based Distortion Correction')
                
                distcorr_img = distcorr.fugue_fsl(input_dwi         = dwi_image,
                                                  input_fm          = fmap_image,
                                                  input_fm_ref      = fmap_ref_image,
                                                  fieldmap_dir      = working_dir,
                                                  unwarpdir         =
                                                  dwellTime         =               )
                                    


        return distcorr_img

    else:
        return dwi_image
