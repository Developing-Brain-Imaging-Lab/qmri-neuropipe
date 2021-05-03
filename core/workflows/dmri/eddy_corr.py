import os,sys, shutil, json, argparse, copy
import nibabel as nib

from bids.layout import writing, parse_file_entities
import core.utils.dmri.eddy_correction as eddycorr
import core.utils.dmri.qc as dmri_qc

def perform_eddy(dwi_image, working_dir, topup_base, method='eddy', gpu=False, cuda_device=0, nthreads=1, data_shelled=True, repol=False, estimate_move_by_suscept=False, mporder=0, slspec=None, fsl_eddy_options=None, verbose=False):

    if not os.path.exists(working_dir):
        os.makedirs(working_dir)

    parsed_filename = parse_file_entities(dwi_image._get_filename())

    entities = {
    'extension': '.nii.gz',
    'subject': parsed_filename.get('subject'),
    'session': parsed_filename.get('session'),
    'suffix':  'dwi',
    'desc': 'EddyCurrentCorrected'
    }

    filename_patterns   = working_dir + '/sub-{subject}[_ses-{session}][_desc-{desc}]_{suffix}{extension}'
    outputbase_patterns = working_dir + '/sub-{subject}[_ses-{session}]'

    output_base = writing.build_path(entities, outputbase_patterns)

    eddy_file = writing.build_path(entities, filename_patterns)
    entities['extension'] = '.bvec'
    eddy_bvec = writing.build_path(entities, filename_patterns)

    eddycorrected_img = copy.deepcopy(dwi_image)
    eddycorrected_img._set_filename(eddy_file)
    eddycorrected_img._set_bvecs(eddy_bvec)

    if not eddycorrected_img.exists():

        #First, create Index and Acqparams based on DWI and JSON
        if( ( eddycorrected_img._get_index() == None) or ( eddycorrected_img._get_acqparams() == None ) ):
            dmri_qc.create_index_acqparam_files(input_dwi   = eddycorrected_img,
                                                output_base = output_base)

        if( eddycorrected_img._get_slspec() == None ):
            if slspec != None:
                shutil.copy2(slspec, output_base + '_desc-Slspec_dwi.txt')
            else:
                dmri_qc.create_slspec_file(input_dwi     = degibbs_img,
                                           output_base   = output_base)

        if method == 'eddy':

            if verbose:
                print('Running EDDY...')

            eddycorrected_img = eddycorr.eddy_fsl(input_dwi                  = dwi_image,
                                                  output_base                = output_base,
                                                  topup_base                 = topup_base,
                                                  repol                      = repol,
                                                  data_shelled               = data_shelled,
                                                  cuda                       = gpu,
                                                  cuda_device                = cuda_device,
                                                  estimate_move_by_suscept   = estimate_move_by_suscept,
                                                  fsl_eddy_options           = fsl_eddy_options,
                                                  mporder                    = mporder,
                                                  nthreads                   = nthreads)

        elif method == 'eddy_correct':
            if verbose:
                print('Running Eddy-correct')
            eddycorrected_img = eddycorr.eddy_correct_fsl(input_dwi   = dwi_image,
                                                          output_base = output_base)

    return eddycorrected_img



def perform_outlier_detection(dwi_image, method, percent_threshold, verbose=False):

    working_dir     = os.path.dirname(dwi_image._get_filename())
    parsed_filename = parse_file_entities(dwi_image._get_filename())

    entities = {
    'extension': '.nii.gz',
    'subject': parsed_filename.get('subject'),
    'session': parsed_filename.get('session'),
    'suffix':  'dwi',
    'desc': 'OutlierRemoved'
    }

    filename_patterns   = working_dir + '/sub-{subject}[_ses-{session}][_desc-{desc}]_{suffix}{extension}'
    outputbase_patterns = working_dir + '/sub-{subject}[_ses-{session}]'

    output_base = writing.build_path(entities, outputbase_patterns)

    outlier_file = writing.build_path(entities, filename_patterns)
    entities['extension'] = '.bvec'
    outlier_bvec = writing.build_path(entities, filename_patterns)
    entities['extension'] = '.bval'
    outlier_bval = writing.build_path(entities, filename_patterns)
    entities['extension'] = '.txt'
    entities['desc'] = 'OutlierRemoved-Index'
    outlier_index = writing.build_path(entities, filename_patterns)

    outlier_removed_img = copy.deepcopy(dwi_image)
    outlier_removed_img._set_filename(outlier_file)
    outlier_removed_img._set_bvecs(outlier_bvec)
    outlier_removed_img._set_bvals(outlier_bval)
    outlier_removed_img._set_index(outlier_index)

    if method != None and not outlier_removed_img.exists():

        if verbose:
            print('Removing Outliers from DWIs')

        outlier_removed_img = dmri_qc.remove_outlier_imgs(input_dwi               = dwi_image,
                                                          output_base             = output_base,
                                                          method                  = method,
                                                          percent_threshold       = percent_threshold,
                                                          output_removed_imgs_dir = working_dir + '/outlier-removed-images/')
    return outlier_removed_img
