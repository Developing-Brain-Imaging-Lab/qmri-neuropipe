import os,sys, shutil, json, argparse, copy
import nibabel as nib

from bids.layout import writing, parse_file_entities
import core.dmri.utils.eddy_correction as eddycorr
import core.dmri.utils.qc as dmri_qc

def perform_eddy(dwi_image, working_dir, topup_base, method='eddy', gpu=False, cuda_device=0, nthreads=1, data_shelled=True, repol=False, estimate_move_by_suscept=False, mporder=0, slspec=None, fsl_eddy_options=None, tortoise_options=None, struct_img=None, verbose=False):

    if not os.path.exists(working_dir):
        os.makedirs(working_dir)

    parsed_filename = parse_file_entities(dwi_image.filename)

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
    eddycorrected_img.filename = eddy_file
    eddycorrected_img.bvecs = eddy_bvec

    print(eddycorrected_img.get_type())

    if not eddycorrected_img.exists():

        #First, create Index and Acqparams based on DWI and JSON
        if( ( eddycorrected_img.index == None) or ( eddycorrected_img.acqparams == None ) ):
            dmri_qc.create_index_acqparam_files(input_dwi   = eddycorrected_img,
                                                output_base = output_base)

        if( eddycorrected_img.slspec == None ):
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


        elif method == 'two-pass':
            if verbose:
                print('Running Two-stage Eddy/Motion correction')

            print('Running EDDY')
            eddy_corr_img = eddycorr.eddy_fsl(input_dwi                      = dwi_image,
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
                                                  
            print('Running EDDY-CORRECT')
            eddycorrected_img = eddycorr.eddy_correct_fsl(input_dwi   = eddy_corr_img,
                                                          output_base = output_base)
    
        elif method == 'tortoise-diffprep':
            if verbose:
                print('Running Two-stage Eddy/Motion correction')

            print('Running TORTOISE DIFFPREP')
            eddy_corr_img = eddycorr.diffprep_tortoise(input_dwi                  = dwi_image,
                                                       output_base                = output_base,
                                                       tortoise_options           = tortoise_options,
                                                       struct_img                 = struct_img,
                                                       nthreads                   = nthreads)
        

                                                  


        else:
            print('Incorrect Eddy method, exiting')
            exit(-1)


    return eddycorrected_img



def perform_outlier_detection(dwi_image, working_dir, method, percent_threshold=0.1, manual_report_dir=None, verbose=False):

    parsed_filename = parse_file_entities(dwi_image.filename)

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

    output_dir = os.path.dirname(output_base)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    outlier_file = writing.build_path(entities, filename_patterns)
    entities['extension'] = '.bvec'
    outlier_bvec = writing.build_path(entities, filename_patterns)
    entities['extension'] = '.bval'
    outlier_bval = writing.build_path(entities, filename_patterns)
    entities['extension'] = '.txt'
    entities['desc'] = 'OutlierRemoved-Index'
    outlier_index = writing.build_path(entities, filename_patterns)

    outlier_removed_img = copy.deepcopy(dwi_image)
    outlier_removed_img.filename = outlier_file
    outlier_removed_img.bvecs    = outlier_bvec
    outlier_removed_img.bvals    = outlier_bval
    outlier_removed_img.index    = outlier_index

    if method != None and not outlier_removed_img.exists():

        if verbose:
            print('Removing Outliers from DWIs')

        outlier_removed_img = dmri_qc.remove_outlier_imgs(input_dwi               = dwi_image,
                                                          output_base             = output_base,
                                                          method                  = method,
                                                          percent_threshold       = percent_threshold,
                                                          manual_report_dir       = manual_report_dir,
                                                          output_removed_imgs_dir = working_dir)
    return outlier_removed_img
