import os,sys, shutil, json, argparse, copy
import nibabel as nib

from bids.layout import writing
from core.utils.io import Image

import core.utils.workflows.prep_rawdata as raw_proc
import core.utils.workflows.denoise_degibbs as img_proc

import core.utils.tools as img_tools
import core.utils.mask as mask
import core.utils.denoise as denoise

from core.qmri.despot.utils.json import create_processing_json
import core.qmri.despot.utils.coregistration as despot_coreg
import core.qmri.afi as afi_tools

from core.qmri.despot.models.despot1 import DESPOT1_Model
from core.qmri.despot.models.despot2 import DESPOT2_Model
from core.qmri.despot.models.mcdespot import MCDESPOT_Model

class DESPOTProcessingPipeline:

    def __init__(self, verbose=False):
        if verbose:
            print('Creating DESPOT Processing Pipeline')

    def run(self):

        parser = argparse.ArgumentParser()

        parser.add_argument('--bids_dir',
                            type=str,
                            help='BIDS Data Directory')

        parser.add_argument('--bids_rawdata_dir',
                            type=str, help='BIDS RAWDATA Directory',
                            default='rawdata')

        parser.add_argument('--bids_despot_dir',
                            type=str, help='BIDS DWI RAWDATA Directory Basename',
                            default='despot')

        parser.add_argument('--load_json',
                            type=str, help='Load settings from file in json format. Command line options are overriden by values in file.',
                            default=None)

        parser.add_argument('--subject',
                            type=str,
                            help='Subject ID')

        parser.add_argument('--session',
                            type=str,
                            help='Subject Timepoint',
                            default=None)

        parser.add_argument('--bids_pipeline_name',
                            type=str, help='BIDS PIPELINE Name',
                            default='qmri-neuropipe')

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
                            default=None)

        parser.add_argument('--despot_cleanup',
                            type=bool,
                            help='Clean up the Preprocessing Subdirectories',
                            default=False)

        parser.add_argument('--despot_mask_method',
                            type=str,
                            help='Skull-stripping Algorithm',
                            choices=['bet', 'hd-bet', 'mrtrix', 'ants', 'antspynet'],
                            default='bet')

        parser.add_argument('--despot_ants_mask_template',
                            type=str,
                            help='Image to use for registration based skull-stripping',
                            default=os.environ['FSLDIR']+'/data/standard/MNI152_T1_1mm.nii.gz')

        parser.add_argument('--despot_ants_mask_template_mask',
                            type=str,
                            help='Brain mask to use for registration based skull-stripping',
                            default=os.environ['FSLDIR']+'/data/standard/MNI152_T1_1mm_brain_mask.nii.gz')

        parser.add_argument('--despot_antspynet_modality',
                            type=str,
                            help='ANTsPyNet modality/network name',
                            default='t1')

        parser.add_argument('--despot_denoise_method',
                            type=str,
                            help='Method for Denoising DWIs',
                            choices=['mrtrix', 'dipy-nlmeans', 'dipy-localpca', 'dipy-mppca', 'dipy-patch2self'],
                            default='mrtrix')

        parser.add_argument('--despot_gibbs_correction_method',
                            type=str,
                            help='Method for Gibbs Ringing Correction',
                            choices=['mrtrix', 'dipy'],
                            default='mrtrix')

        parser.add_argument('--despot_biasfield_correction_method',
                            type=str,
                            help='Method for Gibbs Ringing Correction',
                            choices=['ants', 'fsl', 'N4'],
                            default='ants')

        parser.add_argument('--despot_coregistration_method',
                            type=str,
                            help='Method for Coregistration of DESPOT Images',
                            choices=['ANTS', 'FSL'],
                            default='FSL')

        parser.add_argument('--coregister_to_anat',
                            type = bool,
                            help = 'Coregister DESPOT MRI to Structural MRI',
                            default = False)

        parser.add_argument('--coregister_to_anat_method',
                            type = str,
                            help = 'Linear Registration for DESPOT to Anat',
                            default = 'linear')

        parser.add_argument('--coregister_to_anat_linear_method',
                            type = str,
                            help = 'Linear Registration for DESPOT to Anat',
                            default = 'FSL')

        parser.add_argument('--coregister_to_anat_nonlinear_method',
                            type = str,
                            help = 'Linear Registration for DESPOT to Anat',
                            default = 'ANTS')

        parser.add_argument('--despot_b1_method',
                            type=str,
                            help='B1 Field Inhomogeneity Correction Method',
                            choices=['AFI', 'HIFI'],
                            default='AFI')

        parser.add_argument('--despot1_fit_method',
                            type=str,
                            help='Fitting Algorithm for DESPOT1 Model',
                            choices=['LinReg','OLS','WLS', 'NLOPT', 'CERES'],
                            default=None)

        parser.add_argument('--despot2_fit_method',
                            type=str,
                            help='Fitting Algorithm for DESPOT2-FM Model',
                            choices=['NLOPT','CERES', 'SRC', 'GRC'],
                            default=None)

        parser.add_argument('--mcdespot_fit',
                            type=bool,
                            help='Perfrom mcDESPOT Fitting',
                            default=False)

        parser.add_argument('--mcdespot_fit_method',
                            type=str,
                            help='Fitting Algorithm for mcDESPOT Model',
                            choices=['GRC', 'SRC'],
                            default=None)

        parser.add_argument('--mcdespot_model',
                            type=str,
                            help='mcDESPOT Model',
                            choices=[3, 2],
                            default=3)

        parser.add_argument('--mcdespot_use_condor',
                            type=bool,
                            help='Perform fitting using CHTC CONDOR Submit Node (e.g. MEDUDA)',
                            default=False)

        parser.add_argument('--mcdespot_package_condor_data',
                            type=bool,
                            help='Package CHTC CONDOR data',
                            default=False)

        parser.add_argument('--despot_register_to_template',
                            type=bool,
                            help='Normalize DESPOT data to template space',
                            default=False)

        parser.add_argument('--despot_registration_template',
                            type=str,
                            help='Template to Normalize DESPOT Data',
                            default=os.environ['FSLDIR']+'/data/standard/MNI152_T1_2mm_brain.nii.gz')

        parser.add_argument('--despot_registration_template_mask',
                            type=str,
                            help='Mask of template to Normalize DESPOT Data',
                            default=os.environ['FSLDIR']+'/data/standard/MNI152_T1_2mm_brain_mask.nii.gz')

        parser.add_argument('--despot_registration_method',
                            type=str,
                            help='Method for Normalizing DESPOT Data to template',
                            default='ANTS')

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
        'modality': 'despot'
        }

        id_patterns = 'sub-{subject}[_ses-{session}]'
        rawdata_patterns = args.bids_dir + '/'+ args.bids_rawdata_dir + '/sub-{subject}[/ses-{session}]/'
        derivative_patterns = args.bids_dir + '/derivatives/' + args.bids_pipeline_name + '/sub-{subject}[/ses-{session}]/'

        bids_id             = writing.build_path(entities, id_patterns)
        bids_rawdata_dir    = writing.build_path(entities, rawdata_patterns)
        bids_derivative_dir = writing.build_path(entities, derivative_patterns)

        preprocess_dir      = bids_derivative_dir + '/despot/preprocessed/'
        models_dir          = bids_derivative_dir + '/despot/models/'
        despot1_dir         = models_dir + '/DESPOT1/'
        despot2_dir         = models_dir + '/DESPOT2-FM/'
        mcdespot_dir        = models_dir + '/mcDESPOT/'

        if args.despot_b1_method == 'HIFI':
            despot1_dir         = models_dir +'/DESPOT1-HIFI/'

        if not os.path.exists(bids_derivative_dir):
            os.makedirs(bids_derivative_dir)

        if not os.path.exists(preprocess_dir):
            os.makedirs(preprocess_dir)


        despot_json     = preprocess_dir + bids_id + '_desc-FittingParameters.json'
        despot_data     = {}

        raw_spgr = None
        if os.path.exists(os.path.join(bids_rawdata_dir,'despot-spgr','') + bids_id + '_despot-spgr.nii.gz'):
            raw_spgr = Image(file = os.path.join(bids_rawdata_dir,'despot-spgr','') + bids_id + '_despot-spgr.nii.gz',
                             json = os.path.join(bids_rawdata_dir,'despot-spgr','') + bids_id + '_despot-spgr.json')

        raw_ssfp = None
        if os.path.exists(os.path.join(bids_rawdata_dir,'despot-ssfp','') + bids_id + '_despot-ssfp.nii.gz'):
            raw_ssfp = Image(file = os.path.join(bids_rawdata_dir,'despot-ssfp','') + bids_id + '_despot-ssfp.nii.gz',
                           json = os.path.join(bids_rawdata_dir,'despot-ssfp','') + bids_id + '_despot-ssfp.json')

        raw_irspgr = None
        if args.despot_b1_method == 'HIFI':
            raw_irspgr = Image(file = os.path.join(bids_rawdata_dir,'despot-irspgr','') + bids_id + '_despot-irspgr.nii.gz',
                               json = os.path.join(bids_rawdata_dir,'despot-irspgr','') + bids_id + '_despot-irspgr.json')

        raw_afi_ref = None
        raw_afi_b1  = None
        if args.despot_b1_method == 'AFI':
            raw_afi_ref = Image(file = os.path.join(bids_rawdata_dir,'despot-afi','') + bids_id + '_despot-afi.nii.gz',
                                json = os.path.join(bids_rawdata_dir,'despot-afi','') + bids_id + '_despot-afi.json')

            raw_afi_b1 = Image(file = os.path.join(bids_rawdata_dir,'despot-afi','') + bids_id + '_desc-AFI-B1.nii.gz',
                               json = os.path.join(bids_rawdata_dir,'despot-afi','') + bids_id + '_despot-afi.json')



        create_processing_json(despot_json = preprocess_dir + bids_id + '_desc-FittingParameters.json',
                               spgr_img    = raw_spgr,
                               ssfp_img    = raw_ssfp,
                               irspgr_img  = raw_irspgr)


        #Create target image and coregister images to the target
        target_img = Image(file = preprocess_dir + bids_id + '_desc-reference_despot-spgr.nii.gz')

        if not os.path.exists(target_img._get_filename()):
            if args.verbose:
                print("Creating Target Image for Coregistration")

            spgr = nib.load(raw_spgr._get_filename())
            num_spgr = spgr.shape[3]

            ref_img = nib.Nifti1Image(spgr.get_fdata()[:,:,:,num_spgr-1], spgr.affine)
            ref_img.to_filename(target_img._get_filename())


        coreg_spgr   = Image(file = preprocess_dir + bids_id + '_desc-Coregistered_despot-spgr.nii.gz')
        coreg_ssfp   = Image(file = preprocess_dir + bids_id + '_desc-Coregistered_despot-ssfp.nii.gz')
        coreg_irspgr = Image(file = preprocess_dir + bids_id + '_desc-Coregistered_despot-irspgr.nii.gz')
        b1_map       = Image()

        if args.despot_b1_method == 'AFI':
            b1_map._set_filename(preprocess_dir + bids_id + '_desc-Coregistered-AFI-B1.nii.gz')
        elif args.despot_b1_method == 'HIFI':
            b1_map._set_filename(despot1_dir + bids_id + '_model-HIFI_parameter-B1.nii.gz')
        else:
            print('Incorrect B1 map method!')
            exit()

        #Coregister SPGR
        if not os.path.exists(coreg_spgr._get_filename()):
            if args.verbose:
                print('Coregistering SPGR images...')

            if raw_spgr != None:
                despot_coreg.coregister_images(input_img      = raw_spgr,
                                               reference_img  = target_img,
                                               output_img     = coreg_spgr,
                                               method         = args.despot_coregistration_method)

        #Coregister SPGR
        if raw_ssfp != None:
            if not os.path.exists(coreg_ssfp._get_filename()):
                if args.verbose:
                    print('Coregistering SSFP images...')

                despot_coreg.coregister_images(input_img      = raw_ssfp,
                                               reference_img  = target_img,
                                               output_img     = coreg_ssfp,
                                               method         = args.despot_coregistration_method)


        if args.despot_b1_method == 'HIFI':
            if not os.path.exists(coreg_irspgr._get_filename()):
                if args.verbose:
                    print('Coregistering IR-SPGR images...')

                if raw_irspgr != None:
                    despot_coreg.coregister_images(input_img      = raw_irspgr,
                                                   reference_img  = target_img,
                                                   output_img     = coreg_irspgr,
                                                   method         = args.despot_coregistration_method)

        if args.despot_b1_method == 'AFI':
            if not os.path.exists(b1_map._get_filename()):
                if args.verbose:
                    print('Coregistering AFI data')

                if raw_afi_ref != None:
                    afi_tools.register_afi_flirt(input_afi        = raw_afi_ref,
                                                 input_b1         = raw_afi_b1,
                                                 ref_img          = target_img,
                                                 output_b1        = b1_map)

        brain_mask = Image(file = preprocess_dir + bids_id + '_desc-brain-mask.nii.gz')
        if not os.path.exists(brain_mask._get_filename()):
            mask.mask_image(input_img            = target_img,
                            output_mask          = brain_mask,
                            method               = args.despot_mask_method,
                            ref_img              = args.despot_ants_mask_template,
                            ref_mask             = args.despot_ants_mask_template_mask,
                            antspynet_modality   = args.despot_antspynet_modality,
                            nthreads             = args.nthreads)

        if args.despot1_fit_method != None:
            despot1_base = bids_id + '_model-DESPOT1_parameter-'
            despot1_model = 'DESPOT1'

            if args.despot_b1_method == 'HIFI':
                despot1_base = bids_id + '_model-HIFI_parameter-'
                despot1_model = 'HIFI'

            if not os.path.exists(despot1_dir + despot1_base + 'T1.nii.gz'):
                if args.verbose:
                    print('Fitting DESPOT1 model...')


                despot1_model = DESPOT1_Model(spgr_img      = coreg_spgr,
                                              params        = despot_json,
                                              out_dir       = despot1_dir,
                                              out_base      = despot1_base,
                                              b1            = b1_map,
                                              irspgr_img    = coreg_irspgr,
                                              mask          = brain_mask,
                                              model         = despot1_model,
                                              fit_algorithm = args.despot1_fit_method,
                                              nthreads      = args.nthreads,
                                              verbose       = args.verbose)

                despot1_model.fit()
                
                exit()

                if args.despot_b1_method == 'HIFI':
                    #Smooth the output B1 map, and refit using DESPOT1 code
                    print('Smoothing B1')
                    os.system('fslmaths ' + b1_map._get_filename() + ' -s 2.55 ' + despot1_dir + '/tmp_b1.nii.gz')
                    os.system('fslmaths ' + brain_mask._get_filename() + ' -s 2.55 ' + despot1_dir + '/tmp_mask.nii.gz')
                    os.system('fslmaths ' + despot1_dir +'/tmp_b1.nii.gz -div ' + despot1_dir + '/tmp_mask.nii.gz -mas ' + brain_mask._get_filename()+ ' ' + b1_map._get_filename()  )
                    os.remove(despot1_dir + '/tmp_mask.nii.gz')
                    os.remove(despot1_dir + '/tmp_b1.nii.gz')

                    despot1_model.set_b1(b1 = b1_map)
                    despot1_model.set_model('DESPOT1')
                    despot1_model.fit()

        if args.despot2_fit_method != None:
            despot2_base = bids_id + '_model-DESPOT2_parameter-'
            despot2_model = 'DESPOT2-FM'

            if not os.path.exists(despot2_dir + despot2_base + 'T2.nii.gz'):
                if args.verbose:
                    print('Fitting DESPOT2-FM model...')


                despot2_model = DESPOT2_Model(ssfp_img      = coreg_ssfp,
                                              params        = despot_json,
                                              out_dir       = despot2_dir,
                                              out_base      = despot2_base,
                                              t1            = Image(file = despot1_dir + despot1_base + 'T1.nii.gz'),
                                              b1            = b1_map,
                                              mask          = brain_mask,
                                              model         = despot2_model,
                                              fit_algorithm = args.despot2_fit_method,
                                              nthreads      = args.nthreads,
                                              verbose       = args.verbose)

                despot2_model.fit()
                
                if args.verbose:
                    print('Smoothing F0')

                os.system('fslmaths ' + brain_mask._get_filename() + ' -s 2.55 ' + despot2_dir + '/tmp_mask.nii.gz')
                os.system('fslmaths ' + despot2_dir + despot2_base + 'F0.nii.gz -s 2.55 -div ' + despot2_dir + '/tmp_mask.nii.gz -mas ' + brain_mask._get_filename() + ' ' + despot2_dir + despot2_base + 'F0.nii.gz' )
                os.remove(despot2_dir + '/tmp_mask.nii.gz')


        if args.mcdespot_fit_method != None and args.mcdespot_fit != False:
            mcdespot_base = bids_id + '_model-mcDESPOT_parameter-'

            if not os.path.exists(mcdespot_dir + mcdespot_base + 'VFm.nii.gz'):
                if args.verbose:
                    print('Fitting mcDESPOT model...')


                mcdespot_model = MCDESPOT_Model(spgr_img    = coreg_spgr,
                                                ssfp_img      = coreg_ssfp,
                                                params        = despot_json,
                                                out_dir       = mcdespot_dir,
                                                out_base      = mcdespot_base,
                                                b1            = b1_map,
                                                f0            = Image(file=despot2_dir+despot2_base+'F0.nii.gz'),
                                                mask          = brain_mask,
                                                model         = args.mcdespot_model,
                                                fit_algorithm = args.mcdespot_fit_method,
                                                use_condor    = args.mcdespot_use_condor,
                                                nthreads      = args.nthreads,
                                                verbose       = args.verbose)

                if args.mcdespot_package_condor_data == True:
                    mcdespot_model.package_condor_chunks_three_compartments()
                else:
                    mcdespot_model.fit()

        if args.despot_register_to_template == True:
            import core.registration.registration as reg_tools

            registration_dir      = bids_derivative_dir + '/despot/registration/'
            normalized_dir        = bids_derivative_dir + '/despot/normalized/'

            if not os.path.exists(registration_dir):
                os.makedirs(registration_dir)
            if not os.path.exists(normalized_dir):
                os.makedirs(normalized_dir)

            moving_img = Image(file = registration_dir + bids_id + '_desc-reference_despot-spgr.nii.gz')
            os.system('fslmaths ' + target_img._get_filename() + ' -mas ' + brain_mask._get_filename() + ' ' + moving_img._get_filename())
            img_tools.biasfield_correction(input_img    = moving_img,
                                           output_file  = moving_img._get_filename(),
                                           method       = 'N4',
                                           nthreads     = args.nthreads,
                                           iterations   = 3)

            if not os.path.exists(registration_dir + bids_id + '_desc-TemplateRegistration_Warped.nii.gz'):
                if args.verbose:
                    print('Registering to Template')

                reg_tools.nonlinear_reg(input_img       = moving_img,
                                        reference_img   = Image(file = args.despot_registration_template),
                                        reference_mask  = Image(file = args.despot_registration_template_mask),
                                        output_base     = registration_dir + bids_id + '_desc-TemplateRegistration_',
                                        nthreads        = args.nthreads,
                                        method          = args.despot_registration_method)

                reg_tools.create_composite_transform(reference_img = Image(file = args.despot_registration_template),
                                                     output_file   = registration_dir + bids_id + '_desc-TemplateRegistration_ForwardWarp.nii.gz',
                                                     transforms    = [registration_dir + bids_id + '_desc-TemplateRegistration_1Warp.nii.gz', registration_dir + bids_id + '_desc-TemplateRegistration_0GenericAffine.mat'])

            #Now, warp DESPOT maps to the Template
            normalized_despot1_dir = normalized_dir + '/DESPOT1/'
            normalized_despot2_dir = normalized_dir + '/DESPOT2-FM/'
            normalized_mcdespot_dir = normalized_dir + '/mcDESPOT/'

            if args.despot_b1_method == 'HIFI':
                normalized_despot1_dir = normalized_dir + '/DESPOT1-HIFI/'

            #DESPOT1 images
            if os.path.exists(despot1_dir + despot1_base + 'T1.nii.gz'):
                if not os.path.exists(normalized_despot1_dir):
                    os.makedirs(normalized_despot1_dir)

                imgs = ['T1','M0']
                for img in imgs:
                    reg_tools.apply_transform(input_img     = Image(file = despot1_dir + despot1_base + img +'.nii.gz'),
                                              reference_img = Image(file = args.despot_registration_template),
                                              output_file   = normalized_despot1_dir + despot1_base + img + '.nii.gz',
                                              matrix        = registration_dir + bids_id + '_desc-TemplateRegistration_ForwardWarp.nii.gz',
                                              method        = 'ANTS')

            #DESPOT2 images
            if os.path.exists(despot2_dir + despot2_base + 'T2.nii.gz'):
                if not os.path.exists(normalized_despot2_dir):
                    os.makedirs(normalized_despot2_dir)

                imgs = ['T2','M0', 'F0']
                for img in imgs:
                    reg_tools.apply_transform(input_img     = Image(file = despot2_dir + despot2_base + img +'.nii.gz'),
                                              reference_img = Image(file = args.despot_registration_template),
                                              output_file   = normalized_despot2_dir + despot2_base + img + '.nii.gz',
                                              matrix        = registration_dir + bids_id + '_desc-TemplateRegistration_ForwardWarp.nii.gz',
                                              method        = 'ANTS')

                #mcDESPOT images
                mcdespot_base = bids_id + '_model-mcDESPOT_parameter-'
                if os.path.exists(mcdespot_dir + mcdespot_base + 'VFm.nii.gz'):
                    if not os.path.exists(normalized_mcdespot_dir):
                        os.makedirs(normalized_mcdespot_dir)

                    imgs = ['VFm','VFcsf', 'F0', 'T1csf', 'T2csf', 'T1m', 'T2m', 'T1f', 'T2f', 'Tau']

                    for img in imgs:
                        reg_tools.apply_transform(input_img     = Image(file = mcdespot_dir + mcdespot_base + img +'.nii.gz'),
                                                  reference_img = Image(file = args.despot_registration_template),
                                                  output_file   = normalized_mcdespot_dir + mcdespot_base + img + '.nii.gz',
                                                  matrix        = registration_dir + bids_id + '_desc-TemplateRegistration_ForwardWarp.nii.gz',
                                                  method        = 'ANTS')
