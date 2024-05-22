import os, shutil, json, argparse
import numpy as np
import nibabel as nib

from bids.layout import writing
from core.utils.io import Image

import core.utils.tools as img_tools
import core.utils.mask as mask
import core.utils.denoise as denoise
from core.utils.gibbs_correction import gibbs_ringing_correction

from core.qmri.despot.utils.json import create_processing_json
import core.registration.multilinreg as coreg
import core.qmri.afi as afi_tools

from core.qmri.despot.models.despot1 import DESPOT1_Model
from core.qmri.despot.models.despot2 import DESPOT2_Model
#from core.qmri.despot.models.mcdespot import MCDESPOT_Model

from nilearn.image import resample_img


def resample_image(input_img, out_shape):
    in_nii = nib.load(input_img.filename)

    # Initialize target_affine
    target_affine = in_nii.affine.copy()
    input_shape   = np.asarray(in_nii.shape)
    target_shape  = np.asarray(out_shape)
    target_shape  = np.append(target_shape, input_shape[3])
    scale         = np.divide(input_shape, target_shape)

    print(scale)
    print(input_shape)

    # Reconstruct the affine
    target_affine = target_affine @ np.diag(scale)

    resampled_img = resample_img(img = in_nii, 
                                 target_affine=target_affine,
                                 target_shape=out_shape)
                
    return resampled_img

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

        parser.add_argument('--preproc_derivative_dir',
                            type=str, help='Preprocessing Derivative Output',
                            default='despot-neuropipe-preproc')
        
        parser.add_argument('--models_derivative_dir',
                            type=str, help='BIDS PIPELINE Name',
                            default='despot-neuropipe-models')

        parser.add_argument('--nthreads',
                            type=int,
                            help='Number of Threads',
                            default=1)

        parser.add_argument('--gpu',
                            type=bool,
                            help='CUDA GPU Available',
                            default=False)

        parser.add_argument('--despot_cleanup',
                            type=bool,
                            help='Clean up the Preprocessing Subdirectories',
                            default=False)
        
        parser.add_argument('--despot_hybrid',
                            type=bool,
                            help='Hybrid combination of High-res and Low-res DESPOT data',
                            default=False)

        parser.add_argument('--despot_mask_method',
                            type=str,
                            help='Skull-stripping Algorithm',
                            choices=['bet', 'hd-bet', 'mrtrix', 'ants', 'antspynet', 'mri_synthstrip'],
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
                            help='Method for Denoising DESPOT Data',
                            choices=['mrtrix', 'dipy-nlmeans', 'dipy-localpca', 'dipy-mppca'],
                            default='mrtrix')

        parser.add_argument('--despot_gibbs_correction_method',
                            type=str,
                            help='Method for Gibbs Ringing Correction',
                            choices=['mrtrix', 'dipy'],
                            default='mrtrix')

        parser.add_argument('--despot_biasfield_correction_method',
                            type=str,
                            help='Method for Biasfield correction',
                            choices=['ants', 'fsl', 'N4'],
                            default='ants')

        parser.add_argument('--despot_coregistration_method',
                            type=str,
                            help='Method for Coregistration of DESPOT Images',
                            choices=['ants', 'fsl'],
                            default='fsl')

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
                            default = 'fsl')

        parser.add_argument('--coregister_to_anat_nonlinear_method',
                            type = str,
                            help = 'Linear Registration for DESPOT to Anat',
                            default = 'ants')

        parser.add_argument('--despot_b1_method',
                            type=str,
                            help='B1 Field Inhomogeneity Correction Method',
                            choices=['afi', 'hifi'],
                            default='afi')

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

        id_patterns          = "sub-{subject}[_ses-{session}]"
        rawdata_patterns     = os.path.join(args.bids_dir, args.bids_rawdata_dir, "sub-{subject}[/ses-{session}]",)
        derivative_patterns  = os.path.join(args.bids_dir, "derivatives", args.preproc_derivative_dir),
        output_patterns      = os.path.join(args.bids_dir, "derivatives", args.preproc_derivative_dir, "sub-{subject}[/ses-{session}]",)
        
        id              = writing.build_path(entities, id_patterns)
        rawdata_dir     = writing.build_path(entities, rawdata_patterns)
        derivative_dir  = writing.build_path(entities, derivative_patterns)
        output_dir      = writing.build_path(entities, output_patterns)
        
        anat_preproc_dir = os.path.join(output_dir, "anat",)
        fmap_preproc_dir = os.path.join(output_dir, "fmap",)


        if not os.path.exists(derivative_dir):
            os.makedirs(derivative_dir)
        
        if not os.path.exists(anat_preproc_dir):
            os.makedirs(anat_preproc_dir)

        spgr = Image(filename = os.path.join(rawdata_dir, 'anat',id+'_desc-SPGR_VFA.nii.gz'),
                     json     = os.path.join(rawdata_dir, 'anat',id+'_desc-SPGR_VFA.json'))
        ssfp = Image(filename = os.path.join(rawdata_dir, 'anat',id+'_desc-SSFP_VFA.nii.gz'),
                     json     = os.path.join(rawdata_dir, 'anat',id+'_desc-SSFP_VFA.json'))
        
        spgr_highres = Image(filename = os.path.join(rawdata_dir, 'anat',id+'_acq-highres_desc-SPGR_VFA.nii.gz'),
                             json    = os.path.join(rawdata_dir, 'anat',id+'_acq-highres_desc-SPGR_VFA.json'))
        
        ssfp_highres = Image(filename = os.path.join(rawdata_dir, 'anat',id+'_acq-highres_desc-SSFP_VFA.nii.gz'),
                             json     = os.path.join(rawdata_dir, 'anat',id+'_acq-highres_desc-SSFP_VFA.json'))
        
        spgr_preproc = Image(filename = os.path.join(anat_preproc_dir, id+'_desc-SPGR-preproc_VFA.nii.gz'),
                             json     = os.path.join(anat_preproc_dir, id+'_desc-SPGR-preproc_VFA.json'))
        ssfp_preproc = Image(filename = os.path.join(anat_preproc_dir, id+'_desc-SSFP-preproc_VFA.nii.gz'),
                             json     = os.path.join(anat_preproc_dir, id+'_desc-SSFP-preproc_VFA.json'))
        
        irspgr = None
        irspgr_preproc = None
        if args.despot_b1_method.lower() == 'hifi':
            irspgr = Image(filename = os.path.join(rawdata_dir, 'anat',id+'_desc-HIFI_T1w.nii.gz'),
                           json     = os.path.join(rawdata_dir, 'anat',id+'_desc-HIFI_T1w.json'))
            
            irspgr_preproc = Image(filename = os.path.join(anat_preproc_dir, id+'_desc-HIFI-preproc_T1w.nii.gz'),
                                   json     = os.path.join(anat_preproc_dir, id+'_desc-HIFI-preproc_T1w.json'))

        afi = None
        afi_preproc = None
        afi_b1map = None
        if args.despot_b1_method.lower() == 'afi':
            afi = Image(filename = os.path.join(rawdata_dir, 'fmap',id+'_TB1AFI.nii.gz'),
                        json     = os.path.join(rawdata_dir, 'fmap',id+'_TB1AFI.json'))
            
            afi_preproc = Image(filename = os.path.join(fmap_preproc_dir, id+'_desc-preproc_TB1AFI.nii.gz'),
                                json     = os.path.join(fmap_preproc_dir, id+'_desc-preproc_TB1AFI.json'))
            
            afi_b1map = Image(filename = os.path.join(fmap_preproc_dir, id+"_TB1map.nii.gz"))
            
            if not os.path.exists(fmap_preproc_dir):
                os.makedirs(fmap_preproc_dir)
        
        
        #IF performing HYBRID DESPOT, first, resample images to the higher resolution and then combine
        if args.despot_hybrid:
            spgr_highres_img = nib.load(spgr_highres.filename)
            num_spgr = spgr_highres_img.shape[3]
            ref_img = nib.Nifti1Image(spgr_highres_img.get_fdata()[:,:,:,num_spgr-1], spgr_highres_img.affine)
            
            spgr_target    = Image(filename = os.path.join(anat_preproc_dir, id+"_desc-SPGRtarget_VFA.nii.gz"))
            spgr_resampled = Image(filename = os.path.join(anat_preproc_dir, id+"_desc-SPGR-Hybrid_VFA.nii.gz"))
            ref_img.to_filename(spgr_target.filename)

            coreg.multilinreg(input         = spgr,
                              ref           = spgr_target,
                              out           = spgr_resampled,
                              dof           = 6,
                              method        = args.despot_coregistration_method,
                              nthreads      = args.nthreads, 
                              debug         = args.verbose)
            
            #Then merge the highres images
            os.system('fslmerge -t ' + spgr_resampled.filename + " " + spgr_resampled.filename + " " + spgr_highres.filename)


            ssfp_highres_img = nib.load(ssfp_highres.filename)
            num_ssfp = ssfp_highres_img.shape[3]
            ref_img = nib.Nifti1Image(ssfp_highres_img.get_fdata()[:,:,:,num_ssfp-1], ssfp_highres_img.affine)
            
            ssfp_target    = Image(filename = os.path.join(anat_preproc_dir, id+"_desc-SSFPtarget_VFA.nii.gz"))
            ssfp_resampled = Image(filename = os.path.join(anat_preproc_dir, id+"_desc-SSFP-Hybrid_VFA.nii.gz"))
            ref_img.to_filename(ssfp_target.filename)

            coreg.multilinreg(input         = ssfp,
                              ref           = ssfp_target,
                              out           = ssfp_resampled,
                              dof           = 6,
                              method        = args.despot_coregistration_method,
                              nthreads      = args.nthreads, 
                              debug         = args.verbose)
            
            os.system('fslmerge -t ' + ssfp_resampled.filename + " " + ssfp_resampled.filename + " " + ssfp_highres.filename)
        

            exit()


        
        
        
        ##ADD IN OPTIONS FOR DENOISING AND GIBBS RINGING CORRECTION
        if args.despot_denoise_method:

            if not spgr_preproc.exists() and not os.path.exists(os.path.join(anat_preproc_dir, id+"_desc-SPGR-Denoised_VFA.nii.gz")):
                if args.verbose:
                    print("Denoising SPGR Image")

                spgr = denoise.denoise_image(input_img    = spgr,
                                             output_file   = os.path.join(anat_preproc_dir, id+"_desc-SPGR-Denoised_VFA.nii.gz"),
                                             method        = args.despot_denoise_method, 
                                             noise_map     = os.path.join(anat_preproc_dir, id+"_desc-SPGR-NoiseMap.nii.gz"), 
                                             nthreads      = args.nthreads, 
                                             debug         = args.verbose)

            else:
                spgr.filename = os.path.join(anat_preproc_dir, id+"_desc-SPGR-Denoised_VFA.nii.gz")
            
            if not ssfp_preproc.exists() and not os.path.exists(os.path.join(anat_preproc_dir, id+"_desc-SSFP-Denoised_VFA.nii.gz")):
                if args.verbose:
                    print("Denoising SSFP Image")
                ssfp = denoise.denoise_image(input_img    = ssfp,
                                             output_file   = os.path.join(anat_preproc_dir, id+"_desc-SSFP-Denoised_VFA.nii.gz"),
                                             method        = args.despot_denoise_method, 
                                             noise_map     = os.path.join(anat_preproc_dir, id+"_desc-SSFP-NoiseMap.nii.gz"), 
                                             nthreads      = args.nthreads, 
                                             debug         = args.verbose)
            else:
                ssfp.filename = os.path.join(anat_preproc_dir, id+"_desc-SSFP-Denoised_VFA.nii.gz")

            if args.despot_b1_method.lower() == 'hifi':
                if not irspgr_preproc.exists() and not os.path.exists(os.path.join(anat_preproc_dir, id+"_desc-HIFI-Denoised_T1w.nii.gz")):
                    if args.verbose:
                        print("Denoising IR-SPGR Image")

                    irspgr = denoise.denoise_image(input_img     = irspgr,
                                                   output_file   = os.path.join(anat_preproc_dir, id+"_desc-HIFI-Denoised_T1w.nii.gz"),
                                                   method        = args.despot_denoise_method, 
                                                   noise_map     = os.path.join(anat_preproc_dir, id+"_desc-HIFI-NoiseMap_T1w.nii.gz"), 
                                                   nthreads      = args.nthreads, 
                                                   debug         = args.verbose)
                else:
                    irspgr.filename = os.path.join(anat_preproc_dir, id+"_desc-HIFI-Denoised_T1w.nii.gz")
                
        if args.despot_gibbs_correction_method:

            if not spgr_preproc.exists() and not os.path.exists(os.path.join(anat_preproc_dir, id+"_desc-SPGR-GibbsRinging_VFA.nii.gz")):
                if args.verbose:
                    print("Correcting Gibbs Ringing in SPGR Image")

                spgr = gibbs_ringing_correction(input_img   = spgr,
                                                output_file = os.path.join(anat_preproc_dir, id+"_desc-SPGR-GibbsRinging_VFA.nii.gz"),
                                                method      = args.despot_gibbs_correction_method, 
                                                nthreads    = args.nthreads, 
                                                debug       = args.verbose)
            else:
                spgr.filename = os.path.join(anat_preproc_dir, id+"_desc-SPGR-GibbsRinging_VFA.nii.gz")
  
            if not ssfp_preproc.exists() and not os.path.exists(os.path.join(anat_preproc_dir, id+"_desc-SSFP-GibbsRinging_VFA.nii.gz")):
                if args.verbose:
                    print("Correcting Gibbs Ringing in SSFP Image")

                ssfp = gibbs_ringing_correction(input_img   = ssfp,
                                                output_file = os.path.join(anat_preproc_dir, id+"_desc-SSFP-GibbsRinging_VFA.nii.gz"),
                                                method      = args.despot_gibbs_correction_method, 
                                                nthreads    = args.nthreads, 
                                                debug       = args.verbose)
            else:
                ssfp.filename = os.path.join(anat_preproc_dir, id+"_desc-SSFP-GibbsRinging_VFA.nii.gz")

            if args.despot_b1_method.lower() == 'hifi':
                if not irspgr_preproc.exists() and not os.path.exists(os.path.join(anat_preproc_dir, id+"_desc-HIFI-GibbsRinging_T1w.nii.gz")):
                    if args.verbose:
                        print("Correcting Gibbs Ringing in IR-SPGR Image")

                    irspgr = gibbs_ringing_correction(input_img   = irspgr,
                                                      output_file = os.path.join(anat_preproc_dir, id+"_desc-HIFI-GibbsRinging_T1w.nii.gz"),
                                                      method      = args.despot_gibbs_correction_method, 
                                                      nthreads    = args.nthreads, 
                                                      debug       = args.verbose)
                else:
                    irspgr.filename = os.path.join(anat_preproc_dir, id+"_desc-HIFI-GibbsRinging_T1w.nii.gz")


        #Create target image and coregister images to the target
        target_img = Image(filename = os.path.join(anat_preproc_dir, id+'_desc-SPGR-Ref_T1w.nii.gz'))
        if not spgr_preproc.exists() and not target_img.exists():
            if args.verbose:
                print("Creating Target Image for Coregistration")

            spgr_img = nib.load(spgr.filename)
            num_spgr = spgr_img.shape[3]

            ref_img = nib.Nifti1Image(spgr_img.get_fdata()[:,:,:,num_spgr-1], spgr_img.affine)
            ref_img.to_filename(target_img.filename)


        #Coregister SPGR
        if not spgr_preproc.exists():
            if args.verbose:
                print('Coregistering SPGR images...')

            if spgr != None:
                coreg.multilinreg(input         = spgr,
                                  ref           = target_img,
                                  out           = spgr_preproc,
                                  dof           = 6,
                                  method        = args.despot_coregistration_method,
                                  nthreads      = args.nthreads, 
                                  debug         = args.verbose)
                shutil.copy2(spgr.json, spgr_preproc.json)

        #Coregister SSFP
        if not ssfp_preproc.exists():
            if args.verbose:
                print('Coregistering SSFP images...')

            if ssfp != None:
                coreg.multilinreg(input         = ssfp,
                                  ref           = target_img,
                                  out           = ssfp_preproc,
                                  dof           = 6,
                                  method        = args.despot_coregistration_method,
                                  nthreads      = args.nthreads, 
                                  debug         = args.verbose)
                shutil.copy2(ssfp.json, ssfp_preproc.json)

        if args.despot_b1_method.lower() == 'hifi':
            if not irspgr_preproc.exists():
                if args.verbose:
                    print('Coregistering IR-SPGR images...')

                if irspgr != None:
                    coreg.multilinreg(input         = irspgr,
                                      ref           = target_img,
                                      out           = irspgr_preproc,
                                      dof           = 6,
                                      method        = args.despot_coregistration_method,
                                      nthreads      = args.nthreads, 
                                      debug         = args.verbose)
                    shutil.copy2(irspgr.json, irspgr_preproc.json)
        
        elif args.despot_b1_method.lower() == 'afi':
            if not afi_preproc.exists():
                if args.verbose:
                    print('Coregistering AFI data')

                if afi != None:
                    afi_tools.coregister_afi(input_afi = afi, 
                                             ref_img   = target_img, 
                                             out_afi   = afi_preproc)
                    shutil.copy2(afi.json, afi_preproc.json)
                
            if not afi_b1map.exists():
                afi_tools.compute_afi_b1map(afi   = afi_preproc,
                                            b1map = afi_b1map,
                                            fwhm  = 6)
                

        brain_mask = Image(filename = os.path.join(anat_preproc_dir, id+"_desc-brain-mask.nii.gz"))
        if not brain_mask.exists():
            mask.mask_image(input               = target_img,
                            mask                = brain_mask,
                            algo                = args.despot_mask_method,
                            ref_img             = args.despot_ants_mask_template,
                            ref_mask            = args.despot_ants_mask_template_mask,
                            antspynet_modality  = args.despot_antspynet_modality,
                            nthreads            = args.nthreads)

        fitparam_json = os.path.join(anat_preproc_dir, id+'_desc-FittingParameters.json')
        if not os.path.exists(fitparam_json):
            create_processing_json(despot_json = fitparam_json,
                                spgr_img    = spgr_preproc,
                                ssfp_img    = ssfp_preproc,
                                irspgr_img  = irspgr_preproc)
            
        if args.despot_cleanup:
            #Remove all but preproc files
            if os.path.exists(os.path.join(anat_preproc_dir, id+"_desc-SPGR-Denoised_VFA.nii.gz")):
                os.remove(os.path.join(anat_preproc_dir, id+"_desc-SPGR-Denoised_VFA.nii.gz"))
            if os.path.exists(os.path.join(anat_preproc_dir, id+"_desc-SSFP-Denoised_VFA.nii.gz")):
                os.remove(os.path.join(anat_preproc_dir, id+"_desc-SSFP-Denoised_VFA.nii.gz"))
            
            if os.path.exists(os.path.join(anat_preproc_dir, id+"_desc-SPGR-GibbsRinging_VFA.nii.gz")):
                os.remove(os.path.join(anat_preproc_dir, id+"_desc-SPGR-GibbsRinging_VFA.nii.gz"))
            if os.path.exists(os.path.join(anat_preproc_dir, id+"_desc-SSFP-GibbsRinging_VFA.nii.gz")):
                os.remove(os.path.join(anat_preproc_dir, id+"_desc-SSFP-GibbsRinging_VFA.nii.gz"))

            if os.path.exists(os.path.join(anat_preproc_dir, id+"_desc-SPGR-NoiseMap.nii.gz")):
                os.remove(os.path.join(anat_preproc_dir, id+"_desc-SPGR-NoiseMap.nii.gz"))
            if os.path.exists(os.path.join(anat_preproc_dir, id+"_desc-SSFP-NoiseMap.nii.gz")):
                os.remove(os.path.join(anat_preproc_dir, id+"_desc-SSFP-NoiseMap.nii.gz"))

            if args.despot_b1_method.lower() == 'hifi':
                if os.path.exists(os.path.join(anat_preproc_dir, id+"_desc-HIFI-Denoised_T1w.nii.gz")):
                    os.remove(os.path.join(anat_preproc_dir, id+"_desc-HIFI-Denoised_T1w.nii.gz"))
                if os.path.exists(os.path.join(anat_preproc_dir, id+"_desc-HIFI-GibbsRinging_T1w.nii.gz")):
                    os.remove(os.path.join(anat_preproc_dir, id+"_desc-HIFI-GibbsRinging_T1w.nii.gz"))
                if os.path.exists(os.path.join(anat_preproc_dir, id+"_desc-HIFI-NoiseMap_T1w.nii.gz")):
                    os.remove(os.path.join(anat_preproc_dir, id+"_desc-HIFI-NoiseMap_T1w.nii.gz"))

            if os.path.exists(os.path.join(anat_preproc_dir, id+"_desc-SPGR-Ref_T1w.nii.gz")):
                os.remove(os.path.join(anat_preproc_dir, id+"_desc-SPGR-Ref_T1w.nii.gz"))

        ############# DESPOT MODEL FITTING #################
        despot_model_patterns = os.path.join(args.bids_dir, "derivatives", args.models_derivative_dir, "sub-{subject}[/ses-{session}]","anat",)
        despot_models_dir     = writing.build_path(entities, despot_model_patterns)

        b1map = afi_b1map
        
        ###DTI MODELING ###
        if args.despot1_fit_method != None:
            despot1_base_pattern = None
            despot1_model        = None
            if args.despot_b1_method.lower() == 'hifi':
                despot1_base_pattern = "sub-{subject}[_ses-{session}]_model-DESPOT1-HIFI_param-"
                despot1_model = "HIFI"
            else:
                despot1_base_pattern = "sub-{subject}[_ses-{session}]_model-DESPOT1_param-"
                despot1_model = "DESPOT1"

            despot1_base = writing.build_path(entities, despot1_base_pattern)
            if not os.path.exists(os.path.join(despot_models_dir,despot1_base+'T1.nii.gz')):
                if args.verbose:
                    print("Fitting DESPOT1 model with " + args.despot1_fit_method + "...")

                model = DESPOT1_Model(spgr      = spgr_preproc,
                                      params    = fitparam_json,
                                      out_dir   = despot_models_dir,
                                      out_base  = despot1_base,
                                      b1        = afi_b1map,
                                      irspgr    = irspgr_preproc,
                                      mask      = brain_mask,
                                      model     = despot1_model,
                                      algorithm = args.despot1_fit_method,
                                      nthreads  = args.nthreads,
                                      verbose   = args.verbose)

                model.fit()

                if args.despot_b1_method.lower() == 'hifi':
                    if args.verbose:
                        print('Smoothing B1')

                    os.system('fslmaths ' + brain_mask.filename + ' -s 2.55 ' + os.path.join(despot_models_dir, "tmp_mask.nii.gz"))
                    os.system('fslmaths ' + os.path.join(despot_models_dir, despot1_base+"B1.nii.gz") + " -s 2.55 -div " + os.path.join(despot_models_dir, "tmp_mask.nii.gz") + " -mas " + brain_mask.filename + " " + os.path.join(despot_models_dir, despot1_base+"B1.nii.gz") )
                    os.remove(os.path.join(despot_models_dir, "tmp_mask.nii.gz"))

                    #Refit after smoothing and fixing B1:
                    b1map = Image(filename = os.path.join(despot_models_dir, despot1_base+"B1.nii.gz"))
                    model.set_b1(b1map)
                    model.set_model(model = "DESPOT1")
                    model.fit()

        
        if args.despot2_fit_method != None:
            despot2_model = "DESPTO2-FM"
            despot2_base  = writing.build_path(entities, "sub-{subject}[_ses-{session}]_model-DESPOT2-FM_param-")            
            
            if not os.path.exists(os.path.join(despot_models_dir, despot2_base+'T2.nii.gz')):
                if args.verbose:
                    print('Fitting DESPOT2-FM model...')

                model = DESPOT2_Model(ssfp      = ssfp_preproc,
                                      params    = fitparam_json,
                                      out_dir   = despot_models_dir,
                                      out_base  = despot2_base,
                                      t1        = Image(filename = os.path.join(despot_models_dir,despot1_base+"T1.nii.gz")),
                                      b1        = b1map,
                                      mask      = brain_mask,
                                      model     = despot2_model,
                                      algorithm = args.despot2_fit_method,
                                      nthreads  = args.nthreads,
                                      verbose   = args.verbose)

                model.fit()
                

        # if args.mcdespot_fit_method != None and args.mcdespot_fit != False:
        #     mcdespot_base = bids_id + '_model-mcDESPOT_parameter-'

        #     if not os.path.exists(mcdespot_dir + mcdespot_base + 'VFm.nii.gz'):
        #         if args.verbose:
        #             print('Fitting mcDESPOT model...')


        #         mcdespot_model = MCDESPOT_Model(spgr_img    = coreg_spgr,
        #                                         ssfp_img      = coreg_ssfp,
        #                                         params        = despot_json,
        #                                         out_dir       = mcdespot_dir,
        #                                         out_base      = mcdespot_base,
        #                                         b1            = b1_map,
        #                                         f0            = Image(file=despot2_dir+despot2_base+'F0.nii.gz'),
        #                                         mask          = brain_mask,
        #                                         model         = args.mcdespot_model,
        #                                         fit_algorithm = args.mcdespot_fit_method,
        #                                         use_condor    = args.mcdespot_use_condor,
        #                                         nthreads      = args.nthreads,
        #                                         verbose       = args.verbose)

        #         if args.mcdespot_package_condor_data == True:
        #             mcdespot_model.package_condor_chunks_three_compartments()
        #         else:
        #             mcdespot_model.fit()

        # if args.despot_register_to_template == True:
        #     import core.registration.registration as reg_tools

        #     registration_dir      = bids_derivative_dir + '/despot/registration/'
        #     normalized_dir        = bids_derivative_dir + '/despot/normalized/'

        #     if not os.path.exists(registration_dir):
        #         os.makedirs(registration_dir)
        #     if not os.path.exists(normalized_dir):
        #         os.makedirs(normalized_dir)

        #     moving_img = Image(file = registration_dir + bids_id + '_desc-reference_despot-spgr.nii.gz')
        #     os.system('fslmaths ' + target_img._get_filename() + ' -mas ' + brain_mask._get_filename() + ' ' + moving_img._get_filename())
        #     img_tools.biasfield_correction(input_img    = moving_img,
        #                                    output_file  = moving_img._get_filename(),
        #                                    method       = 'N4',
        #                                    nthreads     = args.nthreads,
        #                                    iterations   = 3)

        #     if not os.path.exists(registration_dir + bids_id + '_desc-TemplateRegistration_Warped.nii.gz'):
        #         if args.verbose:
        #             print('Registering to Template')

        #         reg_tools.nonlinear_reg(input_img       = moving_img,
        #                                 reference_img   = Image(file = args.despot_registration_template),
        #                                 reference_mask  = Image(file = args.despot_registration_template_mask),
        #                                 output_base     = registration_dir + bids_id + '_desc-TemplateRegistration_',
        #                                 nthreads        = args.nthreads,
        #                                 method          = args.despot_registration_method)

        #         reg_tools.create_composite_transform(reference_img = Image(file = args.despot_registration_template),
        #                                              output_file   = registration_dir + bids_id + '_desc-TemplateRegistration_ForwardWarp.nii.gz',
        #                                              transforms    = [registration_dir + bids_id + '_desc-TemplateRegistration_1Warp.nii.gz', registration_dir + bids_id + '_desc-TemplateRegistration_0GenericAffine.mat'])

        #     #Now, warp DESPOT maps to the Template
        #     normalized_despot1_dir = normalized_dir + '/DESPOT1/'
        #     normalized_despot2_dir = normalized_dir + '/DESPOT2-FM/'
        #     normalized_mcdespot_dir = normalized_dir + '/mcDESPOT/'

        #     if args.despot_b1_method == 'HIFI':
        #         normalized_despot1_dir = normalized_dir + '/DESPOT1-HIFI/'

        #     #DESPOT1 images
        #     if os.path.exists(despot1_dir + despot1_base + 'T1.nii.gz'):
        #         if not os.path.exists(normalized_despot1_dir):
        #             os.makedirs(normalized_despot1_dir)

        #         imgs = ['T1','M0']
        #         for img in imgs:
        #             reg_tools.apply_transform(input_img     = Image(file = despot1_dir + despot1_base + img +'.nii.gz'),
        #                                       reference_img = Image(file = args.despot_registration_template),
        #                                       output_file   = normalized_despot1_dir + despot1_base + img + '.nii.gz',
        #                                       matrix        = registration_dir + bids_id + '_desc-TemplateRegistration_ForwardWarp.nii.gz',
        #                                       method        = 'ANTS')

        #     #DESPOT2 images
        #     if os.path.exists(despot2_dir + despot2_base + 'T2.nii.gz'):
        #         if not os.path.exists(normalized_despot2_dir):
        #             os.makedirs(normalized_despot2_dir)

        #         imgs = ['T2','M0', 'F0']
        #         for img in imgs:
        #             reg_tools.apply_transform(input_img     = Image(file = despot2_dir + despot2_base + img +'.nii.gz'),
        #                                       reference_img = Image(file = args.despot_registration_template),
        #                                       output_file   = normalized_despot2_dir + despot2_base + img + '.nii.gz',
        #                                       matrix        = registration_dir + bids_id + '_desc-TemplateRegistration_ForwardWarp.nii.gz',
        #                                       method        = 'ANTS')

        #         #mcDESPOT images
        #         mcdespot_base = bids_id + '_model-mcDESPOT_parameter-'
        #         if os.path.exists(mcdespot_dir + mcdespot_base + 'VFm.nii.gz'):
        #             if not os.path.exists(normalized_mcdespot_dir):
        #                 os.makedirs(normalized_mcdespot_dir)

        #             imgs = ['VFm','VFcsf', 'F0', 'T1csf', 'T2csf', 'T1m', 'T2m', 'T1f', 'T2f', 'Tau']

        #             for img in imgs:
        #                 reg_tools.apply_transform(input_img     = Image(file = mcdespot_dir + mcdespot_base + img +'.nii.gz'),
        #                                           reference_img = Image(file = args.despot_registration_template),
        #                                           output_file   = normalized_mcdespot_dir + mcdespot_base + img + '.nii.gz',
        #                                           matrix        = registration_dir + bids_id + '_desc-TemplateRegistration_ForwardWarp.nii.gz',
        #                                           method        = 'ANTS')
