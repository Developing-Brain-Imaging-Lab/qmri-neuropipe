import os
from core.utils.io import Image
from core.registration.nonlinreg import nonlinreg
from core.registration.apply_transform import apply_transform
from core.registration.create_composite_transform import create_composite_transform

def dmri_to_standard(bids_id, dwi_models_dir, dwi_registration_dir, dwi_normalized_dir, template, template_mask, method="ants", nthreads=1, fslopts=None, antsopts=None  ):
     
    if not os.path.exists(dwi_registration_dir):
        os.makedirs(dwi_registration_dir)

    if not os.path.exists(dwi_normalized_dir):
        os.makedirs(dwi_normalized_dir)


    #Use ANTS to do the registration
    dti_dir     = os.path.join(dwi_models_dir, "DTI")
    fa_map      = Image(filename=os.path.join(dti_dir, bids_id+"_model-DTI_parameter-FA.nii.gz"))
    output_base = os.path.join(dwi_registration_dir, bids_id+"_desc-RegistrationToStandard_")
    output_warp = os.path.join(dwi_registration_dir, bids_id+"_desc-RegistrationToStandard_Warp.nii.gz")

    if os.path.exists(fa_map.filename) and not os.path.exists(output_warp):
        nonlinreg(input       = fa_map, 
                  ref         = template, 
                  mask        = template_mask,
                  out_xfm     = output_base,
                  nthreads    = nthreads,
                  method      = method,
                  fsl_options = fslopts,
                  ants_options= antsopts)
        
        create_composite_transform(ref        = template,
                                   out        = output_warp,
                                   transforms = [output_base + '1Warp.nii.gz', output_base+'0GenericAffine.mat'])

    #Warp DTI model parameters
    scalars_to_warp = []
    scalars_to_warp.append("model-DTI_parameter-FA.nii.gz")
    scalars_to_warp.append("model-DTI_parameter-MD.nii.gz") 
    scalars_to_warp.append("model-DTI_parameter-RD.nii.gz")
    scalars_to_warp.append("model-DTI_parameter-AD.nii.gz")
    scalars_to_warp.append("model-DTI_parameter-L1.nii.gz")
    scalars_to_warp.append("model-DTI_parameter-L2.nii.gz")
    scalars_to_warp.append("model-DTI_parameter-L3.nii.gz")

    vectors_to_warp = []
    vectors_to_warp.append("model-DTI_parameter-V1.nii.gz")
    vectors_to_warp.append("model-DTI_parameter-V2.nii.gz") 
    vectors_to_warp.append("model-DTI_parameter-V3.nii.gz")

    tensors_to_warp = []
    tensors_to_warp.append("model-DTI_parameter-TENSOR.nii.gz") 
    tensors_to_warp.append("model-DTI_parameter-FSL_TENSOR.nii.gz") 
    tensors_to_warp.append("model-DTI_parameter-MRTRIX_TENSOR.nii.gz") 

    norm_dti_dir = os.path.join(dwi_normalized_dir, "DTI")

    if os.path.exists(os.path.join(dti_dir, bids_id+"_"+scalars_to_warp[0])) and not os.path.exists(os.path.join(norm_dti_dir, bids_id+"_space-Standard_"+scalars_to_warp[0])):    
        if not os.path.exists(norm_dti_dir):
            os.makedirs(norm_dti_dir)

        for param in scalars_to_warp:
            in_img  = Image(filename = os.path.join(dti_dir, bids_id+"_"+param))
            out_img = Image(filename = os.path.join(norm_dti_dir, bids_id+"_space-Standard_"+param))

            if not os.path.exists(out_img.filename):
                apply_transform(input         = in_img,
                                ref           = template,
                                out           = out_img,
                                transform     = output_warp,
                                nthreads      = nthreads,
                                method        = "ants")
                
        for param in vectors_to_warp:
            in_img  = Image(filename = os.path.join(dti_dir, bids_id+"_"+param))
            out_img = Image(filename = os.path.join(norm_dti_dir, bids_id+"_space-Standard_"+param))

            if not os.path.exists(out_img.filename):
                apply_transform(input         = in_img,
                                ref           = template,
                                out           = out_img,
                                transform     = output_warp,
                                nthreads      = nthreads,
                                method        = "ants", 
                                ants_options  = " -e 1")

        for param in tensors_to_warp:
            in_img  = Image(filename = os.path.join(dti_dir, bids_id+"_"+param))
            out_img = Image(filename = os.path.join(norm_dti_dir, bids_id+"_space-Standard_"+param))

            if not os.path.exists(out_img.filename):
                apply_transform(input         = in_img,
                                ref           = template,
                                out           = out_img,
                                transform     = output_warp,
                                nthreads      = nthreads,
                                method        = "ants", 
                                ants_options  = " -e 2")

    # #Add in Warp of other model parameters (if exists)
    # noddi_dir = os.path.join(models_dir, args.noddi_fit_method)

    # if os.path.exists(noddi_dir):
    #     norm_noddi_dir  = os.path.join(normalization_dir, args.noddi_fit_method)
    #     if not os.path.exists(norm_noddi_dir):
    #         os.makedirs(norm_noddi_dir)

    #     noddi_to_warp = []
    #     noddi_to_warp.append("model-NODDI_parameter-ICVF.nii.gz")
    #     noddi_to_warp.append("model-NODDI_parameter-ODI.nii.gz") 
    #     noddi_to_warp.append("model-NODDI_parameter-EXVF.nii.gz")
    #     noddi_to_warp.append("model-NODDI_parameter-FISO.nii.gz")

    #     for param in noddi_to_warp:
    #         in_img  = os.path.join(noddi_dir, bids_id+"_"+param)
    #         out_img = os.path.join(norm_noddi_dir, bids_id+"_space-Standard_"+param)

    #         if not os.path.exists(out_img):
    #             cmd = "antsApplyTransforms -d 3 -i " + in_img \
    #                 + ' -o ' + out_img \
    #                 + ' -r ' + args.dwi_standard_template \
    #                 + ' -t ' + output_base+"1Warp.nii.gz" \
    #                 + ' -t ' + output_base+"0GenericAffine.mat"
    #             os.system(cmd)

    # #Add in Warp of other model parameters (if exists)
    # dki_dir = os.path.join(models_dir, "DKI")

    # if os.path.exists(dki_dir):
    #     norm_dki_dir  = os.path.join(normalization_dir, "DKI")
    #     if not os.path.exists(norm_dki_dir):
    #         os.makedirs(norm_dki_dir)

    #     dki_to_warp = []
    #     dki_to_warp.append("model-DKI_parameter-FA.nii.gz")
    #     dki_to_warp.append("model-DKI_parameter-MD.nii.gz") 
    #     dki_to_warp.append("model-DKI_parameter-RD.nii.gz")
    #     dki_to_warp.append("model-DKI_parameter-AD.nii.gz")
    #     dki_to_warp.append("model-DKI_parameter-AK.nii.gz")
    #     dki_to_warp.append("model-DKI_parameter-KFA.nii.gz") 
    #     dki_to_warp.append("model-DKI_parameter-MK.nii.gz")
    #     dki_to_warp.append("model-DKI_parameter-RK.nii.gz")
    #     dki_to_warp.append("model-DKI_parameter-MKT.nii.gz")

    #     for param in dki_to_warp:
    #         in_img  = os.path.join(dki_dir, bids_id+"_"+param)
    #         out_img = os.path.join(norm_dki_dir, bids_id+"_space-Standard_"+param)

    #         if not os.path.exists(out_img):
    #             cmd = "antsApplyTransforms -d 3 -i " + in_img \
    #                 + ' -o ' + out_img \
    #                 + ' -r ' + args.dwi_standard_template \
    #                 + ' -t ' + output_base+"1Warp.nii.gz" \
    #                 + ' -t ' + output_base+"0GenericAffine.mat"
    #             os.system(cmd)

    # #Add in Warp of other model parameters (if exists)
    # fwe_dir = os.path.join(models_dir, "FWE-DTI")

    # if os.path.exists(fwe_dir):
    #     norm_fwe_dir  = os.path.join(normalization_dir, "FWE-DTI")
    #     if not os.path.exists(norm_fwe_dir):
    #         os.makedirs(norm_fwe_dir)

    #     fwe_to_warp = []
    #     fwe_to_warp.append("model-FWE-DTI_parameter-FA.nii.gz")
    #     fwe_to_warp.append("model-FWE-DTI_parameter-MD.nii.gz") 
    #     fwe_to_warp.append("model-FWE-DTI_parameter-RD.nii.gz")
    #     fwe_to_warp.append("model-FWE-DTI_parameter-AD.nii.gz")
    #     fwe_to_warp.append("model-FWE-DTI_parameter-F.nii.gz")

    #     for param in fwe_to_warp:
    #         in_img  = os.path.join(fwe_dir, bids_id+"_"+param)
    #         out_img = os.path.join(norm_fwe_dir, bids_id+"_space-Standard_"+param)

    #         if not os.path.exists(out_img):
    #             cmd = "antsApplyTransforms -d 3 -i " + in_img \
    #                 + ' -o ' + out_img \
    #                 + ' -r ' + args.dwi_standard_template \
    #                 + ' -t ' + output_base+"1Warp.nii.gz" \
    #                 + ' -t ' + output_base+"0GenericAffine.mat"
    #             os.system(cmd)
        





    # def __init__(self, verbose=False):
    #     if verbose:
    #         print('qmri-neuropipe Diffusion Normalization Pipeline')

    # def run(self):

    #     parser = argparse.ArgumentParser()

    #     parser.add_argument('--bids_dir',
    #                         type=str,
    #                         help='BIDS Data Directory')
        
    #     parser.add_argument('--bids_pipeline_name',
    #                 type=str, help='BIDS PIPELINE Name',
    #                 default='qmri-neuropipe')
    
    #     parser.add_argument('--load_json',
    #                     type=str, 
    #                     help='Load settings from file in json format. Command line options are overriden by values in file.',
    #                     default=None)
        
    #     parser.add_argument('--nthreads',
    #                     type=int,
    #                     help='Number of Threads',
    #                     default=1)
        
    #     parser.add_argument('--subject',
    #                         type=str,
    #                         help='Subject ID')

    #     parser.add_argument('--session',
    #                         type=str,
    #                         help='Subject Timepoint',
    #                         default=None)
        
    #     parser.add_argument('--dwi_standard_template',
    #                         type=str,
    #                         help='Template to use for registration',
    #                         default=None)
    
    #     args, unknown = parser.parse_known_args()
        
    #     if args.load_json:
    #         with open(args.load_json, 'rt') as f:
    #             t_args = argparse.Namespace()
    #             t_dict = vars(t_args)
    #             t_dict.update(json.load(f))
    #             args, unknown = parser.parse_known_args(namespace=t_args)

    #     #Setup the BIDS Directories and Paths
    #     entities = {
    #     'extension': '.nii.gz',
    #     'subject': args.subject,
    #     'session': args.session,
    #     'modality': 'dwi',
    #     'suffix': 'dwi'
    #     }

    #     id_patterns = 'sub-{subject}[_ses-{session}]'
    #     derivative_patterns = args.bids_dir + '/derivatives/' + args.bids_pipeline_name + '/sub-{subject}[/ses-{session}]/'

    #     bids_id             = writing.build_path(entities, id_patterns)
    #     bids_derivative_dir = writing.build_path(entities, derivative_patterns)

    #     models_dir          = os.path.join(bids_derivative_dir, "dwi", "models" )
    #     registration_dir    = os.path.join(bids_derivative_dir, "dwi", 'registration-to-standard/')
    #     normalization_dir   = os.path.join(bids_derivative_dir, "dwi", 'models-standard/')

    #     i