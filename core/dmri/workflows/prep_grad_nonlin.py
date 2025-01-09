#!/usr/bin/env python
import os, sys, subprocess

from bids.layout import writing, parse_file_entities

#Neuroimaging Modules
from core.utils.io import Image
import core.utils.tools as img_tools
import core.utils.mask as mask


def grad_dev_tensor(dwi_img, gw_coils, working_dir, coregister_dwi_to_anat=False, gpu=False, debug=False, logfile=None):

    if logfile:
        sys.stdout = logfile

    graddev_dir = os.path.join(working_dir, "grad-nonlin-correction",)
    if not os.path.exists(graddev_dir):
        os.makedirs(graddev_dir)

    parsed_filename = parse_file_entities(dwi_img.filename)
    
    entities = {
    'extension': '.nii.gz',
    'subject': parsed_filename.get('subject'),
    'session': parsed_filename.get('session'),
    'suffix':  'dwi',
    'desc': 'GradNonLinTensor'
    }
    
    filename_patterns = os.path.join(working_dir, 'sub-{subject}[_ses-{session}][_desc-{desc}]_{suffix}{extension}')
    dmri_graddev_file = writing.build_path(entities, filename_patterns)
    final_img         = img_tools.calculate_mean_img(dwi_img, os.path.join(graddev_dir, "temp_img.nii.gz"), debug=debug)

    if coregister_dwi_to_anat:
        
        mean_dwi    = Image(filename=os.path.join(working_dir, "coregister-to-anatomy", "mean_dwi.nii.gz")) 
        temp_brain  = Image(filename=os.path.join(working_dir, "coregister-to-anatomy", "mean_dwi_brain.nii.gz"))  
        temp_mask   = Image(filename=os.path.join(working_dir, "coregister-to-anatomy", "mean_dwi_mask.nii.gz"))                 
        os.environ['MKL_THREADING_LAYER'] = 'GNU'
        mask.mask_image(input    = mean_dwi,
                        mask     = temp_mask,
                        mask_img = temp_brain,
                        algo     = 'hd-bet',
                        gpu      = gpu)
    
        final_img_brain = Image(filename=os.path.join(graddev_dir, "temp_img_brain.nii.gz"))
        final_img_mask  = Image(filename=os.path.join(graddev_dir, "temp_img_mask.nii.gz"))
        mask.mask_image(input    = final_img,
                        mask     = final_img_mask,
                        mask_img = final_img_brain,
                        algo     = 'hd-bet',
                        gpu      = gpu)
        
        os.system(f"CreateGradientNonlinearityBMatrix -f {final_img_brain.filename} -i {temp_brain.filename} -g {gw_coils} --isGE 1")
        os.system("gzip " + os.path.join(graddev_dir, "temp_img_brain_graddev_c.nii"))
       
        if debug:
            print(f"CreateGradientNonlinearityBMatrix -f {final_img_brain.filename} -i {temp_brain.filename} -g {gw_coils} --isGE 1")

        os.system("cp " + os.path.join(graddev_dir, "temp_img_brain_graddev_c.nii.gz") + " " + dmri_graddev_file)
        

    else:
        os.system(f"CreateGradientNonlinearityBMatrix -f {final_img.filename}  -g {gw_coils} --isGE 1")
        os.system("gzip " + os.path.join(graddev_dir, "temp_img_graddev_c.nii"))
        os.system("cp " + os.path.join(graddev_dir, "temp_img_graddev_c.nii.gz") + " " + dmri_graddev_file)


    #Clean up Temporary Files
    # if temp_img.exists():
    #     os.remove(temp_img.filename)

    file_names = [
    "temp_img_brain_graddev_c.nii.gz",
    "temp_img_graddev_c.nii.gz",
    "temp_img_brain_gradwarp_field.nii",
    "temp_img_brain_mask.nii.gz",
    "temp_img_brain.nii",
    "temp_img.nii.gz"]

    for file_name in file_names:
        file_path = os.path.join(working_dir, file_name)
        if os.path.exists(file_path):
            os.remove(file_path)

if __name__ == '__main__':

   import argparse

   parser = argparse.ArgumentParser(description='Create grad_dev_tensor file for use in gradient grad_non_lin corr')

   parser.add_argument('--input',
                    type=str,
                    help="Input image to be skull-stripped",
                    default=None)

   parser.add_argument('--gw_coils',
                       type=str,
                       help="Path to scanner spherical harmonics coefficients file gw_coils.dat",
                       default=None)

   parser.add_argument('--working_dir',
                       type=str,
                       help="Path to preprocessed data directory")

   parser.add_argument("--coregister_dwi_to_anat",
                       type=bool,
                       help="has dwi data been aligned to anat",
                       default=False)

   parser.add_argument("--use_gpu",
                       type=bool,
                       help="Use GPU for compatible methods",
                       default=False)

   parser.add_argument("--debug",
                       type=bool,
                       help="Print debugging statements",
                       default=False)

   parser.add_argument("--logfile",
                       type=str,
                       help="Log file to print statements",
                       default=None)

   args, unknown = parser.parse_known_args()


   grad_dev_tensor(                 input                  = Image(filename=args.input),
                                    gw_coils               = args.gw_coils,
                                    subj_id                = args.subject,
                                    sess_id                = args.session,
                                    working_dir            = args.working_dir,
                                    coregister_dwi_to_anat = args.coregister_dwi_to_anat,
                                    gpu                    = args.use_gpu,
                                    debug                  = args.debug,
                                    logfile                = args.logfile)
