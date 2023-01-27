import os,sys, shutil, json, argparse, copy
import nibabel as nib
import numpy as np


from core.utils.io import Image
import core.utils.tools as img_tools
import core.utils.mask as mask


def compute_synthetic_t2w(input_t1w, output_dir, cmd_args):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    synthetic_t2w = Image(file = output_dir + '/synT2w.nii.gz')
    
    t1w = Image(file = output_dir + '/t1w.nii.gz')
    t1w = img_tools.reorient_to_standard(input_img      = input_t1w,
                                         output_file    = t1w._get_filename())

    t1w_mask  = Image(file=output_dir + '/t1w_mask.nii.gz')
    t1w_brain = Image(file=output_dir + '/t1w_brain.nii.gz')
    mask.mask_image(input_img            = t1w,
                    output_mask          = t1w_mask,
                    output_img           = t1w_brain,
                    method               = cmd_args.anat_mask_method,
                    nthreads             = cmd_args.nthreads,
                    ref_img              = cmd_args.anat_ants_mask_template,
                    ref_mask             = cmd_args.anat_ants_mask_template_mask)
                    

    #First normalize the image
    os.system('fslmaths ' + t1w._get_filename() + ' -sub ' + t1w_brain._get_filename() + ' ' + output_dir + '/skull.nii.gz')
    os.system('ImageMath 3 ' + output_dir + '/skull.nii.gz Normalize ' + output_dir + '/skull.nii.gz')
    
    #Now, take the inverse, normalize,
    os.system('ImageMath 3 ' + output_dir + '/t1w_norm.nii.gz Normalize ' + t1w_brain._get_filename())
    os.system('fslmaths ' + output_dir + '/t1w_norm.nii.gz -recip ' + output_dir + '/t1w_recip.nii.gz' )
    os.system('ImageMath 3 ' + output_dir + '/t1w_recip.nii.gz Normalize ' + output_dir + '/t1w_recip.nii.gz')
    os.system('fslmaths ' + output_dir + '/t1w_recip.nii.gz -mas ' + t1w_mask._get_filename() + ' ' + output_dir + '/t1w_recip.nii.gz')
    
    #Now add the skull and recip-T1w_brain image
    os.system('fslmaths ' + output_dir + '/t1w_recip.nii.gz -add ' + output_dir + '/skull.nii.gz ' + synthetic_t2w._get_filename())

    
    return synthetic_t2w

    


