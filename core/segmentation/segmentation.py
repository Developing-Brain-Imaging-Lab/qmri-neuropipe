import string, os, sys, subprocess, shutil, time, copy

import core.registration.registration as reg_tools


def atlas_segmentation(target_img, atlas, label, output_seg_file, nthreads=1, verbose=False):

    #Coregister the atlas to the target image
    output_dir = os.path.dirname(output_seg_file)
    output_prefix = output_dir + '/atlas_segmentation_'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    reg_tools.nonlinear_reg(input_img     = atlas,
                            reference_img = target_img,
                            output_base   = output_prefix,
                            nthreads      = nthreads,
                            method        = 'ANTS-QUICK')

    #Create overall warp file
    composite_transform = output_dir + '/atlas_segmentation_warp.nii.gz'

    reg_tools.create_composite_transform(reference_img  = target_img,
                                         output_file    = composite_transform,
                                         transforms     = [output_prefix + '1Warp.nii.gz', output_prefix + '0GenericAffine.mat'])

    #Warp labels to target
    reg_tools.apply_transform(input_img     = label,
                              reference_img = target_img,
                              output_file   = output_seg_file,
                              matrix        = composite_transform,
                              nthreads      = nthreads,
                              method        = 'ANTS',
                              ants_options  = '-n GenericLabel')

    os.system('rm -rf ' + output_prefix + '*')

def multi_atlas_segmentation(target_img, atlases, labels, output_seg_file, nthreads=1, verbose=False):

    output_dir = os.path.dirname(output_seg_file)
    output_prefix = output_dir + '/ants_multi_seg'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cmd = 'antsJointLabelFusion.sh -d 3 -j ' + str(nthreads)+ ' -t ' + target_img._get_filename() + ' -o ' + output_prefix

    for i in range(0,len(atlases)):
        cmd += ' -g ' + atlases[i] + ' -l ' + labels[i]

    os.system(cmd)
    os.rename(output_prefix+'Labels.nii.gz', output_seg_file)
    
    os.system('rm -rf ' + output_prefix + '*')
