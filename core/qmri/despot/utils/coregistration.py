import string, os, sys, subprocess, shutil, time, json, copy
from glob import glob

import nibabel as nib
from core.utils.io import Image
from core.registration.linreg import linreg

def coregister_images(input_img, reference_img, output_img, method='fsl'):

    data_shape = nib.load(input_img._get_filename()).shape

    num_imgs = 1
    if len(data_shape) == 4:
        num_imgs = data_shape[3]


    output_dir = os.path.dirname(os.path.realpath(output_img._get_filename()))
    output_base = os.path.basename(output_img._get_filename())

    if output_base.endswith('.nii.gz'):
        output_base = output_base[:-7]
    else:
        output_base = output_base[:-3]

    tmp_dir = output_dir + '/tmp/'
    if os.path.exists(tmp_dir):
        os.system('rm -rf ' + tmp_dir)
    os.makedirs(tmp_dir)
    os.system('fslsplit ' + input_img._get_filename() + ' ' + tmp_dir + 'tmp_ -t')

    list_of_imgs = os.listdir(tmp_dir)

    fslmerge_cmd = 'fslmerge -t ' + output_img._get_filename()
    for i in range(0,num_imgs):
        moving_img  = Image(file = tmp_dir + list_of_imgs[i])
        tmp_out_img = tmp_dir + 'coreg_img_'+str(i).zfill(4)
        output_mat = tmp_dir + 'coreg_img_'+str(i).zfill(4)+'.mat'

        linreg(input   = moving_img,
               ref     = reference_img,
               out_mat = output_mat,
               out     = tmp_out_img,
               dof     = 6,
               method  = method)
               
        fslmerge_cmd += ' ' + tmp_out_img

    os.system(fslmerge_cmd)
    os.system('rm -rf ' + tmp_dir)
