import string, os, sys, subprocess, shutil, time, copy
from glob import glob

import numpy as np
import nibabel as nib
import nibabel.processing as nib_proc
import ants

from core.utils.io import Image, DWImage


def binarize(input_img):
    output_img = copy.deepcopy(input_img)
    subprocess.run(['fslmaths', input_img._get_filename(), '-bin', output_img._get_filename()], stderr=subprocess.STDOUT)

    return output_img

def fill_holes(input_img):
    output_img = copy.deepcopy(input_img)
    subprocess.run(['fslmaths', input_img._get_filename(), '-fillh', output_img._get_filename()], stderr=subprocess.STDOUT)

    return output_img


def reorient_to_standard(input_img, output_file, reorient_img=None):
    output_img = copy.deepcopy(input_img)
    output_img._set_filename(output_file)

    subprocess.run(['fslreorient2std',input_img._get_filename(),output_file], stderr=subprocess.STDOUT)

    if reorient_img != None:
        output_dir  = os.path.dirname(output_file)
        filename    = os.path.basename(output_file).split("_")

        reorient_xfm = os.path.join(output_dir, filename[0]+'_'+filename[1]+'_desc-Reorient2Standard_dwi.xfm')

        if os.path.exists(reorient_xfm):
            os.system('flirt -in ' + output_file + ' -ref ' + reorient_img + ' -out ' + output_file + ' -applyxfm -init ' + reorient_xfm + ' -dof 6')
        else:
            os.system('flirt -in ' + output_file + ' -ref ' + reorient_img + ' -out ' + output_file + ' -omat ' + reorient_xfm + ' -dof 6')

    return output_img

def merge_images(list_of_images, output_file):

    cmd_ = 'fslmerge -t ' + output_file

    for img in list_of_images:
        cmd_+= ' ' + img._get_filename()

    os.system(cmd_)
    #subprocess.run(cmd_, stderr=subprocess.STDOUT)

def resample_image(input_img, output_file, target_resolution, interp=3):

    # resampled_img = nib_proc.resample_to_output(in_img = input_img._get_filename(),
    #                                             voxel_sizes = target_resolution,
    #                                             order = interp)
    output_img = copy.deepcopy(input_img)
    output_img._set_filename(output_file)
    tmp_file = output_file.replace("_dwi.nii.gz","_dwi_tmp.nii.gz")

    cmd = 'mrgrid ' + input_img._get_filename() \
        + ' regrid -voxel ' + target_resolution \
        + ' -interp sinc -force ' + tmp_file

    os.system(cmd)

    os.rename(tmp_file, output_file)

    # nib.save(resampled_img, output_img._get_filename())

    return output_img


def calculate_mean_img(input_img, output_file):
    img = nib.load(input_img._get_filename())

    if len(img.shape) > 3:
        mean_data = np.mean(img.get_data(), 3)

    else:
        mean_data = img.get_data()

    mean_img = nib.Nifti1Image(mean_data.astype(np.float32), img.affine, img.header)

    output_img = copy.deepcopy(input_img)
    output_img._set_filename(output_file)
    nib.save(mean_img, output_img._get_filename())

    return output_img

def create_target_img(input_img, output_file, index=0):
    img = nib.load(input_img._get_filename())
    data = img.get_data()
    target_img = nib.Nifti1Image(data[:,:,:, index].astype(np.float32), img.affine, img.header)
    target_img.set_sform(img.get_sform())
    target_img.set_qform(img.get_qform())

    output_img = copy.deepcopy(input_img)
    output_img._set_filename(output_file)
    nib.save(mean_img , output_img._get_filename())

    return output_img

def remove_end_img(input_img, output_file):

    img = nib.load(input_img._get_filename())
    data = img.get_data()

    target_img = nib.Nifti1Image(data[:,:,:,0:img.shape[3]-1].astype(np.float32), img.affine, img.header)
    target_img.set_sform(img.get_sform())
    target_img.set_qform(img.get_qform())

    output_img = copy.deepcopy(input_img)
    output_img._set_filename(output_file)
    nib.save(target_img, output_img._get_filename())

    return output_img

def remove_end_slice(input_img, output_file):
    img = nib.load(input_img._get_filename())
    data = img.get_data()

    target_img = nib.Nifti1Image(data[:,:,0:img.shape[2]-1,:].astype(np.float32), img.affine, img.header)
    target_img.set_sform(img.get_sform())
    target_img.set_qform(img.get_qform())

    output_img = copy.deepcopy(input_img)
    output_img._set_filename(output_file)
    nib.save(target_img, output_img._get_filename())

    return output_img

def check_isotropic_voxels(input_img, output_file, target_resolution=None):

    img = nib.load(input_img._get_filename())
    # voxel_size = img.header.get_zooms()[0:2]
    voxel_size = img.header.get_zooms()[0:3]

    if not (np.all(np.isclose(voxel_size, voxel_size[0]))):

        if not target_resolution:
            target_resolution = np.repeat(max(voxel_size), 3)

        return resample_image(input_img, output_file, target_resolution)

    else:
        return input_img

def correct_header_orientation(img_path, new_x, new_y, new_z):

    img = nib.load(img_path)
    sform = img.get_sform()
    qform = img.get_qform()

    new_sform = img.get_sform()
    new_qform = img.get_qform()

    if new_x == 'y':
        new_sform[0] = sform[1]
        new_qform[0] = qform[1]
    if new_x == 'y-':
        new_sform[0] = -1.00*sform[1]
        new_qform[0] = -1.00*qform[1]
    if new_x == 'z':
        new_sform[0] = sform[2]
        new_qform[0] = qform[2]
    if new_x == 'z-':
        new_sform[0] = -1.00*sform[2]
        new_qform[0] = -1.00*qform[2]

    if new_y == 'x':
        new_sform[1] = sform[0]
        new_qform[1] = qform[0]
    if new_y == 'x-':
        new_sform[1] = -1.00*sform[0]
        new_qform[1] = -1.00*qform[0]
    if new_y == 'z':
        new_sform[1] = sform[2]
        new_qform[1] = qform[2]
    if new_y == 'z-':
        new_sform[1] = -1.00*sform[2]
        new_qform[1] = -1.00*qform[2]

    if new_z == 'x':
        new_sform[2] = sform[0]
        new_qform[2] = qform[0]
    if new_z == 'x-':
        new_sform[2] = -1.00*sform[0]
        new_qform[2] = -1.00*qform[0]
    if new_z == 'y':
        new_sform[2] = sform[1]
        new_qform[2] = qform[1]
    if new_z == 'y-':
        new_sform[2] = -1.00*sform[1]
        new_qform[2] = -1.00*qform[1]

    out_img = img
    out_img.set_sform(new_sform)
    out_img.set_qform(new_qform)
    out_img.to_filename(img_path)
