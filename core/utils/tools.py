import subprocess, copy
from glob import glob

import numpy as np
import nibabel as nib

def binarize(input_img, debug=False):
    output_img = copy.deepcopy(input_img)

    CMD="fslmaths " + input_img.filename + " -bin " + output_img.filename

    if debug:
        print("Binarize Image")
        print(CMD)

    #os.system(CMD)
    subprocess.run([CMD], shell=True, stderr=subprocess.STDOUT)

    return output_img

def fill_holes(input_img, debug=False):
    output_img = copy.deepcopy(input_img)

    CMD="fslmaths " + input_img.filename + " -fillh " + output_img.filename

    if debug:
        print("Fill holes")
        print(CMD)

    subprocess.run([CMD], shell=True, stderr=subprocess.STDOUT)

    return output_img


def reorient_to_standard(input_img, output_file, debug=False):
    
    output_img          = copy.deepcopy(input_img)
    output_img.filename = output_file
    
    CMD="fslreorient2std " + input_img.filename + " " + output_img.filename

    if debug:
        print("Reorienting to standard")
        print(CMD)

    subprocess.run([CMD], shell=True, stderr=subprocess.STDOUT)

    return output_img

def merge_images(list_of_images, output_img, debug=False):

    CMD="fslmerge -t " + output_img.filename

    for img in list_of_images:
        CMD+= ' ' + img.filename

    if debug:
        print("Merging images")
        print(CMD)

    subprocess.run([CMD], shell=True, stderr=subprocess.STDOUT)

def resample_image(input_img, output_file, target_resolution, debug=False):

    output_img          = copy.deepcopy(input_img)
    output_img.filename = output_file

    CMD = "mrgrid " + input_img.filename \
        + " regrid -voxel " + str(target_resolution[0]) + "," +  str(target_resolution[1]) + "," + str(target_resolution[2]) \
        + " -interp sinc " + output_img.filename + " -force -quiet"
    

    if debug:
        print("Resmapling image to target resolution: " + target_resolution)
        print(CMD)

    subprocess.run([CMD], shell=True, stderr=subprocess.STDOUT) 
    
    return output_img 


def calculate_mean_img(input_img, output_file, debug=False):

    img = nib.load(input_img.filename)

    if len(img.shape) > 3:
        mean_data = np.mean(img.get_fdata(), 3)

    else:
        mean_data = img.get_fdata()

    if debug:
        print("Calculating image mean")

    mean_img = nib.Nifti1Image(mean_data.astype(np.float32), img.affine, img.header)

    output_img = copy.deepcopy(input_img)
    output_img.filename = output_file
    nib.save(mean_img, output_img.filename)

    return output_img

def create_target_img(input_img, output_file, index=0, debug=False):
    img = nib.load(input_img.filename)
    data = img.get_fdata()

    if debug:
        print("Extracting target image")

    target_img = nib.Nifti1Image(data[:,:,:, index].astype(np.float32), img.affine, img.header)
    target_img.set_sform(img.get_sform())
    target_img.set_qform(img.get_qform())

    output_img = copy.deepcopy(input_img)
    output_img.filename = output_file
    nib.save(mean_img , output_img.filename)

    return output_img

def remove_end_img(input_img, output_file, debug=False):

    img = nib.load(input_img.filename)
    data = img.get_fdata()

    if debug:
        print("Removing last image from series")

    target_img = nib.Nifti1Image(data[:,:,:,0:img.shape[3]-1].astype(np.float32), img.affine, img.header)
    target_img.set_sform(img.get_sform())
    target_img.set_qform(img.get_qform())

    output_img = copy.deepcopy(input_img)
    output_img.filename = output_file
    nib.save(target_img, output_img.filename)

    return output_img

def remove_end_slice(input_img, output_file, debug=False):
    img = nib.load(input_img.filename)
    data = img.get_fdata()

    if debug:
        print("Removing last slice from image")

    target_img = nib.Nifti1Image(data[:,:,0:img.shape[2]-1,:].astype(np.float32), img.affine, img.header)
    target_img.set_sform(img.get_sform())
    target_img.set_qform(img.get_qform())

    output_img = copy.deepcopy(input_img)
    output_img.filename = output_file
    nib.save(target_img, output_img.filename)

    return output_img

def check_isotropic_voxels(input_img, output_file, target_resolution=None, debug=False):

    img = nib.load(input_img.filename)
    voxel_size = img.header.get_zooms()[0:3]

    if debug:
        print("Checking voxel resolution")

    if not (np.all(np.isclose(voxel_size, voxel_size[0]))):

        if not target_resolution:
            target_resolution = np.repeat(max(voxel_size), 3)

            print(target_resolution)

        return resample_image(input_img, output_file, target_resolution)

    elif target_resolution:
        print ('Resampling Image Voxels')
        return resample_image(input_img, output_file, np.fromstring(target_resolution, dtype=float, sep=' '))
    else:
        return input_img

def correct_header_orientation(img_path, new_x, new_y, new_z, debug=False):

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
