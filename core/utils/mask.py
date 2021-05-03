import string, os, sys, subprocess, shutil, time, copy

#Neuroimaging Modules
import numpy as np
import nibabel as nib
import ants
from dipy.io.image import load_nifti, save_nifti
from dipy.segment.mask import median_otsu

from core.utils.io import Image, DWImage
import core.utils.tools as img_tools

def apply_mask(input_img, mask_img, output_img):
    input = ants.image_read(input_img)
    mask = ants.image_read(mask_img)
    ants.image_write(input*mask, output_img)

def mask_image(input_img, output_mask, method='bet', nthreads=1, output_img=None, ref_img=None, ref_mask=None, bet_options='', ants_lower_threshold=0.2, antspynet_modality='t1'):

    output_root, img = os.path.split(output_mask._get_filename())
    tmp_img    = Image(file=output_root+'/temp_img.nii.gz')

    if method == 'bet':
        tmp_img = img_tools.calculate_mean_img(input_img, tmp_img._get_filename())
        subprocess.run(['bet', tmp_img._get_filename(), output_mask._get_filename(), bet_options], stderr=subprocess.STDOUT)
        output_mask = img_tools.binarize(output_mask)

    elif method == 'dipy':

        data, affine, img = load_nifti(input_img._get_filename(), return_img=True)
        masked_data, mask = median_otsu(data, 2,2)

        #Save these files
        save_nifti(output_mask._get_filename(), mask, affine, img.header)

        if output_img != None:
            save_nifti(output_img._get_filename(), masked_data, affine, img.header)

    elif method == 'afni':

        tmp_img = img_tools.calculate_mean_img(input_img,tmp_img._get_filename())
        subprocess.run(['3dSkullStrip','-input',tmp_img._get_filename(),'-prefix',output_mask._get_filename()], stderr=subprocess.STDOUT)
        output_mask = img_tools.binarize(output_mask)

    elif method == 'mrtrix':

        if input_img._get_bvals() == None or input_img._get_bvecs() == None:
            print('Need to specify B-vectors and B-values to use dwi2mask!')
            exit()

        subprocess.run(['dwi2mask','-quiet', '-force', '-fslgrad',
                        input_img._get_bvecs(),
                        input_img._get_bvals(),
                        input_img._get_filename(),
                        output_mask._get_filename(),
                        '-nthreads',str(nthreads)],stderr=subprocess.STDOUT)

    elif method == 'ants':

        os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = str(nthreads)

        if ref_img == None or ref_mask == None:
            print('Need to specify Template and Template Mask for ANTS skullstriping!')
            exit()
        
        ref_img    = Image(file = ref_img)
        ref_mask   = Image(file = ref_mask)

        ants_output = output_root + '/ants_'

        tmp_img = img_tools.calculate_mean_img(input_img,tmp_img._get_filename())
        
        mov_image       = ants.image_read(tmp_img._get_filename())
        fixed_image     = ants.image_read(ref_img._get_filename())
        fixed_mask      = ants.image_read(ref_mask._get_filename())

        mov2fixed = ants.registration(moving               = mov_image,
                                      fixed                = fixed_image,
                                      type_of_transform    = 'SyNRA',
                                      reg_iterations       = [1000,800,500,400,300,250] )

        warped_mask = ants.apply_transforms(fixed           = mov_image,
                                            moving          = fixed_mask,
                                            transformlist   = mov2fixed['invtransforms'],
                                            interpolator    = 'linear',
                                            whichtoinvert   = [True,False] )

        warped_mask_thresh = ants.threshold_image(warped_mask, ants_lower_threshold, 1, 1, 0)
        ants.image_write(warped_mask_thresh, output_mask._get_filename())
        
        os.system('rm -rf ' + output_root + '/ants*')

    elif method == 'antspynet':

        import antspynet
        
        input = ''
        if type(input_img) is list:
            
            tmp_imgA = img_tools.calculate_mean_img(input_img[0], output_root + '/tmpA.nii.gz')
            tmp_imgB = img_tools.calculate_mean_img(input_img[1], output_root + '/tmpB.nii.gz')

            image_A = ants.image_read(tmp_imgA._get_filename())
            image_B = ants.image_read(tmp_imgB._get_filename())
            input = [image_A, image_B]

        else:
            tmp_imgA = img_tools.calculate_mean_img(input_img, output_root + '/tmpA.nii.gz')
            input = ants.image_read(tmp_imgA._get_filename())


        mask_image = antspynet.brain_extraction(input, antspynet_modality)
        mask_image = ants.threshold_image(mask_image, ants_lower_threshold, 1, 1, 0)
        ants.image_write(mask_image, output_mask._get_filename())

    else:
        print('Incorrect Method for Masking Data...please see available options')
        exit(-1)


    if output_img != None:
        apply_mask(input_dwi, output_mask, output_img._get_filename())


    #Clean up Temporary Files
    os.system('rm -rf ' + output_root + '/tmp*')
    if tmp_img.exists():
        os.remove(tmp_img._get_filename())
