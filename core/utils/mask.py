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
    os.system('fslmaths ' + input_img._get_filename() + ' -mas ' + mask_img._get_filename() + ' ' + output_img._get_filename())

def mask_image(input_img, output_mask, method='bet', nthreads=1, output_img=None, ref_img=None, ref_mask=None, bet_options='', ants_lower_threshold=0.2, antspynet_modality='t1'):

    output_root, img = os.path.split(output_mask._get_filename())
    tmp_img    = Image(file=output_root+'/temp_img.nii.gz')

    if method == 'bet':

        if type(input_img) is list:
            input_img = input_img[0]

        tmp_img = img_tools.calculate_mean_img(input_img, tmp_img._get_filename())
        os.system('bet ' + tmp_img._get_filename() + ' ' + output_mask._get_filename() + ' ' + bet_options)
        output_mask = img_tools.binarize(output_mask)
        output_mask = img_tools.fill_holes(output_mask)

    elif method == 'hd-bet':
        os.environ['MKL_THREADING_LAYER'] = 'GNU'

        if type(input_img) is list:
            input_img = input_img[0]

        tmp_mask = output_root + '/temp_mask'
        tmp_img = img_tools.calculate_mean_img(input_img, tmp_img._get_filename())

        #print('hd-bet -i ' + tmp_img._get_filename() + ' -o ' + tmp_mask)
        os.system('hd-bet -i ' + tmp_img._get_filename() + ' -o ' + tmp_mask)
        os.rename(tmp_mask+'_mask.nii.gz', output_mask._get_filename())

    elif method == 'dipy':

        if type(input_img) is list:
            input_img = input_img[0]

        data, affine, img = load_nifti(input_img._get_filename(), return_img=True)
        masked_data, mask = median_otsu(data, 2,2)

        #Save these files
        save_nifti(output_mask._get_filename(), mask, affine, img.header)

        if output_img != None:
            save_nifti(output_img._get_filename(), masked_data, affine, img.header)

    elif method == 'afni':

        if type(input_img) is list:
            input_img = input_img[0]

        tmp_img = img_tools.calculate_mean_img(input_img,tmp_img._get_filename())
        subprocess.run(['3dSkullStrip','-input',tmp_img._get_filename(),'-prefix',output_mask._get_filename()], stderr=subprocess.STDOUT)
        output_mask = img_tools.binarize(output_mask)

    elif method == 'mrtrix':

        if type(input_img) is list:
            input_img = input_img[0]

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

        if type(input_img) is list:
            input_img = input_img[0]

        ref_img    = Image(file = ref_img)
        ref_mask   = Image(file = ref_mask)

        ants_output = output_root + '/ants_'

        tmp_img = img_tools.calculate_mean_img(input_img,tmp_img._get_filename())
        cmd = 'antsBrainExtraction.sh -d 3 -a ' + tmp_img._get_filename() \
            + ' -e ' + ref_img._get_filename() \
            + ' -m ' + ref_mask._get_filename() \
            + ' -o ' + ants_output \
            + ' -u 0'
        os.system(cmd)

        # mov_image       = ants.image_read(tmp_img._get_filename())
        # fixed_image     = ants.image_read(ref_img._get_filename())
        # fixed_mask      = ants.image_read(ref_mask._get_filename())
        #
        # mov2fixed = ants.registration(moving               = mov_image,
        #                               fixed                = fixed_image,
        #                               type_of_transform    = 'SyNRA',
        #                               reg_iterations       = [1000,800,500,400,300,250] )
        #
        # warped_mask = ants.apply_transforms(fixed           = mov_image,
        #                                     moving          = fixed_mask,
        #                                     transformlist   = mov2fixed['invtransforms'],
        #                                     interpolator    = 'linear',
        #                                     whichtoinvert   = [True,False] )
        #
        # warped_mask_thresh = ants.threshold_image(warped_mask, ants_lower_threshold, 1, 1, 0)
        #ants.image_write(warped_mask_thresh, output_mask._get_filename())

        os.rename(ants_output+'BrainExtractionMask.nii.gz', output_mask._get_filename())
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

        mask_image = antspynet.brain_extraction(image=input,
                                                modality = antspynet_modality)
        mask_image = ants.threshold_image(mask_image, ants_lower_threshold, 1, 1, 0)
        ants.image_write(mask_image, output_mask._get_filename())

    else:
        print('Incorrect Method for Masking Data...please see available options')
        exit(-1)

    if output_img != None:
        apply_mask(input_img, output_mask, output_img._get_filename())


    #Clean up Temporary Files
    os.system('rm -rf ' + output_root + '/tmp*')
    if tmp_img.exists():
        os.remove(tmp_img._get_filename())
