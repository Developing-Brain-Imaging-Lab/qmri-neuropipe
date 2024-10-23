#!/usr/bin/env python
import os, sys, subprocess, shutil

#Neuroimaging Modules
import ants
from dipy.io.image import load_nifti, save_nifti
from dipy.segment.mask import median_otsu

from core.utils.io import Image, DWImage
import core.utils.tools as img_tools

def apply_mask(input, mask, output, debug=False):
    CMD = "fslmaths " + input.filename + " -mas " + mask.filename + " " + output.filename

    if debug:
        print("Applying mask image")
        print(CMD)

    os.system(CMD)

def mask_image(input, mask, mask_img=None, algo="bet", nthreads=1, gpu=False, gpu_device=0, ref_img=None, ref_mask=None, bet_options="", ants_lower_threshold=0.2, antspynet_modality="t1", debug=False, logfile=None):
    
    if logfile:
        sys.stdout = logfile

    input_dir  = os.path.dirname(input.filename)
    output_dir, img = os.path.split(mask.filename)

    temp_img    = img_tools.calculate_mean_img(input, os.path.join(output_dir, "temp_img.nii.gz"), debug=debug)
    CMD=""

    if algo == "bet": 
        CMD = "bet " + temp_img.filename+ " " + mask.filename + " " + bet_options

        if debug:
            print(CMD)

        subprocess.run([CMD], shell=True, stdout=logfile)
        mask = img_tools.binarize(mask, debug=debug)

    elif algo == "hd-bet":
        os.environ['MKL_THREADING_LAYER'] = 'GNU'

        temp_mask = os.path.join(output_dir, "temp_mask")
        CMD = "hd-bet -i " + temp_img.filename + " -mode accurate -tta 0 -o " + temp_mask
        
        if gpu:
            CMD+= f" -device {gpu_device}"
        else:
            CMD+= " -device cpu"

        if debug:
            print(CMD)

        subprocess.run([CMD], shell=True, stdout=logfile)
        os.rename(temp_mask+"_mask.nii.gz", mask.filename)
        os.remove(temp_mask+".nii.gz")

    elif algo == "dipy":

        data, affine, img = load_nifti(input.filename, return_img=True)
        masked_data, binary_mask = median_otsu(data, 2,2)

        #Save these files
        save_nifti(mask.filename, binary_mask, affine, img.header)

        if mask_img != None:
            save_nifti(mask_img.filename, masked_data, affine, img.header)

    elif algo == "afni":

        CMD = "3dSkullStrip -input " + temp_img.filename + " -prefix " + mask.filename

        if debug:
            print(CMD)

        subprocess.run([CMD], shell=True, stdout=logfile)
        mask = img_tools.binarize(mask, debug=debug)

    elif algo == "mrtrix":

        if input.get_type() != "DWImage":
            print("MRtrix3 skull-stripping is only compatible with diffusion data. Change skull-stripping algorithm")
            exit(-1)

        if input.bvals == None or input.bvecs == None:
            print('Need to specify B-vectors and B-values to use dwi2mask!')
            exit(-1)

        CMD = "dwi2mask -quiet -force -fslgrad " + input.bvecs + " " + input.bvals \
            + input.filename + " " + mask.filename + " -nthreads " + str(nthreads) 

        if debug:
            print(CMD)

        subprocess.run([CMD], shell=True, stdout=logfile)
        
    elif algo == "mri_synthstrip":
        
        CMD = "mri_synthstrip -i " + temp_img.filename + " -o " + mask.filename
        
        if debug:
            print(CMD)

        subprocess.run([CMD], shell=True, stdout=logfile)
        mask = img_tools.binarize(mask, debug=debug)


    elif algo == 'ants':

        os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = str(nthreads)

        if ref_img == None or ref_mask == None:
            print('Need to specify Template and Template Mask for ANTS skullstriping!')
            exit(-1)

        ants_output = output_dir + '/ants_'
    
        CMD = "antsRegistrationSyN.sh -d 3 -j 1 -y 1 -n " + str(nthreads) \
                 + " -f " + temp_img.filename \
                 + " -m " + ref_img \
                 + " -o " + ants_output

        if debug:
            print(CMD)

        subprocess.run([CMD], shell=True, stdout=logfile)
    
        #Warp the mask
        CMD = "antsApplyTransforms -d 3 -n NearestNeighbor" \
                 + " -i " + ref_mask \
                 + " -r " + temp_img.filename \
                 + " -o " + mask.filename
        if debug:
            print(CMD)
        subprocess.run([CMD], shell=True, stdout=logfile)
        os.system("rm -rf " + os.path.join(output_dir, "ants*"))

    elif algo == 'antspynet':

        import antspynet

        ants_input = ants.image_read(temp_img.filename)
        mask_image = antspynet.brain_extraction(image=ants_input,
                                                modality = antspynet_modality)
        mask_image = ants.threshold_image(mask_image, ants_lower_threshold, 1, 1, 0)
        ants.image_write(mask_image, mask.filename)

    else:
        print('Incorrect Method for Masking Data...please see available options')
        exit(-1)

    if mask_img !=None and mask_img.filename != None:
        apply_mask(input, mask, mask_img)


    #Clean up Temporary Files
    if temp_img.exists():
        os.remove(temp_img.filename)
    


if __name__ == '__main__':
   
   import argparse
   
   parser = argparse.ArgumentParser(description='QMRI-Neuropipe Skull-stripping function')
   
   parser.add_argument('--input',
                    type=str,
                    help="Input image to be skull-stripped",
                    default=None)
   
   parser.add_argument('--mask',
                       type=str,
                       help="Output binary mask",
                       default=None)
   
   parser.add_argument('--mask_img',
                       type=str,
                       help="Output masked data",
                       default=None)
   
   parser.add_argument('--algo',
                       type=str,
                       help="Skull-stripping algorithm",
                       choices=["bet", "hd-bet", "dipy", "mrtrix", "afni", "ants", "antspynet"],
                       default="bet")
   
   parser.add_argument("--nthreads",
                       type=int,
                       help="Number of threads",
                       default=1)
   
   parser.add_argument("--use_gpu",
                       type=bool,
                       help="Use GPU for compatible methods",
                       default=False)
   
   parser.add_argument("--reference_image",
                       type=str,
                       help="Reference image for registration based methods",
                       default=None)
   
   parser.add_argument("--reference_mask",
                       type=str,
                       help="Reference mask for registration based methods",
                       default=None)    
   
   parser.add_argument("--bet_options",
                       type=str,
                       help="FSL BET options",
                       default="")
   
   parser.add_argument("--ants_lower_threshold",
                       type=float,
                       help="Lower threshold for ANTS",
                       default=0.2)  
   
   parser.add_argument("--antspynet_modality",
                       type=str,
                       help="ANTSPYNET Image Modality",
                       choices=["t1", "t2", "t1infant", "t2infant", "fa"],
                       default="t1")
   
   parser.add_argument("--debug",
                       type=bool,
                       help="Print debugging statements",
                       default=False)
   
   parser.add_argument("--logfile",
                       type=str,
                       help="Log file to print statements",
                       default=None)            
   
   args, unknown = parser.parse_known_args()
   
   mask_image(input                = Image(filename=args.input),
              mask                 = Image(filename=args.mask),
              mask_img             = Image(filename=args.mask_img),
              algo                 = args.algo,
              nthreads             = args.nthreads,
              gpu                  = args.use_gpu,
              ref_img              = Image(filename=args.reference_image),
              ref_mask             = Image(filename=args.reference_mask),
              bet_options          = args.bet_options,
              ants_lower_threshold = args.ants_lower_threshold,
              antspynet_modality   = args.antspynet_modality,
              debug                = args.debug,
              logfile              = args.logfile)



