#!/usr/bin/env python

import os, subprocess
from core.utils.io import Image

from core.registration.create_composite_transform import create_composite_transform
      
def nonlinreg(input, ref, mask=None, out_xfm=None, out_xfm_base=None, out_img=None, nthreads=1, method='ants', fsl_options=None, ants_options=None):

    CMD = ""

    if method == "ants" or method == 'ants-quick':

        os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = str(nthreads)

        if method == 'ants-quick':
            CMD = "antsRegistrationSyNQuick.sh"
        else:
            CMD = "antsRegistrationSyN.sh"


        CMD += " -d 3 -n " + str(nthreads) + " -o " + out_xfm_base

        if type(input) is list:
            for i in range(0,len(input)):
                CMD += " -m " + input[i].filename \
                    +  " -f " + ref[i].filename
        else:
                CMD += " -m " + input.filename \
                    +  " -f " + ref.filename

        if mask != None:
            CMD += " -x " + mask.filename

        if ants_options != None:
            CMD += ' ' + ants_options


    elif method == "fsl":

        #First, use flirt to create affine transform
        flirt_aff =  out_xfm.split(".")[0]+".mat"
        flirt_opts = "-searchrx -180 180 -searchry -180 180 -searchrz -180 180"

        from linreg import linreg
        linreg(input, ref, out_mat=flirt_aff, dof=12, method="fsl", flirt_options=flirt_opts)

        CMD = "fnirt --ref=" + ref[0].filename \
            + " --in=" + input[0].filename \
            + " --aff=" + flirt_aff \
            + " --fout=" + out_xfm
        
        if mask !=None:
            CMD += " --refmask=" + mask.filename
        
        if out_img:
            CMD += " --iout="+out_img[0].filename
    else:
        print("Incorrect nonlinear registration method")
        exit(-1)

    
    print(CMD)
    subprocess.run([CMD], shell=True, stderr=subprocess.STDOUT)
    
    if (method == 'ants' or method == 'ants-quick') and (out_xfm != None):

        ants_transforms = [out_xfm_base + "1Warp.nii.gz", out_xfm_base + "0GenericAffine.mat"]

        ref_img = ref
        if type(ref) is list:
            ref_img = ref[0]
            
        create_composite_transform(ref = ref_img,
                                   out = out_xfm,
                                   transforms = ants_transforms)


if __name__ == '__main__':
   
   import argparse

   parser = argparse.ArgumentParser(description='QMRI-Neuropipe Non-linear Registration Tool')
   
   parser.add_argument('--input',
                    type=str, nargs='+',
                    help="Input image",
                    default=None)

   parser.add_argument('--ref',
                    type=str, nargs='+',
                    help="Reference image",
                    default=None)
   
   parser.add_argument('--mask',
                    type=str,
                    help="Reference brain mask",
                    default=None)
   
   parser.add_argument('--out_xfm',
                       type=str,
                       help="Output matrix transformation",
                       default=None)

   parser.add_argument('--out_img',
                       type=str, nargs='+',
                       help="Output image",
                       default=None)
   
   parser.add_argument('--method',
                       type=str,
                       help="Linear registration method",
                       choices=["fsl", "ants", "ants-quick"],
                       default="fsl")   
   
   parser.add_argument('--nthreads',
                       type=int,
                       help="Number of threads (for multi-threaded applications)",
                       default=1)
   
   parser.add_argument('--fnirt_options',
                       type=str,
                       help="Additinoal FSL FNIRT options",
                       default=None)  
   
   parser.add_argument('--ants_options',
                       type=str,
                       help="Additinoal ANTs options",
                       default=None)
   
   parser.add_argument("--debug",
                       type=bool,
                       help="Print debugging statements",
                       default=False)
   
   args, unknown = parser.parse_known_args()
   
   
   input_imgs = []
   ref_imgs   = []
   out_imgs   = []
   mask_img   = None


   for img in args.input:
       input_imgs.append(Image(filename=img))

   for img in args.ref:
       ref_imgs.append(Image(filename=img))

   for img in args.out_img:
       out_imgs.append(Image(filename=img))

   if args.mask != None:
       mask_img = Image(args.mask) 

       
   nonlinreg(input         = input_imgs,
             ref           = ref_imgs,
             mask          = mask_img, 
             out_xfm       = args.out_xfm,
             out_img       = out_imgs,
             nthreads      = args.nthreads,
             method        = args.method,
             fsl_options   = args.fnirt_options, 
             ants_options  = args.ants_options)
   
   
   
 
