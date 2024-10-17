#!/usr/bin/env python
import os, sys, subprocess, shutil, time, copy


def apply_transform(input, ref, out, transform, nthreads=1, method="fsl", flirt_options=None, ants_options=None):

    CMD = ""

    if method == "fsl":
        CMD = "flirt -in " + input.filename \
              + " -ref " + ref.filename \
              + ' -out ' + out.filename \
              + ' -applyxfm -init ' + transform

        if flirt_options is not None:
            CMD += " " + flirt_options

        os.system(CMD)

    elif method == "ants":
        os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(nthreads)
        CMD = "antsApplyTransforms -d 3 -i " + input.filename \
              + " -r " + ref.filename \
              + " -o " + out.filename \
              + " -t " + transform

        if ants_options is not None:
            CMD += " " + ants_options
        
        os.system(CMD)


    elif method == "mrtrix":

        output_dir = os.path.dirname(out.filename)

        mrtrix_img = os.path.join(output_dir,"img.mif")
        warped_img = os.path.join(output_dir,"img_warped.mif")
        os.system("mrconvert -fslgrad " + input.bvecs + " " + input.bvals + " " + input.filename + " " + mrtrix_img + " -force -quiet -nthreads " + str(nthreads))

        

        if noresample:
            os.system("mrtransform " + mrtrix_img \
                      + " -linear " + transform \
                      + " " + warped_img + " " + " -force " + " -strides " +  ref.filename)

            os.system("mrconvert -force -quiet " + warped_img + " " + out.filename + " -export_grad_fsl " + out.bvecs + " " + out.bvals + " -nthreads " + str(nthreads) + " " + " -force ")

        else:
        
            ident_warp  = os.path.join(output_dir, "identity_warp")
            mrtrix_warp = os.path.join(output_dir, "mrtrix_warp")
            os.system("warpinit " + mrtrix_img + " " + ident_warp+"[].nii -force -quiet")

            for i in range(0,3):
                os.system("antsApplyTransforms -d 3 -e 0 -i " + ident_warp+str(i)+".nii -o " + mrtrix_warp+str(i)+".nii -r " + ref.filename + " -t " + transform)

            mrtrix_corr_warp = os.path.join(output_dir, "mrtrix_warp_corrected.mif")
            os.system("warpcorrect " + mrtrix_warp+"[].nii " +  mrtrix_corr_warp + " -force -quiet")

            os.system("mrtransform " + mrtrix_img \
                + " -warp " + mrtrix_corr_warp \
                + " " + warped_img + " -template " + ref.filename \
                + " -strides " +  ref.filename + " -force -quiet -reorient_fod no -nthreads " + str(nthreads) + " -interp sinc")
            
            os.system("mrconvert -force -quiet " + warped_img + " " + out.filename + " -export_grad_fsl " + out.bvecs + " " + out.bvals + " -nthreads " + str(nthreads))
            
            #Clean up files
            if method=="mrtrix":
                os.system('rm -rf ' + ident_warp+'*')
                os.system('rm -rf ' + mrtrix_warp+'*')
                os.remove(warped_img)


if __name__ == '__main__':
   

   import argparse

   parser = argparse.ArgumentParser(description='QMRI-Neuropipe Apply Normalization Transformation')
   
                    
   parser.add_argument('--input',
                    type=str,
                    help="Input image",
                    default=None)
   
   parser.add_argument('--out',
                       type=str,
                       help="Output image",
                       default=None)
   
   parser.add_argument('--ref',
                    type=str,
                    help="Reference image",
                    default=None)
   
   parser.add_argument('--method',
                       type=str,
                       help="Tool to use for dicom conversion",
                       choices=["fsl", "ants", "mrtrix", "mrtrix"],
                       default="dcm2niix")
   
   parser.add_argument('--transform',
                    type=str,
                    help="Transform to warp image ",
                    choices=["fsl", "ants", "mrtrix", "mrtrix"])
   
   parser.add_argument("--nthreads",
                       type=int,
                       help="Number of threads",
                       default=1)
   
   parser.add_argument("--flirt_options",
                       type=str,
                       help="Additional FSL-flirt options",
                       default=None)
   
   parser.add_argument("--ants_options",
                       type=str,
                       help="Additinal ANTs options",
                       default=None)
   
   parser.add_argument("--debug",
                       type=bool,
                       help="Print debugging statements",
                       default=False)
   
   args, unknown = parser.parse_known_args()
   
   apply_transform(input           = args.input,
                   ref             = args.ref,
                   out             = args.out, 
                   transform       = args.transform,
                   nthreads        = args.nthreads,
                   method          = args.method,
                   flirt_options   = args.flirt_options,
                   ants_options    = args.ants_options)
