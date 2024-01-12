#!/usr/bin/env python
import os, shutil

import nibabel as nib
from core.utils.io import Image
from core.registration.linreg import linreg

def multilinreg(input, ref, out, dof=6, nthreads=1, method="fsl", flirt_options=None, ants_options=None, freesurfer_subjs_dir=None, debug=False):

    output_dir  = os.path.dirname(os.path.realpath(out.filename))
    output_base = os.path.basename(out.filename)

    if output_base.endswith(".nii.gz"):
        output_base = output_base[:-7]
    else:
        output_base = output_base[:-3]

    tmp_dir = os.path.join(output_dir, "tmp",)
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.makedirs(tmp_dir)
    os.system("fslsplit " + input.filename + " " + os.path.join(tmp_dir,"tmp_") + " -t")

    list_of_imgs = os.listdir(tmp_dir)
    fslmerge_cmd = "fslmerge -t " + out.filename
    for i in range(0,len(list_of_imgs)):
        moving_img  = Image(filename = os.path.join(tmp_dir,list_of_imgs[i]))
        tmp_out_img = Image(filename = os.path.join(tmp_dir, "coreg_img_"+str(i).zfill(4)+".nii.gz"))
        output_mat  = os.path.join(tmp_dir,"coreg_img_"+str(i).zfill(4)+".mat")

        print(moving_img.filename)
        print(tmp_out_img.filename)

        linreg(input                = moving_img,
               ref                  = ref,
               out_mat              = output_mat,
               out                  = tmp_out_img,
               dof                  = dof,
               nthreads             = nthreads,
               method               = method, 
               flirt_options        = flirt_options,
               ants_options         = ants_options,
               freesurfer_subjs_dir = freesurfer_subjs_dir,
               debug                = debug)
               
        fslmerge_cmd += " " + tmp_out_img.filename

    subprocess.run([fslmerge_cmd], shell=True, stderr=subprocess.STDOUT)
    #shutil.rmtree(tmp_dir)


if __name__ == '__main__':
   
   import argparse

   parser = argparse.ArgumentParser(description='QMRI-Neuropipe Linear Registration Tool')
   
   parser.add_argument("-i", "--input",
                    type=str,
                    help="Input image",
                    default=None)

   parser.add_argument("-r", "--ref",
                    type=str,
                    help="Reference image",
                    default=None)
   
   parser.add_argument("-o", "--out",
                       type=str, 
                       default=None)
   
   parser.add_argument("-m", "--method",
                       type=str,
                       help="Linear registration method",
                       choices=["fsl", "ants", "bbregister"],
                       default="fsl")   
   
   parser.add_argument("--dof",
                       type=int,
                       help="Degrees of Freedom",
                       default=6)   
   
   parser.add_argument("-n", "--nthreads",
                       type=int,
                       help="Number of threads (for multi-threaded applications)",
                       default=1)
   
   parser.add_argument("--flirt_options",
                       type=str,
                       help="Additinoal FSL Flirt options",
                       default=None)  
   
   parser.add_argument("--ants_options",
                       type=str,
                       help="Additinoal ANTs options",
                       default=None)
   
   parser.add_argument("--freesurfer_subjects_dir",
                       type=str,
                       help="FreeSurfer SUBJECTS_DIR Path",
                       default=None)           
   
   parser.add_argument("-d", "--debug",
                       type=bool,
                       help="Print debugging statements",
                       default=False)
   
   args, unknown = parser.parse_known_args()

   
   multilinreg(input                = Image(filename = args.input),
               ref                  = Image(filename = args.ref),
               out                  = Image(filename = args.out),
               dof                  = args.dof,
               nthreads             = args.nthreads,
               method               = args.method, 
               flirt_options        = args.flirt_options,
               ants_options         = args.ants_options,
               freesurfer_subjs_dir = args.freesurfer_subjects_dir,
               debug                = args.debug)