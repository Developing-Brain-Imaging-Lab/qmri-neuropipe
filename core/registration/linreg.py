#!/usr/bin/env python
import os, subprocess

from core.utils.io import Image
from .convert_fsl2ants import convert_fsl2ants
from .apply_transform import apply_transform
from bids.layout import writing, parse_file_entities

def linreg(input, ref, out_mat, out=None, dof=6, nthreads=1, method="fsl", flirt_options=None, ants_options=None, freesurfer_subjs_dir=None, debug=False):

    CMD=""

    if method == 'fsl':

        if type(input) is list:
            in_img = input[0]
        else:
            in_img = input
        if type(ref) is list:
            ref_img = ref[0]
        else:
            ref_img = ref

        if out != None:
            if type(out) is list:
                out_img = out[0]
            else:
                out_img = out

        CMD = "flirt -in " + in_img.filename \
            + " -ref " +  ref_img.filename \
            + " -omat " + out_mat \
            + " -dof " + str(dof)

        if out != None:
            CMD += " -out " + out_img.filename
        if flirt_options != None:
            CMD += " " + flirt_options

        if debug:
            print("Running FSL FLIRT")
            print(CMD)
        
        subprocess.run([CMD], shell=True, stderr=subprocess.STDOUT)

        if out != None and type(input) is list:
            #Apply transform to other images
            for i in range(0, len(input)):
                apply_transform(input         = input[i], 
                                ref           = ref_img, 
                                out           = out[i], 
                                transform     = out_mat, 
                                method        = "fsl",
                                flirt_options = flirt_options)
                
        return SUCCESS


    elif method == 'ants':
        os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = str(nthreads)

        out_dir         = os.path.dirname(out_mat)
        ants_tmp_out    = os.path.join(out_dir, "tmp_ants_") 

        CMD = "antsRegistrationSyN.sh -d 3 -o " + ants_tmp_out + " -n " + str(nthreads)

        if dof == 6:
            CMD += " -t r"
        else:
            CMD += " -t a"

        if type(input) is list:
            for i in range(0,len(input)):
                CMD += ' -m ' + input[i].filename
                
                if len(ref) > 1:
                    CMD +=  ' -f ' + ref[i].filename
                else:
                    CMD +=  ' -f ' + ref[0].filename
        else:
                CMD += ' -m ' + input.filename \
                    +  ' -f ' + ref.filename

        if ants_options != None:
            CMD += ' ' + ants_options
        
        if debug:
            print("Running ANTs Linear Registration")
            print(CMD)
        
        subprocess.check_call([CMD], shell=True, stderr=subprocess.STDOUT)

        #Change filenames
        os.rename(ants_tmp_out+"0GenericAffine.mat", out_mat)
        os.rename(ants_tmp_out+"Warped.nii.gz", out[0].filename)

        for i in range(1,len(out)):
            apply_transform(input         = input[i], 
                            ref           = ref[0], 
                            out           = out[i], 
                            transform     = out_mat, 
                            method        = "ants",
                            ants_options  = ants_options)
            

        #Clean up remaining ants files
        #os.system("rm -rf " + ants_tmp_out+"*")
        
    elif method == 'bbr':
    
        if type(input) is list:
            input = input[0]
            ref   = ref[0]

        parsed_filename = parse_file_entities(input.filename)
        entities = {
        'subject': parsed_filename.get('subject'),
        'session': parsed_filename.get('session'),
        }
        subid_patterns   = 'sub-{subject}[_ses-{session}]'
        subid = writing.build_path(entities, subid_patterns)
    
        os.environ["SUBJECTS_DIR"] = freesurfer_subjs_dir
        output_dir = os.path.dirname(out_mat)
        
        freesurfer_tmp_dir = os.path.join(output_dir, '/tmp/')
        if not os.path.exists(freesurfer_tmp_dir):
            os.makedirs(freesurfer_tmp_dir)

        ## run bbregister and output transform in fsl format
        b0toT1mat      = os.path.join(output_dir, "b0toT1.mat")
        b0toT1lta      = os.path.join(output_dir, "b0toT1.lta")
        b0toT1flirtmtx = os.path.join(output_dir, "b0toT1flirt.mtx")
        
        CMD = "bbregister --s " + subid + " --mov " + input.filename \
            + " --reg " + b0toT1mat \
            + ' --dti --init-fsl --lta ' + b0toT1lta \
            + ' --fslmat ' + b0toT1flirtmtx \
            + ' --tmp ' + freesurfer_tmp_dir

        subprocess.check_call([CMD], shell=True, stderr=subprocess.STDOUT)
        convert_fsl2ants(input, ref, b0toT1flirtmtx, out_mat)


        
        
if __name__ == '__main__':
   
   import argparse

   parser = argparse.ArgumentParser(description='QMRI-Neuropipe Linear Registration Tool')
   
   parser.add_argument("-i", "--in",
                    type=str, nargs='+',
                    help="Input image",
                    default=None)

   parser.add_argument("-r", "--ref",
                    type=str, nargs='+',
                    help="Reference image",
                    default=None)
   
   parser.add_argument("-x", "--out_xfm",
                       type=str,
                       help="Output matrix transformation",
                       default=None)

   parser.add_argument("-o", "--out",
                       type=str, nargs='+',
                       help="Output image",
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

   input_imgs = []
   ref_imgs   = []
   out_imgs   = []

   for img in args.i:
       print(img)
       input_imgs.append(Image(filename=img))

   for img in args.ref:
       ref_imgs.append(Image(filename=img))

   for img in args.out:
       out_imgs.append(Image(filename=img))

   
   linreg(input                = input_imgs,
          ref                  = ref_imgs,
          out_mat              = args.out_xfm,
          out                  = out_imgs,
          dof                  = args.dof,
          nthreads             = args.nthreads,
          method               = args.method, 
          flirt_options        = args.flirt_options,
          ants_options         = args.ants_options,
          freesurfer_subjs_dir = args.freesurfer_subjects_dir,
          debug                = args.debug)
   
   
