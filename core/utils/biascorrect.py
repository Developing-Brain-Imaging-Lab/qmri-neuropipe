#!/usr/bin/env python
import os, shutil, copy, subprocess
import core.utils.mask as mask

def biasfield_correction(input_img, output_file, method="ants", mask_img=None, nthreads=1, iterations=1, debug=False):

    output_img = copy.deepcopy(input_img)
    output_img.filename = output_file

    CMD=""
    if method=="mrtrix-ants" or method=="mrtrix-fsl":
        if method == "mrtrix-fsl":
            method = "fsl"
        else:
            method = "ants"
        
        CMD = "dwibiascorrect " + method + " " \
                + input_img.filename + " " \
                + output_img.filename \
                + " -force -quiet -nthreads " + str(nthreads)

        if mask_img != None:
            CMD += " -mask " + mask_img.filename

        if input_img.bvals != None and input_img.bvecs != None:
            CMD += " -fslgrad " + input_img.bvecs + " " + input_img.bvals

    elif method=="ants":
        os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = str(nthreads)

        dim = 3
        if input_img.get_type() == "DWImage":
            dim = 4

        CMD = "N4BiasFieldCorrection -d " + str(dim) + " " \
                + "-i " + input_img.filename + " " \
                + "-o " + output_img.filename

        if mask_img != None:
            CMD += " -x " + mask_img.filename
        
        print(CMD)

    elif method == "fsl":
        tmp_dir = os.path.join(os.path.dirname(output_img.filename), "tmp_fast")

        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)

        CMD = "fast -B -o " + tmp_dir + "/fast" + " " + input_img.filename

    else:
        print('Invalid Biasfield correction Method')
        print('Available options are: mrtrix-ants, mrtrix-fsl, ants, fsl')
        exit()

    if debug:
        print("Biasfield correction")
        print(CMD)

    subprocess.run([CMD], shell=True, stderr=subprocess.STDOUT)

    if method == "fsl":
        os.rename(tmp_dir+"/fast_restore.nii.gz", output_img.filename)
        shutil.rmtree(tmp_dir)

    if iterations > 1:

        for i in range(0, iterations -1):

            CMD=""
            if method=="mrtrix-ants" or method=="mrtrix-fsl":
                if method == "mrtrix-fsl":
                    method = "fsl"
                else:
                    method = "ants"
                
                CMD = "dwibiascorrect " + method + " " \
                        + output_img.filename + " " \
                        + output_img.filename \
                        + " -force -quiet -nthreads " + str(nthreads)

                if mask_img != None:
                    CMD += " -mask " + mask_img.filename

                if input_img.bvals != None and input_img.bvecs != None:
                    CMD += " -fslgrad " + input_img.bvecs + " " + input_img.bvals

            elif method=="ants":
                os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = str(nthreads)

                CMD = "N4BiasFieldCorrection -d 3 " \
                        + "-i " + output_img.filename + " " \
                        + "-o " + output_img.filename

                if mask_img != None:
                    CMD += " -x " + mask_img.filename

            elif method == "fsl":
                tmp_dir = os.path.join(os.path.dirname(output_img.filename), "tmp_fast")

                if not os.path.exists(tmp_dir):
                    os.makedirs(tmp_dir)

                CMD = "fast -B -o " + tmp_dir + "/fast" + " " + output_img.filename

                os.rename(tmp_dir+"/fast_restore.nii.gz", output_img.filename)

                shutil.rmtree(tmp_dir)

            else:
                print('Invalid Biasfield correction Method')
                print('Available options are: mrtrix-ants, mrtrix-fsl, ants, fsl')
                exit()

            if debug:
                print("Biasfield correction: Iteration " + str(iteration+1))
                print(CMD)

            subprocess.run([CMD], shell=True, stderr=subprocess.STDOUT)
            if method == "fsl":
                os.rename(tmp_dir+"/fast_restore.nii.gz", output_img.filename)
                shutil.rmtree(tmp_dir)


    return output_img


if __name__ == '__main__':
   
   import argparse
   from core.utils.io import Image, DWImage
   
   parser = argparse.ArgumentParser(description='QMRI-Neuropipe Bias-field correction function')
   
   parser.add_argument('--input',
                       type=str,
                       help="Input image",
                       default=None)
   
   parser.add_argument('--output',
                       type=str,
                       help="Biasfield corrected image",
                       default=None)
   
   parser.add_argument('--bvals',
                       type=str,
                       help="B-values of DWI input",
                       default=None)
   
   parser.add_argument('--bvecs',
                       type=str,
                       help="B-bvectors of DWI input",
                       default=None)
   
   parser.add_argument('--mask',
                       type=str,
                       help="Binary mask",
                       default=None)
   
   parser.add_argument('--method',
                       type=str,
                       help="Biasfield correction algorithm",
                       choices=["mrtrix-ants", "mrtrix-fsl", "ants", "fsl"],
                       default="ants")

   parser.add_argument("--iterations",
                       type=int,
                       help="Number of iterations",
                       default=1)

   parser.add_argument("--nthreads",
                       type=int,
                       help="Number of threads",
                       default=1)
   
   parser.add_argument("--debug",
                       type=bool,
                       help="Print debugging statements",
                       default=False)
   
   args, unknown = parser.parse_known_args()

   if args.bvals and args.bvecs:
       input_img = DWImage(filename = args.input,
                           bvals    = args.bvals,
                           bvecs    = args.bvecs)
   else:
       input_img = Image(filename = args.input)
       
   biasfield_correction(input_img      = input_img,
                        output_file    = args.output,
                        method         = args.method,
                        mask_img       = Image(filename=args.mask), 
                        nthreads       = args.nthreads,
                        iterations     = args.iterations,
                        debug          = args.debug)
