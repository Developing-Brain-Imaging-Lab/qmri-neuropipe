#!/usr/bin/env python
import os, subprocess, copy

import numpy as np
import nibabel as nib

from dipy.denoise.gibbs import gibbs_removal
from core.utils.io import Image, DWImage

def gibbs_ringing_correction(input_img, output_file, method='mrtrix', nthreads=0, debug=False):

    output_img          = copy.deepcopy(input_img)
    output_img.filename = output_file 

    if method=='mrtrix':

        CMD="mrdegibbs " + input_img.filename + " " + output_img.filename + " -nthreads " + str(nthreads) + " -quiet -force"

        if debug:
            print("Gibbs Ringing Correction with MRtrix")
            print(CMD)

        subprocess.run([CMD], shell=True, stderr=subprocess.STDOUT)
                        

    if method=='dipy':
        img = nib.load(input_img.filename)
        data = img.get_fdata()
        data_corrected = gibbs_removal(data, num_processes=nthreads)

        corrected_img = nib.Nifti1Image(data_corrected.astype(np.float32), img.affine, img.header)
        corrected_img.set_sform(img.get_sform())
        corrected_img.set_qform(img.get_qform())
        nib.save(corrected_img, output_img.filename)
        
    #After the gibbs ringing correction, copy the header from the input to ensure proper sizing
    os.system('fslcpgeom ' + input_img.filename + ' ' + output_img.filename )

    return output_img


if __name__ == '__main__':
   
   import argparse
   
   parser = argparse.ArgumentParser(description='QMRI-Neuropipe Gibbs Ringing function')
   
   parser.add_argument('--input',
                    type=str,
                    help="Input image to be apply gibbs ringing correction",
                    default=None)
   
   parser.add_argument('--output',
                       type=str,
                       help="Output image",
                       default=None)
   
   parser.add_argument('--method',
                       type=str,
                       help="Algorithm to use for Gibbs Ringing correction",
                       choices=["mrtrix", "dipy"],
                       default="mrtrix")
   
   parser.add_argument("--nthreads",
                       type=int,
                       help="Number of threads",
                       default=1)
   
   parser.add_argument("--debug",
                       type=bool,
                       help="Print debugging statements",
                       default=False)
   
   args, unknown = parser.parse_known_args()
   gibbs_ringing_correction(input_img   = Image(args.input),
                            output_file = args.output,
                            method      = args.method, 
                            nthreads    = args.nthreads, 
                            debug       = args.debug)