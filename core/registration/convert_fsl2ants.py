#!/usr/bin/env python
import os, subprocess

def convert_fsl2ants(input, ref, fsl_mat, ants_mat, debug=False):

    CMD = "c3d_affine_tool -ref " + ref.filename + " -src " +  input.filename + " " + fsl_mat + " -fsl2ras -oitk " + ants_mat

    if debug:
        print("Converting FSL style transform to ITK style")
        print(CMD)

    subprocess.run([CMD], shell=True, stderr=subprocess.STDOUT)
    

if __name__ == '__main__':
   
   import argparse

   parser = argparse.ArgumentParser(description='QMRI-Neuropipe Convert FSL Matrix to ITK Matrix')
   
                    
   parser.add_argument('--input',
                    type=str,
                    help="Input image",
                    default=None)
   
   parser.add_argument('--ref',
                    type=str,
                    help="Reference image",
                    default=None)
   
   parser.add_argument('--out',
                       type=str,
                       help="Output ITK style transform",
                       default=None)
   
   parser.add_argument('--fsl_mat',
                    type=str,
                    help="FSL style .mat ")
   
   parser.add_argument("--debug",
                       type=bool,
                       help="Print debugging statements",
                       default=False)
   
   args, unknown = parser.parse_known_args()
   
   convert_fsl2ants(input   = Image(filename=args.input),
                    ref     = Image(filename=args.ref),
                    fsl_mat = args.fsl_mat,
                    ants_mat= args.out,
                    debug   = args.debug)
