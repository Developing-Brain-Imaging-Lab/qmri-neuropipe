#!/usr/bin/env python

import numpy as np

def rotate_fsl_bvecs(input_bvecs, output_bvecs, transform):

    print(input_bvecs)
    print(output_bvecs)

    #Rotate bvecs
    trans   = np.loadtxt(transform)
    matrix = trans[:9].reshape((4,4))
    
    bvecs = np.genfromtxt(input_bvecs)
    if bvecs.shape[0] != 3:
        bvecs = bvecs.T

    newbvecs = np.dot(matrix, bvecs)
    np.savetxt(output_bvecs, newbvecs, fmt='%.5f')


if __name__ == '__main__':
   
   import argparse
   parser = argparse.ArgumentParser(description='QMRI-Neuropipe Rotate b-vectors')
   
   parser.add_argument('--input',
                    type=str,
                    help="Input image to be skull-stripped",
                    default=None)
   
   parser.add_argument('--output',
                       type=str,
                       help="Output binary mask",
                       default=None)
   
   parser.add_argument('--transform',
                       type=str,
                       help="Output masked data",
                       default=None)
   
   
   args, unknown = parser.parse_known_args()


   rotate_fsl_bvecs(input_bvecs   = args.input,
                output_bvecs  = args.output,
                transform     = args.transform)








