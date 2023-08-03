#!/usr/bin/env python

import numpy as np
from scipy.io import loadmat

def rotate_bvecs(input_bvecs, output_bvecs, transform, linreg_method):

    #Rotate bvecs
    trans   = loadmat(transform)
    matrix  = ''
    if linreg_method == 'FSL':
        matrix = trans['MatrixOffsetTransformBase_double_3_3'][:9].reshape((3,3))
    elif linreg_method == 'ANTS':
        matrix = trans['AffineTransform_double_3_3'][:9].reshape((3,3))

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
   
   parser.add_argument('--method',
                       type=str,
                       help="Skull-stripping algorithm",
                       choices=["FSL", "ANTS"],
                       default="FSL")
   
   args, unknown = parser.parse_known_args()

   rotate_bvecs(input_bvecs   = args.input,
                output_bvecs  = args.output,
                transform     = args.transform,
                linreg_method = args.method)








