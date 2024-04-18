import os, sys, shutil
import numpy as np
import nibabel as nib
from scipy.ndimage import gaussian_filter

def compute_afi_b1map(input_img1, input_img2, output_b1map, theta=60, n=5, smooth_inputs=False, fwhm=6):

    #Load the images
    img1 = nib.load(input_img1)
    img2 = nib.load(input_img2)

    TR1 = img1.get_fdata()
    TR2 = img2.get_fdata()
    
    if smooth_inputs:
            TR1 = gaussian_filter(img1.get_fdata(), fwhm/2.35)
            TR2 = gaussian_filter(img2.get_fdata(), fwhm/2.35)

    r = TR2 / TR1
    r[r>1] = 1.00

    n=float(n)
    theta=float(theta)

    arg = (r*n-1.0)/(n-r)
    arg[arg>1]=1
    arg[arg<0]=1

    b1 = np.degrees(np.arccos(arg))/theta

    b1map = nib.Nifti1Image(b1.astype(np.float32), img1.affine, img1.header)
    b1map.set_sform(img1.get_sform())
    b1map.set_qform(img1.get_qform())
    nib.save(b1map, output_b1map)


if __name__ == "__main__":
    
    import argparse
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--afi_img1',
                                type=str,
                                help='Subject ID')
    
    parser.add_argument('--afi_img2',
                                type=str,
                                help='Number of Threads')
    
    parser.add_argument('--out_b1map',
                                type=str,
                                help='Number of Threads')
    
    parser.add_argument('--theta',
                        type=int,
                        help='Number of Threads',
                        default=60)
    
    parser.add_argument('--trratio',
                        type=int,
                        help='Number of Threads',
                        default=5)
    
    parser.add_argument('--smooth',
                        type=bool,
                        help='Number of Threads',
                        default=False)
    
    parser.add_argument('--fwhm',
                        type=int,
                        help='Number of Threads',
                        default=6)
    
    args, unknown = parser.parse_known_args()
    
    compute_afi_b1map(input_img1    = args.afi_img1,
                      input_img2    = args.afi_img2,
                      output_b1map  = args.out_b1map,
                      theta         = args.theta,
                      n             = args.trratio,
                      smooth_inputs = args.smooth, 
                      fwhm          = args.fwhm)