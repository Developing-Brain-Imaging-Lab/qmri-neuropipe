import os,sys, shutil, json, argparse, copy
import nibabel as nib
import numpy as np

from core.utils.io import Image, DWImage
from core.dmri.models.dti import DTI_Model



def fit_dti(subject, dwi_img, bvals, bvecs, out_dir, mask=None, fit_method='dipy-WLS', grad_nonlin=None):
      
    DWI = DWImage(file        = dwi_img,
                  bvecs       = bvecs,
                  bvals       = bvals)
                      
    MASK = Image(file = mask)


    dti_model = DTI_Model(dwi_img               = DWI,
                          sub_info              = {subject},
                          out_dir               = out_dir,
                          mask                  = MASK,
                          fit_type              = fit_method,
                          verbose               = args.verbose)
    dti_model.fit()
        
if __name__ == '__main__':

        parser = argparse.ArgumentParser()
       
        parser.add_argument('--subject',
                            type=str,
                            help='Subject ID')

        parser.add_argument('--dwi_img',
                            type=str,
                            help='Subject ID')
        
        parser.add_argument('--bvals',
                            type=str,
                            help='Subject ID')
        
        parser.add_argument('--bvecs',
                            type=str,
                            help='Subject ID')

        parser.add_argument('--mask',
                            type=str,
                            help='Subject ID')
        
        parser.add_argument('--out_dir',
                            type=str,
                            help='Subject ID')
        
        parser.add_argument('--nthreads',
                            type=int,
                            help='Number of Threads',
                            default=1)

        parser.add_argument('--dti_fit_method',
                            type=str,
                            help='Fitting Algorithm for Neurite Orietation Dispersion and Density Imaging Model',
                            default='dipy-WLS')
    
        parser.add_argument('--gradnonlin',
                            type=str,
                            help='Fitting Algorithm for Neurite Orietation Dispersion and Density Imaging Model')

        parser.add_argument('--verbose',
                            type=bool,
                            help='Print out information meassages and progress status',
                            default=False)
        
        parser.add_argument('--debug',
                    type=bool,
                    help='Print out debugging messages',
                    default=False)
        
        
        args, unknown = parser.parse_known_args()
        
        
        if not os.path.exists(args.out_dir):
            os.makedirs(args.out_dir)

        
        
        fit_dti(subject = args.subject,
                dwi_img = args.dwi_img,
                bvals   = args.bvals,
                bvecs   = args.bvecs,
                out_dir = args.out_dir,
                mask    = args.mask,
                fit_method=args.dti_fit_method, 
                grad_nonlin=args.gradnonlin )

            
        
       

        
   
