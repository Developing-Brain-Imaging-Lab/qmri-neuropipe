import os,sys, shutil, json, argparse, copy
import nibabel as nib
import numpy as np

from core.utils.io import Image, DWImage
from core.dmri.models.noddi import NODDI_Model

class DiffusionProcessingPipeline:

    def __init__(self, verbose=False):
        if verbose:
            print('Creating Diffusion Processing Pipeline')

    def run(self):

        parser = argparse.ArgumentParser()

        parser.add_argument('--subject',
                            type=str,
                            help='Subject ID')

        parser.add_argument('--nthreads',
                            type=int,
                            help='Number of Threads',
                            default=1)

        parser.add_argument('--noddi_fit_method',
                            type=str,
                            help='Fitting Algorithm for Neurite Orietation Dispersion and Density Imaging Model',
                            choices=['amico', 'noddi-watson', 'noddi-bingham'],
                            default='noddi-watson')

        parser.add_argument('--noddi_dpar',
                            type=float,
                            help='Parallel diffusivity value to use in the NODDI model fitting',
                            default=1.7e-9)

        parser.add_argument('--noddi_diso',
                            type=float,
                            help='Isotropic diffusivity value to use in the NODDI model fitting',
                            default=3e-9)

        parser.add_argument('--noddi_solver',
                            type=str,
                            help='DMIPY Optimization solver for NODDI model',
                            choices=['brute2fine', 'mix'],
                            default='brute2fine')

        parser.add_argument('--verbose',
                            type=bool,
                            help='Print out information meassages and progress status',
                            default=False)
        
        parser.add_argument('--debug',
                    type=bool,
                    help='Print out debugging messages',
                    default=False)
        
        
        args, unknown = parser.parse_known_args()
        
        
        
        study_root  = "/scratch/sjshort_ebds/BEE-2wk_NODDI_31May2023"
        subject_dir = study_root + "/BEE-"+args.subject+"-2wk_NODDI/"
        
        dwi_img     = subject_dir + "BEE-"+args.subject+"-2wk_NODDI_DWI.nii.gz"
        dwi_bval    = subject_dir + "BEE-"+args.subject+"-2wk_NODDI_protocol.bval"
        dwi_bvec    = subject_dir + "BEE-"+args.subject+"-2wk_NODDI_protocol.bvec"
        mask_img    = subject_dir + "BEE-"+args.subject+"-2wk_brain_mask.nii.gz"
        
        
        noddi_output_dir = subject_dir + "/NODDI_WATSON/"
        
        
        
        if not os.path.exists(noddi_output_dir):
            os.makedirs(noddi_output_dir)
            
        
        DWI = DWImage(file        = dwi_img,
                      bvecs       = dwi_bvec,
                      bvals       = dwi_bval)
                      
        MASK = Image(file = mask_img)
        

        noddi_model = NODDI_Model(dwi_img               = DWI,
                                  out_base              = noddi_output_dir+"BEE-"+args.subject,
                                  fit_type              = args.noddi_fit_method,
                                  mask                  = MASK,
                                  parallel_diffusivity  = args.noddi_dpar,
                                  iso_diffusivity       = args.noddi_diso,
                                  solver                = args.noddi_solver,
                                  nthreads              = args.nthreads,
                                  verbose               = args.verbose)
        noddi_model.fit()
        
        
     
if __name__ == "__main__":
    noddi_fit = DiffusionProcessingPipeline()
    noddi_fit.run()
        

        
        
        
        
   
