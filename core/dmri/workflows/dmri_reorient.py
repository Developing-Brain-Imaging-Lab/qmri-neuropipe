import os, shutil
from core.utils.io import Image, DWImage

import core.dmri.utils.eddy_correction as eddycorr
import core.dmri.tools as dmri_tools
from core.dmri.utils.rotate_fsl_bvecs import rotate_fsl_bvecs 

from core.registration.linreg import linreg


def dmri_reorient( in_dwi, out_dwi, ref_img):

    #First, run eddy-correct to ensure DWIs
    output_dir  = os.path.dirname(out_dwi.filename)
    output_base = output_dir + "/tmp_dwi_eddy"
    
    eddycorrected_img = eddycorr.eddy_correct_fsl(input_dwi   = in_dwi,
                                                  output_base = output_base)
    
    dwi_ref = Image(filename=output_dir+"/meanB0.nii.gz")
    dwi_ref = dmri_tools.extract_b0s(input_dwi    = eddycorrected_img, 
                                     output_b0    = dwi_ref, 
                                     compute_mean = True)

    #Coregister the mean to the ref
    out_mat = output_dir+"/tmp_dwi2ref.mat"
    linreg(input    = dwi_ref, 
           ref      = ref_img, 
           out_mat  = out_mat,
           flirt_options = "-dof 6 -searchrx -180 180 -searchry -180 180 -searchrz -180 180")

    #Apply transform
    os.system("applywarp -i " + eddycorrected_img.filename + " -r " + ref_img.filename + " -o " + out_dwi.filename + " --premat="+out_mat)

    #Rotate bvecs
    rotate_fsl_bvecs(eddycorrected_img.bvecs, out_dwi.bvecs, out_mat)
    
    
    
    #os.remove(output_dir+"/mean.nii.gz")
    #os.system(output_dir+"/tmp_dwi_eddy*")
    #os.remove(output_dir+"/tmp_dwi2ref.mat")
    #os.remove(output_dir+"/tmp_dwi_hdr.nii.gz")
    

    
     
 
