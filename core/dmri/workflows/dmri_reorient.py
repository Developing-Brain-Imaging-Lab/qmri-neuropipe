import os, shutil
from core.utils.io import Image, DWImage

import core.dmri.utils.eddy_correction as eddycorr
import core.utils.tools as img_tools
from core.dmri.utils.rotate_fsl_bvecs import rotate_fsl_bvecs 

from core.registration.linreg import linreg


def dmri_reorient( in_dwi, out_dwi, ref_img):

    #First, run eddy-correct to ensure DWIs
    output_dir  = os.path.dirname(out_dwi.filename)
    output_base = output_dir + "/tmp_dwi_eddy"
    
    eddycorrected_img = eddycorr.eddy_correct_fsl(input_dwi   = in_dwi,
                                                  output_base = output_base)
    
    dwi_mean = Image(filename=output_dir+"/mean.nii.gz")                                         
    mean_img =  img_tools.calculate_mean_img(eddycorrected_img, output_file=dwi_mean.filename) 

    #Coregister the mean to the ref
    out_mat = output_dir+"/tmp_dwi2ref.mat"
    linreg(input    = dwi_mean, 
           ref      = ref_img, 
           out_mat  = out_mat,
           flirt_options = "-dof 6 -searchrx -180 180 -searchry -180 180 -searchrz -180 180")
    
    mean_corr_header = Image(filename=output_dir+"/tmp_dwi_hdr.nii.gz")
    shutil.copy2(mean_img.filename, mean_corr_header.filename)
    os.system("fslcpgeom " + ref_img.filename + " " + mean_corr_header.filename + " -d")

    #Apply transform
    os.system("applywarp -i " + eddycorrected_img.filename + " -r " + mean_corr_header.filename + " -o " + out_dwi.filename + " --premat="+out_mat)

    #Rotate bvecs
    rotate_fsl_bvecs(eddycorrected_img.bvecs, out_dwi.bvecs, out_mat)
    
    
    
    os.remove(output_dir+"/mean.nii.gz")
    os.system(output_dir+"/tmp_dwi_eddy*")
    os.remove(output_dir+"/tmp_dwi2ref.mat")
    os.remove(output_dir+"/tmp_dwi_hdr.nii.gz")
    

    
     
 
