import os,sys, subprocess
from core.utils.io import Image
from core.segmentation.segmentation import create_wmseg

def acpc_align(output_dir, id, T1w=None, T2w=None, T1w_template=None, T2w_template=None, BrainSize="150", logfile=None):

    if logfile:
        sys.stdout = logfile
    
    #First adjust FOV
    T1w_robustroi       = Image(filename=os.path.join(output_dir, id + "_desc-robustfov_T1w.nii.gz"))
    T1w_roi2full_mat    = os.path.join(output_dir, id + "_desc-roi2full_T1w.mat")
    T1w_full2roi_mat    = os.path.join(output_dir, id + "_desc-full2roi_T1w.mat")
    T1w_sub2std_mat     = os.path.join(output_dir, id + "_desc-sub2std_T1w.mat")
    T1w_acpc_mat        = os.path.join(output_dir, id + "_desc-ACPC-Alignment_T1w.mat")
    T1w_acpc_aligned    = Image(filename=os.path.join(output_dir, id + "_desc-ACPC-Aligned_T1w.nii.gz"))
    
    T2w_robustroi       = Image(filename=os.path.join(output_dir, id + "_desc-robustfov_T2w.nii.gz"))
    T2w_roi2full_mat    = os.path.join(output_dir, id + "_desc-roi2full_T2w.mat")
    T2w_full2roi_mat    = os.path.join(output_dir, id + "_desc-full2roi_T2w.mat")
    T2w_sub2std_mat     = os.path.join(output_dir, id + "_desc-sub2std_T2w.mat")
    T2w_acpc_mat        = os.path.join(output_dir, id + "_desc-ACPC-Alignment_T2w.mat")
    T2w_acpc_aligned    = Image(filename=os.path.join(output_dir, id + "_desc-ACPC-Aligned_T2w.nii.gz"))
    
    if T1w is None:
        T1w_acpc_aligned = None
    if T2w is None:
        T2w_acpc_aligned = None
 
    if T1w and not os.path.exists(T1w_acpc_aligned.filename):
        print("##############################################")
        print("############ ACPC Alignment START ############")
        print()
        print("Input T1w: " + T1w.filename )
        print(flush=True)
    
        CMD = "robustfov -i " + T1w.filename \
                  + " -m " + T1w_roi2full_mat \
                  + " -r " + T1w_robustroi.filename \
                  + " -b " + str(BrainSize)
        subprocess.run([CMD], shell=True, stdout=logfile)
                  
        CMD = "convert_xfm -omat " + T1w_full2roi_mat \
                  + " -inverse " + T1w_roi2full_mat
        
        subprocess.run([CMD], shell=True, stdout=logfile)
                  
        #Register to the supplied template.
        if T1w_template is None:
            print("Error: No T1w ACPC-Alignment Supplied!", flush=True)
            exit()
        
        CMD = "flirt -ref " + T1w_template.filename \
                  + " -in " + T1w_robustroi.filename \
                  + " -omat " + T1w_sub2std_mat \
                  + " -interp spline -searchrx -180 180 -searchry -180 180 -searchrz -180 180"
        
        subprocess.run([CMD], shell=True, stdout=logfile)

        CMD = "convert_xfm -omat " + T1w_acpc_mat + " -concat " + T1w_sub2std_mat + " " + T1w_full2roi_mat
        subprocess.run([CMD], shell=True, stdout=logfile)
        
        CMD = "applywarp --rel --interp=spline -i " + T1w.filename \
                  + " -r " + T1w_template.filename \
                  + " --premat=" + T1w_acpc_mat \
                  + " -o " + T1w_acpc_aligned.filename
        subprocess.run([CMD], shell=True, stdout=logfile)
        
        
        print("Output T1w: " + T1w_acpc_aligned.filename )
        print("ACPC Successful")
        print("############ ACPC Alignment END ############")
        print("##############################################")
        print(flush=True)

    if T2w and not os.path.exists(T2w_acpc_aligned.filename):
        print("##############################################")
        print("############ ACPC Alignment START ############")
        print()
        print("Input T2w: " + T2w.filename )
        print(flush=True)
    
        CMD = "robustfov -i " + T2w.filename \
                  + " -m " + T2w_roi2full_mat \
                  + " -r " + T2w_robustroi.filename \
                  + " -b " + str(BrainSize)
        subprocess.run([CMD], shell=True, stdout=logfile)
                  
        CMD = "convert_xfm -omat " + T2w_full2roi_mat \
                  + " -inverse " + T2w_roi2full_mat
        subprocess.run([CMD], shell=True, stdout=logfile)
                  
    
        #Register to the supplied template.
        if T2w_template is None:
            print("Error: No T2w ACPC-Alignment Supplied!", flush=True)
            exit()
        
        CMD = "flirt -ref " + T2w_template.filename \
                  + " -in " + T2w_robustroi.filename \
                  + " -omat " + T2w_sub2std_mat \
                  + " -interp spline -searchrx -180 180 -searchry -180 180 -searchrz -180 180"
        subprocess.run([CMD], shell=True, stdout=logfile)

        CMD = "convert_xfm -omat " + T2w_acpc_mat + " -concat " + T2w_sub2std_mat + " " + T2w_full2roi_mat
        subprocess.run([CMD], shell=True, stdout=logfile)
        
        CMD = "applywarp --rel --interp=spline -i " + T2w.filename \
                  + " -r " + T2w_template.filename \
                  + " --premat=" + T2w_acpc_mat \
                  + " -o " + T2w_acpc_aligned.filename
        subprocess.run([CMD], shell=True, stdout=logfile)
        
        print("Output T2w: " + T2w_acpc_aligned.filename )
        print("ACPC Successful")
        print("############ ACPC Alignment END ############")
        print("##############################################")
        print(flush=True)

    if os.path.exists(T1w_robustroi.filename):
        os.remove(T1w_robustroi.filename)
    if os.path.exists(T1w_roi2full_mat):
        os.remove(T1w_roi2full_mat)
    if os.path.exists(T1w_full2roi_mat):
        os.remove(T1w_full2roi_mat)
    if os.path.exists(T1w_sub2std_mat):
        os.remove(T1w_sub2std_mat)
    if os.path.exists(T1w_acpc_mat):
        os.remove(T1w_acpc_mat)
        
    if os.path.exists(T2w_robustroi.filename):
        os.remove(T2w_robustroi.filename)
    if os.path.exists(T2w_roi2full_mat):
        os.remove(T2w_roi2full_mat)
    if os.path.exists(T2w_full2roi_mat):
        os.remove(T2w_full2roi_mat)
    if os.path.exists(T2w_sub2std_mat):
        os.remove(T2w_sub2std_mat)
    if os.path.exists(T2w_acpc_mat):
        os.remove(T2w_acpc_mat)
    
    return T1w_acpc_aligned, T2w_acpc_aligned


def coregister_images(output_dir, id, T1w, T2w, infant_mode=False, brain_size="150", logfile=None, nthreads=1):

    if logfile:
        sys.stdout = logfile
 
    RefImage    = T1w
    InImage     = T2w
    FullImage   = T2w,
    OutImage    = Image(filename = os.path.join(output_dir, id + "_space-individual-T1w_T2w.nii.gz"))
    T1w_output  = T1w
    T2w_output  = OutImage
    
    if infant_mode:
        RefImage    = T2w
        InImage     = T1w
        FullImage   = T1w
        OutImage    = Image(filename = os.path.join(output_dir, id + "_space-individual-T2w_T1w.nii.gz"))
        T1w_output  = OutImage
        T2w_output  = T2w

    print(FullImage) 
    FullMat     = os.path.join(output_dir, id + "_desc-T2w-2-T1w.mat")
    InitMat     = os.path.join(output_dir, id + "_desc-Coreg.mat")
    BBRMat      = os.path.join(output_dir, id + "_desc-BBR.mat")
    
    #Create FOV for RefImage
    RefImage_robustfov      = Image(filename = os.path.join(output_dir, "RefImage_robustfov.nii.gz"))
    RefImage_roi2full_mat   = os.path.join(output_dir, "RefImage_roi2full.mat")
    RefImage_full2roi_mat   = os.path.join(output_dir, "RefImage_full2roi.mat")
    CMD = "robustfov -i " + RefImage.filename \
          + " -m " + RefImage_roi2full_mat \
          + " -r " + RefImage_robustfov.filename \
          + " -b " + str(brain_size)
    subprocess.run([CMD], shell=True, stdout=logfile)
    CMD = "convert_xfm -omat " + RefImage_full2roi_mat \
              + " -inverse " + RefImage_roi2full_mat
    subprocess.run([CMD], shell=True, stdout=logfile)
    
    InImage_robustfov      = Image(filename = os.path.join(output_dir, "InImage_robustfov.nii.gz"))
    InImage_roi2full_mat   = os.path.join(output_dir, "InImage_roi2full.mat")
    InImage_full2roi_mat   = os.path.join(output_dir, "InImage_full2roi.mat")
    CMD = "robustfov -i " + InImage.filename \
          + " -m " + InImage_roi2full_mat \
          + " -r " + InImage_robustfov.filename \
          + " -b " + str(brain_size)
    subprocess.run([CMD], shell=True, stdout=logfile)
    CMD = "convert_xfm -omat " + InImage_full2roi_mat \
              + " -inverse " + InImage_roi2full_mat
    subprocess.run([CMD], shell=True, stdout=logfile)

              
    CMD = "N4BiasFieldCorrection -d 3 -i " + InImage_robustfov.filename + " -o " + InImage_robustfov.filename
    subprocess.run([CMD], shell=True, stdout=logfile)
    
    CMD = "N4BiasFieldCorrection -d 3 -i " + RefImage_robustfov.filename + " -o " + RefImage_robustfov.filename
    subprocess.run([CMD], shell=True, stdout=logfile)
              
    #Run with FLIRT – can constrain because of the initial ACPC alignment.
    CMD = "flirt -in " + InImage_robustfov.filename \
              + " -ref " + RefImage_robustfov.filename \
              + " -omat " + InitMat \
              + " -init $FSLDIR/etc/flirtsch/ident.mat" \
              + " -dof 6 -interp spline" \
              + " -cost normmi"\
              + " -searchrx -10 10 -searchry -10 10 -searchrz -10 10 -finesearch 2 -coarsesearch 5"
    subprocess.run([CMD], shell=True, stdout=logfile)

    #Create WMseg for BBR
    # WMsegImg = create_wmseg(input_img  = RefImage_robustfov, 
    #                         output_dir = output_dir+"/wmseg/", 
    #                         nthreads   = nthreads)
    
    # CMD = "flirt -in " + InImage_robustfov.filename \
    #         + " -ref " + RefImage_robustfov.filename\
    #         + " -out " + output_dir + "/test.nii.gz" \
    #         + " -omat " + BBRMat \
    #         + " -init " + InitMat \
    #         + " -wmseg " + WMsegImg.filename \
    #         + " -dof 6 -interp spline" \
    #         + " -searchrx -5 5 -searchry -5 5  -searchrz -5 5" \
    #         + " -cost bbr" \
    #         + " -bbrtype global_abs -bbrslope 0.5 -finesearch 10" \
    #         + " -schedule ${FSLDIR}/etc/flirtsch/bbr.sch"
    # subprocess.run([CMD], shell=True, stdout=logfile)

    CMD = "convert_xfm -omat " + FullMat + " -concat " + InitMat + " " + InImage_full2roi_mat
    subprocess.run([CMD], shell=True, stdout=logfile)
    
    CMD = "convert_xfm -omat " + FullMat + " -concat " + RefImage_roi2full_mat + " " + FullMat
    subprocess.run([CMD], shell=True, stdout=logfile)

    CMD = "applywarp --rel --interp=spline -i " + FullImage.filename \
              + " -r " + RefImage.filename \
              + " --premat=" + FullMat \
              + " -o " + OutImage.filename
    subprocess.run([CMD], shell=True, stdout=logfile)

    
    #Clean up files
    if os.path.exists(InImage_robustfov.filename):
        os.remove(InImage_robustfov.filename)
    if os.path.exists(InImage_roi2full_mat):
        os.remove(InImage_roi2full_mat)
    if os.path.exists(InImage_full2roi_mat):
        os.remove(InImage_full2roi_mat)

    if os.path.exists(RefImage_robustfov.filename):
        os.remove(RefImage_robustfov.filename)
    if os.path.exists(RefImage_roi2full_mat):
        os.remove(RefImage_roi2full_mat)
    if os.path.exists(RefImage_full2roi_mat):
        os.remove(RefImage_full2roi_mat)
        
    if os.path.exists(InitMat):
        os.remove(InitMat)
    if os.path.exists(FullMat):
        os.remove(FullMat)


    
    return T1w_output, T2w_output
