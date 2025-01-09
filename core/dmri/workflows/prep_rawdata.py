import os, shutil, json
from bids import BIDSLayout
from bids.layout import writing

from core.utils.io import DWImage
import core.utils.tools as img_tools

import core.dmri.utils.qc as dmri_qc
import core.dmri.workflows.eddy_corr as eddy_proc
from core.dmri.workflows.dmri_reorient import dmri_reorient

def prep_rawdata(bids_dir, preproc_dir, 
                 id, 
                 session=None, 
                 bids_filter=None,
                 check_gradients=False, 
                 reorient_dwi=False, 
                 dwi_reorient_template=None, 
                 resample_resolution=None, 
                 remove_last_vol=False, 
                 outlier_detection=None, 
                 nthreads=1,
                 verbose=False):

    #Setup raw data paths
    layout      = BIDSLayout(bids_dir, validate=False)
    bids_id     = writing.build_path({'subject': id, 'session': session}, "sub-{subject}[_ses-{session}]")
    proc_dir    = os.path.join(preproc_dir, "rawdata/")
    run_topup   = False

    if not os.path.exists(proc_dir):
        os.makedirs(proc_dir)

    #Get the subject's diffusion data
    subj_data = layout.get(subject=id, session=session, datatype='dwi', suffix='dwi', extension='nii.gz', return_type='filename')
    num_dwis  = len(subj_data)

    dwi_img = DWImage(filename  = f"{proc_dir}/{bids_id}_dwi.nii.gz",
                      bvals     = f"{proc_dir}/{bids_id}_dwi.bval",
                      bvecs     = f"{proc_dir}/{bids_id}_dwi.bvec",
                      index     = f"{proc_dir}/{bids_id}_desc-Index_dwi.txt",
                      acqparams = f"{proc_dir}/{bids_id}_desc-Acqparams_dwi.txt",
                      json      = f"{proc_dir}/{bids_id}_dwi.json")

    if num_dwis == 1:
        img     = layout.get(subject=id, session=session, datatype='dwi', suffix='dwi', extension='nii.gz', return_type='filename')[0]
        bvals   = layout.get(subject=id, session=session, datatype='dwi', suffix='dwi', extension='bval', return_type='filename')[0]
        bvecs   = layout.get(subject=id, session=session, datatype='dwi', suffix='dwi', extension='bvec', return_type='filename')[0]
        sidecar = layout.get(subject=id, session=session, datatype='dwi', suffix='dwi', extension='json', return_type='filename')[0]
        
        shutil.copy2(img, dwi_img.filename)
        shutil.copy2(bvals, dwi_img.bvecs)
        shutil.copy2(bvecs, dwi_img.bvals)
        shutil.copy2(sidecar, dwi_img.json)

    else:
        
        if bids_filter is None:
            print("Please provide a BIDS filter to select the appropriate DWI data")
            exit(-1)

        imgs_to_merge = []
        dwi_filter = json.load(open(bids_filter, 'r'))['dwi']
        
        if "rpe_direction" in dwi_filter:
            dwi_dirs = dwi_filter['rpe_direction']

            for rpe_dir in dwi_dirs:
                
                img     = layout.get(subject=id, session=session, datatype='dwi', direction=rpe_dir, suffix='dwi', extension='nii.gz', return_type='filename')[0]
                bvals   = layout.get(subject=id, session=session, datatype='dwi', direction=rpe_dir, suffix='dwi', extension='bval', return_type='filename')[0]
                bvecs   = layout.get(subject=id, session=session, datatype='dwi', direction=rpe_dir, suffix='dwi', extension='bvec', return_type='filename')[0]
                sidecar = layout.get(subject=id, session=session, datatype='dwi', direction=rpe_dir, suffix='dwi', extension='json', return_type='filename')[0]

                imgs_to_merge.append(DWImage(filename=img, bvals=bvals, bvecs=bvecs, json=sidecar))

        elif "description" in dwi_filter:          
            dwi_desc = dwi_filter['description']

            img     = layout.get(subject=id, session=session, datatype='dwi', suffix='dwi', extension='nii.gz', return_type='filename')
            bvals   = layout.get(subject=id, session=session, datatype='dwi', suffix='dwi', extension='bval', return_type='filename')
            bvecs   = layout.get(subject=id, session=session, datatype='dwi', suffix='dwi', extension='bvec', return_type='filename')
            sidecar = layout.get(subject=id, session=session, datatype='dwi', suffix='dwi', extension='json', return_type='filename')
                
            for img_desc in dwi_desc:
                for i in range(len(img)):
                    if img_desc in img[i]:
                        imgs_to_merge.append(DWImage(filename=img[i], bvals=bvals[i], bvecs=bvecs[i], json=sidecar[i]))
                        break
        else:
            print("Please provide a valid BIDS filter")
            exit(-1)      

        
        dwi_img = dmri_qc.merge_phase_encodes(DWI_pepolar0 = imgs_to_merge[0], 
                                              DWI_pepolar1 = imgs_to_merge[1], 
                                              output_base  = f"{proc_dir}/{bids_id}")
        if len(imgs_to_merge) > 2:
            for i in range(2, len(imgs_to_merge)):
                dwi_img = dmri_qc.merge_phase_encodes(DWI_pepolar0 = dwi_img, 
                                                      DWI_pepolar1 = imgs_to_merge[i], 
                                                      output_base  = f"{proc_dir}/{bids_id}")
        run_topup  = True

    

    #Ensure ISOTROPIC voxels prior to processing
    if verbose:
        print('Ensuring DWIs have isotropic voxels')

    dwi_img = img_tools.check_isotropic_voxels(input_img          = dwi_img,
                                               output_file        = dwi_img.filename,
                                               target_resolution  = resample_resolution,
                                               debug              = verbose)

    #Remove Last DWI volume before processing further
    if remove_last_vol:
        if verbose:
            print('Removing Last DWI in volume')

        dwi_img = img_tools.remove_end_img(input_img   = dwi_img,
                                           output_file = dwi_img.filename)

    #Check the Image Sizes to Ensure Proper Length:
    if verbose:
        print('Checking DWI Acquisition Size and Gradient Orientations')

    dmri_qc.check_bvals_bvecs(input_dwi   = dwi_img,
                              output_base = f"{proc_dir}/{bids_id}")

    if check_gradients:
        dmri_qc.check_gradient_directions(input_dwi   = dwi_img,
                                          nthreads    = nthreads)

    dwi_img.index     = f"{proc_dir}/{bids_id}_desc-Index_dwi.txt"
    dwi_img.acqparams = f"{proc_dir}/{bids_id}_desc-Acqparams_dwi.txt"
    
    if not os.path.exists(dwi_img.index) or not os.path.exists(dwi_img.acqparams):
        dwi_img.index, dwi_img.acqparams = dmri_qc.create_index_acqparam_files(input_dwi   = dwi_img,
                                                                               output_base = f"{proc_dir}/{bids_id}")

    dwi_img.slspec = f"{proc_dir}/{bids_id}_desc-Slspec_dwi.txt"
    if not os.path.exists( dwi_img.slspec ):
        dwi_img.slspec = dmri_qc.create_slspec_file(input_dwi        = dwi_img,
                                                    output_base      = f"{proc_dir}/{bids_id}")

    if outlier_detection == 'Manual':
        outlier_detection_dir = os.path.join(preproc_dir, 'outlier-removed-images/')

        if verbose:
            print('Removing DWIs from manual selection')

        dwi_img = eddy_proc.perform_outlier_detection(dwi_image         = dwi_img,
                                                      working_dir       = outlier_detection_dir,
                                                      method            = outlier_detection,
                                                      manual_report_dir = f"{bids_dir}/{bids_id}/dwi",
                                                      verbose           = verbose )
        
    if reorient_dwi:
        dmri_reorient(in_dwi  = dwi_img,
                      out_dwi = dwi_img,
                      ref_img = dwi_reorient_template)

  
        
    
    return dwi_img
