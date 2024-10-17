import os, shutil
from bids import BIDSLayout
from bids.layout import writing

from core.utils.io import Image
import core.utils.tools as img_tools

def prep_anat_rawdata(bids_dir, preproc_dir,
                      id,
                      session=None,
                      t1w_type='t1w',
                      resample_resolution=None, 
                      mpnrage_derivatives_dir='', 
                      verbose=True):

    #Setup raw data paths
    layout      = BIDSLayout(bids_dir, validate=False)
    bids_id     = writing.build_path({'subject': id, 'session': session}, "sub-{subject}[_ses-{session}]")
    proc_dir    = os.path.join(preproc_dir, "rawdata/")

    if not os.path.exists(proc_dir):
        os.makedirs(proc_dir)

    #See if there are any T1w images
    subj_data = layout.get(subject=id, session=session, datatype='anat', suffix='T1w', extension='nii.gz', return_type='filename')
    num_t1w  = len(subj_data)

    if num_t1w == 1:
        shutil.copy2(subj_data[0], os.path.join(proc_dir, os.path.basename(subj_data[0])))
        shutil.copy2(subj_data[0].replace('.nii.gz', '.json'), os.path.join(proc_dir, os.path.basename(subj_data[0].replace('.nii.gz', '.json'))))

        t1w = Image(filename = os.path.join(proc_dir, os.path.basename(subj_data[0])),
                    json     = os.path.join(proc_dir, os.path.basename(subj_data[0].replace('.nii.gz', '.json'))))

    elif t1w_type.lower() == 'mpnrage':
        if mpnrage_derivatives_dir == '':
            raise ValueError("If using MPnRAGE images, please provide the derivatives directory")

        t1w = Image(filename = os.path.join(mpnrage_derivatives_dir, id+'_acq-MPnRAGE_rec-MoCo_T1w.nii.gz'),
                    json     = os.path.join(mpnrage_derivatives_dir, id+'_acq-MPnRAGE_rec-MoCo_T1w.json'))

    elif t1w_type.lower() == 'mp2rage':
        t1w = Image(filename = os.path.join(rawdata_dir, "anat", id+'_inv-2_part-mag_MP2RAGE.nii.gz'),
                    json     = os.path.join(rawdata_dir, "anat", id+'_inv-2_MP2RAGE.json'))
    else:   
        t1w = None
        if verbose:
            print("WARNING: No anatomical T1w image found")


    subj_data = layout.get(subject=id, session=session, datatype='anat', suffix='T2w', extension='nii.gz', return_type='filename')
    num_t2w  = len(subj_data)

    if num_t2w >= 1:
        shutil.copy2(subj_data[0], os.path.join(proc_dir, os.path.basename(subj_data[0])))
        shutil.copy2(subj_data[0].replace('.nii.gz', '.json'), os.path.join(proc_dir, os.path.basename(subj_data[0].replace('.nii.gz', '.json'))))

        t2w = Image(filename = os.path.join(proc_dir, os.path.basename(subj_data[0])),
                    json     = os.path.join(proc_dir, os.path.basename(subj_data[0].replace('.nii.gz', '.json'))))
    else:
        t2w = None
        if verbose:
            print("WARNING: No anatomical T2w image found")

    

    
    if resample_resolution:
        if t1w:
            t1w = img_tools.check_isotropic_voxels(input_img          = t1w,
                                                   output_file        = t1w.filename,
                                                   target_resolution  = resample_resolution,
                                                   debug              = verbose)
        
        if t2w:
            print(resample_resolution)
            t2w = img_tools.check_isotropic_voxels(input_img          = t2w,
                                                   output_file        = t2w.filename,
                                                   target_resolution  = resample_resolution,
                                                   debug              = verbose)
        


    return t1w, t2w
