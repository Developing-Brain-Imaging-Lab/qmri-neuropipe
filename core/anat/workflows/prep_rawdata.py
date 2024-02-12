import os
from core.utils.io import Image

def prep_anat_rawdata(id, rawdata_dir, t1w_type='t1w', mpnrage_derivatives_dir='', verbose=True):
 
    #Setup Paths for T1w Images of different types
    if t1w_type == 'MP2RAGE' or t1w_type == 'mp2rage':
        t1w = Image(filename = os.path.join(rawdata_dir, "anat", id+'_inv-2_part-mag_MP2RAGE.nii.gz'),
                    json     = os.path.join(rawdata_dir, "anat", id+'_inv-2_MP2RAGE.json'))
        
    elif t1w_type == 'mpnrage' or t1w_type == 'MPnRAGE' or t1w_type == 'MPNRAGE':
        t1w = Image(filename = os.path.join(mpnrage_derivatives_dir, id+'_acq-MPnRAGE_rec-MoCo_T1w.nii.gz'),
                    json     = os.path.join(mpnrage_derivatives_dir, id+'_acq-MPnRAGE_rec-MoCo_T1w.json'))
        
    else:
        t1w = Image(filename = os.path.join(rawdata_dir, "anat", id+'_T1w.nii.gz'),
                    json     = os.path.join(rawdata_dir, "anat", id+'_T1w.json'))
    
    #Setup Paths for T2w Images
    t2w = Image(filename = os.path.join(rawdata_dir, "anat", id+'_T2w.nii.gz'),
                json     = os.path.join(rawdata_dir, "anat", id+'_T2w.json'))


    if not os.path.exists(t1w.filename):
        t1w = None
        if verbose:
            print("WARNING: No anatomical T1w image found")
    if not os.path.exists(t2w.filename):
        t2w = None
        if verbose:
            print("WARNING: No anatomical T2w image found")

    return t1w, t2w
