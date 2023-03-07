import os,sys, shutil, copy

from bids.layout import writing
from core.utils.io import Image, DWImage
import core.utils.tools as img_tools
import core.utils.mask as mask
import core.utils.biascorrect as bias_tools

import core.dmri.utils.qc as dmri_qc
import core.dmri.utils.distortion_correction as distcorr
import core.dmri.workflows.eddy_corr as eddy_proc
import core.dmri.workflows.distort_corr as distort_proc

import core.registration.registration as reg_tools
import core.segmentation.segmentation as seg_tools


def prep_anat_rawdata(bids_id, bids_rawdata_dir, bids_t1w_dir='anat', bids_t2w_dir='anat', t1w_type='t1w', verbose=True):

    #Setup raw data paths
    bids_t1w_rawdata_dir         = os.path.join(bids_rawdata_dir, bids_t1w_dir,'')
    bids_t2w_rawdata_dir         = os.path.join(bids_rawdata_dir, bids_t2w_dir,'')

    #Setup Paths for T1w Images of different types
    if t1w_type == 'mp2rage':
        raw_t1w = Image(file = bids_t1w_rawdata_dir + bids_id + '_inv-2_part-mag_MP2RAGE.nii.gz',
                        json = bids_t1w_rawdata_dir + bids_id + '_inv-2_MP2RAGE.json ')
    elif t1w_type == 'mpnrage':
        raw_t1w = Image(file = bids_t1w_rawdata_dir + bids_id + '_acq-MPnRAGE_rec-MoCo_T1w.nii.gz',
                        json = bids_t1w_rawdata_dir + bids_id + '_acq-MPnRAGE_T1w.json')
    else:
        raw_t1w = Image(file = bids_t1w_rawdata_dir + bids_id + '_T1w.nii.gz',
                        json = bids_t1w_rawdata_dir + bids_id + '_T1w.json')

    #Setup Paths for T2w Images
    raw_t2w = Image(file = bids_t2w_rawdata_dir + bids_id + '_T2w.nii.gz',
                    json = bids_t2w_rawdata_dir + bids_id + '_T2w.json')


    if not os.path.exists(raw_t1w._get_filename()):
        raw_t1w = None
        if verbose:
            print("WARNING: No anatomical T1w image found")
    if not os.path.exists(raw_t2w._get_filename()):
        raw_t2w = None
        if verbose:
            print("WARNING: No anatomical T2w image found")

    return raw_t1w, raw_t2w
