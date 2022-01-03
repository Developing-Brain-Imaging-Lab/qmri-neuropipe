import os,sys, shutil, json, argparse, copy
import nibabel as nib
import numpy as np

from dipy.io.image import load_nifti, save_nifti
from bids.layout import writing, parse_file_entities
from core.utils.io import Image, DWImage
import core.utils.tools as img_tools
import core.utils.mask as mask

def prepocess_mp2rage(bids_id, bids_rawdata_dir, bids_derivative_dir, brain_mask, reorient_img=None, cleanup_files=True, nthreads=1, verbose=False):

    #Setup raw data paths
    bids_mp2rage_rawdata_dir         = os.path.join(bids_rawdata_dir, 'mp2rage','')
    bids_mp2rage_derivative_dir      = os.path.join(bids_derivative_dir, 'mp2rage','')

    raw_t1w   = Image(file = bids_mp2rage_rawdata_dir + bids_id + '_T1w.nii.gz')
    raw_t1map = Image(file = bids_mp2rage_rawdata_dir + bids_id + '_T1map.nii.gz')
    raw_r1map = Image(file = bids_mp2rage_rawdata_dir + bids_id + '_R1map.nii.gz')
    raw_m0map = Image(file = bids_mp2rage_rawdata_dir + bids_id + '_M0map.nii.gz')

    t1w   = Image(file = bids_mp2rage_derivative_dir + bids_id + '_T1w.nii.gz')
    t1map = Image(file = bids_mp2rage_derivative_dir + bids_id + '_T1map.nii.gz')
    r1map = Image(file = bids_mp2rage_derivative_dir + bids_id + '_R1map.nii.gz')
    m0map = Image(file = bids_mp2rage_derivative_dir + bids_id + '_M0map.nii.gz')


    t1w = img_tools.reorient_to_standard(input_img     = raw_t1w,
                                         output_file    = t1w._get_filename(),
                                         reorient_img   = reorient_img)

    if not os.path.exists(t1map._get_filename()):
        t1map = img_tools.reorient_to_standard(input_img     = raw_t1map,
                                              output_file    = t1map._get_filename(),
                                              reorient_img   = reorient_img)

    if not os.path.exists(r1map._get_filename()):
        r1map = img_tools.reorient_to_standard(input_img      = raw_r1map,
                                               output_file    = r1map._get_filename(),
                                               reorient_img   = reorient_img)

    if not os.path.exists(m0map._get_filename()):
        m0map = img_tools.reorient_to_standard(input_img      = raw_m0map,
                                               output_file    = m0map._get_filename(),
                                               reorient_img   = reorient_img)

    if not os.path.exists(bids_mp2rage_derivative_dir + bids_id + '_desc-Masked_T1map.nii.gz'):
        if verbose:
            print('Applying Brain Mask to T1map')

        mask.apply_mask(input_img   = t1map,
                        mask_img    = brain_mask,
                        output_img  = bids_mp2rage_derivative_dir + bids_id + '_desc-Masked_T1map.nii.gz')

    if not os.path.exists(bids_mp2rage_derivative_dir + bids_id + '_desc-Masked_R1map.nii.gz'):
        if verbose:
            print('Applying Brain Mask to R1map')

        mask.apply_mask(input_img   = r1map,
                        mask_img    = brain_mask,
                        output_img  = bids_mp2rage_derivative_dir + bids_id + '_desc-Masked_R1map.nii.gz')

    if not os.path.exists(bids_mp2rage_derivative_dir + bids_id + '_desc-Masked_M0map.nii.gz'):
        if verbose:
            print('Applying Brain Mask to M0map')

        mask.apply_mask(input_img   = m0map,
                        mask_img    = brain_mask,
                        output_img  = bids_mp2rage_derivative_dir + bids_id + '_desc-Masked_M0map.nii.gz')


    if cleanup_files:
        os.remove(t1map._get_filename())
        os.remove(r1map._get_filename())
        os.remove(m0map._get_filename())
            

def prepocess_mpnrage(bids_id, bids_rawdata_dir, bids_derivative_dir, brain_mask, reorient_img=None, nthreads=1, verbose=False):

    #Setup raw data paths
    bids_mpnrage_rawdata_dir         = os.path.join(bids_rawdata_dir, 'mpnrage','')
    bids_mpnrage_derivative_dir      = os.path.join(bids_derivative_dir, 'mpnrage','')

    raw_t1w   = Image(file = bids_mpnrage_rawdata_dir + bids_id + '_T1w.nii.gz')
    raw_t1map = Image(file = bids_mpnrage_rawdata_dir + bids_id + '_T1map.nii.gz')

    t1w   = Image(file = bids_mpnrage_derivative_dir + bids_id + '_T1w.nii.gz')
    t1map = Image(file = bids_mpnrage_derivative_dir + bids_id + '_T1map.nii.gz')

    if not os.path.exists(t1map._get_filename()):
        t1map = img_tools.reorient_to_standard(input_img     = raw_t1map,
                                              output_file    = t1map._get_filename(),
                                              reorient_img   = reorient_img)

    if not os.path.exists(bids_mpnrage_derivative_dir + bids_id + '_desc-Masked_T1map.nii.gz'):
        if verbose:
            print('Applying Brain Mask to T1map')

        mask.apply_mask(input_img   = t1map,
                        mask_img    = brain_mask,
                        output_img  = bids_mpnrage_derivative_dir + bids_id + '_desc-Masked_T1map.nii.gz')
