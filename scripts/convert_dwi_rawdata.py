#!/usr/bin/env python
import os,sys, shutil, time, argparse, json

from core.utils.io import Image, DWImage
import core.dicom.dicom_data as rawdata
import core.dmri.utils.qc as dmri_qc

parser = argparse.ArgumentParser(description='Convert DIFFUSION DICOM Data to NIFTI Data')
parser.add_argument('--subject', type=str, help='Subject ID')
parser.add_argument('--session', type=str, help='Subject Timepoint', default=None)
parser.add_argument('--bids_dir', type=str, help='BIDS Directory')
parser.add_argument('--nthreads', type=int, help='Number of Threads', default=1)
parser.add_argument('--nifti_convert_method', type=str, help='Method to use for NIFTI conversion', default="dcm2niix", choices=["dcm2niix", "dcm2nii", "mrtrix", "mri-convert"])

parser.add_argument('--dwi_dcm_dir', type=str, help='Path to DICOM DWI directory')
parser.add_argument('--dwi_bvals', type=str, help='Path to DWI B-values', default=None)
parser.add_argument('--dwi_bvecs', type=str, help='Path to DWI B-vectors', default=None)

parser.add_argument('--dwi_pepolar1_dcm_dir', type=str, help='Path to Reversed Phase Encode DWI DICOM directory', default=None)
parser.add_argument('--dwi_pepolar1_bvals', type=str, help='Path to Reverse Phase Encode B-values', default=None)
parser.add_argument('--dwi_pepolar1_bvecs', type=str, help='Path to Reverse Phase Encode B-vectors', default=None)

parser.add_argument('--dwi_fieldmap_dcm_dir', type=str, help='Path to DICOM data for B0 Fieldmap', default=None)
parser.add_argument('--dwi_fieldmap_ref_dcm_dir', type=str, help='Path to DICOM data for B0 Fieldmap reference image', default=None)

args = parser.parse_args()

bids_id=""
bids_sub_dir=""

if args.session:
	bids_id                = 'sub-' + args.subject + '_ses-' + args.session.zfill(2)
	bids_sub_dir           = 'sub-' + args.subject + '/ses-' + args.session.zfill(2)
else:
	bids_id                = 'sub-' + args.subject
	bids_sub_dir           = 'sub-' + args.subject

bids_rawdata_dir       = args.bids_dir + '/rawdata/' + bids_sub_dir + '/'


##################################
##################################
##### PROCESSING STARTS HERE #####
##################################
##################################

bids_dwi_dir    = bids_rawdata_dir + 'dwi/'

if not os.path.exists(bids_dwi_dir):
    os.makedirs(bids_dwi_dir)

if args.dwi_pepolar1_dcm_dir != None:
    bids_dwi        = bids_dwi_dir + bids_id + '_desc-pepolar-0_dwi.nii.gz'
    bids_dwi_json   = bids_dwi_dir + bids_id + '_desc-pepolar-0_dwi.json'
    bids_dwi_bvals  = bids_dwi_dir + bids_id + '_desc-pepolar-0_dwi.bval'
    bids_dwi_bvecs  = bids_dwi_dir + bids_id + '_desc-pepolar-0_dwi.bvec'
else:
    bids_dwi        = bids_dwi_dir + bids_id + '_dwi.nii.gz'
    bids_dwi_json   = bids_dwi_dir + bids_id + '_dwi.json'
    bids_dwi_bvals  = bids_dwi_dir + bids_id + '_dwi.bval'
    bids_dwi_bvecs  = bids_dwi_dir + bids_id + '_dwi.bvec'

print('Converting DWI Images...')
rawdata.dicom_to_nifti(dcm_dir     = args.dwi_dcm_dir,
                       method      = args.nifti_convert_method,
                       output_img  = bids_dwi)
if args.dwi_bvals != None:
    shutil.copy2(args.dwi_bvals, bids_dwi_bvals)
if args.dwi_bvecs != None:
    shutil.copy2(args.dwi_bvecs, bids_dwi_bvecs)

#Check to make sure the Numebr of Bvals/Bvecs match acquired data
dmri_qc.check_bvals_bvecs(DWImage(filename=bids_dwi, bvals=bids_dwi_bvals, bvecs=bids_dwi_bvecs))

###CONVERT THE REVERSE PHASE ENCODE DATA ###
if args.dwi_pepolar1_dcm_dir != None:
    bids_dwi        = bids_dwi_dir + bids_id + '_desc-pepolar-1_dwi.nii.gz'
    bids_dwi_json   = bids_dwi_dir + bids_id + '_desc-pepolar-1_dwi.json'
    bids_dwi_bvals  = bids_dwi_dir + bids_id + '_desc-pepolar-1_dwi.bval'
    bids_dwi_bvecs  = bids_dwi_dir + bids_id + '_desc-pepolar-1_dwi.bvec'

    print('Converting Reversed-Phase Encode DWI Images...')
    rawdata.dicom_to_nifti_dcm2niix(dcm_dir     = args.dwi_pepolar1_dcm_dir,
                                    output_img  = bids_dwi)

    if args.dwi_pepolar1_bvals != None:
        shutil.copy2(args.dwi_pepolar1_bvals, bids_dwi_bvals)
    if args.dwi_pepolar1_bvecs != None:
        shutil.copy2(args.dwi_pepolar1_bvecs, bids_dwi_bvecs)

    #Check to make sure the Numebr of Bvals/Bvecs match acquired data
    dmri_qc.check_bvals_bvecs(DWImage(file = bids_dwi, bvals=bids_dwi_bvals, bvecs=bids_dwi_bvecs))


###CONVERT FMAP DATA IF AVAILABLE
if args.dwi_fieldmap_dcm_dir != None:

    bids_fmap_dir    = bids_rawdata_dir + 'fmap-dwi/'

    if not os.path.exists(bids_fmap_dir):
        os.makedirs(bids_fmap_dir)

    bids_fmap       = bids_fmap_dir + bids_id + '_fieldmap.nii.gz'
    bids_fmap_ref   = bids_fmap_dir + bids_id + '_magnitude.nii.gz'

    print('Converting DWI Fieldmap Images...')
    rawdata.dicom_to_nifti_dcm2niix(dcm_dir     = args.dwi_fieldmap_dcm_dir,
                                    output_img  = bids_fmap)

    rawdata.dicom_to_nifti_dcm2niix(dcm_dir     = args.dwi_fieldmap_ref_dcm_dir,
                                    output_img  = bids_fmap_ref)

    os.remove(bids_fmap_dir + bids_id + '_magnitude.json')
