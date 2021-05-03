import os,sys, shutil, json, argparse, copy
import nibabel as nib
import numpy as np

from dipy.io.image import load_nifti, save_nifti
from bids.layout import writing, parse_file_entities
from core.utils.io import Image, DWImage


def compute_t1w_t2w_ratio(t1w, t2w, mask, template):

    #First, coregister template to
    


t1w_path = '/study/dean_k99/Studies/infmri/processed-data/derivatives/qmri-prep/sub-10021/ses-02/anat/sub-10021_ses-02_T1w.nii.gz'
t2w_path = '/study/dean_k99/Studies/infmri/processed-data/derivatives/qmri-prep/sub-10021/ses-02/anat/sub-10021_ses-02_space-individual-T1w_T2w.nii.gz'

bias_t1_path = '/study/dean_k99/Studies/infmri/processed-data/derivatives/qmri-prep/sub-10021/ses-02/anat/sub-10021_ses-02_desc-BiasFieldCorrected_T1w.nii.gz'
bias_t2_path = '/study/dean_k99/Studies/infmri/processed-data/derivatives/qmri-prep/sub-10021/ses-02/anat/sub-10021_ses-02_desc-BiasFieldCorrected_T2w.nii.gz'

mask = '/study/dean_k99/Studies/infmri/processed-data/derivatives/qmri-prep/sub-10021/ses-02/anat/sub-10021_ses-02_desc-brain_mask.nii.gz'

tm_label = '/study/dean_k99/Studies/infmri/processed-data/derivatives/qmri-prep/sub-10021/ses-02/anat/test/tm_label.nii.gz'
eye_label = '/study/dean_k99/Studies/infmri/processed-data/derivatives/qmri-prep/sub-10021/ses-02/anat/test/eye_label.nii.gz'

t1w = ants.image_read(t1w_path)
t2w = ants.image_read(t2w_path)

tm = ants.image_read(tm_label).numpy()
eye = ants.image_read(eye_label).numpy()

T1_TM_prod = np.multiply(t1w.numpy(), tm)
T1_EY_prod = np.multiply(t1w.numpy(), eye)

T2_TM_prod = np.multiply(t2w.numpy(), tm)
T2_EY_prod = np.multiply(t2w.numpy(), eye)

Xs_t1 = np.mean(T1_TM_prod[T1_TM_prod!=0])
Ys_t1 = np.mean(T1_EY_prod[T1_EY_prod!=0])

Xs_t2 = np.mean(T2_TM_prod[T2_TM_prod!=0])
Ys_t2 = np.mean(T2_EY_prod[T2_EY_prod!=0])



#X = TM; Y = EYE
Xr_t1 = 2.415277
Yr_t1 = 0.894427

Xr_t2 = 1.263179
Yr_t2 = 12.090323


T1w_bias = ants.image_read(bias_t1_path).numpy()
T2w_bias = ants.image_read(bias_t2_path).numpy()


slope_t1     = (Xr_t1-Yr_t1)/(Xs_t1 - Ys_t1)
intercept_t1 = (Xs_t1*Yr_t1 - Xr_t1*Ys_t1)/(Xs_t1 - Ys_t1)

slope_t2     = (Xr_t2-Yr_t2)/(Xs_t2 - Ys_t2)
intercept_t2 = (Xs_t2*Yr_t2 - Xr_t2*Ys_t2)/(Xs_t2 - Ys_t2)

T1w_corr = ants.from_numpy( (slope_t1*T1w_bias + intercept_t1),
                            spacing = t1w.spacing,
                            origin  = t1w.origin,
                            direction = t1w.direction)

T2w_corr = ants.from_numpy( (slope_t2*T2w_bias + intercept_t2),
                            spacing = t1w.spacing,
                            origin  = t1w.origin,
                            direction = t1w.direction)
Ratio_corr = ants.from_numpy( T1w_corr.numpy() / T2w_corr.numpy(),
                            spacing = t1w.spacing,
                            origin  = t1w.origin,
                            direction = t1w.direction)

T1w_corr_path = '/study/dean_k99/Studies/infmri/processed-data/derivatives/qmri-prep/sub-10021/ses-02/anat/test/T1w_corr.nii.gz'
T2w_corr_path = '/study/dean_k99/Studies/infmri/processed-data/derivatives/qmri-prep/sub-10021/ses-02/anat/test/T2w_corr.nii.gz'
Ratio_corr_path = '/study/dean_k99/Studies/infmri/processed-data/derivatives/qmri-prep/sub-10021/ses-02/anat/test/T1wT2wRatio_corr.nii.gz'
ants.image_write(T1w_corr, T1w_corr_path)
ants.image_write(T2w_corr, T2w_corr_path)
ants.image_write(Ratio_corr, Ratio_corr_path)
