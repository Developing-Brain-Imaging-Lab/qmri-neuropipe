import string, os, sys, subprocess, shutil, time

#Neuroimaging Modules
import nibabel as nib
import numpy as np
from dipy.io.image import load_nifti, save_nifti
from dipy.core.gradients import gradient_table
from dipy.io import read_bvals_bvecs
import dipy.reconst.dki as dki
import scipy.ndimage.filters as filters

class DKI_Model():
    def __init__(self, dwi_img, out_base, fit_type='dipy-WLS', mask=None, include_micro_fit=False, nthreads=1):
        self._inputs = {}
        self._inputs['dwi_img']     = dwi_img
        self._inputs['out_base']    = out_base
        self._inputs['fit_type']    = fit_type
        self._inputs['mask']        = mask
        self._inputs['micro']       = include_micro_fit
        self._inputs['nthreads']    = nthreads

        self._outputs = {}
        self._outputs['fa']               = out_base + '_model-DKI_parameter-FA.nii.gz'
        self._outputs['md']               = out_base + '_model-DKI_parameter-MD.nii.gz'
        self._outputs['rd']               = out_base + '_model-DKI_parameter-RD.nii.gz'
        self._outputs['ad']               = out_base + '_model-DKI_parameter-AD.nii.gz'
        self._outputs['mk']               = out_base + '_model-DKI_parameter-MK.nii.gz'
        self._outputs['rk']               = out_base + '_model-DKI_parameter-RK.nii.gz'
        self._outputs['ak']               = out_base + '_model-DKI_parameter-AK.nii.gz'
        self._outputs['mkt']              = out_base + '_model-DKI_parameter-MKT.nii.gz'
        self._outputs['kfa']              = out_base + '_model-DKI_parameter-KFA.nii.gz'
        self._outputs['awf']              = out_base + '_model-DKI_parameter-AWF.nii.gz'
        self._outputs['tort']             = out_base + '_model-DKI_parameter-TORTUSITY.nii.gz'


    def fit(self):

        dwi_img = self._inputs['dwi_img']
        output_dir = os.path.dirname(self._inputs['out_base'])

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        img = nib.load(dwi_img._get_filename())
        data = img.get_fdata()
        bvals, bvecs = read_bvals_bvecs(dwi_img._get_bvals(), dwi_img._get_bvecs())
        gtab = gradient_table(bvals, bvecs)

        if self._inputs['mask'] != None:
            mask_data = nib.load(self._inputs['mask']._get_filename()).get_fdata()

        values = np.array(bvals)
        ii = np.where(values == bvals.min())[0]
        b0_average = np.mean(data[:,:,:,ii], axis=3)

        #Recommended to smooth data prior to fitting:
        fwhm = 2.00
        gauss_std = fwhm / np.sqrt(8 * np.log(2))  # converting fwhm to Gaussian std
        data_smooth = np.zeros(data.shape)
        for v in range(data.shape[-1]):
            data_smooth[..., v] = filters.gaussian_filter(data[..., v], sigma=gauss_std)

        fit_type = self._inputs['fit_type'].split('-')[1]
        dkimodel = dki.DiffusionKurtosisModel(gtab, fit_type)

        if self._inputs['mask'] != None:
            dkifit = dkimodel.fit(data_smooth, mask_data)
        else:
            dkifit = dkimodel.fit(data_smooth)

        save_nifti(self._outputs['fa'], dkifit.fa.astype(np.float32), img.affine, img.header)
        save_nifti(self._outputs['md'], dkifit.md.astype(np.float32), img.affine, img.header)
        save_nifti(self._outputs['rd'], dkifit.rd.astype(np.float32), img.affine, img.header)
        save_nifti(self._outputs['ad'], dkifit.ad.astype(np.float32), img.affine, img.header)
        save_nifti(self._outputs['mk'], dkifit.mk(0,3).astype(np.float32), img.affine, img.header)
        save_nifti(self._outputs['rk'], dkifit.rk(0,3).astype(np.float32), img.affine, img.header)
        save_nifti(self._outputs['ak'], dkifit.ak(0,3).astype(np.float32), img.affine, img.header)
        save_nifti(self._outputs['mkt'], dkifit.mkt(0,3).astype(np.float32), img.affine, img.header)
        save_nifti(self._outputs['kfa'], dkifit.kfa.astype(np.float32), img.affine, img.header)

        if self._inputs['micro']:

            import dipy.reconst.dki_micro as dki_micro
            well_aligned_mask = np.ones(data.shape[:-1], dtype='bool')

            # Diffusion coefficient of linearity (cl) has to be larger than 0.4, thus
            # we exclude voxels with cl < 0.4.
            cl = dkifit.linearity.copy()
            well_aligned_mask[cl < 0.2] = False

            # Diffusion coefficient of planarity (cp) has to be lower than 0.2, thus
            # we exclude voxels with cp > 0.2.
            cp = dkifit.planarity.copy()
            well_aligned_mask[cp > 0.2] = False

            # Diffusion coefficient of sphericity (cs) has to be lower than 0.35, thus
            # we exclude voxels with cs > 0.35.
            cs = dkifit.sphericity.copy()
            well_aligned_mask[cs > 0.2] = False

            # Removing nan associated with background voxels
            well_aligned_mask[np.isnan(cl)] = False
            well_aligned_mask[np.isnan(cp)] = False
            well_aligned_mask[np.isnan(cs)] = False

            dki_micro_model = dki_micro.KurtosisMicrostructureModel(gtab, fit_type)
            dki_micro_fit = dki_micro_model.fit(data_smooth, mask=well_aligned_mask)

            save_nifti(self._outputs['awf'], dki_micro_fit.awf.astype(np.float32), img.affine, img.header)
            save_nifti(self._outputs['tort'], dki_micro_fit.tortuosity.astype(np.float32), img.affine, img.header)
