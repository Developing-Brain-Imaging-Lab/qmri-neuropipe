import string, os, sys, subprocess, shutil, time

#Neuroimaging Modules
import nibabel as nib
import numpy as np
from dipy.io.image import load_nifti, save_nifti
from dipy.core.gradients import gradient_table
from dipy.io import read_bvals_bvecs
import dipy.reconst.dki as dki
import scipy.ndimage.filters as filters
from bids.layout import writing

class DKI_Model():
    def __init__(self, dwi_img, sub_info, out_dir, fit_type='dipy-WLS', mask=None, include_micro_fit=False, fwhm=2, nthreads=1):
        self._inputs = {}
        self._inputs['dwi_img']     = dwi_img
        self._inputs['out_dir']     = out_dir
        self._inputs['fit_type']    = fit_type
        self._inputs['mask']        = mask
        self._inputs['micro']       = include_micro_fit
        self._inputs['fwhm']        = fwhm  
        self._inputs['nthreads']    = nthreads
             
        map_entities = {}
        map_entities['subject'] = sub_info['subject']
        map_entities['session'] = sub_info['session']
        map_entities['model']   = "DKI"
        
        map_pattern = os.path.join(out_dir, "sub-{subject}[_ses-{session}][_model-{model}]_param-{map}.nii.gz")         
        map_entities['map']               = "FA"
        self._outputs['fa']               = writing.build_path(map_entities, map_pattern)
        map_entities['map']               = "MD"
        self._outputs['md']               = writing.build_path(map_entities, map_pattern)
        map_entities['map']               = "RD"
        self._outputs['rd']               = writing.build_path(map_entities, map_pattern)
        map_entities['map']               = "AD"
        self._outputs['ad']               = writing.build_path(map_entities, map_pattern)
        
        map_entities['map']               = "MK"
        self._outputs['mk']               = writing.build_path(map_entities, map_pattern)
        map_entities['map']               = "RL"
        self._outputs['rk']               = writing.build_path(map_entities, map_pattern)
        map_entities['map']               = "AK"
        self._outputs['ak']               = writing.build_path(map_entities, map_pattern)
        map_entities['map']               = "MKT"
        self._outputs['mkt']              = writing.build_path(map_entities, map_pattern)
        map_entities['map']               = "kFA"
        self._outputs['kfa']              = writing.build_path(map_entities, map_pattern)
        map_entities['map']               = "AWF"
        self._outputs['awf']              = writing.build_path(map_entities, map_pattern)
        map_entities['map']               = "TORTUSITY"
        self._outputs['tort']              = writing.build_path(map_entities, map_pattern)
        
    def fit(self):

        dwi_img = self._inputs['dwi_img']
        output_dir = self._inputs['out_dir']

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        img          = nib.load(dwi_img.filename)
        data         = img.get_fdata()
        bvals, bvecs = read_bvals_bvecs(dwi_img.bvals, dwi_img.bvecs)
        gtab         = gradient_table(bvals, bvecs)

        if self._inputs['mask'] != None:
            mask_data = nib.load(self._inputs['mask'].filename).get_fdata()

        values = np.array(bvals)
        ii = np.where(values == bvals.min())[0]
        b0_average = np.mean(data[:,:,:,ii], axis=3)

        #Recommended to smooth data prior to fitting:
        gauss_std = self._inputs['fwhm'] / np.sqrt(8 * np.log(2))  # converting fwhm to Gaussian std
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
