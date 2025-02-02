import string, os, sys, subprocess, shutil, time

#Neuroimaging Modules
import nibabel as nib
import numpy as np
import scipy.ndimage.filters as filters
from bids.layout import writing

from dipy.io.image import save_nifti
from dipy.core.gradients import gradient_table
from dipy.io import read_bvals_bvecs

import dipy.reconst.dti as dti
import dipy.reconst.dki as dki

from core.dmri.utils.correct_bvals_bvecs import correct_bvals_bvecs

class DKI_Model():
    def __init__(self, dwi_img, sub_info, out_dir, fit_type='dipy-WLS', mask=None, grad_nonlin=None, include_micro_fit=False, smooth_data=True, fwhm=2, nthreads=1):
        self._inputs = {}
        self._outputs = {}

        self._inputs['dwi_img']     = dwi_img
        self._inputs['out_dir']     = out_dir
        self._inputs['fit_type']    = fit_type
        self._inputs['mask']        = mask
        self._inputs['micro']       = include_micro_fit
        self._inputs['smooth_data'] = smooth_data
        self._inputs['fwhm']        = fwhm  
        self._inputs['nthreads']    = nthreads
        self._inputs['grad_nonlin'] = grad_nonlin
             
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
        map_entities['map']               = "RK"
        self._outputs['rk']               = writing.build_path(map_entities, map_pattern)
        map_entities['map']               = "AK"
        self._outputs['ak']               = writing.build_path(map_entities, map_pattern)
        map_entities['map']               = "MKT"
        self._outputs['mkt']              = writing.build_path(map_entities, map_pattern)
        map_entities['map']               = "RTK"
        self._outputs['rtk']              = writing.build_path(map_entities, map_pattern)
        map_entities['map']               = "kFA"
        self._outputs['kfa']              = writing.build_path(map_entities, map_pattern)
        map_entities['map']               = "AWF"
        self._outputs['awf']              = writing.build_path(map_entities, map_pattern)
        map_entities['map']               = "TORTUSITY"
        self._outputs['tort']              = writing.build_path(map_entities, map_pattern)
        
    def fit(self):

        dwi_img = self._inputs['dwi_img']
        output_dir = self._inputs['out_dir']

        npa = 27

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

        data_to_fit = data

        #Recommended to smooth data prior to fitting:
        if self._inputs['smooth_data']:
            gauss_std = self._inputs['fwhm'] / np.sqrt(8 * np.log(2))  # converting fwhm to Gaussian std
            data_to_fit = np.zeros(data.shape)
            for v in range(data.shape[-1]):
                data_to_fit[..., v] = filters.gaussian_filter(data[..., v], sigma=gauss_std)

        #Loop over all voxels
        img_shape   = data_to_fit.shape[:-1]
        flat_data   = data_to_fit.reshape(-1, data_to_fit.shape[-1])
        flat_params = np.empty((flat_data.shape[0], npa))
        flat_mask   = mask_data.reshape(-1)
        gtab        = gradient_table(bvals, bvecs, atol=0.1)

        grad_nonlin_data = None
        if self._inputs['grad_nonlin'] != None:
            grad_nonlin_data = nib.load(self._inputs['grad_nonlin'].filename).get_fdata().reshape(flat_data.shape[0], 9)

        for vox in range(flat_data.shape[0]):
            if flat_mask[vox] > 0:
                if self._inputs['grad_nonlin'] != None:
                    corr_bvals, corr_bvecs = correct_bvals_bvecs(bvals, bvecs, grad_nonlin_data[vox])
                    gtab = gradient_table(corr_bvals, corr_bvecs, atol=0.1)

                fit_type = self._inputs['fit_type'].split('-')[1]
                dkimodel = dki.DiffusionKurtosisModel(gtab, fit_type)
                dkifit   = dkimodel.fit(flat_data[vox])
                flat_params[vox] = dkifit.model_params

                # voxel_E  = flat_data[vox]
                # fit_args = (voxel_E)
                # 

        #     if use_parallel_processing:
        #         fitted_parameters_lin[vox] = pool.apipe(dkimodel.fit, flat_data[vox])
        #         #print('fitted parameters: {}'.format(fitted_parameters_lin[idx]))
        #     else:
        #         fitted_parameters_lin[idx] = dkimodel.fit(flat_data[vox])
        # if use_parallel_processing:
        #     fitted_parameters_lin = np.array(
        #         [p.get() for p in fitted_parameters_lin])
        #     pool.close()
        #     pool.join()
        #     pool.clear()

        #Reshape the parameters
        params = flat_params.reshape((img_shape + (npa,)))
        evals = params[..., :3]
        evecs = params[..., 3:12].reshape(params.shape[:-1] + (3, 3))

        fa = dti.fractional_anisotropy(evals)
        md = dti.mean_diffusivity(evals)
        rd = dti.radial_diffusivity(evals)
        ad = dti.axial_diffusivity(evals)

        mk = dki.mean_kurtosis(params)
        rk = dki.radial_kurtosis(params)
        ak = dki.axial_kurtosis(params)
        kfa = dki.kurtosis_fractional_anisotropy(params)

        mkt = dki.mean_kurtosis_tensor(params)
        #rtk = dki.radial_tensor_kurtosis(params)
  
        save_nifti(self._outputs['fa'], fa.astype(np.float32), img.affine, img.header)
        save_nifti(self._outputs['md'], md.astype(np.float32), img.affine, img.header)
        save_nifti(self._outputs['rd'], rd.astype(np.float32), img.affine, img.header)
        save_nifti(self._outputs['ad'], ad.astype(np.float32), img.affine, img.header)
        save_nifti(self._outputs['mk'], mk.astype(np.float32), img.affine, img.header)
        save_nifti(self._outputs['rk'], rk.astype(np.float32), img.affine, img.header)
        save_nifti(self._outputs['ak'], ak.astype(np.float32), img.affine, img.header)
        save_nifti(self._outputs['mkt'],mkt.astype(np.float32), img.affine, img.header)
        #save_nifti(self._outputs['rtk'],rtk.astype(np.float32), img.affine, img.header)
        save_nifti(self._outputs['kfa'],kfa.astype(np.float32), img.affine, img.header)

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
            dki_micro_fit = dki_micro_model.fit(data_to_fit, mask=well_aligned_mask)

            save_nifti(self._outputs['awf'], dki_micro_fit.awf.astype(np.float32), img.affine, img.header)
            save_nifti(self._outputs['tort'], dki_micro_fit.tortuosity.astype(np.float32), img.affine, img.header)
