import numpy as np

def compute_bval_bvec_maps(bvals, bvecs, grad_nonlin, mask=None):
    """
    Corrects b-values and b-vectors for gradient non-linearity effects in Python.

    Args:
        bvals (np.array): 1D numpy array containing the original b-values.
        bvecs (np.array): 2D numpy array containing the original b-vectors (shape: [3, N]).
        grad_nonlin (np.array): 1D numpy array containing the parameters describing the gradient
                                non-linearity correction (length: 9).

    Returns:
        Tuple[np.array, np.array]: Corrected b-values and b-vectors.
    """
    dwi_img = self._inputs['dwi_img']
    output_dir = self._inputs['out_dir']

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if self._inputs['mask'] != None:
        mask_data = nib.load(self._inputs['mask'].filename).get_fdata()

    img = nib.load(dwi_img.filename)
    data = img.get_fdata()
    bvals, bvecs = read_bvals_bvecs(dwi_img.bvals, dwi_img.bvecs)
    gtab = gradient_table(bvals, bvecs)

    #Loop over all voxels
    img_shape = data.shape[:-1]

    flat_data   = data.reshape(-1, data.shape[-1])
    flat_params = np.empty((flat_data.shape[0], 13))
    flat_tensor = np.empty((flat_data.shape[0], 6))
    flat_mask   = mask_data.reshape(-1)
    gtab = gradient_table(bvals, bvecs, atol=0.1)
    bval_map    = np.empty(flat_data.shape)
    bvec_map    = np.empty((flat_data.shape + (3,))) 

    grad_nonlin_data = None
    if self._inputs['grad_nonlin'] != None:
        grad_nonlin_data = nib.load(self._inputs['grad_nonlin'].filename).get_fdata().reshape(flat_data.shape[0], 9)

    for vox in range(flat_data.shape[0]):
        if flat_mask[vox] > 0:
            if self._inputs['grad_nonlin'] != None:
                corr_bvals, corr_bvecs = correct_bvals_bvecs(bvals, bvecs, grad_nonlin_data[vox])
                gtab = gradient_table(corr_bvals, corr_bvecs, atol=0.1)
                bval_map[vox] = corr_bvals
                bvec_map[vox] = corr_bvecs

            fwidtimodel = fwdti.FreeWaterTensorModel(gtab, self._inputs['fit_type'])
            fwidti_fit = fwidtimodel.fit(flat_data[vox])
            flat_params[vox] = fwidti_fit.model_params
            
    params = flat_params.reshape((img_shape + (13,)))
    evals = params[...,:3].astype(np.float32)
    evecs = params[...,3:12].reshape((img_shape + (3,3))).astype(np.float32)
    f     = params[...,12].astype(np.float32)

    fa = fractional_anisotropy(evals)
    md = mean_diffusivity(evals)
    ad = axial_diffusivity(evals)
    rd = radial_diffusivity(evals)

    #Remove any nan
    fa[np.isnan(fa)] = 0
    md[np.isnan(md)] = 0
    rd[np.isnan(rd)] = 0
    ad[np.isnan(ad)] = 0
    f[np.isnan(f)]   = 0    

    # #Calculate Parameters for FWDTI Model
    save_nifti(self._outputs['fa'], fa.astype(np.float32), img.affine, img.header)
    save_nifti(self._outputs['md'], md.astype(np.float32), img.affine, img.header)
    save_nifti(self._outputs['rd'], rd.astype(np.float32), img.affine, img.header)
    save_nifti(self._outputs['ad'], ad.astype(np.float32), img.affine, img.header)
    save_nifti(self._outputs['f'],  f.astype(np.float32),  img.affine, img.header)
    save_nifti(self._outputs['l1'], evals[:,:,:,0], img.affine, img.header)
    save_nifti(self._outputs['l2'], evals[:,:,:,1], img.affine, img.header)
    save_nifti(self._outputs['l3'], evals[:,:,:,2], img.affine, img.header)

    save_nifti(self._outputs['bvals'], bval_map.reshape(data.shape), img.affine, img.header)
    save_nifti(self._outputs['bvec_1'], bvec_map.reshape((data.shape+(3,)))[...,0], img.affine, img.header)
    save_nifti(self._outputs['bvec_2'], bvec_map.reshape((data.shape+(3,)))[...,1], img.affine, img.header)
    save_nifti(self._outputs['bvec_3'], bvec_map.reshape((data.shape+(3,)))[...,2], img.affine, img.header)

    return bvals_c, bvecs_c
