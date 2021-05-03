import string, os, sys, subprocess, shutil, time

import numpy as np
import nibabel as nib

class CSD_Model():
    def __init__(self, dwi_img, out_base, response_algo='tournier', fod_algo='csd', mask=None, nthreads=1, verbose=False):
        self._inputs = {}
        self._inputs['dwi_img']         = dwi_img
        self._inputs['out_base']        = out_base
        self._inputs['response_algo']   = response_algo
        self._inputs['fod_algo']        = fod_algo
        self._inputs['mask']            = mask
        self._inputs['nthreads']        = nthreads

        self._outputs = {}
        self._outputs['fod']            = out_base + '_model-CSD_parameter-FOD.nii.gz'

    def fit(self):

        dwi_img = self._inputs['dwi_img']
        output_dir = os.path.dirname(self._inputs['out_base'])

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        dwi_mif = output_dir + '/tmp_dwi.mif'
        response_file = output_dir + '/response_function.txt'
        dwi_fod = output_dir + '/fod.mif'
        dwi_fod_nifti = output_dir + '/fod.nii.gz'

        os.system('mrconvert -quiet -force -fslgrad ' + dwi_img._get_bvecs() + ' ' + dwi_img._get_bvals() + ' ' + dwi_img._get_filename() + ' ' + dwi_mif + ' -nthreads ' + str(self._inputs['nthreads']))
        os.system('dwi2response ' + self._inputs['response_algo'] + ' ' + dwi_mif + ' ' + response_file + ' -nthreads ' + str(self._inputs['nthreads']) + ' -quiet -force')
        os.system('dwi2fod ' + self._inputs['fod_algo'] + ' ' + dwi_mif + ' ' + response_file + ' ' + dwi_fod + ' -mask ' + self._inputs['mask']._get_filename() + ' -nthreads ' + str(self._inputs['nthreads']) + ' -quiet -force')
        os.system('mrconvert ' + dwi_fod + ' ' + self._outputs['fod'] + ' -nthreads ' + str(self._inputs['nthreads'])+ ' -quiet -force')

        os.system('rm -rf ' + output_dir+'/tmp*')
        os.system('rm -rf ' + output_dir+'/fod.mif')
