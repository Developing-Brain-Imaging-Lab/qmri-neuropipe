import string, os, sys, subprocess, shutil, time

import numpy as np
import nibabel as nib

class CSD_Model():
    def __init__(self, dwi_img, out_base, mask=None, struct_img=None, response=None, wm_response=None, gm_response=None, csf_response=None, response_algo='tournier', fod_algo='csd',  nthreads=1, verbose=False):
        self._inputs = {}
        self._inputs['dwi_img']         = dwi_img
        self._inputs['struct_img']      = struct_img
        self._inputs['out_base']        = out_base
        self._inputs['response_algo']   = response_algo
        self._inputs['fod_algo']        = fod_algo
        self._inputs['mask']            = mask
        self._inputs['nthreads']        = nthreads
        self._inputs['response_func']    = response
        self._inputs['wm_response_func'] = wm_response
        self._inputs['gm_response_func'] = gm_response
        self._inputs['csf_response_func']= csf_response

        self._outputs = {}
        self._outputs['fod']            = out_base + '_model-CSD_parameter-FOD.nii.gz'

    def fit(self):

        dwi_img = self._inputs['dwi_img']
        mask_img = self._inputs['mask']

        output_dir = os.path.dirname(self._inputs['out_base'])

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        dwi_mif = output_dir + '/tmp_dwi.mif'
        os.system('mrconvert -quiet -force -fslgrad ' + dwi_img._get_bvecs() + ' ' + dwi_img._get_bvals() + ' ' + dwi_img._get_filename() + ' ' + dwi_mif + ' -nthreads ' + str(self._inputs['nthreads']) + ' --strides 0,0,0,1')

        if self._inputs['fod_algo'] == 'msmt_csd':

            if self._inputs['response_algo']  == 'msmt_5tt':
                struct_img = self._inputs['struct_img']

                if struct_img == None:
                    print('Need to provide structural image with MSMT_5tt')
                    exit()
                    
                #Run the 5TT masking
                cmd = '5ttgen fsl ' \
                    + '-mask ' + mask_img._get_filename() \
                    + ' -force -quiet -nthreads ' + str(self._inputs['nthreads']) \
                    + ' ' + struct_img._get_filename() \
                    + ' ' + self._inputs['out_base'] + '_desc-5ttgen_mask.nii.gz'
                os.system(cmd)

                if self._inputs['wm_response_func'] == None or self._inputs['gm_response_func'] == None or self._inputs['csf_response_func'] == None:
                    self._inputs['wm_response_func'] = self._inputs['out_base'] + '_desc-wm-response_dwi.txt'
                    self._inputs['gm_response_func'] = self._inputs['out_base'] + '_desc-gm-response_dwi.txt'
                    self._inputs['csf_response_func'] = self._inputs['out_base'] + '_desc-csf-response_dwi.txt'
                    
                    #Generage the response functions
                    cmd = 'dwi2response msmt_5tt ' \
                        + ' -force -quiet -nthreads ' + str(self._inputs['nthreads']) \
                        + ' ' + dwi_mif + ' ' \
                        + self._inputs['out_base'] + '_desc-5ttgen_mask.nii.gz ' \
                        + self._inputs['wm_response_func'] + ' ' \
                        + self._inputs['gm_response_func'] + ' ' \
                        + self._inputs['csf_response_func']
                    os.system(cmd)

            else:
            
                if self._inputs['wm_response_func'] == None or self._inputs['gm_response_func'] == None or self._inputs['csf_response_func'] == None:
                    self._inputs['wm_response_func'] = self._inputs['out_base'] + '_desc-wm-response_dwi.txt'
                    self._inputs['gm_response_func'] = self._inputs['out_base'] + '_desc-gm-response_dwi.txt'
                    self._inputs['csf_response_func'] = self._inputs['out_base'] + '_desc-csf-response_dwi.txt'
                    
                    #Generage the response functions
                    cmd = 'dwi2response ' + self._inputs['response_algo'] \
                        + ' -force -quiet -nthreads ' + str(self._inputs['nthreads']) \
                        + ' ' + dwi_mif + ' ' \
                        + self._inputs['wm_response_func'] + ' ' \
                        + self._inputs['gm_response_func'] + ' ' \
                        + self._inputs['csf_response_func'] + ' ' \
                        + '-mask ' + mask_img._get_filename()
                    print(cmd)
                    os.system(cmd)

            #Now Generage FOD generation
            parameter_base = self._inputs['out_base']
            if self._inputs['response_algo']  == 'msmt_5tt':
                parameter_base += '_model-MSMT-5tt'
            elif self._inputs['response_algo']  == 'dhollander':
                parameter_base += '_model-DHOLLANDER'
            else:
                print('Incorrect Response Function')
                exit()

            cmd = 'dwi2fod msmt_csd ' \
                + ' -force -quiet -nthreads ' + str(self._inputs['nthreads']) + ' ' \
                + dwi_mif + ' ' \
                + self._inputs['wm_response_func'] + ' ' \
                + parameter_base + '_parameter-WMfod.nii.gz ' \
                + self._inputs['gm_response_func'] + ' ' \
                + parameter_base + '_parameter-GMfod.nii.gz ' \
                + self._inputs['csf_response_func'] + ' ' \
                + parameter_base + '_parameter-CSFfod.nii.gz ' \
                + '-mask ' + mask_img._get_filename()
                
            print(cmd)
            os.system(cmd)

        else:
            #Generage the response functions
            if self._inputs['response_algo'] == 'dhollander':
            
                if self._inputs['wm_response_func'] == None or self._inputs['gm_response_func'] == None or self._inputs['csf_response_func'] == None:
                    self._inputs['wm_response_func'] = self._inputs['out_base'] + '_desc-wm-response_dwi.txt'
                    self._inputs['gm_response_func'] = self._inputs['out_base'] + '_desc-gm-response_dwi.txt'
                    self._inputs['csf_response_func'] = self._inputs['out_base'] + '_desc-csf-response_dwi.txt'
                    
                #Generage the response functions
                cmd = 'dwi2response ' + self._inputs['response_algo'] \
                    + ' -force -quiet -nthreads ' + str(self._inputs['nthreads']) \
                    + ' ' + dwi_mif + ' ' \
                    + self._inputs['wm_response_func'] + ' ' \
                    + self._inputs['gm_response_func'] + ' ' \
                    + self._inputs['csf_response_func'] + ' ' \
                    + '-mask ' + mask_img._get_filename()
                    
                print(cmd)
                os.system(cmd)
            
            
            else:
                if self._inputs['response_func'] == None:
                    self._inputs['response_func'] = self._inputs['out_base'] + '_desc-csd-response_dwi.txt'
            
                cmd = 'dwi2response' \
                    + ' ' + self._inputs['response_algo'] \
                    + ' -force -quiet -nthreads ' + str(self._inputs['nthreads']) \
                    + ' -mask ' + mask_img._get_filename() \
                    + ' ' + dwi_mif + ' ' \
                    +  self._inputs['response_func']
                
                print(cmd)
                os.system(cmd)

                #Now Generage FOD generation
                cmd = 'dwi2fod' \
                    + ' ' + self._inputs['fod_algo'] \
                    + ' -force -quiet -nthreads ' + str(self._inputs['nthreads']) \
                    + ' -mask ' + mask_img._get_filename() \
                    + ' ' + dwi_mif + ' ' \
                    + self._inputs['response_func'] + ' ' \
                    + self._inputs['out_base'] + '_model-CSD_parameter-FOD.nii.gz'
                
                print(cmd)
                os.system(cmd)
            


        os.system('rm -rf ' + output_dir+'/tmp*')
