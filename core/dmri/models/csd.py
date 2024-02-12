import string, os, sys, subprocess, shutil, time

import numpy as np
import nibabel as nib
from bids.layout import writing

class CSD_Model():
    def __init__(self, dwi_img, sub_info, out_dir, mask=None, struct_img=None, response_algo='tournier', fod_algo='csd',  nthreads=1, verbose=False):
        self._inputs = {}
        self._inputs['dwi_img']           = dwi_img
        self._inputs['struct_img']        = struct_img
        self._inputs['out_dir']           = out_dir
        self._inputs['response_algo']     = response_algo
        self._inputs['fod_algo']          = fod_algo
        self._inputs['mask']              = mask
        self._inputs['struct_img']        = struct_img
        self._inputs['nthreads']          = nthreads
        self._inputs['verbose']           = verbose
        
        map_entities = {}
        map_entities['subject'] = sub_info['subject']
        map_entities['session'] = sub_info['session']
        map_entities['model']   = "CSD"
        map_pattern             = os.path.join(out_dir, "sub-{subject}[_ses-{session}][_model-{model}]_param-{map}.nii.gz")
        norm_map_pattern        = os.path.join(out_dir, "sub-{subject}[_ses-{session}][_model-{model}]_desc-{desc}_param-{map}.nii.gz")   
        
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        self._outputs = {}
        map_entities['map']                = "FOD"
        map_entities['desc']               = "MTNorm"
        self._outputs['fod']               = writing.build_path(map_entities, map_pattern)
        self._outputs['fod_norm']          = writing.build_path(map_entities, norm_map_pattern)
        
        if self._inputs['response_algo']  == 'msmt_5tt':
            map_entities['model'] = 'MSMT-5tt'
        elif self._inputs['response_algo']  == 'dhollander':
            map_entities['model'] = 'DHOLLANDER'
        elif self._inputs['response_algo']  == 'tournier'  or self._inputs['response_algo']  == 'fa' or self._inputs['response_algo']  == 'tax':
            map_entities['model'] = 'CSD'
        else:
            print('Incorrect Response Function')
            exit()
    
        map_entities['map']                = "WMfod"
        self._outputs['wmfod']             = writing.build_path(map_entities, map_pattern)
        self._outputs['wmfod_norm']        = writing.build_path(map_entities, norm_map_pattern)
        
        map_entities['map']                = "GMfod"
        self._outputs['gmfod']             = writing.build_path(map_entities, map_pattern)
        self._outputs['gmfod_norm']        = writing.build_path(map_entities, norm_map_pattern)
        
        map_entities['map']                = "CSFfod"
        self._outputs['csffod']            = writing.build_path(map_entities, map_pattern)
        self._outputs['csffod_norm']        = writing.build_path(map_entities, norm_map_pattern)
        
        
        map_pattern                        = os.path.join(out_dir, "sub-{subject}[_ses-{session}][_model-{model}]_desc-{desc}_dwi.txt")
        map_entities['desc']               = "csd-response"
        self._outputs['response_func']  = writing.build_path(map_entities, map_pattern)

        map_entities['desc']               = "wm-response"
        self._outputs['wm_response_func']  = writing.build_path(map_entities, map_pattern)
        
        map_entities['desc']               = "gm-response"
        self._outputs['gm_response_func']  = writing.build_path(map_entities, map_pattern)
        
        map_entities['desc']               = "csf-response"
        self._outputs['csf_response_func'] = writing.build_path(map_entities, map_pattern)  
        
        
        mask55t_pattern              = os.path.join(out_dir, "sub-{subject}[_ses-{session}][_model-{model}]_desc-{desc}_mask.nii.gz")
        map_entities['desc']         = "55tgen"
        self._outputs['mask_55tgen'] = writing.build_path(map_entities, mask55t_pattern)  
        

    def fit(self):

        dwi_img = self._inputs['dwi_img']
        mask_img = self._inputs['mask']

        output_dir = os.path.dirname(self._inputs['out_dir'])

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        dwi_mif = output_dir + '/tmp_dwi.mif'
        os.system('mrconvert -quiet -force -fslgrad ' + dwi_img.bvecs + ' ' + dwi_img.bvals + ' ' + dwi_img.filename + ' ' + dwi_mif + ' -nthreads ' + str(self._inputs['nthreads']) + ' --strides 0,0,0,1')

        if self._inputs['fod_algo'] == 'msmt_csd':
       
            if self._inputs['response_algo']  == 'msmt_5tt':
                struct_img = self._inputs['struct_img']

                if struct_img == None:
                    print('Need to provide structural image with MSMT_5tt')
                    exit()
                    
                #Run the 5TT masking
                cmd = '5ttgen fsl ' \
                    + '-mask ' + mask_img.filename \
                    + ' -force -quiet -nthreads ' + str(self._inputs['nthreads']) \
                    + ' ' + struct_img.filename \
                    + ' ' + self._outputs['mask_55tgen']
                os.system(cmd)

                #Generage the response functions
                cmd = 'dwi2response msmt_5tt ' \
                    + ' -force -quiet -nthreads ' + str(self._inputs['nthreads']) \
                    + ' ' + dwi_mif + ' ' \
                    + self._outputs['mask_55tgen'] + ' ' \
                    + self._outputs['wm_response_func'] + ' ' \
                    + self._outputs['gm_response_func'] + ' ' \
                    + self._outputs['csf_response_func']
                os.system(cmd)
                    
            else:
                #Generage the response functions
                cmd = 'dwi2response dhollander ' \
                    + ' -force -quiet -nthreads ' + str(self._inputs['nthreads']) \
                    + ' ' + dwi_mif + ' ' \
                    + ' -mask ' + mask_img.filename + ' ' \
                    + self._outputs['wm_response_func'] + ' ' \
                    + self._outputs['gm_response_func'] + ' ' \
                    + self._outputs['csf_response_func']
                    
                os.system(cmd)
        

            cmd = 'dwi2fod msmt_csd ' \
                + ' -force -quiet -nthreads ' + str(self._inputs['nthreads']) + ' ' \
                + dwi_mif + ' ' \
                + self._outputs['wm_response_func'] + ' ' \
                + self._outputs['wmfod'] + ' ' \
                + self._outputs['gm_response_func'] + ' ' \
                + self._outputs['gmfod'] + ' ' \
                + self._outputs['csf_response_func'] + ' ' \
                + self._outputs['csffod'] + ' ' \
                + '-mask ' + mask_img.filename
            os.system(cmd)
            
            cmd = 'mtnormalise ' \
                + ' -force -quiet -nthreads ' + str(self._inputs['nthreads']) + ' ' \
                + self._outputs['wmfod'] + ' ' \
                + self._outputs['wmfod_norm'] + ' ' \
                + self._outputs['gmfod'] + ' ' \
                + self._outputs['gmfod_norm'] + ' ' \
                + self._outputs['csffod'] + ' ' \
                + self._outputs['csffod_norm'] + ' ' \
                + '-mask ' + mask_img.filename
            os.system(cmd)
            
            
        else:
            #Generage the response functions            
            cmd = 'dwi2response' \
                + ' ' + self._inputs['response_algo'] \
                + ' -force -quiet -nthreads ' + str(self._inputs['nthreads']) \
                + ' -mask ' + mask_img.filename \
                + ' ' + dwi_mif + ' ' \
                +  self._outputs['response_func']
            
            if self._inputs['verbose']:
                print(cmd)
            os.system(cmd)

            #Now Generage FOD generation
            cmd = 'dwi2fod csd ' \
                + ' -force -quiet -nthreads ' + str(self._inputs['nthreads']) \
                + ' -mask ' + mask_img.filename \
                + ' ' + dwi_mif + ' ' \
                + self._outputs['response_func'] + ' ' \
                + self._outputs['fod']
                
            if self._inputs['verbose']:
                print(cmd)
            os.system(cmd)
            
            cmd = 'mtnormalise ' \
                + ' -force -quiet -nthreads ' + str(self._inputs['nthreads']) + ' ' \
                + self._outputs['fod'] + ' '\
                + self._outputs['fod_norm'] + ' ' \
                + '-mask ' + mask_img.filename
            os.system(cmd)
            
        os.system('rm -rf ' + output_dir+'/tmp*')
