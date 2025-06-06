import string, os, sys, subprocess, shutil, time

import numpy as np
import nibabel as nib
import dipy.reconst.dti as dti

from bids.layout import writing
import core.utils.create_dataset_json as create_dataset_json

from dipy.denoise.noise_estimate import estimate_sigma
from dipy.core.gradients import gradient_table, reorient_vectors
from dipy.io import  read_bvals_bvecs
from dipy.io.image import load_nifti, save_nifti
from dipy.reconst.dti import fractional_anisotropy, axial_diffusivity, radial_diffusivity, mean_diffusivity
from dipy.io.utils import nifti1_symmat

from core.utils.io import Image, DWImage
from core.dmri.utils.correct_bvals_bvecs import correct_bvals_bvecs

#def calculate_dti_skewness(input_tensor, output_dir):
#    tmp_dir = output_dir + '/tmp/'
#    if not os.path.exists(tmp_dir):
#        os.makedirs(tmp_dir)
#
#    tmp_tensor = tmp_dir+'tensor.nii.gz'
#    md = tmp_dir+'md.nii.gz'
#    shutil.copy2(input_tensor, tmp_tensor)
#    os.chdir(tmp_dir)
#    os.system('TVEigenSystem -in ' + tmp_tensor + ' -type FSL')
#    os.system('TVtool -in ' + tmp_tensor + ' -out ' + md + ' -tr')
#    os.system('fslmaths ' + md + ' -div 3.00 ' + md)
#
#    l1 = tmp_dir+'l1.nii.gz'
#    l2 = tmp_dir+'l2.nii.gz'
#    l3 = tmp_dir+'l3.nii.gz'
#
#    l13 = tmp_dir+'l13.nii.gz'
#    l23 = tmp_dir+'l23.nii.gz'
#    l33 = tmp_dir+'l33.nii.gz'
#    skewness = output_dir + 'dti_SKEWNESS.nii.gz'
#
#    os.system('fslmaths ' + tmp_dir+'tensor_L1.nii.gz -sub ' + md + ' ' + l1)
#    os.system('fslmaths ' + tmp_dir+'tensor_L2.nii.gz -sub ' + md + ' ' + l2)
#    os.system('fslmaths ' + tmp_dir+'tensor_L3.nii.gz -sub ' + md + ' ' + l3)
#    os.system('fslmaths ' + l1 + ' -mul ' + l1 + ' -mul ' + l1 + ' ' + l13)
#    os.system('fslmaths ' + l2 + ' -mul ' + l2 + ' -mul ' + l2 + ' ' + l23)
#    os.system('fslmaths ' + l3 + ' -mul ' + l3 + ' -mul ' + l3 + ' ' + l33)
#
#    os.system('fslmaths ' + l13 + ' -add ' + l23 + ' -add ' + l33 + ' -div 3.0 ' + skewness)
#    os.system('rm -rf ' + tmp_dir)

class DTI_Model():
    def __init__(self, dwi_img, sub_info, out_dir, fit_type='dipy-WLS', mask=None, bmax=None, grad_nonlin=None, full_output=False, nthreads=1):
        self._inputs = {}
        self._inputs['dwi_img']     = dwi_img
        self._inputs['out_dir']     = out_dir
        self._inputs['fit_type']    = fit_type
        self._inputs['mask']        = mask
        self._inputs['bmax']        = bmax
        self._inputs['nthreads']    = nthreads
        self._inputs['full_output'] = full_output
        self._inputs['grad_nonlin'] = grad_nonlin
        
        dti_entities = {}
        dti_entities['subject'] = sub_info['subject']
        dti_entities['session'] = sub_info['session']
        dti_entities['model']   = "DTI"
        
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        
        map_pattern = os.path.join(out_dir, "sub-{subject}[_ses-{session}][_model-{model}]_param-{map}.nii.gz")   
            
        self._outputs = {}
        
        dti_entities['map']               = "FA"
        self._outputs['fa']               = writing.build_path(dti_entities, map_pattern)
        dti_entities['map']               = "MD"
        self._outputs['md']               = writing.build_path(dti_entities, map_pattern)
        dti_entities['map']               = "RD"
        self._outputs['rd']               = writing.build_path(dti_entities, map_pattern)
        dti_entities['map']               = "AD"
        self._outputs['ad']               = writing.build_path(dti_entities, map_pattern)
        dti_entities['map']               = "FSL_TENSOR"
        self._outputs['tensor-fsl']       = writing.build_path(dti_entities, map_pattern)
        dti_entities['map']               = "MRTRIX_TENSOR"
        self._outputs['tensor-mrtrix']    = writing.build_path(dti_entities, map_pattern)
        dti_entities['map']               = "L1" 
        self._outputs['l1']               = writing.build_path(dti_entities, map_pattern)
        dti_entities['map']               = "L2"
        self._outputs['l2']               = writing.build_path(dti_entities, map_pattern)
        dti_entities['map']               = "L3"
        self._outputs['l3']               = writing.build_path(dti_entities, map_pattern)
        dti_entities['map']               = "ITK_TENSOR"
        self._outputs['tensor']           = writing.build_path(dti_entities, map_pattern)
        dti_entities['map']               = "V1"
        self._outputs['v1']               = writing.build_path(dti_entities, map_pattern)
        dti_entities['map']               = "V2"
        self._outputs['v2']               = writing.build_path(dti_entities, map_pattern)
        dti_entities['map']               = "V3"
        self._outputs['v3']               = writing.build_path(dti_entities, map_pattern)

        dti_entities['map']               = 'GA'
        self._outputs['ga']               = writing.build_path(dti_entities, map_pattern)
        dti_entities['map']               = 'COLOR_FA'
        self._outputs['cfa']              = writing.build_path(dti_entities, map_pattern)
        dti_entities['map']               = 'TRACE'
        self._outputs['tr']               = writing.build_path(dti_entities, map_pattern)
        dti_entities['map']               = 'PLANARITY'
        self._outputs['pl']               = writing.build_path(dti_entities, map_pattern)
        dti_entities['map']               = 'SPHERICITY'
        self._outputs['sp']               = writing.build_path(dti_entities, map_pattern)
        dti_entities['map']               = 'MODE'
        self._outputs['mo']               = writing.build_path(dti_entities, map_pattern)
        dti_entities['map']               = 'RESIDUALS'
        self._outputs['res']              = writing.build_path(dti_entities, map_pattern)
        
        
    def create_param_json(self, map):
        create_dataset_json.create_bids_sidecar_json(image = map, 
                                                     data = {"RawSources": [self._inputs['dwi_img'].filename, 
                                                                            self._inputs['dwi_img'].bvals,
                                                                            self._inputs['dwi_img'].bvecs],
                                                             "Estimation Algorithm": self._inputs['fit_type'],
                                                             "Estimation Software": "qmri-neuropipe"})
    
    def fit(self):
        dwi_img = self._inputs['dwi_img']
        output_dir = os.path.dirname(self._inputs['out_dir'])

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        if self._inputs['fit_type'][0:4]=='dipy':
            img = nib.load(dwi_img.filename)
            axis_orient = nib.aff2axcodes(img.affine)
            ras_img = nib.as_closest_canonical(img)
            data = ras_img.get_fdata()
            
            bvals, bvecs = read_bvals_bvecs(dwi_img.bvals, dwi_img.bvecs)
            bvecs = reorient_vectors(bvecs, axis_orient[0]+axis_orient[1]+axis_orient[2],'RAS',axis=1)

            if self._inputs['mask'] != None:
                mask_data = nib.as_closest_canonical(nib.load(self._inputs['mask'].filename)).get_fdata()

            if self._inputs['bmax'] != None:
                jj = np.where(bvals >= self._inputs['bmax'])
                bvals = np.delete(bvals, jj)
                bvecs = np.delete(bvecs, jj, 0)
                data = np.delete(data, jj , axis=3)

            ii = np.where(np.array(bvals) == bvals.min())[0]
            b0_average = np.mean(data[:,:,:,ii], axis=3)

            #Loop over all voxels
            img_shape = data.shape[:-1]

            flat_data   = data.reshape(-1, data.shape[-1])
            flat_params = np.empty((flat_data.shape[0], 12))
            flat_tensor = np.empty((flat_data.shape[0], 6))
            flat_mask   = mask_data.reshape(-1)
            gtab = gradient_table(bvals, bvecs, atol=0.1)

            grad_nonlin_data = None
            if self._inputs['grad_nonlin'] != None:
                grad_nonlin_data = nib.as_closest_canonical(nib.load(self._inputs['grad_nonlin'].filename)).get_fdata().reshape(flat_data.shape[0], 9)

            for vox in range(flat_data.shape[0]):
                if flat_mask[vox] > 0:

                    if self._inputs['grad_nonlin'] != None:
                        corr_bvals, corr_bvecs = correct_bvals_bvecs(bvals, bvecs, grad_nonlin_data[vox])
                        gtab = gradient_table(corr_bvals, corr_bvecs, atol=0.1)

                    dti_model = None
                    if self._inputs['fit_type'] == 'dipy-RESTORE':
                        sigma = estimate_sigma(data)
                        dti_model = dti.TensorModel(gtab, fit_method='RESTORE', sigma=sigma)
                    else:
                        dti_model = dti.TensorModel(gtab, fit_method=self._inputs['fit_type'][5:])

                    dti_fit = dti_model.fit(flat_data[vox])

                    flat_params[vox, :3]   = dti_fit.evals.astype(np.float32)
                    flat_params[vox, 3:12] = dti_fit.evecs.astype(np.float32).ravel()
                    flat_tensor[vox]       = dti.lower_triangular(dti_fit.quadratic_form.astype(np.float32))

            params = flat_params.reshape((img_shape + (12,)))
            evals  = params[...,:3]
            evecs  = params[...,3:12].reshape((img_shape + (3,3)))
            tensor = flat_tensor.reshape((img_shape + (6,)))

            fa = fractional_anisotropy(evals)
            md = mean_diffusivity(evals)
            ad = axial_diffusivity(evals)
            rd = radial_diffusivity(evals)

            ga         = dti.geodesic_anisotropy(evals)
            trace      = dti.trace(evals)
            planarity  = dti.planarity(evals)
            sphericity = dti.sphericity(evals)
            color_fa   = dti.color_fa(fa, evecs)

            if self._inputs['full_output']:
                tensor_img = nifti1_symmat(tensor, ras_img.affine, ras_img.header)
                tensor_img.header.set_intent = 'NIFTI_INTENT_SYMMATRIX'
                tensor_img.to_filename(self._outputs['tensor'])

                tensor_fsl          = np.empty(tensor.shape)
                tensor_fsl[:,:,:,0] = tensor[:,:,:,0]
                tensor_fsl[:,:,:,1] = tensor[:,:,:,1]
                tensor_fsl[:,:,:,2] = tensor[:,:,:,3]
                tensor_fsl[:,:,:,3] = tensor[:,:,:,2]
                tensor_fsl[:,:,:,4] = tensor[:,:,:,4]
                tensor_fsl[:,:,:,5] = tensor[:,:,:,5]
                save_nifti(self._outputs['tensor-fsl'], tensor_fsl, ras_img.affine, ras_img.header)

                tensor_mrtrix           = np.empty(tensor.shape)
                tensor_mrtrix[:,:,:,0]  = tensor[:,:,:,0]
                tensor_mrtrix[:,:,:,1]  = tensor[:,:,:,2]
                tensor_mrtrix[:,:,:,2]  = tensor[:,:,:,5]
                tensor_mrtrix[:,:,:,3]  = tensor[:,:,:,1]
                tensor_mrtrix[:,:,:,4]  = tensor[:,:,:,3]
                tensor_mrtrix[:,:,:,5]  = tensor[:,:,:,4]
                save_nifti(self._outputs['tensor-mrtrix'], tensor_mrtrix, ras_img.affine, ras_img.header)

            #Remove any nan
            fa[np.isnan(fa)]                            = 0
            md[np.isnan(md)]                            = 0
            rd[np.isnan(rd)]                            = 0
            ad[np.isnan(ad)]                            = 0
            ga[np.isnan(ga)]                            = 0
            trace[np.isnan(trace)]                      = 0
            color_fa[np.isnan(color_fa)]                = 0
            planarity[np.isnan(planarity)]              = 0
            sphericity[np.isnan(sphericity)]            = 0
             # dti_mode[np.isnan(dti_mode)]                = 0

            save_nifti(self._outputs['v1'], evecs[:,:,:,:,0], ras_img.affine, ras_img.header)
            save_nifti(self._outputs['v2'], evecs[:,:,:,:,1], ras_img.affine, ras_img.header)
            save_nifti(self._outputs['v3'], evecs[:,:,:,:,2], ras_img.affine, ras_img.header)

            save_nifti(self._outputs['l1'], evals[:,:,:,0], ras_img.affine, ras_img.header)
            save_nifti(self._outputs['l2'], evals[:,:,:,1], ras_img.affine, ras_img.header)
            save_nifti(self._outputs['l3'], evals[:,:,:,2], ras_img.affine, ras_img.header)

            save_nifti(self._outputs['fa'], fa, ras_img.affine, ras_img.header)
            save_nifti(self._outputs['md'], md, ras_img.affine, ras_img.header)
            save_nifti(self._outputs['rd'], rd, ras_img.affine, ras_img.header)
            save_nifti(self._outputs['ad'], ad, ras_img.affine, ras_img.header)
            
            self.create_param_json(Image(self._outputs['fa']))
            self.create_param_json(Image(self._outputs['md']))
            self.create_param_json(Image(self._outputs['rd']))
            self.create_param_json(Image(self._outputs['ad']))
            self.create_param_json(Image(self._outputs['v1']))
            self.create_param_json(Image(self._outputs['v2']))
            self.create_param_json(Image(self._outputs['v3']))
            self.create_param_json(Image(self._outputs['l1']))
            self.create_param_json(Image(self._outputs['l2']))
            self.create_param_json(Image(self._outputs['l3']))

            if self._inputs['full_output']:
                save_nifti(self._outputs['ga'], ga, ras_img.affine, ras_img.header)
                save_nifti(self._outputs['cfa'], color_fa, ras_img.affine, ras_img.header)
                save_nifti(self._outputs['tr'], trace, ras_img.affine, ras_img.header)
                save_nifti(self._outputs['pl'], planarity, ras_img.affine, ras_img.header)
                save_nifti(self._outputs['sp'], sphericity, ras_img.affine, ras_img.header)
                #save_nifti(self._outputs['mo'], dti_mode, ras_img.affine, ras_img.header)
                #save_nifti(self._outputs['res'], residuals, ras_img.affine, ras_img.header)

            orig_ornt   = nib.io_orientation(ras_img.affine)
            targ_ornt   = nib.io_orientation(img.affine)
            transform   = nib.orientations.ornt_transform(orig_ornt, targ_ornt)
            affine_xfm  = nib.orientations.inv_ornt_aff(transform, ras_img.shape)
            trans_mat = affine_xfm[0:3,0:3]

            for key in self._outputs:                
                if os.path.exists(self._outputs[key]):
                    orig_img    = nib.load(self._outputs[key])
                    reoriented  = orig_img.as_reoriented(transform)
                    reoriented.to_filename(self._outputs[key])

            #Correct FSL tensor for orientation
            dirs = []
            dirs.append(np.array([[1],[0],[0]]))
            dirs.append(np.array([[1],[1],[0]]))
            dirs.append(np.array([[1],[0],[1]]))
            dirs.append(np.array([[0],[1],[0]]))
            dirs.append(np.array([[0],[1],[1]]))
            dirs.append(np.array([[0],[0],[1]]))

            if os.path.exists(self._outputs['tensor-fsl']):
                tensor_fsl = nib.load(self._outputs['tensor-fsl'])
                corr_fsl_tensor = np.empty(tensor_fsl.get_fdata().shape)

                for i in range(0,len(dirs)):
                    rot_dir = np.matmul(trans_mat, dirs[i])
                    sign = 1.0
                    if np.sum(rot_dir) == 0.0:
                        sign = -1.0

                    if (np.absolute(rot_dir) == np.array([[1],[0],[0]])).all():
                        tensor_ind = 0
                    elif (np.absolute(rot_dir) == np.array([[1],[1],[0]])).all():
                        tensor_ind = 1
                    elif (np.absolute(rot_dir) == np.array([[1],[0],[1]])).all():
                        tensor_ind = 2
                    elif (np.absolute(rot_dir) == np.array([[0],[1],[0]])).all():
                        tensor_ind = 3
                    elif ( np.absolute(rot_dir) == np.array([[0],[1],[1]])).all():
                        tensor_ind = 4
                    elif ( np.absolute(rot_dir) == np.array([[0],[0],[1]])).all():
                        tensor_ind = 5

                    corr_fsl_tensor[:,:,:,i] = sign*tensor_fsl.get_fdata()[:,:,:,tensor_ind]

                save_nifti(self._outputs['tensor-fsl'], corr_fsl_tensor, tensor_fsl.affine, tensor_fsl.header)

            #Now correct the eigenvectors
            #Determine the order to rearrange
            vec_order = np.transpose(targ_ornt[:,0]).astype(int)
            sign_order = np.transpose(targ_ornt[:,1]).astype(int)

            fsl_v1 = nib.load(self._outputs['v1'])
            corr_fsl_v1 = fsl_v1.get_fdata()[:,:,:,vec_order]
            for i in range(0,2):
                corr_fsl_v1[:,:,:,i] = sign_order[i]*corr_fsl_v1[:,:,:,i]

            save_nifti(self._outputs['v1'], corr_fsl_v1, fsl_v1.affine, fsl_v1.header)

            fsl_v2 = nib.load(self._outputs['v2'])
            corr_fsl_v2 = fsl_v2.get_fdata()[:,:,:,vec_order]
            for i in range(0,2):
                corr_fsl_v2[:,:,:,i] = sign_order[i]*corr_fsl_v2[:,:,:,i]

            save_nifti(self._outputs['v2'], corr_fsl_v2, fsl_v2.affine, fsl_v2.header)

            fsl_v3 = nib.load(self._outputs['v3'])
            corr_fsl_v3 = fsl_v3.get_fdata()[:,:,:,vec_order]
            for i in range(0,2):
                corr_fsl_v3[:,:,:,i] = sign_order[i]*corr_fsl_v3[:,:,:,i]

            save_nifti(self._outputs['v3'], corr_fsl_v3, fsl_v3.affine, fsl_v3.header)

        elif self._inputs['fit_type'][0:6] == 'mrtrix':

            command = 'dwi2tensor -quiet -nthreads ' + str(self._inputs['nthreads'])

            if self._inputs['bmax'] != None:
               img = nib.load(dwi_img.filename)
               data = img.get_fdata()
               bvals, bvecs = read_bvals_bvecs(dwi_img.bvals, dwi_img.bvecs)

               aff = img.get_affine()
               sform = img.get_sform()
               qform = img.get_qform()

               jj = np.where(bvals >= bmax)
               bvals = np.delete(bvals, jj)
               bvecs = np.delete(bvecs, jj, 0)
               data = np.delete(data, jj , axis=3)

               #Save the dwi data
               tmp_dwi_img = nib.Nifti1Image(data,aff,img.header)
               tmp_dwi_img.set_sform(sform)
               tmp_dwi_img.set_qform(qform)
               nib.save(tmp_dwi_img, output_dir+'/tmp_dwi.nii.gz')
               np.savetxt(output_dir+'/tmp_bvals.bval', bvals, fmt='%i')
               np.savetxt(output_dir+'/tmp_bvecs.bvec', np.transpose(bvecs), fmt='%.5f')

               #Run the tensor fitting using MRTRIX:
               command += ' -fslgrad ' + output_dir+'/tmp_bvecs.bvec ' + output_dir+'/tmp_bvals.bval ' + output_dir+'/tmp_dwi.nii.gz'

            else:
               command += ' -fslgrad ' + dwi_img.bvecs + ' ' +  dwi_img.bvals + ' ' + dwi_img.filename

            command += ' ' + self._outputs['tensor-mrtrix']

            if self._inputs['mask'] != None:
               os.system(command+' -mask ' + self._inputs['mask'].filename)
            else:
               os.system(command)

            #Write out the parameters
            os.system('tensor2metric -quiet -force -nthreads ' + str(self._inputs['nthreads']) + ' -adc ' + self._outputs['md'] + ' ' + self._outputs['tensor-mrtrix'])
            os.system('tensor2metric -quiet -force -nthreads ' + str(self._inputs['nthreads']) + ' -fa ' + self._outputs['fa'] + ' ' + self._outputs['tensor-mrtrix'])
            os.system('tensor2metric -quiet -force -nthreads ' + str(self._inputs['nthreads']) + ' -ad ' + self._outputs['ad'] + ' ' + self._outputs['tensor-mrtrix'])
            os.system('tensor2metric -quiet -force -nthreads ' + str(self._inputs['nthreads']) + ' -rd ' + self._outputs['rd'] + ' ' +self._outputs['tensor-mrtrix'])

            os.system('tensor2metric -quiet -force -nthreads ' + str(self._inputs['nthreads']) + ' -value ' + self._outputs['l1'] + ' -num 1 ' + self._outputs['tensor-mrtrix'])
            os.system('tensor2metric -quiet -force -nthreads ' + str(self._inputs['nthreads']) + ' -value ' + self._outputs['l2'] + ' -num 2 ' + self._outputs['tensor-mrtrix'])
            os.system('tensor2metric -quiet -force -nthreads ' + str(self._inputs['nthreads']) + ' -value ' + self._outputs['l3'] + ' -num 3 ' + self._outputs['tensor-mrtrix'])

            os.system('tensor2metric -quiet -force -nthreads ' + str(self._inputs['nthreads']) + ' -vector ' + self._outputs['v1'] + ' -num 1 ' + self._outputs['tensor-mrtrix'])
            os.system('tensor2metric -quiet -force -nthreads ' + str(self._inputs['nthreads']) + ' -vector ' + self._outputs['v2'] + ' -num 2 ' + self._outputs['tensor-mrtrix'])
            os.system('tensor2metric -quiet -force -nthreads ' + str(self._inputs['nthreads']) + ' -vector ' + self._outputs['v3'] + ' -num 3 ' + self._outputs['tensor-mrtrix'])

            os.system('rm -rf ' + output_dir + '/tmp*')

        elif self._inputs['fit_type'][0:6] == 'camino':
            camino_dwi = output_dir + '/tmp_dwi.Bfloat'
            camino_scheme = output_dir + '/tmp_dwi.scheme'
            camino_tensor = output_dir + '/tmp_dti.Bfloat'
            os.system('image2voxel -4dimage ' + dwi_img.filename + ' -outputfile ' + camino_dwi)
            os.system('fsl2scheme -bvecfile ' + dwi_img.bvecs + ' -bvalfile ' + dwi_img.bvals + ' > ' + camino_scheme)

            command = 'modelfit -inputfile ' + camino_dwi + ' -schemefile ' + camino_scheme + ' -bgmask ' + self._inputs['mask']._get_filename() + ' -outputfile ' + camino_tensor

            if self._inputs['fit_type'][7:] == 'RESTORE':
                data = nib.load(dwi_img.filename).get_fdata()
                bvals, bvecs = read_bvals_bvecs(dwi_img.bvals, dwi_img.bvecs)
                values = np.array(bvals)
                ii = np.where(values == bvals.min())[0]
                sigma = estimate_sigma(data)
                sigma = np.mean(sigma[ii])

                #FIT TENSOR
                command += ' -model restore -sigma ' + str(sigma)

            elif self._inputs['fit_type'][7:] == 'WLS':
                command += ' -model ldt_wtd'

            elif self._inputs['fit_type'][7:] == 'NLLS':
                command += ' -model nldt_pos'

            else:
                command += ' -model ldt'

            os.system(command)

            #Convert the data back to NIFTI
            os.system('dt2nii -inputfile ' + camino_tensor + ' -gzip -inputdatatype double -header ' + dwi_img.filename + ' -outputroot ' + output_dir + '/')

            os.system('TVtool -in ' + output_dir + '/dt.nii.gz -scale 1e9 -spd -out ' + self._outputs['tensor'])
            os.system('TVtool -in ' + self._outputs['tensor'] + ' -mask ' + self._inputs['mask'].filename + ' -out ' + self._outputs['tensor'])
            os.system('TVFromEigenSystem -basename dti -type FSL -out ' + self._outputs['tensor'])

            #Calculate FA, MD, RD, AD
            os.system('TVtool -in ' + self._outputs['tensor'] + ' -fa -out ' + self._outputs['fa'])
            os.system('TVtool -in ' + self._outputs['tensor'] + ' -rd -out ' + self._outputs['rd'])
            os.system('TVtool -in ' + self._outputs['tensor'] + ' -ad -out ' + self._outputs['ad'])
            os.system('TVtool -in ' + self._outputs['tensor'] + ' -tr -out ' + self._outputs['tr'])
            os.system('fslmaths ' + self._outputs['tr'] + ' -div 3.00 ' + self._outputs['md'])

            #Output the eigenvectors and eigenvalues
            os.system('TVEigenSystem -in ' + self._outputs['tensor'] + ' -type FSL')
            dti_basename=nib.filename_parser.splitext_addext(self._outputs['tensor'])[0]
            os.system('mv ' + dti_basename + '_V1.nii.gz ' + self._outputs['v1'])
            os.system('mv ' + dti_basename + '_V2.nii.gz ' + self._outputs['v2'])
            os.system('mv ' + dti_basename + '_V3.nii.gz ' + self._outputs['v3'])
            os.system('mv ' + dti_basename + '_L1.nii.gz ' + self._outputs['l1'])
            os.system('mv ' + dti_basename + '_L2.nii.gz ' + self._outputs['l2'])
            os.system('mv ' + dti_basename + '_L3.nii.gz ' + self._outputs['l3'])

            #Clean up files
            os.system('rm -rf ' + dti_basename +'_[V,L]* ' + output_dir + '/tmp*')
            os.system('rm -rf ' + output_dir + '/dt.nii.gz')
            os.system('rm -rf ' + output_dir + '/lns0.nii.gz')
            os.system('rm -rf ' + output_dir + '/exitcode.nii.gz')

class FWEDTI_Model():
    def __init__(self, dwi_img, sub_info, out_dir, fit_type='dipy-WLS', mask=None, bmax=None, grad_nonlin=None, nthreads=1):
        self._inputs = {}
        self._inputs['dwi_img']     = dwi_img
        self._inputs['out_dir']     = out_dir
        self._inputs['fit_type']    = fit_type
        self._inputs['mask']        = mask
        self._inputs['bmax']        = bmax
        self._inputs['nthreads']    = nthreads
        self._inputs['grad_nonlin'] = grad_nonlin
   
        dti_entities = {}
        dti_entities['subject'] = sub_info['subject']
        dti_entities['session'] = sub_info['session']
        dti_entities['model']   = "FWE-DTI"
        
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        
        map_pattern = os.path.join(out_dir, "sub-{subject}[_ses-{session}][_model-{model}]_param-{map}.nii.gz")   
            
        self._outputs = {}
        
        dti_entities['map']               = "FA"
        self._outputs['fa']               = writing.build_path(dti_entities, map_pattern)
        dti_entities['map']               = "MD"
        self._outputs['md']               = writing.build_path(dti_entities, map_pattern)
        dti_entities['map']               = "RD"
        self._outputs['rd']               = writing.build_path(dti_entities, map_pattern)
        dti_entities['map']               = "AD"
        self._outputs['ad']               = writing.build_path(dti_entities, map_pattern)
        dti_entities['map']               = "F"
        self._outputs['f']                = writing.build_path(dti_entities, map_pattern)
        dti_entities['map']               = "L1" 
        self._outputs['l1']               = writing.build_path(dti_entities, map_pattern)
        dti_entities['map']               = "L2"
        self._outputs['l2']               = writing.build_path(dti_entities, map_pattern)
        dti_entities['map']               = "L3"
        self._outputs['l3']               = writing.build_path(dti_entities, map_pattern)
        dti_entities['map']               = "BVALS" 
        self._outputs['bvals']            = writing.build_path(dti_entities, map_pattern)
        dti_entities['map']               = "BVEC_1"
        self._outputs['bvec_1']           = writing.build_path(dti_entities, map_pattern)
        dti_entities['map']               = "BVEC_2"
        self._outputs['bvec_2']           = writing.build_path(dti_entities, map_pattern)
        dti_entities['map']               = "BVEC_3"
        self._outputs['bvec_3']           = writing.build_path(dti_entities, map_pattern)

    def fit(self):

        import dipy.reconst.fwdti as fwdti

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

        if self._inputs['bmax'] != None:
            jj = np.where(bvals >= self._inputs['bmax'])
            bvals = np.delete(bvals, jj)
            bvecs = np.delete(bvecs, jj, 0)
            data = np.delete(data, jj , axis=3)

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
