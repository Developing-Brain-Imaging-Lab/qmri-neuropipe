import string, os, sys, subprocess, shutil, time

class NODDI_Model():
    def __init__(self, dwi_img, out_base, fit_type='NODDI-WATSON', mask=None, solver='brute2fine', parallel_diffusivity=1.7e-9, iso_diffusivity=3e-9, nthreads=1, verbose=False):
        self._inputs = {}
        self._inputs['dwi_img']     = dwi_img
        self._inputs['out_base']    = out_base
        self._inputs['fit_type']    = fit_type
        self._inputs['mask']        = mask
        self._inputs['solver']      = solver
        self._inputs['dpar']        = parallel_diffusivity
        self._inputs['diso']        = iso_diffusivity
        self._inputs['nthreads']    = nthreads

        self._outputs = {}
        self._outputs['ficvf']               = out_base + '_model-NODDI_parameter-ICVF.nii.gz'
        self._outputs['odi']                 = out_base + '_model-NODDI_parameter-ODI.nii.gz'
        self._outputs['fiso']                = out_base + '_model-NODDI_parameter-FISO.nii.gz'
        self._outputs['exvf']                = out_base + '_model-NODDI_parameter-EXVF.nii.gz'

    def fit(self):
        os.environ["OMP_NUM_THREADS"] = str(self._inputs['nthreads'])
        os.environ["MKL_NUM_THREADS"] = str(self._inputs['nthreads'])
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

        dwi_img = self._inputs['dwi_img']
        output_dir = os.path.dirname(self._inputs['out_base'])

        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        if self._inputs['fit_type'][0:5] == 'NODDI':
            import nibabel as nib
            from dmipy.signal_models import cylinder_models, gaussian_models
            from dmipy.distributions.distribute_models import SD1WatsonDistributed, SD2BinghamDistributed
            from dmipy.core.modeling_framework import MultiCompartmentModel
            from dmipy.core import modeling_framework
            from dmipy.core.acquisition_scheme import acquisition_scheme_from_bvalues
            from dipy.io import read_bvals_bvecs
            from dipy.io.image import load_nifti, save_nifti

            #Setup the acquisition scheme
            bvals, bvecs = read_bvals_bvecs(dwi_img._get_bvals(), dwi_img._get_bvecs())
            bvals_SI = bvals*1e6
            acq_scheme = acquisition_scheme_from_bvalues(bvals_SI, bvecs)
            acq_scheme.print_acquisition_info

            #Load the data
            img = nib.load(dwi_img._get_filename())
            data = img.get_data()

            #Load the mask
            img = nib.funcs.squeeze_image(nib.load(self._inputs['mask']._get_filename()))
            mask_data = img.get_data()

            ball = gaussian_models.G1Ball() #CSF
            stick = cylinder_models.C1Stick() #Intra-axonal diffusion
            zeppelin = gaussian_models.G2Zeppelin() #Extra-axonal diffusion

            if self._inputs['fit_type'] == 'NODDI-BINGHAM':
                dispersed_bundle = SD2BinghamDistributed(models=[stick, zeppelin])
            else:
                dispersed_bundle = SD1WatsonDistributed(models=[stick, zeppelin])

            dispersed_bundle.set_tortuous_parameter('G2Zeppelin_1_lambda_perp','C1Stick_1_lambda_par','partial_volume_0')
            dispersed_bundle.set_equal_parameter('G2Zeppelin_1_lambda_par', 'C1Stick_1_lambda_par')
            dispersed_bundle.set_fixed_parameter('G2Zeppelin_1_lambda_par', self._inputs['dpar'])

            NODDI_mod = MultiCompartmentModel(models=[ball, dispersed_bundle])
            NODDI_mod.set_fixed_parameter('G1Ball_1_lambda_iso', self._inputs['diso'])
            NODDI_fit = NODDI_mod.fit(acq_scheme, data, mask=mask_data, number_of_processors=int(self._inputs['nthreads']), solver=self._inputs['solver'])

            fitted_parameters = NODDI_fit.fitted_parameters

            if self._inputs['fit_type'] == 'NODDI-BINGHAM':
                # get total Stick signal contribution
                vf_intra = (fitted_parameters['SD2BinghamDistributed_1_partial_volume_0'] * fitted_parameters['partial_volume_1'])

                # get total Zeppelin signal contribution
                vf_extra = ((1 - fitted_parameters['SD2BinghamDistributed_1_partial_volume_0'])*fitted_parameters['partial_volume_1'])
                vf_iso = fitted_parameters['partial_volume_0']
                odi = fitted_parameters['SD2BinghamDistributed_1_SD2Bingham_1_odi']

            else:
                # get total Stick signal contribution
                vf_intra = (fitted_parameters['SD1WatsonDistributed_1_partial_volume_0'] * fitted_parameters['partial_volume_1'])

                # get total Zeppelin signal contribution
                vf_extra = ((1 - fitted_parameters['SD1WatsonDistributed_1_partial_volume_0'])*fitted_parameters['partial_volume_1'])
                vf_iso = fitted_parameters['partial_volume_0']
                odi = fitted_parameters['SD1WatsonDistributed_1_SD1Watson_1_odi']

            #Save the images
            save_nifti(self._outputs['odi'], odi, img.affine, img.header)
            save_nifti(self._outputs['ficvf'], vf_intra, img.affine, img.header)
            save_nifti(self._outputs['exvf'], vf_extra, img.affine, img.header)
            save_nifti(self._outputs['fiso'], vf_iso, img.affine, img.header)

        elif self._inputs['fit_type'] == 'AMICO':
            import amico
            amico.core.setup()
            ae = amico.Evaluation(output_dir, '')

            os.chdir(output_dir)
            amico_dwi = output_dir + '/NODDI_data.nii.gz'
            amico_bval = output_dir + '/NODDI_protocol.bvals'
            amico_bvec = output_dir + '/NODDI_protocol.bvecs'
            amico_scheme = output_dir + '/NODDI_protocol.scheme'
            amico_mask = output_dir + '/roi_mask.nii.gz'
            shutil.copy2(dwi_img._get_filename(), amico_dwi)
            shutil.copy2(dwi_img._get_bvals(), amico_bval)
            shutil.copy2(dwi_img._get_bvecs(), amico_bvec)
            shutil.copy2(self._inputs['mask']._get_filename(), amico_mask)

            amico.util.fsl2scheme(amico_bval, amico_bvec)
            ae.load_data(dwi_filename = 'NODDI_data.nii.gz', scheme_filename = 'NODDI_protocol.scheme', mask_filename = 'roi_mask.nii.gz', b0_thr = 0)
            ae.set_model('NODDI')
            ae.CONFIG['solver_params']['numThreads'] = int(self._inputs['nthreads'])

            ae.generate_kernels()
            ae.load_kernels()
            ae.fit()
            ae.save_results()

            amico_ICVF  = output_dir + '/AMICO/NODDI/FIT_ICVF.nii.gz'
            amico_ISOVF = output_dir + '/AMICO/NODDI/FIT_ISOVF.nii.gz'
            amico_OD    = output_dir + '/AMICO/NODDI/FIT_OD.nii.gz'

            shutil.copy2(amico_ICVF, self._outputs['ficvf'])
            shutil.copy2(amico_ISOVF, self._outputs['fiso'])
            shutil.copy2(amico_OD, self._outputs['odi'])

            shutil.rmtree(output_dir + '/AMICO')
            shutil.rmtree(output_dir + '/kernels')
            os.system('rm -rf ' + amico_dwi + ' ' + amico_bval + ' ' + amico_bvec + ' ' + amico_scheme + ' ' + amico_mask)
        else:
            print('Invalid Method')
            exit()

# def fit_noddi_amico(input_dwi, input_bval, input_bvec, input_mask, output_dir, kernel_dir='', nthreads=1, bids_fmt=False, bids_id=''):
#
#     if not os.path.exists(output_dir):
#         os.mkdir(output_dir)
#
#     os.environ["OMP_NUM_THREADS"] = str(nthreads)
#     os.environ["MKL_NUM_THREADS"] = str(nthreads)
#     os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
#
#     import amico
#     amico.core.setup()
#     ae = amico.Evaluation(output_dir, '')
#
#     os.chdir(output_dir)
#     amico_dwi = output_dir + '/NODDI_data.nii.gz'
#     amico_bval = output_dir + '/NODDI_protocol.bvals'
#     amico_bvec = output_dir + '/NODDI_protocol.bvecs'
#     amico_scheme = output_dir + '/NODDI_protocol.scheme'
#     amico_mask = output_dir + '/roi_mask.nii.gz'
#     shutil.copy2(input_dwi, amico_dwi)
#     shutil.copy2(input_bval, amico_bval)
#     shutil.copy2(input_bvec, amico_bvec)
#     shutil.copy2(input_mask, amico_mask)
#
#     amico.util.fsl2scheme(amico_bval, amico_bvec)
#     ae.load_data(dwi_filename = 'NODDI_data.nii.gz', scheme_filename = 'NODDI_protocol.scheme', mask_filename = 'roi_mask.nii.gz', b0_thr = 0)
#     ae.set_model('NODDI')
#     ae.CONFIG['solver_params']['numThreads'] = int(nthreads)
#
#     if kernel_dir != '':
#         shutil.copytree(kernel_dir, output_dir+'/kernels/')
#
#     ae.generate_kernels()
#     ae.load_kernels()
#     ae.fit()
#     ae.save_results()
#
#     amico_dir   = output_dir + '/AMICO/NODDI/FIT_dir.nii.gz'
#     amico_ICVF  = output_dir + '/AMICO/NODDI/FIT_ICVF.nii.gz'
#     amico_ISOVF = output_dir + '/AMICO/NODDI/FIT_ISOVF.nii.gz'
#     amico_OD    = output_dir + '/AMICO/NODDI/FIT_OD.nii.gz'
#
#     if bids_fmt:
#         noddi_dir   = output_dir + '/' + bids_id + '_model-AMICO_NODDI_parameter-DIRECTIONS.nii.gz'
#         noddi_ICVF  = output_dir + '/' + bids_id + '_model-AMICO_NODDI_parameter-FICVF.nii.gz'
#         noddi_ISOVF = output_dir + '/' + bids_id + '_model-AMICO_NODDI_parameter-FISO.nii.gz'
#         noddi_OD    = output_dir + '/' + bids_id + '_model-AMICO_NODDI_parameter-ODI.nii.gz'
#     else:
#         noddi_dir   = output_dir + '/noddi_directions.nii.gz'
#         noddi_ICVF  = output_dir + '/noddi_FICVF.nii.gz'
#         noddi_ISOVF = output_dir + '/noddi_FISO.nii.gz'
#         noddi_OD    = output_dir + '/noddi_ODI.nii.gz'
#
#     shutil.copy2(amico_dir, noddi_dir)
#     shutil.copy2(amico_ICVF, noddi_ICVF)
#     shutil.copy2(amico_ISOVF, noddi_ISOVF)
#     shutil.copy2(amico_OD, noddi_OD)
#
#     shutil.rmtree(output_dir + '/AMICO')
#     shutil.rmtree(output_dir + '/kernels')
#     os.system('rm -rf ' + amico_dwi + ' ' + amico_bval + ' ' + amico_bvec + ' ' + amico_scheme + ' ' + amico_mask)
#
# def fit_noddi_dmipy(input_dwi, input_bval, input_bvec, input_mask, output_dir, nthreads=1, solver='brute2fine', model_type='WATSON', parallel_diffusivity=1.7e-9, iso_diffusivity=3e-9, bids_fmt=False, bids_id=''):
#
#     import nibabel as nib
#     from dmipy.signal_models import cylinder_models, gaussian_models
#     from dmipy.distributions.distribute_models import SD1WatsonDistributed, SD2BinghamDistributed
#     from dmipy.core.modeling_framework import MultiCompartmentModel
#     from dmipy.core import modeling_framework
#     from dmipy.core.acquisition_scheme import acquisition_scheme_from_bvalues
#     from dipy.io import read_bvals_bvecs
#     from dipy.io.image import load_nifti, save_nifti
#
#     if not os.path.exists(output_dir):
#         os.mkdir(output_dir)
#
#     #Setup the acquisition scheme
#     bvals, bvecs = read_bvals_bvecs(input_bval, input_bvec)
#     bvals_SI = bvals*1e6
#     acq_scheme = acquisition_scheme_from_bvalues(bvals_SI, bvecs)
#     acq_scheme.print_acquisition_info
#
#     #Load the data
#     img = nib.load(input_dwi)
#     data = img.get_data()
#
#     #Load the mask
#     img = nib.funcs.squeeze_image(nib.load(input_mask))
#     mask_data = img.get_data()
#
#     ball = gaussian_models.G1Ball() #CSF
#     stick = cylinder_models.C1Stick() #Intra-axonal diffusion
#     zeppelin = gaussian_models.G2Zeppelin() #Extra-axonal diffusion
#
#     if model_type == 'Bingham' or model_type == 'BINGHAM':
#         dispersed_bundle = SD2BinghamDistributed(models=[stick, zeppelin])
#     else:
#         dispersed_bundle = SD1WatsonDistributed(models=[stick, zeppelin])
#
#     dispersed_bundle.set_tortuous_parameter('G2Zeppelin_1_lambda_perp','C1Stick_1_lambda_par','partial_volume_0')
#     dispersed_bundle.set_equal_parameter('G2Zeppelin_1_lambda_par', 'C1Stick_1_lambda_par')
#     dispersed_bundle.set_fixed_parameter('G2Zeppelin_1_lambda_par', parallel_diffusivity)
#
#     NODDI_mod = MultiCompartmentModel(models=[ball, dispersed_bundle])
#     NODDI_mod.set_fixed_parameter('G1Ball_1_lambda_iso', iso_diffusivity)
#     NODDI_fit = NODDI_mod.fit(acq_scheme, data, mask=mask_data, number_of_processors=int(nthreads), solver=solver)
#
#     fitted_parameters = NODDI_fit.fitted_parameters
#
#     if model_type == 'Bingham' or model_type == 'BINGHAM':
#         # get total Stick signal contribution
#         vf_intra = (fitted_parameters['SD2BinghamDistributed_1_partial_volume_0'] * fitted_parameters['partial_volume_1'])
#
#         # get total Zeppelin signal contribution
#         vf_extra = ((1 - fitted_parameters['SD2BinghamDistributed_1_partial_volume_0'])*fitted_parameters['partial_volume_1'])
#         vf_iso = fitted_parameters['partial_volume_0']
#         odi = fitted_parameters['SD2BinghamDistributed_1_SD2Bingham_1_odi']
#
#     else:
#         # get total Stick signal contribution
#         vf_intra = (fitted_parameters['SD1WatsonDistributed_1_partial_volume_0'] * fitted_parameters['partial_volume_1'])
#
#         # get total Zeppelin signal contribution
#         vf_extra = ((1 - fitted_parameters['SD1WatsonDistributed_1_partial_volume_0'])*fitted_parameters['partial_volume_1'])
#         vf_iso = fitted_parameters['partial_volume_0']
#         odi = fitted_parameters['SD1WatsonDistributed_1_SD1Watson_1_odi']
#
#     if bids_fmt:
#         output_odi      = output_dir + '/' + bids_id + '_model-NODDI_parameter-ODI.nii.gz'
#         output_vf_intra = output_dir + '/' + bids_id + '_model-NODDI_parameter-ICVF.nii.gz'
#         output_vf_extra = output_dir + '/' + bids_id + '_model-NODDI_parameter-EXVF.nii.gz'
#         output_vf_iso   = output_dir + '/' + bids_id + '_model-NODDI_parameter-ISO.nii.gz'
#     else:
#         output_odi      = output_dir + '/noddi_ODI.nii.gz'
#         output_vf_intra = output_dir + '/noddi_ICVF.nii.gz'
#         output_vf_extra = output_dir + '/noddi_EXVF.nii.gz'
#         output_vf_iso   = output_dir + '/noddi_ISO.nii.gz'
#
#     #Save the images
#     ras_img = nib.as_closest_canonical(img)
#
#     save_nifti(output_odi, odi, ras_img.affine, ras_img.header)
#     save_nifti(output_vf_intra, vf_intra, ras_img.affine, ras_img.header)
#     save_nifti(output_vf_extra, vf_extra, ras_img.affine, ras_img.header)
#     save_nifti(output_vf_iso, vf_iso, ras_img.affine, ras_img.header)

#def fit_noddi_matlab(noddi_bin, username, input_dwi, input_bval, input_bvec, input_mask, output_dir):
#
#    if not os.path.exists(output_dir):
#        os.mkdir(output_dir)
#
#    shutil.copyfile(input_dwi, output_dir+'/mergedImages.nii.gz')
#    os.system('fslchfiletype NIFTI ' + output_dir+'/mergedImages.nii.gz')
#
#    shutil.copyfile(input_mask, output_dir+'/roi_mask.nii.gz')
#    os.system('fslchfiletype NIFTI ' + output_dir+'/roi_mask.nii.gz')
#
#    shutil.copyfile(input_bval, output_dir+'/noddi_bvals.bval')
#    shutil.copyfile(input_bvec, output_dir+'/noddi_bvecs.bvec')
#
#
#    #if the condor analysis directory is there, remove it to avoid duplicating results
#    if os.path.exists(output_dir + '/CONDOR_NODDI/'):
#        os.system('rm -rf ' + output_dir + '/CONDOR_NODDI/')
#
#
#    #Next, to run the NODDI on CONDOR, we need to first, run Nagesh's bash scripts to:
#    #1 Prep using MATLAB function (performs chunking, etc...)
#    #2 Copy noddi_fitting_condor and other needed condor files
#    #3 Make Dag
#    #4 Submit Dag
#
#    print('\tSubmitting dataset to CONDOR for processing....')
#    #First, change directory to the directory where the condor scripts are located
#    os.chdir(noddi_bin + '/noddiCondor/')
#
#    print('\t\tPrepping data for CONDOR....')
#    #Next, run the prep script
#    os.system('matlab -nodesktop -nosplash -nojvm -r \'noddiCondorPrep(''+noddi_bin+'','' + output_dir +'')\'')
#
#    print('\t\tCopying noddi_fitting_condor executable....')
#    #Run the copy script
#    os.system('sh copy_noddi_fitting_condor.sh ' + noddi_bin + ' ' + output_dir + '/CONDOR_NODDI/')
#
#    print('\t\tMaking DAG FILE....')
#    #Make the DAG file
#    os.system('sh makedag.sh ' + output_dir + '/CONDOR_NODDI/')
#
#    #Submit the dag file to the condor node
#    print('\t\tSUBMITTING DAG FILE....')
#    #Submit the DAG to CONDOR
#    os.system('ssh '+username+'@medusa.keck.waisman.wisc.edu ''sh ' + noddi_bin + '/noddiCondor/submit_dag.sh ' + output_dir + '/CONDOR_NODDI/'+'')
