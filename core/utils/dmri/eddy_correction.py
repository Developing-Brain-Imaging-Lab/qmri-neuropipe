import string, os, sys, subprocess, shutil, time, copy
import numpy as np

from core.utils.io import Image, DWImage
import core.utils.mask as mask

if sys.platform == 'linux':
    eddy='eddy_openmp'
    eddy_cuda='eddy_cuda11.1'
else:
    eddy='eddy'

def eddy_correct_fsl(input_dwi, output_base):

    log_file = output_base + '.ecclog'

    if os.path.exists(log_file):
        os.remove(log_file)

    output_img = copy.deepcopy(input_dwi)
    output_img._set_filename(output_base+'_desc-EddyCurrentCorrected_dwi.nii.gz')
    output_img._set_bvecs(output_base+'_desc-EddyCurrentCorrected_dwi.bvec')

    subprocess.run(['eddy_correct',
                    input_dwi._get_filename(),
                    output_base,
                    '0'], stderr=subprocess.STDOUT)

    subprocess.run(['fdt_rotate_bvecs',
                    input_dwi._get_bvecs(),
                    output_dwi._get_bvecs(),
                    log_file], stderr=subprocess.STDOUT)

    #Rotate b-vecs after doing the eddy correction
    os.remove(log_file)

    return output_img

def eddy_fsl(input_dwi, output_base, mask_img=None, topup_base=None, external_b0=None, repol=0, data_shelled=0, mb=None, cuda=False, mporder=0, ol_type='sw', mb_off='1', estimate_move_by_suscept=False, cuda_device=None, nthreads='1', fsl_eddy_options=''):

    output_dir = os.path.dirname(output_base)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    eddy_output_base = output_base +'_desc-EddyCurrentCorrected_dwi'

    output_img = copy.deepcopy(input_dwi)
    output_img._set_filename(eddy_output_base+'.nii.gz')
    output_img._set_bvecs(eddy_output_base+'.bvec')

    if mask_img == None:
        mask_img = Image(file = output_dir + '/mask.nii.gz')
        mask.mask_image(input_dwi, mask_img, method='bet')

    exe = ''
    if cuda:
        if cuda_device:
            exe = 'CUDA_VISIBLE_DEVICES='+str(cuda_device)+ ' ' + eddy_cuda
        else:
            exe = eddy_cuda
    else:
        exe = 'OMP_NUM_THREADS='+str(nthreads)+ ' ' + eddy

    command = exe + ' --imain=' + input_dwi._get_filename() \
              + ' --mask='  + mask_img._get_filename() \
              + ' --index=' + input_dwi._get_index() \
              + ' --acqp='  + input_dwi._get_acqparams() \
              + ' --bvecs=' + input_dwi._get_bvecs() \
              + ' --bvals=' + input_dwi._get_bvals() \
              + ' --slspec=' + input_dwi._get_slspec() \
              + ' --out='   + eddy_output_base \
              + ' --cnr_maps --residuals --ol_type='+ol_type

    if topup_base != None:
        command += ' --topup='+topup_base
    if external_b0 != None:
        command += ' --field='+external_b0
    if repol != 0:
        command += ' --repol '
    if data_shelled != 0:
        command += ' --data_is_shelled '
    if mb != None:
        command += ' --mb=' + str(mb)
    if mporder != 0:
        command += ' --mporder='+str(mporder)
    if estimate_move_by_suscept == True:
        command += ' --estimate_move_by_susceptibility'

    command += ' ' + fsl_eddy_options

    os.system(command)
    #Rotate b-vecs after doing the eddy correction
    os.system('mv ' + eddy_output_base +'.eddy_rotated_bvecs ' + output_img._get_bvecs())
    os.remove(output_dir + '/mask.nii.gz')

    return output_img

def compute_average_motion(eddy_basename):
    movement_rms_file = eddy_basename + '.eddy_movement_rms'
    restricted_movement_rms_file = eddy_basename + '.eddy_restricted_movement_rms'

    movement_rms = np.loadtxt(movement_rms_file)
    restricted_movement_rms = np.loadtxt(restricted_movement_rms_file)


    avg_movement_rms = np.mean(movement_rms, axis=0)
    avg_restricted_movement_rms = np.mean(restricted_movement_rms, axis=0)

    avg_global_displacement = avg_movement_rms[0]
    avg_slice_displacement  = avg_movement_rms[1]

    avg_restricted_displacement = avg_restricted_movement_rms[0]
    avg_restricted_slice_displacement = avg_restricted_movement_rms[1]

    return ('Average Total Movement', avg_movement_rms[0],
            'Average Slice Movement', avg_movement_rms[1],
            'Average Restricted Movement', avg_restricted_movement_rms[0],
            'Average Restricted Slice Movement', avg_restricted_movement_rms[1])
