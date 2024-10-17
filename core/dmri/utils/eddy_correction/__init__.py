import string, os, sys, subprocess, shutil, time, copy
import numpy as np

from core.utils.io import Image, DWImage
import core.utils.mask as mask


eddy='eddy_cpu'
eddy_cuda='eddy_cuda10.2'

def eddy_correct_fsl(input_dwi, output_base):

    output_dir = os.path.dirname(output_base)
    log_file   = os.path.join(output_dir, "dwi.ecclog")
    bvecs_file = os.path.join(output_dir, "dwi.bvecs")
    dwi_file   = os.path.join(output_dir, "dwi.nii.gz")

    if os.path.exists(log_file):
        os.remove(log_file)
        
    #Implementing FSL eddy_correct
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    os.system(f"eddy_correct {input_dwi.filename} {dwi_file} 0")
    os.system(f"fdt_rotate_bvecs {input_dwi.bvecs} {bvecs_file} {log_file}")

    output_img = copy.deepcopy(input_dwi)
    output_img.filename = output_base+'_desc-EddyCurrentCorrected_dwi.nii.gz'
    output_img.bvecs    = output_base+'_desc-EddyCurrentCorrected_dwi.bvec'

    os.rename(dwi_file, output_img.filename)
    os.rename(bvecs_file, output_img.bvecs)

    #Rotate b-vecs after doing the eddy correction
    os.remove(log_file)

    return output_img

def eddy_fsl(input_dwi, output_base, mask_img=None, topup_base=None, external_b0=None, cuda=False, cuda_device=None, nthreads=1, fsl_eddy_options='', debug=False):

    output_dir = os.path.dirname(output_base)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    eddy_output_base = output_base +'_desc-EddyCurrentCorrected_dwi'

    output_img = copy.deepcopy(input_dwi)
    output_img.filename = eddy_output_base+'.nii.gz'
    output_img.bvecs    = eddy_output_base+'.bvec'

    if mask_img == None:
        mask_img = Image(filename = os.path.join(output_dir, "mask.nii.gz"))
        mask.mask_image(input_dwi, mask_img, algo='bet', bet_options='-f 0.1')

    exe = ''
    if cuda:
        if cuda_device:
            exe = 'CUDA_VISIBLE_DEVICES='+str(cuda_device)+ ' ' + eddy_cuda
        else:
            exe = eddy_cuda
    else:
        os.environ["OMP_NUM_THREADS"] = str(nthreads)
        exe = 'OMP_NUM_THREADS='+str(nthreads)+ " " + eddy

    command = exe + ' --imain=' + input_dwi.filename \
              + ' --mask='  + mask_img.filename \
              + ' --index=' + input_dwi.index \
              + ' --acqp='  + input_dwi.acqparams \
              + ' --bvecs=' + input_dwi.bvecs \
              + ' --bvals=' + input_dwi.bvals \
              + ' --slspec=' + input_dwi.slspec \
              + ' --out='   + eddy_output_base \
              + ' --nthr='   + str(nthreads)

    if topup_base != None:
        command += ' --topup='+topup_base
    if external_b0 != None:
        command += ' --field='+external_b0
   
    command += " " + fsl_eddy_options

    if debug:
        print(command)
        
    os.system(command)
    #Rotate b-vecs after doing the eddy correction
    os.rename(eddy_output_base+'.eddy_rotated_bvecs', output_img.bvecs)
    os.remove(os.path.join(output_dir, "mask.nii.gz"))

    return output_img

def compute_average_motion(eddy_basename):
    movement_rms_file = os.path.join(eddy_basename, '.eddy_movement_rms')
    restricted_movement_rms_file = os.path.join(eddy_basename, '.eddy_restricted_movement_rms')

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

def diffprep_tortoise(input_dwi, output_base, phase='horizontal', tortoise_options=None, struct_img=None, nthreads=1, verbose=False):
    
    current_dir = os.getcwd()
    output_dir = os.path.dirname(output_base)

    proc_dir = output_dir + '/tort_tmp/'
    
    if not os.path.exists(proc_dir):
        os.makedirs(proc_dir)
    os.chdir(proc_dir)
    
    dwi_img_base = input_dwi.filename.split('/')[-1]
    tort_proc_img   = proc_dir + dwi_img_base.split('.')[0]+'_DMC.nii'
    tort_proc_bmtxt = proc_dir + dwi_img_base.split('.')[0]+'_DMC.bmtxt'
    tort_proc_bval  = proc_dir + dwi_img_base.split('.')[0]+'_DMC.bvals'
    tort_proc_bvec  = proc_dir + dwi_img_base.split('.')[0]+'_DMC.bvecs'
    
    
    shutil.copy(input_dwi.filename, proc_dir)
    shutil.copy(input_dwi.bvecs, proc_dir)
    shutil.copy(input_dwi.bvals, proc_dir)
    
    if struct_img:
        shutil.copy(struct_img.filename, proc_dir)
    
    diffprep_cmd = 'DIFFPREP --dwi ' + dwi_img_base \
                 + ' --bvecs ' + input_dwi.bvecs.split('/')[-1]  \
                 + ' --bvals ' + input_dwi.bvals.split('/')[-1]  \
                 + ' --phase ' + phase
    
    if struct_img:
        diffprep_cmd += ' --structural ' + struct_img.filename.split('/')[-1]
        
    if tortoise_options:
        diffprep_cmd += ' ' + tortoise_options
        
    if verbose:
        os.print(diffprep_cmd)
        
        
    os.system('OMP_NUM_THREADS='+str(nthreads)+ ' ' + diffprep_cmd)
    os.system('TORTOISEBmatrixToFSLBVecs ' + tort_proc_bmtxt)
    
    #After DIFF PREP, copy processed data back
    eddy_output_base = output_base +'_desc-EddyCurrentCorrected_dwi'
    eddy_output_img  = eddy_output_base + '.nii.gz'
    eddy_output_bvec = eddy_output_base + '.bvec'
    
    shutil.copy(tort_proc_img, eddy_output_img)
    shutil.copy(tort_proc_bvec, eddy_output_bvec)
    
    output_img = copy.deepcopy(input_dwi)
    output_img.filename = eddy_output_img
    output_img.bvec     = eddy_output_bvec
    
    if struct_img:
        os.system('mri_convert ' + output_img.filename + ' --reslice_like ' + struct_img.filename + ' ' + output_img.filename )

    #shutil.rmtree(proc_dir)
    
    return output_img
