import string, os, sys, subprocess, shutil, time

if sys.platform == 'linux2':
    fitmcmd_exe = os.path.dirname(__file__)+'/bin/linux/fitmcmicro'
    fitmicro_exe = os.path.dirname(__file__)+'/bin/linux/fitmicrodt'
else:
    fitmcmd_exe = os.path.dirname(__file__)+'/bin/mac/fitmcmicro'
    fitmicro_exe = os.path.dirname(__file__)+'/bin/mac/fitmicrodt'

def fit_mcmd_model(input_dwi, input_bval, input_bvec, input_mask, output_dir):
    
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    tmp_output_image = output_dir + 'tmp.nii.gz'
    command = fitmcmd_exe + ' ' + input_dwi + ' ' + tmp_output_image + ' --bvals ' + input_bval + ' --bvecs ' + input_bvec + ' --mask ' + input_mask
    os.system(command)

    tmp_dir=output_dir + '/tmp/'
    tmp_basename = tmp_dir + 'img_'
    os.mkdir(tmp_dir)
    os.system('fslsplit ' + tmp_output_image + ' ' + tmp_basename + ' -t')

    mcmd_intra = output_dir + '/mcmd_INTRA.nii.gz'
    mcmd_diff = output_dir + '/mcmd_DIFF.nii.gz'
    mcmd_extratrans = output_dir + '/mcmd_EXTRATRANS.nii.gz'
    mcmd_extramd = output_dir + '/mcmd_EXTRAMD.nii.gz'

    shutil.copy2(tmp_basename+'0000.nii.gz', mcmd_intra)
    shutil.copy2(tmp_basename+'0001.nii.gz', mcmd_diff)
    shutil.copy2(tmp_basename+'0002.nii.gz', mcmd_extratrans)
    shutil.copy2(tmp_basename+'0003.nii.gz', mcmd_extramd)

    shutil.rmtree(tmp_dir)
    os.remove(tmp_output_image)


def fit_microdt_model(input_dwi, input_bval, input_bvec, input_mask, output_dir):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    tmp_output_image = output_dir + 'tmp.nii.gz'
    command = fitmicro_exe + ' ' + input_dwi + ' ' + tmp_output_image + ' --bvals ' + input_bval + ' --bvecs ' + input_bvec + ' --mask ' + input_mask
    os.system(command)

    tmp_dir=output_dir + '/tmp/'
    tmp_basename = tmp_dir + 'img_'
    os.mkdir(tmp_dir)
    os.system('fslsplit ' + tmp_output_image + ' ' + tmp_basename + ' -t')
    
    micro_long = output_dir + '/micro_LONG.nii.gz'
    micro_trans = output_dir + '/micro_TRANS.nii.gz'
    micro_fa = output_dir + '/micro_FA.nii.gz'
    micro_fapow = output_dir + '/micro_faPow3.nii.gz'
    micro_md = output_dir + '/micro_MD.nii.gz'
    
    shutil.copy2(tmp_basename+'0000.nii.gz', micro_long)
    shutil.copy2(tmp_basename+'0001.nii.gz', micro_trans)
    shutil.copy2(tmp_basename+'0002.nii.gz', micro_fa)
    shutil.copy2(tmp_basename+'0003.nii.gz', micro_fapow)
    shutil.copy2(tmp_basename+'0004.nii.gz', micro_md)
    
    shutil.rmtree(tmp_dir)
    os.remove(tmp_output_image)
