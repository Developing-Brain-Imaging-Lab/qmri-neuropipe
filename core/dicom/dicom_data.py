import string, os, sys, subprocess, shutil, time, json, paramiko
from glob import glob

#Neuroimaging Modules
import pydicom as dcm
import numpy as np
import nibabel as nib

def create_index_acqparam_files(dcm_file, output_dwi, output_index, output_acqparams):

    dcmData = dicom.read_file(dcm_file)
    echoSpacing = dcmData[0x0043, 0x102c].value
    phaseEncodeDir = dcmData[0x0043, 0x108a].value
    assetFactor = dcmData[0x0043, 0x1083].value

    #Grab PEPOLAR from the series description
    series_description = dcmData[0x0008,0x103e].value
    pepolar_flag = series_description.split('_')[len(series_description.split('_'))-1]
    pepolar = int(pepolar_flag[len(pepolar_flag)-1])

    nii = nib.load(output_dwi)
    xDim = nii.header.get_data_shape()[0]
    numImages = nii.header.get_data_shape()[3]

    #Only if parallel imaging is turned on...
    acqFourthColumn = float(assetFactor[0])*float(xDim)*(0.001)*float(float(echoSpacing)/1000)

    indexFile = open(output_index, 'w')
    acqFile = open(output_acqparams, 'w')

    for i in range(int(numImages)):
        indexFile.write('1 ')

    if 'COL' in phaseEncodeDir:
        if int(pepolar) == 0:
            acqFile.write('0 1 0 ' + str(acqFourthColumn) + '\n')
        else:
            acqFile.write('0 -1 0 ' + str(acqFourthColumn) + '\n')
    else:
        if int(pepolar) == 0:
            acqFile.write('1 0 0 ' + str(acqFourthColumn) + '\n')
        else:
            acqFile.write('-1 0 0 ' + str(acqFourthColumn) + '\n')

    indexFile.close()
    acqFile.close()


def dicom_to_nifti_mri_convert(dwi_dcm_dir, output_dwi, output_index='', output_acqparams=''):

    src_dcms = glob(dwi_dcm_dir+'/*')
    os.system('mri_convert -i ' + src_dcms[0] + ' -o ' + output_dwi)

    if (output_index!='') and (output_acqparams!=''):
        create_index_acqparam_files(src_dcms[0], output_dwi, output_index, output_acqparams)

def dicom_to_nifti_dcm2nii(dcm_dir, output_img, **keywords):

    index=''
    acqparams=''
    params = ''
    type = ''
    pe = ''

    if 'params' in keywords:
        params = keywords['params']
    if 'type' in keywords:
        type = keywords['type']
    if 'pe' in keywords:
        pe = keywords['pe']
    if 'index' in keywords:
        index = keywords['index']
    if 'acqparams' in keywords:
        acqparams = keywords['acqparams']

    output_root, img = os.path.split(output_img)

    tmp_dir = output_root+'/tmp/'
    os.makedirs(tmp_dir)

    filetype = 0
    raw_data = ''

    for file in os.listdir(dcm_dir):
        if file.endswith('.tgz'):
            filetype = 1;
            raw_data = dcm_dir + '/' + file
        if file.endswith('.bz2'):
            filetype = 2;
            raw_data = dcm_dir + '/' + file
        if file.endswith('.dcm'):
            filetype = 3;
            raw_data = dcm_dir + '/' + file

    if filetype == 1:
        os.system('tar -xf ' + raw_data + ' -C ' + tmp_dir)

    elif filetype == 2:
        os.system('ls ' + dcm_dir + '/*.bz2 | xargs -P8 -I{} bash -c "rsync -a {} ' + tmp_dir + '/"')
        os.system('ls ' + tmp_dir + '/*.bz2 | xargs -P8 -I{} bash -c "bunzip2 {}"')

    elif filetype == 3:
        os.system('rsync -a ' + dcm_dir + '/*.dcm ' + tmp_dir)

    else:
        print('Unknown DICOM filetype!')
        exit(-1);

    dicom_imgs = os.listdir(tmp_dir)
    tmp_img = tmp_dir + '/' + dicom_imgs[0]

    ds = dcm.read_file(tmp_img)
    tr = ds.RepetitionTime
    fa = ds.FlipAngle
    ti=''
    n=''

    try:
        ti = ds.InversionTime
    except:
        print('No Inversion Time')

    try:
        n = ds[0x0043,0x1038].value[3]
    except:
        print('No TR Ratio')

    if pe == '':
        pe = ds.Rows

    #Write out the acquisition params
    if params != '':
        f = open(params, 'w')
        f.write(str(tr)+'\n')
        if type == 'irspgr':
            f.write(str(ti)+'\n')
            f.write(str(fa)+'\n')
            f.write(str(pe)+'\n')
        f.close()

    output_filename=''
    extension = ''
    if img.endswith('.nii.gz'):
        output_filename = os.path.splitext(os.path.splitext(img)[0])[0]
        extension = 'nii.gz'
    else:
        output_filename = os.path.splitext(img)[0]
        extension = os.path.splitext(img)[1]

    os.chdir(tmp_dir)
    os.system('dcm2nii -g Y -r N *')
    os.system('mv *.nii.gz ' + output_img)
    os.system('rm -rf ' + tmp_dir)

    output_img = ''
    for file in os.listdir(output_root):
        if(output_filename in file):
            if('_t' in file):
                os.rename(output_root+'/'+file, output_root+'/'+output_filename+'.'+extension)

    return tr,fa,pe,ti,n

def dicom_to_nifti_dcm2niix(dcm_dir, output_img, output_json='', **keywords):

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    index=''
    acqparams=''
    params = ''
    type = ''
    pe = ''

    if 'params' in keywords:
        params = keywords['params']
    if 'type' in keywords:
        type = keywords['type']
    if 'pe' in keywords:
        pe = keywords['pe']
    if 'index' in keywords:
        index = keywords['index']
    if 'acqparams' in keywords:
        acqparams = keywords['acqparams']

    output_root, img = os.path.split(output_img)
    tmp_dir = output_root+'/tmp/'
    os.system('mkdir ' + tmp_dir)

    filetype = 0
    raw_data = ''

    for file in os.listdir(dcm_dir):
        if file.endswith('.tgz'):
            filetype = 1;
            raw_data = dcm_dir + '/' + file
        if file.endswith('.bz2'):
            filetype = 2;
            raw_data = dcm_dir + '/' + file
        if file.endswith('.dcm'):
            filetype = 3;
            raw_data = dcm_dir + '/' + file

    if filetype == 1:
        os.system('tar -xf ' + raw_data + ' -C ' + tmp_dir)

    elif filetype == 2:
        os.system('ls ' + dcm_dir + '/*.bz2 | xargs -P8 -I{} bash -c "rsync -a {} ' + tmp_dir + '/"')
        os.system('ls ' + tmp_dir + '/*.bz2 | xargs -P8 -I{} bash -c "bunzip2 {}"')

    elif filetype == 3:
        os.system('rsync -a ' + dcm_dir + '/*.dcm ' + tmp_dir)

    else:
        print('Unknown DICOM filetype!')
        exit(-1);

    #Use mri_convert to extract the data
    dicom_imgs = os.listdir(tmp_dir)
    tmp_img = tmp_dir + '/' + dicom_imgs[0]
    
    ds = dcm.read_file(tmp_img)
    tr = ds.RepetitionTime
    fa = ds.FlipAngle
    ti=''
    n=''

    try:
        ti = ds.InversionTime
    except:
        print('No Inversion Time')

    try:
        n = ds[0x0043,0x1038].value[3]
    except:
        print('No TR Ratio')

    if pe == '':
        pe = ds.Rows

    #Write out the acquisition params
    if params != '':
        f = open(params, 'w')
        f.write(str(tr)+'\n')
        if type == 'irspgr':
            f.write(str(ti)+'\n')
            f.write(str(fa)+'\n')
            f.write(str(pe)+'\n')
        f.close()

    output_filename=''
    extension = ''
    if img.endswith('.nii.gz'):
        output_filename = os.path.splitext(os.path.splitext(img)[0])[0]
        extension = 'nii.gz'
    else:
        output_filename = os.path.splitext(img)[0]
        extension = os.path.splitext(img)[1]

    os.system('dcm2niix -b y -m 1 -z y -w 1 -x i -f ' + output_filename + ' -o ' + output_root + ' ' + tmp_dir)
    os.system('rm -rf ' + tmp_dir)

    output_img = ''
    for file in os.listdir(output_root):
        if(output_filename in file):
            file_ext = os.path.splitext(file)[1]
            if('_t' in file and file_ext != '.json'):
                os.rename(output_root+'/'+file, output_root+'/'+output_filename+'.'+extension)
            if('_t' in file and file_ext == '.json'):
                os.rename(output_root+'/'+file, output_root+'/'+output_filename+'.json')


    return tr,fa,pe,ti,n

def nifti_to_ras(input_img, input_json = '', input_bvals='', input_bvecs=''):

    if input_json != '':
        #Need to adjust phase encoding direction to match new orientation
        mrinfo_result = subprocess.check_output(["mrinfo", input_img], stderr=subprocess.STDOUT)

        transform = np.array([[0, 0, 0],
                              [0, 0, 0],
                              [0, 0, 0]])

        for line in str(mrinfo_result).split('\\n'):
            if 'Data strides' in line:
                tmp = line.split('  ')[len(line.split('  '))-1].split(' ')
                data_strides = tmp[1:4]

        for i in range(0,len(data_strides)):
            transform[i,abs(int(data_strides[i]))-1] = 1
            if int(data_strides[i]) < 0:
                transform[i,abs(int(data_strides[i]))-1] = -1

        #Read json and get phase encode
        with open(input_json) as f:
            json_file = json.load(f)

        if(json_file["PhaseEncodingDirection"] == 'i'):
            pe_vec = np.array([1, 0, 0])
        elif(json_file["PhaseEncodingDirection"] == 'i-'):
            pe_vec = np.array([-1, 0, 0])
        elif(json_file["PhaseEncodingDirection"] == 'j'):
            pe_vec = np.array([0, 1, 0])
        elif(json_file["PhaseEncodingDirection"] == 'j-'):
            pe_vec = np.array([0, -1, 0])

        new_pe_vec = np.matmul(transform, pe_vec)

        if( (new_pe_vec == [1,0,0]).all() ):
            json_file["PhaseEncodingDirection"] = 'i'
        elif( (new_pe_vec == [-1,0,0]).all() ):
            json_file["PhaseEncodingDirection"] = 'i-'
        elif( (new_pe_vec == [0,1,0]).all() ):
            json_file["PhaseEncodingDirection"] = 'j'
        elif( (new_pe_vec == [0,-1,0]).all() ):
            json_file["PhaseEncodingDirection"] = 'j-'

        with open(input_json, "w") as outfile:
            json.dump(json_file, outfile)


    if input_bvals != '':
        tmp_img = os.path.dirname(input_img) + '/tmp.mif'
        os.system('mrconvert -force -quiet -fslgrad ' + input_bvecs + ' ' + input_bvals + ' ' + input_img + ' ' + tmp_img)
        os.system('mrconvert -force -quiet -strides -1,2,3,4 -export_grad_fsl ' + input_bvecs + ' ' + input_bvals + ' ' + tmp_img + ' ' + input_img)

    else:
        os.system('mrconvert -force -quiet -strides -1,2,3,4 ' + input_img + ' ' + input_img)
