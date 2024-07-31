#!/usr/bin/env python
import os, shutil, subprocess, json, paramiko
from glob import glob

#Neuroimaging Modules
import pydicom as dcm
import numpy as np
import nibabel as nib

def create_index_acqparam_files(dcm_file, output_dwi, output_index, output_acqparams):

    dcmData = dcm.read_file(dcm_file)
    echoSpacing = dcmData[0x0043, 0x102c].value
    phaseEncodeDir = dcmData[0x0043, 0x108a].value
    assetFactor = dcmData[0x0043, 0x1083].value

    #Grab PEPOLAR from the series description
    series_description = dcmData[0x0008,0x103e].value
    pepolar_flag = series_description.split('_')[len(series_description.split('_'))-1]
    pepolar = int(pepolar_flag[len(pepolar_flag)-1])

    nii = nib.load(output_dwi)
    xDim = nii.header.get_fdata_shape()[0]
    numImages = nii.header.get_fdata_shape()[3]

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

def dicom_to_nifti(dcm_dir, output_img, method="dcm2niix", nthreads=1, **keywords):

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    output_dir = os.path.dirname(output_img)
    tmp_dir = output_dir+'/tmp/'

    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)

    filetype = 0
    raw_data = ""

    if dcm_dir.endswith(".tgz"):
        filetype = 1
        raw_data = dcm_dir+".tgz"
    elif os.path.exists(dcm_dir+".tgz"):
        filetype = 1
        raw_data = dcm_dir+".tgz"
    else:
        for file in os.listdir(dcm_dir):
            if file.endswith('.tgz'):
                filetype = 1
                raw_data = os.path.join(dcm_dir, file)
            if file.endswith('.bz2'):
                filetype = 2
                raw_data = os.path.join(dcm_dir, file)
            if file.endswith('.dcm'):
                filetype = 3
                raw_data = os.path.join(dcm_dir, file)

    if filetype == 1:
        os.system('tar -xf ' + raw_data + ' -C ' + tmp_dir)
    elif filetype == 2:
        os.system('ls ' + tmp_dir + '/*.bz2 | xargs -P'+nthreads+' -I{} bash -c "bunzip2 {}"')
        os.system('ls ' + dcm_dir + '/*.bz2 | xargs -P'+nthreads+' -I{} bash -c "rsync -a {} ' + tmp_dir + '/"')
    elif filetype == 3:
        os.system('rsync -a ' + dcm_dir + '/*.dcm ' + tmp_dir)
    else:
        print('Unknown DICOM filetype!')
        exit(-1)

    dicom_imgs = os.listdir(tmp_dir)
    tmp_img = tmp_dir + '/' + dicom_imgs[0]
    
    ds = dcm.read_file(tmp_img)
    tr = ds.RepetitionTime
    flip = ds.FlipAngle
    ti=""
    n=""
    pe=""

    try:
        ti = ds.InversionTime
    except:
        print('No Inversion Time')

    try:
        n = ds[0x0043,0x1038].value[3]
    except:
        print('No TR Ratio')

    if pe == "":
        pe = ds.Rows


    output_filename=''
    output_basename=''
    extension = ''
    if output_img.endswith('.nii.gz'):
        output_basename=os.path.splitext(os.path.splitext(os.path.basename(output_img))[0])[0]
        output_filename = os.path.splitext(os.path.splitext(output_img)[0])[0]
        extension = 'nii.gz'
    else:
        output_basename=os.path.splitext(os.path.basename(output_img))[0]
        output_filename = os.path.splitext(output_img)[0]
        extension = os.path.splitext(output_img)[1]

    CMD=""
    if method == "dcm2niix":
        CMD="dcm2niix -b y -m 1 -z y -w 1 -x i -f " + output_basename + " -o " + output_dir + " " + tmp_dir
    elif method == "dcm2nii":
        CMD="dcm2nii -g Y -r N " + tmp_dir+"/*"
    elif method =="mrtrix":
        CMD="mrconvert -quiet -force -json_export " + output_filename+".json -export_grad_fsl " + output_filename+".bvecs " + output_filename+".bvals " + tmp_dir + " " + output_filename+".nii.gz" 
    elif method=="mri-convert":
        src_dcms = glob(tmp_dir+'/*')
        CMD="mri_convert -i " + src_dcms[0] + " -o " + output_img
    else:
        print("Invalid DICOM conversion method!")
        exit(-1)

    print(CMD)
    subprocess.run([CMD], shell=True, stderr=subprocess.STDOUT)

    output_img = ''
    for file in os.listdir(output_dir):
        if(output_filename in file):
            file_ext = os.path.splitext(file)[1]
            if('_t' in file and file_ext != '.json'):
                os.rename(output_dir+'/'+file, output_dir+'/'+output_filename+'.'+extension)
            if('_t' in file and file_ext == '.json'):
                os.rename(output_dir+'/'+file, output_dir+'/'+output_filename+'.json')

    # if (output_index!=None) and (output_acqparams!=None):
    #     create_index_acqparam_files(src_dcms[0], output_img, output_index, output_acqparams)

    shutil.rmtree(tmp_dir)

    return tr,flip,pe,ti,n

def nifti_to_ras(input_img, input_json='', input_bvals='', input_bvecs=''):

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


if __name__ == '__main__':
   
   import argparse
   
   parser = argparse.ArgumentParser(description='QMRI-Neuropipe DICOM To NIFTI Conversion')
   
   parser.add_argument('--dcm_dir',
                    type=str,
                    help="Input image to be apply gibbs ringing correction",
                    default=None)
   
   parser.add_argument('--out',
                       type=str,
                       help="Output image",
                       default=None)
   
   parser.add_argument('--method',
                       type=str,
                       help="Tool to use for dicom conversion",
                       choices=["dcm2niix", "dcm2nii", "mrtrix", "mri-convert"],
                       default="dcm2niix")
   
   parser.add_argument("--nthreads",
                       type=int,
                       help="Number of threads",
                       default=1)
   
   parser.add_argument("--debug",
                       type=bool,
                       help="Print debugging statements",
                       default=False)
   
   args, unknown = parser.parse_known_args()

   dicom_to_nifti(dcm_dir       = args.dcm_dir, 
                  output_img    = args.out, 
                  method        = args.method, 
                  nthreads      = args.nthreads)
   
