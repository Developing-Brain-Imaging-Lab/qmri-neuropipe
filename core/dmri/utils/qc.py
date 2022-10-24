import string, os, sys, subprocess, shutil, time, json, copy
from glob import glob

import nibabel as nib
import numpy as np
from scipy.io import loadmat

from dipy.segment.mask import median_otsu
from dipy.denoise.nlmeans import nlmeans
from dipy.denoise.noise_estimate import estimate_sigma
from dipy.io import read_bvals_bvecs
from dipy.io.bvectxt import reorient_vectors

from core.utils.io import Image, DWImage
import core.utils.mask as mask
import core.utils.tools as img_tools

#from PNGViewer import PNGViewer

def merge_phase_encodes(DWI_pepolar0, DWI_pepolar1, output_base):

    DWI_out = DWImage(file       = output_base + '_dwi.nii.gz',
                      bvals      = output_base + '_dwi.bval',
                      bvecs      = output_base + '_dwi.bvec',
                      index      = output_base + '_desc-Index_dwi.txt',
                      acqparams  = output_base + '_desc-Acqparams_dwi.txt',
                      json       = output_base + '_dwi.json')

    #First, get the size of the images
    dwi_pepolar0 = nib.load(DWI_pepolar0._get_filename())
    dwi_pepolar1 = nib.load(DWI_pepolar1._get_filename())

    bvals_pepolar0, bvecs_pepolar0 = read_bvals_bvecs(DWI_pepolar0._get_bvals(), DWI_pepolar0._get_bvecs())
    bvals_pepolar1, bvecs_pepolar1 = read_bvals_bvecs(DWI_pepolar1._get_bvals(), DWI_pepolar1._get_bvecs())

    nImages_pepolar0 = dwi_pepolar0.header.get_data_shape()[3]
    nImages_pepolar1 = dwi_pepolar1.header.get_data_shape()[3]

    if bvals_pepolar0.shape[0] != nImages_pepolar0:
        indices_to_remove = np.arange(nImages_pepolar0, bvals_pepolar0.shape[0])
        bvals_pepolar0 = np.delete(bvals_pepolar0, indices_to_remove)
        bvecs_pepolar0 = np.delete(bvecs_pepolar0, indices_to_remove, 0)

    if bvals_pepolar1.shape[0] != nImages_pepolar1:
        indices_to_remove = np.arange(nImages_pepolar1, bvals_pepolar1.shape[0])
        bvals_pepolar1 = np.delete(bvals_pepolar1, indices_to_remove)
        bvecs_pepolar1 = np.delete(bvecs_pepolar1, indices_to_remove, 0)

    #Read in the DWI ACQPARAMS FILE, DETERMINE WHICH IMAGES CORRESPOND TO UP AND DOWN, AND MERGE INTO SEPARATE FILES
    img_tools.merge_images([DWI_pepolar0, DWI_pepolar1], DWI_out._get_filename())
    bvals = np.concatenate((bvals_pepolar0, bvals_pepolar1), axis=0)
    bvecs = np.concatenate((bvecs_pepolar0, bvecs_pepolar1), axis=0)

    acqparams = np.empty([2,4])
    acqparams_list = [DWI_pepolar0._get_json(), DWI_pepolar1._get_json()]

    for i in range(0,len(acqparams_list)):
        with open(acqparams_list[i]) as f:
            dwi_json = json.load(f)

            phase_encode_dir = ''
            try:
                phase_encode_dir = dwi_json["PhaseEncodingDirection"]
            except KeyError:
                try:
                    phase_encode_dir = dwi_json["PhaseEncodingAxis"]
                except KeyError:
                    print('No phase encoding direction information')
                    exit()

            if(phase_encode_dir == 'i'):
                acqparams[i] = np.array(['1', '0', '0', str(dwi_json["TotalReadoutTime"])])
            elif(phase_encode_dir == 'i-'):
                acqparams[i] = np.array(['-1', '0', '0', str(dwi_json["TotalReadoutTime"])])
            elif(phase_encode_dir== 'j'):
                acqparams[i] = np.array(['0', '1', '0', str(dwi_json["TotalReadoutTime"])])
            elif(phase_encode_dir == 'j-'):
                acqparams[i] = np.array(['0', '-1', '0', str(dwi_json["TotalReadoutTime"])])


    np.savetxt(DWI_out._get_index(), np.concatenate((np.ones(nImages_pepolar0, dtype=int), 2*np.ones(nImages_pepolar1, dtype=int)), axis=0), fmt='%i', newline = ' ')
    np.savetxt(DWI_out._get_acqparams(), acqparams, delimiter = ' ', fmt='%s')
    np.savetxt(DWI_out._get_bvals(), bvals, fmt='%i', newline=' ')
    np.savetxt(DWI_out._get_bvecs(), bvecs.transpose(), fmt='%.8f')
    shutil.copy2(DWI_pepolar0._get_json(), DWI_out._get_json())

    return DWI_out


def check_bvals_bvecs(input_dwi, output_base=None):

    dwi_img = nib.load(input_dwi._get_filename())
    bvals,bvecs=read_bvals_bvecs(input_dwi._get_bvals(),input_dwi._get_bvecs())
    numberOfVolumes = dwi_img.header.get_data_shape()[3]
    numberOfSlices  = dwi_img.header.get_data_shape()[2]

    if bvals.shape[0] != numberOfVolumes:
        indices_to_remove = np.arange(numberOfVolumes, bvals.shape[0])
        bvals = np.delete(bvals, indices_to_remove)
        bvecs = np.delete(bvecs, indices_to_remove, 0)

    if output_base == None:
        np.savetxt(input_dwi._get_bvals(), bvals, fmt='%i', newline=' ')
    else:
        input_dwi._set_bvals(output_base + '_dwi.bval')
        np.savetxt(output_base + '_dwi.bval', bvals, fmt='%i', newline=' ')

    if output_base == None:
        np.savetxt(input_dwi._get_bvecs(), np.transpose(bvecs), fmt='%.5f')
    else:
        input_dwi._set_bvecs(output_base + '_dwi.bvec')
        np.savetxt(output_base + '_dwi.bvec', np.transpose(bvecs), fmt='%.5f')


def check_gradient_directions(input_dwi, nthreads=1):

    dir = os.path.dirname(os.path.abspath(input_dwi._get_filename()))

    tmp_mask  = Image(file=dir+'/mask.nii.gz')
    tmp_bvals = dir+'/tmp_bvals.bval'
    tmp_bvecs = dir+'/tmp_bvecs.bvec'

    mask.mask_image(input_dwi, tmp_mask, method='bet', bet_options='-f 0.25',)
    subprocess.run(['dwigradcheck',
                    '-force',
                    '-quiet',
                    '-mask', tmp_mask._get_filename(),
                    '-fslgrad',input_dwi._get_bvecs(), input_dwi._get_bvals(),
                    '-export_grad_fsl',tmp_bvecs, tmp_bvals,
                    '-nthreads', str(nthreads), input_dwi._get_filename()], stderr=subprocess.STDOUT)

    bvals,bvecs=read_bvals_bvecs(input_dwi._get_bvals(),tmp_bvecs)
    np.savetxt(input_dwi._get_bvals(), bvals, fmt='%i', newline=' ')
    np.savetxt(input_dwi._get_bvecs(), np.transpose(bvecs), fmt='%.5f')

    os.system('rm -rf ' + tmp_mask._get_filename() + ' ' + tmp_bvals + ' ' + tmp_bvecs)


#def rotate_bvecs(input_bvecs, output_bvecs, transform, linreg_method):
#
#    #Rotate bvecs
#    trans   = loadmat(transform)
#    matrix  = ''
#    if linreg_method == 'FSL':
#        matrix = trans['MatrixOffsetTransformBase_double_3_3'][:9].reshape((3,3))
#    elif linreg_method == 'ANTS':
#        matrix = trans['AffineTransform_double_3_3'][:9].reshape((3,3))
#
#    bvecs = np.genfromtxt(input_bvecs)
#    if bvecs.shape[0] != 3:
#        bvecs = bvecs.T
#
#    newbvecs = np.dot(matrix, bvecs)
#    np.savetxt(output_bvecs, newbvecs, fmt='%.5f')


def rotate_bvecs(input_img, ref_img, output_bvec, transform, nthreads=1):

    output_dir = os.path.dirname(output_bvec)

    #Convert image to mif
    tmp_img = output_dir + '/input_img.mif'
    os.system('mrconvert -force -quiet -fslgrad ' + input_img._get_bvecs() + ' ' + input_img._get_bvals() + ' ' + input_img._get_filename() + ' ' + tmp_img + ' -nthreads ' + str(nthreads))

    mrtrix_xfm = output_dir + '/mrtrix_xfm.txt'
    os.system('transformconvert -force -quiet ' + transform + '  itk_import ' + mrtrix_xfm )

    warped_img = output_dir + '/warped_img.mif'
    os.system('mrtransform -force -quiet -linear ' + mrtrix_xfm + ' -template ' + ref_img._get_filename() + ' -reorient_fod no ' + ' -strides ' + ref_img._get_filename() + ' ' + tmp_img + ' ' + warped_img)

    output_bval = output_dir + '/bval.bval'
    output_img  = output_dir + '/dwi.nii.gz'

    os.system('mrconvert -force -quiet -export_grad_fsl ' + output_bvec + ' ' + output_bval + ' ' + warped_img + ' ' + output_img)

    os.remove(tmp_img)
    os.remove(mrtrix_xfm)
    os.remove(warped_img)
    os.remove(output_bval)
    os.remove(output_img)


def create_index_acqparam_files(input_dwi, output_base):

    dwi_img = nib.load(input_dwi._get_filename())
    numberOfVolumes = dwi_img.header.get_data_shape()[3]

    with open(input_dwi._get_json()) as f:
        dwi_json = json.load(f)

    phase_encode_dir = ''
    try:
        phase_encode_dir = dwi_json["PhaseEncodingDirection"]
    except KeyError:
        try:
            phase_encode_dir = dwi_json["PhaseEncodingAxis"]
        except KeyError:
            print('No phase encoding direction information')
            exit()


    acqparams = np.empty(4)
    if(phase_encode_dir == 'i'):
        try:
            acqparams = np.array(['1', '0', '0', str(dwi_json["TotalReadoutTime"])])
        except: KeyError

        try:
            acqparams = np.array(['1', '0', '0', str(dwi_json["EffectiveEchoSpacing"]*(dwi_img.header.get_data_shape()[1]-1))])
        except: KeyError

    elif(phase_encode_dir == 'i-'):
        try:
            acqparams = np.array(['-1', '0', '0', str(dwi_json["TotalReadoutTime"])])
        except: KeyError

        try:
            acqparams = np.array(['-1', '0', '0', str(dwi_json["EffectiveEchoSpacing"]*(dwi_img.header.get_data_shape()[1]-1))])
        except: KeyError

    elif(phase_encode_dir == 'j'):

        try:
            acqparams = np.array(['0', '1', '0', str(dwi_json["TotalReadoutTime"])])
        except: KeyError

        try:
            acqparams = np.array(['0', '1', '0', str(dwi_json["EffectiveEchoSpacing"]*(dwi_img.header.get_data_shape()[2]-1))])
        except: KeyError

    elif(phase_encode_dir == 'j-'):
        try:
            acqparams = np.array(['0', '-1', '0', str(dwi_json["TotalReadoutTime"])])
        except: KeyError

        try:
            acqparams = np.array(['0', '-1', '0', str(dwi_json["EffectiveEchoSpacing"]*(dwi_img.header.get_data_shape()[2]-1))])
        except: KeyError


    acqparams_file = output_base+'_desc-Acqparams_dwi.txt'
    index_file     = output_base+'_desc-Index_dwi.txt'

    np.savetxt(index_file, np.ones(numberOfVolumes, dtype=int), fmt='%i', newline = ' ')
    f=open(index_file,'a')
    f.write('\n')
    f.close()

    np.savetxt(acqparams_file, acqparams, delimiter = ' ', fmt='%s', newline = ' ')
    f=open(acqparams_file,'a')
    f.write('\n')
    f.close()

    return index_file, acqparams_file

def create_slspec_file(input_dwi, output_base):

    from scipy.stats import rankdata
    slspec_file = output_base+'_desc-Slspec_dwi.txt'

    with open(input_dwi._get_json()) as f:
        dwi_json = json.load(f)

        try:
            slice_times       = dwi_json["SliceTiming"]
            sorted_slicetimes = np.sort(slice_times)
            sorted_indices    = np.argsort(slice_times)
            mb                = int(len(sorted_slicetimes)/(np.sum(np.diff(sorted_slicetimes)!=0)+1));
            slspec            = np.reshape(sorted_indices,[int(len(sorted_indices)/mb), mb]);

        except KeyError:
            print('WARNING: Creating default Slice Timing file...Please check to ensure correct')
            img = nib.load(input_dwi._get_filename())
            even = np.arange(0, img.shape[2], 2)
            odd  = np.arange(1, img.shape[2], 2)
            slspec = np.concatenate((even, odd), axis=0)

        np.savetxt(slspec_file, slspec, fmt='%s')

    return slspec_file

# def manually_review_dwi(input_dwi, manual_corr_dir, output_file):
#     if os.path.exists(manual_corr_dir):
#         shutil.rmtree(manual_corr_dir)
#
#     os.mkdir(manual_corr_dir)
#
#     #First split the DWIs into individual volumes
#     os.system('fslsplit ' + input_dwi + ' ' + manual_corr_dir + '/img_ -t')
#
#     for nii in glob(manual_corr_dir + '*.nii*'):
#         basename = nii.split('/')[len(nii.split('/'))-1]
#         slice = basename.split('.')[0]
#         outputPNG = manual_corr_dir + slice + '.png'
#         os.system('slicer ' + nii + ' -L -a ' + outputPNG)
#
#     #Run the manual correction
#     png_viewer = PNGViewer(manual_corr_dir, subject_id)
#     png_viewer.runPNGViewer()
#
#     try:
#         input('Please press enter after reviewing DWIs...')
#     except SyntaxError:
#         pass
#
#     png_viewer.cleanupURL()
#     os.system('mv ~/Downloads/Unknown* ' + output_file)
#     shutil.rmtree(manual_corr_dir)

def remove_outlier_imgs(input_dwi, output_base, output_removed_imgs_dir, mask_img=None, method='Threshold', percent_threshold=0.1, input_topup_field=None, manual_report_dir=None):

    output_img = copy.deepcopy(input_dwi)
    output_img._set_filename(output_base + '_desc-OutlierRemoved_dwi.nii.gz' )
    output_img._set_bvals(output_base + '_desc-OutlierRemoved_dwi.bval')
    output_img._set_bvecs(output_base + '_desc-OutlierRemoved_dwi.bvec')
    output_img._set_index(output_base + '_desc-OutlierRemoved-Index_dwi.txt')

    #Now, correct the DWI data
    dwi_img         = nib.load(input_dwi._get_filename())
    bvals, bvecs    = read_bvals_bvecs(input_dwi._get_bvals(), input_dwi._get_bvecs())
    index           = np.loadtxt(input_dwi._get_index())
    aff             = dwi_img.get_affine()
    sform           = dwi_img.get_sform()
    qform           = dwi_img.get_qform()
    dwi_data        = dwi_img.get_data()

    numberOfVolumes = dwi_img.shape[3]

    if method == 'Manual':
        #Read the manual report
        imgs_to_remove = np.fromfile(manual_report_dir+'/imgs_to_remove.txt', sep=' ')
        vols_to_remove = []
        for img in imgs_to_remove:
            vols_to_remove.append(int(img))

        vols_to_keep = np.delete(np.arange(numberOfVolumes),vols_to_remove).flatten()

    else:

        eddy_output_basename = input_dwi._get_filename().split('.')[0]
        input_report_file = eddy_output_basename+'.eddy_outlier_map'

        report_data = np.loadtxt(input_report_file, skiprows=1) #Skip the first row in the file as it contains text information

        if method == 'Threshold':
            numberOfSlices = report_data.shape[1] #Calculate the number of slices per volume

            #Calculate the threshold at which we will deem acceptable/unacceptable.
            threshold=np.round(float(percent_threshold)*numberOfSlices)
            sum_data = np.sum(report_data, axis=1);
            badVols = sum_data>=threshold
            goodVols=sum_data<threshold
            vols_to_remove = np.asarray(np.where(badVols)).flatten()
            vols_to_keep = np.asarray(np.where(goodVols)).flatten()

        elif method == 'EDDY-QUAD':
            if os.path.exists(output_removed_imgs_dir+'/eddy-qc/'):
                os.system('rm -rf ' + output_removed_imgs_dir+'/eddy-qc/')

            if mask_img==None:
                mask_img = Image(file = os.path.dirname(output_base) + '/mask.nii.gz')
                mask.mask_image(input_dwi, mask_img, method='bet')

            eddy_quad_cmd = 'eddy_quad ' + eddy_output_basename \
                            + ' -idx ' + input_dwi._get_index() \
                            + ' -par ' + input_dwi._get_acqparams() \
                            + ' -m '   + mask_img._get_filename() \
                            + ' -b '   + input_dwi._get_bvals() \
                            + ' -g '   + input_dwi._get_bvecs() \
                            + ' -o '   + output_removed_imgs_dir + '/eddy-qc/'

            if input_topup_field != None:
                eddy_quad_cmd += ' -f ' + input_topup_field

            os.system(eddy_quad_cmd)
            vols_to_keep = np.loadtxt(output_removed_imgs_dir+'/eddy-qc/vols_no_outliers.txt')
            vols_to_remove = sorted(list(set(range(0, numberOfVolumes)) - set(vols_to_keep)))

    #Remove the DWIs, Bvals, Bvecs from the files
    data_to_keep  = np.delete(dwi_data, vols_to_remove, 3)
    bvals_to_keep = np.delete(bvals, vols_to_remove)
    bvecs_to_keep = np.delete(bvecs, vols_to_remove, 0)
    index_to_keep = np.delete(index, vols_to_remove)

    data_to_remove= dwi_data[:,:,:,vols_to_remove]
    bvals_to_remove = bvals[vols_to_remove,]

    ##Write the bvals, bvecs, index, and corrected image data
    np.savetxt(output_img._get_index(), index_to_keep, fmt='%i')
    np.savetxt(output_img._get_bvals(), bvals_to_keep, fmt='%i')
    np.savetxt(output_img._get_bvecs(), np.transpose(bvecs_to_keep), fmt='%.5f')

    corr_img = nib.Nifti1Image(data_to_keep.astype(np.float32), aff, dwi_img.header)
    corr_img.set_sform(sform)
    corr_img.set_qform(qform)
    nib.save(corr_img , output_img._get_filename())

    if len(vols_to_remove) != 0:
        if not os.path.exists(output_removed_imgs_dir):
            os.mkdir(output_removed_imgs_dir)

        imgs_to_remove= nib.Nifti1Image(data_to_remove.astype(np.float32), aff, dwi_img.header)
        imgs_to_remove.set_sform(sform)
        imgs_to_remove.set_qform(qform)
        nib.save(imgs_to_remove, output_removed_imgs_dir+'/RemovedImages.nii.gz')
        np.savetxt(output_removed_imgs_dir+'/bvals_removed.txt', bvals_to_remove, fmt='%i', newline=" ")
        np.savetxt(output_removed_imgs_dir+'/volumes_removed.txt', vols_to_remove, fmt='%i', newline=" ")

    if os.path.exists(os.path.dirname(output_base) + '/mask.nii.gz'):
        os.remove(os.path.dirname(output_base) + '/mask.nii.gz')

    return output_img


# def reorient_dwi_imgs(input_dwi, input_bval, input_bvec, output_dwi, output_bval, output_bvec, new_x, new_y, new_z, new_r, new_a, new_s):
#
#     os.system('fslswapdim ' + input_dwi + ' ' + new_x + ' ' + new_y + ' ' + new_z + ' ' + output_dwi)
#
#     #Now reorient the bvecs
#     bvals, bvecs = read_bvals_bvecs(input_bval, input_bvec)
#
#     new_orient = new_r+new_a+new_s
#     r_bvecs = reorient_vectors(bvecs, 'ras', new_orient, axis=1)
#
#     N = len(bvals)
#     fmt = '   %e' * N + '\n'
#
#     open(output_bval, 'wt').write(fmt % tuple(bvals))
#
#     bvf = open(output_bvec, 'wt')
#     for dim_vals in r_bvecs.T:
#         bvf.write(fmt % tuple(dim_vals))
#     bvf.close()
#
# def reorient_bvecs(input_bvecs, output_bvecs, new_x, new_y, new_z):
#
#     bvecs = np.loadtxt(input_bvecs)
#     permute = np.array([0, 1, 2])
#
#     if new_x[0] == "x" and new_y[0] == "z" and new_z[0] == "y":
#         permute = np.array([0, 2, 1])
#     elif new_x[0] == "y" and new_y[0] == "x" and new_z[0] == "z":
#         permute = np.array([1, 0, 2])
#     elif new_x[0]== "y" and new_y[0] == "z" and new_z[0] == "x":
#         permute = np.array([1, 2, 0])
#     elif new_x[0] == "z" and new_y[0] == "y" and new_z[0] == "x":
#         permute = np.array([2, 1, 0])
#     elif new_x[0] == "z" and new_y[0] == "x" and new_z[0] == "y":
#         permute = np.array([2, 0, 1])
#
#     new_bvecs = np.empty(bvecs.shape)
#     new_bvecs[0] = bvecs[permute[0]]
#     new_bvecs[1] = bvecs[permute[1]]
#     new_bvecs[2] = bvecs[permute[2]]
#
#
#     if len(new_x) == 2 and new_x[1] == "-":
#         new_bvecs[0] = -1.00*new_bvecs[0]
#     if len(new_y) == 2 and new_y[1] == "-":
#         new_bvecs[1] = -1.00*new_bvecs[1]
#     if len(new_z) == 2 and new_z[1] == "-":
#         new_bvecs[2] = -1.00*new_bvecs[2]
#
#     np.savetxt(output_bvecs, new_bvecs, fmt='%.10f')
#
# def convert_bvals_bvecs_to_fsl(input_bval_file, input_bvec_file, output_bval_file, output_bvec_file):
#     input_bval = open(input_bval_file).read().splitlines()
#     input_bvec = open(input_bvec_file).read().splitlines()
#
#     number_of_volumes = len(input_bval)
#
#     bvals = np.empty([number_of_volumes, 1])
#     bvecs = np.empty([number_of_volumes, 3])
#
#     for i in range(0,len(input_bval)):
#         bvals[i] = int(float(input_bval[i].split(" ")[2]))
#
#         bvecs[i,0] = float(input_bvec[i].split(" ")[2])
#         bvecs[i,1] = float(input_bvec[i].split(" ")[3])
#         bvecs[i,2] = float(input_bvec[i].split(" ")[4])
#
#     np.savetxt(output_bval_file, bvals, fmt='%i')
#     np.savetxt(output_bvec_file, np.transpose(bvecs), fmt='%.5f')
#
# def create_pseudoT1_img(fa_img, fiso_img, mask_img, pseudoT1_img):
#
#     base_dir = os.path.dirname(pseudoT1_img)
#     if not os.path.exists(base_dir):
#         os.makedirs(base_dir)
#
#     segment_img = base_dir+'/segment.nii.gz'
#     prob_img = base_dir+'/prob.nii.gz'
#
#     os.system('Atropos -d 3 -a '+fa_img+' -i KMeans[2] -o ['+segment_img+','+ prob_img+'] -x '+ mask_img)
#
#     gm_fraction_img = base_dir+'/gm_fraction.nii.gz'
#     #Ensure Mask in binary
#     os.system('fslmaths ' + mask_img + ' -fillh -bin ' + ' -sub ' + prob_img + ' -sub ' + fiso_img + ' ' + gm_fraction_img)
#     os.system('fslmaths ' + prob_img + ' -mul 2.00 -add ' + gm_fraction_img + ' ' + pseudoT1_img)
