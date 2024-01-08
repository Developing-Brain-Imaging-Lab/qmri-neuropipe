import string, os, sys, subprocess, shutil, time, copy

from core.utils.io import Image
import core.utils.mask as mask_tools

from core.registration.linreg import linreg
from core.registration.nonlinreg import nonlinreg
from core.registration.apply_transform import apply_transform
from core.registration.convert_fsl2ants import convert_fsl2ants
from core.registration.create_composite_transform import create_composite_transform



def atlas_segmentation(target_img, target_mask, atlas, label, output_seg_file, nthreads=1, verbose=False):

    #Coregister the atlas to the target image
    output_dir = os.path.dirname(output_seg_file)
    output_prefix = output_dir + '/atlas_segmentation_'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    #First, use FLIRT to run linear registration
    ants_atropos(input_img    = target_img,
                 brain_mask   = target_mask,
                 output_dir   = output_dir + '/atropos/')

    #Create WM seg image
    WM_Seg = Image(output_dir + '/atropos/atropos_WM.nii.gz')
    os.system('fslmaths ' + output_dir + '/atropos/atropos_seg.nii.gz -thr 1.9 -uthr 2.1 -bin ' + WM_Seg.filename )
    
    linreg(input        = atlas,
           ref          = target_img,
           out_mat      = output_dir + '/flirt_atlas2target.mat',
           method       = 'fsl',
           dof          = 12,
          flirt_options =  ' -cost normmi -interp sinc -searchrx -180 180 -searchry -180 180 -searchrz -180 180 -usesqform')

    bbr_options = ' -cost bbr -wmseg ' + WM_Seg.filename + ' -schedule $FSLDIR/etc/flirtsch/bbr.sch -interp sinc -bbrtype global_abs -bbrslope 0.25 -finesearch 18 -init ' + output_dir + '/flirt_atlas2target.mat -usesqform'

    coreg_atlas = Image(file = output_dir + '/atlas_coreg.nii.gz')
    linreg(input         = atlas,
           ref           = target_img,
           out_mat       = output_dir + '/flirt_atlas2target.mat',
           out           = coreg_atlas,
           method        = 'fsl',
           dof           = 12,
           flirt_options = bbr_options)

    fslmat_ants = output_dir + '/flirt_atlas2target.txt'
    convert_fsl2ants(input    = atlas,
                     ref      = target_img,
                     fsl_mat  = output_dir + '/flirt_atlas2target.mat',
                     ants_mat = fslmat_ants )

    nonlinreg(input    = coreg_atlas,
              ref      = target_img,
              mask     = target_mask,
              out_xfm  = output_prefix,
              nthreads = nthreads,
              method   = 'ants-quick')

    #Create overall warp file
    composite_transform = output_dir + '/atlas_segmentation_warp.nii.gz'
    create_composite_transform(ref        = target_img,
                               out        = composite_transform,
                               transforms = [output_prefix + '1Warp.nii.gz', output_prefix + '0GenericAffine.mat', fslmat_ants])

    #Warp labels to target
    apply_transform(input         = label,
                    ref           = target_img,
                    out           = output_seg_file,
                    transform     = composite_transform,
                    nthreads      = nthreads,
                    method        = 'ANTS',
                    ants_options  = '-n GenericLabel')

    os.system('rm -rf ' + output_prefix + '*')
    os.system('rm -rf ' + output_dir + '/atropos')

def multi_atlas_segmentation(target_img, atlases, labels, output_seg_file, nthreads=1, verbose=False):

    output_dir = os.path.dirname(output_seg_file)
    output_prefix = output_dir + '/ants_multi_seg'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    atlases_dir = output_dir + '/coreg_multi_seg_atlases/'
    if not os.path.exists(atlases_dir):
        os.makedirs(atlases_dir)

    coreg_atlases_list=[]
    coreg_labels_list=[]

    #First, run TWO-STAGE FLIRT between ATLAS and TARGET to get good initial alignment and then run ANTSJoingLabelFusion
    for i in range(0,len(atlases)):

        if verbose:
            print('Running FSL FLIRT to initially align Atlases to Target Image')

        atlas_dir = atlases_dir + str(i) + '/'
        if not os.path.exists(atlas_dir):
            os.makedirs(atlas_dir)        

        #First, use FLIRT to run linear registration
        ants_atropos(input_img    = target_img,
                    brain_mask   = target_mask,
                    output_dir   = output_dir + '/atropos/')

        WM_Seg = Image(output_dir + '/atropos/atropos_WM.nii.gz')
        os.system('fslmaths ' + output_dir + '/atropos/atropos_seg.nii.gz -thr 1.9 -uthr 2.1 -bin ' + WM_Seg._get_filename() )

        linreg(input         = Image(file=atlases[i]),
               ref           = target_img,
               out_mat       = atlas_dir + '/flirt_atlas2target.mat',
               method        = 'fsl',
               dof           = 12,
               flirt_options =  ' -cost normmi -interp sinc -searchrx -180 180 -searchry -180 180 -searchrz -180 180 -usesqform')

        bbr_options = ' -cost bbr -wmseg ' + WM_Seg._get_filename() + ' -schedule $FSLDIR/etc/flirtsch/bbr.sch -interp sinc -bbrtype global_abs -bbrslope 0.25 -finesearch 18 -init ' + atlas_dir + '/flirt_atlas2target.mat -usesqform'

        coreg_atlas = Image(file = atlas_dir + '/atlas_coreg.nii.gz')
        linreg(input         = Image(file=atlases[i]),
               ref           = target_img,
               out_mat       = atlas_dir + '/flirt_atlas2target.mat',
               out           = coreg_atlas,
               method        = 'FSL',
               dof           = 12,
               flirt_options =  bbr_options)
        
        fslmat_ants = atlas_dir + '/flirt_atlas2target.txt'
        convert_fsl2ants(input    = Image(file=atlases[i]),
                         ref      = target_img,
                         fsl_mat  = atlas_dir + '/flirt_atlas2target.mat',
                         ants_mat = fslmat_ants )

        coreg_labels = Image(file = atlas_dir +'/coreg_atlas_labels.nii.gz')
        
        #Warp labels to target
        apply_transform(input         = Image(labels[i]),
                        ref           = target_img,
                        out           = coreg_labels,
                        transform     = fslmat_ants,
                        nthreads      = nthreads,
                        method        = 'ants',
                        ants_options  = '-n GenericLabel')
        
        
        coreg_atlases_list.append(coreg_atlas.filename)
        coreg_labels_list.append(coreg_labels.filename)

    cmd = 'antsJointLabelFusion.sh -d 3 -j ' + str(nthreads)+ ' -t ' + target_img._get_filename() + ' -o ' + output_prefix

    for i in range(0,len(coreg_atlases_list)):
        cmd += ' -g ' + coreg_atlases_list[i] + ' -l ' + coreg_labels_list[i]

    os.system(cmd)
    os.rename(output_prefix+'Labels.nii.gz', output_seg_file)
    
    os.system('rm -rf ' + output_prefix + '*')
    os.system('rm -rf ' + output_dir + '/atropos')


def fsl_fast(input_img, output_dir, fast_options=None):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    os.system('fast ' + fast_options + ' -o ' + output_dir+'/fast ' + input_img._get_filename())


def ants_atropos(input_img, output_dir, brain_mask=None,  atropos_options='-i \'KMeans[3]\''):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    if not brain_mask:
        brain_mask = Image(file = output_dir +'/brain_mask.nii.gz')
        os.system('fslmaths ' + input_img._get_filename() + ' -thr 10 -bin ' + brain_mask._get_filename())

    os.system('Atropos -d 3 -a ' + input_img._get_filename() + ' -x ' + brain_mask._get_filename() + ' ' + atropos_options + ' -o ' + output_dir + '/atropos_seg.nii.gz')


def create_wmseg(input_img, output_dir, nthreads=1):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    output_seg = os.path.join(output_dir, "seg.nii.gz")
    os.system("mri_synthseg --i " + input_img.filename + " --o " + output_seg + " --robust --cpu --threads " + str(nthreads))

    output_left_wm      = os.path.join(output_dir, "left_wm.nii.gz")
    output_left_cb_wm   = os.path.join(output_dir, "left_cb_wm.nii.gz")
    output_right_wm     = os.path.join(output_dir, "right_wm.nii.gz")
    output_right_cb_wm  = os.path.join(output_dir, "right_cb_wm.nii.gz")

    os.system("fslmaths " + output_seg + " -thr 1.9 -uthr 2.1 -bin " + output_left_wm)
    os.system("fslmaths " + output_seg + " -thr 6.9 -uthr 7.1 -bin " + output_left_cb_wm)
    os.system("fslmaths " + output_seg + " -thr 40.9 -uthr 41.1 -bin " + output_right_wm)
    os.system("fslmaths " + output_seg + " -thr 45.9 -uthr 46.1 -bin " + output_right_cb_wm)

    wmseg_img = Image(os.path.join(output_dir, "wmseg.nii.gz"))
    os.system("fslmaths " + output_left_wm + " -add " + output_left_cb_wm + " -add " + output_right_wm + " -add " + output_right_cb_wm + " -bin " + wmseg_img.filename)
        
    
    return wmseg_img
