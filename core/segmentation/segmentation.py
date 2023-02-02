import string, os, sys, subprocess, shutil, time, copy

from core.utils.io import Image
import core.registration.registration as reg_tools
import core.utils.mask as mask_tools


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
    os.system('fslmaths ' + output_dir + '/atropos/atropos_seg.nii.gz -thr 1.9 -uthr 2.1 -bin ' + WM_Seg._get_filename() )
    
    reg_tools.linear_reg(input_img      = atlas,
                         reference_img  = target_img,
                         output_matrix  = output_dir + '/flirt_atlas2target.mat',
                         method         = 'FSL',
                         dof            = 12,
                         flirt_options =  ' -cost normmi -interp sinc -searchrx -180 180 -searchry -180 180 -searchrz -180 180 -usesqform')

    bbr_options = ' -cost bbr -wmseg ' + WM_Seg._get_filename() + ' -schedule $FSLDIR/etc/flirtsch/bbr.sch -interp sinc -bbrtype global_abs -bbrslope 0.25 -finesearch 18 -init ' + output_dir + '/flirt_atlas2target.mat -usesqform'

    coreg_atlas = Image(file = output_dir + '/atlas_coreg.nii.gz')
    reg_tools.linear_reg(input_img     = atlas,
                        reference_img  = target_img,
                        output_matrix  = output_dir + '/flirt_atlas2target.mat',
                        output_file    = coreg_atlas._get_filename(),
                        method         = 'FSL',
                        dof            = 12,
                        flirt_options =  bbr_options)

    fslmat_ants = output_dir + '/flirt_atlas2target.txt'
    reg_tools. convert_fsl2ants(atlas, target_img, output_dir + '/flirt_atlas2target.mat', fslmat_ants)

    reg_tools.nonlinear_reg(input_img       = coreg_atlas,
                            reference_img   = target_img,
                            reference_mask  = target_mask,
                            output_base     = output_prefix,
                            nthreads        = nthreads,
                            method          = 'ANTS-QUICK')

    #Create overall warp file
    composite_transform = output_dir + '/atlas_segmentation_warp.nii.gz'

    reg_tools.create_composite_transform(reference_img  = target_img,
                                         output_file    = composite_transform,
                                         transforms     = [output_prefix + '1Warp.nii.gz', output_prefix + '0GenericAffine.mat', fslmat_ants])

    #Warp labels to target
    reg_tools.apply_transform(input_img     = label,
                              reference_img = target_img,
                              output_file   = output_seg_file,
                              matrix        = composite_transform,
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

        reg_tools.linear_reg(input_img      = Image(file=atlases[i]),
                             reference_img  = target_img,
                             output_matrix  = atlas_dir + '/flirt_atlas2target.mat',
                             method         = 'FSL',
                             dof            = 12,
                             flirt_options =  ' -cost normmi -interp sinc -searchrx -180 180 -searchry -180 180 -searchrz -180 180 -usesqform')

        bbr_options = ' -cost bbr -wmseg ' + WM_Seg._get_filename() + ' -schedule $FSLDIR/etc/flirtsch/bbr.sch -interp sinc -bbrtype global_abs -bbrslope 0.25 -finesearch 18 -init ' + atlas_dir + '/flirt_atlas2target.mat -usesqform'

        coreg_atlas = Image(file = atlas_dir + '/atlas_coreg.nii.gz')
        reg_tools.linear_reg(input_img     = Image(file=atlases[i]),
                             reference_img  = target_img,
                             output_matrix  = atlas_dir + '/flirt_atlas2target.mat',
                             output_file    = coreg_atlas._get_filename(),
                             method         = 'FSL',
                             dof            = 12,
                             flirt_options =  bbr_options)
        
        fslmat_ants = atlas_dir + '/flirt_atlas2target.txt'
        reg_tools. convert_fsl2ants(Image(file=atlases[i]), target_img, atlas_dir + '/flirt_atlas2target.mat', fslmat_ants)
        
        coreg_labels = Image(file = atlas_dir +'/coreg_atlas_labels.nii.gz')
        
        #Warp labels to target
        reg_tools.apply_transform(input_img     = Image(labels[i]),
                                  reference_img = target_img,
                                  output_file   = coreg_labels._get_filename(),
                                  matrix        = fslmat_ants,
                                  nthreads      = nthreads,
                                  method        = 'ANTS',
                                  ants_options  = '-n GenericLabel')
        
        coreg_atlases_list.append(coreg_atlas._get_filename())
        coreg_labels_list.append(coreg_labels._get_filename())



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


def create_wmseg(input_img, output_dir, brain_mask=None, modality='t1w'):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    if not brain_mask:
        brain_mask = Image(file = output_dir +'/brain_mask.nii.gz')
        mask_tools.mask_image(input_img   = input_img,
                              output_mask = brain_mask,
                              method='hd-bet')
                              
    os.system('N4BiasFieldCorrection -d 3 -i ' + input_img._get_filename() + ' -o ' + output_dir +'/bias_corr.nii.gz -x ' + brain_mask._get_filename() )
    os.system('Atropos -d 3 -a ' + output_dir +'/bias_corr.nii.gz -x ' + brain_mask._get_filename() + ' -i \'KMeans[3]\' -o ' + output_dir + '/atropos_seg.nii.gz')
    
    wmseg_img = Image(output_dir + '/wmseg.nii.gz')
    
    if modality='t2w':
        os.system('fslmaths ' + output_dir + '/atropos_seg.nii.gz -thr 0.9 -uthr 1.1 -bin ' + wmseg_img._get_filename() )
    else:
        os.system('fslmaths ' + output_dir + '/atropos_seg.nii.gz -thr 2.9 -uthr 3.1 -bin ' + wmseg_img._get_filename() )
    
    return wmseg_img
