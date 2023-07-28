#!/usr/bin/env python
import os

from core.utils.io import Image
import core.utils.biascorrect as biascorr
import core.utils.mask as mask

def compute_synthetic_t2w(input_t1w, output_dir, cmd_args, syn_t2w="synthetic_T2w.nii.gz", t1w_mask=None, debug=False):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    t1w           = input_t1w
    t1w_brain     = Image(filename = os.path.join(output_dir, "t1w_brain.nii.gz"))
    t1w_norm      = Image(filename = os.path.join(output_dir, "t1w_norm.nii.gz"))
    t1w_recip     = Image(filename = os.path.join(output_dir, "t1w_recip.nii.gz"))
    synthetic_t2w = Image(filename = os.path.join(output_dir, syn_t2w))
    skull_img     = Image(filename = os.path.join(output_dir, "skull.nii.gz"))

    if t1w_mask:
        mask.apply_mask(input   = t1w,
                        mask    = t1w_mask,
                        output  = t1w_brain)
    else:
        t1w_mask  = Image(filename  = os.path.join(output_dir, "t1w_mask.nii.gz"))
        mask.mask_image(input       = t1w,
                        mask        = t1w_mask,
                        mask_img    = t1w_brain,
                        method      = cmd_args.anat_mask_method,
                        nthreads    = cmd_args.nthreads,
                        ref_img     = cmd_args.anat_t1w_ants_mask_template,
                        ref_mask    = cmd_args.anat_t1w_ants_mask_template_mask)
                    

    #Create a image of the skull
    os.system("fslmaths " + t1w.filename + " -sub "  + t1w_brain.filename + " " + skull_img.filename)
    os.system("ImageMath 3 " + skull_img.filename + " Normalize " + skull_img.filename)
        
    #Norimalize the T1w
    os.system("ImageMath 3 " + t1w_norm.filename + " Normalize " + t1w_brain.filename)
    os.system("fslmaths " + t1w_norm.filename + " -mas " + t1w_mask.filename + " -recip -nan " +  t1w_recip.filename )
    os.system("ImageMath 3 " + t1w_recip.filename + " Normalize " + t1w_recip.filename)

    
    #Now add the skull and recip-T1w_brain image
    os.system("fslmaths " + t1w_recip.filename + " -add " + skull_img.filename + " " + synthetic_t2w.filename)
    
    
    synthetic_t2w = biascorr.biasfield_correction(input_img     = synthetic_t2w,
                                                  output_file   = synthetic_t2w.filename,
                                                  method        = 'ants',
                                                  iterations    = 3)
    
    os.remove(t1w_brain.filename)
    os.remove(t1w_norm.filename)
    os.remove(t1w_recip.filename)
    os.remove(skull_img.filename)

    return synthetic_t2w

    

if __name__ == '__main__':
   
   import argparse

   parser = argparse.ArgumentParser(description='QMRI-Neuropipe Create Synthetic T2w from T1w')
                    
   parser.add_argument('--t1w',
                    type=str,
                    help="Input T1w image",
                    default=None)
   
   parser.add_argument('--out_dir',
                    type=str,
                    help="Output directory",
                    default=None)
   
   parser.add_argument('--syn_t2w_filename',
                       type=str,
                       help="Synthetic T2w filename",
                       default=None)
   
   parser.add_argument('--t1w_mask',
                    type=str,
                    help="Brain mask for T1w image",
                    default=None)
   
   parser.add_argument("--debug",
                       type=bool,
                       help="Print debugging statements",
                       default=False)
   
   args, unknown = parser.parse_known_args()
   
   compute_synthetic_t2w(input_t1w  = Image(filename=args.t1w),
                         output_dir = args.out_dir,
                         cmd_args   = args,
                         syn_t2w    = args.syn_t2w_filename,  
                         t1w_mask   = Image(filename=args.t1w_mask),
                         debug      = args.debug)
   


