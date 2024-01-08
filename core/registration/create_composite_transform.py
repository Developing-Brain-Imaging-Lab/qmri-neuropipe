import os, subprocess


def create_composite_transform(ref, out, transforms, linear=False, inverse=0, debug=False):

    CMD=""
    if linear:
        CMD = "antsApplyTransforms -d 3 -o Linear[" + out + ","+str(inverse)+"] -r " + ref.filename
    else:
        CMD = "antsApplyTransforms -d 3 -o [" + out + ","+str(inverse)+"] -r " + ref.filename

    for xfm in transforms:
        CMD += " -t " + xfm


    if debug:
        print("Creating composite transform from list")
        print(CMD)

    subprocess.run([CMD], shell=True, stderr=subprocess.STDOUT)


if __name__ == '__main__':
   
   import argparse

   parser = argparse.ArgumentParser(description='QMRI-Neuropipe Create Composite transformation')
   
                    
 
   parser.add_argument('--ref',
                    type=str,
                    help="Reference image",
                    default=None)
   
   parser.add_argument('--out',
                       type=str,
                       help="Output composite transformation",
                       default=None)
   
   parser.add_argument('--transforms',
                    type=list,
                    help="List of transformations (Need to be in ITK Format)")
   
   parser.add_argument("--debug",
                       type=bool,
                       help="Print debugging statements",
                       default=False)
   
   args, unknown = parser.parse_known_args()
   
   create_composite_transform(ref           = args.ref,
                              out           = args.out,
                              transforms    = args.transforms, 
                              debug         = args.debug)
