import os,sys, shutil, json, argparse, copy
from distutils.util import strtobool

#import core.pipelines.diffusion as dwi_pipe
#from core.pipelines.anatomical import AnatomicalPrepPipeline
#from core.pipelines.diffusion import DiffusionProcessingPipeline

parser = argparse.ArgumentParser(description='Waisman Center Processing for Quantitative MRI Data in BIDS format')

parser.add_argument('--load_json',
                    type=str, help='Load settings from file in json format. Command line options are overriden by values in file.', default=None)

parser.add_argument('--anat_preproc_pipeline',
                    type=bool,
                    help='Preprocess the Anataomical Imaging Data',
                    default=False)

parser.add_argument('--dwi_proc_pipeline',
                    type=bool,
                    help='Process Diffusion Imaging Data',
                    default=False)

parser.add_argument('--verbose',
                    type=bool,
                    help='Print out information meassages and progress status',
                    default=False)

args, unknown = parser.parse_known_args()

if args.load_json:
    with open(args.load_json, 'rt') as f:
        t_args = argparse.Namespace()
        t_dict = vars(t_args)
        t_dict.update(json.load(f))
        args, unknown = parser.parse_known_args(namespace=t_args)


##################################
##################################
##### PROCESSING STARTS HERE #####
##################################
##################################


print(args.anat_preproc_pipeline)
print(args.dwi_proc_pipeline)

print(type(args.anat_preproc_pipeline))
print(type(args.dwi_proc_pipeline))

if (args.anat_preproc_pipeline):
    print("TEST!")

exit()

#if args.anat_preproc_pipeline:
#    anat_pipeline = AnatomicalPrepPipeline()
#    anat_pipeline.run()
#
#
#if args.dwi_proc_pipeline:
#    dwi_pipeline = DiffusionProcessingPipeline()
#    dwi_pipeline.run()
