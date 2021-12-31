import os,sys, shutil, json, argparse, copy
import nibabel as nib

from bids.layout import writing
from core.utils.io import Image, DWImage
from core.pipelines.anatomical import AnatomicalPrepPipeline

import core.segmentation.segmentation as seg_tools

class SegmentationPipeline:

    def __init__(self, verbose=False):
        self._atlases = {}
        self._multiseg_atlases = {}

        print(os.path.abspath(__file__))

        self._atlases['JHU']                                       = os.path.dirname(os.path.abspath(__file__))+'/../../data/atlases/JHU/'
        self._atlases['IIT']                                       = os.path.dirname(os.path.abspath(__file__))+'/../../data/atlases/IIT/'
        self._atlases['HarvardOxford']                             = os.path.dirname(os.path.abspath(__file__))+'/../../data/atlases/HarvardOxford/'
        self._atlases['JHU-Infant']                                = os.path.dirname(os.path.abspath(__file__))+'/../../data/atlases/JHU-Infant/'
        self._atlases['UNC-Neonate-WMFibers']                      = os.path.dirname(os.path.abspath(__file__))+'/../../data/atlases/UNC-Neonate-WMFibers/'
        self._atlases['UNC-Toddler-WMFibers']                      = os.path.dirname(os.path.abspath(__file__))+'/../../data/atlases/UNC-Toddler-WMFibers/'

        self._multiseg_atlases['Neonate-MCRIB-Parcellation']        = os.path.dirname(os.path.abspath(__file__))+'/../../data/atlases/multi-atlas/neonate/MCRIB-Parcellation/'
        self._multiseg_atlases['Neonate-UNC-HippocampusAmygdala']   = os.path.dirname(os.path.abspath(__file__))+'/../../data/atlases/multi-atlas/neonate/UNC-HippocampusAmygdala/'
        self._multiseg_atlases['Neonate-UNC-Parcellation']          = os.path.dirname(os.path.abspath(__file__))+'/../../data/atlases/multi-atlas/neonate/UNC-Parcellation/'
        self._multiseg_atlases['Neonate-UNC-TissueSegmentation']    = os.path.dirname(os.path.abspath(__file__))+'/../../data/atlases/multi-atlas/neonate/UNC-TissueSegmentation/'
        self._multiseg_atlases['Toddler-UNC-Subcortical']           = os.path.dirname(os.path.abspath(__file__))+'/../../data/atlases/multi-atlas/toddler/UNC-Subcortical/'
        self._multiseg_atlases['Toddler-UNC-TissueSegmentation']    = os.path.dirname(os.path.abspath(__file__))+'/../../data/atlases/multi-atlas/toddler/UNC-TissueSegmentation/'

    def run(self):

        parser = argparse.ArgumentParser()

        parser.add_argument('--bids_dir',
                            type=str,
                            help='BIDS Data Directory')

        parser.add_argument('--bids_rawdata_dir',
                            type=str, help='BIDS RAWDATA Directory',
                            default='rawdata')

        parser.add_argument('--bids_pipeline_name',
                            type=str, help='BIDS PIPELINE Name',
                            default='qmri_prep')

        parser.add_argument('--load_json',
                            type=str, help='Load settings from file in json format. Command line options are overriden by values in file.',
                            default=None)

        parser.add_argument('--subject',
                            type=str,
                            help='Subject ID')

        parser.add_argument('--session',
                            type=str,
                            help='Subject Timepoint',
                            default=None)

        parser.add_argument('--modality',
                            type=str,
                            help='Input Image Modality',
                            default = 't1',
                            choices = ['t1', 't2'])

        parser.add_argument('--atlases',
                            help = 'Atlases to use for segmentation',
                            action = 'append',
                            default = [])

        parser.add_argument('--multi_atlases',
                            help = 'Multi-atlas segmentation',
                            action = 'append',
                            default = [])

        parser.add_argument('--nthreads',
                            type=int,
                            help='Number of Threads',
                            default=1)

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


        #Setup the BIDS Directories and Paths
        entities = {
        'extension': '.nii.gz',
        'subject': args.subject,
        'session': args.session,
        'modality': 'dwi',
        'suffix': 'dwi'
        }

        id_patterns = 'sub-{subject}[_ses-{session}]'
        derivative_patterns = args.bids_dir + '/derivatives/' + args.bids_pipeline_name + '/sub-{subject}[/ses-{session}]/'

        bids_id             = writing.build_path(entities, id_patterns)
        bids_derivative_dir = writing.build_path(entities, derivative_patterns)

        #Create final processed DWI dataset
        final_base = os.path.join(bids_derivative_dir, 'segmentations/')

        #Setup the Anatomical Imaging Data if needed
        anat_pipeline = AnatomicalPrepPipeline()
        t1w, t2w, anat_mask = anat_pipeline.run()

        #Loop through the atlases and multi-atlases and run the segmentations

        #Start with the atlases
        for i in args.atlases:
            target_img = ''
            atlas = Image()
            label = Image(file = self._atlases[i] + '/Seg.nii.gz')

            if t1w != None and args.modality == 't1':
                target_img = t1w
                atlas._set_filename(self._atlases[i] + '/T1.nii.gz')
            elif t1w == None and t2w != None and args.modality == 't1':
                print('WARNING - No T1w image found, using available T2w')
                target_img = t2w
                atlas._set_filename(self._atlases[i] + '/T2.nii.gz')
            elif t2w != None and args.modality == 't2':
                target_img = t2w
                atlas._set_filename(self._atlases[i] + '/T2.nii.gz')
            elif t2w == None and t1w != None and args.modality == 't2':
                print('WARNING - No T2w image found, using available T1w')
                target_img = t1w
                atlas._set_filename(self._atlases[i] + '/T1.nii.gz')
            else:
                print('No Anatomical Image!')
                exit()

            if not os.path.exists(final_base + '/' + i + '/' + bids_id + '_desc-'+i+'_Labels.nii.gz'):
                seg_tools.atlas_segmentation(target_img         = target_img,
                                             atlas              = atlas,
                                             label              = label,
                                             output_seg_file    = final_base + '/' + i + '/' + bids_id + '_desc-'+i+'_Labels.nii.gz',
                                             nthreads           = args.nthreads,
                                             verbose            = args.verbose)

        for i in args.multi_atlases:
            target_img = ''
            atlases = []
            labels = []

            #Collect the Multi-Atlases
            for atlas_dir in os.listdir(self._multiseg_atlases[i]):
                if os.path.isdir(self._multiseg_atlases[i] + '/' + atlas_dir):

                    labels.append(self._multiseg_atlases[i] + '/' + atlas_dir + '/Seg.nii.gz')

                    if t1w != None and args.modality == 't1':
                        target_img = t1w
                        atlases.append(self._multiseg_atlases[i] + '/' + atlas_dir + '/T1.nii.gz')
                    elif t1w == None and t2w != None and args.modality == 't1':
                        print('WARNING - No T1w image found, using available T2w')
                        target_img = t2w
                        atlases.append(self._multiseg_atlases[i] + '/' + atlas_dir + '/T2.nii.gz')
                    elif t2w != None and args.modality == 't2':
                        target_img = t2w
                        atlases.append(self._multiseg_atlases[i] + '/' + atlas_dir + '/T2.nii.gz')
                    elif t2w == None and t1w != None and args.modality == 't2':
                        print('WARNING - No T2w image found, using available T1w')
                        target_img = t1w
                        atlases.append(self._multiseg_atlases[i] + '/' + atlas_dir + '/T1.nii.gz')
                    else:
                        print('No Anatomical Image!')
                        exit()

            #Run the segmentation
            if not os.path.exists(final_base + '/' + i + '/' + bids_id + '_desc-'+i+'_Labels.nii.gz'):
                seg_tools.multi_atlas_segmentation(target_img        = target_img,
                                                   atlases           = atlases,
                                                   labels            = labels,
                                                   output_seg_file   = final_base + '/' + i + '/' + bids_id + '_desc-'+i+'_Labels.nii.gz',
                                                   nthreads          = args.nthreads,
                                                   verbose           = args.verbose)
