import string, os, sys, subprocess, shutil, time, json, argparse
import pandas
import numpy as np
from glob import glob

class Subject(object):
    def __init__(self, id, rawdata_id, pepolar0_dcm_dir, pepolar1_dcm_dir):
        self.id                 = str(id)
        self.rawdata_id         = str(rawdata_id)
        self.pepolar0_dcm_dir   = str(pepolar0_dcm_dir)
        self.pepolar1_dcm_dir   = str(pepolar1_dcm_dir)

    def __str__(self):
        return("Subject:\n"
               "  ID = {0}\n"
               "  RAW_ID = {1}\n"
               "  DWI_RAWDATA_PEPOLAR0_DIR = {2}\n"
               "  DWI_RAWDATA_PEPOLAR1_DIR = {3}\n"
               .format(self.id, self.rawdata_id, self.pepolar0_dcm_dir, self.pepolar1_dcm_dir))

def create_database(workbook_file):
    workbook = pandas.read_excel(workbook_file, sheet_name=0, engine='openpyxl')

    ids             = workbook['ID']
    rawdata_ids     = workbook['RAWDATA_ID']
    
    pepolar0_dir    = workbook['DWI_RAWDATA_PEPOLAR0_DIR']
    pepolar1_dir    = workbook['DWI_RAWDATA_PEPOLAR1_DIR']
        

    database = []
    for i in range(0,len(ids)):
        subject = Subject(id                = ids[i],
                          rawdata_id        = rawdata_ids[i],
                          pepolar0_dcm_dir  = pepolar0_dir[i],
                          pepolar1_dcm_dir  = pepolar1_dir[i])
        database.append(subject)

    return database



parser = argparse.ArgumentParser(description='Waisman Center Processing for Quantitative MRI Data in BIDS format')

parser.add_argument('--load_json',
                    type=str,
                    help='Load settings from file in json format. Command line options are overriden by values in file.',
                    default=None)
                    
parser.add_argument('--bids_dir',
                    type=str,
                    help='Load settings from file in json format. Command line options are overriden by values in file.',
                    default=None)
                    
parser.add_argument('--dwi_pepolar0_bvals',
                    type=str,
                    help='Load settings from file in json format. Command line options are overriden by values in file.',
                    default=None)
                
parser.add_argument('--dwi_pepolar0_bvecs',
                    type=str,
                    help='Load settings from file in json format. Command line options are overriden by values in file.',
                    default=None)
                    
parser.add_argument('--dwi_pepolar1_bvals',
                    type=str,
                    help='Load settings from file in json format. Command line options are overriden by values in file.',
                    default=None)
                    
parser.add_argument('--dwi_pepolar1_bvecs',
                    type=str,
                    help='Load settings from file in json format. Command line options are overriden by values in file.',
                    default=None)

parser.add_argument('--session',
                    type=str,
                    help='Session number to use for data (Optional)',
                    default=None)
                    
parser.add_argument('--workbook_file',
                    type=str,
                    help='Load settings from file in json format. Command line options are overriden by values in file.',
                    default=None)

args, unknown = parser.parse_known_args()


if args.load_json:
    with open(args.load_json, 'rt') as f:
        t_args = argparse.Namespace()
        t_dict = vars(t_args)
        t_dict.update(json.load(f))
        args, unknown = parser.parse_known_args(namespace=t_args)
        
dwi_database = create_database(args.workbook_file)


for i in range(0, len(dwi_database)):

    convert_raw_script = 'python convert_dwi_rawdata.py '
    convert_raw_script += ' --subject=' + str(dwi_database[i].id).zfill(3)
    convert_raw_script += ' --bids_dir=' + args.bids_dir
    
    if args.session:
        convert_raw_script += ' --sesssion=' + str(args.session.zfill(2))
        

    if not pandas.isna(dwi_database[i].pepolar0_dcm_dir):
        convert_raw_script += ' --dwi_dcm_dir='+str(dwi_database[i].pepolar0_dcm_dir)
        convert_raw_script += ' --dwi_bvals='+args.dwi_pepolar0_bvals
        convert_raw_script += ' --dwi_bvecs='+args.dwi_pepolar0_bvecs

    if not dwi_database[i].pepolar1_dcm_dir:
        convert_raw_script += ' --dwi_pepolar1_dcm_dir='+str(dwi_database[i].pepolar1_dcm_dir)
        convert_raw_script += ' --dwi_pepolar1_bvals='+args.dwi_pepolar1_bvals
        convert_raw_script += ' --dwi_pepolar1_bvecs='+args.dwi_pepolar1_bvecs


    print('Working on Subject: ' + str(dwi_database[i].id).zfill(3))
    print()
    print(convert_raw_script)
    #os.system(convert_raw_script)
