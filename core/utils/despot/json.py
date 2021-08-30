import string, os, sys, subprocess, shutil, time, json, copy
from glob import glob

def create_processing_json(despot_json, spgr_img, ssfp_img, irspgr_img=None, afi_img=None):

    despot_data     = {}

    #Create the JSON file used for processing data
    if not os.path.exists(despot_json):

        if spgr_img != None:
            with open(spgr_img._get_json(), 'r+') as spgr_file:
                data = json.load(spgr_file)
                despot_data['SPGR'] = []
                despot_data['SPGR'].append({
                    'RepetitionTime' : data["RepetitionTime"],
                    'EchoTime'  : data["EchoTime"],
                    'FlipAngle' : data["FlipAngle"]
                })


                with open(despot_json, 'w+') as outfile:
                    json.dump(despot_data, outfile, indent=4, sort_keys=True)

        if ssfp_img != None:
            with open(ssfp_img._get_json(), 'r+') as ssfp_file:
                data = json.load(ssfp_file)
                despot_data['SSFP'] = []
                despot_data['SSFP'].append({
                    'RepetitionTime': data["RepetitionTime"],
                    'EchoTime': data["EchoTime"],
                    'FlipAngle': data["FlipAngle"],
                    'PhaseCycling': data["PhaseCycling"],
                    'PhaseAngles': data["PhaseAngles"]
                })
                with open(despot_json, 'w+') as outfile:
                    json.dump(despot_data, outfile, indent=4, sort_keys=True)

        if irspgr_img != None:
            with open(irspgr_img._get_json(), 'r+') as irspgr_file:
                data = json.load(irspgr_file)
                despot_data['IRSPGR'] = []
                despot_data['IRSPGR'].append({
                    'RepetitionTime': data["RepetitionTime"],
                    'EchoTime': data["EchoTime"],
                    'FlipAngle': data["FlipAngle"],
                    'InversionTime': data["InversionTime"],
                    'EchoTrainLength': ((data["PercentPhaseFOV"]/100.00)*(data["AcquisitionMatrixPE"]/2.00))
                })
                with open(despot_json, 'w+') as outfile:
                    json.dump(despot_data, outfile, indent=4, sort_keys=True)
