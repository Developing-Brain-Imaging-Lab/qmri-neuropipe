import os, json
import numpy as np
import nibabel as nib

def create_processing_json(despot_json, spgr_img, ssfp_img, irspgr_img=None, afi_img=None):

    despot_data     = {}

    #Create the JSON file used for processing data
    if not os.path.exists(despot_json):

        if spgr_img != None:

            spgr  = nib.load(spgr_img.filename)
            nspgr = spgr.header.get_data_shape()[-1]

            with open(spgr_img.json, 'r+') as spgr_file:
                data = json.load(spgr_file)
                despot_data['SPGR'] = []
                despot_data['SPGR'].append({
                    'RepetitionTime' : np.repeat(data["RepetitionTime"], nspgr).tolist(),
                    'EchoTime'  : np.repeat(data["EchoTime"], nspgr).tolist(),
                    'FlipAngle' : data["FlipAngle"]
                })

                with open(despot_json, 'w+') as outfile:
                    json.dump(despot_data, outfile, indent=4, sort_keys=True)

        if ssfp_img != None:
            ssfp = nib.load(ssfp_img.filename)
            nssfp = ssfp.header.get_data_shape()[-1]

            with open(ssfp_img.json, 'r+') as ssfp_file:
                data = json.load(ssfp_file)
                despot_data['SSFP'] = []
                despot_data['SSFP'].append({
                    'RepetitionTime': np.repeat(data["RepetitionTime"], nssfp).tolist(),
                    'EchoTime': np.repeat(data["EchoTime"], nssfp).tolist(),
                    'FlipAngle': data["FlipAngle"],
                    'PhaseCycling': data["PhaseCycling"],
                    'PhaseAngles': data["PhaseAngles"]
                })
                with open(despot_json, 'w+') as outfile:
                    json.dump(despot_data, outfile, indent=4, sort_keys=True)

        if irspgr_img != None:
            irspgr = nib.load(irspgr_img.filename)
            nirspgr = irspgr.header.get_data_shape()[-1]

            with open(irspgr_img.json, 'r+') as irspgr_file:
                data = json.load(irspgr_file)
                despot_data['IRSPGR'] = []
                despot_data['IRSPGR'].append({
                    'RepetitionTime': np.repeat(data["RepetitionTime"], nirspgr).tolist(),
                    'EchoTime': np.repeat(data["EchoTime"], nirspgr).tolist(),
                    'FlipAngle': np.repeat(data["FlipAngle"], nirspgr).tolist(),
                    'InversionTime': data["InversionTime"],
                    'EchoTrainLength': np.repeat(((data["PercentPhaseFOV"]/100.00)*(data["AcquisitionMatrixPE"]/2.00)), nirspgr).tolist()
                })
                with open(despot_json, 'w+') as outfile:
                    json.dump(despot_data, outfile, indent=4, sort_keys=True)
