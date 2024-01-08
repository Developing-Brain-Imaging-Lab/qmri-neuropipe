import json
import os
import sys

def create_preproc_bids_dataset_description_json(path=None, bids_pipeline="qmri-neuropipe-preproc"):
    """
    Read the provided BIDS structure and import it into the database.

    :param data     : dictionary object to save into dataset_description.json
     :type data     : dict
    :param path     : path to save dataset_description.json
     :type path     : str
    """
    
    dateset_description = {
    "Name": "QMRI-Neuropipe - Preprocessing of Quantitative MRI Data",
    "BIDSVersion": "1.4.0",
    "DatasetType": "derivative",
    "GeneratedBy": [
        {
            "Name": bids_pipeline,
            "Version": "1.0",
            "CodeURL": "https://github.com/Developing-Brain-Imaging-Lab/qmri-neuropipe"
        }
    ]
    }

    # validate if 'GeneratedBy' key is in the data dict
    # try:
    #     if not 'GeneratedBy' in dateset_description or not dateset_description['GeneratedBy'][0]['Name']:
    #         raise Exception
    # except:
    #     print("\nPlease make sure the data passed into the save_dataset_description_jsonfucntion\n"
    #         "includes key, value and format 'GeneratedBy' : [{'Name': 'PIPELINE NAME'}]\n")
    #     sys.exit(1)

    if not path:
        path = os.getcwd()
    path = path if path.endswith('/') else path + '/'

    with open(path + 'dataset_description.json', 'w') as outfile:
        json.dump(dateset_description, outfile, indent=2)
        
def create_bids_sidecar_json(image, data):
    """
    Read the provided BIDS structure and import it into the database.

    :param data     : dictionary object to save into dataset_description.json
     :type data     : dict
    :param path     : path to save dataset_description.json
     :type path     : str
    :param filename : filename to save sidecar json
     :type filename : str
    """

    if not image.json:
        image.json = image.filename.replace('.nii.gz', '.json')
        

    with open(image.json, 'w+') as outfile:
        json.dump(data, outfile, indent=2)
        
#def update_bids_sidecar_json(image, data)