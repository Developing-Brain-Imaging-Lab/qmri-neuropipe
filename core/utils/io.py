import string, os, sys, subprocess, shutil, time
from glob import glob
import nibabel as nib

class Image:
    def __init__(self, file=None, json=None):
        self._inputs = {}
        self._inputs['filename']    = file
        self._inputs['json']        = json

    def __copy__(self):
        return Image(self)

    def exists(self):
        try:
            if os.path.exists(self._inputs['filename']):
                return True
            else:
                return False
        except AttributeError:
            return False

    def _set_filename(self, file):
        self._inputs['filename']    = file

    def _set_json(self, json):
        self._inputs['json']        = json

    def _get_filename(self):
        return self._inputs['filename']

    def _get_json(self):
        return self._inputs['json']

class DWImage(Image):

    def __init__(self, file=None, bvals=None, bvecs=None, json=None, index=None, acqparams=None, slspec=None):
        self._inputs = {}
        self._inputs['filename']    = file
        self._inputs['bvals']       = bvals
        self._inputs['bvecs']       = bvecs
        self._inputs['json']        = json
        self._inputs['index']       = index
        self._inputs['acqparams']   = acqparams
        self._inputs['slspec']      = slspec

    def __copy__(self):
        return DWImage(self)

    def copy_image(self, dwi2copy, datatype=False):
        shutil.copy2(dwi2copy._get_bvals(), self._get_bvals())
        shutil.copy2(dwi2copy._get_bvecs(), self._get_bvecs())
        shutil.copy2(dwi2copy._get_index(), self._get_index())
        shutil.copy2(dwi2copy._get_acqparams(), self._get_acqparams())
        shutil.copy2(dwi2copy._get_slspec(), self._get_slspec())

        if datatype != False:
            out_img = nib.load(dwi2copy._get_filename())
            out_img.set_data_dtype(datatype)
            out_img.to_filename(self._get_filename())
        else:
            shutil.copy2(dwi2copy._get_filename(), self._get_filename())


    def exists(self):
        try:
            if os.path.exists(self._inputs['filename']) and os.path.exists(self._inputs['bvals']) and os.path.exists(self._inputs['bvecs']):
                return True
            else:
                return False
        except AttributeError:
            return False

    def _set_members(self, file=None, bvals=None, bvecs=None, json=None, index=None, acqparams=None, slspec=None):
        self._inputs['filename']    = file
        self._inputs['bvals']       = bvals
        self._inputs['bvecs']       = bvecs
        self._inputs['json']        = json
        self._inputs['index']       = index
        self._inputs['acqparams']   = acqparams
        self._inputs['slspec']      = slspec

    def _set_bvals(self, bvals):
        self._inputs['bvals']       = bvals

    def _set_bvecs(self, bvecs):
        self._inputs['bvecs']       = bvecs

    def _set_index(self, index):
        self._inputs['index']       = index

    def _set_acqparams(self, acqparams):
        self._inputs['acqparams']   = acqparams

    def _set_slspec(self, slspec):
        self._inputs['slspec']   = slspec

    def _get_bvals(self):
        return self._inputs['bvals']

    def _get_bvecs(self):
        return self._inputs['bvecs']

    def _get_index(self):
        return self._inputs['index']

    def _get_acqparams(self):
        return self._inputs['acqparams']

    def _get_slspec(self):
        return self._inputs['slspec']
