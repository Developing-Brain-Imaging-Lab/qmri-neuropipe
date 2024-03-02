import os, shutil
import nibabel as nib

class Image:
    def __init__(self, filename=None, json=None):
        self.filename = filename
        self.json     = json

        if filename:
            self.img_dir  = os.path.dirname(filename)
        else:
            self.img_dir  = None

    def __copy__(self):
        return Image(self)

    def exists(self):
        try:
            if os.path.exists(self.filename):
                return True
            else:
                return False
        except AttributeError:
            return False
    
    def copy_image(self, img2copy, datatype=False):
        
        if img2copy.json != None and os.path.exists(img2copy.json):
            if self.json != None:
                self.json = self.filename.replace('.nii.gz', '.json')
            
            shutil.copy2(img2copy.json, self.json)
       
        if datatype != False:
            out_img = nib.load(img2copy.filename)
            out_img.set_data_dtype(datatype)
            out_img.to_filename(self.filename)
        else:
            shutil.copy2(img2copy.filename, self.filename)

    def remove(self):
        if self.exists():
            os.remove(self.filename)
            if self.json != None:
                os.remove(self.json)

        
    def get_type(self):
        return "Image"
        
        

class DWImage(Image):

    def __init__(self, filename=None, bvals=None, bvecs=None, json=None, index=None, acqparams=None, slspec=None):
        Image.__init__(self, filename, json)
        self.bvals      = bvals        
        self.bvecs      = bvecs
        self.acqparams  = acqparams
        self.slspec     = slspec
        self.index      = index

    def __copy__(self):
        return DWImage(self)

    def copy_image(self, dwi2copy, datatype=False):
        shutil.copy2(dwi2copy.bvals, self.bvals)
        shutil.copy2(dwi2copy.bvecs, self.bvecs)
        shutil.copy2(dwi2copy.index, self.index)
        shutil.copy2(dwi2copy.acqparams, self.acqparams)
        shutil.copy2(dwi2copy.slspec, self.slspec)

        if datatype != False:
            out_img = nib.load(dwi2copy.filename)
            out_img.set_data_dtype(datatype)
            out_img.to_filename(self.filename)
        else:
            shutil.copy2(dwi2copy.filename, self.filename)

    def exists(self):
        try:
            if os.path.exists(self.filename) and os.path.exists(self.bvals) and os.path.exists(self.bvecs):
                return True
            else:
                return False
        except AttributeError:
            return False
    
    def remove(self):
        if self.exists():
            os.remove(self.filename)
            os.remove(self.bvals)
            os.remove(self.bvecs)

            if self.index != None:
                os.remove(self.index)
            if self.acqparams != None:
                os.remove(self.acqparams)
            if self.slspec != None:
                os.remove(self.slspec)
            if self.json != None:
                os.remove(self.json)
    
    def get_type(self):
        return "DWImage"
