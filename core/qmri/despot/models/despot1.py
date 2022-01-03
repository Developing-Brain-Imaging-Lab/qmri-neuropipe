import os, sys, subprocess, shutil, time

#GLOBAL EXECUTABLES HERE
mcd_dir, file = os.path.split(__file__)
platform = sys.platform

if "linux" in platform:
    despot1_exe = mcd_dir + '/bin/linux/despot1 '
else:
    despot1_exe = mcd_dir + "/bin/mac/despot1 "

class DESPOT1_Model():
    def __init__(self, spgr_img, params, out_dir, b1=None, irspgr_img=None, out_base=None, model='DESPOT1', fit_algorithm='Ceres', mask=None, logfile=None, nthreads=1, verbose=False):
        self._inputs = {}
        self._inputs['spgr_img']    = spgr_img
        self._inputs['irspgr_img']  = irspgr_img
        self._inputs['b1']          = b1
        self._inputs['spgr_img']    = spgr_img
        self._inputs['model']       = model
        self._inputs['fit_params']  = params
        self._inputs['out_dir']     = out_dir
        self._inputs['out_base']    = out_base
        self._inputs['algo']        = fit_algorithm
        self._inputs['mask']        = mask
        self._inputs['nthreads']    = nthreads
        self._inputs['logfile']     = logfile
        self._inputs['verbose']     = verbose

    def set_b1(self, b1):
        self._inputs['b1']          = b1
    def set_model(self, model):
        self._inputs['model']       = model

    def fit(self):

        despot1_cmd = despot1_exe \
                    + ' --spgr=' + self._inputs['spgr_img']._get_filename() \
                    + ' --params=' + self._inputs['fit_params'] \
                    + ' --out_dir='+ self._inputs['out_dir'] \
                    + ' --algo=' + self._inputs['algo'] \
                    + ' --threads=' + str(self._inputs['nthreads'])

        if self._inputs['mask'] != None:
            despot1_cmd += ' --mask=' + self._inputs['mask']._get_filename()

        if self._inputs['model'] == 'HIFI':
            despot1_cmd += ' --irspgr=' + self._inputs['irspgr_img']._get_filename()
        elif self._inputs['model'] == 'DESPOT1':
            despot1_cmd += ' --b1='+self._inputs['b1']._get_filename()
        else:
            print('Need to specify a B1-Map or IR-SPGR for HIFI')
            exit()

        if self._inputs['out_base'] != None:
            despot1_cmd+= ' --out_base='+self._inputs['out_base']
        if self._inputs['verbose']:
            despot1_cmd += ' -v'

        if self._inputs['logfile'] != None:
            if not os.path.exists(os.path.dirname(self._inputs['logfile'])):
                os.makedirs(os.path.dirname(self._inputs['logfile']))

            despot1_cmd += ' > ' + self._inputs['logfile']

        os.system(despot1_cmd)
