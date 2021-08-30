import os, sys, subprocess, shutil, time

#GLOBAL EXECUTABLES HERE
mcd_dir, file = os.path.split(__file__)
platform = sys.platform

if "linux" in platform:
    despot2_exe = mcd_dir + '/bin/linux/despot2 '
else:
    despot2_exe = mcd_dir + "/bin/mac/despot2 "

class DESPOT2_Model():
    def __init__(self, ssfp_img, params, t1, b1, out_dir, out_base=None, model='DESPOT2-FM', fit_algorithm='Ceres', mask=None, logfile=None, nthreads=1, verbose=False):
        self._inputs = {}
        self._inputs['ssfp_img']    = ssfp_img
        self._inputs['t1']          = t1
        self._inputs['b1']          = b1
        self._inputs['model']       = model
        self._inputs['fit_params']  = params
        self._inputs['out_dir']     = out_dir
        self._inputs['out_base']    = out_base
        self._inputs['algo']        = fit_algorithm
        self._inputs['mask']        = mask
        self._inputs['nthreads']    = nthreads
        self._inputs['logfile']     = logfile
        self._inputs['verbose']     = verbose

    def fit(self):

        despot2_cmd = despot2_exe \
                    + ' --ssfp=' + self._inputs['ssfp_img']._get_filename() \
                    + ' --params=' + self._inputs['fit_params'] \
                    + ' --t1='+ self._inputs['t1']._get_filename() \
                    + ' --b1='+self._inputs['b1']._get_filename() \
                    + ' --out_dir='+ self._inputs['out_dir'] \
                    + ' --algo=' + self._inputs['algo'] \
                    + ' --threads=' + str(self._inputs['nthreads'])

        if self._inputs['mask'] != None:
            despot2_cmd += ' --mask=' + self._inputs['mask']._get_filename()
        if self._inputs['out_base'] != None:
            despot2_cmd+= ' --out_base='+self._inputs['out_base']
        if self._inputs['verbose']:
            despot2_cmd += ' -v'

        if self._inputs['logfile'] != None:
            if not os.path.exists(os.path.dirname(self._inputs['logfile'])):
                os.makedirs(os.path.dirname(self._inputs['logfile']))

            despot2_cmd += ' > ' + self._inputs['logfile']

        os.system(despot2_cmd)
