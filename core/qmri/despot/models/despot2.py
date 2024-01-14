import os

#GLOBAL EXECUTABLES HERE
DESPOT_PATH = os.getenv('DESPOT_PATH')
despot2_exe = os.path.join(DESPOT_PATH, "despot2")

class DESPOT2_Model():
    def __init__(self, ssfp, params, t1, b1, out_dir, f0=None, out_base=None, model='DESPOT2-FM', algorithm='Ceres', mask=None, logfile=None, nthreads=1, verbose=False):
        self._inputs = {}
        self._inputs['ssfp']        = ssfp
        self._inputs['t1']          = t1
        self._inputs['b1']          = b1
        self._inputs['f0']          = f0
        self._inputs['model']       = model
        self._inputs['fit_params']  = params
        self._inputs['out_dir']     = out_dir
        self._inputs['out_base']    = out_base
        self._inputs['algo']        = algorithm
        self._inputs['mask']        = mask
        self._inputs['nthreads']    = nthreads
        self._inputs['logfile']     = logfile
        self._inputs['verbose']     = verbose
    
    def set_f0(self, f0):
        self._inputs['f0']          = f0
    def set_model(self, model):
        self._inputs['model']       = model

    def fit(self):

        despot2_cmd = despot2_exe \
                    + ' --ssfp=' + self._inputs['ssfp'].filename \
                    + ' --params=' + self._inputs['fit_params'] \
                    + ' --t1='+ self._inputs['t1'].filename \
                    + ' --b1='+self._inputs['b1'].filename \
                    + ' --out_dir='+ self._inputs['out_dir'] \
                    + ' --algo=' + self._inputs['algo'] \
                    + ' --nthreads=' + str(self._inputs['nthreads'])
        
        if self._inputs['model'].lower() == "despot2":
            despot2_cmd += ' --f0=' + self._inputs['f0'].filename
        
        if self._inputs['mask'] != None:
            despot2_cmd += ' --mask=' + self._inputs['mask'].filename
        if self._inputs['out_base'] != None:
            despot2_cmd+= ' --out_base='+self._inputs['out_base']
        if self._inputs['verbose']:
            despot2_cmd += ' -v'

        if self._inputs['logfile'] != None:
            if not os.path.exists(os.path.dirname(self._inputs['logfile'])):
                os.makedirs(os.path.dirname(self._inputs['logfile']))

            despot2_cmd += ' > ' + self._inputs['logfile']

        os.system(despot2_cmd)
