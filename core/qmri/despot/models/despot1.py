import os

#GLOBAL EXECUTABLES HERE
DESPOT_PATH = os.getenv('DESPOT_PATH')
despot1_exe = os.path.join(DESPOT_PATH, "despot1")

class DESPOT1_Model():
    def __init__(self, spgr, params, out_dir, b1=None, irspgr=None, out_base=None, model='despot1', algorithm='Ceres', mask=None, param_type="double", logfile=None, nthreads=1, verbose=False):
        self._inputs = {}
        self._inputs['spgr']        = spgr
        self._inputs['irspgr']      = irspgr
        self._inputs['b1']          = b1
        self._inputs['model']       = model
        self._inputs['fit_params']  = params
        self._inputs['out_dir']     = out_dir
        self._inputs['out_base']    = out_base
        self._inputs['algo']        = algorithm
        self._inputs['mask']        = mask
        self._inputs['nthreads']    = nthreads
        self._inputs['logfile']     = logfile
        self._inputs['verbose']     = verbose
        self._inputs['param_type']  = param_type

    def set_b1(self, b1):
        self._inputs['b1']          = b1
    def set_model(self, model):
        self._inputs['model']       = model

    def fit(self):
        despot1_cmd = despot1_exe \
                    + ' --spgr=' + self._inputs['spgr'].filename \
                    + ' --params=' + self._inputs['fit_params'] \
                    + ' --out_dir='+ self._inputs['out_dir'] \
                    + ' --algo=' + self._inputs['algo'] \
                    + ' --nthreads=' + str(self._inputs['nthreads'])

        if self._inputs['mask'] != None:
            despot1_cmd += ' --mask=' + self._inputs['mask'].filename

        if self._inputs['model'].lower() == 'hifi':
            despot1_cmd += ' --irspgr=' + self._inputs['irspgr'].filename
        elif self._inputs['model'].lower() == 'despot1':
            despot1_cmd += ' --b1='+self._inputs['b1'].filename
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

        if self._inputs['verbose']:
            print(despot1_cmd)

        os.system(despot1_cmd)
