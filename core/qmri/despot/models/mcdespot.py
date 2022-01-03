import os, sys, subprocess, shutil, time

#GLOBAL EXECUTABLES HERE
mcd_dir, file = os.path.split(__file__)
platform = sys.platform

mcdespot_submit_file = mcd_dir + '/bin/linux/mcdespot_threePool.submit'
mcdespot_condor_script =  mcd_dir + '/bin/linux/mcdespot_threePool.sh'

if "linux" in platform:
    mcdespot_exe = mcd_dir + '/bin/linux/mcdespot '
    mcdespot_condor_exe = mcd_dir + '/bin/linux/mcdespot_condor'
    chunk_img_exe = mcd_dir + '/bin/linux/chunkImage '
    package_chunk_exe = mcd_dir + '/bin/linux/packageImage '
else:
    mcdespot_exe = mcd_dir + "/bin/mac/mcdespot "
    mcdespot_condor_exe = mcd_dir + "/bin/mac/mcdespot_condor "
    chunk_img_exe = mcd_dir + "/bin/mac/chunkImage "
    package_chunk_exe = mcd_dir + "/bin/mac/packageImage "

class MCDESPOT_Model():
    def __init__(self, spgr_img, ssfp_img, params, b1, f0, out_dir, out_base='mcDESPOT_', model=3, fit_algorithm='Ceres', mask=None, logfile=None, use_condor=False, chunk_size=500, nthreads=1, verbose=False):
        self._inputs = {}
        self._inputs['spgr_img']    = spgr_img
        self._inputs['ssfp_img']    = ssfp_img
        self._inputs['b1']          = b1
        self._inputs['f0']          = f0
        self._inputs['fit_params']  = params
        self._inputs['out_dir']     = out_dir
        self._inputs['out_base']    = out_base
        self._inputs['model']       = model
        self._inputs['algo']        = fit_algorithm
        self._inputs['mask']        = mask
        self._inputs['use_condor']  = use_condor
        self._inputs['chunk_size']  = chunk_size
        self._inputs['nthreads']    = nthreads
        self._inputs['logfile']     = logfile
        self._inputs['verbose']     = verbose

    def fit(self):

        if self._inputs['use_condor']:
            self.fit_condor()
        else:
            mcdespot_cmd = mcdespot_exe \
                        + ' --spgr=' + self._inputs['spgr_img']._get_filename() \
                        + ' --ssfp=' + self._inputs['ssfp_img']._get_filename() \
                        + ' --params=' + self._inputs['fit_params'] \
                        + ' --b1='+ self._inputs['b1']._get_filename() \
                        + ' --f0='+self._inputs['f0']._get_filename() \
                        + ' --out_dir='+ self._inputs['out_dir'] \
                        + ' --model='+ str(self._inputs['model']) \
                        + ' --algo=' + self._inputs['algo'] \
                        + ' --threads=' + str(self._inputs['nthreads'])

            if self._inputs['mask'] != None:
                mcdespot_cmd += ' --mask=' + self._inputs['mask']._get_filename()
            if self._inputs['out_base'] != None:
                mcdespot_cmd+= ' --out_base='+self._inputs['out_base']
            if self._inputs['verbose']:
                mcdespot_cmd += ' -v'

            if self._inputs['logfile'] != None:
                if not os.path.exists(os.path.dirname(self._inputs['logfile'])):
                    os.makedirs(os.path.dirname(self._inputs['logfile']))

                mcdespot_cmd += ' > ' + self._inputs['logfile']

            os.system(mcdespot_cmd)


    def fit_condor(self):

        condor_dir = self._inputs['out_dir'] + '/CONDOR/'

        if self._inputs['verbose']:
            print('Chunking Data')
        self.prepare_condor_chunks(condor_dir)

        if self._inputs['verbose']:
            print('Creating CONDOR DAG file')

        self.create_condor_dag(condor_dir)

        #Submit the submit file
        if self._inputs['verbose']:
            print('Submitting DAG File')

        os.system("ssh dean@medusa.keck.waisman.wisc.edu condor_submit_dag " + condor_dir + "/mcdespot_proc.dag")


    def prepare_condor_chunks(self, condor_dir):

        condorInput_dir = condor_dir + '/INPUTS/'
        condorOutput_dir = condor_dir + '/OUTPUTS/'
        sharedProcessing_dir = condorInput_dir + '/shared/'
        nChunks = self._inputs['chunk_size']

        if not os.path.exists(condor_dir):
            os.makedirs(condor_dir)
            os.makedirs(condorInput_dir)
            os.makedirs(condorOutput_dir)
            os.makedirs(sharedProcessing_dir)

        shutil.copy2(self._inputs['fit_params'], sharedProcessing_dir+'mcd_params.json');
        shutil.copy2(mcdespot_condor_exe, sharedProcessing_dir)

        spgr_img = self._inputs['spgr_img']._get_filename()
        ssfp_img = self._inputs['ssfp_img']._get_filename()
        mask_img = self._inputs['mask']._get_filename()
        b1_img   = self._inputs['b1']._get_filename()
        f0_img   = self._inputs['f0']._get_filename()

        #Next, chunk the data
        os.system(chunk_img_exe + ' --in='+ spgr_img + ' --mask='+mask_img+' --out_dir='+condor_dir+' --out=multiflipSPGR --chunks='+str(nChunks))
        os.system(chunk_img_exe + ' --in='+ ssfp_img + ' --mask='+mask_img+' --out_dir='+condor_dir+' --out=multiflipSSFP --chunks='+str(nChunks))
        os.system(chunk_img_exe + ' --in='+ b1_img + ' --mask='+mask_img+' --out_dir='+condor_dir+' --out=b1 --chunks='+str(nChunks))
        os.system(chunk_img_exe + ' --in='+ f0_img + ' --mask='+mask_img+' --out_dir='+condor_dir+' --out=f0 --chunks='+str(nChunks))

    def create_condor_dag(self, condor_dir):

        cmd='mcdespot_proc'
        dagFile=open('%s%s.dag' % (condor_dir, cmd),'w')

        condorInput_dir = condor_dir + 'INPUTS/'
        condorOutput_dir = condor_dir + 'OUTPUTS/'
        sharedProcessing_dir = condorInput_dir + 'shared'

        os.chdir(condorInput_dir)
        jobCount=0
        for dirName in os.listdir('.'):
            if 'Chunk_' in dirName:
                jobID='Chunk'+str(jobCount)
                in_dir = condorInput_dir+dirName
                out_dir = condorOutput_dir+dirName
                dagFile.write('JOB '+ jobID+ ' ' + mcdespot_submit_file+' \n')
                dagFile.write('VARS '+ jobID + ' executable= \"' + mcdespot_condor_script+'\"\n')
                dagFile.write('VARS '+ jobID + ' initialDir= \"' + out_dir+'\" \n')
                dagFile.write('VARS '+ jobID + ' logFile= \"' + jobID + '.log\" \n')
                dagFile.write('VARS '+ jobID + ' errFile= \"' + jobID + '.err\" \n')
                dagFile.write('VARS '+ jobID + ' outFile= \"' + jobID + '.out\" \n')
                dagFile.write('VARS '+ jobID + ' transferInputFiles=\"'+in_dir+','+sharedProcessing_dir+'\"\n')
                dagFile.write('VARS '+ jobID + ' transferOutputFiles=\"ModelParams\" \n')
                dagFile.write('VARS '+ jobID + ' args=\"'+dirName+' ModelParams ' + str(self._inputs['chunk_size'])+'\" \n')
                dagFile.write('\n\n')
                jobCount+=1

        dagFile.close()


    def package_condor_chunks_three_compartments(self):

        if self._inputs['verbose']:
            print('Checking if CONDOR Processing is Complete')
        proc_complete = self.check_if_complete()

        if proc_complete:
            if self._inputs['verbose']:
                print('Packing CONDOR Files...')
            condor_dir = self._inputs['out_dir'] + '/CONDOR/'

            condorInput_dir = condor_dir + '/INPUTS/'
            condorOutput_dir = condor_dir + '/OUTPUTS/'

            if not os.path.exists(self._inputs['out_dir']):
                os.makedirs(self._inputs['out_dir'])

            NUMBER_OF_VOXELS_TO_CHUNK=int(self._inputs['chunk_size'])

            imagesToPackage = []
            imagesToPackage.append('csfT1')
            imagesToPackage.append('csfT2')
            imagesToPackage.append('vCSF')
            imagesToPackage.append('freeT1')
            imagesToPackage.append('freeT2')
            imagesToPackage.append('tau')
            imagesToPackage.append('mT1')
            imagesToPackage.append('mT2')
            imagesToPackage.append('VFm')
            imagesToPackage.append('freeWaterOffResonance')

            output_imgs = []
            output_imgs.append('T1csf.nii.gz')
            output_imgs.append('T2csf.nii.gz')
            output_imgs.append('VFcsf.nii.gz')
            output_imgs.append('T1f.nii.gz')
            output_imgs.append('T2f.nii.gz')
            output_imgs.append('Tau.nii.gz')
            output_imgs.append('T1m.nii.gz')
            output_imgs.append('T2m.nii.gz')
            output_imgs.append('VFm.nii.gz')
            output_imgs.append('F0.nii.gz')

            for image in range(0, len(imagesToPackage)):

                listOfFiles = self._inputs['out_dir'] + imagesToPackage[image] + '.package'
                file = open(listOfFiles, 'w')

                for chunk in os.listdir(condorOutput_dir):
                    if('Chunk_' in chunk):
                        outputChunkDirectory = condorOutput_dir + chunk
                        measurementFile = outputChunkDirectory + '/ModelParams/' + imagesToPackage[image] + '.mcd'
                        voxPositionsFile = outputChunkDirectory + '/voxPositions.mcd'
                        file.write(measurementFile+'\n')
                        file.write(voxPositionsFile+'\n')

                #Package the image
                file.close()

                outputImage = self._inputs['out_dir'] + '/' + self._inputs['out_base'] + output_imgs[image]
                packageCommand = package_chunk_exe + '--in='+listOfFiles+' --out='+outputImage+ ' --mask='+self._inputs['mask']._get_filename() + ' --chunks='+str(NUMBER_OF_VOXELS_TO_CHUNK)

                os.system(packageCommand)
                os.remove(listOfFiles)

            command = "rm -rf " + condor_dir
            os.system(command)

        else:
            print('Check CONDOR to see if jobs still running')
            exit()

    def check_if_complete(self, verbose=False):

        condor_dir = self._inputs['out_dir'] + '/CONDOR/'
        condorOutput_dir = condor_dir + '/OUTPUTS/'

        NUMBER_OF_VOXELS_TO_CHUNK=int(self._inputs['chunk_size'])

        imageToCheck = 'VFm'
        chunksMissing=[]
        subjectComplete = True

        for chunk in os.listdir(condorOutput_dir):
            if('Chunk_' in chunk):
                outputChunkDirectory = condorOutput_dir + chunk
                measurementFile = outputChunkDirectory + '/ModelParams/' + imageToCheck + '.mcd'

                if not os.path.exists(measurementFile):
                    chunksMissing.append(chunk)
                    subjectComplete = False

        if(verbose):
            if(subjectComplete):
                print('\tCondor_Processing Complete!')
            else:
                print('\tMissing Following Chunks')
                for chunk in chunksMissing:
                    print('\t'+chunk)

        return(subjectComplete)
