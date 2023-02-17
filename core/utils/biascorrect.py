import os, copy
import core.utils.mask as mask

def biasfield_correction(input_img, output_file, method='N4', mask_img=None, nthreads=0, iterations=3):

    output_img = copy.deepcopy(input_img)
    output_img._set_filename(output_file)

    if method=='ants' or method=='fsl':
        command ='dwibiascorrect ' + method + ' ' \
                + input_img._get_filename() + ' ' \
                + output_img._get_filename() \
                + ' -force -quiet -nthreads ' + str(nthreads)

        
        if mask_img != None:
            command += ' -mask ' + mask_img._get_filename()

        if input_img._get_bvals() != None and input_img._get_bvecs() != None:
            command += ' -fslgrad ' + input_img._get_bvecs() + ' ' + input_img._get_bvals()

        print(command)
        os.system(command)

    elif method=='N4':
        os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = str(nthreads)
        command = 'N4BiasFieldCorrection -d 3 ' \
                + '-i ' + input_img._get_filename() + ' ' \
                + '-o ' + output_img._get_filename()

        if mask_img != None:
            command += ' -x ' + mask_img._get_filename()

        os.system(command)

        for i in range(0, iterations -1):
            command = 'N4BiasFieldCorrection -d 3 ' \
                    + '-i ' + output_img._get_filename() + ' ' \
                    + '-o ' + output_img._get_filename()

            if mask_img != None:
                command += ' -x ' + mask_img._get_filename()

            os.system(command)


    else:
        print('Invalid Biasfield correction Method')
        print('Available options are: ants, fsl, N4')
        exit()


    return output_img
