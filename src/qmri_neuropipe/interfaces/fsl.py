from pathlib import Path
from typing import Optional

from ..core.run import run_cmd
from ..core.types import ImageFile, DWIFile, ImageLike


def bet(in_img: Path, out: Path, frac: float = 0.5, mask: bool = True):

    out_p  = Path(out); out_p.parent.mkdir(parents=True, exist_ok=True)
    mask_p = out_p.with_name(Path(out_p.stem).stem + "_mask.nii.gz") if out_p.suffix == ".gz" else out_p.with_name(out_p.stem + "_mask.nii")
    
    bet_cmd = f"bet {in_img} {out} -f {frac} " + ("-m" if mask else "")
    
    if not out_p.exists():
        run_cmd(bet_cmd, label="bet")  
        
    return out_p


def eddy_correct(in_dwi: DWIFile, out: DWIFile):

    out_p    = Path(out); out_p.parent.mkdir(parents=True, exist_ok=True)    
    out_bvec = out_p.with_name(Path(out_p.stem).stem + ".bvec") if out_p.suffix == ".gz" else out_p.with_name(out_p.stem + ".bvec")
    ecclog   = out_p.with_name(Path(out_p.stem).stem + ".ecclog") if out_p.suffix == ".gz" else out_p.with_name(out_p.stem + ".ecclog")

    ecc_cmd = f"eddy_correct {in_img} {out} 0"
    fdt_cmd = f"fdt_rotate_bvecs {in_bvec} {out_bvec} {ecclog}"

    if not out_p.exists():
        run_cmd(ecc_cmd, label="eddy_correct")  
        
    if not out_bvec.exists():
        run_cmd(fdt_cmd, label="fdt_rotate_bvecs")
   
 
    return out_p, out_bvec


def eddy(in_dwi: DWIFile, out: Path, mask: Optional[Path]=None, topup_base: Optional[str]=None, external_b0: Optional[Path]=None, cuda: Optional[bool]=False, cuda_device: Optional[int]=0, nthreads: Optional[int]=1, eddy_opts: Optional[str]=""):

    out_p = Path(out); out_p.parent.mkdir(parents=True, exist_ok=True)
    out_base = str(out_p.with_suffix('').with_suffix(''))  # remove .nii.gz or .nii
    
    if not mask:
        mask = out_p.with_name("mask.nii.gz")
        mask = bet(in_img=in_img, out=mask)

    # cmd = "" 
    # if cuda:
    #     exe = 'CUDA_VISIBLE_DEVICES='+str(cuda_device)+ ' ' + eddy_cuda
    # else:
    #     os.environ["OMP_NUM_THREADS"] = str(nthreads)
    #     exe = 'OMP_NUM_THREADS='+str(nthreads)+ " " + eddy

    # command = exe + ' --imain=' + input_dwi.filename \
    #           + ' --mask='  + mask_img.filename \
    #           + ' --index=' + input_dwi.index \
    #           + ' --acqp='  + input_dwi.acqparams \
    #           + ' --bvecs=' + input_dwi.bvecs \
    #           + ' --bvals=' + input_dwi.bvals \
    #           + ' --slspec=' + input_dwi.slspec \
    #           + ' --out='   + eddy_output_base
    
    # if not cuda:
    #     command += ' --nthr=' + str(nthreads)
    # if topup_base != None:
    #     command += ' --topup='+topup_base
    # if external_b0 != None:
    #     command += ' --field='+external_b0
   
    # command += " " + fsl_eddy_options

    # if debug:
    #     print(command)
        
    # print(command)
    # os.system(command)
    # #Rotate b-vecs after doing the eddy correction
    # os.rename(eddy_output_base+'.eddy_rotated_bvecs', output_img.bvecs)
    # os.remove(os.path.join(output_dir, "mask.nii.gz"))

    # return output_img


# def eddy_openmp(in_img: Path, out_img: Path, bvecs: Path, bvals: Path, mask: Optional[Path]=None, acq_params: Optional[Path]=None, index: Optional[Path]=None, nthreads: int = 2) -> ShellCommandTask:
    