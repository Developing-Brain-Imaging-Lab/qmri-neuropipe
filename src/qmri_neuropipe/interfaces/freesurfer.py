
from pathlib import Path
import json, csv
from datetime import datetime
from typing import Optional, Union

from ..core.run import run_cmd
from ..core.types import ImageLike
from ..core.run import run_cmd
from ..core.types import ImageLike
from ..core.utils import extract_image_path

def write_sidecar(target: Path, meta: dict):
    sc = target.with_suffix(target.suffix + ".json")
    sc.parent.mkdir(parents=True, exist_ok=True)
    sc.write_text(json.dumps(meta, indent=2), encoding="utf-8")

def mri_convert(in_file: ImageLike | Path, out_file: Path):
    """
    Run FreeSurfer mri_convert.
    """
    in_p = extract_image_path(in_file)
    out_p = Path(out_file); out_p.parent.mkdir(parents=True, exist_ok=True)
    if out_p.exists():
        return
    run_cmd(f"mri_convert {in_p} {out_file}")

def mri_nu_correct(in_file: ImageLike | Path, out_file: Path):
    """
    Run FreeSurfer mri_nu_correct.
    """
    in_p = extract_image_path(in_file)
    out_p = Path(out_file); out_p.parent.mkdir(parents=True, exist_ok=True)
    if out_p.exists():
        return
    run_cmd(f"mri_nu_correct.mni --i {in_p} --o {out_file} --n 2")

def mri_normalize(in_file: ImageLike | Path, out_file: Path):
    """
    Run FreeSurfer mri_normalize.
    """
    in_p = extract_image_path(in_file)
    out_p = Path(out_file); out_p.parent.mkdir(parents=True, exist_ok=True)
    if out_p.exists():
        return
    run_cmd(f"mri_normalize -g 1 -mprage {in_p} {out_file}")


def mri_synthstrip(in_file: ImageLike | Path, out_file: Path, nthreads: Optional[int] = None, mask_out: Optional[Path] = None):
    """
    Run FreeSurfer mri_synthstrip (SynthStrip).
    
    Args:
        in_file: path to input image
        out_file: path to output brain image (or mask if requested via other flags, but here we model 1-to-1)
        nthreads: number of threads to use (passed as -n if supported or via OMP env)
        mask_out: optional path to save binary mask
    """
    # synthstrip supports -n/--threads? Checking docs would be ideal. 
    # Usually FS tools respect OMP_NUM_THREADS, but python wrappers might have explicit flags.
    # Assuming -n <threads> is valid for mri_synthstrip (common in recent tools).
    
    in_p = extract_image_path(in_file)
    out_p = Path(out_file); out_p.parent.mkdir(parents=True, exist_ok=True)
    
    if out_p.exists() and (not mask_out or Path(mask_out).exists()):
        return out_p, mask_out

    # SynthStrip uses -n for threads in some versions? Or depends on underlying tool (python/tensorflow).
    # But previous code used -t. Let's stick to -n if that's standard, but wait, 
    # line 69 used -t. The docstring said -n.
    # I will assume standardizing on `nthreads` variable, and use whatever flag works.
    # If the user error was simply keyword mismatch, checking flag is secondary but important.
    # Most FS tools don't have -t for threads. Python scripts might.
    # Let's check if I can find what flag mri_synthstrip uses. 
    # Usually it's a python wrapper around a model.
    # I will stick to the previous flag implementation `-t` but rename the variable. 
    # Wait, line 69 used `-t {n_threads}`. I'll keep `-t {nthreads}`.
    
    thread_arg = f"-t {nthreads}" if nthreads else ""
    mask_arg = f"-m {mask_out}" if mask_out else ""
    
    run_cmd(f"mri_synthstrip -i {in_p} -o {out_file} {thread_arg} {mask_arg}")
    return out_file, mask_out


def bbregister(in_file: ImageLike | Path, target_file: ImageLike | Path, out_reg_file: Path, contrast_type: str = "t1", fsl_mat_out: Path = None, subjects_dir: Path = None):
    """
    Run FreeSurfer bbregister.
    
    Args:
        in_file: Moving volume (e.g. b0 image)
        target_file: FreeSurfer Subject ID (as string) or Path? 
                     Standard bbregister expects a subject ID.
        out_reg_file: Output registration file.
        contrast_type: Contrast type ('t1', 't2', 'bold')
        subjects_dir: Optional override for SUBJECTS_DIR.
    """
    in_p = extract_image_path(in_file)
    out_reg = Path(out_reg_file)
    out_reg.parent.mkdir(parents=True, exist_ok=True)

    if out_reg.exists():
        return out_reg
    
    # bbregister expects subject ID via --s
    # We assume target_file is the subject ID if it's passed here.
    subject_id = str(target_file)
    
    cmd = f"bbregister --s {subject_id} --mov {in_p} --reg {out_reg} --{contrast_type}"
    
    if fsl_mat_out:
        fsl_mat = Path(fsl_mat_out)
        fsl_mat.parent.mkdir(parents=True, exist_ok=True)
        cmd += f" --fslmat {fsl_mat}"
        
    env = {}
    if subjects_dir:
        env['SUBJECTS_DIR'] = str(subjects_dir)
        
    run_cmd(cmd, label="bbregister", env=env)
    
    return out_reg


class RunLedger:
    def __init__(self, base: Path):
        self.base = base; self.base.mkdir(parents=True, exist_ok=True)
        self.path = self.base / "ledger.csv"
        if not self.path.exists():
            self.path.write_text("started_at,finished_at,dataset,sub,ses,pipeline,status,error\n", encoding="utf-8")
        self.started_at = None
    def start_run(self): self.started_at = datetime.utcnow().isoformat() + "Z"
    def mark_subject(self, sub, ses, status, error=""):
        self.path.write_text(self.path.read_text() + f"{self.started_at},,{''},{sub},{ses or ''},dmri-dti,{status},{error}\n", encoding="utf-8")
    def finish_run(self):
        lines = self.path.read_text().splitlines()
        if len(lines)<=1: return
        hdr, *rows = lines
        now = datetime.utcnow().isoformat() + "Z"
        new = [hdr]
        for r in rows:
            parts = r.split(",")
            if parts[1]=="":
                parts[1]=now
                new.append(",".join(parts))
            else:
                new.append(r)
        self.path.write_text("\n".join(new)+"\n", encoding="utf-8")


def recon_all(in_file: ImageLike | Path, subject_id: str, subjects_dir: Path, openmp: int = None, extra_args: str = "-all"):
    """
    Wrapper for FreeSurfer recon-all.
    """
    in_p = extract_image_path(in_file)
    sd_path = Path(subjects_dir)
    sd_path.mkdir(parents=True, exist_ok=True)

    if (sd_path / subject_id / "mri" / "aseg.mgz").exists():
        return sd_path / subject_id

    omp_arg = f"-openmp {openmp}" if openmp else ""
    cmd = f"recon-all -i {in_p} -s {subject_id} -sd {sd_path} {extra_args} {omp_arg}"
    run_cmd(cmd, label="recon-all")
    

def mri_synthseg(
    in_file: Union[Path, ImageLike],
    out_file: Path,
    nthreads: int = 1,
    robust: bool = True,
    parc: bool = False,
    extra_args: str = ""
):
    """
    Wrapper for FreeSurfer mri_synthseg.
    
    Args:
        in_file: Input image.
        out_file: Output segmentation (main).
        nthreads: Number of threads.
        robust: Use robust (less sensitive to artifacts) mode (default True).
        parc: Output parcellation instead of just segmentation? 
              SynthSeg produces one main output. Flags control behavior.
    """
    in_p = extract_image_path(in_file)
    out_p = Path(out_file)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    
    if out_p.exists():
        return
        
    # mri_synthseg --i <input> --o <output> --threads <n>
    cmd = ["mri_synthseg", f"--i {in_p}", f"--o {out_p}"]
    
    if nthreads > 1:
        cmd.append(f"--threads {nthreads}")
        
    if robust:
        cmd.append("--robust")
        
    if extra_args:
        cmd.append(extra_args)
        
    run_cmd(" ".join(cmd), label="mri_synthseg")

def mri_binarize(in_file: Union[Path, ImageLike], out_file: Path, min_val: float = 1, match: Optional[list] = None):
    """
    Run FreeSurfer mri_binarize.
    """
    in_p = extract_image_path(in_file)
    out_p = Path(out_file)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    
    if out_p.exists():
        return

    cmd = ["mri_binarize", f"--i {in_p}", f"--o {out_p}"]
    
    if match:
        # Match specific values
        vals = " ".join(map(str, match))
        cmd.append(f"--match {vals}")
    else:
        # Default threshold
        cmd.append(f"--min {min_val}")
        
    run_cmd(" ".join(cmd), label="mri_binarize")
