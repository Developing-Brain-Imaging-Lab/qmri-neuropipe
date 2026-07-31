
from pathlib import Path
import json
from datetime import datetime
import logging
from typing import Optional, Union

from ..core.run import run_cmd
from ..core.types import ImageLike
from ..core.utils import extract_image_path

LOGGER = logging.getLogger(__name__)


def synthmorph_transform_extension(
    model: Optional[str] = None,
    requested: Optional[str] = None,
) -> str:
    """Resolve a SynthMorph transform extension compatible with its model."""
    model_name = str(model or "joint").strip().lower()
    linear = any(
        candidate.startswith(model_name)
        for candidate in ("affine", "rigid")
    )
    expected = ".lta" if linear else ".nii.gz"
    if requested is None or not str(requested).strip():
        return expected

    extension = str(requested).strip().lower()
    if not extension.startswith("."):
        extension = f".{extension}"
    compatible = (
        extension == ".lta"
        if linear
        else extension in {".nii", ".nii.gz"}
    )
    if not compatible:
        transform_kind = "linear" if linear else "deformable"
        raise ValueError(
            f"SynthMorph model {model_name!r} produces a {transform_kind} "
            f"transform and requires {expected!r} output, not {extension!r}."
        )
    return extension


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
    
    from ..core.utils import check_nifti_integrity
    if out_p.exists() and check_nifti_integrity(out_p):
        return
        
    if out_p.exists(): out_p.unlink() # Clean up bad file
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


def mri_synthstrip(in_file: ImageLike | Path, out_file: Path, nthreads: Optional[int] = None, mask_out: Optional[Path] = None, gpu: bool = False):
    """
    Run FreeSurfer mri_synthstrip (SynthStrip).
    
    Args:
        in_file: path to input image
        out_file: path to output brain image
        nthreads: number of threads to use
        mask_out: optional path to save binary mask
        gpu: whether to use GPU acceleration
    """
    in_p = extract_image_path(in_file)
    out_p = Path(out_file); out_p.parent.mkdir(parents=True, exist_ok=True)
    
    if out_p.exists() and (not mask_out or Path(mask_out).exists()):
        return out_p, mask_out
    
    thread_arg = f"-t {nthreads}" if nthreads else ""
    mask_arg = f"-m {mask_out}" if mask_out else ""
    gpu_arg = "--gpu" if gpu else ""
    
    cmd = f"mri_synthstrip -i {in_p} -o {out_file} {thread_arg} {mask_arg} {gpu_arg}"
    try:
        run_cmd(cmd)
    except RuntimeError as exc:
        if not gpu or "CUDA is not available" not in str(exc):
            raise

        LOGGER.warning("mri_synthstrip GPU requested but CUDA is unavailable; retrying on CPU.")
        for path in (out_p, Path(mask_out) if mask_out else None):
            if path and path.exists():
                path.unlink()
        cpu_cmd = f"mri_synthstrip -i {in_p} -o {out_file} {thread_arg} {mask_arg}"
        run_cmd(cpu_cmd)
    return out_file, mask_out


def mri_synthmorph_register(
    moving: ImageLike | Path,
    target: ImageLike | Path,
    transform_out: Path,
    output_image: Optional[Path] = None,
    inverse_transform_out: Optional[Path] = None,
    model: Optional[str] = None,
    extra_args: str = "",
    overwrite: bool = False
):
    """
    Run FreeSurfer mri_synthmorph register to estimate a transform.

    Args:
        moving: moving image
        target: target/template image
        transform_out: path to save transform (-t)
        output_image: optional warped moving image in fixed space (-o)
        inverse_transform_out: optional inverse transform in fixed-to-moving direction (-T)
        model: synthmorph model to use (e.g. joint, deform, affine, rigid)
        extra_args: additional CLI args to pass to mri_synthmorph register
        overwrite: replace outputs if they already exist
    """
    mov_p = extract_image_path(moving)
    targ_p = extract_image_path(target)
    tx_p = Path(transform_out)
    tx_p.parent.mkdir(parents=True, exist_ok=True)
    synthmorph_transform_extension(model, "".join(tx_p.suffixes))
    if inverse_transform_out is not None:
        synthmorph_transform_extension(
            model,
            "".join(Path(inverse_transform_out).suffixes),
        )

    if tx_p.exists():
        if not overwrite:
            if output_image is None or Path(output_image).exists():
                return tx_p

    out_arg = f"-o {output_image}" if output_image else ""
    inv_arg = f"-T {inverse_transform_out}" if inverse_transform_out else ""
    model_arg = f"-m {model}" if model else ""
    extra = extra_args or ""
    cmd = f"mri_synthmorph register -t {tx_p} {inv_arg} {model_arg} {out_arg} {extra} {mov_p} {targ_p}".strip()
    run_cmd(cmd, label="mri_synthmorph_register")
    return tx_p


def mri_synthmorph_apply(
    moving: ImageLike | Path,
    target: ImageLike | Path,
    transform_in: Path,
    out_file: Path,
    extra_args: str = "",
    overwrite: bool = False
):
    """
    Run FreeSurfer mri_synthmorph apply to warp a moving image using a transform.

    Args:
        moving: moving image
        target: target/template image
        transform_in: transform to apply
        out_file: output warped image
        extra_args: additional CLI args to pass to mri_synthmorph apply
        overwrite: replace out_file if it already exists
    """
    mov_p = extract_image_path(moving)
    _ = extract_image_path(target)
    out_p = Path(out_file)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    if out_p.exists():
        if not overwrite:
            return out_p
        out_p.unlink()

    extra = extra_args or ""
    cmd = f"mri_synthmorph apply {transform_in} {mov_p} {out_p} {extra}".strip()
    run_cmd(cmd, label="mri_synthmorph_apply")
    return out_p


def mri_warp_convert_to_itk(
    in_transform: Path,
    out_file: Path,
    src_geom: Optional[ImageLike | Path] = None,
    lta1: Optional[Path] = None,
    lta1_inv: bool = False,
    lta2: Optional[Path] = None,
    lta2_inv: bool = False,
):
    tx_p = Path(in_transform)
    out_p = Path(out_file)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    if out_p.exists():
        return out_p

    suffixes = "".join(tx_p.suffixes).lower()
    if suffixes.endswith(".m3z"):
        in_arg = f"--inm3z {tx_p}"
    elif suffixes.endswith(".mgz"):
        in_arg = f"--inmgzwarp {tx_p}"
    elif suffixes.endswith(".nii.gz") or suffixes.endswith(".nii"):
        in_arg = f"--inlps {tx_p}"
    else:
        raise ValueError(f"Unsupported warp format for mri_warp_convert: {tx_p}")

    extra = []
    if src_geom:
        extra.append(f"--insrcgeom {extract_image_path(src_geom)}")
    if lta1:
        extra.append(f"{'--lta1-inv' if lta1_inv else '--lta1'} {Path(lta1)}")
    if lta2:
        extra.append(f"{'--lta2-inv' if lta2_inv else '--lta2'} {Path(lta2)}")

    cmd = f"mri_warp_convert {in_arg} --outitk {out_p} {' '.join(extra)}".strip()
    run_cmd(cmd, label="mri_warp_convert")
    return out_p


def lta_to_itk(in_lta: Path, out_file: Path, src: ImageLike | Path, trg: ImageLike | Path, invert: bool = False):
    in_p = Path(in_lta)
    out_p = Path(out_file)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    if out_p.exists():
        return out_p

    invert_arg = "--invert" if invert else ""
    src_p = extract_image_path(src)
    trg_p = extract_image_path(trg)
    cmd = f"lta_convert --inlta {in_p} --outitk {out_p} --src {src_p} --trg {trg_p} {invert_arg}".strip()
    run_cmd(cmd, label="lta_convert")
    return out_p


def lta_to_fsl(
    in_lta: Path,
    out_file: Path,
    src: ImageLike | Path,
    trg: ImageLike | Path,
    invert: bool = False,
    force: bool = False,
) -> Path:
    """Convert an LTA to an FSL matrix for the requested source/target grids."""
    in_p = Path(in_lta)
    out_p = Path(out_file)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    if out_p.exists():
        if not force:
            return out_p
        out_p.unlink()

    invert_arg = "--invert" if invert else ""
    src_p = extract_image_path(src)
    trg_p = extract_image_path(trg)
    cmd = (
        f"lta_convert --inlta {in_p} --outfsl {out_p} "
        f"--src {src_p} --trg {trg_p} {invert_arg}"
    ).strip()
    run_cmd(cmd, label="lta_convert")
    return out_p


def mri_coreg(
    moving_file: ImageLike | Path,
    reference_file: ImageLike | Path,
    out_lta: Path,
    *,
    dof: int = 6,
    nthreads: int = 1,
    force: bool = False,
) -> Path:
    """Estimate a linear FreeSurfer registration between arbitrary volumes."""
    moving_p = extract_image_path(moving_file)
    reference_p = extract_image_path(reference_file)
    out_p = Path(out_lta)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    if out_p.exists():
        if not force:
            return out_p
        out_p.unlink()

    cmd = (
        f"mri_coreg --mov {moving_p} --ref {reference_p} "
        f"--reg {out_p} --dof {int(dof)} --threads {int(nthreads)}"
    )
    run_cmd(cmd, label="mri_coreg")
    return out_p


def bbregister(in_file: ImageLike | Path, target_file: ImageLike | Path, out_reg_file: Path, contrast_type: str = "t1", fsl_mat_out: Path = None, subjects_dir: Path = None):
    """
    Run FreeSurfer bbregister.
    
    Args:
        in_file: Moving volume (e.g. b0 image)
        target_file: FreeSurfer Subject ID (as string) or Path? 
                     Standard bbregister expects a subject ID.
        out_reg_file: Output registration file.
        contrast_type: Moving-image contrast type ('t1', 't2', 'bold', 'dti')
        subjects_dir: Optional override for SUBJECTS_DIR.
    """
    in_p = extract_image_path(in_file)
    out_reg = Path(out_reg_file)
    out_reg.parent.mkdir(parents=True, exist_ok=True)

    fsl_mat = Path(fsl_mat_out) if fsl_mat_out else None
    def _valid_fsl_mat(path: Path) -> bool:
        try:
            import numpy as np
            mat = np.loadtxt(path)
            return mat.shape == (4, 4) and np.all(np.isfinite(mat))
        except Exception:
            return False

    if out_reg.exists():
        # Reuse existing registration only if all requested outputs are present.
        if not fsl_mat or _valid_fsl_mat(fsl_mat):
            return out_reg
        if fsl_mat.exists():
            fsl_mat.unlink()
        out_reg.unlink()
    
    # bbregister expects a FreeSurfer subject ID via --s, not a structural
    # image path. Accept a subject directory path as a convenience, but reject
    # image paths because they make bbregister look in the wrong place.
    subject_id = str(target_file)
    target_path = Path(subject_id)
    if target_path.exists() and target_path.is_dir():
        subjects_dir = subjects_dir or target_path.parent
        subject_id = target_path.name
    elif "/" in subject_id or "\\" in subject_id:
        raise ValueError(
            "bbregister target_file must be a FreeSurfer subject ID, not a file path. "
            f"Received: {target_file}"
        )
    
    cmd = f"bbregister --s {subject_id} --mov {in_p} --reg {out_reg} --{contrast_type}"
    
    if fsl_mat:
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
    subj_dir = sd_path / subject_id

    if (subj_dir / "mri" / "aseg.mgz").exists():
        return subj_dir

    omp_arg = f"-openmp {openmp}" if openmp else ""
    if subj_dir.exists():
        cmd = f"recon-all -s {subject_id} -sd {sd_path} {extra_args} {omp_arg}"
    else:
        cmd = f"recon-all -i {in_p} -s {subject_id} -sd {sd_path} {extra_args} {omp_arg}"
    run_cmd(cmd, label="recon-all")


def recon_all_clinical(in_file: ImageLike | Path, subject_id: str, subjects_dir: Path, nthreads: int = 1):
    """
    Wrapper for FreeSurfer recon-all-clinical.sh.
    
    Syntax: recon-all-clinical.sh <INPUT_SCAN> <SUBJECT_ID> <THREADS> [SUBJECT_DIR]
    """
    in_p = extract_image_path(in_file)
    sd_path = Path(subjects_dir)
    sd_path.mkdir(parents=True, exist_ok=True)

    # Key output for clinical script is slightly different but brain.mgz should still exist
    if (sd_path / subject_id / "mri" / "brain.mgz").exists():
        return sd_path / subject_id

    cmd = f"recon-all-clinical.sh {in_p} {subject_id} {nthreads} {sd_path}"
    run_cmd(cmd, label="recon-all-clinical")
    

def mri_synthseg(
    in_file: Union[Path, ImageLike],
    out_file: Path,
    nthreads: int = 1,
    robust: bool = True,
    parc: bool = False,
    extra_args: str = "",
    gpu: bool = False
):
    """
    Wrapper for FreeSurfer mri_synthseg.
    
    Args:
        in_file: Input image.
        out_file: Output segmentation (main).
        nthreads: Number of threads.
        robust: Use robust (less sensitive to artifacts) mode (default True).
        parc: Output parcellation instead of just segmentation? 
        extra_args: Additional arguments for mri_synthseg.
        gpu: Use GPU for processing. If False, adds --cpu flag.
    """
    in_p = extract_image_path(in_file)
    out_p = Path(out_file)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    
    if out_p.exists():
        return
        
    cmd = ["mri_synthseg", f"--i {in_p}", f"--o {out_p}"]
    
    if nthreads > 1:
        cmd.append(f"--threads {nthreads}")
        
    if robust:
        cmd.append("--robust")
        
    if not gpu:
        cmd.append("--cpu")
        
    if extra_args:
        cmd.append(extra_args)
        
    run_cmd(" ".join(cmd), label="mri_synthseg")


def mri_super_synth(
    in_file: Union[Path, ImageLike, str],
    out_dir: Path,
    mode: str = "invivo",
    threads: int = -1,
    device: Optional[str] = None,
    sharpen_synths: bool = False,
    overwrite: bool = False,
) -> Path:
    """
    Wrapper for FreeSurfer mri_super_synth (SuperSynth).

    Produces brain region segmentation, MNI atlas registration, and synthetic
    1mm isotropic T1w, T2w, and FLAIR images from any 3D brain volume.
    Supports in vivo, ex vivo, single-hemisphere, and cerebrum-only inputs.

    Requires FreeSurfer development build newer than October 2025.

    Args:
        in_file: Input image (single scan mode) or path to a CSV file (batch
            mode).  In batch mode ``out_dir`` and ``mode`` are ignored because
            they are encoded per-row in the CSV.
        out_dir: Output directory (single scan mode only).
        mode: Type of input volume.  One of: ``invivo``, ``exvivo``,
            ``cerebrum``, ``left-hemi``, ``right-hemi``.
        threads: Number of CPU cores.  Pass ``-1`` to use all available cores.
        device: Compute device — ``"cpu"`` or ``"cuda"``.  If *None*, the
            tool default is used (cuda when available).
        sharpen_synths: Apply sharpening to the synthetic T1w/T2w/FLAIR
            predictions.
        overwrite: Re-run even if outputs already exist.

    Returns:
        Path to the output directory (single scan mode) or to the CSV file
        that was passed as input (batch mode).
    """
    # Determine whether this is CSV batch mode or single-scan mode
    is_csv = isinstance(in_file, (str, Path)) and Path(in_file).suffix.lower() == ".csv"

    if is_csv:
        csv_p = Path(in_file)
        if not csv_p.exists():
            raise FileNotFoundError(f"mri_super_synth CSV not found: {csv_p}")
        cmd = f"mri_super_synth --i {csv_p} --threads {threads}"
        if device:
            cmd += f" --device {device}"
        if sharpen_synths:
            cmd += " --sharpen_synths"
        run_cmd(cmd, label="mri_super_synth")
        return csv_p

    # Single-scan mode
    in_p = extract_image_path(in_file)
    out_p = Path(out_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    # Skip if outputs already exist. Filenames differ across FreeSurfer
    # SuperSynth builds, and downstream Synb0/coregistration need SynthT1.
    seg_sentinels = (
        out_p / "segmentation.mgz",
        out_p / "seg.nii.gz",
        out_p / "seg.mgz",
        out_p / "segmentation.nii.gz",
    )
    synth_t1_sentinels = (
        out_p / "SynthT1.mgz",
        out_p / "T1w.nii.gz",
        out_p / "SynthT1.nii.gz",
        out_p / "T1w.mgz",
    )
    if (
        any(path.exists() for path in seg_sentinels)
        and any(path.exists() for path in synth_t1_sentinels)
        and not overwrite
    ):
        return out_p

    cmd = f"mri_super_synth --i {in_p} --o {out_p} --mode {mode} --threads {threads}"
    if device:
        cmd += f" --device {device}"
    if sharpen_synths:
        cmd += " --sharpen_synths"

    run_cmd(cmd, label="mri_super_synth")
    return out_p


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
