"""
Segmentation utilities for anatomical processing.
"""

from pathlib import Path
from typing import Union, Literal
import logging
import os
import shutil

from ...core.run import run_cmd
from ...core.types import ImageLike
from ...core.utils import extract_image_path, ensure_dir, get_nifti_stem
from ...interfaces import fsl, freesurfer

_log = logging.getLogger(__name__)


def _supersynth_available() -> bool:
    """Return True if mri_super_synth is present in PATH or $FREESURFER_HOME/bin."""
    if shutil.which("mri_super_synth"):
        return True
    fs_home = os.environ.get("FREESURFER_HOME", "")
    if fs_home:
        return (Path(fs_home) / "bin" / "mri_super_synth").exists()
    return False


def generate_wm_segmentation(
    in_file: Union[Path, ImageLike],
    out_dir: Path,
    method: Literal['fast', 'synthseg', 'supersynth'] = 'fast',
    nthreads: int = 1,
    gpu: bool = False
) -> Path:
    """
    Generate a White Matter binary segmentation mask from a T1w/Anatomical image.

    Args:
        in_file: Input anatomical image
        out_dir: Output directory
        method: Segmentation method: 'fast' (FSL FAST), 'synthseg' (mri_synthseg),
            or 'supersynth' (mri_super_synth — requires FreeSurfer build newer than
            Oct 2025; falls back to 'synthseg' automatically when not available).
        nthreads: Number of threads
        gpu: Allow GPU use (synthseg/supersynth only)

    Returns:
        Path to white matter binary mask
    """
    out_dir = ensure_dir(out_dir)
    in_path = extract_image_path(in_file)

    stem = get_nifti_stem(in_path)
    wm_mask = out_dir / f"{stem}_wmseg.nii.gz"

    if wm_mask.exists():
        return wm_mask

    def _generate_with_method(selected_method: str) -> bool:
        if selected_method == 'fast':
            fast_base = out_dir / f"{stem}_fast"
            fast_seg = out_dir / f"{stem}_fast_seg.nii.gz"
            if not fast_seg.exists():
                fsl.fast(in_file, fast_base, img_type=1, num_classes=3)
            if fast_seg.exists():
                run_cmd(f"fslmaths {fast_seg} -thr 3 -uthr 3 -bin {wm_mask}", label="extract_wm_fast")

        elif selected_method == 'synthseg':
            synth_seg = out_dir / f"{stem}_synthseg.nii.gz"
            if not synth_seg.exists():
                freesurfer.mri_synthseg(in_file, synth_seg, nthreads=nthreads, gpu=gpu)
            if synth_seg.exists():
                run_cmd(
                    f"mri_binarize --i {synth_seg} --match 2 41 7 46 --o {wm_mask}",
                    label="extract_wm_synthseg",
                )

        elif selected_method == 'supersynth':
            supersynth_out = out_dir / f"{stem}_supersynth"
            from .super_synth import expected_supersynth_output, find_supersynth_outputs

            ss_outputs = find_supersynth_outputs(supersynth_out)
            seg_file = ss_outputs.get("seg", expected_supersynth_output(supersynth_out, "seg"))
            if not seg_file.exists():
                freesurfer.mri_super_synth(
                    in_file,
                    supersynth_out,
                    threads=nthreads,
                    device=None if gpu else "cpu",
                )
                ss_outputs = find_supersynth_outputs(supersynth_out)
                seg_file = ss_outputs.get("seg", seg_file)
            if seg_file.exists():
                # mri_super_synth uses the same FreeSurfer label scheme as SynthSeg:
                # 2/41 = left/right cerebral WM, 7/46 = left/right cerebellar WM.
                run_cmd(
                    f"mri_binarize --i {seg_file} --match 2 41 7 46 --o {wm_mask}",
                    label="extract_wm_supersynth",
                )

        else:
            raise ValueError(f"Unknown segmentation method: {selected_method}. Choose 'fast', 'synthseg', or 'supersynth'.")

        return wm_mask.exists()

    if method == 'fast':
        fast_base = out_dir / f"{stem}_fast"
        fast_seg = out_dir / f"{stem}_fast_seg.nii.gz"
        if not fast_seg.exists():
            fsl.fast(in_file, fast_base, img_type=1, num_classes=3)
        if fast_seg.exists():
            run_cmd(f"fslmaths {fast_seg} -thr 3 -uthr 3 -bin {wm_mask}", label="extract_wm_fast")

    elif method == 'synthseg':
        synth_seg = out_dir / f"{stem}_synthseg.nii.gz"
        if not synth_seg.exists():
            freesurfer.mri_synthseg(in_file, synth_seg, nthreads=nthreads, gpu=gpu)
        if synth_seg.exists():
            run_cmd(
                f"mri_binarize --i {synth_seg} --match 2 41 7 46 --o {wm_mask}",
                label="extract_wm_synthseg",
            )

    elif method == 'supersynth':
        # Fall back to synthseg when mri_super_synth is not available in this
        # FreeSurfer release (requires development build newer than Oct 2025).
        if not _supersynth_available():
            _log.warning(
                "mri_super_synth not found in this FreeSurfer installation. "
                "Falling back to mri_synthseg for WM segmentation."
            )
            return generate_wm_segmentation(
                in_file, out_dir, method='synthseg', nthreads=nthreads, gpu=gpu
            )

        errors: list[str] = []
        for fallback_method in ("supersynth", "synthseg", "fast"):
            try:
                if _generate_with_method(fallback_method):
                    if fallback_method != "supersynth":
                        _log.warning(
                            "SuperSynth WM segmentation failed; using %s fallback for BBR.",
                            fallback_method,
                        )
                    return wm_mask
            except Exception as exc:
                errors.append(f"{fallback_method}: {exc}")
                _log.warning(
                    "WM segmentation with %s failed while setting up BBR: %s",
                    fallback_method,
                    exc,
                )
        raise RuntimeError(
            "Failed to generate WM segmentation using method 'supersynth' "
            "or fallbacks. " + "; ".join(errors)
        )

    else:
        raise ValueError(f"Unknown segmentation method: {method}. Choose 'fast', 'synthseg', or 'supersynth'.")

    if not wm_mask.exists():
        raise RuntimeError(f"Failed to generate WM segmentation using method '{method}'.")

    return wm_mask
