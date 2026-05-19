"""
Wrapper for Convert3D (c3d) tools.
"""

from pathlib import Path
from ..core.run import run_cmd


def _check_transform_file(transform_file: Path) -> None:
    transform_file = Path(transform_file)
    if not transform_file.exists():
        raise FileNotFoundError(f"Transform file does not exist: {transform_file}")
    if transform_file.stat().st_size == 0:
        raise RuntimeError(f"Transform file is empty: {transform_file}")


def _rewrite_transform_with_antspy(transform_file: Path) -> Path:
    """
    Re-write an ANTsPy/ITK transform through antspyx.

    Some ANTsPy affine files are readable by ANTs itself but rejected by
    c3d_affine_tool. Re-writing often normalizes the on-disk representation.
    """
    import ants

    transform_file = Path(transform_file)
    rewritten = transform_file.with_name(f"{transform_file.stem}_itk{transform_file.suffix}")
    tx = ants.read_transform(str(transform_file))
    ants.write_transform(tx, str(rewritten))
    _check_transform_file(rewritten)
    return rewritten


def _check_fsl_affine_file(transform_file: Path) -> None:
    import numpy as np

    _check_transform_file(transform_file)
    matrix = np.loadtxt(transform_file)
    if matrix.shape != (4, 4):
        raise RuntimeError(f"FSL affine is not a 4x4 matrix: {transform_file} has shape {matrix.shape}")
    if not np.all(np.isfinite(matrix)):
        raise RuntimeError(f"FSL affine contains non-finite values: {transform_file}")


def is_valid_fsl_affine(transform_file: Path) -> bool:
    try:
        _check_fsl_affine_file(transform_file)
        return True
    except Exception:
        return False


def fsl2ants(ref_file: Path, in_file: Path, transform_file: Path, out_file: Path):
    """
    Convert FSL affine matrix to ITK format using c3d_affine_tool.
    
    Command:
      c3d_affine_tool -ref <ref> -src <src> <fsl_mat> -fsl2ras -oitk <itk_mat>
    """
    cmd = f"c3d_affine_tool -ref {ref_file} -src {in_file} {transform_file} -fsl2ras -oitk {out_file}"
    run_cmd(cmd, label="c3d_affine_tool")

def ants2fsl(ref_file: Path, in_file: Path, transform_file: Path, out_file: Path):
    """
    Convert ITK affine matrix to FSL format using c3d_affine_tool.
    
    Command:
      c3d_affine_tool -ref <ref> -src <src> <itk_mat> -ras2fsl -o <fsl_mat>
    """
    # Note: c3d_affine_tool usually takes -ref and -src to define the space
    # For ITK to FSL, we typically need:
    # c3d_affine_tool -ref <fixed> -src <moving> -itk <ants_mat> -ras2fsl -o <fsl_mat>
    
    # Wait, c3d_affine_tool syntax can be tricky.
    # From help: c3d_affine_tool -ref <ref> -src <src> <transform> -ras2fsl -o <out>
    # <transform> is assumed ITK unless specified otherwise? Or do we need explicit flag?
    # Usually it auto-detects or assumes ITK if not specified with -fsl etc?
    # But for explicit:
    # c3d_affine_tool -ref <fixed> -src <moving> -itk <ant_aff> -ras2fsl -o <fsl_out>
    
    transform_file = Path(transform_file)
    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    _check_transform_file(transform_file)

    cmd = f"c3d_affine_tool -ref {ref_file} -src {in_file} -itk {transform_file} -ras2fsl -o {out_file}"
    try:
        run_cmd(cmd, label="c3d_affine_tool_ants2fsl")
        _check_fsl_affine_file(out_file)
        return
    except Exception as first_error:
        try:
            rewritten = _rewrite_transform_with_antspy(transform_file)
            cmd = f"c3d_affine_tool -ref {ref_file} -src {in_file} -itk {rewritten} -ras2fsl -o {out_file}"
            run_cmd(cmd, label="c3d_affine_tool_ants2fsl_retry")
            _check_fsl_affine_file(out_file)
            return
        except Exception as retry_error:
            raise RuntimeError(
                "Unable to convert ANTs affine transform to FSL format. "
                f"Original transform: {transform_file}. "
                f"Initial c3d error: {first_error}. "
                f"Retry after ANTsPy rewrite failed: {retry_error}"
            ) from retry_error
