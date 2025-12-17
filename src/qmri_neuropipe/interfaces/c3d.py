"""
Wrapper for Convert3D (c3d) tools.
"""

from pathlib import Path
from ..core.run import run_cmd


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
    
    cmd = f"c3d_affine_tool -ref {ref_file} -src {in_file} -itk {transform_file} -ras2fsl -o {out_file}"
    run_cmd(cmd, label="c3d_affine_tool_ants2fsl")
