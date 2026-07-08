
from pathlib import Path
from typing import Optional, Dict, Union
import json

import shutil
# Optional dependencies imported locally


from ..core import ProcessingError
from ..core.utils import ensure_dir, extract_image_path
from ..core.types import ImageLike, DWIFile
from ..io.bids import build_bids_name, get_entities_from_path


def _normalize_sandi_scheme_header(scheme_file: Path) -> None:
    lines = scheme_file.read_text().splitlines()
    if not lines:
        raise ProcessingError(f"AMICO SANDI scheme file is empty: {scheme_file}")
    lines[0] = "VERSION: STEJSKALTANNER"
    scheme_file.write_text("\n".join(lines) + "\n")


def _move_amico_metric_outputs(
    res_dir: Path,
    out_dir: Path,
    ent_base: Dict[str, str],
    mapping: Dict[str, tuple[str, ...]],
    sidecar: Dict[str, str],
) -> Dict[str, Path]:
    output_files = {}
    for suffix, source_names in mapping.items():
        src = next((res_dir / name for name in source_names if (res_dir / name).exists()), None)
        if src is None:
            continue

        out_name = build_bids_name({**ent_base, 'suffix': suffix})
        out_path = out_dir / out_name
        if out_path.exists():
            out_path.unlink()
        shutil.move(str(src), str(out_path))
        output_files[suffix] = out_path

        with open(str(out_path).replace('.nii.gz', '.json'), 'w') as f:
            json.dump({**sidecar, "Metric": suffix}, f, indent=4)

    return output_files


def fit_noddi(
    in_file: Union[Path, ImageLike],
    out_dir: Path,
    bval_file: Optional[Path] = None,
    bvec_file: Optional[Path] = None,
    mask_file: Optional[Path] = None,
    subject_id: str = "subject",
    n_cpus: int = 1,
    **kwargs
) -> Dict[str, Path]:
    try:
        import os
        os.environ["TQDM_DISABLE"] = "1"
        import amico
    except ImportError:
        raise ProcessingError("AMICO not installed.")
        
    out_dir = ensure_dir(out_dir).resolve()
    studies_path = out_dir / "AMICO_studies"
    studies_path.mkdir(exist_ok=True)

    n_cpus = int(kwargs.pop("nthreads", n_cpus))
    
    amico.core.setup()
    
    if bval_file is None or bvec_file is None:
        if isinstance(in_file, DWIFile):
            bval_file = bval_file or in_file.bval
            bvec_file = bvec_file or in_file.bvec
            
    if not bval_file or not bvec_file:
         raise ValueError("Gradient files (bval/bvec) are required but not provided or found in input DWIFile.")
    
    amico.core.setup()
    
    bval_file = Path(bval_file).resolve()
    bvec_file = Path(bvec_file).resolve()
    mask_file = Path(mask_file).resolve() if mask_file else None
    scheme_file = (out_dir / 'scheme.txt').resolve()
    amico.util.fsl2scheme(str(bval_file), str(bvec_file), str(scheme_file))
    
    in_path = extract_image_path(in_file).resolve()
    ae = amico.Evaluation(str(studies_path), subject_id) # Initialize ae here
    ae.load_data(
        dwi_filename=str(in_path),
        scheme_filename=str(scheme_file),
        mask_filename=str(mask_file) if mask_file else None,
    )
    ae.set_model("NODDI")
    ae.set_config('nthreads', n_cpus)
    ae.generate_kernels()
    ae.load_kernels()
    ae.fit()
    ae.save_results()
    
    # Results are in studies_path/subject_id/AMICO/NODDI
    res_dir = studies_path / subject_id / "AMICO" / "NODDI"
    
    ent_base = get_entities_from_path(in_path)
    if 'desc' in ent_base: del ent_base['desc']
    ent_base['model'] = 'NODDI'
    
    # Map AMICO outputs
    mapping = {
        "ICVF": ("fit_NDI.nii.gz", "fit_ICVF.nii.gz"),
        "ODI": ("fit_ODI.nii.gz",),
        "FISO": ("fit_FWF.nii.gz", "fit_ISOVF.nii.gz", "fit_FISO.nii.gz"),
    }

    sidecar = {
        "ModelName": "NODDI (Neurite Orientation Dispersion and Density Imaging)",
        "FittingSoftware": "AMICO",
        "InputData": in_path.name,
        "FittingMethod": "AMICO accelerated fitting"
    }

    output_files = _move_amico_metric_outputs(res_dir, out_dir, ent_base, mapping, sidecar)
    
    # Cleanup
    os.environ["TQDM_DISABLE"] = "0"
    shutil.rmtree(studies_path)
    
    return output_files

def fit_sandi(
    in_file: Union[Path, ImageLike],
    out_dir: Path,
    bval_file: Optional[Path] = None,
    bvec_file: Optional[Path] = None,
    mask_file: Optional[Path] = None,
    subject_id: str = "subject",
    n_cpus: int = 1,
    **kwargs
) -> Dict[str, Path]:
    try:
        import amico
    except ImportError:
        raise ProcessingError("AMICO not installed.")
        
    out_dir = ensure_dir(out_dir).resolve()
    studies_path = out_dir / "AMICO_studies"
    studies_path.mkdir(exist_ok=True)

    n_cpus = int(kwargs.pop("nthreads", n_cpus))
    
    out_dir = ensure_dir(out_dir).resolve()
    studies_path = out_dir / "AMICO_studies"
    studies_path.mkdir(exist_ok=True)
    
    if bval_file is None or bvec_file is None:
        if isinstance(in_file, DWIFile):
            bval_file = bval_file or in_file.bval
            bvec_file = bvec_file or in_file.bvec
            
    if not bval_file or not bvec_file:
         raise ValueError("Gradient files (bval/bvec) are required but not provided or found in input DWIFile.")
    
    bval_file = Path(bval_file).resolve()
    bvec_file = Path(bvec_file).resolve()
    mask_file = Path(mask_file).resolve() if mask_file else None

    amico.core.setup()
    ae = amico.Evaluation(str(studies_path), subject_id)
    
    scheme_file = (out_dir / 'scheme.txt').resolve()
    
    Delta_file = kwargs.get('Delta_file')
    delta_file = kwargs.get('delta_file')
    if isinstance(in_file, DWIFile):
        Delta_file = Delta_file or getattr(in_file, 'Delta', None)
        delta_file = delta_file or getattr(in_file, 'delta', None)

    if not Delta_file or not delta_file:
        raise ProcessingError(
            "AMICO SANDI requires diffusion timing files so the scheme can be "
            "written as VERSION: STEJSKALTANNER. Provide both Delta_file and "
            "delta_file, or use the dmipy SANDI backend."
        )

    amico.util.sandi2scheme(
        str(bval_file),
        str(bvec_file),
        str(Path(Delta_file).resolve()),
        str(Path(delta_file).resolve()),
        schemeFilename=str(scheme_file),
    )
    _normalize_sandi_scheme_header(scheme_file)
    
    in_path = extract_image_path(in_file).resolve()
    ae.load_data(
        dwi_filename=str(in_path),
        scheme_filename=str(scheme_file),
        mask_filename=str(mask_file) if mask_file else None,
    )
    ae.set_model("SANDI")
    ae.set_config('nthreads', n_cpus)
    ae.generate_kernels()
    ae.load_kernels()
    ae.fit()
    ae.save_results()
    
    res_dir = studies_path / subject_id / "AMICO" / "SANDI"
    
    ent_base = get_entities_from_path(in_path)
    if 'desc' in ent_base: del ent_base['desc']
    ent_base['model'] = 'SANDI'
    
    mapping = {
        'fsoma': ('fit_fsoma.nii.gz', 'SANDI_fsoma.nii.gz'),
        'fneurite': ('fit_fneurite.nii.gz', 'SANDI_fneurite.nii.gz'),
        'fextra': ('fit_fextra.nii.gz', 'SANDI_fextra.nii.gz'),
        'Rsoma': ('fit_Rsoma.nii.gz', 'fit_rsoma.nii.gz', 'SANDI_Rsoma.nii.gz'),
    }
    
    sidecar = {
        "ModelName": "SANDI (Soma and Neurite Density Imaging)",
        "FittingSoftware": "AMICO",
        "InputData": in_path.name
    }
    
    output_files = _move_amico_metric_outputs(res_dir, out_dir, ent_base, mapping, sidecar)
            
    return output_files
