
from pathlib import Path
from typing import Optional, Dict, Union
import json

import shutil
# Optional dependencies imported locally


from ..core import ProcessingError
from ..core.utils import ensure_dir, extract_image_path
from ..core.types import ImageLike, DWIFile
from ..io.bids import build_bids_name, get_entities_from_path

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
        
    out_dir = ensure_dir(out_dir)
    studies_path = out_dir / "AMICO_studies"
    studies_path.mkdir(exist_ok=True)
    
    amico.core.setup()
    
    if bval_file is None or bvec_file is None:
        if isinstance(in_file, DWIFile):
            bval_file = bval_file or in_file.bval
            bvec_file = bvec_file or in_file.bvec
            
    if not bval_file or not bvec_file:
         raise ValueError("Gradient files (bval/bvec) are required but not provided or found in input DWIFile.")
    
    amico.core.setup()
    
    scheme_file = out_dir / 'scheme.txt'
    amico.util.fsl2scheme(str(bval_file), str(bvec_file), str(scheme_file))
    
    in_path = extract_image_path(in_file)
    ae = amico.Evaluation(str(studies_path), subject_id) # Initialize ae here
    ae.load_data(dwi_filename=str(in_path), scheme_filename=str(scheme_file), mask_filename=str(mask_file))
    ae.set_model("NODDI")
    ae.set_config('nthreads', n_cpus)
    ae.generate_kernels()
    ae.load_kernels()
    ae.fit()
    ae.save_results()
    
    # Results are in studies_path/subject_id/AMICO/NODDI
    res_dir = studies_path / subject_id / "AMICO" / "NODDI"
    
    output_files = {}
    output_files = {}
    ent_base = get_entities_from_path(in_path)
    if 'desc' in ent_base: del ent_base['desc']
    ent_base['model'] = 'NODDI'
    
    # Map AMICO outputs
    mapping = {
        "fit_NDI.nii.gz": "icvf",
        "fit_ODI.nii.gz": "odi",
        "fit_FWF.nii.gz": "fiso"
    }

    sidecar = {
        "ModelName": "NODDI (Neurite Orientation Dispersion and Density Imaging)",
        "FittingSoftware": "AMICO",
        "InputData": in_path.name,
        "FittingMethod": "AMICO accelerated fitting"
    }

    for src_name, suff in mapping.items():
        src = res_dir / src_name
        if src.exists():
             out_name = build_bids_name({**ent_base, 'suffix': suff.upper() if suff != 'fiso' else 'FISO'})             
             out_path = out_dir / out_name
             shutil.move(str(src), str(out_path))
             output_files[suff] = out_path
             
             # Save sidecar
             with open(str(out_path).replace('.nii.gz', '.json'), 'w') as f:
                  json.dump(sidecar, f, indent=4)
    
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
        
    out_dir = ensure_dir(out_dir)
    studies_path = out_dir / "AMICO_studies"
    studies_path.mkdir(exist_ok=True)
    
    out_dir = ensure_dir(out_dir)
    studies_path = out_dir / "AMICO_studies"
    studies_path.mkdir(exist_ok=True)
    
    if bval_file is None or bvec_file is None:
        if isinstance(in_file, DWIFile):
            bval_file = bval_file or in_file.bval
            bvec_file = bvec_file or in_file.bvec
            
    if not bval_file or not bvec_file:
         raise ValueError("Gradient files (bval/bvec) are required but not provided or found in input DWIFile.")
    
    amico.core.setup()
    ae = amico.Evaluation(str(studies_path), subject_id)
    
    scheme_file = out_dir / 'scheme.txt'
    
    Delta_file = in_file.Delta if isinstance(in_file, DWIFile) else kwargs.get('Delta_file')
    delta_file = in_file.delta if isinstance(in_file, DWIFile) else kwargs.get('delta_file')
    
    if Delta_file and delta_file:
        amico.util.sandi2scheme(str(bval_file), str(bvec_file), str(Delta_file), str(delta_file), schemeFilename=str(scheme_file))
    else:
        # Fallback if no explicit time parameters
        amico.util.fsl2scheme(str(bval_file), str(bvec_file), str(scheme_file))
    
    in_path = extract_image_path(in_file)
    ae.load_data(dwi_filename=str(in_path), scheme_filename=str(scheme_file), mask_filename=str(mask_file))
    ae.set_model("SANDI")
    ae.generate_kernels()
    ae.load_kernels()
    ae.fit()
    ae.save_results()
    
    res_dir = studies_path / subject_id / "AMICO" / "SANDI"
    
    output_files = {}
    output_files = {}
    ent_base = get_entities_from_path(in_path)
    if 'desc' in ent_base: del ent_base['desc']
    ent_base['model'] = 'SANDI'
    
    mapping = {
        'SANDI_fsoma.nii.gz': 'fsoma',
        'SANDI_fneurite.nii.gz': 'fneurite',
        'SANDI_fextra.nii.gz': 'fextra',
        'SANDI_Rsoma.nii.gz': 'Rsoma'
    }
    
    sidecar = {
        "ModelName": "SANDI (Soma and Neurite Density Imaging)",
        "FittingSoftware": "AMICO",
        "InputData": in_path.name
    }
    
    for src_name, suff in mapping.items():
        src = res_dir / src_name
        if src.exists():
            # Standardize suffix casing?
            # fsoma -> FSOMA?
            # Let's keep existing capitalization from mapping logic?
            # suffix arg in build_bids_name usually capitalized in BIDS?
            # let's assume we pass what we want.
            
            # NOTE: suffix passed to build_bids_name shouldn't have .nii.gz extension
            # previous code had: suffix=suff + '.nii.gz' which is wrong if build_bids_name adds extension!
            # build_bids_name adds default extension .nii.gz which is fine.
            
            out_name = build_bids_name({**ent_base, 'suffix': suff})
            out_path = out_dir / out_name
            shutil.move(str(src), str(out_path))
            output_files[suff] = out_path
            
            # Save sidecar
            with open(str(out_path).replace('.nii.gz', '.json'), 'w') as f:
                 json.dump(sidecar, f, indent=4)
            
    return output_files
