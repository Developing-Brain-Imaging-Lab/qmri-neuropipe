
from __future__ import annotations
from pathlib import Path
from typing import Optional, Dict, Any
from pydra import Workflow, mark
from pydra.tasks.shelltask import ShellCommandTask
from ..core.logging import log
from ..core.deriv import write_sidecar

def _mk(name, cmd, cache_dir: Path):
    t = ShellCommandTask(name=name, executable="bash", args=["-lc", cmd])
    t.cache_dir = cache_dir
    return t

def build_dmri_dti_workflow(bids_dir: Path, output_dir: Path, work_dir: Path, sub: str, ses: Optional[str], n_cpus: int, omp_nthreads: int, config: Dict[str, Any], dry_run: bool) -> Workflow:
    qderiv = output_dir / "derivatives" / "qmri-neuropipe" / "dmri-dti" / f"sub-{sub}"
    if ses: qderiv = qderiv / f"ses-{ses}"
    qderiv.mkdir(parents=True, exist_ok=True)
    cache_dir = work_dir / "pydra" / f"sub-{sub}" / (f"ses-{ses}" if ses else "nosession")
    cache_dir.mkdir(parents=True, exist_ok=True)

    wf = Workflow(name=f"dmri_dti_sub_{sub}_{ses or 'noses'}", cache_dir=cache_dir)
    wf.add(_mk("denoise", "echo denoise; sleep 0.05", cache_dir))
    wf.add(_mk("eddy", "echo eddy; sleep 0.05", cache_dir))
    wf.add(_mk("tensorfit", "echo tensor; sleep 0.05", cache_dir))
    wf.set_dependencies(dependencies=[(wf.denoise, wf.eddy), (wf.eddy, wf.tensorfit)])

    out_dir = qderiv / "dti"; out_dir.mkdir(parents=True, exist_ok=True)
    fa_target = out_dir / ("sub-{}_".format(sub) + (f"ses-{ses}_" if ses else "") + "FA.nii.gz")
    @mark.task
    def finalize(target: Path):
        target.write_text("stub FA", encoding="utf-8")
        write_sidecar(target, {"GeneratedBy":[{"Name":"qmri-neuropipe","Version":"0.0.1"}],"PipelineDescription":{"Name":"dmri-dti"},"Sources":[f"bids:sub-{sub}/"+(f"ses-{ses}/" if ses else "")+"dwi/..."]})
        log.info({"event":"sidecar_written","path":str(target)})
        return str(target)
    wf.add(finalize(name="finalize", target=fa_target))
    wf.set_dependencies(dependencies=[(wf.tensorfit, wf.finalize)])
    return wf
