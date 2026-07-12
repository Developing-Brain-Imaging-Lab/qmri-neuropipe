#!/usr/bin/env python3
from __future__ import annotations

import gzip
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path


def eprint(*args: object) -> None:
    print(*args, file=sys.stderr)


def norm_label(value: str, prefix: str) -> str:
    value = str(value or "").strip()
    return value[len(prefix):] if value.startswith(prefix) else value


def is_empty_label(value: str) -> bool:
    return value.strip().lower() in {"", "none", "null", "n/a", "na"}


def command_exists(name: str) -> bool:
    return shutil.which(name) is not None


def find_config(cwd: Path, config_name: str) -> Path:
    if config_name and (cwd / config_name).is_file():
        return cwd / config_name
    candidates = sorted(
        p for p in cwd.iterdir()
        if p.is_file() and p.suffix.lower() in {".yaml", ".yml", ".json"}
    )
    if not candidates:
        raise FileNotFoundError(
            f"No YAML/JSON qmri-neuropipe config was transferred into {cwd}"
        )
    return candidates[0]


def resolve_qmri_command() -> list[str]:
    if command_exists("qmri-neuropipe"):
        return ["qmri-neuropipe"]
    if Path("/opt/conda/bin/qmri-neuropipe").is_file():
        return ["/opt/conda/bin/qmri-neuropipe"]
    for python in ("/opt/conda/bin/python", shutil.which("python") or ""):
        if python and Path(python).exists():
            probe = subprocess.run(
                [python, "-c", "import qmri_neuropipe.cli"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            if probe.returncode == 0:
                return [python, "-m", "qmri_neuropipe.cli"]
    raise RuntimeError(
        "qmri-neuropipe was not found. The job may not be running inside qneuro.sif "
        "or the container does not provide the installed package."
    )


def copy_freesurfer_license(cwd: Path, license_name: str) -> None:
    candidates: list[Path] = []
    if license_name:
        candidates.append(cwd / license_name)
    candidates.append(cwd / "license.txt")
    candidates.extend(sorted(cwd.glob("*license*.txt")))

    for candidate in candidates:
        if candidate.is_file():
            target_dir = cwd / "freesurfer"
            target_dir.mkdir(exist_ok=True)
            target = target_dir / "license.txt"
            shutil.copy2(candidate, target)
            os.environ["FS_LICENSE"] = str(target)
            print(f"Using FreeSurfer license at FS_LICENSE={target}")
            return

    eprint(
        "Warning: no transferred FreeSurfer license found. FreeSurfer tools may fail "
        "unless the container/site provides FS_LICENSE."
    )


def extract_input_archives(cwd: Path, subject: str, session_for_name: str) -> None:
    output_name = f"qneuro_outputs_sub-{subject}_ses-{session_for_name}.tar.gz"
    for archive in sorted(cwd.iterdir()):
        if not archive.is_file() or archive.name == output_name:
            continue
        if not (
            archive.name.endswith(".tar.gz")
            or archive.name.endswith(".tgz")
            or archive.name.endswith(".tar")
        ):
            continue
        print(f"Extracting transferred BIDS archive: {archive.name}")
        with tarfile.open(archive, "r:*") as tf:
            tf.extractall(cwd)


def move_if_exists(src: Path, dst: Path) -> bool:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        return True
    return False


def stage_bids_tree(cwd: Path, subject: str, session: str) -> Path:
    data_dir = cwd / "data"
    data_dir.mkdir(exist_ok=True)

    for root_file in ("dataset_description.json", "participants.tsv", "participants.json", "README", "CHANGES"):
        src = cwd / root_file
        if src.exists():
            shutil.move(str(src), str(data_dir / root_file))

    if session:
        if (cwd / f"sub-{subject}").is_dir():
            shutil.move(str(cwd / f"sub-{subject}"), str(data_dir / f"sub-{subject}"))
        elif (cwd / f"ses-{session}").is_dir():
            target = data_dir / f"sub-{subject}"
            target.mkdir(parents=True, exist_ok=True)
            shutil.move(str(cwd / f"ses-{session}"), str(target / f"ses-{session}"))
        elif any((cwd / m).exists() for m in ("anat", "dwi", "fmap", "func", "perf")):
            target = data_dir / f"sub-{subject}" / f"ses-{session}"
            target.mkdir(parents=True, exist_ok=True)
            for modality in ("anat", "dwi", "fmap", "func", "perf"):
                src = cwd / modality
                if src.exists():
                    shutil.move(str(src), str(target / modality))
        else:
            eprint(f"Could not find transferred BIDS input sub-{subject}/ses-{session} or ses-{session}")
            for p in sorted(cwd.glob("*")):
                if p.is_dir():
                    eprint(p)
            sys.exit(5)
    else:
        if not move_if_exists(cwd / f"sub-{subject}", data_dir / f"sub-{subject}"):
            eprint(f"Could not find transferred BIDS input sub-{subject}")
            sys.exit(5)

    desc = data_dir / "dataset_description.json"
    if not desc.exists():
        desc.write_text('{"Name":"condor-staged-qmri-neuropipe","BIDSVersion":"1.8.0"}\n')
    return data_dir


def preflight_dwi(data_dir: Path, subject: str, session: str, require_dwi: str) -> None:
    dwi_dir = data_dir / f"sub-{subject}"
    if session:
        dwi_dir = dwi_dir / f"ses-{session}"
    dwi_dir = dwi_dir / "dwi"

    print(f"DWI candidate files under {dwi_dir}:")
    for p in sorted(dwi_dir.glob("*_dwi.*")) if dwi_dir.exists() else []:
        print(f"  {p.name}")

    if require_dwi.lower() != "false":
        if not any(dwi_dir.glob("*_dwi.nii.gz")) and not any(dwi_dir.glob("*_dwi.nii")):
            eprint(f"No BIDS DWI image found at {dwi_dir}/*_dwi.nii[.gz].")
            sys.exit(7)


def make_output_tar(cwd: Path, subject: str, session_for_name: str) -> None:
    tarball = cwd / f"qneuro_outputs_sub-{subject}_ses-{session_for_name}.tar.gz"
    out_dir = cwd / "out"
    with tarfile.open(tarball, "w:gz") as tf:
        if out_dir.exists():
            for item in out_dir.rglob("*"):
                tf.add(item, arcname=item.relative_to(out_dir))
    print(f"Wrote {tarball.name}")


def main(argv: list[str]) -> int:
    if len(argv) < 5:
        eprint(
            "Usage: run_qneuro_condor.py <subject> <session|none> <cpus> "
            "<memory_gb> <pipeline> [config_name] [require_dwi] [license_name]"
        )
        return 2

    subject = norm_label(argv[0], "sub-")
    session_raw = norm_label(argv[1], "ses-")
    session = "" if is_empty_label(session_raw) else session_raw
    session_for_name = session or "none"
    cpus = argv[2]
    memory_gb = argv[3]
    pipeline = argv[4]
    config_name = argv[5] if len(argv) > 5 else ""
    require_dwi = argv[6] if len(argv) > 6 else "true"
    license_name = argv[7] if len(argv) > 7 else "license.txt"

    cwd = Path(os.environ.get("_CONDOR_SCRATCH_DIR", os.getcwd())).resolve()
    os.chdir(cwd)
    print(f"Running in Condor scratch directory: {cwd}")
    print("Transferred top-level files and directories:")
    for p in sorted(cwd.iterdir()):
        print(f"  {p.name}")

    env_file = Path("/.singularity.d/env/90-environment.sh")
    # Do not source shell from Python; set the paths we need explicitly.
    if env_file.exists():
        print(f"Container environment file exists: {env_file}")

    conda_dir = os.environ.setdefault("CONDA_DIR", "/opt/conda")
    fsldir = os.environ.setdefault("FSLDIR", "/usr/local/fsl")
    fs_home = os.environ.setdefault("FREESURFER_HOME", "/usr/local/freesurfer/8.2.0")
    c3d = os.environ.setdefault("C3DPATH", "/opt/c3d/bin")
    os.environ["PATH"] = ":".join([
        f"{conda_dir}/bin",
        f"{fsldir}/bin",
        f"{fs_home}/bin",
        f"{fs_home}/python/bin",
        f"{fs_home}/python/scripts",
        c3d,
        os.environ.get("PATH", ""),
    ])

    config = find_config(cwd, config_name)
    qmri_cmd = resolve_qmri_command()

    for dirname in ("data", "config", "out", "work"):
        (cwd / dirname).mkdir(exist_ok=True)
    staged_config = cwd / "config" / config.name
    shutil.copy2(config, staged_config)

    copy_freesurfer_license(cwd, license_name)
    extract_input_archives(cwd, subject, session_for_name)
    data_dir = stage_bids_tree(cwd, subject, session)

    print("Staged BIDS tree:")
    for p in sorted(x for x in data_dir.rglob("*") if x.is_dir()):
        print(p)

    if pipeline == "dmri":
        preflight_dwi(data_dir, subject, session, require_dwi)

    for key in (
        "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[key] = str(cpus)

    cmd = [
        *qmri_cmd,
        "--config", str(staged_config.relative_to(cwd)),
        "--bids-dir", "data",
        "--output-dir", "out",
        "--work-dir", "work",
        "--pipeline", pipeline,
        "--participant-label", subject,
        "--n-cpus", str(cpus),
        "--memory-gb", str(memory_gb),
    ]
    if session:
        cmd += ["--session-label", session]

    print(" ".join(cmd))
    try:
        subprocess.run(cmd, check=True, cwd=cwd, env=os.environ.copy())
    finally:
        make_output_tar(cwd, subject, session_for_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
