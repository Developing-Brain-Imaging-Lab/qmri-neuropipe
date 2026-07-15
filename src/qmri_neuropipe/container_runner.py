from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def eprint(*args: object) -> None:
    print(*args, file=sys.stderr)


def norm_label(value: str | None, prefix: str) -> str:
    value = str(value or "").strip()
    return value[len(prefix):] if value.startswith(prefix) else value


def is_empty_label(value: str) -> bool:
    return value.strip().lower() in {"", "none", "null", "n/a", "na"}


def split_csvish(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in str(value).split(",") if item.strip()]


def parse_subjects_file(path: Path) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    with path.open(newline="") as f:
        reader = csv.reader(line for line in f if line.strip() and not line.lstrip().startswith("#"))
        for row in reader:
            if not row:
                continue
            subject = row[0].strip()
            if subject.lower() in {"subject", "sub", "participant", "participant_label"}:
                continue
            session = row[1].strip() if len(row) > 1 else ""
            rows.append((subject, session))
    return rows


def selected_pairs(args: argparse.Namespace) -> list[tuple[str, str]]:
    if args.subjects_file:
        return parse_subjects_file(Path(args.subjects_file).expanduser())
    if args.subjects:
        subjects = split_csvish(args.subjects)
    elif args.subject:
        subjects = [args.subject]
    else:
        return [("", "")]

    sessions = split_csvish(args.sessions)
    if not sessions:
        sessions = [args.session] * len(subjects)
    elif len(sessions) == 1:
        sessions = sessions * len(subjects)
    elif len(subjects) == 1:
        subjects = subjects * len(sessions)
    elif len(sessions) != len(subjects):
        raise ValueError(
            "--sessions must contain either one session or the same number of "
            "comma-separated values as --subjects"
        )
    return list(zip(subjects, sessions))


def find_container_runtime(explicit: str = "") -> str:
    if explicit:
        runtime = shutil.which(explicit) if "/" not in explicit else explicit
        if runtime and Path(runtime).exists():
            return runtime
        raise FileNotFoundError(f"Container runtime not found: {explicit}")

    for candidate in ("apptainer", "singularity"):
        runtime = shutil.which(candidate)
        if runtime:
            return runtime
    raise FileNotFoundError("Neither apptainer nor singularity was found on PATH.")


def check_container_image(container_image: str) -> None:
    if "://" in container_image:
        return
    path = Path(container_image).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Container image not found: {path}")


def copy_support_file(source: str, support_dir: Path, *, label: str, required: bool = False) -> str:
    if not source:
        if required:
            raise ValueError(f"{label} is required")
        return ""

    src = Path(source).expanduser().resolve()
    if not src.is_file():
        raise FileNotFoundError(f"{label} not found: {src}")
    dest = support_dir / src.name
    shutil.copy2(src, dest)
    return dest.name


def bind_arg(host: Path | str, container: str, mode: str = "") -> str:
    src = str(host)
    bind = f"{src}:{container}"
    if mode:
        bind += f":{mode}"
    return bind


def build_container_command(
    *,
    args: argparse.Namespace,
    runtime: str,
    support_dir: Path,
    config_name: str,
    fs_license_name: str,
    subject: str,
    session: str,
) -> list[str]:
    bids_dir = Path(args.bids_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    work_dir = Path(args.work_dir).expanduser().resolve() if args.work_dir else output_dir / "work"

    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    cmd: list[str] = [
        runtime,
        "exec",
        "--pwd",
        "/",
    ]
    if args.cleanenv:
        cmd.append("--cleanenv")
    if args.nv:
        cmd.append("--nv")

    cmd.extend([
        "--bind",
        bind_arg(bids_dir, "/data", "ro"),
        "--bind",
        bind_arg(output_dir, "/out"),
        "--bind",
        bind_arg(work_dir, "/work"),
        "--bind",
        bind_arg(support_dir, "/config", "ro"),
    ])

    for extra_bind in args.bind:
        cmd.extend(["--bind", extra_bind])

    if fs_license_name:
        cmd.extend(["--env", f"FS_LICENSE=/config/{fs_license_name}"])

    cmd.append(args.container_image)
    cmd.extend([
        "qmri-neuropipe",
        "--config",
        f"config/{config_name}",
        "--bids-dir",
        "data",
        "--output-dir",
        "out",
        "--work-dir",
        "work",
        "--pipeline",
        args.pipeline,
        "--n-cpus",
        str(args.cpus),
        "--memory-gb",
        str(args.memory_gb),
    ])

    subject = norm_label(subject, "sub-")
    session = norm_label(session, "ses-")
    if subject:
        cmd.extend(["--participant-label", subject])
    if not is_empty_label(session):
        cmd.extend(["--session-label", session])

    cmd.extend(args.extra_arg)
    return cmd


def run(args: argparse.Namespace) -> int:
    runtime = find_container_runtime(args.runtime)
    check_container_image(args.container_image)

    bids_dir = Path(args.bids_dir).expanduser().resolve()
    if not bids_dir.is_dir():
        raise FileNotFoundError(f"BIDS directory not found: {bids_dir}")

    pairs = selected_pairs(args)
    failures = 0

    with tempfile.TemporaryDirectory(prefix="qneuro_container.") as tmp:
        support_dir = Path(tmp) / "config"
        support_dir.mkdir()
        config_name = copy_support_file(args.config_file, support_dir, label="Config file", required=True)
        fs_license_name = copy_support_file(args.freesurfer_license, support_dir, label="FreeSurfer license")
        if args.gnl_coeff_file:
            copy_support_file(args.gnl_coeff_file, support_dir, label="GNL coefficient file")

        for subject, session in pairs:
            label = "all subjects" if not subject else f"sub-{norm_label(subject, 'sub-')}"
            session_norm = norm_label(session, "ses-")
            if not is_empty_label(session_norm):
                label += f" ses-{session_norm}"

            cmd = build_container_command(
                args=args,
                runtime=runtime,
                support_dir=support_dir,
                config_name=config_name,
                fs_license_name=fs_license_name,
                subject=subject,
                session=session,
            )
            print(f"Running {label}:")
            print("  " + " ".join(shlex_quote(part) for part in cmd))
            if args.dry_run:
                continue

            result = subprocess.run(cmd)
            if result.returncode != 0:
                failures += 1
                if not args.keep_going:
                    return result.returncode

    return 1 if failures else 0


def shlex_quote(value: str) -> str:
    import shlex

    return shlex.quote(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run qmri-neuropipe directly through an Apptainer/Singularity container "
            "with automatic path binds."
        )
    )
    parser.add_argument("--container-image", required=True, help="Path or URI to qneuro/qmri-neuropipe SIF/image.")
    parser.add_argument("--config-file", required=True, help="qmri-neuropipe YAML/JSON config file.")
    parser.add_argument("--bids-dir", required=True, help="Host BIDS dataset directory.")
    parser.add_argument("--output-dir", required=True, help="Host output directory.")
    parser.add_argument("--work-dir", default="", help="Host work directory. Defaults to <output-dir>/work.")
    parser.add_argument("--freesurfer-license", default="", help="Optional FreeSurfer license file.")
    parser.add_argument("--gnl-coeff-file", default="", help="Optional GNL coefficient file copied into /config.")

    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--subject", help="Single subject label, with or without sub- prefix.")
    selection.add_argument("--subjects", help="Comma-separated subject labels.")
    selection.add_argument("--subjects-file", help="CSV/text file with subject,session rows.")
    parser.add_argument("--session", default="none", help="Single session label, with or without ses- prefix.")
    parser.add_argument("--sessions", default="", help="Comma-separated sessions paired with --subjects.")

    parser.add_argument("--pipeline", default="dmri", help="Pipeline passed to qmri-neuropipe.")
    parser.add_argument("--cpus", type=int, default=8, help="CPU count passed to qmri-neuropipe.")
    parser.add_argument("--memory-gb", type=int, default=32, help="Memory GB passed to qmri-neuropipe.")
    parser.add_argument("--runtime", default="", help="Container runtime executable. Defaults to apptainer, then singularity.")
    parser.add_argument("--bind", action="append", default=[], help="Additional bind spec, e.g. /host/path:/container/path:ro.")
    parser.add_argument("--extra-arg", action="append", default=[], help="Additional argument passed through to qmri-neuropipe.")
    parser.add_argument("--dry-run", action="store_true", help="Print container command(s) without running.")
    parser.add_argument("--keep-going", action="store_true", help="Continue after a failed subject/session.")
    parser.add_argument("--cleanenv", action="store_true", help="Run container with --cleanenv.")
    parser.add_argument("--no-nv", dest="nv", action="store_false", help="Do not pass --nv to the container runtime.")
    parser.set_defaults(nv=True)
    return parser


def main(argv: list[str]) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return run(args)
    except Exception as exc:
        eprint(f"Error: {exc}")
        return 1


def console_main() -> int:
    return main(sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(console_main())
