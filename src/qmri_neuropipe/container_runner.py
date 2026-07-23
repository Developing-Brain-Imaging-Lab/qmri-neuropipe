from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


SETTING_KEYS = {
    "container_image",
    "config_file",
    "bids_dir",
    "output_dir",
    "work_dir",
    "freesurfer_license",
    "gnl_coeff_file",
    "subject",
    "subjects",
    "subjects_file",
    "session",
    "sessions",
    "pipeline",
    "cpus",
    "n_cpus",
    "memory_gb",
    "runtime",
    "bind",
    "extra_arg",
    "dry_run",
    "keep_going",
    "cleanenv",
    "nv",
    "participant_label",
    "session_label",
    "config",
}

SETTING_ALIASES = {
    "config_file": "config",
    "cpus": "n_cpus",
}


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


def load_settings_file(path: Path) -> dict[str, object]:
    text = path.read_text()
    if path.suffix.lower() == ".json":
        data = json.loads(text)
    else:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError("YAML settings require PyYAML to be installed.") from exc
        data = yaml.safe_load(text)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Settings file must contain a mapping: {path}")
    return data


def pipeline_path_defaults(path: Path) -> dict[str, object]:
    """Read wrapper-relevant top-level paths from a pipeline config."""
    data = load_settings_file(path)
    return {
        key: data[key]
        for key in ("bids_dir", "output_dir", "work_dir")
        if data.get(key) not in (None, "")
    }


def pop_settings_arg(argv: list[str]) -> tuple[list[str], Path | None]:
    cleaned: list[str] = []
    settings: Path | None = None
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--settings":
            if i + 1 >= len(argv):
                raise ValueError("--settings requires a YAML/JSON file path")
            settings = Path(argv[i + 1]).expanduser()
            i += 2
            continue
        if arg.startswith("--settings="):
            settings = Path(arg.split("=", 1)[1]).expanduser()
            i += 1
            continue
        cleaned.append(arg)
        i += 1
    return cleaned, settings


def value_to_arg(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple)):
        return ",".join(str(v) for v in value)
    return str(value)


def has_flag(argv: list[str], flag: str) -> bool:
    return any(arg == flag or arg.startswith(f"{flag}=") for arg in argv)


def settings_values(settings: dict[str, object]) -> dict[str, object]:
    values: dict[str, object] = {}
    common = settings.get("common")
    if isinstance(common, dict):
        values.update(common)

    for key, value in settings.items():
        if key == "common" or isinstance(value, dict):
            continue
        values[key] = value

    for section_name in ("container", "qneuro_container", "qneuro-container"):
        section = settings.get(section_name)
        if isinstance(section, dict):
            values.update(section)
    return normalize_subject_session_settings(values)


def normalize_subject_session_settings(values: dict[str, object]) -> dict[str, object]:
    values = dict(values)
    session_value = values.get("session")
    if isinstance(session_value, (list, tuple)) and "sessions" not in values:
        values["sessions"] = session_value
        values.pop("session", None)

    sessions_value = values.get("sessions")
    has_subjects = bool(values.get("subjects") or values.get("subjects_file"))
    if (
        values.get("subject")
        and isinstance(sessions_value, (list, tuple))
        and len(sessions_value) > 1
        and not has_subjects
    ):
        values["subjects"] = [values["subject"]] * len(sessions_value)
        values.pop("subject", None)
    return values


def expand_settings_args(argv: list[str]) -> list[str]:
    argv, settings_path = pop_settings_arg(argv)
    if settings_path is None:
        return argv

    values = settings_values(load_settings_file(settings_path))
    inserted: list[str] = []
    subject_flags = {"--subject", "--subjects", "--subjects-file"}
    append_flags = {"bind", "extra_arg"}

    for key, value in values.items():
        normalized = str(key).replace("-", "_")
        if normalized not in SETTING_KEYS or value is None or value == "":
            continue

        normalized = SETTING_ALIASES.get(normalized, normalized)

        if normalized in {"subject", "subjects", "subjects_file", "participant_label"} and any(
            has_flag(argv, flag) for flag in subject_flags | {"--participant-label", "-p"}
        ):
            continue

        if normalized == "nv":
            if value is False and not has_flag(argv, "--no-nv"):
                inserted.append("--no-nv")
            continue

        flag = "--" + normalized.replace("_", "-")
        if normalized in {"dry_run", "keep_going", "cleanenv"}:
            if bool(value) and not has_flag(argv, flag):
                inserted.append(flag)
            continue

        if normalized in append_flags:
            if isinstance(value, (list, tuple)):
                for item in value:
                    inserted.extend([flag, value_to_arg(item)])
            elif not has_flag(argv, flag):
                inserted.extend([flag, value_to_arg(value)])
            continue

        if has_flag(argv, flag):
            continue
        inserted.extend([flag, value_to_arg(value)])

    return [*inserted, *argv]


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
    # Preserve site-visible aliases such as /study rather than resolving them to
    # backing paths such as /study2. The same spelling must remain valid inside
    # the container for paths embedded elsewhere in the pipeline config.
    bids_dir = Path(args.bids_dir).expanduser().absolute()
    output_dir = Path(args.output_dir).expanduser().absolute()
    work_dir = Path(args.work_dir).expanduser().absolute() if args.work_dir else output_dir / "work"

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
        bind_arg(bids_dir, str(bids_dir), "ro"),
        "--bind",
        bind_arg(output_dir, str(output_dir)),
        "--bind",
        bind_arg(work_dir, str(work_dir)),
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
        str(bids_dir),
        "--output-dir",
        str(output_dir),
        "--work-dir",
        str(work_dir),
    ])

    if args.pipeline:
        cmd.extend(["--pipeline", args.pipeline])
    if args.n_cpus is not None:
        cmd.extend(["--n-cpus", str(args.n_cpus)])
    if args.memory_gb is not None:
        cmd.extend(["--memory-gb", str(args.memory_gb)])

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

    config_path = Path(args.config).expanduser()
    config_defaults = pipeline_path_defaults(config_path)
    args.bids_dir = args.bids_dir or config_defaults.get("bids_dir")
    args.output_dir = args.output_dir or config_defaults.get("output_dir")
    args.work_dir = args.work_dir or config_defaults.get("work_dir") or ""
    if not args.bids_dir or not args.output_dir:
        raise ValueError(
            "bids_dir and output_dir must be provided either in --config or as "
            "--bids-dir/--output-dir wrapper overrides"
        )

    bids_dir = Path(args.bids_dir).expanduser().absolute()
    if not bids_dir.is_dir():
        raise FileNotFoundError(f"BIDS directory not found: {bids_dir}")

    pairs = selected_pairs(args)
    failures = 0

    with tempfile.TemporaryDirectory(prefix="qneuro_container.") as tmp:
        support_dir = Path(tmp) / "config"
        support_dir.mkdir()
        config_name = copy_support_file(args.config, support_dir, label="Config file", required=True)
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
    parser.add_argument("--settings", help="YAML/JSON defaults for this container wrapper. CLI flags override settings values.")
    parser.add_argument("--container-image", required=True, help="Path or URI to qneuro/qmri-neuropipe SIF/image.")
    parser.add_argument("--config", "-c", "--config-file", dest="config", required=True, help="qmri-neuropipe YAML/JSON config file.")
    parser.add_argument("--bids-dir", default="", help="Host BIDS dataset directory. Defaults to bids_dir in --config.")
    parser.add_argument("--output-dir", default="", help="Host output directory. Defaults to output_dir in --config.")
    parser.add_argument("--work-dir", default="", help="Host work directory. Defaults to work_dir in --config, then <output-dir>/work.")
    parser.add_argument("--freesurfer-license", default="", help="Optional FreeSurfer license file.")
    parser.add_argument("--gnl-coeff-file", default="", help="Optional GNL coefficient file copied into /config.")

    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--participant-label", "-p", "--subject", dest="subject", help="Single subject label, with or without sub- prefix.")
    selection.add_argument("--subjects", help="Comma-separated participant labels.")
    selection.add_argument("--subjects-file", help="CSV/text file with subject,session rows.")
    parser.add_argument("--session-label", "-s", "--session", dest="session", default="none", help="Single session label, with or without ses- prefix.")
    parser.add_argument("--sessions", default="", help="Comma-separated sessions paired with --subjects.")

    parser.add_argument("--pipeline", default=None, help="Pipeline passed to qmri-neuropipe (otherwise read from config).")
    parser.add_argument("--n-cpus", "--cpus", dest="n_cpus", type=int, default=None, help="CPU count passed to qmri-neuropipe (otherwise read from config).")
    parser.add_argument("--memory-gb", type=float, default=None, help="Memory GB passed to qmri-neuropipe (otherwise read from config).")
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
    argv = expand_settings_args(argv)
    parser = build_parser()
    args, passthrough = parser.parse_known_args(argv)
    args.extra_arg.extend(passthrough)
    try:
        return run(args)
    except Exception as exc:
        eprint(f"Error: {exc}")
        return 1


def console_main() -> int:
    return main(sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(console_main())
