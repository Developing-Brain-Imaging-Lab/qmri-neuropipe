#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path, PurePosixPath


ROOT_METADATA = ("dataset_description.json", "participants.tsv", "participants.json", "README", "CHANGES")
COMMANDS = {"submit-local", "push-submit", "stage-remote", "submit-staged", "collect-outputs"}
OUTPUT_ARCHIVE_RE = re.compile(r"^qneuro_outputs_sub-(?P<subject>.+?)_ses-(?P<session>.+?)\.tar\.gz$")
STAGED_BIDS_ARCHIVE_RE = re.compile(r"^sub-(?P<subject>.+?)(?:_ses-(?P<session>.+?))?_bids\.tar\.gz$")

COMMAND_SETTINGS = {
    "submit-local": {
        "container_image", "config_file", "freesurfer_license", "gnl_coeff_file", "bids_dir",
        "subject", "subjects", "subjects_file", "session", "sessions",
        "bids_include_dirs", "bids_exclude_dirs", "anat_derivatives_dir", "t1w_match_acq", "t1w_match_desc",
        "transfer_uri", "pipeline", "cpus", "gpus", "memory_gb", "disk_gb",
        "require_dwi", "submit_file_name", "queue_file_name", "no_submit",
        "package_dir", "requirements", "getenv", "gpu_minimum_capability",
        "want_flocking", "want_glidein", "want_gpu_lab", "gpu_job_length",
        "notification", "notify_user", "log_dir", "output_directory",
        "output_destination",
    },
    "push-submit": {
        "container_image", "config_file", "freesurfer_license", "gnl_coeff_file", "bids_dir",
        "subject", "subjects", "subjects_file", "session", "sessions",
        "bids_include_dirs", "bids_exclude_dirs", "anat_derivatives_dir", "t1w_match_acq", "t1w_match_desc",
        "transfer_uri", "pipeline", "cpus", "gpus", "memory_gb", "disk_gb",
        "require_dwi", "submit_file_name", "queue_file_name", "no_submit",
        "submit_host", "remote_submit_dir", "remote_package_dir",
        "requirements", "getenv", "gpu_minimum_capability", "want_flocking",
        "want_glidein", "want_gpu_lab", "gpu_job_length", "notification",
        "notify_user", "log_dir", "output_directory", "output_destination",
    },
    "stage-remote": {
        "remote_host", "remote_stage_dir", "bids_dir", "subject", "subjects",
        "subjects_file", "session", "sessions", "bids_include_dirs", "bids_exclude_dirs",
        "anat_derivatives_dir", "manifest_name",
        "bundle_name", "create_remote_dir", "bundle", "no_bundle",
        "include_support_files", "config_file", "freesurfer_license", "gnl_coeff_file",
        "copy_script", "no_copy_script",
    },
    "submit-staged": {
        "staging_dir", "submit_dir", "manifest", "manifest_name",
        "container_image", "config_file", "freesurfer_license", "gnl_coeff_file", "transfer_uri",
        "subject", "subjects", "subjects_file", "session", "sessions",
        "pipeline", "cpus", "gpus", "memory_gb", "disk_gb", "require_dwi",
        "t1w_match_acq", "t1w_match_desc",
        "submit_file_name", "queue_file_name", "no_submit", "requirements",
        "getenv", "gpu_minimum_capability", "want_flocking", "want_glidein",
        "want_gpu_lab", "gpu_job_length", "notification", "notify_user",
        "log_dir", "output_directory", "output_destination",
    },
    "collect-outputs": {
        "outputs_dir", "remote_host", "remote_outputs_dir", "bids_dir",
        "derivatives_dir", "derivative_name", "subject", "subjects",
        "subjects_file", "session", "sessions", "overwrite", "dry_run",
    },
}

BOOLEAN_FLAGS = {
    "no_submit": "--no-submit",
    "create_remote_dir": "--create-remote-dir",
    "no_bundle": "--no-bundle",
    "include_support_files": "--include-support-files",
    "no_copy_script": "--no-copy-script",
    "overwrite": "--overwrite",
    "dry_run": "--dry-run",
}

NEGATED_BOOLEAN_FLAGS = {
    "bundle": (False, "--no-bundle"),
    "copy_script": (False, "--no-copy-script"),
}


def eprint(*args: object) -> None:
    print(*args, file=sys.stderr)


def parse_scalar(value: str) -> object:
    value = value.strip()
    if not value:
        return ""
    if value[0:1] in {"'", '"'} and value[-1:] == value[0]:
        return value[1:-1]
    lower = value.lower()
    if lower in {"true", "yes", "on"}:
        return True
    if lower in {"false", "no", "off"}:
        return False
    if lower in {"null", "none", "~"}:
        return ""
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [str(parse_scalar(part.strip())) for part in inner.split(",")]
    if len(value) > 1 and value[0] == "0" and value[1].isdigit():
        return value
    try:
        return int(value)
    except ValueError:
        return value


def parse_simple_yaml(text: str) -> dict[str, object]:
    data: dict[str, object] = {}
    current_key: str | None = None
    current_dict: dict[str, object] | None = None
    current_list: list[object] | None = None

    for raw_line in text.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        line = raw_line.split("#", 1)[0].rstrip()
        stripped = line.strip()

        if indent == 0:
            current_key = None
            current_dict = None
            current_list = None
            if ":" not in stripped:
                raise ValueError(f"Unsupported YAML line: {raw_line}")
            key, value = stripped.split(":", 1)
            key = key.strip()
            value = value.strip()
            if value:
                data[key] = parse_scalar(value)
            else:
                current_key = key
                data[key] = {}
                current_dict = data[key]  # type: ignore[assignment]
            continue

        if current_key is None:
            raise ValueError(f"Indented YAML line has no parent key: {raw_line}")
        if stripped.startswith("- "):
            if current_list is None:
                current_list = []
                data[current_key] = current_list
                current_dict = None
            current_list.append(parse_scalar(stripped[2:]))
            continue
        if ":" in stripped:
            if current_dict is None:
                current_dict = {}
                data[current_key] = current_dict
                current_list = None
            key, value = stripped.split(":", 1)
            current_dict[key.strip()] = parse_scalar(value.strip())
            continue
        raise ValueError(f"Unsupported YAML line: {raw_line}")

    return data


def load_settings_file(path: Path) -> dict[str, object]:
    text = path.read_text()
    if path.suffix.lower() == ".json":
        loaded = json.loads(text)
    else:
        loaded = parse_simple_yaml(text)
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Settings file must contain a YAML mapping: {path}")
    return loaded


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


def command_from_argv(argv: list[str], settings: dict[str, object]) -> str | None:
    for arg in argv:
        if arg in COMMANDS:
            return arg
    command = settings.get("command")
    return str(command) if command else None


def has_flag(argv: list[str], flag: str) -> bool:
    return any(arg == flag or arg.startswith(f"{flag}=") for arg in argv)


def has_any_flag(argv: list[str], flags: set[str]) -> bool:
    return any(has_flag(argv, flag) for flag in flags)


def settings_for_command(settings: dict[str, object], command: str) -> dict[str, object]:
    merged: dict[str, object] = {}
    common = settings.get("common")
    if isinstance(common, dict):
        merged.update(common)

    for key, value in settings.items():
        if key in {"command", "common"} or isinstance(value, dict):
            continue
        merged[key] = value

    for section_name in (command, command.replace("-", "_")):
        section = settings.get(section_name)
        if isinstance(section, dict):
            merged.update(section)
    return merged


def value_to_arg(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple)):
        return ",".join(str(v) for v in value)
    return str(value)


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

    settings = load_settings_file(settings_path)
    command = command_from_argv(argv, settings)
    if not command:
        raise ValueError("Settings file must include command, or command must be given on the CLI")
    if command not in COMMANDS:
        raise ValueError(f"Unsupported settings command: {command}")

    if not argv or argv[0] not in COMMANDS:
        argv = [command, *argv]

    values = normalize_subject_session_settings(settings_for_command(settings, command))
    allowed = COMMAND_SETTINGS[command]
    inserted: list[str] = []

    subject_flags = {"--subject", "--subjects", "--subjects-file"}
    for key, value in values.items():
        normalized = str(key).replace("-", "_")
        if normalized not in allowed or value is None or value == "":
            continue
        if normalized in {"subject", "subjects", "subjects_file"} and has_any_flag(argv, subject_flags):
            continue

        if normalized in BOOLEAN_FLAGS:
            flag = BOOLEAN_FLAGS[normalized]
            if bool(value) and not has_flag(argv, flag):
                inserted.append(flag)
            continue
        if normalized in NEGATED_BOOLEAN_FLAGS:
            trigger_value, flag = NEGATED_BOOLEAN_FLAGS[normalized]
            if bool(value) == trigger_value and not has_flag(argv, flag):
                inserted.append(flag)
            continue

        flag = "--" + normalized.replace("_", "-")
        if has_flag(argv, flag):
            continue
        inserted.extend([flag, value_to_arg(value)])

    return [argv[0], *inserted, *argv[1:]]


def norm_label(value: str | None, prefix: str) -> str:
    value = str(value or "").strip()
    return value[len(prefix):] if value.startswith(prefix) else value


def is_empty_label(value: str) -> bool:
    return value.strip().lower() in {"", "none", "null", "n/a", "na"}


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
        return parse_subjects_file(Path(args.subjects_file))
    if args.subjects:
        subjects = [s.strip() for s in args.subjects.split(",") if s.strip()]
        sessions = [s.strip() for s in args.sessions.split(",") if s.strip()] if args.sessions else []
        if not sessions:
            sessions = [args.session] * len(subjects)
        elif len(sessions) == 1:
            sessions = sessions * len(subjects)
        elif len(sessions) != len(subjects):
            raise ValueError(
                "--sessions must contain either one session or the same number of "
                "comma-separated values as --subjects"
            )
        return list(zip(subjects, sessions))
    return [(args.subject, args.session)]


def has_subject_filter(args: argparse.Namespace) -> bool:
    return bool(
        getattr(args, "subject", "")
        or getattr(args, "subjects", "")
        or getattr(args, "subjects_file", "")
    )


def filter_manifest_rows(
    rows: list[list[str]],
    args: argparse.Namespace,
) -> list[list[str]]:
    if not has_subject_filter(args):
        return rows

    requested: set[tuple[str, str]] = set()
    wildcard_subjects: set[str] = set()
    for subject_raw, session_raw in selected_pairs(args):
        subject = norm_label(subject_raw, "sub-")
        session = norm_label(session_raw, "ses-")
        if is_empty_label(session):
            wildcard_subjects.add(subject)
        else:
            requested.add((subject, session))

    filtered: list[list[str]] = []
    for row in rows:
        subject = norm_label(row[0], "sub-")
        session = norm_label(row[1], "ses-")
        if subject in wildcard_subjects or (subject, session) in requested:
            filtered.append(row)

    if not filtered:
        requested_text = ", ".join(
            [f"sub-{s}/ses-{se}" for s, se in sorted(requested)]
            + [f"sub-{s}/<any-session>" for s in sorted(wildcard_subjects)]
        )
        raise ValueError(f"No staged input rows matched requested filter: {requested_text}")
    return filtered


def tar_members(subject: str, session: str) -> tuple[str, str]:
    subject = norm_label(subject, "sub-")
    session = norm_label(session, "ses-")
    subject_dir = f"sub-{subject}"
    if not is_empty_label(session):
        session_dir = f"ses-{session}"
        return f"{subject_dir}/{session_dir}", f"{subject_dir}_{session_dir}_bids.tar.gz"
    return subject_dir, f"{subject_dir}_bids.tar.gz"


def split_csvish(value: object) -> list[str]:
    if not value:
        return []
    if isinstance(value, (list, tuple)):
        parts = value
    else:
        parts = str(value).split(",")
    return [str(part).strip() for part in parts if str(part).strip()]


def add_empty_dir(tf: tarfile.TarFile, source: Path, arcname: str) -> None:
    info = tf.gettarinfo(str(source), arcname=arcname)
    info.type = tarfile.DIRTYPE
    tf.addfile(info)


def add_direct_files(tf: tarfile.TarFile, source_dir: Path, arc_prefix: str) -> None:
    for item in sorted(source_dir.iterdir()):
        if item.is_file():
            tf.add(
                item,
                arcname=f"{arc_prefix}/{item.name}",
                filter=tarinfo_without_macos_metadata,
            )


def add_filtered_bids_tree(
    tf: tarfile.TarFile,
    bids_dir: Path,
    input_rel: str,
    include_dirs: list[str],
    exclude_dirs: list[str],
) -> None:
    input_dir = bids_dir / input_rel
    include_set = {d.strip("/") for d in include_dirs}
    exclude_set = {d.strip("/") for d in exclude_dirs}
    add_empty_dir(tf, input_dir, input_rel)
    add_direct_files(tf, input_dir, input_rel)

    if include_set:
        datatype_dirs = sorted(include_set)
    else:
        datatype_dirs = sorted(
            item.name for item in input_dir.iterdir()
            if item.is_dir() and item.name not in exclude_set
        )

    for dirname in datatype_dirs:
        if dirname in exclude_set:
            continue
        src = input_dir / dirname
        if src.is_dir():
            tf.add(src, arcname=f"{input_rel}/{dirname}", filter=tarinfo_without_macos_metadata)
        elif include_set:
            print(f"Warning: requested BIDS directory not found, skipping: {src}", file=sys.stderr)


def add_filtered_subject_tree(
    tf: tarfile.TarFile,
    bids_dir: Path,
    subject_rel: str,
    include_dirs: list[str],
    exclude_dirs: list[str],
) -> None:
    subject_dir = bids_dir / subject_rel
    include_set = {d.strip("/") for d in include_dirs}
    exclude_set = {d.strip("/") for d in exclude_dirs}
    add_empty_dir(tf, subject_dir, subject_rel)
    add_direct_files(tf, subject_dir, subject_rel)

    for item in sorted(subject_dir.iterdir()):
        if not item.is_dir():
            continue
        if item.name.startswith("ses-"):
            add_filtered_bids_tree(tf, bids_dir, f"{subject_rel}/{item.name}", include_dirs, exclude_dirs)
        elif include_set:
            if item.name in include_set and item.name not in exclude_set:
                tf.add(item, arcname=f"{subject_rel}/{item.name}", filter=tarinfo_without_macos_metadata)
        elif item.name not in exclude_set:
            tf.add(item, arcname=f"{subject_rel}/{item.name}", filter=tarinfo_without_macos_metadata)


def create_bids_archive(
    bids_dir: Path,
    subject: str,
    session: str,
    archive_dir: Path,
    include_dirs: list[str] | None = None,
    exclude_dirs: list[str] | None = None,
    anat_derivatives_dir: Path | None = None,
) -> Path:
    input_rel, archive_name = tar_members(subject, session)
    if not (bids_dir / input_rel).is_dir():
        raise FileNotFoundError(f"Requested BIDS input not found: {bids_dir / input_rel}")

    archive = archive_dir / archive_name
    include_dirs = include_dirs or []
    exclude_dirs = exclude_dirs or []
    members = [m for m in ROOT_METADATA if (bids_dir / m).exists()]
    print(f"Creating BIDS archive: {archive}")
    if include_dirs:
        print(f"  Including BIDS directories only: {', '.join(include_dirs)}")
    if exclude_dirs:
        print(f"  Excluding BIDS directories: {', '.join(exclude_dirs)}")
    with tarfile.open(archive, "w:gz") as tf:
        for member in members:
            tf.add(bids_dir / member, arcname=member, filter=tarinfo_without_macos_metadata)
        if include_dirs or exclude_dirs:
            if "/" in input_rel:
                add_filtered_bids_tree(tf, bids_dir, input_rel, include_dirs, exclude_dirs)
            else:
                add_filtered_subject_tree(tf, bids_dir, input_rel, include_dirs, exclude_dirs)
        else:
            tf.add(bids_dir / input_rel, arcname=input_rel, filter=tarinfo_without_macos_metadata)
        if anat_derivatives_dir is not None:
            derivative_anat = anat_derivatives_dir / input_rel / "anat"
            if not derivative_anat.is_dir():
                raise FileNotFoundError(f"Requested anatomical derivatives not found: {derivative_anat}")
            target_anat = f"{input_rel}/anat"
            print(f"  Overlaying anatomical derivatives from: {derivative_anat}")
            for item in sorted(derivative_anat.iterdir()):
                if item.is_file():
                    tf.add(
                        item,
                        arcname=f"{target_anat}/{item.name}",
                        filter=tarinfo_without_macos_metadata,
                    )
    return archive


def tarinfo_without_macos_metadata(tarinfo: tarfile.TarInfo) -> tarfile.TarInfo | None:
    name = Path(tarinfo.name).name
    if name == ".DS_Store" or name.startswith("._"):
        return None
    return tarinfo


def write_queue_file(rows: list[list[str]], path: Path) -> None:
    with path.open("w", newline="") as f:
        csv.writer(f, lineterminator="\n").writerows(rows)


def read_stage_manifest(path: Path) -> list[list[str]]:
    rows: list[list[str]] = []
    with path.open(newline="") as f:
        reader = csv.reader(line for line in f if line.strip() and not line.lstrip().startswith("#"))
        for row in reader:
            if not row:
                continue
            if row[0].strip().lower() == "subject":
                continue
            if len(row) < 4:
                raise ValueError(f"Manifest row must have 4 columns: {row}")
            rows.append([c.strip() for c in row[:4]])
    if not rows:
        raise ValueError(f"No staged inputs found in manifest: {path}")
    return rows


def parse_staged_bids_archive_name(path: Path) -> list[str] | None:
    match = STAGED_BIDS_ARCHIVE_RE.match(path.name)
    if not match:
        return None
    subject = norm_label(match.group("subject"), "sub-")
    session = norm_label(match.group("session"), "ses-")
    session_name = "none" if is_empty_label(session) else session
    return [subject, session_name, session_name, path.name]


def infer_stage_manifest_rows(staging_dir: Path) -> list[list[str]]:
    rows: list[list[str]] = []
    for archive in sorted(staging_dir.glob("sub-*_bids.tar.gz")):
        row = parse_staged_bids_archive_name(archive)
        if row:
            rows.append(row)
    for archive in sorted(staging_dir.glob("sub-*_ses-*_bids.tar.gz")):
        row = parse_staged_bids_archive_name(archive)
        if row:
            rows.append(row)
    rows = sorted({tuple(row): row for row in rows}.values())
    if not rows:
        raise FileNotFoundError(
            f"No staged BIDS archives found in {staging_dir}. Expected names like "
            "sub-10021_ses-01_bids.tar.gz or sub-10021_bids.tar.gz."
        )
    return rows


def yaml_key_from_line(line: str) -> str | None:
    stripped = line.split("#", 1)[0].rstrip()
    if not stripped.strip() or stripped.lstrip().startswith("- "):
        return None
    content = stripped.strip()
    if ":" not in content:
        return None
    key = content.split(":", 1)[0].strip()
    if not key or " " in key and not (key.startswith(("'", '"')) and key.endswith(("'", '"'))):
        return None
    return key.strip("'\"")


def check_yaml_duplicate_keys(path: Path) -> None:
    if path.suffix.lower() not in {".yaml", ".yml"}:
        return
    stack: list[tuple[int, tuple[str, ...], dict[str, tuple[int, int]]]] = [
        (-1, (), {})
    ]

    for lineno, raw_line in enumerate(path.read_text().splitlines(), start=1):
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        key = yaml_key_from_line(raw_line)
        if key is None:
            continue

        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent_seen = stack[-1][2]
        if key in parent_seen:
            first_lineno, first_col = parent_seen[key]
            raise ValueError(
                f"Duplicate YAML key '{key}' in {path} at line {lineno}, column {indent + 1}; "
                f"first seen at line {first_lineno}, column {first_col}. Merge repeated sections."
            )
        parent_seen[key] = (lineno, indent + 1)
        stack.append((indent, (*stack[-1][1], key), {}))


def find_default_config(staging_dir: Path) -> Path:
    candidates = sorted(
        p for p in staging_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".yaml", ".yml", ".json"}
    )
    if not candidates:
        raise FileNotFoundError(
            f"No YAML/JSON config found in {staging_dir}. Pass --config-file explicitly."
        )
    return candidates[0]


def find_default_license(staging_dir: Path) -> Path:
    candidates = [staging_dir / "license.txt"] + sorted(staging_dir.glob("*license*.txt"))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"No FreeSurfer license found in {staging_dir}. Pass --freesurfer-license explicitly."
    )


def htcondor_transfer_source(value: str) -> str:
    if "://" in value or not value.startswith("/"):
        return value
    if value.startswith("/staging/"):
        return "osdf:///chtc" + value
    if value.startswith("/projects/"):
        return "file://" + value
    return value


def transfer_source_name(value: str) -> str:
    if not value:
        return ""
    cleaned = value.rstrip("/")
    if "://" in cleaned:
        cleaned = cleaned.rsplit("?", 1)[0].rsplit("#", 1)[0]
    return cleaned.rsplit("/", 1)[-1]


def prepare_optional_transfer_file(value: str, *, label: str) -> tuple[str, str]:
    if not value:
        return "", ""
    if "://" in value:
        name = transfer_source_name(value)
        if not name:
            raise ValueError(f"{label} URL must end with a filename: {value}")
        return value, name
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    return str(path), path.name


def local_path_from_transfer_source(value: str) -> str:
    if value.startswith("osdf:///chtc/staging/"):
        return "/staging/" + value.removeprefix("osdf:///chtc/staging/")
    if value.startswith("file:///"):
        return value.removeprefix("file://")
    return value


def infer_transfer_uri(staging_value: str) -> str:
    if "://" in staging_value:
        return staging_value.rstrip("/")
    return htcondor_transfer_source(staging_value).rstrip("/")


def submit_bool(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return "true"
    if text in {"0", "false", "no", "off"}:
        return "false"
    return str(value)


def positive_int(value: object) -> bool:
    try:
        return int(str(value))
    except ValueError:
        return False


def generate_submit_file(
    path: Path,
    *,
    executable: str,
    container_image: str,
    config_file: str,
    config_name: str,
    freesurfer_license: str,
    license_name: str,
    gnl_coeff_file: str,
    gnl_coeff_name: str,
    queue_file: str,
    cpus: int,
    gpus: int,
    memory_gb: int,
    disk_gb: int,
    pipeline: str,
    require_dwi: str,
    t1w_match_acq: str,
    t1w_match_desc: str,
    requirements: str,
    getenv: str,
    gpu_minimum_capability: str,
    want_flocking: str,
    want_glidein: str,
    want_gpu_lab: str,
    gpu_job_length: str,
    notification: str,
    notify_user: str,
    log_dir: str,
    output_directory: str,
    output_destination: str,
) -> None:
    container_image = htcondor_transfer_source(container_image)
    config_file = htcondor_transfer_source(config_file)
    freesurfer_license = htcondor_transfer_source(freesurfer_license)
    gnl_coeff_file = htcondor_transfer_source(gnl_coeff_file) if gnl_coeff_file else ""
    gnl_transfer = ", $(gnl_coeff_file)" if gnl_coeff_file else ""
    gpu_lines = ""
    if positive_int(gpus):
        gpu_lines = f"request_gpus = $(gpus)\n"
        if gpu_minimum_capability:
            gpu_lines += f"gpus_minimum_capability = {gpu_minimum_capability}\n"
    classad_lines = ""
    if want_flocking:
        classad_lines += f"+WantFlocking = {submit_bool(want_flocking)}\n"
    if want_glidein:
        classad_lines += f"+WantGlideIn = {submit_bool(want_glidein)}\n"
    if want_gpu_lab:
        classad_lines += f"+WantGPULab = {submit_bool(want_gpu_lab)}\n"
    if gpu_job_length:
        classad_lines += f'+GPUJobLength = "{gpu_job_length}"\n'
    notification_lines = ""
    if notification:
        notification_lines += f"notification = {notification}\n"
    if notify_user:
        notification_lines += f"notify_user = {notify_user}\n"
    log_prefix = log_dir.rstrip("/")
    if log_prefix:
        output_line = f"output = {log_prefix}/$(SUBJECT)_$(SESSION_NAME).out"
        error_line = f"error = {log_prefix}/$(SUBJECT)_$(SESSION_NAME).err"
        log_line = f"log = {log_prefix}/$(SUBJECT)_$(SESSION_NAME).log"
    else:
        output_line = "output = qneuro_sub-$(SUBJECT)_ses-$(SESSION_NAME).$(Cluster).$(Process).out"
        error_line = "error = qneuro_sub-$(SUBJECT)_ses-$(SESSION_NAME).$(Cluster).$(Process).err"
        log_line = "log = qneuro_sub-$(SUBJECT)_ses-$(SESSION_NAME).$(Cluster).log"
    output_target = output_destination or output_directory
    if output_target and "://" in output_target:
        output_directory_line = f"output_destination = {output_target.rstrip('/')}\n"
    elif output_target:
        output_directory_line = f"output_directory = {output_target.rstrip('/')}\n"
    else:
        output_directory_line = ""
    requirements_line = f"requirements = {requirements}\n" if requirements else ""
    text = f"""# Generated by qneuro_condor.py. Edit the Python script inputs, not this file.
universe = vanilla
executable = {executable}
getenv = {submit_bool(getenv)}
{requirements_line}

container_image = {container_image}
config_file = {config_file}
config_name = {config_name}
freesurfer_license = {freesurfer_license}
license_name = {license_name}
gnl_coeff_file = {gnl_coeff_file}
gnl_coeff_name = {gnl_coeff_name}

pipeline = {pipeline}
cpus = {cpus}
gpus = {gpus}
memory_gb = {memory_gb}
disk_gb = {disk_gb}
require_dwi = {require_dwi}
t1w_match_acq = {t1w_match_acq}
t1w_match_desc = {t1w_match_desc}

arguments = run $(SUBJECT) $(SESSION) $(cpus) $(memory_gb) $(pipeline) $(config_name) $(require_dwi) $(license_name) $(gnl_coeff_name) $(t1w_match_acq) $(t1w_match_desc)

should_transfer_files = YES
when_to_transfer_output = ON_EXIT
transfer_input_files = $(config_file), $(freesurfer_license){gnl_transfer}, $(bids_input)
transfer_output_files = qneuro_outputs_sub-$(SUBJECT)_ses-$(SESSION_NAME).tar.gz
{output_directory_line}

request_cpus = $(cpus)
request_memory = $(memory_gb)GB
request_disk = $(disk_gb)GB
{gpu_lines}
{classad_lines}
{notification_lines}

{output_line}
{error_line}
{log_line}

queue SUBJECT,SESSION,SESSION_NAME,bids_input from {queue_file}
"""
    path.write_text(text)


def build_queue_rows(args: argparse.Namespace, archive_dir: Path) -> tuple[list[list[str]], list[Path]]:
    bids_dir = Path(args.bids_dir).resolve()
    transfer_uri = (args.transfer_uri or "").rstrip("/")
    include_dirs = split_csvish(getattr(args, "bids_include_dirs", ""))
    exclude_dirs = split_csvish(getattr(args, "bids_exclude_dirs", ""))
    anat_derivatives_dir = getattr(args, "anat_derivatives_dir", "")
    derivative_root = Path(anat_derivatives_dir).expanduser().resolve() if anat_derivatives_dir else None
    rows: list[list[str]] = []
    archives: list[Path] = []

    for subject_raw, session_raw in selected_pairs(args):
        subject = norm_label(subject_raw, "sub-")
        session = norm_label(session_raw, "ses-")
        session_arg = "none" if is_empty_label(session) else session
        archive = create_bids_archive(
            bids_dir,
            subject,
            session_arg,
            archive_dir,
            include_dirs=include_dirs,
            exclude_dirs=exclude_dirs,
            anat_derivatives_dir=derivative_root,
        )
        archives.append(archive)
        bids_input = f"{transfer_uri}/{archive.name}" if transfer_uri else htcondor_transfer_source(str(archive))
        rows.append([subject, session_arg, session_arg, bids_input])

    if not rows:
        raise RuntimeError("No subject/session rows were selected.")
    return rows, archives


def build_stage_rows(args: argparse.Namespace, archive_dir: Path) -> tuple[list[list[str]], list[Path]]:
    bids_dir = Path(args.bids_dir).resolve()
    include_dirs = split_csvish(getattr(args, "bids_include_dirs", ""))
    exclude_dirs = split_csvish(getattr(args, "bids_exclude_dirs", ""))
    anat_derivatives_dir = getattr(args, "anat_derivatives_dir", "")
    derivative_root = Path(anat_derivatives_dir).expanduser().resolve() if anat_derivatives_dir else None
    rows: list[list[str]] = []
    archives: list[Path] = []

    for subject_raw, session_raw in selected_pairs(args):
        subject = norm_label(subject_raw, "sub-")
        session = norm_label(session_raw, "ses-")
        session_arg = "none" if is_empty_label(session) else session
        archive = create_bids_archive(
            bids_dir,
            subject,
            session_arg,
            archive_dir,
            include_dirs=include_dirs,
            exclude_dirs=exclude_dirs,
            anat_derivatives_dir=derivative_root,
        )
        archives.append(archive)
        rows.append([subject, session_arg, session_arg, archive.name])

    if not rows:
        raise RuntimeError("No subject/session rows were selected.")
    return rows, archives


def create_stage_bundle(bundle: Path, files: list[Path]) -> None:
    used_names: set[str] = set()
    with tarfile.open(bundle, "w:gz") as tf:
        for file in files:
            arcname = file.name
            if arcname in used_names:
                raise ValueError(f"Cannot stage duplicate bundle member name: {arcname}")
            used_names.add(arcname)
            tf.add(file, arcname=arcname)


def add_submit_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--container-image", required=True)
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--freesurfer-license", required=True)
    parser.add_argument(
        "--gnl-coeff-file",
        default="",
        help=(
            "Optional gradient nonlinearity coefficient file to transfer with each job. "
            "Reference it from the container YAML as either its basename or config/<basename>."
        ),
    )
    parser.add_argument("--bids-dir", required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--subject")
    selection.add_argument("--subjects", help="Comma-separated subject labels, for example: 10021,10022")
    selection.add_argument("--subjects-file")
    parser.add_argument("--session", default="none")
    parser.add_argument(
        "--sessions",
        default="",
        help=(
            "Comma-separated session labels for --subjects. Use one value for all "
            "subjects, or one value per subject."
        ),
    )
    parser.add_argument("--transfer-uri", default="")
    parser.add_argument("--pipeline", default="dmri")
    parser.add_argument(
        "--bids-include-dirs",
        default="",
        help="Comma-separated BIDS datatype directories to package, for example: anat,dwi.",
    )
    parser.add_argument(
        "--bids-exclude-dirs",
        default="",
        help="Comma-separated BIDS datatype directories to omit when packaging.",
    )
    parser.add_argument(
        "--anat-derivatives-dir",
        default="",
        help="Derivative dataset root whose sub-*/[ses-*]/anat files are added to each archive.",
    )
    parser.add_argument(
        "--t1w-match-acq",
        default="",
        help="Select the staged T1w by its BIDS acq entity, for example MPnRAGE.",
    )
    parser.add_argument(
        "--t1w-match-desc",
        default="",
        help="Select the staged T1w by its BIDS desc entity, for example MPnRAGE.",
    )
    parser.add_argument("--cpus", type=int, default=8)
    parser.add_argument("--gpus", type=int, default=0)
    parser.add_argument("--memory-gb", type=int, default=32)
    parser.add_argument("--disk-gb", type=int, default=40)
    parser.add_argument("--require-dwi", default="true", choices=("true", "false"))
    parser.add_argument(
        "--requirements",
        default='(OpSys == "LINUX") && (Arch == "X86_64") && (HasCHTCStaging == true)',
    )
    parser.add_argument("--getenv", default="true")
    parser.add_argument("--gpu-minimum-capability", default="8.0")
    parser.add_argument("--want-flocking", default="true")
    parser.add_argument("--want-glidein", default="true")
    parser.add_argument("--want-gpu-lab", default="false")
    parser.add_argument("--gpu-job-length", default="medium")
    parser.add_argument("--notification", default="")
    parser.add_argument("--notify-user", default="")
    parser.add_argument("--log-dir", default="")
    parser.add_argument("--output-directory", default="")
    parser.add_argument("--output-destination", default="")
    parser.add_argument("--submit-file-name", default="qneuro_generated.sub")
    parser.add_argument("--queue-file-name", default="qneuro_inputs.csv")
    parser.add_argument("--no-submit", action="store_true")


def cmd_submit_local(args: argparse.Namespace) -> int:
    package_dir = Path(args.package_dir).resolve()
    package_dir.mkdir(parents=True, exist_ok=True)

    submit_file = Path(args.submit_file_name).resolve()
    queue_file = submit_file.parent / args.queue_file_name
    rows, _archives = build_queue_rows(args, package_dir)
    write_queue_file(rows, queue_file)

    script_path = Path(__file__).resolve()
    config_file = Path(args.config_file).resolve()
    license_file = Path(args.freesurfer_license).resolve()
    gnl_coeff_file, gnl_coeff_name = prepare_optional_transfer_file(
        args.gnl_coeff_file,
        label="GNL coefficient file",
    )
    check_yaml_duplicate_keys(config_file)
    generate_submit_file(
        submit_file,
        executable=str(script_path),
        container_image=args.container_image,
        config_file=str(config_file),
        config_name=config_file.name,
        freesurfer_license=str(license_file),
        license_name=license_file.name,
        gnl_coeff_file=gnl_coeff_file,
        gnl_coeff_name=gnl_coeff_name,
        queue_file=queue_file.name,
        cpus=args.cpus,
        gpus=args.gpus,
        memory_gb=args.memory_gb,
        disk_gb=args.disk_gb,
        pipeline=args.pipeline,
        require_dwi=args.require_dwi,
        t1w_match_acq=getattr(args, "t1w_match_acq", ""),
        t1w_match_desc=getattr(args, "t1w_match_desc", ""),
        requirements=args.requirements,
        getenv=args.getenv,
        gpu_minimum_capability=args.gpu_minimum_capability,
        want_flocking=args.want_flocking,
        want_glidein=args.want_glidein,
        want_gpu_lab=args.want_gpu_lab,
        gpu_job_length=args.gpu_job_length,
        notification=args.notification,
        notify_user=args.notify_user,
        log_dir=args.log_dir,
        output_directory=args.output_directory,
        output_destination=args.output_destination,
    )

    print(f"Wrote submit file: {submit_file}")
    print(f"Wrote queue file: {queue_file}")
    if not args.no_submit:
        subprocess.run(["condor_submit", submit_file.name], cwd=submit_file.parent, check=True)
    return 0


def cmd_push_submit(args: argparse.Namespace) -> int:
    config_file = Path(args.config_file).expanduser().resolve()
    license_file = Path(args.freesurfer_license).expanduser().resolve()
    if not config_file.is_file():
        raise FileNotFoundError(f"Config file not found: {config_file}")
    if not license_file.is_file():
        raise FileNotFoundError(f"FreeSurfer license not found: {license_file}")
    gnl_coeff_file, gnl_coeff_name = prepare_optional_transfer_file(
        args.gnl_coeff_file,
        label="GNL coefficient file",
    )
    remote_gnl_coeff_file = gnl_coeff_file
    gnl_scp_files: list[str] = []
    if gnl_coeff_file and "://" not in gnl_coeff_file:
        remote_gnl_coeff_file = gnl_coeff_name
        gnl_scp_files.append(gnl_coeff_file)
    check_yaml_duplicate_keys(config_file)

    work_dir = Path(tempfile.mkdtemp(prefix="qneuro_condor."))
    archive_dir = work_dir / "archives"
    archive_dir.mkdir()
    queue_file = work_dir / args.queue_file_name
    submit_file = work_dir / args.submit_file_name
    remote_submit_dir = args.remote_submit_dir.rstrip("/")
    remote_package_dir = args.remote_package_dir.rstrip("/")
    remote_script_name = Path(__file__).name

    rows, archives = build_queue_rows(args, archive_dir)
    write_queue_file(rows, queue_file)
    generate_submit_file(
        submit_file,
        executable=remote_script_name,
        container_image=args.container_image,
        config_file=config_file.name,
        config_name=config_file.name,
        freesurfer_license=license_file.name,
        license_name=license_file.name,
        gnl_coeff_file=remote_gnl_coeff_file,
        gnl_coeff_name=gnl_coeff_name,
        queue_file=queue_file.name,
        cpus=args.cpus,
        gpus=args.gpus,
        memory_gb=args.memory_gb,
        disk_gb=args.disk_gb,
        pipeline=args.pipeline,
        require_dwi=args.require_dwi,
        t1w_match_acq=getattr(args, "t1w_match_acq", ""),
        t1w_match_desc=getattr(args, "t1w_match_desc", ""),
        requirements=args.requirements,
        getenv=args.getenv,
        gpu_minimum_capability=args.gpu_minimum_capability,
        want_flocking=args.want_flocking,
        want_glidein=args.want_glidein,
        want_gpu_lab=args.want_gpu_lab,
        gpu_job_length=args.gpu_job_length,
        notification=args.notification,
        notify_user=args.notify_user,
        log_dir=args.log_dir,
        output_directory=args.output_directory,
        output_destination=args.output_destination,
    )

    mkdir_cmd = f"mkdir -p {shlex.quote(remote_submit_dir)} {shlex.quote(remote_package_dir)}"
    subprocess.run(["ssh", args.submit_host, mkdir_cmd], check=True)
    subprocess.run(["scp", *map(str, archives), f"{args.submit_host}:{remote_package_dir}/"], check=True)
    subprocess.run(
        [
            "scp",
            str(Path(__file__).resolve()),
            str(submit_file),
            str(queue_file),
            str(config_file),
            str(license_file),
            *gnl_scp_files,
            f"{args.submit_host}:{remote_submit_dir}/",
        ],
        check=True,
    )
    chmod_cmd = f"chmod +x {shlex.quote(remote_submit_dir + '/' + remote_script_name)}"
    subprocess.run(["ssh", args.submit_host, chmod_cmd], check=True)

    print(f"Wrote generated submit file locally: {submit_file}")
    print(f"Pushed submit bundle to: {args.submit_host}:{remote_submit_dir}")
    if not args.no_submit:
        submit_cmd = f"cd {shlex.quote(remote_submit_dir)} && condor_submit {shlex.quote(submit_file.name)}"
        subprocess.run(["ssh", args.submit_host, submit_cmd], check=True)
    return 0


def cmd_stage_remote(args: argparse.Namespace) -> int:
    work_dir = Path(tempfile.mkdtemp(prefix="qneuro_stage."))
    archive_dir = work_dir / "archives"
    archive_dir.mkdir()
    manifest = work_dir / args.manifest_name

    rows, archives = build_stage_rows(args, archive_dir)
    write_queue_file([["subject", "session", "session_name", "archive_name"], *rows], manifest)

    remote_stage_dir = args.remote_stage_dir.rstrip("/")
    if args.create_remote_dir:
        mkdir_cmd = f"mkdir -p {shlex.quote(remote_stage_dir)}"
        subprocess.run(["ssh", args.remote_host, mkdir_cmd], check=True)

    payload = [*archives, manifest]
    if args.include_support_files:
        if not args.config_file or not args.freesurfer_license:
            raise ValueError(
                "--include-support-files requires --config-file and --freesurfer-license"
            )
        config_file = Path(args.config_file).expanduser().resolve()
        license_file = Path(args.freesurfer_license).expanduser().resolve()
        if not config_file.is_file():
            raise FileNotFoundError(f"Config file not found: {config_file}")
        if not license_file.is_file():
            raise FileNotFoundError(f"FreeSurfer license not found: {license_file}")
        payload.extend([config_file, license_file])
        if args.gnl_coeff_file:
            gnl_coeff_file, _gnl_coeff_name = prepare_optional_transfer_file(
                args.gnl_coeff_file,
                label="GNL coefficient file",
            )
            if "://" in gnl_coeff_file:
                raise ValueError(
                    "--include-support-files with stage-remote requires a local --gnl-coeff-file"
                )
            payload.append(Path(gnl_coeff_file))
        if args.copy_script:
            payload.append(Path(__file__).resolve())

    if args.bundle:
        bundle = work_dir / args.bundle_name
        create_stage_bundle(bundle, payload)
        subprocess.run(["scp", str(bundle), f"{args.remote_host}:{remote_stage_dir}/"], check=True)
        print(f"Created local staging bundle in: {bundle}")
        print(f"Copied one staging bundle to: {args.remote_host}:{remote_stage_dir}/{bundle.name}")
        print("Next, log into CHTC and extract it:")
        print(f"  cd {remote_stage_dir}")
        print(f"  tar -xzf {bundle.name}")
        print("Then submit from the extracted staged files:")
        print(f"  python /path/on/chtc/qneuro_condor.py submit-staged --staging-dir {remote_stage_dir} --container-image /staging/<user>/qneuro.sif")
        return 0

    subprocess.run(["scp", *map(str, payload), f"{args.remote_host}:{remote_stage_dir}/"], check=True)
    print(f"Created local staging bundle in: {work_dir}")
    print(f"Copied staged payload to: {args.remote_host}:{remote_stage_dir}")
    print(f"Next, log into CHTC and run submit-staged against: {remote_stage_dir}/{manifest.name}")
    return 0


def cmd_submit_staged(args: argparse.Namespace) -> int:
    staging_source = args.staging_dir.rstrip("/")
    staging_dir = Path(local_path_from_transfer_source(staging_source)).expanduser().resolve()
    submit_dir = Path(args.submit_dir).expanduser().resolve()
    submit_dir.mkdir(parents=True, exist_ok=True)

    explicit_manifest = bool(args.manifest)
    manifest = Path(args.manifest).expanduser() if explicit_manifest else staging_dir / args.manifest_name
    if not manifest.is_absolute():
        manifest = staging_dir / manifest
    if explicit_manifest and not manifest.is_file():
        raise FileNotFoundError(f"Stage manifest not found: {manifest}")

    config_file = Path(args.config_file).expanduser().resolve() if args.config_file else find_default_config(staging_dir)
    license_file = (
        Path(args.freesurfer_license).expanduser().resolve()
        if args.freesurfer_license
        else find_default_license(staging_dir)
    )
    if not config_file.is_file():
        raise FileNotFoundError(f"Config file not found: {config_file}")
    if not license_file.is_file():
        raise FileNotFoundError(f"FreeSurfer license not found: {license_file}")
    gnl_coeff_file, gnl_coeff_name = prepare_optional_transfer_file(
        args.gnl_coeff_file,
        label="GNL coefficient file",
    )
    check_yaml_duplicate_keys(config_file)

    transfer_uri = args.transfer_uri.rstrip("/") or infer_transfer_uri(staging_source)
    queue_rows: list[list[str]] = []
    if manifest.is_file():
        manifest_rows = filter_manifest_rows(read_stage_manifest(manifest), args)
        print(f"Using staged manifest: {manifest}")
    else:
        manifest_rows = filter_manifest_rows(infer_stage_manifest_rows(staging_dir), args)
        print(f"No staged manifest found at {manifest}; inferred inputs from staged archive names.")
    for subject, session, session_name, archive_name in manifest_rows:
        archive_path = staging_dir / archive_name
        if transfer_uri:
            bids_input = f"{transfer_uri}/{archive_name}"
        else:
            if not archive_path.is_file():
                raise FileNotFoundError(f"Staged archive not found: {archive_path}")
            bids_input = htcondor_transfer_source(str(archive_path))
        queue_rows.append([subject, session, session_name, bids_input])

    queue_file = submit_dir / args.queue_file_name
    submit_file = submit_dir / args.submit_file_name
    executable_file = submit_dir / Path(__file__).name
    current_script = Path(__file__).resolve()
    if current_script != executable_file.resolve():
        shutil.copy2(current_script, executable_file)
        executable_file.chmod(0o755)

    write_queue_file(queue_rows, queue_file)
    generate_submit_file(
        submit_file,
        executable=str(executable_file),
        container_image=args.container_image,
        config_file=str(config_file),
        config_name=config_file.name,
        freesurfer_license=str(license_file),
        license_name=license_file.name,
        gnl_coeff_file=gnl_coeff_file,
        gnl_coeff_name=gnl_coeff_name,
        queue_file=queue_file.name,
        cpus=args.cpus,
        gpus=args.gpus,
        memory_gb=args.memory_gb,
        disk_gb=args.disk_gb,
        pipeline=args.pipeline,
        require_dwi=args.require_dwi,
        t1w_match_acq=getattr(args, "t1w_match_acq", ""),
        t1w_match_desc=getattr(args, "t1w_match_desc", ""),
        requirements=args.requirements,
        getenv=args.getenv,
        gpu_minimum_capability=args.gpu_minimum_capability,
        want_flocking=args.want_flocking,
        want_glidein=args.want_glidein,
        want_gpu_lab=args.want_gpu_lab,
        gpu_job_length=args.gpu_job_length,
        notification=args.notification,
        notify_user=args.notify_user,
        log_dir=args.log_dir,
        output_directory=args.output_directory,
        output_destination=args.output_destination,
    )

    print(f"Wrote submit file: {submit_file}")
    print(f"Wrote queue file: {queue_file}")
    if not args.no_submit:
        subprocess.run(["condor_submit", submit_file.name], cwd=submit_dir, check=True)
    return 0


def parse_output_archive_name(path: Path) -> tuple[str, str] | None:
    match = OUTPUT_ARCHIVE_RE.match(path.name)
    if not match:
        return None
    subject = norm_label(match.group("subject"), "sub-")
    session = norm_label(match.group("session"), "ses-")
    session = "" if is_empty_label(session) else session
    return subject, session


def output_archive_patterns(args: argparse.Namespace) -> list[str]:
    if not has_subject_filter(args):
        return ["qneuro_outputs_sub-*_ses-*.tar.gz"]

    patterns: list[str] = []
    for subject_raw, session_raw in selected_pairs(args):
        subject = norm_label(subject_raw, "sub-")
        session = norm_label(session_raw, "ses-")
        if is_empty_label(session):
            patterns.append(f"qneuro_outputs_sub-{subject}_ses-*.tar.gz")
        else:
            patterns.append(f"qneuro_outputs_sub-{subject}_ses-{session}.tar.gz")
    return patterns


def archive_selected(path: Path, args: argparse.Namespace) -> bool:
    parsed = parse_output_archive_name(path)
    if parsed is None:
        return False
    if not has_subject_filter(args):
        return True

    subject, session = parsed
    requested: set[tuple[str, str]] = set()
    wildcard_subjects: set[str] = set()
    for subject_raw, session_raw in selected_pairs(args):
        requested_subject = norm_label(subject_raw, "sub-")
        requested_session = norm_label(session_raw, "ses-")
        if is_empty_label(requested_session):
            wildcard_subjects.add(requested_subject)
        else:
            requested.add((requested_subject, requested_session))
    return subject in wildcard_subjects or (subject, session) in requested


def download_remote_outputs(args: argparse.Namespace, outputs_dir: Path) -> None:
    if not args.remote_host and not args.remote_outputs_dir:
        return
    if not args.remote_host or not args.remote_outputs_dir:
        raise ValueError("--remote-host and --remote-outputs-dir must be used together")

    outputs_dir.mkdir(parents=True, exist_ok=True)
    remote_dir = args.remote_outputs_dir.rstrip("/")
    patterns = output_archive_patterns(args)
    remote_command = " ".join(
        [
            "cd",
            shlex.quote(remote_dir),
            "&&",
            "tar",
            "-cf",
            "-",
            "--wildcards",
            "--",
            *(shlex.quote(pattern) for pattern in patterns),
        ]
    )
    print(f"Copying Condor output archive(s) in one SSH transfer to: {outputs_dir}")
    process = subprocess.Popen(
        ["ssh", args.remote_host, remote_command],
        stdout=subprocess.PIPE,
    )
    if process.stdout is None:
        process.kill()
        raise RuntimeError("Could not read the remote output tar stream")

    received = 0
    try:
        with process.stdout, tarfile.open(fileobj=process.stdout, mode="r|") as tf:
            for member in tf:
                name = PurePosixPath(member.name)
                if (
                    not member.isfile()
                    or name.is_absolute()
                    or len(name.parts) != 1
                    or parse_output_archive_name(Path(name.name)) is None
                ):
                    raise ValueError(f"Unexpected file in remote output bundle: {member.name}")
                destination = outputs_dir / name.name
                source = tf.extractfile(member)
                if source is None:
                    raise ValueError(f"Could not read remote output archive: {member.name}")
                print(f"Receiving {name.name} ({member.size / (1024 ** 3):.2f} GiB)")
                temporary_path: Path | None = None
                try:
                    with source, tempfile.NamedTemporaryFile(
                        dir=outputs_dir,
                        prefix=f".{name.name}.",
                        suffix=".part",
                        delete=False,
                    ) as output:
                        temporary_path = Path(output.name)
                        shutil.copyfileobj(source, output)
                    temporary_path.replace(destination)
                    received += 1
                except BaseException:
                    if temporary_path is not None:
                        temporary_path.unlink(missing_ok=True)
                    raise
    except BaseException:
        process.kill()
        process.wait()
        raise

    returncode = process.wait()
    if returncode:
        raise subprocess.CalledProcessError(
            returncode,
            ["ssh", args.remote_host, remote_command],
        )
    print(f"Received {received} Condor output archive(s).")


def collect_target_dir(args: argparse.Namespace) -> Path:
    if args.derivatives_dir:
        return Path(args.derivatives_dir).expanduser().resolve()
    if not args.bids_dir:
        raise ValueError("Pass either --bids-dir or --derivatives-dir")
    return Path(args.bids_dir).expanduser().resolve() / "derivatives" / args.derivative_name


def safe_tar_member_parts(member_name: str) -> list[str]:
    parts = [
        part
        for part in PurePosixPath(member_name).parts
        if part not in {"", "."}
    ]
    if not parts:
        return []
    if PurePosixPath(member_name).is_absolute() or any(part == ".." for part in parts):
        raise ValueError(f"Unsafe archive member path: {member_name}")
    return parts


def normalize_output_member_path(
    member_name: str,
    *,
    subject: str,
    session: str,
    derivative_name: str,
) -> Path | None:
    parts = safe_tar_member_parts(member_name)
    if not parts:
        return None

    while parts and parts[0] in {"out", "."}:
        parts = parts[1:]
    if not parts:
        return None

    if parts[0] == "derivatives":
        parts = parts[1:]
        if parts and parts[0] == derivative_name:
            parts = parts[1:]
    elif parts[0] == derivative_name:
        parts = parts[1:]
    if not parts:
        return None

    expected_subject = f"sub-{subject}"
    expected_session = f"ses-{session}" if session else ""
    if parts[0].startswith("sub-"):
        if parts[0] != expected_subject:
            raise ValueError(
                f"Archive member subject {parts[0]} does not match archive name {expected_subject}: "
                f"{member_name}"
            )
        if expected_session:
            if len(parts) < 2 or not parts[1].startswith("ses-"):
                parts = [parts[0], expected_session, *parts[1:]]
            elif parts[1] != expected_session:
                raise ValueError(
                    f"Archive member session {parts[1]} does not match archive name {expected_session}: "
                    f"{member_name}"
                )
        return Path(*parts)

    if parts[0] in {"dataset_description.json", "README", "CHANGES", "logs", "figures"}:
        return Path(*parts)

    prefix = [expected_subject]
    if expected_session:
        prefix.append(expected_session)
    return Path(*prefix, *parts)


def ensure_derivative_dataset_description(target_dir: Path, derivative_name: str) -> None:
    desc = target_dir / "dataset_description.json"
    if desc.exists():
        return
    desc.write_text(
        json.dumps(
            {
                "Name": derivative_name,
                "BIDSVersion": "1.8.0",
                "DatasetType": "derivative",
                "GeneratedBy": [{"Name": "qmri-neuropipe"}],
            },
            indent=2,
        )
        + "\n"
    )


def install_output_archive(
    archive: Path,
    target_dir: Path,
    *,
    derivative_name: str,
    overwrite: bool,
    dry_run: bool,
) -> int:
    parsed = parse_output_archive_name(archive)
    if parsed is None:
        raise ValueError(f"Output archive name does not match qneuro pattern: {archive.name}")
    subject, session = parsed
    installed = 0

    with tarfile.open(archive, "r:*") as tf:
        for member in tf.getmembers():
            if member.issym() or member.islnk():
                print(f"Skipping archive link member: {member.name}")
                continue
            rel = normalize_output_member_path(
                member.name,
                subject=subject,
                session=session,
                derivative_name=derivative_name,
            )
            if rel is None:
                continue
            dest = target_dir / rel
            if member.isdir():
                if dry_run:
                    print(f"Would create directory: {dest}")
                else:
                    dest.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                continue
            if dest.exists() and not overwrite:
                if dest.is_file() and tar_member_matches_file(tf, member, dest):
                    continue
                raise FileExistsError(f"Refusing to overwrite existing file without --overwrite: {dest}")
            if dry_run:
                print(f"Would install {archive.name}:{member.name} -> {dest}")
                installed += 1
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            source = tf.extractfile(member)
            if source is None:
                continue
            with source, dest.open("wb") as out:
                shutil.copyfileobj(source, out)
            installed += 1
    return installed


def tar_member_matches_file(tf: tarfile.TarFile, member: tarfile.TarInfo, path: Path) -> bool:
    try:
        if member.size != path.stat().st_size:
            return False
    except OSError:
        return False

    source = tf.extractfile(member)
    if source is None:
        return False
    with source, path.open("rb") as existing:
        while True:
            source_chunk = source.read(1024 * 1024)
            existing_chunk = existing.read(1024 * 1024)
            if source_chunk != existing_chunk:
                return False
            if not source_chunk:
                return True


def discover_output_archives(outputs_dir: Path, args: argparse.Namespace) -> list[Path]:
    archives = [
        path for path in sorted(outputs_dir.glob("qneuro_outputs_sub-*_ses-*.tar.gz"))
        if path.is_file() and archive_selected(path, args)
    ]
    if not archives:
        raise FileNotFoundError(
            f"No qneuro output archives matched the requested subject/session selection in: {outputs_dir}"
        )
    return archives


def cmd_collect_outputs(args: argparse.Namespace) -> int:
    outputs_dir = Path(args.outputs_dir).expanduser().resolve()
    download_remote_outputs(args, outputs_dir)
    if not outputs_dir.is_dir():
        raise FileNotFoundError(f"Output archive directory not found: {outputs_dir}")

    target_dir = collect_target_dir(args)
    archives = discover_output_archives(outputs_dir, args)
    print(f"Installing {len(archives)} output archive(s) into: {target_dir}")

    total_files = 0
    if not args.dry_run:
        target_dir.mkdir(parents=True, exist_ok=True)
    for archive in archives:
        count = install_output_archive(
            archive,
            target_dir,
            derivative_name=args.derivative_name,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
        )
        total_files += count
        print(f"Installed {count} file(s) from {archive.name}")
    if not args.dry_run:
        ensure_derivative_dataset_description(target_dir, args.derivative_name)

    action = "Would install" if args.dry_run else "Installed"
    print(f"{action} {total_files} file(s) total.")
    return 0


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
        raise FileNotFoundError(f"No YAML/JSON qmri-neuropipe config was transferred into {cwd}")
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

    eprint("Warning: no transferred FreeSurfer license found. FreeSurfer tools may fail.")


def stage_gnl_coeff_file(cwd: Path, gnl_coeff_name: str) -> None:
    if not gnl_coeff_name:
        return
    candidate = cwd / gnl_coeff_name
    if not candidate.is_file():
        eprint(f"Warning: requested GNL coefficient file was not transferred: {gnl_coeff_name}")
        return
    target = cwd / "config" / candidate.name
    target.parent.mkdir(exist_ok=True)
    if candidate.resolve() != target.resolve():
        shutil.copy2(candidate, target)
    print(f"Using GNL coefficient file at {candidate.name} and config/{candidate.name}")


def apply_t1w_match_entities(config_path: Path, *, acq: str = "", desc: str = "") -> None:
    """Set requested anat.input.t1w_match entities in the job-local config."""
    requested = {key: value.strip() for key, value in {"acq": acq, "desc": desc}.items() if value.strip()}
    if not requested:
        return
    import yaml

    payload = yaml.safe_load(config_path.read_text()) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Pipeline config must contain a mapping: {config_path}")
    anat = payload.setdefault("anat", {})
    if not isinstance(anat, dict):
        raise ValueError("Cannot apply --t1w-match-desc because config 'anat' is not a mapping")
    anat_input = anat.setdefault("input", {})
    if not isinstance(anat_input, dict):
        raise ValueError("Cannot apply --t1w-match-desc because config 'anat.input' is not a mapping")
    match = anat_input.setdefault("t1w_match", {})
    if not isinstance(match, dict):
        raise ValueError("Cannot apply --t1w-match-desc because config 'anat.input.t1w_match' is not a mapping")
    entities = match.setdefault("entities", {})
    if not isinstance(entities, dict):
        raise ValueError("Cannot apply --t1w-match-desc because t1w_match.entities is not a mapping")
    entities.update(requested)
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False))
    print(f"Configured anat.input.t1w_match.entities={requested!r}")


def apply_t1w_match_desc(config_path: Path, desc: str) -> None:
    """Backward-compatible wrapper for the original desc-only option."""
    apply_t1w_match_entities(config_path, desc=desc)


def extract_input_archives(cwd: Path, subject: str, session_for_name: str) -> None:
    output_name = f"qneuro_outputs_sub-{subject}_ses-{session_for_name}.tar.gz"
    for archive in sorted(cwd.iterdir()):
        if not archive.is_file() or archive.name == output_name:
            continue
        if archive.name.endswith((".tar.gz", ".tgz", ".tar")):
            print(f"Extracting transferred BIDS archive: {archive.name}")
            with tarfile.open(archive, "r:*") as tf:
                tf.extractall(cwd)


def stage_bids_tree(cwd: Path, subject: str, session: str) -> Path:
    data_dir = cwd / "data"
    data_dir.mkdir(exist_ok=True)

    for root_file in ROOT_METADATA:
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
    elif not (cwd / f"sub-{subject}").is_dir():
        eprint(f"Could not find transferred BIDS input sub-{subject}")
        sys.exit(5)
    else:
        shutil.move(str(cwd / f"sub-{subject}"), str(data_dir / f"sub-{subject}"))

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
                tf.add(
                    item,
                    arcname=item.relative_to(out_dir),
                    recursive=False,
                    filter=tarinfo_without_macos_metadata,
                )
    print(f"Wrote {tarball.name}")


def cmd_run(argv: list[str]) -> int:
    if len(argv) < 5:
        eprint(
            "Usage: qneuro_condor.py run <subject> <session|none> <cpus> "
            "<memory_gb> <pipeline> [config_name] [require_dwi] [license_name] "
            "[gnl_coeff_name] [t1w_match_acq] [t1w_match_desc]"
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
    gnl_coeff_name = argv[8] if len(argv) > 8 else ""
    t1w_match_acq = argv[9] if len(argv) > 9 else ""
    t1w_match_desc = argv[10] if len(argv) > 10 else ""

    cwd = Path(os.environ.get("_CONDOR_SCRATCH_DIR", os.getcwd())).resolve()
    os.chdir(cwd)
    print(f"Running in Condor scratch directory: {cwd}")
    print("Transferred top-level files and directories:")
    for p in sorted(cwd.iterdir()):
        print(f"  {p.name}")

    if Path("/.singularity.d/env/90-environment.sh").exists():
        print("Container environment file exists: /.singularity.d/env/90-environment.sh")

    conda_dir = os.environ.setdefault("CONDA_DIR", "/opt/conda")
    fsldir = os.environ.setdefault("FSLDIR", "/usr/local/fsl")
    fs_home = os.environ.setdefault("FREESURFER_HOME", "/usr/local/freesurfer/8.2.0")
    c3d = os.environ.setdefault("C3DPATH", "/opt/c3d/bin")
    tortoise_home = os.environ.setdefault("TORTOISE_HOME", "/opt/tortoise")
    qmri_fit_home = os.environ.setdefault("QMRI_FIT_HOME", "/opt/qmri-fit")
    tortoise_lib = f"{tortoise_home}/lib"
    qmri_fit_lib = f"{qmri_fit_home}/lib"
    os.environ["LD_LIBRARY_PATH"] = ":".join([
        tortoise_lib,
        qmri_fit_lib,
        os.environ.get("LD_LIBRARY_PATH", ""),
    ])
    os.environ["PATH"] = ":".join([
        f"{conda_dir}/bin",
        f"{fsldir}/bin",
        f"{fs_home}/bin",
        f"{fs_home}/python/bin",
        f"{fs_home}/python/scripts",
        c3d,
        f"{tortoise_home}/bin",
        f"{qmri_fit_home}/bin",
        os.environ.get("PATH", ""),
    ])
    tortoise_gnl = shutil.which("CreateGradientNonlinearityBMatrix")
    if tortoise_gnl:
        print(f"Found TORTOISE GNL executable: {tortoise_gnl}")
    else:
        print("TORTOISE GNL executable not found on PATH after environment setup")
    qmri_fit = shutil.which("qmri-fit") or shutil.which("qmri_fit")
    if qmri_fit:
        print(f"Found qmri-fit executable: {qmri_fit}")
    else:
        print("qmri-fit executable not found on PATH after environment setup")

    config = find_config(cwd, config_name)
    qmri_cmd = resolve_qmri_command()

    for dirname in ("data", "config", "out", "work"):
        (cwd / dirname).mkdir(exist_ok=True)
    staged_config = cwd / "config" / config.name
    shutil.copy2(config, staged_config)
    apply_t1w_match_entities(staged_config, acq=t1w_match_acq, desc=t1w_match_desc)

    copy_freesurfer_license(cwd, license_name)
    stage_gnl_coeff_file(cwd, gnl_coeff_name)
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


def main(argv: list[str]) -> int:
    argv = expand_settings_args(argv)
    if argv and argv[0] == "run":
        return cmd_run(argv[1:])

    parser = argparse.ArgumentParser(
        description=(
            "qneuro HTCondor helper. Use stage-remote from the BIDS server, then "
            "submit-staged after logging into CHTC."
        ),
        epilog="Any command can read defaults from YAML/JSON with --settings settings.yaml.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    local = subparsers.add_parser("submit-local", help="Package local BIDS data and submit on this machine.")
    add_submit_common_args(local)
    local.add_argument("--package-dir", required=True)
    local.set_defaults(func=cmd_submit_local)

    push = subparsers.add_parser("push-submit", help="Package BIDS data, push to CHTC, generate submit file, and submit.")
    add_submit_common_args(push)
    push.add_argument("--submit-host", required=True)
    push.add_argument("--remote-submit-dir", required=True)
    push.add_argument("--remote-package-dir", required=True)
    push.set_defaults(func=cmd_push_submit)

    stage = subparsers.add_parser(
        "stage-remote",
        help="Package BIDS data and copy the staged payload to CHTC without submitting.",
    )
    stage.add_argument("--remote-host", required=True)
    stage.add_argument("--remote-stage-dir", required=True)
    stage.add_argument("--config-file", default="")
    stage.add_argument("--freesurfer-license", default="")
    stage.add_argument("--gnl-coeff-file", default="")
    stage.add_argument("--bids-dir", required=True)
    stage.add_argument(
        "--bids-include-dirs",
        default="",
        help="Comma-separated BIDS datatype directories to package, for example: anat,dwi.",
    )
    stage.add_argument(
        "--bids-exclude-dirs",
        default="",
        help="Comma-separated BIDS datatype directories to omit when packaging.",
    )
    stage.add_argument(
        "--anat-derivatives-dir",
        default="",
        help="Derivative dataset root whose sub-*/[ses-*]/anat files are added to each archive.",
    )
    selection = stage.add_mutually_exclusive_group(required=True)
    selection.add_argument("--subject")
    selection.add_argument("--subjects", help="Comma-separated subject labels, for example: 10021,10022")
    selection.add_argument("--subjects-file")
    stage.add_argument("--session", default="none")
    stage.add_argument(
        "--sessions",
        default="",
        help="Comma-separated session labels for --subjects.",
    )
    stage.add_argument("--manifest-name", default="qneuro_stage.csv")
    stage.add_argument("--bundle-name", default="qneuro_stage_bundle.tar.gz")
    stage.add_argument(
        "--create-remote-dir",
        action="store_true",
        help="Create the remote stage directory first. This costs one extra SSH/Duo login.",
    )
    stage.add_argument(
        "--no-bundle",
        dest="bundle",
        action="store_false",
        help="Copy individual staged files instead of one tar.gz bundle.",
    )
    stage.add_argument(
        "--include-support-files",
        action="store_true",
        help="Also include config, FreeSurfer license, and optionally this script in the staging bundle.",
    )
    stage.add_argument(
        "--no-copy-script",
        dest="copy_script",
        action="store_false",
        help="With --include-support-files, do not copy qneuro_condor.py alongside the staged payload.",
    )
    stage.set_defaults(func=cmd_stage_remote, copy_script=True, bundle=True)

    staged = subparsers.add_parser(
        "submit-staged",
        help="Run on CHTC to generate and submit Condor files from an existing staged payload.",
    )
    staged.add_argument("--staging-dir", required=True)
    staged.add_argument("--submit-dir", default=".")
    staged.add_argument(
        "--manifest",
        default="",
        help=(
            "Optional staged CSV. If omitted, submit-staged uses --manifest-name when "
            "present, otherwise infers rows from sub-*_bids.tar.gz archive names."
        ),
    )
    staged.add_argument(
        "--manifest-name",
        default="qneuro_stage.csv",
        help="Default staged CSV name to use if it exists in --staging-dir.",
    )
    staged.add_argument("--container-image", required=True)
    staged.add_argument("--config-file", default="")
    staged.add_argument("--freesurfer-license", default="")
    staged.add_argument(
        "--gnl-coeff-file",
        default="",
        help=(
            "Optional gradient nonlinearity coefficient file to transfer with each job. "
            "May be a local path or osdf:/// URL."
        ),
    )
    staged.add_argument("--transfer-uri", default="")
    staged_selection = staged.add_mutually_exclusive_group()
    staged_selection.add_argument("--subject")
    staged_selection.add_argument("--subjects", help="Comma-separated subject labels to submit from staged inputs.")
    staged_selection.add_argument("--subjects-file", help="CSV/text subject/session rows to submit from staged inputs.")
    staged.add_argument("--session", default="")
    staged.add_argument("--sessions", default="", help="Comma-separated session labels for --subjects.")
    staged.add_argument("--pipeline", default="dmri")
    staged.add_argument("--cpus", type=int, default=8)
    staged.add_argument("--gpus", type=int, default=0)
    staged.add_argument("--memory-gb", type=int, default=32)
    staged.add_argument("--disk-gb", type=int, default=40)
    staged.add_argument("--require-dwi", default="true", choices=("true", "false"))
    staged.add_argument(
        "--t1w-match-acq",
        default="",
        help="Select the staged T1w by its BIDS acq entity, for example MPnRAGE.",
    )
    staged.add_argument(
        "--t1w-match-desc",
        default="",
        help="Select the staged T1w by its BIDS desc entity, for example MPnRAGE.",
    )
    staged.add_argument(
        "--requirements",
        default='(OpSys == "LINUX") && (Arch == "X86_64") && (HasCHTCStaging == true)',
    )
    staged.add_argument("--getenv", default="true")
    staged.add_argument("--gpu-minimum-capability", default="8.0")
    staged.add_argument("--want-flocking", default="true")
    staged.add_argument("--want-glidein", default="true")
    staged.add_argument("--want-gpu-lab", default="false")
    staged.add_argument("--gpu-job-length", default="medium")
    staged.add_argument("--notification", default="")
    staged.add_argument("--notify-user", default="")
    staged.add_argument("--log-dir", default="")
    staged.add_argument("--output-directory", default="")
    staged.add_argument("--output-destination", default="")
    staged.add_argument("--submit-file-name", default="qneuro_generated.sub")
    staged.add_argument("--queue-file-name", default="qneuro_inputs.csv")
    staged.add_argument("--no-submit", action="store_true")
    staged.set_defaults(func=cmd_submit_staged)

    collect = subparsers.add_parser(
        "collect-outputs",
        help=(
            "Copy completed Condor output tarballs back from CHTC if requested, "
            "then install them into a BIDS derivatives folder."
        ),
    )
    collect.add_argument(
        "--outputs-dir",
        default=".",
        help="Local directory containing qneuro_outputs_sub-*_ses-*.tar.gz files.",
    )
    collect.add_argument(
        "--remote-host",
        default="",
        help="Optional CHTC login host to copy output tarballs from before installing.",
    )
    collect.add_argument(
        "--remote-outputs-dir",
        default="",
        help=(
            "Remote directory containing qneuro output tarballs, for example "
            "/staging/<user>/qneuro-outputs."
        ),
    )
    collect_target = collect.add_mutually_exclusive_group(required=True)
    collect_target.add_argument(
        "--bids-dir",
        help="Raw BIDS dataset root; outputs install into <bids-dir>/derivatives/<derivative-name>.",
    )
    collect_target.add_argument(
        "--derivatives-dir",
        help="Exact derivative dataset directory to install into.",
    )
    collect.add_argument(
        "--derivative-name",
        default="qneuro",
        help="Derivative dataset name used under --bids-dir and when stripping archive prefixes.",
    )
    collect_selection = collect.add_mutually_exclusive_group()
    collect_selection.add_argument("--subject")
    collect_selection.add_argument("--subjects", help="Comma-separated subject labels to install.")
    collect_selection.add_argument("--subjects-file", help="CSV/text subject/session rows to install.")
    collect.add_argument("--session", default="")
    collect.add_argument("--sessions", default="", help="Comma-separated session labels for --subjects.")
    collect.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow extracted files to replace existing derivative files.",
    )
    collect.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned installs without writing files.",
    )
    collect.set_defaults(func=cmd_collect_outputs)

    args = parser.parse_args(argv)
    return args.func(args)


def console_main() -> int:
    return main(sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(console_main())
