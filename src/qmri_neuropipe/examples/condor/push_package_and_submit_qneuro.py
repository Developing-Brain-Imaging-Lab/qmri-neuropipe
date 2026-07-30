#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

try:
    from qmri_neuropipe.examples.condor.qneuro_condor import (
        create_bids_archive,
        norm_label,
        parse_subjects_file,
    )
except ImportError:  # Support an unpacked source checkout.
    from qneuro_condor import (  # type: ignore[no-redef]
        create_bids_archive,
        norm_label,
        parse_subjects_file,
    )


def create_archive(bids_dir: Path, subject: str, session: str, archive_dir: Path) -> Path:
    return create_bids_archive(bids_dir, subject, session, archive_dir)


def read_submit_assignments(template: Path) -> dict[str, str]:
    assignments: dict[str, str] = {}
    for line in template.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        assignments[key.strip().lower()] = value.strip()
    return assignments


def resolve_template_file(submit_file: Path, value: str) -> Path | None:
    if not value or "$(" in value:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = submit_file.parent / path
    return path.resolve() if path.is_file() else None


def rewrite_submit(
    template: Path,
    vars_file_name: str,
    output: Path,
    overrides: dict[str, tuple[str, str]] | None = None,
) -> None:
    wrote_queue = False
    overrides = overrides or {}
    wrote_overrides: set[str] = set()
    skip = {"SUBJECT", "subject", "SESSION", "session", "bids_input"}

    def write_missing_overrides(dst) -> None:
        for lhs_lower, (key, value) in overrides.items():
            if lhs_lower not in wrote_overrides:
                dst.write(f"{key} = {value}\n")
                wrote_overrides.add(lhs_lower)

    with template.open() as src, output.open("w") as dst:
        for line in src:
            stripped = line.strip()
            lhs = stripped.split("=", 1)[0].strip() if "=" in stripped else ""
            lhs_lower = lhs.lower()
            if lhs in skip:
                continue
            if lhs_lower in overrides:
                key, value = overrides[lhs_lower]
                dst.write(f"{key} = {value}\n")
                wrote_overrides.add(lhs_lower)
                continue
            if stripped.lower().startswith("queue"):
                if not wrote_queue:
                    write_missing_overrides(dst)
                    dst.write(f"queue SUBJECT,SESSION,bids_input,subject,session from {vars_file_name}\n")
                    wrote_queue = True
                continue
            dst.write(line)
        write_missing_overrides(dst)
        if not wrote_queue:
            dst.write(f"queue SUBJECT,SESSION,bids_input,subject,session from {vars_file_name}\n")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Push BIDS packages to CHTC and submit remotely.")
    parser.add_argument("--submit-host", required=True)
    parser.add_argument("--remote-submit-dir", required=True)
    parser.add_argument("--remote-package-dir", required=True)
    parser.add_argument("--transfer-uri", required=True)
    parser.add_argument("--submit-file", required=True)
    parser.add_argument(
        "--config-file",
        help=(
            "Pipeline YAML/JSON config to copy to the CHTC submit directory. "
            "If omitted, the script tries to use config_file from the submit template."
        ),
    )
    parser.add_argument("--bids-dir", required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--subject")
    selection.add_argument("--subjects-file")
    parser.add_argument("--session", default="")
    parser.add_argument("--copy-executable", action="store_true")
    args = parser.parse_args(argv)

    bids_dir = Path(args.bids_dir).resolve()
    submit_file = Path(args.submit_file).resolve()
    assignments = read_submit_assignments(submit_file)
    config_file = Path(args.config_file).expanduser().resolve() if args.config_file else None
    if config_file is None:
        config_file = resolve_template_file(submit_file, assignments.get("config_file", ""))
    if config_file is not None and not config_file.is_file():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    if args.subjects_file:
        pairs = parse_subjects_file(Path(args.subjects_file))
    else:
        pairs = [(args.subject, args.session)]

    work_dir = Path(tempfile.mkdtemp(prefix="qneuro_push."))
    archive_dir = work_dir / "archives"
    archive_dir.mkdir()
    vars_file = work_dir / "qneuro_inputs.csv"
    remote_submit = args.remote_submit_dir
    remote_package = args.remote_package_dir
    transfer_uri = args.transfer_uri.rstrip("/")

    queued: list[list[str]] = []
    archives: list[Path] = []
    for subject_raw, session_raw in pairs:
        subject = norm_label(subject_raw, "sub-")
        session = norm_label(session_raw, "ses-")
        archive = create_archive(bids_dir, subject, session, archive_dir)
        archives.append(archive)
        queued.append([subject, session, f"{transfer_uri}/{archive.name}", subject, session])

    with vars_file.open("w", newline="") as f:
        csv.writer(f, lineterminator="\n").writerows(queued)

    tmp_submit = work_dir / submit_file.name
    overrides: dict[str, tuple[str, str]] = {}
    submit_payload = [tmp_submit, vars_file]
    if config_file is not None:
        overrides["config_file"] = ("config_file", config_file.name)
        overrides["config_name"] = ("config_name", config_file.name)
        submit_payload.append(config_file)
    else:
        print(
            "Warning: no local config YAML/JSON was found to push. "
            "The remote submit host must already be able to read config_file."
        )
    rewrite_submit(submit_file, vars_file.name, tmp_submit, overrides)

    subprocess.run([
        "ssh", args.submit_host,
        f"mkdir -p {shlex.quote(remote_submit)} {shlex.quote(remote_package)}",
    ], check=True)
    subprocess.run(["scp", *map(str, archives), f"{args.submit_host}:{remote_package}/"], check=True)
    subprocess.run(["scp", *map(str, submit_payload), f"{args.submit_host}:{remote_submit}/"], check=True)

    if args.copy_executable:
        executable = ""
        for line in submit_file.read_text().splitlines():
            if line.lower().lstrip().startswith("executable") and "=" in line:
                executable = line.split("=", 1)[1].strip()
                break
        if executable:
            exe_path = submit_file.parent / executable
            if exe_path.is_file():
                subprocess.run(["scp", str(exe_path), f"{args.submit_host}:{remote_submit}/"], check=True)

    subprocess.run([
        "ssh", args.submit_host,
        f"cd {shlex.quote(remote_submit)} && condor_submit {shlex.quote(submit_file.name)}",
    ], check=True)
    print(f"Local staging work directory: {work_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
