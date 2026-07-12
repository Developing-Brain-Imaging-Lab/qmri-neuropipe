#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import shlex
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path


ROOT_METADATA = ("dataset_description.json", "participants.tsv", "participants.json", "README", "CHANGES")


def norm_label(value: str | None, prefix: str) -> str:
    value = (value or "").strip()
    return value[len(prefix):] if value.startswith(prefix) else value


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


def tar_members_for_pair(subject: str, session: str) -> tuple[str, str]:
    subject = norm_label(subject, "sub-")
    session = norm_label(session, "ses-")
    subject_dir = f"sub-{subject}"
    if session:
        session_dir = f"ses-{session}"
        return f"{subject_dir}/{session_dir}", f"{subject_dir}_{session_dir}_bids.tar.gz"
    return subject_dir, f"{subject_dir}_bids.tar.gz"


def create_local_archive(bids_dir: Path, subject: str, session: str, package_dir: Path) -> Path:
    input_rel, archive_name = tar_members_for_pair(subject, session)
    if not (bids_dir / input_rel).is_dir():
        raise FileNotFoundError(f"Requested BIDS input not found: {bids_dir / input_rel}")

    archive_path = package_dir / archive_name
    members = [input_rel] + [m for m in ROOT_METADATA if (bids_dir / m).exists()]
    print(f"Creating BIDS archive: {archive_path}")
    with tarfile.open(archive_path, "w:gz") as tf:
        for member in members:
            tf.add(bids_dir / member, arcname=member)
    return archive_path


def create_remote_archive(remote_bids: str, subject: str, session: str, package_dir: Path) -> Path:
    if ":" not in remote_bids:
        raise ValueError("--remote-bids must look like user@host:/path/to/bids")
    host, remote_dir = remote_bids.split(":", 1)
    input_rel, archive_name = tar_members_for_pair(subject, session)
    archive_path = package_dir / archive_name

    remote_input = f"{remote_dir.rstrip('/')}/{input_rel}"
    print(f"Checking remote BIDS input: {host}:{remote_input}")
    subprocess.run(["ssh", host, "test", "-d", remote_input], check=True)

    members = [input_rel]
    for meta in ROOT_METADATA:
        remote_meta = f"{remote_dir.rstrip('/')}/{meta}"
        if subprocess.run(["ssh", host, "test", "-e", remote_meta]).returncode == 0:
            members.append(meta)

    quoted_members = " ".join(shlex.quote(m) for m in members)
    cmd = f"tar -C {shlex.quote(remote_dir)} -czf - {quoted_members}"
    print(f"Streaming remote BIDS archive to: {archive_path}")
    with archive_path.open("wb") as out:
        subprocess.run(["ssh", host, cmd], check=True, stdout=out)
    return archive_path


def rewrite_submit_file(template: Path, vars_file: Path, output: Path) -> None:
    vars_name = vars_file.name
    wrote_queue = False
    skip_prefixes = ("SUBJECT", "subject", "SESSION", "session", "bids_input")

    with template.open() as src, output.open("w") as dst:
        for line in src:
            stripped = line.strip()
            lower = stripped.lower()
            lhs = stripped.split("=", 1)[0].strip() if "=" in stripped else ""
            if lhs in skip_prefixes:
                continue
            if lower.startswith("queue"):
                if not wrote_queue:
                    dst.write(f"queue SUBJECT,SESSION,bids_input,subject,session from {vars_name}\n")
                    wrote_queue = True
                continue
            dst.write(line)
        if not wrote_queue:
            dst.write(f"queue SUBJECT,SESSION,bids_input,subject,session from {vars_name}\n")


def write_vars_file(rows: list[tuple[str, str, str]], path: Path) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        for subject, session, bids_input in rows:
            writer.writerow([subject, session, bids_input, subject, session])


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Package BIDS inputs and submit qneuro Condor jobs.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--bids-dir")
    source.add_argument("--remote-bids")
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--subject")
    selection.add_argument("--subjects-file")
    parser.add_argument("--session", default="")
    parser.add_argument("--submit-file", required=True)
    parser.add_argument("--package-dir", required=True)
    parser.add_argument("--transfer-uri", default="")
    args = parser.parse_args(argv)

    submit_file = Path(args.submit_file).resolve()
    submit_dir = submit_file.parent
    package_dir = Path(args.package_dir).resolve()
    package_dir.mkdir(parents=True, exist_ok=True)

    if args.subjects_file:
        pairs = parse_subjects_file(Path(args.subjects_file))
    else:
        pairs = [(args.subject, args.session)]

    queued: list[tuple[str, str, str]] = []
    transfer_uri = args.transfer_uri.rstrip("/")
    for subject_raw, session_raw in pairs:
        subject = norm_label(subject_raw, "sub-")
        session = norm_label(session_raw, "ses-")
        if args.remote_bids:
            archive = create_remote_archive(args.remote_bids, subject, session, package_dir)
        else:
            archive = create_local_archive(Path(args.bids_dir), subject, session, package_dir)
        bids_input = f"{transfer_uri}/{archive.name}" if transfer_uri else str(archive)
        queued.append((subject, session, bids_input))

    if not queued:
        raise RuntimeError("No subject/session rows were packaged.")

    fd, submit_tmp_name = tempfile.mkstemp(prefix=".qneuro_submit.", suffix=".sub", dir=submit_dir)
    os.close(fd)
    fd, vars_tmp_name = tempfile.mkstemp(prefix=".qneuro_inputs.", suffix=".csv", dir=submit_dir)
    os.close(fd)
    submit_tmp = Path(submit_tmp_name)
    vars_tmp = Path(vars_tmp_name)

    write_vars_file(queued, vars_tmp)
    rewrite_submit_file(submit_file, vars_tmp, submit_tmp)

    print(f"Submitting queued inputs from: {vars_tmp}")
    subprocess.run(["condor_submit", submit_tmp.name], cwd=submit_dir, check=True)
    print(f"Temporary submit file: {submit_tmp}")
    print(f"Queue vars file: {vars_tmp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
