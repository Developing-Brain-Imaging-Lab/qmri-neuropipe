"""
R0 — End-to-end golden snapshot harness.

Unlike test_r0_characterization.py (which pins behavior at code seams and runs
anywhere), this harness runs a REAL pipeline subject and snapshots its outputs.
It needs the external binaries (FSL / ANTs / FreeSurfer / MRtrix as configured)
and a small test dataset, so it is skipped by default.

Usage
-----
1. Point the env vars at a tiny BIDS dataset and a config:

       export QMRI_R0_BIDS=/path/to/tiny_bids
       export QMRI_R0_CONFIG=/path/to/config.yaml
       export QMRI_R0_SUBJECT=01
       export QMRI_R0_SESSION=01          # optional
       export QMRI_R0_PIPELINE=dmri       # dmri | anat | relaxometry

2. Record the golden BEFORE refactoring:

       QMRI_R0_RECORD=1 pytest src/qmri_neuropipe/tests/test_r0_golden_snapshot.py

   This writes <repo>/tests/golden/<pipeline>_<subject>.json and commits the
   snapshot of: relative output file paths, sizes, header/affine checksums, and
   tracker status rows.

3. After each refactor commit, run WITHOUT QMRI_R0_RECORD to diff against golden:

       pytest src/qmri_neuropipe/tests/test_r0_golden_snapshot.py

   A non-empty diff = behavior changed = stop and re-scope.

The snapshot deliberately checksums only NIfTI header + affine (not full voxel
data) so it is fast and stable across library float jitter, while still catching
grid/shape/orientation regressions and file-set changes.
"""

import hashlib
import json
import os
from pathlib import Path

import pytest

RECORD = os.environ.get("QMRI_R0_RECORD") == "1"
BIDS = os.environ.get("QMRI_R0_BIDS")
CONFIG = os.environ.get("QMRI_R0_CONFIG")
SUBJECT = os.environ.get("QMRI_R0_SUBJECT", "01")
SESSION = os.environ.get("QMRI_R0_SESSION") or None
PIPELINE = os.environ.get("QMRI_R0_PIPELINE", "dmri")

GOLDEN_DIR = Path(__file__).resolve().parents[3] / "tests" / "golden"


pytestmark = pytest.mark.skipif(
    not (BIDS and CONFIG),
    reason="Set QMRI_R0_BIDS and QMRI_R0_CONFIG to run the end-to-end golden harness.",
)


def _nifti_fingerprint(path: Path) -> str:
    """Hash NIfTI shape + affine + zooms (not voxel data) for a stable signature."""
    import nibabel as nib
    import numpy as np

    try:
        img = nib.load(str(path))
        h = hashlib.sha1()
        h.update(np.asarray(img.shape).tobytes())
        h.update(np.asarray(img.affine, dtype=np.float64).round(4).tobytes())
        h.update(np.asarray(img.header.get_zooms(), dtype=np.float64).round(4).tobytes())
        return "nifti:" + h.hexdigest()
    except Exception as exc:  # non-image or unreadable: fall back to size only
        return f"unreadable:{exc.__class__.__name__}"


def _snapshot(output_dir: Path) -> dict:
    """Build a deterministic snapshot of an output tree."""
    entries = {}
    for path in sorted(output_dir.rglob("*")):
        if path.is_dir():
            continue
        rel = str(path.relative_to(output_dir))
        # Skip logs/timestamps that legitimately vary run to run.
        if "/logs/" in f"/{rel}" or rel.endswith(".log"):
            continue
        info = {"size": path.stat().st_size}
        if path.name.endswith((".nii", ".nii.gz")):
            info["fingerprint"] = _nifti_fingerprint(path)
        entries[rel] = info
    return entries


def _tracker_snapshot(tracker) -> dict:
    """Capture stable status/module values while excluding update timestamps."""
    if tracker is None:
        return {}

    snapshots = {}
    for sheet_name, frame in sorted(getattr(tracker, "_data", {}).items()):
        if not sheet_name.endswith("_Status") or frame.empty:
            continue

        rows = []
        identity_columns = ("Subject_ID", "Session", "Study")
        ignored_columns = {"Last_Update", "Last_Processing_Date"}
        for _, row in frame.iterrows():
            identity = {
                column: None if column not in frame or row.isna().get(column, False) else str(row[column])
                for column in identity_columns
                if column in frame
            }
            statuses = {
                column: str(row[column])
                for column in frame.columns
                if column not in identity_columns
                and column not in ignored_columns
                and not row.isna().get(column, True)
            }
            rows.append({"identity": identity, "statuses": statuses})

        snapshots[sheet_name] = sorted(rows, key=lambda value: json.dumps(value, sort_keys=True))
    return snapshots


def _run_pipeline(output_dir: Path):
    from qmri_neuropipe.core.config import PipelineConfig

    overrides = {
        "bids_dir": BIDS,
        "output_dir": str(output_dir),
        "participant_label": [SUBJECT],
        # Keep any enabled tracker inside the disposable golden output tree.
        "tracker": {"file": str(output_dir / "study_tracker.xlsx")},
    }
    if SESSION:
        overrides["session_label"] = [SESSION]

    config = PipelineConfig.from_file(CONFIG, overrides=overrides)

    if PIPELINE == "dmri":
        from qmri_neuropipe.workflows.pipelines.dmri import DMRIPipeline as P
    elif PIPELINE == "anat":
        from qmri_neuropipe.workflows.pipelines.anat import AnatPipeline as P
    elif PIPELINE == "relaxometry":
        from qmri_neuropipe.workflows.pipelines.relaxometry import RelaxometryPipeline as P
    else:
        raise ValueError(f"Unknown QMRI_R0_PIPELINE: {PIPELINE}")

    pipeline = P(config)
    pairs = [(SUBJECT, SESSION)]
    pipeline.run(pairs=pairs)
    return config.tracker


def _golden_path() -> Path:
    name = f"{PIPELINE}_{SUBJECT}" + (f"_{SESSION}" if SESSION else "")
    return GOLDEN_DIR / f"{name}.json"


def test_golden_snapshot(tmp_path):
    output_dir = tmp_path / "derivatives"
    output_dir.mkdir(parents=True, exist_ok=True)

    tracker = _run_pipeline(output_dir)
    snapshot = {
        "files": _snapshot(output_dir),
        "tracker_rows": _tracker_snapshot(tracker),
    }

    golden_file = _golden_path()

    if RECORD:
        GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
        golden_file.write_text(json.dumps(snapshot, indent=2, sort_keys=True))
        pytest.skip(f"Recorded golden snapshot to {golden_file} ({len(snapshot)} files).")

    assert golden_file.exists(), (
        f"No golden snapshot at {golden_file}. Record one first with "
        f"QMRI_R0_RECORD=1 before refactoring."
    )

    golden = json.loads(golden_file.read_text())

    # Compute a readable file-tree diff.
    actual_files = snapshot["files"]
    golden_files = golden["files"]
    added = sorted(set(actual_files) - set(golden_files))
    removed = sorted(set(golden_files) - set(actual_files))
    changed = sorted(
        k for k in (set(actual_files) & set(golden_files))
        if actual_files[k] != golden_files[k]
    )

    msg_parts = []
    if added:
        msg_parts.append(f"ADDED files: {added}")
    if removed:
        msg_parts.append(f"REMOVED files: {removed}")
    if changed:
        details = {
            k: {"golden": golden_files[k], "now": actual_files[k]}
            for k in changed
        }
        msg_parts.append(f"CHANGED files: {json.dumps(details, indent=2)}")
    if snapshot["tracker_rows"] != golden.get("tracker_rows", {}):
        msg_parts.append(
            "CHANGED tracker rows: "
            + json.dumps(
                {
                    "golden": golden.get("tracker_rows", {}),
                    "now": snapshot["tracker_rows"],
                },
                indent=2,
            )
        )

    assert not msg_parts, "Output changed vs golden:\n" + "\n".join(msg_parts)
