"""Atomic completion manifests for registry-driven dmipy fits."""

from __future__ import annotations

from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Callable, Mapping

from .dmipy_backend import DmipyRuntime
from ..io.bids import _sidecar
from ..utils.serialization import json_ready


MANIFEST_FILENAME = "dmipy-completion.json"
MANIFEST_SCHEMA_VERSION = 1
_SMALL_FILE_HASH_LIMIT = 4 * 1024 * 1024


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def input_signature(path: Path | str | None) -> dict[str, Any] | None:
    """Return a fast, change-sensitive signature for one fit input."""
    if path is None:
        return None
    resolved = Path(path).expanduser().resolve()
    stat = resolved.stat()
    signature: dict[str, Any] = {
        "path": str(resolved),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    if stat.st_size <= _SMALL_FILE_HASH_LIMIT:
        signature["sha256"] = _sha256_file(resolved)
    else:
        digest = sha256()
        with resolved.open("rb") as stream:
            digest.update(stream.read(1024 * 1024))
            stream.seek(max(0, stat.st_size - 1024 * 1024))
            digest.update(stream.read(1024 * 1024))
        signature["sample_sha256"] = digest.hexdigest()
    return signature


def build_dmipy_run_spec(
    *,
    model_name: str,
    in_file: Path,
    bval_file: Path,
    bvec_file: Path,
    mask_file: Path | None,
    grad_nonlin: Path | None,
    delta_file: Path | None,
    Delta_file: Path | None,
    TE_file: Path | None,
    solver: str,
    device: str,
    solver_options: Mapping[str, Any] | None,
    factory_options: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Build the scientific request whose exact completion may be reused."""
    try:
        dmipy_version = version("dmipy-fit")
    except PackageNotFoundError:
        dmipy_version = "not-installed"
    inputs = {
        "dwi": in_file,
        "bval": bval_file,
        "bvec": bvec_file,
        "mask": mask_file,
        "gradient_nonlinearity": grad_nonlin,
        "delta": delta_file,
        "Delta": Delta_file,
        "TE": TE_file,
    }
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "dmipy_fit_version": dmipy_version,
        "model_name": str(model_name).lower(),
        "solver": str(solver).lower(),
        "device": str(device).lower(),
        "solver_options": json_ready(dict(solver_options or {})),
        "factory_options": json_ready(dict(factory_options or {})),
        "inputs": {
            name: input_signature(path)
            for name, path in inputs.items()
        },
    }


def run_fingerprint(run_spec: Mapping[str, Any]) -> str:
    """Return a stable digest for one canonical run specification."""
    serialized = json.dumps(
        json_ready(run_spec),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(serialized).hexdigest()


def manifest_path(out_dir: Path) -> Path:
    return Path(out_dir) / MANIFEST_FILENAME


def invalidate_completion_manifest(out_dir: Path) -> None:
    """Remove only the completion marker before beginning a new fit."""
    path = manifest_path(out_dir)
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def write_completion_manifest(
    out_dir: Path,
    *,
    run_spec: Mapping[str, Any],
    outputs: Mapping[str, Path],
    runtime: DmipyRuntime,
) -> Path:
    """Atomically mark a fit complete after all images and sidecars exist."""
    out_dir = Path(out_dir).resolve()
    records: dict[str, dict[str, Any]] = {}
    for parameter, output in sorted(outputs.items()):
        output = Path(output).resolve()
        sidecar = _sidecar(output, ".json")
        if not output.is_file() or not sidecar.is_file():
            raise FileNotFoundError(
                f"Cannot complete dmipy run: missing output or sidecar for "
                f"{parameter!r}."
            )
        try:
            relative_output = output.relative_to(out_dir)
            relative_sidecar = sidecar.relative_to(out_dir)
        except ValueError as exc:
            raise ValueError("dmipy outputs must be inside their output directory.") from exc
        records[str(parameter)] = {
            "image": str(relative_output),
            "sidecar": str(relative_sidecar),
            "image_size": output.stat().st_size,
            "image_mtime_ns": output.stat().st_mtime_ns,
            "sidecar_size": sidecar.stat().st_size,
            "sidecar_sha256": _sha256_file(sidecar),
        }
    if not records:
        raise ValueError("Cannot complete a dmipy run without derivative outputs.")

    payload = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "status": "complete",
        "fingerprint": run_fingerprint(run_spec),
        "request": json_ready(run_spec),
        "runtime": json_ready(runtime.provenance()),
        "expected_parameters": sorted(records),
        "outputs": records,
    }
    target = manifest_path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    temporary_name = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=out_dir,
            prefix=f".{MANIFEST_FILENAME}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_name = stream.name
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, target)
    finally:
        if temporary_name is not None:
            try:
                Path(temporary_name).unlink()
            except FileNotFoundError:
                pass
    return target


def completed_outputs(
    out_dir: Path,
    *,
    run_spec: Mapping[str, Any],
    validate_image: Callable[[Path], bool] | None = None,
) -> dict[str, Path] | None:
    """Return declared outputs only when the exact run completed successfully."""
    out_dir = Path(out_dir).resolve()
    path = manifest_path(out_dir)
    try:
        with path.open() as stream:
            payload = json.load(stream)
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None
    if (
        payload.get("schema_version") != MANIFEST_SCHEMA_VERSION
        or payload.get("status") != "complete"
        or payload.get("fingerprint") != run_fingerprint(run_spec)
    ):
        return None
    records = payload.get("outputs")
    expected_parameters = payload.get("expected_parameters")
    if not isinstance(records, dict) or not records:
        return None
    if (
        not isinstance(expected_parameters, list)
        or set(expected_parameters) != set(records)
    ):
        return None

    outputs: dict[str, Path] = {}
    for parameter, record in records.items():
        if not isinstance(record, dict):
            return None
        try:
            image = (out_dir / record["image"]).resolve()
            sidecar = (out_dir / record["sidecar"]).resolve()
            image.relative_to(out_dir)
            sidecar.relative_to(out_dir)
        except (KeyError, TypeError, ValueError):
            return None
        if not image.is_file() or not sidecar.is_file():
            return None
        if (
            image.stat().st_size != record.get("image_size")
            or image.stat().st_mtime_ns != record.get("image_mtime_ns")
            or sidecar.stat().st_size != record.get("sidecar_size")
            or _sha256_file(sidecar) != record.get("sidecar_sha256")
        ):
            return None
        if validate_image is not None and not validate_image(image):
            return None
        outputs[str(parameter)] = image
    return outputs
