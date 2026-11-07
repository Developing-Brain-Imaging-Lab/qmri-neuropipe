"""Optional FastSurfer and anatomical-only DeepPrep processing stages."""
from pathlib import Path
import json
import math
import uuid
import re
import shutil

from ...core import BaseProcessingStep
from ...interfaces.deep_anatomical import execute_anatomical


def _label(value, prefix):
    value = str(value or "").removeprefix(prefix)
    if not re.fullmatch(r"[A-Za-z0-9]+", value):
        raise ValueError(f"Invalid {prefix} label: {value!r}")
    return value


def _nonempty(path):
    return path.is_file() and path.stat().st_size > 0


class DeepAnatomicalStep(BaseProcessingStep):
    """Keep backend outputs separate from recon-all and native-space images."""
    backend = ""

    @property
    def settings(self):
        return (self.config.get("anat", {}) or {}).get(self.backend, {}) or {}

    def _paths(self, context):
        subject = _label(context.get("subject"), "sub-")
        session = context.get("session")
        sid = f"sub-{subject}"
        if session:
            sid += f"_ses-{_label(session, 'ses-')}"
        root = Path(self.settings.get("output_dir") or Path(self.config.output_dir) / self.backend).expanduser().resolve()
        if self.backend == "deepprep":
            root = root / sid  # DeepPrep's reconstruction IDs do not encode sessions.
            subject_dir = root / "Recon" / f"sub-{subject}"
        else:
            subject_dir = root / sid
        return subject, sid, root, subject_dir

    def _input(self, context):
        if self.settings.get("input", "raw") == "preprocessed":
            image = context.get("preprocessed_t1w")
        else:
            candidates = context.get("t1w_files") or []
            if len(candidates) != 1:
                raise ValueError(f"{self.backend} requires exactly one selected T1w; use anat.input.t1w_match to select a scan")
            image = candidates[0]
        if image is None:
            raise ValueError(f"{self.backend} requires a T1w image")
        path = Path(getattr(image, "img", image)).expanduser().resolve()
        if not _nonempty(path):
            raise FileNotFoundError(f"Missing or empty T1w: {path}")
        if not (path.name.endswith('.nii.gz') or path.suffix == '.nii'):
            raise ValueError(f"{self.backend} requires a NIfTI T1w image")
        return image, path

    def _products(self, subject_dir):
        if self.backend == "fastsurfer":
            candidates = [subject_dir / "mri/aparc.DKTatlas+aseg.deep.mgz"]
        else:
            candidates = [subject_dir / "mri/aparc.DKTatlas+aseg.deep.mgz",
                          subject_dir / "mri/aparc+aseg.mgz", subject_dir / "mri/aseg.mgz"]
        products = {"segmentation": next((p for p in candidates if _nonempty(p)), candidates[0])}
        if self.backend == "deepprep" or self.settings.get("mode", "full") == "full":
            for hemi in ("lh", "rh"):
                for name in ("white", "pial", "thickness", "sphere.reg"):
                    products[f"{hemi}_{name}"] = subject_dir / "surf" / f"{hemi}.{name}"
                names = (f"{hemi}.aparc.DKTatlas.mapped.stats", f"{hemi}.aparc.stats",
                         f"{hemi}.aparc.DKTatlas.stats")
                stats = [subject_dir / "stats" / name for name in names]
                products[f"{hemi}_stats"] = next((p for p in stats if _nonempty(p)), stats[0])
        return products

    def _signature(self, context):
        _, path = self._input(context)
        stat = path.stat()
        return {"schema": 1, "input": str(path), "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns, "settings": self.settings,
                "n_cpus": self.config.n_cpus, "memory_gb": self.config.memory_gb,
                "use_gpu": self.config.use_gpu}

    def is_complete(self, context):
        try:
            _, _, _, subject_dir = self._paths(context)
            signature = json.loads(json.dumps(self._signature(context), default=str))
            marker = subject_dir / "scripts/qmri_neuropipe_complete.json"
            return (json.loads(marker.read_text()) == signature
                    and all(_nonempty(p) for p in self._products(subject_dir).values()))
        except (OSError, ValueError, TypeError):
            return False

    def outputs(self, context):
        _, _, root, subject_dir = self._paths(context)
        return {f"{self.backend}_dir": root,
                f"{self.backend}_subject_dir": subject_dir,
                f"{self.backend}_outputs": self._products(subject_dir)}

    def run(self, context, output_dir, force=False, **kwargs):
        settings = self.settings
        if not settings.get("enabled", False):
            return context
        force = force or self.config.get("force", False) or not self.config.skip_existing
        if not force and self.is_complete(context):
            return {**context, **self.outputs(context)}
        subject, sid, root, subject_dir = self._paths(context)
        image, t1 = self._input(context)
        signature = self._signature(context)
        marker = subject_dir / "scripts/qmri_neuropipe_complete.json"
        # Never mark a failed rerun as complete, even when older products remain.
        marker.unlink(missing_ok=True)
        root.mkdir(parents=True, exist_ok=True)
        device = str(settings.get("device") or ("auto" if self.config.use_gpu else "cpu"))
        license_file = settings.get("fs_license")
        license_path = Path(license_file).expanduser().resolve() if license_file else None
        full = self.backend == "deepprep" or settings.get("mode", "full") == "full"
        if full and (license_path is None or not _nonempty(license_path)):
            raise ValueError(f"anat.{self.backend}.fs_license must point to a valid FreeSurfer license file")
        mounts = [(root, False)]
        if license_path:
            if not _nonempty(license_path):
                raise FileNotFoundError(license_path)
            mounts.append((license_path, True))
        if self.backend == "fastsurfer":
            mounts.append((t1.parent, True))
            args = ["run_fastsurfer.sh", "--t1", str(t1), "--sid", sid, "--sd", str(root),
                    "--device", device, "--threads", str(self.config.n_cpus)]
            if not full:
                args.append("--seg_only")
            if license_path:
                args += ["--fs_license", str(license_path)]
        else:
            # A real copy, not a symlink, keeps staged inputs accessible in containers.
            stage = Path(output_dir).resolve() / "deepprep_input"
            if stage.exists():
                shutil.rmtree(stage)
            anat = stage / f"sub-{subject}" / "anat"
            anat.mkdir(parents=True)
            suffix = ".nii.gz" if t1.name.endswith(".nii.gz") else ".nii"
            shutil.copy2(t1, anat / f"sub-{subject}_T1w{suffix}")
            sidecar = getattr(image, "json", None)
            if sidecar and Path(sidecar).is_file():
                shutil.copy2(sidecar, anat / f"sub-{subject}_T1w.json")
            (stage / "dataset_description.json").write_text(json.dumps({
                "Name": "qmri-neuropipe selected anatomical input", "BIDSVersion": "1.9.0"}))
            mounts.append((stage, True))
            args = ["deepprep-docker", str(stage), str(root), "participant", "--anat_only",
                    "--participant_label", subject, "--fs_license_file", str(license_path),
                    "--device", device, "--cpus", str(self.config.n_cpus),
                    "--memory", str(math.ceil(self.config.memory_gb))]
        # External tools can silently reuse their own cached files. Start clean
        # whenever our completion contract is not satisfied, preserving old work.
        previous = root if self.backend == "deepprep" else subject_dir
        if previous.is_dir() and any(previous.iterdir()):
            backup = previous.with_name(previous.name + ".previous-" + uuid.uuid4().hex[:12])
            previous.rename(backup)
            self.logger.info("Preserved previous %s outputs in %s", self.backend, backup)
        root.mkdir(parents=True, exist_ok=True)
        execute_anatomical(args, settings, mounts, label=self.backend,
                           gpu=device not in {"cpu", "mps"})
        missing = [str(p) for p in self._products(subject_dir).values() if not _nonempty(p)]
        if missing:
            raise RuntimeError(f"{self.backend} did not produce required outputs: {', '.join(missing)}")
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps(signature, indent=2, default=str))
        return {**context, **self.outputs(context)}


class FastSurferStep(DeepAnatomicalStep):
    backend = "fastsurfer"


class DeepPrepStep(DeepAnatomicalStep):
    backend = "deepprep"
