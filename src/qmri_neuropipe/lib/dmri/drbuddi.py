"""
Native DRBUDDI-like distortion refinement for reverse phase-encoded dMRI.

This is a pragmatic, ANTs-based approximation intended to run after Eddy
correction on a merged DWI series. It estimates a midpoint b0 target from
paired reverse-PE groups and warps each PE segment to that midpoint.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, List, Optional, Tuple

import nibabel as nib
import numpy as np

from ...core import BaseProcessingStep, ProcessingError
from ...core.types import DWIFile
from ...interfaces import ants
from ...io.bids import build_bids_name
from ..common.json_metadata import copy_json_with_metadata


class NativeDrbuddiStep(BaseProcessingStep):
    """
    ANTs-based reverse-PE distortion refinement inspired by DRBUDDI.

    Expected usage:
    - run after Eddy on a merged DWI series
    - use merge_source_info from MergeStep to recover PE-specific volume ranges
    """

    def __init__(
        self,
        config,
        logger=None,
        provenance=None,
        transform_type: str = "SyNOnly",
        interpolator: str = "linear",
        registration_options: Optional[dict[str, Any]] = None,
        symmetric_pairwise: bool = True,
        pe_axis_constraint: float = 1.0,
    ):
        super().__init__(config, logger, provenance)
        self.transform_type = transform_type
        self.interpolator = interpolator
        self.registration_options = registration_options or {}
        self.symmetric_pairwise = symmetric_pairwise
        self.pe_axis_constraint = float(pe_axis_constraint)

    def validate_inputs(self, first_arg, **kwargs) -> None:
        pass

    def validate_outputs(self, result) -> None:
        pass

    def run(self, first_arg, output_dir: Path, **kwargs):
        context, _ = self.unpack_input(first_arg)
        if context is None:
            raise ProcessingError("NativeDrbuddiStep must run in pipeline context mode.")

        if not context.get("do_drbuddi", False):
            self.logger.info("Native DRBUDDI skipped (do_drbuddi=False).")
            return context

        dwi_files: List[DWIFile] = context.get("dwi_files", [])
        if len(dwi_files) != 1:
            self.logger.warning(
                "Native DRBUDDI expects a single merged DWI after Eddy; found %d input files. Skipping.",
                len(dwi_files),
            )
            return context

        input_dwi = dwi_files[0]
        merge_source_info = context.get("merge_source_info", [])
        if not merge_source_info:
            self.logger.warning("Native DRBUDDI requires merge_source_info from MergeStep. Skipping.")
            return context

        output_dir = self.get_step_output_dir(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        grouped_infos = self._paired_phase_groups(merge_source_info)
        if not grouped_infos:
            self.logger.warning("Native DRBUDDI found no paired reverse-PE groups. Skipping.")
            return context

        corrected, qc_summary = self._run_native_drbuddi(input_dwi, grouped_infos, output_dir)
        self._propagate_sidecars(input_dwi, corrected)

        topup_map = context.get("topup_map", {})
        if input_dwi.img in topup_map:
            topup_map[corrected.img] = topup_map[input_dwi.img]
            topup_map[str(corrected.img)] = topup_map[input_dwi.img]
            context["topup_map"] = topup_map
        elif str(input_dwi.img) in topup_map:
            topup_map[corrected.img] = topup_map[str(input_dwi.img)]
            topup_map[str(corrected.img)] = topup_map[str(input_dwi.img)]
            context["topup_map"] = topup_map

        context["dwi_files"] = [corrected]
        context["current_image"] = corrected
        context["drbuddi_qc"] = qc_summary
        return context

    def _paired_phase_groups(self, merge_source_info: list[dict[str, Any]]) -> list[dict[str, Any]]:
        axis_groups: dict[str, dict[int, list[dict[str, Any]]]] = {}

        for info in merge_source_info:
            ped = str(info.get("phase_encoding_direction") or "").strip()
            if not ped:
                continue
            axis = ped[0]
            sign = -1 if ped.endswith("-") else 1
            axis_groups.setdefault(axis, {1: [], -1: []})[sign].append(info)

        paired = []
        for axis, sign_groups in axis_groups.items():
            if sign_groups[1] and sign_groups[-1]:
                paired.append({"axis": axis, "positive": sign_groups[1], "negative": sign_groups[-1]})
        return paired

    def _run_native_drbuddi(
        self,
        input_dwi: DWIFile,
        grouped_infos: list[dict[str, Any]],
        output_dir: Path,
    ) -> Tuple[DWIFile, list[dict[str, Any]]]:
        merged_img = nib.load(str(input_dwi.img))
        merged_data = merged_img.get_fdata(dtype=np.float32)
        corrected_data = np.array(merged_data, copy=True)
        merged_bvals = np.atleast_1d(np.loadtxt(input_dwi.bval)) if input_dwi.bval else None
        if merged_bvals is None:
            raise ProcessingError("Native DRBUDDI requires a merged bval file.")
        qc_summary: list[dict[str, Any]] = []

        mean_b0_before = self._mean_b0_from_data(merged_data, merged_bvals)
        if mean_b0_before is not None:
            nib.save(
                nib.Nifti1Image(mean_b0_before, merged_img.affine, merged_img.header),
                str(output_dir / "drbuddi_before_meanb0.nii.gz"),
            )

        for group_idx, group in enumerate(grouped_infos):
            pos_mean = self._mean_b0_for_infos(merged_data, merged_bvals, group["positive"])
            neg_mean = self._mean_b0_for_infos(merged_data, merged_bvals, group["negative"])
            if pos_mean is None or neg_mean is None:
                self.logger.warning("Skipping native DRBUDDI group %s due to missing b0 volumes.", group["axis"])
                continue

            midpoint = 0.5 * (pos_mean + neg_mean)
            midpoint_path = output_dir / f"drbuddi_axis-{group['axis']}_midpoint.nii.gz"
            pos_mean_path = output_dir / f"drbuddi_axis-{group['axis']}_pos_meanb0.nii.gz"
            neg_mean_path = output_dir / f"drbuddi_axis-{group['axis']}_neg_meanb0.nii.gz"
            residual_before_path = output_dir / f"drbuddi_axis-{group['axis']}_residual_before.nii.gz"
            nib.save(nib.Nifti1Image(midpoint, merged_img.affine, merged_img.header), str(midpoint_path))
            nib.save(nib.Nifti1Image(pos_mean, merged_img.affine, merged_img.header), str(pos_mean_path))
            nib.save(nib.Nifti1Image(neg_mean, merged_img.affine, merged_img.header), str(neg_mean_path))
            nib.save(
                nib.Nifti1Image(np.abs(pos_mean - neg_mean).astype(np.float32), merged_img.affine, merged_img.header),
                str(residual_before_path),
            )

            pos_warp, neg_warp = self._build_group_transforms(
                pos_mean_path=pos_mean_path,
                neg_mean_path=neg_mean_path,
                midpoint_path=midpoint_path,
                axis=group["axis"],
                output_dir=output_dir,
            )

            for info in group["positive"]:
                corrected_data = self._apply_segment_transform(
                    corrected_data, merged_img, midpoint_path, pos_warp, info, output_dir
                )
            for info in group["negative"]:
                corrected_data = self._apply_segment_transform(
                    corrected_data, merged_img, midpoint_path, neg_warp, info, output_dir
                )

            pos_mean_after = self._mean_b0_for_infos(corrected_data, merged_bvals, group["positive"])
            neg_mean_after = self._mean_b0_for_infos(corrected_data, merged_bvals, group["negative"])
            if pos_mean_after is not None and neg_mean_after is not None:
                pos_mean_after_path = output_dir / f"drbuddi_axis-{group['axis']}_pos_meanb0_after.nii.gz"
                neg_mean_after_path = output_dir / f"drbuddi_axis-{group['axis']}_neg_meanb0_after.nii.gz"
                residual_after_path = output_dir / f"drbuddi_axis-{group['axis']}_residual_after.nii.gz"
                residual_before = np.abs(pos_mean - neg_mean)
                residual_after = np.abs(pos_mean_after - neg_mean_after)
                nib.save(nib.Nifti1Image(pos_mean_after, merged_img.affine, merged_img.header), str(pos_mean_after_path))
                nib.save(nib.Nifti1Image(neg_mean_after, merged_img.affine, merged_img.header), str(neg_mean_after_path))
                nib.save(
                    nib.Nifti1Image(residual_after.astype(np.float32), merged_img.affine, merged_img.header),
                    str(residual_after_path),
                )
                before_mean = float(np.mean(residual_before))
                after_mean = float(np.mean(residual_after))
                improvement = 0.0
                if before_mean > 0:
                    improvement = 100.0 * (before_mean - after_mean) / before_mean
                qc_summary.append(
                    {
                        "axis": group["axis"],
                        "positive_series": len(group["positive"]),
                        "negative_series": len(group["negative"]),
                        "residual_before_mean": before_mean,
                        "residual_after_mean": after_mean,
                        "residual_before_p95": float(np.percentile(residual_before, 95)),
                        "residual_after_p95": float(np.percentile(residual_after, 95)),
                        "improvement_percent": improvement,
                        "residual_before_path": str(residual_before_path),
                        "residual_after_path": str(residual_after_path),
                        "pos_mean_before_path": str(pos_mean_path),
                        "neg_mean_before_path": str(neg_mean_path),
                        "pos_mean_after_path": str(pos_mean_after_path),
                        "neg_mean_after_path": str(neg_mean_after_path),
                    }
                )

        ents = dict(input_dwi.entities)
        old_desc = ents.get("desc", "")
        ents["desc"] = f"{old_desc}drbuddi" if old_desc else "drbuddi"
        out_img = output_dir / build_bids_name(ents)
        nib.save(nib.Nifti1Image(corrected_data, merged_img.affine, merged_img.header), str(out_img))

        out_json = out_img.with_suffix("").with_suffix(".json")
        out_bval = out_img.with_suffix("").with_suffix(".bval")
        out_bvec = out_img.with_suffix("").with_suffix(".bvec")

        result = DWIFile(
            img=out_img,
            entities=ents,
            json=out_json,
            bval=out_bval if input_dwi.bval else None,
            bvec=out_bvec if input_dwi.bvec else None,
            Delta=getattr(input_dwi, "Delta", None),
            delta=getattr(input_dwi, "delta", None),
        )
        mean_b0_after = self._mean_b0_from_data(corrected_data, merged_bvals)
        if mean_b0_after is not None:
            nib.save(
                nib.Nifti1Image(mean_b0_after, merged_img.affine, merged_img.header),
                str(output_dir / "drbuddi_after_meanb0.nii.gz"),
            )
        with (output_dir / "drbuddi_qc.json").open("w") as f:
            json.dump(qc_summary, f, indent=2)
            f.write("\n")
        return result, qc_summary

    def _mean_b0_from_data(
        self,
        data: np.ndarray,
        bvals: np.ndarray,
        b0_threshold: float = 50.0,
    ) -> Optional[np.ndarray]:
        b0_idx = np.where(np.asarray(bvals) < b0_threshold)[0]
        if b0_idx.size == 0:
            return None
        return np.mean(data[..., b0_idx], axis=-1).astype(np.float32)

    def _mean_b0_for_infos(
        self,
        merged_data: np.ndarray,
        merged_bvals: np.ndarray,
        infos: list[dict[str, Any]],
        b0_threshold: float = 50.0,
    ) -> Optional[np.ndarray]:
        b0_volumes = []
        for info in infos:
            start = int(info["start"])
            stop = int(info["stop"])
            seg_bvals = merged_bvals[start:stop]
            if seg_bvals.size == 0:
                continue
            b0_idx = np.where(seg_bvals < b0_threshold)[0]
            for idx in b0_idx:
                b0_volumes.append(merged_data[..., start + int(idx)])

        if not b0_volumes:
            return None
        return np.mean(np.stack(b0_volumes, axis=-1), axis=-1)

    def _build_group_transforms(
        self,
        pos_mean_path: Path,
        neg_mean_path: Path,
        midpoint_path: Path,
        axis: str,
        output_dir: Path,
    ) -> Tuple[list[Path], list[Path]]:
        if self.symmetric_pairwise:
            try:
                return self._build_symmetric_halfway_transforms(
                    pos_mean_path=pos_mean_path,
                    neg_mean_path=neg_mean_path,
                    axis=axis,
                    output_dir=output_dir,
                )
            except Exception as e:
                self.logger.warning(
                    "Native DRBUDDI symmetric pairwise registration failed for axis %s (%s). Falling back to midpoint registration.",
                    axis,
                    e,
                )

        pos_warp = self._register_to_midpoint(
            moving_path=pos_mean_path,
            midpoint_path=midpoint_path,
            output_dir=output_dir,
            prefix=f"drbuddi_axis-{axis}_pos_",
        )
        neg_warp = self._register_to_midpoint(
            moving_path=neg_mean_path,
            midpoint_path=midpoint_path,
            output_dir=output_dir,
            prefix=f"drbuddi_axis-{axis}_neg_",
        )
        return pos_warp, neg_warp

    def _build_symmetric_halfway_transforms(
        self,
        pos_mean_path: Path,
        neg_mean_path: Path,
        axis: str,
        output_dir: Path,
    ) -> Tuple[list[Path], list[Path]]:
        os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(self.config.get("n_cpus", 1))
        import ants as ants_lib

        reg_prefix = output_dir / f"drbuddi_axis-{axis}_pairwise_"
        fixed_img = ants_lib.image_read(str(neg_mean_path))
        moving_img = ants_lib.image_read(str(pos_mean_path))
        reg = ants_lib.registration(
            fixed=fixed_img,
            moving=moving_img,
            type_of_transform=self.transform_type,
            outprefix=str(reg_prefix),
            **self.registration_options,
        )

        fwd = [Path(t) for t in reg.get("fwdtransforms", [])]
        inv = [Path(t) for t in reg.get("invtransforms", [])]
        pos_half = self._prepare_halfway_transforms(fwd, axis, output_dir, f"drbuddi_axis-{axis}_pos")
        neg_half = self._prepare_halfway_transforms(inv, axis, output_dir, f"drbuddi_axis-{axis}_neg")
        if not pos_half or not neg_half:
            raise ProcessingError("Symmetric pairwise registration did not produce usable nonlinear warp fields.")
        return pos_half, neg_half

    def _register_to_midpoint(
        self,
        moving_path: Path,
        midpoint_path: Path,
        output_dir: Path,
        prefix: str,
    ) -> list[Path]:
        reg_prefix = output_dir / prefix
        _, transforms = ants.registration(
            fixed_file=midpoint_path,
            moving_file=moving_path,
            out_prefix=reg_prefix,
            transform_type=self.transform_type,
            interpolator=self.interpolator,
            nthreads=self.config.get("n_cpus", 1),
            **self.registration_options,
        )
        return [Path(t) for t in transforms]

    def _prepare_halfway_transforms(
        self,
        transforms: list[Path],
        axis: str,
        output_dir: Path,
        label: str,
    ) -> list[Path]:
        prepared: list[Path] = []
        saw_warp = False

        for idx, transform in enumerate(transforms):
            transform = Path(transform)
            if "".join(transform.suffixes).endswith(".nii.gz") or transform.suffix == ".nii":
                half_warp = output_dir / f"{label}_halfwarp_{idx}.nii.gz"
                self._write_constrained_half_warp(transform, half_warp, axis)
                prepared.append(half_warp)
                saw_warp = True
            else:
                self.logger.warning(
                    "Native DRBUDDI symmetric mode is ignoring non-warp transform component %s. "
                    "Use transform_type=SyNOnly for the cleanest PE-constrained behavior.",
                    transform.name,
                )

        return prepared if saw_warp else []

    def _write_constrained_half_warp(self, src_warp: Path, dst_warp: Path, axis: str) -> None:
        warp_img = nib.load(str(src_warp))
        warp_data = warp_img.get_fdata(dtype=np.float32)
        axis_idx = self._axis_index(axis)
        ortho_scale = max(0.0, 1.0 - self.pe_axis_constraint)

        if warp_data.ndim < 4:
            raise ProcessingError(f"Unexpected warp field shape for {src_warp.name}: {warp_data.shape}")

        if warp_data.shape[-1] == 3:
            constrained = np.array(warp_data, copy=True)
            for idx in range(3):
                if idx == axis_idx:
                    constrained[..., idx] *= 0.5
                else:
                    constrained[..., idx] *= 0.5 * ortho_scale
        elif warp_data.ndim >= 5 and warp_data.shape[-2] == 1 and warp_data.shape[-1] == 3:
            constrained = np.array(warp_data, copy=True)
            for idx in range(3):
                if idx == axis_idx:
                    constrained[..., 0, idx] *= 0.5
                else:
                    constrained[..., 0, idx] *= 0.5 * ortho_scale
        else:
            raise ProcessingError(f"Unsupported warp field layout for {src_warp.name}: {warp_data.shape}")

        nib.save(nib.Nifti1Image(constrained, warp_img.affine, warp_img.header), str(dst_warp))

    @staticmethod
    def _axis_index(axis: str) -> int:
        axis = str(axis).lower()
        if axis == "i":
            return 0
        if axis == "j":
            return 1
        if axis == "k":
            return 2
        raise ProcessingError(f"Unsupported phase-encoding axis for native DRBUDDI: {axis}")

    def _apply_segment_transform(
        self,
        corrected_data: np.ndarray,
        merged_img: nib.Nifti1Image,
        midpoint_path: Path,
        transforms: list[Path],
        info: dict[str, Any],
        output_dir: Path,
    ) -> np.ndarray:
        start = int(info["start"])
        stop = int(info["stop"])
        seg = corrected_data[..., start:stop]
        if seg.ndim != 4 or seg.shape[-1] == 0:
            return corrected_data

        segment_name = Path(str(info.get("img", f"segment_{start}_{stop}"))).stem
        segment_path = output_dir / f"{segment_name}_segment_{start}_{stop}.nii.gz"
        warped_path = output_dir / f"{segment_name}_segment_{start}_{stop}_warped.nii.gz"
        nib.save(nib.Nifti1Image(seg, merged_img.affine, merged_img.header), str(segment_path))

        try:
            ants.apply_transforms(
                fixed_file=midpoint_path,
                moving_file=segment_path,
                out_file=warped_path,
                transforms=transforms,
                interpolator=self.interpolator,
                nthreads=self.config.get("n_cpus", 1),
                imagetype=3,
            )
            warped = nib.load(str(warped_path)).get_fdata(dtype=np.float32)
        except Exception:
            warped = self._apply_segment_transform_fallback(seg, midpoint_path, transforms, output_dir, segment_name)

        if warped.ndim == 3:
            warped = warped[..., np.newaxis]
        corrected_data[..., start:stop] = warped
        return corrected_data

    def _apply_segment_transform_fallback(
        self,
        segment_data: np.ndarray,
        midpoint_path: Path,
        transforms: list[Path],
        output_dir: Path,
        segment_name: str,
    ) -> np.ndarray:
        warped_vols = []
        for vol_idx in range(segment_data.shape[-1]):
            in_vol = output_dir / f"{segment_name}_vol{vol_idx:04d}.nii.gz"
            out_vol = output_dir / f"{segment_name}_vol{vol_idx:04d}_warped.nii.gz"
            midpoint_img = nib.load(str(midpoint_path))
            nib.save(nib.Nifti1Image(segment_data[..., vol_idx], midpoint_img.affine, midpoint_img.header), str(in_vol))
            ants.apply_transforms(
                fixed_file=midpoint_path,
                moving_file=in_vol,
                out_file=out_vol,
                transforms=transforms,
                interpolator=self.interpolator,
                nthreads=self.config.get("n_cpus", 1),
            )
            warped_vols.append(nib.load(str(out_vol)).get_fdata(dtype=np.float32))
        return np.stack(warped_vols, axis=-1)

    def _propagate_sidecars(self, input_dwi: DWIFile, result: DWIFile) -> None:
        if input_dwi.bval and result.bval:
            shutil.copyfile(input_dwi.bval, result.bval)
        if input_dwi.bvec and result.bvec:
            shutil.copyfile(input_dwi.bvec, result.bvec)

        out_json = copy_json_with_metadata(getattr(input_dwi, "json", None), result.json)
        payload = {}
        if out_json and Path(out_json).exists():
            payload = json.loads(Path(out_json).read_text())
        payload["DistortionCorrection"] = {
            "Method": "native-drbuddi",
            "TransformType": self.transform_type,
            "Interpolator": self.interpolator,
            "SymmetricPairwise": self.symmetric_pairwise,
            "PeAxisConstraint": self.pe_axis_constraint,
        }
        with Path(result.json).open("w") as f:
            json.dump(payload, f, indent=2)
            f.write("\n")
