from pathlib import Path
from typing import Optional

from qmri_neuropipe.core import BaseWorkflow
from qmri_neuropipe.lib.common.segmentation import SegmentationStep


class SegmentationWorkflow(BaseWorkflow):
    """
    Workflow wrapper for ROI segmentation and stats extraction.
    """

    def _initialize_steps(self) -> None:
        self.modality = "Diffusion"
        self.steps = []

    def build_pipeline(self, context: dict) -> None:
        self.steps = []

        cfg = (self.config.get("dmri") or {}).get("analysis", {})
        if not cfg:
            return

        enabled = cfg.get("enabled")
        if enabled is False:
            return

        step = SegmentationStep(
            self.config,
            self.logger,
            self.provenance,
            atlas_file=cfg.get("atlas_file"),
            atlas_labels=cfg.get("atlas_labels"),
            metrics=cfg.get("metrics"),
            atlas_threshold=cfg.get("atlas_threshold"),
        )
        self.add_step(step)

    def run(
        self,
        output_dir: Path,
        context: dict,
        reporter=None,
        final_output_dir: Optional[Path] = None,
    ) -> dict:
        if not self.steps:
            return context

        out_dir = final_output_dir or output_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        result = self.steps[0].run(context, out_dir)
        return result
