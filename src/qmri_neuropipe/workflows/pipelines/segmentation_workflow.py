from pathlib import Path
from typing import Optional

from qmri_neuropipe.core import BaseWorkflow, PipelineContext
from qmri_neuropipe.lib.common.segmentation import SegmentationStep
from qmri_neuropipe.lib.common.analysis import AtlasRegistrationStep, StatsExtractionStep


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

        if cfg.get("atlases"):
            self.add_step(AtlasRegistrationStep(self.config, self.logger, self.provenance))
            self.add_step(StatsExtractionStep(self.config, self.logger, self.provenance))
        else:
            step = SegmentationStep(
                self.config,
                self.logger,
                self.provenance,
                atlas_file=cfg.get("atlas_file"),
                atlas_labels=cfg.get("atlas_labels"),
                metrics=cfg.get("metrics"),
                atlas_threshold=cfg.get("atlas_threshold"),
                include_zero_label=cfg.get("include_zero_label"),
                background_label=cfg.get("background_label"),
            )
            self.add_step(step)

    def run(
        self,
        output_dir: Path,
        context: dict,
        reporter=None,
        final_output_dir: Optional[Path] = None,
    ) -> PipelineContext:
        context = PipelineContext.ensure(context)
        if not self.steps:
            return context

        out_dir = final_output_dir or output_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        result = context
        for step in self.steps:
            result = step.run(result, out_dir)
        return PipelineContext.ensure(result)
