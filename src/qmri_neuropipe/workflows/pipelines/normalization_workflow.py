from pathlib import Path
from typing import Optional

from qmri_neuropipe.core import BaseWorkflow
from qmri_neuropipe.lib.dmri.normalization import NormalizationStep


class NormalizationWorkflow(BaseWorkflow):
    """
    Workflow wrapper for diffusion metric normalization.
    """

    def _initialize_steps(self) -> None:
        self.modality = "Diffusion"
        self.steps = []

    def build_pipeline(self, context: dict) -> None:
        self.steps = []

        cfg = (self.config.get("dmri") or {}).get("normalization", {})
        if not cfg:
            return

        enabled = cfg.get("enabled")
        if enabled is False:
            return

        step = NormalizationStep(
            self.config,
            self.logger,
            self.provenance,
            template=cfg.get("template"),
            driving_metric=cfg.get("driving_metric", "FA"),
            space_name=cfg.get("space_name", cfg.get("space", "Standard")),
            tool=cfg.get("tool", "ants"),
            save_transforms=cfg.get("save_transforms", True),
            transform_type=cfg.get("transform_type", "SyN"),
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

        # Only one step for now, keep it explicit for clarity.
        result = self.steps[0].run(context, out_dir)
        return result
