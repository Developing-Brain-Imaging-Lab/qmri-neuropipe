import logging
from pathlib import Path
from src.qmri_neuropipe.lib.dmri.eddy import EddyCorrectionStep
from src.qmri_neuropipe.core.types import DWIFile

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Test")
class MockConfig:
    def get(self, *args, **kwargs):
        if args[0] == 'eddy': return {}
        return False
config = MockConfig()

step = EddyCorrectionStep(config=config, logger=logger, provenance=None)
dwi = DWIFile(img=Path("/tmp/qmri_skip_test/sub-01/ses-01/dwi/sub-01.nii.gz"), bval=Path("dummy"), bvec=Path("dummy"), entities={"sub": "01", "ses": "01"})

out_dir = Path("/tmp/qmri_skip_test/sub-01/ses-01/dwi")

try:
    step.run(context={"current_image": dwi}, output_dir=out_dir)
except Exception as e:
    pass

