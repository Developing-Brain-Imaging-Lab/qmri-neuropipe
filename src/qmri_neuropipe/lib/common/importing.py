
from pathlib import Path
from typing import Optional, Dict, Any
import logging

from ...core import BaseProcessingStep, ValidationError, ProcessingError
from ...interfaces import dcm2niix, dcm2bids
from .gnl_metadata import GEGnlMetadataEnrichmentStep

class Dcm2NiixStep(BaseProcessingStep):
    """
    Step to convert DICOMs to NIfTI using dcm2niix.
    """
    def run(self, dicom_dir: Path, output_dir: Path, **kwargs) -> Path:
        self.logger.info(f"Converting DICOMs in {dicom_dir} to NIfTI...")
        
        # Get options from config
        cfg = self.config.get("import", {}).get("dcm2niix", {})
        filename = cfg.get("filename", "%p_%s_%t")
        compress = cfg.get("compress", True)
        bids = cfg.get("bids", True)
        
        # Override with kwargs if provided
        filename = kwargs.get("filename", filename)
        
        try:
            dcm2niix.dcm2niix(
                in_dir=dicom_dir,
                out_dir=output_dir,
                filename=filename,
                compress=compress,
                bids=bids,
                verbose=self.config.debug
            )
            return output_dir
        except Exception as e:
            self.logger.error(f"dcm2niix conversion failed: {e}")
            if self.config.stop_on_error:
                raise ProcessingError(f"dcm2niix failed: {e}")
            return dicom_dir

class Dcm2BidsStep(BaseProcessingStep):
    """
    Step to convert DICOMs to BIDS structure using dcm2bids.
    """
    def run(self, dicom_dir: Path, bids_dir: Path, **kwargs) -> Path:
        self.logger.info(f"Running dcm2bids on {dicom_dir}...")
        
        # Subject and Session from context or kwargs
        sub = kwargs.get("subject") or self.config.get("subject")
        ses = kwargs.get("session") or self.config.get("session")
        
        if not sub:
            raise ValidationError("Subject ID required for dcm2bids.")

        # Config file is mandatory for dcm2bids
        import_cfg = self.config.get("import", {}).get("dcm2bids", {})
        config_file = import_cfg.get("config_file")
        if not config_file:
            # Check for a default in common locations
            default_config = self.config.bids_dir / "code" / "dcm2bids_config.json"
            if default_config.exists():
                config_file = default_config
            else:
                raise ValidationError("dcm2bids config_file not specified in configuration.")
        
        config_file = Path(config_file)
        if not config_file.exists():
            raise ValidationError(f"dcm2bids config file not found: {config_file}")

        try:
            dcm2bids.dcm2bids(
                dicom_dir=dicom_dir,
                participant_id=sub,
                config_file=config_file,
                output_dir=bids_dir,
                session_id=ses,
                clobber=kwargs.get("clobber", False)
            )
            return bids_dir
        except Exception as e:
            self.logger.error(f"dcm2bids failed: {e}")
            if self.config.stop_on_error:
                raise ProcessingError(f"dcm2bids failed: {e}")
            return bids_dir


class ImportGnlMetadataStep(GEGnlMetadataEnrichmentStep):
    """
    Alias step for import workflow readability.
    """
    pass
