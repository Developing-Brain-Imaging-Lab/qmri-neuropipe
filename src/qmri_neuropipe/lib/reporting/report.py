from pathlib import Path
from typing import List, Dict, Optional, Any
import logging
from datetime import datetime
import base64
import os
import contextlib
import json

try:
    from jinja2 import Environment, FileSystemLoader
except ImportError:
    Environment = None

# WeasyPrint check
try:
    # On macOS, WeasyPrint might need help finding Pango if installed via Homebrew
    import sys
    import os
    if sys.platform == "darwin":
        # Add common Homebrew paths to DYLD_FALLBACK_LIBRARY_PATH if not already there
        brew_lib = "/opt/homebrew/lib"
        if os.path.exists(brew_lib):
            fallback = os.environ.get("DYLD_FALLBACK_LIBRARY_PATH", "")
            if brew_lib not in fallback:
                os.environ["DYLD_FALLBACK_LIBRARY_PATH"] = f"{fallback}:{brew_lib}".strip(":")

    with open(os.devnull, 'w') as f, contextlib.redirect_stderr(f), contextlib.redirect_stdout(f):
        import weasyprint
except Exception as e:
    # Log the reason why valid import failed (e.g. missing libraries)
    logging.getLogger("ReportGenerator").warning(f"WeasyPrint import failed/unavailable (PDF generation will be disabled): {e}")
    weasyprint = None

class ReportGenerator:
    """
    Generates HTML and PDF reports using Jinja2 templates.
    Supports structured hierarchical data and persistence across runs.
    """
    
    def __init__(self, output_dir: Path, title: str = "QMRI-Neuropipe Report"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.title = title
        self.logger = logging.getLogger("ReportGenerator")
        self.data_file = self.output_dir / "report_data.json"
        
        # Default Structured Data Storage
        self.data = {
            "header": {
                "title": title,
                "generated_at": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "subject": "N/A",
                "session": "",
                "bids_dir": "N/A",
                "work_dir": "N/A"
            },
            "participant": {
                 "summary": "N/A",
                 "details": {}
            },
            "anat": {
                "inputs": [],
                "steps": [],
                "summary_table": None,
                "outputs": []
            },
            "dmri": {
                "inputs": {"summary": "", "figures": []},
                "steps": [],
                "summary_table": None,
                "outputs": []
            },
            "commands": []
        }
        
        # Load existing data if available
        self._load_data()
        
        # Always update generation time and title for new run
        self.data["header"]["generated_at"] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.data["header"]["title"] = title
        
        # Template Setup
        if Environment:
            template_dir = Path(__file__).parent
            self.env = Environment(loader=FileSystemLoader(str(template_dir)))
            try:
                self.template = self.env.get_template("report_template.html")
            except Exception as e:
                self.logger.error(f"Failed to load report_template.html: {e}")
                self.template = None
        else:
            self.logger.error("Jinja2 not installed. Reporting will use fallback (not implemented) or fail.")
            self.template = None

    def _load_data(self):
        """Load data from JSON file if it exists."""
        if self.data_file.exists():
            try:
                with open(self.data_file, 'r') as f:
                    loaded_data = json.load(f)
                    # Deep merge or just update top-level keys?
                    # For a robust merge, we update keys that exist.
                    for key in loaded_data:
                        if key in self.data:
                            if isinstance(self.data[key], dict) and isinstance(loaded_data[key], dict):
                                self.data[key].update(loaded_data[key])
                            else:
                                self.data[key] = loaded_data[key]
                self.logger.info(f"Loaded existing report data from {self.data_file}")
            except Exception as e:
                self.logger.warning(f"Failed to load existing report data: {e}")

    def _save_data(self):
        """Save data to JSON file."""
        try:
            # We don't want to save the base64 images in the JSON? 
            # If we don't, then re-runs won't have the images from previous runs.
            # But the JSON will become massive.
            # However, the user wants the information retained.
            # For now, let's save everything. 
            # Note: Figures contain 'src' which is the base64 string.
            with open(self.data_file, 'w') as f:
                json.dump(self.data, f, indent=4)
        except Exception as e:
            self.logger.warning(f"Failed to save report data: {e}")

    def set_header_info(self, subject: str, session: Optional[str] = None, bids_dir: str = "", work_dir: str = ""):
        """Update header information."""
        self.data["header"]["subject"] = subject
        self.data["header"]["session"] = session or ""
        self.data["header"]["bids_dir"] = bids_dir
        self.data["header"]["work_dir"] = work_dir
        self._save_data()

    def set_participant_summary(self, summary: str, details: Optional[Dict[str, Any]] = None):
        """Set participant summary information."""
        self.data["participant"]["summary"] = summary
        if details:
            self.data["participant"]["details"].update(details)
        self._save_data()

    # --- Anatomical Reporting Methods ---
    
    def add_anat_input(self, modality: str, path: Path, figure_path: Optional[Path] = None, caption: str = ""):
        """Add anatomical input image info."""
        # Avoid duplicates based on modality
        self.data["anat"]["inputs"] = [item for item in self.data["anat"]["inputs"] if item.get("modality") != modality]
        
        item = {
            "modality": modality,
            "path": str(path)
        }
        if figure_path and figure_path.exists():
            item["figure"] = self._create_figure_obj(figure_path, f"{modality} Input", caption)
        self.data["anat"]["inputs"].append(item)
        self._save_data()

    def add_anat_step(self, step_name: str, details: Dict[str, Any], figures: List[Dict[str, str]] = None, commands: List[Dict[str, Any]] = None):
        """
        Add an anatomical processing step.
        figures: List of dicts with keys 'path', 'title', 'caption'
        """
        # Overwrite if step with same name and modality exists, else append
        existing = None
        modality = details.get("Modality")
        for s in self.data["anat"]["steps"]:
            if s["name"] == step_name:
                if modality and s.get("details", {}).get("Modality") == modality:
                     existing = s
                     break
                elif not modality and not s.get("details", {}).get("Modality"):
                     existing = s
                     break
        
        step = {
            "name": step_name,
            "details": details,
            "figures": [],
            "commands": commands or []
        }
        if figures:
            for fig in figures:
                path = Path(fig["path"])
                if path.exists():
                    step["figures"].append(self._create_figure_obj(path, fig.get("title", ""), fig.get("caption", "")))
        
        if existing:
            existing.update(step)
        else:
            self.data["anat"]["steps"].append(step)
        
        self._save_data()

    def add_anat_summary(self, title: str, data: List[Dict[str, str]]):
        """Add anatomical summary table."""
        if not data: return
        self.data["anat"]["summary_table"] = {
            "title": title,
            "columns": list(data[0].keys()),
            "rows": data
        }
        self._save_data()

    # --- dMRI Reporting Methods ---
    
    def set_dmri_input_summary(self, summary_text: str):
        self.data["dmri"]["inputs"]["summary"] = summary_text
        self._save_data()

    def add_dmri_input_figure(self, path: Path, caption: str):
        if path.exists():
             # Avoid duplicate figures based on caption or path? 
             # Let's just append for now but maybe clear if it's the main input?
             self.data["dmri"]["inputs"]["figures"].append(self._create_figure_obj(path, "dMRI Input", caption))
        self._save_data()

    def add_dmri_step(self, step_name: str, details: Dict[str, Any], figures: List[Dict[str, str]] = None, tables: List[Dict[str, Any]] = None, commands: List[Dict[str, Any]] = None):
        """
        Add a dMRI processing step.
        tables: List of dicts {title: str, data: List[Dict]}
        """
        # Overwrite if step with same name exists? 
        # For dMRI, sometimes we have multiple instances of same step (per image).
        # Actually PreprocessingWorkflow._report_step uses the step name.
        # But for per-image steps, the name is same.
        # If we overwrite, we only see the LAST image.
        # So we should only overwrite if it's a GLOBAL step.
        # Or better: check if step+details matches? Too complex.
        
        # Let's check if the user wants "all steps run".
        # If per-image, maybe we want to append.
        
        step = {
            "name": step_name,
            "details": details,
            "figures": [],
            "tables": [],
            "commands": commands or []
        }
        if figures:
            for fig in figures:
                path = Path(fig["path"])
                if path.exists():
                    step["figures"].append(self._create_figure_obj(path, fig.get("title", ""), fig.get("caption", "")))
        
        if tables:
            for tbl in tables:
                if tbl.get("data"):
                     step["tables"].append({
                         "title": tbl.get("title", ""),
                         "columns": list(tbl["data"][0].keys()),
                         "rows": tbl["data"]
                     })
        
        # Check if we should overwrite or append.
        # For now, if details has 'File' or 'Stem', it's per-image.
        # If it's a global step (like Topup), we overwrite.
        
        is_global = step_name in ["TopupStep", "Synb0EstimationStep", "MergeStep", "DMRIReorientStep", "EddyQuadStep", "EddyCorrectionStep", "NativeDrbuddiStep"]
        
        if is_global:
            existing = None
            for s in self.data["dmri"]["steps"]:
                if s["name"] == step_name:
                    existing = s
                    break
            if existing:
                existing.update(step)
            else:
                self.data["dmri"]["steps"].append(step)
        else:
            # For per-image steps, we might want to avoid exact duplicates (same step, same file)
            file_ref = details.get("File") or details.get("Stem") or details.get("Image")
            if file_ref:
                existing = None
                for s in self.data["dmri"]["steps"]:
                    if s["name"] == step_name:
                        s_ref = s.get("details", {}).get("File") or s.get("details", {}).get("Stem") or s.get("details", {}).get("Image")
                        if s_ref == file_ref:
                            existing = s
                            break
                if existing:
                    existing.update(step)
                else:
                    self.data["dmri"]["steps"].append(step)
            else:
                self.data["dmri"]["steps"].append(step)
        
        self._save_data()

    def add_dmri_summary(self, title: str, data: List[Dict[str, str]]):
        if not data: return
        self.data["dmri"]["summary_table"] = {
            "title": title,
            "columns": list(data[0].keys()),
            "rows": data
        }
        self._save_data()

    def set_dmri_outputs(self, outputs: List[Dict[str, str]]):
        """Set final dMRI output files list."""
        self.data["dmri"]["outputs"] = outputs
        self._save_data()

    def set_anat_outputs(self, outputs: List[Dict[str, str]]):
        """Set final anatomical output files list."""
        self.data["anat"]["outputs"] = outputs
        self._save_data()

    # --- Generic/Helpers ---

    def _create_figure_obj(self, path: Path, title: str, caption: str) -> Dict[str, str]:
        """Read image and create base64 src object."""
        try:
            with open(path, "rb") as img_f:
                encoded = base64.b64encode(img_f.read()).decode('utf-8')
                mime = "image/png"
                if path.suffix.lower() in ['.jpg', '.jpeg']: mime = "image/jpeg"
                elif path.suffix.lower() == '.svg': mime = "image/svg+xml"
                src = f"data:{mime};base64,{encoded}"
                return {"src": src, "title": title, "caption": caption}
        except Exception as e:
            self.logger.warning(f"Failed to embed image {path}: {e}")
            return {"src": "", "title": "Error", "caption": "Image load failed"}

    def generate(self, filename: str = "report.html"):
        """Generate the HTML report."""
        if not self.template:
            self.logger.error("Template not loaded. Cannot generate report.")
            return None
            
        out_file = self.output_dir / filename
        try:
            self._refresh_command_history()
            html_content = self.template.render(**self.data)
            out_file.write_text(html_content, encoding='utf-8')
            self.logger.info(f"Report generated: {out_file}")
            return out_file
        except Exception as e:
            self.logger.error(f"Failed to render report: {e}")
            return None

    def generate_pdf(self, filename: str = "report.pdf"):
        """Generate PDF report (requires weasyprint)."""
        if not weasyprint:
            self.logger.debug("weasyprint not installed. Skipping PDF generation.")
            return None
            
        # Re-render to ensure latest data
        if not self.template: return None
        self._refresh_command_history()
        html_content = self.template.render(**self.data)
        
        out_file = self.output_dir / filename
        try:
            weasyprint.HTML(string=html_content).write_pdf(out_file)
            self.logger.info(f"PDF Report generated: {out_file}")
            return out_file
        except Exception as e:
            self.logger.error(f"Failed to generate PDF: {e}")
            return None

    def _refresh_command_history(self):
        """Attach all commands captured during this process to report data."""
        try:
            from qmri_neuropipe.core.run import get_command_history
            self.data["commands"] = get_command_history()
            self._save_data()
        except Exception as e:
            self.logger.debug(f"Could not refresh command history for report: {e}")
