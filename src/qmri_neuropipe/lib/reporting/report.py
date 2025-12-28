from pathlib import Path
from typing import List, Dict, Optional, Union, Any
import logging
from datetime import datetime
import base64
import os
import contextlib

try:
    from jinja2 import Environment, FileSystemLoader
except ImportError:
    Environment = None

# WeasyPrint check
try:
    with open(os.devnull, 'w') as f, contextlib.redirect_stderr(f), contextlib.redirect_stdout(f):
        import weasyprint
except Exception as e:
    # Log the reason why valid import failed (e.g. missing libraries)
    logging.getLogger("ReportGenerator").warning(f"WeasyPrint import failed/unavailable (PDF generation will be disabled): {e}")
    weasyprint = None

class ReportGenerator:
    """
    Generates HTML and PDF reports using Jinja2 templates.
    Supports structured hierarchical data.
    """
    
    def __init__(self, output_dir: Path, title: str = "qmri-neuropipe Pipeline Report"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.title = title
        self.logger = logging.getLogger("ReportGenerator")
        
        # Structured Data Storage
        self.data = {
            "header": {
                "title": title,
                "generated_at": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "subject": "N/A",
                "session": "",
                "bids_dir": "N/A",
                "work_dir": "N/A"
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
            }
        }
        
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

    def set_header_info(self, subject: str, session: Optional[str] = None, bids_dir: str = "", work_dir: str = ""):
        """Update header information."""
        self.data["header"]["subject"] = subject
        self.data["header"]["session"] = session or ""
        self.data["header"]["bids_dir"] = bids_dir
        self.data["header"]["work_dir"] = work_dir

    # --- Anatomical Reporting Methods ---
    
    def add_anat_input(self, modality: str, path: Path, figure_path: Optional[Path] = None, caption: str = ""):
        """Add anatomical input image info."""
        item = {
            "modality": modality,
            "path": str(path)
        }
        if figure_path and figure_path.exists():
            item["figure"] = self._create_figure_obj(figure_path, f"{modality} Input", caption)
        self.data["anat"]["inputs"].append(item)

    def add_anat_step(self, step_name: str, details: Dict[str, Any], figures: List[Dict[str, str]] = None):
        """
        Add an anatomical processing step.
        figures: List of dicts with keys 'path', 'title', 'caption'
        """
        step = {
            "name": step_name,
            "details": details,
            "figures": []
        }
        if figures:
            for fig in figures:
                path = Path(fig["path"])
                if path.exists():
                    step["figures"].append(self._create_figure_obj(path, fig.get("title", ""), fig.get("caption", "")))
        
        self.data["anat"]["steps"].append(step)

    def add_anat_summary(self, title: str, data: List[Dict[str, str]]):
        """Add anatomical summary table."""
        if not data: return
        self.data["anat"]["summary_table"] = {
            "title": title,
            "columns": list(data[0].keys()),
            "rows": data
        }

    # --- dMRI Reporting Methods ---
    
    def set_dmri_input_summary(self, summary_text: str):
        self.data["dmri"]["inputs"]["summary"] = summary_text

    def add_dmri_input_figure(self, path: Path, caption: str):
        if path.exists():
             self.data["dmri"]["inputs"]["figures"].append(self._create_figure_obj(path, "dMRI Input", caption))

    def add_dmri_step(self, step_name: str, details: Dict[str, Any], figures: List[Dict[str, str]] = None, tables: List[Dict[str, Any]] = None):
        """
        Add a dMRI processing step.
        tables: List of dicts {title: str, data: List[Dict]}
        """
        step = {
            "name": step_name,
            "details": details,
            "figures": [],
            "tables": []
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
        
        self.data["dmri"]["steps"].append(step)

    def add_dmri_summary(self, title: str, data: List[Dict[str, str]]):
        if not data: return
        self.data["dmri"]["summary_table"] = {
            "title": title,
            "columns": list(data[0].keys()),
            "rows": data
        }

    def set_dmri_outputs(self, outputs: List[Dict[str, str]]):
        """Set final dMRI output files list."""
        self.data["dmri"]["outputs"] = outputs

    def set_anat_outputs(self, outputs: List[Dict[str, str]]):
        """Set final anatomical output files list."""
        self.data["anat"]["outputs"] = outputs

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
        html_content = self.template.render(**self.data)
        
        out_file = self.output_dir / filename
        try:
            weasyprint.HTML(string=html_content).write_pdf(out_file)
            self.logger.info(f"PDF Report generated: {out_file}")
            return out_file
        except Exception as e:
            self.logger.error(f"Failed to generate PDF: {e}")
            return None
