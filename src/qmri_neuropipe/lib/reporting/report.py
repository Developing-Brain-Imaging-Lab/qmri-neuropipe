"""
Report Generation Module
"""
from pathlib import Path
from typing import List, Dict, Optional, Union
import logging
from datetime import datetime
import base64

try:
    from jinja2 import Template
except ImportError:
    Template = None

try:
    import weasyprint
except Exception: # ImportError or OSError (dll missing)
    weasyprint = None

class ReportGenerator:
    """
    Generates HTML and PDF reports for pipeline execution.
    """
    
    def __init__(self, output_dir: Path, title: str = "Pipeline Report"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.title = title
        self.sections = []
        self.logger = logging.getLogger("ReportGenerator")

    def add_section(self, title: str, content: str = ""):
        """Add a text section."""
        self.sections.append({
            "type": "text",
            "title": title,
            "content": content
        })

    def add_figure(self, title: str, image_path: Path, caption: str = ""):
        """Add a figure section."""
        if not Path(image_path).exists():
             self.logger.warning(f"Figure not found: {image_path}")
             return
             
        self.sections.append({
            "type": "figure",
            "title": title,
            "path": Path(image_path), # absolute path needed? or relative to report?
            # Browser needs relative or accessible path. 
            # Ideally verify readability.
            "caption": caption
        })

    def add_summary_table(self, title: str, data: List[Dict[str, str]]):
        """
        Add a summary table section.
        data: List of dicts, e.g. [{"Step": "Denoise", "Status": "Done", "Duration": "10s"}]
        """
        self.sections.append({
            "type": "table",
            "title": title,
            "data": data
        })

    def _render_html(self) -> str:
        """Render the report to HTML string."""
        
        # Simple CSS
        css = """
        body { font-family: sans-serif; margin: 40px; color: #333; }
        h1 { color: #2c3e50; border-bottom: 2px solid #eee; padding-bottom: 10px; }
        h2 { color: #34495e; margin-top: 30px; }
        .section { margin-bottom: 20px; }
        .figure { text-align: center; margin: 20px 0; border: 1px solid #eee; padding: 10px; background: #fafafa; }
        img { max-width: 70%; height: auto; border-radius: 4px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
        .caption { margin-top: 10px; font-style: italic; color: #666; font-size: 0.9em; }
        .footer { margin-top: 50px; font-size: 0.8em; color: #999; border-top: 1px solid #eee; padding-top: 10px; }
        table { width: 100%; border-collapse: collapse; margin-top: 10px; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
        tr:nth-child(even) { background-color: #f9f9f9; }
        """
        
        html_parts = [
            f"<!DOCTYPE html><html><head><title>{self.title}</title><style>{css}</style></head><body>",
            f"<h1>{self.title}</h1>",
            f"<p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>"
        ]
        
        for section in self.sections:
            if section["type"] == "text":
                html_parts.append(f"<div class='section'><h2>{section['title']}</h2><p>{section['content']}</p></div>")
            elif section["type"] == "figure":
                # Handle image path logic: 
                # If render locally, absolute path might work in some browsers via file:// but forbidden in others.
                # Embedding as base64 is safest for portable single-file HTML/PDF.
                img_path = section["path"]
                try:
                    with open(img_path, "rb") as img_f:
                        encoded = base64.b64encode(img_f.read()).decode('utf-8')
                        mime = "image/png" # detect?
                        if img_path.suffix.lower() in ['.jpg', '.jpeg']: mime = "image/jpeg"
                        elif img_path.suffix.lower() == '.svg': mime = "image/svg+xml"
                        
                        src = f"data:{mime};base64,{encoded}"
                        
                except Exception as e:
                    self.logger.warning(f"Failed to embed image {img_path}: {e}")
                    continue
                    
                html_parts.append(
                    f"<div class='figure'>"
                    f"<h2>{section['title']}</h2>"
                    f"<img src='{src}' alt='{section['title']}'>"
                    f"<p class='caption'>{section['caption']}</p>"
                    f"</div>"
                )
            elif section["type"] == "table":
                # Render table
                if not section['data']: continue
                
                rows = section['data']
                cols = list(rows[0].keys())
                
                # Check for Status column for coloring
                
                thead = "".join(f"<th>{c}</th>" for c in cols)
                tbody = ""
                for row in rows:
                    tbody += "<tr>"
                    for c in cols:
                        # Optional: colorize specific values
                        val = row.get(c, "")
                        tbody += f"<td>{val}</td>"
                    tbody += "</tr>"
                
                html_parts.append(
                    f"<div class='section'><h2>{section['title']}</h2>"
                    f"<table><thead><tr>{thead}</tr></thead><tbody>{tbody}</tbody></table>"
                    f"</div>"
                )
        
        html_parts.append("<div class='footer'>qmri-neuropipe report</div></body></html>")
        return "\n".join(html_parts)

    def generate(self, filename: str = "report.html"):
        """Generate the HTML report."""
        out_file = self.output_dir / filename
        html_content = self._render_html()
        out_file.write_text(html_content, encoding='utf-8')
        self.logger.info(f"Report generated: {out_file}")
        return out_file

    def generate_pdf(self, filename: str = "report.pdf"):
        """Generate PDF report (requires weasyprint)."""
        if not weasyprint:
            self.logger.warning("weasyprint not installed. Skipping PDF generation.")
            return None
            
        html_content = self._render_html()
        out_file = self.output_dir / filename
        try:
            weasyprint.HTML(string=html_content).write_pdf(out_file)
            self.logger.info(f"PDF Report generated: {out_file}")
            return out_file
        except Exception as e:
            self.logger.error(f"Failed to generate PDF: {e}")
            return None
