import os
import sys
from pathlib import Path
import logging
from datetime import datetime

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from qmri_neuropipe.lib.reporting.report import ReportGenerator, weasyprint

def test_report_cycle():
    print(f"Testing Report Cycle...")
    print(f"WeasyPrint status: {'Available' if weasyprint else 'NOT AVAILABLE'}")
    
    output_dir = Path("test_report_output")
    if output_dir.exists():
        import shutil
        shutil.rmtree(output_dir)
    output_dir.mkdir()
    
    # 1. Initialize and save
    reporter = ReportGenerator(output_dir, title="Verification Report")
    reporter.set_header_info(subject="sub-test", bids_dir="/tmp/bids", work_dir="/tmp/work")
    reporter.add_dmri_step("Denoising", details={"method": "MP-PCA"})
    reporter.save = lambda: reporter._save_data() # Add save shim if not public
    reporter._save_data()
    
    print(f"Data saved to {reporter.data_file}")
    
    # 2. Reload and update
    reporter2 = ReportGenerator(output_dir, title="Verification Report")
    # Verify data reloaded
    if reporter2.data["header"]["subject"] == "sub-test":
        print("Data reload: SUCCESS")
    else:
        print("Data reload: FAILED")
        
    reporter2.add_dmri_step("Coregistration", details={"method": "ANTs Rigid"})
    reporter2._save_data()
    
    # 3. Generate PDF
    if weasyprint:
        print("Attempting PDF generation...")
        reporter2.generate_pdf("report.pdf")
        pdf_path = output_dir / "report.pdf"
        if pdf_path.exists():
            print(f"PDF generation: SUCCESS ({pdf_path})")
        else:
            print("PDF generation: FAILED (file not created)")
    else:
        print("Skipping PDF generation (WeasyPrint unavailable)")

if __name__ == "__main__":
    test_report_cycle()
