import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging
import os

class NeuroimagingTracker:
    """
    Manages a multi-sheet Excel file for tracking neuroimaging data,
    processing status, and quality metrics.
    """
    
    CORE_SHEETS = [
        'Subject_Metadata', 'Processing_Status', 'Quality_Metrics', 
        'Data_Files', 'Processing_Times', 'Errors_Notes', 
        'Software_Versions', 'Alert_History'
    ]

    @staticmethod
    def create_empty_tracker(path: Path):
        """Create a new Excel file with the standard tracking structure."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Define core structures
        sheets = {
            'Subject_Metadata': ['Subject_ID', 'Session', 'Study', 'Age', 'Sex', 'Group', 'Scan_Date'],
            'Processing_Status': ['Subject_ID', 'Session', 'Study', 'Overall_Pipeline_Status', 'Last_Processing_Date'],
            'Quality_Metrics': ['Subject_ID', 'Session', 'Study', 'Motion_FD_Mean', 'DWI_SNR'],
            'Data_Files': ['Subject_ID', 'Session', 'Study', 'T1w_Present', 'DWI_Present'],
            'Processing_Times': ['Subject_ID', 'Session', 'Study', 'Total_Pipeline_Time_Min'],
            'Errors_Notes': ['Subject_ID', 'Session', 'Study', 'Has_Processing_Errors', 'Error_Message'],
            'Software_Versions': ['Subject_ID', 'Session', 'Study', 'Pipeline_Version'],
            'Alert_History': ['Alert_ID', 'Subject_ID', 'Study', 'Alert_Type', 'Alert_Status']
        }
        
        with pd.ExcelWriter(path, engine='openpyxl') as writer:
            for name, cols in sheets.items():
                pd.DataFrame(columns=cols).to_excel(writer, sheet_name=name, index=False)
                
            # Add README
            readme = pd.DataFrame({
                'Sheet': list(sheets.keys()),
                'Description': ['Subject info', 'Process status', 'QC stats', 'File paths', 'Timings', 'Errors', 'Versions', 'Alerts']
            })
            readme.to_excel(writer, sheet_name='README', index=False)
        
        return NeuroimagingTracker(path)

    def __init__(self, excel_path: Path, logger: Optional[logging.Logger] = None, auto_save: bool = True):
        self.excel_path = Path(excel_path)
        self.logger = logger or logging.getLogger("Tracker")
        self.auto_save = auto_save
        self._data: Dict[str, pd.DataFrame] = {}
        
        if self.excel_path.exists():
            self.load()

    def load(self):
        """Load all sheets from the Excel file."""
        if not self.excel_path.exists():
            return
            
        if self.excel_path.stat().st_size == 0:
            self.logger.warning(f"Tracker file {self.excel_path} is empty. Skipping load.")
            return

        try:
            # Use openpyxl engine explicitly for better multi-sheet support
            with pd.ExcelFile(self.excel_path, engine='openpyxl') as xls:
                for sheet_name in xls.sheet_names:
                    self._data[sheet_name] = pd.read_excel(xls, sheet_name=sheet_name)
            self.logger.info(f"Loaded tracker from {self.excel_path}")
        except Exception as e:
            self.logger.error(f"Failed to load tracker: {e}")
            raise

    def save(self, force: bool = False):
        """Save all data back to the multi-sheet Excel file."""
        if not self._data:
            return
            
        if not self.auto_save and not force:
            return

        try:
            # Create directory if it doesn't exist
            self.excel_path.parent.mkdir(parents=True, exist_ok=True)
            
            with pd.ExcelWriter(self.excel_path, engine='openpyxl') as writer:
                for sheet_name, df in self._data.items():
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
            self.logger.debug(f"Saved tracker to {self.excel_path}")
        except Exception as e:
            self.logger.error(f"Failed to save tracker: {e}")
            raise

    def _ensure_row(self, sheet_name: str, subject_id: str, session: str, study: Optional[str] = None) -> int:
        """Ensure a row exists for the subject/session/study and return its index."""
        if sheet_name not in self._data:
            # Create empty df with core columns if it doesn't exist
            cols = ['Subject_ID', 'Session']
            if study: cols.append('Study')
            self._data[sheet_name] = pd.DataFrame(columns=cols)
            
        df = self._data[sheet_name]
        
        # Check if row exists
        mask = (df['Subject_ID'] == subject_id) & (df['Session'] == session)
        if study and 'Study' in df.columns:
            mask = mask & (df['Study'] == study)
            
        matches = df.index[mask]
        
        if len(matches) > 0:
            return matches[0]
        else:
            # Create new row
            new_row = {'Subject_ID': subject_id, 'Session': session}
            if study: new_row['Study'] = study
            
            # Use concat instead of deprecated append
            self._data[sheet_name] = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            return len(self._data[sheet_name]) - 1

    def update_status(self, subject_id: str, session: str, module: str, status: str, study: Optional[str] = None):
        """Update the status of a specific processing module."""
        idx = self._ensure_row('Processing_Status', subject_id, session, study)
        col_name = f"{module}_Status"
        self._data['Processing_Status'].at[idx, col_name] = status
        self._data['Processing_Status'].at[idx, 'Last_Processing_Date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    def add_metrics(self, subject_id: str, session: str, metrics: Dict[str, Any], study: Optional[str] = None):
        """Add QC or biological metrics to the tracker."""
        idx = self._ensure_row('Quality_Metrics', subject_id, session, study)
        for key, val in metrics.items():
            self._data['Quality_Metrics'].at[idx, key] = val

    def add_roi_stats(self, subject_id: str, session: str, tsv_path: Path, sheet_name: str, study: Optional[str] = None):
        """Parse an ROI stats TSV and append/update the corresponding sheet."""
        if not tsv_path.exists():
            return
            
        new_stats = pd.read_csv(tsv_path, sep='\t')
        
        # We handle ROI data differently: usually long-format or specific columns
        # For simplicity, let's assume we want to store Mean values for each LabelName
        # as columns like ROI_[LabelName]_Mean
        
        idx = self._ensure_row(sheet_name, subject_id, session, study)
        df = self._data[sheet_name]
        
        for _, row in new_stats.iterrows():
            label = row['LabelName']
            mean_col = f"ROI_{label}_Mean"
            std_col = f"ROI_{label}_Std"
            
            # Ensure columns exist
            if mean_col not in df.columns: df[mean_col] = None
            if std_col not in df.columns: df[std_col] = None
            
            df.at[idx, mean_col] = row['Mean']
            df.at[idx, std_col] = row['Std']
            
        self._data[sheet_name] = df

    def update_metadata(self, subject_id: str, session: str, metadata: Dict[str, Any], study: Optional[str] = None):
        """Update subject demographic or scan metadata."""
        idx = self._ensure_row('Subject_Metadata', subject_id, session, study)
        for key, val in metadata.items():
            self._data['Subject_Metadata'].at[idx, key] = val
            
    def log_error(self, subject_id: str, session: str, module: str, error_msg: str, study: Optional[str] = None):
        """Log a processing error."""
        idx = self._ensure_row('Errors_Notes', subject_id, session, study)
        self._data['Errors_Notes'].at[idx, 'Has_Processing_Errors'] = True
        self._data['Errors_Notes'].at[idx, 'Error_Module'] = module
        self._data['Errors_Notes'].at[idx, 'Error_Message'] = error_msg
        self._data['Errors_Notes'].at[idx, 'Error_Timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    def log_time(self, subject_id: str, session: str, module: str, time_min: float, study: Optional[str] = None):
        """Log processing time for a module."""
        idx = self._ensure_row('Processing_Times', subject_id, session, study)
        col_name = f"{module}_Time_Min"
        self._data['Processing_Times'].at[idx, col_name] = time_min
