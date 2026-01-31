import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging
import os
import time
import shutil

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

        # Acquire lock for reading too, just in case someone is writing
        lock_path = self.excel_path.with_suffix(self.excel_path.suffix + ".lock")
        try:
            self._acquire_lock(lock_path)
            # Use openpyxl engine explicitly for better multi-sheet support
            with pd.ExcelFile(self.excel_path, engine='openpyxl') as xls:
                for sheet_name in xls.sheet_names:
                    self._data[sheet_name] = pd.read_excel(xls, sheet_name=sheet_name)
            self.logger.info(f"Loaded tracker from {self.excel_path}")
        except Exception as e:
            self.logger.error(f"Failed to load tracker: {e}")
            raise
        finally:
            self._release_lock(lock_path)

    def _acquire_lock(self, lock_path: Path, timeout: int = 60):
        """
        Robust file-based lock for multi-process safety.
        Writes current PID to the lock file and detects stale locks.
        """
        start_time = time.time()
        pid = os.getpid()
        
        while True:
            try:
                # Try to create the lock file
                fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                with os.fdopen(fd, 'w') as f:
                    f.write(str(pid))
                return True
            except FileExistsError:
                # Check for stale lock
                try:
                    with open(lock_path, 'r') as f:
                        lock_pid = int(f.read().strip())
                    
                    # Check if process is still running
                    if not self._pid_exists(lock_pid):
                        self.logger.warning(f"Detected stale lock from PID {lock_pid}. Removing it.")
                        self._release_lock(lock_path)
                        continue # Try again immediately
                except (ValueError, OSError, PermissionError):
                    # If we can't read it, assume it's being written or locked by someone else
                    pass

                if time.time() - start_time > timeout:
                    self.logger.warning(f"Timeout waiting for tracker lock on {lock_path}")
                    return False
                time.sleep(0.5)

    def _pid_exists(self, pid: int) -> bool:
        """Check if a process ID exists."""
        if pid <= 0: return False
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True # Exists but no permission
        else:
            return True

    def _release_lock(self, lock_path: Path):
        """Release the file-based lock."""
        if lock_path.exists():
            try:
                # Optional: check if we own the lock before removing?
                # For now, simple remove is fine as we only release in finally blocks or on stale detection
                os.remove(lock_path)
            except OSError:
                pass

    def save(self, force: bool = False):
        """
        Save all data back to the multi-sheet Excel file.
        Uses atomic replace and lightweight backups to prevent corruption.
        """
        if not self._data:
            return
            
        if not self.auto_save and not force:
            return

        lock_path = self.excel_path.with_suffix(self.excel_path.suffix + ".lock")
        temp_path = self.excel_path.with_suffix(self.excel_path.suffix + ".tmp")
        bak_path = self.excel_path.with_suffix(self.excel_path.suffix + ".bak")

        try:
            # Create directory if it doesn't exist
            self.excel_path.parent.mkdir(parents=True, exist_ok=True)
            
            if self._acquire_lock(lock_path):
                # Before saving, re-load to catch changes from other processes!
                if self.excel_path.exists() and self.excel_path.stat().st_size > 0:
                    try:
                        # CREATE BACKUP BEFORE MODIFYING
                        shutil.copy2(self.excel_path, bak_path)

                        with pd.ExcelFile(self.excel_path, engine='openpyxl') as xls:
                            for sheet_name in xls.sheet_names:
                                try:
                                    existing_df = pd.read_excel(xls, sheet_name=sheet_name)
                                    if sheet_name in self._data:
                                        # UPSERT: Merge existing with current
                                        if 'Subject_ID' in existing_df.columns and 'Session' in existing_df.columns:
                                            subset = ['Subject_ID', 'Session']
                                            if 'Study' in existing_df.columns and 'Study' in self._data[sheet_name].columns:
                                                subset.append('Study')
                                            
                                            if 'Alert_ID' in existing_df.columns:
                                                subset = ['Alert_ID']

                                            merged = pd.concat([existing_df, self._data[sheet_name]], ignore_index=True)
                                            self._data[sheet_name] = merged.drop_duplicates(subset=subset, keep='last')
                                        elif sheet_name == 'README':
                                            pass
                                        else:
                                            self._data[sheet_name] = existing_df
                                except Exception as e_sheet:
                                    self.logger.warning(f"Error merging sheet {sheet_name}: {e_sheet}")
                    except Exception as e:
                        self.logger.error(f"FATAL: Tracker corruption detected in {self.excel_path}. Aborting save to prevent data loss. Original error: {e}")
                        self.logger.info(f"Please restore from backup: {bak_path}")
                        return

                # ATOMIC SAVE via temp file
                try:
                    with pd.ExcelWriter(temp_path, engine='openpyxl') as writer:
                        for sheet_name, df in self._data.items():
                            df.to_excel(writer, sheet_name=sheet_name, index=False)
                    
                    # Final atomic swap
                    os.replace(temp_path, self.excel_path)
                    self.logger.debug(f"Atomically saved tracker to {self.excel_path}")
                except Exception as e_save:
                    self.logger.error(f"Failed to write temporary tracker file: {e_save}")
                    if temp_path.exists(): os.remove(temp_path)
                    raise
            else:
                self.logger.error(f"Could not acquire lock for saving tracker: {self.excel_path}")
        except Exception as e:
            self.logger.error(f"Failed to save tracker: {e}")
            raise
        finally:
            self._release_lock(lock_path)
            if temp_path.exists():
                try: os.remove(temp_path)
                except: pass

    def _ensure_row(self, sheet_name: str, subject_id: str, session: str, study: Optional[str] = None, extra_keys: Optional[Dict[str, Any]] = None) -> int:
        """
        Ensure a row exists for the subject/session/study and optional extra keys.
        Returns the index of the row.
        """
        if sheet_name not in self._data:
            # Create empty df with core columns
            cols = ['Subject_ID', 'Session']
            if study: cols.append('Study')
            if extra_keys:
                for k in extra_keys.keys():
                    if k not in cols: cols.append(k)
            self._data[sheet_name] = pd.DataFrame(columns=cols)
            
        df = self._data[sheet_name]
        
        # Build mask for matching
        mask = (df['Subject_ID'] == subject_id) & (df['Session'] == session)
        if study and 'Study' in df.columns:
            mask = mask & (df['Study'] == study)
            
        if extra_keys:
            for k, v in extra_keys.items():
                if k in df.columns:
                    mask = mask & (df[k] == v)
                else:
                    # Column doesn't exist, so row definitely doesn't exist
                    mask = pd.Series([False] * len(df))
                    break
            
        matches = df.index[mask]
        
        if len(matches) > 0:
            return matches[0]
        else:
            # Create new row
            new_row = {'Subject_ID': subject_id, 'Session': session}
            if study: new_row['Study'] = study
            if extra_keys:
                new_row.update(extra_keys)
                
            # Ensure any new columns from extra_keys are added to the DF
            new_cols = [c for c in new_row.keys() if c not in df.columns]
            if new_cols:
                df = df.reindex(columns=list(df.columns) + new_cols)
                for c in new_cols: df[c] = df[c].astype(object)
            
            self._data[sheet_name] = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            return len(self._data[sheet_name]) - 1

    def update_status(self, subject_id: str, session: str, module: str, status: str, study: Optional[str] = None):
        """Update the status of a specific processing module."""
        idx = self._ensure_row('Processing_Status', subject_id, session, study)
        df = self._data['Processing_Status']
        
        col = f"{module}_Status"
        if col not in df.columns:
            # Ensure it's object dtype from the start to avoid float64 FutureWarning
            df = df.reindex(columns=list(df.columns) + [col])
            df[col] = df[col].astype(object)
            
        df.at[idx, col] = status
        df.at[idx, 'Last_Processing_Date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self._data['Processing_Status'] = df.copy()

    def add_metrics(self, subject_id: str, session: str, metrics: Dict[str, Any], study: Optional[str] = None):
        """Add QC or biological metrics to the tracker."""
        idx = self._ensure_row('Quality_Metrics', subject_id, session, study)
        df = self._data['Quality_Metrics']
        
        # Batch add new columns to avoid fragmentation warnings
        new_cols = [c for c in metrics.keys() if c not in df.columns]
        if new_cols:
            df = df.reindex(columns=list(df.columns) + new_cols)
            # Ensure new columns can handle various dtypes
            for c in new_cols: df[c] = df[c].astype(object)
            
        for key, val in metrics.items():
            df.at[idx, key] = val
        
        # De-fragment
        self._data['Quality_Metrics'] = df.copy()

    def add_roi_stats(self, subject_id: str, session: str, tsv_path: Path, atlas_name: str, study: Optional[str] = None):
        """
        Parse an ROI stats TSV and update the tracker in a tidy "Metrics-as-Columns" format.
        Rows: Subject, Session, Atlas, ROI, Statistic
        Columns: Metrics (FA, MD, T1, T2, etc.)
        """
        if not tsv_path.exists():
            return
            
        new_stats = pd.read_csv(tsv_path, sep='\t')
        
        # Ensure we have a 'model' column. If not, fallback to atlas_name
        if 'model' not in new_stats.columns:
            # Try to infer model from 'metric' if it has underscore (e.g. DTI_fa)
            if 'metric' in new_stats.columns:
                def _infer_model(m):
                    if '_' in str(m): return str(m).rsplit('_', 1)[0]
                    return 'Other'
                new_stats['model'] = new_stats['metric'].apply(_infer_model)
                new_stats['metric'] = new_stats['metric'].apply(lambda x: str(x).rsplit('_', 1)[-1] if '_' in str(x) else x)
            else:
                new_stats['model'] = 'ROI_Stats'
        
        # We handle 'Mean', 'Median', 'Std' as separate rows for each (Model, Atlas, ROI)
        for stat_type in ['Mean', 'Median', 'Std']:
            stat_col = stat_type.lower()
            if stat_col not in new_stats.columns:
                # Try finding exact case if stat_col missing
                if stat_type in new_stats.columns:
                    stat_col = stat_type
                else:
                    continue

            # Group by Model to handle separate sheets
            for model, model_df in new_stats.groupby('model'):
                sheet_name = f"{model}_Metrics"
                
                # Group by ROI to update row-by-row
                for roi_name, roi_df in model_df.groupby('roi_name'):
                    keys = {
                        'Atlas': atlas_name,
                        'ROI_Name': roi_name,
                        'Statistic': stat_type
                    }
                    
                    idx = self._ensure_row(sheet_name, subject_id, session, study, extra_keys=keys)
                    df = self._data[sheet_name]
                    
                    # Updates: each metric for this ROI/Stat
                    updates = {}
                    for _, row in roi_df.iterrows():
                        metric = row['metric']
                        val = row[stat_col]
                        updates[metric] = val
                    
                    # Batch add metric columns
                    new_cols = [c for c in updates.keys() if c not in df.columns]
                    if new_cols:
                        df = df.reindex(columns=list(df.columns) + new_cols)
                        for c in new_cols: df[c] = df[c].astype(object)
                    
                    for metric, val in updates.items():
                        df.at[idx, metric] = val
                        
                    self._data[sheet_name] = df.copy()

    def update_metadata(self, subject_id: str, session: str, metadata: Dict[str, Any], study: Optional[str] = None):
        """Update subject demographic or scan metadata."""
        idx = self._ensure_row('Subject_Metadata', subject_id, session, study)
        df = self._data['Subject_Metadata']
        
        new_cols = [c for c in metadata.keys() if c not in df.columns]
        if new_cols:
            df = df.reindex(columns=list(df.columns) + new_cols)
            for c in new_cols: df[c] = df[c].astype(object)
            
        for key, val in metadata.items():
            df.at[idx, key] = val
            
        self._data['Subject_Metadata'] = df.copy()
            
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
