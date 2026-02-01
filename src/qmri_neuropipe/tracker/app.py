import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys
import os

# Add the project root to sys.path to allow imports from qmri_neuropipe
# (Assuming the app is launched via a wrapper that sets this up, or we handle it here)
# For now, let's assume qmri_neuropipe is installable or reachable.
try:
    from qmri_neuropipe.lib.common.tracker import NeuroimagingTracker
except ImportError:
    # Fallback/Debug path - adjust as needed
    sys.path.append(str(Path(__file__).parents[2]))
    from qmri_neuropipe.lib.common.tracker import NeuroimagingTracker

def color_status(val):
    if not isinstance(val, str): return ""
    val_l = val.lower()
    if "complete" in val_l: return "background-color: #C6EFCE; color: #006100; font-weight: bold;"
    if "progress" in val_l: return "background-color: #BEE5EB; color: #0C5460; font-weight: bold;"
    if "pending" in val_l or "queued" in val_l: return "background-color: #E2E3E5; color: #383d41;"
    if "warning" in val_l: return "background-color: #FFEB9C; color: #9C5700; font-weight: bold;"
    if "error" in val_l or "failed" in val_l: return "background-color: #FFC7CE; color: #9C0006; font-weight: bold;"
    return ""

def style_status_df(df):
    exclude = ['Subject_ID', 'Session', 'Study', 'Last_Update', 'Last_Processing_Date', 'Segmentation_Method', 'B1_Mapping_Method', 'Atlases', 'Model_Fits']
    status_cols = [c for c in df.columns if c not in exclude]
    if not status_cols: return df
    try:
        return df.style.map(color_status, subset=status_cols)
    except:
        # Fallback for older pandas
        return df.style.applymap(color_status, subset=status_cols)

st.set_page_config(page_title="qMRI Tracker Dashboard", page_icon="🧠", layout="wide")

st.title("🧠 qMRI Neuroimaging Tracker")

# Sidebar for file selection
st.sidebar.header("Data Source")
tracker_file = st.sidebar.file_uploader("Upload Tracker Excel", type=["xlsx"])

# Determine final path
final_tracker_path = None

if tracker_file:
    # Save uploaded file to a temporary location to use with NeuroimagingTracker
    final_tracker_path = Path("temp_tracker.xlsx")
    with open(final_tracker_path, "wb") as f:
        f.write(tracker_file.getbuffer())
else:
    # Check for environment variable (from CLI)
    env_path = os.environ.get("TRACKER_PATH")
    if env_path:
        p = Path(env_path)
        if p.exists():
            final_tracker_path = p
        else:
            st.sidebar.error(f"Environment tracker not found: {env_path}")

if final_tracker_path:
    try:
        tracker = NeuroimagingTracker(final_tracker_path)
        data = tracker._data
        
        if not data:
            st.warning(f"Tracker file is empty: {final_tracker_path}")
            st.info("Please initialize it using: `qmri-tools tracker-init --output path/to/tracker.xlsx`")
            st.stop()
    except Exception as e:
        st.error(f"Failed to load tracker: {e}")
        st.info("This usually happens if the file is not a valid Excel file or is empty.")
        st.info("Try initializing a new tracker: `qmri-tools tracker-init --output path/to/tracker.xlsx`")
        st.stop()
    
    # Study selector
    studies = ["All"]
    if "Subject_Metadata" in data and "Study" in data["Subject_Metadata"].columns:
        studies.extend(data["Subject_Metadata"]["Study"].unique().tolist())
    
    selected_study = st.sidebar.selectbox("Filter by Study", studies)
    
    # Tabs for different views
    tab_summary, tab_overview, tab_subject, tab_distribution, tab_correlation, tab_raw = st.tabs([
        "📊 Summary", "⚙️ Processing Status", "👤 Subject Details", "📈 Distributions", "🔗 Correlations", "📋 Raw Data"
    ])

    with tab_summary:
        st.header("Executive Summary")
        if "Summary" in data:
            df_sum = data["Summary"]
            col1, col2 = st.columns([1, 2])
            with col1:
                st.dataframe(df_sum, width='stretch', hide_index=True)
            with col2:
                # Add a pie chart of overall completion if data allows
                if "Processing_Status" in data:
                    df_ps = data["Processing_Status"]
                    if "Overall_Pipeline_Status" in df_ps.columns:
                        counts = df_ps["Overall_Pipeline_Status"].value_counts().reset_index()
                        counts.columns = ["Status", "Count"]
                        
                        # Apply custom colors
                        color_map = {
                            "Complete": "#C6EFCE",
                            "In Progress": "#BEE5EB",
                            "Failed": "#FFC7CE",
                            "Pending": "#E2E3E5",
                            "Error": "#FFC7CE",
                            "Warning": "#FFEB9C"
                        }
                        
                        fig = px.pie(counts, values="Count", names="Status", title="Overall Pipeline Status Distribution",
                                     color="Status", color_discrete_map=color_map)
                        st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No summary data found. Re-save your tracker to generate a summary.")
    
    with tab_overview:
        st.header("Modality Processing Status")
        status_sheets = ["Processing_Status", "Anatomical_Status", "Diffusion_Status", "Relaxometry_Status"]
        available_status = [s for s in status_sheets if s in data]
        
        if available_status:
            sel_status_sheet = st.selectbox("Select Status View", available_status)
            df_status = data[sel_status_sheet]
            if selected_study != "All" and "Study" in df_status.columns:
                df_status = df_status[df_status["Study"] == selected_study]
            
            # Module Status heatmap/table
            st.dataframe(style_status_df(df_status), use_container_width=True)
        else:
            st.warning("No status sheets found in tracker.")

    with tab_subject:
        st.header("Individual Subject Details")
        
        # Subject Selection (no session dropdown - we show all sessions)
        subjects = []
        if "Processing_Status" in data:
             subjects = sorted(data["Processing_Status"]["Subject_ID"].unique().tolist())
        else:
             first_df = next(iter(data.values()))
             if "Subject_ID" in first_df.columns:
                  subjects = sorted(first_df["Subject_ID"].unique().tolist())
        
        if subjects:
             selected_subj = st.selectbox("Select Subject", subjects, key="subj_detail_selector")
             
             # Get all sessions for this subject
             sessions = []
             if "Processing_Status" in data:
                  sessions = data["Processing_Status"][data["Processing_Status"]["Subject_ID"] == selected_subj]["Session"].unique().tolist()
             sessions = [str(s) if pd.notna(s) else "N/A" for s in sessions]
             
             st.info(f"Showing data for **{len(sessions)}** session(s): {', '.join(sessions) if sessions else 'N/A'}")
             
             # Processing Status - All Sessions Table
             st.subheader("📋 Processing Status (All Sessions)")
             if "Processing_Status" in data:
                  df_status = data["Processing_Status"][data["Processing_Status"]["Subject_ID"] == selected_subj].copy()
                  if not df_status.empty:
                       # Format session column
                       df_status["Session"] = df_status["Session"].apply(lambda x: str(x) if pd.notna(x) else "N/A")
                       st.dataframe(style_status_df(df_status), use_container_width=True, hide_index=True)
                  else:
                       st.info("No processing status found for this subject.")
             
             # Modality-Specific Status Tables
             modality_sheets = [s for s in data.keys() if s.endswith("_Status") and s != "Processing_Status"]
             if modality_sheets:
                  st.subheader("🔬 Modality-Specific Status (All Sessions)")
                  tabs_modality = st.tabs([s.replace("_Status", "") for s in modality_sheets])
                  for i, sheet_name in enumerate(modality_sheets):
                       with tabs_modality[i]:
                            df_mod = data[sheet_name][data[sheet_name]["Subject_ID"] == selected_subj].copy()
                            if not df_mod.empty:
                                 df_mod["Session"] = df_mod["Session"].apply(lambda x: str(x) if pd.notna(x) else "N/A")
                                 st.dataframe(style_status_df(df_mod), use_container_width=True, hide_index=True)
                            else:
                                 st.info(f"No {sheet_name.replace('_Status', '')} data for this subject.")
             
             st.markdown("---")
             col_meta, col_qc = st.columns(2)
             
             # Metadata - All Sessions Table
             with col_meta:
                  st.subheader("📝 Metadata (All Sessions)")
                  if "Subject_Metadata" in data:
                       df_meta = data["Subject_Metadata"][data["Subject_Metadata"]["Subject_ID"] == selected_subj].copy()
                       if not df_meta.empty:
                            df_meta["Session"] = df_meta["Session"].apply(lambda x: str(x) if pd.notna(x) else "N/A")
                            # Drop columns that are all NaN
                            df_meta = df_meta.dropna(axis=1, how='all')
                            st.dataframe(df_meta, use_container_width=True, hide_index=True)
                       else:
                            st.info("No metadata found for this subject.")
             
             # Quality Metrics - All Sessions Table
             with col_qc:
                  st.subheader("📊 Quality Metrics (All Sessions)")
                  if "Quality_Metrics" in data:
                       df_qc = data["Quality_Metrics"][data["Quality_Metrics"]["Subject_ID"] == selected_subj].copy()
                       if not df_qc.empty:
                            df_qc["Session"] = df_qc["Session"].apply(lambda x: str(x) if pd.notna(x) else "N/A")
                            df_qc = df_qc.dropna(axis=1, how='all')
                            st.dataframe(df_qc, use_container_width=True, hide_index=True)
                       else:
                            st.info("No QC metrics found for this subject.")
             
             st.markdown("---")
             st.subheader("🧠 ROI Statistics (All Sessions)")
             # Find all sheets with "Metrics" in name but not "Quality_Metrics"
             roi_sheets = [s for s in data.keys() if "Metrics" in s and s != "Quality_Metrics"]
             if roi_sheets:
                  sel_roi_sheet = st.selectbox("View ROI Stats from Sheet", roi_sheets, key="roi_sheet_selector")
                  df_roi = data[sel_roi_sheet]
                  df_roi_sub = df_roi[df_roi["Subject_ID"] == selected_subj].copy()
                  
                  if not df_roi_sub.empty:
                       df_roi_sub["Session"] = df_roi_sub["Session"].apply(lambda x: str(x) if pd.notna(x) else "N/A")
                       
                       # If tidy format, show pivot option
                       if "Metric" in df_roi_sub.columns and "Statistic" in df_roi_sub.columns:
                            view_mode = st.radio("View Mode", ["Pivoted (Wide)", "Long (Tidy)"], horizontal=True)
                            if view_mode == "Pivoted (Wide)":
                                 try:
                                      pivot_df = df_roi_sub.pivot_table(
                                           index=["Session", "Atlas", "ROI_Name"], 
                                           columns=["Metric", "Statistic"], 
                                           values="Value",
                                           aggfunc='first'
                                      )
                                      st.dataframe(pivot_df, use_container_width=True)
                                 except Exception as e:
                                      st.warning(f"Could not pivot: {e}")
                                      st.dataframe(df_roi_sub, use_container_width=True, hide_index=True)
                            else:
                                 st.dataframe(df_roi_sub, use_container_width=True, hide_index=True)
                       else:
                            st.dataframe(df_roi_sub, use_container_width=True, hide_index=True)
                  else:
                       st.info("No ROI stats found for this subject in selected sheet.")
             else:
                  st.info("No ROI metric sheets found in tracker.")
        else:
             st.warning("No subjects found in tracker.")


    with tab_distribution:
        st.header("Metric Distributions")
        
        # Select sheet and column
        sheet_names = list(data.keys())
        selected_sheet = st.selectbox("Select Data Sheet", sheet_names)
        
        df = data[selected_sheet]
        if selected_study != "All" and "Study" in df.columns:
            df = df[df["Study"] == selected_study]
            
        # Tidy Sheet Handling
        sel_roi = None
        is_tidy = "Metric" in df.columns and "Statistic" in df.columns
        if is_tidy:
            col1, col2, col3 = st.columns(3)
            if "Model" in df.columns:
                models = ["All"] + df["Model"].unique().tolist()
                sel_model = col1.selectbox("Filter by Model", models)
                if sel_model != "All":
                    df = df[df["Model"] == sel_model]
            
            metrics = df["Metric"].unique().tolist()
            sel_metric = col2.selectbox("Select Metric", metrics)
            df = df[df["Metric"] == sel_metric]
            
            stats = df["Statistic"].unique().tolist()
            sel_stat = col3.selectbox("Select Statistic", stats)
            df = df[df["Statistic"] == sel_stat]

            # Atlas and ROI Filters
            cola, colr = st.columns(2)
            if "Atlas" in df.columns:
                 atlases = ["All"] + df["Atlas"].unique().tolist()
                 sel_atlas = cola.selectbox("Filter by Atlas", atlases)
                 if sel_atlas != "All":
                      df = df[df["Atlas"] == sel_atlas]
            
            if "ROI_Name" in df.columns:
                 rois = sorted(df["ROI_Name"].unique().tolist())
                 sel_roi = colr.multiselect("Filter by ROI (Leave empty for All)", rois)
                 if sel_roi:
                      df = df[df["ROI_Name"].isin(sel_roi)]
            else:
                 sel_roi = None

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if numeric_cols:
            default_metric = "Value" if "Value" in numeric_cols else numeric_cols[0]
            selected_metric = st.selectbox("Select Metric to Visualize", numeric_cols, index=numeric_cols.index(default_metric))
            
            # Grouping Option
            group_options = ["None"] + [c for c in df.columns if df[c].dtype == object and c not in [selected_metric]]
            if sel_roi and isinstance(sel_roi, list) and len(sel_roi) > 1:
                 # Default to ROI if multiple selected
                 sel_group = st.selectbox("Group By", group_options, index=group_options.index("ROI_Name") if "ROI_Name" in group_options else 0)
            else:
                 sel_group = st.selectbox("Group By", group_options)

            col1, col2 = st.columns(2)
            
            with col1:
                fig_hist = px.histogram(df, x=selected_metric, nbins=20, 
                                       color=None if sel_group == "None" else sel_group,
                                       title=f"Distribution of {selected_metric}",
                                       marginal="box", barmode="overlay")
                st.plotly_chart(fig_hist, use_container_width=True)
                
            with col2:
                fig_box = px.box(df, y=selected_metric, 
                                x=None if sel_group == "None" else sel_group,
                                color=None if sel_group == "None" else sel_group,
                                title=f"Boxplot of {selected_metric}",
                                points="all")
                st.plotly_chart(fig_box, use_container_width=True)
        else:
            st.warning("No numeric metrics found in this sheet.")

    with tab_correlation:
        st.header("Metric Correlations")
        
        # Merge data from multiple sheets? 
        # For simplicity, let's correlate within a sheet or join Metadata (Age/Sex) with others
        
        selected_sheet_corr = st.selectbox("Select Sheet for Correlation", sheet_names, key="corr_sheet")
        df_corr = data[selected_sheet_corr]
        
        # Allow merging with Metadata if desired
        if "Subject_Metadata" in data and selected_sheet_corr != "Subject_Metadata":
            if st.checkbox("Merge with Subject Metadata (Age, Sex, etc.)"):
                df_meta = data["Subject_Metadata"]
                df_corr = df_corr.merge(df_meta, on=["Subject_ID", "Session"], suffixes=('', '_meta'))

        if selected_study != "All" and "Study" in df_corr.columns:
            df_corr = df_corr[df_corr["Study"] == selected_study]

        # Tidy Sheet Handling (Correlation Tab)
        is_tidy_corr = "Metric" in df_corr.columns and "Statistic" in df_corr.columns
        if is_tidy_corr:
            st.info("Tidy ROI Sheet detected. Filter for a specific Metric/Statistic combination.")
            col1, col2, col3 = st.columns(3)
            if "Model" in df_corr.columns:
                models = ["All"] + df_corr["Model"].unique().tolist()
                sel_model = col1.selectbox("Filter Model (Corr)", models)
                if sel_model != "All":
                    df_corr = df_corr[df_corr["Model"] == sel_model]
            
            metrics = df_corr["Metric"].unique().tolist()
            sel_metric = col2.selectbox("Select Metric (Corr)", metrics)
            df_corr = df_corr[df_corr["Metric"] == sel_metric]
            
            stats = df_corr["Statistic"].unique().tolist()
            sel_stat = col3.selectbox("Select Statistic (Corr)", stats)
            df_corr = df_corr[df_corr["Statistic"] == sel_stat]

            # Atlas and ROI Filters (Corr)
            cola, colr = st.columns(2)
            if "Atlas" in df_corr.columns:
                 atlases = ["All"] + df_corr["Atlas"].unique().tolist()
                 sel_atlas = cola.selectbox("Filter Atlas (Corr)", atlases)
                 if sel_atlas != "All":
                      df_corr = df_corr[df_corr["Atlas"] == sel_atlas]
            
            if "ROI_Name" in df_corr.columns:
                 rois = sorted(df_corr["ROI_Name"].unique().tolist())
                 sel_roi = colr.multiselect("Filter ROI (Leave empty for All)", rois, key="roi_corr")
                 if sel_roi:
                      df_corr = df_corr[df_corr["ROI_Name"].isin(sel_roi)]

        numeric_cols_corr = df_corr.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols_corr) >= 2:
            col1, col2 = st.columns(2)
            metric_x = col1.selectbox("X Axis", numeric_cols_corr, index=0)
            metric_y = col2.selectbox("Y Axis", numeric_cols_corr, index=1 if len(numeric_cols_corr) > 1 else 0)
            
            color_by = st.selectbox("Color By (optional)", ["None"] + [c for c in df_corr.columns if df_corr[c].dtype == object])
            
            fig_scatter = px.scatter(df_corr, x=metric_x, y=metric_y, 
                                    color=None if color_by == "None" else color_by,
                                    hover_data=["Subject_ID", "Session"],
                                    title=f"{metric_x} vs {metric_y}",
                                    trendline="ols")
            st.plotly_chart(fig_scatter, use_container_width=True)
        else:
            st.warning("Need at least 2 numeric metrics for correlation.")

    with tab_raw:
        st.header("Raw Data Sheets")
        for sheet, df in data.items():
            with st.expander(f"Sheet: {sheet}"):
                st.dataframe(df, use_container_width=True)

else:
    st.info("Please upload a tracker Excel file from the sidebar to begin.")
    
    # Show an example if we have one in the repo?
    # Or just a welcoming message.
    st.markdown("""
    ### Features:
    - **Real-time Status Tracking**: Monitor your pipeline's progress.
    - **Quality Control**: Visualize motion and SNR across your entire study.
    - **ROI Analysis**: Compare DTI/NODDI metrics between groups or correlate with demographics.
    - **Study-wide distributions**: Identify outliers and data trends.
    """)
