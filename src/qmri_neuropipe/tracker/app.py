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
    tab_overview, tab_subject, tab_distribution, tab_correlation, tab_raw = st.tabs([
        "📊 Overview", "👤 Subject Details", "📈 Distributions", "🔗 Correlations", "📋 Raw Data"
    ])
    
    with tab_overview:
        st.header("Study Overview")
        if "Processing_Status" in data:
            df_status = data["Processing_Status"]
            if selected_study != "All":
                df_status = df_status[df_status["Study"] == selected_study]
            
            # Summary Metrics
            cols = st.columns(4)
            cols[0].metric("Total Subjects", len(df_status))
            
            if "Overall_Pipeline_Status" in df_status.columns:
                completed = len(df_status[df_status["Overall_Pipeline_Status"] == "completed"])
                cols[1].metric("Completed", completed)
                failed = len(df_status[df_status["Overall_Pipeline_Status"] == "failed"])
                cols[2].metric("Failed", failed, delta=failed, delta_color="inverse")
            
            st.subheader("Module Status Heatmap")
            status_cols = [c for c in df_status.columns if c.endswith("_Status")]
            if status_cols:
                # Convert status to numeric for heatmap? or just use a table
                st.dataframe(df_status[["Subject_ID", "Session"] + status_cols], use_container_width=True)

    with tab_subject:
        st.header("Individual Subject Details")
        
        # Subject and Session Selection
        # Get list of subjects from Processing_Status if available, else first data sheet
        subjects = []
        if "Processing_Status" in data:
             subjects = sorted(data["Processing_Status"]["Subject_ID"].unique().tolist())
        else:
             # Fallback to first sheet
             first_df = next(iter(data.values()))
             if "Subject_ID" in first_df.columns:
                  subjects = sorted(first_df["Subject_ID"].unique().tolist())
        
        if subjects:
             selected_subj = st.selectbox("Select Subject", subjects, key="subj_detail_selector")
             
             # Filter sessions for this subject
             sessions = []
             if "Processing_Status" in data:
                  sessions = data["Processing_Status"][data["Processing_Status"]["Subject_ID"] == selected_subj]["Session"].unique().tolist()
             
             # Clean up sessions list (handle NaN)
             sessions = [str(s) if pd.notna(s) else "N/A" for s in sessions]
             if not sessions: sessions = ["N/A"]
             
             selected_ses = st.selectbox("Select Session", sessions)
             # Map back "N/A" to NaN if needed for filtering, but usually sessions are strings
             actual_ses = selected_ses if selected_ses != "N/A" else np.nan
             
             # Show Subject Profile
             col_status, col_meta, col_qc = st.columns(3)
             
             with col_status:
                  st.subheader("Processing Status")
                  if "Processing_Status" in data:
                       # Handle sessions correctly in filter
                       if pd.isna(actual_ses):
                            s_row = data["Processing_Status"][(data["Processing_Status"]["Subject_ID"] == selected_subj) & (data["Processing_Status"]["Session"].isna())]
                       else:
                            s_row = data["Processing_Status"][(data["Processing_Status"]["Subject_ID"] == selected_subj) & (data["Processing_Status"]["Session"] == actual_ses)]
                       
                       if not s_row.empty:
                            status_cols = [c for c in s_row.columns if c.endswith("_Status")]
                            for c in status_cols:
                                 val = s_row.iloc[0][c]
                                 color = "green" if val == "completed" else "red" if val == "failed" else "orange"
                                 st.markdown(f"**{c.replace('_Status', '')}**: :{color}[{val}]")
                       else:
                            st.info("No status information found.")

             with col_meta:
                  st.subheader("Metadata")
                  if "Subject_Metadata" in data:
                       if pd.isna(actual_ses):
                            m_row = data["Subject_Metadata"][(data["Subject_Metadata"]["Subject_ID"] == selected_subj) & (data["Subject_Metadata"]["Session"].isna())]
                       else:
                            m_row = data["Subject_Metadata"][(data["Subject_Metadata"]["Subject_ID"] == selected_subj) & (data["Subject_Metadata"]["Session"] == actual_ses)]
                       
                       if not m_row.empty:
                            m_data = m_row.iloc[0].dropna().to_dict()
                            for k, v in m_data.items():
                                 if k not in ["Subject_ID", "Session", "Study"]:
                                      st.text(f"{k}: {v}")
                       else:
                            st.info("No metadata found.")

             with col_qc:
                  st.subheader("Quality Metrics")
                  if "Quality_Metrics" in data:
                       if pd.isna(actual_ses):
                            q_row = data["Quality_Metrics"][(data["Quality_Metrics"]["Subject_ID"] == selected_subj) & (data["Quality_Metrics"]["Session"].isna())]
                       else:
                            q_row = data["Quality_Metrics"][(data["Quality_Metrics"]["Subject_ID"] == selected_subj) & (data["Quality_Metrics"]["Session"] == actual_ses)]
                       
                       if not q_row.empty:
                            q_data = q_row.iloc[0].dropna().to_dict()
                            # Highlight specific QC
                            important_qc = ["QC_DWI_b0_SNR", "QC_DWI_Outliers_Total_Pct", "QC_DWI_Outliers_Removed", "QC_DWI_Motion_Abs_mm"]
                            for k in important_qc:
                                 if k in q_data:
                                      st.metric(k.replace("QC_DWI_", ""), q_data[k])
                            
                            with st.expander("Show all QC"):
                                 for k, v in q_data.items():
                                      if k not in ["Subject_ID", "Session", "Study"]:
                                           st.text(f"{k}: {v}")
                       else:
                            st.info("No QC metrics found.")
             
             st.markdown("---")
             st.subheader("ROI Statistics")
             # Find all sheets with "Metrics" in name
             metric_sheets = [s for s in data.keys() if "Metrics" in s]
             if metric_sheets:
                  sel_metric_sheet = st.selectbox("View ROI Stats from Sheet", metric_sheets)
                  df_roi = data[sel_metric_sheet]
                  
                  if pd.isna(actual_ses):
                       df_roi_sub = df_roi[(df_roi["Subject_ID"] == selected_subj) & (df_roi["Session"].isna())]
                  else:
                       df_roi_sub = df_roi[(df_roi["Subject_ID"] == selected_subj) & (df_roi["Session"] == actual_ses)]
                  
                  if not df_roi_sub.empty:
                       # If tidy, it might be better to pivot for this view?
                       if "Metric" in df_roi_sub.columns and "Statistic" in df_roi_sub.columns:
                            st.info("Displaying pivoted view of ROI metrics.")
                            pivot_df = df_roi_sub.pivot(index=["Atlas", "ROI_Name"], columns=["Metric", "Statistic"], values="Value")
                            st.dataframe(pivot_df, use_container_width=True)
                       else:
                            st.dataframe(df_roi_sub, use_container_width=True)
                  else:
                       st.info("No ROI stats found for this subject/session in selected sheet.")
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
