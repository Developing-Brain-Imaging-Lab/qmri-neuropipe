import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys
import os

# Add the project root to sys.path to allow imports from qmri_neuropipe
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
    # Save uploaded file to a temporary location
    final_tracker_path = Path("temp_tracker.xlsx")
    with open(final_tracker_path, "wb") as f:
        f.write(tracker_file.getbuffer())
else:
    # Check for environment variable
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
    tab_summary, tab_study_details, tab_subject, tab_overview, tab_correlation, tab_raw = st.tabs([
        "📊 Summary", "📊 Study Details", "👤 Subject Details", "⚙️ Processing Status", "🔗 Correlations", "📋 Raw Data"
    ])

    with tab_summary:
        st.header("Executive Summary")
        
        # 1. High-Level Metric Cards
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        
        # Calculate live metrics
        total_subjs = 0
        total_sess = 0
        anat_comp_rate = "0%"
        diff_comp_rate = "0%"
        avg_snr = "N/A"
        avg_mot = "N/A"
        
        if "Processing_Status" in data:
            df_ps_all = data["Processing_Status"]
            if selected_study != "All" and "Study" in df_ps_all.columns:
                 df_ps_all = df_ps_all[df_ps_all["Study"] == selected_study]
            
            total_subjs = df_ps_all["Subject_ID"].nunique()
            total_sess = len(df_ps_all)
            
        if "Anatomical_Status" in data:
             df_anat = data["Anatomical_Status"]
             if selected_study != "All" and "Study" in df_anat.columns: df_anat = df_anat[df_anat["Study"] == selected_study]
             if "Overall_Status" in df_anat.columns:
                  comp = (df_anat["Overall_Status"].astype(str).str.contains("Complete", case=False)).sum()
                  anat_comp_rate = f"{(comp/total_sess)*100:.1f}%" if total_sess > 0 else "0%"
                  
        if "Diffusion_Status" in data:
             df_diff = data["Diffusion_Status"]
             if selected_study != "All" and "Study" in df_diff.columns: df_diff = df_diff[df_diff["Study"] == selected_study]
             if "Overall_Status" in df_diff.columns:
                  comp = (df_diff["Overall_Status"].astype(str).str.contains("Complete", case=False)).sum()
                  diff_comp_rate = f"{(comp/total_sess)*100:.1f}%" if total_sess > 0 else "0%"
        
        if "Quality_Metrics" in data:
             df_qm = data["Quality_Metrics"]
             if selected_study != "All" and "Study" in df_qm.columns:
                  df_qm = df_qm[df_qm["Study"] == selected_study]
             
             if "DWI_SNR" in df_qm.columns:
                  val = df_qm["DWI_SNR"].dropna().mean()
                  if pd.notna(val): avg_snr = f"{val:.1f}"
             if "Motion_FD_Mean" in df_qm.columns:
                  val = df_qm["Motion_FD_Mean"].dropna().mean()
                  if pd.notna(val): avg_mot = f"{val:.3f}"

        col_m1.metric("Total Subjects", total_subjs)
        col_m2.metric("Total Sessions", total_sess)
        col_m3.metric("Anatomical Success", anat_comp_rate)
        col_m4.metric("Diffusion Success", diff_comp_rate)
        
        # Second row for quality metrics
        q_col1, q_col2 = st.columns(2)
        q_col1.metric("Avg DWI SNR", avg_snr)
        q_col2.metric("Avg Mean FD", avg_mot)

        st.markdown("---")

        # 2. Main Content Row
        if "Summary" in data:
            col1, col2 = st.columns([1, 2])
            with col1:
                st.subheader("📋 Study Summary")
                df_sum = data["Summary"]
                st.dataframe(df_sum, width=None, use_container_width=True, hide_index=True)
                
                # Recently Processed
                if "Processing_Status" in data:
                     st.subheader("🕒 Recent Activity")
                     df_recent = data["Processing_Status"].copy()
                     if "Last_Processing_Date" in df_recent.columns:
                          df_recent = df_recent.sort_values("Last_Processing_Date", ascending=False).head(5)
                          st.dataframe(df_recent[["Subject_ID", "Session", "Overall_Pipeline_Status"]], use_container_width=True, hide_index=True)
                     else:
                          st.info("No recent activity timestamps found.")

            with col2:
                st.subheader("📊 Pipeline Status Distribution")
                
                # Gather all status data to show side-by-side or selectable
                status_plots = []
                
                # 1. Overall
                if "Processing_Status" in data:
                    df_ps = data["Processing_Status"]
                    if selected_study != "All" and "Study" in df_ps.columns:
                         df_ps = df_ps[df_ps["Study"] == selected_study]
                    if "Overall_Pipeline_Status" in df_ps.columns:
                         counts = df_ps["Overall_Pipeline_Status"].value_counts().reset_index()
                         counts.columns = ["Status", "Count"]
                         status_plots.append(("Overall", counts))
                
                # 2. Modalities
                for mod_name, sheet in [("Anatomical", "Anatomical_Status"), ("Diffusion", "Diffusion_Status"), ("Relaxometry", "Relaxometry_Status")]:
                     if sheet in data:
                          df_mod = data[sheet]
                          if selected_study != "All" and "Study" in df_mod.columns:
                               df_mod = df_mod[df_mod["Study"] == selected_study]
                          if "Overall_Status" in df_mod.columns:
                               counts = df_mod["Overall_Status"].value_counts().reset_index()
                               counts.columns = ["Status", "Count"]
                               status_plots.append((mod_name, counts))
                
                if status_plots:
                     color_map = {
                         "Complete": "#C6EFCE", "In Progress": "#BEE5EB", "Failed": "#FFC7CE",
                         "Pending": "#E2E3E5", "Error": "#FFC7CE", "Warning": "#FFEB9C"
                     }
                     
                     # If we have multiple, use a columns or a facet-able df
                     if len(status_plots) > 1:
                          view_type = st.radio("Chart View", ["Grid View", "Slide View"], horizontal=True, label_visibility="collapsed")
                          
                          if view_type == "Grid View":
                               sub_col1, sub_col2 = st.columns(2)
                               for i, (p_name, p_counts) in enumerate(status_plots):
                                    target_col = sub_col1 if i % 2 == 0 else sub_col2
                                    fig = px.pie(p_counts, values="Count", names="Status", title=f"{p_name} Status",
                                                 color="Status", color_discrete_map=color_map, hole=0.4)
                                    fig.update_layout(title_x=0.5, margin=dict(t=30, b=0, l=0, r=0), showlegend=False if i > 0 else True)
                                    target_col.plotly_chart(fig, use_container_width=True)
                          else:
                               # Combined for slide or dropdown
                               selected_plot = st.selectbox("Select Modality Status", [p[0] for p in status_plots])
                               p_name, p_counts = next(p for p in status_plots if p[0] == selected_plot)
                               fig = px.pie(p_counts, values="Count", names="Status", title=f"{p_name} Status Distribution",
                                            color="Status", color_discrete_map=color_map, hole=0.4)
                               fig.update_layout(title_x=0.5, margin=dict(t=50, b=0, l=0, r=0))
                               st.plotly_chart(fig, use_container_width=True)
                     else:
                          # Just one
                          p_name, p_counts = status_plots[0]
                          fig = px.pie(p_counts, values="Count", names="Status", title=f"{p_name} Pipeline Status Distribution",
                                       color="Status", color_discrete_map=color_map, hole=0.4)
                          fig.update_layout(title_x=0.5, margin=dict(t=50, b=0, l=0, r=0))
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
            st.dataframe(style_status_df(df_status), use_container_width=True)
        else:
            st.warning("No status sheets found in tracker.")

    with tab_subject:
        st.header("Individual Subject Details")
        subjects = []
        if "Processing_Status" in data:
             subjects = sorted(data["Processing_Status"]["Subject_ID"].unique().tolist())
        else:
             first_df = next(iter(data.values()))
             if "Subject_ID" in first_df.columns:
                  subjects = sorted(first_df["Subject_ID"].unique().tolist())
        
        if subjects:
             selected_subj = st.selectbox("Select Subject", subjects, key="subj_detail_selector")
             sessions = []
             if "Processing_Status" in data:
                  sessions = data["Processing_Status"][data["Processing_Status"]["Subject_ID"] == selected_subj]["Session"].unique().tolist()
             sessions = [str(s) if pd.notna(s) else "N/A" for s in sessions]
             
             st.info(f"Showing data for **{len(sessions)}** session(s): {', '.join(sessions) if sessions else 'N/A'}")
             
             st.subheader("📋 Processing Status (All Sessions)")
             if "Processing_Status" in data:
                  df_status = data["Processing_Status"][data["Processing_Status"]["Subject_ID"] == selected_subj].copy()
                  if not df_status.empty:
                       df_status["Session"] = df_status["Session"].apply(lambda x: str(x) if pd.notna(x) else "N/A")
                       st.dataframe(style_status_df(df_status), use_container_width=True, hide_index=True)
                  else:
                       st.info("No processing status found for this subject.")
             
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
             with col_meta:
                  st.subheader("📝 Metadata (All Sessions)")
                  if "Subject_Metadata" in data:
                       df_meta = data["Subject_Metadata"][data["Subject_Metadata"]["Subject_ID"] == selected_subj].copy()
                       if not df_meta.empty:
                            df_meta["Session"] = df_meta["Session"].apply(lambda x: str(x) if pd.notna(x) else "N/A")
                            df_meta = df_meta.dropna(axis=1, how='all')
                            st.dataframe(df_meta, use_container_width=True, hide_index=True)
                       else:
                            st.info("No metadata found for this subject.")
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
             roi_sheets = [s for s in data.keys() if "Metrics" in s and s != "Quality_Metrics"]
             if roi_sheets:
                  sel_roi_sheet = st.selectbox("View ROI Stats from Sheet", roi_sheets, key="roi_sheet_selector")
                  df_roi = data[sel_roi_sheet]
                  df_roi_sub = df_roi[df_roi["Subject_ID"] == selected_subj].copy()
                  
                  if not df_roi_sub.empty:
                       df_roi_sub["Session"] = df_roi_sub["Session"].apply(lambda x: str(x) if pd.notna(x) else "N/A")
                       
                       # Format Awareness
                       is_wide = "ROI_Name" not in df_roi_sub.columns
                       
                       # Data Table View
                       exp_table = st.expander("Explore Raw Subject ROI Data", expanded=False)
                       with exp_table:
                            if not is_wide and "Metric" in df_roi_sub.columns and "Statistic" in df_roi_sub.columns:
                                 view_mode = st.radio("View Mode", ["Pivoted (Wide)", "Long (Tidy)"], horizontal=True, key="view_mode_legacy")
                                 if view_mode == "Pivoted (Wide)":
                                      try:
                                           idx_cols = ["Session"]
                                           if "Atlas" in df_roi_sub.columns: idx_cols.append("Atlas")
                                           if "ROI_Name" in df_roi_sub.columns: idx_cols.append("ROI_Name")
                                           pivot_df = df_roi_sub.pivot_table(index=idx_cols, columns=["Metric", "Statistic"], values="Value", aggfunc='first')
                                           st.dataframe(pivot_df, use_container_width=True)
                                      except Exception as e:
                                           st.warning(f"Could not pivot: {e}")
                                           st.dataframe(df_roi_sub, use_container_width=True, hide_index=True)
                                 else:
                                      st.dataframe(df_roi_sub, use_container_width=True, hide_index=True)
                            else:
                                 st.dataframe(df_roi_sub, use_container_width=True, hide_index=True)

                       # Subject Analysis Suite
                       st.markdown("---")
                       st.subheader("📈 Subject ROI Analysis")
                       st.info("Visualize metric distributions across ROIs for this subject.")
                       
                       c1, c2, c3, c4 = st.columns(4)
                       avail_models = sorted(df_roi_sub["Model"].unique().tolist()) if "Model" in df_roi_sub.columns else ["N/A"]
                       avail_metrics = sorted(df_roi_sub["Metric"].unique().tolist()) if "Metric" in df_roi_sub.columns else ["N/A"]
                       avail_stats = sorted(df_roi_sub["Statistic"].unique().tolist()) if "Statistic" in df_roi_sub.columns else ["Mean"]
                       
                       sel_models = c1.multiselect("Models", avail_models, default=avail_models[:1] if avail_models else [])
                       sel_metrics = c2.multiselect("Metrics", avail_metrics, default=avail_metrics[:1] if avail_metrics else [])
                       sel_stats = c3.multiselect("Statistics", avail_stats, default=["Mean"] if "Mean" in avail_stats else avail_stats[:1])
                       
                       plot_df = df_roi_sub.copy()
                       if "Model" in plot_df.columns and sel_models: plot_df = plot_df[plot_df["Model"].isin(sel_models)]
                       if "Metric" in plot_df.columns and sel_metrics: plot_df = plot_df[plot_df["Metric"].isin(sel_metrics)]
                       if "Statistic" in plot_df.columns and sel_stats: plot_df = plot_df[plot_df["Statistic"].isin(sel_stats)]
                       
                       meta_cols = ['Subject_ID', 'Session', 'Study', 'Model', 'Metric', 'Statistic', 'Modality', 'ROI_Source', 'Timestamp']
                       if is_wide:
                            roi_cols = [c for c in plot_df.columns if c not in meta_cols and pd.api.types.is_numeric_dtype(plot_df[c])]
                            sel_rois = c4.multiselect("ROIs", sorted(roi_cols), default=roi_cols[:10] if len(roi_cols) > 10 else roi_cols)
                            if sel_rois:
                                 plot_df = plot_df.melt(id_vars=[c for c in plot_df.columns if c not in sel_rois], value_vars=sel_rois, var_name="ROI_Name", value_name="Value")
                       else:
                            if "ROI_Name" in plot_df.columns:
                                 avail_rois = sorted(plot_df["ROI_Name"].unique().tolist())
                                 sel_rois = c4.multiselect("ROIs", avail_rois, default=avail_rois[:10] if len(avail_rois) > 10 else avail_rois)
                                 if sel_rois: plot_df = plot_df[plot_df["ROI_Name"].isin(sel_rois)]

                       if not plot_df.empty and "Value" in plot_df.columns:
                            # Dynamic grouping based on what the user selected multiple of
                            if len(sel_stats) > 1: plot_group = "Statistic"
                            elif len(sel_metrics) > 1: plot_group = "Metric"
                            elif len(sel_models) > 1: plot_group = "Model"
                            else: plot_group = "Session"
                            
                            fig = px.bar(plot_df, x="ROI_Name", y="Value", color=plot_group, barmode="group",
                                         facet_row="Session" if len(plot_df["Session"].unique()) > 1 else None,
                                         title=f"ROI Metrics for {selected_subj}")
                            fig.update_layout(xaxis={'categoryorder':'total descending'})
                            st.plotly_chart(fig, use_container_width=True)
                            
                            fig_box = px.box(plot_df, x="Metric", y="Value", color=plot_group, points="all", hover_name="ROI_Name",
                                             title=f"Distribution across regions for {selected_subj}")
                            st.plotly_chart(fig_box, use_container_width=True)
                       else:
                            st.warning("No data matches the selected filters.")
                  else:
                       st.info("No ROI stats found for this subject.")
             else:
                  st.info("No ROI metric sheets found.")
             
             st.markdown("---")
             st.subheader("⚙️ Processing Details (All Sessions)")
             if "Processing_Details" in data:
                  df_details = data["Processing_Details"][data["Processing_Details"]["Subject_ID"] == selected_subj].copy()
                  if not df_details.empty:
                       df_details["Session"] = df_details["Session"].apply(lambda x: str(x) if pd.notna(x) else "N/A")
                       modalities = df_details["Modality"].unique().tolist()
                       if modalities:
                            tabs_details = st.tabs(modalities)
                            for i, mod in enumerate(modalities):
                                 with tabs_details[i]:
                                      df_mod = df_details[df_details["Modality"] == mod]
                                      view_mode = st.radio(f"View Mode ({mod})", ["By Step (Wide)", "Long (Tidy)"], horizontal=True, key=f"detail_view_{mod}")
                                      if view_mode == "By Step (Wide)":
                                           try:
                                                pivot_df = df_mod.pivot_table(index=["Session", "Step_Name"], columns="Parameter", values="Value", aggfunc='first')
                                                st.dataframe(pivot_df, use_container_width=True)
                                           except Exception as e:
                                                st.warning(f"Could not pivot: {e}")
                                                st.dataframe(df_mod, use_container_width=True, hide_index=True)
                                      else:
                                           st.dataframe(df_mod, use_container_width=True, hide_index=True)
                       else:
                            st.dataframe(df_details, use_container_width=True, hide_index=True)
                  else:
                       st.info("No processing details found.")
             else:
                  st.info("Processing Details sheet not found.")
        else:
             st.warning("No subjects found in tracker.")


    with tab_study_details:
        st.header("📊 Study-Wide Metric Analysis")
        st.info("Filter and visualize metrics across the entire study population.")
        
        # 1. Global Filters for this Tab
        sheet_names = list(data.keys())
        metric_sheets = [s for s in sheet_names if s.endswith("_Metrics") or s == "Volume_Statistics"]
        
        if not metric_sheets:
            st.warning("No metric or volume sheets found in tracker.")
        else:
            c1, c2, c3, c4 = st.columns(4)
            sel_sheet = c1.selectbox("Select Data Sheet", metric_sheets, key="sd_sheet")
            df = data[sel_sheet].copy()
            
            if selected_study != "All" and "Study" in df.columns:
                df = df[df["Study"] == selected_study]
            
            # Dynamic Filters based on Sheet
            is_volume = sel_sheet == "Volume_Statistics"
            is_tidy = "Metric" in df.columns and "Statistic" in df.columns
            is_wide = sel_sheet.endswith("_Metrics") and not is_tidy 
            
            models = ["All"]
            metrics = []
            stats = ["All"]
            
            if "Model" in df.columns:
                models = ["All"] + sorted(df["Model"].unique().tolist())
            if "Metric" in df.columns:
                metrics = sorted(df["Metric"].unique().tolist())
            if is_volume:
                metrics = sorted(df["Structure"].unique().tolist()) if "Structure" in df.columns else []
            if "Statistic" in df.columns:
                stats = ["All"] + sorted(df["Statistic"].unique().tolist())

            sel_model = c2.selectbox("Filter by Model", models, key="sd_model")
            sel_metric = c3.selectbox("Select Metric/Structure", metrics, key="sd_metric")
            sel_stat = c4.selectbox("Filter by Statistic", stats, key="sd_stat")
            
            # Apply Filters
            if sel_model != "All" and "Model" in df.columns: df = df[df["Model"] == sel_model]
            if is_volume:
                df = df[df["Structure"] == sel_metric]
                val_col = "Volume_mm3"
                roi_col = "Structure"
            else:
                if "Metric" in df.columns: df = df[df["Metric"] == sel_metric]
                if sel_stat != "All" and "Statistic" in df.columns: df = df[df["Statistic"] == sel_stat]
                val_col = "Value"
                roi_col = "ROI_Name"

            # ROI/Region Selection
            meta_cols = ['Subject_ID', 'Session', 'Study', 'Model', 'Metric', 'Statistic', 'Modality', 'ROI_Source', 'Timestamp', 'Structure', 'Method', 'Atlas']
            potential_rois = []
            if is_wide:
                potential_rois = sorted([c for c in df.columns if c not in meta_cols and pd.api.types.is_numeric_dtype(df[c])])
            elif roi_col in df.columns:
                potential_rois = sorted(df[roi_col].unique().tolist())
            
            sel_rois = st.multiselect("Filter by Region(s)", potential_rois, key="sd_rois")
            
            if is_wide and sel_rois:
                df = df.melt(id_vars=[c for c in df.columns if c not in sel_rois], value_vars=sel_rois, var_name='ROI_Name', value_name='Value')
                roi_col = "ROI_Name"
                val_col = "Value"
            elif sel_rois:
                df = df[df[roi_col].isin(sel_rois)]

            if not df.empty and val_col in df.columns:
                # 2. Study-Wide Averages
                st.markdown("---")
                st.subheader(f"📈 Overall Averages: {sel_metric}")
                try:
                    # Determine grouping for averages
                    avg_group = roi_col if roi_col in df.columns else "Metric"
                    if avg_group in df.columns:
                        means = df.groupby(avg_group)[val_col].mean().reset_index()
                        means.columns = ['Region', 'Mean_Value']
                        means = means.sort_values('Mean_Value', ascending=False).head(30)
                        fig_avg = px.bar(means, x='Region', y='Mean_Value', color='Mean_Value',
                                       title=f"Mean {sel_metric} across Subjects (Top 30)", color_continuous_scale="Viridis")
                        st.plotly_chart(fig_avg, use_container_width=True)
                except Exception as e:
                    st.error(f"Error calculating averages: {e}")

                # 3. Distributions
                st.markdown("---")
                st.subheader("📊 Metric Distributions")
                
                group_options = ["None"] + [c for c in df.columns if df[c].dtype == object and c not in [val_col]]
                default_group = roi_col if roi_col in group_options else "None"
                sel_group = st.selectbox("Group Distributions By", group_options, index=group_options.index(default_group))
                
                dist_col1, dist_col2 = st.columns(2)
                with dist_col1:
                    fig_hist = px.histogram(df, x=val_col, nbins=20, color=None if sel_group == "None" else sel_group,
                                           title=f"Distribution of {sel_metric}", marginal="box", barmode="overlay")
                    st.plotly_chart(fig_hist, use_container_width=True)
                with dist_col2:
                    fig_box = px.box(df, y=val_col, x=None if sel_group == "None" else sel_group,
                                    color=None if sel_group == "None" else sel_group, 
                                    title=f"Variation in {sel_metric}", points="all", hover_data=["Subject_ID"])
                    st.plotly_chart(fig_box, use_container_width=True)

                # 4. Data Table
                with st.expander("📋 View Filtered Data Table", expanded=False):
                    st.dataframe(df, use_container_width=True, hide_index=True)
            else:
                st.warning("No data matches the selected filters.")

    with tab_correlation:
        st.header("Metric Correlations")
        selected_sheet_corr = st.selectbox("Select Sheet for Correlation", sheet_names, key="corr_sheet")
        df_corr = data[selected_sheet_corr]
        if "Subject_Metadata" in data and selected_sheet_corr != "Subject_Metadata":
            if st.checkbox("Merge with Subject Metadata"):
                df_meta = data["Subject_Metadata"]
                df_corr = df_corr.merge(df_meta, on=["Subject_ID", "Session"], suffixes=('', '_meta'))
        if selected_study != "All" and "Study" in df_corr.columns:
            df_corr = df_corr[df_corr["Study"] == selected_study]
        is_tidy_corr = "Metric" in df_corr.columns and "Statistic" in df_corr.columns
        if is_tidy_corr:
            c1, c2, c3 = st.columns(3)
            if "Model" in df_corr.columns:
                models = ["All"] + df_corr["Model"].unique().tolist()
                sel_model = c1.selectbox("Filter Model (Corr)", models)
                if sel_model != "All": df_corr = df_corr[df_corr["Model"] == sel_model]
            metrics = df_corr["Metric"].unique().tolist()
            sel_metric = c2.selectbox("Select Metric (Corr)", metrics)
            df_corr = df_corr[df_corr["Metric"] == sel_metric]
            stats = df_corr["Statistic"].unique().tolist()
            sel_stat = c3.selectbox("Select Statistic (Corr)", stats)
            df_corr = df_corr[df_corr["Statistic"] == sel_stat]
            cola, colr = st.columns(2)
            if "Atlas" in df_corr.columns:
                 atlases = ["All"] + df_corr["Atlas"].unique().tolist()
                 sel_atlas = cola.selectbox("Filter Atlas (Corr)", atlases)
                 if sel_atlas != "All": df_corr = df_corr[df_corr["Atlas"] == sel_atlas]
            if "ROI_Name" in df_corr.columns:
                 rois = sorted(df_corr["ROI_Name"].unique().tolist())
                 sel_roi = colr.multiselect("Filter ROI (Corr)", rois)
                 if sel_roi: df_corr = df_corr[df_corr["ROI_Name"].isin(sel_roi)]
        numeric_cols_corr = df_corr.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols_corr) >= 2:
            c1, c2 = st.columns(2)
            metric_x = c1.selectbox("X Axis", numeric_cols_corr, index=0)
            metric_y = c2.selectbox("Y Axis", numeric_cols_corr, index=1 if len(numeric_cols_corr) > 1 else 0)
            color_by = st.selectbox("Color By", ["None"] + [c for c in df_corr.columns if df_corr[c].dtype == object])
            fig_scatter = px.scatter(df_corr, x=metric_x, y=metric_y, color=None if color_by == "None" else color_by,
                                    hover_data=["Subject_ID", "Session"], title=f"{metric_x} vs {metric_y}", trendline="ols")
            st.plotly_chart(fig_scatter, use_container_width=True)
        else:
            st.warning("Need at least 2 numeric metrics.")

    with tab_raw:
        st.header("Raw Data Sheets")
        for sheet, df in data.items():
            with st.expander(f"Sheet: {sheet}"):
                st.dataframe(df, use_container_width=True)
else:
    st.info("Please upload a tracker Excel file from the sidebar to begin.")
    st.markdown("""
    ### Features:
    - **Real-time Status Tracking**: Monitor your pipeline's progress.
    - **Quality Control**: Visualize motion and SNR.
    - **ROI Analysis**: Compare metrics between groups.
    - **Study-wide distributions**: Identify outliers.
    """)
