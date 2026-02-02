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
    tab_summary, tab_overview, tab_subject, tab_volumes, tab_distribution, tab_correlation, tab_raw = st.tabs([
        "📊 Summary", "⚙️ Processing Status", "👤 Subject Details", "🧠 Volumes & ROIs", "📈 Distributions", "🔗 Correlations", "📋 Raw Data"
    ])

    with tab_summary:
        st.header("Executive Summary")
        
        # 1. High-Level Metric Cards
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        
        # Calculate live metrics
        total_subjs = 0
        total_sess = 0
        comp_rate = "0%"
        avg_snr = "N/A"
        avg_mot = "N/A"
        
        if "Processing_Status" in data:
            df_ps_all = data["Processing_Status"]
            if selected_study != "All" and "Study" in df_ps_all.columns:
                 df_ps_all = df_ps_all[df_ps_all["Study"] == selected_study]
            
            total_subjs = df_ps_all["Subject_ID"].nunique()
            total_sess = len(df_ps_all)
            
            if "Overall_Pipeline_Status" in df_ps_all.columns:
                 complete = (df_ps_all["Overall_Pipeline_Status"].astype(str).str.contains("Complete", case=False)).sum()
                 comp_rate = f"{(complete/total_sess)*100:.1f}%" if total_sess > 0 else "0%"
        
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
        col_m3.metric("Completion Rate", comp_rate)
        col_m4.metric("Avg SNR / Motion", f"{avg_snr} / {avg_mot}")

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
                if "Processing_Status" in data:
                    df_ps = data["Processing_Status"]
                    if selected_study != "All" and "Study" in df_ps.columns:
                         df_ps = df_ps[df_ps["Study"] == selected_study]
                         
                    if "Overall_Pipeline_Status" in df_ps.columns:
                        counts = df_ps["Overall_Pipeline_Status"].value_counts().reset_index()
                        counts.columns = ["Status", "Count"]
                        color_map = {
                            "Complete": "#C6EFCE", "In Progress": "#BEE5EB", "Failed": "#FFC7CE",
                            "Pending": "#E2E3E5", "Error": "#FFC7CE", "Warning": "#FFEB9C"
                        }
                        fig = px.pie(counts, values="Count", names="Status", title="Overall Pipeline Status Distribution",
                                     color="Status", color_discrete_map=color_map)
                        # Center the title
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


    with tab_volumes:
        st.header("🧠 Anatomical Volumes & ROI Metrics")
        vol_tab, roi_tab = st.tabs(["Volume Statistics", "ROI Metrics"])
        with vol_tab:
            st.subheader("Anatomical Structure Volumes")
            if "Volume_Statistics" in data:
                df_vol = data["Volume_Statistics"].copy()
                if selected_study != "All" and "Study" in df_vol.columns:
                    df_vol = df_vol[df_vol["Study"] == selected_study]
                if not df_vol.empty:
                    c1, c2 = st.columns(2)
                    methods = ["All"] + df_vol["Method"].unique().tolist() if "Method" in df_vol.columns else ["All"]
                    structures = ["All"] + df_vol["Structure"].unique().tolist() if "Structure" in df_vol.columns else ["All"]
                    sel_method = c1.selectbox("Filter by Method", methods)
                    sel_struct = c2.selectbox("Filter by Structure", structures)
                    if sel_method != "All": df_vol = df_vol[df_vol["Method"] == sel_method]
                    if sel_struct != "All": df_vol = df_vol[df_vol["Structure"] == sel_struct]
                    st.dataframe(df_vol, use_container_width=True, hide_index=True)
                    if len(df_vol) > 0 and "Structure" in df_vol.columns and "Volume_mm3" in df_vol.columns:
                        st.markdown("---")
                        st.subheader("Volume Comparison")
                        try:
                            chart_df = df_vol.groupby("Structure")["Volume_mm3"].mean().reset_index()
                            chart_df = chart_df.sort_values("Volume_mm3", ascending=False).head(20)
                            st.bar_chart(chart_df.set_index("Structure")["Volume_mm3"])
                        except Exception as e:
                            st.warning(f"Could not create chart: {e}")
                else:
                    st.info("No volume statistics available.")
            else:
                st.info("Volume Statistics sheet not found.")
        
        with roi_tab:
            st.subheader("Region of Interest (ROI) Metrics")
            metric_sheets = [s for s in data.keys() if s.endswith("_Metrics")]
            if metric_sheets:
                sel_metric_sheet = st.selectbox("Select Atlas / Metric Sheet", metric_sheets)
                df_roi = data[sel_metric_sheet].copy()
                if selected_study != "All" and "Study" in df_roi.columns:
                    df_roi = df_roi[df_roi["Study"] == selected_study]
                if not df_roi.empty:
                    c1, c2, c3 = st.columns(3)
                    models = ["All"] + df_roi["Model"].unique().tolist() if "Model" in df_roi.columns else ["All"]
                    metrics = ["All"] + df_roi["Metric"].unique().tolist() if "Metric" in df_roi.columns else ["All"]
                    stats = ["All"] + df_roi["Statistic"].unique().tolist() if "Statistic" in df_roi.columns else ["All"]
                    sel_model = c1.selectbox("Filter by Model", models, key=f"roi_mod_{sel_metric_sheet}")
                    sel_met = c2.selectbox("Filter by Metric", metrics, key=f"roi_met_{sel_metric_sheet}")
                    sel_stat = c3.selectbox("Filter by Statistic", stats, key=f"roi_stat_{sel_metric_sheet}")
                    if sel_model != "All": df_roi = df_roi[df_roi["Model"] == sel_model]
                    if sel_met != "All": df_roi = df_roi[df_roi["Metric"] == sel_met]
                    if sel_stat != "All": df_roi = df_roi[df_roi["Statistic"] == sel_stat]
                    st.dataframe(df_roi, use_container_width=True, hide_index=True)
                    meta_cols = ['Subject_ID', 'Session', 'Study', 'Model', 'Metric', 'Statistic', 'Modality', 'ROI_Source', 'Timestamp']
                    roi_cols = [c for c in df_roi.columns if c not in meta_cols and pd.api.types.is_numeric_dtype(df_roi[c])]
                    if roi_cols:
                        st.markdown("---")
                        st.subheader(f"Mean Values")
                        try:
                            means = df_roi[roi_cols].mean().reset_index()
                            means.columns = ['ROI', 'Mean_Value']
                            means = means.sort_values('Mean_Value', ascending=False).head(30)
                            fig = px.bar(means, x='ROI', y='Mean_Value', title=f"Top 30 Regions: {sel_met}")
                            st.plotly_chart(fig, use_container_width=True)
                        except Exception as e:
                            st.warning(f"Could not create chart: {e}")
                else:
                    st.info(f"No data found in {sel_metric_sheet}.")
            else:
                st.info("No ROI Metric sheets found.")

    with tab_distribution:
        st.header("Metric Distributions")
        sheet_names = list(data.keys())
        selected_sheet = st.selectbox("Select Data Sheet", sheet_names)
        df = data[selected_sheet]
        if selected_study != "All" and "Study" in df.columns:
            df = df[df["Study"] == selected_study]
        sel_roi = None
        is_tidy = "Metric" in df.columns and "Statistic" in df.columns
        is_wide = selected_sheet.endswith("_Metrics") and not is_tidy 
        if is_tidy or is_wide:
            c1, c2, c3 = st.columns(3)
            if "Model" in df.columns:
                models = ["All"] + df["Model"].unique().tolist()
                sel_model = c1.selectbox("Filter by Model (Dist)", models)
                if sel_model != "All": df = df[df["Model"] == sel_model]
            if "Metric" in df.columns:
                metrics = df["Metric"].unique().tolist()
                sel_metric = c2.selectbox("Select Metric (Dist)", metrics)
                df = df[df["Metric"] == sel_metric]
            if "Statistic" in df.columns:
                stats = df["Statistic"].unique().tolist()
                sel_stat = c3.selectbox("Select Statistic (Dist)", stats)
                df = df[df["Statistic"] == sel_stat]
            cola, colr = st.columns(2)
            if "Atlas" in df.columns:
                 atlases = ["All"] + df["Atlas"].unique().tolist()
                 sel_altas = cola.selectbox("Filter by Atlas (Dist)", atlases)
                 if sel_altas != "All": df = df[df["Atlas"] == sel_altas]
            if "ROI_Name" in df.columns:
                 rois = sorted(df["ROI_Name"].unique().tolist())
                 sel_roi = colr.multiselect("Filter by ROI (Dist)", rois)
                 if sel_roi: df = df[df["ROI_Name"].isin(sel_roi)]
            elif is_wide:
                 meta_cols = ['Subject_ID', 'Session', 'Study', 'Model', 'Metric', 'Statistic', 'Modality', 'ROI_Source', 'Timestamp']
                 potential_rois = sorted([c for c in df.columns if c not in meta_cols and pd.api.types.is_numeric_dtype(df[c])])
                 sel_roi_cols = colr.multiselect("Select regions to compare (Dist)", potential_rois)
                 if sel_roi_cols:
                      df = df.melt(id_vars=[c for c in df.columns if c not in sel_roi_cols], value_vars=sel_roi_cols, var_name='ROI_Name', value_name='ROI_Value')
                      sel_roi = sel_roi_cols
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            if 'ROI_Value' in df.columns: selected_metric = 'ROI_Value'
            else:
                 default_metric = "Value" if "Value" in numeric_cols else numeric_cols[0]
                 selected_metric = st.selectbox("Select Metric to Visualize", numeric_cols, index=numeric_cols.index(default_metric))
            group_options = ["None"] + [c for c in df.columns if df[c].dtype == object and c not in [selected_metric]]
            if sel_roi and "ROI_Name" in df.columns:
                 sel_group = st.selectbox("Group By (Dist)", group_options, index=group_options.index("ROI_Name") if "ROI_Name" in group_options else 0)
            else:
                 sel_group = st.selectbox("Group By (Dist)", group_options)
            c1, c2 = st.columns(2)
            with c1:
                fig_hist = px.histogram(df, x=selected_metric, nbins=20, color=None if sel_group == "None" else sel_group,
                                       title=f"Distribution of {selected_metric}", marginal="box", barmode="overlay")
                st.plotly_chart(fig_hist, use_container_width=True)
            with c2:
                fig_box = px.box(df, y=selected_metric, x=None if sel_group == "None" else sel_group,
                                color=None if sel_group == "None" else sel_group, title=f"Boxplot of {selected_metric}", points="all")
                st.plotly_chart(fig_box, use_container_width=True)
        else:
            st.warning("No numeric metrics found.")

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
