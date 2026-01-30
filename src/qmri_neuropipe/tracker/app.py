import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

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

if tracker_file:
    # Save uploaded file to a temporary location to use with NeuroimagingTracker
    temp_path = Path("temp_tracker.xlsx")
    with open(temp_path, "wb") as f:
        f.write(tracker_file.getbuffer())
    
    tracker = NeuroimagingTracker(temp_path)
    data = tracker._data
    
    # Study selector
    studies = ["All"]
    if "Subject_Metadata" in data and "Study" in data["Subject_Metadata"].columns:
        studies.extend(data["Subject_Metadata"]["Study"].unique().tolist())
    
    selected_study = st.sidebar.selectbox("Filter by Study", studies)
    
    # Tabs for different views
    tab_overview, tab_distribution, tab_correlation, tab_raw = st.tabs([
        "📊 Overview", "📈 Distributions", "🔗 Correlations", "📋 Raw Data"
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

    with tab_distribution:
        st.header("Metric Distributions")
        
        # Select sheet and column
        sheet_names = list(data.keys())
        selected_sheet = st.selectbox("Select Data Sheet", sheet_names)
        
        df = data[selected_sheet]
        if selected_study != "All" and "Study" in df.columns:
            df = df[df["Study"] == selected_study]
            
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if numeric_cols:
            selected_metric = st.selectbox("Select Metric to Visualize", numeric_cols)
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig_hist = px.histogram(df, x=selected_metric, nbins=20, title=f"Distribution of {selected_metric}",
                                       marginal="box", color_discrete_sequence=['#636EFA'])
                st.plotly_chart(fig_hist, use_container_width=True)
                
            with col2:
                fig_box = px.box(df, y=selected_metric, title=f"Boxplot of {selected_metric}",
                                points="all", color_discrete_sequence=['#EF553B'])
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
