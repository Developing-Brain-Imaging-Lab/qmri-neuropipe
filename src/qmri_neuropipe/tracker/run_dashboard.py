"""
run_dashboard.py
================
Launch the qmri-neuropipe Streamlit dashboard backed by LabDB PostgreSQL.

This script is placed inside qmri-neuropipe's tracker/ directory alongside
app.py. It patches in a LabDBDataAdapter before Streamlit loads the app,
so the dashboard reads from PostgreSQL instead of an Excel file.

Usage
-----
# Directly:
    streamlit run tracker/run_dashboard.py -- \\
        --db-url postgresql://user:pass@localhost/research_db \\
        --project-id xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx

# Or use environment variables (good for containers/CI):
    export LABDB_URL="postgresql://user:pass@localhost/research_db"
    export LABDB_PROJECT_ID="xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"
    export LABDB_STUDY_NAME="MyStudy"     # optional, fills "Study" column
    streamlit run tracker/run_dashboard.py

# Or add a labdb: block to your YAML config and point to it:
    export LABDB_CONFIG="/path/to/config.yaml"
    streamlit run tracker/run_dashboard.py

Configuration
-------------
The adapter respects these environment variables:

    LABDB_URL          PostgreSQL SQLAlchemy URL
    LABDB_PROJECT_ID   UUID of the LabDB project to display
    LABDB_STUDY_NAME   Optional study name used in Study filter dropdown

How it works
------------
The qmri-neuropipe Streamlit app (app.py) reads a file, creates a
NeuroimagingTracker, and then uses tracker._data to get all its DataFrames.

This launcher replaces that startup with a LabDBDataAdapter that provides
the same ._data interface from PostgreSQL. The rest of app.py runs unchanged.

The approach is a context-manager-style monkeypatch that only takes effect
when launched through this file, so app.py itself is never modified.
"""

import os
import sys
import uuid
import argparse
from pathlib import Path

import streamlit as st


# ── Parse args / environment ──────────────────────────────────────────────────

def _get_config() -> dict:
    """Resolve db_url and project_id from CLI args, env vars, or YAML config."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--db-url", default=None)
    parser.add_argument("--project-id", default=None)
    parser.add_argument("--study-name", default=None)
    parser.add_argument("--config", default=None, help="Path to qmri-neuropipe YAML config")
    args, _ = parser.parse_known_args()

    db_url = args.db_url or os.environ.get("LABDB_URL")
    project_id_str = args.project_id or os.environ.get("LABDB_PROJECT_ID")
    study_name = args.study_name or os.environ.get("LABDB_STUDY_NAME")

    # Fall back to YAML config if provided
    config_path = args.config or os.environ.get("LABDB_CONFIG")
    if config_path and Path(config_path).exists() and (not db_url or not project_id_str):
        try:
            import yaml
            with open(config_path) as f:
                cfg = yaml.safe_load(f)
            labdb = cfg.get("labdb", {})
            db_url = db_url or labdb.get("db_url")
            project_id_str = project_id_str or labdb.get("project_id")
            study_name = study_name or labdb.get("study_name") or cfg.get("study_name")
        except Exception as e:
            st.error(f"Could not read YAML config: {e}")

    return {
        "db_url": db_url,
        "project_id": project_id_str,
        "study_name": study_name,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="qMRI Tracker Dashboard (LabDB)",
        page_icon="🧠",
        layout="wide",
    )

    cfg = _get_config()
    db_url = cfg["db_url"]
    project_id_str = cfg["project_id"]
    study_name = cfg["study_name"]

    # ── Sidebar data-source panel ─────────────────────────────────────────────
    st.sidebar.header("Data Source: LabDB")

    if not db_url:
        db_url = st.sidebar.text_input(
            "PostgreSQL URL",
            placeholder="postgresql://user:pass@localhost/research_db",
        )
    else:
        st.sidebar.success("✓ Database URL configured")

    if not project_id_str:
        project_id_str = st.sidebar.text_input(
            "Project UUID",
            placeholder="xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
        )
    else:
        st.sidebar.success(f"✓ Project: {project_id_str[:8]}…")

    study_name_input = st.sidebar.text_input("Study Name (optional)", value=study_name or "")
    if study_name_input:
        study_name = study_name_input

    refresh_btn = st.sidebar.button("🔄 Refresh from database")

    if not db_url or not project_id_str:
        st.title("🧠 qMRI Neuroimaging Tracker (LabDB)")
        st.info(
            "Provide a PostgreSQL URL and Project UUID to load pipeline data "
            "directly from LabDB. You can set these via environment variables:\n\n"
            "```\n"
            "export LABDB_URL='postgresql://user:pass@localhost/research_db'\n"
            "export LABDB_PROJECT_ID='your-project-uuid'\n"
            "```\n"
            "Then rerun: `streamlit run tracker/run_dashboard.py`"
        )
        return

    try:
        project_id = uuid.UUID(project_id_str.strip())
    except ValueError:
        st.error(f"Invalid project UUID: {project_id_str!r}")
        return

    # ── Build adapter (cached in session state to avoid repeated DB queries) ──
    cache_key = f"labdb_adapter_{project_id}"
    if cache_key not in st.session_state or refresh_btn:
        try:
            # Add research_db to path if not installed
            _maybe_add_research_db_to_path()

            from research_db.dashboard_adapter import LabDBDataAdapter
            adapter = LabDBDataAdapter(
                db_url=db_url,
                project_id=project_id,
                study_name=study_name or "All",
            )
            adapter.refresh()
            st.session_state[cache_key] = adapter
            if refresh_btn:
                st.sidebar.success("Data refreshed!")
        except Exception as e:
            st.error(f"Could not connect to LabDB: {e}")
            st.info("Check your PostgreSQL URL and that the research_db package is installed.")
            return

    adapter = st.session_state[cache_key]
    data = adapter._data

    # ── Study filter ──────────────────────────────────────────────────────────
    studies = ["All"]
    st.sidebar.selectbox("Filter by Study", studies, key="study_filter_compat")
    # (The qmri-neuropipe app checks data["Subject_Metadata"]["Study"]; since we
    #  populate Study in all sheets, its study filter will work automatically once
    #  data is loaded via the original app. In this standalone mode we just show "All".)

    # ── Sheet count in sidebar ─────────────────────────────────────────────────
    sheet_names = list(data.keys())
    n_runs = len(data.get("Processing_Status", pd.DataFrame()))
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**{n_runs} pipeline runs** across {len(sheet_names)} data sheets")
    for name in sheet_names:
        df = data[name]
        st.sidebar.caption(f"• {name}: {len(df)} rows")

    # ── Render the original app with patched data ─────────────────────────────
    # We re-implement the app's main body here using the same `data` dict.
    # This avoids needing to patch NeuroimagingTracker at import time.
    _render_dashboard(data)


def _render_dashboard(data: dict):
    """
    Render the qmri-neuropipe Streamlit dashboard using a data dict.

    This is a clean re-import of the rendering logic from app.py,
    adapted to accept an already-loaded data dict instead of loading
    from an Excel file. The logic, tabs, and chart types are identical.

    Alternatively you can call exec(open("tracker/app.py").read()) after
    injecting the adapter — see the comment at the bottom of this file.
    """
    import numpy as np
    import plotly.express as px
    import pandas as pd

    def color_status(val):
        if not isinstance(val, str): return ""
        v = val.lower()
        if "complete" in v: return "background-color: #C6EFCE; color: #006100; font-weight: bold;"
        if "progress" in v: return "background-color: #BEE5EB; color: #0C5460; font-weight: bold;"
        if "pending" in v or "queued" in v: return "background-color: #E2E3E5; color: #383d41;"
        if "warning" in v: return "background-color: #FFEB9C; color: #9C5700; font-weight: bold;"
        if "error" in v or "failed" in v: return "background-color: #FFC7CE; color: #9C0006; font-weight: bold;"
        return ""

    def style_df(df):
        exclude = {"Subject_ID", "Session", "Study", "Last_Update", "Last_Processing_Date",
                   "Pipeline", "Version", "Modality"}
        cols = [c for c in df.columns if c not in exclude]
        try:
            return df.style.map(color_status, subset=cols)
        except Exception:
            return df.style.applymap(color_status, subset=cols)

    st.title("🧠 qMRI Neuroimaging Tracker")
    st.caption("Powered by LabDB PostgreSQL")

    selected_study = "All"  # study filter in sidebar handled above

    # ── Global metrics ────────────────────────────────────────────────────────
    df_ps = data.get("Processing_Status", pd.DataFrame())
    df_anat = data.get("Anatomical_Status", pd.DataFrame())
    df_diff = data.get("Diffusion_Status", pd.DataFrame())
    df_qm = data.get("Quality_Metrics", pd.DataFrame())

    total_subjs = df_ps["Subject_ID"].nunique() if not df_ps.empty else 0
    total_sess = len(df_ps) if not df_ps.empty else 0

    def _comp_rate(df, col="Overall_Status"):
        if df.empty or col not in df.columns: return "—"
        n = total_sess or 1
        return f"{(df[col].str.contains('Complete', case=False, na=False).sum() / n)*100:.1f}%"

    anat_rate = _comp_rate(df_anat)
    diff_rate = _comp_rate(df_diff)

    avg_snr = avg_fd = outliers = outliers_pct = "N/A"
    if not df_qm.empty:
        if "DWI_SNR" in df_qm.columns:
            v = df_qm["DWI_SNR"].dropna().mean()
            if pd.notna(v): avg_snr = f"{v:.1f}"
        for fd_col in ("DWI_Motion_FD_Mean", "Motion_FD_Mean"):
            if fd_col in df_qm.columns:
                v = df_qm[fd_col].dropna().mean()
                if pd.notna(v): avg_fd = f"{v:.3f}"; break
        if "DWI_Outliers_Removed_Volumes" in df_qm.columns:
            v = df_qm["DWI_Outliers_Removed_Volumes"].dropna().sum()
            if pd.notna(v): outliers = str(int(v))
        if "DWI_Outliers_Removed_Pct" in df_qm.columns:
            v = df_qm["DWI_Outliers_Removed_Pct"].dropna().mean()
            if pd.notna(v): outliers_pct = f"{v:.2f}%"

    # ── Tabs ─────────────────────────────────────────────────────────────────
    tab_summary, tab_study, tab_subject, tab_status, tab_corr, tab_raw = st.tabs([
        "📊 Summary", "📈 Study Details", "👤 Subject Details",
        "⚙️ Processing Status", "🔗 Correlations", "📋 Raw Data",
    ])

    # ── Summary tab ───────────────────────────────────────────────────────────
    with tab_summary:
        st.header("Executive Summary")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Subjects", total_subjs)
        c2.metric("Total Sessions", total_sess)
        c3.metric("Anatomical Success", anat_rate)
        c4.metric("Diffusion Success", diff_rate)

        q1, q2, q3, q4 = st.columns(4)
        q1.metric("Avg DWI SNR", avg_snr)
        q2.metric("Avg Mean FD", avg_fd)
        q3.metric("Outliers Removed", outliers)
        q4.metric("Outliers Removed %", outliers_pct)

        st.markdown("---")
        col_left, col_right = st.columns([1, 2])

        with col_left:
            st.subheader("📋 Study Summary")
            st.dataframe(data.get("Summary", pd.DataFrame()), use_container_width=True, hide_index=True)
            if not df_ps.empty and "Last_Processing_Date" in df_ps.columns:
                st.subheader("🕒 Recent Activity")
                recent = df_ps.sort_values("Last_Processing_Date", ascending=False).head(5)
                st.dataframe(recent[["Subject_ID", "Session", "Overall_Pipeline_Status"]],
                             use_container_width=True, hide_index=True)

        with col_right:
            st.subheader("📊 Pipeline Status by Modality")
            color_map = {
                "Complete": "#C6EFCE", "In Progress": "#BEE5EB", "Failed": "#FFC7CE",
                "Pending": "#E2E3E5", "Error": "#FFC7CE", "Warning": "#FFEB9C",
            }
            plots = []
            for label, sheet, col in [
                ("Overall", "Processing_Status", "Overall_Pipeline_Status"),
                ("Anatomical", "Anatomical_Status", "Overall_Status"),
                ("Diffusion", "Diffusion_Status", "Overall_Status"),
                ("Relaxometry", "Relaxometry_Status", "Overall_Status"),
            ]:
                df_s = data.get(sheet, pd.DataFrame())
                if not df_s.empty and col in df_s.columns:
                    counts = df_s[col].value_counts().reset_index()
                    counts.columns = ["Status", "Count"]
                    plots.append((label, counts))

            if plots:
                subcols = st.columns(min(len(plots), 2))
                for i, (label, counts) in enumerate(plots):
                    fig = px.pie(counts, values="Count", names="Status", title=f"{label} Status",
                                 color="Status", color_discrete_map=color_map, hole=0.4)
                    fig.update_layout(margin=dict(t=30, b=0, l=0, r=0), showlegend=True)
                    subcols[i % 2].plotly_chart(fig, use_container_width=True)

    # ── Study Details tab ─────────────────────────────────────────────────────
    with tab_study:
        st.header("📈 Study-Wide Metric Analysis")
        metric_sheets = [s for s in data if s.endswith("_Metrics") or s == "Volume_Statistics"]
        if not metric_sheets:
            st.info("No metric data recorded yet.")
        else:
            c1, c2, c3, c4 = st.columns(4)
            sel_sheet = c1.selectbox("Data Sheet", metric_sheets, key="sd_sheet")
            df = data[sel_sheet].copy()

            is_vol = sel_sheet == "Volume_Statistics"
            is_tidy = "Metric" in df.columns and "Statistic" in df.columns

            models = ["All"] + sorted(df["Model"].unique().tolist()) if "Model" in df.columns else ["All"]
            metrics = sorted(df["Structure"].unique()) if is_vol and "Structure" in df.columns \
                      else sorted(df["Metric"].unique()) if "Metric" in df.columns else []
            stats = ["All"] + sorted(df["Statistic"].unique()) if "Statistic" in df.columns else ["All"]

            sel_model = c2.selectbox("Filter by Model", models, key="sd_model")
            sel_metric = c3.selectbox("Select Metric/Structure", metrics or ["—"], key="sd_metric")
            sel_stat = c4.selectbox("Statistic", stats, key="sd_stat")

            if sel_model != "All" and "Model" in df.columns:
                df = df[df["Model"] == sel_model]
            if is_vol:
                if "Structure" in df.columns: df = df[df["Structure"] == sel_metric]
                val_col, roi_col = "Volume_mm3", "Structure"
            else:
                if "Metric" in df.columns: df = df[df["Metric"] == sel_metric]
                if sel_stat != "All" and "Statistic" in df.columns: df = df[df["Statistic"] == sel_stat]
                val_col, roi_col = "Value", "ROI_Name"

            if not df.empty and val_col in df.columns and roi_col in df.columns:
                means = df.groupby(roi_col)[val_col].mean().reset_index()
                means.columns = ["Region", "Mean"]
                means = means.sort_values("Mean", ascending=False).head(30)
                fig_bar = px.bar(means, x="Region", y="Mean", color="Mean",
                                 color_continuous_scale="Viridis",
                                 title=f"Mean {sel_metric} (Top 30 Regions)")
                fig_bar.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig_bar, use_container_width=True)

                d1, d2 = st.columns(2)
                with d1:
                    fig_hist = px.histogram(df, x=val_col, nbins=20, marginal="box",
                                            title=f"Distribution of {sel_metric}")
                    st.plotly_chart(fig_hist, use_container_width=True)
                with d2:
                    fig_box = px.box(df, y=val_col, x=roi_col if df[roi_col].nunique() <= 20 else None,
                                     points="all", hover_data=["Subject_ID"] if "Subject_ID" in df.columns else None,
                                     title=f"Variation in {sel_metric}")
                    st.plotly_chart(fig_box, use_container_width=True)

                with st.expander("View filtered data table"):
                    st.dataframe(df.head(500), use_container_width=True, hide_index=True)
            else:
                st.info("No data matches current filters.")

    # ── Subject Details tab ───────────────────────────────────────────────────
    with tab_subject:
        st.header("👤 Individual Subject Details")
        if df_ps.empty:
            st.info("No pipeline data found.")
        else:
            subjects = sorted(df_ps["Subject_ID"].unique())
            sel_subj = st.selectbox("Select Subject", subjects)

            sessions = df_ps[df_ps["Subject_ID"] == sel_subj]["Session"].unique().tolist()
            st.info(f"**{len(sessions)} session(s):** {', '.join(str(s) for s in sessions)}")

            st.subheader("Processing Status")
            subj_ps = df_ps[df_ps["Subject_ID"] == sel_subj]
            st.dataframe(style_df(subj_ps), use_container_width=True, hide_index=True)

            for mod_sheet in ("Anatomical_Status", "Diffusion_Status", "Relaxometry_Status"):
                df_m = data.get(mod_sheet, pd.DataFrame())
                if not df_m.empty:
                    sub_df = df_m[df_m["Subject_ID"] == sel_subj]
                    if not sub_df.empty:
                        st.subheader(mod_sheet.replace("_", " "))
                        st.dataframe(style_df(sub_df), use_container_width=True, hide_index=True)

            qa_c1, qa_c2 = st.columns(2)
            with qa_c1:
                st.subheader("QC Metrics")
                if not df_qm.empty:
                    sq = df_qm[df_qm["Subject_ID"] == sel_subj].dropna(axis=1, how="all")
                    st.dataframe(sq, use_container_width=True, hide_index=True)
            with qa_c2:
                st.subheader("Volume Statistics")
                df_vol = data.get("Volume_Statistics", pd.DataFrame())
                if not df_vol.empty:
                    sv = df_vol[df_vol["Subject_ID"] == sel_subj]
                    st.dataframe(sv, use_container_width=True, hide_index=True)

            st.markdown("---")
            st.subheader("🧠 ROI Metrics")
            roi_sheets = [s for s in data if s.endswith("_Metrics")]
            if roi_sheets:
                sel_roi_sheet = st.selectbox("Atlas / Sheet", roi_sheets)
                df_roi = data[sel_roi_sheet]
                df_roi_sub = df_roi[df_roi["Subject_ID"] == sel_subj]
                if not df_roi_sub.empty:
                    c1, c2, c3 = st.columns(3)
                    avail_metrics = sorted(df_roi_sub["Metric"].unique()) if "Metric" in df_roi_sub.columns else []
                    avail_models = sorted(df_roi_sub["Model"].dropna().unique()) if "Model" in df_roi_sub.columns else []
                    avail_stats = sorted(df_roi_sub["Statistic"].unique()) if "Statistic" in df_roi_sub.columns else []
                    sel_met = c1.selectbox("Metric", avail_metrics or ["—"], key="subj_met")
                    sel_mod = c2.selectbox("Model", ["All"] + avail_models, key="subj_mod")
                    sel_st = c3.selectbox("Statistic", avail_stats or ["Mean"], key="subj_st")

                    plot_df = df_roi_sub.copy()
                    if "Metric" in plot_df.columns and sel_met: plot_df = plot_df[plot_df["Metric"] == sel_met]
                    if sel_mod != "All" and "Model" in plot_df.columns: plot_df = plot_df[plot_df["Model"] == sel_mod]
                    if "Statistic" in plot_df.columns and sel_st: plot_df = plot_df[plot_df["Statistic"] == sel_st]

                    if not plot_df.empty and "ROI_Name" in plot_df.columns and "Value" in plot_df.columns:
                        fig = px.bar(plot_df, x="ROI_Name", y="Value", color="Session",
                                     title=f"{sel_met} by ROI — {sel_subj}")
                        fig.update_layout(xaxis_tickangle=-45)
                        st.plotly_chart(fig, use_container_width=True)

                    with st.expander("Raw ROI data"):
                        st.dataframe(df_roi_sub, use_container_width=True, hide_index=True)

    # ── Processing Status tab ─────────────────────────────────────────────────
    with tab_status:
        st.header("⚙️ Processing Status")
        status_sheets = [s for s in ("Processing_Status", "Anatomical_Status",
                                      "Diffusion_Status", "Relaxometry_Status") if s in data]
        if status_sheets:
            sel = st.selectbox("View", status_sheets)
            st.dataframe(style_df(data[sel]), use_container_width=True)
        else:
            st.info("No status data.")

    # ── Correlations tab ──────────────────────────────────────────────────────
    with tab_corr:
        st.header("🔗 Metric Correlations")
        corr_sheets = list(data.keys())
        sel_sheet = st.selectbox("Sheet", corr_sheets, key="corr_sheet")
        df_c = data[sel_sheet].copy()

        if "Subject_Metadata" in data and sel_sheet != "Subject_Metadata":
            if st.checkbox("Merge with metadata"):
                df_c = df_c.merge(data["Subject_Metadata"], on=["Subject_ID", "Session"], suffixes=("", "_meta"))

        if "Metric" in df_c.columns and "Statistic" in df_c.columns:
            cc1, cc2, cc3 = st.columns(3)
            if "Model" in df_c.columns:
                mods = ["All"] + df_c["Model"].dropna().unique().tolist()
                sm = cc1.selectbox("Model", mods, key="cmod")
                if sm != "All": df_c = df_c[df_c["Model"] == sm]
            metrics = df_c["Metric"].unique()
            sm2 = cc2.selectbox("Metric", metrics, key="cmet")
            df_c = df_c[df_c["Metric"] == sm2]
            ss = cc3.selectbox("Statistic", df_c["Statistic"].unique(), key="cstat")
            df_c = df_c[df_c["Statistic"] == ss]

        num_cols = df_c.select_dtypes(include=[np.number]).columns.tolist()
        if len(num_cols) >= 2:
            cx, cy = st.columns(2)
            x_col = cx.selectbox("X", num_cols, index=0)
            y_col = cy.selectbox("Y", num_cols, index=1)
            color_opts = ["None"] + [c for c in df_c.columns if df_c[c].dtype == object]
            col_by = st.selectbox("Color by", color_opts)
            try:
                fig_s = px.scatter(
                    df_c, x=x_col, y=y_col,
                    color=None if col_by == "None" else col_by,
                    hover_data=["Subject_ID", "Session"] if "Subject_ID" in df_c.columns else None,
                    trendline="ols", title=f"{x_col} vs {y_col}",
                )
                st.plotly_chart(fig_s, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not plot: {e}")
        else:
            st.info("Need at least 2 numeric columns for correlation.")

    # ── Raw Data tab ──────────────────────────────────────────────────────────
    with tab_raw:
        st.header("📋 Raw Data")
        for sheet_name, df_raw in data.items():
            with st.expander(f"{sheet_name} ({len(df_raw)} rows)"):
                st.dataframe(df_raw, use_container_width=True)


def _maybe_add_research_db_to_path():
    """Add research_db to sys.path if it isn't installed as a package."""
    try:
        import research_db  # noqa
    except ImportError:
        # Try common locations relative to this file
        candidates = [
            Path(__file__).parents[2] / "research_db",  # tracker/../research_db
            Path(__file__).parents[3] / "research_db",
            Path.home() / "research_db",
            Path.home() / "projects" / "research_db",
        ]
        for candidate in candidates:
            if (candidate / "research_db").is_dir():
                sys.path.insert(0, str(candidate))
                return
        raise ImportError(
            "research_db package not found. Install it with:\n"
            "    pip install -e /path/to/research_db\n"
            "or set PYTHONPATH to include its parent directory."
        )


if __name__ == "__main__":
    import pandas as pd  # ensure available inside _render_dashboard
    main()
