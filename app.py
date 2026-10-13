import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="EPCS — Explainable AI CPU Scheduler",
    page_icon="⚙️",
    layout="wide",
)

st.title("⚙️ EPCS — Explainable AI CPU Scheduler")
st.subheader("Intelligent Scheduling | Performance Analysis | Explainable AI")
st.caption(
    "Upload the exported dashboard ZIP from the Kaggle notebook. "
    "This dashboard visualizes exported results; it does not silently retrain the XGBoost model."
)
st.divider()

# ---------- Data loading ----------
@st.cache_data(show_spinner=False)
def load_zip_csvs(zip_bytes: bytes) -> dict[str, pd.DataFrame]:
    data: dict[str, pd.DataFrame] = {}
    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zf:
        for member in zf.namelist():
            if member.lower().endswith(".csv") and not member.startswith("__MACOSX/"):
                basename = Path(member).name.lower()
                try:
                    with zf.open(member) as f:
                        data[basename] = pd.read_csv(f)
                except Exception as exc:
                    data[basename] = pd.DataFrame()
                    st.warning(f"Could not read {basename}: {exc}")
    return data


def as_numeric(df: pd.DataFrame, col: str) -> pd.Series | None:
    if col not in df.columns:
        return None
    return pd.to_numeric(df[col], errors="coerce")


def mean_metric(df: pd.DataFrame | None, col: str) -> float:
    if df is None or col not in df.columns:
        return np.nan
    values = pd.to_numeric(df[col], errors="coerce").dropna()
    return float(values.mean()) if not values.empty else np.nan


def get_job_id_col(df: pd.DataFrame) -> str | None:
    for col in ("Job_ID", "Process_ID", "Job_ID ", "ProcessID", "JobID"):
        if col in df.columns:
            return col
    return None


def get_prediction_col(df: pd.DataFrame) -> str | None:
    for col in ("Predicted_Execution_Time", "Predicted_Burst_Time", "Burst_Time", "Execution_Time(Seconds)"):
        if col in df.columns:
            return col
    return None


def add_metric_row(name: str, df: pd.DataFrame) -> dict:
    return {
        "Algorithm": name,
        "Jobs": int(len(df)),
        "Average Waiting Time": mean_metric(df, "Waiting_Time"),
        "Average Turnaround Time": mean_metric(df, "Turnaround_Time"),
        "Average Response Time": mean_metric(df, "Response_Time"),
    }


st.sidebar.header("Dashboard data")
uploaded = st.sidebar.file_uploader("Upload dashboard_data.zip", type=["zip"])
if uploaded is None:
    st.info("Upload your latest `dashboard_data.zip` in the sidebar to open the project dashboard.")
    st.markdown(
        "**ZIP should ideally include:** `epcs_results.csv`, `fcfs_results.csv`, "
        "`sjf_results_713.csv`, `priority_results.csv`, `rr_results.csv`, "
        "`scheduler_workload.csv`, and `shap_feature_importance.csv`."
    )
    st.stop()

try:
    files = load_zip_csvs(uploaded.getvalue())
except zipfile.BadZipFile:
    st.error("The uploaded file is not a valid ZIP. Re-download dashboard_data.zip from Kaggle and try again.")
    st.stop()

st.success(f"Loaded {len(files)} CSV file(s) successfully.")
st.sidebar.caption("Files detected")
for filename in sorted(files):
    st.sidebar.write(f"✓ {filename}")

# Filename lookup: exact preferred, then substring fallback.
def get_df(exact: str, fallback: str | None = None) -> pd.DataFrame | None:
    if exact.lower() in files:
        return files[exact.lower()]
    if fallback:
        for name, frame in files.items():
            if fallback.lower() in name:
                return frame
    return None


epcs = get_df("epcs_results.csv", "epcs_results")
fcfs = get_df("fcfs_results.csv", "fcfs_results")
sjf = get_df("sjf_results_713.csv", "sjf_results")
priority = get_df("priority_results.csv", "priority_results")
rr = get_df("rr_results.csv", "rr_results")
workload = get_df("scheduler_workload.csv", "scheduler_workload")
shap_df = get_df("shap_feature_importance.csv", "shap_feature_importance")
local_shap_df = get_df("shap_local_explanations.csv", "shap_local_explanations")

schedulers = {
    "EPCS": epcs,
    "FCFS": fcfs,
    "SJF": sjf,
    "Priority": priority,
    "Round Robin": rr,
}

# ---------- 1. Workload overview ----------
st.header("1. Workload Overview")
job_count = len(workload) if workload is not None else (len(epcs) if epcs is not None else 0)
available_schedulers = sum(df is not None and not df.empty for df in schedulers.values())
epcs_wait = mean_metric(epcs, "Waiting_Time")
epcs_tat = mean_metric(epcs, "Turnaround_Time")
cols = st.columns(4)
cols[0].metric("Jobs in Workload", f"{job_count:,}" if job_count else "Not loaded")
cols[1].metric("Algorithms Available", f"{available_schedulers}/5")
cols[2].metric("EPCS Avg Waiting Time", f"{epcs_wait:.2f}" if np.isfinite(epcs_wait) else "N/A")
cols[3].metric("EPCS Avg Turnaround Time", f"{epcs_tat:.2f}" if np.isfinite(epcs_tat) else "N/A")

missing_algorithms = [name for name, df in schedulers.items() if df is None or df.empty]
if missing_algorithms:
    st.warning("Missing scheduler result(s): " + ", ".join(missing_algorithms) + ". Check the ZIP file names and contents.")

# ---------- 2. XGBoost metrics ----------
st.header("2. XGBoost Model Performance")
st.write("The model predicts job execution time from workload characteristics.")
metric_cols = st.columns(3)
metric_cols[0].metric("MAE", "43.55 seconds")
metric_cols[1].metric("RMSE", "55.55 seconds")
metric_cols[2].metric("R² score", "0.9898")
st.caption(
    "These are the previously reported notebook evaluation values. They are displayed as recorded results; "
    "this dashboard does not load the trained model or recalculate the scores. Keep this note unless you connect "
    "and validate the exported model/evaluation artifacts."
)
with st.expander("What these model metrics mean"):
    st.markdown(
        "- **MAE:** mean absolute difference between predicted and observed execution time.\n"
        "- **RMSE:** square-root of mean squared prediction error; larger errors receive greater weight.\n"
        "- **R²:** proportion of target variance explained on the evaluated test set. A high score should still be checked "
        "with chronological validation and leakage checks for workload-trace data."
    )

# ---------- 3. Scheduling comparison ----------
st.header("3. Scheduling Algorithm Comparison")
rows = [add_metric_row(name, df) for name, df in schedulers.items() if df is not None and not df.empty]
if rows:
    comparison = pd.DataFrame(rows)
    st.dataframe(comparison.round(3), use_container_width=True, hide_index=True)

    available_metrics = [
        c for c in ("Average Waiting Time", "Average Turnaround Time", "Average Response Time")
        if c in comparison.columns and comparison[c].notna().any()
    ]
    if available_metrics:
        metric = st.selectbox("Select performance metric", available_metrics)
        chart_data = comparison.dropna(subset=[metric])
        fig = px.bar(
            chart_data,
            x="Algorithm",
            y=metric,
            color="Algorithm",
            text=chart_data[metric].map(lambda x: f"{x:.2f}"),
            title=f"Algorithm Comparison: {metric}",
            template="plotly_white",
        )
        fig.update_layout(showlegend=False, yaxis_title="Average time (same unit as input data)")
        st.plotly_chart(fig, use_container_width=True)

        all_long = comparison.melt(
            id_vars=["Algorithm"], value_vars=available_metrics,
            var_name="Metric", value_name="Average time"
        ).dropna(subset=["Average time"])
        fig_all = px.bar(
            all_long, x="Algorithm", y="Average time", color="Metric",
            barmode="group", title="All Scheduling Metrics", template="plotly_white"
        )
        st.plotly_chart(fig_all, use_container_width=True)

    if len(comparison) > 1 and comparison["Jobs"].nunique() > 1:
        st.warning(
            "The algorithms have different job counts. Do not claim a fair performance comparison until all algorithms "
            "are run on the same workload and use consistent arrival/burst/priority assumptions."
        )
    if epcs is not None and sjf is not None:
        same_wait = np.isclose(mean_metric(epcs, "Waiting_Time"), mean_metric(sjf, "Waiting_Time"), equal_nan=True)
        same_tat = np.isclose(mean_metric(epcs, "Turnaround_Time"), mean_metric(sjf, "Turnaround_Time"), equal_nan=True)
        if same_wait and same_tat:
            st.warning(
                "EPCS and SJF currently have identical average waiting and turnaround times to the stored precision. "
                "This can happen, but inspect the selection/order logic and job-by-job schedules before claiming EPCS improves over SJF."
            )
else:
    st.warning("No scheduler result CSVs were found. Check the ZIP and the expected filenames.")

# ---------- 4. Job explorer ----------
st.header("4. Job Explorer")
choices = {name: df for name, df in schedulers.items() if df is not None and not df.empty}
if choices:
    selected_name = st.selectbox("Select scheduling algorithm", list(choices.keys()), key="job_explorer_choice")
    selected_df = choices[selected_name].copy()
    st.write(f"Showing {len(selected_df):,} jobs for {selected_name}.")
    id_col = get_job_id_col(selected_df)
    if id_col:
        search_text = st.text_input("Search by Job ID / Process ID")
        if search_text.strip():
            selected_df = selected_df[
                selected_df[id_col].astype(str).str.contains(search_text.strip(), case=False, na=False)
            ]
    st.dataframe(selected_df.head(500), use_container_width=True, hide_index=True)
    st.download_button(
        "Download displayed job results",
        data=selected_df.to_csv(index=False).encode("utf-8"),
        file_name=f"{selected_name.lower().replace(' ', '_')}_displayed_results.csv",
        mime="text/csv",
    )
else:
    st.info("Job-level scheduler results are not available in this ZIP.")

# ---------- 5. Scheduling timeline (numeric time; avoids 1970 date interpretation) ----------
st.header("5. Scheduling Timeline")
timeline_choices = {name: df for name, df in schedulers.items() if df is not None and not df.empty}
if timeline_choices:
    timeline_name = st.selectbox("Select algorithm for timeline", list(timeline_choices.keys()), key="timeline_choice")
    timeline_df = timeline_choices[timeline_name].copy()
    id_col = get_job_id_col(timeline_df)
    start_col = "Start_Time" if "Start_Time" in timeline_df.columns else None
    end_col = "Completion_Time" if "Completion_Time" in timeline_df.columns else None
    if id_col and start_col and end_col:
        timeline_df[start_col] = pd.to_numeric(timeline_df[start_col], errors="coerce")
        timeline_df[end_col] = pd.to_numeric(timeline_df[end_col], errors="coerce")
        timeline_df = timeline_df.dropna(subset=[start_col, end_col]).sort_values(start_col).head(100).copy()
        timeline_df["Duration"] = timeline_df[end_col] - timeline_df[start_col]
        timeline_df["Job label"] = timeline_df[id_col].astype(str)
        timeline_df = timeline_df[timeline_df["Duration"] >= 0]
        if not timeline_df.empty:
            fig_timeline = go.Figure()
            fig_timeline.add_trace(go.Bar(
                x=timeline_df["Duration"],
                y=timeline_df["Job label"],
                base=timeline_df[start_col],
                orientation="h",
                name=timeline_name,
                hovertemplate="Job=%{y}<br>Start=%{base}<br>Duration=%{x}<extra></extra>",
            ))
            fig_timeline.update_layout(
                title=f"{timeline_name} Scheduling Timeline — First 100 Jobs",
                xaxis_title="Time (numeric units from workload)",
                yaxis_title="Job",
                template="plotly_white",
                height=700,
            )
            fig_timeline.update_xaxes(type="linear")
            fig_timeline.update_yaxes(autorange="reversed")
            st.plotly_chart(fig_timeline, use_container_width=True)
        else:
            st.warning("No valid start/completion times were available for the timeline.")
    else:
        st.warning("Timeline needs job ID, Start_Time and Completion_Time columns.")

# ---------- 6. Workload analysis ----------
st.header("6. Workload Data Analysis")
if workload is not None and not workload.empty:
    st.subheader("Dataset Preview")
    st.dataframe(workload.head(20), use_container_width=True, hide_index=True)
    numeric_cols = workload.select_dtypes(include=np.number).columns.tolist()
    if numeric_cols:
        feat = st.selectbox("Select a numeric workload feature", numeric_cols, key="workload_feature")
        values = pd.to_numeric(workload[feat], errors="coerce").dropna()
        fig_hist = px.histogram(values, x=feat, nbins=30, title=f"Distribution of {feat}", template="plotly_white")
        st.plotly_chart(fig_hist, use_container_width=True)
        fig_box = px.box(values, y=feat, title=f"Spread of {feat}", template="plotly_white")
        st.plotly_chart(fig_box, use_container_width=True)
else:
    st.warning("scheduler_workload.csv was not found in this ZIP.")

# ---------- 7. Global and local explainability ----------
st.header("7. Explainable AI — SHAP")
st.write(
    "Global SHAP importance summarizes how strongly transformed input features contribute to model predictions on average. "
    "It does not, by itself, prove why a particular job was selected at one scheduling decision."
)
if shap_df is not None and not shap_df.empty:
    required = {"Feature", "Mean_SHAP_Importance"}
    if required.issubset(shap_df.columns):
        shap_plot = shap_df.copy()
        shap_plot["Mean_SHAP_Importance"] = pd.to_numeric(shap_plot["Mean_SHAP_Importance"], errors="coerce")
        shap_plot = shap_plot.dropna(subset=["Mean_SHAP_Importance"]).sort_values("Mean_SHAP_Importance", ascending=False)
        st.subheader("Global Feature Importance")
        st.dataframe(shap_plot.round(5), use_container_width=True, hide_index=True)
        fig_shap = px.bar(
            shap_plot.sort_values("Mean_SHAP_Importance"),
            x="Mean_SHAP_Importance", y="Feature", orientation="h",
            title="Mean Absolute SHAP Importance", template="plotly_white",
            labels={"Mean_SHAP_Importance": "Mean |SHAP value|", "Feature": "Encoded feature"},
        )
        st.plotly_chart(fig_shap, use_container_width=True)
        if not shap_plot.empty:
            top = shap_plot.iloc[0]
            st.success(f"Highest global importance in this exported table: {top['Feature']}.")
        st.caption("Absolute SHAP importance shows contribution magnitude, not whether a feature pushes a prediction up or down.")
    else:
        st.error("SHAP CSV must include Feature and Mean_SHAP_Importance columns.")
else:
    st.warning("shap_feature_importance.csv is missing. Export it from the Kaggle notebook and add it to the ZIP.")

st.subheader("Per-job explanation")
if local_shap_df is not None and not local_shap_df.empty:
    st.success("A local SHAP explanation file was found.")
    st.dataframe(local_shap_df.head(200), use_container_width=True, hide_index=True)
else:
    st.warning(
        "Per-job SHAP explanations are not yet included. The ZIP currently contains global feature importance only. "
        "To satisfy the 'why was this job selected?' objective fully, export local SHAP contributions joined to each job ID "
        "and record the candidate jobs and scheduler decision at each dispatch time."
    )

# ---------- 8. Integrity / scope notes ----------
st.header("8. Validation Notes")
if workload is not None and len(workload) and epcs is not None:
    st.write(f"Workload rows: **{len(workload)}**; EPCS rows: **{len(epcs)}**.")
    if len(workload) != len(epcs):
        st.warning("Workload and EPCS row counts differ. Check whether jobs were filtered or omitted.")
else:
    st.write("Workload/EPCS row-count validation could not run because one of those files is missing.")

st.markdown(
    """
    **Research/implementation caveats**
    - Use the same jobs, arrival times, burst-time definition and units for every baseline.
    - Confirm whether priority number 1 is the highest priority and document Round Robin quantum.
    - Evaluate prediction with a chronological holdout in addition to a random split where time order matters.
    - Do not claim EPCS is better solely from a dashboard; show repeated-run statistics or confidence intervals where appropriate.
    - This prototype visualizes exported CSVs. Live prediction requires exporting/loading the fitted preprocessing pipeline and trained model.
    """
)
st.divider()
st.caption("EPCS | Explainable AI CPU Scheduler | XGBoost + SHAP | Streamlit dashboard")
