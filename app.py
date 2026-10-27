import hashlib
import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# -----------------------------------------------------------------------------
# EPCS Decision Intelligence Dashboard
# Data-driven descriptions are generated from uploaded CSVs. No live model is
# loaded or retrained by this dashboard.
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="EPCS | Decision Intelligence",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .stApp { background: linear-gradient(180deg, #081426 0%, #0b1b31 48%, #0b1220 100%); color: #edf4ff; }
      [data-testid="stHeader"] { background: rgba(0,0,0,0); }
      [data-testid="stSidebar"] { background: #0b1526; border-right: 1px solid #243754; }
      h1, h2, h3 { color: #eff6ff !important; letter-spacing: -0.02em; }
      p, label, li, [data-testid="stMarkdownContainer"] { color: #d7e4f7; }
      [data-testid="stMetric"] { background: linear-gradient(135deg, #122742, #142d46); padding: 16px 18px; border: 1px solid #2a496b; border-radius: 15px; }
      [data-testid="stMetricLabel"] { color: #a9c4e5 !important; }
      [data-testid="stMetricValue"] { color: #f2f8ff !important; }
      div[data-testid="stExpander"] { background: #101f34; border: 1px solid #2a415f; border-radius: 12px; }
      div[data-testid="stAlert"] { border-radius: 12px; }
      .hero { padding: 24px 26px; background: linear-gradient(110deg, #12335a, #0d253d 58%, #123a47); border: 1px solid #315b7e; border-radius: 20px; margin-bottom: 14px; }
      .hero-title { font-size: 2rem; font-weight: 800; margin: 0; color: #f1f7ff; }
      .hero-sub { color: #bed5ef; margin-top: 7px; font-size: 1rem; }
      .tag { display:inline-block; background:#123c50; color:#9ee7e0; padding:5px 10px; margin-right:6px; border-radius:999px; font-size:0.78rem; border:1px solid #266475; }
      .insight { background:#10243a; border-left:4px solid #38c8c2; border-radius:10px; padding:12px 15px; margin:8px 0; }
      .muted { color:#9db3cf; font-size:0.87rem; }
      .novelty-card { min-height:170px; background:linear-gradient(145deg,#142842,#102237); border:1px solid #2c4e6d; border-radius:16px; padding:18px; }
      .novelty-title { font-weight:750; font-size:1.05rem; color:#e8f5ff; margin-bottom:8px; }
      .novelty-body { color:#bdd0e5; font-size:0.91rem; }
      hr { border-color:#263c58 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <div><span class="tag">XGBoost</span><span class="tag">SHAP</span><span class="tag">Decision Intelligence</span></div>
      <p class="hero-title">⚙️ EPCS — Explainable AI CPU Scheduler</p>
      <div class="hero-sub">Workload intelligence · Counterfactual comparison · Reliability guard · Dataset shift radar</div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.caption(
    "This dashboard analyses exported CSV results. It does not retrain XGBoost or silently run a new scheduler. "
    "Every insight below is generated from the uploaded data and is labelled when a feature cannot be estimated."
)

ACCENT = "#32d3c7"
PALETTE = ["#37d6cd", "#60a5fa", "#fbbf24", "#f472b6", "#a78bfa", "#fb923c"]
PLOT_TEMPLATE = "plotly_dark"


@st.cache_data(show_spinner=False)
def load_zip_csvs(zip_bytes: bytes) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zf:
        for member in zf.namelist():
            if member.lower().endswith(".csv") and not member.startswith("__MACOSX/"):
                name = Path(member).name.lower()
                try:
                    with zf.open(member) as f:
                        out[name] = pd.read_csv(f)
                except Exception:
                    out[name] = pd.DataFrame()
    return out


def numeric_series(df: pd.DataFrame | None, col: str) -> pd.Series:
    if df is None or col not in df.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(df[col], errors="coerce").dropna()


def mean_metric(df: pd.DataFrame | None, col: str) -> float:
    s = numeric_series(df, col)
    return float(s.mean()) if not s.empty else np.nan


def find_col(df: pd.DataFrame | None, candidates: tuple[str, ...]) -> str | None:
    if df is None:
        return None
    lookup = {str(c).strip().lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lookup:
            return lookup[cand.lower()]
    return None


def job_key(df: pd.DataFrame | None) -> str | None:
    return find_col(df, ("Job_ID", "Process_ID", "Job_ID ", "ProcessID", "JobID", "Process"))


def metric_row(name: str, df: pd.DataFrame) -> dict:
    return {
        "Algorithm": name,
        "Jobs": int(len(df)),
        "Average Waiting Time": mean_metric(df, "Waiting_Time"),
        "Average Turnaround Time": mean_metric(df, "Turnaround_Time"),
        "Average Response Time": mean_metric(df, "Response_Time"),
    }


def fmt(value: float, unit: str = "") -> str:
    return f"{value:,.2f}{unit}" if pd.notna(value) and np.isfinite(value) else "N/A"


def percent_better(epcs_value: float, comparison_value: float) -> float:
    if not (np.isfinite(epcs_value) and np.isfinite(comparison_value)) or comparison_value == 0:
        return np.nan
    return 100.0 * (comparison_value - epcs_value) / abs(comparison_value)


def metric_key_insights(comparison: pd.DataFrame, metric_names: list[str]) -> list[str]:
    notes: list[str] = []
    if comparison.empty or "EPCS" not in comparison["Algorithm"].values:
        return ["EPCS results are missing, so EPCS-specific conclusions cannot be generated."]
    epcs_row = comparison.loc[comparison["Algorithm"] == "EPCS"].iloc[0]
    for metric in metric_names:
        valid = comparison.dropna(subset=[metric]).sort_values(metric)
        if valid.empty or not np.isfinite(epcs_row[metric]):
            continue
        best = valid.iloc[0]
        if str(best["Algorithm"]) == "EPCS":
            runners = valid[valid["Algorithm"] != "EPCS"]
            if runners.empty:
                notes.append(f"{metric}: EPCS has a recorded value of {epcs_row[metric]:.2f}; no baseline is available to compare.")
            else:
                next_best = float(runners.iloc[0][metric])
                gain = percent_better(float(epcs_row[metric]), next_best)
                if np.isclose(float(epcs_row[metric]), next_best, rtol=1e-5, atol=1e-8):
                    notes.append(f"{metric}: EPCS is tied with {runners.iloc[0]['Algorithm']} at {epcs_row[metric]:.2f} (within numeric tolerance).")
                else:
                    notes.append(f"{metric}: EPCS has the lowest observed mean ({epcs_row[metric]:.2f}); it is {gain:.1f}% lower than the next-best baseline, {runners.iloc[0]['Algorithm']} ({next_best:.2f}).")
        else:
            best_value = float(best[metric])
            epcs_value = float(epcs_row[metric])
            gap = percent_better(epcs_value, best_value)
            notes.append(f"{metric}: {best['Algorithm']} is lowest ({best_value:.2f}); EPCS is {abs(gap):.1f}% {'higher' if gap < 0 else 'lower'} ({epcs_value:.2f}).")
    return notes or ["Not enough comparable numeric metrics were found to generate performance notes."]


def numeric_profile(df: pd.DataFrame | None) -> dict:
    if df is None or df.empty:
        return {}
    nums = df.select_dtypes(include=np.number)
    return {str(c): nums[c].dropna().copy() for c in nums.columns if nums[c].dropna().size > 0}


def categorical_profile(df: pd.DataFrame | None, max_categories: int = 20) -> dict:
    if df is None or df.empty:
        return {}
    out = {}
    for c in df.columns:
        if df[c].dtype == "object" or str(df[c].dtype).startswith("category"):
            s = df[c].dropna().astype(str)
            if 1 < s.nunique() <= max_categories:
                out[str(c)] = s.value_counts(normalize=True)
    return out


def make_shift_tables(current: pd.DataFrame, previous: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    numeric_rows = []
    categorical_rows = []
    shared_numeric = [c for c in current.select_dtypes(include=np.number).columns if c in previous.columns and pd.api.types.is_numeric_dtype(previous[c])]
    for col in shared_numeric:
        a = pd.to_numeric(previous[col], errors="coerce").dropna()
        b = pd.to_numeric(current[col], errors="coerce").dropna()
        if len(a) < 2 or len(b) < 2:
            continue
        denom = np.sqrt((a.std(ddof=1) ** 2 + b.std(ddof=1) ** 2) / 2.0)
        smd = (b.mean() - a.mean()) / denom if denom and np.isfinite(denom) else 0.0
        numeric_rows.append({
            "Feature": col,
            "Previous mean": float(a.mean()),
            "Current mean": float(b.mean()),
            "Standardized mean shift": float(smd),
            "Absolute shift": float(abs(smd)),
            "Signal": "Large shift" if abs(smd) >= 0.5 else ("Moderate shift" if abs(smd) >= 0.25 else "Small shift"),
        })
    shared_cats = [c for c in current.columns if c in previous.columns and (current[c].dtype == "object" or str(current[c].dtype).startswith("category"))]
    for col in shared_cats:
        a = previous[col].dropna().astype(str).value_counts(normalize=True)
        b = current[col].dropna().astype(str).value_counts(normalize=True)
        cats = set(a.index) | set(b.index)
        if not cats or len(cats) > 50:
            continue
        tvd = 0.5 * sum(abs(float(a.get(k, 0.0)) - float(b.get(k, 0.0))) for k in cats)
        categorical_rows.append({"Feature": col, "Share distribution change": float(tvd), "Change (percentage points)": float(100 * tvd)})
    return pd.DataFrame(numeric_rows), pd.DataFrame(categorical_rows)


# Sidebar: uploading data
st.sidebar.markdown("## 📦 Dataset input")
uploaded = st.sidebar.file_uploader("Upload dashboard_data.zip", type=["zip"])
st.sidebar.markdown("---")
st.sidebar.markdown("### Expected files")
st.sidebar.caption("epcs_results.csv · fcfs_results.csv · sjf_results_713.csv · priority_results.csv · rr_results.csv · scheduler_workload.csv · shap_feature_importance.csv")
if uploaded is None:
    st.info("Upload the exported `dashboard_data.zip` from Kaggle in the sidebar to open all analysis panels.")
    st.markdown("#### New in Decision Intelligence")
    st.markdown("- **Counterfactual Job Mirror:** compare a job's observed outcomes under available algorithms.\n- **Prediction Reliability Guard:** examine retrospective prediction error when actual burst times are included.\n- **Workload Shift Radar:** compare this upload with the previous ZIP in the same browser session.")
    st.stop()

zip_bytes = uploaded.getvalue()
dataset_hash = hashlib.sha256(zip_bytes).hexdigest()
try:
    files = load_zip_csvs(zip_bytes)
except zipfile.BadZipFile:
    st.error("This is not a valid ZIP. Re-export `dashboard_data.zip` from Kaggle and upload it again.")
    st.stop()


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
schedulers = {"EPCS": epcs, "FCFS": fcfs, "SJF": sjf, "Priority": priority, "Round Robin": rr}
choices = {name: df for name, df in schedulers.items() if df is not None and not df.empty}

# Track a previous upload for a within-session workload-shift comparison.
if "epcs_previous_workload" not in st.session_state:
    st.session_state.epcs_previous_workload = None
    st.session_state.epcs_previous_name = None
    st.session_state.epcs_current_hash = dataset_hash
    st.session_state.epcs_current_workload = workload.copy() if workload is not None else None
    st.session_state.epcs_current_name = uploaded.name
elif st.session_state.epcs_current_hash != dataset_hash:
    st.session_state.epcs_previous_workload = (
        st.session_state.epcs_current_workload.copy()
        if st.session_state.epcs_current_workload is not None else None
    )
    st.session_state.epcs_previous_name = st.session_state.epcs_current_name
    st.session_state.epcs_current_hash = dataset_hash
    st.session_state.epcs_current_workload = workload.copy() if workload is not None else None
    st.session_state.epcs_current_name = uploaded.name

st.sidebar.success(f"Loaded {len(files)} CSV file(s)")
with st.sidebar.expander("Files detected", expanded=False):
    for filename in sorted(files):
        st.write(f"✓ {filename}")

# ------------------ 1. Dataset overview ------------------
st.markdown("## 01 · Workload command centre")
job_count = len(workload) if workload is not None else (len(epcs) if epcs is not None else 0)
available_n = len(choices)
epcs_wait = mean_metric(epcs, "Waiting_Time")
epcs_tat = mean_metric(epcs, "Turnaround_Time")
epcs_resp = mean_metric(epcs, "Response_Time")
k1, k2, k3, k4 = st.columns(4)
k1.metric("Jobs in workload", f"{job_count:,}" if job_count else "N/A")
k2.metric("Algorithms loaded", f"{available_n}/5")
k3.metric("EPCS mean waiting", fmt(epcs_wait))
k4.metric("EPCS mean turnaround", fmt(epcs_tat))
st.caption(f"Uploaded file: **{uploaded.name}** · CSV tables detected: **{len(files)}** · Dataset fingerprint: `{dataset_hash[:10]}`")

# Innovation cards: project contribution framing, not universal first-ever novelty claims.
st.markdown("## 02 · Three proposed innovation modules")
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("<div class='novelty-card'><div class='novelty-title'>🔁 1. Counterfactual Job Mirror</div><div class='novelty-body'>Select one job and compare its observed waiting, turnaround and response outcomes across schedulers. Highlights the best observed alternative for each metric.</div></div>", unsafe_allow_html=True)
with c2:
    st.markdown("<div class='novelty-card'><div class='novelty-title'>🛡️ 2. Prediction Reliability Guard</div><div class='novelty-body'>When recorded burst times are available, measures retrospective prediction error, flags high-error cases and warns that this is not a calibrated confidence probability.</div></div>", unsafe_allow_html=True)
with c3:
    st.markdown("<div class='novelty-card'><div class='novelty-title'>🛰️ 3. Workload Shift Radar</div><div class='novelty-body'>Compares numerical and categorical workload distributions with the previous ZIP uploaded in this same session. Spots potential changes that may deserve model re-validation.</div></div>", unsafe_allow_html=True)
st.caption("Novelty framing: these are proposed EPCS-specific integrations. Explainable/counterfactual scheduling, uncertainty-aware prediction, and drift monitoring already exist in research; do not claim the individual concepts are globally unprecedented without a literature and patent search.")

# ------------------ 3. XGBoost metrics ------------------
st.markdown("## 03 · XGBoost model scorecard")
st.write("The dashboard shows previously recorded notebook metrics; it does not load the fitted model or recalculate these scores.")
m1, m2, m3 = st.columns(3)
m1.metric("Reported MAE", "43.55 s")
m2.metric("Reported RMSE", "55.55 s")
m3.metric("Reported R²", "0.9898")
with st.expander("Metric definitions & validation caution"):
    st.markdown("- **MAE:** average absolute prediction error.\n- **RMSE:** gives larger errors extra weight.\n- **R²:** fit score on the evaluated test set.\n- These historic values should be re-verified with a leakage-safe chronological holdout before using them as evidence of production performance.")

# ------------------ 4. Scheduler comparison & dynamic key points ------------------
st.markdown("## 04 · Scheduler performance arena")
rows = [metric_row(name, df) for name, df in schedulers.items() if df is not None and not df.empty]
comparison = pd.DataFrame(rows)
metrics = ["Average Waiting Time", "Average Turnaround Time", "Average Response Time"]
if not comparison.empty:
    st.dataframe(comparison.round(3), use_container_width=True, hide_index=True)
    usable_metrics = [m for m in metrics if m in comparison.columns and comparison[m].notna().any()]
    a, b = st.columns(2)
    if usable_metrics:
        with a:
            metric_pick = st.selectbox("Metric focus", usable_metrics, key="metric_focus")
            plot_df = comparison.dropna(subset=[metric_pick]).sort_values(metric_pick)
            fig = px.bar(plot_df, x="Algorithm", y=metric_pick, color="Algorithm", text=plot_df[metric_pick].map(lambda x: f"{x:.2f}"), title=f"Lowest is better · {metric_pick}", template=PLOT_TEMPLATE, color_discrete_sequence=PALETTE)
            fig.update_layout(showlegend=False, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", yaxis_title="Mean time (input units)")
            st.plotly_chart(fig, use_container_width=True)
        with b:
            melt = comparison.melt(id_vars=["Algorithm"], value_vars=usable_metrics, var_name="Metric", value_name="Mean time").dropna(subset=["Mean time"])
            fig2 = px.bar(melt, x="Algorithm", y="Mean time", color="Metric", barmode="group", title="All three scheduling metrics", template=PLOT_TEMPLATE, color_discrete_sequence=PALETTE)
            fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig2, use_container_width=True)
    st.markdown("### 🧠 Dataset-specific key notes")
    notes = metric_key_insights(comparison, usable_metrics)
    for note in notes:
        st.markdown(f"<div class='insight'>• {note}</div>", unsafe_allow_html=True)
    if comparison["Jobs"].nunique() > 1:
        st.warning("The algorithms contain different job counts. Treat the comparison as non-equivalent until the same jobs and assumptions are confirmed.")
    if epcs is not None and sjf is not None:
        same_w = np.isclose(mean_metric(epcs, "Waiting_Time"), mean_metric(sjf, "Waiting_Time"), equal_nan=True)
        same_t = np.isclose(mean_metric(epcs, "Turnaround_Time"), mean_metric(sjf, "Turnaround_Time"), equal_nan=True)
        if same_w and same_t:
            st.warning("EPCS and SJF have identical mean waiting and turnaround metrics in this ZIP. Inspect job-by-job order and EPCS policy before claiming a gain over SJF.")
else:
    st.warning("No scheduler result CSVs found. Confirm the ZIP contains the expected result files.")

# ------------------ 5. Counterfactual Job Mirror ------------------
st.markdown("## 05 · Innovation 1 — Counterfactual Job Mirror")
st.caption("This panel compares stored outcomes for the same job across available algorithms. It is a cross-algorithm what-if view, not a re-simulation of the full scheduler after changing a dispatch decision.")
if epcs is not None and not epcs.empty and choices:
    epcs_id = job_key(epcs)
    if epcs_id:
        id_values = epcs[epcs_id].dropna().astype(str).unique().tolist()
        selected_job = st.selectbox("Choose a job", id_values, key="counterfactual_job") if id_values else None
        if selected_job is not None:
            cf_rows = []
            for alg, df in choices.items():
                jid = job_key(df)
                if jid is None:
                    continue
                matched = df[df[jid].astype(str) == str(selected_job)]
                if matched.empty:
                    continue
                row = matched.iloc[0]
                cf_rows.append({
                    "Algorithm": alg,
                    "Waiting Time": pd.to_numeric(row.get("Waiting_Time", np.nan), errors="coerce"),
                    "Turnaround Time": pd.to_numeric(row.get("Turnaround_Time", np.nan), errors="coerce"),
                    "Response Time": pd.to_numeric(row.get("Response_Time", np.nan), errors="coerce"),
                    "Start Time": pd.to_numeric(row.get("Start_Time", np.nan), errors="coerce"),
                    "Completion Time": pd.to_numeric(row.get("Completion_Time", np.nan), errors="coerce"),
                })
            cf = pd.DataFrame(cf_rows)
            if not cf.empty:
                st.dataframe(cf.round(3), use_container_width=True, hide_index=True)
                plot_cf = cf.melt(id_vars="Algorithm", value_vars=["Waiting Time", "Turnaround Time", "Response Time"], var_name="Metric", value_name="Time").dropna(subset=["Time"])
                if not plot_cf.empty:
                    fig_cf = px.bar(plot_cf, x="Algorithm", y="Time", color="Metric", barmode="group", title=f"Observed alternative outcomes for job {selected_job}", template=PLOT_TEMPLATE, color_discrete_sequence=PALETTE)
                    fig_cf.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig_cf, use_container_width=True)
                for m in ["Waiting Time", "Turnaround Time", "Response Time"]:
                    ss = cf.dropna(subset=[m]).sort_values(m)
                    if len(ss):
                        st.markdown(f"<div class='insight'>• Lowest observed <b>{m.lower()}</b> for this job: <b>{ss.iloc[0]['Algorithm']}</b> ({ss.iloc[0][m]:.2f}). This is an observed cross-scheduler result, not a causal counterfactual guarantee.</div>", unsafe_allow_html=True)
            else:
                st.info("This job ID could not be matched across scheduler CSVs. Check whether all result files use the same ID scheme.")
    else:
        st.info("Job-level what-if view needs a Job_ID or Process_ID column in EPCS results.")
else:
    st.info("Upload EPCS and scheduler result CSVs to enable the job-level what-if view.")

# ------------------ 6. Prediction Reliability Guard ------------------
st.markdown("## 06 · Innovation 2 — Prediction Reliability Guard")
st.caption("This guard reviews observed prediction errors only when a predicted execution-time column can be matched to recorded burst/execution time. It does not invent a probability or calibrated confidence score.")
if epcs is not None and workload is not None and not epcs.empty and not workload.empty:
    pred_col = find_col(epcs, ("Predicted_Execution_Time", "Predicted_Burst_Time", "PredictedExecutionTime"))
    actual_col = find_col(workload, ("Burst_Time", "Execution_Time(Seconds)", "Actual_Execution_Time", "Execution_Time"))
    e_id, w_id = job_key(epcs), job_key(workload)
    if pred_col and actual_col and e_id and w_id:
        pred_join = epcs[[e_id, pred_col]].copy()
        pred_join["_join_id"] = pred_join[e_id].astype(str)
        actual_join = workload[[w_id, actual_col]].copy()
        actual_join["_join_id"] = actual_join[w_id].astype(str)
        joined = pred_join.merge(actual_join[["_join_id", actual_col]], on="_join_id", how="inner")
        joined[pred_col] = pd.to_numeric(joined[pred_col], errors="coerce")
        joined[actual_col] = pd.to_numeric(joined[actual_col], errors="coerce")
        joined = joined.dropna(subset=[pred_col, actual_col])
        joined = joined[(joined[pred_col] >= 0) & (joined[actual_col] >= 0)]
        if len(joined) >= 3:
            joined["Signed Error"] = joined[pred_col] - joined[actual_col]
            joined["Absolute Error"] = joined["Signed Error"].abs()
            q90 = float(joined["Absolute Error"].quantile(0.90))
            med = float(joined["Absolute Error"].median())
            err1, err2, err3 = st.columns(3)
            err1.metric("Matched jobs", f"{len(joined):,}")
            err2.metric("Observed MAE", fmt(float(joined["Absolute Error"].mean())))
            err3.metric("90th percentile absolute error", fmt(q90))
            joined["Error band"] = np.where(joined["Absolute Error"] >= q90, "Highest-error 10% (approx.)", "Other jobs")
            fig_err = px.histogram(joined, x="Absolute Error", nbins=30, color="Error band", title="Observed absolute prediction-error distribution", template=PLOT_TEMPLATE, color_discrete_sequence=PALETTE)
            fig_err.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_err, use_container_width=True)
            fig_scatter = px.scatter(joined, x=actual_col, y=pred_col, color="Error band", hover_data=[e_id], title="Predicted vs recorded execution time", labels={actual_col: "Recorded time", pred_col: "Predicted time"}, template=PLOT_TEMPLATE, color_discrete_sequence=PALETTE)
            low = min(float(joined[actual_col].min()), float(joined[pred_col].min()))
            high = max(float(joined[actual_col].max()), float(joined[pred_col].max()))
            fig_scatter.add_shape(type="line", x0=low, y0=low, x1=high, y1=high, line=dict(color="#fbbf24", dash="dash"))
            fig_scatter.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_scatter, use_container_width=True)
            flagged = joined.sort_values("Absolute Error", ascending=False).head(10)
            st.markdown(f"<div class='insight'>• Largest absolute error in the matched jobs: <b>{joined['Absolute Error'].max():.2f}</b>. Median absolute error: <b>{med:.2f}</b>. The highest-error band is descriptive and should be used for review, not interpreted as a calibrated confidence score.</div>", unsafe_allow_html=True)
            st.markdown("**Jobs with largest observed prediction errors**")
            st.dataframe(flagged[[e_id, pred_col, actual_col, "Signed Error", "Absolute Error"]].round(3), use_container_width=True, hide_index=True)
        else:
            st.info("Fewer than 3 matched numeric predictions and recorded times were found; reliability analysis was skipped.")
    else:
        missing_bits = []
        if not pred_col: missing_bits.append("Predicted_Execution_Time in epcs_results.csv")
        if not actual_col: missing_bits.append("Burst_Time or recorded execution-time column in scheduler_workload.csv")
        if not e_id or not w_id: missing_bits.append("matching job ID columns")
        st.warning("Reliability guard unavailable because the ZIP is missing: " + ", ".join(missing_bits) + ". No confidence score is fabricated.")
else:
    st.warning("Reliability guard needs both EPCS results and workload data.")

# ------------------ 7. Workload Shift Radar ------------------
st.markdown("## 07 · Innovation 3 — Workload Shift Radar")
st.caption("Compares this upload with the previous different ZIP uploaded in the same browser session. The baseline is session-only and resets if the app session restarts.")
current_workload = workload
previous_workload = st.session_state.get("epcs_previous_workload")
if current_workload is not None and previous_workload is not None and not current_workload.empty and not previous_workload.empty:
    st.write(f"Current dataset: **{uploaded.name}** · Baseline dataset: **{st.session_state.get('epcs_previous_name') or 'previous upload'}**")
    num_shift, cat_shift = make_shift_tables(current_workload, previous_workload)
    s1, s2, s3 = st.columns(3)
    if not num_shift.empty:
        s1.metric("Numeric features compared", f"{len(num_shift)}")
        s2.metric("Moderate/large numeric shifts", f"{int((num_shift['Absolute shift'] >= 0.25).sum())}")
        s3.metric("Largest absolute shift", fmt(float(num_shift['Absolute shift'].max()), " SMD"))
        fig_shift = px.bar(num_shift.sort_values("Absolute shift"), x="Absolute shift", y="Feature", orientation="h", color="Signal", title="Numeric workload shift (standardized mean difference)", template=PLOT_TEMPLATE, color_discrete_sequence=PALETTE)
        fig_shift.add_vline(x=0.25, line_dash="dash", line_color="#fbbf24", annotation_text="moderate signal")
        fig_shift.add_vline(x=0.5, line_dash="dot", line_color="#fb7185", annotation_text="large signal")
        fig_shift.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_shift, use_container_width=True)
        st.dataframe(num_shift.sort_values("Absolute shift", ascending=False).round(3), use_container_width=True, hide_index=True)
        top_shift = num_shift.sort_values("Absolute shift", ascending=False).iloc[0]
        st.markdown(f"<div class='insight'>• Largest numeric distribution change: <b>{top_shift['Feature']}</b> (standardized shift {top_shift['Standardized mean shift']:.2f}; {top_shift['Signal'].lower()}). This flags a possible workload shift; it does not prove model degradation.</div>", unsafe_allow_html=True)
    else:
        st.info("No shared numeric features with enough valid values were available for comparison.")
    if not cat_shift.empty:
        fig_cat = px.bar(cat_shift.sort_values("Share distribution change"), x="Share distribution change", y="Feature", orientation="h", title="Categorical mix change (total variation distance; 0–1)", template=PLOT_TEMPLATE, color_discrete_sequence=[ACCENT])
        fig_cat.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_cat, use_container_width=True)
        st.dataframe(cat_shift.sort_values("Share distribution change", ascending=False).round(3), use_container_width=True, hide_index=True)
    else:
        st.info("No shared categorical workload columns were available for category-mix comparison.")
else:
    st.info("Upload a second, different dashboard ZIP in this same browser session to activate shift comparison. The previous dataset is not permanently stored.")
    if current_workload is not None and not current_workload.empty:
        # Still provide a within-upload profile while no previous dataset exists.
        numeric_cols = current_workload.select_dtypes(include=np.number).columns.tolist()
        if numeric_cols:
            selected_profile_col = st.selectbox("Inspect a workload feature while waiting for a second upload", numeric_cols, key="shift_first_profile")
            profile_values = pd.to_numeric(current_workload[selected_profile_col], errors="coerce").dropna()
            fig_profile = px.histogram(profile_values.to_frame(name=selected_profile_col), x=selected_profile_col, nbins=30, title=f"Current workload profile: {selected_profile_col}", template=PLOT_TEMPLATE, color_discrete_sequence=[ACCENT])
            fig_profile.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_profile, use_container_width=True)

# ------------------ 8. Workload analysis ------------------
st.markdown("## 08 · Workload fingerprint")
if workload is not None and not workload.empty:
    st.dataframe(workload.head(15), use_container_width=True, hide_index=True)
    ncols = workload.select_dtypes(include=np.number).columns.tolist()
    if ncols:
        feature = st.selectbox("Choose a numeric workload feature", ncols, key="workload_hist_feat")
        values = pd.to_numeric(workload[feature], errors="coerce").dropna()
        left, right = st.columns(2)
        with left:
            f_hist = px.histogram(values, x=feature, nbins=30, title=f"Distribution: {feature}", template=PLOT_TEMPLATE, color_discrete_sequence=[ACCENT])
            f_hist.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(f_hist, use_container_width=True)
        with right:
            f_box = px.box(values.to_frame(name=feature), y=feature, points="outliers", title=f"Spread and outliers: {feature}", template=PLOT_TEMPLATE, color_discrete_sequence=["#60a5fa"])
            f_box.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(f_box, use_container_width=True)
    cats = [c for c in workload.columns if (workload[c].dtype == "object" or str(workload[c].dtype).startswith("category")) and 1 < workload[c].nunique(dropna=True) <= 30]
    if cats:
        cat_col = st.selectbox("Choose a categorical workload feature", cats, key="workload_cat_feat")
        counts = workload[cat_col].fillna("Missing").astype(str).value_counts().rename_axis(cat_col).reset_index(name="Jobs")
        fig_cat = px.bar(counts, x=cat_col, y="Jobs", color=cat_col, title=f"Workload composition: {cat_col}", template=PLOT_TEMPLATE, color_discrete_sequence=PALETTE)
        fig_cat.update_layout(showlegend=False, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_cat, use_container_width=True)
else:
    st.warning("scheduler_workload.csv is missing from this ZIP.")

# ------------------ 9. SHAP explainability ------------------
st.markdown("## 09 · Explainable AI — SHAP")
st.caption("Global SHAP values describe feature contribution magnitude across evaluated observations; they do not, by themselves, explain a specific scheduling decision.")
if shap_df is not None and not shap_df.empty:
    feature_col = find_col(shap_df, ("Feature", "feature"))
    importance_col = find_col(shap_df, ("Mean_SHAP_Importance", "Mean SHAP Importance", "Importance"))
    if feature_col and importance_col:
        shap_plot = shap_df.copy()
        shap_plot[importance_col] = pd.to_numeric(shap_plot[importance_col], errors="coerce")
        shap_plot = shap_plot.dropna(subset=[importance_col]).sort_values(importance_col, ascending=False)
        if not shap_plot.empty:
            shap_left, shap_right = st.columns([1, 1.2])
            with shap_left:
                st.dataframe(shap_plot[[feature_col, importance_col]].round(5), use_container_width=True, hide_index=True)
            with shap_right:
                fig_shap = px.bar(shap_plot.sort_values(importance_col), x=importance_col, y=feature_col, orientation="h", title="Global feature importance · mean |SHAP|", template=PLOT_TEMPLATE, color=importance_col, color_continuous_scale=[[0, "#22577a"], [0.5, "#38a3a5"], [1, "#80ed99"]])
                fig_shap.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", yaxis_title="Feature", xaxis_title="Mean absolute SHAP value")
                st.plotly_chart(fig_shap, use_container_width=True)
            top = shap_plot.iloc[0]
            st.markdown(f"<div class='insight'>• Highest global importance in this exported SHAP table: <b>{top[feature_col]}</b>. Absolute importance indicates magnitude, not positive/negative direction.</div>", unsafe_allow_html=True)
    else:
        st.error("The SHAP CSV needs feature-name and importance columns (typically Feature and Mean_SHAP_Importance).")
else:
    st.warning("shap_feature_importance.csv not found. Add the global SHAP export to the ZIP to enable this section.")

if local_shap_df is not None and not local_shap_df.empty:
    st.subheader("Per-job SHAP records")
    st.dataframe(local_shap_df.head(200), use_container_width=True, hide_index=True)
else:
    st.info("Per-job SHAP is not included. To explain why one job was selected by the EPCS policy, export local SHAP contributions and decision/candidate logs from the scheduler.")

# ------------------ 10. Scheduling timeline ------------------
st.markdown("## 10 · Schedule timeline")
if choices:
    timeline_name = st.selectbox("Algorithm", list(choices.keys()), key="timeline_algorithm")
    tdf = choices[timeline_name].copy()
    tid = job_key(tdf)
    start_col = find_col(tdf, ("Start_Time", "Start Time"))
    end_col = find_col(tdf, ("Completion_Time", "Completion Time"))
    if tid and start_col and end_col:
        tdf[start_col] = pd.to_numeric(tdf[start_col], errors="coerce")
        tdf[end_col] = pd.to_numeric(tdf[end_col], errors="coerce")
        tdf = tdf.dropna(subset=[start_col, end_col]).sort_values(start_col).head(100).copy()
        tdf["Duration"] = tdf[end_col] - tdf[start_col]
        tdf["Job label"] = tdf[tid].astype(str)
        tdf = tdf[tdf["Duration"] >= 0]
        if not tdf.empty:
            fig_time = go.Figure(go.Bar(x=tdf["Duration"], y=tdf["Job label"], base=tdf[start_col], orientation="h", marker_color=ACCENT, hovertemplate="Job=%{y}<br>Start=%{base}<br>Duration=%{x}<extra></extra>"))
            fig_time.update_layout(title=f"{timeline_name} schedule · first 100 valid jobs", xaxis_title="Time (input units)", yaxis_title="Job", template=PLOT_TEMPLATE, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=650)
            fig_time.update_xaxes(type="linear")
            fig_time.update_yaxes(autorange="reversed")
            st.plotly_chart(fig_time, use_container_width=True)
        else:
            st.info("No valid start/completion time rows found.")
    else:
        st.info("Timeline requires a job ID, Start_Time and Completion_Time columns.")
else:
    st.info("No scheduler results are available for a timeline.")

# ------------------ 11. Validation & scientific boundary ------------------
st.markdown("## 11 · Validation and research notes")
if workload is not None and epcs is not None:
    st.write(f"Workload rows: **{len(workload):,}** · EPCS result rows: **{len(epcs):,}**")
    if len(workload) != len(epcs):
        st.warning("Workload and EPCS row counts differ. Check whether any jobs were filtered or dropped.")
if epcs is not None and sjf is not None:
    if np.isclose(mean_metric(epcs, "Waiting_Time"), mean_metric(sjf, "Waiting_Time"), equal_nan=True) and np.isclose(mean_metric(epcs, "Turnaround_Time"), mean_metric(sjf, "Turnaround_Time"), equal_nan=True):
        st.warning("EPCS and SJF are tied on mean waiting/turnaround time in these saved results. Verify their job-level orders before claiming EPCS is superior.")

with st.expander("Research contribution ideas and limitations"):
    st.markdown(
        "**Project-specific contributions to test:**\n"
        "1. A job-level comparison view that combines scheduler outcomes and concise metric-based reasons.\n"
        "2. An error-monitoring guard that identifies high-error jobs when predicted and recorded times can be matched.\n"
        "3. A session-based workload-shift monitor that highlights feature-distribution changes across uploads.\n\n"
        "**Limitations:** this dashboard visualizes exported CSVs, not live inference; shift comparison is session-only; error monitoring needs matching ground-truth time; and a job outcome comparison is not a full counterfactual re-simulation. Confirm technical novelty through literature/patent search and ablation experiments before making a novelty claim."
    )
st.divider()
st.caption("EPCS · Explainable AI CPU Scheduler · XGBoost + SHAP · Dataset-driven decision intelligence")
