from __future__ import annotations

import hashlib
import io
import zipfile
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# EPCS — Explainable AI CPU Scheduler
# Created by Mangal Sakthi
# This dashboard analyses exported CSVs; it does not silently retrain a model.
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="EPCS | CPU Scheduler Analysis",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Restrained, student-built academic dashboard style: light paper background,
# navy text, blue/teal accents, modest borders, no heavy AI/neon visual effects.
st.markdown(
    """
    <style>
    :root { --ink:#1e2b38; --muted:#596a78; --line:#dbe3e8; --blue:#245a7a; --teal:#247c78; --paper:#f6f7f5; }
    .stApp { background: #f6f7f5; color: var(--ink); }
    [data-testid="stHeader"] { background: rgba(246,247,245,.96); }
    [data-testid="stSidebar"] { background:#eef2f3; border-right:1px solid #d6e0e5; }
    h1,h2,h3,h4 { color:#1c3445 !important; letter-spacing:-.02em; }
    p, li, label, [data-testid="stMarkdownContainer"] { color:#273845; }
    [data-testid="stMetric"] { background:#ffffff; padding:14px 16px; border:1px solid #dce4e8; border-radius:10px; box-shadow:0 1px 2px rgba(28,52,69,.035); }
    [data-testid="stMetricLabel"] { color:#5c6d79 !important; }
    [data-testid="stMetricValue"] { color:#1f4964 !important; }
    div[data-testid="stExpander"] { background:#ffffff; border:1px solid #dbe3e8; border-radius:9px; }
    .page-head { border-bottom:2px solid #245a7a; padding:5px 0 17px 0; margin-bottom:15px; }
    .eyebrow { text-transform:uppercase; letter-spacing:.12em; font-size:.75rem; font-weight:750; color:#247c78; margin-bottom:6px; }
    .main-title { font-size:2.15rem; font-weight:800; color:#1c3445; line-height:1.12; margin:0; }
    .subtitle { color:#596a78; font-size:1.02rem; margin-top:8px; }
    .byline { margin-top:11px; font-size:.86rem; color:#3e5969; font-weight:650; }
    .section-kicker { color:#247c78; font-size:.76rem; font-weight:800; letter-spacing:.10em; text-transform:uppercase; margin-bottom:3px; }
    .insight { background:#ffffff; border:1px solid #dce4e8; border-left:4px solid #247c78; padding:11px 14px; border-radius:7px; margin:7px 0; line-height:1.5; }
    .insight-warn { border-left-color:#c18b2f; }
    .plain-card { background:#fff; border:1px solid #dbe3e8; border-radius:10px; padding:15px 16px; height:100%; }
    .card-title { font-size:1rem; font-weight:750; color:#1e4862; margin-bottom:7px; }
    .muted { color:#607381; font-size:.88rem; }
    .small-note { font-size:.82rem; color:#647682; }
    .number-chip { display:inline-block; min-width:28px; text-align:center; border-radius:5px; padding:3px 7px; background:#e7f1f4; color:#245a7a; font-weight:800; margin-right:7px; }
    hr { border-color:#dbe3e8 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="page-head">
      <div class="eyebrow">Operating Systems · Workload analysis · Explainable machine learning</div>
      <p class="main-title">⚙️ EPCS — Explainable AI CPU Scheduler</p>
      <div class="subtitle">A dataset-driven study of scheduling performance, prediction behaviour, fairness and workload change.</div>
      <div class="byline">CREATED BY MANGAL SAKTHI</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.info(
    "Reading guide: this dashboard calculates findings from the CSVs you upload. "
    "It will not assume EPCS is best. It will show where EPCS wins, ties or loses on this dataset."
)

COLORS = {
    "EPCS": "#247c78",
    "FCFS": "#447da2",
    "SJF": "#c08a34",
    "Priority": "#8b6aa8",
    "Round Robin": "#c56856",
}
EXTRA_COLORS = ["#247c78", "#447da2", "#c08a34", "#8b6aa8", "#c56856", "#689a79", "#7b8794"]
PLOT_TEMPLATE = "plotly_white"


def normalized_name(value: object) -> str:
    return "".join(ch.lower() for ch in str(value) if ch.isalnum())


def find_col(df: pd.DataFrame | None, aliases: tuple[str, ...] | list[str]) -> str | None:
    if df is None:
        return None
    lookup = {normalized_name(c): c for c in df.columns}
    for alias in aliases:
        key = normalized_name(alias)
        if key in lookup:
            return lookup[key]
    return None


def series_num(df: pd.DataFrame | None, aliases: tuple[str, ...] | list[str]) -> pd.Series:
    col = find_col(df, aliases)
    if col is None or df is None:
        return pd.Series(dtype=float)
    return pd.to_numeric(df[col], errors="coerce").dropna()


def value_num(row: pd.Series, aliases: tuple[str, ...] | list[str]) -> float:
    keys = {normalized_name(c): c for c in row.index}
    for alias in aliases:
        actual = keys.get(normalized_name(alias))
        if actual is not None:
            val = pd.to_numeric(pd.Series([row[actual]]), errors="coerce").iloc[0]
            if pd.notna(val):
                return float(val)
    return np.nan


def fmt(value: float, digits: int = 2, suffix: str = "") -> str:
    return f"{value:,.{digits}f}{suffix}" if pd.notna(value) and np.isfinite(value) else "N/A"


def safe_mean(df: pd.DataFrame | None, aliases: tuple[str, ...] | list[str]) -> float:
    s = series_num(df, aliases)
    return float(s.mean()) if len(s) else np.nan


def job_id_col(df: pd.DataFrame | None) -> str | None:
    return find_col(df, ["Job_ID", "Process_ID", "JobID", "ProcessID", "Job"])


def canonical_time_col(df: pd.DataFrame | None, key: str) -> str | None:
    options = {
        "arrival": ["Arrival_Time", "Arrival Time", "Arrival"],
        "burst": ["Burst_Time", "Predicted_Execution_Time", "Execution_Time_Seconds", "Execution_Time", "CPU_Time", "Burst Time", "Predicted Execution Time"],
        "predicted": ["Predicted_Execution_Time", "Predicted Execution Time", "Predicted_Burst_Time"],
        "start": ["Start_Time", "Start Time"],
        "completion": ["Completion_Time", "Completion Time"],
        "waiting": ["Waiting_Time", "Waiting Time"],
        "turnaround": ["Turnaround_Time", "Turnaround Time"],
        "response": ["Response_Time", "Response Time"],
        "priority": ["Priority_Number", "Priority", "Priority_Level"],
        "job_type": ["Job_Type", "Job Type", "Type"],
    }
    return find_col(df, options[key])


def read_zip(zip_bytes: bytes) -> dict[str, pd.DataFrame]:
    loaded: dict[str, pd.DataFrame] = {}
    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as archive:
        for name in archive.namelist():
            if name.lower().endswith(".csv") and not name.startswith("__MACOSX/"):
                try:
                    with archive.open(name) as fp:
                        loaded[Path(name).name.lower()] = pd.read_csv(fp)
                except Exception as exc:
                    loaded[Path(name).name.lower()] = pd.DataFrame()
    return loaded


def get_file(files: dict[str, pd.DataFrame], *names: str) -> pd.DataFrame | None:
    for wanted in names:
        if wanted.lower() in files and not files[wanted.lower()].empty:
            return files[wanted.lower()]
    for actual, frame in files.items():
        for wanted in names:
            if wanted.lower().replace(".csv", "") in actual.lower() and not frame.empty:
                return frame
    return None


def compute_summary_row(label: str, df: pd.DataFrame) -> dict:
    id_col = job_id_col(df)
    return {
        "Algorithm": label,
        "Jobs": int(df[id_col].nunique()) if id_col else int(len(df)),
        "Average Waiting Time": safe_mean(df, ["Waiting_Time", "Average_Waiting_Time"]),
        "Average Turnaround Time": safe_mean(df, ["Turnaround_Time", "Average_Turnaround_Time"]),
        "Average Response Time": safe_mean(df, ["Response_Time", "Average_Response_Time"]),
        "P95 Waiting Time": float(series_num(df, ["Waiting_Time"]).quantile(.95)) if len(series_num(df, ["Waiting_Time"])) else np.nan,
        "Maximum Waiting Time": float(series_num(df, ["Waiting_Time"]).max()) if len(series_num(df, ["Waiting_Time"])) else np.nan,
    }


def build_comparison(files: dict[str, pd.DataFrame], schedulers: dict[str, pd.DataFrame | None]) -> pd.DataFrame:
    # Prefer per-job result CSVs, so the table is recalculated from actual rows.
    rows = [compute_summary_row(name, df) for name, df in schedulers.items() if df is not None and not df.empty]
    if rows:
        return pd.DataFrame(rows)
    summary = get_file(files, "scheduler_comparison.csv")
    if summary is None:
        return pd.DataFrame()
    alg_col = find_col(summary, ["Algorithm", "Scheduler"])
    if alg_col is None:
        return pd.DataFrame()
    data = []
    for _, row in summary.iterrows():
        data.append({
            "Algorithm": str(row[alg_col]),
            "Jobs": value_num(row, ["Jobs", "Jobs_Completed", "Completed_Jobs"]),
            "Average Waiting Time": value_num(row, ["Average_Waiting_Time", "Avg_Waiting_Time"]),
            "Average Turnaround Time": value_num(row, ["Average_Turnaround_Time", "Avg_Turnaround_Time"]),
            "Average Response Time": value_num(row, ["Average_Response_Time", "Avg_Response_Time"]),
            "P95 Waiting Time": np.nan,
            "Maximum Waiting Time": np.nan,
        })
    return pd.DataFrame(data)


def percent_delta(epcs_value: float, baseline_value: float) -> float:
    if not np.isfinite(epcs_value) or not np.isfinite(baseline_value) or abs(baseline_value) < 1e-12:
        return np.nan
    return (baseline_value - epcs_value) / abs(baseline_value) * 100.0


def build_metric_notes(comparison: pd.DataFrame, metrics: list[str]) -> tuple[list[str], dict[str, str], int]:
    notes: list[str] = []
    status: dict[str, str] = {}
    wins = 0
    if comparison.empty or "EPCS" not in comparison["Algorithm"].values:
        return ["EPCS summary is unavailable. Ensure epcs_results.csv is in the uploaded ZIP."], status, wins
    e = comparison.loc[comparison["Algorithm"] == "EPCS"].iloc[0]
    others = comparison[comparison["Algorithm"] != "EPCS"]
    for metric in metrics:
        if metric not in comparison.columns or not np.isfinite(e.get(metric, np.nan)):
            continue
        valid = comparison.dropna(subset=[metric]).sort_values(metric)
        if len(valid) == 0:
            continue
        e_val = float(e[metric])
        min_val = float(valid.iloc[0][metric])
        winners = valid[np.isclose(valid[metric].astype(float), min_val, rtol=1e-5, atol=1e-8)]
        winner_names = ", ".join(winners["Algorithm"].astype(str).tolist())
        if np.isclose(e_val, min_val, rtol=1e-5, atol=1e-8):
            if len(winners) > 1:
                status[metric] = "TIED FIRST" if "EPCS" in winners["Algorithm"].values else "TIED"
                others_best = valid[valid["Algorithm"] != "EPCS"]
                delta = percent_delta(e_val, float(others_best.iloc[0][metric])) if len(others_best) else np.nan
                if np.isfinite(delta):
                    notes.append(f"{metric}: EPCS is tied for the lowest observed mean with {winner_names} at {e_val:.2f}. The uploaded results do not demonstrate a strict EPCS advantage on this metric.")
                else:
                    notes.append(f"{metric}: EPCS has the lowest observed mean at {e_val:.2f}; no separate baseline value is available.")
            else:
                status[metric] = "BEST"
                wins += 1
                runners = valid[valid["Algorithm"] != "EPCS"].sort_values(metric)
                if len(runners):
                    delta = percent_delta(e_val, float(runners.iloc[0][metric]))
                    notes.append(f"{metric}: EPCS is lowest at {e_val:.2f}, {delta:.1f}% below the next-best baseline {runners.iloc[0]['Algorithm']} ({float(runners.iloc[0][metric]):.2f}).")
                else:
                    notes.append(f"{metric}: EPCS is lowest at {e_val:.2f}; no other algorithm is available for comparison.")
        else:
            status[metric] = "NOT BEST"
            best = valid.iloc[0]
            delta = percent_delta(e_val, float(best[metric]))
            notes.append(f"{metric}: {best['Algorithm']} is lowest at {float(best[metric]):.2f}; EPCS is {abs(delta):.1f}% {'higher' if delta < 0 else 'lower'} at {e_val:.2f}. This metric does not support saying EPCS is best.")
    return notes, status, wins


def metric_explanation(metric: str, status: str) -> str:
    theory = {
        "Average Waiting Time": "Waiting time measures how long a job stays in the ready queue before and between CPU service. Lower values indicate less queue delay on this workload.",
        "Average Turnaround Time": "Turnaround time is completion minus arrival. It includes both waiting and execution, so it reflects how quickly jobs finish after entering the system.",
        "Average Response Time": "Response time is first CPU start minus arrival. It matters when a job needs early service, even if its full execution takes longer.",
    }.get(metric, "This metric helps compare scheduler outcomes.")
    if status == "BEST":
        verdict = "Observed result: EPCS is best on this metric among the available uploaded algorithms."
    elif status in ("TIED FIRST", "TIED"):
        verdict = "Observed result: EPCS is tied; the current results do not establish a unique win."
    else:
        verdict = "Observed result: EPCS is not best on this metric in this upload."
    return f"{theory} {verdict}"


def find_job_rows(selected_id: str, schedulers: dict[str, pd.DataFrame | None]) -> pd.DataFrame:
    rows = []
    for name, df in schedulers.items():
        if df is None or df.empty:
            continue
        jid = job_id_col(df)
        if not jid:
            continue
        matched = df[df[jid].astype(str) == str(selected_id)]
        if matched.empty:
            continue
        r = matched.iloc[0]
        rows.append({
            "Algorithm": name,
            "Waiting Time": value_num(r, ["Waiting_Time"]),
            "Turnaround Time": value_num(r, ["Turnaround_Time"]),
            "Response Time": value_num(r, ["Response_Time"]),
            "Start Time": value_num(r, ["Start_Time"]),
            "Completion Time": value_num(r, ["Completion_Time"]),
        })
    return pd.DataFrame(rows)


def workload_profiles(current: pd.DataFrame, previous: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    numeric_rows, categorical_rows = [], []
    shared_num = [
        c for c in current.select_dtypes(include=np.number).columns
        if c in previous.columns and pd.api.types.is_numeric_dtype(previous[c])
    ]
    for col in shared_num:
        a = pd.to_numeric(previous[col], errors="coerce").dropna()
        b = pd.to_numeric(current[col], errors="coerce").dropna()
        if len(a) < 2 or len(b) < 2:
            continue
        denom = float(np.sqrt((a.std(ddof=1) ** 2 + b.std(ddof=1) ** 2) / 2))
        smd = (float(b.mean()) - float(a.mean())) / denom if denom > 1e-12 and np.isfinite(denom) else 0.0
        numeric_rows.append({
            "Feature": col, "Previous mean": float(a.mean()), "Current mean": float(b.mean()),
            "Standardized shift": float(smd), "Abs shift": abs(float(smd)),
            "Signal": "Large" if abs(smd) >= .5 else ("Moderate" if abs(smd) >= .25 else "Small"),
        })
    shared_cat = [c for c in current.columns if c in previous.columns and (current[c].dtype == object or str(current[c].dtype).startswith("category"))]
    for col in shared_cat:
        a = previous[col].dropna().astype(str).value_counts(normalize=True)
        b = current[col].dropna().astype(str).value_counts(normalize=True)
        categories = set(a.index) | set(b.index)
        if not categories or len(categories) > 60:
            continue
        tvd = 0.5 * sum(abs(float(a.get(v, 0)) - float(b.get(v, 0))) for v in categories)
        categorical_rows.append({"Feature": col, "Mix change (0–1)": float(tvd), "Change points": 100 * float(tvd)})
    return pd.DataFrame(numeric_rows), pd.DataFrame(categorical_rows)


def run_nonpreemptive(frame: pd.DataFrame, algorithm: str) -> dict:
    """Small, deterministic educational simulator used only in the stress lab."""
    d = frame.copy().reset_index(drop=True)
    arrivals = pd.to_numeric(d["Arrival"], errors="coerce").to_numpy(dtype=float)
    bursts = pd.to_numeric(d["Burst"], errors="coerce").to_numpy(dtype=float)
    priorities = pd.to_numeric(d["Priority"], errors="coerce").fillna(999).to_numpy(dtype=float)
    n = len(d)
    remaining = set(range(n))
    start = np.full(n, np.nan)
    completion = np.full(n, np.nan)
    now = float(np.min(arrivals)) if n else 0.0
    while remaining:
        ready = [i for i in remaining if arrivals[i] <= now + 1e-9]
        if not ready:
            now = min(arrivals[i] for i in remaining)
            ready = [i for i in remaining if arrivals[i] <= now + 1e-9]
        if algorithm == "FCFS":
            chosen = min(ready, key=lambda i: (arrivals[i], i))
        elif algorithm == "SJF":
            chosen = min(ready, key=lambda i: (bursts[i], arrivals[i], i))
        elif algorithm == "Priority":
            chosen = min(ready, key=lambda i: (priorities[i], arrivals[i], i))
        else:
            raise ValueError("Unsupported non-preemptive algorithm")
        start[chosen] = now
        now += max(0.0, bursts[chosen])
        completion[chosen] = now
        remaining.remove(chosen)
    waiting = start - arrivals
    turnaround = completion - arrivals
    response = waiting.copy()
    return {
        "Algorithm": algorithm, "Average Waiting Time": float(np.mean(waiting)),
        "Average Turnaround Time": float(np.mean(turnaround)),
        "Average Response Time": float(np.mean(response)),
        "P95 Waiting Time": float(np.quantile(waiting, .95)),
        "Maximum Waiting Time": float(np.max(waiting)),
        "Jobs": n,
    }


def run_round_robin(frame: pd.DataFrame, quantum: float) -> dict:
    d = frame.copy().reset_index(drop=True)
    arrivals = pd.to_numeric(d["Arrival"], errors="coerce").to_numpy(dtype=float)
    bursts = pd.to_numeric(d["Burst"], errors="coerce").to_numpy(dtype=float)
    n = len(d)
    if n == 0:
        return {"Algorithm": f"RR q={quantum:.2f}", "Average Waiting Time": np.nan,
                "Average Turnaround Time": np.nan, "Average Response Time": np.nan,
                "P95 Waiting Time": np.nan, "Maximum Waiting Time": np.nan, "Jobs": 0,
                "Context Switches": 0}
    order = np.argsort(arrivals, kind="stable")
    rem = bursts.copy()
    first = np.full(n, np.nan)
    completion = np.full(n, np.nan)
    ready = deque()
    idx = 0
    now = float(arrivals[order[0]])
    context_switches = 0
    last_proc = None
    eps = 1e-9
    while idx < n or ready:
        if not ready and idx < n and now < arrivals[order[idx]]:
            now = float(arrivals[order[idx]])
        while idx < n and arrivals[order[idx]] <= now + eps:
            ready.append(int(order[idx]))
            idx += 1
        if not ready:
            continue
        i = ready.popleft()
        if np.isnan(first[i]):
            first[i] = now
        if last_proc is not None and last_proc != i:
            context_switches += 1
        last_proc = i
        step = min(float(quantum), rem[i])
        now += step
        rem[i] -= step
        while idx < n and arrivals[order[idx]] <= now + eps:
            ready.append(int(order[idx]))
            idx += 1
        if rem[i] > eps:
            ready.append(i)
        else:
            completion[i] = now
    waiting = completion - arrivals - bursts
    turnaround = completion - arrivals
    response = first - arrivals
    return {
        "Algorithm": f"Round Robin (q={quantum:.1f})", "Average Waiting Time": float(np.mean(waiting)),
        "Average Turnaround Time": float(np.mean(turnaround)),
        "Average Response Time": float(np.mean(response)),
        "P95 Waiting Time": float(np.quantile(waiting, .95)),
        "Maximum Waiting Time": float(np.max(waiting)),
        "Jobs": n, "Context Switches": context_switches,
    }


def section(title: str, subtitle: str | None = None):
    st.markdown(f"<div class='section-kicker'>{title}</div>", unsafe_allow_html=True)
    if subtitle:
        st.caption(subtitle)


# -------------------------------- Upload -----------------------------------
st.sidebar.markdown("## Project data")
st.sidebar.caption("Upload the ZIP exported from the Kaggle notebook.")
uploaded = st.sidebar.file_uploader("dashboard_data.zip", type=["zip"])
st.sidebar.divider()
st.sidebar.markdown("### Project details")
st.sidebar.write("**Project:** EPCS")
st.sidebar.write("**Created by:** Mangal Sakthi")
st.sidebar.write("**Core methods:** XGBoost · SHAP · CPU scheduling")
st.sidebar.caption("ZIP data is analysed in this session. This app does not persist uploaded data after the session ends.")

if uploaded is None:
    st.markdown("## Dashboard guide")
    st.markdown(
        "- Upload `dashboard_data.zip` to load the scheduling results.\n"
        "- The dashboard compares EPCS with FCFS, SJF, Priority and Round Robin when their files are present.\n"
        "- Six decision-intelligence modules provide dataset-specific analysis and proposed research directions.\n"
        "- Use the generated findings as evidence; do not claim EPCS is best unless the results show it."
    )
    st.stop()

zip_bytes = uploaded.getvalue()
fingerprint = hashlib.sha256(zip_bytes).hexdigest()
try:
    files = read_zip(zip_bytes)
except zipfile.BadZipFile:
    st.error("This is not a valid ZIP. Re-export dashboard_data.zip from Kaggle and try again.")
    st.stop()

epcs = get_file(files, "epcs_results.csv")
fcfs = get_file(files, "fcfs_results.csv")
sjf = get_file(files, "sjf_results_713.csv", "sjf_results.csv", "sjf_results.csv")
priority = get_file(files, "priority_results.csv")
rr = get_file(files, "rr_results.csv", "round_robin_results.csv")
workload = get_file(files, "scheduler_workload.csv", "scheduler_workload")
shap_df = get_file(files, "shap_feature_importance.csv", "shap_feature_importance")
local_shap = get_file(files, "shap_local_explanations.csv", "individual_shap_explanations.csv")
schedulers: dict[str, pd.DataFrame | None] = {
    "EPCS": epcs, "FCFS": fcfs, "SJF": sjf, "Priority": priority, "Round Robin": rr,
}
available = {name: df for name, df in schedulers.items() if df is not None and not df.empty}
summary = build_comparison(files, schedulers)
metric_names = ["Average Waiting Time", "Average Turnaround Time", "Average Response Time"]
notes, metric_status, epcs_wins = build_metric_notes(summary, metric_names)

# Session-scoped comparison of uploaded workload files for drift view.
if "epcs_upload_fingerprint" not in st.session_state:
    st.session_state.epcs_upload_fingerprint = fingerprint
    st.session_state.epcs_current_workload = workload.copy() if workload is not None else None
    st.session_state.epcs_current_file_name = uploaded.name
    st.session_state.epcs_previous_workload = None
    st.session_state.epcs_previous_file_name = None
elif st.session_state.epcs_upload_fingerprint != fingerprint:
    st.session_state.epcs_previous_workload = (
        st.session_state.epcs_current_workload.copy()
        if st.session_state.epcs_current_workload is not None else None
    )
    st.session_state.epcs_previous_file_name = st.session_state.epcs_current_file_name
    st.session_state.epcs_upload_fingerprint = fingerprint
    st.session_state.epcs_current_workload = workload.copy() if workload is not None else None
    st.session_state.epcs_current_file_name = uploaded.name

with st.sidebar.expander("Files detected", expanded=False):
    for filename in sorted(files):
        st.write(f"✓ {filename}")
st.sidebar.success(f"{len(files)} CSV files loaded")

# ----------------------------- 1 Overview ----------------------------------
section("01 / Executive summary", "All findings below are calculated from the uploaded CSVs where possible.")
job_count = len(workload) if workload is not None else (len(epcs) if epcs is not None else 0)
epcs_row = summary[summary["Algorithm"] == "EPCS"].iloc[0] if not summary.empty and "EPCS" in summary["Algorithm"].values else None
a, b, c, d = st.columns(4)
a.metric("Jobs in workload", f"{job_count:,}" if job_count else "N/A")
b.metric("Algorithms present", f"{len(available)} / 5")
c.metric("EPCS average waiting", fmt(float(epcs_row["Average Waiting Time"])) if epcs_row is not None else "N/A")
d.metric("EPCS average turnaround", fmt(float(epcs_row["Average Turnaround Time"])) if epcs_row is not None else "N/A")
st.caption(f"Uploaded: **{uploaded.name}** · Files: **{len(files)}** · Data fingerprint: `{fingerprint[:12]}`")

section("02 / Dataset verdict", "The conclusion changes automatically when you upload a different result ZIP.")
if summary.empty or "EPCS" not in summary["Algorithm"].values:
    st.warning("EPCS results are missing. Add epcs_results.csv to the ZIP.")
else:
    available_metrics = [m for m in metric_names if m in summary.columns and summary[m].notna().any()]
    score_cols = st.columns(max(1, len(available_metrics)))
    for idx, metric in enumerate(available_metrics):
        state = metric_status.get(metric, "UNAVAILABLE")
        label = {"BEST": "Best", "TIED FIRST": "Tied first", "TIED": "Tied", "NOT BEST": "Not best"}.get(state, state)
        score_cols[idx].metric(metric.replace("Average ", ""), label)
    if epcs_wins == len(available_metrics) and available_metrics:
        st.success("Dataset verdict: EPCS records the lowest mean for all available time metrics in this upload. Confirm identical job sets and scheduling assumptions before claiming overall superiority.")
    elif epcs_wins > 0:
        st.info(f"Dataset verdict: EPCS leads on {epcs_wins} of {len(available_metrics)} available time metrics. The best scheduler depends on the metric being prioritised.")
    else:
        st.warning("Dataset verdict: these results do not show a strict EPCS win on the average time metrics. The dashboard will explain the actual ties/losses instead of claiming EPCS is best.")
    for note in notes:
        st.markdown(f"<div class='insight'>{note}</div>", unsafe_allow_html=True)
    if len(available):
        row_counts = {name: int(len(df)) for name, df in available.items()}
        if len(set(row_counts.values())) > 1:
            st.warning(f"Job counts differ across result files: {row_counts}. A fair comparison requires the same jobs and assumptions.")
        if epcs is not None and sjf is not None:
            epcs_ids, sjf_ids = job_id_col(epcs), job_id_col(sjf)
            if epcs_ids and sjf_ids:
                ei = set(epcs[epcs_ids].dropna().astype(str))
                si = set(sjf[sjf_ids].dropna().astype(str))
                if ei and si and ei != si:
                    st.warning(f"EPCS and SJF job IDs do not match exactly (overlap: {len(ei & si):,} of EPCS {len(ei):,} / SJF {len(si):,}). Do not interpret their aggregate difference as a controlled head-to-head result.")

# -------------------------- 3 Main comparison ------------------------------
st.divider()
section("03 / Scheduling algorithm comparison", "Lower waiting, turnaround and response times are generally preferable; no single metric tells the whole story.")
if not summary.empty:
    display_cols = ["Algorithm", "Jobs", *metric_names, "P95 Waiting Time", "Maximum Waiting Time"]
    display_cols = [x for x in display_cols if x in summary.columns]
    st.dataframe(summary[display_cols].round(3), use_container_width=True, hide_index=True)
    p1, p2 = st.columns(2)
    with p1:
        metric_choice = st.selectbox("Choose the metric to compare", metric_names, key="comparison_metric")
        plot_data = summary.dropna(subset=[metric_choice]).sort_values(metric_choice)
        fig = px.bar(
            plot_data, x="Algorithm", y=metric_choice, color="Algorithm", text=plot_data[metric_choice].map(lambda x: f"{x:.2f}"),
            title=f"{metric_choice} · lower is better", color_discrete_map=COLORS, template=PLOT_TEMPLATE,
        )
        fig.update_layout(showlegend=False, margin=dict(l=10,r=10,t=60,b=10), yaxis_title="Time units", xaxis_title="")
        fig.update_traces(textposition="outside", cliponaxis=False)
        st.plotly_chart(fig, use_container_width=True, key="epcs_chart_1")
    with p2:
        melt = summary.melt(id_vars=["Algorithm"], value_vars=metric_names, var_name="Metric", value_name="Mean time").dropna()
        fig = px.bar(melt, x="Algorithm", y="Mean time", color="Metric", barmode="group", title="All average time metrics", template=PLOT_TEMPLATE, color_discrete_sequence=EXTRA_COLORS)
        fig.update_layout(margin=dict(l=10,r=10,t=60,b=10), yaxis_title="Time units", xaxis_title="")
        st.plotly_chart(fig, use_container_width=True, key="epcs_chart_2")
    p3, p4 = st.columns(2)
    with p3:
        p95 = summary.dropna(subset=["P95 Waiting Time"]) if "P95 Waiting Time" in summary else pd.DataFrame()
        if not p95.empty:
            fig = px.bar(p95, x="Algorithm", y="P95 Waiting Time", color="Algorithm", title="95th percentile waiting time · tail-delay view", color_discrete_map=COLORS, template=PLOT_TEMPLATE)
            fig.update_layout(showlegend=False, margin=dict(l=10,r=10,t=60,b=10), yaxis_title="Time units")
            st.plotly_chart(fig, use_container_width=True, key="epcs_chart_3")
    with p4:
        maxwait = summary.dropna(subset=["Maximum Waiting Time"]) if "Maximum Waiting Time" in summary else pd.DataFrame()
        if not maxwait.empty:
            fig = px.bar(maxwait, x="Algorithm", y="Maximum Waiting Time", color="Algorithm", title="Maximum waiting time · worst observed case", color_discrete_map=COLORS, template=PLOT_TEMPLATE)
            fig.update_layout(showlegend=False, margin=dict(l=10,r=10,t=60,b=10), yaxis_title="Time units")
            st.plotly_chart(fig, use_container_width=True, key="epcs_chart_4")

    st.markdown("### Why these results matter — theory in short points")
    theory_items = [
        ("Waiting time", metric_status.get("Average Waiting Time", "UNAVAILABLE"), "Measures queue delay before first CPU service and any later waiting. A lower average means less queue delay over the tested workload, but average alone can hide a few jobs that wait much longer."),
        ("Turnaround time", metric_status.get("Average Turnaround Time", "UNAVAILABLE"), "Measures arrival-to-completion duration. It combines waiting and execution time, so it is useful for end-to-end job completion."),
        ("Response time", metric_status.get("Average Response Time", "UNAVAILABLE"), "Measures arrival-to-first-service. It is especially meaningful for interactive work; a scheduler may improve response without minimizing total completion time."),
    ]
    for title, state, definition in theory_items:
        verdict = {"BEST":"EPCS leads on this metric in the uploaded results.", "TIED FIRST":"EPCS ties for best; no strict advantage is shown.", "TIED":"Results show a tie.", "NOT BEST":"Another scheduler has a lower observed mean."}.get(state, "The metric is unavailable.")
        st.markdown(f"<div class='insight'><b>{title} · {state}</b><br>{definition}<br><b>Dataset reading:</b> {verdict}</div>", unsafe_allow_html=True)


# ----------------------------- XGBoost scorecard ---------------------------
st.divider()
section("03A / XGBoost model scorecard", "Previously reported notebook evaluation values. These are shown as recorded and are not recalculated from the uploaded ZIP.")
x1, x2, x3 = st.columns(3)
x1.metric("Previously reported MAE", "43.55 seconds")
x2.metric("Previously reported RMSE", "55.55 seconds")
x3.metric("Previously reported R²", "0.9898")
st.caption("Interpretation: MAE is mean absolute error; RMSE gives greater weight to large errors; R² describes the evaluated test-set fit. Revalidate these numbers with a leakage-safe chronological holdout before using them as evidence of real-world performance.")

# -------------------------- 4 Innovation 1 ---------------------------------
st.divider()
section("04 / Innovation 1 — Counterfactual Job Mirror", "Question: 'For this same job, how did the recorded outcomes differ under each available scheduler?'")
st.markdown("**Theory:** A decision is easier to trust when the user can inspect a plausible alternative. Here, we compare the recorded result rows for one job across scheduler outputs.")
if epcs is not None and not epcs.empty and job_id_col(epcs):
    ids = epcs[job_id_col(epcs)].dropna().astype(str).unique().tolist()
    if ids:
        chosen = st.selectbox("Select a job ID", ids, key="job_mirror_id")
        mirror = find_job_rows(chosen, schedulers)
        if mirror.empty:
            st.info("The selected job ID was not matched to any scheduler result rows.")
        else:
            st.dataframe(mirror.round(3), use_container_width=True, hide_index=True)
            mp = mirror.melt(id_vars=["Algorithm"], value_vars=["Waiting Time","Turnaround Time","Response Time"], var_name="Metric", value_name="Time").dropna()
            if not mp.empty:
                fig = px.bar(mp, x="Algorithm", y="Time", color="Metric", barmode="group", title=f"Observed alternatives for {chosen}", template=PLOT_TEMPLATE, color_discrete_sequence=EXTRA_COLORS)
                fig.update_layout(margin=dict(l=10,r=10,t=60,b=10), yaxis_title="Time units")
                st.plotly_chart(fig, use_container_width=True, key="epcs_chart_5")
            for m in ["Waiting Time", "Turnaround Time", "Response Time"]:
                valid = mirror.dropna(subset=[m]).sort_values(m)
                if len(valid):
                    st.markdown(f"<div class='insight'>• Lowest recorded {m.lower()} for this job: <b>{valid.iloc[0]['Algorithm']}</b> ({valid.iloc[0][m]:.2f}). This is a comparison of saved outcomes, not a full counterfactual re-simulation.</div>", unsafe_allow_html=True)
else:
    st.info("This panel needs epcs_results.csv with a Job_ID or Process_ID column.")

# -------------------------- 5 Innovation 2 ---------------------------------
st.divider()
section("05 / Innovation 2 — Prediction Reliability Guard", "Question: 'Where are execution-time predictions most wrong?'")
st.markdown("**Theory:** A scheduler should not blindly trust a predictor. Retrospective errors help identify where further inspection or retraining may be needed. This is an error monitor, not a calibrated confidence probability.")
if epcs is not None and workload is not None and not epcs.empty and not workload.empty:
    pred_col = canonical_time_col(epcs, "predicted")
    actual_col = find_col(workload, ["Burst_Time", "Execution_Time(Seconds)", "Actual_Execution_Time", "Execution_Time", "CPU_Time"])
    e_id, w_id = job_id_col(epcs), job_id_col(workload)
    if pred_col and actual_col and e_id and w_id:
        pred_data = epcs[[e_id, pred_col]].copy()
        pred_data["_job_key"] = pred_data[e_id].astype(str)
        actual_data = workload[[w_id, actual_col]].copy()
        actual_data["_job_key"] = actual_data[w_id].astype(str)
        joined = pred_data.merge(actual_data[["_job_key", actual_col]], on="_job_key", how="inner")
        joined[pred_col] = pd.to_numeric(joined[pred_col], errors="coerce")
        joined[actual_col] = pd.to_numeric(joined[actual_col], errors="coerce")
        joined = joined.dropna(subset=[pred_col, actual_col])
        joined = joined[(joined[pred_col] >= 0) & (joined[actual_col] >= 0)]
        if len(joined) >= 3:
            joined["Signed Error"] = joined[pred_col] - joined[actual_col]
            joined["Absolute Error"] = joined["Signed Error"].abs()
            z1,z2,z3 = st.columns(3)
            z1.metric("Matched job rows", f"{len(joined):,}")
            z2.metric("Recalculated MAE", fmt(float(joined["Absolute Error"].mean())))
            z3.metric("90th percentile absolute error", fmt(float(joined["Absolute Error"].quantile(.90))))
            q90 = float(joined["Absolute Error"].quantile(.90))
            joined["Review band"] = np.where(joined["Absolute Error"] >= q90, "Top error tail (approx.)", "Other")
            e1,e2 = st.columns(2)
            with e1:
                fig = px.histogram(joined, x="Absolute Error", nbins=32, color="Review band", title="Absolute prediction-error distribution", template=PLOT_TEMPLATE, color_discrete_sequence=EXTRA_COLORS)
                fig.update_layout(margin=dict(l=10,r=10,t=60,b=10), xaxis_title="Absolute error")
                st.plotly_chart(fig, use_container_width=True, key="epcs_chart_6")
            with e2:
                fig = px.scatter(joined, x=actual_col, y=pred_col, color="Review band", hover_data=[e_id], title="Predicted vs recorded execution time", template=PLOT_TEMPLATE, color_discrete_sequence=EXTRA_COLORS)
                low = min(float(joined[actual_col].min()), float(joined[pred_col].min()))
                high = max(float(joined[actual_col].max()), float(joined[pred_col].max()))
                fig.add_shape(type="line", x0=low, y0=low, x1=high, y1=high, line=dict(color="#c08a34", dash="dash"))
                fig.update_layout(margin=dict(l=10,r=10,t=60,b=10), xaxis_title="Recorded time", yaxis_title="Predicted time")
                st.plotly_chart(fig, use_container_width=True, key="epcs_chart_7")
            st.markdown("**Jobs to review first**")
            st.dataframe(joined.sort_values("Absolute Error", ascending=False)[[e_id,pred_col,actual_col,"Signed Error","Absolute Error"]].head(12).round(3), use_container_width=True, hide_index=True)
            st.markdown(f"<div class='insight'>• The largest absolute error is {joined['Absolute Error'].max():.2f}. The top error tail is selected using this upload's 90th percentile; it is a review heuristic, not a confidence score.</div>", unsafe_allow_html=True)
        else:
            st.info("Fewer than three matching jobs were available for prediction-error analysis.")
    else:
        missing = []
        if not pred_col: missing.append("Predicted_Execution_Time in EPCS results")
        if not actual_col: missing.append("recorded Burst_Time / execution-time target in workload")
        if not e_id or not w_id: missing.append("matching job IDs")
        st.warning("Cannot calculate reliability metrics because the ZIP is missing: " + ", ".join(missing) + ".")
else:
    st.info("Upload both EPCS results and workload rows to enable the reliability guard.")

# -------------------------- 6 Innovation 3 ---------------------------------
st.divider()
section("06 / Innovation 3 — Workload Shift Radar", "Question: 'Has the workload changed compared with the previous uploaded dataset?'")
st.markdown("**Theory:** Model behaviour can change when workload feature distributions change. This panel flags input shifts; a shift does not by itself prove predictive accuracy has fallen.")
current_workload = workload
previous_workload = st.session_state.get("epcs_previous_workload")
if current_workload is not None and previous_workload is not None and not current_workload.empty and not previous_workload.empty:
    st.caption(f"Current: {uploaded.name} · Baseline: {st.session_state.get('epcs_previous_file_name') or 'previous upload'}")
    nshift, cshift = workload_profiles(current_workload, previous_workload)
    left,right = st.columns(2)
    with left:
        if not nshift.empty:
            nshift_plot = nshift.sort_values("Abs shift", ascending=True)
            fig = px.bar(nshift_plot, x="Abs shift", y="Feature", orientation="h", color="Signal", title="Numeric workload shift (standardized mean difference)", template=PLOT_TEMPLATE, color_discrete_sequence=EXTRA_COLORS)
            fig.add_vline(x=.25, line_dash="dash", line_color="#c08a34", annotation_text="moderate")
            fig.add_vline(x=.5, line_dash="dot", line_color="#c56856", annotation_text="large")
            fig.update_layout(margin=dict(l=10,r=10,t=60,b=10))
            st.plotly_chart(fig, use_container_width=True, key="epcs_chart_8")
            st.dataframe(nshift.sort_values("Abs shift", ascending=False).round(3), use_container_width=True, hide_index=True)
        else:
            st.info("No shared numeric columns were suitable for shift comparison.")
    with right:
        if not cshift.empty:
            fig = px.bar(cshift.sort_values("Mix change (0–1)"), x="Mix change (0–1)", y="Feature", orientation="h", title="Categorical composition change (0–1)", template=PLOT_TEMPLATE, color_discrete_sequence=["#247c78"])
            fig.update_layout(margin=dict(l=10,r=10,t=60,b=10))
            st.plotly_chart(fig, use_container_width=True, key="epcs_chart_9")
            st.dataframe(cshift.sort_values("Mix change (0–1)", ascending=False).round(3), use_container_width=True, hide_index=True)
        else:
            st.info("No shared categorical columns were suitable for shift comparison.")
else:
    st.info("Upload a second, different ZIP in this same browser session to compare workload distributions. The baseline is session-only.")
    if workload is not None and not workload.empty:
        nums = workload.select_dtypes(include=np.number).columns.tolist()
        if nums:
            col = st.selectbox("Current dataset feature preview", nums, key="shift_feature_preview")
            vals = pd.to_numeric(workload[col], errors="coerce").dropna()
            left,right = st.columns(2)
            with left:
                fig = px.histogram(vals.to_frame(name=col), x=col, nbins=30, title=f"Distribution of {col}", template=PLOT_TEMPLATE, color_discrete_sequence=["#247c78"])
                fig.update_layout(margin=dict(l=10,r=10,t=60,b=10))
                st.plotly_chart(fig, use_container_width=True, key="epcs_chart_10")
            with right:
                fig = px.box(vals.to_frame(name=col), y=col, points="outliers", title=f"Spread and outliers: {col}", template=PLOT_TEMPLATE, color_discrete_sequence=["#447da2"])
                fig.update_layout(margin=dict(l=10,r=10,t=60,b=10))
                st.plotly_chart(fig, use_container_width=True, key="epcs_chart_11")

# -------------------------- 7 New idea: Fairness ---------------------------
st.divider()
section("07 / New research direction — Fairness & Starvation Radar", "A CPU scheduler should not only optimize averages; it should also reveal whether a subgroup or a few jobs wait unusually long.")
st.markdown("**Theory:** An average can look good even when some jobs wait much longer than others. P95/max waiting time and subgroup summaries help expose the tail of the waiting-time distribution. They are warning signals, not proof of starvation (which requires observing indefinite waiting over time).")
fair_source = epcs if epcs is not None and not epcs.empty else None
if fair_source is not None and canonical_time_col(fair_source, "waiting"):
    wcol = canonical_time_col(fair_source, "waiting")
    fair_df = fair_source.copy()
    fair_df[wcol] = pd.to_numeric(fair_df[wcol], errors="coerce")
    fair_df = fair_df.dropna(subset=[wcol])
    if len(fair_df):
        q50, q95 = float(fair_df[wcol].quantile(.50)), float(fair_df[wcol].quantile(.95))
        qmax, avg = float(fair_df[wcol].max()), float(fair_df[wcol].mean())
        f1,f2,f3,f4 = st.columns(4)
        f1.metric("Median wait", fmt(q50))
        f2.metric("P95 wait", fmt(q95))
        f3.metric("Maximum wait", fmt(qmax))
        f4.metric("P95 / median", fmt(q95 / q50) if q50 > 1e-12 else ("N/A" if q95 == 0 else "High vs zero median"))
        wl,wr = st.columns(2)
        with wl:
            fig = px.histogram(fair_df, x=wcol, nbins=32, title="EPCS waiting-time distribution", template=PLOT_TEMPLATE, color_discrete_sequence=["#447da2"])
            fig.add_vline(x=q95, line_dash="dash", line_color="#c56856", annotation_text="P95")
            fig.update_layout(margin=dict(l=10,r=10,t=60,b=10), xaxis_title="Waiting time")
            st.plotly_chart(fig, use_container_width=True, key="epcs_chart_12")
        with wr:
            group_col = canonical_time_col(fair_df, "job_type") or canonical_time_col(fair_df, "priority")
            if group_col:
                grouped = fair_df.groupby(group_col, dropna=False)[wcol].agg(["mean","median","max","count"]).reset_index()
                grouped = grouped.rename(columns={group_col:"Group","mean":"Mean wait","median":"Median wait","max":"Maximum wait","count":"Jobs"})
                fig = px.bar(grouped, x="Group", y="Mean wait", color="Group", title=f"Mean waiting time by {group_col}", template=PLOT_TEMPLATE, color_discrete_sequence=EXTRA_COLORS)
                fig.update_layout(showlegend=False, margin=dict(l=10,r=10,t=60,b=10), xaxis_title="")
                st.plotly_chart(fig, use_container_width=True, key="epcs_chart_13")
                st.dataframe(grouped.round(3), use_container_width=True, hide_index=True)
            else:
                st.info("Add Job_Type or Priority_Number to EPCS results to compare waiting-time groups.")
        if q95 > max(avg * 2.0, q50 * 2.0, 1e-9):
            st.markdown("<div class='insight insight-warn'>• Tail-delay review: P95 waiting time is notably above the average/median. Inspect individual high-wait jobs before concluding the scheduler is fair.</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='insight'>• Tail-delay review: the current summary does not meet the dashboard's simple high-tail warning rule. This is not a formal starvation guarantee.</div>", unsafe_allow_html=True)
    else:
        st.info("No valid EPCS waiting-time values were found.")
else:
    st.info("Fairness analysis requires per-job Waiting_Time values in epcs_results.csv.")

# -------------------------- 8 New idea: Policy recommender -----------------
st.divider()
section("08 / New research direction — Goal-Based Policy Recommender", "Let the evaluator choose what matters most. The recommendation changes according to the uploaded dataset and selected trade-offs.")
st.markdown("**Theory:** Scheduling is multi-objective. A policy that minimizes mean waiting time may not minimize turnaround, response time, or long-tail waiting. This transparent score makes the preference explicit rather than hiding it in one chart.")
if not summary.empty and all(m in summary.columns for m in metric_names):
    g1,g2,g3 = st.columns(3)
    w_weight = g1.slider("Waiting-time importance", 0, 100, 45, 5, key="goal_wait")
    t_weight = g2.slider("Turnaround importance", 0, 100, 35, 5, key="goal_turn")
    r_weight = g3.slider("Response-time importance", 0, 100, 20, 5, key="goal_resp")
    raw_weights = np.array([w_weight,t_weight,r_weight], dtype=float)
    if raw_weights.sum() == 0:
        st.warning("Choose at least one metric weight.")
    else:
        weights = raw_weights/raw_weights.sum()
        norm = summary[["Algorithm", *metric_names]].copy()
        for metric in metric_names:
            col = pd.to_numeric(norm[metric], errors="coerce")
            lo,hi = col.min(),col.max()
            norm[metric+"_norm"] = 0.0 if pd.isna(lo) or pd.isna(hi) or np.isclose(lo,hi) else (col-lo)/(hi-lo)
        norm["Weighted score (lower is better)"] = (
            weights[0]*norm[metric_names[0]+"_norm"] +
            weights[1]*norm[metric_names[1]+"_norm"] +
            weights[2]*norm[metric_names[2]+"_norm"]
        )
        norm = norm.sort_values("Weighted score (lower is better)")
        fig = px.bar(norm, x="Algorithm", y="Weighted score (lower is better)", color="Algorithm", title="Goal-weighted policy ranking (relative score)", color_discrete_map=COLORS, template=PLOT_TEMPLATE, text=norm["Weighted score (lower is better)"].map(lambda x:f"{x:.3f}"))
        fig.update_layout(showlegend=False, margin=dict(l=10,r=10,t=60,b=10), yaxis_title="Relative normalized score (lower is better)")
        st.plotly_chart(fig, use_container_width=True, key="epcs_chart_14")
        st.dataframe(norm[["Algorithm",*metric_names,"Weighted score (lower is better)"]].round(3), use_container_width=True, hide_index=True)
        winner = norm.iloc[0]
        st.markdown(f"<div class='insight'>• With the selected weights (waiting {weights[0]*100:.0f}%, turnaround {weights[1]*100:.0f}%, response {weights[2]*100:.0f}%), the lowest weighted score is <b>{winner['Algorithm']}</b>. This is a transparent preference-based recommendation, not proof of universal superiority.</div>", unsafe_allow_html=True)
else:
    st.info("Policy recommendation requires comparable average waiting, turnaround and response metrics.")

# -------------------------- 9 New idea: Stress lab -------------------------
st.divider()
section("09 / New research direction — Workload Stress Lab", "Explore how baseline scheduling results change when execution demand increases or Round Robin quantum changes.")
st.markdown("**Theory:** A result on one workload is not enough to show robustness. Sensitivity tests deliberately perturb job demand and report whether the relative baseline ranking changes.")
if workload is not None and not workload.empty:
    arrival_col = canonical_time_col(workload, "arrival")
    burst_col = find_col(workload, ["Burst_Time", "Execution_Time(Seconds)", "Execution_Time", "CPU_Time", "Predicted_Execution_Time"])
    priority_col = canonical_time_col(workload, "priority")
    if arrival_col and burst_col:
        stress_base = pd.DataFrame({
            "Arrival": pd.to_numeric(workload[arrival_col], errors="coerce"),
            "Burst": pd.to_numeric(workload[burst_col], errors="coerce"),
            "Priority": pd.to_numeric(workload[priority_col], errors="coerce") if priority_col else 999,
        }).dropna(subset=["Arrival","Burst"])
        stress_base = stress_base[(stress_base["Arrival"] >= 0) & (stress_base["Burst"] > 0)].sort_values("Arrival")
        if len(stress_base) > 250:
            sample_idx = np.linspace(0, len(stress_base)-1, 250, dtype=int)
            stress_base = stress_base.iloc[sample_idx].reset_index(drop=True)
            st.caption("For speed, the stress lab uses a deterministic time-spread sample of up to 250 jobs. These are simulated scenarios, not observed EPCS results.")
        else:
            st.caption("Stress lab uses the valid workload rows available. These are simulated scenarios, not observed EPCS results.")
        demand_scale = st.slider("Execution-demand multiplier", 0.50, 2.00, 1.00, 0.10, key="stress_demand")
        scenario = stress_base.copy()
        scenario["Burst"] = scenario["Burst"] * float(demand_scale)
        stress_rows = []
        for alg in ["FCFS","SJF","Priority"]:
            if alg == "Priority" and priority_col is None:
                continue
            stress_rows.append(run_nonpreemptive(scenario, alg))
        median_burst = float(scenario["Burst"].median()) if len(scenario) else 1.0
        quantum_factors = [0.10,0.25,0.50,1.00]
        rr_rows = [run_round_robin(scenario, max(median_burst*factor, 1e-6)) for factor in quantum_factors]
        all_stress = pd.DataFrame(stress_rows + rr_rows)
        if not all_stress.empty:
            st.write("**Simulated baseline results under the selected execution-demand scenario**")
            st.dataframe(all_stress.round(3), use_container_width=True, hide_index=True)
            fig = px.bar(all_stress, x="Algorithm", y="Average Waiting Time", color="Algorithm", title="Stress scenario · average waiting time", color_discrete_sequence=EXTRA_COLORS, template=PLOT_TEMPLATE)
            fig.update_layout(showlegend=False, margin=dict(l=10,r=10,t=60,b=10), yaxis_title="Mean waiting time")
            st.plotly_chart(fig, use_container_width=True, key="epcs_chart_15")
        rr_curve = pd.DataFrame(rr_rows)
        rr_curve["Quantum / median burst"] = quantum_factors
        fig = px.line(rr_curve, x="Quantum / median burst", y=["Average Waiting Time","Average Turnaround Time","Average Response Time"], markers=True, title="Round Robin quantum sensitivity", template=PLOT_TEMPLATE, color_discrete_sequence=EXTRA_COLORS)
        fig.update_layout(margin=dict(l=10,r=10,t=60,b=10), yaxis_title="Mean time", xaxis_title="Quantum as a fraction of median burst")
        st.plotly_chart(fig, use_container_width=True, key="epcs_chart_16")
        st.markdown("<div class='insight insight-warn'>• Scope limit: this sandbox simulates FCFS, non-preemptive SJF, non-preemptive Priority and Round Robin from workload columns. It does not re-run the learned EPCS policy because this ZIP contains result CSVs, not the fitted model plus the exact EPCS dispatch implementation. Do not label these stress results as EPCS performance.</div>", unsafe_allow_html=True)
    else:
        st.warning("Stress Lab needs Arrival_Time and Burst_Time (or another documented execution-time column) in scheduler_workload.csv.")
else:
    st.info("Upload scheduler_workload.csv with arrival and execution-time columns to enable stress testing.")

# ----------------------------- 10 Workload ---------------------------------
st.divider()
section("10 / Workload fingerprint", "Use these charts to explain what kind of jobs exist in this particular uploaded workload.")
if workload is not None and not workload.empty:
    st.dataframe(workload.head(12), use_container_width=True, hide_index=True)
    numeric_cols = workload.select_dtypes(include=np.number).columns.tolist()
    if numeric_cols:
        feature = st.selectbox("Numeric feature", numeric_cols, key="workload_feature")
        series = pd.to_numeric(workload[feature], errors="coerce").dropna()
        p1,p2 = st.columns(2)
        with p1:
            fig = px.histogram(series.to_frame(name=feature), x=feature, nbins=32, title=f"Distribution of {feature}", template=PLOT_TEMPLATE, color_discrete_sequence=["#247c78"])
            fig.update_layout(margin=dict(l=10,r=10,t=60,b=10))
            st.plotly_chart(fig, use_container_width=True, key="epcs_chart_17")
        with p2:
            fig = px.box(series.to_frame(name=feature), y=feature, points="outliers", title=f"Spread and outliers: {feature}", template=PLOT_TEMPLATE, color_discrete_sequence=["#447da2"])
            fig.update_layout(margin=dict(l=10,r=10,t=60,b=10))
            st.plotly_chart(fig, use_container_width=True, key="epcs_chart_18")
    cat_cols = [col for col in workload.columns if (workload[col].dtype == object or str(workload[col].dtype).startswith("category")) and 1 < workload[col].nunique(dropna=True) <= 40]
    if cat_cols:
        cat_feature = st.selectbox("Categorical feature", cat_cols, key="workload_category")
        counts = workload[cat_feature].fillna("Missing").astype(str).value_counts().rename_axis(cat_feature).reset_index(name="Jobs")
        fig = px.bar(counts, x=cat_feature, y="Jobs", color=cat_feature, title=f"Workload composition: {cat_feature}", template=PLOT_TEMPLATE, color_discrete_sequence=EXTRA_COLORS)
        fig.update_layout(showlegend=False, margin=dict(l=10,r=10,t=60,b=10))
        st.plotly_chart(fig, use_container_width=True, key="epcs_chart_19")
    corr_df = workload.select_dtypes(include=np.number)
    if corr_df.shape[1] >= 2:
        corr = corr_df.corr(numeric_only=True)
        fig = px.imshow(corr, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, title="Numeric feature correlation (association, not causation)", template=PLOT_TEMPLATE)
        fig.update_layout(margin=dict(l=10,r=10,t=60,b=10))
        st.plotly_chart(fig, use_container_width=True, key="epcs_chart_20")
        st.markdown("<div class='insight'>• Correlation describes linear association among numeric columns. It cannot by itself prove that one feature causes longer execution time or changes a scheduler decision.</div>", unsafe_allow_html=True)
    # Dataset-specific profile notes
    profile_notes = []
    burst_col = find_col(workload, ["Burst_Time", "Execution_Time(Seconds)", "Execution_Time", "CPU_Time", "Predicted_Execution_Time"])
    if burst_col:
        burst = pd.to_numeric(workload[burst_col], errors="coerce").dropna()
        if len(burst) and abs(float(burst.mean())) > 1e-12:
            cv = float(burst.std() / burst.mean())
            regime = "highly variable" if cv >= 1 else ("moderately variable" if cv >= .5 else "relatively consistent")
            profile_notes.append(f"Execution-time spread: coefficient of variation is {cv:.2f}, indicating a {regime} workload under this simple descriptive rule.")
    if arrival_col := canonical_time_col(workload, "arrival"):
        arr = pd.to_numeric(workload[arrival_col], errors="coerce").dropna()
        if len(arr) > 1:
            gaps = arr.sort_values().diff().dropna()
            profile_notes.append(f"Arrival pattern: median inter-arrival gap is {float(gaps.median()):.2f} time units; this helps contextualise queue build-up but does not prove CPU utilization.")
    for item in profile_notes:
        st.markdown(f"<div class='insight'>• {item}</div>", unsafe_allow_html=True)
else:
    st.warning("scheduler_workload.csv is missing from this ZIP.")


    # Full gallery: every numerical feature is visible in one faceted distribution view.
    all_num_cols = workload.select_dtypes(include=np.number).columns.tolist()
    if all_num_cols:
        st.markdown("### Full feature gallery — every numeric feature")
        st.caption("Small multiples keep every numeric workload field visible together; each panel uses its own x-scale because fields can use different units.")
        melted = workload[all_num_cols].melt(var_name="Feature", value_name="Value").dropna(subset=["Value"])
        if not melted.empty:
            fig_all = px.histogram(
                melted, x="Value", facet_col="Feature", facet_col_wrap=3,
                facet_col_spacing=0.06, facet_row_spacing=0.08,
                nbins=24, title="All numeric workload distributions",
                template=PLOT_TEMPLATE, color_discrete_sequence=["#247c78"],
            )
            fig_all.update_xaxes(matches=None, showgrid=True, gridcolor="#e8edf0")
            fig_all.update_yaxes(matches=None, showgrid=True, gridcolor="#e8edf0")
            fig_all.for_each_annotation(lambda a: a.update(text=a.text.replace("Feature=", "")))
            fig_all.update_layout(
                height=max(420, ((len(all_num_cols) + 2) // 3) * 245),
                showlegend=False, margin=dict(l=10,r=10,t=65,b=10),
            )
            st.plotly_chart(fig_all, use_container_width=True, key="epcs_chart_21")

        stats = workload[all_num_cols].describe().T.reset_index().rename(columns={"index":"Feature"})
        stats["Missing values"] = workload[all_num_cols].isna().sum().values
        st.markdown("**Numerical feature summary**")
        st.dataframe(stats.round(3), use_container_width=True, hide_index=True)

    # Display every useful categorical distribution rather than requiring one dropdown selection.
    all_cat_cols = [
        c for c in workload.columns
        if (workload[c].dtype == object or str(workload[c].dtype).startswith("category"))
        and 1 < workload[c].nunique(dropna=True) <= 40
    ]
    if all_cat_cols:
        st.markdown("### Full feature gallery — every categorical feature")
        for pair_start in range(0, len(all_cat_cols), 2):
            category_cols = st.columns(2)
            for category_col, view_col in zip(all_cat_cols[pair_start:pair_start+2], category_cols):
                with view_col:
                    count_df = workload[category_col].fillna("Missing").astype(str).value_counts().rename_axis(category_col).reset_index(name="Jobs")
                    fig_cat_all = px.bar(
                        count_df, x=category_col, y="Jobs", color=category_col,
                        title=f"Category mix: {category_col}", template=PLOT_TEMPLATE,
                        color_discrete_sequence=EXTRA_COLORS,
                    )
                    fig_cat_all.update_layout(showlegend=False, margin=dict(l=10,r=10,t=60,b=10), xaxis_title="")
                    st.plotly_chart(fig_cat_all, use_container_width=True, key="epcs_chart_22")

# ----------------------------- 11 SHAP -------------------------------------
st.divider()
section("11 / Explainable AI — SHAP", "Global SHAP explains which input features most strongly influence prediction magnitude across evaluated records.")
if shap_df is not None and not shap_df.empty:
    feat_col = find_col(shap_df, ["Feature", "feature_name"])
    importance_col = find_col(shap_df, ["Mean_SHAP_Importance", "Mean SHAP Importance", "Importance", "MeanAbsSHAP"])
    if feat_col and importance_col:
        shap_view = shap_df.copy()
        shap_view[importance_col] = pd.to_numeric(shap_view[importance_col], errors="coerce")
        shap_view = shap_view.dropna(subset=[importance_col]).sort_values(importance_col, ascending=False)
        if not shap_view.empty:
            left,right = st.columns([1,1.25])
            with left:
                st.dataframe(shap_view[[feat_col, importance_col]].round(4), use_container_width=True, hide_index=True)
            with right:
                fig = px.bar(shap_view.sort_values(importance_col), x=importance_col, y=feat_col, orientation="h", title="Global feature importance · mean |SHAP|", template=PLOT_TEMPLATE, color=importance_col, color_continuous_scale=[[0,"#d7e7eb"],[.5,"#75aaa9"],[1,"#245a7a"]])
                fig.update_layout(margin=dict(l=10,r=10,t=60,b=10), xaxis_title="Mean absolute SHAP value", yaxis_title="Feature")
                st.plotly_chart(fig, use_container_width=True, key="epcs_chart_23")
            top = shap_view.iloc[0]
            st.markdown(f"<div class='insight'>• Top global feature by the supplied SHAP importance table: <b>{top[feat_col]}</b>. Mean absolute SHAP reports contribution magnitude, not whether that feature increases or decreases a specific prediction.</div>", unsafe_allow_html=True)
            st.markdown("**Theory notes**")
            st.markdown("- Global importance summarises a collection of predictions.\n- A high score is not proof of causality.\n- To explain one job specifically, use local SHAP values joined to a job ID and the decision log.")
    else:
        st.warning("SHAP file needs feature and importance columns (commonly Feature and Mean_SHAP_Importance).")
else:
    st.info("shap_feature_importance.csv is not in the ZIP. Export it from Kaggle to display SHAP analysis.")
if local_shap is not None and not local_shap.empty:
    st.subheader("Local / per-job explanation records")
    st.dataframe(local_shap.head(100), use_container_width=True, hide_index=True)
else:
    st.caption("Individual-job SHAP and decision logs are not present in the current ZIP. Global feature importance cannot, by itself, explain why a particular job was selected.")

# ----------------------------- 12 Timeline -------------------------------
st.divider()
section("12 / Scheduling timeline", "View the recorded start-to-completion interval for up to the first 100 jobs in an algorithm's result file.")
if available:
    alg_choice = st.selectbox("Timeline scheduler", list(available.keys()), key="timeline_alg")
    timeline = available[alg_choice].copy()
    jid, start_col, end_col = job_id_col(timeline), canonical_time_col(timeline, "start"), canonical_time_col(timeline, "completion")
    if jid and start_col and end_col:
        timeline[start_col] = pd.to_numeric(timeline[start_col], errors="coerce")
        timeline[end_col] = pd.to_numeric(timeline[end_col], errors="coerce")
        timeline = timeline.dropna(subset=[start_col,end_col]).sort_values(start_col).head(100).copy()
        timeline["Duration"] = timeline[end_col] - timeline[start_col]
        timeline["Job label"] = timeline[jid].astype(str)
        timeline = timeline[timeline["Duration"] >= 0]
        if not timeline.empty:
            fig = go.Figure(go.Bar(x=timeline["Duration"], y=timeline["Job label"], base=timeline[start_col], orientation="h", marker_color="#247c78", hovertemplate="Job %{y}<br>Start %{base}<br>Duration %{x}<extra></extra>"))
            fig.update_layout(title=f"{alg_choice} · first 100 recorded intervals", xaxis_title="Time units", yaxis_title="Job", template=PLOT_TEMPLATE, height=620, margin=dict(l=10,r=10,t=60,b=10))
            fig.update_yaxes(autorange="reversed")
            st.plotly_chart(fig, use_container_width=True, key="epcs_chart_24")
        else:
            st.info("No valid timeline rows were found.")
    else:
        st.info("Timeline requires job ID, Start_Time and Completion_Time columns.")
else:
    st.info("No scheduler results are available for timeline plotting.")

# ----------------------------- 13 Research -------------------------------
st.divider()
section("13 / Research ideas & theory", "Three initial decision-intelligence ideas plus three additional ideas. These are proposed EPCS-specific research directions, not verified globally new inventions.")
research_cards = [
    {
        "no":"01", "title":"Counterfactual Job Mirror",
        "problem":"Most summary dashboards say which mean is lower but do not let a reviewer inspect a single job's alternative outcomes.",
        "idea":"For a selected job, compare recorded wait, turnaround, response, start and completion outcomes across algorithms. The next research step is to re-simulate a changed dispatch choice and measure system-wide effects.",
        "measure":"Per-job outcome deltas; whole-schedule makespan; mean and P95 wait.",
        "limit":"Current panel compares stored outcomes; it is not a complete schedule-level counterfactual."
    },
    {
        "no":"02", "title":"Prediction Reliability Guard",
        "problem":"A prediction-driven scheduler can be harmed by large execution-time errors.",
        "idea":"Join predicted and recorded times by job ID; show MAE, high-error cases and predicted-vs-actual plots. A future version can calibrate uncertainty and fall back to a baseline policy for uncertain jobs.",
        "measure":"MAE, RMSE, P90/P95 absolute error, coverage of prediction intervals if calibrated.",
        "limit":"Current version detects observed error; it does not produce calibrated uncertainty intervals."
    },
    {
        "no":"03", "title":"Workload Shift Radar",
        "problem":"Workload mix can change between runs, and previous aggregate performance may not transfer.",
        "idea":"Compare numeric features using standardized mean shift and categorical composition using total variation distance; prompt re-validation when large shifts appear.",
        "measure":"Feature drift, error before/after shift, false-alarm rate.",
        "limit":"Input drift alone does not prove model performance degraded; baseline is the previous ZIP in this session."
    },
    {
        "no":"04", "title":"Fairness & Starvation Radar",
        "problem":"A good average can hide a small number of jobs or job categories suffering very long waits.",
        "idea":"Report P50/P95/max waiting time and group distributions by job type or priority; identify tail-delay jobs for review.",
        "measure":"P95 wait, max wait, group wait gap, Jain's fairness index if a meaningful per-job service metric is defined.",
        "limit":"A finite trace cannot prove or rule out indefinite starvation."
    },
    {
        "no":"05", "title":"Goal-Based Policy Recommender",
        "problem":"There may not be one best scheduler for every use case because waiting, response and turnaround can trade off.",
        "idea":"Let a user select relative weights; normalize metrics on the current dataset and calculate a transparent weighted score. The recommendation changes when the workload or goals change.",
        "measure":"Weighted score, rank stability under changed weights, Pareto-dominated policies.",
        "limit":"The recommendation depends on chosen weights and the available summary metrics; it is not an objective universal winner."
    },
    {
        "no":"06", "title":"Workload Stress & Sensitivity Lab",
        "problem":"A scheduler's ranking on a single static workload may not hold when execution demand or Round Robin time quantum changes.",
        "idea":"Scale workload burst times and sweep Round Robin quantum values, then re-simulate classical baselines to see whether rankings and tail delays change.",
        "measure":"Mean/P95 waiting, turnaround, response and RR context switches across scenarios.",
        "limit":"Current sandbox runs exported workload traces through classical baselines only; EPCS requires the actual model and dispatch code to be replayed fairly."
    },
]

for start in range(0, len(research_cards), 2):
    cols = st.columns(2)
    for col, item in zip(cols, research_cards[start:start+2]):
        with col:
            st.markdown(
                f"<div class='plain-card'><div class='card-title'><span class='number-chip'>{item['no']}</span>{item['title']}</div>"
                f"<b>Problem</b><br>{item['problem']}<br><br>"
                f"<b>Core idea</b><br>{item['idea']}<br><br>"
                f"<b>How to evaluate</b><br>{item['measure']}<br><br>"
                f"<span class='muted'><b>Boundary:</b> {item['limit']}</span></div>",
                unsafe_allow_html=True,
            )
            st.write("")

st.markdown("### Why this dataset can support EPCS analysis")
dataset_reasons = []
if workload is not None and not workload.empty:
    dataset_reasons.append(f"The uploaded workload contains {len(workload):,} records, enabling summary statistics and job-level inspection where IDs match.")
    if canonical_time_col(workload, "arrival"):
        dataset_reasons.append("Arrival-time information is present, which is necessary to reason about which jobs are eligible to be scheduled at a given point in time.")
    if find_col(workload, ["Burst_Time","Execution_Time(Seconds)","Execution_Time","CPU_Time","Predicted_Execution_Time"]):
        dataset_reasons.append("An execution-time field is present, allowing duration-oriented workload profiling and classical baseline simulation (subject to the dataset's documented meaning/units).")
    if canonical_time_col(workload, "priority"):
        dataset_reasons.append("Priority information is present, enabling a priority-based baseline and subgroup inspection.")
if shap_df is not None and not shap_df.empty:
    dataset_reasons.append("A global SHAP feature-importance export is present, so the dashboard can discuss which model features carry the largest average contribution magnitude.")
if len(available) >= 3:
    dataset_reasons.append(f"{len(available)} schedulers have result files in this upload, making a multi-baseline comparison possible.")
if dataset_reasons:
    for reason in dataset_reasons:
        st.markdown(f"<div class='insight'>• {reason}</div>", unsafe_allow_html=True)
else:
    st.info("Add workload and result CSVs to enable dataset-specific reasoning.")

st.markdown("### What would justify saying EPCS is better?")
st.markdown(
    "- **Controlled inputs:** every algorithm must use the same job set, arrival times, units and equivalent execution-time assumptions.\n"
    "- **Measured gains:** the uploaded EPCS metrics must improve the pre-declared objectives; no cherry-picking one metric after looking at the results.\n"
    "- **Fairness check:** average gains should be checked against P95/max waiting time and priority/job-type groups.\n"
    "- **Prediction validity:** evaluate XGBoost on a leakage-safe holdout (preferably chronological for time-ordered workloads).\n"
    "- **Ablation:** compare the complete scheduler against a version without prediction-based dispatch, and explain whether each module actually contributes."
)

st.markdown("### Validation notes")
if workload is not None and epcs is not None:
    st.write(f"- Workload rows: **{len(workload):,}** · EPCS rows: **{len(epcs):,}**")
if epcs is not None and sjf is not None:
    ew, sw = safe_mean(epcs, ["Waiting_Time"]), safe_mean(sjf, ["Waiting_Time"])
    et, stt = safe_mean(epcs, ["Turnaround_Time"]), safe_mean(sjf, ["Turnaround_Time"])
    if np.isfinite(ew) and np.isfinite(sw) and np.isfinite(et) and np.isfinite(stt) and np.isclose(ew, sw, rtol=1e-5, atol=1e-8) and np.isclose(et, stt, rtol=1e-5, atol=1e-8):
        st.warning("EPCS and SJF have equal mean waiting and turnaround times in this upload. Verify job-level schedule order and dispatch logic before claiming a strict gain.")
st.markdown(
    "- The current app visualizes uploaded CSV results; it does not load/retrain the XGBoost pipeline.\n"
    "- Previously reported notebook metrics (MAE 43.55 s, RMSE 55.55 s, R² 0.9898) are not re-evaluated here and are intentionally not presented as freshly computed metrics.\n"
    "- Research directions such as counterfactual explanations, uncertainty-aware prediction and drift monitoring already exist in the broader literature. Any claim of novelty must be validated against current papers and patents."
)

with st.expander("Reference leads for the literature review"):
    st.markdown(
        "- Mehdiyev, Majlatow & Fettke (2024), *Counterfactual Explanations in the Big Picture: An Approach for Process Prediction-Driven Job-Shop Scheduling Optimization*, Cognitive Computation. [Publisher page](https://link.springer.com/article/10.1007/s12559-024-10294-0)\n"
        "- Löfström et al. (2024), *Calibrated explanations: With uncertainty information and counterfactuals*, Expert Systems with Applications. [Publisher page](https://www.sciencedirect.com/science/article/pii/S0957417424000198)\n"
        "- KDD 2024, *Explanatory Model Monitoring to Understand the Effects of Feature Shifts on Performance*. [Proceedings](https://doi.org/10.1145/3637528.3671959)\n"
        "- USENIX OSDI 2024, *When will my ML Job finish? Toward providing Completion Time Estimates through Predictability-Centric Scheduling*. [USENIX page](https://www.usenix.org/conference/osdi24/presentation/bin-faisal)"
    )

st.divider()
st.markdown(
    "<div style='display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap;color:#526776;font-size:.85rem;'>"
    "<span><b>EPCS — Explainable AI CPU Scheduler</b> · Academic prototype</span>"
    "<span>CREATED BY MANGAL SAKTHI</span></div>",
    unsafe_allow_html=True,
)
