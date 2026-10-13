import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import zipfile
import io

# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Explainable AI CPU Scheduler",
    page_icon="⚙️",
    layout="wide"
)

st.title("⚙️ Explainable AI CPU Scheduler")
st.caption(
    "XGBoost Execution-Time Prediction | SHAP Explainability "
    "| Intelligent CPU Scheduling"
)

st.markdown("---")


# =====================================================
# LOAD DASHBOARD DATA
# =====================================================

st.sidebar.header("📂 Dashboard Data")

uploaded_zip = st.sidebar.file_uploader(
    "Upload dashboard_data.zip",
    type=["zip"]
)

if uploaded_zip is None:
    st.info(
        "Upload your updated dashboard_data.zip to display "
        "the scheduling comparison and SHAP analysis."
    )
    st.stop()


@st.cache_data
def load_csv_files(zip_bytes):
    data = {}

    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as archive:
        for filename in archive.namelist():

            if filename.lower().endswith(".csv"):
                try:
                    with archive.open(filename) as file:
                        data[filename.split("/")[-1]] = pd.read_csv(file)
                except Exception as error:
                    st.warning(
                        f"Could not read {filename}: {error}"
                    )

    return data


files = load_csv_files(uploaded_zip.getvalue())


def find_file(keyword):
    """Find a CSV using part of its filename."""
    for filename, dataframe in files.items():
        if keyword.lower() in filename.lower():
            return dataframe

    return None


# =====================================================
# LOAD INDIVIDUAL DATASETS
# =====================================================

epcs_df = files.get("epcs_results.csv")
fcfs_df = files.get("fcfs_results.csv")
sjf_df = files.get("sjf_results_713.csv")
priority_df = files.get("priority_results.csv")
rr_df = files.get("rr_results.csv")
workload_df = files.get("scheduler_workload.csv")
shap_df = files.get("shap_feature_importance.csv")


# =====================================================
# SIDEBAR INFORMATION
# =====================================================

st.sidebar.markdown("---")
st.sidebar.subheader("Available Files")

for filename in files:
    st.sidebar.write("✅ " + filename)


# =====================================================
# PROJECT OVERVIEW
# =====================================================

st.header("1. Project Overview")

c1, c2, c3 = st.columns(3)

c1.metric(
    "Scheduling Algorithms",
    sum(
        df is not None
        for df in [
            fcfs_df, sjf_df, priority_df, rr_df, epcs_df
        ]
    )
)

c2.metric(
    "SHAP Analysis",
    "Available" if shap_df is not None else "Missing"
)

c3.metric(
    "Workload Dataset",
    f"{len(workload_df):,} jobs"
    if workload_df is not None
    else "Not loaded"
)

st.markdown(
    """
    This project predicts job execution time using XGBoost and
    uses explainable AI techniques to interpret predictions.

    The scheduling results are compared using waiting time,
    turnaround time and response time.
    """
)

st.markdown("---")


# =====================================================
# MODEL PERFORMANCE
# =====================================================

st.header("2. XGBoost Model Performance")

st.write(
    "Previously measured model metrics from the notebook:"
)

m1, m2, m3 = st.columns(3)

m1.metric("MAE", "43.55 seconds")
m2.metric("RMSE", "55.55 seconds")
m3.metric("R² Score", "0.9898")

st.caption(
    "These are the previously recorded evaluation results. "
    "They are not recalculated from the uploaded CSV files."
)

st.markdown("---")


# =====================================================
# SCHEDULER METRIC CALCULATION
# =====================================================

st.header("3. Scheduling Algorithm Comparison")

scheduler_datasets = {
    "FCFS": fcfs_df,
    "SJF": sjf_df,
    "Priority": priority_df,
    "Round Robin": rr_df,
    "EPCS": epcs_df
}

metric_columns = {
    "Average Waiting Time": "Waiting_Time",
    "Average Turnaround Time": "Turnaround_Time",
    "Average Response Time": "Response_Time"
}

comparison_rows = []

for algorithm, dataframe in scheduler_datasets.items():

    if dataframe is None:
        continue

    row = {
        "Algorithm": algorithm,
        "Jobs": len(dataframe)
    }

    for metric_name, column_name in metric_columns.items():

        if column_name in dataframe.columns:
            values = pd.to_numeric(
                dataframe[column_name],
                errors="coerce"
            )

            row[metric_name] = values.mean()

        else:
            row[metric_name] = np.nan

    comparison_rows.append(row)


if comparison_rows:

    comparison_df = pd.DataFrame(comparison_rows)

    st.subheader("Average Performance Metrics")

    st.dataframe(
        comparison_df.round(2),
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Compare Scheduling Algorithms")

    selected_metric = st.selectbox(
        "Select performance metric",
        list(metric_columns.keys())
    )

    chart_data = comparison_df.dropna(
        subset=[selected_metric]
    )

    if not chart_data.empty:

        fig = px.bar(
            chart_data,
            x="Algorithm",
            y=selected_metric,
            color="Algorithm",
            text_auto=".2f",
            title=f"{selected_metric} Comparison",
            template="plotly_white"
        )

        fig.update_layout(
            showlegend=False,
            xaxis_title="Scheduling Algorithm",
            yaxis_title="Average Time"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.subheader("All Metrics Comparison")

    long_df = comparison_df.melt(
        id_vars=["Algorithm"],
        value_vars=list(metric_columns.keys()),
        var_name="Metric",
        value_name="Average Time"
    )

    long_df = long_df.dropna(subset=["Average Time"])

    if not long_df.empty:

        fig_all = px.bar(
            long_df,
            x="Algorithm",
            y="Average Time",
            color="Metric",
            barmode="group",
            title="Scheduling Performance Overview",
            template="plotly_white"
        )

        st.plotly_chart(
            fig_all,
            use_container_width=True
        )

else:

    st.warning(
        "No scheduling results found. Check the CSV filenames "
        "inside dashboard_data.zip."
    )


st.markdown("---")


# =====================================================
# EXPLAINABLE AI — SHAP
# =====================================================

st.header("4. Explainable AI — SHAP")

st.write(
    "SHAP feature importance explains which input features "
    "contribute most to the XGBoost model's execution-time predictions."
)

if shap_df is not None:

    if {
        "Feature",
        "Mean_SHAP_Importance"
    }.issubset(shap_df.columns):

        shap_df = shap_df.copy()

        shap_df["Mean_SHAP_Importance"] = pd.to_numeric(
            shap_df["Mean_SHAP_Importance"],
            errors="coerce"
        )

        shap_df = shap_df.dropna(
            subset=["Mean_SHAP_Importance"]
        )

        shap_df = shap_df.sort_values(
            "Mean_SHAP_Importance",
            ascending=False
        )

        st.subheader("Feature Importance Table")

        st.dataframe(
            shap_df.round(4),
            use_container_width=True,
            hide_index=True
        )

        st.subheader("SHAP Feature Importance Chart")

        shap_chart_df = shap_df.sort_values(
            "Mean_SHAP_Importance",
            ascending=True
        )

        fig_shap = px.bar(
            shap_chart_df,
            x="Mean_SHAP_Importance",
            y="Feature",
            orientation="h",
            title="Mean Absolute SHAP Importance",
            labels={
                "Mean_SHAP_Importance": "Mean |SHAP value|",
                "Feature": "Input Feature"
            },
            template="plotly_white"
        )

        st.plotly_chart(
            fig_shap,
            use_container_width=True
        )

        if not shap_df.empty:

            top_feature = shap_df.iloc[0]

            st.success(
                f"Most influential feature in the current SHAP "
                f"ranking: {top_feature['Feature']}."
            )

        st.caption(
            "Mean absolute SHAP importance measures the average "
            "magnitude of a feature's contribution. It does not "
            "show whether that feature increases or decreases "
            "the prediction."
        )

    else:

        st.error(
            "SHAP CSV is missing the required columns: "
            "Feature and Mean_SHAP_Importance."
        )

else:

    st.warning(
        "shap_feature_importance.csv was not found in the ZIP."
    )


st.markdown("---")


# =====================================================
# WORKLOAD ANALYSIS
# =====================================================

st.header("5. Workload Analysis")

if workload_df is not None:

    st.subheader("Workload Dataset")

    st.write(f"Total workload records: {len(workload_df):,}")

    st.dataframe(
        workload_df.head(20),
        use_container_width=True
    )

    numeric_columns = workload_df.select_dtypes(
        include=np.number
    ).columns.tolist()

    if numeric_columns:

        selected_feature = st.selectbox(
            "Select a numeric workload feature",
            numeric_columns
        )

        feature_df = workload_df[
            [selected_feature]
        ].dropna()

        fig_hist = px.histogram(
            feature_df,
            x=selected_feature,
            nbins=30,
            title=f"Distribution of {selected_feature}",
            template="plotly_white"
        )

        st.plotly_chart(
            fig_hist,
            use_container_width=True
        )

        fig_box = px.box(
            feature_df,
            y=selected_feature,
            title=f"Distribution and Outliers: {selected_feature}",
            template="plotly_white"
        )

        st.plotly_chart(
            fig_box,
            use_container_width=True
        )

else:

    st.warning("scheduler_workload.csv was not found.")


st.markdown("---")


# =====================================================
# JOB EXPLORER
# =====================================================

st.header("6. Job Explorer")

available_job_data = {
    name: dataframe
    for name, dataframe in scheduler_datasets.items()
    if dataframe is not None
}

if available_job_data:

    selected_algorithm = st.selectbox(
        "Choose a scheduler",
        list(available_job_data.keys()),
        key="job_explorer_algorithm"
    )

    job_df = available_job_data[selected_algorithm].copy()

    st.write(
        f"Showing results for {selected_algorithm}: "
        f"{len(job_df):,} jobs"
    )

    search_column = next(
        (
            col for col in [
                "Process_ID",
                "Job_ID",
                "Job_ID",
                "Process_ID"
            ]
            if col in job_df.columns
        ),
        None
    )

    if search_column:

        search_term = st.text_input(
            "Search for a job ID"
        )

        if search_term.strip():

            filtered_df = job_df[
                job_df[search_column]
                .astype(str)
                .str.contains(
                    search_term.strip(),
                    case=False,
                    na=False
                )
            ]

        else:

            filtered_df = job_df

    else:

        filtered_df = job_df

    st.dataframe(
        filtered_df.head(200),
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "Download displayed job results",
        data=filtered_df.to_csv(index=False).encode("utf-8"),
        file_name=f"{selected_algorithm.lower().replace(' ', '_')}_results.csv",
        mime="text/csv"
    )

else:

    st.warning("No job-level scheduling results are available.")


st.markdown("---")


# =====================================================
# FOOTER
# =====================================================

st.caption(
    "Explainable AI CPU Scheduler | XGBoost + SHAP | "
    "Scheduling Performance Dashboard"
)

st.caption(
    "Interpret algorithm comparisons only after verifying "
    "that every scheduler used the same workload and consistent "
    "arrival, burst, priority and scheduling assumptions."
)