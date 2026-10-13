# EPCS — Explainable AI CPU Scheduler

This project package contains a Streamlit dashboard source and a research/validation report draft for the selected topic **Explainable AI CPU Scheduler (XGBoost + SHAP)**.

## What is included

- `app.py` — complete dashboard source for the exported CSV files.
- `requirements.txt` — Python dependencies.
- `PROJECT_REPORT.md` — project status, research survey, gap analysis, architecture options, evaluation plan, and remaining-work checklist.

The package does not include the user's private/local `dashboard_data.zip` or the trained model file. Upload the ZIP you already exported from Kaggle when opening the dashboard.

## Run on Windows

1. Install Python 3.10+ and enable **Add Python to PATH** during setup.
2. Open this folder in VS Code.
3. Open Terminal → New Terminal and run:
   ```bash
   python -m pip install -r requirements.txt
   python -m streamlit run app.py
   ```
4. Open the local URL shown in the terminal (normally `http://localhost:8501`).
5. In the dashboard sidebar, upload the latest `dashboard_data.zip`.

## Expected CSV filenames

- `epcs_results.csv`
- `fcfs_results.csv`
- `sjf_results_713.csv`
- `priority_results.csv`
- `rr_results.csv`
- `scheduler_workload.csv`
- `shap_feature_importance.csv`

Optional for per-job explanations: `shap_local_explanations.csv`.

## Important scientific boundary

The dashboard displays the previously recorded model metrics MAE=43.55 seconds, RMSE=55.55 seconds, and R²=0.9898 as reported from the current notebook. It does not independently recompute them or perform live inference. To enable live inference, export the fitted preprocessing pipeline and trained XGBoost model and then wire them into an explicit prediction form.

The global SHAP file summarizes overall feature importance. A global ranking alone does not explain a specific dispatch decision. For the full “why was this job selected?” objective, export local SHAP contributions for each job and log the candidate set and decision rule at each dispatch.
