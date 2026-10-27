# EPCS Decision Intelligence Dashboard update

## Included
- `app.py`: upgraded Streamlit dashboard.
- `requirements.txt`: runtime dependencies.

## Deploy on GitHub + Streamlit Community Cloud
1. Back up your current `app.py`.
2. Replace the repository's existing `app.py` with this version and commit the change. Keep the existing `requirements.txt` or replace it with the included one if it contains no extra required dependencies.
3. Open the deployed Streamlit app and upload your existing `dashboard_data.zip`.
4. Confirm the overview, comparison charts, SHAP graph and timeline display.
5. To test Workload Shift Radar, upload a *different* ZIP in the same browser session. The comparison baseline is session-only; it resets when the app session restarts.

## New modules
1. **Counterfactual Job Mirror**: compares the stored outcomes for the selected job across existing scheduler result CSVs. This is a cross-algorithm comparison, not a full re-simulation after changing a dispatch decision.
2. **Prediction Reliability Guard**: compares `Predicted_Execution_Time` in `epcs_results.csv` with recorded `Burst_Time` (or a supported recorded-time field) in `scheduler_workload.csv`, matched on job ID. It displays retrospective error, not calibrated predictive confidence. It skips the feature if columns or matching IDs are unavailable.
3. **Workload Shift Radar**: compares shared numeric and categorical feature distributions between the current and previous ZIP uploaded in the same session. It indicates possible shift; it does not prove model degradation.

## Expected ZIP contents
At minimum: `epcs_results.csv`, `fcfs_results.csv`, `sjf_results_713.csv`, `priority_results.csv`, `rr_results.csv`, `scheduler_workload.csv`, and `shap_feature_importance.csv`.

## Scientific boundaries
- Dashboard metrics and interpretations are computed from exported CSVs, not from fresh model inference.
- Static XGBoost metrics (MAE=43.55s, RMSE=55.55s, R²=0.9898) are preserved as previously reported values and are labelled as such.
- Use the same jobs and assumptions across scheduling algorithms.
- Do not claim EPCS is better unless the measured results show it.
- These are project-specific integration ideas, not a guarantee of globally novel inventions. Literature and patent searches are needed before making a patent novelty claim.
