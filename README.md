# EPCS — Explainable AI CPU Scheduler
**Created by Mangal Sakthi**

A Streamlit dashboard for analysing exported CPU scheduling results and explaining workload-specific trade-offs.

## Included files
- `app.py` — dashboard source.
- `requirements.txt` — Python dependencies.
- `PROJECT_THEORY.md` — project theory, six proposed research directions, evaluation plan, and limitations.

## Dashboard sections
1. Workload overview and dataset-specific verdict.
2. Scheduler summary table and comparison charts.
3. Previously reported XGBoost model metrics (labelled as historical, not recomputed).
4. Counterfactual Job Mirror — comparison of one job's saved outcomes across available schedulers.
5. Prediction Reliability Guard — prediction error diagnostics when IDs and recorded execution time can be matched.
6. Workload Shift Radar — comparison with the prior ZIP uploaded in the same browser session.
7. Fairness & Starvation Radar — waiting-time distribution, tail delay and group-level comparisons.
8. Goal-Based Policy Recommender — transparent weighted ranking from user-selected objective priorities.
9. Workload Stress & Sensitivity Lab — simulated classical baselines and Round Robin quantum sensitivity.
10. Workload profiles — selected-feature histograms/box plots plus a full gallery of every numeric feature, every useful categorical feature, numerical summary table and correlation heatmap.
11. Global SHAP importance.
12. Scheduling timeline.
13. Research idea cards and theory.

## Expected ZIP files
At minimum, include as many of these as available:
- `epcs_results.csv`
- `fcfs_results.csv`
- `sjf_results_713.csv` (or `sjf_results.csv`)
- `priority_results.csv`
- `rr_results.csv`
- `scheduler_workload.csv`
- `shap_feature_importance.csv`

The app skips panels when their required columns are missing and explains the limitation.

## Deploy / update Streamlit Community Cloud
1. Keep a backup of your current `app.py`.
2. On GitHub, replace the repository's `app.py` with this file.
3. Confirm that `requirements.txt` includes the dependencies above; commit both files.
4. Wait for Streamlit Community Cloud to rebuild the app.
5. Open the public URL and upload `dashboard_data.zip`.
6. Test the counterfactual section by selecting a job. To activate Workload Shift Radar, upload a different ZIP in the same browser session. The comparison baseline is in-session only.
7. Verify the stress lab uses a valid workload `Arrival_Time` and `Burst_Time` or a documented execution-time field.

## Important scientific boundaries
- The dashboard computes results from the uploaded CSVs, not from live model inference.
- The MAE 43.55 s, RMSE 55.55 s, and R² 0.9898 values are displayed as prior notebook metrics; they are not recalculated by this dashboard.
- The “Why EPCS?” notes are generated from measured metric comparisons. The app will explicitly state when EPCS ties or loses.
- The stress lab simulates FCFS, non-preemptive SJF, non-preemptive Priority and Round Robin. It does not re-run EPCS without the trained model artifact and exact EPCS dispatch logic.
- A distribution shift is not proof of model degradation. Prediction-error summaries are retrospective and not calibrated confidence estimates.
- Novelty directions are hypotheses to evaluate, not proof of unprecedented invention. Do a thorough literature and patent search before claiming novelty or filing a patent.
