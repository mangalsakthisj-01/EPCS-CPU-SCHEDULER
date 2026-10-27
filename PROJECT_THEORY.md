# EPCS project theory and research directions
**Project:** Explainable AI CPU Scheduler (EPCS)  
**Created by:** Mangal Sakthi

## 1. Project purpose
- **Problem:** Traditional schedulers use different fixed rules. A single average can hide metric trade-offs and long-waiting jobs.
- **Project approach:** Use a workload dataset, precomputed XGBoost execution-time predictions, traditional scheduler results, SHAP global feature importance, and a dashboard that makes the evidence understandable.
- **Current dashboard boundary:** The app visualizes exported CSVs. It does not load or retrain the trained XGBoost pipeline.
- **What counts as “EPCS is best”:** The uploaded results must show that EPCS wins on predeclared metrics under the same process set, arrivals, units and comparable execution-time assumptions. The app will not force a positive conclusion.

## 2. Theory behind the performance metrics
- **Waiting time:** total time a job spends waiting in the ready queue. Lower mean waiting indicates less average queue delay, but can hide tail-delay jobs.
- **Turnaround time:** completion time minus arrival time. It includes execution plus waiting, so it reflects end-to-end time to finish.
- **Response time:** first CPU start minus arrival time. It measures how soon a job receives its first service.
- **P95 waiting / maximum waiting:** tail metrics that reveal jobs that wait much longer than most jobs.
- **Throughput:** jobs completed per unit elapsed time, provided all compared algorithms use the same interval and job set.
- **Context switches:** important in a Round Robin simulation because smaller quantum often increases process switches. The dashboard stress lab reports this for simulated RR if available.
- **R² is not accuracy percent:** R²=0.9898 must not be described as “98.98% prediction accuracy.” It is a fit statistic on the particular evaluated test set.

## 3. How the dashboard explains the uploaded dataset
The verdict is dataset-specific:
1. Calculate the mean waiting, turnaround and response metrics from each algorithm’s result file.
2. Compare the EPCS mean with the best and next-best baselines.
3. Report percent difference only where a valid baseline value exists.
4. Mark each metric as **Best**, **Tied first**, or **Not best**.
5. Verify job counts and job IDs; warn if algorithms do not share the same jobs.
6. Explain what each metric means and what the observed result supports.
7. Avoid saying that EPCS is universally best when the current upload only supports a metric-specific finding.

## 4. Six research directions — theory, reason and evaluation

### Idea 1 — Counterfactual Job Mirror
**Problem**
- A summary comparison says which scheduler has a lower average but does not show a specific job's alternatives.

**Core idea**
- Choose a job ID and compare its saved waiting, turnaround, response, start and completion results across algorithms.
- Future research version: change one feasible dispatch decision, re-simulate the remaining schedule and calculate both job-level and whole-system impact.

**Why useful**
- Makes a scheduler result easier for a student, evaluator or system operator to inspect.
- Highlights trade-offs; the lowest result for one job may not improve the entire schedule.

**Evaluation**
- Per-job metric difference, whole-system makespan, mean/P95 waiting and feasibility of the alternative.

**Current limitation**
- The current dashboard compares existing stored rows; this is not yet a complete schedule-level counterfactual re-simulation.

### Idea 2 — Prediction Reliability Guard
**Problem**
- Prediction-driven scheduling depends on execution-time estimates. A large error can mis-rank jobs.

**Core idea**
- Join `Predicted_Execution_Time` to recorded actual execution/burst time by the same job ID.
- Plot absolute-error distribution, predicted-versus-recorded scatter, MAE and P90/P95 error.
- Mark high-error jobs for investigation; future work can add calibrated uncertainty intervals and a safe fallback policy.

**Why useful**
- Prevents a dashboard from treating every prediction as equally reliable.
- Provides an operational review path for jobs where predictions are inaccurate.

**Evaluation**
- MAE, RMSE, P90/P95 absolute error, error by job type, fallback rate and scheduling metrics with/without the guard.

**Current limitation**
- Retrospective error monitoring is not a calibrated confidence measure. The section is skipped if reliable ID/time matching is unavailable.

### Idea 3 — Workload Shift Radar
**Problem**
- Workloads can change after model training; historical performance might not transfer.

**Core idea**
- Compare numerical columns across two uploads using standardized mean difference.
- Compare category-mix changes using total variation distance.
- Highlight features with the largest shifts and prompt revalidation.

**Why useful**
- Helps answer “what changed in this dataset?” rather than just showing a new chart.
- Creates a repeatable review step before trusting old model results on a different workload.

**Evaluation**
- Drift score, model error before/after shift, false alarm rate and time to detect.

**Current limitation**
- The current baseline is the previous different ZIP in the same browser session. Distribution shift alone does not prove model degradation.

### Idea 4 — Fairness & Starvation Radar
**Problem**
- Low average waiting time can coexist with poor service for a small subset of jobs, a job type or a low-priority group.

**Core idea**
- Show waiting-time histogram, median, P95, maximum and group-level wait distributions.
- Identify high-tail jobs and inspect job-type/priority disparities.
- Future work can test ageing, bounded waiting, or priority boosts, then measure their effect against the original policy.

**Why useful**
- Adds a service-quality view that averages miss.
- Helps inspect whether the scheduler favours particular categories.

**Evaluation**
- Mean/P95/max waiting, group waiting-time gaps, starvation events in long-running simulations, and Jain's fairness index only when the chosen service/allocation definition is justified.

**Current limitation**
- A finite dataset cannot prove indefinite starvation. Group differences are descriptive and are not proof of policy bias without controlled experiments.

### Idea 5 — Goal-Based Policy Recommender
**Problem**
- There is no universal scheduler winner for every environment. Low turnaround may trade off against response time or tail delay.

**Core idea**
- Let the evaluator set weights for waiting, turnaround and response.
- Normalize the metrics over the current set of algorithms and calculate a transparent weighted score.
- Show the ranking and how it changes when user priorities change.

**Why useful**
- Forces the project to define the optimization objective.
- Makes the recommendation understandable and reproducible instead of presenting a hidden “best” label.

**Evaluation**
- Weighted ranking, ranking stability across weights, Pareto comparison and outcomes on an unseen dataset.

**Current limitation**
- Results depend on selected weights and available summary metrics. A weighted rank is a decision aid, not an objective universal winner.

### Idea 6 — Workload Stress & Sensitivity Lab
**Problem**
- A policy may look effective on one static trace but change behaviour when execution demand or time quantum changes.

**Core idea**
- Scale recorded burst times over a range.
- Re-simulate FCFS, non-preemptive SJF, non-preemptive Priority and Round Robin.
- Sweep Round Robin quantum relative to median burst and compare average response/wait/turnaround plus context switches.

**Why useful**
- Tests if policy rankings are stable or sensitive to workload assumptions.
- Provides a controlled way to demonstrate the classic response-time/context-switch trade-off for Round Robin.

**Evaluation**
- Mean and P95 waiting, turnaround, response, context switches, rank changes across scenarios.

**Current limitation**
- The current sandbox simulates classical baselines only. EPCS must be re-run through the actual trained model and its dispatch code for an apples-to-apples EPCS stress experiment.

## 5. What would genuinely support the claim “EPCS is best for this dataset”?
- Same jobs and job IDs for all algorithm comparisons.
- Same arrival times, units and documented execution-time definition.
- Predictive model evaluated on a leakage-safe split; chronological splitting is preferable for time-ordered workload traces.
- EPCS's predicted times demonstrably influence job selection.
- All relevant metrics disclosed, including P95/max waiting and fairness trade-offs.
- Multiple runs or scenarios and confidence intervals where randomization exists.
- Ablation results showing the contribution of the predictor, explanation/guard and policy logic.
- No hard-coded claim: if EPCS ties or loses, report that transparently.

## 6. Novelty statement — what can be claimed responsibly?
The broad concepts of counterfactual scheduling explanations, uncertainty-aware explanations and explaining model behaviour under feature shift already exist in research. The project's potential contribution is a **specific, transparent integration and evaluation** of:
- job-level cross-scheduler inspection,
- error-tail monitoring,
- session-based workload comparison,
- tail-delay/fairness diagnostics,
- goal-weighted policy ranking, and
- controlled scheduling sensitivity tests
within one explainable OS scheduling dashboard.

This is a proposed contribution, not a guarantee that the architecture is new worldwide. A novelty claim requires a current literature and patent search, comparison with closest related work, and experiments that show the modules add measurable value.

## 7. Research starting points
- Mehdiyev, Majlatow & Fettke (2024), *Counterfactual Explanations in the Big Picture: An Approach for Process Prediction-Driven Job-Shop Scheduling Optimization*, Cognitive Computation. https://link.springer.com/article/10.1007/s12559-024-10294-0
- Löfström et al. (2024), *Calibrated explanations: With uncertainty information and counterfactuals*, Expert Systems with Applications. https://www.sciencedirect.com/science/article/pii/S0957417424000198
- KDD 2024, *Explanatory Model Monitoring to Understand the Effects of Feature Shifts on Performance*. https://doi.org/10.1145/3637528.3671959
- USENIX OSDI 2024, *When will my ML Job finish? Toward providing Completion Time Estimates through Predictability-Centric Scheduling*. https://www.usenix.org/conference/osdi24/presentation/bin-faisal
