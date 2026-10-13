# EPCS — Explainable AI CPU Scheduler

**Project type:** Prototype scheduler + model evaluation + explanation dashboard  
**Core techniques:** XGBoost regression and SHAP  
**Interface:** Streamlit  
**Status:** Working dashboard prototype; research-grade validation and per-decision explanations remain incomplete.

---

## Executive summary

The project predicts job execution time from workload attributes, uses the prediction as an input to a scheduler, compares scheduling outcomes with classical baselines, and presents the outputs with a Streamlit dashboard. The selected topic in the supplied project list explicitly expects a “Scheduler + explanation dashboard” and describes XGBoost + SHAP as its techniques. The same supplied document also asks for a critical literature review, genuine research gaps, multiple architectural directions, novelty validation, an architecture selection matrix, baseline comparisons, and an ablation plan. This report covers those research-planning components without claiming that a new research contribution has already been experimentally proven.

### Results currently reported from the notebook/screenshots

- Dataset table reported in the conversation: **3,562 rows × 15 columns**.
- Scheduling workload used in the latest dashboard: **713 jobs**.
- XGBoost execution-time prediction metrics reported from the existing test evaluation: **MAE = 43.55 seconds, RMSE = 55.55 seconds, R² = 0.9898**. These are recorded values, not independently recalculated by the supplied Streamlit app.
- Global SHAP ranking currently starts with requested memory, MPI job type, and node count. This is a **global prediction explanation**, not a complete explanation of a particular dispatch decision.
- Screenshot metrics: EPCS mean waiting time **94.72**, mean turnaround time **1207.81**; FCFS mean waiting time **98.60**, mean turnaround time **1211.69**; SJF mean waiting time **94.72**, mean turnaround time **1207.81**; Priority mean waiting time **95.64**, mean turnaround time **1208.72**. The exact time unit needs to be confirmed from the dataset documentation. Round Robin output exists as a CSV according to the ZIP listing, but its metric values are not legible in the screenshots and are not invented here.

The current table shows EPCS and SJF have the same displayed waiting and turnaround means. This is not automatically a bug, but it means the current evidence **does not demonstrate that EPCS improves performance over SJF**. Check their per-job ordering, candidate selection logic, and metric calculation before making a performance-improvement claim.

---

## 1. Research problem

### Problem statement

Traditional CPU scheduling baselines such as FCFS, SJF, Priority, and Round Robin are easy to implement and interpret, but their fixed rules may not use workload information to estimate the execution duration of an arriving job. Conversely, a learned execution-time estimator can be inaccurate under workload drift or unfamiliar job types, and a prediction alone does not decide which ready job to dispatch. A scheduler that uses a learned runtime estimate also needs an auditable explanation of why the chosen job was selected.

**Research question:** Can a prediction-aware CPU scheduler provide useful scheduling decisions while producing faithful, understandable explanations for both its execution-time predictions and individual dispatch decisions, without unacceptable overhead or regressions in fairness and tail latency?

### Objectives

1. Predict observed job execution time from workload features.
2. Measure predictive performance on a defensible held-out test set.
3. Use the prediction in a clearly defined scheduler policy.
4. Compare EPCS with FCFS, SJF, Priority, and Round Robin using the same 713-job workload and clearly documented assumptions.
5. Explain both global model behaviour and individual job/dispatch decisions.
6. Quantify the runtime and memory overhead introduced by prediction and explanations.
7. Test whether the proposed scheduler remains useful when the workload distribution changes.

### Current project boundary

The current prototype is a **workload-trace simulator and dashboard**, not a modified operating-system kernel scheduler. Its results should be described as simulation/offline evaluation unless an actual OS integration has been implemented separately.

---

## 2. Literature survey and critical comparison

The studies below are a starting survey, not a claim of an exhaustive systematic review. Before submitting a novelty claim, expand the search across ACM Digital Library, IEEE Xplore, SpringerLink, ScienceDirect, and Google Scholar using defined inclusion/exclusion criteria.

### 2.1 XGBoost for structured workload prediction

**Chen and Guestrin (2016), “XGBoost: A Scalable Tree Boosting System,” KDD 2016.** The paper presents a scalable gradient-boosted tree system with engineering features for sparsity, weighted quantile sketching, and efficient computation. It is foundational for tabular prediction and motivates XGBoost as a practical estimator for mixed workload features. Its contribution is a learning system, not a CPU scheduling policy or a job-selection explanation. Strong prediction accuracy therefore does not establish scheduling quality. [Paper/DOI](https://doi.org/10.1145/2939672.2939785)

### 2.2 SHAP and local/global model explanation

**Lundberg and Lee (2017), “A Unified Approach to Interpreting Model Predictions,” NeurIPS 2017.** SHAP frames feature attribution through a unified additive explanation framework and gives feature attribution values for individual predictions. Aggregating absolute attributions can describe global feature importance, while per-instance SHAP values can explain the components of one prediction. However, a feature’s contribution to a predicted runtime is not identical to a reason for the scheduler choosing a job: the latter also depends on which jobs were ready, the policy’s ranking rule, tie-breaking, priorities, and any constraints. [Paper](https://proceedings.neurips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html)

### 2.3 Deep reinforcement learning for cloud resource scheduling

**Zhou et al. (2024), “Deep reinforcement learning-based methods for resource scheduling in cloud computing: a review and future directions,” Artificial Intelligence Review, 57:124.** This review organizes DRL-based cloud scheduling methods and discusses adaptability to dynamic/high-dimensional states. It also discusses limitations involving computational cost, explainability, convergence, and deployment in realistic systems. This supports the need to evaluate the overhead and operational assumptions of learning-based schedulers instead of reporting only average waiting time. The study reviews a broad cloud-resource-scheduling setting, so its conclusions cannot be taken as a direct benchmark for this 713-job CPU trace. [DOI](https://doi.org/10.1007/s10462-024-10756-9)

### 2.4 AI-driven scheduling in cloud computing

**Sanjalawe et al. (2025), “AI-driven job scheduling in cloud computing: a comprehensive review,” Artificial Intelligence Review, 58:197.** This review discusses current AI-driven scheduling and identifies continuing issues such as latency constraints, multi-objective trade-offs, security, and explainability/auditability. It provides context for studying explanations alongside performance, rather than treating XAI as a decorative chart. Because a review spans diverse systems, its claims should guide hypotheses, not substitute for experiments on the selected dataset. [Article](https://link.springer.com/article/10.1007/s10462-025-11208-8)

### 2.5 Explainable reinforcement learning for scheduling

**“Explainable reinforcement learning in job-shop scheduling: A systematic literature review” (Procedia CIRP, 2025).** This review reports that explainability of RL decisions is still infrequently addressed in the job-shop-scheduling literature and calls for more comprehensive empirical analysis. It is useful evidence that action-level explanation merits investigation. However, manufacturing job-shop scheduling is not the same task as independent CPU job scheduling, so the research gap needs to be validated in CPU/cloud scheduler literature specifically. [DOI](https://doi.org/10.1016/j.procir.2025.01.005)

### Critical synthesis

| Approach | What it contributes | Why it is useful here | Important limitation for this project |
|---|---|---|---|
| Fixed heuristic schedulers | Deterministic, low-overhead reference policies | Essential baselines; behaviour is easy to explain | Fixed assumptions may ignore workload-specific patterns; performance is workload-dependent |
| XGBoost runtime regression | Efficient prediction on structured tabular features | Predicting runtime can improve a duration-aware decision rule | Accurate prediction alone does not define or guarantee a good scheduler |
| Global SHAP importance | Average feature influence across an evaluation set | Shows which encoded features matter most to runtime predictions | Does not explain a particular scheduling choice or show contribution direction when only mean absolute values are exported |
| Per-instance SHAP | Attribution for a single prediction | Can explain why a job’s runtime estimate differs from the baseline prediction | Still does not explain the scheduler’s choice among the ready queue without decision-context logging |
| DRL scheduling | Can learn adaptive policies in dynamic state spaces | Possible future baseline for dynamic workloads | Adds training complexity, stochasticity, and explanation challenges; may be unnecessary for a small offline trace |
| Explainable RL scheduling | Attempts to interpret action selection | Directly relevant to transparency of decisions | Existing literature spans different scheduling domains; novelty claims need task-specific review and fair experiments |

### Survey limitation

No universal performance table is copied into this report because results from different datasets, objective functions, simulations, and resource models are not directly comparable. The current project needs its own same-workload baseline experiments and uncertainty analysis.

---

## 3. Current implementation audit

### Completed or visible in the screenshots

- Workload data prepared for 713 jobs.
- XGBoost prediction model trained in the notebook and previous evaluation values recorded.
- Global SHAP feature-importance table exported to `shap_feature_importance.csv` and displayed in Streamlit.
- EPCS, FCFS, SJF, and Priority aggregate results shown in a comparison table.
- `rr_results.csv` is included in the updated ZIP according to the notebook output, although Round Robin was not displayed by the older dashboard screenshot.
- Job Explorer, workload distribution charts, and a scheduling-timeline section are present.
- Updated ZIP contains nine CSVs according to the most recent Kaggle screenshot.

### Incomplete or requiring validation

1. **XGBoost metrics:** They need to be shown with a clear label that they are recorded from the notebook. For a fully reproducible dashboard, export the evaluation metrics into a file or calculate them from persisted test predictions and labels.
2. **Live inference:** The dashboard presently visualizes CSVs; it does not load the trained XGBoost model and fitted preprocessing pipeline for new input predictions.
3. **Round Robin display:** Ensure the dashboard reads `rr_results.csv` and includes it in the five-policy comparison.
4. **Timeline:** Treat Start_Time and Completion_Time as numeric workload-time units. Do not let a plotting library interpret them as Unix timestamps (which leads to dates around January 1970).
5. **Per-job explanation:** `shap_feature_importance.csv` is global importance only. The project objective says the system should explain why a job was selected, so local SHAP attributions and the scheduler’s candidate/decision log are still needed.
6. **Fair comparison:** Verify that all baselines use the same 713 jobs, arrival times, burst-time target/estimator, time units, priority convention, and identical metric formulas. Confirm Round Robin quantum.
7. **EPCS policy validation:** Document exact selection logic, when it dispatches, how predicted runtime and priority interact, and tie-breaking. EPCS and SJF currently show identical averages, so inspect per-job order/results.
8. **Generalization/leakage checks:** If a random train/test split was used, add chronological or trace-based validation. Features must not encode target or post-completion information.
9. **Reproducible experimental report:** Record dataset source/version, preprocessing, random seed, train/test method, model hyperparameters, system/runtime environment, and repeated-run results.
10. **Research task:** The supplied assignment requires 3–5 architectural options, novelty validation, weighted selection, and ablation plan. A working dashboard by itself does not fulfil those written research deliverables.

---

## 4. Research gaps

These are **candidate gaps to validate**, not final novelty claims.

### Gap G1 — Prediction explanations and dispatch explanations are disconnected

- **What existing techniques do:** Runtime regression predicts a duration; global/local SHAP attributes features to a prediction.
- **What is missing in the prototype:** A trace showing the ready candidates at each dispatch time, the selected job, alternatives not selected, and how the policy used runtime prediction and priority.
- **Why it matters:** A memory feature can dominate runtime prediction without being the actual reason a particular job beat another job in the ready queue.
- **Candidate response:** Store a decision record at each dispatch and pair it with local SHAP for the selected job and, where possible, top alternatives.
- **Validation required:** User studies are beyond the current prototype; at minimum check explanation faithfulness, consistency, overhead, and whether a reviewer can reconstruct the rule from the log.

### Gap G2 — Static prediction accuracy does not guarantee good scheduling under workload shift

- **What existing techniques do:** Fit predictors and report MAE/RMSE/R² on a held-out set.
- **What is missing:** A time-ordered test and shift detection/fallback policy.
- **Why it matters:** If job types, arrival rate, or requested resources change, a historical predictor may become miscalibrated.
- **Candidate response:** Evaluate chronologically and introduce a simple drift monitor with a transparent fallback to a baseline if error or drift exceeds a validated threshold.
- **Validation required:** Synthetic and trace-derived shifts, with predeclared thresholds and false-alarm analysis.

### Gap G3 — Mean waiting time can hide starvation and poor tail outcomes

- **What existing methods often report:** Average waiting/turnaround or a scalar reward.
- **What may be missing here:** P95/P99 waiting time, maximum wait, per-priority outcomes, and fairness/starvation measures.
- **Why it matters:** A lower average can hide a small group of jobs waiting far longer.
- **Candidate response:** Evaluate mean and tail metrics plus fairness, and impose a documented aging or wait-time safeguard if warranted.
- **Validation required:** Quantify any trade-off between average latency and fairness; do not claim fairness improvement without those metrics.

### Gap G4 — Explanation overhead is not accounted for

- **What existing work studies:** Prediction/scheduling quality and explanation methods, not always their operational overhead in a CPU scheduling loop.
- **What is missing:** Measured end-to-end latency for prediction, explanation, and dispatch.
- **Why it matters:** Explanations can cost more than the scheduling decision itself.
- **Candidate response:** Precompute global explanations, compute local explanations asynchronously or only for audited decisions, and measure overhead against a defined budget.
- **Validation required:** Runtime and memory measurements under multiple batch/workload sizes.

---

## 5. Problem characteristics and design requirements

| Characteristic | Evidence/implication | Design requirement |
|---|---|---|
| Mixed numerical and categorical fields | CPU/memory/node counts and job/priority categories are used | Fit preprocessing only on training data; preserve feature names for explanations |
| Jobs arrive over time | Arrival times determine the ready queue | Use correct event-driven scheduling and chronological evaluation |
| Runtime is uncertain | The target is predicted, not known in advance at dispatch | Analyse errors and add a baseline fallback for unreliable cases |
| Multiple objectives | Waiting, turnaround, response, throughput and fairness can conflict | Report several metrics, not a single winner score only |
| Large arrival-time range | Trace timestamps can be large numeric values | Keep timeline axes numeric and label units explicitly |
| Categorical encoding | Six raw columns produced eleven encoded features in the SHAP output | Use encoded feature names for global and local SHAP; do not align 11 values to 6 raw columns |
| Scheduling policy changes decisions | A runtime prediction is not itself a dispatch action | Log candidate set, policy scores, selected job, tie-break and decision time |
| Possible distribution shift | Trace data is historical | Add chronological and shift evaluations before deployment claims |

---

## 6. Rejected or deferred directions

1. **Claiming “XGBoost + SHAP” alone is novel:** Rejected. Both components are established, and XAI in scheduling has already been studied. The value must be a well-motivated, evaluated scheduler architecture and audit mechanism—not just combining two libraries.
2. **Switching to a Transformer simply because it is newer:** Deferred. Current tabular dataset size (3,562 rows as reported) and the project’s present need do not justify increased complexity without a benchmark.
3. **Building a full DRL scheduler immediately:** Deferred. It needs a defensible state/action/reward design, environment simulator, training stability checks, and multiple seeds. It would not be a small extension to the current prototype.
4. **Claiming EPCS wins because one aggregate metric is lower:** Rejected until identical workload, formulas, assumptions, and per-job schedules are audited. Current EPCS and SJF averages tie in the screenshot.

---

## 7. Four candidate architectural directions

### A1. EPCS-LA: Explanation-Logged Predictive CPU Scheduler

**Idea:** Keep a clearly specified runtime-aware policy, but create an immutable decision log containing ready candidates, predicted runtime, priority, policy score, chosen job, tie-break, and local SHAP contributions for the selected prediction. Global SHAP remains a separate model-level view.

**Pipeline:**

`Trace input → preprocessing → XGBoost runtime prediction → ready-queue candidates → explicit EPCS score/rule → selected job → local prediction explanation + decision log → dashboard`

**Strength:** Directly addresses the project’s stated “why was this job selected?” objective and is feasible with the current stack.

**Risk:** Local SHAP explains the runtime prediction; the policy log must separately show how the score/rule selected the job. Do not present SHAP as a direct explanation of the whole policy unless it is actually built and validated for that policy.

### A2. DriftGuard-EPCS: Drift-Aware Runtime Predictor with Fallback

Add a chronologically monitored prediction-quality/drift module. When a prevalidated threshold is exceeded, fall back to an explicit traditional scheduling policy until the predictor is assessed/retrained.

**Strength:** Addresses changing workload behaviour and makes the safety behaviour explicit.

**Risk:** Requires temporal order, threshold calibration, and experiments across controlled shifts.

### A3. FairEPCS: Tail-Latency and Starvation-Aware Scheduler

Combine predicted runtime with a wait-age/fairness safeguard and report mean, p95/p99 and maximum waiting time by priority/job type.

**Strength:** Avoids optimising averages while creating severe outliers.

**Risk:** A fairness term may worsen average waiting or throughput; weights must be justified and sensitivity-tested.

### A4. MetaSched-X: Contextual Baseline/Policy Selector

A transparent selector chooses among FCFS, SJF, Priority, Round Robin, and EPCS based on measurable workload features and an offline-validated policy.

**Strength:** A single predictive policy need not win on every workload.

**Risk:** The selector can overfit the benchmark, and selecting a policy is itself a scheduling decision that needs careful temporal validation and explanation.

---

## 8. Provisional novelty validation

| Candidate | Related work context | Potential difference | Provisional novelty | Incremental-risk level |
|---|---|---|---|---|
| A1: EPCS-LA | XGBoost is established; SHAP is established; explainable scheduling is already an active topic | Explicitly links prediction explanation with decision-state/candidate logging, tie-break reconstruction and measured explanation overhead for this CPU trace | Medium, **not validated yet** | Medium–High |
| A2: DriftGuard-EPCS | Predictive/resource scheduling and adaptive scheduling are established | A transparent, bounded fallback tied to measured prediction drift in the evaluation setup | Medium, **not validated yet** | Medium–High |
| A3: FairEPCS | Fairness-aware scheduling and aging are established topics | Joint reporting of tail latency, fairness and local explanation for this workload/policy | Low–Medium, **not validated yet** | High |
| A4: MetaSched-X | Learned algorithm selection and adaptive scheduling are established ideas | A trace-specific, interpretable policy-selector may be useful, but must outperform individual baselines on held-out temporal workloads | Low–Medium, **not validated yet** | High |

**Important:** These are architecture candidates, not proven original inventions. A database search of title/abstract/keywords and then full texts is still required. The supplied PDF specifically instructs rejecting incremental ideas instead of forcing novelty.

---

## 9. Weighted architecture selection

Weights follow the supplied assignment: novelty 25%, expected performance improvement 20%, research gap 15%, feasibility 15%, dataset availability 10%, computation 5%, publication potential 10%. Scores below are provisional expert judgements on a 1–5 scale, not experimental results.

| Criterion | Weight | A1 EPCS-LA | A2 DriftGuard | A3 FairEPCS | A4 MetaSched-X |
|---|---:|---:|---:|---:|---:|
| Novelty potential | 25% | 3 | 3 | 2 | 2 |
| Expected improvement | 20% | 3 | 4 | 3 | 4 |
| Gap strength | 15% | 4 | 4 | 3 | 3 |
| Technical feasibility | 15% | 5 | 3 | 4 | 2 |
| Data availability | 10% | 5 | 4 | 5 | 4 |
| Computational requirements | 5% | 5 | 3 | 5 | 2 |
| Publication potential | 10% | 3 | 4 | 3 | 3 |
| **Weighted score / 5** | **100%** | **3.75** | **3.55** | **3.10** | **2.85** |

### Selected direction: A1 — EPCS-LA

It is the most feasible extension for the current project because it directly improves auditability, preserves the existing XGBoost/SHAP pipeline, and enables exact reconstruction of each dispatch choice. This is a practical selection for the project deliverable, **not a claim that its novelty has been proven**.

---

## 10. Proposed final architecture

### Component descriptions

1. **Trace loader:** Read raw trace/workload and preserve job identifiers and chronological order.
2. **Preprocessing pipeline:** Numeric transformation plus one-hot encoding fit on training data only. Persist the fitted transformer.
3. **Execution-time predictor:** XGBoost regression estimates runtime. Store model version, prediction, and uncertainty/error-monitor values if available.
4. **Ready-queue builder:** At each simulated decision time, include only jobs that have arrived and are not completed.
5. **Explicit scheduling policy:** Rank candidates using documented predicted-runtime/priority/aging criteria; define tie-breaking and idle-CPU handling.
6. **Decision audit logger:** Record time, all candidate IDs and relevant features, scores/ranks, selected job, predicted duration and the rule that selected it.
7. **Explanation service:** Global SHAP for model behaviour; local SHAP for individual predictions. A separate rule trace/counterfactual comparison explains why one candidate won over the others.
8. **Monitor/fallback (future extension):** Evaluate drift or prediction error and switch to a defined baseline when a preregistered trigger fires.
9. **Metrics evaluator:** Waiting time = start − arrival; turnaround = completion − arrival; response time = first start − arrival. Also calculate throughput, maximum wait, p95/p99 latency, CPU utilisation when the simulation provides the needed fields, and policy/explanation overhead.
10. **Streamlit dashboard:** Show model performance, five-policy comparison, timeline, job explorer, global/local explanations, and validation warnings.

### Decision explanation example (illustrative template)

> “Job J was dispatched at time t because it was in the ready queue and received the best policy score under the configured rule. Its predicted execution time was p; its priority was q; its wait age was a. The next-best candidate was K with score sK. Local SHAP attributes the runtime prediction mainly to features F1 and F2. This explanation describes the model prediction separately from the policy rule.”

Do not populate a real explanation with invented candidate IDs/scores. Those fields must come from the scheduler decision log.

---

## 11. Experimental methodology

### Dataset and splits

- Identify and document the exact Kaggle dataset title, URL, download date/version, target column, and filtering rules.
- Keep job identifiers for joining prediction and schedule outputs, but do not use ID as a predictive feature unless justified.
- Use a time-ordered/chronological split where arrival-time dependence matters. If a random split is also reported, present it as a secondary result.
- Fit all imputers/encoders/scalers only on training data and apply the fitted transform to validation/test data.
- Verify the target is not present indirectly in input features and that no post-completion field leaks into features.

### Baselines

1. FCFS.
2. SJF (document whether it is non-preemptive and whether burst time is actual or predicted; do not give it access to actual future runtime if EPCS only has predictions).
3. Priority (state whether smaller or larger priority number means higher priority and how ties are broken).
4. Round Robin (state the quantum and preserve the same arrivals/CPU time assumptions).
5. EPCS with XGBoost prediction and a precisely specified dispatch rule.
6. Optional model baselines: mean/median predictor, linear regression, random forest, and gradient boosting without the proposed decision log.

### Predictive metrics

- MAE, RMSE, R².
- Residual plot and error by job type/priority/resource range.
- Chronological test performance and an explicit distribution-shift experiment.

### Scheduling metrics

- Mean, median, p95/p99, and maximum waiting time.
- Mean and tail turnaround time.
- Response time.
- Throughput and CPU utilisation where computable from the simulator.
- Fairness/starvation measures by priority and job type.
- Scheduler decision latency, local explanation latency, and memory overhead.

### Fair-comparison protocol

- Same 713 job IDs and same arrival times in all policies.
- Same burst-time knowledge assumption. If SJF is granted actual burst time while EPCS uses predicted burst time, label it an oracle baseline and also compare with predicted-time SJF.
- Same units, tie-breaking conventions, completion formula, and treatment of idle gaps.
- Validate that `Start_Time >= Arrival_Time`, `Completion_Time >= Start_Time`, and each job has one completion record.
- Compute response time from first start, not from a later preempted segment.
- For preemptive Round Robin, compute waiting time as turnaround minus CPU burst, provided there is a single CPU, no I/O blocking, and the trace’s burst definition is compatible.
- Report repeated seeds/runs where any component is stochastic and calculate uncertainty intervals when appropriate.

### Ablation plan

| Variant | Removed/changed component | Question |
|---|---|---|
| B0 | FCFS/SJF/Priority/RR baselines | What do conventional policies achieve? |
| B1 | EPCS using a simple mean/median runtime estimator | Is XGBoost prediction useful? |
| B2 | EPCS without priority/aging factor | What contribution does priority/aging make? |
| B3 | EPCS without drift monitor/fallback (if A2 is implemented) | Does the guardrail reduce degradation under shift? |
| B4 | EPCS without local explanations | What is the added explanation overhead? |
| B5 | Full EPCS-LA | What does the complete design achieve? |

Ablations must use the same test workloads and same parameter tuning budget. Do not infer component contribution from different train/test splits.

---

## 12. Evaluation and acceptance criteria

A final report should only claim the project meets its aims after:

- [ ] Model scores are reproducible from a saved test prediction/label artifact.
- [ ] All five schedulers produce valid results for the same job set.
- [ ] Unit conventions, RR quantum, priority direction and SJF information assumptions are documented.
- [ ] Timeline is numeric, and no invalid negative duration/wait or missing completion exists without explanation.
- [ ] EPCS logic is documented and per-job results are inspected against SJF.
- [ ] Global SHAP chart is present and correctly labelled as global feature importance.
- [ ] Local SHAP and decision-log explanation are implemented if the requirement is literally to explain why a job was selected.
- [ ] Prediction performance is measured on a temporal holdout and leakage checks are recorded.
- [ ] Baseline comparison includes tail latency and at least one fairness/starvation measure, not only averages.
- [ ] Any novelty claim is checked against additional literature and phrased cautiously.

---

## 13. Risks and limitations

- **Trace-to-real-system gap:** Offline trace simulation does not prove behaviour in a real OS kernel or cloud production system.
- **Prediction drift:** Historical runtime relationships may change.
- **Information asymmetry:** SJF often assumes known burst duration; comparing it to a predictor-based policy can be unfair unless assumptions are harmonised.
- **Metric ambiguity:** Units and precise burst/runtime semantics must be confirmed from the dataset source.
- **Global/local explanation confusion:** Global mean absolute SHAP cannot say why a specific job was selected at one dispatch.
- **Reproducibility:** Static metrics typed into a dashboard can become stale; use an exported metrics file or recompute from saved predictions.
- **Novelty risk:** XGBoost, SHAP, and explainable scheduling are established. Research value needs to come from a validated decision-level explanation method, operational constraints, and measured findings.

---

## 14. Conclusion

The prototype is **on the correct topic** and already demonstrates a visible scheduling dashboard, XGBoost evaluation values, baseline metrics, workload charts, and global SHAP importance. It is not yet fully complete against the supplied requirements. The most important remaining technical work is to make Round Robin appear in the comparison, fix and validate the numeric timeline, audit identical EPCS/SJF results, produce a local per-job decision explanation, and strengthen the experiment with chronological validation and fairness/tail metrics. The most important written deliverable is the full literature/gap/architecture/novelty-validation report required by the supplied brief.

---

## References

1. Chen, T., & Guestrin, C. (2016). **XGBoost: A Scalable Tree Boosting System.** Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 785–794. https://doi.org/10.1145/2939672.2939785
2. Lundberg, S. M., & Lee, S.-I. (2017). **A Unified Approach to Interpreting Model Predictions.** Advances in Neural Information Processing Systems 30. https://proceedings.neurips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html
3. Zhou, G., Tian, W., Buyya, R., Xue, R., & Song, J. (2024). **Deep reinforcement learning-based methods for resource scheduling in cloud computing: a review and future directions.** Artificial Intelligence Review, 57, 124. https://doi.org/10.1007/s10462-024-10756-9
4. Sanjalawe, Y., et al. (2025). **AI-driven job scheduling in cloud computing: a comprehensive review.** Artificial Intelligence Review, 58, 197. https://link.springer.com/article/10.1007/s10462-025-11208-8
5. **Explainable reinforcement learning in job-shop scheduling: A systematic literature review.** (2025). Procedia CIRP, 132, 25–30. https://doi.org/10.1016/j.procir.2025.01.005

The references above establish technical context and candidate gaps. They do not complete a systematic novelty search; additional papers must be screened and compared before claiming the final architecture is original.
