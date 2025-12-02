FARM-SPECIFIC MODEL METHODOLOGY
====================================
PROJECT OVERVIEW
================

OBJECTIVE
---------
Develop a systematic, reproducible methodology for predicting wind turbine 
component failures 24 hours in advance using SCADA time-series data.

PRIMARY GOAL
------------
Create a transferable approach that can be applied to any wind farm with 
SCADA data, not optimized for a single competition or dataset.

LEARNING FOCUS
--------------
Build methodology from first principles with zero domain expertise bias,
documenting every decision and validation step for reproducibility.

SUCCESS CRITERIA
----------------
- Systematic approach that transfers across wind farms
- Documented decision-making process
- Reproducible results
- Clear understanding of limitations
- Operational deployment readiness

DATASETS USED
=============

Source: EDP Open Data Competition Dataset (3 wind farms)

Wind Farm A - Development Dataset
----------------------------------
- Location: Onshore, Portugal
- Turbines: 5 assets
- Sensors: 54 base sensors → 81 features (with min/max/std)
- Failures: 12 anomaly events (2022-2023)
- Total SCADA records: 425,205
- Purpose: Methodology development and initial validation

Wind Farm B - Limited Data Test
--------------------------------
- Location: Offshore, Germany 
- Sensors: 86 base sensors → 257 features
- Failures: 6 anomaly events
- Purpose: Test performance with insufficient sample size

Wind Farm C - Primary Validation
---------------------------------
- Location: Offshore, Germany
- Turbines: 22 assets
- Sensors: 317 base sensors → 952 features
- Failures: 27 anomaly events
- Total SCADA records: 1,574,461
- Purpose: Validate transferability to different environment/turbine model

METHODOLOGY
====================================================

Phase 1: Data Understanding & Validation
--------------------------------------------------

Step 1.1: Temporal Pattern Discovery
- Analyzed time gaps between failures and last normal operation
- Discovered two distinct patterns:
- Sudden failures (10min - 18hr gaps): 4 events on Farm A
- Gradual failures (7-day gaps): 8 events on Farm A
- Key insight: event_start = when logged, not when failure began

Step 1.2: Normal Operation Definition Crisis
- Initial approach: status_type_id == 0 (normal status)
- Problem discovered: Included idle/stopped turbines (power = 0)
- Visual validation: Plotted Event 0, saw cooling-down turbine
- Solution: status == 0 AND power > 0.1
- Impact: Event 0 gap changed from 10min → 10.8 hours (real behavior)

Step 1.3: Visual Signal Validation 
- Plotted 3 representative failures with 4 key sensors each
- Event 72 (gearbox): No visible degradation in 24h window
- Event 40 (generator): Clear declining RPM trend (1650→1450)
- Event 0 (generator): Normal appearance despite failure
- Conclusion: Mixed predictability - need statistical features

Phase 2: Feature Engineering Strategy
-------------------------------------------------

Decision Point: Rolling Windows vs Window Aggregation
- Rolling windows: Complex, data leakage risk, doesn't scale
- Window aggregation: Clean, scalable, interpretable

Chosen Approach: Aggregate each 24h window into 3 statistics per sensor
- MEAN: Average operating level
- STD: Variability/stability 
- TREND: Linear slope (increasing/decreasing)

Result: 
- Farm A: 81 sensors × 3 = 243 features
- Farm C: 952 sensors × 3 = 2,856 features
- Scalable to any farm size

Baseline Construction:
- Exclude ±30 days around each failure (avoid contamination)
- Random sample 20 contiguous 24h windows per turbine
- Validate temporal integrity (no gaps >30min)
- Result: Balanced datasets (12:76, 6:180, 27:435 failure:normal)

Phase 3: Feature Selection
------------------------------------

Initial Approach: Effect Size (Cohen's d) Analysis
- Computed discriminative power for all features
- Farm A top features: RPM variability (d=1.30-1.45)
- Farm C top features: Voltage collapse (d=1.62-1.69)

Discovery: Massive Redundancy
- Farm A: 214 correlated pairs (r>0.9) in top 50 features
- Farm C: 213 correlated pairs (r>0.9) in top 50 features
- Generator RPM ≈ Rotor RPM (r=1.00)
- Three voltage phases nearly identical (r=0.998)

Solution: Iterative Selection with Correlation Threshold
- Select highest d feature
- Add next feature only if correlation <0.9 with selected
- Continue until max_features or candidates exhausted
- Result: 15-17 diverse, non-redundant features per farm

Critical Finding: Trend Features Initially Excluded
- Event 40 (clear visual trend) predicted at 0% probability
- Investigation: ZERO trend features in selected set (all std features)
- Understanding: Cohen's d favored variability over trends
- Decision for Farm A: Manually added top 4 trend features
- Farm B/C: Lowered threshold to include some trend features automatically
- Result: Still didn't help Event 40 (fundamental outlier)

Phase 4: Model Development & Validation
---------------------------------------------------

Model Choice: Random Forest Classifier
- n_estimators: 100
- max_depth: 5 (prevent overfitting with small n)
- class_weight: 'balanced' (handle class imbalance)
- random_state: 42 (reproducibility)

Cross-Validation Strategy: Leave-One-Out (LOO)
Rationale:
- Small sample sizes (6-27 failures) make k-fold problematic
- LOO maximizes training data per iteration
- Provides n independent predictions
- Standard approach for small datasets in literature

Farm A Results (Development):
- 8/12 failures detected (67% recall)
- 1/76 false alarms (99% specificity, 89% precision)
- 95% overall accuracy
- Event 40 outlier: 0% probability despite visual degradation

Event 40 Deep Dive:
- Clear visual trend: RPM declining 1650→1450 over 24h
- Aggregated features: All within 2σ of normal
- Distance to normal: 44.6
- Distance to other failures: 487.7
- Conclusion: Different failure physics (smooth vs erratic)
- Learning: Not all failures are predictable with current approach

Phase 5: Transferability Validation
---------------------------------------------

Challenge: Make Pipeline Farm-Independent
- Initial code: Hardcoded Farm C sensor numbers (100, 103, 136...)
- Refactored: Pure data-driven selection, no sensor name assumptions
- Result: Same code runs on any farm with SCADA structure

Farm C Validation (Offshore, 952 sensors):
- 18/27 failures detected (67% recall - SAME as Farm A!)
- 0/435 false alarms (100% precision)
- 98% overall accuracy
- Automated feature selection: 15 features from 2,856 candidates

Farm B Validation (Limited data, 6 failures):
- 2/6 failures detected (33% recall - degraded as expected)
- 0/180 false alarms (100% precision maintained)
- Demonstrated graceful degradation with insufficient data

Key Discovery: Consistent 67% Recall Benchmark
- Farm A: 67% (n=12)
- Farm C: 67% (n=27)
- Different environments, different sensors, SAME performance
- Conclusion: 67% is stable ceiling for this approach with 24h windows

FINAL METHODOLOGY - SYSTEMATIC 5-STEP PIPELINE
==============================================

Step 1: Data Validation
------------------------
INPUT: Raw SCADA data, event logs
PROCESS:
- Validate required columns exist
- Auto-detect power sensor
- Report data quality metrics
- Identify available status codes

OUTPUT: Validated data structure
FARM-SPECIFIC: Power sensor name only

Step 2: Temporal Pattern Analysis
----------------------------------
INPUT: SCADA data, event timestamps
PROCESS:
- Find last production timestamp before each failure
(status==0 AND power>threshold)
- Calculate time gap to event_start
- Report gap statistics (mean, median, min, max)

OUTPUT: Gap analysis, failure pattern classification
PURPOSE: Understand when failures are logged vs when they occur

Step 3: Feature Engineering
----------------------------
INPUT: SCADA data, event logs
PROCESS:
For each failure:
1. Find last_production time (status==0, power>0.1, time<event_start)
2. Extract window [last_production-24h, last_production]
3. Aggregate each sensor: compute mean, std, linear trend
4. Label as failure (1)

For normal baseline:
1. Exclude ±30 days around all failures
2. Filter for production periods (status==0, power>0.1)
3. Random sample 20 contiguous 24h windows per turbine
4. Aggregate each sensor: mean, std, trend
5. Label as normal (0)

OUTPUT: Aggregated feature dataset
- Rows: 24h windows (failure + normal)
- Columns: 3 × n_sensors features + metadata
SCALING: Handles 81 to 952 sensors automatically

Step 4: Feature Selection 
--------------------------
INPUT: Aggregated dataset
- PROCESS:
- Phase A - Discriminative Analysis:
- For each feature:
    - Calculate Cohen's d (effect size)
    - Perform Welch's t-test
    - Sort by effect size

Phase B - Redundancy Removal:
- Initialize selected = []
- For each candidate (sorted by Cohen's d):
    - If selected is empty: add candidate
    - Else: Calculate max correlation with selected features
- If max_corr < 0.9: add candidate
- If len(selected) >= max_features: stop

OUTPUT: Selected feature list (typically 15-17 features)
- PARAMETERS:
- cohens_d_threshold: 0.8 (higher for farms with many sensors)
- max_features: 15 (balance information vs overfitting)
- correlation_threshold: 0.9 (remove near-duplicates)

Step 5: Model Evaluation
-------------------------
INPUT: Dataset, selected features
PROCESS:
- Split features (X) and labels (y)
- Leave-One-Out cross-validation:
For each sample:
- Train on n-1 samples
- Predict held-out sample
- Record true label, prediction, probability
- Calculate metrics:
- Confusion matrix
- Precision, Recall, F1-score
- Per-failure probabilities

- OUTPUT: Performance metrics, per-event predictions
- METRICS PRIORITY: Precision > Recall (avoid false alarms)

PERFORMANCE RESULTS - THREE FARM VALIDATION
============================================

Farm A Farm B Farm C Combined
------ ------ ------ --------
- Environment Onshore Offshore Offshore Mixed
- Location Portugal Germany Germany 
- Sensors (base)        A:54    B:86    C:317 
- Features (engineered) A:81    B:257   C:952
- Selected Features     A:17    B:15    C:15-17

SAMPLE SIZES
- Total Windows A:88 B:186 C:462 Total:736
- Failure Windows A:12 B:6 C:27 Total:45
- Normal Windows A:76 B:180 C:435 Total:691
- Failure % of Data A:14% B:3% C:6% Total:6%

DETECTION PERFORMANCE
- Failures Detected A:8 B:2 C:18 Total:28
- Total Failures A:12 B:6 C:27 Total:45
- RECALL A:67% B:33% C:67% Total:62%

FALSE ALARM PERFORMANCE 
- False Positives A:1 B:0 C:0 Total:1
- True Negatives A:75 B:180 C:435 Total:690
- SPECIFICITY A:99% B:100% C:100% Total:99.9%
- PRECISION A:89% B:100% C:100% Total:96%

OVERALL METRICS
- Accuracy A:95% B:98% C:98% Total:97%
- F1-Score (Failure) A:0.76 B:0.50 C:0.80 Total:0.76

CONFIDENCE ANALYSIS
- Detected - Min Prob A:0.58 B:0.82 C:0.61 Total:0.58
- Detected - Max Prob A:0.99 B:0.84 C:1.00 Total:1.00
- Missed - Min Prob A:0.00 B:0.06 C:0.00 Total:0.00
- Missed - Max Prob A:0.25 B:0.34 C:0.31 Total:0.34

KEY OBSERVATIONS:

1. Consistent 67% Recall on Adequate Data
- Farm A (n=12): 67%
- Farm C (n=27): 67%
- Stable performance across different environments

2. Sample Size Matters 
- Farm B (n=6): 33% recall (bad but expected)
- Minimum ~10-12 failures needed for 60%+ recall

3. Exceptional Precision
- Only 1 false alarm across 691 normal windows (99.9%)
- When model predicts failure, 96% chance it's correct
- Critical for operator trust

4. Decisive Predictions
- Detected failures: High confidence (0.58-1.00)
- Missed failures: Low confidence (0.00-0.34)
- Model rarely "unsure" (few probabilities near 0.5)

KEY FINDINGS & INSIGHTS
========================

Finding 1: Power Threshold is Critical
---------------------------------------
- DISCOVERY: status_type_id alone insufficient to define "normal operation"
- PROBLEM: Idle turbines (wind too low) had status==0 but power==0
- SOLUTION: Require status==0 AND power>0.1
- IMPACT: 
    - Event 0 window changed from idle/cooling to actual production
    - Improved signal quality across all farms
    - Threshold 0.1 worked on all three farms (normalized power)

LEARNING: Domain assumptions must be validated against actual data

Finding 2: Failure Logging ≠ Failure Occurrence 
------------------------------------------------
DISCOVERY: Temporal gaps between last normal operation and event_start vary
FARM A PATTERN:
- 8/12 failures: Exactly 7-day gaps (downtime before logging)
- 4/12 failures: <18 hour gaps (nearly immediate)
FARM C PATTERN:
- Median gap: 0.2 hours (mostly immediate)
- Mean gap: 8.9 hours (few outliers)

IMPLICATION: 
- 24h prediction window captures different scenarios per farm
- Farm A: Last healthy period before downtime
- Farm C: Active operation right before shutdown

LEARNING: "Last normal" is more reliable anchor than event_start timestamp

Finding 3: Two Distinct Failure Physics
----------------------------------------
TYPE 1 - INSTABILITY FAILURES (Most common):
- Increased variability in sensors
- Erratic behavior while still operating
- High RPM std, power fluctuations, temp variability
- Well-detected by model (contributes to 67% recall)

TYPE 2 - SMOOTH DEGRADATION (Rare):
- Gradual performance decline
- Event 40: Steady RPM drop 1650→1450 over 24h
- Aggregated features appear "normal" (within 2σ)
- Poorly detected by current approach (0% probability)

FREQUENCY: 1 smooth degradation in 45 total failures (2%)

LEARNING: Current approach optimized for instability, not gradual trends

Finding 4: Feature Redundancy is Massive
-----------------------------------------
- FARM A: 214 highly correlated pairs (r>0.9) in top 50 features
- FARM C: 213 highly correlated pairs (r>0.9) in top 50 features

EXAMPLES:
- Generator RPM ≈ Rotor RPM (r=1.000)
- Three pitch blades move identically (r=1.000)
- Three voltage phases nearly identical (r=0.998)
- Reactive power sensors duplicate (r=0.998)

SOLUTION: Iterative selection with correlation threshold
RESULT: 15-17 features capture same information as 50+ correlated features

LEARNING: More features ≠ better performance with small datasets

Finding 5: Different Farms, Different Signatures
-------------------------------------------------
FARM A KEY FEATURES:
- RPM variability (generator, rotor)
- Reactive power variability
- Gearbox temperature variability
- Wind/power levels (LOWER in failures)

FARM C KEY FEATURES:
- Generator voltage collapse (1,135V → 0V)
- Pitch angle erratic behavior
- Rotor speed variability
- Bearing temperature variability

SIMILARITY: All involve increased variability or shutdown signatures

DIFFERENCE: Which specific sensors matter

IMPLICATION: 
- Cannot transfer learned features between farms
- CAN transfer feature selection methodology
- Automated selection finds relevant sensors per farm

LEARNING: Methodology transfers, specific features don't

Finding 6: 67% Recall is Stable Ceiling
----------------------------------------
OBSERVATION: Same 67% recall on Farm A (n=12) and Farm C (n=27)

ANALYSIS:
- Not due to insufficient data (doubling samples didn't improve)
- Not due to poor features (Cohen's d all >1.2)
- Not due to model choice (Random Forest well-suited)

HYPOTHESIS: Fundamental limit of 24h window approach

REASONS:
- Some failures are truly sudden (no 24h warning)
- Some failures are smooth degradation (aggregation misses trends) 
- Some failures lack sensor coverage (failure in un-monitored component)

EVIDENCE: 
- 33% of Farm A+C failures missed consistently
- Missed failures have low confidence (0.00-0.34), not borderline
- Different failure types show different detectability

LEARNING: 67% is realistic expectation, not failure of methodology

Finding 7: Zero False Alarms on Validation Farms
-------------------------------------------------
- FARM B: 0/180 false positives (100% specificity)
- FARM C: 0/435 false positives (100% specificity)
- FARM A: 1/76 false positives (99% specificity)

COMBINED: 1/691 false alarms (99.9%)

IMPLICATION:
- High operator trust (predictions are reliable)
- Conservative model (only predicts failure with strong evidence)
- Acceptable for operations (false alarms more costly than missed failures)

TRADE-OFF: Could increase recall by lowering threshold, but at cost of 
false alarms. Current balance appropriate for deployment.

Finding 8: Sample Size Directly Impacts Performance
----------------------------------------------------
- n=6 (Farm B): 33% recall
- n=12 (Farm A): 67% recall 
- n=27 (Farm C): 67% recall

THRESHOLD: Minimum ~10-12 failures for stable 60%+ recall

IMPLICATION FOR DEPLOYMENT:
- New wind farm needs ≥10 failures before reliable predictions
- Can start with smaller dataset but expect lower recall
- Precision remains high even with small n (Farm B: 100%)

LEARNING: Collect data first, deploy model second

CRITICAL DECISIONS DOCUMENTED
==============================

Decision 1: Restart vs Continue with Anomaly Detection
-------------------------------------------------------
SITUATION: Initial attempt felt overwhelming, lost confidence

CHOICE: Restart with systematic validation-first approach

ALTERNATIVE: Continue iterating on anomaly detection

OUTCOME: Clean methodology, reproducible, well-understood

LESSON: Starting over with clear principles beats incremental fixes

Decision 2: Normal Operation Definition
----------------------------------------
SITUATION: What constitutes "normal" for baseline?

CHOICE: status==0 AND power>0.1

ALTERNATIVE: status==0 only, or status IN (0,2)

VALIDATION: Visual inspection of Event 0 showed idle period problem

OUTCOME: Cleaner signals, better windows

LESSON: Validate assumptions visually before statistical analysis

Decision 3: Window Aggregation vs Rolling Features
---------------------------------------------------
SITUATION: How to engineer features from time-series?

CHOICE: Aggregate each 24h window (mean/std/trend)

ALTERNATIVE: Rolling windows across full time-series

RATIONALE:
- Prevents data leakage across train/test split
- Scales to any sensor count
- More interpretable (window = prediction unit)

OUTCOME: Successfully scaled from 81 to 952 sensors

LESSON: Simple, principled approaches often outperform complex ones

Decision 4: Feature Selection Strategy
---------------------------------------
SITUATION: 243-2,856 features, need to reduce dimensionality

CHOICE: Automated selection via Cohen's d + correlation threshold

ALTERNATIVE: Manual physics-based selection, PCA, domain expertise

RATIONALE:
- No domain knowledge available (by design)
- Different farms likely need different features
- Data-driven avoids bias

OUTCOME: Successfully found discriminative features on all farms

LESSON: Let data reveal patterns when domain knowledge limited

Decision 5: Event 40 Outlier - Adjust or Accept?
-------------------------------------------------
SITUATION: Clear visual degradation, 0% model probability

CHOICE: Accept as fundamental limitation, document thoroughly

ALTERNATIVE: Add specialized features, shorter windows, different model

RATIONALE:
- Only 1 of 45 failures (2%) shows this pattern
- Adding complexity for edge case risks overfitting
- Better to understand limitation than hide it

OUTCOME: Honest assessment of 67% recall ceiling

LESSON: Acknowledge limitations rather than force 100% performance

Decision 6: Precision vs Recall Trade-off
------------------------------------------
SITUATION: Could increase recall by lowering prediction threshold

CHOICE: Optimize for precision (minimize false alarms)

ALTERNATIVE: Optimize for recall (catch all failures)

RATIONALE:
- False alarms erode operator trust
- 67% recall catches most failures
- Missed failures tend to be sudden/unpredictable anyway

OUTCOME: 99.9% specificity on validation farms

LESSON: Operational context determines metric priority

Decision 7: Farm-Specific vs Unified Models
--------------------------------------------
SITUATION: Different sensor signatures across farms

CHOICE: Farm-specific models with shared methodology

ALTERNATIVE: Unified multi-farm model with sensor mapping

RATIONALE:
- Different turbine models have different physics
- Sample sizes too small to benefit from pooling (n=6-27)
- Sensor mapping complex and error-prone

OUTCOME: Consistent 67% recall on adequate data per farm

LESSON: Sometimes simpler is better than ambitious

Decision 8: Random Forest over Alternatives
-----------------------------------------------------
SITUATION: Selecting a classification algorithm

CHOICE: Random Forest Classifier

ALTERNATIVE: Log Reg, SVM, XGB, k-NN, Neural Networks

RATIONALE:
- Optimal for small-medum datasets
- Handles high-dimensional data
- Built-in feature importance
- Handles class imbalance
- Minimal hyperparameter sensitivity

OUTCOME: Achieved Project Goals

TRADEOFFS:
- May get higher recall with GBM
- Larger model size (100 trees vs single model)

Decision 9: Leave-One-Out vs K-Fold Cross-Validation
-----------------------------------------------------
SITUATION: Small sample sizes (6-27 failures)

CHOICE: Leave-One-Out cross-validation

ALTERNATIVE: 5-fold or 10-fold stratified CV

RATIONALE:
- LOO maximizes training data per iteration
- K-fold with n=6 means only 4-5 failures per fold
- Standard practice in literature for small n

OUTCOME: Stable performance estimates across farms


LIMITATIONS & EDGE CASES
=========================

Limitation 1: Minimum Sample Size Requirement
----------------------------------------------
- FINDING: Performance degrades below ~10 failures
- EVIDENCE: Farm B (n=6) achieved only 33% recall
- IMPACT: Cannot deploy on new farm without historical failure data
- WORKAROUND: Collect data first, or use unsupervised anomaly detection initially
- SEVERITY: High (fundamental to supervised learning)

Limitation 2: Smooth Degradation Not Captured
----------------------------------------------
- FINDING: Event 40 showed gradual trend but 0% model probability
- EVIDENCE: 10x closer to normal than other failures in feature space
- FREQUENCY: 1 of 45 failures (2%)
- IMPACT: Some failure modes undetectable with current approach
- WORKAROUND: Shorter windows (12h), explicit rate features, or LSTM
- SEVERITY: Medium (rare occurrence, specific failure type)

Limitation 3: 24-Hour Window May Be Too Early
----------------------------------------------
- FINDING: Some failures show no signal 24h before (e.g., Event 72)
- EVIDENCE: Visual inspection shows normal behavior in window
- IMPACT: Contributes to 33% missed failures
- WORKAROUND: Multiple time scales (6h, 12h, 24h), or accept limitation
- SEVERITY: Medium (inherent to prediction horizon choice)

Limitation 4: Cannot Transfer Features Between Farms
-----------------------------------------------------
- FINDING: Farm A and Farm C have completely different sensor signatures
- EVIDENCE: Generator voltage collapse (Farm C) vs RPM variability (Farm A)
- IMPACT: Cannot pre-train on one farm and transfer to another directly
- WORKAROUND: Sensor mapping via descriptions (future work)
- SEVERITY: Medium (methodology transfers, but not learned model)

Limitation 5: No Explanation of Specific Failure Cause
-------------------------------------------------------
- FINDING: Model predicts "failure" but not "hydraulic" vs "gearbox"
- EVIDENCE: Single binary classifier, not multi-class
- IMPACT: Operators know "something will fail" but not what specifically
- WORKAROUND: Per-failure-type models (needs more data), or feature importance
- SEVERITY: Low (24h advance warning is primary value)

Limitation 6: Assumes Static Turbine Behavior
----------------------------------------------
- FINDING: Model trained on historical data, assumes future similar
- RISK: Turbine aging, component replacements, control updates change behavior
- IMPACT: Model may degrade over time without retraining
- WORKAROUND: Periodic retraining (quarterly/annually), drift detection
- SEVERITY: Medium (standard ML limitation, well-understood)

Limitation 7: Requires Clean, Complete SCADA Data
--------------------------------------------------
- FINDING: Missing data, sensor failures would impact predictions
- EVIDENCE: Pipeline assumes <1% missing data (verified on all farms)
- IMPACT: Cannot deploy on farms with poor data quality
- WORKAROUND: Data quality checks, imputation for minor gaps
- SEVERITY: Low (data quality pre-requisite is standard)

Limitation 8: Single Prediction Threshold 
-------------------------------------------
- FINDING: Uses 0.5 probability threshold for all predictions
- EVIDENCE: Could adjust per-farm or per-operator risk tolerance
- IMPACT: Miss some tuning opportunities
- WORKAROUND: Threshold tuning based on operational costs
- SEVERITY: Low (easy to adjust in deployment)





