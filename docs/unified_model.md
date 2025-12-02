UNIFIED MULTI-FARM MODEL METHODOLOGY
====================================

PROJECT CONTEXT
===============

MOTIVATION
----------
Farm-specific models fail with limited data: Farm B achieved only 33% recall 
with just 6 failure examples, falling below the minimum ~10-12 failures needed 
for reliable ML pattern recognition.

HYPOTHESIS
---------------
Can we combine training data from multiple wind farms to improve performance on 
limited-data scenarios while maintaining transferability?

CHALLENGE
---------
Wind farms use different turbine models, sensor configurations, and naming 
conventions. Farm A has 81 sensors, Farm B has 257, Farm C has 952. Need to create a unified feature space?

APPROACH
-----------------
Semantic sensor mapping: Map farm-specific sensors to unified physical categories 
(e.g., "gearbox_mechanical", "generator_thermal") enabling cross-farm learning 
while preserving physical meaning.

SUCCESS CRITERIA
----------------
- Farm B recall improves beyond 33% baseline
- Farm A/C performance remains acceptable (>55% recall)
- Precision maintained (>90%)
- Methodology is reproducible and generalizable

CRITICAL IMPLEMENTATION NOTE
-----------------------------
During methodology development, initial results showed Farm B improvement 
from 33% to 67% recall with no degradation on Farm A (maintaining 67%). 
However, these results were obtained with INCOMPLETE category merging, 
effectively running farm-specific models rather than a truly unified approach.

After proper implementation of unified semantic categories (merging 50 
granular categories into 16 cross-farm categories), actual results show:

ACTUAL UNIFIED MODEL RESULTS:
- Farm A: 67% -> 58% (-9%)
- Farm B: 33% -> 50% (+17%)
- Farm C: 67% -> 56% (-11%)
- Overall: 56% -> 56% (neutral)

This demonstrates the FUNDAMENTAL TRADE-OFF in multi-domain learning: 
generalizability vs. domain-specific optimization. The unified approach 
enables consistent baseline performance across heterogeneous sites but 
sacrifices peak performance on well-instrumented farms.

BASELINE RESULTS (FARM-SPECIFIC MODELS)
========================================

Farm A (Development):
- 12 failures, 76 normal windows
- 8/12 detected (67% recall)
- 1/76 false alarms (89% precision)
- Status: Good performance

Farm B (Limited Data):
- 6 failures, 180 normal windows  
- 2/6 detected (33% recall) ← PROBLEM
- 0/180 false alarms (100% precision)
- Status: Insufficient training data

Farm C (Validation):
- 27 failures, 435 normal windows
- 18/27 detected (67% recall)
- 0/435 false alarms (100% precision)
- Status: Good performance

OVERALL BASELINE: 56% recall (28/45 failures detected)

THE UNIFIED APPROACH
====================

Phase 1: Sensor Configuration Analysis
---------------------------------------

Step 1.1: Inventory Farm-Specific Sensors
- Farm A: 54 base sensors -> 81 with min/max/std suffixes
- Farm B: 86 base sensors -> 257 features
- Farm C: 317 base sensors -> 952 features
- Total unique sensors to map: 283

Naming Convention Discovery:
- Farm A: sensor_5, power_30_avg, wind_speed_3_max
- Farm B: sensor_20_std, reactive_power_11_avg
- Farm C: sensor_146_avg, sensor_100_std_dev
- Pattern: base_name + [_min/_max/_avg/_std/_std_dev]

Step 1.2: Initial Categorization Attempt
Created 50 categories based on:
- Physical system (electrical, gearbox, generator, rotor, pitch, etc.)
- Measurement type (temperature, speed, current, voltage, pressure)
- Functional role (cooling, lubrication, control, monitoring)

Result of Initial Attempt:
- Categories in ALL 3 farms: 3 (too few)
- Categories in 2+ farms: 10
- Farm-specific only: 38
- Expected missing data: 88% (unusable)

Analysis:
- Too granular: "gearbox_bearings" vs "gearbox_main_shaft" as separate
- Inconsistent naming across farms
- Different sensor availability patterns

Step 1.3: Sensor Name Matching Challenge
Direct matching failed:
- SCADA columns: sensor_5_min, power_30_avg, wind_speed_59_max
- Mapping file: sensor_5, power_30, wind_speed_59
- Match rate: Farm A 17%, Farm B 0%, Farm C 0%

Phase 2: Strategic Category Merging
------------------------------------

Objective: Increase shared categories while preserving physical meaning

Merging Principles:
1. Combine subcategories measuring same physical phenomenon
2. Keep mechanically/thermally distinct systems separate
3. Preserve operationally meaningful distinctions (active vs reactive power)
4. Use domain knowledge to guide consolidation

Key Merges Applied:

ELECTRICAL SYSTEM:
- electrical_general (phase currents/voltages) -> electrical_grid
- electrical_grid_connection (frequency, grid voltage) -> electrical_grid
- electrical_internal (consumption currents) -> electrical_grid
- electrical_transformer (transformer temps) -> transformer_temperature
Rationale: All measure grid power quality and electrical health

GEARBOX SYSTEM:
- gearbox_bearings (bearing temperatures) -> gearbox_mechanical
- gearbox_main_shaft (shaft speeds) -> gearbox_mechanical
- gearbox_general (rotational speed) -> gearbox_mechanical
- gearbox_oil_temperature -> gearbox_lubrication
- gearbox_oil_pressure -> gearbox_lubrication
- gearbox_oil_level -> gearbox_lubrication
Rationale: Separate mechanical health from lubrication system

GENERATOR SYSTEM:
- generator_cooling (cooling air temps) -> generator_thermal
- generator_bearings (bearing temps) -> generator_thermal
- generator_stator (stator winding temps) -> generator_thermal
- generator_current (RMS currents) -> generator_electrical
- generator_speed (RPM) -> generator_electrical
- generator_general -> generator_electrical
Rationale: Thermal vs electrical state are distinct failure modes

PITCH SYSTEM:
- pitch_motor (motor current, temp) -> pitch_control
- pitch_battery (battery charge/discharge) -> pitch_control
- pitch_general (DC link voltage, RMS current) -> pitch_control
- pitch_angle -> KEPT SEPARATE (physically meaningful)
Rationale: All control-related except angle which is blade position

ROTOR SYSTEM:
- rotor_bearings (bearing temps) -> rotor_thermal
- rotor_temperature (hub, nose cone) -> rotor_thermal
- rotor_spinner (pressure differential) -> rotor_thermal
- rotor_speed (RPM) -> rotor_mechanical
- rotor_brake (hydraulic pressure) -> rotor_mechanical
- rotor_general (DC voltage) -> rotor_mechanical
Rationale: Thermal degradation vs mechanical operation

ENVIRONMENT:
- meteorological_temperature -> environment_temperature
- meteorological_wind_dir -> environment_wind
- meteorological_wind_speed -> environment_wind
Rationale: Consolidate external conditions

OTHER SYSTEMS:
- control_control_temp + control_cabinet_temp -> control_temperature
- nacelle_vibration + nacelle_general -> nacelle_structural
- tower_vibration + tower_frequency -> tower_structural
- hydraulic_* variations -> hydraulic_system (mostly Farm C only)

Result After Merging:
- Categories in ALL 3 farms: 12 (4x improvement)
- Categories in 2+ farms: 16 (60% more usable)
- Farm-specific only: 10 (75% reduction)
- Expected missing data: 25-38% (65% improvement from 88%)

Final Unified Categories (16 total):

IN ALL 3 FARMS (12 categories):
1. electrical_grid (41 sensors)
2. environment_wind (11 sensors)
3. gearbox_lubrication (13 sensors)
4. gearbox_mechanical (26 sensors)
5. generator_electrical (7 sensors)
6. generator_thermal (27 sensors)
7. pitch_angle (6 sensors)
8. power_active (15 sensors)
9. power_reactive (14 sensors)
10. rotor_mechanical (7 sensors)
11. rotor_thermal (17 sensors)
12. transformer_temperature (17 sensors)

IN 2 FARMS (4 categories):
13. control_temperature (A, C: 10 sensors)
14. environment_temperature (B, C: 4 sensors)
15. nacelle_structural (A, C: 16 sensors)
16. pitch_control (B, C: 30 sensors)

Phase 3: Unified Feature Engineering
-------------------------------------

Step 3.1: Category Aggregation Strategy

For each 24-hour window and each unified category:

1. IDENTIFY applicable sensors in this farm
   Example (gearbox_mechanical):
   - Farm A: sensor_11 (gearbox bearing temp)
   - Farm C: sensor_146, 147 (main shaft speeds), 
             sensor_168, 169 (axial bearings), ...

2. AGGREGATE across sensors within category
   If multiple sensors:
     category_timeseries = mean(sensor_146, sensor_147, sensor_168, ...)
   If single sensor:
     category_timeseries = sensor_value
   
   Result: 144 data points (24h × 6 readings/hour) per category

3. COMPUTE window statistics on aggregated time-series
   category_mean = mean(category_timeseries)
   category_std = std(category_timeseries)
   category_trend = linear_slope(category_timeseries)

Result: Same feature names across all farms
- Farm A: gearbox_mechanical_mean, gearbox_mechanical_std, ...
- Farm B: gearbox_mechanical_mean, gearbox_mechanical_std, ...
- Farm C: gearbox_mechanical_mean, gearbox_mechanical_std, ...

Step 3.2: Feature Space Construction

Expected features: 16 categories × 3 statistics = 48 base features
Actual features created: 150 (some categories have variants)

Comparison to Farm-Specific Approach:
- Farm-specific: 81, 257, 952 features (farm-dependent)
- Unified: 150 features (farm-independent)
- Benefit: Directly comparable feature space

Missing Data Handling:
- Categories in all 3 farms: ~10% missing
- Categories in 2 farms: ~40% missing
- Farm-specific categories: 100% missing for other farms
- Overall: ~25% missing (acceptable for Random Forest)

Step 3.3: Dataset Construction

Combined Training Set:
- Farm A: 12 failures, 84 normal -> 96 windows
- Farm B: 6 failures, 176 normal -> 182 windows
- Farm C: 27 failures, 435 normal -> 462 windows
- TOTAL: 45 failures, 695 normal -> 740 windows

Key Benefit for Farm B:
- Previously: Trained on 6 failures (33% recall)
- Now: Trained on 45 failures (7.5× more data)
- Hypothesis: Should significantly improve recall

Phase 4: Unified Feature Selection
-----------------------------------

Challenge: Select features discriminative across ALL farms

Step 4.1: Discriminative Analysis (Cohen's d)

For each of 150 unified features:
1. Extract normal values: all windows with label=0
2. Extract failure values: all windows with label=1
3. Calculate effect size: d = |mean_failure - mean_normal| / pooled_std
4. Skip features with >75% missing data

Results:
- Features analyzed: 123 (27 excluded for excessive missing data)
- Features with d > 1.0 (very large effect): 15
- Features with d > 0.8 (large effect): 20
- Features with d > 0.6 (medium-large): 29

Step 4.2: Redundancy Removal

Iterative selection with correlation threshold = 0.9:
1. Start with highest Cohen's d feature
2. For each next candidate:
   - Calculate correlation with already-selected features
   - If max_correlation < 0.9: add to selected set
   - Else: skip (redundant)
3. Stop at 20 features (or exhaustion of candidates d>0.6)

Result: 20 diverse, non-redundant features selected

Top 20 Selected Unified Features (ranked by Cohen's d):

1. rotor_speed_std (d=1.387) - Increased speed variability
2. gearbox_main_shaft_std (d=1.381) - Shaft instability
3. generator_current_mean (d=1.365) - Current collapse (shutdown)
4. pitch_angle_std (d=1.231) - Pitch control instability
5. gearbox_main_shaft_mean (d=1.078) - Lower average speed
6. pitch_motor_std (d=1.048) - Motor current variability
7. gearbox_bearings_std (d=1.008) - Bearing temperature spikes
8. gearbox_main_shaft_trend (d=0.990) - Declining shaft speed
9. gearbox_bearings_mean (d=0.990) - Elevated bearing temps
10. meteorological_wind_speed_mean (d=0.956) - Lower wind conditions
11. gearbox_oil_pressure_std (d=0.894) - Oil pressure variability
12. rotor_speed_mean (d=0.885) - Lower average rotor speed
13. rotor_bearings_std (d=0.873) - Bearing temp variability
14. pitch_battery_std (d=0.864) - Battery instability
15. rotor_temperature_std (d=0.856) - Rotor thermal variability
16. generator_bearings_std (d=0.844) - Generator bearing temps
17. meteorological_wind_dir_std (d=0.790) - Wind direction changes
18. generator_cooling_std (d=0.739) - Cooling system variability
19. generator_current_trend (d=0.739) - Declining current
20. generator_cooling_mean (d=0.708) - Elevated cooling temps

Key Observations:

UNIVERSAL MECHANICAL SIGNALS:
- Rotor and gearbox variability dominate (6 of top 10)
- Speed instability is strongest predictor
- Bearing temperature elevation consistent across farms

UNIVERSAL CONTROL SIGNALS:
- Pitch system instability (angle, motor, battery)
- Generator current patterns (collapse, declining trend)

ENVIRONMENTAL CONTEXT:
- Lower wind speeds during failures (turbines operating in calmer conditions)
- Wind direction variability

DIFFERENT FROM FARM-SPECIFIC:
- Farm A top: RPM variability (std features)
- Farm C top: Voltage collapse (generator shutdown)
- Unified top: Rotor speed + gearbox (mechanical degradation)
- Conclusion: Unified model finds common mechanical patterns

Phase 5: Unified Model Training & Evaluation
---------------------------------------------

Model Configuration:
- Random Forest Classifier (same as farm-specific)
- n_estimators: 100
- max_depth: 5 
- class_weight: 'balanced' (14:1 imbalance)

Missing Data Strategy:
- Median imputation for training (sklearn SimpleImputer)
- Features >75% missing already excluded from selection

Cross-Validation: Leave-One-Out (LOO)
- 740 windows -> 740 train-test splits
- Each window held out once for prediction
- Trained on 739 windows (44-45 failures depending on fold)
- Standard approach for small failure counts

Training Process:
- Total samples: 740
- Failures: 45 (6%)
- Normal: 695 (94%)
- Features: 20 (after selection)
- Missing values before imputation: 3,810 (25.7%)
- Missing values after imputation: 0

UNIFIED MODEL RESULTS
=====================

Overall Performance:
--------------------

Metrics:
- Recall: 56% (25/45 failures detected)
- Precision: 96% (25/26 positive predictions correct)
- Specificity: 99.9% (668/669 normal correctly identified)
- Accuracy: 97% (693/714 overall correct)
- F1-Score: 0.70

Farm-by-Farm Breakdown:
-----------------------

Farm A (Development, n=12):
- Detected: 7/12 (58% recall)
- False alarms: 0/65 (100% specificity)
- Precision: 100%
- CHANGE FROM BASELINE: -9% (67% -> 58%)
- Status: ✗ Moderate degradation
- Trade-off: Acceptable for Farm B improvement

Farm B (Limited Data, n=6):
- Detected: 3/6 (50% recall)
- False alarms: 1/180 (99% specificity)
- Precision: 75%
- CHANGE FROM BASELINE: +17% (33% -> 50%)
- Status: ✓ Modest improvement
- Impact: Approaches deployment threshold (>50% achieved)

Farm C (Validation, n=27):
- Detected: 15/27 (56% recall)
- False alarms: 0/424 (100% specificity)
- Precision: 100%
- CHANGE FROM BASELINE: -11% (67% -> 56%)
- Status: -> Moderate degradation
- Trade-off: Acceptable, maintains >50% threshold

Comparison Table:

| Metric | Farm A | Farm B | Farm C | Overall |
|--------|--------|--------|--------|---------|
| Baseline Recall | 67% | 33% | 67% | 56% |
| Unified Recall | 58% | 50% | 56% | 56% |
| Change | -9% | +17% | -11% | 0% |
| Baseline Precision | 89% | 100% | 100% | 96% |
| Unified Precision | 100% | 75% | 100% | 96% |
| False Alarms | 0/65 | 1/180 | 0/424 | 1/669 |

INTERPRETATION OF RESULTS
=========================

Success: Farm B Performance Improved
------------------------------------
From 2/6 to 3/6 failures detected (+17% recall)

Root Cause of Improvement:
- Training data: 6 failures -> 45 failures (7.5× increase)
- More diverse failure patterns learned
- Better generalization from larger sample
- Cross-farm patterns more robust than farm-specific noise

Business Impact:
- Previously: 33% recall = unreliable, cannot deploy
- Now: 50% recall = approaches deployment threshold
- 1 additional failure prevented = $100K saved annually
- Farm B now marginally viable for cautious deployment

Trade-off Analysis: Farm A & C Degradation
-----------------------------------------------
Farm A: 8/12 -> 7/12 detected (-9% recall)
Farm C: 18/27 -> 15/27 detected (-11% recall)

Root Cause of Degradation:
- Unified categories lose some granular information
- Farm A has 81 sensors, Farm C has 952 sensors
- Aggregating sensors within categories -> single value loses nuance
- Example: 20 gearbox sensors averaged -> "gearbox_mechanical" feature
- Some farm-specific subtle signals masked

Trade-off Analysis:
- Farm A loss: -1 failure (67% -> 58%)
- Farm C loss: -3 failures (67% -> 56%)
- Farm B gain: +1 failure (33% -> 50%)
- Net: -3 failures detected overall
- However: All farms now meet minimum threshold (>50%)

Precision Maintained: Near-Zero False Alarms
-----------------------------------------------------
Farm A: 0/65 false alarms (100% specificity)
Farm B: 1/180 false alarms (99% specificity)
Farm C: 0/424 false alarms (100% specificity)
Overall: 1/669 false alarms (99.9% specificity)

Why Precision Matters:
- False alarms cost ~$5K-$10K (unnecessary inspection)
- Operator trust erodes quickly with false positives
- Near-perfect specificity maintains confidence
- Critical for production deployment

Overall System Assessment: Neutral Performance
---------------------------------------
56% baseline -> 56% unified (25/45 total)

Demonstrates:
- Methodology works: Combined training improves limited-data scenarios (+17% Farm B)
- Trade-offs exist: Farm-specific optimization sacrificed for generalizability
- Precision maintained: 99.9% specificity across all farms
- Deployment threshold: All farms now meet >50% minimum for cautious deployment

Key Insight:
Unified approach enables consistent baseline performance across heterogeneous 
sites but does not improve overall system recall. Primary value is enabling 
deployment on previously unsuitable farms (Farm B) at cost of optimized 
performance on well-instrumented farms (A & C).

DISCOVERED UNIVERSAL FAILURE PATTERNS
======================================

Mechanical Degradation Dominates
---------------------------------
Top signals are mechanical, not electrical:
1. Rotor speed variability
2. Gearbox shaft instability
3. Gearbox bearing temps

Observation:
- Farm A top: RPM variability (electrical focus)
- Farm C top: Voltage collapse (electrical shutdown)
- Unified top: Mechanical degradation (rotor/gearbox)

Interpretation:
- Mechanical issues precede electrical
- More universal across turbine models
- Electrical collapse is late-stage (Farm C specific)

Control System Instability
---------------------------
Pitch system features prominent:
- Pitch angle variability 
- Pitch motor current 
- Pitch battery 

Significance:
- Pitch control critical for turbine operation
- Instability indicates control system stress
- Present in 2 of 3 farms (merged from B & C)

Environmental Context Matters
------------------------------
Lower wind speeds during failures 

Hypothesis:
- Failures occur during calmer conditions
- Less wind -> less load -> opportunity for degradation
- Or: Low wind -> unusual operating regime -> stress

Surprising finding from unified model (not prominent in farm-specific)

Generalization Insights
-----------------------

What Transfers:
- Mechanical degradation patterns (rotor, gearbox)
- Control system instability (pitch)
- Bearing temperature elevation
- Speed/RPM variability

What Doesn't Transfer:
- Specific voltage/current collapse patterns (farm-specific)
- Absolute sensor values (turbine-dependent)
- Smooth degradation trends (Event 40 type failures)

Implication:
- Unified model learns physics-based patterns
- Farm-specific models learn turbine-specific signatures
- Hybrid approach could combine strengths

LIMITATIONS & CHALLENGES
========================

Challenge 1: Category Mapping Requires Domain Expertise
-------------------------------------------------------
Issue: Would take many hours to manually map 283 sensors to categories and I had no real domain knowledge of wind turbines.

Process:
- Review each sensor description
- Research physical meaning (LLM assisted)
- Group by system and function
- Validate with domain knowledge (or lack thereof in my case)

Difficulty:
- Ambiguous descriptions
- Inconsistent naming across farms
- Trade-off between granularity and coverage

Mitigation:
- Build reusable category library
- Document merge rationale
- Automate via sensor similarity analysis (future work)

Challenge 2: Missing Data in Partial Categories
-----------------------------------------------
Issue: 4 categories exist in only 2 farms (pitch_control, etc.)

Impact:
- ~40% missing for these categories
- Reduces discriminative power
- Increases imputation uncertainty

Trade-off:
- Include partial categories: More features, more missing data
- Exclude partial categories: Fewer features, less missing data
- Chose to include (60+ features vs 36 if excluded)

Result:
- Overall 25% missing (acceptable)
- Median imputation for other algorithms

Challenge 3: Farm C Degradation from Averaging
----------------------------------------------
Issue: Averaging 20+ sensors -> single value loses information

Example:
- Farm C sensor_137: Generator voltage
- Unified generator_electrical_mean:
- Loss: Voltage-specific collapse signal partially masked

Trade-off Analysis:
Farm C-specific: 67% recall, not transferable
Unified: 59% recall, works on Farm B

Decision: Accept 8% loss for transferability benefit

Future Improvement: Hybrid approach
- Unified features (transferable patterns)
- Top 5 farm-specific features (local optimization)
- Potential: 59% -> 65% on Farm C while maintaining Farm B gain

Challenge 4: Performance Ceiling at ~62% Recall
-----------------------------------------------
Observation: 
- Farm-specific ceiling: ~67%
- Unified ceiling: ~62%
- Only 5% difference

Causes:
- Some failures inherently unpredictable (sudden, no warning)
- Smooth degradation not captured by window aggregation
- 24-hour window may be too short or too long for certain failure modes

Conclusion: 62% is acceptable ceiling for this approach

Challenge 5: Requires Minimum Per-Farm Data
-------------------------------------------
Issue: Still need SOME data from target farm

Evidence:
- Farm B: 6 failures in training set
- If Farm B had 0 failures: Model would likely fail

Requirement: 5-10 failures per farm for basic calibration

Limitation: Cannot deploy on completely new farm with zero historical data

Workaround:
1. Deploy unified model initially (based on other farms)
2. Collect failure data over 6-12 months
3. Retrain with farm-included
4. Performance improves over time

METHODOLOGY DECISIONS & RATIONALE
==================================

Decision 1: Category Merging Strategy
-------------------------------------
CHOSEN: Aggressive merging (50 -> 16 categories)

Alternatives Considered:
A. Keep granular (50 categories): 3 shared -> 88% missing
B. Moderate merging (30 categories): 8 shared -> 60% missing
C. Aggressive merging (16 categories): 12 shared -> 25% missing

Rationale:
- Need >10 shared categories for usable feature space
- Preserve meaningful distinctions (active vs reactive power)
- Combine similar measurements (all bearing temps -> mechanical)
- Basic reason: 16 categories worked, 3 did not

Validation:
- 62% overall recall demonstrates adequacy
- Top features include diverse systems (rotor, gearbox, generator, pitch)
- If merged too aggressively, would lose discriminative power

Decision 2: Window Aggregation within Categories
------------------------------------------------
CHOSEN: Average sensors -> single time-series -> statistics

Alternatives Considered:
A. Keep all sensors separate: Farm C has 952 -> cannot unify
B. Select one representative sensor: Loses information from others
C. Average sensors, then aggregate: CHOSEN

Example (gearbox_mechanical):
- Farm A: 1 sensor -> use directly
- Farm C: 20 sensors -> average -> compute stats

Rationale:
- Averaging preserves signal while reducing dimensionality
- Statistical features (mean, std, trend) capture aggregated behavior
- Scales to any number of sensors automatically
- Physical interpretation: "gearbox mechanical health" not "sensor_146 value"

Trade-off:
- Pro: Transferability, scalability
- Con: Loses sensor-specific signals (Farm C degradation)

Validation:
- Farm B improved (33% -> 67%)
- Farm A maintained (67% -> 67%)
- Farm C acceptable (67% -> 59%)
- Net positive

Decision 3: Combined Training vs Transfer Learning
--------------------------------------------------
CHOSEN: Combine all data into single training set

Alternatives Considered:
A. Transfer learning: Train on A&C, fine-tune on B
   - Pro: Preserves farm-specific patterns
   - Con: Complex, requires more implementation
   - Con: Still limited by B's 6 failures for fine-tuning

B. Simple combination: Treat all farms as one dataset ← CHOSEN
   - Pro: Simple, interpretable
   - Pro: Maximum data for Farm B (6 -> 45)
   - Con: Assumes failures have similar patterns

Rationale:
- Simplicity: Same algorithm as farm-specific
- Data maximization: Farm B benefits from 7.5× more failures
- Interpretability: Same Random Forest, same feature importance
- Proven: Worked (Farm B 33% -> 67%)

Decision 4: Feature Selection on Combined Dataset
-------------------------------------------------
CHOSEN: Select features using Cohen's d on ALL farms combined

Alternatives Considered:
A. Select per-farm, take union: Might miss cross-farm patterns
B. Select per-farm, take intersection: Too restrictive (few features)
C. Select on combined: ← CHOSEN

Process:
- Compute Cohen's d using all 740 windows
- Normal: 695 windows from all farms
- Failure: 45 windows from all farms
- Select top features discriminative across entire dataset

Rationale:
- Finds universally discriminative features
- Balances performance across farms
- Discovered: Mechanical signals most universal

Result:
- Top 20 features span multiple systems
- No single farm dominates feature selection
- Validated by consistent performance

Decision 5: Leave-One-Out CV on Combined Set
--------------------------------------------
CHOSEN: LOO on all 740 windows

Alternatives Considered:
A. LOO per farm separately: Doesn't test transferability
B. Leave-One-Farm-Out: Only 3 folds (insufficient)
C. K-fold with stratification: Risks farm imbalance in folds
D. LOO on combined: ← CHOSEN

Rationale:
- Small n: LOO maximizes training data
- Provides 740 independent predictions
- Tests model on all farms simultaneously
- Standard for small datasets

Validation Strategy:
- Overall metrics (740 windows)
- Per-farm breakdown (evaluate transferability)
- Per-event probabilities 

FINAL UNIFIED PIPELINE - 6 STEPS
=================================

Step 1: Sensor Mapping (One-time Setup)
---------------------------------------
INPUT: Farm sensor configurations
PROCESS:
- Inventory all sensors across farms
- Map to unified physical categories
- Document merge rationale
- Handle different naming conventions (root name extraction)

OUTPUT: sensor_mapping_v2.csv
EFFORT: 4 hours per new farm
REUSABLE: Yes, library grows over time

Step 2: Data Loading & Validation
----------------------------------
INPUT: SCADA data, event logs for all farms
PROCESS:
- Load and validate each farm
- Auto-detect power sensors
- Check required columns
- Report data quality

OUTPUT: Validated multi-farm dataset

Step 3: Unified Feature Extraction
-----------------------------------
INPUT: SCADA data, events, sensor mapping
PROCESS:
For each farm and each window:
1. Identify sensors in each unified category
2. Extract sensor values for 24h window
3. Aggregate within category (if multiple sensors)
4. Compute statistics: mean, std, trend

OUTPUT: Unified feature dataset (740 windows × 150 features)
RESULT: Same feature names across all farms

Step 4: Dataset Combination
----------------------------
INPUT: Feature datasets from all farms
PROCESS:
- Concatenate all windows
- Add farm identifier column
- Mark missing data
- Label failures (1) and normal (0)

OUTPUT: Combined dataset
- 740 windows total
- 45 failures (6% positive class)
- 695 normal (94% negative class)
- 150 unified features

Step 5: Unified Feature Selection
----------------------------------
INPUT: Combined dataset
PROCESS:
- Discriminative analysis (Cohen's d on all farms)
- Redundancy removal (correlation < 0.9)
- Missing data filtering (exclude >75% missing)

OUTPUT: 20 selected features
CHARACTERISTICS:
- Large effect size (d > 0.7)
- Low redundancy (r < 0.9)
- Spans multiple systems
- <50% missing on average

Step 6: Unified Model Training & Evaluation
-------------------------------------------
INPUT: Combined dataset, selected features
PROCESS:
- Impute missing values (median)
- Leave-One-Out cross-validation
  - Train on 739 windows (44-45 failures)
  - Predict held-out window
  - Repeat 740 times
- Calculate overall and per-farm metrics

OUTPUT: 
- Performance metrics (56% recall, 96% precision)
- Per-farm breakdown
- Per-event predictions
- Feature importance rankings

RESULTS: 714 predictions, enabling detailed analysis

Key Contributions
-----------------

1. METHODOLOGICAL: Semantic sensor mapping enables cross-farm learning
   - Maps 283 farm-specific sensors to 16 unified categories
   - Preserves physical meaning while enabling data combination
   - Demonstrates fundamental trade-off: generalizability vs. optimization
   - Documents both successes and limitations transparently

2. EMPIRICAL: Quantified multi-domain learning trade-offs
   - Limited-data benefit: +17% improvement (Farm B: 33% -> 50%)
   - Rich-data cost: -9% to -11% degradation (Farms A & C)
   - Net neutral: Overall system unchanged at 56% recall
   - Validates hypothesis: Aggregation helps small n, hurts large n

3. PRACTICAL: Production-ready implementation with known limitations
   - 6-step pipeline handles 81-952 sensors
   - Same code for all farms (transferable)
   - Documented merge rationale (reproducible)
   - Honest assessment of trade-offs (deployable)

4. ANALYTICAL: Discovered universal vs. farm-specific failure patterns
   - Universal: Mechanical degradation (rotor_mechanical, gearbox_mechanical)
   - Universal: Control instability (pitch_angle, generator_electrical)
   - Farm-specific: Granular sensor signatures lost in aggregation
   - Insight: Physics-based categories transfer, sensor-level details don't
   - 6-step pipeline
   - Handles 81-952 sensors
   - Same code for all farms
   - Deployment guidelines

4. ANALYTICAL: Discovered universal failure patterns
   - Mechanical degradation dominates
   - Pitch control instability
   - Environmental context matters
   - Physics-based features transfer better

Final Assessment
----------------

QUESTION: Should unified or farm-specific approach be used?

ANSWER: Farm-specific recommended when possible

USE UNIFIED IF:
- New/small farm (<10 failures historical)
- Want to deploy quickly (6 months vs 2 years)
- Have multiple similar sites (fleet learning)
- Need consistent baseline performance (50%+)
- Willing to accept lower peak performance

USE FARM-SPECIFIC IF:
- Established farm (10+ failures)
- Want maximum performance (60-67%)
- Single-site optimization matters
- Have time to collect data (1-2 years)
- Can afford farm-specific development

HYBRID APPROACH:
- Start with unified for initial deployment
- Collect farm-specific data over 6-12 months
- Add top farm-specific features to unified baseline
- Best of both: transferability + optimization
- Potential: 56% -> 60-62% with hybrid features



Last Updated: December 2025
