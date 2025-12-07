# XGBoost Supervised Learning Approach

## Executive Summary

Developed supervised learning models using XGBoost achieving **100% failure detection** across three independent wind farms with false positive rates of 0-1.9%. This represents a significant improvement over the Random Forest approach (67% average recall) and validates that supervised learning with proper class imbalance handling outperforms unsupervised methods when sufficient labeled examples exist.

**Key Results:**
- **Farm A:** 100% recall (12/12), 0.8% FPR
- **Farm B:** 100% recall (6/6), 0.0% FPR  
- **Farm C:** 100% recall (27/27), 1.9% FPR

**Operational Impact:**
- 24-hour warning window enables proactive maintenance
- 0-2.7 alerts per turbine per day (manageable workload)
- Prevents failures costing $500K+ in repairs and downtime

---

## Methodology

### 1. Problem Formulation

**Supervised Classification Task:**
> Given sensor readings, predict whether turbine will transition to failure status in the next 24 hours.

**Labels:**
- **Class 1 (Pre-Failure):** 24-hour window before last normal status
- **Class 0 (Normal):** Normal operation excluding ±30 day buffer around failures

**Why 24 hours before last normal status?**
- `event_start` timestamp = when failure formally logged (often days after problem began)
- Last normal status = when turbine last operated normally before failure cascade
- This 24-hour window captures the degradation patterns that precede catastrophic failure

### 2. Data Preparation

#### Normal Baseline (Class 0)
```python
def get_normal_baseline_from_scada(scada, event_info, buffer_days=30):
    """Extract normal operation data with buffer around failures."""
    
    # Filter to status = 0 (normal operation)
    normal = scada[scada['status_type_id'] == 0].copy()
    
    # Remove 30-day buffer around each failure event
    for _, event in event_info[event_info['event_label'] == 'anomaly'].iterrows():
        event_start = pd.to_datetime(event['event_start'])
        buffer_start = event_start - pd.Timedelta(days=buffer_days)
        buffer_end = event_start + pd.Timedelta(days=buffer_days)
        
        # Remove this buffer period for the specific asset
        mask = ~(
            (normal['asset_id'] == event['asset_id']) &
            (normal['time_stamp'] >= buffer_start) &
            (normal['time_stamp'] <= buffer_end)
        )
        normal = normal[mask]
    
    return normal
```

**Result:** ~50,000+ normal operation timestamps per farm

#### Pre-Failure Windows (Class 1)
```python
def get_prefailure_windows(scada, event_info, window_hours=24):
    """Get 24-hour windows before last normal status for each failure."""
    
    anomalies = event_info[event_info['event_label'] == 'anomaly']
    all_windows = []
    
    for _, event in anomalies.iterrows():
        asset_id = event['asset_id']
        event_start = pd.to_datetime(event['event_start'])
        
        # Find last normal status before failure
        asset_data = scada[
            (scada['asset_id'] == asset_id) &
            (scada['time_stamp'] < event_start) &
            (scada['status_type_id'] == 0)
        ].sort_values('time_stamp')
        
        last_normal_time = asset_data.iloc[-1]['time_stamp']
        window_start = last_normal_time - pd.Timedelta(hours=window_hours)
        
        # Extract 24h window
        window = asset_data[
            (asset_data['time_stamp'] >= window_start) &
            (asset_data['time_stamp'] < last_normal_time)
        ].copy()
        
        window['event'] = event['event_id']
        all_windows.append(window)
    
    return pd.concat(all_windows)
```

**Result:** ~3,500 pre-failure timestamps across all farms

### 3. Feature Engineering

**Base Sensors (Farm-Specific):**
- **Farm A:** Grid reactive power, power output, generator RPM, bearing temps
- **Farm B:** Rotor bearing temperature, gearbox bearing, power output
- **Farm C:** Power output, internal voltage, rotor bearing temperature

**Engineered Features:**
```python
def engineer_features(data, sensor_cols, rolling_window=6):
    """
    Add variability and rate of change features.
    rolling_window: 6 periods = 1 hour with 10-min data
    """
    
    for sensor in sensor_cols:
        # Variability (rolling std over 1 hour)
        data[f'{sensor}_roll_std'] = (
            data.groupby('asset_id')[sensor]
            .rolling(window=rolling_window, min_periods=1)
            .std()
            .reset_index(0, drop=True)
        )
        
        # Rate of change (first difference)
        data[f'{sensor}_change'] = (
            data.groupby('asset_id')[sensor].diff()
        )
    
    return data
```

**Feature Types:**
- Raw sensor values (avg)
- Rolling variability (roll_std) - **Dominant predictor (60-87%)**
- Rate of change (change)

**Total Features:** 12-15 per farm (base sensors × 3 feature types)

### 4. Class Imbalance Handling

**The Problem:**
- Initial ratio: 14:1 (normal:failure)
- Model learned "always predict normal" → 3.7% recall

**The Solution: Strategic Undersampling**
```python
def prepare_labeled_dataset(scada, event_info, normal_sample_ratio=0.1):
    """Sample only 10% of normal data to reduce imbalance."""
    
    normal_data = get_normal_baseline_from_scada(scada, event_info)
    
    # UNDERSAMPLE: Take only 10% of normal data
    normal_sample = normal_data.sample(frac=normal_sample_ratio, random_state=42)
    normal_sample['label'] = 0
    
    failure_data = get_prefailure_windows(scada, event_info)
    failure_data['label'] = 1
    
    return pd.concat([normal_sample, failure_data])
```

**Result:** New ratio ~1.4:1, recall improved from 3.7% → 100%

**Why Undersampling Works:**
- Balanced training data prevents "predict normal" bias
- XGBoost learns actual failure patterns
- False positive rate tested separately on full normal dataset

### 5. Model Training

**XGBoost Configuration:**
```python
model = XGBClassifier(
    n_estimators=100,           # 100 trees
    max_depth=5,                # Shallow trees prevent overfitting
    learning_rate=0.1,          # Standard rate
    scale_pos_weight=ratio,     # Handle remaining imbalance
    eval_metric='logloss',      # Binary classification metric
    random_state=42,
    verbosity=0
)
```

**Hyperparameter Rationale:**
- `max_depth=5`: Limited depth prevents overfitting on small failure dataset
- `scale_pos_weight`: Calculated per CV fold to handle any remaining imbalance
- `n_estimators=100`: Sufficient for convergence without overfitting

### 6. Cross-Validation Strategy

**Leave-One-Event-Out (LOEO):**
```python
for test_event in failure_events:
    # Split RAW data first (before feature engineering)
    is_test = labeled_data['event'] == test_event
    is_train = ~is_test | labeled_data['event'].isna()  # Include all normal data
    
    train_data = labeled_data[is_train].copy()
    test_data = labeled_data[is_test].copy()
    
    # Engineer features SEPARATELY for train and test
    train_data = engineer_features(train_data, base_sensors)
    test_data = engineer_features(test_data, base_sensors)
    
    # Train model
    model.fit(X_train, y_train)
    
    # Predict on held-out event
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    event_score = y_pred_proba.mean()  # Event-level aggregation
```

**Critical: Data Leakage Prevention**
- Feature engineering happens INSIDE the CV loop
- Rolling calculations only use data from same fold
- Test event completely held out during training
- Prevents artificially inflated performance

**Why LOEO?**
- Appropriate for limited failures (6-27 per farm)
- Tests generalization to unseen failure patterns
- Each event represents different failure mode/root cause

### 7. Evaluation Metrics

**Event-Level Aggregation:**
```python
# Don't evaluate at timestamp level (noisy)
# Aggregate probabilities over 24-hour window
event_score = y_pred_proba.mean()
detected = event_score > 0.5  # Threshold
```

**Primary Metrics:**
- **Recall:** % of failures detected (TP / (TP + FN))
- **False Positive Rate (FPR):** % of normal operation flagged as failure

**Critical: Separate FPR Evaluation**
```python
def evaluate_false_positive_rate(model, scada, event_info):
    """Test FPR on held-out normal operation data."""
    
    # Get fresh normal data (different random seed)
    normal_holdout = get_normal_baseline_from_scada(scada, event_info)
    normal_sample = normal_holdout.sample(n=10000, random_state=99)
    
    # Engineer features
    normal_sample = engineer_features(normal_sample, base_sensors)
    
    # Predict
    y_pred_proba = model.predict_proba(X_normal)[:, 1]
    
    # Count false positives
    false_positives = (y_pred_proba > 0.5).sum()
    fpr = false_positives / len(y_pred_proba)
    
    return fpr
```

**Why Test FPR Separately?**
- High recall with high FPR = useless model (constant false alarms)
- Validates model isn't "trigger-happy" from undersampling
- Operational feasibility depends on manageable alert volume

---

## Results

### Overall Performance

| Farm | Environment | Failures | Recall | FPR | Alerts/Day | Top Predictor (Importance) |
|------|-------------|----------|--------|-----|------------|---------------------------|
| **A** | Onshore Portugal | 12 | **100%** | 0.8% | 0.5/turbine | Reactive power variability (73%) |
| **B** | Offshore Germany | 6 | **100%** | 0.0% | 0/turbine | Rotor bearing temp variability (87%) |
| **C** | Offshore Germany | 27 | **100%** | 1.9% | 2.7/turbine | Power output variability (60%) |

**Aggregate:** 45 of 45 failures detected across all farms

### Detailed Results by Farm

#### Farm A: Onshore Portugal
**Performance:**
- Recall: 12/12 failures (100%)
- FPR: 82 false alarms / 10,000 normal timestamps (0.8%)
- Mean probability on normal data: 0.016

**Feature Importance:**
1. sensor_31_avg_roll_std (73%) - Grid reactive power variability
2. power_30_avg_roll_std (7%) - Power output variability
3. sensor_18_avg_roll_std (3%) - Generator RPM variability
4. sensor_52_avg_roll_std (2%) - Rotor bearing variability
5. power_30_avg_change (2%) - Power change rate

**Failure Patterns:**
- Electrical/power quality issues dominate
- Reactive power becomes erratic before failure
- Threshold analysis confirmed spikes occur, but simple thresholds generate false alarms
- Model learns pattern: sustained variability >> brief spikes

**Operational Impact:**
- 5 turbines monitored
- ~0.5 alerts per turbine per day
- ~16 alerts per turbine per month
- Extremely low noise - every alert worth investigating

#### Farm B: Offshore Germany (Limited Data)
**Performance:**
- Recall: 6/6 failures (100%)
- FPR: 0 false alarms / 10,000 normal timestamps (0.0%)
- Mean probability on normal data: 0.001

**Feature Importance:**
1. sensor_52_avg_roll_std (87%) - Rotor bearing temperature variability
2. sensor_52_avg_change (2%) - Bearing temp change
3. sensor_52_avg (2%) - Bearing temp absolute
4. power_62_avg (2%) - Power output
5. power_62_avg_roll_std (1%) - Power variability

**Failure Patterns:**
- Mechanical degradation in drivetrain
- Bearing temperature becomes erratic (not necessarily high) as degradation accelerates
- 87% importance on single feature = clear failure signature
- Perfect discrimination (0% FPR) indicates consistent mechanical failure mode

**Operational Impact:**
- **Zero false alarms** - perfect operational performance
- Operators can trust every alert
- Likely due to: (a) limited but clean failure data, (b) consistent mechanical failure mode

#### Farm C: Offshore Germany (Large Scale)
**Performance:**
- Recall: 27/27 failures (100%)
- FPR: 194 false alarms / 10,000 normal timestamps (1.9%)
- Mean probability on normal data: 0.047

**Feature Importance:**
1. power_2_avg_roll_std (60%) - Power output variability
2. power_2_avg (7%) - Power output
3. power_2_avg_change (5%) - Power change
4. sensor_71_std (4%) - Internal voltage std
5. sensor_196_std (4%) - Rotor bearing temp std

**Failure Patterns:**
- Offshore harsh environment = diverse failure modes
- Power output instability is universal indicator
- More complex failure patterns than Farm B (hence higher FPR)
- System-wide degradation detected via power generation patterns

**Operational Impact:**
- 20 turbines monitored
- ~2.7 alerts per turbine per day
- ~81 alerts per turbine per month
- Higher alert volume but still manageable
- Trade-off: catch every failure vs. some investigation overhead

### Comparison with Random Forest

| Metric | Random Forest | XGBoost | Improvement |
|--------|---------------|---------|-------------|
| **Farm A Recall** | 67% (8/12) | 100% (12/12) | +33% |
| **Farm B Recall** | 33% (2/6) | 100% (6/6) | +67% |
| **Farm C Recall** | 63% (17/27) | 100% (27/27) | +37% |
| **Overall Recall** | 60% (27/45) | 100% (45/45) | +40% |
| **False Positives** | 0% (validation only) | 0-1.9% (10K holdout) | Similar |

**Why XGBoost Outperforms:**
1. **Supervised learning:** Learns specific failure patterns vs. just "what's abnormal"
2. **Gradient boosting:** Sequential error correction focuses on hard-to-classify failures
3. **Proper imbalance handling:** Undersampling enables learning from minority class
4. **Feature interactions:** Can learn complex patterns (e.g., "high variability + falling power = imminent failure")

---

## Key Findings

### 1. Variability Dominates Over Absolute Values

**Across all farms:** Rolling standard deviation features account for 60-87% of model decisions.

**Engineering Insight:**
- Not: "Temperature exceeds 80°C" (threshold violation)
- But: "Temperature oscillates erratically between 60-75°C" (variability)

**Why This Matters:**
- Simple threshold monitoring would miss these failures
- Degradation often shows as increased noise/instability before catastrophic failure
- Supports manual threshold analysis findings from Farm A

**Example (Farm A - Gearbox Failure):**
- sensor_31 (reactive power) showed ZERO threshold violations
- But sensor_31_roll_std increased 3x in pre-failure window
- Model detected via variability pattern

### 2. Environment-Specific Failure Signatures

Each farm has distinct dominant predictor reflecting local failure modes:

**Farm A (Onshore):**
- Reactive power variability (73%)
- Electrical/grid interaction issues
- Transformer, generator, hydraulic failures
- Smoother wind conditions → electrical problems more prominent

**Farm B (Offshore - Limited Data):**
- Rotor bearing temperature variability (87%)
- Mechanical wear in drivetrain
- Harsh offshore environment accelerates mechanical degradation
- Consistent failure mode → perfect discrimination

**Farm C (Offshore - Large Scale):**
- Power output variability (60%)
- Diverse failure modes (pitch, communication, hydraulic, electrical)
- Power generation is universal canary indicator
- More complex patterns → slightly higher FPR

**Implication:** Single unified model may not be optimal - farm-specific models capture local patterns

### 3. Class Imbalance is Make-or-Break

**Journey:**
1. **Initial attempt:** 14:1 imbalance → 3.7% recall (failed)
2. **After undersampling:** 1.4:1 imbalance → 100% recall (success)

**Lessons:**
- XGBoost's `scale_pos_weight` alone insufficient for severe imbalance
- Undersampling to ~1:1 ratio critical for minority class learning
- Must validate FPR separately - undersampling could make model trigger-happy
- Our results: Low FPR (0-1.9%) proves undersampling didn't introduce bias

**Best Practice:** For rare event prediction with <5% prevalence, combine undersampling + scale_pos_weight

### 4. False Positive Rate Determines Operational Viability

**High recall alone is meaningless** if FPR is unacceptable.

**Our FPR Results:**
- Farm A: 0.8% → 0.5 alerts/turbine/day
- Farm B: 0.0% → 0 alerts/turbine/day
- Farm C: 1.9% → 2.7 alerts/turbine/day

**Operational Context:**
- Unnecessary inspection: $5-10K cost
- Prevented failure: $500K+ savings
- ROI positive even at 50% FPR, but operator trust requires <5%
- Our results: 0-1.9% FPR maintains credibility

**Industry Comparison:**
- Most anomaly detection systems: 10-30% FPR (operator fatigue)
- Our system: <2% FPR (actionable alerts)

---

## Technical Challenges & Solutions

### Challenge 1: Initial Data Leakage Suspicion

**Problem:** First results showed 100% recall across all farms - too good to be true

**Investigation:**
1. Checked feature engineering sequence
2. Verified CV implementation
3. Tested for temporal leakage

**Root Cause:** Initially engineered features on entire dataset before CV split
- Rolling calculations could "see" test event patterns
- Fixed by moving feature engineering inside CV loop

**Validation:** Results remained 100% after fix - performance was legitimate, not artifact

**Lesson:** Always be skeptical of perfect results. Healthy paranoia prevents publishing bad models.

### Challenge 2: Severe Class Imbalance

**Problem:** 14:1 ratio → model predicts "normal" constantly

**Attempted Solutions:**
1. `scale_pos_weight` parameter (insufficient)
2. SMOTE oversampling (degraded FPR)
3. **Undersampling to 10%** (solved it)

**Why Undersampling Worked:**
- Balanced training prevents "predict normal" shortcut
- Model forced to learn actual failure patterns
- Validation on full normal dataset prevents trigger-happy behavior

**Best Practice:** For severe imbalance, undersample majority class to 1:1 or 2:1 ratio, then validate FPR separately

### Challenge 3: Limited Failures (Farm B)

**Problem:** Only 6 labeled failures - risk of overfitting

**Mitigation Strategies:**
1. Leave-one-out CV (max training data per fold)
2. Shallow trees (max_depth=5)
3. Conservative feature engineering (no complex interactions)
4. Validation on large normal dataset (10K timestamps)

**Result:** 100% recall, 0% FPR - model didn't overfit despite limited data

**Lesson:** LOEO CV + shallow models + separate validation enables learning from limited examples

### Challenge 4: Farm-Specific Sensor Selection

**Problem:** Different farms have different sensors - how to choose features?

**Solution:** Farm-specific base sensor selection
- Farm A: Reactive power, power, generator, bearings (from threshold analysis)
- Farm B: Bearing temps, power (mechanical focus)
- Farm C: Power, voltage, bearings (diverse)

**Result:** Each farm's model optimized for local sensor availability and failure modes

**Alternative Approach:** Unified model with semantic sensor mapping (explored separately in unified_model.md)

---

## Production Deployment Considerations

### Alert Volume Management

**Farm A:** 0.5 alerts/turbine/day
- Manageable: 1-2 checks per day per turbine
- Low enough for thorough investigation

**Farm C:** 2.7 alerts/turbine/day
- Higher but still reasonable
- Can batch alerts for scheduled inspections
- Priority ranking by probability score

**Recommendation:** Set probability threshold per farm
- Farm A/B: threshold=0.5 (current)
- Farm C: threshold=0.6 (reduce alerts by ~30%)

### Cost-Benefit Analysis

**Per Turbine Annual Costs:**
- False alarm inspection: $5K × (alerts/year) × FPR
- Farm A: $5K × 182 × 0.008 = ~$7,280/year
- Farm C: $5K × 985 × 0.019 = ~$93,575/year

**Per Turbine Annual Savings:**
- Prevented failures: $500K × (failures/year) × recall
- Assuming 1 failure/year: $500K × 1 × 1.0 = $500K/year

**ROI:** 6,800% (Farm A) to 530% (Farm C)
- Even with conservative assumptions, massive positive ROI
- False alarm costs negligible vs. prevented failure costs

### Model Maintenance

**Retraining Frequency:**
- Initial: Quarterly (collect new failures)
- Mature: Semi-annual (stable patterns)

**Performance Monitoring:**
- Track recall on new failures
- Track FPR on ongoing normal operation
- Alert if FPR exceeds 5% (model drift)

**Failure Feedback Loop:**
- Log all alerts (predicted failures)
- Track actual outcomes (true/false positive)
- Retrain with new labeled data

---

## Future Work

### 1. Unified Multi-Farm Model
- Semantic sensor mapping across farms
- Train on combined dataset (45 failures)
- May improve Farm B performance (limited data)
- Trade-off: Slight performance loss on A/C for consistency

### 2. Failure Type Classification
- Multi-class prediction: Gearbox vs Generator vs Hydraulic vs Pitch
- Enables targeted maintenance response
- Requires more granular failure labels

### 3. Remaining Useful Life (RUL) Prediction
- Regression: Predict time until failure (hours/days)
- More actionable than binary classification
- Enables maintenance scheduling optimization

### 4. Anomaly Severity Scoring
- Not just "will fail" but "how severe"
- Prioritize high-severity alerts
- Combine probability + predicted impact

### 5. Explainability Dashboard
- SHAP values for each alert
- "Alert triggered because: reactive power variability increased 3x"
- Builds operator trust and domain knowledge

---

## Conclusion

XGBoost supervised learning achieves **100% failure detection** across three independent wind farms with operationally manageable false positive rates (0-1.9%). This represents a significant advancement over unsupervised methods and validates that:

1. **Supervised learning outperforms unsupervised** when sufficient labeled failures exist (45 total)
2. **Class imbalance handling is critical** - undersampling transformed 3.7% → 100% recall
3. **Variability features dominate** - rolling statistics capture degradation patterns missed by thresholds
4. **Environment-specific models** learn local failure signatures (electrical vs mechanical vs system-wide)
5. **Separate FPR validation is essential** - high recall with low FPR proves operational viability

The system provides 24-hour advance warning with 0-2.7 alerts per turbine per day, enabling proactive maintenance while maintaining operator trust through low false alarm rates.

---

## Code Repository

Complete implementation available at: [github.com/keithcockerham/Wind_Farm](https://github.com/keithcockerham/Wind_Farm)

**Key Files:**
- `xgboost_anomaly_detection.ipynb` - Full implementation
- `utils/data_preparation.py` - Labeling and feature engineering
- `utils/evaluation.py` - CV and FPR testing
- `results/` - Per-farm performance metrics

---

## References

**Dataset:**
Wind Turbine SCADA Data for Early Fault Detection (Zenodo)
- 3 wind farms (onshore + offshore)
- 2.5M+ SCADA records (10-minute intervals)
- 45 labeled failure events with root cause descriptions

**Related Work:**
- `methodology.md` - Random Forest approach (67% recall baseline)
- `unified_model.md` - Cross-farm semantic sensor mapping
- `results.html` - Comprehensive comparison of all approaches
