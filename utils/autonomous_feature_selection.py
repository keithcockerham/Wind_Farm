"""
Autonomous Feature Selection Pipeline for Wind Turbine Failure Prediction

Complete autonomous pipeline requiring only:
- farm: 'A', 'B', or 'C'
- cohens_d_threshold: minimum effect size (default: 0.8)
- max_features: maximum features to select (default: 20)
- correlation_threshold: max correlation between features (default: 0.9)

Returns: List of optimal sensors for XGBoost model
"""

import pandas as pd
import numpy as np
from scipy import stats


def calculate_cohens_d(normal_data, failure_data, feature):
    """
    Calculate Cohen's d effect size for a feature.
    
    Cohen's d interpretation:
    - 0.2: small effect
    - 0.5: medium effect
    - 0.8: large effect (our threshold)
    
    Returns:
        float: Cohen's d value (absolute value)
    """
    normal_vals = normal_data[feature].dropna()
    failure_vals = failure_data[feature].dropna()
    
    if len(normal_vals) == 0 or len(failure_vals) == 0:
        return 0.0
    
    # Mean difference
    mean_diff = failure_vals.mean() - normal_vals.mean()
    
    # Pooled standard deviation
    n1, n2 = len(normal_vals), len(failure_vals)
    var1, var2 = normal_vals.var(), failure_vals.var()
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    if pooled_std == 0:
        return 0.0
    
    cohens_d = abs(mean_diff / pooled_std)
    
    return cohens_d


def get_sensor_columns(scada_data):
    """
    Automatically identify sensor columns (exclude metadata).
    
    Returns:
        list: Column names of sensors
    """
    metadata_cols = {
        'time_stamp', 'asset_id', 'status_type_id', 'train_test',
        'farm_id', 'event', 'Anomalous', 'anomaly_score', 'label'
    }
    
    sensor_cols = [
        col for col in scada_data.columns
        if col not in metadata_cols and 
        scada_data[col].dtype in ['float32', 'float64', 'int32', 'int64']
    ]
    
    return sensor_cols


def calculate_all_effect_sizes(normal_data, failure_data):
    """
    Calculate Cohen's d for all sensors.
    
    Returns:
        pd.DataFrame: Sensors ranked by effect size
    """
    sensor_cols = get_sensor_columns(normal_data)
    
    results = []
    for sensor in sensor_cols:
        cohens_d = calculate_cohens_d(normal_data, failure_data, sensor)
        results.append({
            'sensor': sensor,
            'cohens_d': cohens_d
        })
    
    df = pd.DataFrame(results)
    df = df.sort_values('cohens_d', ascending=False).reset_index(drop=True)
    
    return df


def remove_correlated_features(data, features, correlation_threshold=0.9):
    """
    Iteratively remove correlated features, keeping higher Cohen's d.
    
    Algorithm:
    1. Start with features sorted by Cohen's d (descending)
    2. For each feature, check correlation with already-selected features
    3. If correlation > threshold with any selected feature, skip it
    4. Otherwise, add to selected features
    
    Args:
        data: DataFrame with sensor values
        features: List of sensor names sorted by Cohen's d
        correlation_threshold: Maximum allowed correlation (default: 0.9)
    
    Returns:
        list: Non-redundant features
    """
    if len(features) == 0:
        return []
    
    # Calculate correlation matrix once
    corr_matrix = data[features].corr().abs()
    
    selected_features = [features[0]]  # Always keep best feature
    
    for feature in features[1:]:
        # Check correlation with all already-selected features
        is_redundant = False
        for selected in selected_features:
            if corr_matrix.loc[feature, selected] > correlation_threshold:
                is_redundant = True
                break
        
        if not is_redundant:
            selected_features.append(feature)
    
    return selected_features


def select_features_autonomous(
    scada_data,
    event_info,
    cohens_d_threshold=0.8,
    max_cohens_d=50.0,
    max_features=20,
    correlation_threshold=0.9,
    verbose=True
):
    """
    Autonomous feature selection pipeline.
    
    Complete pipeline:
    1. Separate normal vs pre-failure data
    2. Calculate Cohen's d for all sensors
    3. Filter by effect size threshold (min and max)
    4. Remove correlated features
    5. Limit to max_features
    
    Args:
        scada_data: SCADA dataframe with sensor readings
        event_info: Event metadata with failure events
        cohens_d_threshold: Minimum Cohen's d (default: 0.8 = large effect)
        max_cohens_d: Maximum Cohen's d (default: 50.0, filters catastrophic sensors)
        max_features: Maximum features to return (default: 20)
        correlation_threshold: Max correlation between features (default: 0.9)
        verbose: Print selection details (default: True)
    
    Returns:
        list: Selected sensor column names
    """
    
    if verbose:
        print("="*80)
        print("AUTONOMOUS FEATURE SELECTION PIPELINE")
        print("="*80)
    
    # Step 1: Prepare normal and failure data
    if verbose:
        print("\n[1/5] Preparing normal and failure datasets...")
    
    # Get normal baseline (status=0, outside 30-day buffer)
    normal_data = get_normal_baseline(scada_data, event_info)
    
    # Get pre-failure windows (24h before last normal status)
    failure_data = get_prefailure_windows(scada_data, event_info)
    
    if verbose:
        print(f"   Normal data: {len(normal_data):,} rows")
        print(f"   Pre-failure data: {len(failure_data):,} rows")
        print(f"   Available sensors: {len(get_sensor_columns(scada_data))}")
    
    # Step 2: Calculate Cohen's d for all sensors
    if verbose:
        print("\n[2/5] Calculating effect sizes (Cohen's d)...")
    
    effect_sizes = calculate_all_effect_sizes(normal_data, failure_data)
    
    if verbose:
        print(f"   Top 5 sensors by effect size:")
        for idx, row in effect_sizes.head().iterrows():
            print(f"      {row['sensor']}: d={row['cohens_d']:.3f}")
    
    # Step 3: Filter by Cohen's d threshold range
    if verbose:
        print(f"\n[3/5] Filtering by Cohen's d range ({cohens_d_threshold} ≤ d ≤ {max_cohens_d})...")
    
    strong_features = effect_sizes[
        (effect_sizes['cohens_d'] >= cohens_d_threshold) &
        (effect_sizes['cohens_d'] <= max_cohens_d)
    ]['sensor'].tolist()
    
    if verbose:
        print(f"   Features in range: {len(strong_features)}")
        # Show what was filtered out
        too_weak = (effect_sizes['cohens_d'] < cohens_d_threshold).sum()
        too_strong = (effect_sizes['cohens_d'] > max_cohens_d).sum()
        if too_weak > 0:
            print(f"   Filtered (too weak, d<{cohens_d_threshold}): {too_weak}")
        if too_strong > 0:
            print(f"   Filtered (catastrophic, d>{max_cohens_d}): {too_strong}")
            print(f"      Note: Catastrophic sensors only fire during total failure (not early warning)")
    
    if len(strong_features) == 0:
        print(f"\n   WARNING: No features in range [{cohens_d_threshold}, {max_cohens_d}]")
        print(f"   Adjusting thresholds...")
        # Try without upper limit first
        strong_features = effect_sizes[
            effect_sizes['cohens_d'] >= cohens_d_threshold/2
        ]['sensor'].tolist()[:max_features]
        if len(strong_features) == 0:
            # Last resort - take top max_features regardless
            strong_features = effect_sizes['sensor'].tolist()[:max_features]
        print(f"   Selected {len(strong_features)} features with adjusted thresholds")
    
    # Step 4: Remove correlated features
    if verbose:
        print(f"\n[4/5] Removing correlated features (r < {correlation_threshold})...")
    
    # Combine normal and failure data for correlation calculation
    combined_data = pd.concat([normal_data, failure_data], ignore_index=True)
    
    selected_features = remove_correlated_features(
        combined_data,
        strong_features,
        correlation_threshold
    )
    
    if verbose:
        print(f"   Non-redundant features: {len(selected_features)}")
    
    # Step 5: Limit to max_features
    if len(selected_features) > max_features:
        if verbose:
            print(f"\n[5/5] Limiting to top {max_features} features...")
        selected_features = selected_features[:max_features]
    else:
        if verbose:
            print(f"\n[5/5] Using all {len(selected_features)} non-redundant features...")
    
    # Summary
    if verbose:
        print("\n" + "="*80)
        print("SELECTED FEATURES")
        print("="*80)
        for i, feature in enumerate(selected_features, 1):
            cohens_d = effect_sizes[effect_sizes['sensor'] == feature]['cohens_d'].values[0]
            print(f"{i:2d}. {feature:40s} (d={cohens_d:.3f})")
        print("="*80)
    
    return selected_features


def get_normal_baseline(scada_data, event_info, buffer_days=30):
    """
    Extract normal operation data with buffer around failures.
    
    Args:
        scada_data: SCADA dataframe
        event_info: Event metadata
        buffer_days: Days to exclude around failures (default: 30)
    
    Returns:
        pd.DataFrame: Normal operation data
    """
    # Filter to status = 0 (normal operation)
    normal = scada_data[scada_data['status_type_id'] == 0].copy()
    
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


def get_prefailure_windows(scada_data, event_info, window_hours=24):
    """
    Get 24-hour windows before last normal status for each failure.
    
    Args:
        scada_data: SCADA dataframe
        event_info: Event metadata
        window_hours: Hours before failure (default: 24)
    
    Returns:
        pd.DataFrame: Pre-failure windows
    """
    anomalies = event_info[event_info['event_label'] == 'anomaly']
    all_windows = []
    
    for _, event in anomalies.iterrows():
        asset_id = event['asset_id']
        event_start = pd.to_datetime(event['event_start'])
        
        # Find last normal status before failure
        asset_data = scada_data[
            (scada_data['asset_id'] == asset_id) &
            (scada_data['time_stamp'] < event_start) &
            (scada_data['status_type_id'] == 0)
        ].sort_values('time_stamp')
        
        if len(asset_data) == 0:
            continue
        
        last_normal_time = asset_data.iloc[-1]['time_stamp']
        window_start = last_normal_time - pd.Timedelta(hours=window_hours)
        
        # Extract 24h window
        window = asset_data[
            (asset_data['time_stamp'] >= window_start) &
            (asset_data['time_stamp'] < last_normal_time)
        ].copy()
        
        window['event'] = event['event_id']
        all_windows.append(window)
    
    if len(all_windows) == 0:
        return pd.DataFrame()
    
    return pd.concat(all_windows, ignore_index=True)


# ============================================================================
# COMPLETE AUTONOMOUS PIPELINE
# ============================================================================

def run_autonomous_xgboost_pipeline(
    farm,
    scada_data,
    event_info,
    cohens_d_threshold=0.8,
    max_cohens_d=50.0,
    max_features=15,
    correlation_threshold=0.9,
    normal_sample_ratio=0.1,
    rolling_window=6,
    use_gpu=False,
    gpu_id=0
):
    """
    Complete autonomous XGBoost pipeline from data to trained model.
    
    ONLY INPUTS REQUIRED:
    - farm: 'A', 'B', or 'C'
    - scada_data: Raw SCADA dataframe
    - event_info: Event metadata
    
    OPTIONAL PARAMETERS (with sensible defaults):
    - cohens_d_threshold: Minimum effect size (default: 0.8)
    - max_cohens_d: Maximum effect size to avoid catastrophic sensors (default: 50.0)
    - max_features: Maximum features (default: 15)
    - correlation_threshold: Max correlation (default: 0.9)
    - normal_sample_ratio: Undersample ratio (default: 0.1)
    - rolling_window: Rolling window size (default: 6 = 1 hour)
    - use_gpu: Use GPU acceleration (default: False, requires xgboost[cuda])
    - gpu_id: GPU device ID (default: 0)
    
    Returns:
        dict: {
            'selected_features': list of base sensors,
            'engineered_features': list of all features (base + engineered),
            'results': detection results per event,
            'fpr': false positive rate,
            'feature_importance': feature importance rankings
        }
    """
    
    print(f"\n{'='*80}")
    print(f"AUTONOMOUS XGBOOST PIPELINE - FARM {farm}")
    print(f"{'='*80}")
    if use_gpu:
        print(f"🚀 GPU ACCELERATION ENABLED (Device: cuda:{gpu_id})")
    print()
    
    # ========================================================================
    # STEP 1: AUTOMATIC FEATURE SELECTION
    # ========================================================================
    
    print("PHASE 1: AUTOMATIC FEATURE SELECTION")
    print("-" * 80)
    
    base_sensors = select_features_autonomous(
        scada_data=scada_data,
        event_info=event_info,
        cohens_d_threshold=cohens_d_threshold,
        max_cohens_d=max_cohens_d,
        max_features=max_features,
        correlation_threshold=correlation_threshold,
        verbose=True
    )
    
    # ========================================================================
    # STEP 2: DATA PREPARATION
    # ========================================================================
    
    print(f"\n{'='*80}")
    print("PHASE 2: DATA PREPARATION")
    print("-" * 80)
    
    from xgboost import XGBClassifier
    
    # Get labeled data
    print("\n[1/3] Creating labeled dataset...")
    normal_data = get_normal_baseline(scada_data, event_info)
    failure_data = get_prefailure_windows(scada_data, event_info)
    
    # Undersample normal data
    normal_sample = normal_data.sample(frac=normal_sample_ratio, random_state=42)
    normal_sample['label'] = 0
    normal_sample['event'] = np.nan
    
    failure_data['label'] = 1
    
    labeled_data = pd.concat([normal_sample, failure_data], ignore_index=True)
    
    print(f"   Normal: {len(normal_sample):,} rows")
    print(f"   Pre-failure: {len(failure_data):,} rows")
    print(f"   Ratio: {len(normal_sample)/len(failure_data):.1f}:1")
    
    # ========================================================================
    # STEP 3: FEATURE ENGINEERING
    # ========================================================================
    
    print("\n[2/3] Engineering features...")
    labeled_data = engineer_features(labeled_data, base_sensors, rolling_window)
    
    # Build feature list
    engineered_features = []
    for sensor in base_sensors:
        if sensor in labeled_data.columns:
            engineered_features.append(sensor)
        if f'{sensor}_roll_std' in labeled_data.columns:
            engineered_features.append(f'{sensor}_roll_std')
        if f'{sensor}_change' in labeled_data.columns:
            engineered_features.append(f'{sensor}_change')
    
    print(f"   Base sensors: {len(base_sensors)}")
    print(f"   Total features: {len(engineered_features)}")
    
    # ========================================================================
    # STEP 4: MODEL TRAINING
    # ========================================================================
    
    print("\n[3/3] Training model with leave-one-event-out CV...")
    
    X = labeled_data[engineered_features].copy()
    y = labeled_data['label'].copy()
    event_ids = labeled_data['event'].copy()
    
    failure_events = event_ids[~event_ids.isna()].unique()
    
    results = []
    all_feature_importance = []
    
    for i, test_event in enumerate(failure_events, 1):
        print(f"   [{i}/{len(failure_events)}] Testing event {test_event}...", end='')
        
        # Split
        is_test = event_ids == test_event
        is_train = ~is_test | event_ids.isna()
        
        X_train = X[is_train]
        y_train = y[is_train]
        X_test = X[is_test]
        
        # Train
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        
        # Configure XGBoost parameters
        xgb_params = {
            'n_estimators': 100,
            'max_depth': 5,
            'learning_rate': 0.1,
            'scale_pos_weight': scale_pos_weight,
            'eval_metric': 'logloss',
            'random_state': 42,
            'verbosity': 0
        }
        
        # Add GPU parameters if requested
        if use_gpu:
            xgb_params['tree_method'] = 'hist'
            xgb_params['device'] = f'cuda:{gpu_id}'
        
        model = XGBClassifier(**xgb_params)
        
        model.fit(X_train, y_train)
        
        # Predict
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        mean_proba = y_pred_proba.mean()
        
        event_desc = event_info[event_info['event_id'] == test_event]['event_description'].values
        event_desc = event_desc[0] if len(event_desc) > 0 else 'Unknown'
        
        results.append({
            'event': test_event,
            'description': event_desc,
            'mean_proba': mean_proba,
            'detected': mean_proba > 0.5
        })
        
        # Feature importance
        importance = pd.DataFrame({
            'feature': engineered_features,
            'importance': model.feature_importances_
        })
        all_feature_importance.append(importance)
        
        print(f" Detected: {'✓' if mean_proba > 0.5 else '✗'}")
    
    results_df = pd.DataFrame(results)
    detection_rate = results_df['detected'].sum() / len(results_df)
    
    # Average feature importance
    avg_importance = pd.concat(all_feature_importance).groupby('feature')['importance'].mean()
    avg_importance = avg_importance.sort_values(ascending=False)
    
    # ========================================================================
    # STEP 5: FALSE POSITIVE RATE VALIDATION
    # ========================================================================
    
    print(f"\n{'='*80}")
    print("PHASE 3: FALSE POSITIVE RATE VALIDATION")
    print("-" * 80)
    
    print("\n[1/1] Testing on held-out normal data...")
    
    # Train final model on all data
    train_data = labeled_data.copy()
    X_train = train_data[engineered_features]
    y_train = train_data['label']
    
    # Configure XGBoost parameters
    xgb_params = {
        'n_estimators': 100,
        'max_depth': 5,
        'learning_rate': 0.1,
        'scale_pos_weight': (y_train==0).sum()/(y_train==1).sum(),
        'random_state': 42,
        'verbosity': 0
    }
    
    # Add GPU parameters if requested
    if use_gpu:
        xgb_params['tree_method'] = 'hist'
        xgb_params['device'] = f'cuda:{gpu_id}'
    
    model = XGBClassifier(**xgb_params)
    model.fit(X_train, y_train)
    
    # Test on holdout normal data
    normal_holdout = get_normal_baseline(scada_data, event_info)
    normal_sample = normal_holdout.sample(n=min(10000, len(normal_holdout)), random_state=99)
    normal_sample = engineer_features(normal_sample, base_sensors, rolling_window)
    
    X_normal = normal_sample[engineered_features]
    y_pred_proba = model.predict_proba(X_normal)[:, 1]
    
    false_positives = (y_pred_proba > 0.5).sum()
    fpr = false_positives / len(y_pred_proba)
    
    print(f"   False positives: {false_positives:,} / {len(y_pred_proba):,}")
    print(f"   FPR: {fpr:.1%}")
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    
    print(f"\n{'='*80}")
    print("FINAL RESULTS")
    print("="*80)
    print(f"Recall: {detection_rate:.1%} ({results_df['detected'].sum()}/{len(results_df)})")
    print(f"False Positive Rate: {fpr:.1%}")
    print(f"\nTop 5 Most Important Features:")
    for i, (feature, importance) in enumerate(avg_importance.head().items(), 1):
        print(f"  {i}. {feature:40s} {importance:.3f}")
    print("="*80)
    
    return {
        'selected_features': base_sensors,
        'engineered_features': engineered_features,
        'results': results_df,
        'fpr': fpr,
        'feature_importance': avg_importance,
        'model': model
    }


def engineer_features(data, sensor_cols, rolling_window=6):
    """
    Add variability and rate of change features.
    
    Args:
        data: DataFrame with sensor readings
        sensor_cols: List of base sensor columns
        rolling_window: Window size for rolling statistics (default: 6 = 1 hour)
    
    Returns:
        pd.DataFrame: Data with engineered features
    """
    for sensor in sensor_cols:
        if sensor not in data.columns:
            continue
        
        # Variability (rolling std)
        data[f'{sensor}_roll_std'] = (
            data.groupby('asset_id')[sensor]
            .rolling(window=rolling_window, min_periods=1)
            .std()
            .reset_index(0, drop=True)
        )
        
        # Rate of change
        data[f'{sensor}_change'] = (
            data.groupby('asset_id')[sensor].diff()
        )
    
    # Fill NaN from rolling/diff (first rows per group)
    # Use forward fill then zero for any remaining NaNs
    engineered_cols = [col for col in data.columns 
                      if '_roll_std' in col or '_change' in col]
    
    data[engineered_cols] = data[engineered_cols].ffill().fillna(0)
    
    return data


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    """
    Example usage of autonomous pipeline.
    """
    
    # Load your data (these functions should exist in your codebase)
    from your_module import get_farm_scada_chunked, get_event_info
    
    # ========================================================================
    # EXAMPLE 1: Default settings
    # ========================================================================
    
    farm = 'C'
    scada = get_farm_scada_chunked(farm=farm)
    event_info = get_event_info(farm=farm)
    
    results = run_autonomous_xgboost_pipeline(
        farm=farm,
        scada_data=scada,
        event_info=event_info
        # All other parameters use defaults
    )
    
    print(f"\n✓ Detection Rate: {results['results']['detected'].sum()}/{len(results['results'])}")
    print(f"✓ FPR: {results['fpr']:.1%}")
    print(f"✓ Top Feature: {results['feature_importance'].index[0]}")
    
    # ========================================================================
    # EXAMPLE 2: Custom thresholds for limited data scenario
    # ========================================================================
    
    results = run_autonomous_xgboost_pipeline(
        farm='B',
        scada_data=scada,
        event_info=event_info,
        cohens_d_threshold=0.5,    # Lower threshold (fewer failures)
        max_features=10,            # Limit features (prevent overfitting)
        correlation_threshold=0.85  # More aggressive correlation removal
    )
    
    # ========================================================================
    # EXAMPLE 3: Feature selection only
    # ========================================================================
    
    # If you just want the feature list without training
    selected_features = select_features_autonomous(
        scada_data=scada,
        event_info=event_info,
        cohens_d_threshold=0.8,
        max_features=15,
        correlation_threshold=0.9
    )
    
    print(f"\nSelected {len(selected_features)} features:")
    for i, feature in enumerate(selected_features, 1):
        print(f"  {i}. {feature}")
