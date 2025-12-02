class WindTurbineFailurePipeline:
    """
    Reproducible pipeline for wind turbine failure prediction from SCADA data.
    Implements methodology developed on Wind Farm A.
    """
    
    def __init__(self, power_sensor, power_threshold=0.1, buffer_days=30, window_hours=24):
        """
        Parameters:
        -----------
        power_threshold : float
            Minimum power for "actual production" (default 0.1)
        buffer_days : int
            Exclusion buffer around failures for normal baseline (default 30)
        window_hours : int
            Prediction window length in hours (default 24)
        """
        self.power_threshold = power_threshold
        self.power_sensor = power_sensor
        self.buffer_days = buffer_days
        self.window_hours = window_hours
        self.selected_features = None
        self.feature_importance = None
        
    def validate_data_structure(self, scada, event_info):
        """
        Step 1: Validate data has required columns and structure
        """
        print("STEP 1: DATA VALIDATION")
        print("="*60)
        
        # Check SCADA columns
        required_scada = ['time_stamp', 'asset_id', 'status_type_id']
        missing_scada = [col for col in required_scada if col not in scada.columns]
        
        if missing_scada:
            raise ValueError(f"SCADA missing columns: {missing_scada}")
            
        # Check event_info columns
        required_events = ['event_id', 'asset_id', 'event_start', 'event_end', 'event_label']
        missing_events = [col for col in required_events if col not in event_info.columns]
        
        if missing_events:
            raise ValueError(f"Event info missing columns: {missing_events}")
        
        # Find power column (could be power_30_avg, power_avg, etc.)
        power_cols = [col for col in scada.columns if 'power' in col.lower() and 'avg' in col.lower()]
        if not power_cols:
            raise ValueError("No power column found in SCADA data")
        
        self.power_column = power_cols[0]  # Use first power column found
        
        print(f"✓ Data structure valid")
        print(f"  SCADA records: {len(scada):,}")
        print(f"  Unique assets: {scada['asset_id'].nunique()}")
        print(f"  Total events: {len(event_info)}")
        print(f"  Anomaly events: {(event_info['event_label']=='anomaly').sum()}")
        print(f"  Power column: {self.power_column}")
        print(f"  Status values: {sorted(scada['status_type_id'].unique())}")
        
    def analyze_temporal_patterns(self, scada, event_info):
        """
        Step 2: Analyze time gaps between last production and logged failures
        """
        print("\nSTEP 2: TEMPORAL PATTERN ANALYSIS")
        print("="*60)
        
        gaps = []
        for _, failure in event_info[event_info['event_label']=='anomaly'].iterrows():
            asset_id = failure['asset_id']
            failure_start = failure['event_start']
            
            production_data = scada[
                (scada['asset_id'] == asset_id) &
                (scada['time_stamp'] < failure_start) &
                (scada['status_type_id'] == 0) &
                (scada[self.power_column] > self.power_threshold)
            ].sort_values('time_stamp')
            
            if len(production_data) == 0:
                print(f"  WARNING: Event {failure['event_id']} has no production data before failure")
                continue
                
            last_production = production_data.iloc[-1]['time_stamp']
            gap = failure_start - last_production
            gaps.append({
                'event_id': failure['event_id'],
                'gap_hours': gap.total_seconds() / 3600,
                'gap_days': gap.total_seconds() / (3600*24)
            })
        
        gaps_df = pd.DataFrame(gaps)
        print(f"\n  Gap statistics (hours):")
        print(f"    Mean: {gaps_df['gap_hours'].mean():.1f}")
        print(f"    Median: {gaps_df['gap_hours'].median():.1f}")
        print(f"    Min: {gaps_df['gap_hours'].min():.1f}")
        print(f"    Max: {gaps_df['gap_hours'].max():.1f}")
        
        return gaps_df
        
    def build_feature_dataset(self, scada, event_info):
        """
        Step 3: Create aggregated feature dataset
        """
        print("\nSTEP 3: FEATURE ENGINEERING")
        print("="*60)
        
        sensor_cols = [col for col in scada.columns 
                      if col not in ['time_stamp', 'asset_id', 'status_type_id']]
        
        print(f"  Sensor columns: {len(sensor_cols)}")
        
        # Build failure windows
        failure_windows = self._extract_failure_windows(scada, event_info, sensor_cols)
        
        # Build normal baseline
        normal_windows = self._extract_normal_windows(scada, event_info, sensor_cols)
        
        # Combine
        full_dataset = pd.concat([failure_windows, normal_windows], ignore_index=True)
        
        print(f"\n  Dataset created:")
        print(f"    Total windows: {len(full_dataset)}")
        print(f"    Failure windows: {(full_dataset['label']==1).sum()}")
        print(f"    Normal windows: {(full_dataset['label']==0).sum()}")
        print(f"    Features: {len([c for c in full_dataset.columns if c.endswith(('_mean','_std','_trend'))])}")
        
        return full_dataset
    
    def _extract_failure_windows(self, scada, event_info, sensor_cols):
        """Extract 24h windows before each failure"""
        
        sensor_cols = [col for col in scada.columns 
                       if col not in ['time_stamp', 'asset_id', 'status_type_id']]
        
        all_windows = []
        
        # Process each failure event
        for event_id in event_info['event_id']:
            failure = event_info[event_info['event_id'] == event_id].iloc[0]
            asset_id = failure['asset_id']
            failure_start = failure['event_start']
            
            # Find last production time
            production_data = scada[
                (scada['asset_id'] == asset_id) &
                (scada['time_stamp'] < failure_start) &
                (scada['status_type_id'] == 0) &
                (scada[self.power_sensor] > self.power_threshold)
            ].sort_values('time_stamp')
            
            if len(production_data) == 0:
                print(f"Warning: No production data for event {event_id}")
                continue
            
            last_production = production_data.iloc[-1]['time_stamp']
            window_start = last_production - pd.Timedelta(hours=24)
            
            # Extract 24h window
            window_data = scada[
                (scada['asset_id'] == asset_id) &
                (scada['time_stamp'] >= window_start) &
                (scada['time_stamp'] <= last_production)
            ]
            
            if len(window_data) < 100:  # Ensure we have enough data
                print(f"Warning: Insufficient data for event {event_id} ({len(window_data)} records)")
                continue
            
            # Aggregate features
            features = aggregate_window_features(window_data, sensor_cols)
            features['label'] = 1  # Failure
            features['event_id'] = event_id
            features['asset_id'] = asset_id
            features['failure_type'] = failure['event_description']
            
            all_windows.append(features)
        
        # Convert to DataFrame
        feature_df = pd.DataFrame(all_windows)
        
        print(f"\nCreated features for {len(feature_df)} failure windows")
        print(f"Feature count: {len([c for c in feature_df.columns if c not in ['label', 'event_id', 'asset_id', 'failure_type']])}")
        
        return feature_df
        
    
    def _extract_normal_windows(self, scada, event_info, sensor_cols, windows_per_asset=20):
        sensor_cols = [col for col in scada.columns 
                   if col not in ['time_stamp', 'asset_id', 'status_type_id']]
    
        all_normal_windows = []
        
        # Get all unique assets
        assets = scada['asset_id'].unique()
        
        for asset_id in assets:
            print(f"\nProcessing asset {asset_id}...")
            
            # Get all failure timestamps for this asset
            asset_failures = event_info[event_info['asset_id'] == asset_id]
            
            # Build exclusion zones (±30 days around each failure)
            exclusion_zones = []
            for _, failure in asset_failures.iterrows():
                start_exclude = failure['event_start'] - pd.Timedelta(days=self.buffer_days)
                end_exclude = failure['event_end'] + pd.Timedelta(days=self.buffer_days)
                exclusion_zones.append((start_exclude, end_exclude))
            
            # Get asset data in normal operation
            asset_data = scada[
                (scada['asset_id'] == asset_id) &
                (scada['status_type_id'] == 0) &
                (scada[self.power_sensor] > self.power_threshold)
            ].copy()
            
            # Filter out exclusion zones
            mask = pd.Series(True, index=asset_data.index)
            for start_ex, end_ex in exclusion_zones:
                mask &= ~((asset_data['time_stamp'] >= start_ex) & 
                          (asset_data['time_stamp'] <= end_ex))
            
            normal_data = asset_data[mask].sort_values('time_stamp')
            
            if len(normal_data) < 144 * windows_per_asset:
                print(f"  Warning: Limited normal data for asset {asset_id}")
                continue
            
            # Sample random 24h windows
            sampled_windows = 0
            max_attempts = windows_per_asset * 3
            attempts = 0
            
            while sampled_windows < windows_per_asset and attempts < max_attempts:
                attempts += 1
                
                # Pick random starting point
                max_start_idx = len(normal_data) - 145
                if max_start_idx < 0:
                    break
                    
                start_idx = np.random.randint(0, max_start_idx)
                window_candidate = normal_data.iloc[start_idx:start_idx + 145]
                
                # Check if window is contiguous (no big gaps)
                time_diffs = window_candidate['time_stamp'].diff()
                if time_diffs.max() > pd.Timedelta(minutes=30):  # Allow small gaps
                    continue
                
                # Aggregate features
                features = aggregate_window_features(window_candidate, sensor_cols)
                features['label'] = 0  # Normal
                features['event_id'] = None
                features['asset_id'] = asset_id
                features['failure_type'] = 'normal'
                
                all_normal_windows.append(features)
                sampled_windows += 1
            
            print(f"  Created {sampled_windows} normal windows")
        
        normal_df = pd.DataFrame(all_normal_windows)
        print(f"\nTotal normal windows: {len(normal_df)}")
        
        return normal_df
    
    def select_features(self, dataset, feature_descriptions=None, 
                   cohens_d_threshold=0.8, max_features=15, 
                   correlation_threshold=0.9):
        """
        Step 4: Feature selection based on discriminative power
        
        Parameters:
        -----------
        dataset : DataFrame
            Full dataset with features and labels
        feature_descriptions : DataFrame, optional
            Feature description metadata (for interpretability)
        cohens_d_threshold : float
            Minimum effect size for initial filtering
        max_features : int
            Maximum features to select
        correlation_threshold : float
            Correlation threshold for redundancy removal
        """
        print("\nSTEP 4: FEATURE SELECTION")
        print("="*60)
        
        # Get feature columns (exclude metadata)
        feature_cols = [c for c in dataset.columns 
                        if c.endswith(('_mean', '_std', '_trend'))]
        
        print(f"Starting features: {len(feature_cols)}")
        
        X = dataset[feature_cols]
        y = dataset['label']
        
        # STEP 4A: Univariate discriminative analysis
        from scipy import stats
        import numpy as np
        
        feature_analysis = []
        
        for feature in feature_cols:
            normal_vals = dataset[dataset['label'] == 0][feature]
            failure_vals = dataset[dataset['label'] == 1][feature]
            
            # Skip if all NaN
            if normal_vals.isna().all() or failure_vals.isna().all():
                continue
            
            # T-test
            t_stat, p_value = stats.ttest_ind(normal_vals.dropna(), failure_vals.dropna(), 
                                               equal_var=False, nan_policy='omit')
            
            # Cohen's d
            mean_diff = failure_vals.mean() - normal_vals.mean()
            pooled_std = np.sqrt((normal_vals.std()**2 + failure_vals.std()**2) / 2)
            cohens_d = abs(mean_diff / pooled_std) if pooled_std > 0 else 0
            
            feature_analysis.append({
                'feature': feature,
                'cohens_d': cohens_d,
                'p_value': p_value,
                'mean_diff': mean_diff,
                'normal_mean': normal_vals.mean(),
                'failure_mean': failure_vals.mean()
            })
        
        feature_analysis_df = pd.DataFrame(feature_analysis)
        feature_analysis_df = feature_analysis_df.sort_values('cohens_d', ascending=False)
        
        print(f"\nFeatures with effect size calculated: {len(feature_analysis_df)}")
        print(f"\nEffect size distribution:")
        print(f"  Cohen's d > 1.0 (large): {(feature_analysis_df['cohens_d'] > 1.0).sum()}")
        print(f"  Cohen's d > 0.8 (large): {(feature_analysis_df['cohens_d'] > 0.8).sum()}")
        print(f"  Cohen's d > 0.6 (medium): {(feature_analysis_df['cohens_d'] > 0.6).sum()}")
        
        print("\nTop 30 discriminative features:")
        print(feature_analysis_df.head(30)[['feature', 'cohens_d', 'mean_diff']].to_string(index=False))
        
        # STEP 4B: Iterative redundancy removal
        print("\nSTEP 4B: REDUNDANCY REMOVAL")
        print("="*60)
        
        selected_features = []
        candidate_features = feature_analysis_df[
            feature_analysis_df['cohens_d'] > cohens_d_threshold
        ]['feature'].tolist()
        
        print(f"Candidates (d > {cohens_d_threshold}): {len(candidate_features)}")
        
        for candidate in candidate_features:
            if len(selected_features) >= max_features:
                break
            
            if len(selected_features) == 0:
                # First feature - take highest Cohen's d
                selected_features.append(candidate)
                continue
            
            # Check correlation with already selected features
            corr_check = dataset[selected_features + [candidate]].corr()
            max_corr = corr_check[candidate][:-1].abs().max()
            
            if max_corr < correlation_threshold:
                selected_features.append(candidate)
        
        print(f"\nSelected {len(selected_features)} non-redundant features")
        print(f"(max_corr < {correlation_threshold}, Cohen's d > {cohens_d_threshold})")
        
        print("\n\nFINAL FEATURE SET:")
        print("="*60)
        for f in selected_features:
            row = feature_analysis_df[feature_analysis_df['feature']==f].iloc[0]
            print(f"  {f:<40} d={row['cohens_d']:.3f}")
        
        # Save to instance
        self.selected_features = selected_features
        self.feature_analysis = feature_analysis_df
        
        return selected_features
        
    def evaluate_model(self, full_dataset, selected_features, model_params=None):
        """
        Step 5: Model training and Leave-One-Out cross-validation
        
        Parameters:
        -----------
        dataset : DataFrame
            Complete dataset with features and labels
        selected_features : list
            List of feature names to use
        model_params : dict, optional
            RandomForest parameters (default: sensible defaults)
        """
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import LeaveOneOut
        from sklearn.metrics import classification_report, confusion_matrix
        import numpy as np
        
        print("\n\nSTEP 5: MODEL EVALUATION")
        print("="*80)
        
        if model_params is None:
            model_params = {
                'n_estimators': 100,
                'max_depth': 5,
                'random_state': 42,
                'class_weight': 'balanced'
            }
        
        X = dataset[selected_features]
        y = dataset['label']
        
        print(f"Features: {len(selected_features)}")
        print(f"Samples: {len(X)} ({y.sum()} failures, {(~y.astype(bool)).sum()} normal)")
        
        # Leave-One-Out Cross-Validation
        loo = LeaveOneOut()
        y_true = []
        y_pred = []
        y_pred_proba = []
        
        for train_idx, test_idx in loo.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            clf = RandomForestClassifier(**model_params)
            clf.fit(X_train, y_train)
            
            y_true.append(y_test.values[0])
            y_pred.append(clf.predict(X_test)[0])
            y_pred_proba.append(clf.predict_proba(X_test)[0, 1])
        
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_pred_proba = np.array(y_pred_proba)
        
        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_true, y_pred)
        print(cm)
        
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=['Normal', 'Failure']))
        
        # Failure detection details
        failure_indices = np.where(y_true == 1)[0]
        detected = y_pred[failure_indices].sum()
        total = len(failure_indices)
        
        print(f"\n{'='*80}")
        print(f"FAILURES DETECTED: {detected} / {total} ({detected/total*100:.1f}%)")
        print(f"{'='*80}")
        
        print("\nPer-failure probabilities:")
        print(f"{'Event ID':<12} {'Predicted':<12} {'Probability':<12}")
        print("-" * 40)
        
        for idx in failure_indices:
            event_id = dataset.iloc[idx]['event_id']
            prob = y_pred_proba[idx]
            pred = 'FAILURE ✓' if y_pred[idx] == 1 else 'normal ✗'
            print(f"{str(event_id):<12} {pred:<12} {prob:.3f}")
        
        # Summary statistics
        if detected > 0:
            detected_probs = y_pred_proba[failure_indices][y_pred[failure_indices] == 1]
            print(f"\nDetected failures - probability range: {detected_probs.min():.3f} to {detected_probs.max():.3f}")
        
        if detected < total:
            missed_probs = y_pred_proba[failure_indices][y_pred[failure_indices] == 0]
            print(f"Missed failures - probability range: {missed_probs.min():.3f} to {missed_probs.max():.3f}")
        
        # Save results
        self.model_results = {
            'y_true': y_true,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'confusion_matrix': cm,
            'recall': detected / total,
            'precision': cm[1,1] / (cm[1,1] + cm[0,1]) if (cm[1,1] + cm[0,1]) > 0 else 0,
            'accuracy': (cm[0,0] + cm[1,1]) / cm.sum()
        }
        
        return self.model_results

# Usage for Wind Farm B:
# pipeline = WindTurbineFailurePipeline()
# pipeline.validate_data_structure(scada_b, event_info_b)
# gaps = pipeline.analyze_temporal_patterns(scada_b, event_info_b)
# dataset = pipeline.build_feature_dataset(scada_b, event_info_b)