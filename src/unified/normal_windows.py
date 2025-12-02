def extract_normal_windows_unified(scada, events, farm, sensor_mapping,
                                   power_col, power_threshold, window_hours,
                                   buffer_days, windows_per_asset=20, random_seed=42):
    np.random.seed(random_seed)
    windows = []
    
    assets = scada['asset_id'].unique()
    
    for asset_id in assets:
        
        asset_failures = events[
            (events['asset_id'] == asset_id) &
            (events['event_label'] == 'anomaly')
        ]   
        exclusion_zones = []
        for _, failure in asset_failures.iterrows():
            start_exclude = failure['event_start'] - pd.Timedelta(days=buffer_days)
            end_exclude = failure['event_end'] + pd.Timedelta(days=buffer_days)
            exclusion_zones.append((start_exclude, end_exclude))
        
        asset_data = scada[
            (scada['asset_id'] == asset_id) &
            (scada['status_type_id'] == 0) &
            (scada[power_col] > power_threshold)
        ].copy()
        
        mask = pd.Series(True, index=asset_data.index)
        for start_ex, end_ex in exclusion_zones:
            mask &= ~((asset_data['time_stamp'] >= start_ex) & 
                      (asset_data['time_stamp'] <= end_ex))
        
        normal_data = asset_data[mask].sort_values('time_stamp')
        
        if len(normal_data) < 144 * windows_per_asset:
            continue
        
        sampled = 0
        max_attempts = windows_per_asset * 3
        attempts = 0
        
        while sampled < windows_per_asset and attempts < max_attempts:
            attempts += 1
            
            max_start_idx = len(normal_data) - 145
            if max_start_idx < 0:
                break
            
            start_idx = np.random.randint(0, max_start_idx)
            window_candidate = normal_data.iloc[start_idx:start_idx + 145]
            
            time_diffs = window_candidate['time_stamp'].diff()
            if time_diffs.max() > pd.Timedelta(minutes=30):
                continue

            features = extract_unified_window_features_fixed(
                window_data=window_candidate,  
                farm=farm,
                sensor_mapping=sensor_mapping,
                statistics=['mean', 'std', 'trend']
            )
            
            features['label'] = 0
            features['event_id'] = None
            features['asset_id'] = asset_id
            features['farm'] = farm
            features['failure_type'] = 'normal'
            
            windows.append(features)
            sampled += 1
    
    return windows