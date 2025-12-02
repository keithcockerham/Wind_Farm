def extract_failure_windows_unified_fixed(scada, events, farm, sensor_mapping,
                                         power_col, power_threshold, window_hours):    
    windows = []
    
    for _, failure in events[events['event_label'] == 'anomaly'].iterrows():
        
        asset_id = failure['asset_id']
        event_id = failure['event_id']
        failure_start = failure['event_start']
        
        # Find last production time
        production_data = scada[
            (scada['asset_id'] == asset_id) &
            (scada['time_stamp'] < failure_start) &
            (scada['status_type_id'] == 0) &
            (scada[power_col] > power_threshold)
        ].sort_values('time_stamp')
        
        if len(production_data) == 0:
            continue
        
        last_production = production_data.iloc[-1]['time_stamp']
        window_start = last_production - pd.Timedelta(hours=window_hours)
        
        window_data = scada[
            (scada['asset_id'] == asset_id) &
            (scada['time_stamp'] >= window_start) &
            (scada['time_stamp'] <= last_production)
        ]
        
        if len(window_data) < 100:
            continue
        
        features = extract_unified_window_features_fixed(
            window_data, farm, sensor_mapping,
            statistics=['mean', 'std', 'trend']
        )
        
        features['label'] = 1
        features['event_id'] = event_id
        features['asset_id'] = asset_id
        features['farm'] = farm
        features['failure_type'] = failure.get('event_description', 'Unknown')
        
        windows.append(features)
    
    return windows