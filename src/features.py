def aggregate_window_features(window_data, sensor_cols):
    """
    Aggregate a 24-hour window into summary statistics per sensor.
    
    Parameters:
    -----------
    window_data : DataFrame
        24 hours of SCADA data (should be ~144 records at 10-min intervals)
    sensor_cols : list
        List of sensor column names to aggregate
    
    Returns:
    --------
    dict : Feature dictionary for this window
    """
    features = {}
    
    for sensor in sensor_cols:
        # Skip if sensor is all NaN in this window
        if window_data[sensor].isna().all():
            continue
            
        # Mean: average sensor value over 24h
        features[f'{sensor}_mean'] = window_data[sensor].mean()
        
        # Std: variability over 24h
        features[f'{sensor}_std'] = window_data[sensor].std()
        
        # Trend: linear slope over the window
        values = window_data[sensor].dropna().values
        if len(values) > 10:  # Need minimum points for meaningful trend
            time_idx = np.arange(len(values))
            slope, _ = np.polyfit(time_idx, values, 1)
            features[f'{sensor}_trend'] = slope
        else:
            features[f'{sensor}_trend'] = 0
    
    return features