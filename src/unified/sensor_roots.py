def extract_root_sensor_name(col_name):
    
    suffixes = ['_min', '_max', '_avg', '_std', '_std_dev']
    
    for suffix in suffixes:
        if col_name.endswith(suffix):
            return col_name[:-len(suffix)]
    
    # If no suffix found, return as-is
    return col_name


def extract_unified_window_features_fixed(window_data, farm, sensor_mapping, 
                                          statistics=['mean', 'std', 'trend']):

    farm_sensors = sensor_mapping[sensor_mapping['farm'] == farm].copy()
    
    unified_features = {}
    
    for (primary, secondary), group in farm_sensors.groupby(['primary group', 'secondary group']):
        
        category_key = f"{primary}_{secondary}"
        
        mapping_sensor_names = group['sensor_name'].tolist()
        
        available_sensors = []
        for col in window_data.columns:
            root_name = extract_root_sensor_name(col)
            if root_name in mapping_sensor_names:
                available_sensors.append(col)
        
        if len(available_sensors) == 0:
            continue
        
        category_data = window_data[available_sensors]
        
        if len(available_sensors) > 1:
            category_series = category_data.mean(axis=1, skipna=True)
        else:
            category_series = category_data[available_sensors[0]]
        
        if category_series.isna().all():
            continue
        
        if 'mean' in statistics:
            unified_features[f"{category_key}_mean"] = category_series.mean()
        
        if 'std' in statistics:
            unified_features[f"{category_key}_std"] = category_series.std()
        
        if 'trend' in statistics:
            unified_features[f"{category_key}_trend"] = compute_linear_trend(category_series)
    
    return unified_features
