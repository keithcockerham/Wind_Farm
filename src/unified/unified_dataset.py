def build_unified_dataset(scada_dict, event_info_dict, sensor_mapping, 
                         power_sensors, power_threshold=0.1, 
                         window_hours=24, buffer_days=30, random_seed=42):
    
    all_windows = []
    
    for farm in ['A', 'B', 'C']:
        print(f"\nProcessing Farm {farm}...")
        
        scada = scada_dict[farm]
        events = event_info_dict[farm]
        power_col = power_sensors[farm]
        
        failure_windows = extract_failure_windows_unified_fixed(
            scada, events, farm, sensor_mapping, 
            power_col, power_threshold, window_hours
        )
        
        print(f"  Failure windows: {len(failure_windows)}")
        
        normal_windows = extract_normal_windows_unified(
            scada, events, farm, sensor_mapping,
            power_col, power_threshold, window_hours, buffer_days,random_seed=random_seed
        )
        
        print(f"  Normal windows: {len(normal_windows)}")
        
        # Combine
        all_windows.extend(failure_windows)
        all_windows.extend(normal_windows)
    
    dataset = pd.DataFrame(all_windows)
    
    print(f"\n{'='*60}")
    print(f"UNIFIED DATASET CREATED")
    print(f"{'='*60}")
    print(f"Total windows: {len(dataset)}")
    print(f"Failure windows: {(dataset['label']==1).sum()}")
    print(f"Normal windows: {(dataset['label']==0).sum()}")
    print(f"Features: {len([c for c in dataset.columns if c not in ['label', 'event_id', 'asset_id', 'farm', 'failure_type']])}")
    print(f"\nWindows per farm:")
    print(dataset['farm'].value_counts())
    
    return dataset
    print('Unified DB built')