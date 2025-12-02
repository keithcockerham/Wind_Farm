def get_event_info(farm='A'):
    farm_dir = os.path.join(BASE_DIR, 'Wind Farm '+farm)
    event_info = pd.read_csv(farm_dir + '\\event_info.csv', sep=';')
    event_info = event_info.rename(columns={'asset':'asset_id'})
    # Drop events that aren't anomalies
    event_info = event_info[event_info['event_label'] == 'anomaly']
    # Clean Up
    event_info["event_start"] = pd.to_datetime(event_info["event_start"])
    event_info["event_end"] = pd.to_datetime(event_info["event_end"])
    event_info["asset_id"] = pd.to_numeric(event_info["asset_id"], errors="coerce").astype("Int16")
    # Drop rows with missing critical info
    event_info = event_info.dropna(subset=["asset_id", "event_start", 'event_end'])
    # Sort for asset_id
    event_info = event_info.sort_values(["asset_id"]).reset_index(drop=True)

    return event_info
    
def get_feature_info(farm='A'):
    farm_dir = os.path.join(BASE_DIR, 'Wind Farm '+farm)
    feature_info = pd.read_csv(farm_dir + '\\feature_description.csv', sep=';')
    
    return feature_info
    
def get_farm_scada_chunked(farm='A', asset_ids=None, chunksize=50000):
    """
    Process farm data in chunks - for when dataset won't fit in RAM.
    
    asset_ids: Process only specific assets (e.g., [0, 10, 21])
    chunksize: Rows per chunk
    """
    farm_dir = os.path.join(BASE_DIR, 'Wind Farm '+farm)
    farm_dataset_dir = os.path.join(farm_dir, 'datasets')
    all_files = glob.glob(os.path.join(farm_dataset_dir, '*.csv'))
    parquet_path = os.path.join(farm_dir, f'farm_{farm}_optimized.parquet')
    dtype_dict = {
        'asset_id': 'int16',
        'status_type_id': 'int8',
    }
    # Check if already processed
    if os.path.exists(parquet_path):
        print(f"Loading pre-processed Farm {farm}...")
        return pd.read_parquet(parquet_path)
        
    all_data = []
    
    for f in all_files:
        # Read in chunks
        for chunk in pd.read_csv(f, sep=";", dtype=dtype_dict, chunksize=chunksize):
            # Filter to specific assets if provided
            if asset_ids is not None:
                chunk = chunk[chunk['asset_id'].isin(asset_ids)]
            
            # Optimize dtypes
            float_cols = chunk.select_dtypes(include=['float64']).columns
            chunk[float_cols] = chunk[float_cols].astype('float32')
            
            # Clean
            chunk["time_stamp"] = pd.to_datetime(chunk["time_stamp"])
            chunk = chunk.drop(['train_test', 'id'], axis=1, errors='ignore')
            chunk = chunk.dropna(subset=["asset_id", "time_stamp"])
            
            all_data.append(chunk)
    
    scada_data = pd.concat(all_data, ignore_index=True)
    scada_data = scada_data.drop_duplicates(subset=['asset_id', 'time_stamp'], keep='first')
    scada_data = scada_data.sort_values(["asset_id", "time_stamp"]).reset_index(drop=True)

    scada_data.to_parquet(parquet_path, compression='snappy', index=False)
    print(f"Saved to {parquet_path}")
    
    return scada_data