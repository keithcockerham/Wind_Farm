def analyze_unified_features(dataset):
    
    feature_cols = [c for c in dataset.columns 
                   if c not in ['label', 'event_id', 'asset_id', 'farm', 'failure_type']]
    
    print(f"\nTotal unified features: {len(feature_cols)}")
    
    if len(feature_cols) == 0:
        print("ERROR: No feature columns found!")
        print("Available columns:", dataset.columns.tolist())
        return None
    
    stat_types = {}
    for stat in ['mean', 'std', 'trend', 'max', 'min']:
        count = len([c for c in feature_cols if c.endswith(f'_{stat}')])
        stat_types[stat] = count
    
    print("\nFeatures by statistic type:")
    for stat, count in stat_types.items():
        print(f"  {stat}: {count}")
    
    # Check coverage across farms
    print("\n" + "="*60)
    print("Feature Coverage by Farm:")
    print("="*60)
    
    for farm in ['A', 'B', 'C']:
        farm_data = dataset[dataset['farm'] == farm]
        if len(farm_data) == 0:
            print(f"\nFarm {farm}: NO DATA")
            continue
            
        non_null = farm_data[feature_cols].notna().sum()
        coverage = (non_null / len(farm_data) * 100).describe()
        
        print(f"\nFarm {farm} ({len(farm_data)} windows):")
        print(f"  Mean coverage: {coverage['mean']:.1f}%")
        print(f"  Min coverage: {coverage['min']:.1f}%")
        print(f"  Features with 100% coverage: {(non_null == len(farm_data)).sum()}")
    
    # Missing data analysis
    print("\n" + "="*60)
    print("Missing Data Analysis:")
    print("="*60)
    
    missing_pct = (dataset[feature_cols].isna().sum() / len(dataset) * 100).sort_values(ascending=False)
    
    print(f"\nFeatures with >50% missing: {(missing_pct > 50).sum()}")
    print(f"Features with >25% missing: {(missing_pct > 25).sum()}")
    print(f"Features with >10% missing: {(missing_pct > 10).sum()}")
    print(f"Features with 0% missing: {(missing_pct == 0).sum()}")
    
    if (missing_pct > 50).sum() > 0:
        print("\nTop 20 features with most missing data:")
        print(missing_pct.head(20))
    
    print("\n" + "="*60)
    print("Discriminative Power Analysis:")
    print("="*60)
    
    discriminative_features = []
    skipped_missing = 0
    skipped_error = 0
    
    for feature in feature_cols:
        normal_vals = dataset[dataset['label'] == 0][feature]
        failure_vals = dataset[dataset['label'] == 1][feature]
        
        # RELAXED: Skip only if >75% missing (was 50%)
        normal_missing_pct = normal_vals.isna().sum() / len(normal_vals)
        failure_missing_pct = failure_vals.isna().sum() / len(failure_vals)
        
        if normal_missing_pct > 0.75 or failure_missing_pct > 0.75:
            skipped_missing += 1
            continue
        
        normal_clean = normal_vals.dropna()
        failure_clean = failure_vals.dropna()
        
        if len(normal_clean) < 3 or len(failure_clean) < 3:
            skipped_error += 1
            continue
        
        try:
            mean_diff = failure_clean.mean() - normal_clean.mean()
            pooled_std = np.sqrt((normal_clean.std()**2 + failure_clean.std()**2) / 2)
            cohens_d = abs(mean_diff / pooled_std) if pooled_std > 0 else 0
            
            discriminative_features.append({
                'feature': feature,
                'cohens_d': cohens_d,
                'mean_diff': mean_diff,
                'normal_missing_pct': normal_missing_pct,
                'failure_missing_pct': failure_missing_pct
            })
        except Exception as e:
            skipped_error += 1
            continue
    
    print(f"\nFeatures analyzed: {len(discriminative_features)}")
    print(f"Skipped (>75% missing): {skipped_missing}")
    print(f"Skipped (errors/insufficient data): {skipped_error}")
    
    if len(discriminative_features) == 0:
        print("\nERROR: No features could be analyzed!")
        print("This suggests all features have too much missing data.")
        print("\nDEBUG: Check first few feature columns:")
        for col in feature_cols[:5]:
            print(f"  {col}: {dataset[col].isna().sum()}/{len(dataset)} missing")
        return None
    
    df_disc = pd.DataFrame(discriminative_features).sort_values('cohens_d', ascending=False)
    
    print(f"Features with d > 1.0: {(df_disc['cohens_d'] > 1.0).sum()}")
    print(f"Features with d > 0.8: {(df_disc['cohens_d'] > 0.8).sum()}")
    print(f"Features with d > 0.6: {(df_disc['cohens_d'] > 0.6).sum()}")
    print(f"Features with d > 0.5: {(df_disc['cohens_d'] > 0.5).sum()}")
    
    print("\nTop 20 discriminative unified features:")
    print(df_disc.head(20)[['feature', 'cohens_d', 'mean_diff']].to_string(index=False))
    
    return df_disc
