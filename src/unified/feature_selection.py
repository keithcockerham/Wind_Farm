def select_unified_features(dataset, cohens_d_threshold=0.6, max_features=20, 
                           correlation_threshold=0.9):
    
    feature_cols = [c for c in dataset.columns 
                   if c not in ['label', 'event_id', 'asset_id', 'farm', 'failure_type']]
    
    print(f"\nTotal feature columns: {len(feature_cols)}")
    
    feature_analysis = []
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
            
            feature_analysis.append({
                'feature': feature,
                'cohens_d': cohens_d
            })
        except Exception as e:
            skipped_error += 1
            continue
    
    print(f"\nFeatures analyzed: {len(feature_analysis)}")
    print(f"Skipped (>75% missing): {skipped_missing}")
    print(f"Skipped (errors/insufficient data): {skipped_error}")
    
    if len(feature_analysis) == 0:
        print("\nERROR: No features could be analyzed!")
        return []
    
    df_analysis = pd.DataFrame(feature_analysis).sort_values('cohens_d', ascending=False)
    
    selected_features = []
    candidates = df_analysis[df_analysis['cohens_d'] > cohens_d_threshold]['feature'].tolist()
    
    print(f"\nCandidates (d > {cohens_d_threshold}): {len(candidates)}")
    
    if len(candidates) == 0:
        print(f"\nWARNING: No features exceed Cohen's d threshold of {cohens_d_threshold}")
        print("Lowering threshold to include top 10 features...")
        candidates = df_analysis.head(10)['feature'].tolist()
    
    for candidate in candidates:
        if len(selected_features) >= max_features:
            break
        
        if len(selected_features) == 0:
            selected_features.append(candidate)
            continue
        
        # Check correlation with imputation for missing values
        try:
            subset = dataset[selected_features + [candidate]].copy()
            
            # Impute missing values
            from sklearn.impute import SimpleImputer
            imputer = SimpleImputer(strategy='median')
            subset_imputed = pd.DataFrame(
                imputer.fit_transform(subset),
                columns=subset.columns
            )
            
            corr_check = subset_imputed.corr()
            max_corr = corr_check[candidate][:-1].abs().max()
            
            if max_corr < correlation_threshold:
                selected_features.append(candidate)
        except Exception as e:
            continue
    
    print(f"\nSelected {len(selected_features)} features")
    print(f"(max_corr < {correlation_threshold}, d > {cohens_d_threshold})")
    
    if len(selected_features) > 0:
        print("\n\nFINAL UNIFIED FEATURE SET:")
        print("="*60)
        for f in selected_features:
            d = df_analysis[df_analysis['feature']==f]['cohens_d'].iloc[0]
            print(f"  {f:<50} d={d:.3f}")
    else:
        print("\nERROR: No features selected!")
    
    return selected_features