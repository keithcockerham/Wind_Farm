def evaluate_unified_model(dataset, selected_features):

    print("\n" + "="*80)
    print("UNIFIED MODEL EVALUATION")
    print("="*80)
    
    X = dataset[selected_features]
    y = dataset['label']
    
    print(f"\nFeatures: {len(selected_features)}")
    print(f"Total samples: {len(X)}")
    print(f"Failures: {y.sum()}")
    print(f"Normal: {(y==0).sum()}")
    
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy='median')
    X_imputed = pd.DataFrame(
        imputer.fit_transform(X),
        columns=X.columns,
        index=X.index
    )
    
    print(f"\nMissing values before imputation: {X.isna().sum().sum()}")
    print(f"Missing values after imputation: {X_imputed.isna().sum().sum()}")
    
    loo = LeaveOneOut()
    y_true = []
    y_pred = []
    y_pred_proba = []
    
    print("\nRunning Leave-One-Out Cross-Validation...")
    
    for train_idx, test_idx in loo.split(X_imputed):
        X_train, X_test = X_imputed.iloc[train_idx], X_imputed.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        clf = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42,
            class_weight='balanced'
        )
        clf.fit(X_train, y_train)
        
        y_true.append(y_test.values[0])
        y_pred.append(clf.predict(X_test)[0])
        y_pred_proba.append(clf.predict_proba(X_test)[0, 1])
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_pred_proba = np.array(y_pred_proba)
    
    print("\n" + "="*60)
    print("OVERALL PERFORMANCE")
    print("="*60)
    
    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:")
    print(cm)
    
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['Normal', 'Failure']))
    
    failure_indices = np.where(y_true == 1)[0]
    detected = y_pred[failure_indices].sum()
    total = len(failure_indices)
    
    print(f"\n{'='*60}")
    print(f"FAILURES DETECTED: {detected} / {total} ({detected/total*100:.1f}%)")
    print(f"{'='*60}")
    
    print("\n" + "="*60)
    print("PERFORMANCE BY FARM")
    print("="*60)
    
    for farm in ['A', 'B', 'C']:
        farm_mask = dataset['farm'] == farm
        farm_indices = np.where(farm_mask)[0]
        
        y_true_farm = y_true[farm_indices]
        y_pred_farm = y_pred[farm_indices]
        
        failure_mask = y_true_farm == 1
        if failure_mask.sum() == 0:
            continue
        
        farm_detected = y_pred_farm[failure_mask].sum()
        farm_total = failure_mask.sum()
        farm_recall = farm_detected / farm_total if farm_total > 0 else 0
        
        normal_mask = y_true_farm == 0
        farm_fp = (y_pred_farm[normal_mask] == 1).sum()
        farm_tn = (y_pred_farm[normal_mask] == 0).sum()
        farm_precision = farm_detected / (farm_detected + farm_fp) if (farm_detected + farm_fp) > 0 else 0
        
        print(f"\nFarm {farm}:")
        print(f"  Total failures: {farm_total}")
        print(f"  Detected: {farm_detected} ({farm_recall:.0%})")
        print(f"  False alarms: {farm_fp}/{farm_tn + farm_fp}")
        print(f"  Precision: {farm_precision:.0%}")
    
    return {
        'y_true': y_true,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'confusion_matrix': cm
    }