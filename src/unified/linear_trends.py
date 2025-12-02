def compute_linear_trend(series):
    
    series_clean = series.dropna()
    if len(series_clean) < 2:
        return 0.0
    
    x = np.arange(len(series_clean))
    try:
        slope, _ = np.polyfit(x, series_clean, 1)
        return slope
    except:
        return 0.0
    print('linear trend done')