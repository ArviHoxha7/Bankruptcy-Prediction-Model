def remove_high_correlation_features(data, threshold=0.99):
    correlation_matrix = data.corr()
    columns_to_remove = set()
    for i in range(len(correlation_matrix.columns)):
        for j in range(i):
            if abs(correlation_matrix.iloc[i, j]) > threshold:
                columns_to_remove.add(correlation_matrix.columns[j])

    data = data.drop(columns=columns_to_remove)
    return data, columns_to_remove

def remove_low_importance_features(X, y, threshold_importance=0.01):
    from xgboost import XGBClassifier

    xgb_model = XGBClassifier(eval_metric='logloss')
    xgb_model.fit(X, y)

    importances = xgb_model.feature_importances_
    low_importance = X.columns[importances < threshold_importance]
    X_clean = X.drop(columns=low_importance)

    return X_clean, low_importance
