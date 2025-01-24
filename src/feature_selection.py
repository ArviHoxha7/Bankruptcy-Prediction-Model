import shap
from xgboost import XGBClassifier
import numpy as np

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


def remove_low_shap_features(X, y, threshold=0.01):
    model = XGBClassifier(eval_metric='logloss')
    model.fit(X, y)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    low_shap = X.columns[mean_abs_shap < threshold]
    return X.drop(columns=low_shap), low_shap

