import random
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

def train_xgboost_model(X_train, y_train, scale_pos_weight):
    """Εκπαιδεύει το XGBoost μοντέλο."""
    xgb_model = XGBClassifier(
        eval_metric='logloss',
        scale_pos_weight=scale_pos_weight,
        random_state=42
    )
    xgb_model.fit(X_train, y_train)
    return xgb_model
