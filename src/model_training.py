from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
import numpy as np
from xgboost import XGBClassifier
from hyperopt import fmin, tpe, hp, STATUS_OK
from catboost import CatBoostClassifier

def train_catboost_model(X_train, y_train):
    model = CatBoostClassifier(
        auto_class_weights='Balanced',
        verbose=False,
        iterations=500,
        depth=5
    )
    model.fit(X_train, y_train)
    return model


def train_top_features_model(X_train, y_train, X_test, y_test, top_features):
    """Εκπαίδευση μοντέλου με υποσύνολο γνωρισμάτων."""
    # Ensure X_train and X_test are DataFrames
    X_train_sub = X_train[top_features]
    X_test_sub = X_test[top_features]

    model = CatBoostClassifier(
        auto_class_weights='Balanced',
        verbose=False,
        iterations=500,
        depth=5
    )
    model.fit(X_train_sub, y_train)

    # Calculate metrics
    y_pred = model.predict(X_test_sub)
    y_proba = model.predict_proba(X_test_sub)[:, 1]

    metrics = {
        'f1': f1_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba)
    }

    return model, metrics