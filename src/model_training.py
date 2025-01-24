import random
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from hyperopt import fmin, tpe, hp, Trials
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from hyperopt import fmin, tpe, hp, STATUS_OK
from catboost import CatBoostClassifier

def train_xgboost_model(X_train, y_train, scale_pos_weight):
    """XGBoost training with manual hyperparameter tuning."""
    # Simplified search space
    # In model_training.py
    space = {
        'max_depth': hp.choice('max_depth', [3, 4, 5, 6]),
        'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.3)),
        'subsample': hp.uniform('subsample', 0.6, 0.95),
        'colsample_bytree': hp.uniform('colsample_bytree', 0.6, 0.95),
        'reg_alpha': hp.uniform('reg_alpha', 0, 2),
        'reg_lambda': hp.uniform('reg_lambda', 0, 2),
        'scale_pos_weight': hp.uniform('scale_pos_weight', 5, 20)
    }

    def objective(params):
        model = XGBClassifier(
            eval_metric='logloss',
            max_depth=int(params['max_depth']),
            learning_rate=params['learning_rate'],
            subsample=params['subsample'],
            colsample_bytree=params['colsample_bytree'],
            scale_pos_weight=params['scale_pos_weight'],
            n_estimators=100
        )

        # Manual 3-fold cross-validation
        scores = []
        indices = np.arange(len(y_train))
        np.random.shuffle(indices)
        fold_size = len(indices) // 3

        for i in range(3):
            val_indices = indices[i * fold_size: (i + 1) * fold_size]
            train_indices = np.concatenate([indices[:i * fold_size], indices[(i + 1) * fold_size:]])

            model.fit(X_train[train_indices], y_train.iloc[train_indices])
            y_pred = model.predict(X_train[val_indices])
            score = np.mean(y_pred == y_train.iloc[val_indices])  # Simple accuracy
            scores.append(score)

        return {'loss': -np.mean(scores), 'status': STATUS_OK}

    # Increase max_evals to 50
    best_params = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=50)

    # Train final model with best params
    final_model = XGBClassifier(
        eval_metric='logloss',
        max_depth=int(best_params['max_depth']),
        learning_rate=best_params['learning_rate'],
        subsample=best_params['subsample'],
        colsample_bytree=best_params['colsample_bytree'],
        scale_pos_weight=best_params['scale_pos_weight'],
        n_estimators=200
    )
    final_model.fit(X_train, y_train)

    return final_model

def train_catboost_model(X_train, y_train):
    model = CatBoostClassifier(
        auto_class_weights='Balanced',
        verbose=0,
        iterations=500,
        depth=5
    )
    model.fit(X_train, y_train)
    return model