import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import pandas as pd

from sklearn.metrics import roc_curve


def evaluate_model(model, X_test, y_test):
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Find optimal threshold
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]

    y_pred = (y_pred_proba > optimal_threshold).astype(int)

    print("Optimal Threshold:", optimal_threshold)
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    return optimal_threshold


def predict_unseen_data(model, unseen_data, scaler, output_file, threshold=0.5):
    """Κάνει πρόβλεψη για τα άγνωστα δεδομένα και αποθηκεύει τα αποτελέσματα."""
    unseen_data_scaled = scaler.transform(unseen_data)
    predictions_proba = model.predict_proba(unseen_data_scaled)[:, 1]
    predictions = (predictions_proba > threshold).astype(int)

    results = pd.DataFrame({
        'rowid': range(1, len(predictions) + 1),
        'prediction': predictions
    })
    results.to_csv(output_file, index=False)
    print(f"Οι προβλέψεις αποθηκεύτηκαν στο {output_file}.")