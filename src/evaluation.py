import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import pandas as pd


def evaluate_model(model, X_test, y_test):
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Find threshold that maximizes F1-score
    f1_scores = []
    thresholds = np.linspace(0.1, 0.9, 50)

    for thresh in thresholds:
        y_pred = (y_pred_proba > thresh).astype(int)
        f1 = f1_score(y_test, y_pred)
        f1_scores.append(f1)

    best_threshold = thresholds[np.argmax(f1_scores)]
    y_pred = (y_pred_proba > best_threshold).astype(int)

    print(f"Optimal F1 Threshold: {best_threshold:.4f}")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    return best_threshold

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