from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import pandas as pd

def evaluate_model(model, X_test, y_test, threshold=0.5):
    """Αξιολογεί το μοντέλο σε δεδομένα ελέγχου."""
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba > threshold).astype(int)

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    roc_auc = roc_auc_score(y_test, y_pred_proba)
    print(f"\nROC-AUC Score: {roc_auc:.4f}")

    return y_pred_proba


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