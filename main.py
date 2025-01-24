import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE

from src.data_preparation import load_and_clean_data, preprocess_and_split_data, prepare_unseen_data
from src.feature_selection import remove_high_correlation_features, remove_low_importance_features, \
    remove_low_shap_features
from src.model_training import train_xgboost_model, train_catboost_model
from src.evaluation import evaluate_model, predict_unseen_data

# Ορισμός στρατηγικών αντικατάστασης για ελλιπείς τιμές
columns_with_replacements = {
    'X1': 'median', 'X2': 'median', 'X3': 'median', 'X4': 'median',
    'X5': 0, 'X6': 'median', 'X7': 'median', 'X8': 'median',
    'X9': 'median', 'X10': 'median', 'X11': 'median', 'X12': 'median',
    'X13': 'median', 'X14': 'median', 'X15': 'median', 'X16': 'median',
    'X17': 'median', 'X18': 'median', 'X19': 'median', 'X20': 0,
    'X21': 'mean', 'X22': 'median', 'X23': 'median', 'X24': 'median',
    'X25': 'median', 'X26': 'median', 'X27': 'median', 'X28': 'median',
    'X29': 'mean', 'X30': 'median', 'X31': 'median', 'X32': 'median',
    'X33': 'median', 'X34': 'median', 'X35': 'median', 'X36': 'median',
    'X37': 'median', 'X38': 'median', 'X39': 'median', 'X40': 'median',
    'X41': 'median', 'X42': 'median', 'X43': 0, 'X44': 'median',
    'X45': 'median', 'X46': 'median', 'X47': 0, 'X48': 'median',
    'X49': 'median', 'X50': 'median', 'X51': 'median', 'X52': 'median',
    'X53': 'median', 'X54': 'median', 'X55': 'median', 'X56': 'median',
    'X57': 'median', 'X58': 'median', 'X59': 'median', 'X60': 'median',
    'X61': 'median', 'X66': 'median', 'X63': 'median', 'X64': 'median'
}

# Καθαρισμός δεδομένων
train_data, imputation_values = load_and_clean_data("data/training_companydata.csv", columns_with_replacements)
train_data = train_data.sample(frac=1).reset_index(drop=True) # Ανακάτεμα δεδομένων
train_data, removed_columns = remove_high_correlation_features(train_data)

# Διαγραφή low-importance features
X = train_data.drop(columns=['X65'])  # Features only
y = train_data['X65']                # Target

# Apply feature importance filtering
X_clean, low_importance_columns = remove_low_importance_features(X, y, threshold_importance=0.01)
#X_clean, _ = remove_low_shap_features(X, y, threshold=0.02)

# Rebuild the cleaned dataset
train_data = pd.concat([X_clean, y], axis=1)

# Αποθήκευση καθαρισμένων δεδομένων
train_data.to_csv("data/cleaned_training_data.csv", index=False)
print("Τα καθαρισμένα δεδομένα αποθηκεύτηκαν.")

# Προεπεξεργασία και εκπαίδευση μοντέλου
X_train_scaled, X_test_scaled, y_train, y_test, scaler = preprocess_and_split_data(train_data)
smote = SMOTE(sampling_strategy=0.5, random_state=42)  # Balance class 1 to 50% of class 0
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
# scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1]) * 0.5
# print(f"Class weight ratio: {scale_pos_weight:.1f}")

#xgb_model = train_xgboost_model(X_train_scaled, y_train, scale_pos_weight)
cat_model = train_catboost_model(X_train_balanced, y_train_balanced)

# Φόρτωση & προετοιμασία άγνωστων δεδομένων ΜΕ ΤΗΝ ΣΥΝΑΡΤΗΣΗ
required_columns = [f'X{i}' for i in range(1, 65)]  # Correct X1-X64
X = train_data.drop(columns=['X65'])  # Get final training features

unseen_data_scaled = prepare_unseen_data(
    filepath="data/test_unlabeled.csv",
    required_columns=required_columns,
    scaler=scaler,
    X_train_columns=X.columns.tolist(),  # Final features after selection
    imputation_values=imputation_values  # From training phase
)

# Replace the evaluation call:
optimal_threshold = evaluate_model(cat_model, X_test_scaled, y_test)

# Use the optimal threshold for predictions:
predict_unseen_data(cat_model, unseen_data_scaled, scaler, "outputs/test_predictions.csv", threshold=optimal_threshold)

# Δημιουργία αρχείου προγνώσεων
test_predictions = pd.DataFrame({
    'prediction': cat_model.predict(unseen_data_scaled)  # Πρόβλεψη (1 ή 0)
})
test_predictions.to_csv("outputs/test_predictions_final.csv", index=False, header=False)
print("Το αρχείο 'test_predictions_final.csv' δημιουργήθηκε.")
# Εκτυπωνει ποσα 1 εχει
print(test_predictions['prediction'].value_counts())

# Υπολογισμός πιθανοτήτων πτώχευσης
probabilities = cat_model.predict_proba(unseen_data_scaled)[:, 1]

# Δημιουργία DataFrame με rowids και πιθανότητες
rowid_probabilities = pd.DataFrame({
    'rowid': range(1, len(probabilities) + 1),  # Δημιουργία rowid από 1 έως n
    'probability': probabilities
})

# Εύρεση των 50 εταιριών με τη μεγαλύτερη πιθανότητα πτώχευσης
top_50 = rowid_probabilities.sort_values(by='probability', ascending=False).head(50)

# Δημιουργία αρχείου με τα rowids
top_50[['rowid']].to_csv("outputs/top_50_predictions.csv", index=False, header=False)
print("Το αρχείο 'top_50_predictions.csv' δημιουργήθηκε.")
