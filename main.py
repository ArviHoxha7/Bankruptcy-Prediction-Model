import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from src.data_preparation import load_and_clean_data, preprocess_and_split_data, prepare_unseen_data
from src.feature_selection import remove_high_correlation_features, remove_low_importance_features
from src.model_training import train_catboost_model, train_top_features_model
from src.evaluation import evaluate_model, predict_unseen_data

# ======================================================================
# Προεπεξεργασία & Εκπαίδευση Μοντέλου
# ======================================================================

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
train_data = train_data.sample(frac=1).reset_index(drop=True)  # Ανακάτεμα δεδομένων
train_data, removed_columns = remove_high_correlation_features(train_data)
print(len(removed_columns), "highly correlated features removed.")

# Διαγραφή low-importance features
X = train_data.drop(columns=['X65'])  # Features only
y = train_data['X65']                # Target

# Apply feature importance filtering
X_clean, low_importance_columns = remove_low_importance_features(X, y, threshold_importance=0.01)
print(len(low_importance_columns), "low-importance features removed.")
# Rebuild the cleaned dataset
train_data = pd.concat([X_clean, y], axis=1)
print("Remaining Columns: ", len(X_clean.columns))
# Αποθήκευση καθαρισμένων δεδομένων
train_data.to_csv("data/cleaned_training_data.csv", index=False)
print("Τα καθαρισμένα δεδομένα αποθηκεύτηκαν.")

# Προεπεξεργασία και εκπαίδευση μοντέλου
X_train_scaled, X_test_scaled, y_train, y_test, scaler = preprocess_and_split_data(train_data)
smote = SMOTE(sampling_strategy=0.5, random_state=42)  # Balance class 1 to 50% of class 0
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)

cat_model = train_catboost_model(X_train_balanced, y_train_balanced)

# ======================================================================
# Προσθήκες για Αξιολόγηση Γνωρισμάτων
# ======================================================================

# 1. Ανάλυση SHAP για Κατάταξη Γνωρισμάτων
explainer = shap.TreeExplainer(cat_model)
shap_values = explainer.shap_values(X_train_balanced)

# Υπολογισμός μέσης απόλυτης συμβολής (SHAP importance)
shap_importance = np.abs(shap_values).mean(axis=0)
feature_names = X_clean.columns.tolist()
shap_df = pd.DataFrame({
    'Feature': feature_names,
    'SHAP Importance': shap_importance
}).sort_values(by='SHAP Importance', ascending=False)

print("\nTop 10 Γνωρίσματα βάσει SHAP:")
print(shap_df.head(10))

# Αποθήκευση σε αρχείο
shap_df.to_csv("outputs/feature_importance_shap.csv", index=False)

# 2. Συσχέτιση Γνωρισμάτων με τον Στόχο (X65)
corr_with_target = train_data.corr()[['X65']].sort_values(by='X65', ascending=False)
plt.figure(figsize=(10, 6))
plt.plot(corr_with_target.index, corr_with_target['X65'], marker='o')
plt.xticks(rotation=90)
plt.xlabel("Features")
plt.ylabel("Correlation with X65")
plt.title("Correlation of Features with Bankruptcy (X65)")
plt.tight_layout()
plt.savefig("outputs/correlation_graph.png")
plt.close()

top_10_features = shap_df['Feature'].head(10).tolist()

# Train model with top 10 features
cat_model_top10, metrics_top10 = train_top_features_model(
    pd.DataFrame(X_train_balanced, columns=X_train_scaled.columns),  # Ensure DataFrame
    y_train_balanced,
    X_test_scaled,
    y_test,
    top_10_features
)

print("\nΑποτελέσματα μοντέλου με Top 10 Γνωρίσματα:")
evaluate_model(cat_model_top10, X_test_scaled[top_10_features], y_test)

# ======================================================================
# Προετοιμασία & Πρόβλεψη Άγνωστων Δεδομένων
# ======================================================================

# Φόρτωση & προετοιμασία άγνωστων δεδομένων
X = train_data.drop(columns=['X65'])  # Get final training features

unseen_data_scaled = prepare_unseen_data(
    filepath="data/test_unlabeled.csv",
    scaler=scaler,
    X_train_columns=X.columns.tolist(),  # Final features after selection
    imputation_values=imputation_values  # From training phase
)

# Αξιολόγηση μοντέλου και πρόβλεψη
print("\nΑποτελέσματα μοντέλου CatBoost:")
optimal_threshold = evaluate_model(cat_model, X_test_scaled, y_test)
predict_unseen_data(cat_model, unseen_data_scaled, scaler, "outputs/test_predictions.csv", threshold=optimal_threshold)

# Δημιουργία αρχείου προβλέψεων
test_predictions = pd.DataFrame({
    'prediction': cat_model.predict(unseen_data_scaled)
})
test_predictions.to_csv("outputs/test_predictions_final.csv", index=False, header=False)
print("\nΤο αρχείο 'test_predictions_final.csv' δημιουργήθηκε.")
print(test_predictions['prediction'].value_counts())

# Κατάταξη εταιριών βάσει πιθανότητας πτώχευσης
probabilities = cat_model.predict_proba(unseen_data_scaled)[:, 1]
rowid_probabilities = pd.DataFrame({
    'rowid': range(1, len(probabilities) + 1),
    'probability': probabilities
})

# Εύρεση των 50 εταιριών με τη μεγαλύτερη πιθανότητα πτώχευσης
top_50 = rowid_probabilities.sort_values(by='probability', ascending=False).head(50)
top_50[['rowid']].to_csv("outputs/top_50_predictions.csv", index=False, header=False)
print("\nΤο αρχείο 'top_50_predictions.csv' δημιουργήθηκε.")
