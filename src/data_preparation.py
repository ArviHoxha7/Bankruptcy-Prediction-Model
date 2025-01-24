import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_and_clean_data(filepath, columns_with_replacements):
    data = pd.read_csv(filepath)
    data = data.replace("?", np.nan)

    # Store computed imputation values
    imputation_values = {}

    for column, strategy in columns_with_replacements.items():
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors='coerce')
            if strategy == 'median':
                median_val = data[column].median()
                imputation_values[column] = median_val
                data[column] = data[column].fillna(median_val)
            elif strategy == 'mean':
                mean_val = data[column].mean()
                imputation_values[column] = mean_val
                data[column] = data[column].fillna(mean_val)
            elif strategy == 0:
                imputation_values[column] = 0
                data[column] = data[column].fillna(0)

    return data, imputation_values

def preprocess_and_split_data(data, target_column='X65', test_size=0.2):
    X = data.drop(columns=[target_column])
    y = data[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        shuffle=True
    )

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler
def prepare_unseen_data(filepath, required_columns, scaler, X_train_columns, imputation_values):
    """Φορτώνει και προετοιμάζει τα άγνωστα δεδομένα με βάση τις στρατηγικές εκπαίδευσης."""
    # Φόρτωση δεδομένων
    unseen_data = pd.read_csv(filepath, header=None)
    unseen_data = unseen_data.replace("?", np.nan)
    unseen_data.columns = required_columns

    # Εφαρμογή των ίδιων στρατηγικών αντικατάστασης
    for column, strategy_value in imputation_values.items():
        if column in unseen_data.columns:
            unseen_data[column] = unseen_data[column].fillna(strategy_value)

    # Προσθήκη ελλιπουσών στηλών με 0 και διασφάλιση σειράς
    for col in X_train_columns:
        if col not in unseen_data.columns:
            unseen_data[col] = 0

    unseen_data = unseen_data[X_train_columns]  # Ensure column order matches training

    # Κανονικοποίηση
    unseen_data_scaled = scaler.transform(unseen_data)

    return unseen_data_scaled