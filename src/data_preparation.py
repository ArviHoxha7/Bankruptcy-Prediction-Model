import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_clean_data(filepath, columns_with_replacements):
    data = pd.read_csv(filepath)
    data = data.replace("?", np.nan)

    for column, strategy in columns_with_replacements.items():
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors='coerce')
            if strategy == 'median':
                data[column] = data[column].fillna(data[column].median())
            elif strategy == 'mean':
                data[column] = data[column].fillna(data[column].mean())
            elif strategy == 0:
                data[column] = data[column].fillna(0)

    return data

def preprocess_and_split_data(data, target_column='X65', test_size=0.2, random_state=42):
    X = data.drop(columns=[target_column])
    y = data[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def prepare_unseen_data(filepath, required_columns, scaler, X_train_columns):
    """Φορτώνει και προετοιμάζει τα άγνωστα δεδομένα."""
    # Φόρτωση δεδομένων
    unseen_data = pd.read_csv(filepath, header=None)
    unseen_data = unseen_data.replace("?", np.nan)
    unseen_data.columns = required_columns

    # Προσθήκη ελλιπουσών στηλών
    for col in X_train_columns:
        if col not in unseen_data.columns:
            unseen_data[col] = 0

    # Διασφάλιση σωστής σειράς στηλών
    unseen_data = pd.DataFrame(unseen_data, columns=X_train_columns)

    # Κανονικοποίηση δεδομένων
    unseen_data_scaled = scaler.transform(unseen_data)

    return unseen_data_scaled