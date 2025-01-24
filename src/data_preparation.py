import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_clean_data(filepath, columns_with_replacements):
    data = pd.read_csv(filepath)
    data = data.replace("?", np.nan)

    # Identify rows that exceed this threshold
    rows_to_drop = data.index[data.isnull().sum(axis=1) > 0.5 * data.shape[1]]

    # Rows dropped
    print(f"Dropping {len(rows_to_drop)} rows with >50% missing values")

    # Drop these rows from the DataFrame
    data.drop(rows_to_drop, inplace=True)

    # Store computed imputation values
    imputation_values = {}

    # Perform imputation based on the specified strategies
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
def prepare_unseen_data(filepath, scaler, X_train_columns, imputation_values):
    """Φορτώνει και προετοιμάζει τα άγνωστα δεδομένα με βάση τις στρατηγικές εκπαίδευσης."""
    required_columns = ['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9', 'X10',
                        'X11', 'X12', 'X13', 'X14', 'X15', 'X16', 'X17', 'X18', 'X19', 'X20',
                        'X21', 'X22', 'X23', 'X24', 'X25', 'X26', 'X27', 'X28', 'X29', 'X30',
                        'X31', 'X32', 'X33', 'X34', 'X35', 'X36', 'X37', 'X38', 'X39', 'X40',
                        'X41', 'X42', 'X43', 'X44', 'X45', 'X46', 'X47', 'X48', 'X49', 'X50',
                        'X51', 'X52', 'X53', 'X54', 'X55', 'X56', 'X57', 'X58', 'X59', 'X60',
                        'X61', 'X66', 'X63', 'X64']
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