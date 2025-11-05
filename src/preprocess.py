
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder


redundant_column_names = ['id', "quantity_group", "waterpoint_type_group",]


def preprocess_data(X_train, X_test, y_train, y_test=None, make_binary=False):

    print("================== Data Preprocessing ===================")
    # Create copies of original data for preprocessing
    X_train_processed = X_train.copy()
    X_test_processed = X_test.copy()
    print(f"Initial shapes - X_train: {X_train_processed.shape}, X_test: {X_test_processed.shape}")

    if make_binary:
            y_train = y_train.replace({
                'functional needs repair': 'Needs repair or non functional',
                'non functional': 'Needs repair or non functional'
            })
            if y_test is not None:
                y_test = y_test.replace({
                    'functional needs repair': 'Needs repair or non functional',
                    'non functional': 'Needs repair or non functional'
                })

    X_train_processed, X_test_processed = apply_remove_redundant_columns(X_train_processed, X_test_processed)

    X_train_processed, X_test_processed = remove_single_value_columns(X_train_processed, X_test_processed)

    X_train_processed, X_test_processed = add_well_age(X_train_processed, X_test_processed)

    X_train_processed, X_test_processed = handle_missing_values(X_train_processed, X_test_processed)
    
    X_train_processed, X_test_processed = encode_categorial_variables(X_train_processed, X_test_processed, include_label_maps=False)
    
    X_train_processed, X_test_processed = scale_numeric_features(X_train_processed, X_test_processed)



    if y_test is not None:
        y_train_encoded, y_test_encoded, target_label_encoder = encode_labels(y_train, y_test)

        print("================== End ===================")
        return X_train_processed, X_test_processed, y_train_encoded, y_test_encoded, target_label_encoder
    else:
        y_train_encoded, target_label_encoder = encode_labels(y_train)
        print("================== End ===================")
        return X_train_processed, X_test_processed, y_train_encoded, target_label_encoder


def apply_remove_redundant_columns(X_train, X_test):
    # Identify and remove redundant features
    X_train = X_train.drop(columns=redundant_column_names, errors='ignore')
    X_test = X_test.drop(columns=redundant_column_names, errors='ignore')
    print(f"\nRemoved {len(redundant_column_names)} redundant attributes")
    print(f"Shape after removing redundant attributes - X_train: {X_train.shape}, X_test: {X_test.shape}")
    return X_train, X_test


def remove_single_value_columns(X_train, X_test):
    # Drop features that have only one unique value in training data
    single_value_cols = [col for col in X_train.columns if X_train[col].nunique() == 1]
    if single_value_cols:
        X_train = X_train.drop(columns=single_value_cols)
        X_test = X_test.drop(columns=single_value_cols, errors='ignore')
        print(f"\nRemoved {len(single_value_cols)} single-value features: {single_value_cols}")
    return X_train, X_test


def add_well_age(X_train, X_test):
    # Feature engineering: Create 'well_age' from 'construction_year'
    # Use 2025 as current year
    X_train['well_age'] = 2025 - X_train['construction_year']
    X_test['well_age'] = 2025 - X_test['construction_year']
    print(f"\nEngineered 'well_age' feature from 'construction_year'")

    # Now drop construction_year as we have engineered well_age
    X_train = X_train.drop(columns=['construction_year', 'date_recorded'], errors='ignore')
    X_test = X_test.drop(columns=['construction_year', 'date_recorded'], errors='ignore')
    return X_train, X_test


def handle_missing_values(X_train, X_test):
        # Print missing value count and percentage per column for training set
    print("\nMissing values in X_train before imputation:")
    null_counts = X_train.isnull().sum()
    null_percentages = X_train.isnull().mean() * 100
    for col in X_train.columns:
        print(f"{col}: {null_counts[col]} nulls ({null_percentages[col]:.2f}%)")

    # Separate numeric and categorical features
    numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X_train.select_dtypes(include=['object']).columns.tolist()
    print(f"\nNumeric features ({len(numeric_features)}): {numeric_features}")
    print(f"\nCategorical features ({len(categorical_features)}): {categorical_features}")

    # Impute missing values
    # Numeric: use median strategy
    numeric_imputer = SimpleImputer(strategy='median')
    X_train[numeric_features] = numeric_imputer.fit_transform(X_train[numeric_features])
    X_test[numeric_features] = numeric_imputer.transform(X_test[numeric_features])

    # Categorical: use most_frequent strategy
    categorical_imputer = SimpleImputer(strategy='most_frequent')
    X_train[categorical_features] = categorical_imputer.fit_transform(X_train[categorical_features])
    X_test[categorical_features] = categorical_imputer.transform(X_test[categorical_features])

    print(f"\nMissing values after imputation (train): {X_train.isnull().sum().sum()}")
    print(f"Missing values after imputation (test): {X_test.isnull().sum().sum()}")
    return X_train, X_test

def encode_categorial_variables(X_train, X_test, include_label_maps=False):
    # Using pandas .astype('category').cat.codes for ordinal encoding
    # This method: converts values to categories and assigns sequential integer codes (0, 1, 2, ...)
    # Advantage: handles most cases automatically and is more efficient than LabelEncoder
    # Unseen labels in test set: pandas will assign -1 to values not present in training data
    # This provides a robust way to handle data drift between train and test sets
    categorical_features = X_train.select_dtypes(include=['object']).columns.tolist()
    label_maps = {}

    for col in categorical_features:
        # Create category mapping from training data
        # Get unique categories from train set to ensure consistent encoding
        train_categories = X_train[col].astype('category').cat.categories
        
        # Encode training data
        X_train[col] = X_train[col].astype('category').cat.codes
        
        # Encode test data and handle unseen labels
        # Convert test data to same categories as train, unseen values become -1
        X_test[col] = pd.Categorical(
            X_test[col],
            categories=train_categories
        ).codes

        if include_label_maps:
            label_maps[col] = dict(enumerate(train_categories))

    print(f"\nEncoded {len(categorical_features)} categorical features")
    if include_label_maps:
        return X_train, X_test, label_maps
    else:
        return X_train, X_test


def scale_numeric_features(X_train, X_test):
    # Standardize numeric features (mean=0, std=1)
    # Important for algorithms sensitive to feature magnitude (e.g., KNN, SVM, Linear models)
    numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()

    scaler = StandardScaler()
    X_train[numeric_features] = scaler.fit_transform(X_train[numeric_features])
    X_test[numeric_features] = scaler.transform(X_test[numeric_features])

    print(f"\nScaled {len(numeric_features)} numeric features")
    return X_train, X_test


def encode_labels(y_train, y_test=None):
    # Encode target labels using LabelEncoder
    print(f"\nUnique target labels: {sorted(y_train.unique())}")
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train.values.ravel())
    if y_test is not None:
        y_test_encoded = label_encoder.transform(y_test.values.ravel())
    print("Encoded target labels")
    if y_test is not None:
        return y_train_encoded, y_test_encoded, label_encoder
    else:
        return y_train_encoded, label_encoder


if __name__ == "__main__":
    # Example usage
    from load_and_clean_data import load_data

    X_train, X_test, y_train = load_data()
    y_train = y_train.iloc[:, -1]
    X_train_processed, X_test_processed, y_train_encoded, target_label_encoder = preprocess_data(X_train, X_test, y_train, make_binary=True)
    X_train_processed, X_test_processed, y_train_encoded, target_label_encoder = preprocess_data(X_train, X_test, y_train, make_binary=False)
