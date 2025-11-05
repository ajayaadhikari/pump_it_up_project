import os
from pathlib import Path
import pandas as pd

# File name variables
X_train_file = "X_train_with_population.csv"
X_test_file = "X_test_with_population.csv"
y_train_file = "y_train.csv"

def load_data():
    # Load CSVs as DataFrames
    print("================== Data loading ===================")

    X_train = pd.read_csv(Path('data') / X_train_file)
    X_test = pd.read_csv(Path('data') / X_test_file)
    y_train = pd.read_csv(Path('data') / y_train_file)
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_train shape:", y_train.shape)
    print("Example row from X_train:")
    print(X_train.iloc[0])

    print("================== End ===================")
    return X_train, X_test, y_train


if __name__ == "__main__":
    X_train, X_test, y_train = load_data()
