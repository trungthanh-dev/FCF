import pandas as pd

from feature_config import TARGET, get_features

def split_features_target(df):
    X = df[get_features(df)]
    y = df[TARGET]
    return X, y

def time_series_split(X, y, test_ratio = 0.2):
    split_index = int(len(X)*(1- test_ratio))

    X_train = X[:split_index]
    X_test = X[split_index:]

    y_train = y[:split_index]
    y_test = y[split_index:]

    return X_train, X_test, y_train, y_test

def dataset_summary(X_train, X_test):
    print('-'*60)
    print("Dataset Summary")
    print('-'*60)

    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")

    print(f"Training ratio: {len(X_train)/(len(X_train)+len(X_test)):.2%}")
    print(f"Testing ratio: {len(X_test) / (len(X_train) + len(X_test)):.2%}")