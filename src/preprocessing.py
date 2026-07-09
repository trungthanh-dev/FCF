import pandas as pd

from feature_config import *

def remove_leakage(df: pd.DataFrame):
    leakage = detect_leakage_features(df)
    print(f"Removed {len(leakage)} leakage features.")
    return df.drop(
        columns = leakage,
        errors = "ignore"
    )

def remove_empty_columns(df):
    empty_cols = df.columns[df.isnull().all()]

    return df.drop(
        columns = empty_cols
    )

def remove_drop_features(df):
    drop_features = detect_drop_features(df)
    return df.drop(
        columns = DROP_FEATURES,
        errors = "ignore"
    )

def handle_missing(df):
    df = df.ffill()
    df = df.bfill()
    return df

def preprocess(df: pd.DataFrame):
    df = df.copy()

    df = remove_leakage(df)
    df = remove_empty_columns(df)
    df = remove_drop_features(df)
    df = handle_missing(df)

    return df



