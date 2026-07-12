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
    print(f"Removed {len(drop_features)} non-predictive features.")
    # FIX: was dropping the literal keyword list (DROP_FEATURES, e.g. ["FuelType"]),
    # which never matches real column names like "Consumer_Boiler1_FuelType".
    # errors="ignore" silently hid this, so no exception was ever raised.
    # Now drops the actual matched column names returned by detect_drop_features().
    return df.drop(
        columns = drop_features,
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