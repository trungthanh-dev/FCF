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

def add_target_history_features(
    df,
    target=TARGET,
    lag_steps=TARGET_LAG_STEPS,
    rolling_windows=TARGET_ROLLING_WINDOWS,
):
    """
    Add lagged and rolling-window features built from the target itself
    (e.g. Fuel_Lag1, Fuel_RollingMean5).

    IMPORTANT — no leakage: every value uses ONLY past rows. Rolling
    windows call .shift(1) BEFORE .rolling(...), so a rolling mean at row
    i only sees rows [i-window, i-1], never row i. Lags use .shift(lag)
    directly, lag >= 1.

    Why: ships like Triton spread feature importance across ~20 weak
    sensor features with no single strong signal (unlike Poseidon, where
    Ship_SpeedThroughWater alone explains most of the variance). A lagged/
    rolling version of the target gives the model a direct anchor to
    recent fuel consumption that raw sensor features don't provide.

    The first `max(lag_steps + rolling_windows)` rows of each ship will
    have NaN in these new columns (no history yet) — handle_missing()
    should be re-run after this to fill them via ffill/bfill, same as the
    rest of the pipeline.
    """
    df = df.copy()
    for lag in lag_steps:
        df[f"Fuel_Lag{lag}"] = df[target].shift(lag)
    for window in rolling_windows:
        shifted = df[target].shift(1)
        df[f"Fuel_RollingMean{window}"] = shifted.rolling(window).mean()
        df[f"Fuel_RollingStd{window}"] = shifted.rolling(window).std()
    return df

def preprocess(df: pd.DataFrame, add_target_history=True):
    df = df.copy()

    df = remove_leakage(df)
    df = remove_empty_columns(df)
    df = remove_drop_features(df)
    df = handle_missing(df)

    if add_target_history:
        df = add_target_history_features(df)
        # Re-fill: lag/rolling introduce fresh NaNs at the start of the
        # series (no history for the first few rows) that the earlier
        # handle_missing() call couldn't have touched since these columns
        # didn't exist yet.
        df = handle_missing(df)

    return df