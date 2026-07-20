import numpy as np
import pandas as pd

from config import*

def create_sliding_window(
    X: pd.DataFrame,
    y: pd.Series,
    WINDOW_SIZE,
    horizon,
):
    """
    Create sliding-window samples for time-series forecasting.

    Parameters
    ----------
    X : pd.DataFrame
        Input features.
    y : pd.Series
        Target variable.
    window_size : int
        Number of past time steps.
    horizon : int
        Forecast horizon.

    Returns
    -------
    X_window : np.ndarray
        Shape (samples, window_size, features)
    y_window : np.ndarray
        Shape (samples,)
    """
    X_window = []
    y_window = []
    for i in range(len(X) - WINDOW_SIZE - horizon + 1):
        X_window.append(
            X.iloc[i:i + WINDOW_SIZE].values
        )
        y_window.append(
            y.iloc[i + WINDOW_SIZE + horizon - 1]
        )
    return np.array(X_window), np.array(y_window)

def create_sliding_window_delta(
    X: pd.DataFrame,
    y: pd.Series,
    WINDOW_SIZE,
    horizon,
):
    """
    Same windowing as create_sliding_window(), but the target is the CHANGE
    in y over the horizon (y "now" -> y "now + horizon") instead of the raw
    future value. "Now" is the last row of the window, i.e. the most recent
    actual reading available at prediction time -- using it as the anchor
    introduces no leakage, same as the existing Lag1 target-history feature.

    This exists because direct-forecast models trained on the raw target
    consistently learn to lean on the target's own Lag1 feature (RF feature
    importance: ~0.8 for Poseidon/Ceto), which is persistence forecasting in
    disguise -- "predict the last known value" -- and shows up as phase lag
    against the real trajectory (see evalute.dtw_distance). Predicting the
    delta instead removes Lag1 as a free lunch: copying it now gives a
    delta prediction of ~0, which is only right when nothing changes.

    Returns
    -------
    X_window : (samples, window_size, features)
    delta_window : (samples,) -- y(now + horizon) - y(now)
    anchor_window : (samples,) -- y(now), needed to reconstruct the raw
        prediction: y_hat(now + horizon) = anchor + delta_hat
    """
    X_window = []
    delta_window = []
    anchor_window = []
    for i in range(len(X) - WINDOW_SIZE - horizon + 1):
        anchor = y.iloc[i + WINDOW_SIZE - 1]
        target = y.iloc[i + WINDOW_SIZE + horizon - 1]
        X_window.append(X.iloc[i:i + WINDOW_SIZE].values)
        anchor_window.append(anchor)
        delta_window.append(target - anchor)
    return np.array(X_window), np.array(delta_window), np.array(anchor_window)

def create_seq2seq_window_delta(
        X: pd.DataFrame,
        y: pd.Series,
        WINDOW_SIZE,
        horizons,
):
    """
    Delta-target counterpart of create_seq2seq_window(): each horizon's
    target is y(now + h) - y(now) instead of the raw future value, "now"
    being the last row of the window (same anchor definition as
    create_sliding_window_delta()). One anchor per sample, shared across
    all horizon columns, since "now" doesn't depend on which horizon is
    being predicted.

    Returns
    -------
    X_window : (samples, window_size, features)
    delta_window : (samples, n_horizons)
    anchor_window : (samples,)
    """
    max_h = max(horizons)
    X_window = []
    delta_window = []
    anchor_window = []
    for i in range(len(X) - WINDOW_SIZE - max_h + 1):
        anchor = y.iloc[i + WINDOW_SIZE - 1]
        deltas = [
            y.iloc[i + WINDOW_SIZE + h - 1] - anchor
            for h in horizons
        ]
        X_window.append(X.iloc[i: i + WINDOW_SIZE].values)
        anchor_window.append(anchor)
        delta_window.append(deltas)
    return np.array(X_window), np.array(delta_window), np.array(anchor_window)

def reshape_for_random_forest(X: np.ndarray):
    """
    Reshape 3D sliding-window data into 2D for Random Forest.
    """
    return X.reshape(X.shape[0], -1)

def create_seq2seq_window(
        X: pd.DataFrame,
        y: pd.Series,
        WINDOW_SIZE,
        horizons,
):
    max_h = max(horizons)
    X_window = []
    y_window = []
    for i in range (len(X) - WINDOW_SIZE - max_h + 1):
        X_window.append(
            X.iloc[i: i + WINDOW_SIZE].values
        )
        targets = [
            y.iloc[i + WINDOW_SIZE + h - 1]
            for h in horizons
        ]
        y_window.append(targets)
    return np.array(X_window), np.array(y_window)