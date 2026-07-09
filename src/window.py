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

def reshape_for_random_forest(X: np.ndarray):
    """
    Reshape 3D sliding-window data into 2D for Random Forest.
    """
    return X.reshape(X.shape[0], -1)