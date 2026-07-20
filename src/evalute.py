import numpy as np
import pandas as pd

from sklearn.metrics import(
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

def dtw_distance(y_true, y_pred, window=10):
    """
    Dynamic Time Warping distance between two 1D sequences, restricted to a
    Sakoe-Chiba band of the given radius around the diagonal.

    A full O(n*m) DTW matrix is infeasible for this project's test-set sizes
    (Poseidon's test split alone is ~20k points -> a 20k x 20k float64 matrix
    is several GB and the nested Python loop never finishes in practice).
    Forecast/target sequences here are the same length and only ever drift by
    a few steps, so a small band around the diagonal gives the same answer as
    full DTW while keeping memory at O(m) (two rolling rows) and time at
    O(n * window).
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    n = len(y_true)
    m = len(y_pred)

    window = max(window, abs(n - m))

    prev_row = np.full(m + 1, np.inf)
    curr_row = np.full(m + 1, np.inf)
    prev_row[0] = 0.0

    for i in range(1, n + 1):
        curr_row.fill(np.inf)
        j_lo = max(1, i - window)
        j_hi = min(m, i + window)

        for j in range(j_lo, j_hi + 1):
            cost = abs(y_true[i - 1] - y_pred[j - 1])
            curr_row[j] = cost + min(
                prev_row[j],        # insertion
                curr_row[j - 1],    # deletion
                prev_row[j - 1]     # match
            )

        prev_row, curr_row = curr_row, prev_row

    return prev_row[m] / (n + m)

def evaluate_regression(
        y_true,
        y_pred,
):
    mae = mean_absolute_error(y_true, y_pred)

    rmse = np.sqrt(
        mean_squared_error(y_true, y_pred)
    )

    r2 = r2_score(y_true, y_pred)

    dtw = dtw_distance(y_true, y_pred)

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "DTW": dtw
    }

def print_metrics(
        metrics,
):
    print(f"MAE  : {metrics['MAE']:.6f}")
    print(f"RMSE : {metrics['RMSE']:.6f}")
    print(f"R^2  : {metrics['R2']:.6f}")
    print(f"DTW : {metrics['DTW']:.6f}")

def save_metrics(
        metrics,
        path,
):
    pd.DataFrame(
        [metrics]
    ).to_csv(
        path,
        index=False
    )
