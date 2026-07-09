import numpy as np
import pandas as pd

from sklearn.metrics import(
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


def evaluate_regression(
        y_true,
        y_pred,
):
    mae = mean_absolute_error(
        y_true,
        y_pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred
        )
    )

    r2 = r2_score(
        y_true,
        y_pred,
    )

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
    }

def print_metrics(
        metrics,
):
    print(f"MAE  : {metrics['MAE']:.6f}")
    print(f"RMSE : {metrics['RMSE']:.6f}")
    print(f"R^2  : {metrics['R2']:.6f}")

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
