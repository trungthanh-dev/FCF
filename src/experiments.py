import os

import numpy as np
import pandas as pd

from window import create_sliding_window
from dataset import split_features_target, time_series_split
from evalute import evaluate_regression, print_metrics
from models.random_forest import RandomForestModel
from models.lstm import LSTMModel
from visualization import (
    plot_actual_vs_predicted,
    plot_residuals,
    plot_feature_importance,
    plot_trajectory,
    plot_horizon_comparison,
)


def run_random_forest_experiment(
    cleaned_datasets,
    window_size,
    forecast_horizons,
    plot_dir,
    results_csv_path,
    model_dir=None,
    predictions_dir=None,
    use_cache=True,
):
    os.makedirs(plot_dir, exist_ok=True)
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)
    if predictions_dir:
        os.makedirs(predictions_dir, exist_ok=True)

    results = []

    for name, df in cleaned_datasets.items():
        X, y = split_features_target(df)
        feature_names = list(X.columns)

        for horizon in forecast_horizons:
            tag = f"{name}_h{horizon}"

            model_path = os.path.join(model_dir, f"{tag}.pkl") if model_dir else None
            pred_path = os.path.join(predictions_dir, f"{tag}.npz") if predictions_dir else None
            cache_available = (
                use_cache
                and model_path and pred_path
                and os.path.exists(model_path)
                and os.path.exists(pred_path)
            )

            if cache_available:
                print(f"\n{name} | Horizon = {horizon} — loaded from cache, skipping training")

                rf = RandomForestModel()
                rf.load(model_path)

                cached = np.load(pred_path)
                y_test, y_pred = cached["y_test"], cached["y_pred"]

            else:
                X_window, y_window = create_sliding_window(X, y, window_size, horizon)
                X_train, X_test, y_train, y_test = time_series_split(X_window, y_window)

                print(f"\n{name} | Horizon = {horizon}")
                print("X_train:", X_train.shape)
                print("X_test:", X_test.shape)
                print("y_train:", y_train.shape)
                print("y_test:", y_test.shape)

                rf = RandomForestModel()
                rf.train(X_train, y_train)
                y_pred = rf.predict(X_test)

                if model_path:
                    rf.save(model_path)
                if pred_path:
                    np.savez(pred_path, y_test=y_test, y_pred=y_pred)

            metrics = evaluate_regression(y_test, y_pred)
            print_metrics(metrics)

            metrics["ship"] = name
            metrics["horizon"] = horizon
            results.append(metrics)

            plot_actual_vs_predicted(
                y_test, y_pred,
                title=f"Actual vs Predicted — {name}, horizon={horizon}",
                save_path=os.path.join(plot_dir, f"{tag}_actual_vs_pred.png"),
            )
            plot_residuals(
                y_test, y_pred,
                title=f"Residuals — {name}, horizon={horizon}",
                save_path=os.path.join(plot_dir, f"{tag}_residuals.png"),
            )
            plot_trajectory(
                y_test, y_pred,
                title=f"Fuel Consumption Trajectory: Actual vs. Predicted — {name}, horizon={horizon}",
                save_path=os.path.join(plot_dir, f"{tag}_trajectory.png"),
            )

            flattened_names = [
                f"{feat}_t-{window_size - lag}"
                for lag in range(window_size)
                for feat in feature_names
            ]
            plot_feature_importance(
                rf.feature_importance(),
                flattened_names,
                title=f"Feature importance — {name}, horizon={horizon}",
                save_path=os.path.join(plot_dir, f"{tag}_feature_importance.png"),
                top_n=20,
            )

    results_df = pd.DataFrame(results)[["ship", "horizon", "MAE", "RMSE", "R2"]]
    results_df.to_csv(results_csv_path, index=False)

    print("\nFull results (all ships x all horizons):")
    print(results_df)

    plot_horizon_comparison(results_df, save_dir=plot_dir)

    return results_df


def run_lstm_experiment(
        cleaned_datasets,
        window_size,
        forecast_horizons,
        plot_dir,
        results_csv_path,
        model_dir=None,
        predictions_dir=None,
        use_cache=False,
        hidden_size=128,
        num_layers=2,
        dropout=0.1,
        learning_rate=5e-4,
        epochs=150,
        batch_size=128,
):
    """
    Train, evaluate, and plot an LSTM model for every (ship, horizon)
    combination. Mirrors run_random_forest_experiment(): same caching
    pattern (model + predictions saved to disk, reloaded on later runs),
    same plot set — except feature importance, which Random Forest
    exposes natively but LSTM does not.

    LSTM consumes the sliding-window input directly as a 3D tensor
    (samples, window_size, features); no flattening step is needed
    (handled internally by LSTMModel).

    Unlike Random Forest — whose tree-based splits are invariant to
    feature scale — LSTM is trained via gradient descent through
    sigmoid/tanh activations and is highly sensitive to input scale.
    Feature scaling, a chronological validation split for early
    stopping, and gradient clipping are all handled internally by
    LSTMModel.train()/predict() (see models/lstm.py), so raw windowed
    data is passed here unmodified.

    Parameters
    ----------
    cleaned_datasets, window_size, forecast_horizons, plot_dir,
    results_csv_path, model_dir, predictions_dir, use_cache :
        Same meaning as in run_random_forest_experiment(). use_cache
        defaults to False here since LSTM training configuration
        (architecture, scaler, early stopping) changes more often
        during experimentation than the RF baseline.
    hidden_size, num_layers, dropout, learning_rate, epochs, batch_size :
        Forwarded to LSTMModel(...) for every combination.

    Returns
    -------
    pd.DataFrame
        Columns: ship, horizon, MAE, RMSE, R2 — one row per combination.
    """
    os.makedirs(plot_dir, exist_ok=True)
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)
    if predictions_dir:
        os.makedirs(predictions_dir, exist_ok=True)

    results = []

    for name, df in cleaned_datasets.items():
        X, y = split_features_target(df)

        for horizon in forecast_horizons:
            tag = f"{name}_h{horizon}"

            model_path = os.path.join(model_dir, f"{tag}.pt") if model_dir else None
            pred_path = os.path.join(predictions_dir, f"{tag}.npz") if predictions_dir else None
            cache_available = (
                    use_cache
                    and model_path and pred_path
                    and os.path.exists(model_path)
                    and os.path.exists(pred_path)
            )

            if cache_available:
                print(f"\n[LSTM] {name} | Horizon = {horizon} — loaded from cache, skipping training")

                lstm = LSTMModel(
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout=dropout,
                    learning_rate=learning_rate,
                    epochs=epochs,
                    batch_size=batch_size,
                )
                lstm.load(model_path)

                cached = np.load(pred_path)
                y_test, y_pred = cached["y_test"], cached["y_pred"]

            else:
                X_window, y_window = create_sliding_window(
                    X, y, window_size, horizon
                )
                X_train, X_test, y_train, y_test = time_series_split(
                    X_window, y_window
                )

                print(f"\n[LSTM] {name} | Horizon = {horizon}")
                print("X_train:", X_train.shape)
                print("X_test:", X_test.shape)
                print("y_train:", y_train.shape)
                print("y_test:", y_test.shape)

                lstm = LSTMModel(
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout=dropout,
                    learning_rate=learning_rate,
                    epochs=epochs,
                    batch_size=batch_size,
                )
                lstm.train(X_train, y_train)
                y_pred = lstm.predict(X_test)

                if model_path:
                    lstm.save(model_path)
                if pred_path:
                    np.savez(pred_path, y_test=y_test, y_pred=y_pred)

            metrics = evaluate_regression(y_test, y_pred)
            print_metrics(metrics)

            metrics["ship"] = name
            metrics["horizon"] = horizon
            results.append(metrics)

            plot_actual_vs_predicted(
                y_test, y_pred,
                title=f"[LSTM] Actual vs Predicted — {name}, horizon={horizon}",
                save_path=os.path.join(plot_dir, f"{tag}_actual_vs_pred.png"),
            )
            plot_residuals(
                y_test, y_pred,
                title=f"[LSTM] Residuals — {name}, horizon={horizon}",
                save_path=os.path.join(plot_dir, f"{tag}_residuals.png"),
            )
            plot_trajectory(
                y_test, y_pred,
                title=f"[LSTM] Fuel Consumption Trajectory: Actual vs. Predicted — {name}, horizon={horizon}",
                save_path=os.path.join(plot_dir, f"{tag}_trajectory.png"),
            )

    results_df = pd.DataFrame(results)[["ship", "horizon", "MAE", "RMSE", "R2"]]
    results_df.to_csv(results_csv_path, index=False)

    print("\n[LSTM] Full results (all ships x all horizons):")
    print(results_df)

    plot_horizon_comparison(results_df, save_dir=plot_dir)

    return results_df