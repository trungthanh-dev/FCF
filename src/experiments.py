import os

import numpy as np
import pandas as pd

from window import create_sliding_window
from dataset import split_features_target, time_series_split
from evalute import evaluate_regression, print_metrics
from models.random_forest import RandomForestModel
from models.lstm import LSTMModel
from window import create_seq2seq_window
from models.seq2seq_lstm import Seq2SeqLTSMModel
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
        val_ratio=0.1,
        patience=10,
        loss_delta=1.0,
        per_ship_params=None,
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
    hidden_size, num_layers, dropout, learning_rate, epochs, batch_size,
    val_ratio, patience, loss_delta :
        Default hyperparameters forwarded to LSTMModel(...) for every
        ship/horizon combination, unless overridden per-ship below.
    per_ship_params : dict[str, dict] or None
        Optional per-ship hyperparameter overrides, e.g.
        {"Triton": {"hidden_size": 32, "num_layers": 1, "patience": 20}}.
        Only the keys you want to override need to be present; anything
        missing falls back to the defaults above. This exists because
        smaller/noisier datasets (Triton, Ceto) overfit much faster than
        Poseidon and need a smaller model and/or more patience, rather
        than forcing every ship through identical hyperparameters.

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

    per_ship_params = per_ship_params or {}
    base_params = dict(
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        learning_rate=learning_rate,
        epochs=epochs,
        batch_size=batch_size,
        val_ratio=val_ratio,
        patience=patience,
        loss_delta=loss_delta,
    )

    results = []

    for name, df in cleaned_datasets.items():
        X, y = split_features_target(df)

        # Merge base defaults with any per-ship overrides for this ship.
        ship_params = {**base_params, **per_ship_params.get(name, {})}
        print(f"\n[LSTM] {name} — hyperparameters: {ship_params}")

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

                lstm = LSTMModel(**ship_params)
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

                lstm = LSTMModel(**ship_params)
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


def run_seq2seq_experiment(
        cleaned_datasets,
        window_size,
        forecast_horizons,
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
        val_ratio=0.1,
        patience=10,
        loss_delta=1.0,
        per_ship_params=None,
):
    """
    Train and evaluate ONE seq2seq LSTM per ship (not one per horizon).
    A single encoder-decoder model predicts all of `forecast_horizons`
    simultaneously for a given ship, so predictions across horizons come
    from one shared internal state -- unlike run_lstm_experiment(), which
    trains 4 independent direct-forecast models whose splits/weights
    never have to agree with each other.

    Mirrors run_lstm_experiment()'s conventions: per-ship hyperparameter
    overrides via `per_ship_params`, Huber loss, target/feature scaling
    handled inside Seq2SeqLSTMModel.

    Returns
    -------
    pd.DataFrame
        Columns: ship, horizon, MAE, RMSE, R2 -- one row per (ship, horizon),
        same shape as run_lstm_experiment()'s output, so results are
        directly comparable in the same results table / plots.
    """
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)
    if predictions_dir:
        os.makedirs(predictions_dir, exist_ok=True)

    per_ship_params = per_ship_params or {}
    base_params = dict(
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        learning_rate=learning_rate,
        epochs=epochs,
        batch_size=batch_size,
        val_ratio=val_ratio,
        patience=patience,
        loss_delta=loss_delta,
    )

    results = []

    for name, df in cleaned_datasets.items():
        X, y = split_features_target(df)

        ship_params = {**base_params, **per_ship_params.get(name, {})}
        print(f"\n[Seq2Seq] {name} — hyperparameters: {ship_params}")

        model_path = os.path.join(model_dir, f"{name}_seq2seq.pt") if model_dir else None
        pred_path = os.path.join(predictions_dir, f"{name}_seq2seq.npz") if predictions_dir else None
        cache_available = (
                use_cache
                and model_path and pred_path
                and os.path.exists(model_path)
                and os.path.exists(pred_path)
        )

        if cache_available:
            print(f"[Seq2Seq] {name} — loaded from cache, skipping training")

            model = Seq2SeqLSTMModel(horizons=forecast_horizons, **ship_params)
            model.load(model_path)

            cached = np.load(pred_path)
            y_test, y_pred = cached["y_test"], cached["y_pred"]

        else:
            X_window, y_window = create_seq2seq_window(
                X, y, window_size, forecast_horizons
            )
            X_train, X_test, y_train, y_test = time_series_split(
                X_window, y_window
            )

            print(f"[Seq2Seq] {name} | horizons={forecast_horizons}")
            print("X_train:", X_train.shape)
            print("X_test:", X_test.shape)
            print("y_train:", y_train.shape)
            print("y_test:", y_test.shape)

            model = Seq2SeqLSTMModel(horizons=forecast_horizons, **ship_params)
            model.train(X_train, y_train)
            y_pred = model.predict(X_test)

            if model_path:
                model.save(model_path)
            if pred_path:
                np.savez(pred_path, y_test=y_test, y_pred=y_pred)

        # y_test / y_pred are (samples, n_horizons); evaluate one column
        # (= one horizon) at a time, same metric functions as everywhere
        # else in the pipeline, so results are directly comparable.
        for col_idx, horizon in enumerate(forecast_horizons):
            metrics = evaluate_regression(y_test[:, col_idx], y_pred[:, col_idx])
            print(f"[Seq2Seq] {name} | horizon={horizon}")
            print_metrics(metrics)

            metrics["ship"] = name
            metrics["horizon"] = horizon
            results.append(metrics)

    results_df = pd.DataFrame(results)[["ship", "horizon", "MAE", "RMSE", "R2"]]
    results_df.to_csv(results_csv_path, index=False)

    print("\n[Seq2Seq] Full results (all ships x all horizons):")
    print(results_df)

    return results_df