import os
import numpy as np
import pandas as pd
import optuna

from config import WINDOW_SIZE, TEST_SIZE
from dataset import split_features_target
from window import create_sliding_window_delta
from evalute import evaluate_regression, print_metrics
from experiments import _scale_metrics
from models.tcn import TCNModel
from visualization import plot_actual_vs_predicted, plot_residuals, plot_trajectory

# ---------------------------------------------------------------------------
# Optuna hyperparameter search for TCN-delta on Poseidon. TCN-delta already
# beats Seq2Seq-delta at every horizon with DEFAULT hyperparameters (see
# eda_output_power/tcn_poseidon_delta_results.csv vs
# seq2seq_poseidon_delta_results.csv) -- this searches around that default,
# one study per horizon since the best receptive field/capacity differs a lot
# between h=1 and h=20 (R2 0.986 -> 0.840 with defaults).
#
# Objective is validation-set MAE in raw ShaftPower units (anchor +
# delta_hat reconstruction), NOT the scaled Huber val_loss TCNModel.train()
# minimizes internally -- that loss lives on an arbitrary delta-scaler scale
# and isn't what's reported/compared (MAE/RMSE/R2/DTW in results.csv). The
# val split used for this objective is recomputed with the same VAL_RATIO
# TCNModel uses internally, so it lines up with the exact rows
# TCNModel.train() already set aside and restored best-val-loss weights for.
#
# Trials use a smaller epoch/patience budget than the final retrain to keep
# the search affordable; the winning config per horizon is retrained at full
# budget before evaluating on the held-out test set, mirroring
# main_tcn_poseidon_delta.py's evaluation shape so results are comparable.
# ---------------------------------------------------------------------------

EDA_DIR = "eda_output_power"
DATA_DIR = "data_clean_power"
PLOT_DIR = os.path.join(EDA_DIR, "tcn_poseidon_optuna_plots")
MODEL_DIR = os.path.join("models_saved_power", "tcn_poseidon_optuna")
PRED_DIR = os.path.join("predictions_cache_power", "tcn_poseidon_optuna")
RESULTS_CSV = os.path.join(EDA_DIR, "tcn_poseidon_optuna_results.csv")
BEST_PARAMS_CSV = os.path.join(EDA_DIR, "tcn_poseidon_optuna_best_params.csv")

os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PRED_DIR, exist_ok=True)

UNIT_SCALE = 1e6
UNIT_LABEL = "MW"
FORECAST_HORIZONS = [1, 5, 10, 20]
VAL_RATIO = 0.1
N_TRIALS = 25       # per horizon -- raise if Colab time budget allows
SEARCH_EPOCHS = 60  # smaller budget while searching trials
SEARCH_PATIENCE = 6
FINAL_EPOCHS = 150  # TCNModel default, used only for the winning config
FINAL_PATIENCE = 10

df = pd.read_parquet(os.path.join(DATA_DIR, "poseidon_clean.parquet"))
X, y = split_features_target(df)

all_results = []
best_params_records = []

for horizon in FORECAST_HORIZONS:
    tag = f"Poseidon_h{horizon}_optuna"
    print(f"\n{'=' * 70}\n[Optuna] Poseidon | Horizon = {horizon}\n{'=' * 70}")

    X_window, delta_window, anchor_window = create_sliding_window_delta(
        X, y, WINDOW_SIZE, horizon
    )
    split_index = int(len(X_window) * (1 - TEST_SIZE))
    X_train, X_test = X_window[:split_index], X_window[split_index:]
    delta_train, delta_test = delta_window[:split_index], delta_window[split_index:]
    anchor_test = anchor_window[split_index:]

    # Same chronological val slice TCNModel.train() carves out internally
    # (last VAL_RATIO fraction of X_train, no shuffling) -- recomputed here
    # so the Optuna objective can reconstruct raw-unit predictions on it.
    n_val = int(len(X_train) * VAL_RATIO)
    anchor_val = anchor_window[:split_index][-n_val:]
    y_val_raw = anchor_val + delta_train[-n_val:]
    X_val = X_train[-n_val:]

    def objective(trial, X_train=X_train, delta_train=delta_train,
                  X_val=X_val, anchor_val=anchor_val, y_val_raw=y_val_raw):
        depth = trial.suggest_int("depth", 3, 6)
        width = trial.suggest_categorical("width", [16, 32, 64, 128])
        kernel_size = trial.suggest_categorical("kernel_size", [2, 3, 5, 7])
        dropout = trial.suggest_float("dropout", 0.0, 0.4)
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])

        model = TCNModel(
            num_channels=[width] * depth,
            kernel_size=kernel_size,
            dropout=dropout,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            batch_size=batch_size,
            epochs=SEARCH_EPOCHS,
            patience=SEARCH_PATIENCE,
            val_ratio=VAL_RATIO,
        )
        model.train(X_train, delta_train, verbose=False)

        delta_val_pred = model.predict(X_val)
        y_val_pred_raw = anchor_val + delta_val_pred
        return float(np.mean(np.abs(y_val_raw - y_val_pred_raw)))

    study = optuna.create_study(direction="minimize", study_name=tag)
    study.optimize(objective, n_trials=N_TRIALS)

    print(f"\n[Optuna] Best val MAE (raw units): {study.best_value:.4f}")
    print(f"[Optuna] Best params: {study.best_params}")

    best = study.best_params
    best_params_records.append({"horizon": horizon, "best_val_mae": study.best_value, **best})

    final_model = TCNModel(
        num_channels=[best["width"]] * best["depth"],
        kernel_size=best["kernel_size"],
        dropout=best["dropout"],
        learning_rate=best["learning_rate"],
        weight_decay=best["weight_decay"],
        batch_size=best["batch_size"],
        epochs=FINAL_EPOCHS,
        patience=FINAL_PATIENCE,
        val_ratio=VAL_RATIO,
    )
    final_model.train(X_train, delta_train)
    delta_pred = final_model.predict(X_test)

    y_test_raw = anchor_test + delta_test
    y_pred_raw = anchor_test + delta_pred

    final_model.save(os.path.join(MODEL_DIR, f"{tag}.pt"))
    np.savez(os.path.join(PRED_DIR, f"{tag}.npz"), y_test=y_test_raw, y_pred=y_pred_raw)

    metrics = _scale_metrics(evaluate_regression(y_test_raw, y_pred_raw), UNIT_SCALE)
    print_metrics(metrics)
    metrics["ship"] = "Poseidon"
    metrics["horizon"] = horizon
    all_results.append(metrics)

    plot_actual_vs_predicted(
        y_test_raw, y_pred_raw,
        title=f"[TCN-optuna] Actual vs Predicted — Poseidon, horizon={horizon}",
        save_path=os.path.join(PLOT_DIR, f"{tag}_actual_vs_pred.png"),
    )
    plot_residuals(
        y_test_raw, y_pred_raw,
        title=f"[TCN-optuna] Residuals — Poseidon, horizon={horizon}",
        save_path=os.path.join(PLOT_DIR, f"{tag}_residuals.png"),
    )
    plot_trajectory(
        y_test_raw, y_pred_raw,
        title=f"[TCN-optuna] ShaftPower Trajectory: Actual vs. Predicted — Poseidon, horizon={horizon}",
        save_path=os.path.join(PLOT_DIR, f"{tag}_trajectory.png"),
    )

results_df = pd.DataFrame(all_results)[["ship", "horizon", "MAE", "RMSE", "R2", "DTW"]]
results_df.to_csv(RESULTS_CSV, index=False)
pd.DataFrame(best_params_records).to_csv(BEST_PARAMS_CSV, index=False)

print(f"\n[TCN-optuna] Full results [MAE/RMSE/DTW in {UNIT_LABEL}]:")
print(results_df)
print(f"\nBest params per horizon saved to {BEST_PARAMS_CSV}")
print("Compare against eda_output_power/tcn_poseidon_delta_results.csv (default hyperparameters).")
