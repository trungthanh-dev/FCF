import os
import numpy as np
import pandas as pd
import optuna

from config import WINDOW_SIZE, TEST_SIZE, RANDOM_STATE
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
#
# Noise-robust rework (2026-07-23): two independent runs of this script over
# an IDENTICAL search space picked different "best" hyperparameters and got
# different final-retrain quality out of them (h10 "win" shrank from -10.6%
# to -1.1%, h20 flipped sign entirely) -- see CLAUDE.md. Even with the
# cuDNN-determinism fix (models/tcn.py), a single training run is not a
# trustworthy signal for a config's quality: this is the standard "noisy
# black-box optimization" problem, and the standard fix is not a bigger
# search, it's a noise-robust objective. Every objective evaluation now
# trains N_SEEDS models (same config, different seed) and returns the MEAN
# val MAE across seeds; the winning config's final retrain is likewise an
# N_SEEDS-model ensemble (average of raw-unit test predictions) rather than
# a single retrain, since a single retrain reproducing a search run's
# quality is exactly what failed last time (h20's winning trial: ~1.78MW
# val MAE during search vs. a near-untrained checkpoint on retrain).
# N_TRIALS is cut from 25 to keep total search compute roughly unchanged
# (N_TRIALS * N_SEEDS ~= the old N_TRIALS * 1).
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
N_TRIALS = 8        # per horizon -- N_TRIALS * N_SEEDS ~= old N_TRIALS * 1
N_SEEDS = 3         # models trained per config (search AND final ensemble)
SEARCH_EPOCHS = 60  # smaller budget while searching trials
SEARCH_PATIENCE = 6
FINAL_EPOCHS = 150  # TCNModel default, used only for the winning config
FINAL_PATIENCE = 10


def _train_seed(params, seed, X_train, delta_train, epochs, patience):
    model = TCNModel(
        num_channels=[params["width"]] * params["depth"],
        kernel_size=params["kernel_size"],
        dropout=params["dropout"],
        learning_rate=params["learning_rate"],
        weight_decay=params["weight_decay"],
        batch_size=params["batch_size"],
        epochs=epochs,
        patience=patience,
        val_ratio=VAL_RATIO,
        seed=seed,
    )
    model.train(X_train, delta_train, verbose=False)
    return model


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
        params = dict(
            depth=trial.suggest_int("depth", 3, 6),
            width=trial.suggest_categorical("width", [16, 32, 64, 128]),
            kernel_size=trial.suggest_categorical("kernel_size", [2, 3, 5, 7]),
            dropout=trial.suggest_float("dropout", 0.0, 0.4),
            learning_rate=trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True),
            weight_decay=trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
            batch_size=trial.suggest_categorical("batch_size", [64, 128, 256]),
        )

        # Train N_SEEDS models for this config and score on the MEAN val MAE
        # across seeds, not a single run -- a single run is exactly what
        # produced two contradictory Optuna studies previously (see header
        # comment). seed_maes/std are stashed as user_attrs so a "winning"
        # config that's actually high-variance (unstable across seeds) is
        # visible in BEST_PARAMS_CSV, not hidden behind a single mean number.
        seed_maes = []
        for i in range(N_SEEDS):
            model = _train_seed(params, RANDOM_STATE + i, X_train, delta_train,
                                 SEARCH_EPOCHS, SEARCH_PATIENCE)
            delta_val_pred = model.predict(X_val)
            y_val_pred_raw = anchor_val + delta_val_pred
            seed_maes.append(float(np.mean(np.abs(y_val_raw - y_val_pred_raw))))

        trial.set_user_attr("seed_maes", seed_maes)
        trial.set_user_attr("seed_mae_std", float(np.std(seed_maes)))
        return float(np.mean(seed_maes))

    study = optuna.create_study(direction="minimize", study_name=tag)
    study.optimize(objective, n_trials=N_TRIALS)

    best_trial = study.best_trial
    print(f"\n[Optuna] Best mean val MAE across {N_SEEDS} seeds (raw units): {best_trial.value:.4f}")
    print(f"[Optuna] Best params: {best_trial.params}")
    print(f"[Optuna] Per-seed val MAE: {best_trial.user_attrs.get('seed_maes')} "
          f"(std: {best_trial.user_attrs.get('seed_mae_std'):.4f})")

    best = best_trial.params
    best_params_records.append({
        "horizon": horizon,
        "best_val_mae": best_trial.value,
        "best_val_mae_std": best_trial.user_attrs.get("seed_mae_std"),
        **best,
    })

    # Final evaluation is an N_SEEDS-model ensemble (average of raw-unit test
    # predictions), not a single retrain -- a single retrain of the winning
    # config is exactly what silently failed to reproduce search-time
    # quality before (h20: ~1.78MW val MAE during search vs. a near-untrained
    # checkpoint on retrain). Averaging predictions also gives a lower-
    # variance final result than any single seed, independent of whether
    # that instability is fully gone.
    y_test_raw = anchor_test + delta_test
    seed_test_preds = []
    seed_test_maes = []
    for i in range(N_SEEDS):
        seed = RANDOM_STATE + i
        seed_model = _train_seed(best, seed, X_train, delta_train, FINAL_EPOCHS, FINAL_PATIENCE)
        seed_delta_pred = seed_model.predict(X_test)
        seed_test_preds.append(seed_delta_pred)
        seed_test_maes.append(float(np.mean(np.abs(delta_test - seed_delta_pred))))
        seed_model.save(os.path.join(MODEL_DIR, f"{tag}_seed{i}.pt"))

    print(f"[Optuna] Final-retrain per-seed test delta-MAE: {seed_test_maes}")

    delta_pred = np.mean(seed_test_preds, axis=0)
    y_pred_raw = anchor_test + delta_pred

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
