import os
import numpy as np
import pandas as pd

from config import WINDOW_SIZE, TEST_SIZE, FORECAST_HORIZONS
from dataset import split_features_target
from window import create_sliding_window_delta, reshape_for_random_forest
from evalute import evaluate_regression, print_metrics
from experiments import _scale_metrics
from models.random_forest import RandomForestModel
from visualization import plot_actual_vs_predicted, plot_residuals, plot_trajectory

# ---------------------------------------------------------------------------
# Same delta-target fix as main_tcn_poseidon_delta.py (see that file for the
# full reasoning), applied to Random Forest: train on the delta
# y(now+horizon) - y(now) instead of the raw future value, removing
# Power_Lag1_t-1 as a free lunch (RF feature importance showed it at ~0.80
# for Poseidon -- the model was mostly doing persistence forecasting).
# Random Forest has no gradient descent, so none of the NaN-training risk
# that delta targets triggered in TCN/LSTM/Seq2Seq applies here.
#
# Writes to its own results/model/prediction paths so the raw-target
# baseline (main.py's run_random_forest_experiment output) stays untouched.
# ---------------------------------------------------------------------------

EDA_DIR = "eda_output_power"
DATA_DIR = "data_clean_power"
PLOT_DIR = os.path.join(EDA_DIR, "rf_poseidon_delta_plots")
MODEL_DIR = os.path.join("models_saved_power", "rf_poseidon_delta")
PRED_DIR = os.path.join("predictions_cache_power", "rf_poseidon_delta")
RESULTS_CSV = os.path.join(EDA_DIR, "rf_poseidon_delta_results.csv")

os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PRED_DIR, exist_ok=True)

UNIT_SCALE = 1e6
UNIT_LABEL = "MW"

df = pd.read_parquet(os.path.join(DATA_DIR, "poseidon_clean.parquet"))
X, y = split_features_target(df)

results = []

for horizon in FORECAST_HORIZONS:
    tag = f"Poseidon_h{horizon}_delta"
    print(f"\n[RF-delta] Poseidon | Horizon = {horizon}")

    X_window, delta_window, anchor_window = create_sliding_window_delta(
        X, y, WINDOW_SIZE, horizon
    )

    split_index = int(len(X_window) * (1 - TEST_SIZE))
    X_train, X_test = X_window[:split_index], X_window[split_index:]
    delta_train, delta_test = delta_window[:split_index], delta_window[split_index:]
    anchor_test = anchor_window[split_index:]

    print("X_train:", X_train.shape)
    print("X_test:", X_test.shape)

    model = RandomForestModel()
    model.train(X_train, delta_train)
    delta_pred = model.predict(X_test)

    y_test_raw = anchor_test + delta_test
    y_pred_raw = anchor_test + delta_pred

    model.save(os.path.join(MODEL_DIR, f"{tag}.pkl"))
    np.savez(os.path.join(PRED_DIR, f"{tag}.npz"), y_test=y_test_raw, y_pred=y_pred_raw)

    metrics = _scale_metrics(evaluate_regression(y_test_raw, y_pred_raw), UNIT_SCALE)
    print_metrics(metrics)
    metrics["ship"] = "Poseidon"
    metrics["horizon"] = horizon
    results.append(metrics)

    plot_actual_vs_predicted(
        y_test_raw, y_pred_raw,
        title=f"[RF-delta] Actual vs Predicted — Poseidon, horizon={horizon}",
        save_path=os.path.join(PLOT_DIR, f"{tag}_actual_vs_pred.png"),
    )
    plot_residuals(
        y_test_raw, y_pred_raw,
        title=f"[RF-delta] Residuals — Poseidon, horizon={horizon}",
        save_path=os.path.join(PLOT_DIR, f"{tag}_residuals.png"),
    )
    plot_trajectory(
        y_test_raw, y_pred_raw,
        title=f"[RF-delta] ShaftPower Trajectory: Actual vs. Predicted — Poseidon, horizon={horizon}",
        save_path=os.path.join(PLOT_DIR, f"{tag}_trajectory.png"),
    )

results_df = pd.DataFrame(results)[["ship", "horizon", "MAE", "RMSE", "R2", "DTW"]]
results_df.to_csv(RESULTS_CSV, index=False)

print(f"\n[RF-delta] Full results [MAE/RMSE/DTW in {UNIT_LABEL}]:")
print(results_df)
print("\nCompare against eda_output_power/rf_results.csv (Poseidon rows).")
