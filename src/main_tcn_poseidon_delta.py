import os
import numpy as np
import pandas as pd

from config import WINDOW_SIZE, TEST_SIZE
from dataset import split_features_target
from window import create_sliding_window_delta
from evalute import evaluate_regression, print_metrics
from experiments import _scale_metrics
from models.tcn import TCNModel
from visualization import plot_actual_vs_predicted, plot_residuals, plot_trajectory

# ---------------------------------------------------------------------------
# Root-cause fix, not another checkpoint-selection tweak: RF feature
# importance shows Poseidon/Ceto direct-forecast models leaning on the
# target's own Power_Lag1_t-1 feature (~0.8 importance) -- i.e. they've
# mostly learned persistence forecasting ("predict the last known value"),
# which is exactly what shows up as phase lag in the DTW analysis. Neither
# early_stop_metric="dtw" nor "combined" (main_tcn_poseidon_dtw.py /
# main_tcn_poseidon_combined.py) changes what the model is incentivized to
# learn -- they only pick which already-persistence-leaning epoch to keep.
#
# This script instead changes the TARGET: train TCN to predict the DELTA
# (y(now+horizon) - y(now)) via window.create_sliding_window_delta(), so
# copying Lag1 is no longer a free lunch (it predicts delta=0). Raw-unit
# predictions are reconstructed as anchor + delta_hat before evaluation, so
# MAE/RMSE/R2/DTW stay directly comparable to the baseline run.
#
# Isolated from the DTW-checkpoint experiments on purpose (plain Huber
# early stopping, default TCN hyperparameters) so the effect of the delta
# target alone is visible, not mixed with another change. Writes to its own
# results/model/prediction paths -- baseline and the two DTW variants stay
# untouched for comparison.
# ---------------------------------------------------------------------------

EDA_DIR = "eda_output_power"
DATA_DIR = "data_clean_power"
PLOT_DIR = os.path.join(EDA_DIR, "tcn_poseidon_delta_plots")
MODEL_DIR = os.path.join("models_saved_power", "tcn_poseidon_delta")
PRED_DIR = os.path.join("predictions_cache_power", "tcn_poseidon_delta")
RESULTS_CSV = os.path.join(EDA_DIR, "tcn_poseidon_delta_results.csv")

os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PRED_DIR, exist_ok=True)

UNIT_SCALE = 1e6
UNIT_LABEL = "MW"
FORECAST_HORIZONS = [5, 10]

df = pd.read_parquet(os.path.join(DATA_DIR, "poseidon_clean.parquet"))
X, y = split_features_target(df)

results = []

for horizon in FORECAST_HORIZONS:
    tag = f"Poseidon_h{horizon}_delta"
    print(f"\n[TCN-delta] Poseidon | Horizon = {horizon}")

    X_window, delta_window, anchor_window = create_sliding_window_delta(
        X, y, WINDOW_SIZE, horizon
    )

    split_index = int(len(X_window) * (1 - TEST_SIZE))
    X_train, X_test = X_window[:split_index], X_window[split_index:]
    delta_train, delta_test = delta_window[:split_index], delta_window[split_index:]
    anchor_test = anchor_window[split_index:]

    print("X_train:", X_train.shape)
    print("X_test:", X_test.shape)

    model = TCNModel()  # default hyperparameters, same as Poseidon's baseline TCN run
    model.train(X_train, delta_train)
    delta_pred = model.predict(X_test)

    # Reconstruct raw ShaftPower units so metrics stay comparable to the
    # baseline (raw-target) run -- both "true" and "pred" go through the
    # same anchor + delta reconstruction, so this is not circular: anchor is
    # the same known y(now) in both cases, only delta_pred comes from the
    # model.
    y_test_raw = anchor_test + delta_test
    y_pred_raw = anchor_test + delta_pred

    model.save(os.path.join(MODEL_DIR, f"{tag}.pt"))
    np.savez(os.path.join(PRED_DIR, f"{tag}.npz"), y_test=y_test_raw, y_pred=y_pred_raw)

    metrics = _scale_metrics(evaluate_regression(y_test_raw, y_pred_raw), UNIT_SCALE)
    print_metrics(metrics)
    metrics["ship"] = "Poseidon"
    metrics["horizon"] = horizon
    results.append(metrics)

    plot_actual_vs_predicted(
        y_test_raw, y_pred_raw,
        title=f"[TCN-delta] Actual vs Predicted — Poseidon, horizon={horizon}",
        save_path=os.path.join(PLOT_DIR, f"{tag}_actual_vs_pred.png"),
    )
    plot_residuals(
        y_test_raw, y_pred_raw,
        title=f"[TCN-delta] Residuals — Poseidon, horizon={horizon}",
        save_path=os.path.join(PLOT_DIR, f"{tag}_residuals.png"),
    )
    plot_trajectory(
        y_test_raw, y_pred_raw,
        title=f"[TCN-delta] ShaftPower Trajectory: Actual vs. Predicted — Poseidon, horizon={horizon}",
        save_path=os.path.join(PLOT_DIR, f"{tag}_trajectory.png"),
    )

results_df = pd.DataFrame(results)[["ship", "horizon", "MAE", "RMSE", "R2", "DTW"]]
results_df.to_csv(RESULTS_CSV, index=False)

print(f"\n[TCN-delta] Full results [MAE/RMSE/DTW in {UNIT_LABEL}]:")
print(results_df)
print("\nCompare against eda_output_power/tcn_results.csv (Poseidon, h=5/h=10 rows).")
