import os
import numpy as np
import pandas as pd

from config import WINDOW_SIZE, TEST_SIZE, FORECAST_HORIZONS
from dataset import split_features_target
from window import create_seq2seq_window_delta
from evalute import evaluate_regression, print_metrics
from experiments import _scale_metrics
from models.seq2seq_lstm import Seq2SeqLSTMModel
from visualization import plot_actual_vs_predicted, plot_residuals, plot_trajectory

# ---------------------------------------------------------------------------
# Same delta-target fix as main_tcn_poseidon_delta.py (see that file for the
# full reasoning), applied to the seq2seq LSTM: predict, for every horizon
# simultaneously, the delta y(now+h) - y(now) instead of the raw future
# value, using ONE shared anchor y(now) per sample (window.
# create_seq2seq_window_delta()). Seq2SeqLSTMModel now defaults
# adam_eps=1e-4 for the same reason as LSTMModel/TCNModel -- delta targets
# sit at ~0 for long stretches and triggered NaN training under PyTorch's
# 1e-8 Adam default.
#
# Hyperparameters match Poseidon's entry in main_seq2seq_power.py's
# SEQ2SEQ_PARAMS_BY_SHIP (hidden_size=64, num_layers=1, dropout=0.3,
# weight_decay=1e-4) so this is an apples-to-apples comparison against the
# raw-target baseline, isolating the effect of the delta target alone.
#
# Writes to its own results/model/prediction paths so the raw-target
# baseline stays untouched.
# ---------------------------------------------------------------------------

EDA_DIR = "eda_output_power"
DATA_DIR = "data_clean_power"
PLOT_DIR = os.path.join(EDA_DIR, "seq2seq_poseidon_delta_plots")
MODEL_DIR = os.path.join("models_saved_power", "seq2seq_poseidon_delta")
PRED_DIR = os.path.join("predictions_cache_power", "seq2seq_poseidon_delta")
RESULTS_CSV = os.path.join(EDA_DIR, "seq2seq_poseidon_delta_results.csv")

os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PRED_DIR, exist_ok=True)

UNIT_SCALE = 1e6
UNIT_LABEL = "MW"

df = pd.read_parquet(os.path.join(DATA_DIR, "poseidon_clean.parquet"))
X, y = split_features_target(df)

X_window, delta_window, anchor_window = create_seq2seq_window_delta(
    X, y, WINDOW_SIZE, FORECAST_HORIZONS
)

split_index = int(len(X_window) * (1 - TEST_SIZE))
X_train, X_test = X_window[:split_index], X_window[split_index:]
delta_train, delta_test = delta_window[:split_index], delta_window[split_index:]
anchor_test = anchor_window[split_index:]

print("X_train:", X_train.shape)
print("X_test:", X_test.shape)

model = Seq2SeqLSTMModel(
    horizons=FORECAST_HORIZONS,
    hidden_size=64,
    num_layers=1,
    dropout=0.3,
    weight_decay=1e-4,
)
model.train(X_train, delta_train)
delta_pred = model.predict(X_test)  # (samples, n_horizons)

# anchor is per-sample (shared across horizons), broadcast against the
# (samples, n_horizons) delta arrays to reconstruct raw ShaftPower.
y_test_raw = anchor_test[:, None] + delta_test
y_pred_raw = anchor_test[:, None] + delta_pred

model.save(os.path.join(MODEL_DIR, "Poseidon_seq2seq_delta.pt"))
np.savez(os.path.join(PRED_DIR, "Poseidon_seq2seq_delta.npz"), y_test=y_test_raw, y_pred=y_pred_raw)

results = []
for col_idx, horizon in enumerate(FORECAST_HORIZONS):
    tag = f"Poseidon_h{horizon}_delta"
    metrics = _scale_metrics(
        evaluate_regression(y_test_raw[:, col_idx], y_pred_raw[:, col_idx]), UNIT_SCALE
    )
    print(f"[Seq2Seq-delta] Poseidon | horizon={horizon}")
    print_metrics(metrics)
    metrics["ship"] = "Poseidon"
    metrics["horizon"] = horizon
    results.append(metrics)

    plot_actual_vs_predicted(
        y_test_raw[:, col_idx], y_pred_raw[:, col_idx],
        title=f"[Seq2Seq-delta] Actual vs Predicted — Poseidon, horizon={horizon}",
        save_path=os.path.join(PLOT_DIR, f"{tag}_actual_vs_pred.png"),
    )
    plot_residuals(
        y_test_raw[:, col_idx], y_pred_raw[:, col_idx],
        title=f"[Seq2Seq-delta] Residuals — Poseidon, horizon={horizon}",
        save_path=os.path.join(PLOT_DIR, f"{tag}_residuals.png"),
    )
    plot_trajectory(
        y_test_raw[:, col_idx], y_pred_raw[:, col_idx],
        title=f"[Seq2Seq-delta] ShaftPower Trajectory: Actual vs. Predicted — Poseidon, horizon={horizon}",
        save_path=os.path.join(PLOT_DIR, f"{tag}_trajectory.png"),
    )

results_df = pd.DataFrame(results)[["ship", "horizon", "MAE", "RMSE", "R2", "DTW"]]
results_df.to_csv(RESULTS_CSV, index=False)

print(f"\n[Seq2Seq-delta] Full results [MAE/RMSE/DTW in {UNIT_LABEL}]:")
print(results_df)
print("\nCompare against eda_output_power/seq2seq_results.csv (Poseidon rows).")
