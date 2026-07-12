import os
import pandas as pd

from experiments import run_lstm_experiment
from config import WINDOW_SIZE, FORECAST_HORIZONS

# ---------------------------------------------------------------------------
# CHỈ chạy LSTM. Không load lại data thô từ HF hub, không chạy lại EDA/RF —
# dùng thẳng data_clean/*.parquet đã sinh ra từ main.py chạy ở local trước đó.
# ---------------------------------------------------------------------------

EDA_DIR = "eda_output"
DATA_DIR = "data_clean"
LSTM_PLOT_DIR = os.path.join(EDA_DIR, "lstm_plots")
LSTM_MODEL_DIR = os.path.join("models_saved", "lstm")
LSTM_PRED_DIR = os.path.join("predictions_cache", "lstm")

os.makedirs(LSTM_PLOT_DIR, exist_ok=True)

cleaned_datasets = {
    "Poseidon": pd.read_parquet(os.path.join(DATA_DIR, "poseidon_clean.parquet")),
    "Triton": pd.read_parquet(os.path.join(DATA_DIR, "triton_clean.parquet")),
    "Ceto": pd.read_parquet(os.path.join(DATA_DIR, "ceto_clean.parquet")),
}

# Triton and Ceto have far fewer samples than Poseidon (~25k and ~43k vs
# ~105k) and, per the diagnostic run, overfit fast — val_loss starts
# climbing within a handful of epochs while train_loss keeps dropping.
# Poseidon keeps the defaults defined in run_lstm_experiment(), since
# it's already performing well (R2 0.93 -> 0.76 across horizons).
#
# Triton: smaller model (hidden_size=32, num_layers=1) + higher patience
# worked well here (R2 no longer negative at h=10, much less negative at
# h=20 — see run from 2026-07-13).
#
# Ceto: the same small model (32/1) fixed the h=20 negative R2, but made
# h=5/h=10 WORSE than the previous (128/2) run. Ceto's target is
# positively skewed with occasional high-fuel-consumption spikes, and its
# speed-vs-fuel relationship is noisier than Triton's (see weekly report,
# feature importance slide) — so cutting capacity too far likely removed
# some of the real signal along with the overfitting capacity. Ceto gets
# a mid-sized model (64/1) instead of Triton's 32/1.
#
# Both Triton and Ceto also switch to Huber loss (loss_delta) instead of
# MSE: their targets have occasional outlier spikes that, under MSE,
# produce oversized gradients and destabilize training. Huber caps that
# influence while staying smooth for small errors.
LSTM_PARAMS_BY_SHIP = {
    "Triton": {
        "hidden_size": 32,
        "num_layers": 1,
        "patience": 20,
        "loss_delta": 1.0,
    },
    "Ceto": {
        "hidden_size": 64,
        "num_layers": 1,
        "patience": 15,
        "loss_delta": 1.0,
    },
}

lstm_results_df = run_lstm_experiment(
    cleaned_datasets,
    WINDOW_SIZE,
    FORECAST_HORIZONS,
    LSTM_PLOT_DIR,
    os.path.join(EDA_DIR, "lstm_results.csv"),
    model_dir=LSTM_MODEL_DIR,
    predictions_dir=LSTM_PRED_DIR,
    use_cache=False,
    per_ship_params=LSTM_PARAMS_BY_SHIP,
)