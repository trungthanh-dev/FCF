import os
import pandas as pd

from experiments import run_lstm_experiment
from config import WINDOW_SIZE, FORECAST_HORIZONS

# ---------------------------------------------------------------------------
# Runs LSTM (direct forecasting: one model per ship per horizon) for all
# 3 ships. Uses data_clean/*.parquet produced by main.py -- does NOT
# re-download or re-preprocess raw data.
#
# Saves one model file per (ship, horizon) to LSTM_MODEL_DIR, e.g.:
#   models_saved/lstm/Poseidon_h1.pt
#   models_saved/lstm/Poseidon_h5.pt
#   models_saved/lstm/Poseidon_h10.pt
#   models_saved/lstm/Poseidon_h20.pt
#   models_saved/lstm/Triton_h1.pt ... etc.
#
# These are the files needed by speed_optimization_poseidon.py and
# anomaly_detection_poseidon.py (model.load(...)).
# ---------------------------------------------------------------------------

EDA_DIR = "eda_output"
DATA_DIR = "data_clean"
LSTM_PLOT_DIR = os.path.join(EDA_DIR, "lstm_plots")
LSTM_MODEL_DIR = os.path.join("models_saved", "lstm")
LSTM_PRED_DIR = os.path.join("predictions_cache", "lstm")

os.makedirs(LSTM_PLOT_DIR, exist_ok=True)
os.makedirs(LSTM_MODEL_DIR, exist_ok=True)

cleaned_datasets = {
    "Poseidon": pd.read_parquet(os.path.join(DATA_DIR, "poseidon_clean.parquet")),
    "Triton": pd.read_parquet(os.path.join(DATA_DIR, "triton_clean.parquet")),
    "Ceto": pd.read_parquet(os.path.join(DATA_DIR, "ceto_clean.parquet")),
}

# Triton and Ceto have far fewer samples than Poseidon (~25k and ~43k vs
# ~105k) and overfit faster, so they get a smaller model and higher
# patience. Both also use Huber loss (loss_delta) instead of MSE, since
# their targets have occasional outlier spikes that destabilize training
# under plain MSE. Poseidon keeps the defaults defined in
# run_lstm_experiment() (hidden_size=128, num_layers=2, patience=10).
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

print("\nDone. Models saved to:", LSTM_MODEL_DIR)
print(lstm_results_df)