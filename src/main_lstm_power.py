import os
import pandas as pd

from experiments import run_lstm_experiment
from config import WINDOW_SIZE, FORECAST_HORIZONS

# ---------------------------------------------------------------------------
# Runs direct-forecast LSTM (one model per ship per horizon) for all 3 ships
# on the POWER target (Consumer_Total_ShaftPower). Mirrors main_lstm.py, but
# reads/writes the *_power directories so it never touches the fuel-target
# pipeline's data_clean/, eda_output/, models_saved/, predictions_cache/.
#
# Requires data_clean_power/*.parquet to already exist (produced by
# main_power.py). Does NOT re-download or re-preprocess raw data.
# ---------------------------------------------------------------------------

EDA_DIR = "eda_output_power"
DATA_DIR = "data_clean_power"
LSTM_PLOT_DIR = os.path.join(EDA_DIR, "lstm_plots")
LSTM_MODEL_DIR = os.path.join("models_saved_power", "lstm")
LSTM_PRED_DIR = os.path.join("predictions_cache_power", "lstm")

os.makedirs(LSTM_PLOT_DIR, exist_ok=True)
os.makedirs(LSTM_MODEL_DIR, exist_ok=True)

cleaned_datasets = {
    "Poseidon": pd.read_parquet(os.path.join(DATA_DIR, "poseidon_clean.parquet")),
    "Triton": pd.read_parquet(os.path.join(DATA_DIR, "triton_clean.parquet")),
    "Ceto": pd.read_parquet(os.path.join(DATA_DIR, "ceto_clean.parquet")),
}

# Starting points carried over from the fuel-target LSTM tuning (see
# main_lstm.py) -- Triton/Ceto are far smaller/noisier than Poseidon and
# overfit faster with the 128/2 default, so they start smaller with more
# patience. The power target has a different scale/distribution than fuel
# (see EDA stats: Ceto is heavily right-skewed), so re-check these once
# results come back rather than assuming they transfer as-is.
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
    unit_scale=1e6,
    unit_label="MW",
)

print("\nDone. Models saved to:", LSTM_MODEL_DIR)
print(lstm_results_df)
