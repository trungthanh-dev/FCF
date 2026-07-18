import os
import pandas as pd

from experiments import run_seq2seq_experiment
from config import WINDOW_SIZE, FORECAST_HORIZONS

# ---------------------------------------------------------------------------
# Runs the seq2seq LSTM (one model per ship, predicts all horizons at once)
# for all 3 ships on the POWER target (Consumer_Total_ShaftPower). Mirrors
# main_seq2seq_lstm.py, but reads/writes the *_power directories.
#
# Requires data_clean_power/*.parquet to already exist (produced by
# main_power.py). Does NOT re-download or re-preprocess raw data.
# ---------------------------------------------------------------------------

EDA_DIR = "eda_output_power"
DATA_DIR = "data_clean_power"
SEQ2SEQ_MODEL_DIR = os.path.join("models_saved_power", "seq2seq")
SEQ2SEQ_PRED_DIR = os.path.join("predictions_cache_power", "seq2seq")

os.makedirs(SEQ2SEQ_MODEL_DIR, exist_ok=True)

cleaned_datasets = {
    "Poseidon": pd.read_parquet(os.path.join(DATA_DIR, "poseidon_clean.parquet")),
    "Triton": pd.read_parquet(os.path.join(DATA_DIR, "triton_clean.parquet")),
    "Ceto": pd.read_parquet(os.path.join(DATA_DIR, "ceto_clean.parquet")),
}

# Starting points carried over from the fuel-target seq2seq sweep (see
# main_seq2seq_lstm.py's comments for the tuning history) -- NOT re-verified
# for the power target. Power's distribution differs from fuel's (Ceto is
# far more right-skewed here), so treat these as a reasonable starting
# point, not a confirmed optimum -- re-sweep if results look off.
SEQ2SEQ_PARAMS_BY_SHIP = {
    "Poseidon": {
        "hidden_size": 64,
        "num_layers": 1,
        "dropout": 0.3,
        "weight_decay": 1e-4,
    },
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

seq2seq_results_df = run_seq2seq_experiment(
    cleaned_datasets,
    WINDOW_SIZE,
    FORECAST_HORIZONS,
    os.path.join(EDA_DIR, "seq2seq_results.csv"),
    model_dir=SEQ2SEQ_MODEL_DIR,
    predictions_dir=SEQ2SEQ_PRED_DIR,
    use_cache=False,
    per_ship_params=SEQ2SEQ_PARAMS_BY_SHIP,
    unit_scale=1e6,
    unit_label="MW",
)

print("\nDone. Models saved to:", SEQ2SEQ_MODEL_DIR)
print(seq2seq_results_df)
