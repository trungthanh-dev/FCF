import os
import pandas as pd

from experiments import run_seq2seq_experiment
from config import WINDOW_SIZE, FORECAST_HORIZONS

# ---------------------------------------------------------------------------
# Runs the seq2seq LSTM only. Uses the same data_clean/*.parquet produced by
# main.py, exactly like main_lstm.py does.
# ---------------------------------------------------------------------------

EDA_DIR = "eda_output"
DATA_DIR = "data_clean"
SEQ2SEQ_MODEL_DIR = os.path.join("models_saved", "seq2seq")
SEQ2SEQ_PRED_DIR = os.path.join("predictions_cache", "seq2seq")

os.makedirs(SEQ2SEQ_MODEL_DIR, exist_ok=True)

cleaned_datasets = {
    "Poseidon": pd.read_parquet(os.path.join(DATA_DIR, "poseidon_clean.parquet")),
    "Triton": pd.read_parquet(os.path.join(DATA_DIR, "triton_clean.parquet")),
    "Ceto": pd.read_parquet(os.path.join(DATA_DIR, "ceto_clean.parquet")),
}

# Same per-ship idea as main_lstm.py: Triton/Ceto have far fewer samples
# than Poseidon, so give them a smaller model and more patience. Tune
# these based on what you find, same as the direct-forecast LSTM.
SEQ2SEQ_PARAMS_BY_SHIP = {
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
)