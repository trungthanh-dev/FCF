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

lstm_results_df = run_lstm_experiment(
    cleaned_datasets,
    WINDOW_SIZE,
    FORECAST_HORIZONS,
    LSTM_PLOT_DIR,
    os.path.join(EDA_DIR, "lstm_results.csv"),
    model_dir=LSTM_MODEL_DIR,
    predictions_dir=LSTM_PRED_DIR,
    use_cache=True,
)