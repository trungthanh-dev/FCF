import os
import pandas as pd

from experiments import run_tcn_experiment
from config import WINDOW_SIZE, FORECAST_HORIZONS

# ---------------------------------------------------------------------------
# Runs TCN (direct forecasting: one model per ship per horizon) for all 3
# ships on the POWER target (Consumer_Total_ShaftPower). Unlike
# cps_triton_tcn.py (single-ship, no caching/model saving), this follows the
# same run_*_experiment convention as RF/XGBoost/LSTM/seq2seq -- caching,
# per-ship hyperparameters, results CSV directly comparable to the others.
#
# Requires data_clean_power/*.parquet to already exist (produced by
# main_power.py). Does NOT re-download or re-preprocess raw data.
# ---------------------------------------------------------------------------

EDA_DIR = "eda_output_power"
DATA_DIR = "data_clean_power"
TCN_PLOT_DIR = os.path.join(EDA_DIR, "tcn_plots")
TCN_MODEL_DIR = os.path.join("models_saved_power", "tcn")
TCN_PRED_DIR = os.path.join("predictions_cache_power", "tcn")

os.makedirs(TCN_PLOT_DIR, exist_ok=True)
os.makedirs(TCN_MODEL_DIR, exist_ok=True)

cleaned_datasets = {
    "Poseidon": pd.read_parquet(os.path.join(DATA_DIR, "poseidon_clean.parquet")),
    "Triton": pd.read_parquet(os.path.join(DATA_DIR, "triton_clean.parquet")),
    "Ceto": pd.read_parquet(os.path.join(DATA_DIR, "ceto_clean.parquet")),
}

# Same "smaller model + more patience for Triton/Ceto" idea as the other
# recurrent/conv models -- untuned for the power target specifically
# (cps_triton_tcn.py's (16,16,16) channel sweep was done on the fuel
# target's Poseidon data), so treat as a starting point.
TCN_PARAMS_BY_SHIP = {
    "Triton": {
        "num_channels": (16, 16, 16),
        "patience": 20,
    },
    "Ceto": {
        "num_channels": (16, 16, 16),
        "patience": 15,
    },
}

tcn_results_df = run_tcn_experiment(
    cleaned_datasets,
    WINDOW_SIZE,
    FORECAST_HORIZONS,
    TCN_PLOT_DIR,
    os.path.join(EDA_DIR, "tcn_results.csv"),
    model_dir=TCN_MODEL_DIR,
    predictions_dir=TCN_PRED_DIR,
    use_cache=False,
    per_ship_params=TCN_PARAMS_BY_SHIP,
    unit_scale=1e6,
    unit_label="MW",
)

print("\nDone. Models saved to:", TCN_MODEL_DIR)
print(tcn_results_df)
