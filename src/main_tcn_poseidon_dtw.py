import os
import pandas as pd

from experiments import run_tcn_experiment
from config import WINDOW_SIZE

# ---------------------------------------------------------------------------
# Focused experiment: does selecting the TCN checkpoint by validation DTW
# (instead of validation Huber loss) improve horizons 5 and 10 for Poseidon
# specifically? Poseidon's TCN DTW/MAE ratio was the lowest of any ship/model
# combo at these horizons (see eda_output_power/tcn_results.csv), suggesting
# more recoverable phase-lag error than LSTM/Seq2Seq show on the same data.
#
# Writes to its own results/model/prediction paths so it never overwrites
# the baseline Huber-selected TCN run in main_tcn_power.py -- both runs need
# to stay on disk side by side to compare.
# ---------------------------------------------------------------------------

EDA_DIR = "eda_output_power"
DATA_DIR = "data_clean_power"
TCN_PLOT_DIR = os.path.join(EDA_DIR, "tcn_poseidon_dtw_plots")
TCN_MODEL_DIR = os.path.join("models_saved_power", "tcn_poseidon_dtw")
TCN_PRED_DIR = os.path.join("predictions_cache_power", "tcn_poseidon_dtw")

os.makedirs(TCN_PLOT_DIR, exist_ok=True)
os.makedirs(TCN_MODEL_DIR, exist_ok=True)

cleaned_datasets = {
    "Poseidon": pd.read_parquet(os.path.join(DATA_DIR, "poseidon_clean.parquet")),
}

tcn_results_df = run_tcn_experiment(
    cleaned_datasets,
    WINDOW_SIZE,
    forecast_horizons=[5, 10],
    plot_dir=TCN_PLOT_DIR,
    results_csv_path=os.path.join(EDA_DIR, "tcn_poseidon_dtw_results.csv"),
    model_dir=TCN_MODEL_DIR,
    predictions_dir=TCN_PRED_DIR,
    use_cache=False,
    per_ship_params={
        "Poseidon": {
            "early_stop_metric": "dtw",
            "dtw_window": 10,
        },
    },
    unit_scale=1e6,
    unit_label="MW",
)

print("\nDone. Compare against eda_output_power/tcn_results.csv (Poseidon, h=5/h=10 rows).")
print(tcn_results_df)
