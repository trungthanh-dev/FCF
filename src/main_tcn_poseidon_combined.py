import os
import pandas as pd

from experiments import run_tcn_experiment
from config import WINDOW_SIZE

# ---------------------------------------------------------------------------
# Follow-up to main_tcn_poseidon_dtw.py: pure DTW-based checkpoint selection
# improved DTW/MAE at horizon 10 but made RMSE/R2 noticeably worse (see
# eda_output_power/tcn_poseidon_dtw_results.csv vs the Huber baseline) --
# selecting purely on DTW let a few large-error points slide as long as
# local phase alignment improved. This tries a blended selection criterion
# instead: select_metric = (1 - dtw_weight) * val_MAE + dtw_weight * val_DTW,
# so a checkpoint has to improve on both fronts together, not trade one for
# the other.
#
# Writes to its own results/model/prediction paths -- never overwrites the
# Huber baseline (main_tcn_power.py) or the pure-DTW run
# (main_tcn_poseidon_dtw.py); all three need to stay on disk to compare.
# ---------------------------------------------------------------------------

EDA_DIR = "eda_output_power"
DATA_DIR = "data_clean_power"
TCN_PLOT_DIR = os.path.join(EDA_DIR, "tcn_poseidon_combined_plots")
TCN_MODEL_DIR = os.path.join("models_saved_power", "tcn_poseidon_combined")
TCN_PRED_DIR = os.path.join("predictions_cache_power", "tcn_poseidon_combined")

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
    results_csv_path=os.path.join(EDA_DIR, "tcn_poseidon_combined_results.csv"),
    model_dir=TCN_MODEL_DIR,
    predictions_dir=TCN_PRED_DIR,
    use_cache=False,
    per_ship_params={
        "Poseidon": {
            "early_stop_metric": "combined",
            "dtw_window": 10,
            "dtw_weight": 0.5,
        },
    },
    unit_scale=1e6,
    unit_label="MW",
)

print("\nDone. Compare against:")
print("  eda_output_power/tcn_results.csv               (Huber baseline)")
print("  eda_output_power/tcn_poseidon_dtw_results.csv   (pure DTW selection)")
print(tcn_results_df)
