import os
import pandas as pd

from eda import dataset_overview, target_overview, target_boxplot
from feature_config import print_feature_summary, TARGET
from preprocessing import preprocess
from dataset import split_features_target, time_series_split
from window import *
from experiments import run_random_forest_experiment, run_xgboost_experiment, run_lstm_experiment
from config import (
    WINDOW_SIZE,
    FORECAST_HORIZONS,
    TEST_SIZE
)

#Output folders (power-target run — kept fully separate from the fuel-target
#pipeline's data_clean/ and eda_output/ so this run can't clobber fuel outputs)
EDA_DIR = "eda_output_power"
DATA_DIR = "data_clean_power"
RF_PLOT_DIR = os.path.join(EDA_DIR, "rf_plots")
RF_MODEL_DIR = os.path.join("models_saved_power", "random_forest")
RF_PRED_DIR = os.path.join("predictions_cache_power", "random_forst")
XGB_PLOT_DIR = os.path.join(EDA_DIR, "xgb_plots")
XGB_MODEL_DIR = os.path.join("models_saved_power", "xgboost")
XGB_PRED_DIR = os.path.join("predictions_cache_power", "xgboost")
LSTM_PLOT_DIR = os.path.join(EDA_DIR, "lstm_plots")
LSTM_MODEL_DIR = os.path.join("models_saved_power", "lstm")
LSTM_PRED_DIR = os.path.join("predictions_cache_power", "lstm")
os.makedirs(EDA_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RF_PLOT_DIR, exist_ok=True)
os.makedirs(XGB_PLOT_DIR, exist_ok=True)
os.makedirs(LSTM_PLOT_DIR, exist_ok=True)

LOG_FILE = os.path.join(EDA_DIR, "eda_report.txt")
if os.path.exists(LOG_FILE):
    os.remove(LOG_FILE)

poseidon = pd.read_parquet("hf://datasets/krohnedigital/FuelCast/CPS_Poseidon.parquet")
triton = pd.read_parquet("hf://datasets/krohnedigital/FuelCast/CPS_Triton.parquet")
ceto = df = pd.read_parquet("hf://datasets/krohnedigital/FuelCast/OSS_Ceto.parquet")

datasets = {
    "Poseidon": poseidon,
    "Triton": triton,
    "Ceto": ceto
}

#EDA
for name, df in datasets.items():
    dataset_overview(name,df, log_file = LOG_FILE)
    target_overview(df, TARGET, save_dir = EDA_DIR, name=name)
    target_boxplot(df, TARGET, save_dir = EDA_DIR, name=name)

#feature summary _ leakage remove
cleaned_datasets = {}
for name, df in datasets.items():
    print(f"\n{'=' * 70}\nFeature summary — {name}\n{'=' * 70}")
    print_feature_summary(df)

    df_clean= preprocess(df)
    cleaned_datasets[name] = df_clean

    print(f"Shape after preprocessing: {df_clean.shape}")

for name, df_clean in cleaned_datasets.items():
    out_path = os.path.join(DATA_DIR, f"{name.lower()}_clean.parquet")
    df_clean.to_parquet(out_path, index = False)
    print(f"Save")
print("Done")

rd_results_df = run_random_forest_experiment(
    cleaned_datasets,
    WINDOW_SIZE,
    FORECAST_HORIZONS,
    RF_PLOT_DIR,
    os.path.join(EDA_DIR, "rf_results.csv"),
    model_dir=RF_MODEL_DIR,
    predictions_dir=RF_PRED_DIR,
    use_cache = True,
    unit_scale=1e6,
    unit_label="MW",
)

xgb_results_df = run_xgboost_experiment(
    cleaned_datasets,
    WINDOW_SIZE,
    FORECAST_HORIZONS,
    XGB_PLOT_DIR,
    os.path.join(EDA_DIR, "xgb_results.csv"),
    model_dir=XGB_MODEL_DIR,
    predictions_dir=XGB_PRED_DIR,
    use_cache = True,
    unit_scale=1e6,
    unit_label="MW",
)
