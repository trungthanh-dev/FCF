# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

FCF (Fuel Consumption Forecasting) predicts ship fuel consumption (`Consumer_Total_MomentaryFuel`) at multiple forecast horizons (1, 5, 10, 20 steps) using time-series sensor data from three vessels — Poseidon, Triton, Ceto — from the FuelCast dataset (loaded from `hf://datasets/krohnedigital/FuelCast/...` via `pandas.read_parquet`). Three model families are compared: Random Forest, direct-forecast LSTM (one model per horizon), and a seq2seq encoder-decoder LSTM (one model per ship, predicts all horizons at once).

There is no requirements.txt/pyproject.toml — dependencies (pandas, numpy, scikit-learn, torch, streamlit, matplotlib, joblib) come from the Anaconda base environment. Python 3.12.

## Running

All entrypoint scripts live in `src/` and use **flat, non-package imports** (e.g. `from eda import dataset_overview`, not `from src.eda import ...`). Run them from inside `src/`, or ensure `src/` is on `PYTHONPATH`:

```bash
cd src
python main.py              # full pipeline: fetch raw data from HF hub, EDA, preprocess, save data_clean/*.parquet, run Random Forest
python main_lstm.py         # LSTM only — reuses existing data_clean/*.parquet, does NOT refetch/redo EDA
python main_seq2seq_lstm.py # seq2seq LSTM only — same data_clean/*.parquet
python diagnose_lstm.py     # checks predictions_cache/lstm/*.npz for "collapsed" (near-constant) predictions
streamlit run dashboard_app.py  # interactive what-if dashboard (Random Forest models only)
```

`main.py` must be run at least once (or `data_clean/*.parquet` must already exist) before `main_lstm.py` / `main_seq2seq_lstm.py` / `dashboard_app.py` will work.

Files under `src/models/*.py` append their parent directory to `sys.path` at import time (`sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))`) specifically so `from config import RANDOM_STATE` resolves regardless of the caller's working directory — preserve this pattern if adding new model modules.

There is no test suite and no linter configured in this repo.

## Pipeline architecture

The pipeline is a strict linear chain; each stage's output feeds the next:

1. **Load** (`main.py`) — raw parquet per ship from the HF hub.
2. **EDA** (`eda.py`) — dataset/target overviews, writes `eda_output/eda_report.txt` and per-ship stats/plots. Side-effecting only, doesn't transform data.
3. **Feature policy** (`feature_config.py`) — the single source of truth for what a "feature" is. `LEAKAGE_KEYWORDS` (ShaftPower, ShaftTorque, RotationSpeed, MomentaryFuel) identify propulsion-related columns that would leak the target; `get_features()` is the canonical feature list used everywhere downstream (never hand-roll a feature list elsewhere).
4. **Preprocess** (`preprocessing.py`, `preprocess()`) — applied per-ship, in this fixed order: remove leakage columns → remove all-null columns → remove non-predictive columns (e.g. FuelType) → `ffill`/`bfill` → add relative-bearing sin/cos direction features → add target lag/rolling/slope history features → re-run `ffill`/`bfill` (the lag/rolling step introduces fresh NaNs at series start). Output is cached to `data_clean/{ship}_clean.parquet`.
   - **Target history features are look-back only, by construction**: `add_target_history_features()` always calls `.shift()` before `.rolling()`, so a rolling feature at row *i* only ever sees rows `[i-window, i-1]`. When adding any new feature derived from the target, preserve this shift-before-rolling order — this is the project's core anti-leakage invariant, called out explicitly in both `feature_config.py` and `preprocessing.py` comments.
5. **Windowing** (`window.py`) — turns `(X, y)` into overlapping sliding windows of `WINDOW_SIZE` (`config.py`, default 20) past steps. `create_sliding_window()` produces one `(window, single-horizon-target)` pair per direct-forecast model call; `create_seq2seq_window()` produces one `(window, all-horizons-target)` pair for the seq2seq model. Random Forest additionally flattens the 3D window via `reshape_for_random_forest()` since it has no notion of sequence.
6. **Split** (`dataset.py`) — `time_series_split()` is a chronological (non-shuffled) train/test split — always split by index, never `train_test_split` with shuffling, since this is time-series data.
7. **Train/evaluate** (`experiments.py`) — `run_random_forest_experiment()`, `run_lstm_experiment()`, `run_seq2seq_experiment()` all follow the same shape: iterate ships × horizons, check for cached model+predictions on disk, otherwise window → split → train → predict → cache → evaluate (`evalute.py`: MAE/RMSE/R2) → plot (`visualization.py`). Results always end up as a DataFrame with columns `[ship, horizon, MAE, RMSE, R2]`, written to `eda_output/*_results.csv`, so RF/LSTM/seq2seq results are directly comparable.

## Model conventions (`src/models/`)

`RandomForestModel`, `LSTMModel`, and `Seq2SeqLSTMModel` all expose the same `train`/`predict`/`save`/`load` interface, letting `experiments.py` treat them interchangeably.

For the two LSTM models specifically:
- Both X and y are standardized (`StandardScaler`), fit **only on the training split** (never val/test) — predictions are inverse-transformed back to raw fuel units before evaluation, so metrics stay comparable across model types.
- Seq2seq's target scaler is fit jointly across all horizon columns (not one scaler per horizon), so the decoder learns one consistent output scale.
- Loss is `nn.HuberLoss` (not MSE) — several ships have positively-skewed targets with occasional high-consumption spikes; Huber caps the gradient impact of those outliers.
- Early stopping uses a **chronological** validation split (last `val_ratio` fraction of the train set, no shuffling) and restores the best-val-loss weights, not the final epoch's.
- Gradient clipping (`max_norm=1.0`) is applied every step.
- `hidden_size`/`num_layers`/`patience`/`loss_delta` are tunable per-ship via `per_ship_params` in `run_lstm_experiment()`/`run_seq2seq_experiment()` (see `main_lstm.py`/`main_seq2seq_lstm.py`) — Triton and Ceto are much smaller/noisier datasets than Poseidon and need smaller models + more patience to avoid overfitting; tune by ship, don't force shared hyperparameters.
- Model checkpoints (`.pt` via `torch.save`) bundle the fitted scalers alongside `state_dict`, since `predict()` needs the exact fit-time scaler to inverse-transform correctly.

## Known inconsistency

`dashboard_app.py` loads Random Forest models from `eda_output/rf_plots/` because `main.py`'s call to `run_random_forest_experiment()` passes `model_dir=RF_PLOT_DIR` instead of the `RF_MODEL_DIR` variable it defines but never uses (see comment in `dashboard_app.py`). If you fix `main.py` to use `RF_MODEL_DIR`, update `dashboard_app.py`'s `MODEL_DIR` (and retrain/re-save models) to match.

## Caching

`models_saved/`, `predictions_cache/`, and `data_clean/` act as on-disk caches keyed by `{ship}_h{horizon}` (or `{ship}_seq2seq` for the seq2seq model). `use_cache=True` skips training and reloads from disk if both the model and predictions files exist. LSTM/seq2seq experiments default `use_cache=False` since hyperparameters change frequently during experimentation; Random Forest defaults `use_cache=True`.
