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
#
# Poseidon: unlike the direct-forecast LSTM, the seq2seq model stacks an
# encoder LSTM AND a decoder LSTM at the same hidden_size/num_layers --
# roughly double the recurrent capacity for the same data -- and it
# showed textbook overfitting at the 128/2 default (val_loss bottomed at
# epoch 1 then rose every epoch while train_loss kept falling). A sweep
# on Poseidon (see conversation / commit history) confirmed
# hidden_size=64, num_layers=1, dropout=0.3, weight_decay=1e-4 improves
# R2 at every horizon over the 128/2 default (h1: 0.972->0.977, h20:
# 0.843->0.862) without underfitting -- pushing further (32/1, wd=1e-3)
# starts giving worse results despite training longer before early
# stopping, so this is a real optimum, not "smaller is always better."
#
# NOTE -- GPU/CPU determinism: Seq2SeqLSTMModel seeds random/np.random/
# torch, but that does NOT make results bit-identical across devices --
# cuDNN's LSTM kernels are not deterministic vs. CPU even with a fixed
# seed. Same config retrained on GPU vs. CPU gave h1=0.976/h20=0.851
# (GPU) vs. h1=0.977/h20=0.862 (CPU) here -- same direction and
# magnitude of improvement over the 128/2 baseline, but not an exact
# match. When comparing results across runs, note which device (see
# LSTMModel/Seq2SeqLSTMModel's self.device) produced them.
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
)