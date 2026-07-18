import os
import numpy as np
import pandas as pd

from dataset import split_features_target, time_series_split
from window import create_sliding_window
from evalute import evaluate_regression, print_metrics
from config import WINDOW_SIZE, FORECAST_HORIZONS
from models.tcn import TCNModel
# ---------------------------------------------------------------------------
# Runs TCN for Triton ONLY, across all 4 horizons in FORECAST_HORIZONS.
# Mirrors the structure of main_lstm.py / main_seq2seq_lstm.py, but scoped
# to one ship at a time, per the "finish one ship before moving on" plan.
# ---------------------------------------------------------------------------

EDA_DIR = "eda_output"
DATA_DIR = "data_clean"
os.makedirs(EDA_DIR, exist_ok=True)

SHIP = "Triton"
df = pd.read_parquet(os.path.join(DATA_DIR, "triton_clean.parquet"))
X, y = split_features_target(df)

# Smaller than the default (32,32,32,32) since Triton has the fewest
# samples of the three ships -- same reasoning as the smaller
# hidden_size used for Triton's LSTM/Seq2Seq models.
TCN_PARAMS = dict(
    num_channels=(16, 16, 16),
    kernel_size=3,
    dropout=0.1,
    learning_rate=5e-4,
    epochs=150,
    batch_size=128,
    val_ratio=0.1,
    patience=10,
    loss_delta=1.0,
)

results = []

for horizon in FORECAST_HORIZONS:
    print(f"\n[TCN] {SHIP} | horizon={horizon}")

    # NOTE: TCN, like LSTM/Seq2Seq, consumes the 3D window directly
    # (samples, WINDOW_SIZE, features) -- no flattening like XGBoost/RF.
    X_window, y_window = create_sliding_window(X, y, WINDOW_SIZE, horizon)
    X_train, X_test, y_train, y_test = time_series_split(X_window, y_window)

    print("X_train:", X_train.shape)
    print("X_test:", X_test.shape)

    model = TCNModel(**TCN_PARAMS)
    model.train(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = evaluate_regression(y_test, y_pred)
    print_metrics(metrics)
    corr = np.corrcoef(y_pred, y_test)[0, 1]
    print(f"corr(y_pred, y_test): {corr:.6f}")

    metrics["ship"] = SHIP
    metrics["horizon"] = horizon
    results.append(metrics)

results_df = pd.DataFrame(results)[["ship", "horizon", "MAE", "RMSE", "R2"]]
results_df.to_csv(os.path.join(EDA_DIR, "tcn_triton_results.csv"), index=False)

print("\n[TCN] Full results (Triton, all horizons):")
print(results_df)
print("y_pred std:", y_pred.std(), "y_pred mean:", y_pred.mean())
print("y_test std:", y_test.std(), "y_test mean:", y_test.mean())
