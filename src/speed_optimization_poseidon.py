import os
import numpy as np
import pandas as pd

from feature_config import get_features
from config import WINDOW_SIZE, FORECAST_HORIZONS
from models.lstm import LSTMModel

# ---------------------------------------------------------------------------
# For each forecast horizon, load the already-trained Poseidon LSTM model
# and run a simple grid search over ship speed: try many candidate speed
# values (all within the range actually observed in the data), keep every
# other feature at its most recent real value, and find the speed that
# minimizes predicted fuel consumption.
#
# This assumes main_lstm.py has already been run, so models are saved at
# models_saved/lstm/{Ship}_h{horizon}.pt (per experiments.py's naming:
# tag = f"{name}_h{horizon}").
# ---------------------------------------------------------------------------

DATA_DIR = "data_clean"
LSTM_MODEL_DIR = os.path.join("models_saved", "lstm")
SHIP = "Poseidon"
SPEED_COL = "Ship_SpeedThroughWater"
N_CANDIDATES = 30  # number of speed values to try in the grid search

df = pd.read_parquet(os.path.join(DATA_DIR, f"{SHIP.lower()}_clean.parquet"))
features = get_features(df)

if SPEED_COL not in features:
    raise ValueError(
        f"{SPEED_COL} not found in features -- check the actual column name "
        f"in data_clean/{SHIP.lower()}_clean.parquet"
    )

# Speed search range: only within values actually observed historically,
# so recommendations stay physically meaningful (no extrapolation beyond
# what the ship has actually done).
speed_min = df[SPEED_COL].min()
speed_max = df[SPEED_COL].max()
candidate_speeds = np.linspace(speed_min, speed_max, N_CANDIDATES)

# The "current situation" to optimize around: the most recent WINDOW_SIZE
# rows of real data, same idea as the dashboard's baseline window.
recent_window = df[features].tail(WINDOW_SIZE).copy()
baseline_speed = recent_window[SPEED_COL].mean()

results = []

for horizon in FORECAST_HORIZONS:
    model_path = os.path.join(LSTM_MODEL_DIR, f"{SHIP}_h{horizon}.pt")

    model = LSTMModel()
    model.load(model_path)

    predicted_fuel = []
    for speed in candidate_speeds:
        modified_window = recent_window.copy()
        modified_window[SPEED_COL] = speed  # hold speed fixed across the window
        X_input = modified_window.values.reshape(1, WINDOW_SIZE, -1)
        pred = model.predict(X_input)[0]
        predicted_fuel.append(pred)

    predicted_fuel = np.array(predicted_fuel)

    # Optimize fuel PER UNIT SPEED (a proxy for fuel-per-distance, assuming
    # the ship holds this speed for some fixed duration: distance = speed
    # x time, so fuel/distance = (fuel/time) / speed). This avoids the
    # trivial "always pick the slowest speed" result that minimizing raw
    # fuel-per-time would always produce -- slower is almost always lower
    # fuel/time, but it also takes longer to cover the same distance, so
    # it isn't actually "more efficient" in the way that matters.
    # NOTE: this dataset has no real route/distance data, so this is a
    # simplified proxy, not a true per-voyage optimization -- see
    # discussion/limitations.
    specific_consumption = predicted_fuel / candidate_speeds
    best_idx = np.argmin(specific_consumption)
    best_speed = candidate_speeds[best_idx]
    best_fuel = predicted_fuel[best_idx]
    best_specific = specific_consumption[best_idx]

    # Baseline: fuel prediction using the ship's actual recent speed,
    # unmodified, for comparison against the optimized speed.
    baseline_window = recent_window.copy()  # SPEED_COL already at real values
    X_baseline = baseline_window.values.reshape(1, WINDOW_SIZE, -1)
    baseline_fuel = model.predict(X_baseline)[0]
    baseline_specific = baseline_fuel / baseline_speed

    pct_change = (best_specific - baseline_specific) / baseline_specific * 100

    print(f"[Speed Optimization] {SHIP} | horizon={horizon}")
    print(f"  baseline speed = {baseline_speed:.2f} -> fuel = {baseline_fuel:.4f}, fuel/speed = {baseline_specific:.5f}")
    print(f"  best speed     = {best_speed:.2f} -> fuel = {best_fuel:.4f}, fuel/speed = {best_specific:.5f}")
    print(f"  change in fuel/speed vs baseline: {pct_change:+.2f}%")

    results.append({
        "horizon": horizon,
        "baseline_speed": baseline_speed,
        "baseline_fuel": baseline_fuel,
        "baseline_specific": baseline_specific,
        "best_speed": best_speed,
        "best_fuel": best_fuel,
        "best_specific": best_specific,
        "pct_change": pct_change,
    })

results_df = pd.DataFrame(results)
print("\n[Speed Optimization] Full results (Poseidon, all horizons):")
print(results_df)

os.makedirs("eda_output", exist_ok=True)
results_df.to_csv("eda_output/speed_optimization_poseidon.csv", index=False)