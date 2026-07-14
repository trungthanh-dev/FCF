"""
Fuel Consumption Forecasting Dashboard
---------------------------------------
Run with: streamlit run dashboard_app.py

Assumes this file sits in your project's `src/` folder, next to:
  - config.py, feature_config.py, dataset.py, window.py
  - models/random_forest.py
  - data_clean/{ship}_clean.parquet
  - eda_output/rf_plots/{Ship}_h{horizon}.pkl
    (main.py currently passes model_dir=RF_PLOT_DIR when training RF,
    so trained models end up here rather than in models_saved/random_forest/)

Shows:
  1. Baseline forecast (using the ship's real recent history) across all horizons.
  2. A "what-if" scenario: override one or more operating variables
     (speed, ocean current velocity, wind speed) and see how the forecast
     changes at every horizon.

Why more than just speed: for ships like Ceto, which spend time in
dynamic-positioning mode (holding position against current/wind rather
than moving), speed is often near zero and is NOT the main driver of
fuel use. Letting the user vary current/wind instead makes the what-if
tool meaningful for that operating mode too.
"""

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from config import WINDOW_SIZE, FORECAST_HORIZONS
from feature_config import get_features

DATA_DIR = "data_clean"
# NOTE: main.py's run_random_forest_experiment() call actually passes
# model_dir=RF_PLOT_DIR (eda_output/rf_plots), not the RF_MODEL_DIR
# variable it defines but never uses. Models are saved there, not in
# models_saved/random_forest/. Fix this here to match, or better, fix
# main.py's model_dir=RF_PLOT_DIR -> model_dir=RF_MODEL_DIR and retrain.
MODEL_DIR = os.path.join("eda_output", "rf_plots")

SHIPS = ["Poseidon", "Triton", "Ceto"]

# Candidate what-if variables: (display label, list of possible column
# names to try, since exact naming can vary slightly by ship/dataset).
# Only variables whose column actually exists for the selected ship will
# show a slider.
WHATIF_CANDIDATES = [
    ("Ship speed", ["Ship_SpeedThroughWater", "Ship_SpeedOverGround"]),
    ("Ocean current velocity", ["Weather_OceanCurrentVelocity"]),
    ("Wind speed", ["Weather_WindSpeed10M"]),
]


@st.cache_data
def load_ship_data(ship: str) -> pd.DataFrame:
    path = os.path.join(DATA_DIR, f"{ship.lower()}_clean.parquet")
    return pd.read_parquet(path)


@st.cache_resource
def load_rf_model(ship: str, horizon: int):
    path = os.path.join(MODEL_DIR, f"{ship}_h{horizon}.pkl")
    return joblib.load(path)


def find_column(df, candidates):
    for col in candidates:
        if col in df.columns:
            return col
    return None


def predict_all_horizons(df, ship, overrides=None):
    """
    Build the most recent WINDOW_SIZE-row window, optionally override one
    or more columns for every row in that window (a simple what-if
    scenario: "if the ship had experienced these conditions for the
    recent history"), then predict fuel consumption at every horizon.
    """
    features = get_features(df)
    window_df = df[features].tail(WINDOW_SIZE).copy()

    if overrides:
        for col, value in overrides.items():
            if col in window_df.columns:
                window_df[col] = value

    X = window_df.values.reshape(1, -1)

    rows = []
    for horizon in FORECAST_HORIZONS:
        model = load_rf_model(ship, horizon)
        pred = model.predict(X)[0]
        rows.append({"horizon": horizon, "predicted_fuel": pred})

    return pd.DataFrame(rows)


def main():
    st.set_page_config(page_title="Fuel Forecast Dashboard", layout="wide")
    st.title("Ship Fuel Consumption Forecast")
    st.caption(
        "Random Forest forecasts across horizons, with a multi-variable what-if scenario."
    )

    ship = st.sidebar.selectbox("Ship", SHIPS)
    df = load_ship_data(ship)

    st.subheader(f"{ship} — recent operating window")
    st.dataframe(df[get_features(df)].tail(WINDOW_SIZE), height=200)

    # --- Baseline forecast (no override) ---
    baseline = predict_all_horizons(df, ship, overrides=None)
    baseline = baseline.rename(columns={"predicted_fuel": "Baseline"})

    # --- Build what-if sliders for every variable available on this ship ---
    st.sidebar.subheader("What-if: change operating conditions")

    overrides = {}
    slider_info = []  # (label, column, current_avg, new_value)

    for label, candidates in WHATIF_CANDIDATES:
        col = find_column(df, candidates)
        if col is None:
            continue
        current_avg = float(df[col].tail(WINDOW_SIZE).mean())
        col_min = float(df[col].min())
        col_max = float(df[col].max())
        if col_min == col_max:
            continue  # no meaningful range to slide over

        new_value = st.sidebar.slider(
            f"{label} ({col}) — recent avg = {current_avg:.2f}",
            min_value=col_min,
            max_value=col_max,
            value=current_avg,
            step=(col_max - col_min) / 100 or 0.1,
        )
        overrides[col] = new_value
        slider_info.append((label, col, current_avg, new_value))

    if not slider_info:
        st.warning(
            "No known what-if columns (speed / current / wind) were found "
            "for this ship. Check WHATIF_CANDIDATES in this script against "
            "your data_clean parquet's actual column names."
        )
        st.subheader("Baseline forecast")
        st.line_chart(baseline.set_index("horizon"))
        st.dataframe(baseline)
        return

    # --- What-if forecast using all adjusted sliders together ---
    whatif = predict_all_horizons(df, ship, overrides=overrides)
    whatif = whatif.rename(columns={"predicted_fuel": "What-if"})

    merged = baseline.merge(whatif, on="horizon")
    merged["Change (%)"] = (
        (merged["What-if"] - merged["Baseline"]) / merged["Baseline"] * 100
    )

    st.subheader("Forecast: baseline vs. what-if")
    st.line_chart(merged.set_index("horizon")[["Baseline", "What-if"]])
    st.dataframe(merged.style.format({
        "Baseline": "{:.4f}",
        "What-if": "{:.4f}",
        "Change (%)": "{:+.1f}%",
    }))

    avg_change = merged["Change (%)"].mean()

    # Warn if the model barely reacts to the chosen overrides — this can
    # happen for ships/conditions where these variables aren't the main
    # driver of fuel use (e.g. Ceto in dynamic-positioning mode, where
    # speed is near zero and current/wind resistance dominates instead).
    if abs(avg_change) < 1.0:
        st.warning(
            "The model shows very low sensitivity to these changes for "
            f"**{ship}** (average change: {avg_change:+.1f}%). This can be "
            "expected for ships or operating modes where these variables "
            "aren't the main driver of fuel use (e.g. dynamic positioning, "
            "where current/wind resistance matters more than speed) — but "
            "it also means this what-if scenario may not be very "
            "informative here. Try adjusting a different variable."
        )
    else:
        direction = "increase" if avg_change > 0 else "decrease"
        changed_desc = ", ".join(
            f"{label} to {val:.2f} (was {avg:.2f})"
            for label, col, avg, val in slider_info
            if abs(val - avg) > 1e-9
        )
        if changed_desc:
            st.info(
                f"Setting {changed_desc} is predicted to **{direction} fuel "
                f"consumption by about {abs(avg_change):.1f}%** on average "
                f"across all horizons."
            )


if __name__ == "__main__":
    main()