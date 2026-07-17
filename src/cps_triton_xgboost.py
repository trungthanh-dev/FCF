import pandas
import pandas as pd
import numpy as np
import os

from pandas.core.common import random_state
from xgboost import XGBRegressor
from dataset import split_features_target, time_series_split
from window import create_sliding_window, reshape_for_random_forest
from config import WINDOW_SIZE, RANDOM_STATE, FORECAST_HORIZONS
from evalute import evaluate_regression, print_metrics


cps_triton = pd.read_parquet(os.path.join("data_clean", "triton_clean.parquet")) #Read cps_triton_processed_data
X, y = split_features_target(cps_triton)

results = []
for horizon in FORECAST_HORIZONS:
    X_window, y_window = create_sliding_window(X, y, WINDOW_SIZE, horizon)
    X_window_2d = reshape_for_random_forest(X_window)
    X_train, X_test, y_train, y_test = time_series_split(X_window_2d, y_window)
    model = XGBRegressor(n_estimators=100, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = evaluate_regression(y_test, y_pred)
    print(f"horizon = {horizon}")
    print_metrics(metrics)

    metrics["horizon"] = horizon
    results.append(metrics)










