"""
Diagnostic: check if LSTM predictions "collapsed" to a near-constant value.

Run this BEFORE retraining, using your existing cached predictions
(predictions_cache/lstm/*.npz), to confirm the hypothesis behind the
negative R2 on Triton/Ceto. Then run it again AFTER retraining with the
target-scaling fix in lstm.py, to check the fix actually helped.

Usage:
    python diagnose_lstm_collapse.py
"""
import numpy as np
import os

PRED_DIR = "predictions_cache/lstm"

def check(ship, horizon):
    path = os.path.join(PRED_DIR, f"{ship}_h{horizon}.npz")
    if not os.path.exists(path):
        print(f"{ship} h={horizon}: no cached predictions found at {path}")
        return
    data = np.load(path)
    y_test, y_pred = data["y_test"], data["y_pred"]

    std_ratio = y_pred.std() / (y_test.std() + 1e-12)
    print(
        f"{ship:10s} h={horizon:<3d} | "
        f"std(y_test)={y_test.std():.5f}  std(y_pred)={y_pred.std():.5f}  "
        f"ratio={std_ratio:.3f}  "
        f"{'<-- LIKELY COLLAPSED (near-constant predictions)' if std_ratio < 0.3 else ''}"
    )

if __name__ == "__main__":
    ships = ["Poseidon", "Triton", "Ceto"]
    horizons = [1, 5, 10, 20]
    for ship in ships:
        for h in horizons:
            check(ship, h)