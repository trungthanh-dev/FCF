import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



def plot_actual_vs_predicted(y_true, y_pred, title, save_path):
    """Scatter plot of actual vs predicted values with a y = x reference line."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.3, s=8)

    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1, label="y = x")

    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_residuals(y_true, y_pred, title, save_path):
    """Residuals over sample order + residuals vs predicted value."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    residuals = y_true - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].plot(residuals, linewidth=0.6)
    axes[0].axhline(0, color="r", linestyle="--", linewidth=1)
    axes[0].set_xlabel("Sample index (time order)")
    axes[0].set_ylabel("Residual (actual - predicted)")
    axes[0].set_title("Residuals over time")

    axes[1].scatter(y_pred, residuals, alpha=0.3, s=8)
    axes[1].axhline(0, color="r", linestyle="--", linewidth=1)
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Residual")
    axes[1].set_title("Residuals vs predicted")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_feature_importance(importances, feature_names, title, save_path, top_n=20):
    """Horizontal bar chart of the top-N most important features."""
    importances = np.asarray(importances)
    n = min(top_n, len(importances))
    order = np.argsort(importances)[::-1][:n]

    names = [feature_names[i] for i in order]
    values = importances[order]

    fig, ax = plt.subplots(figsize=(8, max(4, 0.3 * n)))
    ax.barh(range(n), values[::-1])
    ax.set_yticks(range(n))
    ax.set_yticklabels(names[::-1], fontsize=8)
    ax.set_xlabel("Importance")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)

def plot_trajectory(y_true, y_pred, title, save_path, n_points = 300):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = min(n_points, len(y_true))
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(y_true[:n], color="blue", linewidth = 1, label = "Real")
    ax.plot(y_pred[:n], color="red",linewidth=1, label="Predict")
    ax.set_xlabel("Sample index (time order)")
    ax.set_ylabel("Fuel consumption")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)

def plot_horizon_comparison(results_df: pd.DataFrame, save_dir):
    """For each metric (MAE, RMSE, R2), plot metric vs horizon, one line per ship."""
    os.makedirs(save_dir, exist_ok=True)
    for metric in ["MAE", "RMSE", "R2"]:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for ship, group in results_df.groupby("ship"):
            group = group.sort_values("horizon")
            ax.plot(group["horizon"], group[metric], marker="o", label=ship)

        ax.set_xlabel("Forecast horizon")
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} vs horizon, by ship")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, f"horizon_comparison_{metric}.png"), dpi=150)
        plt.close(fig)