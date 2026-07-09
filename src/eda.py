import os
import matplotlib.pyplot as plt
import pandas as pd


def dataset_overview(name, df, log_file=None):
    lines = []
    lines.append("=" * 70)
    lines.append(name)
    lines.append("=" * 70)
    lines.append(f"Shape: {df.shape}")
    lines.append("\nData Types")
    lines.append(str(df.dtypes.value_counts()))
    lines.append("\nMemory Usage")
    lines.append(f"{df.memory_usage(deep=True).sum() / 1024**2} MB")
    lines.append("\nMissing Values")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if len(missing) == 0:
        lines.append("No missing values")
    else:
        lines.append(str(missing.sort_values(ascending=False)))
    lines.append("\n")

    text = "\n".join(lines)
    print(text)
    if log_file is not None:
        with open(log_file, "a") as f:
            f.write(text + "\n")


def target_overview(df, target, save_dir=None, name=""):
    text = []
    text.append("-" * 70)
    text.append(target)
    text.append("-" * 70)
    text.append(str(df[target].describe()))
    text.append(f"\nSkewness: {df[target].skew()}")
    text.append(f"Kurtosis: {df[target].kurt()}")
    print("\n".join(text))

    plt.figure(figsize=(8, 4))
    df[target].hist(bins=50)
    plt.title(target)
    plt.xlabel(target)
    plt.ylabel("Count")

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        fig_path = os.path.join(save_dir, f"{name}_{target}_hist.png")
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {fig_path}")
        with open(os.path.join(save_dir, f"{name}_{target}_stats.txt"), "w") as f:
            f.write("\n".join(text))
    plt.show()
    plt.close()


def target_boxplot(df, target, save_dir=None, name=""):
    plt.figure(figsize=(8, 2))
    plt.boxplot(df[target].dropna(), vert=False)
    plt.title(target)

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        fig_path = os.path.join(save_dir, f"{name}_{target}_boxplot.png")
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {fig_path}")
    plt.show()
    plt.close()