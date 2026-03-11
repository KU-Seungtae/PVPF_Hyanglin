#!/usr/bin/env python3
"""
Power prediction using the trained XGBoost model (window_flatten, win_16, 4 features).
Usage:
  python predict_power.py                    # show usage and run example
  python predict_power.py input.csv         # predict from CSV (columns: 4 features, 16+ rows)
  python predict_power.py input.xlsx       # predict from Excel
  python predict_power.py --example         # run with built-in example scenario
Output: predictions printed and optionally saved to predictions.csv
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


@dataclass
class Scalers:
    """Must match XGBoost.ipynb so joblib can load model_artifacts.joblib."""
    x_scaler: StandardScaler
    y_scaler: StandardScaler


# Paths relative to this script's directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DEFAULT_MODEL_DIR = os.path.join(
    PROJECT_ROOT,
    "3. XGBoost Training",
    "xgb_hyangrin_results",
    "holdout",
    "window_flatten",
    "win_16",
    "high_none_low_0.3",
)
WINDOW_LENGTH = 16


def load_artifacts(model_dir):
    """Load model, scalers, and feature list from the best run."""
    artifacts_path = os.path.join(model_dir, "model_artifacts.joblib")
    if not os.path.isfile(artifacts_path):
        raise FileNotFoundError(f"Model not found: {artifacts_path}")

    with open(os.path.join(model_dir, "feature_selection.json")) as f:
        fs = json.load(f)
    selected = fs["selected_features"]

    artifacts = joblib.load(artifacts_path)
    model = artifacts["model"]
    scalers = artifacts["scalers"]

    return model, scalers, selected


def build_window_input(df, feature_cols, t, window_len):
    """Build one flattened window at index t: rows [t-window_len+1, t] inclusive."""
    start = t - window_len + 1
    block = df.iloc[start : t + 1][feature_cols].to_numpy(dtype=float)
    return block.reshape(-1)


def predict_from_dataframe(df, model, scalers, feature_cols, window_len=WINDOW_LENGTH):
    """Sliding-window predictions: one prediction per valid window (from row window_len-1 onwards)."""
    n = len(df)
    if n < window_len:
        raise ValueError(f"Need at least {window_len} rows; got {n}.")

    X_list = []
    for t in range(window_len - 1, n):
        x = build_window_input(df, feature_cols, t, window_len)
        X_list.append(x)
    X = np.asarray(X_list, dtype=float)

    X_s = scalers.x_scaler.transform(X)
    y_s = model.predict(X_s)
    y_pred = scalers.y_scaler.inverse_transform(y_s.reshape(-1, 1)).ravel()
    return y_pred


def run_example(model_dir):
    """Run a single prediction with a built-in example (16 rows of 4 features)."""
    model, scalers, feature_cols = load_artifacts(model_dir)

    # Example: 16 time steps with plausible daytime values (same order as feature_selection.json)
    np.random.seed(42)
    n_steps = WINDOW_LENGTH
    example = pd.DataFrame({
        "surface_temperature": np.linspace(22, 28, n_steps) + np.random.randn(n_steps) * 0.5,
        "global_horizontal_irradiance": np.linspace(200, 600, n_steps) + np.random.randn(n_steps) * 20,
        "plane_of_array_irradiance": np.linspace(250, 650, n_steps) + np.random.randn(n_steps) * 25,
        "relative_humidity": np.linspace(55, 45, n_steps) + np.random.randn(n_steps) * 2,
    })

    preds = predict_from_dataframe(example, model, scalers, feature_cols)
    next_power = float(preds[-1])
    print("Example scenario (16 steps of 4 features):")
    print(example.to_string())
    print(f"\nPredicted next-step power (kW): {next_power:.4f}")
    return next_power


def main():
    parser = argparse.ArgumentParser(
        description="Predict PV power using trained XGBoost (window_flatten, win_16)."
    )
    parser.add_argument(
        "input_file",
        nargs="?",
        default=None,
        help="CSV or Excel file with columns: surface_temperature, global_horizontal_irradiance, plane_of_array_irradiance, relative_humidity (min 16 rows).",
    )
    parser.add_argument(
        "--example",
        action="store_true",
        help="Run with built-in example instead of a file.",
    )
    parser.add_argument(
        "--model-dir",
        default=DEFAULT_MODEL_DIR,
        help=f"Directory containing model_artifacts.joblib (default: {DEFAULT_MODEL_DIR}).",
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output CSV path for predictions (default: print only).",
    )
    args = parser.parse_args()

    if args.example or args.input_file is None:
        if args.input_file is not None and not args.example:
            parser.print_help()
            sys.exit(1)
        run_example(args.model_dir)
        return

    path = os.path.abspath(args.input_file)
    if not os.path.isfile(path):
        print(f"Error: file not found: {path}", file=sys.stderr)
        sys.exit(1)

    if path.lower().endswith(".xlsx") or path.lower().endswith(".xls"):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)

    model, scalers, feature_cols = load_artifacts(args.model_dir)
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        print(f"Error: missing columns: {missing}", file=sys.stderr)
        sys.exit(1)

    preds = predict_from_dataframe(df, model, scalers, feature_cols)

    # Align with row index: prediction at row t is for "next" step, so attach to row t
    out_df = pd.DataFrame({"row_index": np.arange(WINDOW_LENGTH - 1, len(df)), "predicted_power_kw": preds})
    print(out_df.to_string(index=False))

    if args.output:
        out_path = os.path.abspath(args.output)
        out_df.to_csv(out_path, index=False)
        print(f"\nSaved to: {out_path}")

    # If only one prediction, also print a single line
    if len(preds) == 1:
        print(f"\nPredicted next-step power (kW): {preds[0]:.4f}")


if __name__ == "__main__":
    main()
