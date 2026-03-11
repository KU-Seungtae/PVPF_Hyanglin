# Visualization (Denormalized Features)

Notebooks that visualize model outputs and feature importance using **original (denormalized) feature values**. Feature axes and colorbars show physical units (e.g. kW, °C, Wh/m²) for interpretability.

## Contents

| File | Description |
|------|-------------|
| `FI_SHAP.ipynb` | Feature importance and SHAP summary/dependency plots with unscaled features |
| `Machine_learning_results.ipynb` | Test-set actual vs. predicted plots (4×4 daily grid) |
| `Correlation.ipynb` | Correlation heatmap |
| `shap_dependency_points.xlsx` | Per-feature dependency points; sheet names use abbreviated form (e.g. `GHI_t-0`, `POA_t-1`) |

## Prerequisites

- Run `3. XGBoost Training` first to generate `model_artifacts.joblib` and `test_predictions.csv`.
