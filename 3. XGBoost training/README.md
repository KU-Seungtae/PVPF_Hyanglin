# XGBoost Model Training

Hyperparameter optimization and model training for short-term PV power prediction using Optuna and XGBoost. Supports holdout validation and various window/feature-selection configurations.

## Contents

| File | Description |
|------|-------------|
| `XGBoost.ipynb` | Main pipeline: load data, feature selection, window flatten, Optuna tuning, test evaluation, artifact saving |
| `XGB_Hyanglin_results/` | Output directory for all experiment runs |

## Configuration

- **Input**: `../2. PV_Power_Data/Hyanglin_dataset_merged_daytime_5_19.xlsx`
- **Target**: `output_power`
- **Validation**: Holdout (train: Jan–Aug, val: Sep–Oct, test: Nov–Dec)
- **Window mode**: `window_flatten` with length 16
- **Feature selection**: Correlation-based (low/high thresholds)

## Output structure

```
xgb_hyangrin_results/
  holdout/
    window_flatten/
      win_16/
        high_none_low_0.3/
          model_artifacts.joblib
          best_params.json
          feature_selection.json
          test_predictions.csv
          metrics.xlsx
          ...
```

The best configuration (by test RMSE) is saved under `holdout/window_flatten/win_16/high_none_low_0.3/` and is used by `5. Program` for inference.
