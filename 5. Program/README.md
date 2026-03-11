# Prediction Program

Web interface and CLI for PV power prediction using the trained XGBoost model.

## Contents

| File | Description |
|------|-------------|
| `app.py` | Streamlit web app for interactive prediction |
| `predict_power.py` | Command-line prediction from CSV/Excel |

## Web app

```bash
streamlit run app.py
```

Run from the project root or from this folder. Requires the best model in `3. XGBoost Training/xgb_hyangrin_results/holdout/window_flatten/win_16/high_none_low_0.3/`.

## CLI

```bash
python predict_power.py                      # usage and built-in example
python predict_power.py input.csv            # predict from CSV
python predict_power.py input.xlsx           # predict from Excel
python predict_power.py --example            # run with example scenario
```

Input must include the four features (surface temperature, GHI, POA irradiance, relative humidity) with a window of at least 16 rows for the model’s window-flatten format.
