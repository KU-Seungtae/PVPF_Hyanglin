# PV Power Prediction Pipeline (Hyanglin)

End-to-end pipeline for short-term photovoltaic (PV) power forecasting using XGBoost. Data flows from raw PV measurements and meteorological observations through integration, model training, and visualization to a deployment-ready prediction app.

## Pipeline overview

| Folder | Description |
|----------|-------------|
| **1. KMA** | Meteorological data acquisition via Korea Meteorological Administration API |
| **2. PV Power Data** | Raw PV power CSV aggregation and merge with weather data |
| **3. XGBoost Training** | Optuna hyperparameter tuning, feature selection, and model training |
| **4. Visualization** | Feature importance, SHAP (full-dataset sample, denormalized), correlation, and test-set actual vs predicted plots |
| **5. Program** | Streamlit web app and CLI prediction script |

## Execution order

1. **1. KMA** → fetch weather data (API key required)
2. **2. PV Power Data** → `data_integration.ipynb` produces merged datasets
3. **3. XGBoost Training** → `XGBoost.ipynb` trains best model and saves artifacts
4. **4. Visualization** → run notebooks for FI, SHAP, and ML result figures
5. **5. Program** → `streamlit run app.py` or `python predict_power.py` for inference

## Environment

Python 3.12. Install dependencies:

```bash
pip install -r requirements.txt
```

Requires `numpy<2.4` if using SHAP.
