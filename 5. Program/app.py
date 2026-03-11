#!/usr/bin/env python3
"""
PV power prediction app — browser UI.
Run: streamlit run app.py
"""

import os
import sys
from dataclasses import dataclass

# Ensure backend can be imported when run via streamlit
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


# Must match XGBoost.ipynb so joblib can load model_artifacts when this file is __main__
@dataclass
class Scalers:
    x_scaler: StandardScaler
    y_scaler: StandardScaler


from predict_power import (
    load_artifacts,
    predict_from_dataframe,
    WINDOW_LENGTH,
    DEFAULT_MODEL_DIR,
)

FEATURE_LABELS = {
    "surface_temperature": "Surface temperature (°C)",
    "global_horizontal_irradiance": "Global horizontal irradiance (Wh/m²)",
    "plane_of_array_irradiance": "Plane-of-array irradiance (Wh/m²)",
    "relative_humidity": "Relative humidity (%)",
}

FEATURE_DEFAULTS = {
    "surface_temperature": 25.0,
    "global_horizontal_irradiance": 400.0,
    "plane_of_array_irradiance": 450.0,
    "relative_humidity": 50.0,
}


def main():
    st.set_page_config(
        page_title="PV Power Prediction",
        page_icon="☀️",
        layout="centered",
        initial_sidebar_state="expanded",
    )

    # Simple custom style for a cleaner look
    st.markdown("""
        <style>
        .big-result {
            font-size: 2rem;
            font-weight: 700;
            color: #1e88e5;
            text-align: center;
            padding: 1rem;
            border-radius: 8px;
            background: #e3f2fd;
            margin: 1rem 0;
        }
        .section-title {
            font-size: 1.25rem;
            color: #37474f;
            margin-bottom: 0.5rem;
        }
        </style>
    """, unsafe_allow_html=True)

    st.title("☀️ PV Power Prediction")
    st.caption("Trained XGBoost model (window_flatten, win_16, 4 features)")

    try:
        model, scalers, feature_cols = load_artifacts(DEFAULT_MODEL_DIR)
    except FileNotFoundError as e:
        st.error(f"Model not found. {e}")
        st.stop()

    sidebar = st.sidebar
    sidebar.header("Input")
    mode = sidebar.radio(
        "Mode",
        [
            "16-step window (edit table)",
            "Quick (1 condition × 16, steady-state)",
            "Upload CSV/Excel",
        ],
        index=0,
    )

    if mode == "16-step window (edit table)":
        st.subheader("16 time steps (model input)")
        st.caption("Edit the table: exactly 16 rows, one per time step (oldest at top, newest at bottom). Row 16 is the current step; the model predicts the next step.")

        default_df = pd.DataFrame({
            name: np.linspace(FEATURE_DEFAULTS[name] * 0.9, FEATURE_DEFAULTS[name] * 1.1, WINDOW_LENGTH)
            for name in feature_cols
        })
        edited = st.data_editor(
            default_df,
            use_container_width=True,
            num_rows="fixed",
            column_config={
                name: st.column_config.NumberColumn(
                    FEATURE_LABELS.get(name, name),
                    min_value=-20.0 if "temperature" in name else 0.0,
                    max_value=1200.0 if "irradiance" in name else 100.0,
                    format="%.2f",
                )
                for name in feature_cols
            },
        )
        if st.button("Predict power", type="primary"):
            if len(edited) != WINDOW_LENGTH:
                st.warning(f"Exactly {WINDOW_LENGTH} rows required; you have {len(edited)}.")
            else:
                preds = predict_from_dataframe(edited, model, scalers, feature_cols)
                power_kw = float(preds[-1])
                st.markdown(f'<div class="big-result">Predicted power: {power_kw:.2f} kW</div>', unsafe_allow_html=True)

    elif mode == "Quick (1 condition × 16, steady-state)":
        st.subheader("Quick (steady-state)")
        st.caption("One set of values is repeated 16×. Use this only as an approximation when the last 16 steps are assumed constant.")

        cols = st.columns(2)
        inputs = {}
        for i, name in enumerate(feature_cols):
            with cols[i % 2]:
                label = FEATURE_LABELS.get(name, name)
                default = FEATURE_DEFAULTS.get(name, 0.0)
                if "temperature" in name.lower():
                    inputs[name] = st.number_input(label, value=float(default), min_value=-20.0, max_value=60.0, step=0.5, format="%.1f")
                elif "irradiance" in name.lower():
                    inputs[name] = st.number_input(label, value=float(default), min_value=0.0, max_value=1200.0, step=10.0, format="%.0f")
                else:
                    inputs[name] = st.number_input(label, value=float(default), min_value=0.0, max_value=100.0, step=1.0, format="%.1f")

        if st.button("Predict power", type="primary"):
            row = pd.DataFrame([inputs])
            df_16 = pd.concat([row] * WINDOW_LENGTH, ignore_index=True)
            preds = predict_from_dataframe(df_16, model, scalers, feature_cols)
            power_kw = float(preds[-1])
            st.markdown(f'<div class="big-result">Predicted power: {power_kw:.2f} kW</div>', unsafe_allow_html=True)
            st.caption("(Approximation: same conditions over the last 16 steps)")

    else:
        st.subheader("Upload file")
        st.caption("CSV or Excel with columns: surface_temperature, global_horizontal_irradiance, plane_of_array_irradiance, relative_humidity (min 16 rows).")

        uploaded = st.file_uploader("Choose a file", type=["csv", "xlsx", "xls"])
        if uploaded is not None:
            try:
                if uploaded.name.lower().endswith(".xlsx") or uploaded.name.lower().endswith(".xls"):
                    df = pd.read_excel(uploaded)
                else:
                    df = pd.read_csv(uploaded)
            except Exception as e:
                st.error(f"Could not read file: {e}")
            else:
                missing = [c for c in feature_cols if c not in df.columns]
                if missing:
                    st.error(f"Missing columns: {missing}")
                elif len(df) < WINDOW_LENGTH:
                    st.warning(f"Need at least {WINDOW_LENGTH} rows; file has {len(df)}.")
                else:
                    preds = predict_from_dataframe(df, model, scalers, feature_cols)
                    out = pd.DataFrame({
                        "row_index": np.arange(WINDOW_LENGTH - 1, len(df)),
                        "predicted_power_kw": preds,
                    })
                    st.markdown(f'<div class="big-result">Predicted power (last window): {float(preds[-1]):.2f} kW</div>', unsafe_allow_html=True)
                    st.dataframe(out, use_container_width=True, hide_index=True)
                    csv = out.to_csv(index=False)
                    st.download_button("Download predictions (CSV)", csv, file_name="predictions.csv", mime="text/csv")

    st.sidebar.divider()
    st.sidebar.caption("Model: holdout / window_flatten / win_16 / high_none_low_0.3")


if __name__ == "__main__":
    main()
