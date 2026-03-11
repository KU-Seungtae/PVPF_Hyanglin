# PV Power Data Integration

Aggregates raw PV power CSV files and merges them with meteorological data to produce a single, time-aligned dataset for model training.

## Contents

| File | Description |
|------|-------------|
| `data_integration.ipynb` | Loads daily CSVs from `Power_raw/`, applies column mapping, merges with KMA data, and exports merged CSV/Excel |
| `Power_raw/` | Directory for raw daily PV power files (e.g. `_2024-01-01.csv`, `_2024-01-02.csv`) |
| `hyanglin_power_*.csv` | Output: merged power data for the analysis period |
| `Hyanglin_dataset_merged_daytime_5_19.xlsx` | Output used by XGBoost training (daytime 05:00–19:00 only) |

## Column mapping

Raw columns are mapped to standard names: `surface_temperature`, `global_horizontal_irradiance`, `plane_of_array_irradiance`, `output_power`, etc. Exact mapping is defined in `data_integration.ipynb`.

## Prerequisites

- Raw PV power CSVs in `Power_raw/`
- KMA weather data from `1. KMA` merged into the workflow (if applicable)
