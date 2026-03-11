# KMA Data Acquisition

Retrieves minute-level meteorological data from the Korea Meteorological Administration (KMA) API for the Hyanglin site. Required to align weather observations with PV power measurements in the data integration step.

## Contents

| File | Description |
|------|-------------|
| `KMA_API_hyanglin.ipynb` | Notebook to request and save KMA weather data for the target period |
| `secret_key.py` | Placeholder for API key (add your KMA API key; do not commit) |

## Usage

1. Obtain an API key from the [KMA Open Data Portal](https://data.kma.go.kr/).
2. Configure credentials in `secret_key.py`.
3. Run `KMA_API_hyanglin.ipynb` to fetch and save hourly/minute-level observations.

Output is used by `2. PV_Power_Data/data_integration.ipynb` to merge weather with PV power.
