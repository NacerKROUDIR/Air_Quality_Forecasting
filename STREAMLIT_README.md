# France Air Quality Forecast Dashboard - Streamlit App

## Overview

This Streamlit application provides a comprehensive dashboard for visualizing and evaluating air quality forecasts across 13 regions in France. The app connects to Hopsworks feature stores to retrieve real-time data and predictions.

## Features

### 📊 Overview Dashboard (Tab 1)

1. **🗺️ Interactive France Map**
   - Visual representation of all 13 French regions
   - Color-coded by forecasted AQI levels (green → red scale)
   - Time slider (0h → +12h) to view forecasts at different horizons
   - Click on regions to see detailed information

2. **📈 Regional Forecast Explorer**
   - Select a region to view:
     - Historical AQI values (past 24 hours)
     - 12-hour forecast line chart
     - Key metrics (current AQI, expected change, max forecast)
   - Compare multiple regions side-by-side

3. **⚠️ Alerts & Highlights**
   - Top 3 regions with worsening air quality (next 6 hours)
   - Top 3 regions expected to improve most
   - National AQI average forecast

### 🔍 Evaluation & Insights (Tab 2)

1. **User-Interactive Evaluation Panel**
   - Select prediction horizon (+1h to +12h)
   - Choose date range for evaluation
   - Select one or multiple regions

2. **📉 AQI Forecasts vs Actual Values**
   - Visualize actual aqi values.
   - Visualize predictions corresponding to the selected horizon.

3. **Performance Metrics**
   - Error metrics: MSE, RMSE, MAE
   - Performance by region (bar chart)
   - Error heatmap: regions × horizons
   - Performance trend over time

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Make sure you have Hopsworks credentials configured (either through environment variables or `.hopsworksrc` file).

## Running the App

Start the Streamlit app:
```bash
python -m streamlit run app.py
```

The app will automatically open in your browser at `http://localhost:8501`.

## Data Sources

The app connects to the following Hopsworks feature stores:

1. **france_regions_coordinates** (version 1)
   - Columns: `region_id`, `region`, `latitude`, `longitude`

2. **aqi_data_france** (version 1)
   - Columns: `region_id`, `date`, `european_aqi`
   - Hourly data

3. **predictions** or **predictions_test2** (version 1)
   - Columns: `region_id`, `date`, `target_t_plus_1`, ..., `target_t_plus_12`
   - 12-hour forecasts

## AQI Color Scheme

- 🟢 **Green (0-20)**: Very Good
- 🟢 **Light Green (20-40)**: Good
- 🟡 **Yellow (40-60)**: Moderate
- 🟠 **Orange (60-80)**: Unhealthy for Sensitive Groups
- 🔴 **Red (80-100)**: Unhealthy
- 🔴 **Dark Red (100+)**: Hazardous

## Notes

- Data is cached for 1 hour to improve performance
- Make sure your Hopsworks project has the required feature groups before running the app
