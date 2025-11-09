# France Air Quality Index (AQI) Forecasting - MLOps Project

## Overview

This project is an end-to-end MLOps system for forecasting the European Air Quality Index (AQI) across 13 regions in France for the next 12 hours (+1h, +2h, ..., +12h). The system retrieves real-time air quality data from the Open-Meteo Air Quality API, stores it in Hopsworks feature stores, trains machine learning models, and serves predictions through a Streamlit dashboard.

## Features

- **🌍 Multi-Region Forecasting**: Predicts AQI for 13 French regions simultaneously
- **⏱️ Multi-Horizon Predictions**: Generates forecasts for 12 time steps ahead (1h to 12h)
- **🔄 Automated Pipeline**: Scheduled data fetching and prediction updates via GitHub Actions
- **📊 Interactive Dashboard**: Real-time visualization and evaluation through Streamlit
- **🏗️ MLOps Infrastructure**: Feature store, model registry, and deployment managed through Hopsworks
- **📈 Model Evaluation**: Comprehensive metrics and performance tracking

## Architecture

### Data Pipeline

1. **Data Ingestion**: Air quality data is retrieved from the [Open-Meteo Air Quality API](https://open-meteo.com/en/docs/air-quality-api) using the `openmeteo-requests` library
2. **Feature Store**: Data is stored in Hopsworks feature groups:
   - `france_regions_coordinates`: Region metadata (coordinates, names)
   - `aqi_data_france`: Historical and real-time AQI observations
   - `predictions`: Model predictions for all 13 regions
3. **Feature Engineering**: Lag features are created from historical AQI values (lags: 1, 2, 3, 4, 5, 6, 23, 24, 25 hours)

### Machine Learning Pipeline

1. **Model Training**: XGBoost-based Multi-Output Regressor trained on historical data
2. **Model Registry**: Trained models are versioned and stored in Hopsworks Model Registry
3. **Model Deployment**: Models are deployed via Hopsworks Model Serving for batch predictions
4. **Prediction Pipeline**: Automated predictions are generated hourly and stored in the feature store

### MLOps Workflow

- **Automated Data Fetching**: GitHub Actions workflow runs every hour to fetch latest AQI data
- **Automated Predictions**: Predictions are generated automatically after data updates
- **Model Retraining**: Manual process for retraining models with updated data

## Technology Stack

### Core Technologies
- **Python 3.12.4**: Primary programming language
- **XGBoost**: Gradient boosting framework for time series forecasting
- **scikit-learn**: Machine learning utilities (MultiOutputRegressor)
- **Pandas & NumPy**: Data manipulation and numerical computations

### MLOps & Infrastructure
- **Hopsworks**: Feature store, model registry, and model serving
- **GitHub Actions**: CI/CD pipeline for automated data fetching and predictions
- **Open-Meteo API**: Air quality data source

### Visualization & Frontend
- **Streamlit**: Interactive web dashboard
- **Plotly**: Interactive charts and visualizations
- **Folium**: Interactive map visualizations

### Data Processing
- **openmeteo-requests**: API client for Open-Meteo
- **requests-cache**: HTTP caching for API requests
- **retry-requests**: Retry logic for robust API calls
- **pyarrow**: Efficient data serialization

## Data Sources

### Open-Meteo Air Quality API

The project uses the [Open-Meteo Air Quality API](https://open-meteo.com/en/docs/air-quality-api) to retrieve European Air Quality Index data. The API provides:

- **European AQI**: Air quality index based on European Environment Agency (EEA) standards
- **Spatial Resolution**: ~11 km (CAMS European Air Quality Forecast)
- **Temporal Resolution**: Hourly data
- **Coverage**: 13 regions in France
- **Update Frequency**: Every 24 hours with 4-day forecasts

The European AQI ranges from:
- **0-20**: Good
- **20-40**: Fair
- **40-60**: Moderate
- **60-80**: Poor
- **80-100**: Very Poor
- **100+**: Extremely Poor

## Installation

### Prerequisites

- Python 3.12.4 or higher
- Hopsworks account and API key
- Git (for GitHub Actions workflow)

### Setup

1. **Clone the repository**:
```bash
git clone <repository-url>
cd MLOps_project
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Configure Hopsworks**:
   - Set up your Hopsworks API key as an environment variable:
   ```bash
   export HOPSWORKS_API_KEY=your_api_key_here
   ```
   - Or create a `.hopsworksrc` file in your home directory with your credentials

4. **Set up GitHub Actions** (for automated pipeline):
   - Add `HOPSWORKS_API_KEY` to your GitHub repository secrets
   - The workflow will automatically run on schedule (every hour)

## Usage

### Running the Streamlit Dashboard

Start the interactive dashboard to visualize forecasts and evaluate model performance:

```bash
streamlit run app.py
```

The dashboard will open at `http://localhost:8501`. For detailed information about the dashboard features, see [STREAMLIT_README.md](STREAMLIT_README.md).

### Training a Model

Train a new model using historical data:

```bash
python src/train_model.py
```

This script will:
1. Load data from Hopsworks feature store
2. Create features with lag variables
3. Train an XGBoost Multi-Output Regressor
4. Evaluate performance on test set
5. Save the model to Hopsworks Model Registry

### Fetching Data Manually

Fetch the latest air quality data:

```bash
python src/fetch_data.py
```

### Generating Predictions Manually

Generate predictions using the deployed model:

```bash
python src/predict.py
```

### Automated Pipeline

The project includes a GitHub Actions workflow (`.github/workflows/fetch_and_predict.yml`) that:
- Runs every hour (at 2 minutes past the hour)
- Fetches the latest AQI data from Open-Meteo API
- Generates predictions for all regions
- Stores predictions in Hopsworks feature store

To trigger manually, use the "workflow_dispatch" option in GitHub Actions.

## Project Structure

```
MLOps_project/
├── .github/
│   └── workflows/
│       └── fetch_and_predict.yml    # GitHub Actions workflow
├── air_quality_data/                # Historical air quality data (parquet files)
├── weather_data/                    # Weather data files
├── models/                          # Local model files
│   └── model.joblib
├── src/
│   ├── fetch_data.py               # Data fetching script
│   ├── fetch_and_predict.py        # Combined fetch and predict pipeline
│   ├── train_model.py              # Model training script
│   ├── predict.py                  # Prediction script
│   ├── predictor_file.py           # Model serving predictor
│   └── predictor_file_test.py      # Test predictor implementation
├── app.py                          # Streamlit dashboard application
├── requirements.txt                # Python dependencies
├── STREAMLIT_README.md            # Detailed Streamlit dashboard documentation
├── README.md                       # This file
└── france_regions_coordinates.csv  # Region metadata
```

## MLOps Components

### Hopsworks Feature Store

The project uses three main feature groups:

1. **france_regions_coordinates** (version 1)
   - Columns: `region_id`, `region`, `latitude`, `longitude`
   - Stores geographic information for 13 French regions

2. **aqi_data_france** (version 1)
   - Columns: `region_id`, `date`, `european_aqi`
   - Stores hourly AQI observations
   - Primary key: `region_id`, `date`

3. **predictions** (version 1)
   - Columns: `region_id`, `date`, `target_t_plus_1`, ..., `target_t_plus_12`
   - Stores 12-hour horizon predictions
   - Primary key: `region_id`, `date`
   - Online-enabled for real-time access

### Hopsworks Model Registry

- **Model Name**: `xgboost`
- **Model Type**: Multi-Output XGBoost Regressor
- **Metrics**: MAE, RMSE for each forecast horizon (1h to 12h)
- **Feature View**: `aqi_forecast` (version 1)

### Model Serving

Models are deployed via Hopsworks Model Serving for batch predictions. The deployment:
- Loads the latest model from the registry
- Accepts feature arrays as input
- Returns predictions for all 12 horizons
- Automatically scales based on demand

## Model Details

### Model Architecture

- **Algorithm**: XGBoost Regressor
- **Wrapper**: Multi-Output Regressor (scikit-learn)
- **Input Features**: 
  - Current AQI value
  - Lag features: 1, 2, 3, 4, 5, 6, 23, 24, 25 hours
- **Output**: 12 predictions (one for each forecast horizon: +1h to +12h)

### Training Configuration

- **Forecast Horizon**: 12 hours
- **Input Lags**: [1, 2, 3, 4, 5, 6, 23, 24, 25]
- **Training Cutoff**: 6 months before the last recorded date (configurable)
- **Evaluation Metrics**: MAE, RMSE, R2 Score

## Dashboard

The Streamlit dashboard provides:

1. **Overview Dashboard**: Interactive map of France with color-coded AQI forecasts
2. **Regional Forecast Explorer**: Detailed forecasts for selected regions
3. **Evaluation & Insights**: Model performance metrics and evaluation tools

For detailed dashboard documentation, see [STREAMLIT_README.md](STREAMLIT_README.md).

## Monitoring & Logging

- **GitHub Actions Logs**: Stored as artifacts for each workflow run
- **Hopsworks Logs**: Model training and serving logs available in Hopsworks UI
- **Application Logs**: Python logging configured throughout the codebase

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Acknowledgments

- **Open-Meteo**: For providing the Air Quality API
- **Hopsworks**: For the MLOps platform and infrastructure
- **CAMS (Copernicus Atmosphere Monitoring Service)**: For the air quality forecast data
- **European Environment Agency (EEA)**: For AQI standards and thresholds

## References

- [Open-Meteo Air Quality API Documentation](https://open-meteo.com/en/docs/air-quality-api)
- [Hopsworks Documentation](https://docs.hopsworks.ai/)
- [Streamlit Dashboard Documentation](STREAMLIT_README.md)
- [CAMS European Air Quality Forecast](https://ads.atmosphere.copernicus.eu/datasets/cams-europe-air-quality-forecasts)

## Contact

<div align="left">

📧 **Email:** [nacerkroudir@gmail.com](mailto:nacerkroudir@gmail.com)  
📱 **Phone:** [+33 6 02 54 97 94](tel:+33602549794)  
🔗 **LinkedIn:** [linkedin.com/in/nacerkroudir](https://www.linkedin.com/in/nacerkroudir)  
🌐 **Portfolio:** [nacerkroudir.github.io](https://nacerkroudir.github.io)  
✍️ **Medium:** [medium.com/@nacerkroudir](https://medium.com/@nacerkroudir)  

</div>

