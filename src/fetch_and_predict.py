import numpy as np
import pandas as pd
import openmeteo_requests
import requests_cache
from retry_requests import retry
from datetime import datetime, timedelta
import hopsworks
from hopsworks.client.exceptions import RestAPIError
import logging
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



def prepare_data(df, start_date, training_config):
    """
    Get training data with configurable lags and horizon.
    Stops training data 2 months before the last recorded date.
    
    Args:
        aqi_df: Retireved observations from api
        
    Returns:
        X: Feature dataframe
        extra: extra dataframe
    """
    df = df.sort_values(['region_id', 'date']).reset_index(drop=True)
    
    # Convert date to datetime if it's string
    df['date'] = pd.to_datetime(df['date'])
    
    if len(df) == 0:
        logger.error(f"⚠️  No data")
    
    # Create lag features
    for lag in training_config['input_lags']:
        df[f'lag_{lag}'] = df['european_aqi'].shift(lag)
    
    df_to_pred = df[df['date'] > start_date].reset_index(drop=True)
    
    logger.info(f"   Data to pred: {len(df_to_pred)}")
    logger.info(f"📅 Date range: {df_to_pred['date'].min()} to {df_to_pred['date'].max()}")

    # Define feature columns (lag features)
    feature_cols = ['european_aqi']+[f'lag_{lag}' for lag in training_config['input_lags']]
    
    X = df_to_pred[feature_cols]
    extra = df_to_pred[['region_id', 'date']]
    
    return X, extra

def start_deployment(project):
    # Access the Model Serving
    ms = project.get_model_serving()
    mr = project.get_model_registry()

    # Specify the deployment name
    deployment_name = "xgboost"
    
    # Check if deployment exists
    try:
        deployment = ms.get_deployment(deployment_name)
        
        # Check if deployment actually exists (get_deployment might return None)
        if deployment is None:
            raise ValueError(f"Deployment '{deployment_name}' not found")
            
        logger.info(f"✅ Found existing deployment: {deployment_name}")
        
        # Check if deployment is stopped
        if deployment.is_stopped():
            logger.info("🔄 Deployment is stopped, starting it...")
            deployment.start(await_running=300)
            logger.info("✅ Deployment started successfully")
        else:
            logger.info("✅ Deployment is already running")
            
    except (ValueError, Exception) as e:
        # Deployment doesn't exist, create it
        logger.info(f"⚠️  Deployment '{deployment_name}' not found. Creating new deployment...")
        
        # Get the model from registry
        model = mr.get_model("xgboost", version=1)
        logger.info("✅ Model retrieved from registry")
        
        # Create predictor with script file
        predictor = ms.create_predictor(model, script_file="src/predictor_file.py")
        logger.info("✅ Predictor created from predictor_file.py")
        
        # Create deployment with predictor
        deployment = ms.create_deployment(predictor, name=deployment_name)
        deployment.save()
        logger.info("✅ Deployment created and saved")
        
        # Start the deployment
        deployment.start(await_running=300)
        logger.info("✅ Deployment started successfully")

    return deployment

def fetch_data(aqi_fg, aqi_df, regions_df, last_dates, start_hour, end_hour):
    # Get last 24 hours from aqi_df
    aqi_df['date'] = pd.to_datetime(aqi_df['date'])
    last_24h = aqi_df[aqi_df['date'] >= aqi_df['date'].max() - pd.Timedelta(hours=24)].copy()
    last_24h['date'] = last_24h['date'].astype(str)

    # Setup the Open-Meteo API client with cache and retry on error
    cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    # ========== AIR QUALITY REQUEST SETUP ==========
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    hourly_vars = ["european_aqi"]

    # Prepare lists for all coordinates
    latitudes = regions_df["latitude"].tolist()
    longitudes = regions_df["longitude"].tolist()

    # Build a single request for all regions
    params = {
        "latitude": latitudes,
        "longitude": longitudes,
        "start_hour": start_hour,
        "end_hour": end_hour,
        "hourly": hourly_vars,
    }

    logger.info(f"  Fetching air quality data for {len(latitudes)} regions...")
    logger.info(f"📅 Between {start_hour} to {end_hour}")

    # Perform one API call for all locations
    responses = openmeteo.weather_api(url, params=params)
    retrieved_data = []
    # ========== PROCESS EACH REGION ==========
    for i, response in enumerate(responses):
        region = regions_df.loc[i, "region"]
        region_id = regions_df.loc[i, "region_id"]

        hourly = response.Hourly()

        hourly_data = {
            "date": pd.date_range(
                start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
                end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
                freq=pd.Timedelta(seconds=hourly.Interval()),
                inclusive="left"
            ),
            "european_aqi": hourly.Variables(0).ValuesAsNumpy(),
        }

        hourly_df = pd.DataFrame(hourly_data)
        hourly_df['region_id'] = region_id

        retrieved_data.append(hourly_df)

    retrieved_df = pd.concat(retrieved_data, ignore_index=True)
    retrieved_df['date'] = retrieved_df['date'].dt.tz_localize(None).astype(str)
    retrieved_df['region_id'] = np.int8(retrieved_df['region_id'])

    # Filter out existing data
    filtered_df = retrieved_df.merge(last_dates, on="region_id", how="left")
    filtered_df = filtered_df[filtered_df["date"] > filtered_df["last_date"]].drop(columns=["last_date"])

    aqi_fg.insert(filtered_df)

    logger.info("✅ All regions processed and uploaded successfully!")

    combined_df = pd.concat([last_24h, filtered_df], ignore_index=True)

    return combined_df


def predict_aqi(project, pred_fg, df, last_dates, start_hour):
    # Configuration
    training_config = {
        'input_lags': [1, 2, 3, 4, 5, 6, 23, 24, 25],
        'forecast_horizon': 12,
        'random_state': 42,
    }

    X, extra = prepare_data(df, start_hour, training_config)

    deployment = start_deployment(project)

    logger.info("   Making predictions...")
    predictions = deployment.predict(data={"instances": X.values.tolist()})
    logger.info("✅ Predictions calculated successfully")
    deployment.stop(await_stopped=60)

    target_cols = [f'target_t_plus_{step}' for step in range(1, training_config['forecast_horizon'] + 1)]

    predictions_df = pd.DataFrame(predictions['predictions'], columns=target_cols)

    predictions_df = pd.concat([extra.reset_index(drop=True), predictions_df], axis=1)

    if last_dates is not None:
        predictions_df = predictions_df.merge(last_dates, on="region_id", how="left")
        predictions_df = predictions_df[predictions_df["date"] > predictions_df["last_date"]].drop(columns=["last_date"])

    predictions_df['date'] = predictions_df['date'].dt.tz_localize(None).astype(str)
    predictions_df['region_id'] = np.int8(predictions_df['region_id'])


    logger.info("   Uploading predictions to feature group...")

    pred_fg.insert(predictions_df)

    logger.info("✅ Predictions uploaded successfully")
    logger.info(f"📅 Date range: {predictions_df['date'].min()} to {predictions_df['date'].max()}")

def main():
    project = hopsworks.login()
    fs = project.get_feature_store()

    regions_fg = fs.get_feature_group(
        name='france_regions_coordinates', 
        version=1,
    )
    regions_df = regions_fg.read()

    pred_fg = fs.get_feature_group(
        name=f"predictions",
        version=1,
    )

    try:
        pred_df = pred_fg.read()

        last_dates = pred_df.loc[
            pred_df.groupby("region_id")["date"].idxmax()
        ].reset_index(drop=True)[['region_id', 'date']]
        last_dates.rename(columns={"date": "last_date"}, inplace=True)

        start_hour_pred = last_dates['last_date'].min()
    except RestAPIError:
        last_dates = None

    aqi_fg = fs.get_feature_group(
        name='aqi_data_france', 
        version=1,
    )

    aqi_df = aqi_fg.read()

    last_dates = aqi_df.loc[
        aqi_df.groupby("region_id")["date"].idxmax()
    ].reset_index(drop=True)[['region_id', 'date']]
    last_dates.rename(columns={"date": "last_date"}, inplace=True)

    start_hour = pd.to_datetime(last_dates['last_date'].min()).strftime("%Y-%m-%dT%H:%M") 
    end_hour = (datetime.now() + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0).strftime("%Y-%m-%dT%H:%M") 

    if end_hour > start_hour:
        df = fetch_data(aqi_fg, aqi_df, regions_df, last_dates, start_hour, end_hour)
        predict_aqi(project, pred_fg, df, last_dates, start_hour_pred)

    else:
        logger.info("  Data is up to date")


def run_with_retry(max_attempts=5):
    """Execute main function with retry logic."""
    for attempt in range(1, max_attempts + 1):
        try:
            logger.info(f"Attempt {attempt}/{max_attempts}")
            main()
            logger.info("Execution completed successfully")
            return True
        except Exception as e:
            logger.error(f"Attempt {attempt} failed with error: {e}")
            if attempt < max_attempts:
                wait_time = 2 ** attempt  # Exponential backoff
                logger.info(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                logger.error(f"All {max_attempts} attempts failed")
                raise


if __name__ == "__main__":
    run_with_retry(max_attempts=5)