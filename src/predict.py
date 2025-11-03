import numpy as np
import pandas as pd
from datetime import datetime
import hopsworks
from hopsworks.client.exceptions import RestAPIError
import logging
import joblib
from hsml.transformer import Transformer
import requests

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
training_config = {
    'input_lags': [1, 2, 3, 4, 5, 6, 23, 24, 25],
    'forecast_horizon': 12,
    'random_state': 42,
}

def prepare_data(feature_view, start_date):
    """
    Get training data with configurable lags and horizon.
    Stops training data 2 months before the last recorded date.
    
    Args:
        feature_view: Hopsworks feature view object
        config: Dictionary with training configuration
        
    Returns:
        X: Feature dataframe
    """
    # Get all batch data
    df_all = feature_view.get_batch_data()

    temp_date = str(pd.to_datetime(start_date) - pd.Timedelta(days=7))

    df = (
        df_all[df_all['date'] >= temp_date]
        .sort_values(['region_id', 'date'])
        .reset_index(drop=True)
    )
    
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
    

def start_deployment():
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




project = hopsworks.login()
fs = project.get_feature_store()
mr = project.get_model_registry()

aqi_fg = fs.get_feature_group(
    name='aqi_data_france', 
    version=1,
)

feature_view = fs.get_or_create_feature_view(
    name="aqi_forecast",
    version=1,
    query=aqi_fg.select(['region_id', 'date', 'european_aqi']),
)

pred_fg = fs.get_or_create_feature_group(
    name=f"predictions_test2",
    version=1,
    online_enabled=True,
    description=f"Predictions of the Air Quality Index in 13 regions in France",
    primary_key=["region_id", "date"],
)    

try:
    pred_df = pred_fg.read()

    last_dates = pred_df.loc[
        pred_df.groupby("region_id")["date"].idxmax()
    ].reset_index(drop=True)[['region_id', 'date']]
    last_dates.rename(columns={"date": "last_date"}, inplace=True)

    start_hour = last_dates['last_date'].min()
except RestAPIError:
    df_all = feature_view.get_batch_data()
    # Convert date to datetime if it's string
    df_all['date'] = pd.to_datetime(df_all['date'])
    
    # Find the last date in the dataset
    last_dates = None
    last_hour = df_all['date'].max()
    start_hour = last_hour - pd.Timedelta(weeks=8)

X, extra = prepare_data(feature_view, start_hour)

deployment = start_deployment()

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

# predictions_df.to_csv("predictions.csv", index=False)

logger.info("   Uploading predictions to feature group...")
pred_fg.insert(predictions_df)
logger.info("✅ Predictions uploaded successfully")
logger.info(f"📅 Date range: {predictions_df['date'].min()} to {predictions_df['date'].max()}")