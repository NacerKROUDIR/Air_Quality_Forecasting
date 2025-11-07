import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import hopsworks
import logging

# Configuration
training_config = {
    'region_id': 7,
    'input_lags': [1, 2, 3, 4, 5, 6, 23, 24, 25],
    'forecast_horizon': 12,
    'random_state': 42,
    'months_before_last': 6  # parameter to set the number of months to cut from training data
}

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_features_and_targets(feature_view, config):
    """
    Get training data with configurable lags and horizon.
    Stops training data 2 months before the last recorded date.
    
    Args:
        feature_view: Hopsworks feature view object
        config: Dictionary with training configuration
        
    Returns:
        DataFrame with features and targets for all specified regions
    """
    # Get all batch data
    df_all = feature_view.get_batch_data()

    df_region = df_all[df_all['region_id'] == config['region_id']].sort_values('date').reset_index(drop=True)
    
    # Convert date to datetime if it's string
    df_region['date'] = pd.to_datetime(df_region['date'])
    
    # Find the last date in the dataset
    last_date = df_region['date'].max()
    
    # Calculate cutoff date (2 months before last date)
    cutoff_date = last_date - pd.DateOffset(months=config['months_before_last'])
    
    if len(df_region) == 0:
        logger.error(f"⚠️  No training data")
    
    # Create lag features
    for lag in config['input_lags']:
        df_region[f'lag_{lag}'] = df_region['european_aqi'].shift(lag)
    
    # Create target features (forecast horizon)
    for step in range(1, config['forecast_horizon'] + 1):
        df_region[f'target_t_plus_{step}'] = df_region['european_aqi'].shift(-step)
    
    # Drop rows with NaN values (due to shifting)
    df_training = df_region[df_region['date'] <= cutoff_date].dropna().reset_index(drop=True)
    df_test = df_region[df_region['date'] > cutoff_date].dropna().reset_index(drop=True)
    
    logger.info(f"   Training records: {len(df_training)}")
    logger.info(f"   Test records: {len(df_test)}")
    logger.info(f"📅 Training date range: {df_training['date'].min()} to {df_training['date'].max()}")
    logger.info(f"📅 Test date range: {df_test['date'].min()} to {df_test['date'].max()}")
    
    return df_training, df_test


def separate_features_and_targets(df, config):
    """
    Separate features (X) and targets (Y) from the prepared dataframe.
    
    Args:
        df: DataFrame with lag features and targets
        config: Configuration dictionary
        
    Returns:
        X: Feature dataframe
        Y: Target dataframe
    """
    # Define feature columns (lag features)
    feature_cols = ['european_aqi']+[f'lag_{lag}' for lag in config['input_lags']]
    
    # Define target columns
    target_cols = [f'target_t_plus_{step}' for step in range(1, config['forecast_horizon'] + 1)]
    
    X = df[feature_cols]
    Y = df[target_cols]
    
    return X, Y


def train_model(X_train, Y_train):
    """
    Train the XGBoost multi-output model.
    
    Args:
        X_train, Y_train: Training data
        
    Returns:
        model
    """
    logger.info("🚀 Starting model training...")
    
    # Initialize XGBoost base regressor
    base_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        booster='gbtree',
        n_estimators=500,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,      # L1 regularization (Lasso)
        reg_lambda=1.0,     # L2 regularization (Ridge)
        gamma=0,            # Minimum loss reduction for split (also acts as regularization)
        max_depth=6,        # Tree depth (limits complexity)
        min_child_weight=1, # Minimum sum of instance weight in a child (regularization)
        random_state=42,
        verbosity=0         # Suppress warnings (0=silent, 1=warning, 2=info, 3=debug)
    )
    
    # Wrap it for multi-step output
    model = MultiOutputRegressor(base_model, n_jobs=-1)
    
    # Fit model
    model.fit(X_train, Y_train)
    
    logger.info("✅ Model training completed!")
    
    return model


def evaluate_forecasts(Y_test: pd.DataFrame, Y_pred: np.ndarray):
    """
    Evaluate multi-horizon AQI forecasts using common regression metrics.
    
    Parameters
    ----------
    Y_test : pd.DataFrame
        True values (n_samples x n_horizons) with DatetimeIndex.
    Y_pred : np.ndarray
        Predicted values (n_samples x n_horizons).
        
    Returns
    -------
    pd.DataFrame
        DataFrame with metrics (MAE, RMSE, R²) for each horizon and the average.
    """
    assert Y_test.shape == Y_pred.shape, "Y_test and Y_pred must have the same shape"
    
    horizons = Y_test.columns if isinstance(Y_test.columns, pd.Index) else [f"+{i+1}h" for i in range(Y_test.shape[1])]
    
    metrics = []
    for i, col in enumerate(horizons):
        y_true = Y_test.iloc[:, i]
        y_hat = Y_pred[:, i]
        
        mae = mean_absolute_error(y_true, y_hat)
        rmse = np.sqrt(mean_squared_error(y_true, y_hat))
        r2 = r2_score(y_true, y_hat)
        
        metrics.append({"Horizon": col, "MAE": mae, "RMSE": rmse, "R2": r2})
    
    # Compute averages
    avg_mae = np.mean([m["MAE"] for m in metrics])
    avg_rmse = np.mean([m["RMSE"] for m in metrics])
    avg_r2 = np.mean([m["R2"] for m in metrics])
    
    metrics.append({"Horizon": "Average", "MAE": avg_mae, "RMSE": avg_rmse, "R2": avg_r2})
    
    results = pd.DataFrame(metrics)
    return results


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

train_df, test_df = create_features_and_targets(feature_view, training_config)

X_train, Y_train = separate_features_and_targets(train_df, training_config)
X_test, Y_test = separate_features_and_targets(test_df, training_config)

model = train_model(X_train, Y_train)

# Get predictions
train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)

# Evaluate using the custom function
print("\n📈 Training Set Performance:")
train_results = evaluate_forecasts(Y_train, train_predictions)
print(train_results.to_string(index=False))

print("\n📈 Test Set Performance:")
test_results = evaluate_forecasts(Y_test, test_predictions)
print(test_results.to_string(index=False))

joblib.dump(model, "models/model.joblib")

# Convert metrics to a flat dictionary format
metrics_dict = {}
for record in test_results.to_dict(orient="records"):
    horizon = record['Horizon']
    for metric_name in ['MAE', 'RMSE', 'R2']:
        key = f"{horizon}_{metric_name}"
        metrics_dict[key] = record[metric_name]

py_model = mr.python.create_model("xgboost", metrics=metrics_dict, feature_view=feature_view)
py_model.save("models/model.joblib")