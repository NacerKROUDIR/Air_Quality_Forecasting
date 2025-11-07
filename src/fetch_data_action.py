import numpy as np
import pandas as pd
import openmeteo_requests
import requests_cache
from retry_requests import retry
from datetime import datetime, timedelta
import hopsworks
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

project = hopsworks.login()
fs = project.get_feature_store()

regions_fg = fs.get_feature_group(
    name='france_regions_coordinates', 
    version=1,
)

aqi_fg = fs.get_feature_group(
    name='aqi_data_france', 
    version=1,
)

aqi_df = aqi_fg.read()

last_dates = aqi_df.loc[
    aqi_df.groupby("region_id")["date"].idxmax()
].reset_index(drop=True)[['region_id', 'date']]
last_dates.rename(columns={"date": "last_date"}, inplace=True)

# Setup the Open-Meteo API client with cache and retry on error
cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

# Load the CSV with all regions
regions_df = regions_fg.read()

# ========== AIR QUALITY REQUEST SETUP ==========
url = "https://air-quality-api.open-meteo.com/v1/air-quality"
hourly_vars = ["european_aqi"]
start_hour = pd.to_datetime(last_dates['last_date'].min()).strftime("%Y-%m-%dT%H:%M") 
end_hour = (datetime.now() + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0).strftime("%Y-%m-%dT%H:%M") 

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

if end_hour > start_hour:
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
else:
    logger.info("  Data is up to date")
