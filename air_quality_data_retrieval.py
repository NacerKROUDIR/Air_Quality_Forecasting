import os
import pandas as pd
import openmeteo_requests
import requests_cache
from retry_requests import retry

# ========== SETUP ==========
# Create a folder for outputs
os.makedirs("air_quality_data", exist_ok=True)

# Setup the Open-Meteo API client with cache and retry on error
cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

# Load the CSV with all regions
regions_df = pd.read_csv("france_regions_coordinates.csv")

# ========== AIR QUALITY REQUEST SETUP ==========
url = "https://air-quality-api.open-meteo.com/v1/air-quality"
start_date = "2021-10-10"
end_date = "2025-10-10"
hourly_vars = ["pm10", "pm2_5", "ozone", "nitrogen_dioxide", "sulphur_dioxide", "european_aqi"]

# Prepare lists for all coordinates
latitudes = regions_df["Latitude"].tolist()
longitudes = regions_df["Longitude"].tolist()

# Build a single request for all regions
params = {
    "latitude": latitudes,
    "longitude": longitudes,
    "start_date": start_date,
    "end_date": end_date,
    "hourly": hourly_vars,
}

print(f"Fetching air quality data for {len(latitudes)} regions...")

# Perform one API call for all locations
responses = openmeteo.weather_api(url, params=params)

# ========== PROCESS EACH REGION ==========
for i, response in enumerate(responses):
    region = regions_df.loc[i, "Region"]

    hourly = response.Hourly()

    hourly_data = {
        "date": pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left"
        ),
        "pm10": hourly.Variables(0).ValuesAsNumpy(),
        "pm2_5": hourly.Variables(1).ValuesAsNumpy(),
        "ozone": hourly.Variables(2).ValuesAsNumpy(),
        "nitrogen_dioxide": hourly.Variables(3).ValuesAsNumpy(),
        "sulphur_dioxide": hourly.Variables(4).ValuesAsNumpy(),
        "european_aqi": hourly.Variables(5).ValuesAsNumpy(),
        "latitude": response.Latitude(),
        "longitude": response.Longitude(),
    }

    hourly_df = pd.DataFrame(hourly_data)

    # Save as compressed Parquet
    output_path = f"air_quality_data/{region.replace(' ', '_')}.parquet"
    hourly_df.to_parquet(output_path, index=False, compression="snappy")

    print(f"✅ Saved {region} data to {output_path}")

print("🎉 All regions processed and saved successfully with a single API call!")
