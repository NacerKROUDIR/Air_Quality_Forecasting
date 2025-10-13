import os
import pandas as pd
import openmeteo_requests
import requests_cache
from retry_requests import retry

# ========== SETUP ==========
# Create a folder for outputs
os.makedirs("weather_data", exist_ok=True)

# Setup the Open-Meteo API client with cache and retry on error
cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

# Load the CSV with all regions
regions_df = pd.read_csv("france_regions_coordinates.csv")

# ========== WEATHER REQUEST SETUP ==========
url = "https://archive-api.open-meteo.com/v1/archive"
start_date = "2021-10-10"
end_date = "2025-10-10"
hourly_vars = [
    "temperature_2m",
    "precipitation", 
    "relative_humidity_2m",
    "surface_pressure",
    "cloud_cover",
    "direct_radiation",
    "sunshine_duration",
    "wind_speed_10m",
    "wind_speed_100m",
    "wind_gusts_10m",
    "vapour_pressure_deficit"
    ]

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

print(f"Fetching weather data for {len(latitudes)} regions...")

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
        "temperature_2m": hourly.Variables(0).ValuesAsNumpy(),
        "precipitation": hourly.Variables(1).ValuesAsNumpy(),
        "relative_humidity_2m": hourly.Variables(2).ValuesAsNumpy(),
        "surface_pressure": hourly.Variables(3).ValuesAsNumpy(),
        "cloud_cover": hourly.Variables(4).ValuesAsNumpy(),
        "direct_radiation": hourly.Variables(5).ValuesAsNumpy(),
        "sunshine_duration": hourly.Variables(6).ValuesAsNumpy(),
        "wind_speed_10m": hourly.Variables(7).ValuesAsNumpy(),
        "wind_speed_100m": hourly.Variables(8).ValuesAsNumpy(),
        "wind_gusts_10m": hourly.Variables(9).ValuesAsNumpy(),
        "vapour_pressure_deficit": hourly.Variables(10).ValuesAsNumpy(),
        "region": region,
        "latitude": response.Latitude(),
        "longitude": response.Longitude(),
    }

    hourly_df = pd.DataFrame(hourly_data)

    # Save as compressed Parquet
    output_path = f"weather_data/{region.replace(' ', '_')}.parquet"
    hourly_df.to_parquet(output_path, index=False, compression="snappy")

    print(f"✅ Saved {region} data to {output_path}")

print("🎉 All regions processed and saved successfully with a single API call!")