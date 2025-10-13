import pandas as pd
import folium

# Load the CSV file (replace with your path if different)
df_regions = pd.read_csv("france_regions_coordinates.csv")

# Initialize the map centered around France
m = folium.Map(location=[46.603354, 1.888334], zoom_start=6)

# Add a marker for each region
for _, row in df_regions.iterrows():
    folium.Marker(
        location=[row["Latitude"], row["Longitude"]],
        popup=row["Region"],
        tooltip=row["Region"]
    ).add_to(m)

# Save the map as an HTML file
m.save("france_regions_map.html")

print("Map saved as france_regions_map.html")
