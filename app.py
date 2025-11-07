import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import hopsworks
from sklearn.metrics import mean_squared_error, mean_absolute_error
import logging

# Configure page
st.set_page_config(
    page_title="France Air Quality Forecast Dashboard",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize session state for data caching
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'regions_df' not in st.session_state:
    st.session_state.regions_df = None
if 'aqi_df' not in st.session_state:
    st.session_state.aqi_df = None
if 'predictions_df' not in st.session_state:
    st.session_state.predictions_df = None

@st.cache_data(ttl=900)  # Cache for 15 minutes
def load_data_from_hopsworks():
    """Load data from Hopsworks feature stores"""
    try:
        project = hopsworks.login()
        fs = project.get_feature_store()
        
        # Load regions data
        regions_fg = fs.get_feature_group(
            name='france_regions_coordinates', 
            version=1,
        )
        regions_df = regions_fg.read()
        
        # Load AQI data
        aqi_fg = fs.get_feature_group(
            name='aqi_data_france', 
            version=1,
        )
        aqi_df = aqi_fg.read()
        if not aqi_df.empty:
            aqi_df['date'] = pd.to_datetime(aqi_df['date'])
        
        # Load predictions
        pred_fg = fs.get_feature_group(name='predictions', version=1)
        predictions_df = pred_fg.read()
        
        if not predictions_df.empty:
            predictions_df['date'] = pd.to_datetime(predictions_df['date'])
        
        return regions_df, aqi_df, predictions_df
    except Exception as e:
        st.error(f"Error loading data from Hopsworks: {str(e)}")
        return None, None, None

def get_aqi_color(aqi_value):
    """Get color based on AQI value"""
    if pd.isna(aqi_value):
        return '#808080'  # Gray for missing data
    if aqi_value <= 20:
        return '#00FF00'  # Green - Good
    elif aqi_value <= 40:
        return '#90EE90'  # Light Green
    elif aqi_value <= 60:
        return '#FFFF00'  # Yellow - Moderate
    elif aqi_value <= 80:
        return '#FFA500'  # Orange - Unhealthy for Sensitive Groups
    elif aqi_value <= 100:
        return '#FF6347'  # Tomato - Unhealthy
    else:
        return '#FF0000'  # Red - Hazardous

def get_aqi_category(aqi_value):
    """Get AQI category"""
    if pd.isna(aqi_value):
        return "Unknown"
    if aqi_value <= 20:
        return "🟢 Very Good"
    elif aqi_value <= 40:
        return "🟢 Good"
    elif aqi_value <= 60:
        return "🟡 Moderate"
    elif aqi_value <= 80:
        return "🟠 Unhealthy for Sensitive Groups"
    elif aqi_value <= 100:
        return "🔴 Unhealthy"
    else:
        return "🔴 Hazardous"

def _create_france_map(regions_df, aqi_values, selected_hour):
    """Create interactive France map with AQI colors"""
    # Center map on France
    m = folium.Map(
        location=[46.603354, 2.213749],
        zoom_start=6,
        tiles='OpenStreetMap',
        scrollWheelZoom=False
    )
    
    # Add markers for each region
    for _, row in regions_df.iterrows():
        region_id = row['region_id']
        region_name = row['region']
        lat = row['latitude']
        lon = row['longitude']
        
        aqi = aqi_values.get(region_id, np.nan)
        color = get_aqi_color(aqi)
        category = get_aqi_category(aqi)
        
        # Create popup text
        popup_text = f"""
        <div style='font-family: Arial; font-size: 12px; min-width: 150px;'>
            <h4 style='margin: 0 0 10px 0;'>{region_name.title().replace('_', ' ')}</h4>
            <p style='margin: 5px 0;'><strong>AQI:</strong> {f"{aqi:.1f}" if not pd.isna(aqi) else 'N/A'}</p>
            <p style='margin: 5px 0;'><strong>Status:</strong> {category}</p>
            <p style='margin: 5px 0;'><strong>Forecast Hour:</strong> +{selected_hour}h</p>
        </div>
        """
        
        folium.CircleMarker(
            location=[lat, lon],
            radius=15,
            popup=folium.Popup(popup_text, max_width=300),
            tooltip=f"{region_name.title().replace('_', ' ')}: AQI {f"{aqi:.1f}" if not pd.isna(aqi) else 'N/A'}",
            color='black',
            weight=2,
            fillColor=color,
            fillOpacity=0.7,
        ).add_to(m)
    
    return m

@st.cache_data
def calculate_metrics_for_evaluation(actual_df, pred_df, horizon, date_range=None):
    """Calculate evaluation metrics by matching predictions at time T with actuals at time T+h"""
    pred_col = f'target_t_plus_{horizon}'
    
    if pred_col not in pred_df.columns:
        return None
    
    # Filter actual and predictions by date range if provided
    actual_filtered = actual_df.copy()
    pred_filtered = pred_df.copy()
    
    if date_range:
        pred_start_date, pred_end_date = date_range

        actual_start_date = pred_start_date + pd.Timedelta(hours=horizon)
        actual_end_date = pred_end_date + pd.Timedelta(hours=horizon)

        actual_filtered = actual_filtered[
            (actual_filtered['date'] >= actual_start_date) & 
            (actual_filtered['date'] <= actual_end_date)
        ]

        pred_filtered = pred_filtered[
            (pred_filtered['date'] >= pred_start_date) & 
            (pred_filtered['date'] <= pred_end_date)
        ][['date', 'region_id', f'target_t_plus_{horizon}']]
    
    if actual_filtered.empty or pred_filtered.empty:
        return None
    
    actual_filtered.sort_values(by=['region_id', 'date'], inplace=True)
    pred_filtered.sort_values(by=['region_id', 'date'], inplace=True)

    y_true = actual_filtered['european_aqi'].values
    y_pred = pred_filtered[f'target_t_plus_{horizon}'].values

    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'n_samples': len(pred_filtered)
    }

# Main app
def main():
    st.title("🌍 France Air Quality Forecast Dashboard")
    
    # Load data
    with st.spinner("Loading data from Hopsworks..."):
        regions_df, aqi_df, predictions_df = load_data_from_hopsworks()
        
        if regions_df is None or aqi_df is None:
            st.error("Failed to load data. Please check your Hopsworks connection.")
            return
        
        st.session_state.regions_df = regions_df
        st.session_state.aqi_df = aqi_df
        st.session_state.predictions_df = predictions_df
        st.session_state.data_loaded = True
    
    # Create tabs
    tab1, tab2 = st.tabs(["📊 Overview Dashboard", "🔍 Evaluation & Insights"])
    
    with tab1:
        render_overview_dashboard(regions_df, aqi_df, predictions_df)
    
    with tab2:
        render_evaluation_tab(regions_df, aqi_df, predictions_df)

def render_interactive_map(aqi_df, regions_df, predictions_df,latest_pred_date):
    col1, col2 = st.columns([3, 1])
    
    with col1:
        selected_hour = st.slider(
            "Forecast Horizon (hours ahead)",
            min_value=0,
            max_value=12,
            value=0,
            step=1,
            help="Select how many hours into the future to view forecasts"
        )
    
    with col2:
        st.metric("Selected Forecast", f"+{selected_hour}h")
        st.caption(f"Latest prediction date: {latest_pred_date.strftime('%Y-%m-%d %H:%M')}")
    
    # Get AQI values for selected hour
    aqi_values = {}
    if selected_hour == 0:
        # Current/latest AQI
        latest_aqi = aqi_df[aqi_df['date'] == aqi_df['date'].max()]
        for _, row in latest_aqi.iterrows():
            aqi_values[row['region_id']] = row['european_aqi']
    else:
        # Forecasted AQI
        pred_col = f'target_t_plus_{selected_hour}'
        if pred_col in predictions_df.columns:
            latest_pred = predictions_df[predictions_df['date'] == latest_pred_date]
            for _, row in latest_pred.iterrows():
                aqi_values[row['region_id']] = row[pred_col]
    
    # Create and display map
    map_obj = _create_france_map(regions_df, aqi_values, selected_hour)
    map_data = st_folium(map_obj, width=1200, height=700)

def render_regional_forecast(aqi_df, regions_df, predictions_df, latest_pred_date):
    # Region selector
    region_list = regions_df['region'].str.title().str.replace('_', ' ').tolist()
    selected_region_display = st.selectbox(
        "Select a Region",
        region_list,
        index=3
    )
    
    # Get region_id from selected region
    selected_region = selected_region_display.lower().replace(' ', '_')
    region_row = regions_df[regions_df['region'].str.lower() == selected_region]
    if not region_row.empty:
        selected_region_id = region_row.iloc[0]['region_id']
    else:
        selected_region_id = 0
    
    # Single region view
    if selected_region_id is not None:
        # Get historical AQI (last 24 hours)
        historical = aqi_df[aqi_df['region_id'] == selected_region_id].sort_values('date')
        historical = historical[historical['date'] >= historical['date'].max() - pd.Timedelta(hours=72)]
        
        # Get predictions
        forecast_values = []
        forecast_dates = []
        
        if not predictions_df.empty:
            preds = predictions_df[
                (predictions_df['region_id'] == selected_region_id) &
                (predictions_df['date'] == latest_pred_date)
            ]
            
            if not preds.empty:
                for h in range(1, 13):
                    forecast_values.append(preds.iloc[0][f'target_t_plus_{h}'])
                    forecast_dates.append(latest_pred_date + pd.Timedelta(hours=h))
        
        # Create combined chart
        fig = go.Figure()
        
        # Historical data
        if not historical.empty:
            fig.add_trace(go.Scatter(
                x=historical['date'],
                y=historical['european_aqi'],
                mode='lines+markers',
                name='Historical',
                line=dict(color='blue', width=2),
                marker=dict(size=6)
            ))
        
        # Forecast data
        if forecast_dates:
            fig.add_trace(go.Scatter(
                x=forecast_dates,
                y=forecast_values,
                mode='lines+markers',
                name='Forecast',
                line=dict(color='red', width=2),
                marker=dict(size=6)
            ))
        
        fig.update_layout(
            title=f"AQI Forecast for {selected_region_display}",
            xaxis_title="Date",
            yaxis_title="European AQI",
            hovermode='x unified',
            height=400,
            template="plotly_white"
        )
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            current_aqi = historical['european_aqi'].iloc[-1] if not historical.empty else np.nan
            st.metric("Current AQI", f"{current_aqi:.1f}" if not pd.isna(current_aqi) else "N/A")
        
        with col2:
            if forecast_values:
                change_1h = forecast_values[0] - current_aqi if not pd.isna(current_aqi) and len(forecast_values) > 0 else np.nan
                st.metric(
                    "Expected AQI (1h)",
                    f"{forecast_values[0]:.1f}" if not pd.isna(change_1h) else "N/A",
                    delta=round(change_1h, 1) if not pd.isna(change_1h) else None,
                    delta_color="inverse"
                )

        with col3:
            if forecast_values:
                change_6h = forecast_values[5] - current_aqi if not pd.isna(current_aqi) and len(forecast_values) > 5 else np.nan
                st.metric(
                    "Expected AQI (6h)",
                    f"{forecast_values[5]:.1f}" if not pd.isna(change_6h) else "N/A",
                    delta=round(change_6h, 1) if not pd.isna(change_6h) else None,
                    delta_color="inverse"
                )
        
        with col4:
            if forecast_values:
                max_forecast = max(forecast_values)
                st.metric("Highest AQI Forecast (12h)", f"{max_forecast:.1f}")
        
        st.plotly_chart(fig, width='stretch')

def render_alerts_and_highlights(aqi_df, regions_df, predictions_df, latest_pred_date):
    if not predictions_df.empty and not aqi_df.empty:
        # Get latest predictions
        latest_preds = predictions_df[predictions_df['date'] == latest_pred_date]
        
        # Calculate changes for each region
        changes = []
        for _, pred_row in latest_preds.iterrows():
            region_id = pred_row['region_id']
            current_aqi = aqi_df[
                (aqi_df['region_id'] == region_id) &
                (aqi_df['date'] == aqi_df['date'].max())
            ]['european_aqi']
            
            if not current_aqi.empty:
                current = current_aqi.iloc[0]
                forecast_6h = pred_row['target_t_plus_6']
                change = forecast_6h - current
                
                region_name = regions_df[regions_df['region_id'] == region_id]['region'].iloc[0]
                changes.append({
                    'region': region_name.title().replace('_', ' '),
                    'region_id': region_id,
                    'current_aqi': current,
                    'forecast_6h': forecast_6h,
                    'change': change
                })
        
        changes_df = pd.DataFrame(changes)

        if not changes_df.empty:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("##### 🔴 Top 3 Worsening Regions (Next 6h)")
                worsening = changes_df.nlargest(3, 'change')[['region', 'change', 'forecast_6h']]
                worsening = worsening[worsening['change'] > 0]
                for idx, row in worsening.iterrows():
                    st.write(f"**{row['region']}**: {row['change']:+.1f} → AQI {row['forecast_6h']:.1f}")
            
            with col2:
                st.markdown("##### 🟢 Top 3 Improving Regions (Next 6h)")
                improving = changes_df.nsmallest(3, 'change')[['region', 'change', 'forecast_6h']]
                improving = improving[improving['change'] < 0]
                for idx, row in improving.iterrows():
                    st.write(f"**{row['region']}**: {row['change']:+.1f} → AQI {row['forecast_6h']:.1f}")
            
            with col3:
                st.markdown("##### 📊 National AQI Average Forecast")
                national_avg = changes_df['forecast_6h'].mean()
                st.metric("6h Forecast Average", f"{national_avg:.1f}", get_aqi_category(national_avg))

def plot_error_heatmaps(aqi_df, predictions_df, regions_df, selected_region_ids, date_range_dt):
    """
    Generate side-by-side RMSE and MAE heatmaps for regions × forecast horizons.
    
    Parameters:
    -----------
    aqi_df : pd.DataFrame
        Actual AQI data
    predictions_df : pd.DataFrame
        Predicted AQI data
    regions_df : pd.DataFrame
        Region information
    selected_region_ids : list
        List of region IDs to include
    date_range_dt : tuple
        Date range for evaluation (start_date, end_date)
    """
    
    # Calculate metrics for all horizons
    heatmap_data = []
    for horizon in range(1, 13):
        for region_id in selected_region_ids:
            region_aqi = aqi_df[aqi_df['region_id'] == region_id]
            region_pred = predictions_df[predictions_df['region_id'] == region_id]
            
            metrics = calculate_metrics_for_evaluation(
                region_aqi,
                region_pred,
                horizon,
                date_range_dt
            )
            
            if metrics:
                region_name = regions_df[regions_df['region_id'] == region_id]['region'].iloc[0]
                heatmap_data.append({
                    'region': region_name.title().replace('_', ' '),
                    'horizon': f'+{horizon}h',
                    'rmse': metrics['rmse'],
                    'mae': metrics['mae']
                })
    
    if not heatmap_data:
        st.warning("No data available for heatmap generation.")
        return
    
    # Convert to DataFrame and create pivot tables
    heatmap_df = pd.DataFrame(heatmap_data)
    pivot_rmse = heatmap_df.pivot(index='region', columns='horizon', values='rmse')
    pivot_mae = heatmap_df.pivot(index='region', columns='horizon', values='mae')
    
    # Sort horizons numerically
    sorted_horizons = sorted(pivot_rmse.columns, key=lambda x: int(x.strip('+h')))
    pivot_rmse = pivot_rmse[sorted_horizons]
    pivot_mae = pivot_mae[sorted_horizons]
    
    # Create side-by-side heatmaps
    col1, col2 = st.columns(2)
    
    with col1:
        fig_rmse = _create_heatmap(
            pivot_rmse,
            metric_name="RMSE",
            color_range=[0, 10]
        )
        st.plotly_chart(fig_rmse, width='stretch')
    
    with col2:
        fig_mae = _create_heatmap(
            pivot_mae,
            metric_name="MAE",
            color_range=[0, 10]
        )
        st.plotly_chart(fig_mae, width='stretch')


def _create_heatmap(pivot_data, metric_name, color_range):
    """
    Create a single heatmap figure.
    
    Parameters:
    -----------
    pivot_data : pd.DataFrame
        Pivoted data with regions as rows and horizons as columns
    metric_name : str
        Name of the metric (e.g., 'RMSE', 'MAE')
    color_range : list
        Min and max values for color scale
    
    Returns:
    --------
    plotly.graph_objects.Figure
        Heatmap figure
    """
    fig = px.imshow(
        pivot_data.values,
        labels=dict(x="Forecast Horizon", y="Region", color=metric_name),
        x=pivot_data.columns,
        y=pivot_data.index,
        color_continuous_scale='RdYlGn_r',
        range_color=color_range,
        aspect="auto",
        title=f"{metric_name}: Regions × Horizons"
    )
    fig.update_layout(height=500)
    return fig

def plot_performance_per_region(metrics_df, prediction_horizon):
    fig = px.bar(
        metrics_df.sort_values('rmse'),
        x='region',
        y='rmse',
        title=f'RMSE by Region (Horizon: +{prediction_horizon}h)',
        labels={'region': 'Region', 'rmse': 'RMSE'},
        color='rmse',
        color_continuous_scale='RdYlGn_r',
        range_color=[0, 10]
    )
    fig.update_layout(
        height=500, 
        xaxis_tickangle=-45,
        xaxis=dict(
            tickfont=dict(size=14)
        )
    )
    st.plotly_chart(fig, width='stretch')

def plot_performance_trend_over_time(aqi_df, predictions_df, start_date, end_date, prediction_horizon, selected_region_ids):
    # Calculate daily RMSE and MAE for the selected horizon
    daily_metrics = []
    current_date = start_date
    
    while current_date <= end_date:
        date_start = pd.to_datetime(current_date)
        date_end = date_start + pd.Timedelta(days=1)
        
        day_rmse = []
        day_mae = []
        for region_id in selected_region_ids:
            region_aqi = aqi_df[aqi_df['region_id'] == region_id]
            region_pred = predictions_df[predictions_df['region_id'] == region_id]
            
            metrics = calculate_metrics_for_evaluation(
                region_aqi,
                region_pred,
                prediction_horizon,
                (date_start, date_end)
            )
            
            if metrics:
                day_rmse.append(metrics['rmse'])
                day_mae.append(metrics['mae'])
        
        if day_rmse:
            daily_metrics.append({
                'date': current_date,
                'rmse': np.mean(day_rmse),
                'mae': np.mean(day_mae),
                'n_samples': len(day_rmse)
            })
        
        current_date += timedelta(days=1)
    
    if daily_metrics:
        trend_df = pd.DataFrame(daily_metrics)
        
        fig = go.Figure()
        
        # Add RMSE trace with red color
        fig.add_trace(go.Scatter(
            x=trend_df['date'],
            y=trend_df['rmse'],
            mode='lines+markers',
            name=f'RMSE (+{prediction_horizon}h)',
            line=dict(width=2, color='#E74C3C'),  # Red for RMSE
            marker=dict(size=6)
        ))
        
        # Add MAE trace with orange color
        fig.add_trace(go.Scatter(
            x=trend_df['date'],
            y=trend_df['mae'],
            mode='lines+markers',
            name=f'MAE (+{prediction_horizon}h)',
            line=dict(width=2, color='#F39C12'),  # Orange for MAE
            marker=dict(size=6)
        ))
        
        fig.update_layout(
            title=f'Error Metrics Trend Over Time (Horizon: +{prediction_horizon}h)',
            xaxis_title="Date",
            yaxis_title="Error Value",
            yaxis=dict(range=[0, 10]),
            hovermode='x unified',
            hoverlabel=dict(
                font_size=14
            ),
            height=500,
            template="plotly_white",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                font_size=14
            )
        )
        
        st.plotly_chart(fig, width='stretch')

def show_metrics(metrics_df):
    col1, col2, col3, col4 = st.columns(4)
            
    with col1:
        avg_rmse = metrics_df['rmse'].mean()
        st.metric("Average RMSE", f"{avg_rmse:.2f}")
    
    with col2:
        avg_mae = metrics_df['mae'].mean()
        st.metric("Average MAE", f"{avg_mae:.2f}")
    
    with col3:
        avg_mse = metrics_df['mse'].mean()
        st.metric("Average MSE", f"{avg_mse:.2f}")
    
    with col4:
        total_samples = metrics_df['n_samples'].sum()
        st.metric("Total Samples", f"{total_samples:,}")

def plot_forecasts(aqi_df, predictions_df, start_date, end_date, prediction_horizon, selected_region_ids, regions_df):
    """Plot true AQI values vs predictions for a specific forecast horizon across selected regions"""
    
    # Define color palette for regions
    colors = ['#3498DB', '#2ECC71', '#9B59B6', '#E67E22', '#1ABC9C', '#E74C3C', '#F39C12', '#34495E']
    
    def lighten_color(hex_color, factor=0.5):
        """Convert hex color to a lighter version by blending with white"""
        # Remove '#' if present
        hex_color = hex_color.lstrip('#')
        
        # Convert hex to RGB
        r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
        
        # Blend with white (255, 255, 255)
        r = int(r + (255 - r) * factor)
        g = int(g + (255 - g) * factor)
        b = int(b + (255 - b) * factor)
        
        # Convert back to hex
        return f'#{r:02x}{g:02x}{b:02x}'
    
    fig = go.Figure()
    
    actual_date_start = pd.to_datetime(start_date)
    actual_date_end = pd.to_datetime(end_date)

    pred_date_start = actual_date_start - pd.Timedelta(hours=prediction_horizon)
    pred_date_end = actual_date_end - pd.Timedelta(hours=prediction_horizon)
    
    # Plot for each selected region
    for idx, region_id in enumerate(selected_region_ids):
        color = colors[idx % len(colors)]
        pale_color = lighten_color(color, factor=0.75)
        
        # Get region name
        region_name = regions_df[regions_df['region_id'] == region_id]['region'].iloc[0]
        region_display_name = region_name.title().replace('_', ' ')
        
        # Filter data for this region and date range
        region_aqi = aqi_df[
            (aqi_df['region_id'] == region_id) & 
            (aqi_df['date'] >= actual_date_start) & 
            (aqi_df['date'] <= actual_date_end)
        ].sort_values('date')
        

        region_pred = predictions_df[
            (predictions_df['region_id'] == region_id) & 
            (predictions_df['date'] >= pred_date_start) & 
            (predictions_df['date'] <= pred_date_end)
        ].sort_values('date')
        
        # Determine visibility - only first region visible by default
        visible = True if idx == 0 else 'legendonly'
        
        # Add true AQI values (solid line with circle markers)
        if not region_aqi.empty:
            fig.add_trace(go.Scatter(
                x=region_aqi['date'],
                y=region_aqi['european_aqi'],
                mode='lines+markers',
                name=f'{region_display_name} - True AQI',
                line=dict(color=color, width=2),
                marker=dict(symbol='circle', size=6),
                legendgroup=region_display_name,
                visible=visible,
                opacity=0.8,
            ))
        
        # Add predicted AQI values (dashed line with different marker shape and pale color)
        if not region_pred.empty:
            pred_col = f'target_t_plus_{prediction_horizon}'
            if pred_col in region_pred.columns:
                fig.add_trace(go.Scatter(
                    x=region_aqi['date'],
                    y=region_pred[pred_col],
                    mode='lines+markers',
                    name=f'{region_display_name} - Prediction +{prediction_horizon}h',
                    line=dict(color=pale_color, width=2),
                    marker=dict(symbol='circle', size=6, color=pale_color),
                    legendgroup=region_display_name,
                    visible=visible,
                    opacity=0.8,
                ))
    
    # Customize the layout
    fig.update_layout(
        title=f'AQI Forecasts vs True Values (Horizon: +{prediction_horizon}h)',
        xaxis_title='Date',
        yaxis_title='AQI',
        hovermode='x unified',
        template='plotly_white',
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
            font_size=12,
        ),
        height=500
    )
    
    st.plotly_chart(fig, width="stretch")

def render_overview_dashboard(regions_df, aqi_df, predictions_df):
    """Render the Overview Dashboard tab"""
    
    if predictions_df.empty:
        st.warning("⚠️ No predictions data available. Please run predictions first.")
        return
    
    # Get latest prediction date
    latest_pred_date = predictions_df['date'].max()
    
    # Time slider for forecast horizon
    st.subheader("🗺️ Interactive France Map - AQI Forecasts")

    with st.spinner("Loading map..."):
        render_interactive_map(aqi_df, regions_df, predictions_df, latest_pred_date)

    st.divider()
    
    # Alerts & Highlights
    st.subheader("⚠️ Alerts & Highlights")

    with st.spinner("Making analysis ..."):
        render_alerts_and_highlights(aqi_df, regions_df, predictions_df, latest_pred_date)

    st.divider()
    
    # Regional Forecast Explorer
    st.subheader("📈 Regional Forecast Explorer")
    
    with st.spinner("Visualizing forecasts..."):
        render_regional_forecast(aqi_df, regions_df, predictions_df, latest_pred_date)
    

def render_evaluation_tab(regions_df, aqi_df, predictions_df):
    """Render the Evaluation & Model Performance tab"""
    
    st.subheader("🔍 User-Interactive Evaluation Panel")
    
    if predictions_df.empty or aqi_df.empty:
        st.warning("⚠️ No predictions or AQI data available for evaluation.")
        return
    
    # User controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        prediction_horizon = st.selectbox(
            "Prediction Horizon",
            options=list(range(1, 13)),
            format_func=lambda x: f"+{x}h",
            index=0
        )
    
    with col2:
        # Date range selector
        min_date = predictions_df['date'].min().date() + pd.Timedelta(days=2)
        max_date = predictions_df['date'].max().date() - pd.Timedelta(days=2)
        
        date_range = st.date_input(
            "Date Range",
            value=(max_date - timedelta(days=30), max_date),
            min_value=min_date,
            max_value=max_date
        )
    
    with col3:
        # Region selector (multi-select)
        region_list = regions_df['region'].str.title().str.replace('_', ' ').tolist()
        selected_regions = st.multiselect(
            "Select Regions",
            region_list,
            default=region_list  # All regions by default
        )
    
    # Calculate metrics
    if len(date_range) == 2:
        start_date, end_date = date_range
        date_range_dt = (pd.to_datetime(start_date), pd.to_datetime(end_date))
        
        # Get region IDs
        selected_region_ids = []
        for reg_display in selected_regions:
            reg = reg_display.lower().replace(' ', '_')
            reg_row = regions_df[regions_df['region'].str.lower() == reg]
            if not reg_row.empty:
                selected_region_ids.append(reg_row.iloc[0]['region_id'])
        
        # Calculate metrics for selected regions
        all_metrics = []
        for region_id in selected_region_ids:
            region_aqi = aqi_df[aqi_df['region_id'] == region_id]
            region_pred = predictions_df[predictions_df['region_id'] == region_id]
            
            metrics = calculate_metrics_for_evaluation(
                region_aqi,
                region_pred,
                prediction_horizon,
                date_range_dt
            )
            
            if metrics:
                region_name = regions_df[regions_df['region_id'] == region_id]['region'].iloc[0]
                metrics['region'] = region_name.title().replace('_', ' ')
                metrics['region_id'] = region_id
                all_metrics.append(metrics)
        
        if all_metrics:
            metrics_df = pd.DataFrame(all_metrics)
            
            # Display forecasts
            st.subheader("📉 AQI Forecasts vs Actual Values")
            
            with st.spinner("Creating forecast plot..."):
                plot_forecasts(
                    aqi_df, 
                    predictions_df, 
                    start_date, 
                    end_date, 
                    prediction_horizon, 
                    selected_region_ids,
                    regions_df
                )
            
            st.divider()

            # Display metrics
            st.subheader("📊 Error Metrics")
            
            with st.spinner("Calculating metrics..."):
                show_metrics(metrics_df)
            
            st.divider()
            
            # Performance by region
            st.subheader("📈 Performance by Region")
            
            with st.spinner("Creating bar chart..."):
                plot_performance_per_region(metrics_df, prediction_horizon)

            st.divider()
            
            # Heatmap of errors across horizons and regions
            st.subheader("🔥 Error Heatmap: Regions × Horizons")

            with st.spinner("Creating heatmaps..."):
                plot_error_heatmaps(aqi_df, predictions_df, regions_df, selected_region_ids, date_range_dt)

            st.divider()

            # Performance trend over time
            st.subheader("📉 Performance Trend Over Time")
            
            with st.spinner("Visualizing performance over time..."):
                plot_performance_trend_over_time(aqi_df, predictions_df, start_date, end_date, prediction_horizon, selected_region_ids)
        else:
            st.info("No metrics available for the selected criteria. Try adjusting the date range or regions.")
    else:
        st.info("Please select a date range for evaluation.")

if __name__ == "__main__":
    main()

