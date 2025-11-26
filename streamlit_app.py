import streamlit as st
import pandas as pd
import plotly.express as px
import os

PROJECT_DIR = r"D:\Projects\Wind_Turbine"
DATA_DIR = r"D:\Data\SCADA\Wind_Turbine"

slit_dir = os.path.join(PROJECT_DIR, 'Streamlit')

#event_score_path = os.path.join(slit_dir, 'event_scores.csv')
event_scores = pd.read_csv('event_scores.csv')  
features = pd.read_csv('features.csv')
#missed_path = os.path.join(slit_dir, 'missed.csv')
missed = pd.read_csv('missed.csv')

#test_data_path = os.path.join(slit_dir, 'test_data_clean.parquet')
test_data = pd.read_parquet('test_data_clean.parquet')

#event_info = pd.read_csv('d:\\Data\\SCADA\\Wind_Turbine\\Wind Farm A\\event_info.csv', sep=';')
event_info = pd.read_csv('event_info.csv', sep=';')
event_info = event_info[event_info['event_label'] == 'anomaly']

st.title("Wind Turbine Predictive Maintenance System")
st.write("Detecting failures 24 hours in advance using multivariate anomaly detection")

# Sidebar
st.sidebar.header("About This Project")
st.sidebar.write("""
This system detects wind turbine failures by identifying subtle sensor patterns 
that simple threshold alerts miss.

**Key Findings:**
- 66% of failures detected (8 of 12)
- Reduced sensor variability is primary indicator
- Simple thresholds miss gearbox failures entirely
""")

# Main content
st.header("Model Performance")

col1, col2, col3 = st.columns(3)
col1.metric("Failures Detected", "8 of 12", "66%")
col2.metric("Detection Method", "ML Anomaly Detection")
col3.metric("Key Signal", "Reduced Sensor Variability")
 

st.header("Event-Level Anomaly Scores")
st.dataframe(event_scores)

# Simple bar chart
fig = px.bar(event_scores, x='event', y='mean', 
             title='Average Anomaly Score by Failure Event',
             labels={'mean': 'Anomaly Score', 'event': 'Event ID'})
st.plotly_chart(fig)

st.header("Why This Matters")
st.write("""
**Traditional threshold alerts would miss these failures:**
- Gearbox failures show no threshold violations
- Only subtle reductions in sensor variability (30-40%)
- ML detects patterns human operators wouldn't notice
""")

st.header("Sensor Patterns Before Failure")
st.write(""" 
**Event ID and Description Information**
""")
st.dataframe(event_info[['event_id', 'event_description']].sort_values(by='event_id'))
st.write(""" 
**Events That Failed to be Detected**
""")
st.dataframe(missed[['event','mean','min']])
# Let user select an event
event_id = st.selectbox("Select Event to Examine:", event_scores['event'])

# Load that event's data

event_data = test_data[test_data['event'] == event_id]


# Plot key sensors

sensor_descriptions = pd.read_csv('features.csv', index_col='sensor_name')['description'].to_dict()

sensor_cols = ['sensor_5_std', 'sensor_18_std', 'sensor_26_avg', 'sensor_2_avg']

for sensor in sensor_cols:
    # Extract the root sensor name (e.g., 'sensor_26' from 'sensor_26_avg')
    root_sensor = re.match(r'(sensor_\d+|wind_speed_\d+|reactive_power_\d+|power_\d+)', sensor).group(1)
    
    # Get the description
    description = sensor_descriptions.get(root_sensor, 'Unknown')
    
    # Create the plot with enhanced title
    fig = px.line(event_data, x=event_data.index, y=sensor,
                  title=f'{root_sensor}: {description} - 24h Before Failure')
    st.plotly_chart(fig)

############################################################################
