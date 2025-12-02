import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="Feature Explorer",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Feature Explorer")
st.markdown("### Interactive Analysis of What Predicts Failures")

# Load feature data
@st.cache_data
def load_feature_data():
    """Load feature importance and characteristics for all farms"""
    
    farm_a_features = {
        'feature': [
            'sensor_18_avg_std', 'sensor_18_min_std', 'sensor_31_avg_std',
            'reactive_power_27_avg_std', 'reactive_power_28_avg_std',
            'sensor_22_avg_std', 'sensor_44_std', 'sensor_11_avg_std',
            'sensor_8_avg_std', 'power_30_max_mean', 'wind_speed_3_max_mean',
            'wind_speed_3_std_mean', 'sensor_53_avg_mean',
            'sensor_1_avg_trend', 'sensor_42_avg_trend',
            'sensor_8_avg_trend', 'sensor_11_avg_trend'
        ],
        'cohens_d': [1.30, 1.45, 1.43, 1.35, 1.29, 1.37, 1.23, 1.38, 1.23, 
                     1.23, 1.28, 1.22, 1.23, 0.83, 0.80, 0.72, 0.70],
        'sensor_name': [
            'Generator RPM', 'Generator RPM', 'Grid reactive power',
            'Capacitive reactive power', 'Inductive reactive power',
            'Phase displacement', 'Active power counter', 'Gearbox bearing temp',
            'Choke coils temp', 'Maximum grid power', 'Maximum wind speed',
            'Wind variability', 'Nose cone temp',
            'Wind direction trend', 'Nacelle position trend',
            'Choke temp trend', 'Gearbox temp trend'
        ],
        'statistic': [
            'std', 'std', 'std', 'std', 'std', 'std', 'std', 'std', 'std',
            'mean', 'mean', 'mean', 'mean', 'trend', 'trend', 'trend', 'trend'
        ],
        'system': [
            'Generator', 'Generator', 'Electrical', 'Electrical', 'Electrical',
            'Electrical', 'Power', 'Gearbox', 'Electrical', 'Power', 'Environmental',
            'Environmental', 'Environmental', 'Environmental', 'Yaw', 'Electrical', 'Gearbox'
        ],
        'direction': [
            'Higher', 'Higher', 'Higher', 'Higher', 'Higher', 'Higher', 'Higher',
            'Higher', 'Higher', 'Lower', 'Lower', 'Lower', 'Higher',
            'Declining', 'Declining', 'Declining', 'Declining'
        ]
    }
    
    farm_c_features = {
        'feature': [
            'sensor_100_std_std', 'sensor_104_max_std', 'sensor_137_min_mean',
            'sensor_137_max_std', 'sensor_144_min_mean', 'sensor_8_max_mean',
            'sensor_8_min_std', 'sensor_144_avg_std', 'sensor_135_max_mean',
            'sensor_147_max_std', 'sensor_196_std_mean', 'sensor_132_avg_mean',
            'sensor_147_min_std', 'sensor_134_std_mean', 'sensor_230_std_std'
        ],
        'cohens_d': [1.86, 1.73, 1.69, 1.57, 1.54, 1.52, 1.51, 1.46, 1.45,
                     1.44, 1.39, 1.38, 1.37, 1.35, 1.29],
        'sensor_name': [
            'Position motor axis 1', 'Position rotor blade axis 2',
            'Generator RMS voltage L1-L2', 'Generator RMS voltage L2-L3',
            'Rotor speed 1', 'Generator angle speed', 'Generator angle speed',
            'Rotor speed 1', 'RMS line current axis 3', 'Rotor speed gearbox shaft 2',
            'Rotor bearing temperature 1', 'RMS line current axis 2',
            'Rotor speed gearbox shaft 2', 'RMS line current axis 1',
            'Rotor azimuth position'
        ],
        'statistic': [
            'std_std', 'std', 'mean', 'std', 'mean', 'mean', 'std', 'std',
            'mean', 'std', 'mean', 'mean', 'std', 'mean', 'std_std'
        ],
        'system': [
            'Pitch', 'Pitch', 'Generator', 'Generator', 'Rotor', 'Generator',
            'Generator', 'Rotor', 'Electrical', 'Gearbox', 'Bearing', 'Electrical',
            'Gearbox', 'Electrical', 'Rotor'
        ],
        'direction': [
            'Higher', 'Higher', 'Lower', 'Higher', 'Lower', 'Lower', 'Higher',
            'Higher', 'Lower', 'Higher', 'Higher', 'Lower', 'Higher', 'Lower', 'Higher'
        ]
    }
    
    return {
        'Farm A': pd.DataFrame(farm_a_features),
        'Farm C': pd.DataFrame(farm_c_features)
    }

feature_data = load_feature_data()

# Farm selector
st.markdown("---")
farm_choice = st.selectbox(
    "Select Wind Farm:",
    ['Farm A (Onshore Portugal)', 'Farm C (Offshore Germany)'],
    index=0
)

farm_key = 'Farm A' if 'Farm A' in farm_choice else 'Farm C'
df_features = feature_data[farm_key]

st.info(f"""
**{farm_choice}**  
{len(df_features)} features selected from {'243' if farm_key == 'Farm A' else '2,856'} candidates  
All features have Cohen's d > {'0.7' if farm_key == 'Farm A' else '1.2'} (large effect size)
""")

# Overview metrics
st.markdown("---")
st.markdown("## 📊 Feature Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Features", len(df_features))
col2.metric("Avg Effect Size", f"{df_features['cohens_d'].mean():.2f}")
col3.metric("Max Effect Size", f"{df_features['cohens_d'].max():.2f}")
col4.metric("Systems Covered", df_features['system'].nunique())

# Feature importance chart
st.markdown("---")
st.markdown("### 🎯 Feature Importance Ranking")

# Sort by Cohen's d
df_sorted = df_features.sort_values('cohens_d', ascending=True)

fig = go.Figure()

# Color by system
system_colors = {
    'Generator': '#667eea',
    'Electrical': '#764ba2',
    'Pitch': '#f093fb',
    'Rotor': '#4facfe',
    'Gearbox': '#43e97b',
    'Power': '#00f2fe',
    'Environmental': '#ffd93d',
    'Bearing': '#ff6b6b',
    'Yaw': '#a8dadc'
}

for system in df_sorted['system'].unique():
    mask = df_sorted['system'] == system
    fig.add_trace(go.Bar(
        y=df_sorted[mask]['sensor_name'],
        x=df_sorted[mask]['cohens_d'],
        name=system,
        orientation='h',
        marker_color=system_colors.get(system, '#cccccc'),
        text=df_sorted[mask]['cohens_d'].round(2),
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>Effect Size: %{x:.3f}<extra></extra>'
    ))

fig.update_layout(
    title=f'Feature Discriminative Power: {farm_key}',
    xaxis_title="Cohen's d (Effect Size)",
    yaxis_title=None,
    height=600,
    showlegend=True,
    barmode='relative',
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.markdown("### 🔧 Breakdown by Physical System")

col1, col2 = st.columns(2)

with col1:
    system_counts = df_features['system'].value_counts()
    
    fig = px.pie(
        values=system_counts.values,
        names=system_counts.index,
        title='Features by Physical System',
        color=system_counts.index,
        color_discrete_map=system_colors
    )
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(height=400)
    
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # Bar chart - avg effect size by system
    system_effect = df_features.groupby('system')['cohens_d'].mean().sort_values(ascending=True)
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=system_effect.index,
        x=system_effect.values,
        orientation='h',
        marker_color=[system_colors.get(s, '#cccccc') for s in system_effect.index],
        text=system_effect.values.round(2),
        textposition='outside'
    ))
    
    fig.update_layout(
        title='Average Effect Size by System',
        xaxis_title="Average Cohen's d",
        yaxis_title=None,
        height=400,
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Statistic type breakdown
st.markdown("---")
st.markdown("### 📈 Feature Statistic Types")

col1, col2 = st.columns(2)

with col1:
    stat_counts = df_features['statistic'].value_counts()
    
    fig = px.bar(
        x=stat_counts.index,
        y=stat_counts.values,
        title='Features by Statistic Type',
        labels={'x': 'Statistic', 'y': 'Count'},
        color=stat_counts.values,
        color_continuous_scale='Blues'
    )
    
    fig.update_layout(height=350, showlegend=False)
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.caption("""
    **Statistic types:**  
    • **std** = Variability over 24h window  
    • **mean** = Average level over 24h window  
    • **trend** = Linear slope over 24h window
    """)

with col2:
    direction_counts = df_features['direction'].value_counts()
    
    fig = px.bar(
        x=direction_counts.index,
        y=direction_counts.values,
        title='Feature Direction in Pre-Failure',
        labels={'x': 'Direction', 'y': 'Count'},
        color=direction_counts.values,
        color_continuous_scale='Reds'
    )
    
    fig.update_layout(height=350, showlegend=False)
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.caption("""
    **Direction interpretation:**  
    • **Higher** = Increases before failure  
    • **Lower** = Decreases before failure  
    • **Declining** = Negative trend
    """)

# Interactive feature table
st.markdown("---")
st.markdown("### 📋 Interactive Feature Table")

# Add filters
col1, col2, col3 = st.columns(3)

with col1:
    system_filter = st.multiselect(
        "Filter by System:",
        options=['All'] + sorted(df_features['system'].unique().tolist()),
        default=['All']
    )

with col2:
    stat_filter = st.multiselect(
        "Filter by Statistic:",
        options=['All'] + sorted(df_features['statistic'].unique().tolist()),
        default=['All']
    )

with col3:
    min_effect = st.slider(
        "Minimum Effect Size:",
        min_value=float(df_features['cohens_d'].min()),
        max_value=float(df_features['cohens_d'].max()),
        value=float(df_features['cohens_d'].min()),
        step=0.1
    )

# Apply filters
df_filtered = df_features.copy()

if 'All' not in system_filter:
    df_filtered = df_filtered[df_filtered['system'].isin(system_filter)]

if 'All' not in stat_filter:
    df_filtered = df_filtered[df_filtered['statistic'].isin(stat_filter)]

df_filtered = df_filtered[df_filtered['cohens_d'] >= min_effect]

# Display filtered table
df_display = df_filtered[['sensor_name', 'system', 'statistic', 'cohens_d', 'direction']].copy()
df_display.columns = ['Sensor', 'System', 'Statistic', 'Effect Size', 'Direction']
df_display = df_display.sort_values('Effect Size', ascending=False).reset_index(drop=True)

st.dataframe(
    df_display.style.background_gradient(subset=['Effect Size'], cmap='Blues'),
    use_container_width=True,
    height=400
)

st.caption(f"Showing {len(df_filtered)} of {len(df_features)} features")

# Key insights section
st.markdown("---")
st.markdown("### 💡 Key Feature Insights")

if farm_key == 'Farm A':
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Dominant Signal: Variability (std features)**
        
        - 13 of 17 features are variability measures
        - Pre-failure windows show INCREASED instability
        - Generator/rotor RPM variability strongest (d=1.30-1.45)
        - Reactive power variability important (d=1.29-1.43)
        
        **Pattern:** Failures preceded by erratic behavior
        """)
    
    with col2:
        st.markdown("""
        **Surprising Finding: Lower Wind/Power**
        
        - Maximum power LOWER in pre-failure (d=1.23)
        - Maximum wind speed LOWER in pre-failure (d=1.28)
        - Wind variability LOWER in pre-failure (d=1.22)
        
        **Hypothesis:** Failures occur during calmer conditions  
        (Less wind → less load → opportunity for failure?)
        """)
    
    st.success("""
    **Farm A Signature:**  
    Failures characterized by increased variability in rotational/electrical systems  
    combined with lower operating wind/power conditions.
    """)

elif farm_key == 'Farm C':
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Dominant Signal: Generator Shutdown**
        
        - Voltage collapse: 1,135V → 0V (d=1.69)
        - Angular speed drops to zero (d=1.52)
        - Phase currents drop (d=1.38-1.45)
        
        **Pattern:** Complete generator shutdown signature
        """)
    
    with col2:
        st.markdown("""
        **Pitch System Instability**
        
        - Blade position variability extreme (d=1.86)
        - Motor position variability high (d=1.73)
        - Pitch blades moving erratically
        
        **Pattern:** Loss of pitch control before failure
        """)
    
    st.success("""
    **Farm C Signature:**  
    Failures characterized by generator shutdown (voltage/speed collapse) and  
    erratic pitch behavior, suggesting electrical/control system failures.
    """)

# Comparison section
if st.checkbox("🔄 Compare Farm A vs Farm C Features"):
    st.markdown("---")
    st.markdown("### 🔄 Cross-Farm Feature Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Farm A (Onshore)**")
        st.markdown("""
        **Top Systems:**
        - Electrical (6 features)
        - Generator (2 features)
        - Environmental (4 features)
        
        **Top Signals:**
        - RPM variability
        - Reactive power variability
        - Temperature trends
        """)
    
    with col2:
        st.markdown("**Farm C (Offshore)**")
        st.markdown("""
        **Top Systems:**
        - Generator (5 features)
        - Electrical (4 features)
        - Pitch (2 features)
        
        **Top Signals:**
        - Voltage collapse
        - Pitch erratic behavior
        - Speed variability
        """)
    
    st.warning("""
    **Critical Finding:**  
    Different turbine models show DIFFERENT sensor signatures for failures.  
    This is why automated, data-driven feature selection is essential.  
    Cannot transfer learned features between farms, but CAN transfer methodology.
    """)

# Feature selection methodology
st.markdown("---")
st.markdown("### ⚙️ How Features Were Selected")

with st.expander("📖 Feature Selection Process"):
    st.markdown("""
    **Phase 1: Univariate Discriminative Analysis**
    
    For each of the 243-2,856 candidate features:
    1. Calculate Cohen's d (effect size) comparing normal vs pre-failure windows
    2. Perform Welch's t-test for statistical significance
    3. Sort features by discriminative power (Cohen's d)
    
    **Phase 2: Redundancy Removal**
    
    Iterative selection process:
    1. Start with highest Cohen's d feature
    2. For each next candidate:
       - Calculate correlation with already-selected features
       - If max correlation < 0.9: add to selected set
       - Else: skip (redundant)
    3. Stop when reaching max_features (15-17) or exhausting candidates
    
    **Result:**
    - Farm A: 17 features from 243 (7% selected)
    - Farm C: 15 features from 2,856 (0.5% selected)
    - All selected features independent (correlation < 0.9)
    - All selected features discriminative (Cohen's d > 0.7-1.2)
    """)

with st.expander("🔍 Why Massive Redundancy Exists"):
    st.markdown("""
    **Examples of Redundant Sensors:**
    
    1. **Pitch Blades (Farm C):**
       - 3 blades move identically (correlation r=1.000)
       - Keep 1 representative → remove 2
    
    2. **Generator/Rotor RPM (Farm A):**
       - Generator RPM ≈ Rotor RPM (r=1.000)
       - Mechanically coupled → measure same thing
    
    3. **Three-Phase Voltages (Farm C):**
       - Phase L1-L2, L2-L3, L3-L1 nearly identical (r=0.998)
       - Keep 1 phase → remove 2
    
    4. **Power Measurements (Farm A):**
       - Multiple power sensors duplicate (r=0.999)
       - Different measurement points, same information
    
    **Impact:**
    - Farm A: 214 correlated pairs (r>0.9) in top 50 features
    - Farm C: 213 correlated pairs (r>0.9) in top 50 features
    - Without removal: Overfitting, slower training, worse performance
    - With removal: Better generalization, faster, more interpretable
    """)

st.markdown("---")
st.markdown("## 📥 Download Feature Data")

st.markdown(f"""
[📊 {farm_key} Selected Features CSV](https://github.com/keithcockerham/Wind_Farm/results/features_{farm_key.replace(' ', '_').lower()}.txt)

""")

