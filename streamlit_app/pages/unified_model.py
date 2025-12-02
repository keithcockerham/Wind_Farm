import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="Unified Model",
    page_icon="🔬",
    layout="wide"
)

st.title("Unified Multi-Farm Model")
st.markdown("### Cross-Farm Learning with Semantic Sensor Mapping")


st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 20px; border-radius: 10px; color: white; text-align: center;'>
        <h2>Farm B Impact</h2>
        <h3>33% → 50%</h3>
        <p>Modest improvement (+17%)</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div style='background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); 
                padding: 20px; border-radius: 10px; color: white; text-align: center;'>
        <h2>Training Data</h2>
        <h3>6 → 45 failures</h3>
        <p>Combined across all farms</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div style='background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); 
                padding: 20px; border-radius: 10px; color: white; text-align: center;'>
        <h2>Precision</h2>
        <h3>96%</h3>
        <p>99.9% specificity overall</p>
    </div>
    """, unsafe_allow_html=True)

st.info("""
Semantic sensor mapping enables knowledge transfer between different turbine models.
By mapping farm-specific sensors to unified physical categories (e.g., "gearbox_lubrication", "rotor_thermal"),
the model learns generalizable failure patterns that work across manufacturers and environments.
""")

# The Problem
st.markdown("---")
st.markdown("## ❌ The Problem: Limited Data Performance")

problem_data = pd.DataFrame({
    'Farm': ['Farm A\n(n=12)', 'Farm B\n(n=6)', 'Farm C\n(n=27)'],
    'Baseline Recall': [0.67, 0.33, 0.67],
    'Issues': [
        'Good performance',
        'POOR: Insufficient training data',
        'Good performance'
    ]
})

fig = go.Figure()

colors = ['#667eea', '#f5576c', '#00f2fe']

for i, row in problem_data.iterrows():
    fig.add_trace(go.Bar(
        x=[row['Farm']],
        y=[row['Baseline Recall']],
        name=row['Farm'],
        marker_color=colors[i],
        text=[f"{row['Baseline Recall']:.0%}"],
        textposition='outside',
        showlegend=False,
        hovertemplate=f"<b>{row['Farm']}</b><br>Recall: {row['Baseline Recall']:.0%}<br>{row['Issues']}<extra></extra>"
    ))

fig.add_hline(y=0.6, line_dash="dash", line_color="green", 
              annotation_text="Target: 60%", annotation_position="right")

fig.update_layout(
    title='Farm-Specific Models: Farm B Severely Limited',
    xaxis_title='Wind Farm',
    yaxis_title='Recall',
    yaxis_range=[0, 1],
    height=400
)

st.plotly_chart(fig, use_container_width=True)

col1, col2 = st.columns(2)

with col1:
    st.error("""
    **Farm B Challenge:**
    - Only 6 failure events available
    - Model cannot learn reliable patterns
    - 33% recall = misses 4 of 6 failures
    - Unable to deploy with confidence
    """)

with col2:
    st.markdown("""
    **Why This Happens:**
    - ML requires minimum ~10-12 samples
    - With n=6, model underfits
    - High variance in predictions
    - Cannot distinguish signal from noise
    """)

# The Solution
st.markdown("---")
st.markdown("## ✅ The Solution: Unified Multi-Farm Training")

st.markdown("### Step 1: Semantic Sensor Mapping")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    **Before (Farm-Specific):**
    
    - Farm A: `sensor_18` = "Generator RPM variability"
    - Farm B: `sensor_20` = "Gearbox rotational speed"
    - Farm C: `sensor_146` = "Rotor speed gearbox main shaft 1"
    
    ❌ **Different sensors, cannot combine**
    """)

with col2:
    st.markdown("""
    **After (Unified Categories):**
    
    - Farm A: `sensor_18` → **`gearbox_mechanical`**
    - Farm B: `sensor_20` → **`gearbox_mechanical`**
    - Farm C: `sensor_146` → **`gearbox_mechanical`**
    
    ✅ **Same category, can combine!**
    """)

st.markdown("### Step 2: Category Merging Process")

merge_stats = pd.DataFrame({
    'Stage': ['Original', 'After Merging', 'Final Usable'],
    'Categories in All 3 Farms': [3, 12, 12],
    'Categories in 2+ Farms': [10, 16, 16],
    'Farm-Specific Only': [38, 10, 10],
    'Expected Missing Data': ['88%', '38%', '25%']
})

st.dataframe(merge_stats, use_container_width=True, hide_index=True)

with st.expander("📋 View Category Merging Examples"):
    st.markdown("""
    **Electrical System:**
    - `electrical_general` + `electrical_grid_connection` + `electrical_internal` → **`electrical_grid`**
    
    **Gearbox:**
    - `gearbox_bearings` + `gearbox_main_shaft` + `gearbox_general` → **`gearbox_mechanical`**
    - `gearbox_oil_temperature` + `gearbox_oil_pressure` + `gearbox_oil_level` → **`gearbox_lubrication`**
    
    **Generator:**
    - `generator_cooling` + `generator_bearings` + `generator_stator` → **`generator_thermal`**
    - `generator_current` + `generator_speed` → **`generator_electrical`**
    
    **Pitch System:**
    - `pitch_motor` + `pitch_battery` + `pitch_general` → **`pitch_control`**
    - `pitch_angle` kept separate (physically meaningful)
    
    **Result:** 16 categories shared across farms vs. 3 originally
    """)

st.markdown("### Step 3: Feature Aggregation Strategy")

st.code("""
# For each unified category (e.g., "gearbox_mechanical"):

# 1. Find all sensors in this category for this farm
Farm A: sensor_11 (gearbox bearing temp)
Farm C: sensor_146, sensor_147 (main shaft speeds), 
        sensor_168, sensor_169 (axial bearings), ...

# 2. Average across sensors within category (if multiple)
category_timeseries = mean(sensor_146, sensor_147, sensor_168, ...)

# 3. Compute window statistics
gearbox_mechanical_mean = category_timeseries.mean()
gearbox_mechanical_std = category_timeseries.std()
gearbox_mechanical_trend = linear_slope(category_timeseries)

# Result: SAME features for all farms!
""", language='python')

# Results Comparison
st.markdown("---")
st.markdown("## 📈 Results: Farm-Specific vs Unified")

# Side-by-side comparison
comparison_data = {
    'Farm': ['Farm A', 'Farm B', 'Farm C', 'Overall'],
    'Failures': [12, 6, 27, 45],
    'Baseline Recall': [0.67, 0.33, 0.63, 0.56],
    'Unified Recall': [0.58, 0.50, 0.56, 0.56],
    'Change': [-0.09, 0.17, -0.07, 0.00],
    'Baseline Precision': [1.00, 1.00, 1.00, 1.00],
    'Unified Precision': [1.00, 0.75, 1.00, 0.96]
}

df_comparison = pd.DataFrame(comparison_data)

# Create subplots
fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=('Recall Comparison', 'Change from Baseline'),
    specs=[[{'type': 'bar'}, {'type': 'bar'}]]
)

# Left: Recall comparison
farms = df_comparison['Farm'][:3]  # Exclude 'Overall'

fig.add_trace(
    go.Bar(
        x=farms,
        y=df_comparison['Baseline Recall'][:3],
        name='Farm-Specific',
        marker_color='lightblue',
        text=[f"{v:.0%}" for v in df_comparison['Baseline Recall'][:3]],
        textposition='outside'
    ),
    row=1, col=1
)

fig.add_trace(
    go.Bar(
        x=farms,
        y=df_comparison['Unified Recall'][:3],
        name='Unified Model',
        marker_color='darkblue',
        text=[f"{v:.0%}" for v in df_comparison['Unified Recall'][:3]],
        textposition='outside'
    ),
    row=1, col=1
)

# Right: Change
colors_change = ['gray' if c == 0 else 'green' if c > 0 else 'orange' 
                for c in df_comparison['Change'][:3]]

fig.add_trace(
    go.Bar(
        x=farms,
        y=df_comparison['Change'][:3],
        marker_color=colors_change,
        text=[f"{v:+.0%}" for v in df_comparison['Change'][:3]],
        textposition='outside',
        showlegend=False
    ),
    row=1, col=2
)

fig.add_hline(y=0, line_dash="dash", line_color="black", row=1, col=2)

fig.update_xaxes(title_text="Farm", row=1, col=1)
fig.update_yaxes(title_text="Recall", range=[0, 1], row=1, col=1)
fig.update_xaxes(title_text="Farm", row=1, col=2)
fig.update_yaxes(title_text="Change", row=1, col=2)

fig.update_layout(height=400, showlegend=True, barmode='group')

st.plotly_chart(fig, use_container_width=True)

# Detailed metrics table
st.markdown("### 📋 Detailed Performance Metrics")

display_data = df_comparison.copy()
display_data['Baseline Recall'] = display_data['Baseline Recall'].apply(lambda x: f"{x:.0%}")
display_data['Unified Recall'] = display_data['Unified Recall'].apply(lambda x: f"{x:.0%}")
display_data['Change'] = display_data['Change'].apply(lambda x: f"{x:+.0%}")
display_data['Baseline Precision'] = display_data['Baseline Precision'].apply(lambda x: f"{x:.0%}")
display_data['Unified Precision'] = display_data['Unified Precision'].apply(lambda x: f"{x:.0%}")

st.dataframe(display_data, use_container_width=True, hide_index=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.warning("""
    **→ Farm A: Degraded**
    - 67% → 58% recall (-9%)
    - 100% precision (0 false alarms)
    - Trade-off for Farm B improvement
    """)

with col2:
    st.success("""
    **✓ Farm B: Improved**
    - 33% → 50% recall (+17%)
    - 75% precision (1 false alarm)
    - Approaches deployment threshold
    """)

with col3:
    st.warning("""
    **→ Farm C: Moderate Trade-off**
    - 63% → 56% recall (-7%)
    - 100% precision maintained
    - Acceptable for Farm B gain
    """)

# Feature Importance
st.markdown("---")
st.markdown("## Universal Failure Signals Discovered")

st.markdown("""
The unified model identified failure patterns that generalize across all three wind farms,
despite different turbine models, manufacturers, and environments.
""")

feature_importance = pd.DataFrame({
    'Feature': [
        'pitch_angle_std',
        'generator_electrical_std',
        'rotor_mechanical_std',
        'pitch_control_std',
        'pitch_angle_mean',
        'control_temperature_std',
        'generator_thermal_std',
        'power_reactive_std',
        'gearbox_mechanical_mean',
        'rotor_thermal_std'
    ],
    'Category': [
        'Pitch', 'Generator', 'Rotor', 'Pitch', 'Pitch',
        'Control', 'Generator', 'Power', 'Gearbox', 'Rotor'
    ],
    'Cohens_d': [1.228, 0.858, 0.759, 0.716, 0.688, 0.680, 0.616, 0.565, 0.537, 0.524],
    'Signal': [
        'Increased pitch instability',
        'Generator electrical variability',
        'Rotor mechanical instability',
        'Pitch control system stress',
        'Elevated pitch angle',
        'Control cabinet temperature spike',
        'Generator thermal variability',
        'Reactive power variability',
        'Lower gearbox speed',
        'Rotor thermal variability'
    ]
})

category_colors = {
    'Rotor': '#667eea',
    'Gearbox': '#43e97b',
    'Generator': '#764ba2',
    'Pitch': '#f093fb',
    'Control': '#fa709a',
    'Power': '#feca57',
    'Environment': '#ffd93d',
    'Gearbox': '#43e97b',
    'Generator': '#764ba2',
    'Pitch': '#f093fb',
    'Environment': '#ffd93d'
}

fig = go.Figure()

for category in feature_importance['Category'].unique():
    mask = feature_importance['Category'] == category
    subset = feature_importance[mask]
    
    fig.add_trace(go.Bar(
        y=subset['Feature'],
        x=subset['Cohens_d'],
        name=category,
        orientation='h',
        marker_color=category_colors[category],
        text=subset['Cohens_d'].round(2),
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>Effect Size: %{x:.3f}<br>' + 
                     subset['Signal'] + '<extra></extra>'
    ))

fig.update_layout(
    title='Top 10 Universal Failure Predictors (Cohen\'s d)',
    xaxis_title="Effect Size (Cohen's d)",
    yaxis_title=None,
    height=500,
    showlegend=True,
    barmode='relative'
)

st.plotly_chart(fig, use_container_width=True)

st.info("""
**Key Finding:** The strongest signals are pitch control and generator systems (thermal/electrical).
Unified categories emphasize control system health over granular sensor readings.
These patterns work across:
- Onshore (Farm A) vs Offshore (Farms B & C)  
- Different manufacturers
- 81 to 952 sensor configurations

**Trade-off:** Aggregation within categories loses some farm-specific granularity (-7 to -9% on Farms A & C),
but enables deployment on limited-data farms (+17% on Farm B).
""")

# Technical Deep Dive
st.markdown("---")
st.markdown("## 🔬 Technical Implementation")

tab2, tab3, tab4 = st.tabs([
    "Sensor Mapping", "Training Strategy", "Limitations"
])

with tab2:
    st.markdown("""
    ### Sensor Mapping Process
    
    **Challenge:** Wind farms use different sensor configurations:
    - Different manufacturers (Siemens, Vestas, Enercon)
    - Different sensor counts (81 to 952)
    - Different naming conventions
    
    **Solution:** Manual semantic mapping with domain knowledge
    
    **Example Categories Created:**
    
    | Primary Group | Secondary Group | Description | Farms |
    |---------------|-----------------|-------------|-------|
    | electrical | grid | Phase currents, voltages, frequency | All 3 |
    | gearbox | mechanical | Shaft speeds, bearing temps | All 3 |
    | gearbox | lubrication | Oil pressure, temperature, level | All 3 |
    | generator | thermal | Stator temps, bearing temps, cooling | All 3 |
    | generator | electrical | Currents, voltages, speed | All 3 |
    | pitch | control | Motor current, battery, position | B |
    | pitch | angle | Blade pitch angle | All 3 |
    | rotor | mechanical | Rotor speed, brake pressure | All 3 |
    | rotor | thermal | Bearing temps, hub temp | All 3 |
    | power | active | Active power output | All 3 |
    | power | reactive | Reactive power | All 3 |
    | environment | wind | Wind speed, direction | All 3 |
    
    **Merging Strategy:**
    1. Started with 50 granular categories (3 shared across all farms)
    2. Merged overly-specific subcategories using domain knowledge
    3. Ended with 16 categories (12 in all farms, 4 in 2+ farms)
    4. Reduced missing data from 88% to 25%
    """)

with tab3:
    st.markdown("""
    ### Training Strategy
    
    **Data Combination:**
    - Farm A: 12 failures, 84 normal windows
    - Farm B: 6 failures, 176 normal windows
    - Farm C: 27 failures, 435 normal windows
    - **Total: 45 failures, 695 normal windows**
    
    **Cross-Validation:**
    - Leave-One-Out (LOO) strategy
    - Each of 740 windows held out once
    - Trained on remaining 739 windows
    - Appropriate for small failure counts
    
    **Model Configuration:**
    - Random Forest Classifier
    - 100 trees, max depth 5
    - Balanced class weights (14:1 imbalance)
    - Handles missing values via surrogate splits
    
    **Feature Selection:**
    - Phase 1: Cohen's d > 0.6 (discriminative power)
    - Phase 2: Correlation < 0.9 (remove redundancy)
    - Result: 20 features from 150 candidates
    
    **Missing Data Handling:**
    - Median imputation for model training
    - Features with >75% missing excluded
    - Final dataset: ~25% missing on average
    """)

with tab4:
    st.markdown("""
    ### Known Limitations
    
    **1. Farm C Degradation (-8% recall)**
    - Cause: Averaging across many sensors loses some granularity
    - Farm C has 952 sensors (most detailed)
    - Trade-off: Generalizability vs. farm-specific optimization
    - Mitigation: Hybrid approach (unified + farm-specific features)
    
    **2. Category Mapping Requires Domain Expertise**
    - Manual process to map sensors to categories
    - Requires understanding of turbine physics
    - Time-consuming for new farms
    - Mitigation: Build category library over time
    
    **3. Missing Data in Non-Shared Categories**
    - 4 categories exist in only 2 farms
    - 10 categories farm-specific
    - ~25% missing data overall
    - Mitigation: Focus on 12 universally-shared categories
    
    **4. Performance Ceiling at ~62% Recall**
    - Similar to farm-specific models (~67%)
    - Some failures inherently unpredictable
    - Sudden failures with no advance signal
    - Smooth degradation patterns not captured
    
    **5. Requires Minimum Data from Each Farm**
    - Still needs SOME data from target farm
    - Cannot deploy on completely new farm with zero data
    - Mitigation: Start with 5-10 failures for initial calibration
    """)

# Business Value
st.markdown("---")
st.markdown("## 💰 Business Value")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### Operational Benefits
    
    **Faster Deployment**
    - New wind farms can benefit from existing knowledge
    - Reduce data collection period from 2+ years to 6-12 months
    - Start with basic coverage, improve over time
    
    **Consistent Performance**
    - 62% recall across diverse environments
    - Onshore and offshore
    - Different manufacturers and models
    
    **Reduced False Alarms**
    - 97% precision overall
    - 100% on validation farms
    - Maintains operator trust
    
    **Knowledge Transfer**
    - Learn from fleet-wide failures
    - Share insights across portfolio
    - Continuous improvement as fleet grows
    """)

with col2:
    st.markdown("""
    ### ROI Considerations
    
    **Farm B Case Study:**
    - Previously: 33% recall (2 of 6 detected)
    - Now: 67% recall (4 of 6 detected)
    - **2 additional failures prevented**
    
    **Scalability:**
    - Add Farm D, E, F → performance improves
    - More training data → better generalization
    - Shared model across N farms
    - Marginal cost decreases with scale
    """)

# Comparison to Alternatives
st.markdown("---")
st.markdown("## 🔄 Comparison to Alternative Approaches")

alternatives = pd.DataFrame({
    'Approach': [
        'Farm-Specific Models',
        'Transfer Learning (Fine-tuning)',
        'Domain Adaptation',
        'Unified Model (This Work)'
    ],
    'Farm B Recall': ['33%', '~45%', '~50%', '67%'],
    'Interpretability': ['High', 'Low', 'Medium', 'High'],
    'Implementation Complexity': ['Low', 'High', 'High', 'Medium'],
    'Data Requirements': ['High per farm', 'Medium', 'Medium', 'Low per farm'],
    'Pros': [
        'Simple, optimal per farm',
        'Leverages deep learning',
        'Handles distribution shift',
        'Combines data, maintains interpretability'
    ],
    'Cons': [
        'Fails with limited data',
        'Black box, requires GPUs',
        'Complex, hard to debug',
        'Requires sensor mapping'
    ]
})

st.dataframe(alternatives, use_container_width=True, hide_index=True)

st.warning("""
**Unified Model Trade-offs:**
- Modest improvement on limited-data (Farm B: 33% → 50%, +17%)
- Performance cost on well-instrumented farms (Farms A & C: -7 to -9%)
- Maintains interpretability (Random Forest + physical categories)
- Enables consistent baseline across heterogeneous sites (all farms >50%)

**Conclusion:** Unified approach best for limited-data scenarios and rapid deployment. 
Farm-specific models preferred when adequate data exists (n>10 failures).
""")

# Downloads
st.markdown("---")
st.markdown("## 📥 Downloads & Resources")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    [📊 Unified Model Results](https://github.com/keithcockerham/Wind_Farm/results/unified_model.txt)
    
    Complete predictions and metrics.
    """)

with col2:
    st.markdown("""
    [🗺️ Sensor Mapping File](https://github.com/keithcockerham/Wind_Farm/data/sensor_mapping_v2.csv)
    
    Category definitions for all sensors.
    """)

with col3:
    st.markdown("""
    [📓 Implementation Notebook](https://github.com/keithcockerham/Wind_Farm/notebooks/Unified_Model.ipynb)
    
    Complete code walkthrough.
    """)

st.markdown("---")
st.info("""
**Research Note:** This unified modeling approach demonstrates that semantic abstraction of domain-specific 
features enables effective knowledge transfer between heterogeneous data sources. The methodology is 
generalizable to any multi-site industrial monitoring scenario where physical principles are consistent 
but sensor implementations vary.
""")