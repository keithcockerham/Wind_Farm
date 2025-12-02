import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# Page config
st.set_page_config(
    page_title="Wind Turbine Failure Prediction",
    page_icon="🌬️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .success-card {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
    }
    .stAlert {
        background-color: #f0f2f6;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<p class="main-header">🌬️ Wind Turbine Failure Prediction</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Systematic Methodology with 67% Recall, 100% Precision</p>', unsafe_allow_html=True)


st.sidebar.markdown("**Quick Stats**")

# Load summary data
@st.cache_data
def load_summary_stats():
    return {
        'farms_validated': 3,
        'total_failures': 45,
        'avg_recall': 0.67,
        'avg_precision': 0.96,
        'total_sensors': '81-952',
        'false_alarms': '1 in 691'
    }

stats = load_summary_stats()

st.sidebar.metric("Farms Validated", stats['farms_validated'])
st.sidebar.metric("Failures Analyzed", stats['total_failures'])
st.sidebar.metric("Average Recall", f"{stats['avg_recall']:.0%}")
st.sidebar.metric("Average Precision", f"{stats['avg_precision']:.0%}")

st.sidebar.markdown("---")
st.sidebar.markdown("""
**About This Project**

Developed a systematic, transferable methodology for predicting wind turbine failures 24 hours in advance using SCADA data.

[📁 GitHub Repo](https://github.com/keithcockerham/Wind_Farm)  
""")

# Main content
st.markdown("## Project Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="metric-card">
        <h2>67%</h2>
        <p>Recall on Validation</p>
        <small>18 of 27 failures detected</small>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-card">
        <h2>100%</h2>
        <p>Precision (Farm C)</p>
        <small>0 false alarms on 435 tests</small>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-card">
        <h2>24hr</h2>
        <p>Advance Warning</p>
        <small>Time to schedule maintenance</small>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Key Value Proposition
st.markdown("### 💡 Why This Matters")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    **Problem:**
    - Wind turbine failures are expensive
    - Unplanned downtime loses revenue
    - Reactive maintenance is expensive and risky
    """)

with col2:
    st.markdown("""
    **Solution:**
    - 24-hour advance warning for 2 of 3 failures
    - Near-zero false alarms (operator trust)
    - Goal - Transferable to any wind farm with SCADA data
    """)

st.markdown("---")

# Results Summary
st.markdown("### 📊 Performance Across Three Wind Farms")

results_data = {
    'Farm': ['A (Development)', 'B (Limited Data)', 'C (Validation)'],
    'Environment': ['Onshore Portugal', 'Offshore Germany', 'Offshore Germany'],
    'Sensors': [81, 257, 952],
    'Failures': [12, 6, 27],
    'Recall': [0.67, 0.33, 0.67],
    'Precision': [0.89, 1.00, 1.00],
    'False Alarms': ['1/76', '0/180', '0/435'],
    'Accuracy': [0.95, 0.98, 0.98]
}

df_results = pd.DataFrame(results_data)

# Style the dataframe
def color_recall(val):
    if val >= 0.6:
        color = '#38ef7d'
    elif val >= 0.4:
        color = '#ffd93d'
    else:
        color = '#ff6b6b'
    return f'background-color: {color}; color: black; font-weight: bold'

styled_df = df_results.style.map(
    color_recall, 
    subset=['Recall']
).format({
    'Recall': '{:.0%}',
    'Precision': '{:.0%}',
    'Accuracy': '{:.0%}'
})

st.dataframe(styled_df, use_container_width=True)

st.info("""
✓ **Consistent 67% recall** on adequately-sized datasets (Farms A & C)  
✓ **100% precision** on independent validation farms (B & C)  
✓ **Methodology transfers** across different environments and sensor configurations
""")

st.markdown("---")

# Methodology Preview
st.markdown("### 🧠 Methodology Overview")

st.markdown("""
This project developed a **systematic 5-step pipeline** that works on any wind farm:

1. **Data Validation** - Verify structure, identify power sensor, check quality
2. **Temporal Analysis** - Understand failure patterns (sudden vs gradual)
3. **Feature Engineering** - 24h window aggregation (mean/std/trend per sensor)
4. **Feature Selection** - Automated selection via Cohen's d + correlation removal
5. **Model Evaluation** - Leave-One-Out CV with Random Forest

**Result:** Farm-independent approach - same code works on 81 or 952 sensors.
""")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**✅ What Worked:**")
    st.markdown("""
    - Power threshold filtering (status AND power>0.1)
    - 24h windows capture both sudden and gradual failures
    - Aggressive redundancy removal (r>0.9)
    - Zero false alarms on validation
    """)

with col2:
    st.markdown("**⚠️ Challenges Discovered:**")
    st.markdown("""
    - Smooth degradation hard to detect (1 of 45 failures)
    - Minimum 10-12 failures needed for training
    - Different sensor signatures per farm
    - 67% appears to be stable ceiling
    """)

st.markdown("---")

# Interactive Feature Importance Preview
st.markdown("### 🔍 Feature Importance Example (Farm C)")

# Sample feature importance data
feature_data = {
    'Feature': [
        'Blade Position Variability',
        'Generator Voltage (avg)',
        'Pitch Motor Variability',
        'Rotor Speed Variability',
        'Generator Angular Speed',
        'Phase Current (max)',
        'Rotor Bearing Temp (std)',
        'Gearbox Speed Variability'
    ],
    'Cohen\'s d': [1.86, 1.65, 1.73, 1.59, 1.62, 1.45, 1.39, 1.37],
    'Type': ['Pitch', 'Generator', 'Pitch', 'Rotor', 'Generator', 'Electrical', 'Bearing', 'Gearbox']
}

df_features = pd.DataFrame(feature_data)

fig = px.bar(
    df_features, 
    x='Cohen\'s d', 
    y='Feature',
    color='Type',
    orientation='h',
    title='Top Discriminative Features (Effect Size)',
    labels={'Cohen\'s d': 'Effect Size (Cohen\'s d)'},
    color_discrete_map={
        'Pitch': '#667eea',
        'Generator': '#764ba2',
        'Rotor': '#f093fb',
        'Electrical': '#4facfe',
        'Bearing': '#00f2fe',
        'Gearbox': '#43e97b'
    }
)

fig.update_layout(
    height=400,
    showlegend=True,
    xaxis_title="Effect Size (Cohen's d)",
    yaxis_title=None
)

st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# Call to Action
st.markdown("### Explore Further")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.page_link(
        "pages/methodology.py",
        label="📖 Explore Methodology →",
        icon="📘"
    )

with col2:
    st.page_link(
        "pages/farm_results.py",
        label="📖 Farm Results →",
        icon="📘"
    )

with col3:
    st.page_link(
        "pages/features_explorer.py",
        label="📖 Features Explorer →",
        icon="📘"
    )
with col4:
    st.page_link(
        "pages/model_performance.py",
        label="📖 Model Performance →",
        icon="📘"
    )

st.markdown("---")

# GitHub and Documentation Links
st.markdown("### 📚 Resources")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    [📁 GitHub Repository](https://github.com/keithcockerham/Wind_Farm)
    """)

with col2:
    st.markdown("""
    [📄 Full Documentation](https://github.com/keithcockerham/Wind_Farm/docs)
    """)

with col3:
    st.markdown("""
    [📊 Notebooks](https://github.com/keithcockerham/Wind_Farm/notebooks)
    """)


# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>Built as a learning project for systematic ML methodology, 
    transferability validation, and production deployment readiness.</p>
    <p><strong>Windfarm Data Science Project | 2025</p>
</div>
""", unsafe_allow_html=True)