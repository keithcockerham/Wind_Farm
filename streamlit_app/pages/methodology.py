import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

st.set_page_config(page_title="Methodology", page_icon="📖", layout="wide")

st.title("📖 Systematic Methodology")

# 5-Step Pipeline
st.markdown("## ⚙️ The 5-Step Pipeline")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "1️⃣ Data Validation",
    "2️⃣ Temporal Analysis", 
    "3️⃣ Feature Engineering",
    "4️⃣ Feature Selection",
    "5️⃣ Model Evaluation"
])

with tab1:
    st.markdown("### Step 1: Data Validation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Purpose:**  
        Verify data structure and quality before any analysis.
        
        **Process:**
        1. Validate required columns exist
        2. Auto-detect power sensor
        3. Check status codes available
        4. Report data quality metrics
        
        **Key Discovery:**
        Power threshold essential! Status==0 alone included idle turbines.
        """)
    
    with col2:
        # Sample validation output
        st.code("""
✓ Data structure valid
  SCADA records: 1,574,461
  Unique assets: 22
  Total events: 27
  Anomaly events: 27
  Power column: power_2_avg
  Status values: [0, 3, 4, 5]
        """, language="text")
    
    st.info("**Learning:** Always validate assumptions visually before statistics")

with tab2:
    st.markdown("### Step 2: Temporal Pattern Analysis")
    
    st.markdown("""
    **Purpose:**  
    Understand when failures are logged vs when they actually occur.
    
    **Method:**
    - Find last production timestamp (status==0 AND power>0.1)
    - Calculate gap to event_start
    - Classify failure patterns
    """)
    
    # Sample gap visualization
    gap_data = pd.DataFrame({
        'Farm': ['A', 'A', 'A', 'C', 'C', 'C', 'B'],
        'Pattern': ['7-day', 'Sudden', 'Other', 'Sudden', 'Sudden', 'Sudden', 'Sudden'],
        'Gap_Hours': [168, 0.17, 10.8, 0.2, 0.2, 8.9, 0.5],
        'Count': [8, 3, 1, 20, 5, 2, 4]
    })
    
    fig = go.Figure()
    
    for pattern in gap_data['Pattern'].unique():
        data = gap_data[gap_data['Pattern'] == pattern]
        fig.add_trace(go.Scatter(
            x=data['Farm'],
            y=data['Gap_Hours'],
            mode='markers',
            name=pattern,
            marker=dict(size=data['Count']*3)
        ))
    
    fig.update_layout(
        title='Time Gap Between Last Production and Logged Failure',
        xaxis_title='Wind Farm',
        yaxis_title='Gap (hours, log scale)',
        yaxis_type='log',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.success("""
    **Finding:** Farm A showed 7-day pattern (downtime before logging).  
    Farm C showed mostly sudden failures (<1 hour).
    """)

with tab3:
    st.markdown("### Step 3: Feature Engineering")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        **Window Aggregation Approach:**
        
        For each 24-hour window, compute per sensor:
        - **MEAN:** Average operating level
        - **STD:** Variability/stability  
        - **TREND:** Linear slope (increasing/decreasing)
        
        **Why This Works:**
        - Reduces 144 timesteps → 1 row per window
        - Scales to any sensor count (81 to 952)
        - Prevents data leakage across train/test
        - More interpretable than raw time-series
        """)
    
    with col2:
        st.metric("Farm A", "243 features", "81 sensors × 3")
        st.metric("Farm C", "2,856 features", "952 sensors × 3")
        st.caption("Automatic scaling!")

with tab4:
    st.markdown("### Step 4: Feature Selection")
    
    st.markdown("""
    **Challenge:** 243-2,856 features, need to reduce dimensionality
    
    **Solution:** Automated selection via effect size + correlation removal
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Phase A: Discriminative Analysis**
        
        For each feature:
        1. Calculate Cohen's d (effect size)
        2. Measure difference: normal vs pre-failure
        3. Sort by discriminative power
        
        Cohen's d interpretation:
        - d > 1.0 = Large effect (strong signal)
        - d > 0.5 = Medium effect
        - d < 0.3 = Small effect
        """)
    
    with col2:
        st.markdown("""
        **Phase B: Redundancy Removal**
        
        Iterative selection:
        1. Start with highest Cohen's d
        2. Add next feature IF correlation <0.9
        3. Continue until max_features reached
        
        **Result:** 15-17 diverse features
        
        **Discovery:** 214 correlated pairs (r>0.9)  
        Example: Generator RPM ≈ Rotor RPM (r=1.00)
        """)
    
    # Show redundancy example
    st.markdown("**Redundancy Example: Farm C Top Features**")
    
    redundancy_data = {
        'Feature Pair': [
            'sensor_100 ↔ sensor_105',
            'sensor_103 ↔ sensor_101',
            'sensor_136 ↔ sensor_137',
            'power_29 ↔ power_30'
        ],
        'Correlation': [1.000, 1.000, 0.998, 0.999],
        'Interpretation': [
            'Pitch blades move identically',
            'Pitch blades move identically',
            'Voltage phases nearly identical',
            'Power measurements duplicate'
        ]
    }
    
    st.table(pd.DataFrame(redundancy_data))
    
    st.warning("""
    **Lesson:** More features ≠ better performance.  
    Massive redundancy in sensor data requires aggressive removal.
    """)

with tab5:
    st.markdown("### Step 5: Model Evaluation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Model Choice:**
        - Random Forest Classifier
        - n_estimators: 100
        - max_depth: 5 (prevent overfitting)
        - class_weight: 'balanced'
        
        **Why Random Forest?**
        - Handles non-linear relationships
        - Robust to outliers
        - Feature importance interpretable
        - Works well with small datasets
        """)
    
    with col2:
        st.markdown("""
        **Cross-Validation:**
        - Leave-One-Out (LOO) for small n
        - Maximizes training data per iteration
        - Provides n independent predictions
        
        **Metrics Priority:**
        1. **Precision** (avoid false alarms)
        2. **Recall** (catch failures)
        3. Accuracy (overall)
        """)
    
    # Show sample confusion matrix
    st.markdown("**Sample Result: Farm C**")
    
    conf_matrix = pd.DataFrame(
        [[435, 0], [9, 18]],
        columns=['Predicted Normal', 'Predicted Failure'],
        index=['Actual Normal', 'Actual Failure']
    )
    
    fig = go.Figure(data=go.Heatmap(
        z=conf_matrix.values,
        x=conf_matrix.columns,
        y=conf_matrix.index,
        colorscale='Blues',
        text=conf_matrix.values,
        texttemplate='%{text}',
        textfont={"size": 20},
        showscale=False
    ))
    
    fig.update_layout(
        title='Confusion Matrix: Farm C (27 failures, 435 normal)',
        height=300
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Recall", "67%", "18/27")
    col2.metric("Precision", "100%", "0 false alarms")
    col3.metric("Accuracy", "98%", "453/462")

st.markdown("---")

# Key Decisions
st.markdown("## 🎯 Critical Decisions Documented")

decisions = [
    {
        'decision': 'Restart vs Continue',
        'choice': '✓ Restart with systematic validation',
        'alternative': 'Continue with anomaly detection',
        'outcome': 'Clean methodology, reproducible'
    },
    {
        'decision': 'Normal Operation Definition',
        'choice': '✓ status==0 AND power>0.1',
        'alternative': 'status==0 only',
        'outcome': 'Cleaner signals, better windows'
    },
    {
        'decision': 'Window Aggregation vs Rolling',
        'choice': '✓ Aggregate each 24h window',
        'alternative': 'Rolling windows on full series',
        'outcome': 'Scales to 952 sensors, no leakage'
    },
    {
        'decision': 'Feature Selection Strategy',
        'choice': '✓ Automated via Cohen\'s d',
        'alternative': 'Manual physics-based',
        'outcome': 'Farm-independent, data-driven'
    },
    {
        'decision': 'Precision vs Recall',
        'choice': '✓ Optimize for precision',
        'alternative': 'Maximize recall',
        'outcome': '100% precision on validation'
    }
]

for i, d in enumerate(decisions):
    with st.expander(f"**Decision {i+1}: {d['decision']}**"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"**Chosen:**  \n{d['choice']}")
        with col2:
            st.markdown(f"**Alternative:**  \n{d['alternative']}")
        with col3:
            st.success(f"**Outcome:**  \n{d['outcome']}")

st.markdown("---")

# Downloads
st.markdown("## 📥 Documentation Downloads")

st.markdown("""
[📄 Complete Methodology](https://github.com/keithcockerham/Wind_Farm/docs/methodology.md)
""")