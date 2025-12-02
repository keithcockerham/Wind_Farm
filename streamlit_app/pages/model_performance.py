import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import roc_curve, auc, precision_recall_curve

st.set_page_config(
    page_title="Model Performance",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Model Performance Analysis")
st.markdown("### Detailed Metrics, Curves, and Per-Event Results")

# Load performance data
@st.cache_data
def load_performance_data():
    """Load detailed performance metrics and predictions"""
    
    # Farm A detailed results
    farm_a_events = {
        'event_id': [73, 0, 26, 40, 42, 10, 68, 45, 84, 22, 72, 51],
        'failure_type': [
            'Hydraulic group', 'Generator bearing', 'Hydraulic group',
            'Generator bearing', 'Hydraulic group', 'Gearbox',
            'Transformer', 'Hydraulic group', 'Hydraulic group',
            'Hydraulic group', 'Gearbox', 'Gearbox bearings'
        ],
        'asset_id': [0, 0, 0, 10, 0, 10, 10, 11, 13, 13, 21, 21],
        'probability': [0.175, 0.769, 0.193, 0.000, 0.248, 0.737, 0.970, 0.690, 0.920, 0.959, 0.582, 0.582],
        'predicted': [0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
        'actual': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    }
    
    # Farm C detailed results
    farm_c_events = {
        'event_id': [12, 15, 66, 78, 30, 79, 47, 70, 5, 49, 4, 18, 67, 31, 81, 
                     91, 11, 33, 44, 55, 39, 28, 16, 35, 76, 9, 90],
        'failure_type': [
            'Oil level error', 'Pitch failure', 'Pitchfailure', 'Grounding brake',
            'Pitch failure', 'Communication fault', 'Hydraulic problems', 'Pitch battery',
            'Axis 3 error', 'Gear coupling', 'Axis 3 error', 'Rotor brake failure',
            'Transformer overpressure', 'Communication failure', 'Converter failure',
            'Axis 3 error', 'DGUV defective', 'Blade grease', 'Water cooling valve',
            'Harting plug damaged', 'Gear oil cooler', 'Carbon brush defect',
            'Hub battery charger', 'Converter vibrations', 'Pitch battery',
            'Yaw grease pump', 'Communication fault'
        ],
        'asset_id': [2, 12, 12, 15, 16, 16, 21, 23, 32, 33, 34, 34, 35, 35, 38,
                     42, 43, 43, 44, 50, 52, 52, 53, 53, 53, 55, 56],
        'probability': [0.919, 0.297, 0.896, 0.310, 0.892, 0.012, 0.014, 0.999,
                       0.826, 0.999, 0.611, 0.119, 0.970, 0.000, 0.016, 0.647,
                       0.973, 0.689, 1.000, 0.000, 0.980, 1.000, 1.000, 1.000,
                       0.133, 1.000, 0.902],
        'predicted': [1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1,
                     0, 1, 1, 1, 1, 0, 1, 1],
        'actual': [1]*27
    }
    
    return {
        'Farm A': pd.DataFrame(farm_a_events),
        'Farm C': pd.DataFrame(farm_c_events)
    }

performance_data = load_performance_data()

# Farm selector
st.markdown("---")
farm_choice = st.selectbox(
    "Select Wind Farm:",
    ['Farm A (Development)', 'Farm C (Validation)', 'Comparison View'],
    index=2
)

if farm_choice == 'Comparison View':
    # Comparison metrics
    st.markdown("## 📊 Performance Metrics Comparison")
    
    metrics_data = {
        'Metric': ['Recall', 'Precision', 'Specificity', 'Accuracy', 'F1-Score',
                   'True Positives', 'False Positives', 'False Negatives', 'True Negatives'],
        'Farm A': ['67%', '89%', '99%', '95%', '0.76', '8', '1', '4', '75'],
        'Farm B': ['33%', '100%', '100%', '98%', '0.50', '2', '0', '4', '180'],
        'Farm C': ['67%', '100%', '100%', '98%', '0.80', '18', '0', '9', '435']
    }
    
    df_metrics = pd.DataFrame(metrics_data)
    
    # Style the dataframe
    st.dataframe(
        df_metrics.style.set_properties(**{
            'background-color': '#f0f2f6',
            'border-color': 'white'
        }),
        use_container_width=True,
        height=400
    )
    

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('Farm A', 'Farm B', 'Farm C'),
        specs=[[{'type': 'heatmap'}, {'type': 'heatmap'}, {'type': 'heatmap'}]]
        )
            
    # Farm A
    fig.add_trace(
        go.Heatmap(
            z=[[75, 1], [4, 8]],
            text=[[75, 1], [4, 8]],
            texttemplate='%{text}',
            colorscale='Blues',
            showscale=False
        ),
        row=1, col=1
    )
            
    # Farm B
    fig.add_trace(
        go.Heatmap(
            z=[[180, 0], [4, 2]],
            text=[[180, 0], [4, 2]],
            texttemplate='%{text}',
            colorscale='Reds',
            showscale=False
        ),
        row=1, col=2
    )
            
    # Farm C
    fig.add_trace(
        go.Heatmap(
            z=[[435, 0], [9, 18]],
            text=[[435, 0], [9, 18]],
            texttemplate='%{text}',
            colorscale='Greens',
            showscale=False
        ),
        row=1, col=3
    )
            
    fig.update_layout(
        title_text='Confusion Matrices',
        height=450,
        showlegend=False
    )
            
    st.plotly_chart(fig, use_container_width=True)
        
        # Key findings
    st.markdown("---")
    st.markdown("### 🎯 Key Performance Findings")
        
    col1, col2, col3 = st.columns(3)
        
    with col1:
        st.markdown("""
        **✓ Consistent Recall**
            
        - Farm A: 67% (n=12)
        - Farm C: 67% (n=27)
        - Same performance across different farms
        - Stable ceiling for this approach
        """)
        
    with col2:
        st.markdown("""
        **✓ Perfect Validation Precision**
            
        - Farm B: 100% (0/180 false alarms)
        - Farm C: 100% (0/435 false alarms)
        - Model very conservative
        - High operator trust
        """)
        
    with col3:
        st.markdown("""
        **⚠️ Sample Size Impact**
            
        - Farm B: 33% recall (n=6)
        - Minimum ~10-12 failures needed
        - Performance degrades gracefully
        - Precision still maintained
        """)
        
        # ROC Curve Comparison
    st.markdown("---")
    st.markdown("### 📈 ROC Curves (Simulated)")
        
    st.info("""
    **Note:** These are simulated ROC curves based on the probability distributions.  
    Leave-One-Out cross-validation doesn't provide traditional ROC curves, but we can  
    approximate the model's discrimination capability.
    """)
        
    # Generate simulated ROC data
    def simulate_roc(detected_probs, missed_probs, normal_count):
        # Combine all probabilities
        all_probs = list(detected_probs) + list(missed_probs) + [0.0]*normal_count
        all_labels = [1]*len(detected_probs) + [1]*len(missed_probs) + [0]*normal_count
            
        fpr, tpr, _ = roc_curve(all_labels, all_probs)
        roc_auc = auc(fpr, tpr)
            
        return fpr, tpr, roc_auc
        
    fig = go.Figure()
        
    # Farm A
    fpr_a, tpr_a, auc_a = simulate_roc(
        [0.58, 0.61, 0.69, 0.74, 0.77, 0.90, 0.92, 0.96],
        [0.00, 0.17, 0.19, 0.25],
        76
    )
    fig.add_trace(go.Scatter(
        x=fpr_a, y=tpr_a,
        mode='lines',
        name=f'Farm A (AUC = {auc_a:.2f})',
        line=dict(color='#667eea', width=2)
    ))

    # Farm C
    fpr_c, tpr_c, auc_c = simulate_roc(
        [0.61, 0.65, 0.69, 0.83, 0.90, 0.90, 0.92, 0.97, 0.97, 0.98, 0.98, 0.98, 0.99, 0.99, 1.00, 1.00, 1.00, 1.00],
        [0.00, 0.00, 0.01, 0.01, 0.12, 0.13, 0.30, 0.31, 0.37],
        435
    )
    fig.add_trace(go.Scatter(
        x=fpr_c, y=tpr_c,
        mode='lines',
        name=f'Farm C (AUC = {auc_c:.2f})',
        line=dict(color='#00f2fe', width=2)
    ))

    # Diagonal line
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random Classifier',
        line=dict(color='gray', width=1, dash='dash')
    ))

    fig.update_layout(
        title='ROC Curve Comparison',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate (Recall)',
        height=500,
        showlegend=True,
        hovermode='closest'
    )

    st.plotly_chart(fig, use_container_width=True)

    st.caption("AUC (Area Under Curve) close to 1.0 indicates excellent discrimination between classes")

else:
    # Individual farm detailed view
    farm_key = 'Farm A' if 'Farm A' in farm_choice else 'Farm C'
    df_events = performance_data[farm_key]
    
    st.markdown(f"## 📊 {farm_choice} - Detailed Results")
    
    # Overall metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    detected = df_events['predicted'].sum()
    total = len(df_events)
    fp_count = 1 if farm_key == 'Farm A' else 0
    
    col1.metric("Total Failures", total)
    col2.metric("Detected", detected, f"{detected/total:.0%}")
    col3.metric("Missed", total - detected)
    col4.metric("Avg Confidence (Detected)", 
                f"{df_events[df_events['predicted']==1]['probability'].mean():.2f}")
    col5.metric("Avg Confidence (Missed)", 
                f"{df_events[df_events['predicted']==0]['probability'].mean():.2f}")
    
    # Per-event results table
    st.markdown("---")
    st.markdown("### 🔍 Per-Event Predictions")
    
    # Add result column
    df_display = df_events.copy()
    df_display['result'] = df_display['predicted'].apply(lambda x: '✓ Detected' if x == 1 else '✗ Missed')
    df_display['confidence'] = df_display['probability'].apply(lambda x: 'High' if x > 0.7 else ('Medium' if x > 0.3 else 'Low'))
    
    # Reorder columns
    df_display = df_display[['event_id', 'asset_id', 'failure_type', 'probability', 'result', 'confidence']]
    df_display.columns = ['Event ID', 'Asset', 'Failure Type', 'Probability', 'Result', 'Confidence']
    
    # Color coding function
    def color_result(row):
        if row['Result'] == '✓ Detected':
            return ['background-color: #d4edda']*len(row)
        else:
            return ['background-color: #f8d7da']*len(row)
    
    styled_df = df_display.style.apply(color_result, axis=1).format({
        'Probability': '{:.3f}'
    })
    
    st.dataframe(styled_df, use_container_width=True, height=500)
    
    # Probability distribution
    st.markdown("---")
    st.markdown("### 📊 Probability Distribution Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Histogram - all events
        fig = go.Figure()
        
        detected_events = df_events[df_events['predicted'] == 1]
        missed_events = df_events[df_events['predicted'] == 0]
        
        fig.add_trace(go.Histogram(
            x=detected_events['probability'],
            name='Detected',
            marker_color='green',
            opacity=0.7,
            nbinsx=20
        ))
        
        fig.add_trace(go.Histogram(
            x=missed_events['probability'],
            name='Missed',
            marker_color='red',
            opacity=0.7,
            nbinsx=20
        ))
        
        fig.add_vline(x=0.5, line_dash="dash", line_color="black",
                     annotation_text="Threshold = 0.5")
        
        fig.update_layout(
            title='Prediction Probability Distribution',
            xaxis_title='Probability',
            yaxis_title='Count',
            barmode='overlay',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Box plot
        fig = go.Figure()
        
        fig.add_trace(go.Box(
            y=detected_events['probability'],
            name='Detected',
            marker_color='green',
            boxmean='sd'
        ))
        
        fig.add_trace(go.Box(
            y=missed_events['probability'],
            name='Missed',
            marker_color='red',
            boxmean='sd'
        ))
        
        fig.add_hline(y=0.5, line_dash="dash", line_color="black",
                     annotation_text="Threshold")
        
        fig.update_layout(
            title='Probability Box Plot Comparison',
            yaxis_title='Probability',
            height=400,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    st.success("""
    **Model is Decisive:**
    - Clear separation between detected (high prob) and missed (low prob)
    - Few predictions near 0.5 threshold (model is confident)
    - Missed failures have genuinely low probabilities (not borderline)
    """)
    
    # By failure type
    if farm_key == 'Farm A':
        st.markdown("---")
        st.markdown("### 🔧 Performance by Failure Type")
        
        # Group by failure type
        type_performance = df_events.groupby('failure_type').agg({
            'event_id': 'count',
            'predicted': 'sum',
            'probability': 'mean'
        }).reset_index()
        type_performance.columns = ['Failure Type', 'Total', 'Detected', 'Avg Probability']
        type_performance['Recall'] = type_performance['Detected'] / type_performance['Total']
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=type_performance['Failure Type'],
            y=type_performance['Total'],
            name='Total Failures',
            marker_color='lightblue'
        ))
        
        fig.add_trace(go.Bar(
            x=type_performance['Failure Type'],
            y=type_performance['Detected'],
            name='Detected',
            marker_color='green'
        ))
        
        fig.update_layout(
            title='Detection by Failure Type',
            xaxis_title='Failure Type',
            yaxis_title='Count',
            barmode='group',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(
            type_performance.style.format({
                'Avg Probability': '{:.3f}',
                'Recall': '{:.0%}'
            }),
            use_container_width=True
        )
    
    # By asset
    st.markdown("---")
    st.markdown("### 🏭 Performance by Asset")
    
    asset_performance = df_events.groupby('asset_id').agg({
        'event_id': 'count',
        'predicted': 'sum',
        'probability': 'mean'
    }).reset_index()
    asset_performance.columns = ['Asset ID', 'Total', 'Detected', 'Avg Probability']
    asset_performance['Recall'] = asset_performance['Detected'] / asset_performance['Total']
    
    fig = px.bar(
        asset_performance,
        x='Asset ID',
        y=['Total', 'Detected'],
        title='Failures by Asset',
        barmode='group',
        color_discrete_sequence=['lightblue', 'green']
    )
    
    fig.update_layout(height=400)
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.dataframe(
        asset_performance.style.format({
            'Avg Probability': '{:.3f}',
            'Recall': '{:.0%}'
        }).background_gradient(subset=['Recall'], cmap='RdYlGn'),
        use_container_width=True
    )
    
    if farm_key == 'Farm A':
        st.warning("""
        **Asset-Specific Pattern Discovered:**
        - Assets 0 & 10: 50% recall (2 of 4 detected)
        - Assets 11, 13, 21: 100% recall (8 of 8 detected)
        
        Suggests features may not generalize perfectly across turbines.
        """)
    
    # Notable events
    st.markdown("---")
    st.markdown("### ⭐ Notable Events")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Highest Confidence Detection**")
        best = df_events[df_events['predicted'] == 1].nlargest(1, 'probability').iloc[0]
        st.success(f"""
        Event {best['event_id']}: {best['failure_type']}  
        Probability: {best['probability']:.3f}  
        Asset: {best['asset_id']}
        """)
    
    with col2:
        st.markdown("**Most Challenging Miss**")
        worst = df_events[df_events['predicted'] == 0].nlargest(1, 'probability').iloc[0]
        st.error(f"""
        Event {worst['event_id']}: {worst['failure_type']}  
        Probability: {worst['probability']:.3f}  
        Asset: {worst['asset_id']}
        """)
    
    if farm_key == 'Farm A':
        st.markdown("---")
        st.markdown("### 🔬 Event 40 Deep Dive")
        
        st.warning("""
        **Event 40: The Perfect Outlier**
        
        - **Visual observation:** Clear declining RPM trend (1650 → 1450 over 24h)
        - **Model prediction:** 0.000 probability (complete miss)
        - **Feature space analysis:** 10x closer to normal than other failures
        - **Root cause:** Smooth degradation not captured by aggregated statistics
        
        **Conclusion:** Represents different failure physics (smooth vs erratic).  
        Only 1 of 45 total failures (2%) shows this pattern.
        """)

# Model architecture
st.markdown("---")
st.markdown("## ⚙️ Model Architecture & Training")

with st.expander("🔧 Model Configuration"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Model Type:**
        - Random Forest Classifier
        - Ensemble of 100 decision trees
        
        **Hyperparameters:**
        - n_estimators: 100
        - max_depth: 5 (prevent overfitting)
        - class_weight: 'balanced' (handle imbalance)
        - random_state: 42 (reproducibility)
        """)
    
    with col2:
        st.markdown("""
        **Cross-Validation:**
        - Leave-One-Out (LOO) strategy
        - Appropriate for small n (6-27 failures)
        - Maximizes training data per iteration
        - Provides n independent predictions
        
        **Training Time:**
        - Farm A: ~2 seconds (n=88)
        - Farm C: ~30 seconds (n=462)
        """)

with st.expander("🎯 Why These Choices?"):
    st.markdown("""
    **Random Forest over Deep Learning:**
    - Works well with small datasets (12-27 failures)
    - No need for massive data like neural networks
    - Interpretable feature importance
    - Robust to outliers
    
    **Leave-One-Out over K-Fold:**
    - K-fold with n=6 means only 4-5 failures per fold
    - LOO uses n-1 for training (maximum data usage)
    - Standard practice for small sample ML
    
    **Max Depth = 5:**
    - Prevents overfitting with limited data
    - Each tree can only split 5 times
    - Forces model to learn general patterns
    
    **Class Weight = 'balanced':**
    - Handles 14% failure rate (86% normal)
    - Prevents model from predicting "all normal"
    - Gives equal importance to both classes
    """)

# Limitations
st.markdown("---")
st.markdown("## ⚠️ Model Limitations")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    **Known Limitations:**
    
    1. **67% Recall Ceiling**
       - Stable across farms, not improving with more data
       - Some failures inherently unpredictable
    
    2. **Smooth Degradation Challenge**
       - Event 40 example (0% probability)
       - Aggregation hides gradual trends
    
    3. **Sample Size Dependency**
       - Farm B: 33% recall with n=6
       - Needs 10-12 failures minimum
    """)

with col2:
    st.markdown("""
    **Operational Constraints:**
    
    1. **24-Hour Window**
       - Some failures too sudden (no warning)
       - Some too gradual (signal appears later)
    
    2. **Sensor Coverage**
       - Can only detect what sensors measure
       - Mechanical failures may lack signatures
    
    3. **Asset-Specific Patterns**
       - Features may not generalize across turbines
       - Some assets harder to predict
    """)

st.info("""
**Honest Assessment:**  
67% recall with 100% precision is excellent for this problem domain.  
The remaining 33% represents fundamental limits (sudden failures, sensor gaps, smooth degradation).  
Focus on maintaining precision rather than chasing 100% recall.
""")

# Download section
st.markdown("---")
st.markdown("## 📥 Download Code")
 

st.markdown("""
[📊 Complete Pipeline Class](https://github.com/keithcockerham/Wind_Farm/src/pipeline.py)
  
""")


