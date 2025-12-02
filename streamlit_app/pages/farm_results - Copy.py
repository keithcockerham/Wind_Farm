import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="Farm Results",
    page_icon="🏭",
    layout="wide"
)

st.title("Wind Farm Performance Results")
st.markdown("### Comparative Analysis Across Three Independent Datasets")

# Load results data
@st.cache_data
def load_farm_results():
    """Load performance metrics for all farms"""
    return {
        'Farm A': {
            'environment': 'Onshore Portugal',
            'turbines': 5,
            'sensors': 81,
            'features_engineered': 243,
            'features_selected': 17,
            'failures': 12,
            'normal_windows': 76,
            'recall': 0.67,
            'precision': 0.89,
            'specificity': 0.99,
            'accuracy': 0.95,
            'f1_score': 0.76,
            'detected': 8,
            'missed': 4,
            'false_positives': 1,
            'true_negatives': 75,
            'conf_matrix': [[75, 1], [4, 8]],
            'failure_types': {
                'Hydraulic group': 6,
                'Gearbox': 3,
                'Generator bearing': 2,
                'Transformer': 1
            },
            'detected_probs': [0.58, 0.61, 0.69, 0.74, 0.77, 0.90, 0.92, 0.96],
            'missed_probs': [0.00, 0.17, 0.19, 0.25]
        },
        'Farm B': {
            'environment': 'Offshore Germany',
            'turbines': 'Not specified',
            'sensors': 257,
            'features_engineered': 257,
            'features_selected': 15,
            'failures': 6,
            'normal_windows': 180,
            'recall': 0.33,
            'precision': 1.00,
            'specificity': 1.00,
            'accuracy': 0.98,
            'f1_score': 0.50,
            'detected': 2,
            'missed': 4,
            'false_positives': 0,
            'true_negatives': 180,
            'conf_matrix': [[180, 0], [4, 2]],
            'failure_types': {'Mixed': 6},
            'detected_probs': [0.82, 0.84],
            'missed_probs': [0.06, 0.12, 0.33, 0.34]
        },
        'Farm C': {
            'environment': 'Offshore Germany',
            'turbines': 22,
            'sensors': 952,
            'features_engineered': 2856,
            'features_selected': 15,
            'failures': 27,
            'normal_windows': 435,
            'recall': 0.67,
            'precision': 1.00,
            'specificity': 1.00,
            'accuracy': 0.98,
            'f1_score': 0.80,
            'detected': 18,
            'missed': 9,
            'false_positives': 0,
            'true_negatives': 435,
            'conf_matrix': [[435, 0], [9, 18]],
            'failure_types': {
                'Pitch system': 8,
                'Hydraulic': 6,
                'Communication': 4,
                'Converter': 3,
                'Other': 6
            },
            'detected_probs': [0.61, 0.65, 0.69, 0.83, 0.89, 0.90, 0.92, 0.97, 0.98, 0.99, 0.99, 0.99, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
            'missed_probs': [0.00, 0.00, 0.01, 0.01, 0.12, 0.13, 0.30, 0.31, 0.37]
        }
    }

results = load_farm_results()

# Farm selector
st.markdown("---")
st.markdown("## 🎯 Select Farm to Analyze")

farm_choice = st.selectbox(
    "Choose a wind farm:",
    ['All Farms (Comparison)', 'Farm A (Development)', 'Farm B (Limited Data)', 'Farm C (Validation)'],
    index=0
)

if farm_choice == 'All Farms (Comparison)':
    # Comparative view
    st.markdown("---")
    st.markdown("## 📊 Performance Comparison")
    
    # Key metrics comparison
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 20px; border-radius: 10px; color: white; text-align: center;'>
            <h2>Farm A</h2>
            <h3>67% Recall</h3>
            <p>Onshore Portugal<br>Development Dataset</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                    padding: 20px; border-radius: 10px; color: white; text-align: center;'>
            <h2>Farm B</h2>
            <h3>33% Recall</h3>
            <p>Offshore Germany<br>Limited Data (n=6)</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); 
                    padding: 20px; border-radius: 10px; color: white; text-align: center;'>
            <h2>Farm C</h2>
            <h3>67% Recall</h3>
            <p>Offshore Germany<br>Primary Validation</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Create comparison dataframe
    comparison_data = {
        'Metric': ['Environment', 'Turbines', 'Sensors', 'Failures', 'Recall', 'Precision', 
                   'Specificity', 'Accuracy', 'F1-Score', 'False Alarms'],
        'Farm A': [
            results['Farm A']['environment'],
            results['Farm A']['turbines'],
            results['Farm A']['sensors'],
            results['Farm A']['failures'],
            f"{results['Farm A']['recall']:.0%}",
            f"{results['Farm A']['precision']:.0%}",
            f"{results['Farm A']['specificity']:.0%}",
            f"{results['Farm A']['accuracy']:.0%}",
            f"{results['Farm A']['f1_score']:.2f}",
            f"{results['Farm A']['false_positives']}/{results['Farm A']['normal_windows']}"
        ],
        'Farm B': [
            results['Farm B']['environment'],
            results['Farm B']['turbines'],
            results['Farm B']['sensors'],
            results['Farm B']['failures'],
            f"{results['Farm B']['recall']:.0%}",
            f"{results['Farm B']['precision']:.0%}",
            f"{results['Farm B']['specificity']:.0%}",
            f"{results['Farm B']['accuracy']:.0%}",
            f"{results['Farm B']['f1_score']:.2f}",
            f"{results['Farm B']['false_positives']}/{results['Farm B']['normal_windows']}"
        ],
        'Farm C': [
            results['Farm C']['environment'],
            results['Farm C']['turbines'],
            results['Farm C']['sensors'],
            results['Farm C']['failures'],
            f"{results['Farm C']['recall']:.0%}",
            f"{results['Farm C']['precision']:.0%}",
            f"{results['Farm C']['specificity']:.0%}",
            f"{results['Farm C']['accuracy']:.0%}",
            f"{results['Farm C']['f1_score']:.2f}",
            f"{results['Farm C']['false_positives']}/{results['Farm C']['normal_windows']}"
        ]
    }
    
    df_comparison = pd.DataFrame(comparison_data)
    
    # Display styled table
    st.dataframe(df_comparison, use_container_width=True, height=400)
    
    st.info("""
    **Key Observations:**
    - ✓ **Consistent 67% recall** on adequately-sized datasets (Farms A & C)
    - ✓ **100% precision** on validation farms (B & C) - zero false alarms
    - ⚠️ **Farm B degraded** to 33% recall due to insufficient sample size (n=6)
    """)
    
    # Visualizations
    st.markdown("---")
    st.markdown("### 📈 Visual Performance Comparison")
    
    # Metrics radar chart
    col1, col2 = st.columns(2)
    
    with col1:
        # Radar chart
        categories = ['Recall', 'Precision', 'Specificity', 'Accuracy', 'F1-Score']
        
        fig = go.Figure()
        
        for farm in ['Farm A', 'Farm B', 'Farm C']:
            values = [
                results[farm]['recall'],
                results[farm]['precision'],
                results[farm]['specificity'],
                results[farm]['accuracy'],
                results[farm]['f1_score']
            ]
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name=farm
            ))
        
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True,
            title='Performance Metrics Comparison',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Bar chart - Recall by farm
        recall_data = pd.DataFrame({
            'Farm': ['Farm A\n(n=12)', 'Farm B\n(n=6)', 'Farm C\n(n=27)'],
            'Recall': [
                results['Farm A']['recall'],
                results['Farm B']['recall'],
                results['Farm C']['recall']
            ],
            'Color': ['#667eea', '#f5576c', '#00f2fe']
        })
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=recall_data['Farm'],
            y=recall_data['Recall'],
            marker_color=recall_data['Color'],
            text=[f"{r:.0%}" for r in recall_data['Recall']],
            textposition='outside'
        ))
        
        fig.add_hline(y=0.67, line_dash="dash", line_color="green", 
                      annotation_text="Target: 67%")
        
        fig.update_layout(
            title='Recall by Farm (Sample Size Impact)',
            yaxis_title='Recall',
            yaxis_range=[0, 1],
            height=400,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Scalability demonstration
    st.markdown("---")
    st.markdown("### 🔧 Scalability: From 81 to 952 Sensors")
    
    scale_data = pd.DataFrame({
        'Farm': ['Farm A', 'Farm B', 'Farm C'],
        'Sensors': [81, 257, 952],
        'Features Engineered': [243, 257, 2856],
        'Features Selected': [17, 15, 15],
        'Recall': [0.67, 0.33, 0.67]
    })
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Feature Count by Farm', 'Performance vs Sensor Count')
    )
    
    # Left: Feature counts
    fig.add_trace(
        go.Bar(name='Engineered', x=scale_data['Farm'], y=scale_data['Features Engineered'],
               marker_color='lightblue'),
        row=1, col=1
    )
    fig.add_trace(
        go.Bar(name='Selected', x=scale_data['Farm'], y=scale_data['Features Selected'],
               marker_color='darkblue'),
        row=1, col=1
    )
    
    # Right: Recall vs sensors
    fig.add_trace(
        go.Scatter(x=scale_data['Sensors'], y=scale_data['Recall'], 
                   mode='markers+lines', marker=dict(size=15, color='#667eea'),
                   name='Recall'),
        row=1, col=2
    )
    
    fig.update_xaxes(title_text="Farm", row=1, col=1)
    fig.update_yaxes(title_text="Feature Count", row=1, col=1)
    fig.update_xaxes(title_text="Number of Sensors", row=1, col=2)
    fig.update_yaxes(title_text="Recall", range=[0, 1], row=1, col=2)
    
    fig.update_layout(height=400, showlegend=True)
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.success("""
    **Scalability Validated:** Same methodology works on 81 to 952 sensors.  
    Automated feature selection prevents feature explosion (2,856 → 15 features).
    """)

else:
    # Individual farm view
    farm_key = farm_choice.split(' ')[0] + ' ' + farm_choice.split(' ')[1]
    farm_data = results[farm_key]
    
    st.markdown("---")
    st.markdown(f"## {farm_choice}")
    
    # Farm overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Environment", farm_data['environment'])
        st.metric("Turbines", farm_data['turbines'])
    
    with col2:
        st.metric("Base Sensors", farm_data['sensors'])
        st.metric("Features Engineered", farm_data['features_engineered'])
    
    with col3:
        st.metric("Features Selected", farm_data['features_selected'])
        st.metric("Total Failures", farm_data['failures'])
    
    with col4:
        st.metric("Normal Windows", farm_data['normal_windows'])
        st.metric("Total Windows", farm_data['failures'] + farm_data['normal_windows'])
    
    st.markdown("---")
    
    # Performance metrics
    st.markdown("### 📊 Performance Metrics")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    col1.metric("Recall", f"{farm_data['recall']:.0%}", 
                f"{farm_data['detected']}/{farm_data['failures']}")
    col2.metric("Precision", f"{farm_data['precision']:.0%}",
                "0 FP" if farm_data['false_positives'] == 0 else f"{farm_data['false_positives']} FP")
    col3.metric("Specificity", f"{farm_data['specificity']:.0%}",
                f"{farm_data['true_negatives']}/{farm_data['normal_windows']}")
    col4.metric("Accuracy", f"{farm_data['accuracy']:.0%}")
    col5.metric("F1-Score", f"{farm_data['f1_score']:.2f}")
    
    # Confusion Matrix
    st.markdown("---")
    st.markdown("### 🎯 Confusion Matrix")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        conf_matrix = farm_data['conf_matrix']
        
        fig = go.Figure(data=go.Heatmap(
            z=conf_matrix,
            x=['Predicted Normal', 'Predicted Failure'],
            y=['Actual Normal', 'Actual Failure'],
            text=conf_matrix,
            texttemplate='%{text}',
            textfont={"size": 24},
            colorscale='Blues',
            showscale=False
        ))
        
        fig.update_layout(
            title=f'Confusion Matrix: {farm_key}',
            height=400,
            xaxis=dict(side='bottom')
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("**Interpretation:**")
        st.markdown(f"""
        - **True Negatives:** {conf_matrix[0][0]}  
          (Correctly identified normal)
        
        - **False Positives:** {conf_matrix[0][1]}  
          (False alarms)
        
        - **False Negatives:** {conf_matrix[1][0]}  
          (Missed failures)
        
        - **True Positives:** {conf_matrix[1][1]}  
          (Correctly detected failures)
        """)
        
        if farm_data['false_positives'] == 0:
            st.success("✓ **Zero false alarms!**  \nPerfect precision")
    
    # Probability distributions
    st.markdown("---")
    st.markdown("### 📈 Prediction Probability Distributions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Detected failures
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=farm_data['detected_probs'],
            nbinsx=20,
            marker_color='green',
            opacity=0.7,
            name='Detected Failures'
        ))
        
        fig.update_layout(
            title=f'Detected Failures (n={farm_data["detected"]})',
            xaxis_title='Prediction Probability',
            yaxis_title='Count',
            xaxis_range=[0, 1],
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.caption(f"Range: {min(farm_data['detected_probs']):.3f} to {max(farm_data['detected_probs']):.3f}")
    
    with col2:
        # Missed failures
        if len(farm_data['missed_probs']) > 0:
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=farm_data['missed_probs'],
                nbinsx=20,
                marker_color='red',
                opacity=0.7,
                name='Missed Failures'
            ))
            
            fig.update_layout(
                title=f'Missed Failures (n={farm_data["missed"]})',
                xaxis_title='Prediction Probability',
                yaxis_title='Count',
                xaxis_range=[0, 1],
                height=300
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.caption(f"Range: {min(farm_data['missed_probs']):.3f} to {max(farm_data['missed_probs']):.3f}")
        else:
            st.info("No missed failures!")
    
    st.info("""
    **Model is decisive, not uncertain:**
    - Detected failures have high confidence (>0.6)
    - Missed failures have low confidence (<0.4)
    - Few predictions near 0.5 threshold
    """)
    
    # Failure type breakdown
    if len(farm_data['failure_types']) > 1:
        st.markdown("---")
        st.markdown("### 🔧 Failure Types Distribution")
        
        fig = px.pie(
            values=list(farm_data['failure_types'].values()),
            names=list(farm_data['failure_types'].keys()),
            title=f'Failure Types: {farm_key}'
        )
        
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(height=400)
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Farm-specific insights
    st.markdown("---")
    st.markdown("### 💡 Key Insights")
    
    if farm_key == 'Farm A':
        st.markdown("""
        **Development Dataset Findings:**
        
        ✓ **Temporal Pattern Discovery:**
        - 8 failures showed 7-day gap (downtime before logging)
        - 4 failures were nearly immediate (<18 hours)
        
        ✓ **Feature Insights:**
        - RPM variability strongest signal (Cohen's d=1.30-1.45)
        - Reactive power variability important
        - Wind/power LOWER in pre-failure windows
        
        ⚠️ **Event 40 Outlier:**
        - Clear visual degradation (declining RPM trend)
        - 0% model probability (smooth degradation not captured)
        - 10x closer to normal than other failures
        """)
    
    elif farm_key == 'Farm B':
        st.markdown("""
        **Limited Data Challenge:**
        
        ⚠️ **Sample Size Impact:**
        - Only 6 failures available for training
        - Recall degraded to 33% (vs 67% on larger datasets)
        - Demonstrates minimum data requirement
        
        ✓ **Precision Maintained:**
        - 100% precision (0 false alarms on 180 tests)
        - Model remains conservative with limited data
        
        📊 **Lesson Learned:**
        - Minimum 10-12 failures needed for 60%+ recall
        - Performance degrades gracefully (doesn't fail completely)
        """)
    
    elif farm_key == 'Farm C':
        st.markdown("""
        **Primary Validation Success:**
        
        ✓ **Transferability Confirmed:**
        - Same 67% recall as Farm A (different environment)
        - Offshore vs onshore, 952 vs 81 sensors
        - Zero manual tuning required
        
        ✓ **Different Sensor Signatures:**
        - Generator voltage collapse (1,135V → 0V)
        - Pitch angle erratic behavior
        - Different physics than Farm A
        
        ✓ **Scalability Demonstrated:**
        - Automated selection: 2,856 → 15 features
        - Massive redundancy removal (213 correlated pairs)
        """)

st.markdown("---")

# Download results
st.markdown("## 📥 Download Results")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    [📊 Performance Metrics CSV](https://github.com/keithcockerham/Wind_Farm/results/metrics.csv)
    
    Complete metrics for all farms.
    """)

with col2:
    st.markdown("""
    [🔍 Detailed Analysis](https://github.com/keithcockerham/Wind_Farm/notebooks/)
    
    Jupyter notebooks with full analysis.
    """)

with col3:
    st.markdown("""
    [📄 Transferability Report](https://github.com/keithcockerham/Wind_Farm/docs/transferability.md)
    
    Complete validation assessment.
    """)