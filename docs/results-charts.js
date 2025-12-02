// Farm Comparison Chart
function createFarmComparisonChart() {
    const data = [
        {
            x: ['Farm A', 'Farm B', 'Farm C'],
            y: [0.67, 0.33, 0.63],
            name: 'Recall',
            type: 'bar',
            marker: {
                color: ['#667eea', '#f5576c', '#00f2fe']
            },
            text: ['67%', '33%', '63%'],
            textposition: 'outside'
        }
    ];

    const layout = {
        title: 'Farm-Specific Model Performance',
        xaxis: { title: 'Wind Farm' },
        yaxis: { 
            title: 'Recall',
            range: [0, 1],
            tickformat: '.0%'
        },
        showlegend: false,
        plot_bgcolor: '#f7fafc',
        paper_bgcolor: '#ffffff',
        height: 400,
        shapes: [{
            type: 'line',
            x0: -0.5,
            x1: 2.5,
            y0: 0.6,
            y1: 0.6,
            line: {
                color: '#38ef7d',
                width: 2,
                dash: 'dash'
            }
        }],
        annotations: [{
            x: 2.3,
            y: 0.6,
            text: 'Target: 60%',
            showarrow: false,
            font: { color: '#38ef7d' }
        }]
    };

    const config = { responsive: true, displayModeBar: false };
    Plotly.newPlot('farmComparisonChart', data, layout, config);
}

// Unified Model Comparison
function createUnifiedComparisonChart() {
    const trace1 = {
        x: ['Farm A', 'Farm B', 'Farm C'],
        y: [0.67, 0.33, 0.63],
        name: 'Farm-Specific',
        type: 'bar',
        marker: { color: '#667eea' }
    };

    const trace2 = {
        x: ['Farm A', 'Farm B', 'Farm C'],
        y: [0.58, 0.50, 0.56],
        name: 'Unified',
        type: 'bar',
        marker: { color: '#764ba2' }
    };

    const data = [trace1, trace2];

    const layout = {
        title: 'Farm-Specific vs Unified Model',
        xaxis: { title: 'Wind Farm' },
        yaxis: { 
            title: 'Recall',
            range: [0, 1],
            tickformat: '.0%'
        },
        barmode: 'group',
        plot_bgcolor: '#f7fafc',
        paper_bgcolor: '#ffffff',
        height: 400
    };

    const config = { responsive: true, displayModeBar: false };
    Plotly.newPlot('unifiedComparisonChart', data, layout, config);
}

// Feature Importance Chart
function createFeatureImportanceChart() {
    const unifiedFeatures = [
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
    ];

    const cohensD = [1.228, 0.858, 0.759, 0.716, 0.688, 0.680, 0.616, 0.565, 0.537, 0.524];
    
    const categories = [
        'Pitch', 'Generator', 'Rotor', 'Pitch', 'Pitch',
        'Control', 'Generator', 'Power', 'Gearbox', 'Rotor'
    ];

    const categoryColors = {
        'Pitch': '#f093fb',
        'Generator': '#764ba2',
        'Rotor': '#667eea',
        'Control': '#fa709a',
        'Power': '#feca57',
        'Gearbox': '#43e97b'
    };

    const colors = categories.map(cat => categoryColors[cat]);

    const data = [{
        x: cohensD,
        y: unifiedFeatures,
        type: 'bar',
        orientation: 'h',
        marker: { color: colors },
        text: cohensD.map(d => `d=${d.toFixed(2)}`),
        textposition: 'outside'
    }];

    const layout = {
        title: 'Top 10 Unified Model Features (Cohen\'s d)',
        xaxis: { title: 'Effect Size (Cohen\'s d)' },
        yaxis: { 
            title: '',
            automargin: true
        },
        showlegend: false,
        plot_bgcolor: '#f7fafc',
        paper_bgcolor: '#ffffff',
        height: 500,
        margin: { l: 200, r: 50, t: 60, b: 60 }
    };

    const config = { responsive: true, displayModeBar: false };
    Plotly.newPlot('featureImportanceChart', data, layout, config);
}

// Load all charts when DOM is ready
document.addEventListener('DOMContentLoaded', function() {
    if (document.getElementById('farmComparisonChart')) {
        createFarmComparisonChart();
    }
    if (document.getElementById('unifiedComparisonChart')) {
        createUnifiedComparisonChart();
    }
    if (document.getElementById('featureImportanceChart')) {
        createFeatureImportanceChart();
    }
});
