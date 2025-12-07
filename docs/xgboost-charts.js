// XGBoost Performance Chart - Split into two subplots for clarity
function createXGBoostPerformanceChart() {
    // Recall subplot
    const recallTrace = {
        x: ['Farm A', 'Farm B', 'Farm C'],
        y: [1.0, 1.0, 1.0],
        type: 'bar',
        marker: { 
            color: '#38ef7d',
            line: { color: '#2ecc71', width: 2 }
        },
        text: ['✓ 100%<br>(12/12)', '✓ 100%<br>(6/6)', '✓ 100%<br>(27/27)'],
        textposition: 'inside',
        textfont: { size: 14, color: 'white', weight: 'bold' },
        name: 'Recall',
        xaxis: 'x',
        yaxis: 'y'
    };

    // False Positive Rate subplot (inverted to show "accuracy on normal")
    const accuracyOnNormal = {
        x: ['Farm A', 'Farm B', 'Farm C'],
        y: [99.2, 100.0, 98.1],  // 100 - FPR
        type: 'bar',
        marker: { 
            color: '#667eea',
            line: { color: '#5a67d8', width: 2 }
        },
        text: ['99.2%<br>(FPR: 0.8%)', '100%<br>(FPR: 0.0%)', '98.1%<br>(FPR: 1.9%)'],
        textposition: 'inside',
        textfont: { size: 12, color: 'white', weight: 'bold' },
        name: 'Accuracy on Normal Operation',
        xaxis: 'x2',
        yaxis: 'y2'
    };

    const data = [recallTrace, accuracyOnNormal];

    const layout = {
        title: {
            text: 'XGBoost Performance: Perfect Detection with Minimal False Alarms',
            font: { size: 18 }
        },
        grid: {
            rows: 1, 
            columns: 2,
            pattern: 'independent',
            subplots: [['xy'], ['x2y2']]
        },
        xaxis: { 
            title: 'Wind Farm',
            domain: [0, 0.45]
        },
        yaxis: {
            title: 'Recall (%)',
            range: [0, 105],
            tickformat: '.0f',
            ticksuffix: '%'
        },
        xaxis2: { 
            title: 'Wind Farm',
            domain: [0.55, 1]
        },
        yaxis2: {
            title: 'Accuracy on Normal (%)',
            range: [95, 101],
            tickformat: '.1f',
            ticksuffix: '%'
        },
        plot_bgcolor: '#f7fafc',
        paper_bgcolor: '#ffffff',
        height: 400,
        showlegend: false,
        annotations: [
            {
                text: '<b>Failure Detection</b>',
                xref: 'x domain',
                yref: 'y domain',
                x: 0.5,
                y: 1.15,
                xanchor: 'center',
                showarrow: false,
                font: { size: 14, color: '#2d3748' }
            },
            {
                text: '<b>Normal Operation Accuracy</b>',
                xref: 'x2 domain',
                yref: 'y2 domain',
                x: 0.5,
                y: 1.15,
                xanchor: 'center',
                showarrow: false,
                font: { size: 14, color: '#2d3748' }
            }
        ]
    };

    const config = { responsive: true, displayModeBar: false };
    Plotly.newPlot('xgboostPerformanceChart', data, layout, config);
}

// Feature Importance Comparison
function createFeatureImportanceComparison() {
    const farmA = {
        x: [0.732, 0.071, 0.031, 0.023, 0.018],
        y: ['Reactive Power Variability', 'Power Output Variability', 'Generator RPM Variability', 'Rotor Bearing Variability', 'Power Change'],
        name: 'Farm A',
        type: 'bar',
        orientation: 'h',
        marker: { color: '#667eea' }
    };

    const farmB = {
        x: [0.873, 0.021, 0.018, 0.017, 0.013],
        y: ['Rotor Bearing Temp Variability', 'Rotor Bearing Temp Change', 'Rotor Bearing Temp', 'Power Output', 'Power Variability'],
        name: 'Farm B',
        type: 'bar',
        orientation: 'h',
        marker: { color: '#f5576c' }
    };

    const farmC = {
        x: [0.604, 0.074, 0.047, 0.043, 0.039],
        y: ['Power Output Variability', 'Power Output', 'Power Change', 'Internal Voltage Std', 'Rotor Bearing Temp Std'],
        name: 'Farm C',
        type: 'bar',
        orientation: 'h',
        marker: { color: '#00f2fe' }
    };

    const data = [farmA, farmB, farmC];

    const layout = {
        title: 'Top 5 Features by Farm (Feature Importance)',
        xaxis: { 
            title: 'Feature Importance',
            tickformat: '.0%'
        },
        yaxis: {
            automargin: true,
            visible: false
        },
        barmode: 'group',
        plot_bgcolor: '#f7fafc',
        paper_bgcolor: '#ffffff',
        height: 500,
        showlegend: true,
        legend: {
            x: 0.7,
            y: 1.0
        },
        annotations: [
            {
                x: 0.85,
                y: 1.1,
                xref: 'paper',
                yref: 'paper',
                text: '<b>Key Insight:</b> Variability features dominate all farms (60-87%)',
                showarrow: false,
                font: { size: 12, color: '#38ef7d' },
                bgcolor: '#f0fff4',
                bordercolor: '#38ef7d',
                borderwidth: 2,
                borderpad: 8
            }
        ]
    };

    const config = { responsive: true, displayModeBar: false };
    Plotly.newPlot('featureImportanceComparison', data, layout, config);
}

// Model Comparison Chart
function createModelComparisonChart() {
    const rfRecall = {
        x: ['Farm A', 'Farm B', 'Farm C'],
        y: [0.67, 0.33, 0.63],
        name: 'Random Forest',
        type: 'bar',
        marker: { color: '#764ba2' },
        text: ['67%', '33%', '63%'],
        textposition: 'outside'
    };

    const xgbRecall = {
        x: ['Farm A', 'Farm B', 'Farm C'],
        y: [1.0, 1.0, 1.0],
        name: 'XGBoost',
        type: 'bar',
        marker: { color: '#38ef7d' },
        text: ['100%', '100%', '100%'],
        textposition: 'outside'
    };

    const data = [rfRecall, xgbRecall];

    const layout = {
        title: 'Random Forest vs XGBoost: Recall Comparison',
        xaxis: { title: 'Wind Farm' },
        yaxis: {
            title: 'Recall (% Failures Detected)',
            range: [0, 1.1],
            tickformat: '.0%'
        },
        barmode: 'group',
        plot_bgcolor: '#f7fafc',
        paper_bgcolor: '#ffffff',
        height: 450,
        annotations: [
            {
                x: 2.5,
                y: 0.6,
                text: 'RF Target: 60%',
                showarrow: false,
                font: { color: '#764ba2', size: 11 }
            },
            {
                x: 2.5,
                y: 1.05,
                text: 'XGB: Perfect Detection',
                showarrow: false,
                font: { color: '#38ef7d', size: 11, weight: 'bold' }
            }
        ],
        shapes: [
            {
                type: 'line',
                x0: -0.5,
                x1: 2.5,
                y0: 0.6,
                y1: 0.6,
                line: { color: '#764ba2', width: 2, dash: 'dash' }
            }
        ]
    };

    const config = { responsive: true, displayModeBar: false };
    Plotly.newPlot('modelComparisonChart', data, layout, config);
}

// Load all charts when DOM is ready
document.addEventListener('DOMContentLoaded', function() {
    if (document.getElementById('xgboostPerformanceChart')) {
        createXGBoostPerformanceChart();
    }
    if (document.getElementById('featureImportanceComparison')) {
        createFeatureImportanceComparison();
    }
    if (document.getElementById('modelComparisonChart')) {
        createModelComparisonChart();
    }
});
