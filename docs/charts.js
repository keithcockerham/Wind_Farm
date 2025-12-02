// Farm Comparison Chart on Homepage
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
        title: {
            text: 'Farm-Specific Model Performance',
            font: { size: 20, family: 'Arial, sans-serif' }
        },
        xaxis: { title: 'Wind Farm' },
        yaxis: { 
            title: 'Recall',
            range: [0, 1],
            tickformat: '.0%'
        },
        showlegend: false,
        plot_bgcolor: '#f7fafc',
        paper_bgcolor: '#ffffff',
        margin: { t: 60, b: 60, l: 60, r: 40 },
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

    const config = {
        responsive: true,
        displayModeBar: false
    };

    Plotly.newPlot('farmComparisonChart', data, layout, config);
}

// Load chart when DOM is ready
document.addEventListener('DOMContentLoaded', function() {
    if (document.getElementById('farmComparisonChart')) {
        createFarmComparisonChart();
    }
});
