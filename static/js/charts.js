// 添加交互式图表
const createRiskDistributionChart = () => {
    new Chart(ctx, {
        type: 'bubble',
        data: {
            datasets: [{
                label: 'Risk Distribution',
                data: riskData,
                backgroundColor: 'rgba(54, 162, 235, 0.5)'
            }]
        }
    });
}; 