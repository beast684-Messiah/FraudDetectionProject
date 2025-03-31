// 页面加载完成后执行
document.addEventListener('DOMContentLoaded', function() {
    // 表单验证
    const fraudForm = document.getElementById('fraud-form');
    
    if (fraudForm) {
        fraudForm.addEventListener('submit', function(event) {
            // 自定义验证规则
            const premium = parseFloat(document.getElementById('policy_annual_premium').value);
            
            if (premium <= 0) {
                event.preventDefault();
                alert('Annual premium must be greater than 0');
                return;
            }
            
            // 添加表单提交时的加载动画
            const submitButton = fraudForm.querySelector('button[type="submit"]');
            submitButton.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Processing...';
            submitButton.disabled = true;
        });
    }
    
    // 添加表单输入自动格式化
    const numberInputs = document.querySelectorAll('input[type="number"]');
    numberInputs.forEach(input => {
        input.addEventListener('blur', function() {
            if (this.value) {
                this.value = parseFloat(this.value).toFixed(2);
            }
        });
    });
});

// 添加帮助工具提示
function initializeTooltips() {
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
} 