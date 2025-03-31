import os
import sys
import logging

def check_environment():
    """检查环境设置"""
    required_files = [
        'model/random_forest_model2.pkl',
        'templates/index.html',
        'templates/result.html',
        'static/css/style.css',
        'static/js/main.js'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("错误: 以下文件缺失:")
        for file in missing_files:
            print(f"  - {file}")
        return False
    
    return True

if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 检查环境
    if not check_environment():
        sys.exit(1)
    
    # 如果环境检查通过，启动应用
    from app import app
    app.run(debug=True) 