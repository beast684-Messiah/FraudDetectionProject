import numpy as np
import logging
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib

logger = logging.getLogger(__name__)

# 这些应该与您训练模型时使用的类别相匹配
severity_categories = ['Minor Damage', 'Major Damage', 'Total Loss', 'Trivial Damage']
hobbies_categories = ['chess', 'cross-fit', 'skydiving', 'bungie-jumping', 'base-jumping', 'golf', 'exercise', 'camping', 'dancing', 'movies', 'hiking', 'yachting', 'paintball', 'polo', 'reading']
occupation_categories = ['armed-forces', 'craft-repair', 'exec-managerial', 'farming-fishing', 'handlers-cleaners', 'machine-op-inspct', 'other-service', 'priv-house-serv', 'prof-specialty', 'protective-serv', 'sales', 'tech-support', 'transport-moving']
education_categories = ['High School', 'College', 'Masters', 'Associate', 'JD', 'MD', 'PhD']
state_categories = ['NY', 'OH', 'IL', 'NC', 'PA', 'VA', 'CA', 'OR', 'WV', 'SC']
relationship_categories = ['husband', 'other-relative', 'own-child', 'unmarried', 'wife', 'not-in-family']

def preprocess_input(input_data):
    """预处理输入数据"""
    try:
        # 加载预处理器和特征信息
        scaler = joblib.load('model/scaler.joblib')
        encoders = joblib.load('model/encoders.joblib')
        feature_info = joblib.load('model/feature_info.joblib')
        
        # 准备数据结构
        numerical_features = feature_info['numerical_features']
        categorical_features = feature_info['categorical_features']
        
        # 处理数值特征
        numerical_values = []
        for feature in numerical_features:
            numerical_values.append(float(input_data[feature]))
        numerical_values = np.array(numerical_values).reshape(1, -1)
        numerical_values_scaled = scaler.transform(numerical_values)[0]
        
        # 处理分类特征
        categorical_values = []
        for feature in categorical_features:
            encoder = encoders[feature]
            value = input_data[feature]
            encoded_value = encoder.transform([value])[0]
            categorical_values.append(encoded_value)
        
        # 合并所有特征
        all_features = np.concatenate([numerical_values_scaled, categorical_values])
        
        return all_features
        
    except Exception as e:
        print(f"预处理错误: {str(e)}")
        raise

def get_feature_importance(model):
    """
    获取模型特征重要性
    
    Args:
        model: 训练好的随机森林模型
    
    Returns:
        dict: 特征重要性字典
    """
    # 假设模型有feature_importances_属性
    importances = model.feature_importances_
    
    # 创建特征名称列表（应与训练模型时使用的相同）
    numerical_features = ['policy_annual_premium', 'umbrella_limit', 'capital_gains', 'capital_loss']
    categorical_features = [f'severity_{cat}' for cat in severity_categories] + \
                          [f'hobby_{cat}' for cat in hobbies_categories] + \
                          [f'occupation_{cat}' for cat in occupation_categories] + \
                          [f'education_{cat}' for cat in education_categories] + \
                          [f'state_{cat}' for cat in state_categories] + \
                          [f'relationship_{cat}' for cat in relationship_categories]
    
    all_features = numerical_features + categorical_features
    
    # 将特征与重要性配对
    importance_dict = {feature: importance for feature, importance in zip(all_features, importances)}
    
    # 按重要性降序排序
    sorted_importance = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
    
    return sorted_importance 