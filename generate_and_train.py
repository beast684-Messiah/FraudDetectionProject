import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

# 创建必要的目录
os.makedirs('model', exist_ok=True)

# 生成示例数据
def generate_sample_data(n_samples=1000):
    np.random.seed(42)
    
    data = {
        'policy_annual_premium': np.random.uniform(500, 2000, n_samples),
        'umbrella_limit': np.random.choice([1000000, 2000000, 3000000, 4000000, 5000000], n_samples),
        'capital_gains': np.random.uniform(0, 100000, n_samples),
        'capital_loss': np.random.uniform(0, 50000, n_samples),
        'incident_severity': np.random.choice(['Minor Damage', 'Major Damage', 'Total Loss', 'Trivial Damage'], n_samples),
        'insured_hobbies': np.random.choice(['chess', 'cross-fit', 'skydiving', 'golf', 'reading'], n_samples),
        'insured_occupation': np.random.choice(['exec-managerial', 'prof-specialty', 'sales', 'tech-support'], n_samples),
        'insured_education_level': np.random.choice(['High School', 'College', 'Masters', 'PhD'], n_samples),
        'incident_state': np.random.choice(['NY', 'CA', 'TX', 'FL', 'IL'], n_samples),
        'insured_relationship': np.random.choice(['husband', 'wife', 'unmarried', 'own-child'], n_samples)
    }
    
    # 生成目标变量
    fraud_prob = (
        (data['policy_annual_premium'] > 1500) * 0.3 +
        (data['incident_severity'] == 'Total Loss') * 0.4 +
        (data['insured_hobbies'] == 'skydiving') * 0.2 +
        np.random.random(n_samples) * 0.1
    )
    
    data['fraud_reported'] = ['Y' if p > 0.5 else 'N' for p in fraud_prob]
    return pd.DataFrame(data)

# 生成数据
df = generate_sample_data()

# 准备特征
features = [
    'policy_annual_premium',
    'umbrella_limit',
    'capital_gains',
    'capital_loss',
    'incident_severity',
    'insured_hobbies',
    'insured_occupation',
    'insured_education_level',
    'incident_state',
    'insured_relationship'
]

X = df[features]
y = df['fraud_reported'].map({'Y': 1, 'N': 0})

# 分离数值和分类特征
numerical_features = ['policy_annual_premium', 'umbrella_limit', 'capital_gains', 'capital_loss']
categorical_features = [f for f in features if f not in numerical_features]

# 创建和保存预处理器
scaler = StandardScaler()
X[numerical_features] = scaler.fit_transform(X[numerical_features])

encoders = {}
for feature in categorical_features:
    encoders[feature] = LabelEncoder()
    X[feature] = encoders[feature].fit_transform(X[feature])

# 训练模型
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# 保存所有必要的组件
joblib.dump(model, 'model/rf_model.joblib')
joblib.dump(scaler, 'model/scaler.joblib')
joblib.dump(encoders, 'model/encoders.joblib')
joblib.dump({'numerical_features': numerical_features, 
             'categorical_features': categorical_features}, 
            'model/feature_info.joblib')

print("模型和预处理器已保存!")
