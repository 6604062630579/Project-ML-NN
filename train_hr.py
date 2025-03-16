import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('WA_Fn-UseC_-HR-Employee-Attrition.csv')

# Map target: Attrition (Yes -> 1, No -> 0)
df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})

# Select features ที่เราต้องการใช้ 
selected_features = [
    'Age', 'Gender', 'MaritalStatus', 'JobRole', 'OverTime',
    'DistanceFromHome', 'MonthlyIncome', 'BusinessTravel', 'Department',
    'WorkLifeBalance', 'NumCompaniesWorked'
]

# Keep only selected features + target
df = df[selected_features + ['Attrition']]

# For categorical features, apply Label Encoding
categorical_cols = ['Gender', 'MaritalStatus', 'JobRole', 'OverTime', 'BusinessTravel', 'Department']
for col in categorical_cols:
    if df[col].dtype == 'object':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

# (Note: WorkLifeBalance, Age, DistanceFromHome, MonthlyIncome, NumCompaniesWorked อยู่ในรูปแบบตัวเลขแล้ว)

# Define X and y
X = df[selected_features]
y = df['Attrition']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build pipeline: scaling + XGBoostClassifier
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1,
                                 subsample=0.8, colsample_bytree=0.8, random_state=42,
                                 use_label_encoder=False, eval_metric='logloss'))
])

pipe.fit(X_train, y_train)

# ดู Feature Importance จากโมเดล XGBoost
xgb_model = pipe.named_steps['classifier']
importances = xgb_model.feature_importances_

# แสดงผล Feature Importance แบบตัวเลข (เรียงลำดับจากมากไปน้อย)
indices = np.argsort(importances)[::-1]
print("\nFeature Importances:")
for idx in indices:
    print(f"{selected_features[idx]}: {importances[idx]:.4f}")

# แสดงผล Feature Importance แบบกราฟ
plt.figure(figsize=(10, 6))
plt.barh(range(len(importances)), importances[indices], align='center')
plt.yticks(range(len(importances)), [selected_features[i] for i in indices])
plt.xlabel("Feature Importance")
plt.title("XGBoost Feature Importance")
plt.gca().invert_yaxis()  # ให้ฟีเจอร์ที่สำคัญที่สุดอยู่ด้านบน
plt.show()

y_pred = pipe.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

# Save the trained model
joblib.dump(pipe, 'hr_attrition_model.pkl')
print("Model saved as hr_attrition_model.pkl")
