import pandas as pd
import joblib
import numpy as np

# Load model và scaler
model = joblib.load('ddos_model.pkl')
scaler = joblib.load('scaler.pkl')

# Load dữ liệu mới
df = pd.read_csv('UDPSYNNEW.csv')

# Tiền xử lý
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

X = df.drop(columns=['Label'], errors='ignore')  # nếu chưa có cột Label thì bỏ qua
X_scaled = scaler.transform(X)

# Dự đoán
y_pred = model.predict(X_scaled)

# Kết quả
print("Số lượng gói DDoS dự đoán:", sum(y_pred))
print("Số lượng gói Benign dự đoán:", len(y_pred) - sum(y_pred))
