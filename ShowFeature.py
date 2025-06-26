import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

# Đọc dữ liệu
df = pd.read_csv("cutting.csv")

# Xử lý lỗi NaN hoặc vô cùng
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

# Tách X và y
X = df.drop("Label", axis=1)
y = df["Label"]

# Load mô hình đã huấn luyện
model = joblib.load("modelchuan.pkl")

# Lấy độ quan trọng của từng feature
importances = model.feature_importances_
feature_names = X.columns
indices = np.argsort(importances)[::-1]  # Sắp xếp giảm dần

# Hiển thị top 15 feature
plt.figure(figsize=(12, 6))
plt.title("Top 15 Feature Importances for DDoS Detection")
plt.bar(range(15), importances[indices][:15], align="center", color='skyblue')
plt.xticks(range(15), [feature_names[i] for i in indices[:15]], rotation=90)
plt.tight_layout()
plt.show()
