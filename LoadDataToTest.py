import pandas as pd
import numpy as np
import joblib

# Load model
model = joblib.load("modelrechuan.pkl")

# Đọc file traffic mới
df = pd.read_csv("test40.csv")

# Lấy phần dữ liệu số
X_new = df.select_dtypes(include=['number'])

# Xử lý NaN & Inf
X_new = X_new.replace([np.inf, -np.inf], pd.NA)
X_new = X_new.dropna()
df = df.loc[X_new.index]  # đồng bộ index

# Dự đoán
y_pred = model.predict(X_new)
df["Prediction"] = y_pred

# Lưu kết quả
df.to_csv("t40datest.csv", index=False)
print("✅ Dự đoán xong. Kết quả lưu tại: t40datest.csv")
