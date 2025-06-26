from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import joblib

# Đọc dữ liệu
df = pd.read_csv("cutting.csv")
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)
X = df.drop("Label", axis=1)
y = df["Label"]

# Tách dữ liệu
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Thiết lập tham số để tìm best max_depth
param_grid = {
    'max_depth': [5, 10, 15, 20, 25, 30, 40, 50, 100]
}

cv_strategy = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Grid Search để tìm max_depth tốt nhất
grid_search = GridSearchCV(
    estimator=RandomForestClassifier(n_estimators=100, random_state=42),
    param_grid=param_grid,
    scoring='f1_macro',
    cv=cv_strategy,
    n_jobs=-1
)
grid_search.fit(X_train, y_train)

# Lấy max_depth tốt nhất
best_depth = grid_search.best_params_['max_depth']
print(f"Best max_depth: {best_depth}")

# Đánh giá mô hình với cross-validation
clf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=best_depth)
scores = cross_val_score(clf, X, y, cv=cv_strategy, scoring='accuracy', n_jobs=-1)
print(f"Accuracy trung bình: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")

# Huấn luyện lại mô hình với max_depth tốt nhất
clf_final = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=best_depth)
clf_final.fit(X_train, y_train)

# Dự đoán và in kết quả
y_pred = clf_final.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Hiển thị ma trận nhầm lẫn
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Confusion Matrix")
plt.show()

# Lưu mô hình
joblib.dump(clf_final, "modelrechuan.pkl")
