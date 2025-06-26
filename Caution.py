import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler # Thêm scaler nếu cần
import matplotlib.pyplot as plt
import joblib
import time

# --- 1. Hàm hỗ trợ (Tương tự như code ARP trước) ---

def add_noise_to_numerical_features(X, noise_factor=0.15, random_state=42):
    """Thêm nhiễu Gaussian vào tất cả các features số."""
    np.random.seed(random_state)
    X_noisy = X.copy()
    numerical_cols = X_noisy.select_dtypes(include=np.number).columns.tolist()

    noise_applied_cols = []
    for col in numerical_cols:
        if X_noisy[col].std() > 1e-6: # Chỉ thêm nhiễu nếu có độ biến thiên
            noise = np.random.normal(0, noise_factor * X_noisy[col].std(), size=len(X_noisy))
            X_noisy[col] = X_noisy[col] + noise
            noise_applied_cols.append(col)
    print(f"🎲 Đã thêm nhiễu cho {len(noise_applied_cols)} features số: {noise_applied_cols}")
    return X_noisy

# --- 2. Hàm chính để huấn luyện và đánh giá ---

def train_and_evaluate_model(labeled_csv_file, test_size=0.2, cv_folds=10, noise_factor=0.15, random_state=42):
    """
    Huấn luyện và đánh giá mô hình RandomForestClassifier với Grid Search và kỹ thuật chống overfitting.
    """
    print("🚀 Bắt đầu quá trình huấn luyện và đánh giá mô hình...")
    start_total_time = time.time()

    # --- Bước 1: Đọc và tiền xử lý dữ liệu cơ bản ---
    print("\n--- Bước 1: Đọc và tiền xử lý dữ liệu cơ bản ---")
    try:
        df = pd.read_csv(labeled_csv_file)
        print(f"📊 Đã đọc {len(df)} mẫu dữ liệu từ '{labeled_csv_file}'")
    except FileNotFoundError:
        print(f"❌ Lỗi: Không tìm thấy file '{labeled_csv_file}'. Vui lòng đảm bảo file nằm cùng thư mục với script.")
        return

    # Xử lý giá trị vô hạn và NaN
    initial_rows = len(df)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    if len(df) < initial_rows:
        print(f"✅ Đã xử lý NaN/Inf: Loại bỏ {initial_rows - len(df)} dòng. Còn lại {len(df)} dòng.")
    else:
        print("✅ Không có giá trị NaN/Inf nào được tìm thấy và loại bỏ.")

    if 'Label' not in df.columns:
        print(f"❌ Lỗi: Thiếu cột 'Label' trong file '{labeled_csv_file}'. Các cột có sẵn: {df.columns.tolist()}")
        return

    # Tách X và y
    X = df.drop("Label", axis=1)
    y = df["Label"]

    print(f"🔍 Các cột Feature (X): {X.columns.tolist()}")
    print(f"🎯 Cột Nhãn (y): Label")

    # Kiểm tra phân bố nhãn
    print(f"\n📈 Phân bố nhãn trong toàn bộ tập dữ liệu: {dict(y.value_counts())}")
    if len(y.unique()) < 2:
        print("⚠️  Cảnh báo: Chỉ có một lớp duy nhất trong cột 'Label'. Mô hình sẽ không thể học phân loại.")
        return
    elif y.value_counts().min() / y.value_counts().max() < 0.1: # Nếu lớp thiểu số < 10% lớp đa số
        print("⚠️  Cảnh báo: Dữ liệu bị mất cân bằng nghiêm trọng. Hãy chú ý đến F1-score hơn Accuracy.")

    # Thêm nhiễu vào features số (áp dụng cho toàn bộ X trước khi chia train/test để CV nhất quán)
    print(f"\n🎲 Thêm nhiễu vào features với hệ số {noise_factor} để chống overfitting...")
    X_noisy = add_noise_to_numerical_features(X, noise_factor=noise_factor, random_state=random_state)


    # --- Bước 2: Chia tập Train/Test và Preprocessing ---
    print("\n--- Bước 2: Chia tập Train/Test và Preprocessing ---")
    # Chia tập train và test một cách stratify để giữ tỷ lệ nhãn
    X_train, X_test, y_train, y_test = train_test_split(
        X_noisy, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"📦 Tập huấn luyện: {len(X_train)} mẫu | Tập kiểm tra: {len(X_test)} mẫu")

    # Xác định các cột số (numerical) và cột nhị phân/phân loại (nếu có)
    numerical_features = X_train.select_dtypes(include=np.number).columns.tolist()
    # Nếu có cột cần mã hóa one-hot hoặc passthrough, bạn có thể định nghĩa ở đây
    # Ví dụ: categorical_features = X_train.select_dtypes(include='object').columns.tolist()

    # Tạo pipeline với StandardScaler (chuẩn hóa dữ liệu số) và RandomForestClassifier
    # StandardScaler giúp các thuật toán khác nhạy cảm với thang đo hoạt động tốt hơn
    # RandomForest ít nhạy cảm, nhưng vẫn là practice tốt nếu có các thuật toán khác trong pipeline
    preprocessing_pipeline = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    # Pipeline đầy đủ bao gồm preprocessing và classifier
    # Lưu ý: GridSearchCV sẽ tìm kiếm trên các tham số của clf trong pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessing_pipeline),
        ('clf', RandomForestClassifier(
            n_estimators=100,
            random_state=random_state,
            class_weight='balanced' # Cân bằng class, hữu ích cho dữ liệu mất cân bằng
        ))
    ])

    # --- Bước 3: Grid Search để tìm max_depth tối ưu ---
    print("\n--- Bước 3: Grid Search để tìm max_depth tối ưu ---")
    # Thiết lập tham số cho Grid Search
    param_grid = {
        'clf__max_depth': [3, 5, 7, 10, 15, 20, 25, 30] # Thêm các giá trị nhỏ hơn cho max_depth
    }

    # Stratified K-Fold cho Cross-validation trong Grid Search
    cv_strategy_gs = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    print(f"🔍 Bắt đầu Grid Search với {cv_folds}-Fold Cross-validation và scoring='f1_macro'...")
    grid_search = GridSearchCV(
        estimator=pipeline, # Sử dụng pipeline làm estimator
        param_grid=param_grid,
        scoring='f1_macro', # Sử dụng F1-macro cho cân bằng hơn, tốt cho bài toán phân loại
        cv=cv_strategy_gs,
        n_jobs=-1, # Sử dụng tất cả các lõi CPU
        verbose=1 # In tiến độ
    )
    grid_search.fit(X_train, y_train)

    best_depth = grid_search.best_params_['clf__max_depth']
    print(f"🏆 Best max_depth được tìm thấy: {best_depth}")
    print(f"🏆 Best F1-macro score từ Grid Search: {grid_search.best_score_:.4f}")

    # --- Bước 4: Đánh giá mô hình với Best max_depth và Cross-validation ---
    print("\n--- Bước 4: Đánh giá mô hình với Best max_depth và Cross-validation ---")

    # Thiết lập pipeline với max_depth tốt nhất
    final_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessing_pipeline),
        ('clf', RandomForestClassifier(
            n_estimators=100,
            random_state=random_state,
            max_depth=best_depth, # Sử dụng best_depth
            class_weight='balanced'
        ))
    ])

    # Đánh giá lại với cross_val_score trên TẤT CẢ DỮ LIỆU (X_noisy, y)
    # để có cái nhìn tổng quan về hiệu suất của cấu hình tối ưu.
    # Sử dụng cùng chiến lược CV như Grid Search
    print(f"📊 Đánh giá cuối cùng bằng {cv_folds}-Fold Cross-validation trên toàn bộ dữ liệu...")
    cv_scores_final = cross_val_score(final_pipeline, X_noisy, y, cv=cv_strategy_gs, scoring='f1_macro', n_jobs=-1)
    print(f"📊 F1-macro trung bình qua các folds: {cv_scores_final.mean():.4f} (+/- {cv_scores_final.std() * 2:.4f})")


    # --- Bước 5: Huấn luyện mô hình cuối cùng và Đánh giá trên tập test ---
    print("\n--- Bước 5: Huấn luyện mô hình cuối cùng và Đánh giá trên tập test ---")

    # Huấn luyện mô hình cuối cùng trên toàn bộ tập huấn luyện đã tối ưu
    start_train_time = time.time()
    final_pipeline.fit(X_train, y_train)
    training_time = time.time() - start_train_time
    print(f"✅ Huấn luyện mô hình cuối cùng hoàn tất sau {training_time:.2f} giây.")

    # Dự đoán và đánh giá
    y_train_pred = final_pipeline.predict(X_train)
    y_test_pred = final_pipeline.predict(X_test)

    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    train_f1 = f1_score(y_train, y_train_pred, average='macro')
    test_f1 = f1_score(y_test, y_test_pred, average='macro')

    print(f"\n🎯 KẾT QUẢ ĐÁNH GIÁ TRÊN TẬP HUẤN LUYỆN VÀ KIỂM TRA:")
    print(f"   Training Accuracy: {train_acc:.4f}")
    print(f"   Test Accuracy: {test_acc:.4f}")
    print(f"   Training F1-score (macro): {train_f1:.4f}")
    print(f"   Test F1-score (macro): {test_f1:.4f}")
    print(f"   Accuracy Gap (Train - Test): {train_acc - test_acc:.4f}")
    print(f"   F1-score Gap (Train - Test): {train_f1 - test_f1:.4f}")

    if (train_acc - test_acc) > 0.05 or (train_f1 - test_f1) > 0.05: # Ngưỡng 0.05 cho gap
        print("⚠️  CẢNH BÁO: Có dấu hiệu overfitting! (Gap > 0.05)")
    else:
        print("✅ Model có vẻ ổn định, không overfitting nghiêm trọng.")

    print(f"\n📋 Classification Report (Test Set):")
    print(classification_report(y_test, y_test_pred))

    # --- Bước 6: Hiển thị Ma trận nhầm lẫn và Quan trọng của Features ---
    print("\n--- Bước 6: Hiển thị Ma trận nhầm lẫn và Quan trọng của Features ---")
    cm = confusion_matrix(y_test, y_test_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=final_pipeline.classes_)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix (Test Set)")
    plt.show()

    # Feature importance (chỉ từ classifier trong pipeline)
    if hasattr(final_pipeline.named_steps['clf'], 'feature_importances_'):
        importances = final_pipeline.named_steps['clf'].feature_importances_
        feature_imp = pd.DataFrame({
            'feature': X.columns,
            'importance': importances
        }).sort_values('importance', ascending=False)

        print(f"\n🔍 TOP 10 FEATURES QUAN TRỌNG NHẤT:")
        print(feature_imp.head(10))

    # --- Bước 7: Lưu mô hình ---
    print("\n--- Bước 7: Lưu mô hình ---")
    model_save_path = "model_cau.joblib"
    joblib.dump(final_pipeline, model_save_path)
    print(f"✅ Đã lưu mô hình tối ưu: '{model_save_path}'")

    end_total_time = time.time()
    print(f"\n🎉 Toàn bộ quá trình hoàn tất sau {end_total_time - start_total_time:.2f} giây.")

# Chạy script
if __name__ == "__main__":
    train_and_evaluate_model(
        labeled_csv_file='train.csv',
        test_size=0.2,
        cv_folds=10,
        noise_factor=0.05, # Giảm noise factor một chút, hoặc giữ 0.15 nếu dữ liệu vẫn quá separable
        random_state=42
    )