import os
import glob
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import ipaddress
from collections import Counter

MODEL_PATH = "modelrechuan.pkl"
INPUT_FOLDER = "Data"
OUTPUT_FILE = "predicted_output.csv"
SENDER_EMAIL = "mff1182003@gmail.com"
SENDER_PASSWORD = "kvcb dtvo umwr tdlr"
RECEIVER_EMAIL = "danghuutoan1182003@gmail.com"

# Ngưỡng số lượng flow cần có để 1 IP được coi là nghi ngờ tấn công
THRESHOLD = 5

# ===== CẤU HÌNH WHITELIST THEO YÊU CẦU =====
# Danh sách IP được tin cậy (không cảnh báo nhưng vẫn ghi nhận)
WHITELIST_IPS = [
    "8.6.0.1",  # IP theo yêu cầu
    "192.168.1.1",  # Gateway thông thường
    "192.168.1.5",  # Gateway có thể của mạng hiện tại
    "127.0.0.1",  # Localhost
    "0.0.0.0"  # Broadcast
]

# Dải IP nội bộ
INTERNAL_NETWORKS = [
    "192.168.0.0/16",
    "10.0.0.0/8",
    "172.16.0.0/12"
]

# Ngưỡng cao hơn cho IP nội bộ
INTERNAL_THRESHOLD = 20

# ===== NGƯỠNG CẢNH BÁO MỚI =====
ATTACK_FLOW_WARNING_THRESHOLD = 50  # Cảnh báo khi >50 attack flows
ATTACK_RATIO_WARNING = 0.02  # Cảnh báo khi >2% tổng flows là attack


def is_internal_ip(ip_str):
    """Kiểm tra IP có phải là IP nội bộ không"""
    try:
        ip = ipaddress.ip_address(ip_str)
        for network in INTERNAL_NETWORKS:
            if ip in ipaddress.ip_network(network):
                return True
        return False
    except:
        return False


def get_network_stats(df):
    """Phân tích thống kê mạng"""
    print("🔍 Phân tích thống kê network...")

    # Đếm số lần xuất hiện của mỗi IP
    src_counts = df['Src IP'].value_counts()

    # IP xuất hiện nhiều nhất
    top_src_ips = src_counts.head(5)
    print(f"📊 Top 5 Source IPs:")
    for ip, count in top_src_ips.items():
        internal = "🏠" if is_internal_ip(ip) else "🌐"
        print(f"   {internal} {ip}: {count} flows")

    # Tự động phát hiện IP host (có nhiều traffic)
    auto_whitelist = []
    total_flows = len(df)
    for ip, count in src_counts.items():
        if is_internal_ip(ip) and count > total_flows * 0.1:  # >10% total traffic
            auto_whitelist.append(ip)
            print(f"🛡️  Auto-whitelist: {ip} (có {count}/{total_flows} flows)")

    return auto_whitelist


def advanced_attack_detection(df, base_threshold=THRESHOLD):
    """
    Phát hiện tấn công nâng cao với logic cải thiện
    """
    print("🔍 Phân tích tấn công nâng cao...")

    # Tự động phát hiện IP có thể là host
    auto_whitelist = get_network_stats(df)
    combined_whitelist = WHITELIST_IPS + auto_whitelist

    # Đếm số flow được dự đoán là tấn công cho mỗi IP
    attack_flows = df[df["Prediction"] == 1]
    attack_counts = attack_flows["Src IP"].value_counts()
    total_attack_flows = len(attack_flows)

    print(f"📊 Tổng số IP có flow tấn công: {len(attack_counts)}")
    print(f"📊 Tổng attack flows: {total_attack_flows}")

    suspected_attackers = []
    whitelisted_attackers = []  # Track whitelist IPs có attack flows

    for ip, count in attack_counts.items():
        # Kiểm tra IP trong whitelist
        if ip in combined_whitelist:
            whitelisted_attackers.append({
                'ip': ip,
                'attack_flows': count,
                'total_flows': len(df[df["Src IP"] == ip]),
                'is_internal': is_internal_ip(ip)
            })
            print(f"🛡️  Bỏ qua IP whitelist: {ip} ({count} attack flows)")
            continue

        # Áp dụng threshold khác nhau cho IP nội bộ vs ngoại bộ
        threshold = INTERNAL_THRESHOLD if is_internal_ip(ip) else base_threshold

        if count >= threshold:
            # Thêm phân tích chi tiết
            ip_flows = df[df["Src IP"] == ip]
            total_flows = len(ip_flows)
            attack_ratio = count / total_flows

            print(f"🚨 IP nghi ngờ: {ip}")
            print(f"   📈 Attack flows: {count}/{total_flows} ({attack_ratio:.1%})")
            print(f"   🏠 Internal: {'Yes' if is_internal_ip(ip) else 'No'}")

            # Chỉ cảnh báo nếu tỷ lệ tấn công đủ cao
            if attack_ratio >= 0.3:  # >=30% flows là tấn công
                suspected_attackers.append({
                    'ip': ip,
                    'attack_flows': count,
                    'total_flows': total_flows,
                    'attack_ratio': attack_ratio,
                    'is_internal': is_internal_ip(ip)
                })
            else:
                print(f"   ✅ Tỷ lệ tấn công thấp, có thể là false positive")

    return suspected_attackers, whitelisted_attackers, total_attack_flows


def evaluate_security_status(total_flows, attack_flows, suspected_attackers, whitelisted_attackers):
    """
    Đánh giá tình trạng bảo mật tổng thể - LOGIC MỚI
    """
    print(f"\n🔍 ĐÁNH GIÁ TÌNH TRẠNG BẢO MẬT:")

    # Tính tỷ lệ attack
    attack_ratio = attack_flows / total_flows if total_flows > 0 else 0

    print(f"   📊 Tỷ lệ attack flows: {attack_ratio:.2%}")
    print(f"   📊 Số IP nghi ngờ: {len(suspected_attackers)}")
    print(f"   📊 Số IP whitelist có attack: {len(whitelisted_attackers)}")

    # Đánh giá mức độ nguy hiểm
    risk_level = "LOW"
    should_alert = False

    # Kiểm tra các điều kiện cảnh báo
    if attack_flows >= ATTACK_FLOW_WARNING_THRESHOLD:
        print(f"   ⚠️  Cảnh báo: Số attack flows ({attack_flows}) vượt ngưỡng ({ATTACK_FLOW_WARNING_THRESHOLD})")
        risk_level = "MEDIUM"
        should_alert = True

    if attack_ratio >= ATTACK_RATIO_WARNING:
        print(f"   ⚠️  Cảnh báo: Tỷ lệ attack ({attack_ratio:.2%}) vượt ngưỡng ({ATTACK_RATIO_WARNING * 100:.1f}%)")
        risk_level = "MEDIUM"
        should_alert = True

    if len(suspected_attackers) > 0:
        print(f"   🚨 Nguy hiểm: Phát hiện {len(suspected_attackers)} IP tấn công thực sự")
        risk_level = "HIGH"
        should_alert = True

    # Cảnh báo về whitelist IPs
    for attacker in whitelisted_attackers:
        if attacker['attack_flows'] > 20:  # Ngưỡng đặc biệt cho whitelist
            print(f"   ⚠️  Chú ý: IP whitelist {attacker['ip']} có {attacker['attack_flows']} attack flows")
            if risk_level == "LOW":
                risk_level = "MEDIUM"
                should_alert = True

    return risk_level, should_alert


def get_latest_csv(folder):
    files = glob.glob(os.path.join(folder, "*.csv"))
    if not files:
        raise FileNotFoundError("Không có file .csv nào trong thư mục.")
    latest_file = max(files, key=os.path.getmtime)
    return latest_file


def detect_attack(input_file, output_file):
    print(f"🔍 Đang xử lý file: {input_file}")

    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(input_file)

    print(f"📋 Dữ liệu gốc: {len(df)} flows")
    print(f"📋 Columns: {list(df.columns)}")

    df.dropna(inplace=True)
    df.replace([float("inf"), float("-inf")], 0, inplace=True)
    print(f"📋 Sau cleanup: {len(df)} flows")

    # Giữ lại các cột IP nếu có
    src_ips = df["Src IP"] if "Src IP" in df.columns else None
    dst_ips = df["Dst IP"] if "Dst IP" in df.columns else None

    # Loại bỏ columns không cần cho prediction
    columns_to_drop = []
    for col in ["Label", "Src IP", "Dst IP"]:
        if col in df.columns:
            columns_to_drop.append(col)

    X = df.drop(columns=columns_to_drop)
    print(f"🔮 Features cho prediction: {X.shape[1]} columns")

    # Dự đoán
    y_pred = model.predict(X)
    print(f"📊 Prediction results: {Counter(y_pred)}")

    # Ghép kết quả
    df["Prediction"] = y_pred
    if src_ips is not None:
        df["Src IP"] = src_ips
    if dst_ips is not None:
        df["Dst IP"] = dst_ips

    # Lưu kết quả chi tiết
    df.to_csv(output_file, index=False)
    print(f"💾 Đã lưu kết quả vào: {output_file}")

    # ===== PHÁT HIỆN VÀ ĐÁNH GIÁ =====
    suspected_attackers, whitelisted_attackers, total_attack_flows = advanced_attack_detection(df)

    return df, suspected_attackers, whitelisted_attackers, total_attack_flows


def send_alert_email(num_attacks, num_safe, total, suspected_attackers, whitelisted_attackers, risk_level):


    # Tạo danh sách IP nghi ngờ
    ip_details = []
    if suspected_attackers:
        ip_details.append("🚨 IP TẤN CÔNG THỰC SỰ:")
        for attacker in suspected_attackers:
            internal_flag = "🏠" if attacker['is_internal'] else "🌐"
            ip_details.append(
                f"   {internal_flag} {attacker['ip']}: "
                f"{attacker['attack_flows']}/{attacker['total_flows']} flows "
                f"({attacker['attack_ratio']:.1%} attack rate)"
            )

    # Thêm thông tin whitelist có vấn đề
    if whitelisted_attackers:
        ip_details.append("\n⚠️  IP WHITELIST CÓ ATTACK FLOWS:")
        for attacker in whitelisted_attackers:
            internal_flag = "🏠" if attacker['is_internal'] else "🌐"
            ip_details.append(
                f"   {internal_flag} {attacker['ip']}: "
                f"{attacker['attack_flows']}/{attacker['total_flows']} flows"
            )

    ip_list = "\n".join(ip_details) if ip_details else "Không có IP đáng ngờ"

    # Xác định emoji theo mức độ
    risk_emoji = {"LOW": "✅", "MEDIUM": "⚠️", "HIGH": "🚨"}[risk_level]

    msg_text = f"""
🔔 BÁO CÁO PHÁT HIỆN TẤN CÔNG DDoS

{risk_emoji} MỨC ĐỘ NGUY HIỂM: {risk_level}

📊 THỐNG KÊ TỔNG QUAN:
   • Tổng số Flow: {total}
   • Flow không bị tấn công (Benign): {num_safe}
   • Flow bị tấn công (DDoS): {num_attacks} ({num_attacks / total:.2%})
   • Tỷ lệ attack: {num_attacks / total:.2%}

{ip_list}

🛡️  LƯU Ý:
   • 🏠 = IP nội bộ | 🌐 = IP ngoại bộ
   • Whitelist hiện tại: {', '.join(WHITELIST_IPS)}
   • Ngưỡng cảnh báo: {ATTACK_FLOW_WARNING_THRESHOLD} attack flows hoặc {ATTACK_RATIO_WARNING * 100:.1f}% tỷ lệ

⏰ Thời gian phát hiện: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

    print(msg_text)

    try:
        msg = MIMEMultipart()
        msg["From"] = SENDER_EMAIL
        msg["To"] = RECEIVER_EMAIL
        msg["Subject"] = f"{risk_emoji} DDoS Alert [{risk_level}]: {num_attacks} attack flows detected"
        msg.attach(MIMEText(msg_text, "plain"))

        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.send_message(msg)
        server.quit()

        print("📬 Email cảnh báo đã được gửi thành công!")
    except Exception as e:
        print(f"❌ Gửi email thất bại: {e}")


# MAIN
if __name__ == "__main__":
    try:
        print("🚀 Bắt đầu phát hiện DDoS...")

        latest_file = get_latest_csv(INPUT_FOLDER)
        print(f"📄 File mới nhất: {latest_file}")

        result_df, suspected_attackers, whitelisted_attackers, total_attack_flows = detect_attack(latest_file,
                                                                                                  OUTPUT_FILE)

        # Tính toán số liệu thống kê
        total_flows = len(result_df)
        attack_flows = len(result_df[result_df["Prediction"] == 1])
        safe_flows = total_flows - attack_flows

        print(f"\n📊 KẾT QUẢ CUỐI CÙNG:")
        print(f"   • Tổng flows: {total_flows}")
        print(f"   • Attack flows: {attack_flows}")
        print(f"   • Safe flows: {safe_flows}")
        print(f"   • Suspected attackers: {len(suspected_attackers)}")
        print(f"   • Whitelisted IPs with attacks: {len(whitelisted_attackers)}")

        # ===== ĐÁNH GIÁ TÌNH TRẠNG BẢO MẬT MỚI =====
        risk_level, should_alert = evaluate_security_status(
            total_flows, attack_flows, suspected_attackers, whitelisted_attackers
        )

        # Quyết định gửi email dựa trên đánh giá tổng thể
        if should_alert:
            print(f"\n🚨 CẢNH BÁO: Hệ thống có dấu hiệu bất thường (Mức độ: {risk_level})")
            send_alert_email(attack_flows, safe_flows, total_flows, suspected_attackers, whitelisted_attackers,
                             risk_level)
        else:
            print(f"\n✅ Hệ thống an toàn (Mức độ: {risk_level})")

    except Exception as e:
        print(f"🚨 Lỗi: {e}")
        import traceback

        traceback.print_exc()