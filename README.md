# Machine-Learning-DDOS
# 🛡️ Machine Learning DDoS Detection

This project uses a Machine Learning model (Random Forest) to detect DDoS attacks from network traffic. It includes tools for packet capturing, preprocessing, labeling, and attack detection.

---

## 📦 1. Development Environment Setup

### 🔧 Install PyCharm
- Download and install from: [https://www.jetbrains.com/pycharm/download](https://www.jetbrains.com/pycharm/download)

### 🐍 Install Python (if not already installed)
- Recommended version: **Python 3.8+**
- Download from: [https://www.python.org/downloads](https://www.python.org/downloads)

---

## 🧪 2. Install Required Python Libraries

You can install all required libraries using:

```bash
pip install -r requirements.txt
🧰 3. Tools for Data Collection and Processing
📡 Install Wireshark
Used to capture raw network traffic.

Download from: https://www.wireshark.org/download.html

Save captured packets as .pcap file

🧪 Install CICFlowMeter
This tool converts .pcap files to .csv format for ML processing.

Download and setup: https://github.com/ahlashkari/CICFlowMeter

🛠 How to Use:
Open CICFlowMeter

Select your .pcap file

Convert to .csv (output will be saved in bin/ folder)

🏷️ 4. Label the Data
Run the script to add labels or clean the CSV file for model input:

bash
Sao chép
Chỉnh sửa
python label.py
Ensure the output .csv is placed inside the Data/ folder as expected by the script.

📧 5. Configure Email Alert System (Gmail)
Open the file check_status.py and find the section to configure Gmail credentials:

python
Sao chép
Chỉnh sửa
sender_email = "your_sender_email@gmail.com"
receiver_email = "your_receiver_email@gmail.com"
app_password = "your_gmail_app_password"
🔐 How to get Gmail App Password
Go to: https://myaccount.google.com/security

Enable 2-Step Verification

Scroll down to App passwords

Generate a password for “Mail” and “Windows Computer” (or your choice)

Copy and paste the password into check_status.py

⚠️ Never share your app password publicly

🔍 6. Run DDoS Detection
Once everything is configured and labeled data is available in the Data/ folder, run:

bash
Sao chép
Chỉnh sửa
python check_status.py
The script will analyze the data and send an email alert if an attack is detected.

📁 Folder Structure
bash
Sao chép
Chỉnh sửa
├── Data/                    # Contains processed CSV files
├── model/                   # Trained Random Forest model (.joblib)
├── label.py                 # Script to label and clean data
├── check_status.py          # DDoS detection and email alert
├── requirements.txt         # List of required Python packages
├── README.md                # This file
✅ System Requirements
Windows or Linux

Python 3.8+

PyCharm or VSCode

Wireshark

CICFlowMeter

📬 Contact
For questions or issues, open an issue on GitHub or contact the project owner.
