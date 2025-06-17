# VNStokePredict

Dự án dự đoán giá cổ phiếu ngắn hạn bằng mô hình LSTM và XGBoost.

## 🧩 Cấu trúc thư mục

- `src/`: mã nguồn crawl, xử lý, ứng dụng Streamlit
- `notebooks/`: các notebook quá trình xử lý dữ liệu, huấn luyện mô hình
- `utils.py`, `predictor.py`: các hàm hỗ trợ và xử lý dự đoán
- `app.py`: giao diện người dùng bằng Streamlit

## 🛠 Cài đặt

### 1. Clone project và tạo môi trường

```bash
git clone <link-repo>
cd VNStokePredict
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Cài đặt ChromeDriver (nếu crawl lại dữ liệu)
Tải tại: https://sites.google.com/a/chromium.org/chromedriver/downloads
Copy vào thư mục browserDrivers/

### 3. Chạy ứng dụng
cd src
streamlit run app.py

## 📌 Ghi chú
Dữ liệu mẫu được cung cấp trong thư mục data/
Nếu cần thu thập lại dữ liệu, chạy file crawl.ipynb trong crawl/
Nếu cần huấn luyện lại, chạy các notebook theo thứ tự trong notebooks/