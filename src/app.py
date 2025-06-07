import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
sys.path.append("utils")
from utils import load_csv
from predictor import predict_lstm, predict_xgb

# Cấu hình trang
st.set_page_config(page_title="Stock Viewer", layout="wide")
st.title("📈 Ứng dụng phân tích cổ phiếu")

# Dropdown chọn cổ phiếu
stock_map = {"FPT": "fpt", "VNM": "vnm", "HPG": "hpg"}
stock_choice = st.selectbox("Chọn cổ phiếu", list(stock_map.keys()))

# Load dữ liệu
symbol = stock_map[stock_choice]
file_path = f"../data/clean/{symbol}.csv"

try:
    df = load_csv(file_path)
except Exception as e:
    st.error(f"❌ Lỗi khi load dữ liệu: {e}")
    st.stop()

# Hiển thị bảng dữ liệu
st.subheader(f"Dữ liệu cổ phiếu {stock_choice}")
st.dataframe(df)

# Biểu đồ Close theo thời gian
st.subheader("Biểu đồ giá đóng cửa theo thời gian")
if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'])
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df['date'], df['close'], label="Close", linewidth=2)
    ax.set_xlabel("Thời gian")
    ax.set_ylabel("Giá đóng cửa")
    ax.set_title(f"Giá đóng cửa của {stock_choice}")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)
else:
    st.warning("Không tìm thấy cột 'date' trong dữ liệu.")

# Dự đoán giá đóng cửa
st.subheader("Dự đoán giá đóng cửa theo ngày")
input_date = st.date_input("Chọn ngày cần dự đoán")
model_type = st.selectbox("Chọn mô hình dự đoán", ["LSTM", "XGB"])

if model_type == "LSTM":
    price, error = predict_lstm(symbol, input_date)
    if error:
        st.warning(error)
    else:
        st.success(f"✅ [LSTM] Giá dự đoán cho ngày {input_date.strftime('%d/%m/%Y')} là **{price:,.2f}** nghìn VND.")
elif model_type == "XGB":
    if symbol == "fpt":
        st.warning("⚠️ Mô hình XGB không hoạt động tốt với cổ phiếu FPT. Kết quả có thể sai lệch lớn.")

    price, error = predict_xgb(symbol, input_date)
    if error:
        st.warning(error)
    else:
        st.success(f"✅ [XGB] Giá dự đoán cho ngày {input_date.strftime('%d/%m/%Y')} là **{price:,.2f}** nghìn VND.")
