import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.append("utils")
from utils import load_csv
import os
from datetime import datetime
from tensorflow.keras.models import load_model
import numpy as np

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
    st.error(f"Lỗi khi load dữ liệu: {e}")
    st.stop()

# Hiển thị bảng dữ liệu
st.subheader(f"Dữ liệu cổ phiếu {stock_choice}")
st.dataframe(df)

# Biểu đồ Close theo thời gian
st.subheader("📊 Biểu đồ giá đóng cửa theo thời gian")
if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')

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

# Biểu đồ giá dự đoán vs giá thực tế (nếu có)
if 'predict' in df.columns:
    st.subheader("🤖 So sánh giá dự đoán và giá thực tế")

    fig2, ax2 = plt.subplots(figsize=(12, 6))
    ax2.plot(df['date'], df['close'], label="Thực tế", linewidth=2)
    ax2.plot(df['date'], df['predict'], label="Dự đoán", linestyle="--")
    ax2.set_xlabel("Thời gian")
    ax2.set_ylabel("Giá đóng cửa")
    ax2.set_title("Giá đóng cửa: Thực tế vs Dự đoán")
    ax2.legend()
    st.pyplot(fig2)

# Dự đoán giá theo ngày
st.subheader("🔮 Dự đoán giá đóng cửa theo ngày")

input_date = st.date_input("Chọn ngày cần dự đoán")

model_path = f"../model/{symbol}_lstm_model.keras"
if os.path.exists(model_path):
    try:
        model = load_model(model_path)

        st.info("📌 Mô hình LSTM sẽ dự đoán lặp từng bước đến ngày bạn chọn (multi-step prediction).")

        lookback = 100
        last_date = df['date'].max().date()

        if input_date <= last_date:
            st.warning(f"Ngày {input_date} đã có trong dữ liệu. Vui lòng chọn ngày **sau** {last_date}.")
        elif len(df) >= lookback:
            # Sắp xếp lại theo ngày tăng dần
            df_sorted = df.sort_values('date')

            # Chuẩn bị dữ liệu đầu vào ban đầu
            recent_data = df_sorted[['close']].values[-lookback:]
            input_seq = recent_data.copy()

            steps = (input_date - last_date).days
            for _ in range(steps):
                input_array = input_seq.reshape(1, lookback, 1)  # Đảm bảo shape là (1, lookback, 1)
                next_pred = model.predict(input_array, verbose=0)[0][0]
                # Cập nhật chuỗi đầu vào cho bước tiếp theo
                input_seq = np.append(input_seq[1:], [[next_pred]], axis=0)

            st.success(f"✅ Giá đóng cửa dự đoán cho {input_date} là: {next_pred:.2f}")
        else:
            st.warning("Không đủ dữ liệu để dự đoán (cần ít nhất 100 ngày).")

    except Exception as e:
        st.error(f"Lỗi khi dự đoán bằng mô hình LSTM: {e}")
else:
    st.info("Chưa có mô hình LSTM cho cổ phiếu này.")