import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import sys
sys.path.append("utils")
from utils import load_csv
import os
from datetime import datetime, timedelta
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

# 🔮 Dự đoán giá đóng cửa theo ngày
st.subheader("Dự đoán giá đóng cửa theo ngày")
input_date = st.date_input("Chọn ngày cần dự đoán")

model_path = f"../model/{symbol}_lstm_model.keras"
data_path = f"../data/clean/{symbol}.csv"
scaler_path = f"../model/{symbol}_scaler.pkl"

if os.path.exists(model_path) and os.path.exists(data_path) and os.path.exists(scaler_path):
    try:
        model = load_model(model_path)
        df = pd.read_csv(data_path)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")

        last_date = df["date"].max().date()
        target_date = input_date

        if target_date <= last_date:
            st.warning(f"Ngày {target_date} đã có trong dữ liệu. Vui lòng chọn ngày **sau** {last_date}.")
        elif len(df) < 100:
            st.warning("Không đủ dữ liệu để dự đoán (cần ít nhất 100 ngày).")
        else:
            # Dự đoán
            data = df["close"].values.reshape(-1, 1)
            scaler = joblib.load(scaler_path)
            data_scaled = scaler.transform(data)

            look_back = 100
            current_input = data_scaled[-look_back:]

            days_to_predict = (target_date - last_date).days
            predictions = []

            for _ in range(days_to_predict):
                input_batch = np.expand_dims(current_input, axis=0)
                pred_scaled = model.predict(input_batch, verbose=0)[0][0]
                predictions.append(pred_scaled)
                current_input = np.append(current_input, [[pred_scaled]], axis=0)[-look_back:]

            # Chuyển về giá gốc
            predictions_unscaled = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()

            # Tạo kết quả
            result_dates = [last_date + timedelta(days=i+1) for i in range(days_to_predict)]
            result_df = pd.DataFrame({
                "date": result_dates,
                "predicted_close": predictions_unscaled
            })

            predicted_price = result_df[result_df["date"] == target_date]["predicted_close"].values
            if predicted_price.size > 0:
                st.success(f"✅ Giá đóng cửa dự đoán cho ngày {target_date.strftime('%d/%m/%Y')} là **{predicted_price[0]:,.2f}** VND.")
            else:
                st.warning("Không tìm thấy kết quả dự đoán cho ngày đã chọn.")

    except Exception as e:
        st.error(f"Lỗi khi dự đoán: {e}")
else:
    st.info("⛔ Chưa có đủ dữ liệu/mô hình để dự đoán cho cổ phiếu này.")
