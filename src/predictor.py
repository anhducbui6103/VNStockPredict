# controllers/predictor.py

import os
import joblib
import numpy as np
import pandas as pd
from datetime import timedelta
from tensorflow.keras.models import load_model

def predict_lstm(symbol, input_date):
    model_path = f"../model/{symbol}_lstm_model.keras"
    scaler_path = f"../model/{symbol}_scaler.pkl"
    data_path = f"../data/clean/{symbol}.csv"

    if not all(os.path.exists(p) for p in [model_path, scaler_path, data_path]):
        return None, "Thiếu model hoặc dữ liệu."

    df = pd.read_csv(data_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    last_date = df["date"].max().date()

    if input_date <= last_date:
        return None, f"Ngày {input_date} đã có trong dữ liệu. Vui lòng chọn ngày sau {last_date}."
    if len(df) < 100:
        return None, "Không đủ dữ liệu để dự đoán (cần ít nhất 100 ngày)."

    data = df["close"].values.reshape(-1, 1)
    scaler = joblib.load(scaler_path)
    model = load_model(model_path)

    data_scaled = scaler.transform(data)
    look_back = 100
    current_input = data_scaled[-look_back:]
    days_to_predict = (input_date - last_date).days
    predictions = []

    for _ in range(days_to_predict):
        input_batch = np.expand_dims(current_input, axis=0)
        pred_scaled = model.predict(input_batch, verbose=0)[0][0]
        predictions.append(pred_scaled)
        current_input = np.append(current_input, [[pred_scaled]], axis=0)[-look_back:]

    predictions_unscaled = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
    result_date = last_date + timedelta(days=days_to_predict)
    return predictions_unscaled[-1], None


def predict_xgb(symbol, input_date):
    model_path = f"../model/{symbol}_xgb_model.pkl"
    feature_path = f"../data/features/{symbol}_features.csv"

    if not os.path.exists(model_path) or not os.path.exists(feature_path):
        return None, "Thiếu model hoặc dữ liệu."

    # Load dữ liệu đặc trưng
    df_feat = pd.read_csv(feature_path)
    df_feat["date"] = pd.to_datetime(df_feat["date"])
    df_feat = df_feat.sort_values("date")
    last_date = df_feat["date"].max().date()

    if input_date <= last_date:
        return None, f"Ngày {input_date} đã có trong dữ liệu. Vui lòng chọn ngày sau {last_date}."

    # Lấy dòng mới nhất để dự đoán bước tiếp theo
    input_row = df_feat[df_feat["date"] == df_feat["date"].max()]
    if input_row.empty:
        return None, "Không tìm thấy dữ liệu đặc trưng."

    model = joblib.load(model_path)

    # Lấy đúng các cột đặc trưng từ model (scikit-learn 1.0+ hỗ trợ feature_names_in_)
    if hasattr(model, "feature_names_in_"):
        feature_cols = model.feature_names_in_
    else:
        # Nếu model không có thông tin cột đầu vào, loại trừ các cột không dùng
        feature_cols = [col for col in df_feat.columns if col not in ["date", "close"]]

    try:
        X_input = input_row[feature_cols].values
        prediction = model.predict(X_input)[0]
        return prediction, None
    except Exception as e:
        return None, f"Lỗi khi xử lý dữ liệu đầu vào: {e}"