import os
import pandas as pd
import json
import logging

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def _check_file_exists(file_path: str):
    """Kiểm tra xem file có tồn tại không"""
    if not os.path.exists(file_path):
        logging.error(f"Không tìm thấy file: {file_path}")
        raise FileNotFoundError(f"File không tồn tại: {file_path}")


def load_csv(file_path: str, encoding: str = "utf-8") -> pd.DataFrame:
    """
    Load file CSV thành DataFrame.
    """
    _check_file_exists(file_path)

    try:
        df = pd.read_csv(file_path, encoding=encoding)
        logging.info(f"✅ Load CSV thành công: {file_path} | shape = {df.shape}")
        return df
    except Exception as e:
        logging.exception(f"❌ Lỗi khi đọc file CSV: {file_path}")
        raise e


def load_excel(file_path: str, sheet_name: str = 0) -> pd.DataFrame:
    """
    Load file Excel thành DataFrame.
    """
    _check_file_exists(file_path)

    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        logging.info(f"✅ Load Excel thành công: {file_path} | shape = {df.shape}")
        return df
    except Exception as e:
        logging.exception(f"❌ Lỗi khi đọc file Excel: {file_path}")
        raise e


def load_json(file_path: str) -> dict:
    """
    Load file JSON thành dict.
    """
    _check_file_exists(file_path)

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logging.info(f"✅ Load JSON thành công: {file_path}")
        return data
    except Exception as e:
        logging.exception(f"❌ Lỗi khi đọc file JSON: {file_path}")
        raise e


def save_data(df: pd.DataFrame, path: str):
    """
    Lưu DataFrame thành file CSV, tự tạo thư mục nếu chưa tồn tại.
    """
    dir_path = os.path.dirname(path)
    os.makedirs(dir_path, exist_ok=True)

    try:
        df.to_csv(path, index=False)
        logging.info(f"💾 Đã lưu dữ liệu thành công: {path}")
    except Exception as e:
        logging.exception(f"❌ Lỗi khi lưu dữ liệu: {path}")
        raise e
