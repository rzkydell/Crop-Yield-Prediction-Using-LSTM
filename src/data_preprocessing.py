import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from .config import FEATURE_COLS, TARGET_COL, SEED, TIME_STEPS

def load_and_clean_data(filepath):
    """Memuat CSV, membuang duplikat, dan mengurutkan berdasarkan entitas & waktu."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset tidak ditemukan di {filepath}")
        
    df = pd.read_csv(filepath)
    df = df.drop_duplicates()
    
    # Sort data
    df = df.sort_values(by=["Entity_ID", "Year", "Month"]).reset_index(drop=True)
    
    # Feature Engineering (Bulan ke Cyclical)
    df["Month_Sin"] = np.sin(2 * np.pi * df["Month"] / 12)
    df["Month_Cos"] = np.cos(2 * np.pi * df["Month"] / 12)
    
    return df

def get_fitted_scalers(df):
    """
    Fit scalers *hanya* pada 70% bagian awal dari setiap Entity_ID
    untuk mencegah Data Leakage.
    """
    train_data_list = []
    
    for entity_id, group in df.groupby("Entity_ID"):
        n_rows = len(group)
        if n_rows > TIME_STEPS:
            n_train = int(n_rows * 0.70)
            train_portion = group.iloc[:n_train]
            train_data_list.append(train_portion)
            
    if not train_data_list:
        raise ValueError("Data tidak cukup untuk mem-fit scaler!")
        
    df_train_portion = pd.concat(train_data_list, ignore_index=True)
    
    scaler_X = MinMaxScaler()
    scaler_X.fit(df_train_portion[FEATURE_COLS].values)
    
    scaler_y = MinMaxScaler()
    scaler_y.fit(df_train_portion[[TARGET_COL]].values)
    
    return scaler_X, scaler_y

def create_sequences_from_scaled(df, scaler_X, scaler_y):
    """Membuat sekues data berukuran TIME_STEPS secara terisolasi per Entity"""
    X_scaled = scaler_X.transform(df[FEATURE_COLS].values)
    y_scaled = scaler_y.transform(df[[TARGET_COL]].values)
    
    X_list, y_list = [], []
    
    for entity_id, group in df.groupby("Entity_ID"):
        idx = group.index
        group_X = X_scaled[idx]
        group_y = y_scaled[idx]
        
        n_rows = len(group_X)
        for i in range(n_rows - TIME_STEPS):
            X_list.append(group_X[i : i + TIME_STEPS])
            y_list.append(group_y[i + TIME_STEPS])
            
    return np.array(X_list), np.array(y_list)

def prepare_data(filepath):
    """Fungsi utama pipeline data preprocessing"""
    df = load_and_clean_data(filepath)
    scaler_X, scaler_y = get_fitted_scalers(df)
    
    X_all, y_all = create_sequences_from_scaled(df, scaler_X, scaler_y)
    
    # 70% / 15% / 15% Randomized Split
    X_train, X_tmp, y_train, y_tmp = train_test_split(X_all, y_all, test_size=0.30, random_state=SEED)
    X_val, X_test, y_val, y_test   = train_test_split(X_tmp, y_tmp, test_size=0.50, random_state=SEED)
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test), scaler_X, scaler_y
