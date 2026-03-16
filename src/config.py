import os

# Hyperparameters & Paths
DATA_PATH     = "dataset/Master_LSTM_Final_Large.csv"
MODEL_DIR     = "models"
OUTPUT_DIR    = "output"

TIME_STEPS    = 12
EPOCHS        = 300
BATCH_SIZE    = 8
LSTM_UNITS_1  = 64
LSTM_UNITS_2  = 32
DROPOUT_RATE  = 0.2
LEARNING_RATE = 0.001
SEED          = 42

FEATURE_COLS = [
    "Production_Kg", "Area_Ha", "Palm_Age",
    "Rainfall_CH", "RainyDays_HH",
    "CH_Lag_6", "HH_Lag_6",
    "CH_Lag_12", "HH_Lag_12",
    "CH_Lag_24", "HH_Lag_24",
    "Month_Sin", "Month_Cos",
    "Yield_Ton_Ha"
]
TARGET_COL = "Yield_Ton_Ha"
