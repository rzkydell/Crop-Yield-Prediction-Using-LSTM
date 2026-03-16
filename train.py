# -*- coding: utf-8 -*-
import os
import joblib
from src.config import DATA_PATH, MODEL_DIR, EPOCHS, BATCH_SIZE, TIME_STEPS, LSTM_UNITS_1, LSTM_UNITS_2
from src.data_preprocessing import prepare_data
from src.model_builder import build_dual_lstm_model, get_callbacks
from src.evaluate import evaluate_and_plot

def main():
    print("=" * 70)
    print("🚀 TRAINING JARINGAN SARAF TIRUAN (LSTM) PREDIKSI YIELD SAWIT 🚀")
    print("=" * 70)
    
    # 1. Data Preparation
    print("[1/5] Memuat dan memproses dataset...")
    try:
        (X_train, y_train), (X_val, y_val), (X_test, y_test), scaler_X, scaler_y = prepare_data(DATA_PATH)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return
        
    print(f"      Train seq: {X_train.shape[0]} | Val seq: {X_val.shape[0]} | Test seq: {X_test.shape[0]}")
    
    # 2. Build Model
    print("\n[2/5] Membangun Arsitektur LSTM...")
    num_features = X_train.shape[2]
    model = build_dual_lstm_model(TIME_STEPS, num_features)
    model.summary()
    
    # 3. Training
    print("\n[3/5] Memulai Proses Training (Pelatihan Memori)...")
    callbacks = get_callbacks()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )
    
    # 4. Evaluation
    print("\n[4/5] Mengeksekusi Evaluasi pada Data Uji (Testing)...")
    accuracy, mape = evaluate_and_plot(model, history, X_test, y_test, scaler_y)
    
    # 5. Export Artifacts
    print("\n[5/5] Mengekspor Artefak Model untuk Produksi (Aplikasi Web)...")
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
        
    model.save(f"{MODEL_DIR}/lstm_yield_model.keras")
    joblib.dump(scaler_X, f"{MODEL_DIR}/scaler_X.pkl")
    joblib.dump(scaler_y, f"{MODEL_DIR}/scaler_y.pkl")
    
    print("      [OK] Berhasil menyimpan .keras dan .pkl")
    
    print("\n" + "=" * 70)
    print("✨ TRAINING, EVALUATION & EXPORT SELESAI ✨")
    print("=" * 70)
    print(f"  Model    : Dual-Layer LSTM ({LSTM_UNITS_1} -> {LSTM_UNITS_2})")
    print(f"  MAPE     : {mape:.2f}%")
    print(f"  Akurasi  : {accuracy:.2f}%")
    if mape < 20:
        print("  Status   : [TARGET TERCAPAI] Performa model sudah sangat baik!")
    print("=" * 70)

if __name__ == "__main__":
    main()
