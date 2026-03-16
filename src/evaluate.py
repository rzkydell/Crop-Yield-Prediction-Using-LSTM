import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from .config import OUTPUT_DIR

def evaluate_and_plot(model, history, X_test, y_test_scaled, scaler_y):
    """Menghitung metrik (MAE, RMSE, R2, MAPE) dan membuat grafik"""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    # Predict
    y_pred_scaled = model.predict(X_test)
    
    # Inverse Transform
    y_test_actual = scaler_y.inverse_transform(y_test_scaled.reshape(-1, 1))
    y_pred_actual = scaler_y.inverse_transform(y_pred_scaled)
    
    # Metrik
    mae  = mean_absolute_error(y_test_actual, y_pred_actual)
    rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_actual))
    r2   = r2_score(y_test_actual, y_pred_actual)
    
    # MAPE & Accuracy
    mask = y_test_actual != 0
    y_t_safe = y_test_actual[mask]
    y_p_safe = y_pred_actual[mask]
    
    mape = np.mean(np.abs((y_t_safe - y_p_safe) / y_t_safe)) * 100
    accuracy = 100.0 - mape
    
    print("\n" + "=" * 70)
    print("STEP 8 -- Model Evaluation (Test Set - Scaled Back to Actual)")
    print("=" * 70)
    print(f"  MAE       : {mae:.4f} Ton/Ha")
    print(f"  RMSE      : {rmse:.4f} Ton/Ha")
    print(f"  R-Squared : {r2:.4f}")
    print(f"  MAPE      : {mape:.2f}%")
    print(f"  AKURASI   : {accuracy:.2f}%")
    
    _plot_charts(history, y_test_actual, y_pred_actual)
    return accuracy, mape

def _plot_charts(history, y_test_actual, y_pred_actual):
    # Chart 1: Line plot
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(y_test_actual, label="Aktual (Ton/Ha)", marker="o", color="#1f77b4", linestyle="-", linewidth=2)
    ax.plot(y_pred_actual, label="Prediksi", marker="x", color="#ff7f0e", linestyle="--", linewidth=2)
    ax.set_title("Aktual vs Prediksi - Produksi Kelapa Sawit")
    ax.set_xlabel("Test Sequence Index")
    ax.set_ylabel("Yield (Ton/Ha)")
    ax.legend()
    plt.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/hasil_prediksi.png", dpi=200, bbox_inches="tight")
    
    # Chart 2: Loss History
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.plot(history.history["loss"], label="Training Loss (MSE)", color="#2ca02c", linewidth=2)
    ax2.plot(history.history["val_loss"], label="Validation Loss (MSE)", color="#d62728", linewidth=2)
    ax2.set_title("Kurva Pembelajaran: Training & Validation Loss")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Mean Squared Error (Scaled)")
    ax2.legend()
    plt.tight_layout()
    fig2.savefig(f"{OUTPUT_DIR}/training_history.png", dpi=200, bbox_inches="tight")
    
    # Chart 3: Scatter Plot
    fig3, ax3 = plt.subplots(figsize=(7, 7))
    ax3.scatter(y_test_actual, y_pred_actual, alpha=0.7, color="#9467bd", edgecolor="w", s=60)
    min_val = min(np.min(y_test_actual), np.min(y_pred_actual))
    max_val = max(np.max(y_test_actual), np.max(y_pred_actual))
    ax3.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label="Garis Ideal (Y = X)")
    ax3.set_title("Korelasi Nilai Aktual vs Prediksi")
    ax3.set_xlabel("Aktual")
    ax3.set_ylabel("Prediksi")
    ax3.legend()
    plt.tight_layout()
    fig3.savefig(f"{OUTPUT_DIR}/scatter_prediksi.png", dpi=200, bbox_inches="tight")
