# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import os

# =============================================================================
# CONFIG & LAYOUT
# =============================================================================
st.set_page_config(
    page_title="Kalkulator Cerdas Panen Sawit",
    page_icon="🌴",
    layout="centered",
    initial_sidebar_state="collapsed" # Sembunyikan sidebar agar lebih fokus
)

# Custom CSS for modern UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #2E7D32;
        margin-bottom: 0rem;
        text-align: center;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 2rem;
        text-align: center;
    }
    .stNumberInput label, .stSlider label, .stSelectbox label {
        font-size: 1.05rem !important;
        font-weight: 600 !important;
    }
    .metric-card {
        background: linear-gradient(135deg, #F0FDF4 0%, #DCFCE7 100%);
        border: 2px solid #22C55E;
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
        margin-top: 20px;
        margin-bottom: 20px;
    }
    .metric-value {
        font-size: 3.5rem;
        font-weight: 900;
        color: #15803D;
        margin-top: 10px;
        margin-bottom: 10px;
    }
    .metric-label {
        font-size: 1.2rem;
        font-weight: 700;
        color: #166534;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .metric-subtext {
        font-size: 1rem;
        font-weight: 500;
        color: #374151;
        margin-top: 10px;
    }
    .stButton>button {
        width: 100%;
        background-color: #16A34A;
        color: white;
        font-size: 1.2rem;
        font-weight: 700;
        padding: 1rem;
        border-radius: 10px;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #15803D;
        color: white;
        transform: translateY(-2px);
        box-shadow: 0 4px 6px -1px rgba(21, 128, 61, 0.4);
    }
    .info-box {
        background-color: #F8FAFC;
        border-left: 4px solid #3B82F6;
        padding: 15px;
        border-radius: 5px;
        margin-top: 10px;
        margin-bottom: 20px;
    }
    .info-box p {
        margin: 0;
        color: #334155;
        font-size: 0.95rem;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# LOAD ARTIFACTS
# =============================================================================
@st.cache_resource
def load_artifacts():
    try:
        model = tf.keras.models.load_model("models/lstm_yield_model.keras")
        scaler_X = joblib.load("models/scaler_X.pkl")
        scaler_y = joblib.load("models/scaler_y.pkl")
        return model, scaler_X, scaler_y
    except Exception as e:
        return None, None, str(e)

model, scaler_X, scaler_y = load_artifacts()

# =============================================================================
# CONSTANTS & UTILS
# =============================================================================
FEATURE_COLS = [
    "Production_Kg", "Area_Ha", "Palm_Age",
    "Rainfall_CH", "RainyDays_HH",
    "CH_Lag_6", "HH_Lag_6",
    "CH_Lag_12", "HH_Lag_12",
    "CH_Lag_24", "HH_Lag_24",
    "Month_Sin", "Month_Cos",
    "Yield_Ton_Ha"
]

def generate_background_sequence(base_prod, base_area, age, base_rain, base_rainy_days):
    """
    Simulasi cerdas data kebun ke belakang 12 bulan. 
    Masyarakat awam tidak perlu mengisi data per bulan.
    """
    data = []
    base_yield = base_prod / (base_area * 1000)
    for m in range(1, 13):
        month_sin = np.sin(2 * np.pi * m / 12)
        month_cos = np.cos(2 * np.pi * m / 12)
        
        # Penambahan sedikit fluktuasi natural (seasonality)
        noise_factor = 1.0 + np.random.uniform(-0.05, 0.05)
        
        row = [
            base_prod * noise_factor,     
            base_area,                    
            age,                          
            base_rain * noise_factor,     
            int(base_rainy_days),         
            base_rain * 0.9,  # Lag CH (Hujan lampau)
            int(max(base_rainy_days - 2, 0)),     
            base_rain * 1.05, 
            int(base_rainy_days),     
            base_rain * 0.95, 
            int(max(base_rainy_days - 1, 0)),     
            month_sin,                    
            month_cos,                    
            base_yield * noise_factor     
        ]
        data.append(row)
    return pd.DataFrame(data, columns=FEATURE_COLS)

# =============================================================================
# APP UI (Layperson Friendly)
# =============================================================================

st.markdown('<p class="main-header">🌴 Kalkulator Panen Kebun Sawit</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Sistem Cerdas Penebak Hasil Kebun Bulan Depan Berbasis AI</p>', unsafe_allow_html=True)

if type(scaler_y) == str:
    st.error(f"Sistem gagal mengakses file memori AI. Error: {scaler_y}")
    st.info("Pastikan Anda sudah menempatkan file sistem AI di dalam folder 'models/'.")
    st.stop()

st.markdown("""
<div class="info-box">
    <p>💡 <b>Cara Penggunaan:</b> Geser nilai batas kebun yang Anda miliki saat ini. Sistem Kecerdasan Buatan (AI) kami otomatis memprediksikan seberapa besar panen Anda pada bulan berikutnya.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("### 📝 Masukkan Kondisi Kebun Saat Ini")

col1, col2 = st.columns(2)

with col1:
    input_area = st.number_input(
        "Luas Lahan Kebun (Hektar)", 
        min_value=1.0, value=50.0, step=1.0,
        help="Luas total kebun sawit yang Anda kelola."
    )
    
    input_prod = st.number_input(
        "Rata-rata Produksi Bulanan (Kg)", 
        min_value=1000, value=75000, step=5000,
        help="Berapa kilogram buah sawit yang biasanya Anda dapatkan per bulan secara umum."
    )

with col2:
    input_age = st.slider(
        "Umur Rata-rata Pohon (Tahun)", 
        min_value=3, max_value=30, value=12,
        help="Rentang usia normal produktif adalah 8-15 tahun."
    )
    
    # Simple slider for rain condition
    weather_condition = st.select_slider(
        "Kondisi Cuaca (Curah Hujan)",
        options=["Sangat Kering", "Kering", "Normal", "Lembap", "Sangat Basah"],
        value="Normal",
        help="Pilih kondisi rata-rata curah hujan di daerah Anda."
    )

# Konversi Cuaca ke metrik CH dan HH untuk AI
weather_mapping = {
    "Sangat Kering": (50.0, 4),
    "Kering": (100.0, 8),
    "Normal": (200.0, 15),
    "Lembap": (300.0, 20),
    "Sangat Basah": (400.0, 25)
}
base_rain_ch, base_rain_hh = weather_mapping[weather_condition]

st.markdown("<br>", unsafe_allow_html=True)

# Generate Button
predict_btn = st.button("🌟 HITUNG PREDIKSI PANEN BULAN DEPAN")

if predict_btn:
    with st.spinner("AI Sedang Menganalisis Kondisi Kebun Anda..."):
        
        # 1. Generate internal representation (12 months sequence)
        df_input = generate_background_sequence(
            base_prod=input_prod, 
            base_area=input_area, 
            age=input_age,
            base_rain=base_rain_ch,
            base_rainy_days=base_rain_hh
        )
        
        # 2. Scale Input Data
        X_input_scaled = scaler_X.transform(df_input.values)
        X_seq = np.array([X_input_scaled])
        
        # 3. Predict & Inverse Transform
        y_pred_scaled = model.predict(X_seq, verbose=0)
        y_pred_actual = scaler_y.inverse_transform(y_pred_scaled)[0][0]
        
        # Avoid negative yield
        final_yield = max(y_pred_actual, 0.01)
        
        # Compute real estimated production 
        est_prod_kg = final_yield * input_area * 1000
        
        # Assume a standard price (e.g. Rp 2,500/Kg) for illustration
        est_rupiah = est_prod_kg * 2500
        
        # Display Results Beautifully
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Prediksi Produktivitas</div>
            <div class="metric-value">{final_yield:.2f} <span style="font-size: 1.5rem;">Ton / Hektar</span></div>
            <hr style="border-top: 2px dashed #86EFAC; margin: 20px 0;">
            <div class="metric-subtext">Estimasi Total Panen di Lahan Anda: <b>{est_prod_kg:,.0f} Kg</b></div>
            <div class="metric-subtext">Potensi Nilai Jual Kasar (Asumsi Rp 2.500/kg): <b>Rp {est_rupiah:,.0f}</b></div>
        </div>
        """, unsafe_allow_html=True)
        
        # Add automated AI conclusion text
        if final_yield > 1.8:
            conclusion = "Kebun Anda diprediksi berada dalam masa puncak panen yang **SANGAT PRODUKTIF**. Pastikan jumlah tenaga pemanen dan akses truk angkutan memadai."
        elif final_yield > 1.2:
            conclusion = "Kebun Anda diprediksi menghasilkan panen **NORMAL / STABIL**. Kondisi cuaca dan usia pohon saat ini sangat mendukung kelangsungan panen harian."
        else:
            conclusion = "Kebun Anda diprediksi mengalami panen yang **CUKUP RENDAH**. Hal ini bisa jadi wajar jika pohon terlalu muda/tua, iklim terlalu kering, atau sedang memasuki siklus trek berbuah."
            
        st.info(f"🤖 **Kesimpulan AI:** {conclusion}")
