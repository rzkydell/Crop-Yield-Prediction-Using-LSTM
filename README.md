# 🌴 Prediksi Hasil Panen Kelapa Sawit Berbasis AI (LSTM)

Repositori ini berisi skrip pemrosesan data, pelatihan model *Deep Learning* (LSTM), dan aplikasi *dashboard web* interaktif (Streamlit) yang dikembangkan untuk memprediksi **Hasil Panen Sawit (Yield Ton/Ha)** berdasarkan data historis 12 bulan terakhir.

Proyek ini dibangun berdasarkan kombinasi Laporan Operasional Kebun (Produksi, Luas Lahan, Umur Pohon) dan Data Iklim (Curah Hujan & Hari Hujan), dengan mengimplementasikan teknik *Lagging Features* untuk menangkap memori fisiologis tanaman kelapa sawit terhadap cuaca di masa lampau.

---

## 🚀 Fitur Utama

- **Sequence Generation & Anti-Leakage Scaling**: Teknik pembagian data *time-series* secara granular per Blok/Afdeling dan mekanisme penskalaan data yang ketat (hanya me-*fit* pada data training).
- **Arsitektur Model LSTM Ganda**: 
  - *Layer 1*: 64 Unit LSTM
  - *Layer 2*: 32 Unit LSTM 
  - *Regulasi*: 20% Dropout
- **Performa Tinggi**: Model telah dievaluasi pada dataset uji (testing) *unseen data* dan mampu menekan angka kegagalan prediksi (*Mean Absolute Percentage Error / MAPE*) menembus batas psikologis **~20%** (Sangat akurat untuk konteks fluktuasi panen iklim ekstrem).
- **Kalkulator Cerdas Interaktif**: Dashboard ramah pemula yang dibangun dengan **Streamlit**, memungkinkan pengguna awam (non-teknis) untuk menyimulasikan profil kebun mereka dan mendapatkan estimasi panen secara instan.

---

## 📂 Struktur Direktori Proyek

```text
J:/prediksi hasil panen/
│
├── dataset/                  # Data operasional mentah & final
│   ├── Master_LSTM_Final_Large.csv 
│   └── Data_Dictionary.md      # Kamus resmi definisi variabel riset
├── models/                   # Bobot arsitektur AI (Diekstrak pasca training)
│   ├── lstm_yield_model.keras  # Jaringan Saraf Tiruan LSTM
│   ├── scaler_X.pkl            # Objek normalisasi input
│   └── scaler_y.pkl            # Objek normalisasi target
├── notebooks/                # Jupyter Notebooks pelengkap skripsi
│   ├── EDA_Analisis_Data.ipynb    # Eksplorasi Visual (Korelasi & Trend)
│   └── Model_Interpretation.ipynb # Analisis Error & Residual AI
├── output/                   # Grafik performa dan rekapan metrik
│   ├── hasil_training.txt      # Catatan metrik (MAPE, RMSE, Akurasi)
│   ├── hasil_prediksi.png      # Plot garis Aktual vs Prediksi
│   ├── scatter_prediksi.png    # Plot sebaran nilai linear
│   └── training_history.png    # Kurva performa Loss Function 
├── src/                      # Modul Source Code (Logika AI Modular)
│   ├── config.py               # Hyperparameters & Konstanta
│   ├── data_preprocessing.py   # Logika Normalisasi & Sebaris Waktu
│   ├── model_builder.py        # Topologi Pembentukan Layer Keras
│   └── evaluate.py             # Logika Skor Analisis Matematika
│
├── app.py                    # Aplikasi Front-End Kalkulator Cerdas (Streamlit)
├── train.py                  # Skrip Utama (Orkestrator/Runner) Pelatihan Model
├── .gitignore                # Perisai file rahasia dari repositori luar
├── requirements.txt          # Daftar spesifikasi Library Python
└── README.md                 # Dokumentasi Induk Proyek (Anda di sini)
```

---

## ⚙️ Persyaratan Instalasi Lingkungan (*Environment*)

Pastikan Anda telah menginstal **Python 3.8+**. Disarankan untuk menggunakan lingkungan virtual (Anaconda / venv).

1. Buka Terminal atau Command Prompt.
2. Arahkan ke folder proyek ini.
3. Instal seluruh *library* pendukung dengan menjalankan satu perintah berikut:

```bash
pip install -r requirements.txt
```

---

## 🏃 Cara Menjalankan Sistem

### 1. Melatih Ulang Model (Training)

Jika Anda memperoleh dataset baru atau ingin mengubah arsitektur parameter (seperti `BATCH_SIZE`, `LEARNING_RATE`, atau `EPOCHS` di dalam `src/config.py`), jalankan skrip pusat komando (Orkestrator):

```bash
python train.py
```

Skrip ini akan secara otomatis:
- Memproses data di folder `dataset/`.
- Membangun memori pada otak LSTM dan melakukan training.
- Mengekspor ulang *model* dan *scaler* ke folder `models/`.
- Menyimpan grafik performa terbaru dan file metrik di folder `output/`.

### 2. Membuka Aplikasi Kalkulator Panen (Web Dashboard)

Untuk membuka antarmuka antar pengguna interaktif berbasis web, pastikan terminal (*Anaconda Prompt* / PowerShell) Anda berada di lokasi folder yang sama, lalu ketik:

```bash
streamlit run app.py
```

Browser Anda akan secara otomatis membuka halaman web di **http://localhost:8501/**. Fitur kalkulator web interaktif sudah siap memfasilitasi penebakan angka produktivitas kebun!

---

## 📝 Catatan Tambahan
Proyek ini dipersiapkan sebagai bagian dari implementasi penelitian akademis. Konfigurasi *Lagging* periode hujan ekstensif (6-24 bulan) merupakan kontribusi agronomi signifikan dalam model komputasi yang direalisasikan pada tesis ini. Dilarang menduplikasi metodologi tanpa penyertaan sitasi (*citation*).

BY rzkydell