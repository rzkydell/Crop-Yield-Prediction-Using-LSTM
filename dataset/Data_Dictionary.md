# Data Dictionary - Prediksi Yield Kelapa Sawit

File ini merupakan kamus resmi untuk mendeskripsikan setiap variabel yang terdapat di dalam dataset utama (`Master_LSTM_Final_Large.csv`). Kamus ini sangat penting sebagai referensi dalam penulisan Bab 3 (Metodologi) atau Bab 4 (Hasil Analisis) pada laporan Skripsi Anda.

## 1. Variabel Identitas
- **Entity_ID**  
  *Deskripsi*: Kode unik yang mewakili kombinasi Afdeling dan Blok kebun. Digunakan sebagai pemisah entitas historis agar *time-series* antar area kebun tidak bercampur.
- **Year**  
  *Deskripsi*: Tahun perekaman data operasional.
- **Month**  
  *Deskripsi*: Bulan perekaman data (1-12).

## 2. Variabel Agronomi (Input)
- **Production_Kg**  
  *Deskripsi*: Total produksi Tandan Buah Segar (TBS) kelapa sawit dalam satuan Kilogram (Kg) pada bulan tersebut di blok tertentu.
- **Area_Ha**  
  *Deskripsi*: Luas areal tertanam kelapa sawit pada blok terkait dalam satuan Hektare (Ha).
- **Palm_Age**  
  *Deskripsi*: Umur ekonomis tanaman kelapa sawit (tahun). Rentang produktif optimal biasanya 8-15 tahun. 

## 3. Variabel Klimatologi & Lagging (Input)
- **Rainfall_CH**  
  *Deskripsi*: Rata-rata curah hujan bulanan dalam milimeter (mm).
- **RainyDays_HH**  
  *Deskripsi*: Total Hari Hujan (HH) dalam satu bulan. Mengindikasikan frekuensi turunnya hujan.
- **CH_Lag_6**, **CH_Lag_12**, **CH_Lag_24**  
  *Deskripsi*: Variabel *Lagging* untuk curah hujan (CH) dari 6 bulan, 12 bulan, dan 24 bulan yang lalu. Kelapa sawit memiliki memori fisiologis (contoh: proses inisiasi bunga hingga buah matang bisa memakan waktu hingga 24 bulan). Kekeringan 24 bulan lalu bisa berdampak pada panen bulan ini.
- **HH_Lag_6**, **HH_Lag_12**, **HH_Lag_24**  
  *Deskripsi*: Variabel *Lagging* untuk Hari Hujan (HH) dari periode lampau.

## 4. Variabel Artificial / Transformasi Waktu (Input)
- **Month_Sin**  
  *Deskripsi*: Transformasi nilai trigonometri Sinus dari kolom `Month`. Digunakan oleh model *Deep Learning* agar algoritma mengerti bahwa bulan 12 (Desember) dan bulan 1 (Januari) berdekatan posisinya di kalender.
- **Month_Cos**  
  *Deskripsi*: Transformasi nilai Cosinus dari rentang bulan sebagai pasangan vektor dari koordinat waktu Siklus (*Cyclical Encoding*).

## 5. Variabel Target (Output)
- **Yield_Ton_Ha**  
  *Deskripsi*: Hasil panen per hektar lahan (dalam satuan Ton). Ini merupakan rumus komposit dari `Production_Kg` dibagi dengan (`Area_Ha` x 1000). Nilai ini adalah **Indikator Utama (Target Variabel)** yang diprediksi oleh cerdas buatan (*Artificial Intelligence* / AI) berbasis memori LSTM.
