# 📈 Prediksi Saham LSTM & GRU - Aplikasi Streamlit

Aplikasi web interaktif untuk memprediksi harga saham Roblox (RBLX) dan Unity (U) menggunakan model deep learning LSTM dan GRU.

## 🎯 Fitur Utama

- **Data Fetching**: Mengambil data historis real-time dari Yahoo Finance
- **Preprocessing Otomatis**: Normalisasi data dan pembentukan time series sequences
- **Model Training**: Melatih model LSTM dan GRU dengan hyperparameter yang dapat disesuaikan
- **Evaluasi Model**: Menampilkan metrik RMSE, MAE, dan MAPE
- **Prediksi Multi-Hari**: Memprediksi harga saham untuk N hari ke depan
- **Visualisasi Interaktif**: Grafik harga historis dan prediksi dengan matplotlib
- **UI Responsif**: Sidebar untuk konfigurasi parameter yang mudah digunakan

## 📋 Prerequisites

- Python 3.8 atau lebih tinggi
- pip (Python Package Installer)
- Koneksi internet (untuk mengambil data dari Yahoo Finance)

## 🚀 Instalasi & Setup

### 1. Clone atau unduh project
```bash
cd "c:\Users\marwn\Documents\Tugas\Semester 8\SKRIPSI\Program Prediksi"
```

### 2. Buat Virtual Environment (Opsional tapi Recommended)

**Menggunakan venv:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Menggunakan conda:**
```bash
conda create -n prediksi-saham python=3.10
conda activate prediksi-saham
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

Jika ada error dengan TensorFlow, coba:
```bash
pip install --upgrade tensorflow
pip install --upgrade numpy
```

## 🏃 Menjalankan Aplikasi

### Cara 1: Menggunakan Terminal/PowerShell

```bash
streamlit run streamlit_app.py
```

Aplikasi akan terbuka otomatis di browser pada URL: `http://localhost:8501`

### Cara 2: Menggunakan VS Code

1. Buka terminal terintegrasi di VS Code (Ctrl + `)
2. Pastikan sudah di folder project
3. Jalankan:
```bash
streamlit run streamlit_app.py
```

## 📊 Cara Menggunakan Aplikasi

### Sidebar Configuration

1. **Pilih Saham**: 
   - Roblox (RBLX)
   - Unity Software (U)
   - Dua-duanya

2. **Periode Data Historis**: 
   - Atur tanggal awal dan akhir untuk mengambil data historis

3. **Hyperparameter Model**:
   - **Lookback Window**: Jumlah hari historis sebagai input model (default: 30)
   - **Rasio Data Latih**: Persentase data untuk training vs testing (default: 80%)
   - **Jumlah Unit**: Neuron dalam layer LSTM/GRU (default: 64)
   - **Dropout Rate**: Regularisasi untuk mencegah overfitting (default: 0.2)
   - **Jumlah Epoch**: Iterasi training (default: 100)
   - **Batch Size**: Jumlah sampel per batch (default: 32)

4. **Periode Prediksi**: Pilih berapa hari ke depan yang ingin diprediksi (7, 14, 30, 60, 90, 120 hari)

5. **Pilih Model**: 
   - LSTM
   - GRU
   - Bandingkan Keduanya

### Main Display

1. **Data Historis**: Lihat data OHLCV dari Yahoo Finance
2. **Grafik Historis**: Visualisasi tren harga
3. **Hasil Evaluasi**: Metrik RMSE, MAE, MAPE untuk setiap model
4. **Prediksi**: Tabel dan grafik prediksi harga N hari ke depan

## 🔧 Troubleshooting

### Error: "No module named 'streamlit'"
**Solusi**: Install streamlit dengan:
```bash
pip install streamlit
```

### Error: "No internet connection" atau data tidak terambil
**Solusi**: 
- Pastikan koneksi internet aktif
- Update yfinance:
```bash
pip install --upgrade yfinance
```

### Error: TensorFlow atau CUDA issues
**Solusi** (untuk Windows):
```bash
# Uninstall dan reinstall TensorFlow
pip uninstall tensorflow
pip install tensorflow

# Atau gunakan CPU version
pip install tensorflow-cpu
```

### Aplikasi berjalan lambat
**Solusi**:
- Kurangi jumlah epoch
- Kurangi ukuran batch
- Gunakan periode data yang lebih kecil (contoh: 2-3 tahun daripada 5 tahun)

## 📁 Struktur File

```
Program Prediksi/
├── streamlit_app.py          # File aplikasi Streamlit utama
├── requirements.txt          # Daftar dependencies
├── README.md                # File dokumentasi ini
└── prediksi_saham_lstm_gru.ipynb  # Notebook original
```

## 🎓 Penjelasan Model

### LSTM (Long Short-Term Memory)
- **Kelebihan**: Menangani dependensi jangka panjang dengan baik
- **Kelemahan**: Lebih kompleks, membutuhkan lebih banyak parameter
- **Gerbang**: Forget gate, Input gate, Output gate

### GRU (Gated Recurrent Unit)
- **Kelebihan**: Lebih efisien, parameter lebih sedikit, training lebih cepat
- **Kelemahan**: Mungkin kurang akurat untuk data yang sangat kompleks
- **Gerbang**: Reset gate, Update gate

## 📈 Tips untuk Hasil Terbaik

1. **Gunakan periode data yang cukup**: Minimal 1-2 tahun untuk hasil yang konsisten
2. **Adjust hyperparameter**: Experiment dengan berbagai nilai untuk dataset Anda
3. **Monitor overfitting**: Perhatikan perbedaan loss train vs validation
4. **Validasi hasil**: Bandingkan prediksi dengan data actual
5. **Update data secara berkala**: Data saham berkembang, retrain model dengan data terbaru

## ⚠️ Disclaimer

Model ini dibuat untuk **tujuan edukasi dan penelitian** saja. Prediksi harga saham:
- Mengandung risiko tinggi
- Tidak dijamin akurat
- Jangan gunakan sebagai satu-satunya basis pengambilan keputusan investasi
- Konsultasikan dengan financial advisor profesional

## 📚 Referensi

- [Streamlit Documentation](https://docs.streamlit.io/)
- [TensorFlow Keras Documentation](https://www.tensorflow.org/guide/keras)
- [yfinance Documentation](https://github.com/ranaroussi/yfinance)
- [LSTM & GRU Overview](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)

## 👤 Metadata

- **Nama**: Marwan Karim Samayfa
- **NPM**: 10122746
- **Program Studi**: Sistem Informasi
- **Universitas**: Gunadarma
- **Tahun**: 2026

## 📞 Support

Jika mengalami masalah, silakan:
1. Cek terminal untuk error message
2. Pastikan semua dependencies terinstall dengan benar
3. Coba restart aplikasi
4. Update semua packages ke versi terbaru

---

**Selamat menggunakan! 🚀**
