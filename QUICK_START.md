# 🚀 Quick Start Guide

## Langkah 1: Buka PowerShell
Tekan `Win + R`, ketik `powershell`, lalu Enter

## Langkah 2: Navigasi ke Folder Project
```powershell
cd "C:\Users\marwn\Documents\Tugas\Semester 8\SKRIPSI\Program Prediksi"
```

## Langkah 3: Buat Virtual Environment
```powershell
python -m venv venv
```

## Langkah 4: Aktifkan Virtual Environment
```powershell
.\venv\Scripts\activate
```

Anda akan melihat `(venv)` di awal baris perintah

## Langkah 5: Install Dependencies
```powershell
pip install -r requirements.txt
```

Tunggu sampai selesai (1-5 menit tergantung kecepatan internet)

## Langkah 6: Jalankan Aplikasi
```powershell
streamlit run streamlit_app.py
```

## Langkah 7: Buka di Browser
Aplikasi akan otomatis membuka di `http://localhost:8501`

---

## Untuk Menjalankan Lagi Nanti

Cukup buka PowerShell di folder project dan jalankan:

```powershell
# Aktifkan virtual environment
.\venv\Scripts\activate

# Jalankan aplikasi
streamlit run streamlit_app.py
```

---

## Panduan Sidebar

| Parameter | Default | Range | Penjelasan |
|-----------|---------|-------|-----------|
| Saham | Roblox | Roblox / Unity / Keduanya | Pilih saham yang ingin diprediksi |
| Lookback | 30 | 10-60 | Berapa hari historis untuk input model |
| Rasio Latih | 80% | 70-90% | Persentase data untuk training |
| Units | 64 | 32-128 | Jumlah neuron dalam layer |
| Dropout | 0.2 | 0.0-0.5 | Regularisasi untuk menghindari overfitting |
| Epoch | 100 | 20-200 | Iterasi training |
| Batch Size | 32 | 16-64 | Data per batch training |
| Periode Prediksi | 30 | 7/14/30/60/90/120 | Berapa hari ke depan yang diprediksi |
| Model | LSTM | LSTM / GRU / Keduanya | Model mana yang ingin digunakan |

---

## Tips Performa

- Jika training terlalu lambat:
  - Kurangi **Epoch** (coba 50 atau 30)
  - Kurangi **Batch Size** (coba 16)
  - Gunakan periode data yang lebih pendek (1-2 tahun)

- Untuk hasil lebih akurat:
  - Gunakan periode data lebih panjang (3-5 tahun)
  - Tingkatkan **Units** (coba 96 atau 128)
  - Coba **Bandingkan Keduanya** untuk membandingkan LSTM dan GRU

---

## Troubleshooting Cepat

**Error: Module not found**
```powershell
pip install --upgrade -r requirements.txt
```

**Data tidak terambil**
- Pastikan internet aktif
- Coba update yfinance: `pip install --upgrade yfinance`

**Aplikasi crash**
- Tutup aplikasi (Ctrl + C di PowerShell)
- Buka kembali
- Reduce epoch/batch size

---

Selamat! Aplikasi Anda sudah siap digunakan! 🎉
