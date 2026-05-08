import random
import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ============================================================
# KONFIGURASI STREAMLIT
# ============================================================
st.set_page_config(
    page_title="Prediksi Saham LSTM/GRU",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main { padding-top: 2rem; }
    </style>
""", unsafe_allow_html=True)

# ============================================================
# SIDEBAR - KONFIGURASI
# ============================================================
st.sidebar.title("⚙️ Konfigurasi Parameter")
st.sidebar.write("---")

saham_pilihan = st.sidebar.selectbox(
    "Pilih Saham:",
    ["Roblox (RBLX)", "Unity Software (U)", "Dua-duanya"]
)

ticker_map = {
    "Roblox (RBLX)": "RBLX",
    "Unity Software (U)": "U",
}

col1, col2 = st.sidebar.columns(2)
with col1:
    tanggal_mulai = st.date_input("Dari tanggal:", pd.to_datetime("2021-03-01"))
with col2:
    tanggal_akhir = st.date_input("Hingga tanggal:", pd.to_datetime("2026-03-20"))

st.sidebar.write("---")

st.sidebar.subheader("🧠 Hyperparameter Model")
lookback_window = st.sidebar.slider("Lookback Window (hari):", 10, 60, 30)
rasio_latih     = st.sidebar.slider("Rasio Data Latih (%):", 70, 90, 80) / 100
jumlah_unit     = st.sidebar.slider("Jumlah Unit LSTM/GRU:", 32, 128, 64, step=16)
dropout_rate    = st.sidebar.slider("Dropout Rate:", 0.0, 0.5, 0.2, step=0.05)
jumlah_epoch    = st.sidebar.slider("Jumlah Epoch:", 20, 200, 100, step=10)
ukuran_batch    = st.sidebar.slider("Batch Size:", 16, 64, 32, step=16)

st.sidebar.write("---")

periode_prediksi_list = st.sidebar.multiselect(
    "Periode Prediksi (hari ke depan):",
    [7, 14, 30, 60, 90, 120],
    default=[30, 60, 120]
)
if not periode_prediksi_list:
    periode_prediksi_list = [30]
periode_prediksi_list = sorted(periode_prediksi_list)

model_pilihan = st.sidebar.selectbox(
    "Pilih Model:",
    ["LSTM", "GRU", "Bandingkan Keduanya"]
)

st.sidebar.write("---")

st.sidebar.subheader("🎲 Reproducibility")
seed_value = st.sidebar.number_input(
    "Set Seed:",
    min_value=0,
    max_value=99999,
    value=18,
    step=1,
    help="Seed yang sama menghasilkan output yang identik."
)

# ============================================================
# FUNGSI UTILITY
# ============================================================

@st.cache_data
def get_stock_data(ticker, start, end):
    try:
        data = yf.download(ticker, start=str(start), end=str(end), auto_adjust=True, progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        data.index = pd.to_datetime(data.index)
        return data
    except Exception as e:
        st.error(f"Error mengambil data {ticker}: {str(e)}")
        return None


def normalisasi_data(harga_close):
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_ternormalisasi = scaler.fit_transform(harga_close.reshape(-1, 1))
    return data_ternormalisasi, scaler


def buat_dataset_sekuensial(data_ternormalisasi, lookback):
    X, y = [], []
    for i in range(lookback, len(data_ternormalisasi)):
        X.append(data_ternormalisasi[i - lookback:i, 0])
        y.append(data_ternormalisasi[i, 0])
    return np.array(X).reshape(-1, lookback, 1), np.array(y)


def bagi_data_latih_uji(X, y, rasio):
    ukuran_latih = int(len(X) * rasio)
    return X[:ukuran_latih], X[ukuran_latih:], y[:ukuran_latih], y[ukuran_latih:]


def buat_model_lstm(lookback, unit, dropout):
    model = Sequential([
        LSTM(unit, return_sequences=True, input_shape=(lookback, 1)),
        Dropout(dropout),
        LSTM(unit // 2, return_sequences=False),
        Dropout(dropout),
        Dense(25, activation='relu'),
        Dense(1)
    ], name='model_lstm')
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


def buat_model_gru(lookback, unit, dropout):
    model = Sequential([
        GRU(unit, return_sequences=True, input_shape=(lookback, 1)),
        Dropout(dropout),
        GRU(unit // 2, return_sequences=False),
        Dropout(dropout),
        Dense(25, activation='relu'),
        Dense(1)
    ], name='model_gru')
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


def latih_model(model, X_latih, y_latih, epoch, batch):
    cb_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=0)
    cb_lr   = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=0)
    riwayat = model.fit(
        X_latih, y_latih,
        epochs=epoch,
        batch_size=batch,
        validation_split=0.1,
        callbacks=[cb_stop, cb_lr],
        verbose=0
    )
    return riwayat


def prediksi_ke_depan(model, data_ternormalisasi, scaler, lookback, jumlah_hari):
    # Sesuai notebook: rolling prediction dengan np.append pada axis=1
    input_sekuens = data_ternormalisasi[-lookback:].reshape(1, lookback, 1)
    hasil = []
    for _ in range(jumlah_hari):
        pred = model.predict(input_sekuens, verbose=0)[0, 0]
        hasil.append(pred)
        input_sekuens = np.append(input_sekuens[:, 1:, :], [[[pred]]], axis=1)
    return scaler.inverse_transform(np.array(hasil).reshape(-1, 1)).flatten()


def hitung_rmse(y_aktual, y_pred):
    return np.sqrt(mean_squared_error(y_aktual, y_pred))


def hitung_mae(y_aktual, y_pred):
    return mean_absolute_error(y_aktual, y_pred)


def hitung_mape(y_aktual, y_pred):
    y_aktual = np.array(y_aktual)
    y_pred   = np.array(y_pred)
    mask     = y_aktual != 0
    return np.mean(np.abs((y_aktual[mask] - y_pred[mask]) / y_aktual[mask])) * 100


# ============================================================
# MAIN APP
# ============================================================
st.title("📈 Prediksi Harga Saham dengan LSTM & GRU")
st.write("Model pembelajaran mendalam untuk memprediksi harga saham Roblox (RBLX) dan Unity (U)")
st.write("---")

# Tentukan ticker yang diproses
if saham_pilihan == "Dua-duanya":
    tickers_to_process = ["RBLX", "U"]
    ticker_names = {"RBLX": "Roblox (RBLX)", "U": "Unity (U)"}
else:
    tickers_to_process = [ticker_map[saham_pilihan]]
    ticker_names = {ticker_map[saham_pilihan]: saham_pilihan}

ticker_colors = {"RBLX": "#1f77b4", "U": "#ff7f0e"}

# Tentukan model yang dilatih
if model_pilihan == "LSTM":
    model_names = ["LSTM"]
elif model_pilihan == "GRU":
    model_names = ["GRU"]
else:
    model_names = ["LSTM", "GRU"]

# ============================================================
# FASE 2 — LOAD DATA
# ============================================================
st.info("📥 Mengambil data historis dari Yahoo Finance...")

data_saham = {}
for ticker in tickers_to_process:
    with st.spinner(f"Mengambil data {ticker}..."):
        df = get_stock_data(ticker, tanggal_mulai, tanggal_akhir)
        if df is not None and not df.empty:
            data_saham[ticker] = df
        else:
            st.error(f"Gagal mengambil data {ticker}")

if not data_saham:
    st.error("❌ Tidak ada data yang berhasil diambil")
    st.stop()

# ============================================================
# DATA HISTORIS
# ============================================================
st.subheader("📊 Data Historis Harga Saham")

for ticker in tickers_to_process:
    if ticker not in data_saham:
        continue
    df = data_saham[ticker]
    with st.expander(f"📋 Lihat Data {ticker_names[ticker]}", expanded=False):
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Hari Perdagangan", len(df))
        c2.metric("Harga Akhir (USD)", f"${df['Close'].iloc[-1]:.2f}")
        c3.metric("Harga Min/Max (USD)", f"${df['Close'].min():.2f} / ${df['Close'].max():.2f}")
        st.dataframe(df[['Open', 'High', 'Low', 'Close', 'Volume']].tail(10), use_container_width=True)

# Grafik historis (sesuai notebook: 2 subplot, warna beda per saham)
st.subheader("📈 Grafik Harga Historis")
fig, axes = plt.subplots(len(tickers_to_process), 1, figsize=(14, 5 * len(tickers_to_process)))
fig.suptitle(
    f'Harga Penutupan Historis Saham ({tanggal_mulai} — {tanggal_akhir})',
    fontsize=14, fontweight='bold'
)

if len(tickers_to_process) == 1:
    axes = [axes]

for idx, ticker in enumerate(tickers_to_process):
    if ticker not in data_saham:
        continue
    df = data_saham[ticker]
    axes[idx].plot(df.index, df['Close'], color=ticker_colors[ticker], linewidth=1.2)
    axes[idx].set_title(ticker_names[ticker], fontsize=12)
    axes[idx].set_ylabel('Harga Penutupan (USD)')
    axes[idx].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
st.pyplot(fig)
plt.close()

st.write("---")

# ============================================================
# FASE 3+4 — PREPROCESSING & TRAINING
# ============================================================
st.subheader("🧠 Preprocessing & Melatih Model")

hasil_semua = {}

for ticker in tickers_to_process:
    if ticker not in data_saham:
        continue

    st.info(f"🔄 Memproses {ticker_names[ticker]}...")
    df        = data_saham[ticker]
    harga_close = df['Close'].values

    # Preprocessing
    data_norm, scaler = normalisasi_data(harga_close)
    X, y              = buat_dataset_sekuensial(data_norm, lookback_window)
    X_latih, X_uji, y_latih, y_uji = bagi_data_latih_uji(X, y, rasio_latih)

    # Tanggal uji: sesuai notebook — hitung dari panjang df asli
    ukuran_latih_df = int(len(df) * rasio_latih)
    tanggal_uji_df  = df.index[ukuran_latih_df + lookback_window:]

    hasil_ticker = {}

    for nama_model in model_names:
        # Reset seed sebelum setiap model (sesuai notebook)
        random.seed(seed_value)
        np.random.seed(seed_value)
        tf.random.set_seed(seed_value)

        with st.spinner(f"Melatih {nama_model} — {ticker_names[ticker]}..."):
            if nama_model == "LSTM":
                model = buat_model_lstm(lookback_window, jumlah_unit, dropout_rate)
            else:
                model = buat_model_gru(lookback_window, jumlah_unit, dropout_rate)

            riwayat = latih_model(model, X_latih, y_latih, jumlah_epoch, ukuran_batch)

            # Prediksi data uji
            pred_uji_norm   = model.predict(X_uji, verbose=0)
            pred_uji_aktual = scaler.inverse_transform(pred_uji_norm)
            y_uji_aktual    = scaler.inverse_transform(y_uji.reshape(-1, 1))

            # Sinkronisasi panjang (sesuai notebook)
            n_min          = min(len(tanggal_uji_df), len(y_uji_aktual))
            tanggal_uji_ok = tanggal_uji_df[:n_min]

            rmse = hitung_rmse(y_uji_aktual[:n_min], pred_uji_aktual[:n_min])
            mae  = hitung_mae(y_uji_aktual[:n_min],  pred_uji_aktual[:n_min])
            mape = hitung_mape(y_uji_aktual[:n_min], pred_uji_aktual[:n_min])

            # Prediksi ke depan per periode
            prediksi_per_periode = {
                hari: prediksi_ke_depan(model, data_norm, scaler, lookback_window, hari)
                for hari in periode_prediksi_list
            }

            hasil_ticker[nama_model] = {
                "model":               model,
                "riwayat_latih":       riwayat,
                "tanggal_uji":         tanggal_uji_ok,
                "pred_uji_aktual":     pred_uji_aktual[:n_min],
                "y_uji_aktual":        y_uji_aktual[:n_min],
                "rmse":                rmse,
                "mae":                 mae,
                "mape":                mape,
                "prediksi_per_periode": prediksi_per_periode,
            }

            epoch_selesai = len(riwayat.history['loss'])
            st.success(
                f"✅ {nama_model} — {ticker_names[ticker]} selesai "
                f"(epoch: {epoch_selesai} | RMSE: {rmse:.4f} | MAPE: {mape:.4f}%)"
            )

    hasil_semua[ticker] = {
        "hasil":     hasil_ticker,
        "df":        df,
        "data_norm": data_norm,
        "scaler":    scaler,
    }

st.write("---")

# ============================================================
# FASE 5 — EVALUASI
# ============================================================
st.subheader("📊 Hasil Evaluasi Model")

baris_tabel = []
for ticker in tickers_to_process:
    if ticker not in hasil_semua:
        continue
    for nm in model_names:
        if nm not in hasil_semua[ticker]["hasil"]:
            continue
        d = hasil_semua[ticker]["hasil"][nm]
        baris_tabel.append({
            "Saham":    ticker_names[ticker],
            "Model":    nm,
            "RMSE ($)": round(d["rmse"], 4),
            "MAE ($)":  round(d["mae"],  4),
            "MAPE (%)": round(d["mape"], 4),
        })

if baris_tabel:
    st.dataframe(pd.DataFrame(baris_tabel), use_container_width=True, hide_index=True)

# Bar chart perbandingan metrik (muncul jika lebih dari 1 kombinasi)
if len(baris_tabel) > 1:
    st.subheader("📊 Perbandingan Metrik Evaluasi")
    label_bar  = [f"{r['Saham'].split('(')[1].rstrip(')')}\n{r['Model']}" for r in baris_tabel]
    warna_bar  = ['#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78'][:len(baris_tabel)]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle('Perbandingan Metrik Evaluasi Model LSTM vs GRU', fontsize=13, fontweight='bold')

    for ax, (metrik, satuan) in zip(axes, [("RMSE ($)", "USD"), ("MAE ($)", "USD"), ("MAPE (%)", "%")]):
        nilai = [r[metrik] for r in baris_tabel]
        bars  = ax.bar(label_bar, nilai, color=warna_bar, edgecolor='white', width=0.6)
        ax.bar_label(bars, fmt='%.3f', padding=3, fontsize=9)
        ax.set_title(f'Perbandingan {metrik}', fontsize=11, fontweight='bold')
        ax.set_ylabel(f'Nilai ({satuan})')
        ax.set_ylim(0, max(nilai) * 1.25)
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

st.write("---")

# ============================================================
# GRAFIK PREDIKSI vs AKTUAL (DATA UJI)  — sesuai plot_per_model notebook
# ============================================================
st.subheader("📉 Prediksi vs Aktual pada Data Uji")

for ticker in tickers_to_process:
    if ticker not in hasil_semua:
        continue
    hasil_ticker = hasil_semua[ticker]["hasil"]
    ticker_name  = ticker_names[ticker]

    for nm in model_names:
        if nm not in hasil_ticker:
            continue
        d = hasil_ticker[nm]

        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(d["tanggal_uji"], d["y_uji_aktual"].flatten(),
                label='Harga Aktual', linewidth=1.8)
        ax.plot(d["tanggal_uji"], d["pred_uji_aktual"].flatten(),
                label=f'Prediksi {nm}', linestyle='--', linewidth=1.5)
        ax.set_title(f'Prediksi vs Aktual ({nm}) — {ticker_name}', fontsize=13, fontweight='bold')
        ax.set_xlabel('Tanggal')
        ax.set_ylabel('Harga Penutupan (USD)')
        ax.legend()
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

st.write("---")

# ============================================================
# FASE 6 — PREDIKSI KE DEPAN  — sesuai plot_prediksi_per_horizon notebook
# ============================================================
st.subheader("🔮 Prediksi Harga ke Depan")

for ticker in tickers_to_process:
    if ticker not in hasil_semua:
        continue
    hasil_ticker      = hasil_semua[ticker]["hasil"]
    df_saham          = hasil_semua[ticker]["df"]
    ticker_name       = ticker_names[ticker]
    tanggal_akhir_data = df_saham.index[-1]
    harga_historis    = df_saham['Close'].values[-90:]
    tanggal_historis  = df_saham.index[-90:]

    for hari in periode_prediksi_list:
        # pd.bdate_range = hari kerja (sesuai notebook)
        tanggal_prediksi = pd.bdate_range(
            start=tanggal_akhir_data + pd.Timedelta(days=1),
            periods=hari
        )

        if len(model_names) > 1 and all(nm in hasil_ticker for nm in ["LSTM", "GRU"]):
            # LSTM dan GRU berdampingan (sesuai notebook)
            fig, axes = plt.subplots(1, 2, figsize=(16, 5))
            fig.suptitle(
                f'Prediksi Harga {ticker_name} — {hari} Hari ke Depan',
                fontsize=14, fontweight='bold'
            )
            for ax, (nm, warna) in zip(axes, [('LSTM', '#1f77b4'), ('GRU', '#ff7f0e')]):
                pred_harga = hasil_ticker[nm]['prediksi_per_periode'][hari]

                ax.plot(tanggal_historis, harga_historis,
                        label='Historis (90 Hari)', color='#2ca02c', linewidth=1.5)
                ax.plot([tanggal_akhir_data, tanggal_prediksi[0]],
                        [harga_historis[-1], pred_harga[0]],
                        color=warna, linewidth=1.2, linestyle='--')
                ax.plot(tanggal_prediksi, pred_harga,
                        label=f'Prediksi {hari} Hari',
                        color=warna, linewidth=2, marker='o', markersize=4)
                ax.annotate(
                    f'${pred_harga[-1]:.2f}',
                    xy=(tanggal_prediksi[-1], pred_harga[-1]),
                    xytext=(-20, 18), ha='center', textcoords='offset points',
                    fontsize=14, color=warna, fontweight='bold'
                )
                ax.set_title(f'Model {nm}', fontsize=11)
                ax.set_xlabel('Tanggal')
                ax.set_ylabel('Harga Penutupan (USD)')
                ax.legend(fontsize=9)
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                ax.tick_params(axis='x', rotation=30)
                ax.grid(True, alpha=0.3)

            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            # Tabel LSTM vs GRU berdampingan (sesuai notebook)
            pred_lstm = hasil_ticker['LSTM']['prediksi_per_periode'][hari]
            pred_gru  = hasil_ticker['GRU']['prediksi_per_periode'][hari]
            selisih   = pred_lstm - pred_gru
            df_tabel  = pd.DataFrame({
                'Tanggal':             tanggal_prediksi.strftime('%Y-%m-%d'),
                'Prediksi LSTM ($)':   [f'{h:.4f}' for h in pred_lstm],
                'Prediksi GRU ($)':    [f'{h:.4f}' for h in pred_gru],
                'Selisih ($)':         [f'{s:+.4f}' for s in selisih],
            })
            st.write(f"**Tabel Prediksi {hari} Hari ke Depan — {ticker_name}**")
            st.dataframe(df_tabel, use_container_width=True, hide_index=True)

        else:
            # Single model
            nm   = model_names[0]
            if nm not in hasil_ticker:
                continue
            warna      = '#1f77b4' if nm == 'LSTM' else '#ff7f0e'
            pred_harga = hasil_ticker[nm]['prediksi_per_periode'][hari]

            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(tanggal_historis, harga_historis,
                    label='Historis (90 Hari)', color='#2ca02c', linewidth=1.5)
            ax.plot([tanggal_akhir_data, tanggal_prediksi[0]],
                    [harga_historis[-1], pred_harga[0]],
                    color=warna, linewidth=1.2, linestyle='--')
            ax.plot(tanggal_prediksi, pred_harga,
                    label=f'Prediksi {hari} Hari',
                    color=warna, linewidth=2, marker='o', markersize=4)
            ax.annotate(
                f'${pred_harga[-1]:.2f}',
                xy=(tanggal_prediksi[-1], pred_harga[-1]),
                xytext=(-20, 18), ha='center', textcoords='offset points',
                fontsize=14, color=warna, fontweight='bold'
            )
            ax.set_title(
                f'Prediksi Harga {ticker_name} ({nm}) — {hari} Hari ke Depan',
                fontsize=13, fontweight='bold'
            )
            ax.set_xlabel('Tanggal')
            ax.set_ylabel('Harga Penutupan (USD)')
            ax.legend(fontsize=9)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.tick_params(axis='x', rotation=30)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            # Tabel single model
            df_tabel = pd.DataFrame({
                'Tanggal':                  tanggal_prediksi.strftime('%Y-%m-%d'),
                f'Prediksi {nm} ($)':       [f'{h:.4f}' for h in pred_harga],
                'Perubahan dari Sebelumnya (%)': np.concatenate(
                    [[np.nan], np.diff(pred_harga) / pred_harga[:-1] * 100]
                ).round(2),
            })
            st.write(f"**Tabel Prediksi {hari} Hari ke Depan — {ticker_name}**")
            st.dataframe(df_tabel, use_container_width=True, hide_index=True)

st.write("---")

# ============================================================
# KURVA LOSS PELATIHAN  — sesuai notebook (2x2 atau 1xN)
# ============================================================
st.subheader("📉 Kurva Loss Pelatihan")

warna_model = {'LSTM': '#1f77b4', 'GRU': '#ff7f0e'}

for ticker in tickers_to_process:
    if ticker not in hasil_semua:
        continue
    hasil_ticker = hasil_semua[ticker]["hasil"]
    ticker_name  = ticker_names[ticker]
    n_models     = len(model_names)

    fig, axes = plt.subplots(1, n_models, figsize=(7 * n_models, 4))
    if n_models == 1:
        axes = [axes]

    for ax, nm in zip(axes, model_names):
        if nm not in hasil_ticker:
            continue
        riwayat = hasil_ticker[nm]['riwayat_latih']
        ax.plot(riwayat.history['loss'],     label='Loss Latih',    color=warna_model[nm], linewidth=1.5)
        ax.plot(riwayat.history['val_loss'], label='Loss Validasi', color='gray', linewidth=1.2, linestyle='--')
        ax.set_title(f'{nm} — {ticker_name}', fontsize=11)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('MSE Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

st.write("---")

# ============================================================
# INFO MODEL
# ============================================================
st.subheader("ℹ️ Informasi Model")

info_col1, info_col2 = st.columns(2)
with info_col1:
    st.write("**LSTM (Long Short-Term Memory)**")
    st.write("- Arsitektur RNN dengan memory cell")
    st.write("- 3 gerbang: forget, input, output")
    st.write("- Cocok untuk data dengan dependensi jangka panjang")

with info_col2:
    st.write("**GRU (Gated Recurrent Unit)**")
    st.write("- Versi sederhana dari LSTM")
    st.write("- 2 gerbang: reset, update")
    st.write("- Komputasi lebih efisien, parameter lebih sedikit")

st.write("**Metrik Evaluasi:**")
st.write("- **RMSE**: Root Mean Squared Error (sensitif terhadap outlier)")
st.write("- **MAE**: Mean Absolute Error (tidak sensitif terhadap outlier)")
st.write("- **MAPE**: Mean Absolute Percentage Error (persentase error)")

st.write("---")
st.info("⚠️ **Disclaimer**: Model ini hanya untuk tujuan edukasi. Prediksi harga saham mengandung risiko tinggi. Jangan gunakan sebagai satu-satunya basis pengambilan keputusan investasi.")
