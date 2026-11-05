
# =============================================================================
# PREDICTION ENGINE - APOTEK STOCK FORECASTING
# Adaptasi dari Google Colab untuk aplikasi Streamlit
# =============================================================================

import io
import warnings

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

# =============================================================================
# KONFIGURASI MODEL
# =============================================================================

N_LAG = 4
MIN_BULAN_TERISI = 4
RANDOM_STATE = 42

XGB_PARAMS = {
    "n_estimators": 200,
    "learning_rate": 0.05,
    "max_depth": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": RANDOM_STATE,
    "verbosity": 0,
}

KOLOM_LAG = [f"lag{i}" for i in range(1, N_LAG + 1)]


# =============================================================================
# 1. MEMBACA FILE CSV
# =============================================================================

def baca_csv(file_bytes):
    """Membaca CSV dengan beberapa pilihan separator dan encoding."""

    kombinasi = [
        {"sep": None, "engine": "python", "encoding": "utf-8"},
        {"sep": ";", "encoding": "utf-8"},
        {"sep": ",", "encoding": "utf-8"},
        {"sep": "\t", "encoding": "utf-8"},
        {"sep": None, "engine": "python", "encoding": "latin-1"},
    ]

    error_terakhir = None

    for opsi in kombinasi:
        try:
            df = pd.read_csv(io.BytesIO(file_bytes), **opsi)

            if df.shape[1] < 2:
                continue

            df.columns = [str(k).strip() for k in df.columns]

            return df

        except Exception as err:
            error_terakhir = err

    raise ValueError(
        f"Gagal membaca file CSV. Periksa format file. "
        f"Error: {error_terakhir}"
    )


# =============================================================================
# 2. DETEKSI KOLOM BULAN
# =============================================================================

def adalah_kolom_bulan(nama):
    """Mendeteksi kolom dengan format YYYY-MM."""

    return pd.notna(
        pd.to_datetime(
            str(nama),
            format="%Y-%m",
            errors="coerce"
        )
    )


# =============================================================================
# 3. PROSES PREDIKSI UTAMA
# =============================================================================

def jalankan_prediksi(file_bytes, item_col=None):
    """
    Menjalankan seluruh proses forecasting.

    Parameter:
    file_bytes : isi file CSV dalam bentuk bytes
    item_col   : nama kolom kode obat, None untuk deteksi otomatis

    Menghasilkan:
    - forecast_df
    - evaluasi_df
    - df_long
    - metrics
    - ringkasan_obat
    - bulan_target
    """

    # -------------------------------------------------------------------------
    # LANGKAH 1 - MEMBACA DATA
    # -------------------------------------------------------------------------

    df_asli = baca_csv(file_bytes)

    if df_asli.empty:
        raise ValueError("File CSV tidak memiliki data.")

    # -------------------------------------------------------------------------
    # LANGKAH 2 - DETEKSI KOLOM BULAN
    # -------------------------------------------------------------------------

    kolom_bulan = [
        k for k in df_asli.columns
        if adalah_kolom_bulan(k)
    ]

    if not kolom_bulan:
        raise ValueError(
            "Kolom bulan tidak ditemukan. "
            "Gunakan format YYYY-MM, contoh 2025-01."
        )

    kolom_non_bulan = [
        k for k in df_asli.columns
        if k not in kolom_bulan
    ]

    if item_col is None:
        if not kolom_non_bulan:
            raise ValueError("Kolom kode obat tidak ditemukan.")
        item_col = kolom_non_bulan[0]

    if item_col not in df_asli.columns:
        raise ValueError(
            f"Kolom '{item_col}' tidak ditemukan dalam data."
        )

    # -------------------------------------------------------------------------
    # LANGKAH 3 - WIDE TO LONG
    # -------------------------------------------------------------------------

    df_pakai = df_asli[[item_col] + kolom_bulan].copy()

    df_pakai[item_col] = (
        df_pakai[item_col]
        .astype(str)
        .str.strip()
    )

    df_long = df_pakai.melt(
        id_vars=item_col,
        var_name="bulan",
        value_name="sales"
    )

    df_long["bulan"] = pd.to_datetime(
        df_long["bulan"],
        format="%Y-%m"
    )

    df_long = df_long.rename(
        columns={item_col: "kode_obat"}
    )

    df_long["sales"] = pd.to_numeric(
        df_long["sales"],
        errors="coerce"
    )

    if df_long["kode_obat"].eq("").any():
        raise ValueError("Terdapat kode obat kosong.")

    if df_long["sales"].dropna().lt(0).any():
        raise ValueError(
            "Terdapat nilai penjualan negatif. "
            "Periksa kembali data CSV."
        )

    df_long = df_long.sort_values(
        ["kode_obat", "bulan"]
    ).reset_index(drop=True)

    # -------------------------------------------------------------------------
    # LANGKAH 4 - PENANGANAN MISSING DATA
    # -------------------------------------------------------------------------

    df_long["tercatat"] = df_long["sales"].notna()

    jumlah_kosong = int(
        df_long["sales"].isna().sum()
    )

    df_long["sales"] = df_long["sales"].fillna(0)

    # -------------------------------------------------------------------------
    # LANGKAH 5 - DETEKSI OUTLIER DENGAN IQR
    # Tidak mengubah nilai penjualan asli
    # -------------------------------------------------------------------------

    data_tercatat = df_long[
        df_long["tercatat"]
    ]

    q1 = data_tercatat.groupby(
        "kode_obat"
    )["sales"].quantile(0.25)

    q3 = data_tercatat.groupby(
        "kode_obat"
    )["sales"].quantile(0.75)

    iqr = q3 - q1

    batas_bawah = (q1 - 1.5 * iqr).clip(lower=0)
    batas_atas = q3 + 1.5 * iqr

    df_long["is_outlier"] = (
        df_long["tercatat"]
        & (
            (df_long["sales"] < df_long["kode_obat"].map(batas_bawah))
            | (df_long["sales"] > df_long["kode_obat"].map(batas_atas))
        )
    )

    jumlah_outlier = int(
        df_long["is_outlier"].sum()
    )

    # -------------------------------------------------------------------------
    # LANGKAH 6 - MEMBUAT FITUR LAG
    # -------------------------------------------------------------------------

    for i, nama in enumerate(KOLOM_LAG, start=1):
        df_long[nama] = (
            df_long.groupby("kode_obat")["sales"].shift(i)
        )

    df_fitur = df_long.dropna(
        subset=KOLOM_LAG
    ).reset_index(drop=True)

    # -------------------------------------------------------------------------
    # LANGKAH 7 - CEK KECUKUPAN DATA PER OBAT
    # -------------------------------------------------------------------------

    jumlah_bulan_terisi = (
        df_long.groupby("kode_obat")["tercatat"]
        .sum()
        .astype(int)
    )

    jumlah_baris_latih = (
        df_fitur.groupby("kode_obat")
        .size()
    )

    ringkasan_obat = pd.DataFrame({
        "jumlah_bulan_terisi": jumlah_bulan_terisi,
        "jumlah_baris_latih": jumlah_baris_latih,
    }).fillna(0)

    ringkasan_obat["jumlah_baris_latih"] = (
        ringkasan_obat["jumlah_baris_latih"].astype(int)
    )

    ringkasan_obat["cukup_data"] = (
        (ringkasan_obat["jumlah_bulan_terisi"] >= MIN_BULAN_TERISI)
        & (ringkasan_obat["jumlah_baris_latih"] >= 2)
    )

    # -------------------------------------------------------------------------
    # LANGKAH 8 - MELATIH MODEL
    # -------------------------------------------------------------------------

    grup_fitur = {
        kode: grup.sort_values("bulan")
        for kode, grup in df_fitur.groupby("kode_obat")
    }

    rata_rata_semua_bulan = (
        df_long.groupby("kode_obat")["sales"].mean()
    )

    models = {}
    metode_forecast = {}
    lag_terakhir = {}
    nilai_fallback = {}

    for kode, baris in ringkasan_obat.iterrows():

        if baris["cukup_data"]:

            grup = grup_fitur[kode]

            model = XGBRegressor(**XGB_PARAMS)

            model.fit(
                grup[KOLOM_LAG],
                grup["sales"]
            )

            models[kode] = model

            metode_forecast[kode] = "XGBoost"

            riwayat = (
                df_long[
                    df_long["kode_obat"] == kode
                ]
                .sort_values("bulan")["sales"]
                .to_numpy(dtype=float)
            )

            lag_terakhir[kode] = {
                f"lag{j}": float(riwayat[-j])
                for j in range(1, N_LAG + 1)
            }

        else:

            nilai_fallback[kode] = float(
                rata_rata_semua_bulan.get(kode, 0.0)
            )

            metode_forecast[kode] = "Rata-rata (Data Minim)"

    # -------------------------------------------------------------------------
    # LANGKAH 9 - FORECAST SATU BULAN BERIKUTNYA
    # -------------------------------------------------------------------------

    bulan_terakhir = df_long["bulan"].max()

    bulan_target = (
        bulan_terakhir + pd.DateOffset(months=1)
    )

    future_predictions = {}

    for kode, model in models.items():

        lag_input = [
            lag_terakhir[kode][f"lag{i}"]
            for i in range(1, N_LAG + 1)
        ]

        X_input = pd.DataFrame(
            [lag_input],
            columns=KOLOM_LAG
        )

        pred = float(
            model.predict(X_input)[0]
        )

        future_predictions[kode] = max(pred, 0.0)

    for kode, nilai in nilai_fallback.items():

        future_predictions[kode] = max(
            float(nilai), 0.0
        )

    semua_prediksi = np.array(
        list(future_predictions.values()),
        dtype=float
    )

    if not np.isfinite(semua_prediksi).all():
        raise ValueError(
            "Ditemukan hasil prediksi NaN atau Inf."
        )

    if len(future_predictions) != len(ringkasan_obat):
        raise ValueError(
            "Jumlah SKU hasil prediksi tidak sesuai."
        )

    # -------------------------------------------------------------------------
    # LANGKAH 10 - MEMBUAT TABEL FORECAST
    # -------------------------------------------------------------------------

    label_bulan = bulan_target.strftime("%Y-%m")

    forecast_df = pd.DataFrame({
        "kode_obat": list(future_predictions.keys()),
        "prediksi_unit": list(future_predictions.values())
    })

    forecast_df["bulan_prediksi"] = label_bulan

    forecast_df["metode_forecast"] = (
        forecast_df["kode_obat"].map(metode_forecast)
    )

    forecast_df["jumlah_bulan_terisi"] = (
        forecast_df["kode_obat"].map(
            ringkasan_obat["jumlah_bulan_terisi"]
        )
    )

    forecast_df["prediksi_unit"] = (
        forecast_df["prediksi_unit"].round(4)
    )

    forecast_df["prediksi_unit_bulat"] = np.floor(
        forecast_df["prediksi_unit"] + 0.5
    ).astype(int)

    forecast_df = forecast_df.sort_values(
        "prediksi_unit",
        ascending=False
    ).reset_index(drop=True)

    # -------------------------------------------------------------------------
    # LANGKAH 11 - EVALUASI ONE-STEP-AHEAD
    # -------------------------------------------------------------------------

    df_eval = df_long.sort_values(
        ["kode_obat", "bulan"]
    ).copy()

    bulan_uji = df_eval["bulan"].max()

    data_latih = df_eval[
        df_eval["bulan"] < bulan_uji
    ].copy()

    data_uji = df_eval[
        df_eval["bulan"] == bulan_uji
    ].copy()

    hasil_evaluasi = []

    for kode, grup in data_latih.groupby("kode_obat"):

        grup = grup.sort_values("bulan").copy()

        aktual_baris = data_uji[
            data_uji["kode_obat"] == kode
        ]

        if aktual_baris.empty:
            continue

        aktual = float(
            aktual_baris["sales"].iloc[0]
        )

        jumlah_terisi = int(
            grup["tercatat"].sum()
        )

        for j in range(1, N_LAG + 1):
            grup[f"lag{j}"] = (
                grup["sales"].shift(j)
            )

        fitur_latih = grup.dropna(
            subset=KOLOM_LAG
        ).copy()

        cukup_data = (
            jumlah_terisi >= MIN_BULAN_TERISI
            and len(fitur_latih) >= 2
        )

        if cukup_data:

            model_uji = XGBRegressor(**XGB_PARAMS)

            model_uji.fit(
                fitur_latih[KOLOM_LAG],
                fitur_latih["sales"]
            )

            riwayat = grup["sales"].to_numpy(
                dtype=float
            )

            lag_input = [
                float(riwayat[-j])
                for j in range(1, N_LAG + 1)
            ]

            X_input = pd.DataFrame(
                [lag_input],
                columns=KOLOM_LAG
            )

            prediksi = float(
                model_uji.predict(X_input)[0]
            )

            metode = "XGBoost"

        else:

            prediksi = float(
                grup["sales"].mean()
            )

            metode = "Rata-rata (Data Minim)"

        prediksi = max(prediksi, 0.0)

        hasil_evaluasi.append({
            "kode_obat": kode,
            "aktual": aktual,
            "prediksi": prediksi,
            "metode": metode,
            "bulan_aktif_latih": int(
                (grup["sales"] > 0).sum()
            )
        })

    evaluasi_df = pd.DataFrame(
        hasil_evaluasi
    )

    if evaluasi_df.empty:
        raise ValueError(
            "Tidak ada data yang dapat dievaluasi."
        )

    evaluasi_df["absolute_error"] = (
        evaluasi_df["aktual"] - evaluasi_df["prediksi"]
    ).abs()

    evaluasi_df["ape"] = np.where(
        evaluasi_df["aktual"] > 0,
        evaluasi_df["absolute_error"]
        / evaluasi_df["aktual"].replace(0, np.nan) * 100,
        np.nan
    )

    MAE = float(
        evaluasi_df["absolute_error"].mean()
    )

    MAPE = float(
        evaluasi_df["ape"].mean()
    )

    total_aktual = evaluasi_df["aktual"].sum()

    WAPE = (
        float(
            evaluasi_df["absolute_error"].sum()
            / total_aktual * 100
        )
        if total_aktual > 0 else np.nan
    )

    # -------------------------------------------------------------------------
    # LANGKAH 12 - RINGKASAN HASIL
    # -------------------------------------------------------------------------

    metrics = {
        "MAE": MAE,
        "MAPE": MAPE,
        "WAPE": WAPE,
        "total_sku": len(forecast_df),
        "sku_xgboost": len(models),
        "sku_rata_rata": len(nilai_fallback),
        "jumlah_kosong": jumlah_kosong,
        "jumlah_outlier": jumlah_outlier,
        "bulan_terakhir": bulan_terakhir.strftime("%Y-%m"),
        "bulan_prediksi": label_bulan,
        "bulan_evaluasi": bulan_uji.strftime("%Y-%m"),
    }

    return {
        "forecast_df": forecast_df,
        "evaluasi_df": evaluasi_df,
        "df_long": df_long,
        "metrics": metrics,
        "ringkasan_obat": ringkasan_obat,
        "bulan_target": bulan_target,
    }
