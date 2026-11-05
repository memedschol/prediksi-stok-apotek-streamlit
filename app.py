
# =============================================================================
# APOTEK STOCK FORECASTING
# Web Application menggunakan Streamlit
# =============================================================================

import io
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from prediction import jalankan_prediksi


# =============================================================================
# KONFIGURASI HALAMAN
# =============================================================================

st.set_page_config(
    page_title="Apotek Stock Forecasting",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =============================================================================
# HEADER
# =============================================================================

st.title("💊 Apotek Stock Forecasting")
st.subheader("Monthly Stock Prediction System")

st.markdown(
    """
    Aplikasi prediksi kebutuhan stok obat berdasarkan data historis
    penjualan bulanan menggunakan algoritma XGBoost dan metode rata-rata.
    """
)

st.divider()


# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:

    st.title("📊 Menu Utama")

    st.markdown("---")

    st.markdown("### Upload Data")

    file_upload = st.file_uploader(
        "Pilih file data penjualan (CSV)",
        type=["csv"],
        help="Gunakan data dengan kolom bulan berformat YYYY-MM."
    )

    st.markdown("---")

    st.markdown("### Informasi Model")

    st.info(
        """
        **Algoritma:**
        - XGBoost Regressor
        
        **Konfigurasi:**
        - Forecast: 1 bulan berikutnya
                """
    )


# =============================================================================
# PETUNJUK DATA
# =============================================================================

with st.expander("ℹ️ Petunjuk format data CSV"):

    st.markdown(
        """
        Data harus menggunakan format **wide**, yaitu satu baris untuk
        setiap obat dan satu kolom untuk setiap bulan.

        Contoh:
        """
    )

    contoh_data = pd.DataFrame({
        "kode_obat": [
            "OBAT A",
            "OBAT B",
            "OBAT C"
        ],
        "2025-01": [100, 50, 20],
        "2025-02": [120, 40, 25],
        "2025-03": [110, 60, None],
        "2025-04": [130, 55, 30],
        "2025-05": [125, 65, 35],
    })

    st.dataframe(
        contoh_data,
        use_container_width=True
    )

    st.caption(
        "Nama kolom bulan harus menggunakan format YYYY-MM. "
        "Kolom identitas obat boleh menggunakan nama lain."
    )


# =============================================================================
# PROSES DATA
# =============================================================================

if file_upload is None:

    st.info(
        "👈 Silakan upload file CSV melalui menu di sebelah kiri "
        "untuk memulai proses prediksi."
    )

    st.markdown("### Alur Penggunaan Aplikasi")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### 1. Upload Data")
        st.write(
            "Masukkan data historis penjualan obat dalam format CSV."
        )

    with col2:
        st.markdown("#### 2. Forecasting")
        st.write(
            "Sistem memproses data dan menjalankan model prediksi."
        )

    with col3:
        st.markdown("#### 3. Hasil Prediksi")
        st.write(
            "Lihat hasil forecast, evaluasi model, dan unduh hasil."
        )

else:

    st.success(
        f"File berhasil diunggah: {file_upload.name}"
    )

    # -------------------------------------------------------------------------
    # PEMILIHAN KOLOM IDENTITAS OBAT
    # -------------------------------------------------------------------------

    try:

        file_bytes = file_upload.getvalue()

        # Membaca data untuk menampilkan pilihan kolom
        from prediction import baca_csv

        df_preview = baca_csv(file_bytes)

        kolom_bulan = [
            k for k in df_preview.columns
            if pd.notna(
                pd.to_datetime(
                    str(k),
                    format="%Y-%m",
                    errors="coerce"
                )
            )
        ]

        kolom_identitas = [
            k for k in df_preview.columns
            if k not in kolom_bulan
        ]

        if not kolom_bulan:
            st.error(
                "Tidak ditemukan kolom bulan dengan format YYYY-MM."
            )
            st.stop()

        if not kolom_identitas:
            st.error(
                "Tidak ditemukan kolom identitas obat."
            )
            st.stop()

        item_col = st.selectbox(
            "Pilih kolom kode/nama obat:",
            options=kolom_identitas,
            index=0
        )

        # ---------------------------------------------------------------------
        # TAMPILAN PREVIEW DATA
        # ---------------------------------------------------------------------

        with st.expander("Lihat preview data yang diunggah"):

            st.dataframe(
                df_preview.head(10),
                use_container_width=True
            )

            st.write(
                f"Jumlah baris: {len(df_preview):,}"
            )

            st.write(
                f"Jumlah bulan: {len(kolom_bulan):,}"
            )

        st.divider()

        # ---------------------------------------------------------------------
        # TOMBOL FORECASTING
        # ---------------------------------------------------------------------

        if st.button(
            "🚀 Jalankan Prediksi Stok",
            type="primary",
            use_container_width=True
        ):

            with st.spinner(
                "Sedang memproses data dan melatih model prediksi..."
            ):

                hasil = jalankan_prediksi(
                    file_bytes=file_bytes,
                    item_col=item_col
                )

                st.session_state["hasil_prediksi"] = hasil

            st.success(
                "Proses forecasting berhasil diselesaikan!"
            )

        # ---------------------------------------------------------------------
        # MENAMPILKAN HASIL
        # ---------------------------------------------------------------------

        if "hasil_prediksi" in st.session_state:

            hasil = st.session_state["hasil_prediksi"]

            forecast_df = hasil["forecast_df"]
            evaluasi_df = hasil["evaluasi_df"]
            df_long = hasil["df_long"]
            metrics = hasil["metrics"]

            st.divider()

            
            # -----------------------------------------------------------------
            # RINGKASAN PREDIKSI
            # -----------------------------------------------------------------

            st.divider()

            st.subheader("Hasil Prediksi Seluruh Obat")

            forecast_tampil = forecast_df.drop(
    columns=["prediksi_unit"],
    errors="ignore"
)

st.dataframe(
    forecast_tampil,
    use_container_width=True,
    hide_index=True
)

            # -----------------------------------------------------------------
            # VISUALISASI TOP 10 SKU
            # -----------------------------------------------------------------

            st.divider()

            st.subheader(
                "10 Obat dengan Prediksi Tertinggi"
            )

            top10 = forecast_df.head(10).copy()

            fig, ax = plt.subplots(figsize=(11, 5))

            ax.barh(
                top10["kode_obat"].astype(str)[::-1],
                top10["prediksi_unit"][::-1]
            )

            ax.set_xlabel("Prediksi (Unit)")
            ax.set_ylabel("Nama Obat")
            ax.set_title(
                f"Forecast Bulan {metrics['bulan_prediksi']}"
            )

            plt.tight_layout()

            st.pyplot(fig)
            plt.close(fig)

            # -----------------------------------------------------------------
            # VISUALISASI RIWAYAT DAN FORECAST
            # -----------------------------------------------------------------

            st.divider()

            st.subheader("Grafik Historis dan Forecast")

            daftar_obat = forecast_df["kode_obat"].tolist()

            obat_dipilih = st.selectbox(
                "Pilih obat untuk melihat grafik:",
                options=daftar_obat
            )

            riwayat = df_long[
                df_long["kode_obat"] == obat_dipilih
            ].sort_values("bulan")

            prediksi_obat = forecast_df[
                forecast_df["kode_obat"] == obat_dipilih
            ].iloc[0]

            fig, ax = plt.subplots(figsize=(11, 4))

            ax.plot(
                riwayat["bulan"],
                riwayat["sales"],
                marker="o",
                linewidth=2,
                label="Penjualan Aktual"
            )

            ax.scatter(
                [hasil["bulan_target"]],
                [prediksi_obat["prediksi_unit"]],
                color="red",
                marker="D",
                s=100,
                label="Prediksi"
            )

            ax.plot(
                [
                    riwayat["bulan"].iloc[-1],
                    hasil["bulan_target"]
                ],
                [
                    riwayat["sales"].iloc[-1],
                    prediksi_obat["prediksi_unit"]
                ],
                linestyle="--",
                color="red",
                alpha=0.7
            )

            ax.set_xlabel("Bulan")
            ax.set_ylabel("Penjualan (Unit)")

            ax.set_title(
                f"{obat_dipilih} - "
                f"Prediksi {metrics['bulan_prediksi']}"
            )

            ax.legend()
            ax.grid(alpha=0.3)

            plt.tight_layout()

            st.pyplot(fig)
            plt.close(fig)

            # -----------------------------------------------------------------
            # HASIL EVALUASI PER OBAT
            # -----------------------------------------------------------------

            with st.expander("Lihat detail evaluasi per obat"):

                st.dataframe(
                    evaluasi_df,
                    use_container_width=True,
                    hide_index=True
                )

            # -----------------------------------------------------------------
            # DOWNLOAD HASIL
            # -----------------------------------------------------------------

            st.divider()

            st.subheader("⬇️ Download Hasil")

            col1, col2 = st.columns(2)

            with col1:

                csv_forecast = forecast_df.to_csv(
                    index=False
                ).encode("utf-8-sig")

                st.download_button(
                    label="Download Hasil Forecast (CSV)",
                    data=csv_forecast,
                    file_name=(
                        f"forecast_{metrics['bulan_prediksi']}.csv"
                    ),
                    mime="text/csv",
                    use_container_width=True
                )

            with col2:

                csv_evaluasi = evaluasi_df.to_csv(
                    index=False
                ).encode("utf-8-sig")

                st.download_button(
                    label="Download Evaluasi (CSV)",
                    data=csv_evaluasi,
                    file_name=(
                        f"evaluasi_{metrics['bulan_evaluasi']}.csv"
                    ),
                    mime="text/csv",
                    use_container_width=True
                )

    except Exception as e:

        st.error(
            f"Terjadi kesalahan saat memproses data: {e}"
        )

        st.info(
            "Periksa kembali format CSV, kolom bulan, "
            "dan nilai penjualan Anda."
        )


# =============================================================================
# FOOTER
# =============================================================================

st.divider()

st.caption(
    "Apotek Stock Forecasting | Monthly Prediction System | "
    "Powered by Python, XGBoost & Streamlit"
)
