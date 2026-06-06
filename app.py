# -*- coding: utf-8 -*-
"""
Lensa Pelanggan — Sistem Segmentasi Pelanggan E-Commerce
Metode: K-Means Clustering berbasis Pola Pembelian (Total Belanja, Jumlah Barang)
        dan Lokasi Geografis (Ongkos Kirim sebagai proksi wilayah).

Dijalankan dengan:  streamlit run app.py
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ----------------------------------------------------------------------------
# KONFIGURASI & KONSTANTA
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Lensa Pelanggan — Segmentasi K-Means",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Tiga fitur yang dipakai untuk clustering (sesuai batasan masalah skripsi)
FITUR = ["Total Pembayaran", "total_qty", "Perkiraan Ongkos Kirim"]
KOLOM_WAJIB = ["Total Pembayaran", "total_qty", "Perkiraan Ongkos Kirim", "Provinsi"]

# Metadata fitur — dipakai bersama oleh pembaca CSV dan form "Isi manual",
# sehingga form manual selalu mengikuti kolom yang diambil sistem dari data.
FITUR_META = {
    "Total Pembayaran":       dict(label="Total pembayaran (Rp)", min=0, value=25000, step=1000,
                                   help="Total nilai belanja satu pelanggan dalam satu pesanan."),
    "total_qty":              dict(label="Jumlah barang (total_qty)", min=1, value=1, step=1,
                                   help="Total barang yang dibeli pelanggan tersebut."),
    "Perkiraan Ongkos Kirim": dict(label="Ongkos kirim (Rp)", min=0, value=10000, step=1000,
                                   help="Perkiraan ongkos kirim — penanda lokasi/wilayah."),
}

# Provinsi yang termasuk Pulau Jawa (untuk analisis geografis)
JAWA = {"JAWA BARAT", "DKI JAKARTA", "BANTEN", "JAWA TENGAH",
        "JAWA TIMUR", "DI YOGYAKARTA"}

# Daftar provinsi untuk pilihan pada mode "Isi manual"
PROVINSI_LIST = [
    "JAWA BARAT", "DKI JAKARTA", "BANTEN", "JAWA TENGAH", "DI YOGYAKARTA", "JAWA TIMUR",
    "ACEH", "SUMATERA UTARA", "SUMATERA BARAT", "RIAU", "KEPULAUAN RIAU", "JAMBI",
    "SUMATERA SELATAN", "KEPULAUAN BANGKA BELITUNG", "BENGKULU", "LAMPUNG",
    "BALI", "NUSA TENGGARA BARAT", "NUSA TENGGARA TIMUR",
    "KALIMANTAN BARAT", "KALIMANTAN TENGAH", "KALIMANTAN SELATAN", "KALIMANTAN TIMUR",
    "KALIMANTAN UTARA", "SULAWESI UTARA", "GORONTALO", "SULAWESI TENGAH", "SULAWESI BARAT",
    "SULAWESI SELATAN", "SULAWESI TENGGARA", "MALUKU", "MALUKU UTARA", "PAPUA", "PAPUA BARAT",
]

# Palet warna segmen — dipilih agar elegan & mudah dibedakan
SEG_COLORS = ["#0E7C6B", "#C2683A", "#3B5BA5", "#9A4C95",
              "#B5843A", "#6E8B3D", "#A23B53", "#2F6E8F"]

DATA_BAWAAN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_penjualan.csv")


# ----------------------------------------------------------------------------
# CSS — gaya antarmuka (editorial / fintech yang rapi)
# ----------------------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600&family=Plus+Jakarta+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root{
  --bg:#FAF7F2; --surface:#FFFFFF; --ink:#211E1B; --muted:#6F6A62;
  --hairline:#E9E3D9; --accent:#0E7C6B; --accent-soft:#E7F0EC;
}
html, body, [class*="css"], .stApp, button, input, textarea{
  font-family:'Plus Jakarta Sans', sans-serif;
}
.stApp{ background:var(--bg); }
[data-testid="stHeader"]{ background:transparent; }
[data-testid="stToolbar"], #MainMenu, footer, [data-testid="stDecoration"]{ display:none !important; }
.block-container{ padding-top:1.6rem; padding-bottom:3rem; max-width:1180px; }
h1,h2,h3,h4{ font-family:'Fraunces', Georgia, serif !important; color:var(--ink) !important;
  letter-spacing:-0.01em; font-weight:600 !important; }
.stApp [data-testid="stMarkdownContainer"] p{ color:var(--ink); line-height:1.7; }

[data-testid="stSidebar"]{ background:var(--surface); border-right:1px solid var(--hairline); }
[data-testid="stSidebar"] .block-container{ padding-top:1.1rem; }

.lp-kicker{ font-size:12px; font-weight:600; letter-spacing:.18em; color:var(--accent); text-transform:uppercase; }
.lp-hero h1{ font-size:2.5rem; line-height:1.08; margin:.4rem 0 .55rem; }
.lp-hero p{ color:var(--muted); font-size:1.02rem; max-width:700px; }

.lp-card{ background:var(--surface); border:1px solid var(--hairline); border-radius:16px;
  padding:20px 22px; transition:transform .18s ease, box-shadow .18s ease; height:100%; }
.lp-card:hover{ transform:translateY(-2px); box-shadow:0 14px 32px -20px rgba(33,30,27,.40); }

.lp-kpi-label{ font-size:12.5px; color:var(--muted); font-weight:500; letter-spacing:.02em; }
.lp-kpi-value{ font-family:'JetBrains Mono', monospace; font-size:2rem; font-weight:500;
  color:var(--ink); line-height:1.1; margin-top:6px; }
.lp-kpi-sub{ font-size:12px; color:var(--muted); margin-top:4px; }

.lp-section-title{ font-family:'Fraunces',serif; font-size:1.55rem; font-weight:600; margin:.1rem 0 .25rem; }
.lp-lead{ color:var(--muted); margin-bottom:1.15rem; max-width:780px; line-height:1.7; }
.lp-pill{ display:inline-block; padding:3px 11px; border-radius:999px; font-size:12px; font-weight:600; }

[data-baseweb="tab-list"]{ gap:2px; border-bottom:1px solid var(--hairline); }
button[data-baseweb="tab"]{ font-family:'Plus Jakarta Sans'; font-weight:500; color:var(--muted); padding:10px 18px; }
button[data-baseweb="tab"][aria-selected="true"]{ color:var(--ink); }
[data-baseweb="tab-highlight"]{ background:var(--accent) !important; height:2px; }

.stDownloadButton button, .stButton button{ border-radius:10px; border:1px solid var(--accent);
  background:var(--accent); color:#fff !important; font-weight:600; }
.stDownloadButton button:hover, .stButton button:hover{ background:#0c6a5b; border-color:#0c6a5b; }

.lp-step-no{ font-family:'JetBrains Mono',monospace; font-size:13px; color:var(--accent); font-weight:500; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# FUNGSI BANTU
# ----------------------------------------------------------------------------
def fmt_int(n):
    """Format angka dengan pemisah ribuan gaya Indonesia (1.234.567)."""
    if pd.isna(n):
        return "–"
    return f"{int(round(float(n))):,}".replace(",", ".")


def rp(n):
    if pd.isna(n):
        return "–"
    return "Rp " + fmt_int(n)


def style_fig(fig, h=430, legend_top=True, top=64):
    """Memberi gaya konsisten ke seluruh grafik Plotly.

    Judul grafik TIDAK ditaruh di dalam grafik (agar tidak bertabrakan dengan
    legend); judul ditampilkan terpisah sebagai teks di atas grafik.
    """
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Plus Jakarta Sans, sans-serif", color="#211E1B", size=13),
        margin=dict(l=10, r=10, t=(top if legend_top else 24), b=10),
        height=h,
        hoverlabel=dict(font_family="Plus Jakarta Sans", bgcolor="#211E1B", font_color="#fff"),
        title="",
    )
    if legend_top:
        fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.04,
                                      xanchor="left", x=0, font=dict(size=12)))
    fig.update_xaxes(gridcolor="#EFE9DF", zeroline=False, linecolor="#E1DACE", ticks="outside",
                     tickcolor="#E1DACE")
    fig.update_yaxes(gridcolor="#EFE9DF", zeroline=False, linecolor="#E1DACE", ticks="outside",
                     tickcolor="#E1DACE")
    return fig


def chart_title(text):
    """Judul kecil di atas grafik (menggantikan judul bawaan Plotly)."""
    st.markdown(f"<div style='font-family:Fraunces,serif;font-weight:600;font-size:1.04rem;"
                f"margin:2px 0 4px;color:#211E1B;'>{text}</div>", unsafe_allow_html=True)


def render_form_manual(caption):
    """Formulir tambah data manual. Field dibuat otomatis dari kolom yang dipakai
    sistem (FITUR), sehingga selalu mengikuti data yang diambil dari CSV.
    Mengembalikan DataFrame baris manual (atau None bila kosong)."""
    if "manual_rows" not in st.session_state:
        st.session_state["manual_rows"] = []
    st.caption(caption)
    with st.form("form_manual", clear_on_submit=True):
        nilai = {}
        for kolom in FITUR:                       # ikut kolom yang diambil dari CSV
            meta = FITUR_META[kolom]
            nilai[kolom] = st.number_input(
                meta["label"], min_value=meta["min"], value=meta["value"],
                step=meta["step"], help=meta["help"])
        m_prov = st.selectbox("Provinsi", PROVINSI_LIST)
        tambah = st.form_submit_button("➕  Tambah pelanggan")
    if tambah:
        baris = {kolom: float(nilai[kolom]) for kolom in FITUR}
        if "total_qty" in baris:
            baris["total_qty"] = int(baris["total_qty"])
        baris["Provinsi"] = m_prov
        baris["Status Pesanan"] = "Selesai"
        st.session_state["manual_rows"].append(baris)
    n_manual = len(st.session_state["manual_rows"])
    st.caption(f"✓ {n_manual} baris manual ditambahkan.")
    if n_manual > 0:
        if st.button("🗑  Kosongkan data manual"):
            st.session_state["manual_rows"] = []
            (st.rerun if hasattr(st, "rerun") else st.experimental_rerun)()
        return pd.DataFrame(st.session_state["manual_rows"])
    return None


# ----------------------------------------------------------------------------
# PEMROSESAN DATA (dengan cache agar cepat)
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data(file_bytes=None, name=""):
    """Memuat data dari unggahan pengguna (CSV/Excel) atau dari file bawaan.

    Pembaca CSV dibuat tahan banting: pemisah kolom (koma/titik koma/tab)
    dideteksi otomatis, beberapa encoding dicoba, dan nama kolom dibersihkan
    dari spasi serta penanda BOM.
    """
    def _bersihkan(df):
        if df is not None:
            df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]
        return df

    name = (name or "").lower()

    if file_bytes is not None:
        if name.endswith((".xlsx", ".xls")):
            return _bersihkan(pd.read_excel(file_bytes))
        for enc in ("utf-8-sig", "utf-8", "latin-1"):
            try:
                file_bytes.seek(0)
            except Exception:
                pass
            try:
                return _bersihkan(pd.read_csv(file_bytes, sep=None, engine="python",
                                              encoding=enc))
            except Exception:
                continue
        try:
            file_bytes.seek(0)
        except Exception:
            pass
        return _bersihkan(pd.read_csv(file_bytes))

    if os.path.exists(DATA_BAWAAN):
        for enc in ("utf-8-sig", "utf-8", "latin-1"):
            try:
                return _bersihkan(pd.read_csv(DATA_BAWAAN, sep=None, engine="python",
                                              encoding=enc))
            except Exception:
                continue
        return _bersihkan(pd.read_csv(DATA_BAWAAN))
    return None


@st.cache_data(show_spinner=False)
def preprocess(df):
    """
    Tahap Data Preparation:
    1. Ambil hanya pesanan berstatus 'Selesai' (transaksi valid).
    2. Buang baris kosong & pembayaran <= 0.
    3. Transformasi log (meredam outlier ekstrem) lalu standarisasi skala.
    """
    d = df.copy()
    if "Status Pesanan" in d.columns:
        d = d[d["Status Pesanan"].astype(str).str.strip().str.lower() == "selesai"]
    # pastikan kolom fitur bertipe angka (mis. bila terbaca sebagai teks)
    for c in FITUR:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=FITUR)
    d = d[d["Total Pembayaran"] > 0].reset_index(drop=True)

    X_log = np.log1p(d[FITUR].values.astype(float))
    Xs = StandardScaler().fit_transform(X_log)
    return d, Xs


@st.cache_data(show_spinner=False)
def compute_k_metrics(Xs, k_min=2, k_max=8):
    """Hitung WCSS (Elbow) dan Silhouette untuk setiap K."""
    ks, wcss, sils = [], [], []
    rng = np.random.RandomState(42)
    idx = rng.choice(len(Xs), size=min(4000, len(Xs)), replace=False)
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(Xs)
        ks.append(k)
        wcss.append(km.inertia_)
        sils.append(silhouette_score(Xs[idx], km.labels_[idx]))
    return ks, wcss, sils


@st.cache_data(show_spinner=False)
def run_kmeans(Xs, k):
    km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(Xs)
    return km.labels_


def make_name(tier, bulk):
    """Nama segmen yang mudah dipahami, dari tingkat nilai & pola beli."""
    if bulk and tier == "Premium":
        return "Pelanggan Grosir Premium"
    if bulk:
        return "Pembeli Grosir"
    if tier == "Premium":
        return "Pelanggan Premium"
    if tier == "Hemat":
        return "Pembeli Hemat"
    return "Pembeli Reguler"


def geo_of(pct_jawa):
    if pct_jawa >= 85:
        return "Dominan Pulau Jawa"
    if pct_jawa <= 60:
        return "Jangkauan Luar Jawa"
    return "Sebaran Campuran"


def profile_segments(d, labels):
    """Membuat profil tiap segmen, diurutkan dari belanja terendah ke tertinggi."""
    d = d.copy()
    d["_seg_raw"] = labels

    rows = []
    for raw in sorted(d["_seg_raw"].unique()):
        g = d[d["_seg_raw"] == raw]
        rows.append(dict(
            raw=raw, n=len(g), pct=len(g) / len(d) * 100,
            med_pay=g["Total Pembayaran"].median(),
            med_qty=g["total_qty"].median(),
            med_ong=g["Perkiraan Ongkos Kirim"].median(),
            pct_jawa=g["Provinsi"].isin(JAWA).mean() * 100,
            top_prov=", ".join(g["Provinsi"].value_counts().head(3).index.str.title()),
        ))
    rows.sort(key=lambda x: x["med_pay"])

    K = len(rows)
    for i, r in enumerate(rows):
        pos = i / (K - 1) if K > 1 else 0.0
        tier = "Premium" if pos >= 0.67 else ("Hemat" if pos <= 0.33 else "Reguler")
        bulk = r["med_qty"] >= 4
        r["tier"] = tier
        r["geo"] = geo_of(r["pct_jawa"])
        r["name"] = make_name(tier, bulk)
        r["color"] = SEG_COLORS[i % len(SEG_COLORS)]
        r["idx"] = i

    # Hindari nama segmen kembar saat K besar: beri akhiran angka bila perlu.
    base_counts = {}
    for r in rows:
        base_counts[r["name"]] = base_counts.get(r["name"], 0) + 1
    used = {}
    for r in rows:
        if base_counts[r["name"]] > 1:
            used[r["name"]] = used.get(r["name"], 0) + 1
            r["name"] = f"{r['name']} {used[r['name']]}"

    remap = {r["raw"]: r["idx"] for r in rows}
    return rows, remap


# ----------------------------------------------------------------------------
# SIDEBAR — kontrol
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("<div class='lp-kicker'>Lensa Pelanggan</div>", unsafe_allow_html=True)
    st.markdown("<div style='font-family:Fraunces,serif;font-size:1.35rem;font-weight:600;"
                "margin:.1rem 0 .2rem;'>Panel Kontrol</div>", unsafe_allow_html=True)
    st.caption("Segmentasi pelanggan dengan metode K-Means Clustering.")
    st.markdown("---")

    st.markdown("**Sumber data**")
    sumber = st.radio(
        "Pilih sumber data",
        ["Dataset bawaan", "Unggah file (CSV/Excel)", "Isi manual (form)"],
        label_visibility="collapsed",
    )

    up = None
    manual_df = None
    manual_extra = None

    if sumber == "Unggah file (CSV/Excel)":
        up = st.file_uploader("Unggah data Anda", type=["csv", "xlsx", "xls"],
                              help="File harus memuat kolom: Total Pembayaran, total_qty, "
                                   "Perkiraan Ongkos Kirim, Provinsi.")
        if up is not None:
            st.caption(f"✓ Memakai: {up.name}")
        else:
            st.caption("Belum ada file. Unggah CSV/Excel untuk melanjutkan.")
        st.markdown("---")
        st.markdown("**Tambah data manual (opsional)**")
        manual_extra = render_form_manual(
            "Lupa memasukkan beberapa transaksi ke file? Tambahkan langsung di sini — "
            "barisnya akan digabung dengan data yang diunggah.")

    elif sumber == "Isi manual (form)":
        manual_df = render_form_manual(
            "Isi data satu per satu seperti formulir — kolomnya sama dengan "
            "yang dibaca dari Excel.")

    else:
        st.caption("✓ Memakai dataset bawaan (data penjualan 2023–2025).")

    # --- muat & proses data dari sumber terpilih (sebelum menentukan jumlah segmen) ---
    if sumber == "Isi manual (form)":
        raw_df = manual_df
        if raw_df is None or len(raw_df) == 0:
            st.info("Tambahkan minimal beberapa baris lewat formulir **Isi manual** "
                    "untuk menjalankan segmentasi.")
            st.stop()
    elif sumber == "Unggah file (CSV/Excel)":
        if up is None:
            st.info("Unggah file CSV/Excel di atas untuk melanjutkan, "
                    "atau pilih sumber data lain.")
            st.stop()
        raw_df = load_data(up, up.name)
        if manual_extra is not None and len(manual_extra) > 0:
            extra = manual_extra.copy()
            if "Status Pesanan" not in raw_df.columns:
                extra = extra.drop(columns=["Status Pesanan"], errors="ignore")
            raw_df = pd.concat([raw_df, extra], ignore_index=True)
    else:
        raw_df = load_data(None, "")

    if raw_df is None:
        st.error("Dataset tidak ditemukan. Pastikan file **data_penjualan.csv** berada di folder "
                 "yang sama dengan app.py, atau unggah data Anda.")
        st.stop()
    _missing = [c for c in KOLOM_WAJIB if c not in raw_df.columns]
    if _missing:
        st.error("Data tidak memiliki kolom yang dibutuhkan: " + ", ".join(_missing) +
                 ".  Kolom wajib: " + ", ".join(KOLOM_WAJIB))
        st.stop()

    d, Xs = preprocess(raw_df)
    n_rows = len(d)
    if n_rows < 3:
        st.warning("Data valid kurang dari 3 baris (setelah menyaring transaksi selesai & "
                   "nilai yang sah). Tambahkan lebih banyak data untuk menjalankan segmentasi.")
        st.stop()

    st.markdown("---")
    st.markdown("**Jumlah segmen (K)**")
    maks_k = min(8, n_rows)
    K = st.slider("Pilih jumlah kelompok", 2, maks_k, min(3, maks_k),
                  label_visibility="collapsed")
    if maks_k < 8:
        st.caption(f"Maksimum **{maks_k}** segmen karena data yang dipakai hanya **{n_rows}** "
                   f"baris — jumlah segmen tidak boleh melebihi jumlah data.")
    else:
        st.caption("Disarankan **K = 3** (keseimbangan kualitas & kemudahan interpretasi). "
                   "Lihat tab *Penentuan Segmen* untuk dasarnya.")

    st.markdown("---")
    st.markdown("**Tentang**")
    st.caption("Variabel: total belanja, jumlah barang, dan ongkos kirim (proksi lokasi). "
               "Provinsi dipakai untuk menjelaskan profil tiap segmen.")


# ----------------------------------------------------------------------------
# JALANKAN K-MEANS (data sudah dimuat & dipraproses di panel kiri)
# ----------------------------------------------------------------------------
with st.spinner("Menjalankan K-Means…"):
    k_metric_max = min(8, n_rows - 1)   # silhouette butuh K <= (jumlah data - 1)
    ks, wcss, sils = compute_k_metrics(Xs, 2, k_metric_max)
    labels = run_kmeans(Xs, K)
    profiles, remap = profile_segments(d, labels)

d = d.copy()
d["_seg_raw"] = labels
d["Segmen"] = d["_seg_raw"].map({p["raw"]: p["name"] for p in profiles})
sil_now = sils[ks.index(K)] if K in ks else float("nan")


# ----------------------------------------------------------------------------
# HERO
# ----------------------------------------------------------------------------
st.markdown(
    "<div class='lp-hero'>"
    "<div class='lp-kicker'>Segmentasi Pelanggan E-Commerce</div>"
    "<h1>Memahami pelanggan lewat<br>pola belanja &amp; lokasi.</h1>"
    "<p>Sistem ini mengelompokkan transaksi secara otomatis menggunakan "
    "<b>K-Means Clustering</b> berdasarkan tiga hal: berapa besar belanja, berapa banyak barang, "
    "dan ongkos kirim sebagai penanda wilayah — lalu menerjemahkannya menjadi segmen yang "
    "mudah dipahami dan ditindaklanjuti.</p>"
    "</div>",
    unsafe_allow_html=True,
)
st.markdown("<div style='height:18px;'></div>", unsafe_allow_html=True)

# KPI
n_prov = d["Provinsi"].nunique()
_sil_txt = f"{sil_now:.2f}" if not pd.isna(sil_now) else "–"
kpis = [
    ("Transaksi dianalisis", fmt_int(len(d)), "pesanan selesai"),
    ("Segmen terbentuk", str(len(profiles)), "kelompok pelanggan"),
    ("Belanja (median)", rp(d["Total Pembayaran"].median()), "nilai per transaksi"),
    ("Skor kualitas", _sil_txt, "silhouette (−1 s/d 1)"),
]
cols = st.columns(4)
for c, (lab, val, sub) in zip(cols, kpis):
    c.markdown(f"<div class='lp-card'><div class='lp-kpi-label'>{lab}</div>"
               f"<div class='lp-kpi-value'>{val}</div>"
               f"<div class='lp-kpi-sub'>{sub}</div></div>", unsafe_allow_html=True)

st.markdown("<div style='height:26px;'></div>", unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# TABS
# ----------------------------------------------------------------------------
t_metode, t_k, t_hasil, t_geo = st.tabs([
    "  Metode  ", "  Penentuan Segmen  ", "  Hasil Segmentasi  ",
    "  Sebaran Geografis  ",
])

# ---- TAB METODE ----
with t_metode:
    st.markdown("<div class='lp-section-title'>Bagaimana sistem ini bekerja</div>",
                unsafe_allow_html=True)
    st.markdown("<p class='lp-lead'>K-Means mengelompokkan data dengan cara menaruh setiap "
                "transaksi sebagai titik, lalu mengumpulkan titik-titik yang berdekatan ke dalam "
                "kelompok yang sama. Prosesnya diulang sampai kelompoknya stabil.</p>",
                unsafe_allow_html=True)

    steps = [
        ("01", "Tentukan jumlah kelompok", "Kita memilih akan ada berapa segmen (K). Komputer menaruh titik pusat awal secara acak."),
        ("02", "Kelompokkan ke pusat terdekat", "Setiap transaksi bergabung ke titik pusat yang paling dekat dengannya."),
        ("03", "Geser pusat ke tengah", "Tiap pusat dipindahkan ke posisi rata-rata anggota kelompoknya."),
        ("04", "Ulangi sampai stabil", "Langkah 2–3 diulang hingga posisi pusat tidak berubah lagi. Segmen final terbentuk."),
    ]
    cs = st.columns(4)
    for c, (no, title, desc) in zip(cs, steps):
        c.markdown(f"<div class='lp-card'><div class='lp-step-no'>{no}</div>"
                   f"<div style='font-family:Fraunces,serif;font-weight:600;font-size:1.05rem;"
                   f"margin:6px 0 6px;'>{title}</div>"
                   f"<div style='color:var(--muted);font-size:13px;line-height:1.6;'>{desc}</div></div>",
                   unsafe_allow_html=True)

    st.markdown("<div style='height:26px;'></div>", unsafe_allow_html=True)
    st.markdown("<div class='lp-section-title'>Tiga variabel yang dipakai</div>",
                unsafe_allow_html=True)
    feats = [
        ("Total belanja", "Seberapa besar uang yang dibelanjakan dalam satu pesanan — menggambarkan nilai pelanggan."),
        ("Jumlah barang", "Berapa banyak barang yang dibeli — membedakan pembeli eceran dan pembeli borongan."),
        ("Ongkos kirim", "Penanda lokasi: ongkir murah berarti dekat (Jawa), ongkir mahal berarti jauh (luar Jawa)."),
    ]
    cf = st.columns(3)
    for c, (title, desc) in zip(cf, feats):
        c.markdown(f"<div class='lp-card'><div style='font-family:Fraunces,serif;font-weight:600;"
                   f"font-size:1.08rem;margin-bottom:7px;'>{title}</div>"
                   f"<div style='color:var(--muted);font-size:13px;line-height:1.6;'>{desc}</div></div>",
                   unsafe_allow_html=True)

    st.markdown("<div style='height:26px;'></div>", unsafe_allow_html=True)
    cL, cR = st.columns([1, 1])
    with cL:
        st.markdown("<div class='lp-section-title'>Rumus yang dipakai</div>", unsafe_allow_html=True)
        st.markdown("**1. Jarak Euclidean** — mengukur seberapa dekat dua titik.")
        st.latex(r"d(x,y)=\sqrt{\sum_{i=1}^{n}(x_i-y_i)^2}")
        st.markdown("**2. WCSS** — total kerapatan kelompok (dipakai oleh Elbow). Makin kecil makin rapat.")
        st.latex(r"WCSS=\sum_{j=1}^{k}\sum_{x\in C_j}\lVert x-\mu_j\rVert^{2}")
        st.markdown("**3. Silhouette** — nilai kualitas pemisahan (−1 s/d 1). Makin mendekati 1 makin baik.")
        st.latex(r"S=\frac{b-a}{\max(a,\,b)}")
    with cR:
        st.markdown("<div class='lp-section-title'>Kenapa K-Means?</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='lp-card'><div style='color:var(--ink);line-height:1.7;font-size:14px;'>"
            "Data ini tidak memiliki label jawaban (tidak ada kolom yang menyatakan pelanggan "
            "termasuk segmen apa). Karena itu metode <b>tanpa pengawasan</b> (unsupervised) seperti "
            "clustering paling cocok — ia menemukan sendiri kelompoknya.<br><br>"
            "K-Means dipilih karena ringan, cepat untuk data transaksi yang besar, dan hasilnya "
            "mudah dijelaskan ke orang non-teknis. Perbandingan teknis dengan metode lain "
            "(seperti DBSCAN atau Hierarchical) akan dibahas terpisah di notebook Google Colab.</div></div>",
            unsafe_allow_html=True)

# ---- TAB PENENTUAN SEGMEN ----
with t_k:
    st.markdown("<div class='lp-section-title'>Berapa jumlah segmen yang ideal?</div>",
                unsafe_allow_html=True)
    st.markdown("<p class='lp-lead'>Dua alat bantu dipakai bersama. <b>Elbow</b> mencari titik "
                "di mana penambahan kelompok tidak lagi banyak menurunkan WCSS (membentuk siku). "
                "<b>Silhouette</b> menilai kualitas pemisahan. Geser nilai K di panel kiri untuk "
                "melihat efeknya — pilihan saat ini ditandai.</p>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        chart_title("Metode Elbow (WCSS)")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ks, y=wcss, mode="lines+markers",
                                 line=dict(color="#3B5BA5", width=2.5),
                                 marker=dict(size=8, color="#3B5BA5"), name="WCSS"))
        if K in ks:
            fig.add_trace(go.Scatter(x=[K], y=[wcss[ks.index(K)]], mode="markers",
                                     marker=dict(size=15, color="#C2683A",
                                                 line=dict(color="#fff", width=2)),
                                     name=f"K terpilih = {K}"))
        fig.update_xaxes(title="Jumlah kelompok (K)", dtick=1)
        fig.update_yaxes(title="WCSS")
        st.plotly_chart(style_fig(fig, 400), use_container_width=True)
    with c2:
        chart_title("Skor Silhouette per K")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ks, y=sils, mode="lines+markers",
                                 line=dict(color="#0E7C6B", width=2.5),
                                 marker=dict(size=8, color="#0E7C6B"), name="Silhouette"))
        if K in ks:
            fig.add_trace(go.Scatter(x=[K], y=[sils[ks.index(K)]], mode="markers",
                                     marker=dict(size=15, color="#C2683A",
                                                 line=dict(color="#fff", width=2)),
                                     name=f"K terpilih = {K}"))
        fig.update_xaxes(title="Jumlah kelompok (K)", dtick=1)
        fig.update_yaxes(title="Silhouette")
        st.plotly_chart(style_fig(fig, 400), use_container_width=True)

    best_k = ks[int(np.argmax(sils))]
    if 3 in ks:
        st.info(f"Secara matematis, skor silhouette tertinggi ada di **K = {best_k}** "
                f"({max(sils):.2f}). Namun **K = 3** dipilih sebagai default karena memberi segmen "
                f"yang lebih kaya dan actionable, sementara skornya masih tergolong baik "
                f"({sils[ks.index(3)]:.2f}) dan grafik Elbow juga menekuk di sekitar titik itu.")
    else:
        st.info(f"Secara matematis, skor silhouette tertinggi ada di **K = {best_k}** "
                f"({max(sils):.2f}).")

# ---- TAB HASIL SEGMENTASI ----
with t_hasil:
    st.markdown("<div class='lp-section-title'>Peta sebaran transaksi</div>", unsafe_allow_html=True)
    st.markdown("<p class='lp-lead'>Setiap titik adalah satu transaksi, diposisikan menurut tiga "
                "variabelnya dan diwarnai sesuai segmen. Tanda berlian besar adalah pusat tiap "
                "segmen. (Sebagian kecil outlier ekstrem disembunyikan agar tampilan lebih jelas; "
                "Anda bisa memutar grafik 3D ini.)</p>", unsafe_allow_html=True)

    # batasi tampilan ke <= persentil 99 supaya tidak terdistorsi outlier
    qp = d["Total Pembayaran"].quantile(0.99)
    qq = d["total_qty"].quantile(0.99)
    qo = d["Perkiraan Ongkos Kirim"].quantile(0.99)
    view = d[(d["Total Pembayaran"] <= qp) & (d["total_qty"] <= qq) &
             (d["Perkiraan Ongkos Kirim"] <= qo)]
    if len(view) > 3500:
        view = view.sample(3500, random_state=42)

    fig = go.Figure()
    for p in profiles:
        g = view[view["_seg_raw"] == p["raw"]]
        fig.add_trace(go.Scatter3d(
            x=g["Total Pembayaran"], y=g["total_qty"], z=g["Perkiraan Ongkos Kirim"],
            mode="markers", name=p["name"],
            marker=dict(size=2.6, color=p["color"], opacity=0.72),
            hovertemplate="Belanja Rp %{x:,.0f}<br>Barang %{y}<br>Ongkir Rp %{z:,.0f}<extra></extra>"))
        fig.add_trace(go.Scatter3d(
            x=[p["med_pay"]], y=[p["med_qty"]], z=[p["med_ong"]], mode="markers",
            marker=dict(size=7, color=p["color"], symbol="diamond",
                        line=dict(color="#211E1B", width=1)),
            showlegend=False, hovertemplate=f"Pusat {p['name']}<extra></extra>"))
    fig.update_layout(
        height=540, paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Plus Jakarta Sans, sans-serif", color="#211E1B", size=12),
        margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=0.0, xanchor="left", x=0),
        scene=dict(
            xaxis=dict(title="Total belanja", backgroundcolor="rgba(0,0,0,0)",
                       gridcolor="#E9E3D9", showbackground=True),
            yaxis=dict(title="Jumlah barang", backgroundcolor="rgba(0,0,0,0)",
                       gridcolor="#E9E3D9", showbackground=True),
            zaxis=dict(title="Ongkos kirim", backgroundcolor="rgba(0,0,0,0)",
                       gridcolor="#E9E3D9", showbackground=True)))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
    st.markdown("<div class='lp-section-title'>Profil setiap segmen</div>", unsafe_allow_html=True)
    st.markdown("<p class='lp-lead'>Diurutkan dari belanja terendah ke tertinggi.</p>",
                unsafe_allow_html=True)

    cards = "<div style='display:flex;flex-wrap:wrap;gap:16px;'>"
    for p in profiles:
        cards += (
            f"<div class='lp-card' style='flex:1 1 300px;min-width:280px;border-top:3px solid {p['color']};'>"
            f"<div style='display:flex;align-items:center;gap:8px;'>"
            f"<span style='width:10px;height:10px;border-radius:50%;background:{p['color']};"
            f"display:inline-block;'></span>"
            f"<span style='font-weight:600;font-size:1.08rem;font-family:Fraunces,serif;'>{p['name']}</span></div>"
            f"<div style='color:var(--muted);font-size:12.5px;margin:5px 0 16px;'>"
            f"{p['geo']} · {p['pct']:.0f}% transaksi ({fmt_int(p['n'])} pesanan)</div>"
            f"<div style='display:flex;gap:20px;flex-wrap:wrap;'>"
            f"<div><div style='font-size:11.5px;color:var(--muted);'>Belanja</div>"
            f"<div style='font-family:\"JetBrains Mono\",monospace;font-weight:500;'>{rp(p['med_pay'])}</div></div>"
            f"<div><div style='font-size:11.5px;color:var(--muted);'>Barang</div>"
            f"<div style='font-family:\"JetBrains Mono\",monospace;font-weight:500;'>{int(p['med_qty'])}</div></div>"
            f"<div><div style='font-size:11.5px;color:var(--muted);'>Ongkir</div>"
            f"<div style='font-family:\"JetBrains Mono\",monospace;font-weight:500;'>{rp(p['med_ong'])}</div></div></div>"
            f"<div style='margin-top:15px;font-size:12.5px;color:var(--muted);line-height:1.55;'>"
            f"Wilayah teratas: {p['top_prov']}</div></div>")
    cards += "</div>"
    st.markdown(cards, unsafe_allow_html=True)

    st.markdown("<div style='height:18px;'></div>", unsafe_allow_html=True)
    st.markdown("<div class='lp-section-title'>Tabel data &amp; hasil segmen</div>",
                unsafe_allow_html=True)
    st.markdown("<p class='lp-lead'>Setiap baris data yang dipakai beserta segmen hasil "
                "pengelompokannya. Klik judul kolom untuk mengurutkan.</p>",
                unsafe_allow_html=True)
    tampil = d[[c for c in ["Total Pembayaran", "total_qty", "Perkiraan Ongkos Kirim",
                            "Provinsi", "Segmen"] if c in d.columns]].copy()
    tampil = tampil.rename(columns={
        "Total Pembayaran": "Total belanja (Rp)", "total_qty": "Jumlah barang",
        "Perkiraan Ongkos Kirim": "Ongkos kirim (Rp)"})
    st.dataframe(tampil, use_container_width=True, height=360)
    st.caption(f"Menampilkan {fmt_int(len(tampil))} baris data.")

    st.markdown("<div style='height:18px;'></div>", unsafe_allow_html=True)
    out_cols = [c for c in ["order_id", "Total Pembayaran", "total_qty",
                            "Perkiraan Ongkos Kirim", "Provinsi", "Segmen"] if c in d.columns]
    csv = d[out_cols].to_csv(index=False).encode("utf-8")
    st.download_button("⬇  Unduh hasil segmentasi (CSV)", csv,
                       file_name="hasil_segmentasi.csv", mime="text/csv")

# ---- TAB SEBARAN GEOGRAFIS ----
with t_geo:
    st.markdown("<div class='lp-section-title'>Sebaran segmen menurut wilayah</div>",
                unsafe_allow_html=True)
    st.markdown("<p class='lp-lead'>Inilah sisi kebaruan penelitian: melihat bagaimana tiap segmen "
                "tersebar secara geografis. Terlihat jelas pasar masih terpusat di Pulau Jawa.</p>",
                unsafe_allow_html=True)

    c1, c2 = st.columns([1.7, 1])
    with c1:
        chart_title("12 provinsi teratas — komposisi tiap segmen")
        top_prov = d["Provinsi"].value_counts().head(12).index.tolist()
        sub = d[d["Provinsi"].isin(top_prov)]
        fig = go.Figure()
        for p in profiles:
            g = sub[sub["_seg_raw"] == p["raw"]]
            counts = g.groupby("Provinsi").size().reindex(top_prov).fillna(0)
            fig.add_trace(go.Bar(y=[x.title() for x in top_prov], x=counts.values,
                                 name=p["name"], orientation="h", marker_color=p["color"]))
        fig.update_layout(barmode="stack")
        fig.update_xaxes(title="Jumlah transaksi")
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(style_fig(fig, 500, top=96), use_container_width=True)
    with c2:
        chart_title("Jawa vs Luar Jawa")
        jawa_share = d["Provinsi"].isin(JAWA).mean()
        fig = go.Figure(go.Pie(values=[jawa_share, 1 - jawa_share],
                               labels=["Pulau Jawa", "Luar Jawa"], hole=0.62,
                               marker_colors=["#0E7C6B", "#C2683A"], sort=False,
                               textinfo="label+percent",
                               textfont=dict(family="Plus Jakarta Sans", size=13)))
        fig.update_layout(showlegend=False)
        st.plotly_chart(style_fig(fig, 320, legend_top=False), use_container_width=True)
        og_jawa = d[d["Provinsi"].isin(JAWA)]["Perkiraan Ongkos Kirim"].median()
        og_luar = d[~d["Provinsi"].isin(JAWA)]["Perkiraan Ongkos Kirim"].median()
        st.markdown(
            f"<div class='lp-card'><div style='font-size:13px;color:var(--muted);line-height:1.7;'>"
            f"Ongkir median di <b>Jawa {rp(og_jawa)}</b> vs <b>Luar Jawa {rp(og_luar)}</b>. "
            f"Selisih inilah yang membuat lokasi penting dalam strategi harga & promo.</div></div>",
            unsafe_allow_html=True)

st.markdown("<div style='height:30px;'></div>", unsafe_allow_html=True)
st.markdown("<div style='text-align:center;color:var(--muted);font-size:12px;'>"
            "Lensa Pelanggan · Segmentasi K-Means · Pola Belanja &amp; Lokasi Geografis</div>",
            unsafe_allow_html=True)
