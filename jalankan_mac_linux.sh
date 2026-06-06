#!/bin/bash
echo "============================================"
echo "  LENSA PELANGGAN - Segmentasi Pelanggan"
echo "============================================"
echo ""
echo "[1/2] Menginstall kebutuhan (hanya perlu sekali)..."
python3 -m pip install -r requirements.txt
echo ""
echo "[2/2] Menjalankan aplikasi..."
echo "Aplikasi akan terbuka otomatis di browser. Untuk berhenti, tekan Ctrl+C."
echo ""
python3 -m streamlit run app.py
