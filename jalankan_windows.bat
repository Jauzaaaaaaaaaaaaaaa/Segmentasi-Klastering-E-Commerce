@echo off
title Lensa Pelanggan - Segmentasi K-Means
echo ============================================
echo   LENSA PELANGGAN - Segmentasi Pelanggan
echo ============================================
echo.
echo [1/2] Menginstall kebutuhan (hanya perlu sekali)...
python -m pip install -r requirements.txt
echo.
echo [2/2] Menjalankan aplikasi...
echo Aplikasi akan terbuka otomatis di browser. Untuk berhenti, tutup jendela ini.
echo.
python -m streamlit run app.py
pause
