# Pengolahan Citra Digital (PCD) - Web Application

Aplikasi web ini adalah proyek Pengolahan Citra Digital yang dikembangkan menggunakan kerangka kerja web FastAPI di Python. Aplikasi ini memungkinkan pengguna untuk melakukan berbagai operasi pengolahan citra dasar dan lanjutan melalui antarmuka web yang sederhana.

-----

### Fitur Utama

Aplikasi ini mencakup berbagai operasi pengolahan citra yang dapat diakses melalui antarmuka pengguna:

  * **Operasi Aritmatika**: Lakukan operasi penambahan, pengurangan, maksimum, minimum, dan invers pada citra.
  * **Operasi Logika**: Lakukan operasi logika seperti AND, OR, dan XOR dengan dua gambar, atau operasi NOT dengan satu gambar.
  * **Konversi Grayscale**: Konversikan gambar berwarna menjadi citra grayscale.
  * **Analisis Histogram**: Hasilkan histogram skala abu-abu dan berwarna untuk menganalisis distribusi intensitas piksel.
  * **Equalisasi Histogram**: Tingkatkan kontras gambar menggunakan teknik equalisasi histogram.
  * **Spesifikasi Histogram**: Sesuaikan histogram gambar sumber agar cocok dengan histogram gambar referensi.
  * **Statistik Citra**: Hitung rata-rata intensitas piksel dan standar deviasi citra grayscale yang diunggah.

-----

### Teknologi yang Digunakan

  * **FastAPI**: Kerangka kerja web yang cepat dan modern untuk membangun API dengan Python.
  * **OpenCV (`cv2`)**: Pustaka utama untuk operasi pengolahan citra.
  * **NumPy (`np`)**: Digunakan untuk manipulasi data citra sebagai array numerik.
  * **Scikit-image (`skimage`)**: Pustaka untuk algoritma pengolahan citra tingkat lanjut, khususnya `match_histograms` untuk spesifikasi histogram.
  * **Matplotlib (`plt`)**: Digunakan untuk menghasilkan dan menyimpan plot histogram.
  * **Jinja2Templates**: Digunakan untuk rendering template HTML.
  * **Bootstrap & Font Awesome**: Digunakan untuk desain responsif dan ikon pada antarmuka web.

-----

### Cara Menjalankan Proyek

1.  **Instalasi Dependensi**: Pastikan Anda telah menginstal pustaka yang diperlukan. Anda dapat menginstalnya menggunakan pip:
    ```bash
    pip install fastapi "uvicorn[standard]" opencv-python numpy scikit-image matplotlib
    ```
2.  **Jalankan Aplikasi**: Dari direktori root proyek, jalankan server FastAPI menggunakan Uvicorn:
    ```bash
    uvicorn main:app --reload
    ```
3.  **Akses Aplikasi**: Buka peramban web Anda dan navigasi ke `http://127.0.0.1:8000`.

-----

### Identitas Proyek

  * **Nama**: ZAHRA HILYATUL JANNAH
  * **NIM**: 231524031
  * **Institusi**: POLBAN
  * **Dosen Praktikum**: Rizqi (MR), Trisna (TG)
