# PraTubes

Sistem Pengenalan Wajah dan Deteksi Suku/Etnis

Sistem ini adalah aplikasi berbasis Streamlit untuk pengenalan wajah, deteksi suku/etnis, dan deteksi gender menggunakan metode seperti FaceNet, ArcFace, Siamese Network, dan DeepFace. Sistem ini mendukung deteksi wajah dengan MTCNN, RetinaFace, atau Haar Cascade, serta klasifikasi suku/etnis menggunakan model ResNet50.

Fitur Utama
Tambah Subjek: Unggah atau ambil foto subjek baru dengan variasi ekspresi, sudut, dan pencahayaan.
Split Dataset: Bagi dataset menjadi data training, validasi, dan testing.
Preprocessing: Terapkan normalisasi, rotasi, flip, brightness/contrast adjustment, dan noise pada gambar.
Statistik: Tampilkan distribusi suku/etnis dalam dataset.
Face Similarity: Bandingkan kemiripan wajah menggunakan Siamese Network, FaceNet, atau ArcFace.
Deteksi Suku/Etnis: Prediksi suku/etnis dari gambar wajah menggunakan model ResNet50.
Deteksi Gender: Deteksi gender menggunakan DeepFace.
Evaluasi Sistem: Evaluasi performa model pada data validasi dan testing, termasuk Classification Report, Confusion Matrix, ROC Curve, dan perbandingan waktu komputasi.

Prasyarat
Python: Versi 3.9 atau lebih tinggi (Python 3.8 juga didukung, tetapi disarankan menggunakan 3.9+ untuk kompatibilitas yang lebih baik).
Sistem Operasi: Windows, Linux, atau macOS.
GPU (opsional): Untuk performa lebih baik, gunakan GPU dengan CUDA dan cuDNN terinstal (jika menggunakan TensorFlow atau PyTorch dengan dukungan GPU).
Kamera (opsional): Untuk fitur pengambilan foto langsung melalui aplikasi Streamlit.

Langkah-langkah Setup
1. Clone atau Unduh Repositori
Clone repositori ini atau unduh kode sumbernya ke komputer Anda:

git clone <URL_REPOSITORI_ANDA>
cd <NAMA_DIREKTORI>

2. Buat Lingkungan Virtual
Buat dan aktifkan lingkungan virtual untuk mengisolasi dependensi:

python -m venv praktikum_env_new
Windows:
praktikum_env_new\Scripts\activate

Linux/Mac:
source praktikum_env_new/bin/activate

3. Instal Dependensi
Instal semua dependensi yang diperlukan menggunakan requirements.txt:
pip install -r requirements.txt

4. Siapkan Dataset (Opsional)
Sistem ini menggunakan folder dataset_kel_9 untuk menyimpan gambar wajah. Jika Anda ingin menggunakan dataset Anda sendiri, buat folder dataset_kel_9 di direktori proyek.

Struktur folder:

dataset_kel_9/
├── NamaSubjek1/
│   ├── Suku1/
│   │   ├── img1.jpg
│   │   ├── img2.jpg
│   │   └── ...
├── NamaSubjek2/
│   ├── Suku2/
│   │   ├── img1.jpg
│   │   └── ...

Metadata gambar akan disimpan dalam dataset_metadata.csv secara otomatis saat Anda menambahkan subjek baru melalui aplikasi.
Folder tambahan seperti processed_dataset_kel_9, train_dataset_kel_9, val_dataset_kel_9, dan test_dataset_kel_9 akan dibuat saat preprocessing atau splitting dataset.
Folder consents akan dibuat untuk menyimpan formulir persetujuan dalam format .txt.

5. Jalankan Aplikasi
Setelah dependensi terinstal, jalankan aplikasi Streamlit:
streamlit run app.py

Aplikasi akan terbuka di browser default Anda (biasanya di http://localhost:8501).
Gunakan sidebar untuk navigasi ke fitur yang diinginkan (Tambah Subjek, Split Dataset, Preprocessing, dll.).

Contoh Penggunaan

Tambah Subjek Baru:
Navigasikan ke tab "Tambah Subjek".
Masukkan nama dan suku/etnis subjek.
Unggah atau ambil foto untuk setiap variasi (tersenyum, serius, dll.).
Klik "Simpan Foto" untuk menyimpan gambar ke dataset.

Split Dataset:
Navigasikan ke tab "Split Dataset".
Klik tombol "Split Dataset" untuk membagi dataset menjadi training (70%), validasi (15%), dan testing (15%).

Deteksi Suku/Etnis:
Navigasikan ke tab "Deteksi Suku/Etnis".
Latih model terlebih dahulu dengan klik "Latih Model Suku/Etnis".
Pilih gambar dari dataset, unggah gambar baru, atau gunakan kamera untuk mendeteksi suku/etnis.

Deteksi Gender:
Navigasikan ke tab "Deteksi Gender".
Pilih gambar dari dataset, unggah gambar baru, atau gunakan kamera.
Klik "Deteksi Gender" untuk melihat prediksi gender menggunakan DeepFace.

Catatan Penting
GPU Support: Jika menggunakan GPU, pastikan CUDA dan cuDNN terinstal, lalu instal versi TensorFlow dan PyTorch yang mendukung GPU:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install tensorflow-gpu==2.12.0

Kamera: Fitur kamera hanya berfungsi di browser yang mendukung akses kamera (seperti Chrome atau Firefox).

Dependensi Berat: Sistem ini menggunakan banyak pustaka berat (TensorFlow, PyTorch, OpenCV, DeepFace, dll.). Pastikan Anda memiliki cukup RAM (minimal 8GB) dan ruang disk.

Kompatibilitas: Jika mengalami konflik dependensi, periksa versi pustaka di requirements.txt dan sesuaikan dengan versi Python Anda.

Model Penyimpanan: Model yang dilatih disimpan sebagai ethnic_model.pth (untuk deteksi suku/etnis) dan siamese_model.h5 (untuk Siamese Network).

Troubleshooting
Error "cannot import name 'builder' from 'google.protobuf.internal'":

Pastikan versi protobuf sesuai (4.24.4 direkomendasikan). Coba:
pip uninstall protobuf -y
pip install protobuf==4.24.4

TensorFlow/TensorBoard Konflik:
Jika ada konflik dengan protobuf, perbarui TensorFlow:
pip install tensorflow==2.12.0

Gagal Instal Pustaka:
Pastikan Anda menggunakan Python 3.8/3.9 atau lebih tinggi. Untuk Windows, beberapa pustaka seperti retinaface atau deepface mungkin memerlukan Microsoft Visual C++ Build Tools.
