# 📊 Analisis Segmentasi Kepuasan Masyarakat (Diskominfo)

*A Secure, Data-Driven Machine Learning Framework for Evaluating Community Satisfaction in West Kalimantan.*

Proyek skripsi ini menganalisis kepuasan masyarakat terhadap pengelolaan informasi publik menggunakan **K-Means Clustering** dan **SHAP Explainability**. Sistem ini juga menyediakan **REST API** interaktif untuk prediksi *real-time* dan pengelolaan data berkelanjutan (MLOps) dengan standar privasi yang ketat.

---

## 🧠 Overview & Objectives

Dalam konteks pemerintahan digital, analisis survei tradisional seringkali hanya berhenti pada statistik deskriptif. Penelitian ini melangkah lebih jauh dengan mengungkap **pola perilaku tersembunyi**, mengidentifikasi **faktor utama ketidakpuasan**, dan mengubah data survei mentah menjadi **wawasan kebijakan yang dapat ditindaklanjuti**.

**Tujuan Utama:**
- Mensegmentasi tingkat kepuasan masyarakat menggunakan algoritma *clustering* tanpa pengawasan (Unsupervised Learning).
- Mengidentifikasi faktor (pertanyaan survei) yang paling berpengaruh terhadap kepuasan.
- Memberikan *Interpretable Insights* menggunakan **Explainable AI (SHAP)** untuk transparansi keputusan AI.
- Membangun REST API yang siap diimplementasikan (*deployment-ready*) untuk Diskominfo.

---

## 🔒 Keamanan & Privasi Data (Data Anonymization)

Mengingat dataset mengandung *Personally Identifiable Information* (PII) atau identitas pribadi responden, proyek ini menerapkan standar **Data Anonymization** tingkat produksi.

- Data asli dari pemerintah (Diskominfo) **TIDAK** disertakan dalam repositori ini demi menjaga kerahasiaan.
- **Pipeline Anonimisasi:** Kami menggunakan *script* otomatis (`scripts/anonymize.py`) untuk memproses data mentah, menghapus atribut sensitif (seperti Nama dan Nomor HP), dan menghasilkan dataset yang aman (`Data-Aman.csv`).
- Model Machine Learning **hanya dilatih menggunakan data yang sudah dianonimkan**, sehingga menjamin 100% keamanan privasi responden apabila *source code* ini dikembangkan oleh *developer* selanjutnya.

---

## 📁 Struktur Proyek

```text
Project/
├── api/
│   ├── main.py           # Endpoint aplikasi FastAPI & konfigurasi server
│   ├── model.py          # Logika Machine Learning, Training, & Caching Model
│   ├── schemas.py        # Pydantic Schema (Request/Response validation)
│   └── requirements.txt  # Daftar pustaka Python yang dibutuhkan
├── data/
│   ├── raw/              # (Diabaikan Git) Tempat meletakkan CSV rahasia dari Diskominfo
│   └── processed/        # Tempat menyimpan 'Data-Aman.csv' yang sudah bersih
├── scripts/
│   └── anonymize.py      # Script otomatis untuk membuang identitas dari raw ke processed
├── diskominfo_public_information_satisfaction.py  # Script dasar (Eksperimen awal)
├── test_api.py                                    # Pengujian pipeline ML secara lokal
├── .env.example                                   # Template Environment Variables
└── README.md                                      # Dokumentasi Proyek
```

---

## 💻 Instalasi & Persiapan Lingkungan

Proyek ini menggunakan **Python 3.10+**. Sangat disarankan untuk menggunakan *Virtual Environment* (venv) agar pustaka tidak bentrok dengan proyek lain di komputer Anda.

### 1. Clone & Setup Environment
```bash
# Clone repositori (jika di-hosting)
git clone https://github.com/username/repo-anda.git
cd repo-anda

# Buat dan aktifkan Virtual Environment
python -m venv venv

# Windows:
.\venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Install semua dependensi
pip install -r api/requirements.txt
```

### 2. Konfigurasi Environment (Opsional)
Salin file template `.env` jika Anda ingin mengkustomisasi path pembacaan CSV secara manual:
```bash
cp .env.example .env
```

### 3. Persiapan Data (Bagi Pengembang Lanjutan)
Jika Anda memiliki data baru dari *web scraper* SIKEDIP:
1. Masukkan file mentah berformat `.csv` ke dalam folder `data/raw/`.
2. Jalankan script anonimisasi:
   ```bash
   python scripts/anonymize.py
   ```
3. File akan diproses menjadi `data/processed/Data-Aman.csv` dan siap digunakan oleh Model.

---

## 🚀 Cara Menjalankan REST API

Setelah instalasi selesai, jalankan server **Uvicorn** melalui terminal:

```bash
# Set encoding (Khusus pengguna Windows PowerShell)
$env:PYTHONIOENCODING='utf-8'

# Jalankan server
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```

Akses Dokumentasi API Interaktif (*Swagger UI*):
👉 **http://127.0.0.1:8000/docs**

---

## 🔑 Endpoint API & Autentikasi

Semua *endpoint* baca (GET) dan prediksi (POST `/predict`) bersifat **publik**. 
Namun, *endpoint* administratif untuk memperbarui data dan melatih ulang model membutuhkan **API Key**.

**API Key Default:** `skripsi123`  
*(Cara pakai di Swagger UI: Klik 'Try it out', lalu masukkan kata sandi di kotak `x-api-key` sebelum mengeksekusi).*

| Method | Endpoint | Deskripsi | Auth |
|--------|----------|-----------|------|
| **GET** | `/` | Pengecekan status server (*Health Check*). | ❌ |
| **GET** | `/model/info` | Melihat metrik performa model (Silhouette Score) & hyperparameter. | ❌ |
| **POST** | `/predict` | Memprediksi kelompok (*cluster*) kepuasan responden baru. | ❌ |
| **GET** | `/shap/global` | Menampilkan faktor pelayanan yang secara umum paling berpengaruh. | ❌ |
| **GET** | `/clusters` | Menampilkan ringkasan profil & rata-rata indeks tiap *cluster*. | ❌ |
| **POST** | `/data/upload_and_retrain` | Mengunggah CSV baru dan langsung melatih ulang (*Retrain*) model AI. | 🔒 **Ya** |
| **POST** | `/model/retrain` | Memaksa sistem melatih ulang model dari data lokal yang ada. | 🔒 **Ya** |

---

## 🔄 End-to-End MLOps Pipeline

Proyek ini dibangun di atas fondasi otomasi AI (*MLOps*) dengan tahapan berikut:
1. **Data Ingestion**: Menerima file CSV dari unggahan Admin atau *Local Directory*.
2. **Anonymization**: Menghilangkan *Personally Identifiable Information* (PII).
3. **Preprocessing**: Normalisasi *Z-Score* menggunakan `StandardScaler`.
4. **K-Means Clustering**: Membagi masyarakat ke dalam segmentasi yang tersembunyi.
5. **Surrogate Modeling**: Melatih Random Forest Classifier di atas label *clustering* untuk menangkap logika prediksi matematis.
6. **SHAP Explainability**: Menggunakan *TreeExplainer* untuk membedah *black-box* Random Forest agar AI dapat menjelaskan "Kenapa orang ini masuk ke cluster tertentu?".
7. **Model Caching**: Menyimpan artefak model ke bentuk `.joblib` agar API merespon dalam satuan milidetik tanpa harus melatih ulang.

---

## 📈 Key Findings (Hasil Segmentasi)

Berdasarkan dataset uji:
| ID | Cluster | Interpretasi Karakteristik |
|---|---|---|
| **0** | **Puas Merata** | Memberikan nilai tinggi dan sangat konsisten di semua instrumen survei. |
| **1** | **Puas Tidak Merata** | Secara garis besar puas, namun ada beberapa poin instrumen yang dinilai kurang maksimal. |
| **2** | **Kurang Puas** | Responden yang merasa kecewa pada sebagian besar pelayanan informasi. Harus menjadi target intervensi utama. |

**Insight Pendorong Kepuasan (SHAP):**
Faktor **Aksesibilitas Digital (Q10)** merupakan pendorong paling kritis yang menentukan apakah seseorang akan berada di *Cluster* Puas atau Tidak. Sedangkan faktor **Profesionalisme Staf (Q15)** berperan sebagai pendorong krusial kedua. Ini menegaskan bahwa sinergi *Sistem IT + Pelayanan Manusia* adalah kunci utama kepuasan masyarakat.

---

## 🎓 Kontak & Kredit
**Peneliti:** Riza Haryadi  
**Email:** rizaharyadi13@gmail.com  
Proyek ini diajukan sebagai bagian dari tugas akhir (Skripsi). Jika ada pertanyaan mengenai implementasi arsitektur atau adopsi teknologi ini untuk instansi Anda, silakan hubungi via email di atas.
