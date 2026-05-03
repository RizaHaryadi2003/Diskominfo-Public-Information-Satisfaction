# 📊 Analisis Segmentasi Kepuasan Masyarakat (Diskominfo)

*A Data-Driven Machine Learning Framework for Evaluating Community Satisfaction in West Kalimantan.*

Proyek skripsi ini menganalisis kepuasan masyarakat terhadap pengelolaan informasi publik menggunakan **K-Means Clustering** dan **SHAP Explainability**, serta menyediakan **REST API** untuk prediksi secara *real-time*.

---

## 🧠 Overview & Objectives

Dalam konteks pemerintahan digital, analisis survei tradisional seringkali hanya berhenti pada statistik deskriptif. Penelitian ini melangkah lebih jauh dengan mengungkap **pola perilaku tersembunyi**, mengidentifikasi **faktor utama ketidakpuasan**, dan mengubah data survei mentah menjadi **wawasan kebijakan yang dapat ditindaklanjuti**.

Tujuan utama penelitian ini:
- Mensegmentasi tingkat kepuasan masyarakat menggunakan *clustering* berbasis data.
- Mengidentifikasi faktor paling berpengaruh terhadap kepuasan.
- Memberikan penjelasan yang dapat diinterpretasi (*interpretable insights*) untuk perbaikan kebijakan dan sistem.

---

## 📁 Struktur Proyek

```
Project/
├── Data-Responden (1).csv                         # Dataset survei SKM SIKEDIP 2023-2026
├── diskominfo_public_information_satisfaction.py  # Script analisis lengkap (Jupyter/Python)
├── test_api.py                                    # Test pipeline ML tanpa server HTTP
├── README.md                                      # Dokumentasi ini
└── api/
    ├── __init__.py
    ├── main.py           # FastAPI application
    ├── model.py          # Pipeline ML (ModelStore singleton)
    ├── schemas.py        # Pydantic request/response schemas
    └── requirements.txt  # Dependensi Python & API
```

---

## 💻 Cara Menjalankan Aplikasi (REST API)

API dibangun menggunakan **FastAPI** yang mengekspos model K-Means + SHAP. Model dioptimalkan dengan *caching* (`.joblib`) dan mendukung *retrain* data secara dinamis.

### 1. Install Dependensi
Karena proyek ini menggunakan banyak pustaka, **sangat disarankan menggunakan Virtual Environment**.
```bash
python -m venv venv
.\venv\Scripts\activate
pip install -r api/requirements.txt
```

### 2. Jalankan Server
```powershell
$env:PYTHONIOENCODING='utf-8'
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```
Buka dokumentasi interaktif di browser: **http://127.0.0.1:8000/docs**

---

## 🔄 End-to-End Data Science Pipeline

Proyek ini dibangun di atas 10 tahapan *pipeline* untuk memastikan ketahanan dan interpretabilitas:
1. **Data Acquisition**: Mengumpulkan data (Q1-Q15) dari layanan informasi.
2. **EDA**: Memahami distribusi data dan anomali.
3. **Preprocessing**: Pembersihan data dan penanganan *missing values*.
4. **Feature Engineering**: Memilih fitur yang memengaruhi Indeks Kepuasan.
5. **Standardization**: Menggunakan `StandardScaler` (Z-score).
6. **Optimal Cluster Determination**: Menggunakan *Elbow Method* & *Silhouette Score*.
7. **Unsupervised Clustering (K-Means)**: Mensegmentasi populasi.
8. **Model Validation (Random Forest)**: Melatih model *surrogate* untuk memvalidasi konsistensi.
9. **Explainable AI (SHAP)**: Menginterpretasikan pendorong kepuasan secara global dan individual.
10. **Insight Generation**: Mengubah *output* menjadi rekomendasi strategis SIKEDIP.

---

## 📈 Segmentasi Cluster & Key Findings

| Cluster | Nama | Interpretasi |
|---------|------|--------------|
| **0** | Puas Merata | Kepuasan tinggi dan konsisten di semua dimensi. |
| **1** | Puas Tidak Merata | Kepuasan tinggi tapi tidak konsisten antar dimensi. |
| **2** | Kurang Puas | Kepuasan rendah, perlu intervensi prioritas. |

**Temuan Utama:**
- **Aksesibilitas Digital (Q10)** adalah pendorong utama (*primary driver*). Kemudahan akses sistem sangat memengaruhi kepuasan.
- **Profesionalisme Staf (Q15)** adalah pendorong sekunder.
Ini membuktikan bahwa kombinasi **kegunaan teknologi + interaksi manusia** mendefinisikan kepuasan layanan Diskominfo.

---

## ⚠️ Keterbatasan Model & Future Work

Model ini saat ini masih berskala penelitian dan belum sepenuhnya siap pakai untuk keputusan final tingkat produksi berskala besar. 
**Alasan:** Model ini dilatih menggunakan dataset yang terbatas (Survei SKM 2023-2026 sejumlah 477 responden). 

Untuk penerapan optimal di masa depan, pertimbangkan:
1. Menambahkan **Data Operasional (Transaction Log)** dari sistem SIKEDIP.
2. Melakukan riset lanjutan menggunakan metode Kualitatif.
3. Membangun *background task* (seperti Celery) untuk proses *retrain* agar API tidak memblokir saat data membesar.
