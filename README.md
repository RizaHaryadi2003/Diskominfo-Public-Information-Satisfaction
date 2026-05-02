# Analisis Segmentasi Kepuasan Masyarakat
### Pengelolaan Informasi Publik — Diskominfo Provinsi Kalimantan Barat

Proyek skripsi yang menganalisis kepuasan masyarakat terhadap pengelolaan informasi publik menggunakan **K-Means Clustering** dan **SHAP Explainability**.

---

## Struktur Proyek

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
    └── requirements.txt  # Dependensi Python
```

---

## Script Analisis (`diskominfo_public_information_satisfaction.py`)

Script penelitian lengkap yang memuat alur kerja:

| # | Seksi | Deskripsi |
|---|-------|-----------|
| 1 | Load & Preprocessing | Muat CSV, rename kolom |
| 2 | Cleaning | Drop kolom tidak relevan, pisahkan fitur |
| 3 | Normalisasi | StandardScaler (Z-score) |
| 4 | EDA | Rata-rata skor per pertanyaan |
| 5 | K-Means: Elbow + Silhouette | Penentuan K optimal |
| 6 | Final Clustering K=3 | Labeling cluster |
| 7 | Demografi per Cluster | Profil demografis tiap segmen |
| 8 | Validasi Kruskal-Wallis | Uji statistik perbedaan antar cluster |
| 9 | Random Forest Surrogate | Classifier + 5-Fold CV |
| 10 | SHAP Explainability | Kalkulasi SHAP values |
| 11 | Priority Matrix | Quadrant analysis (Impact vs Performance) |
| 12 | Inferensi Real-Time | Fungsi prediksi responden baru |
| 13 | Heatmap Korelasi | Korelasi antar Q1–Q15 |
| 14 | Distribusi Index | Histogram kepuasan per cluster |
| 15 | Radar Chart | Profil cluster multidimensi |
| 16 | Analisis per Dimensi | Grouped bar chart dimensi informasi |
| 17 | Gap Analysis | Performa vs target ideal (85) |

### Cara Menjalankan

```bash
# Pastikan dependensi terinstall
pip install -r api/requirements.txt

# Jalankan script analisis
python diskominfo_public_information_satisfaction.py
```

---

## REST API (`api/`)

FastAPI yang mengekspos model K-Means + SHAP untuk prediksi real-time.

### Cara Menjalankan API

```powershell
# Set encoding (Windows)
$env:PYTHONIOENCODING='utf-8'

# Jalankan server
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```

Server berjalan di: **http://127.0.0.1:8000**

### Endpoints

| Method | Endpoint | Deskripsi |
|--------|----------|-----------|
| `GET` | `/` | Health check & status API |
| `GET` | `/model/info` | Status model, metrik silhouette, distribusi cluster |
| `POST` | `/predict` | Prediksi cluster + SHAP untuk responden baru |
| `GET` | `/shap/global` | Global SHAP feature importance |
| `GET` | `/shap/cluster/{0\|1\|2}` | SHAP feature importance per cluster |
| `GET` | `/clusters` | Profil rata-rata semua cluster |

### Contoh Request `POST /predict`

```json
{
  "Q1": 75, "Q2": 70, "Q3": 80, "Q4": 65, "Q5": 85,
  "Q6": 90, "Q7": 75, "Q8": 70, "Q9": 80, "Q10": 60,
  "Q11": 75, "Q12": 85, "Q13": 70, "Q14": 80, "Q15": 65
}
```

### Contoh Response

```json
{
  "cluster_id": 0,
  "cluster_name": "Puas Merata",
  "confidence": 0.87,
  "shap_values": { "Q1": 0.045, "Q2": -0.012, ... }
}
```

### Dokumentasi Interaktif

Buka browser di: **http://127.0.0.1:8000/docs** (Swagger UI)

---

## Test Pipeline (tanpa server)

```bash
python test_api.py
```

---

## Segmentasi Cluster

| Cluster | Nama | Interpretasi |
|---------|------|--------------|
| 0 | Puas Merata | Kepuasan tinggi dan konsisten di semua dimensi |
| 1 | Puas Tidak Merata | Kepuasan tinggi tapi tidak konsisten antar dimensi |
| 2 | Kurang Puas | Kepuasan rendah, perlu intervensi prioritas |

---

## Dependensi Utama

```
fastapi, uvicorn, pandas, scikit-learn, shap, numpy, pydantic, seaborn, matplotlib
```

Install semua:
```bash
pip install -r api/requirements.txt
```

---

## Dataset

- **Sumber**: Survei Kepuasan Masyarakat (SKM) SIKEDIP 2023–2026
- **Jumlah**: 477 responden
- **Fitur**: Q1–Q15 (skor kepuasan 0–100 per dimensi layanan informasi)

## How to Run

-- $env:PYTHONIOENCODING='utf-8'; python test_api.py

--$env:PYTHONIOENCODING='utf-8'; python -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload

## Model ini belum siap pakai karena beberapa alasan sistem dan data

## Alasan: Model ini dibuat bedasarkan data yang terbatas dimana saya hanya menggunakan data SKM terhadap Informasi Publik tahun 2023-2026 yang hanya berjumlah 477 responden. dalam penerapan model ML yang optimal untuk memberikan keputusan di haruskan mendapatkan data yang sangat banyak beberapa opsi yang perlu di pertimbangkan 
1. Data Oprasional (Transaction Log ) 
2. Lanjut Melakukan Riset Data Menggunakan Metode Kualitatif 
3. 