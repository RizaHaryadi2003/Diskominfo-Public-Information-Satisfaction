"""
main.py
FastAPI application — REST API untuk K-Means Clustering + SHAP Explainability
Diskominfo Provinsi Kalimantan Barat

Endpoints:
  GET  /              → health check
  GET  /model/info    → status & metrik model
  POST /predict       → prediksi cluster + SHAP per instance
  GET  /shap/global   → global SHAP feature importance
  GET  /shap/cluster/{cluster_id} → SHAP per cluster
  GET  /clusters      → profil semua cluster
"""

from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, HTTPException, Path
from fastapi.middleware.cors import CORSMiddleware

from api.model import (
    train_pipeline,
    predict_cluster,
    get_global_shap,
    get_cluster_shap,
    get_all_clusters_profile,
    get_model_info,
    CLUSTER_NAMES,
    Q_LABELS,
    CSV_PATH,
)
from api.schemas import (
    PredictRequest,
    PredictResponse,
    FeatureImportanceResponse,
    ClusterSHAPResponse,
    AllClustersResponse,
    ClusterProfileItem,
    ModelInfoResponse,
)


# ─────────────────────────────────────────
#  LIFESPAN (training saat startup)
# ─────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Jalankan training pipeline sekali saat server pertama kali start."""
    print("=" * 55)
    print("  Diskominfo SKM — Clustering & SHAP REST API")
    print("  [INFO] Memulai training pipeline...")
    print("=" * 55)
    train_pipeline(csv_path=CSV_PATH)
    print("=" * 55)
    print("[API] Training pipeline selesai. API siap menerima request.")
    print("[API] Dokumentasi: http://localhost:8000/docs")
    print("=" * 55)
    yield
    print("\n[API] Shutdown.")


# ─────────────────────────────────────────
#  APLIKASI
# ─────────────────────────────────────────

app = FastAPI(
    title="Diskominfo SKM — K-Means & SHAP API",
    description=(
        "REST API untuk analisis segmentasi kepuasan masyarakat "
        "terhadap pengelolaan informasi publik Diskominfo Provinsi Kalimantan Barat. "
        "Menggunakan K-Means clustering (K=3) dan SHAP explainability "
        "berbasis Random Forest surrogate."
    ),
    version="1.0.0",
    contact={
        "name" : "Riza Haryadi",
        "email": "riza@example.com",
    },
    lifespan=lifespan,
)

# CORS — izinkan semua origin (sesuaikan jika deploy ke production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────
#  ENDPOINT: HEALTH CHECK
# ─────────────────────────────────────────

@app.get(
    "/",
    summary="Health Check",
    tags=["General"],
    response_model=dict,
)
async def root():
    """
    Endpoint dasar untuk mengecek apakah API berjalan.
    Mengembalikan status, versi, dan daftar endpoint.
    """
    info = get_model_info()
    return {
        "service" : "Diskominfo SKM — K-Means & SHAP API",
        "version" : "1.0.0",
        "status"  : info.get("status", "unknown"),
        "docs_url": "/docs",
        "endpoints": {
            "POST /predict"               : "Prediksi cluster untuk satu responden baru",
            "GET  /shap/global"           : "Global SHAP feature importance",
            "GET  /shap/cluster/{id}"     : "SHAP feature importance per cluster (0/1/2)",
            "GET  /clusters"              : "Profil semua cluster",
            "GET  /model/info"            : "Status & metrik model",
        },
    }


# ─────────────────────────────────────────
#  ENDPOINT: MODEL INFO
# ─────────────────────────────────────────

@app.get(
    "/model/info",
    summary="Informasi Status Model",
    tags=["Model"],
    response_model=ModelInfoResponse,
)
async def model_info():
    """
    Kembalikan informasi status model yang sedang aktif:
    - Jumlah data training
    - Silhouette Score hasil clustering
    - Distribusi responden per cluster
    - Parameter utama KMeans & Random Forest
    """
    info = get_model_info()
    if info.get("status") == "not_trained":
        raise HTTPException(status_code=503, detail="Model belum dilatih.")
    return info


# ─────────────────────────────────────────
#  ENDPOINT: PREDICT
# ─────────────────────────────────────────

@app.post(
    "/predict",
    summary="Prediksi Cluster Responden Baru",
    tags=["Prediction"],
    response_model=PredictResponse,
    responses={
        200: {
            "description": "Prediksi berhasil",
            "content": {
                "application/json": {
                    "example": {
                        "cluster_id"  : 2,
                        "cluster_name": "Kurang Puas",
                        "confidence"  : 0.87,
                        "probabilities": {"0": 0.06, "1": 0.07, "2": 0.87},
                        "shap_scores" : {
                            "Q1": 0.032, "Q2": 0.018, "Q3": 0.015,
                            "Q4": 0.012, "Q5": 0.041, "Q6": 0.053,
                            "Q7": 0.021, "Q8": 0.019, "Q9": 0.024,
                            "Q10": 0.075, "Q11": 0.020, "Q12": 0.048,
                            "Q13": 0.016, "Q14": 0.022, "Q15": 0.013,
                        },
                    }
                }
            },
        },
        422: {"description": "Input tidak valid (bukan 15 nilai / nilai di luar 0–100)"},
    },
)
async def predict(request: PredictRequest):
    """
    **Prediksi segmen kepuasan** untuk satu responden baru.

    ### Input
    - `scores`: list berisi **15 nilai float** (Q1–Q15), rentang 0–100.

    ### Output
    - `cluster_id`: ID cluster (0, 1, atau 2)
    - `cluster_name`: nama deskriptif cluster
    - `confidence`: keyakinan model (probabilitas kelas prediksi)
    - `probabilities`: probabilitas untuk setiap cluster
    - `shap_scores`: kontribusi SHAP per fitur untuk instance ini

    ### Mapping Cluster
    | ID | Nama |
    |----|------|
    | 0  | Puas Merata |
    | 1  | Puas Tidak Merata |
    | 2  | Kurang Puas |
    """
    try:
        result = predict_cluster(request.scores)
        return result
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Kesalahan internal: {str(e)}")


# ─────────────────────────────────────────
#  ENDPOINT: SHAP GLOBAL
# ─────────────────────────────────────────

@app.get(
    "/shap/global",
    summary="Global SHAP Feature Importance",
    tags=["SHAP Explainability"],
    response_model=FeatureImportanceResponse,
)
async def shap_global():
    """
    Kembalikan **global SHAP feature importance** — rata-rata |SHAP value|
    untuk seluruh responden dan seluruh cluster.

    Nilai lebih tinggi = fitur tersebut lebih berpengaruh dalam membentuk segmentasi.

    Hasil diurutkan dari fitur paling dominan (tertinggi) ke terendah.
    """
    try:
        return get_global_shap()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))


# ─────────────────────────────────────────
#  ENDPOINT: SHAP PER CLUSTER
# ─────────────────────────────────────────

@app.get(
    "/shap/cluster/{cluster_id}",
    summary="SHAP Feature Importance per Cluster",
    tags=["SHAP Explainability"],
    response_model=ClusterSHAPResponse,
)
async def shap_per_cluster(
    cluster_id: int = Path(
        ...,
        ge=0,
        le=2,
        description="ID cluster: 0 (Puas Merata), 1 (Puas Tidak Merata), 2 (Kurang Puas)",
        example=0,
    )
):
    """
    Kembalikan **SHAP feature importance untuk satu cluster** tertentu.

    Berguna untuk memahami faktor-faktor dominan yang membentuk karakteristik
    masing-masing segmen masyarakat secara spesifik.

    ### Mapping Cluster
    | cluster_id | Nama |
    |------------|------|
    | 0          | Puas Merata |
    | 1          | Puas Tidak Merata |
    | 2          | Kurang Puas |
    """
    try:
        return get_cluster_shap(cluster_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))


# ─────────────────────────────────────────
#  ENDPOINT: PROFIL SEMUA CLUSTER
# ─────────────────────────────────────────

@app.get(
    "/clusters",
    summary="Profil Semua Cluster",
    tags=["Clustering"],
    response_model=AllClustersResponse,
)
async def clusters():
    """
    Kembalikan **profil ringkasan semua cluster**, meliputi:
    - Jumlah responden per cluster
    - Rata-rata Index kepuasan
    - Rata-rata skor Q1–Q15 per cluster

    Berguna sebagai referensi interpretasi hasil clustering.
    """
    try:
        profiles = get_all_clusters_profile()
        return {"clusters": profiles}
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
