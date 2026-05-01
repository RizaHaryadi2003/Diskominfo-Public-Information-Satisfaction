"""
model.py
Core ML pipeline: K-Means clustering + Random Forest surrogate + SHAP explainability.
Logika diekstrak dari diskominfo_public_information_satisfaction.py
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.ensemble import RandomForestClassifier
import shap
from typing import Dict, List, Tuple


# ─────────────────────────────────────────
#  KONSTANTA
# ─────────────────────────────────────────

Q_COLS = [
    "Q1", "Q2", "Q3", "Q4", "Q5",
    "Q6", "Q7", "Q8", "Q9", "Q10",
    "Q11", "Q12", "Q13", "Q14", "Q15",
]

CLUSTER_NAMES: Dict[int, str] = {
    0: "Puas Merata",
    1: "Puas Tidak Merata",
    2: "Kurang Puas",
}

Q_LABELS: Dict[str, str] = {
    "Q1" : "Q1  — Akses via Media Massa",
    "Q2" : "Q2  — Ingat Info 3 Bulan",
    "Q3" : "Q3  — Bisa Sebutkan Info",
    "Q4" : "Q4  — Bisa Jelaskan Info",
    "Q5" : "Q5  — Info Akurat & Bebas Kesalahan",
    "Q6" : "Q6  — Info Mudah Dipahami",
    "Q7" : "Q7  — Info Mudah Diperoleh",
    "Q8" : "Q8  — Info Tersedia saat Dibutuhkan",
    "Q9" : "Q9  — Info Sesuai Kebutuhan",
    "Q10": "Q10 — Akses Info Mudah Digunakan",
    "Q11": "Q11 — Info Sesuai Kebutuhan Saya",
    "Q12": "Q12 — Bisa Bedakan Info Resmi",
    "Q13": "Q13 — Info Ada di Media Pemprov",
    "Q14": "Q14 — Info Bermanfaat Selesaikan Masalah",
    "Q15": "Q15 — Pemprov Libatkan Masyarakat",
}

# Path CSV relatif terhadap direktori Project
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(_BASE_DIR, "Data-Responden (1).csv")

# Jumlah cluster final (ditentukan dari analisis Elbow + Silhouette)
N_CLUSTERS = 3


# ─────────────────────────────────────────
#  CONTAINER GLOBAL (in-memory model store)
# ─────────────────────────────────────────

class ModelStore:
    """
    Menyimpan semua artefak model setelah training.
    Diinstansiasi sekali saat startup aplikasi.
    """

    def __init__(self) -> None:
        self.is_trained: bool = False

        # Data
        self.df: pd.DataFrame | None = None
        self.X: pd.DataFrame | None = None          # fitur asli (Q1–Q15)
        self.X_scaled: np.ndarray | None = None     # setelah StandardScaler

        # Artefak model
        self.scaler: StandardScaler | None = None
        self.kmeans: KMeans | None = None
        self.rf: RandomForestClassifier | None = None
        self.explainer: shap.TreeExplainer | None = None

        # Pre-computed SHAP
        self.shap_values: np.ndarray | None = None   # shape: (n_classes, n_samples, n_features)
        self.global_shap: np.ndarray | None = None   # shape: (n_features,)
        self.cluster_shap: np.ndarray | None = None  # shape: (n_clusters, n_features)

        # Metrik
        self.silhouette: float = 0.0
        self.cluster_counts: Dict[int, int] = {}


# Singleton instance
_store = ModelStore()


def get_store() -> ModelStore:
    """Dependency injection helper untuk FastAPI."""
    return _store


# ─────────────────────────────────────────
#  PIPELINE TRAINING
# ─────────────────────────────────────────

def _load_data(csv_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Muat CSV, rename kolom, pisahkan fitur Q1–Q15 dari demografi.
    Menggunakan separator ';' sesuai script asli.
    """
    df = pd.read_csv(csv_path, sep=";", on_bad_lines="skip")

    df.columns = [
        "Nama", "Pendidikan", "Pekerjaan", "Kontak", "Waktu",
        "Index", "Usia", "Jenis_Kelamin",
        "Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7",
        "Q8", "Q9", "Q10", "Q11", "Q12", "Q13", "Q14", "Q15",
    ]

    # Drop kolom tidak relevan
    df = df.drop(columns=["Nama", "Kontak", "Waktu"])

    # Pastikan kolom Q numerik
    for col in Q_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=Q_COLS)

    X = df[Q_COLS].copy()
    return df, X


def _fit_kmeans(X_scaled: np.ndarray) -> Tuple[KMeans, float]:
    """
    Fit KMeans dengan K=3 (parameter optimal dari analisis Elbow+Silhouette).
    Mengembalikan model dan silhouette score.
    """
    km = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
    km.fit(X_scaled)
    sil = silhouette_score(X_scaled, km.labels_)
    return km, sil


def _fit_random_forest(X: pd.DataFrame, y: pd.Series) -> RandomForestClassifier:
    """
    Fit Random Forest surrogate classifier di atas label cluster.
    Parameter sesuai script asli (address class imbalance & overfitting).
    """
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
    )
    rf.fit(X, y)
    return rf


def _compute_shap(
    rf: RandomForestClassifier,
    X: pd.DataFrame,
) -> Tuple[shap.TreeExplainer, np.ndarray, np.ndarray, np.ndarray]:
    """
    Hitung SHAP values menggunakan TreeExplainer.

    Returns:
        explainer    : shap.TreeExplainer
        shap_array   : shape (n_classes, n_samples, n_features)
        global_shap  : mean |SHAP| semua kelas & sampel, shape (n_features,)
        cluster_shap : mean |SHAP| per kelas, shape (n_clusters, n_features)
    """
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X)           # list[ndarray] panjang n_classes

    shap_array = np.array(shap_values)               # (n_classes, n_samples, n_features)

    # shap_array shape = (n_classes, n_samples, n_features)
    # mean axis=(0,1) => rata-rata global per fitur
    global_shap = np.abs(shap_array).mean(axis=(0, 1))  # (n_features,)

    # Per cluster: mean |SHAP| per kelas (axis 1 = sampel)
    cluster_shap = np.abs(shap_array).mean(axis=1)  # (n_clusters, n_features)

    return explainer, shap_array, global_shap, cluster_shap


def train_pipeline(csv_path: str = CSV_PATH) -> None:
    """
    Jalankan seluruh pipeline training dan simpan hasilnya ke _store.
    Dipanggil sekali saat startup FastAPI (lifespan event).
    """
    global _store

    print(f"[ModelPipeline] Memuat data dari: {csv_path}")
    df, X = _load_data(csv_path)

    # 1. Normalisasi
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 2. K-Means clustering
    print("[ModelPipeline] Melatih KMeans (K=3)...")
    km, sil = _fit_kmeans(X_scaled)
    df["Cluster"] = km.labels_

    # 3. Random Forest surrogate
    print("[ModelPipeline] Melatih Random Forest surrogate...")
    y = df["Cluster"]
    rf = _fit_random_forest(X, y)

    # 4. SHAP
    print("[ModelPipeline] Menghitung SHAP values...")
    explainer, shap_array, global_shap, cluster_shap = _compute_shap(rf, X)

    # 5. Simpan ke store
    _store.is_trained    = True
    _store.df            = df
    _store.X             = X
    _store.X_scaled      = X_scaled
    _store.scaler        = scaler
    _store.kmeans        = km
    _store.rf            = rf
    _store.explainer     = explainer
    _store.shap_values   = shap_array
    _store.global_shap   = global_shap
    _store.cluster_shap  = cluster_shap
    _store.silhouette    = round(float(sil), 4)
    _store.cluster_counts = {
        str(k): int(v) for k, v in df["Cluster"].value_counts().sort_index().items()
    }

    print(
        f"[ModelPipeline] DONE. "
        f"n={len(df)}, silhouette={sil:.4f}, "
        f"distribusi={_store.cluster_counts}"
    )


# ─────────────────────────────────────────
#  INFERENSI
# ─────────────────────────────────────────

def predict_cluster(scores: List[float]) -> Dict:
    """
    Prediksi cluster untuk satu responden baru.

    Args:
        scores: List 15 float (Q1–Q15)

    Returns:
        dict berisi cluster_id, cluster_name, confidence,
        probabilities, dan shap_scores per fitur.
    """
    if not _store.is_trained:
        raise RuntimeError("Model belum dilatih. Panggil train_pipeline() terlebih dahulu.")

    new_data = pd.DataFrame([scores], columns=Q_COLS)

    # Prediksi label & probabilitas
    label_idx: int = int(_store.rf.predict(new_data)[0])
    probs_arr: np.ndarray = _store.rf.predict_proba(new_data)[0]
    probs_dict = {str(i): round(float(p), 4) for i, p in enumerate(probs_arr)}

    # SHAP untuk instance ini
    shap_vals = _store.explainer.shap_values(new_data)  # list of arrays
    shap_arr = np.array(shap_vals)  # (n_classes, 1, n_features)
    # Rata-rata |SHAP| di semua kelas untuk instance tunggal ini
    instance_shap = np.abs(shap_arr[:, 0, :]).mean(axis=0)  # (n_features,)

    shap_scores = {
        q: round(float(v), 6)
        for q, v in zip(Q_COLS, instance_shap)
    }

    return {
        "cluster_id"  : label_idx,
        "cluster_name": CLUSTER_NAMES[label_idx],
        "confidence"  : round(float(probs_arr[label_idx]), 4),
        "probabilities": probs_dict,
        "shap_scores" : shap_scores,
    }


def get_global_shap() -> Dict:
    """
    Kembalikan global SHAP feature importance (mean |SHAP| semua cluster).
    Sudah diurutkan dari tertinggi ke terendah.
    """
    if not _store.is_trained:
        raise RuntimeError("Model belum dilatih.")

    importance = {
        q: round(float(v), 6)
        for q, v in zip(Q_COLS, _store.global_shap)
    }
    # Urutkan dari tertinggi
    importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
    return {
        "feature_importance": importance,
        "feature_labels"    : Q_LABELS,
    }


def get_cluster_shap(cluster_id: int) -> Dict:
    """
    Kembalikan SHAP feature importance untuk cluster tertentu.

    Args:
        cluster_id: 0, 1, atau 2
    """
    if not _store.is_trained:
        raise RuntimeError("Model belum dilatih.")
    if cluster_id not in CLUSTER_NAMES:
        raise ValueError(f"cluster_id harus 0, 1, atau 2. Diterima: {cluster_id}")

    importance = {
        q: round(float(v), 6)
        for q, v in zip(Q_COLS, _store.cluster_shap[cluster_id])
    }
    importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
    return {
        "cluster_id"        : cluster_id,
        "cluster_name"      : CLUSTER_NAMES[cluster_id],
        "feature_importance": importance,
        "feature_labels"    : Q_LABELS,
    }


def get_all_clusters_profile() -> List[Dict]:
    """
    Kembalikan profil rata-rata (mean Q1–Q15 dan mean Index) setiap cluster.
    """
    if not _store.is_trained:
        raise RuntimeError("Model belum dilatih.")

    profiles = []
    df = _store.df

    for cid, cname in CLUSTER_NAMES.items():
        sub = df[df["Cluster"] == cid]
        mean_scores = {
            q: round(float(sub[q].mean()), 2) for q in Q_COLS
        }
        profiles.append({
            "cluster_id"    : cid,
            "cluster_name"  : cname,
            "n_respondents" : int(len(sub)),
            "mean_index"    : round(float(sub["Index"].mean()), 2),
            "mean_scores"   : mean_scores,
        })

    return profiles


def get_model_info() -> Dict:
    """Kembalikan informasi status dan parameter model."""
    if not _store.is_trained:
        return {"status": "not_trained"}

    return {
        "status"              : "ready",
        "n_training_samples"  : int(len(_store.df)),
        "n_features"          : len(Q_COLS),
        "n_clusters"          : N_CLUSTERS,
        "silhouette_score"    : _store.silhouette,
        "cluster_distribution": _store.cluster_counts,
        "model_params"        : {
            "kmeans_n_init"        : 10,
            "kmeans_random_state"  : 42,
            "rf_n_estimators"      : 200,
            "rf_max_depth"         : 6,
            "rf_min_samples_leaf"  : 5,
            "rf_class_weight"      : "balanced",
        },
    }
