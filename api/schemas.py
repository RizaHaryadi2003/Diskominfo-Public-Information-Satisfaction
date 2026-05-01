"""
schemas.py
Pydantic models untuk request dan response FastAPI.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Dict, List, Optional


# ─────────────────────────────────────────
#  REQUEST
# ─────────────────────────────────────────

class PredictRequest(BaseModel):
    """
    Input berupa 15 skor jawaban survei (Q1 – Q15).
    Setiap nilai idealnya berada di rentang 0 – 100.
    """
    scores: List[float] = Field(
        ...,
        min_length=15,
        max_length=15,
        example=[80, 75, 70, 65, 80, 85, 70, 60, 75, 85, 70, 65, 60, 75, 60],
        description="List berisi 15 nilai skor survei (Q1–Q15).",
    )

    @field_validator("scores")
    @classmethod
    def validate_scores(cls, v: List[float]) -> List[float]:
        if len(v) != 15:
            raise ValueError("Harus ada tepat 15 nilai skor (Q1–Q15).")
        for i, score in enumerate(v, start=1):
            if not (0 <= score <= 100):
                raise ValueError(
                    f"Skor Q{i} = {score} di luar rentang yang valid (0–100)."
                )
        return v


# ─────────────────────────────────────────
#  RESPONSE
# ─────────────────────────────────────────

class PredictResponse(BaseModel):
    """Hasil prediksi cluster untuk satu responden."""

    cluster_id: int = Field(..., description="ID cluster (0, 1, atau 2).")
    cluster_name: str = Field(
        ..., description="Nama deskriptif cluster hasil prediksi."
    )
    confidence: float = Field(
        ..., description="Keyakinan model terhadap prediksi (0.0 – 1.0)."
    )
    probabilities: Dict[str, float] = Field(
        ...,
        description="Probabilitas keanggotaan untuk setiap cluster.",
    )
    shap_scores: Dict[str, float] = Field(
        ...,
        description=(
            "Kontribusi SHAP per fitur untuk instance ini "
            "(rata-rata absolut dari semua cluster)."
        ),
    )


class FeatureImportanceResponse(BaseModel):
    """Global SHAP feature importance (rata-rata semua cluster)."""

    feature_importance: Dict[str, float] = Field(
        ...,
        description="Mean |SHAP value| per fitur, diurutkan dari tertinggi.",
    )
    feature_labels: Dict[str, str] = Field(
        ..., description="Label lengkap untuk setiap kode fitur (Q1–Q15)."
    )


class ClusterSHAPResponse(BaseModel):
    """SHAP feature importance untuk satu cluster tertentu."""

    cluster_id: int
    cluster_name: str
    feature_importance: Dict[str, float] = Field(
        ...,
        description="Mean |SHAP value| per fitur untuk cluster ini.",
    )
    feature_labels: Dict[str, str]


class ClusterProfileItem(BaseModel):
    """Profil rata-rata satu cluster."""

    cluster_id: int
    cluster_name: str
    n_respondents: int
    mean_index: float
    mean_scores: Dict[str, float] = Field(
        ..., description="Rata-rata skor Q1–Q15 untuk cluster ini."
    )


class AllClustersResponse(BaseModel):
    """Profil ringkasan semua cluster."""

    clusters: List[ClusterProfileItem]


class ModelInfoResponse(BaseModel):
    """Informasi status dan metrik model yang sudah dilatih."""

    status: str
    n_training_samples: int
    n_features: int
    n_clusters: int
    silhouette_score: float
    cluster_distribution: Dict[str, int] = Field(
        ..., description="Jumlah sampel per cluster."
    )
    model_params: Dict[str, object] = Field(
        ..., description="Parameter utama model KMeans dan RandomForest."
    )
