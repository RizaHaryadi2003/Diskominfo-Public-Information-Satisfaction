"""
test_api.py
Script pengujian standalone — memverifikasi seluruh pipeline tanpa menjalankan HTTP server.
Jalankan dari direktori Project: python test_api.py
"""

import sys
import json

def separator(title=""):
    print("\n" + "=" * 60)
    if title:
        print(f"  {title}")
        print("=" * 60)

separator("MEMULAI PENGUJIAN PIPELINE K-MEANS + SHAP API")

# ── 1. Import ────────────────────────────────────────────────
print("\n[1/5] Import modules...")
try:
    from api.model import (
        train_pipeline,
        predict_cluster,
        get_global_shap,
        get_cluster_shap,
        get_all_clusters_profile,
        get_model_info,
    )
    print("      OK - semua modul berhasil diimport")
except Exception as e:
    print(f"      GAGAL: {e}")
    sys.exit(1)

# ── 2. Training ──────────────────────────────────────────────
separator("2/5 — Training Pipeline")
try:
    train_pipeline()
    info = get_model_info()
    print(f"  Status         : {info['status']}")
    print(f"  Jumlah data    : {info['n_training_samples']} responden")
    print(f"  Silhouette     : {info['silhouette_score']}")
    print(f"  Distribusi     : {info['cluster_distribution']}")
    print("  OK")
except Exception as e:
    print(f"  GAGAL: {e}")
    sys.exit(1)

# ── 3. Prediksi satu responden ───────────────────────────────
separator("3/5 — POST /predict (simulasi)")

test_cases = [
    {
        "label": "Responden skor rendah (harapan: Kurang Puas)",
        "scores": [60, 50, 60, 40, 50, 60, 50, 40, 50, 40, 60, 50, 40, 50, 40],
    },
    {
        "label": "Responden skor tinggi merata (harapan: Puas Merata)",
        "scores": [90, 88, 85, 87, 90, 92, 88, 85, 90, 92, 88, 86, 85, 90, 88],
    },
]

for tc in test_cases:
    print(f"\n  Input  : {tc['label']}")
    try:
        result = predict_cluster(tc["scores"])
        print(f"  Cluster: {result['cluster_id']} — {result['cluster_name']}")
        print(f"  Conf   : {result['confidence']*100:.1f}%")
        print(f"  Prob   : {result['probabilities']}")
        top_shap = sorted(result['shap_scores'].items(), key=lambda x: x[1], reverse=True)[:3]
        print(f"  Top 3 SHAP fitur: {top_shap}")
    except Exception as e:
        print(f"  GAGAL: {e}")

# ── 4. Global SHAP ──────────────────────────────────────────
separator("4/5 — GET /shap/global")
try:
    global_result = get_global_shap()
    importance = global_result["feature_importance"]
    print("\n  Top 5 Faktor Dominan (Global SHAP):")
    for i, (feat, val) in enumerate(list(importance.items())[:5], 1):
        label = global_result["feature_labels"][feat]
        print(f"  {i}. {feat} ({label.strip()}) = {val:.6f}")
    print("\n  JSON output simulasi:")
    print("  " + json.dumps({"feature_importance": dict(list(importance.items())[:3])}, indent=4).replace('\n', '\n  '))
except Exception as e:
    print(f"  GAGAL: {e}")

# ── 5. SHAP per cluster ──────────────────────────────────────
separator("5/5 — GET /shap/cluster/{id}")
for cid in [0, 1, 2]:
    try:
        cr = get_cluster_shap(cid)
        top3 = list(cr["feature_importance"].items())[:3]
        print(f"\n  Cluster {cid} — {cr['cluster_name']}")
        for feat, val in top3:
            print(f"    {feat}: {val:.6f}")
    except Exception as e:
        print(f"  Cluster {cid} GAGAL: {e}")

separator("SEMUA TEST SELESAI")
print()
print("  Untuk menjalankan server HTTP:")
print("  $ python -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload")
print("  Lalu buka: http://127.0.0.1:8000/docs")
print()
