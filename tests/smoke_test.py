"""Deployment smoke test using the locked notebook demo profile."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

config = json.loads((ROOT / "models/champion_configuration.json").read_text())
profile = {
    "Age": 13,
    "BMI": 30.0,
    "GenHlth": 5,
    "HighBP": 1,
    "PhysHlth": 22.0,
    "MentHlth": 2.0,
    "HeartDiseaseorAttack": 0,
    "HighChol": 1,
    "DiffWalk": 1,
    "Education": 6.0,
    "Family_History_Diabetes": 0,
    "Income": 5.0,
    "PhysActivity": 1,
    "Stroke": 1,
    "Smoker": 0,
}
frame = pd.DataFrame([profile])[config["features"]]

model = joblib.load(ROOT / "models/stacking_ensemble.joblib")
probability = float(model.predict_proba(frame)[0, 1])
assert np.isclose(probability, 0.4433281881, atol=1e-8), probability

scaler = joblib.load(ROOT / "models/kmeans_scaler.joblib")
kmeans = joblib.load(ROOT / "models/kmeans_risk_model.joblib")
mapping = json.loads((ROOT / "models/risk_cluster_mapping.json").read_text())
cluster = int(kmeans.predict(scaler.transform(frame))[0])
assert mapping[str(cluster)] == "High Risk"

chunks = pd.read_csv(ROOT / "rag/diabetes_knowledge_chunks.csv")
embeddings = np.load(ROOT / "rag/diabetes_chunk_embeddings.npy")
assert embeddings.shape == (len(chunks), 384)

print("PASS: model, clustering and RAG artifacts are compatible.")
