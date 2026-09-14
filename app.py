from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from google import genai
from google.genai import types
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
RAG_DIR = BASE_DIR / "rag"
TABLE_DIR = BASE_DIR / "tables"
FIGURE_DIR = BASE_DIR / "figures"

st.set_page_config(
    page_title="Diabetes Early-Risk AI",
    page_icon="🩺",
    layout="wide",
)


@st.cache_resource
def load_prediction_assets():
    model = joblib.load(MODEL_DIR / "stacking_ensemble.joblib")
    scaler = joblib.load(MODEL_DIR / "kmeans_scaler.joblib")
    kmeans = joblib.load(MODEL_DIR / "kmeans_risk_model.joblib")
    config = json.loads(
        (MODEL_DIR / "champion_configuration.json").read_text()
    )
    cluster_mapping = json.loads(
        (MODEL_DIR / "risk_cluster_mapping.json").read_text()
    )
    cri_config = json.loads(
        (MODEL_DIR / "risk_index_configuration.json").read_text()
    )
    return model, scaler, kmeans, config, cluster_mapping, cri_config


@st.cache_resource
def load_rag_assets():
    chunks = pd.read_csv(RAG_DIR / "diabetes_knowledge_chunks.csv")
    embeddings = np.load(RAG_DIR / "diabetes_chunk_embeddings.npy")
    embedding_model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )
    return chunks, embeddings, embedding_model


def yes_no(label: str, help_text: str | None = None, key: str | None = None) -> int:
    value = st.selectbox(label, ["No", "Yes"], help=help_text, key=key)
    return 1 if value == "Yes" else 0


def calculate_cri(profile: pd.DataFrame, config: dict):
    contributions = []
    total = 0.0

    for feature in config["features"]:
        value = float(profile.iloc[0][feature])
        minimum = float(config["minimums"][feature])
        maximum = float(config["maximums"][feature])
        direction = int(config["directions"][feature])
        weight = float(config["weights"][feature])

        normalized = 0.0 if maximum == minimum else (
            np.clip(value, minimum, maximum) - minimum
        ) / (maximum - minimum)

        if direction == -1:
            normalized = 1.0 - normalized

        contribution = 100.0 * normalized * weight
        total += contribution
        contributions.append(
            {
                "Feature": feature,
                "Risk contribution": contribution,
            }
        )

    contribution_table = pd.DataFrame(contributions).sort_values(
        "Risk contribution", ascending=False
    )
    return float(total), contribution_table


def retrieve_evidence(query: str, chunks: pd.DataFrame, embeddings: np.ndarray,
                      embedding_model: SentenceTransformer, top_k: int = 5):
    query_embedding = embedding_model.encode(
        [query], normalize_embeddings=True
    )[0]
    scores = embeddings @ query_embedding
    indices = np.argsort(scores)[::-1][:top_k]
    results = chunks.iloc[indices].copy().reset_index(drop=True)
    results.insert(1, "Similarity_Score", scores[indices])
    return results


def build_retrieval_query(profile: dict) -> str:
    descriptions = {
        "HighBP": "high blood pressure",
        "HighChol": "high cholesterol",
        "Smoker": "smoking history",
        "Stroke": "history of stroke",
        "HeartDiseaseorAttack": "heart disease or heart attack history",
        "Family_History_Diabetes": "family history of diabetes",
        "DiffWalk": "difficulty walking",
    }
    factors = [text for feature, text in descriptions.items() if profile[feature] == 1]
    if profile["BMI"] >= 25:
        factors.append("elevated BMI")
    if profile["GenHlth"] >= 4:
        factors.append("self-reported poor general health")
    if profile["PhysActivity"] == 0:
        factors.append("limited physical activity")
    summary = ", ".join(factors) if factors else "general type 2 diabetes risk factors"
    return f"Evidence-based type 2 diabetes prevention guidance for an adult with {summary}."


def generate_recommendation(api_key: str, profile: dict, probability: float,
                            risk_level: str, cri: float, evidence: pd.DataFrame) -> str:
    blocks = []
    for index, row in evidence.iterrows():
        blocks.append(
            f"[Source {index + 1}]\nOrganization: {row['Organization']}\n"
            f"Title: {row['Title']}\nURL: {row['URL']}\nEvidence: {row['Text']}"
        )

    prompt = f"""
Generate concise educational guidance for a diabetes early-risk research prototype.

Estimated questionnaire-based early-risk score: {probability:.3f}
K-Means descriptive risk segment: {risk_level}
Composite Risk Index: {cri:.1f}/100
Feature profile: {json.dumps(profile, indent=2)}

Age is an encoded survey category from 1 to 13, not age in years.
GenHlth ranges from 1 (excellent) to 5 (poor). The model does not use diagnostic
laboratory tests.

Retrieved evidence:
{chr(10).join(blocks)}

Explain the score calmly as an early-risk estimate, identify relevant modifiable
factors, and give 4-6 practical suggestions supported only by the retrieved evidence.
Cite claims with [Source 1], [Source 2], etc., and include a Sources section with URLs.
Do not diagnose, prescribe treatment, or recommend medication changes. Recommend
professional assessment/testing where appropriate. End exactly with:
"This is a research prototype and not a clinical diagnostic tool."
"""

    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=90000),
    )
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.2,
            max_output_tokens=2000,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )
    text = response.text or ""
    if len(text.split()) < 80:
        raise RuntimeError("Gemini returned an incomplete response. Please try again.")
    disclaimer = "This is a research prototype and not a clinical diagnostic tool."
    if disclaimer not in text:
        text = text.rstrip() + "\n\n" + disclaimer
    return text


model, cluster_scaler, kmeans_model, model_config, cluster_mapping, cri_config = (
    load_prediction_assets()
)

st.title("Diabetes Early-Risk AI Framework")
st.caption(
    "Ensemble prediction · Composite Risk Index · K-Means stratification · "
    "explainability · evidence-grounded recommendations"
)
st.warning(
    "Research prototype only. This application does not diagnose diabetes or "
    "replace professional medical assessment."
)

with st.sidebar:
    st.header("Gemini configuration")
    try:
        deployed_key = st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        deployed_key = ""
    api_key = st.text_input(
        "Gemini API key",
        value=deployed_key,
        type="password",
        help="Optional until an evidence-based recommendation is requested.",
    )
    st.caption("The key is masked and is not written to a project file.")
    st.divider()
    st.metric("Champion model", "Stacking Ensemble")
    st.metric("Held-out ROC-AUC", "0.841")
    st.metric("Held-out sensitivity", "82.3%")
    st.metric("CDC weighted ROC-AUC", "0.783")

with st.form("risk_profile_form"):
    st.subheader("Enter questionnaire-based risk factors")
    col1, col2, col3 = st.columns(3)

    age_options = {
        "18–24": 1, "25–29": 2, "30–34": 3, "35–39": 4,
        "40–44": 5, "45–49": 6, "50–54": 7, "55–59": 8,
        "60–64": 9, "65–69": 10, "70–74": 11, "75–79": 12,
        "80+": 13,
    }
    education_options = {
        "Never attended / kindergarten": 1,
        "Grades 1–8": 2,
        "Grades 9–11": 3,
        "Grade 12 or GED": 4,
        "College 1–3 years": 5,
        "College 4+ years": 6,
    }
    income_options = {
        "Less than $10,000": 1, "$10,000–$14,999": 2,
        "$15,000–$19,999": 3, "$20,000–$24,999": 4,
        "$25,000–$34,999": 5, "$35,000–$49,999": 6,
        "$50,000–$74,999": 7, "$75,000 or more": 8,
    }

    with col1:
        age_label = st.selectbox("Age group", list(age_options))
        bmi = st.number_input("BMI (kg/m²)", 14.0, 48.0, 27.0, 0.1)
        general_health = st.select_slider(
            "General health",
            options=[1, 2, 3, 4, 5],
            value=3,
            format_func=lambda x: {
                1: "Excellent", 2: "Very good", 3: "Good",
                4: "Fair", 5: "Poor",
            }[x],
        )
        physical_health = st.slider("Poor physical-health days (past 30)", 0, 30, 3)
        mental_health = st.slider("Poor mental-health days (past 30)", 0, 30, 1)

    with col2:
        high_bp = yes_no("Diagnosed high blood pressure")
        high_chol = yes_no("Diagnosed high cholesterol")
        heart_disease = yes_no("Heart disease or heart attack history")
        stroke = yes_no("History of stroke")
        family_history = yes_no("Family history of diabetes")

    with col3:
        difficulty_walking = yes_no("Difficulty walking or climbing stairs")
        smoker = yes_no("Smoked at least 100 cigarettes in lifetime")
        physical_activity = yes_no("Physical activity in the past 30 days")
        education_label = st.selectbox("Education", list(education_options), index=4)
        income_label = st.selectbox("Annual household income", list(income_options), index=5)

    submitted = st.form_submit_button(
        "Estimate early risk", type="primary", width="stretch"
    )

if submitted:
    st.session_state.pop("guidance", None)
    profile = {
        "Age": age_options[age_label],
        "BMI": float(bmi),
        "GenHlth": general_health,
        "HighBP": high_bp,
        "PhysHlth": physical_health,
        "MentHlth": mental_health,
        "HeartDiseaseorAttack": heart_disease,
        "HighChol": high_chol,
        "DiffWalk": difficulty_walking,
        "Education": education_options[education_label],
        "Family_History_Diabetes": family_history,
        "Income": income_options[income_label],
        "PhysActivity": physical_activity,
        "Stroke": stroke,
        "Smoker": smoker,
    }

    feature_order = model_config["features"]
    profile_frame = pd.DataFrame([profile])[feature_order]
    probability = float(model.predict_proba(profile_frame)[0, 1])
    prediction = int(probability >= float(model_config["threshold"]))

    scaled_profile = cluster_scaler.transform(profile_frame)
    raw_cluster = int(kmeans_model.predict(scaled_profile)[0])
    risk_level = cluster_mapping[str(raw_cluster)]
    cri, contributions = calculate_cri(profile_frame, cri_config)

    st.session_state["result"] = {
        "profile": profile,
        "probability": probability,
        "prediction": prediction,
        "risk_level": risk_level,
        "cri": cri,
        "contributions": contributions,
    }

if "result" in st.session_state:
    result = st.session_state["result"]
    st.divider()
    st.subheader("Risk assessment")

    metric1, metric2, metric3 = st.columns(3)
    metric1.metric("Estimated early-risk score", f"{result['probability']:.1%}")
    metric2.metric("Composite Risk Index", f"{result['cri']:.1f}/100")
    metric3.metric("K-Means risk segment", result["risk_level"])

    if result["prediction"] == 1:
        st.info(
            "The score is above the validation-selected screening threshold. "
            "This indicates elevated questionnaire-based risk, not a diagnosis."
        )
    else:
        st.success(
            "The score is below the validation-selected screening threshold. "
            "This does not rule out diabetes or replace appropriate testing."
        )

    left, right = st.columns([1, 1])
    with left:
        st.markdown("#### Leading profile-based risk contributions")
        chart_data = result["contributions"].head(8).set_index("Feature")
        st.bar_chart(chart_data)
    with right:
        st.markdown("#### Model context")
        st.write(
            "The Stacking Ensemble combines Random Forest, XGBoost, AdaBoost "
            "and Gradient Boosting. The displayed threshold was selected on the "
            "validation set to balance sensitivity and specificity."
        )
        st.caption(
            "Risk segments are descriptive K-Means groups. Their training "
            "silhouette score was 0.113, indicating overlapping clusters."
        )

    st.markdown("### Evidence-grounded guidance")
    st.caption(
        "On request, the app semantically retrieves supporting ADA, CDC and "
        "NIDDK evidence before contacting Gemini."
    )

    if st.button("Generate personalized guidance", type="primary"):
        if not api_key:
            st.error("Enter a Gemini API key in the sidebar first.")
        else:
            with st.spinner("Retrieving evidence and generating guidance..."):
                try:
                    chunks, embeddings, embedding_model = load_rag_assets()
                    query = build_retrieval_query(result["profile"])
                    evidence = retrieve_evidence(
                        query, chunks, embeddings, embedding_model, top_k=5
                    )
                    recommendation = generate_recommendation(
                        api_key,
                        result["profile"],
                        result["probability"],
                        result["risk_level"],
                        result["cri"],
                        evidence,
                    )
                    st.session_state["guidance"] = {
                        "recommendation": recommendation,
                        "evidence": evidence,
                    }
                except Exception as error:
                    st.error(f"Recommendation could not be generated: {error}")

    if "guidance" in st.session_state:
        guidance = st.session_state["guidance"]
        with st.expander("View retrieved evidence sources"):
            st.dataframe(
                guidance["evidence"][
                    ["Similarity_Score", "Organization", "Title", "URL"]
                ],
                hide_index=True,
                width="stretch",
                column_config={"URL": st.column_config.LinkColumn("URL")},
            )
        st.markdown(guidance["recommendation"])

with st.expander("Model evaluation and explainability"):
    image_columns = st.columns(3)
    image_columns[0].image(
        str(FIGURE_DIR / "11_test_roc_and_pr_curves.png"),
        caption="Held-out ROC and precision–recall curves",
        width="stretch",
    )
    image_columns[1].image(
        str(FIGURE_DIR / "14_shap_global_importance.png"),
        caption="Global SHAP importance",
        width="stretch",
    )
    image_columns[2].image(
        str(FIGURE_DIR / "12_kmeans_composite_risk_groups.png"),
        caption="K-Means risk segments and CRI",
        width="stretch",
    )

st.divider()
st.caption(
    "The primary model was developed using synthetic/hypothetical data and "
    "externally evaluated on harmonized CDC BRFSS 2023 survey data. Predictions "
    "must not be used for diagnosis, emergency decisions, or medication changes."
)
