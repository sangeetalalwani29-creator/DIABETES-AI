# Diabetes Early-Risk AI Framework

Deployment-ready Streamlit interface for the research prototype developed in the accompanying notebooks.

## Included capabilities

- Stacking Ensemble early-risk prediction using 15 selected predictors
- Validation-selected decision threshold
- Composite Risk Index from 0 to 100
- K-Means risk stratification with K=4
- Profile-level risk-contribution display
- Semantic retrieval from ADA, CDC and NIDDK evidence
- Gemini-generated evidence-grounded educational guidance
- Model-performance and XAI figures

## Run locally

Use Python 3.11.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

On Windows, activate with `.venv\\Scripts\\activate`.

The Gemini key can be entered in the masked sidebar field. It is optional until guidance generation is requested.

## Streamlit Community Cloud

1. Upload this folder to a GitHub repository without `.env` or `secrets.toml`.
2. Create a Streamlit app with `app.py` as the entry point.
3. Optionally add the following under the deployed app's **Settings → Secrets**:

```toml
GEMINI_API_KEY = "your-key"
```

If no deployment secret is configured, a user can enter a key in the masked sidebar field.

## Verify the deployment package

After installing the dependencies, run:

```bash
python tests/smoke_test.py
```

The test reproduces the locked notebook demo score and validates the clustering and RAG artifacts.

## Important reproducibility note

The serialized scikit-learn models were created with scikit-learn 1.6.1. The pinned dependencies should be retained to avoid model-deserialization incompatibility.

## Scope

This application is an educational research prototype. The primary model was developed using synthetic/hypothetical data and externally evaluated using harmonized CDC BRFSS 2023 survey data. It must not be used for diagnosis, emergency decisions, or medication changes.
