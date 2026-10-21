from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "ml" / "artifacts" / "model.joblib"

st.set_page_config(
    page_title="AEGIS | Fraud Intelligence",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ AEGIS Fraud Intelligence")
st.caption("Batch transaction screening and risk analysis")

if not MODEL_PATH.exists():
    st.error(f"Trained model not found: {MODEL_PATH}")
    st.stop()

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

try:
    model = load_model()
except Exception as exc:
    st.error(f"Unable to load the model: {exc}")
    st.stop()

st.success("Model artifact loaded.")

st.subheader("Upload transactions")
uploaded = st.file_uploader("Upload a transaction CSV", type=["csv"])

if uploaded is not None:
    try:
        data = pd.read_csv(uploaded)
        st.write("Uploaded data preview")
        st.dataframe(data.head(20), use_container_width=True)

        if data.empty:
            st.warning("The uploaded CSV contains no transactions.")
            st.stop()

        expected = getattr(model, "feature_names_in_", None)
        if expected is not None:
            missing = [col for col in expected if col not in data.columns]
            if missing:
                st.error(
                    "Missing model features: " + ", ".join(map(str, missing))
                )
                st.stop()
            features = data.loc[:, list(expected)]
        else:
            features = data.select_dtypes(include="number").copy()
            if features.empty:
                st.error("No numeric feature columns were found.")
                st.stop()

        if not hasattr(model, "predict"):
            st.error("The saved artifact does not expose predict().")
            st.stop()

        predictions = model.predict(features)
        results = data.copy()
        results["prediction"] = predictions

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(features)
            classes = list(getattr(model, "classes_", []))
            if 1 in classes:
                fraud_index = classes.index(1)
                results["fraud_risk_score"] = probabilities[:, fraud_index]
            else:
                st.warning(
                    "Could not identify class 1 as the fraud class; "
                    "no fraud probability score was added."
                )

        st.subheader("Batch results")
        total = len(results)
        flagged = int((results["prediction"] == 1).sum())
        a, b, c = st.columns(3)
        a.metric("Transactions screened", f"{total:,}")
        b.metric("Flagged by model", f"{flagged:,}")
        c.metric("Flagged percentage", f"{100 * flagged / total:.2f}%")

        st.dataframe(results, use_container_width=True)

        st.download_button(
            "Download prediction results",
            data=results.to_csv(index=False).encode("utf-8"),
            file_name="aegis_predictions.csv",
            mime="text/csv",
        )

    except Exception as exc:
        st.error(f"Could not process this CSV: {exc}")
else:
    st.info("Upload a CSV containing the feature columns expected by your model.")
