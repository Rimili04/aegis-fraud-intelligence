# AEGIS Fraud Intelligence

Batch transaction screening dashboard.

## Run locally

Install requirements with `pip install -r requirements.txt`, then run:

`streamlit run frontend/app.py`

The trained model must be available at `ml/artifacts/model.joblib`.
Use transaction CSV files with the feature columns expected by the model.
