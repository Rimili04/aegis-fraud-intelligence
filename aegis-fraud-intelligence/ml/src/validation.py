import pandas as pd
import numpy as np

def validate_payload(df: pd.DataFrame):
    """Validate transaction schema and check for data anomalies."""
    issues = []
    # Check required canonical fields
    required = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
    for col in required:
        if col not in df.columns:
            issues.append(f"Missing critical numerical feature: {col}")
            
    # Check negative or malformed transaction amounts
    if 'amount' in df.columns:
        negatives = df[df['amount'] < 0]
        if len(negatives) > 0:
            issues.append(f"Found {len(negatives)} negative transaction amounts.")
            
    return issues
