import pandas as pd
import numpy as np

def build_features(df: pd.DataFrame):
    """Generate high-fidelity features for fraud scoring."""
    X = pd.DataFrame(index=df.index)
    X['amount'] = df['amount'].astype(float)
    X['oldbalanceOrg'] = df['oldbalanceOrg'].astype(float)
    X['newbalanceOrig'] = df['newbalanceOrig'].astype(float)
    X['oldbalanceDest'] = df['oldbalanceDest'].astype(float)
    X['newbalanceDest'] = df['newbalanceDest'].astype(float)
    
    # Category encoders for transaction types
    if 'type' in df.columns:
        types = ['CASH_OUT', 'PAYMENT', 'CASH_IN', 'TRANSFER', 'DEBIT']
        for t in types:
            X[f'type_{t}'] = (df['type'] == t).astype(int)
    else:
        # Fill dummy columns if column is absent
        for t in ['CASH_OUT', 'PAYMENT', 'CASH_IN', 'TRANSFER', 'DEBIT']:
            X[f'type_{t}'] = 0
            
    # High value flag
    X['is_high_value'] = (X['amount'] > 200000).astype(int)
    
    # Balance discrepancies
    X['orig_balance_discrepancy'] = X['newbalanceOrig'] + X['amount'] - X['oldbalanceOrg']
    X['dest_balance_discrepancy'] = X['oldbalanceDest'] + X['amount'] - X['newbalanceDest']
    
    return X
