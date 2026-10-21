import os
import sys
import pandas as pd

sys.path.append(os.path.abspath('aegis-fraud-intelligence'))
from ml.src.features import build_features
from ml.src.validation import validate_payload

def test_feature_alignment():
    sample_data = pd.DataFrame([{
        'amount': 5000.00,
        'oldbalanceOrg': 10000.00,
        'newbalanceOrig': 5000.00,
        'oldbalanceDest': 0.00,
        'newbalanceDest': 5000.00,
        'type': 'TRANSFER'
    }])
    feats = build_features(sample_data)
    assert len(feats) == 1
    assert 'orig_balance_discrepancy' in feats.columns

    issues = validate_payload(sample_data)
    assert len(issues) == 0
