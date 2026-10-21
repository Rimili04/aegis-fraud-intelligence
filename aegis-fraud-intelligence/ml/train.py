import sys
import os

# Ensure python path is aligned for importing local modules
sys.path.append(os.path.abspath('aegis-fraud-intelligence'))

import pandas as pd
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from ml.src.features import build_features

# Read PaySim dataset
dataset_path = "/content/PS_20174392719_1491204439457_log.csv"
if os.path.exists(dataset_path):
    print("Loading dataset...")
    # Let's read a subset to optimize training time and resources
    df = pd.read_csv(dataset_path, nrows=100000)
    
    X = build_features(df)
    y = df['isFraud'].astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("Training the classifier...")
    clf = HistGradientBoostingClassifier(random_state=42, max_iter=100)
    clf.fit(X_train, y_train)
    
    # Persist artifacts
    os.makedirs('aegis-fraud-intelligence/ml/artifacts', exist_ok=True)
    joblib.dump(clf, 'aegis-fraud-intelligence/ml/artifacts/model.joblib')
    print("Model artifact created successfully!")
else:
    print(f"Selected dataset not found at {dataset_path}")
