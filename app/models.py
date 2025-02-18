import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
import joblib
import os

model = None

def load_or_train_model():
    global model
    if os.path.exists('insurance_fraud_model.pkl'):
        model = joblib.load('insurance_fraud_model.pkl')
    else:
        model = train_model()

def train_model(file=None):
    np.random.seed(42)

    if file:
        df = pd.read_csv(file)
    else:
        data = {
            'claim_amount': np.random.normal(5000, 2000, 5000).astype(int),
            'num_previous_claims': np.random.poisson(2, 5000),
            'customer_age': np.random.randint(18, 80, 5000),
            'policy_age': np.random.randint(1, 30, 5000),
            'incident_severity': np.random.randint(1, 5, 5000),
            'witness_present': np.random.choice([0, 1], 5000, p=[0.7, 0.3]),
            'fraudulent': np.random.choice([0, 1], 5000, p=[0.85, 0.15])
        }
        df = pd.DataFrame(data)

    X = df.drop(columns=['fraudulent'])
    y = df['fraudulent']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = GradientBoostingClassifier(n_estimators=500, learning_rate=0.05, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, 'insurance_fraud_model.pkl')
    
    return model

load_or_train_model()
