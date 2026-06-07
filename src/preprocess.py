import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
import joblib
import os

def load_and_preprocess(filepath="data/interactions.csv"):
    df = pd.read_csv(filepath)
    
    # Drop drug name columns (not used as features)
    df = df.drop(columns=["drug_a", "drug_b"])
    
    # Encode target
    le = LabelEncoder()
    df["interaction_class"] = le.fit_transform(df["interaction_class"])
    
    os.makedirs("models", exist_ok=True)
    joblib.dump(le, "models/label_encoder.pkl")
    
    X = df.drop(columns=["interaction_class"])
    y = df["interaction_class"]
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, "models/scaler.pkl")
    
    print(f"Shape before SMOTE: {X_scaled.shape}, {y.shape}")
    print(f"Class distribution before SMOTE:\n{y.value_counts()}")
    
    # SMOTE for class imbalance
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_scaled, y)
    
    print(f"Shape after SMOTE: {X_resampled.shape}")
    
    return X_resampled, y_resampled, le, scaler, list(X.columns)

if __name__ == "__main__":
    X, y, le, scaler, features = load_and_preprocess()
    print("Preprocessing complete.")
    print(f"Features: {features}")
