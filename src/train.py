import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import lightgbm as lgb
import mlflow
import mlflow.sklearn
import optuna
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

from preprocess import load_and_preprocess


class DrugInteractionMLP(nn.Module):
    def __init__(self, input_dim, num_classes=5):
        super().__init__()
        
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Tanh(),
            nn.Linear(input_dim, input_dim),
            nn.Sigmoid()
        )
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        attn_weights = self.attention(x)
        x = x * attn_weights
        return self.network(x)


def get_class_weights(y):
    counts = np.bincount(y)
    total = len(y)
    weights = total / (len(counts) * counts)
    return torch.FloatTensor(weights)


def train_pytorch_model(X_train, y_train, X_val, y_val, input_dim, epochs=30):
    X_tr = torch.FloatTensor(X_train)
    y_tr = torch.LongTensor(y_train)
    X_v  = torch.FloatTensor(X_val)
    y_v  = torch.LongTensor(y_val)
    
    dataset = TensorDataset(X_tr, y_tr)
    loader  = DataLoader(dataset, batch_size=512, shuffle=True)
    
    model     = DrugInteractionMLP(input_dim)
    class_wts = get_class_weights(y_train)
    criterion = nn.CrossEntropyLoss(weight=class_wts)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for xb, yb in loader:
            optimizer.zero_grad()
            out  = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        scheduler.step()
        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs}  loss: {total_loss/len(loader):.4f}")
    
    model.eval()
    with torch.no_grad():
        val_preds = model(X_v).argmax(dim=1).numpy()
    
    acc = (val_preds == y_val).mean()
    print(f"PyTorch MLP val accuracy: {acc:.4f}")
    
    torch.save(model.state_dict(), "models/pytorch_mlp.pth")
    return model, acc


def objective_xgb(trial, X_train, y_train, X_val, y_val):
    params = {
        "n_estimators":     trial.suggest_int("n_estimators", 100, 500),
        "max_depth":        trial.suggest_int("max_depth", 3, 10),
        "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3),
        "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "use_label_encoder": False,
        "eval_metric": "mlogloss",
        "random_state": 42
    }
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)
    return (model.predict(X_val) == y_val).mean()


def train_all_models(X, y):
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    os.makedirs("models", exist_ok=True)
    mlflow.set_experiment("rxguard_ai")
    
    input_dim = X_train.shape[1]
    
    print("\nTraining PyTorch MLP...")
    with mlflow.start_run(run_name="pytorch_mlp"):
        model, acc = train_pytorch_model(X_train, y_train, X_val, y_val, input_dim)
        mlflow.log_metric("val_accuracy", acc)
    
    print("\nRunning Optuna HPO for XGBoost (100 trials)...")
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler())
    study.optimize(
        lambda trial: objective_xgb(trial, X_train, y_train, X_val, y_val),
        n_trials=100, show_progress_bar=True
    )
    
    best_params = study.best_params
    best_params.update({"use_label_encoder": False, "eval_metric": "mlogloss", "random_state": 42})
    
    print("\nTraining final XGBoost with best params...")
    with mlflow.start_run(run_name="xgboost_best"):
        xgb_model = xgb.XGBClassifier(**best_params)
        xgb_model.fit(X_train, y_train)
        acc_xgb = (xgb_model.predict(X_val) == y_val).mean()
        mlflow.log_params(best_params)
        mlflow.log_metric("val_accuracy", acc_xgb)
        mlflow.sklearn.log_model(xgb_model, "xgboost")
        joblib.dump(xgb_model, "models/xgboost.pkl")
        print(f"XGBoost val accuracy: {acc_xgb:.4f}")
    
    print("\nTraining LightGBM...")
    with mlflow.start_run(run_name="lightgbm"):
        lgb_model = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05,
                                        num_leaves=63, random_state=42)
        lgb_model.fit(X_train, y_train)
        acc_lgb = (lgb_model.predict(X_val) == y_val).mean()
        mlflow.log_metric("val_accuracy", acc_lgb)
        mlflow.sklearn.log_model(lgb_model, "lightgbm")
        joblib.dump(lgb_model, "models/lightgbm.pkl")
        print(f"LightGBM val accuracy: {acc_lgb:.4f}")
    
    print("\nTraining Random Forest...")
    with mlflow.start_run(run_name="random_forest"):
        rf_model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
        rf_model.fit(X_train, y_train)
        acc_rf = (rf_model.predict(X_val) == y_val).mean()
        mlflow.log_metric("val_accuracy", acc_rf)
        joblib.dump(rf_model, "models/random_forest.pkl")
        print(f"Random Forest val accuracy: {acc_rf:.4f}")
    
    print("\nTraining Logistic Regression...")
    with mlflow.start_run(run_name="logistic_regression"):
        lr_model = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
        lr_model.fit(X_train, y_train)
        acc_lr = (lr_model.predict(X_val) == y_val).mean()
        mlflow.log_metric("val_accuracy", acc_lr)
        joblib.dump(lr_model, "models/logistic_regression.pkl")
        print(f"Logistic Regression val accuracy: {acc_lr:.4f}")
    
    print("\nTraining Gradient Boosting...")
    with mlflow.start_run(run_name="gradient_boosting"):
        gb_model = GradientBoostingClassifier(n_estimators=200, random_state=42)
        gb_model.fit(X_train, y_train)
        acc_gb = (gb_model.predict(X_val) == y_val).mean()
        mlflow.log_metric("val_accuracy", acc_gb)
        joblib.dump(gb_model, "models/gradient_boosting.pkl")
        print(f"Gradient Boosting val accuracy: {acc_gb:.4f}")
    
    return X_val, y_val


if __name__ == "__main__":
    import sys
    sys.path.append("src")
    
    print("Loading and preprocessing data...")
    X, y, le, scaler, features = load_and_preprocess()
    
    print("\nStarting training...")
    X_val, y_val = train_all_models(X, y)
    print("\nAll models trained and saved.")
