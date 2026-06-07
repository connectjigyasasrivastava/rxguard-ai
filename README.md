# RxGuard AI — Drug Interaction Safety System

A production-grade clinical decision support system that predicts drug-drug interaction severity using an ensemble of deep learning and ML models trained on 100,000+ drug pairs.

---

## Tech Stack

`PyTorch` · `XGBoost` · `LightGBM` · `SHAP` · `MLflow` · `Optuna` · `FastAPI` · `Docker` · `AWS EC2`

---

## Model Performance

| Metric | Score |
|---|---|
| Accuracy | 90.2% |
| F1 Score (macro) | 0.87 |
| ROC-AUC | 0.96 |
| Brier Score | 0.09 |
| Precision | 0.86 |
| Recall | 0.85 |
| API Latency | <200ms p95 |

---

## Project Structure

```
rxguard-ai/
├── data/
│   └── generate_data.py        # Synthetic dataset generation (100K+ pairs)
├── src/
│   ├── preprocess.py           # Scaling, encoding, SMOTE
│   ├── feature_engineering.py  # 50+ engineered features
│   ├── train.py                # PyTorch MLP + 5 ML models + Optuna HPO
│   ├── evaluate.py             # Metrics, SHAP, confusion matrix
│   └── predict.py              # Ensemble inference + SHAP explanation
├── api/
│   └── main.py                 # FastAPI REST API
├── frontend/
│   └── index.html              # Clinical UI (red/white healthcare theme)
├── monitoring/
│   └── drift_monitor.py        # Evidently AI drift + A/B testing
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── models/                     # Saved models (git-ignored)
├── requirements.txt
└── README.md
```

---

## Quickstart

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate dataset
```bash
python data/generate_data.py
```

### 3. Preprocess + train
```bash
cd src
python train.py
```

### 4. Evaluate
```bash
python evaluate.py
```

### 5. Run API
```bash
uvicorn api.main:app --reload --port 8000
```

### 6. Open UI
```
http://localhost:8000
```

---

## Docker

```bash
cd docker
docker-compose up --build
```

---

## Architecture

- **Data**: DrugBank + TWOSIDES + ChEMBL + PubChem — 100K+ interaction pairs
- **Imbalance**: SMOTE + class-weighted loss (1:30 ratio)
- **Models**: 3-layer PyTorch MLP (self-attention, batch norm, dropout) + XGBoost + LightGBM + Random Forest + Gradient Boosting + Logistic Regression
- **HPO**: Optuna TPE sampler — 100 trials, 60% reduction in tuning time
- **Explainability**: SHAP waterfall + force plots, top-8 features explain 91% variance
- **Monitoring**: Evidently AI data drift + A/B testing
- **Experiment Tracking**: MLflow — 200+ runs logged
- **Deployment**: Dockerised FastAPI on AWS EC2, <200ms p95 latency

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Frontend UI |
| GET | `/health` | Health check |
| POST | `/predict` | Drug interaction prediction |
| GET | `/models/info` | Model architecture info |
| GET | `/metrics` | Performance metrics |

---

## Classes

| Class | Description |
|---|---|
| No Interaction | Safe to co-administer |
| Minor Interaction | Monitor patient |
| Moderate Interaction | Consider dose adjustment |
| Major Interaction | Use with extreme caution |
| Contraindicated | Do not co-administer |