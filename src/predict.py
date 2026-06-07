import numpy as np
import pandas as pd
import torch
import joblib
import shap
import sys
sys.path.append("src")
from train import DrugInteractionMLP


INTERACTION_LABELS = [
    "Contraindicated",
    "Major Interaction",
    "Minor Interaction",
    "Moderate Interaction",
    "No Interaction"
]

SEVERITY_COLOR = {
    "No Interaction":       "green",
    "Minor Interaction":    "yellow",
    "Moderate Interaction": "orange",
    "Major Interaction":    "red",
    "Contraindicated":      "darkred"
}


def load_artifacts():
    scaler = joblib.load("models/scaler.pkl")
    le     = joblib.load("models/label_encoder.pkl")
    xgb    = joblib.load("models/xgboost.pkl")
    lgb    = joblib.load("models/lightgbm.pkl")
    rf     = joblib.load("models/random_forest.pkl")

    feature_names = scaler.feature_names_in_ if hasattr(scaler, "feature_names_in_") else None

    n_features = scaler.n_features_in_
    mlp = DrugInteractionMLP(n_features)
    mlp.load_state_dict(torch.load("models/pytorch_mlp.pth", map_location="cpu"))
    mlp.eval()

    return {"scaler": scaler, "le": le, "xgboost": xgb,
            "lightgbm": lgb, "random_forest": rf, "mlp": mlp,
            "feature_names": feature_names}


def preprocess_input(raw_features: dict, scaler):
    df     = pd.DataFrame([raw_features])
    scaled = scaler.transform(df)
    return scaled


def ensemble_predict(scaled_input, artifacts):
    xgb_proba = artifacts["xgboost"].predict_proba(scaled_input)
    lgb_proba = artifacts["lightgbm"].predict_proba(scaled_input)
    rf_proba  = artifacts["random_forest"].predict_proba(scaled_input)

    xt = torch.FloatTensor(scaled_input)
    with torch.no_grad():
        mlp_logits = artifacts["mlp"](xt)
        mlp_proba  = torch.softmax(mlp_logits, dim=1).numpy()

    avg_proba  = (xgb_proba + lgb_proba + rf_proba + mlp_proba) / 4.0
    pred_class = np.argmax(avg_proba, axis=1)[0]
    confidence = float(avg_proba[0][pred_class])

    label = artifacts["le"].inverse_transform([pred_class])[0]

    return {
        "predicted_class":  label,
        "confidence":       round(confidence*100, 2),
        "severity_color":   SEVERITY_COLOR.get(label, "gray"),
        "class_probabilities": {
            artifacts["le"].inverse_transform([i])[0]: round(float(p)*100, 2)
            for i, p in enumerate(avg_proba[0])
        }
    }


def get_shap_explanation(scaled_input, artifacts, feature_names):
    explainer  = shap.TreeExplainer(artifacts["xgboost"])
    shap_vals  = explainer.shap_values(scaled_input)

    if isinstance(shap_vals, list):
        sv = shap_vals[0]
    else:
        sv = shap_vals[0]

    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(len(sv))]

    shap_dict = {
        name: round(float(val), 4)
        for name, val in zip(feature_names, sv)
    }

    top8 = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:8]
    return dict(top8)


def predict(raw_features: dict):
    artifacts     = load_artifacts()
    scaled_input  = preprocess_input(raw_features, artifacts["scaler"])
    result        = ensemble_predict(scaled_input, artifacts)
    shap_exp      = get_shap_explanation(scaled_input, artifacts,
                                          artifacts["feature_names"])
    result["shap_explanation"] = shap_exp
    return result


if __name__ == "__main__":
    sample = {
        "molecular_weight_a": 325.0,  "molecular_weight_b": 180.0,
        "logp_a": 2.5,                "logp_b": 1.2,
        "hbd_a": 2,                   "hbd_b": 1,
        "hba_a": 4,                   "hba_b": 3,
        "tpsa_a": 75.0,               "tpsa_b": 60.0,
        "rotatable_bonds_a": 5,       "rotatable_bonds_b": 3,
        "cyp3a4_inhibitor_a": 1,      "cyp3a4_inhibitor_b": 1,
        "cyp2d6_inhibitor_a": 0,      "cyp2d6_inhibitor_b": 1,
        "cyp2c9_inhibitor_a": 1,      "cyp2c9_inhibitor_b": 0,
        "pgp_substrate_a": 1,         "pgp_substrate_b": 0,
        "protein_binding_a": 90.0,    "protein_binding_b": 85.0,
        "half_life_a": 12.0,          "half_life_b": 8.0,
        "bioavailability_a": 80.0,    "bioavailability_b": 70.0,
        "renal_clearance_a": 120.0,   "renal_clearance_b": 90.0,
        "vd_a": 2.5,                  "vd_b": 1.8,
        "therapeutic_index_a": 3.0,   "therapeutic_index_b": 2.5,
        "is_anticoagulant_a": 1,      "is_anticoagulant_b": 0,
        "is_nsaid_a": 0,              "is_nsaid_b": 1,
        "is_antidepressant_a": 0,     "is_antidepressant_b": 0,
        "is_antibiotic_a": 0,         "is_antibiotic_b": 0,
        "is_statin_a": 0,             "is_statin_b": 0,
        "same_drug_class": 0,
        "mw_diff": 145.0,
        "logp_diff": 1.3,
        "protein_binding_sum": 175.0,
        "both_cyp3a4": 1,
        "both_cyp2d6": 0,
        "shared_metabolism": 1,
        "num_shared_targets": 3,
        "tanimoto_similarity": 0.4,
        "side_effect_overlap": 12
    }

    result = predict(sample)
    print(f"\nPrediction : {result['predicted_class']}")
    print(f"Confidence : {result['confidence']}%")
    print(f"Top SHAP   : {result['shap_explanation']}")
