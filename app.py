import base64
import os
import pickle
from itertools import combinations

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS


app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
PLOTS_DIR = os.path.join(BASE_DIR, "static", "plots")


def read_csv(name):
    return pd.read_csv(os.path.join(DATA_DIR, name))


drug_df = read_csv("drugs.csv")
interaction_df = read_csv("interactions.csv")
brand_df = read_csv("brand_names.csv")
pregnancy_df = read_csv("pregnancy_unsafe.csv")
allergy_df = read_csv("allergy_classes.csv")

interaction_lookup = {}
for _, row in interaction_df.iterrows():
    pair = frozenset([str(row["drug1"]).strip().lower(), str(row["drug2"]).strip().lower()])
    interaction_lookup[pair] = {
        "severity": str(row.get("severity", "moderate")).strip().lower(),
        "description": str(row.get("description", "Potential interaction detected.")),
        "alternative": str(row.get("alternative", "Consult your physician.")),
        "mechanism": str(row.get("mechanism", "Pharmacokinetic or pharmacodynamic interaction.")),
    }

brand_lookup = {
    str(row["brand"]).strip().lower(): str(row["generic"]).strip()
    for _, row in brand_df.iterrows()
}

drug_names = sorted(set(
    drug_df["name"].dropna().str.strip().tolist()
    + brand_df["brand"].dropna().str.strip().tolist()
))

pregnancy_unsafe = set(pregnancy_df["drug"].str.strip().str.lower().tolist())
allergy_lookup = {
    str(row["drug"]).strip().lower(): str(row["allergy_class"]).strip()
    for _, row in allergy_df.iterrows()
}
category_lookup = drug_df.drop_duplicates("name").set_index("name")["category"].to_dict()

model_bundle = None
model_path = os.path.join(MODEL_DIR, "best_model.pkl")
if os.path.exists(model_path):
    with open(model_path, "rb") as model_file:
        model_bundle = pickle.load(model_file)

SEVERITY_RANK = {"severe": 3, "moderate": 2, "mild": 1, "none": 0, "unknown": 0}
SEVERITY_LABELS = {3: "severe", 2: "moderate", 1: "mild", 0: "none"}


def clean_name(value):
    return value.strip().lower()


def resolve_drug_name(name):
    return brand_lookup.get(clean_name(name), name.strip())


def predict_pair(drug_a, drug_b):
    if not model_bundle:
        return "unknown", 50, {}

    try:
        encoder_a = model_bundle["le1"]
        encoder_b = model_bundle["le2"]
        severity_encoder = model_bundle["le_sev"]
        model = model_bundle["model"]

        category_a = category_lookup.get(drug_a, "Unknown")
        category_b = category_lookup.get(drug_b, "Unknown")
        encoded_a = category_a if category_a in encoder_a.classes_ else "Unknown"
        encoded_b = category_b if category_b in encoder_b.classes_ else "Unknown"

        features = np.array([[
            encoder_a.transform([encoded_a])[0] if encoded_a in encoder_a.classes_ else 0,
            encoder_b.transform([encoded_b])[0] if encoded_b in encoder_b.classes_ else 0,
            0,
            0,
            0,
            int(category_a == category_b),
        ]])

        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
        raw_label = severity_encoder.inverse_transform([prediction])[0]
        confidence = round(float(probabilities.max()) * 100)
        severity = SEVERITY_LABELS.get(int(raw_label), "unknown")
        probability_map = {}

        for index, probability in enumerate(probabilities):
            raw = severity_encoder.inverse_transform([index])[0]
            probability_map[SEVERITY_LABELS.get(int(raw), "none")] = round(float(probability) * 100)

        return severity, confidence, probability_map
    except Exception:
        return "unknown", 50, {}


def check_drug_pairs(drugs):
    results = []

    for drug_a, drug_b in combinations(drugs, 2):
        pair_key = frozenset([clean_name(drug_a), clean_name(drug_b)])
        predicted_severity, confidence, probabilities = predict_pair(drug_a, drug_b)

        if pair_key in interaction_lookup:
            info = interaction_lookup[pair_key]
            severity = info["severity"]
            description = info["description"]
            alternative = info["alternative"]
            mechanism = info["mechanism"]
            source = "database"
        else:
            severity = "none"
            description = "No known interaction found."
            alternative = "-"
            mechanism = "-"
            source = "model"

        results.append({
            "drug1": drug_a,
            "drug2": drug_b,
            "severity": severity,
            "description": description,
            "alternative": alternative,
            "mechanism": mechanism,
            "ml_severity": predicted_severity,
            "ml_confidence": confidence,
            "ml_probs": probabilities,
            "source": source,
        })

    return sorted(results, key=lambda item: SEVERITY_RANK.get(item["severity"], 0), reverse=True)


def overall_risk(interactions):
    if not interactions:
        return "none"
    highest = max(SEVERITY_RANK.get(item["severity"], 0) for item in interactions)
    return ["none", "mild", "moderate", "severe"][highest]


def risk_score(interactions):
    if not interactions:
        return 0

    weights = {"severe": 100, "moderate": 50, "mild": 20, "none": 0}
    scores = [weights.get(item["severity"], 0) for item in interactions]
    return min(100, int(sum(scores) / len(scores) + max(scores) * 0.3))


def patient_warnings(drugs, age, gender, pregnant, conditions, allergies):
    warnings = []
    conditions = [condition.lower() for condition in conditions]

    for drug in drugs:
        key = clean_name(drug)

        if "kidney" in conditions and key in ["metformin", "ibuprofen", "naproxen", "diclofenac", "celecoxib"]:
            warnings.append({"type": "contraindication", "icon": "!", "text": f"{drug} may worsen kidney function."})
        if "liver" in conditions and key in ["paracetamol", "methotrexate", "amiodarone", "atorvastatin"]:
            warnings.append({"type": "contraindication", "icon": "!", "text": f"{drug} requires caution in liver disease."})
        if "heart" in conditions and key in ["ibuprofen", "naproxen", "diclofenac", "celecoxib"]:
            warnings.append({"type": "contraindication", "icon": "!", "text": f"{drug} may increase cardiovascular risk."})
        if "diabetes" in conditions and key in ["prednisone", "prednisolone", "dexamethasone"]:
            warnings.append({"type": "contraindication", "icon": "!", "text": f"{drug} can raise blood glucose; monitor carefully."})
        if pregnant and key in pregnancy_unsafe:
            warnings.append({"type": "pregnancy", "icon": "!", "text": f"{drug} is unsafe during pregnancy."})

        drug_class = allergy_lookup.get(key)
        if drug_class and drug_class.lower() in allergies:
            warnings.append({"type": "allergy", "icon": "!", "text": f"{drug} belongs to {drug_class}; allergy risk."})
        if gender == "female" and key in ["finasteride", "dutasteride"]:
            warnings.append({"type": "gender", "icon": "!", "text": f"{drug} is contraindicated in women."})

    if age:
        try:
            age_value = int(age)
        except ValueError:
            return warnings

        for drug in drugs:
            key = clean_name(drug)
            if age_value > 65 and key in ["diazepam", "zolpidem", "alprazolam", "amitriptyline", "diphenhydramine"]:
                warnings.append({"type": "age", "icon": "!", "text": f"{drug} is high-risk in patients over 65."})
            if age_value < 18 and key == "aspirin":
                warnings.append({"type": "age", "icon": "!", "text": "Aspirin is contraindicated under 18."})

    return warnings


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/drugs")
def get_drugs():
    query = request.args.get("q", "").lower()
    matches = [name for name in drug_names if query in name.lower()][:20]
    return jsonify(matches)


@app.route("/api/check", methods=["POST"])
def check():
    payload = request.get_json() or {}
    raw_drugs = [drug.strip() for drug in payload.get("drugs", []) if drug.strip()]

    if len(raw_drugs) < 2:
        return jsonify({"error": "Please enter at least 2 drugs."}), 400

    drugs = [resolve_drug_name(drug) for drug in raw_drugs]
    resolved = {
        original: resolved_name
        for original, resolved_name in zip(raw_drugs, drugs)
        if clean_name(original) != clean_name(resolved_name)
    }

    interactions = check_drug_pairs(drugs)
    warnings = patient_warnings(
        drugs=drugs,
        age=payload.get("age", ""),
        gender=payload.get("gender", ""),
        pregnant=payload.get("pregnant", False),
        conditions=payload.get("conditions", []),
        allergies=payload.get("allergies", "").strip().lower(),
    )

    nodes = [{"id": drug, "category": category_lookup.get(drug, "Unknown")} for drug in drugs]
    links = [
        {"source": item["drug1"], "target": item["drug2"], "severity": item["severity"]}
        for item in interactions
        if item["severity"] != "none"
    ]

    return jsonify({
        "drugs": drugs,
        "raw_drugs": raw_drugs,
        "resolved": resolved,
        "risk": overall_risk(interactions),
        "risk_score": risk_score(interactions),
        "interactions": interactions,
        "warnings": warnings,
        "graph": {"nodes": nodes, "links": links},
        "model_results": model_bundle.get("results", {}) if model_bundle else {},
    })


@app.route("/api/plots")
def plots_api():
    plots = {}
    for name in ["model_comparison", "feature_importance", "ann_training_curves"]:
        path = os.path.join(PLOTS_DIR, f"{name}.png")
        if os.path.exists(path):
            with open(path, "rb") as image_file:
                plots[name] = base64.b64encode(image_file.read()).decode()
    return jsonify(plots)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)
