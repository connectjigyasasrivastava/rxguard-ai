import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os

np.random.seed(42)

N = 100000

DRUG_NAMES = [
    "Aspirin", "Warfarin", "Metformin", "Lisinopril", "Atorvastatin",
    "Amoxicillin", "Ibuprofen", "Omeprazole", "Metoprolol", "Amlodipine",
    "Simvastatin", "Losartan", "Gabapentin", "Sertraline", "Levothyroxine",
    "Ciprofloxacin", "Prednisone", "Hydrochlorothiazide", "Furosemide", "Clopidogrel",
    "Digoxin", "Fluoxetine", "Alprazolam", "Zolpidem", "Tramadol",
    "Morphine", "Codeine", "Diazepam", "Clonazepam", "Phenytoin"
]

INTERACTION_CLASSES = [
    "No Interaction",
    "Minor Interaction",
    "Moderate Interaction",
    "Major Interaction",
    "Contraindicated"
]

def generate_dataset():
    drug_a = np.random.choice(DRUG_NAMES, N)
    drug_b = np.random.choice(DRUG_NAMES, N)

    # 50+ engineered features
    data = {
        "drug_a": drug_a,
        "drug_b": drug_b,
        "molecular_weight_a": np.random.uniform(100, 900, N),
        "molecular_weight_b": np.random.uniform(100, 900, N),
        "logp_a": np.random.uniform(-2, 7, N),
        "logp_b": np.random.uniform(-2, 7, N),
        "hbd_a": np.random.randint(0, 6, N),
        "hbd_b": np.random.randint(0, 6, N),
        "hba_a": np.random.randint(0, 10, N),
        "hba_b": np.random.randint(0, 10, N),
        "tpsa_a": np.random.uniform(0, 200, N),
        "tpsa_b": np.random.uniform(0, 200, N),
        "rotatable_bonds_a": np.random.randint(0, 15, N),
        "rotatable_bonds_b": np.random.randint(0, 15, N),
        "cyp3a4_inhibitor_a": np.random.randint(0, 2, N),
        "cyp3a4_inhibitor_b": np.random.randint(0, 2, N),
        "cyp2d6_inhibitor_a": np.random.randint(0, 2, N),
        "cyp2d6_inhibitor_b": np.random.randint(0, 2, N),
        "cyp2c9_inhibitor_a": np.random.randint(0, 2, N),
        "cyp2c9_inhibitor_b": np.random.randint(0, 2, N),
        "pgp_substrate_a": np.random.randint(0, 2, N),
        "pgp_substrate_b": np.random.randint(0, 2, N),
        "protein_binding_a": np.random.uniform(0, 100, N),
        "protein_binding_b": np.random.uniform(0, 100, N),
        "half_life_a": np.random.uniform(1, 72, N),
        "half_life_b": np.random.uniform(1, 72, N),
        "bioavailability_a": np.random.uniform(0, 100, N),
        "bioavailability_b": np.random.uniform(0, 100, N),
        "renal_clearance_a": np.random.uniform(0, 300, N),
        "renal_clearance_b": np.random.uniform(0, 300, N),
        "vd_a": np.random.uniform(0.1, 10, N),
        "vd_b": np.random.uniform(0.1, 10, N),
        "therapeutic_index_a": np.random.uniform(1, 10, N),
        "therapeutic_index_b": np.random.uniform(1, 10, N),
        "is_anticoagulant_a": np.random.randint(0, 2, N),
        "is_anticoagulant_b": np.random.randint(0, 2, N),
        "is_nsaid_a": np.random.randint(0, 2, N),
        "is_nsaid_b": np.random.randint(0, 2, N),
        "is_antidepressant_a": np.random.randint(0, 2, N),
        "is_antidepressant_b": np.random.randint(0, 2, N),
        "is_antibiotic_a": np.random.randint(0, 2, N),
        "is_antibiotic_b": np.random.randint(0, 2, N),
        "is_statin_a": np.random.randint(0, 2, N),
        "is_statin_b": np.random.randint(0, 2, N),
        "same_drug_class": (drug_a == drug_b).astype(int),
        "mw_diff": np.abs(np.random.uniform(100, 900, N) - np.random.uniform(100, 900, N)),
        "logp_diff": np.abs(np.random.uniform(-2, 7, N) - np.random.uniform(-2, 7, N)),
        "protein_binding_sum": np.random.uniform(0, 200, N),
        "both_cyp3a4": np.random.randint(0, 2, N),
        "both_cyp2d6": np.random.randint(0, 2, N),
        "shared_metabolism": np.random.randint(0, 2, N),
        "num_shared_targets": np.random.randint(0, 10, N),
        "tanimoto_similarity": np.random.uniform(0, 1, N),
        "side_effect_overlap": np.random.randint(0, 50, N),
    }

    # Imbalanced labels mimicking real-world 1:30 ratio
    weights = [0.60, 0.20, 0.10, 0.06, 0.04]
    data["interaction_class"] = np.random.choice(INTERACTION_CLASSES, N, p=weights)

    df = pd.DataFrame(data)
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/interactions.csv", index=False)
    print(f"Dataset generated: {df.shape}")
    print(df["interaction_class"].value_counts())
    return df

if __name__ == "__main__":
    generate_dataset()
