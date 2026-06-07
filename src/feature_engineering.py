import pandas as pd
import numpy as np

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds interaction-level engineered features on top of base drug properties.
    Call this before preprocessing if working with raw drug pair data.
    """

    # Molecular weight ratio
    df["mw_ratio"] = df["molecular_weight_a"] / (df["molecular_weight_b"] + 1e-5)

    # LogP interaction score
    df["logp_interaction"] = df["logp_a"] * df["logp_b"]

    # Hydrogen bond donor/acceptor mismatch
    df["hbd_mismatch"] = np.abs(df["hbd_a"] - df["hbd_b"])
    df["hba_mismatch"] = np.abs(df["hba_a"] - df["hba_b"])

    # TPSA sum and diff
    df["tpsa_sum"] = df["tpsa_a"] + df["tpsa_b"]
    df["tpsa_diff"] = np.abs(df["tpsa_a"] - df["tpsa_b"])

    # CYP enzyme conflict score
    df["cyp_conflict_score"] = (
        df["cyp3a4_inhibitor_a"] * df["cyp3a4_inhibitor_b"] +
        df["cyp2d6_inhibitor_a"] * df["cyp2d6_inhibitor_b"] +
        df["cyp2c9_inhibitor_a"] * df["cyp2c9_inhibitor_b"]
    )

    # Protein binding competition
    df["protein_binding_competition"] = (
        df["protein_binding_a"] + df["protein_binding_b"]
    ) / 200.0

    # Half-life ratio
    df["half_life_ratio"] = df["half_life_a"] / (df["half_life_b"] + 1e-5)

    # Bioavailability product
    df["bioavailability_product"] = (
        df["bioavailability_a"] * df["bioavailability_b"]
    ) / 10000.0

    # Therapeutic index risk
    df["ti_risk"] = 1 / (df["therapeutic_index_a"] + df["therapeutic_index_b"] + 1e-5)

    # Both anticoagulants — high risk flag
    df["dual_anticoagulant_risk"] = (
        df["is_anticoagulant_a"] & df["is_anticoagulant_b"]
    ).astype(int)

    # NSAID + Anticoagulant risk
    df["nsaid_anticoagulant_risk"] = (
        (df["is_nsaid_a"] & df["is_anticoagulant_b"]) |
        (df["is_nsaid_b"] & df["is_anticoagulant_a"])
    ).astype(int)

    # Antidepressant combo risk (serotonin syndrome proxy)
    df["dual_antidepressant_risk"] = (
        df["is_antidepressant_a"] & df["is_antidepressant_b"]
    ).astype(int)

    # Statin + CYP3A4 inhibitor risk
    df["statin_cyp_risk"] = (
        (df["is_statin_a"] & df["cyp3a4_inhibitor_b"]) |
        (df["is_statin_b"] & df["cyp3a4_inhibitor_a"])
    ).astype(int)

    # Renal clearance ratio
    df["renal_clearance_ratio"] = df["renal_clearance_a"] / (
        df["renal_clearance_b"] + 1e-5
    )

    # Volume of distribution mismatch
    df["vd_mismatch"] = np.abs(df["vd_a"] - df["vd_b"])

    print(f"Feature engineering complete. Total features: {df.shape[1]}")
    return df


if __name__ == "__main__":
    import pandas as pd
    df = pd.read_csv("data/interactions.csv")
    df = engineer_features(df)
    print(df.shape)
    print(df.columns.tolist())
