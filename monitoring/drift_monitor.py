import pandas as pd
import numpy as np
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, ClassificationPreset
from evidently.pipeline.column_mapping import ColumnMapping
import joblib
import os
import json
from datetime import datetime


def generate_reference_data(n=5000):
    np.random.seed(42)
    scaler = joblib.load("models/scaler.pkl")
    n_features = scaler.n_features_in_

    ref_data = pd.DataFrame(
        scaler.inverse_transform(
            np.random.randn(n, n_features)
        ),
        columns=[f"feature_{i}" for i in range(n_features)]
    )
    ref_data["target"] = np.random.randint(0, 5, n)
    ref_data["prediction"] = np.random.randint(0, 5, n)
    return ref_data


def generate_current_data(n=1000, drift=False):
    np.random.seed(99)
    scaler = joblib.load("models/scaler.pkl")
    n_features = scaler.n_features_in_

    if drift:
        current_data = pd.DataFrame(
            scaler.inverse_transform(
                np.random.randn(n, n_features) * 2.5 + 1.5
            ),
            columns=[f"feature_{i}" for i in range(n_features)]
        )
    else:
        current_data = pd.DataFrame(
            scaler.inverse_transform(
                np.random.randn(n, n_features)
            ),
            columns=[f"feature_{i}" for i in range(n_features)]
        )

    current_data["target"]     = np.random.randint(0, 5, n)
    current_data["prediction"] = np.random.randint(0, 5, n)
    return current_data


def run_drift_report(reference, current, output_path="monitoring/drift_report.html"):
    os.makedirs("monitoring", exist_ok=True)

    col_mapping = ColumnMapping(
        target     = "target",
        prediction = "prediction"
    )

    report = Report(metrics=[DataDriftPreset()])
    report.run(
        reference_data = reference,
        current_data   = current,
        column_mapping = col_mapping
    )
    report.save_html(output_path)
    print(f"Drift report saved -> {output_path}")
    return report


def run_ab_test(model_a_preds, model_b_preds, true_labels):
    acc_a = (np.array(model_a_preds) == np.array(true_labels)).mean()
    acc_b = (np.array(model_b_preds) == np.array(true_labels)).mean()

    winner = "Model A" if acc_a > acc_b else "Model B"

    result = {
        "timestamp":   datetime.now().isoformat(),
        "model_a_acc": round(float(acc_a), 4),
        "model_b_acc": round(float(acc_b), 4),
        "winner":      winner,
        "delta":       round(abs(float(acc_a) - float(acc_b)), 4)
    }

    os.makedirs("monitoring", exist_ok=True)
    with open("monitoring/ab_test_results.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nA/B Test Results:")
    print(f"  Model A accuracy : {acc_a:.4f}")
    print(f"  Model B accuracy : {acc_b:.4f}")
    print(f"  Winner           : {winner}")
    return result


if __name__ == "__main__":
    print("Generating reference and current data...")
    reference = generate_reference_data(5000)
    current   = generate_current_data(1000, drift=False)

    print("\nRunning drift report...")
    run_drift_report(reference, current)

    print("\nRunning A/B test simulation...")
    n = 1000
    true_labels  = np.random.randint(0, 5, n)
    model_a_preds = np.where(np.random.rand(n) > 0.1, true_labels,
                             np.random.randint(0, 5, n))
    model_b_preds = np.where(np.random.rand(n) > 0.13, true_labels,
                             np.random.randint(0, 5, n))
    run_ab_test(model_a_preds, model_b_preds, true_labels)
