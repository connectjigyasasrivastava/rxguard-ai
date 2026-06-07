import numpy as np
import pandas as pd
import torch
import shap
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score, brier_score_loss,
    classification_report, confusion_matrix
)
from sklearn.preprocessing import label_binarize
import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.append("src")
from train import DrugInteractionMLP


def load_models(input_dim):
    models = {}

    xgb_model = joblib.load("models/xgboost.pkl")
    models["xgboost"] = xgb_model

    lgb_model = joblib.load("models/lightgbm.pkl")
    models["lightgbm"] = lgb_model

    rf_model = joblib.load("models/random_forest.pkl")
    models["random_forest"] = rf_model

    lr_model = joblib.load("models/logistic_regression.pkl")
    models["logistic_regression"] = lr_model

    gb_model = joblib.load("models/gradient_boosting.pkl")
    models["gradient_boosting"] = gb_model

    mlp = DrugInteractionMLP(input_dim)
    mlp.load_state_dict(torch.load("models/pytorch_mlp.pth"))
    mlp.eval()
    models["pytorch_mlp"] = mlp

    return models


def evaluate_sklearn_model(model, X_val, y_val, model_name, n_classes=5):
    preds  = model.predict(X_val)
    probas = model.predict_proba(X_val)

    acc  = accuracy_score(y_val, preds)
    f1   = f1_score(y_val, preds, average="macro")
    prec = precision_score(y_val, preds, average="macro", zero_division=0)
    rec  = recall_score(y_val, preds, average="macro", zero_division=0)

    y_bin   = label_binarize(y_val, classes=list(range(n_classes)))
    roc_auc = roc_auc_score(y_bin, probas, multi_class="ovr", average="macro")

    brier = np.mean([
        brier_score_loss((y_val==i).astype(int), probas[:, i])
        for i in range(n_classes)
    ])

    print(f"\n{'='*40}")
    print(f"Model: {model_name}")
    print(f"Accuracy  : {acc:.4f}")
    print(f"F1 (macro): {f1:.4f}")
    print(f"Precision : {prec:.4f}")
    print(f"Recall    : {rec:.4f}")
    print(f"ROC-AUC   : {roc_auc:.4f}")
    print(f"Brier     : {brier:.4f}")
    print(f"\n{classification_report(y_val, preds)}")

    return {"model": model_name, "accuracy": acc, "f1": f1,
            "precision": prec, "recall": rec, "roc_auc": roc_auc, "brier": brier}


def plot_confusion_matrix(model, X_val, y_val, model_name, class_names):
    preds = model.predict(X_val)
    cm    = confusion_matrix(y_val, preds)

    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Reds",
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Confusion Matrix — {model_name}")
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(f"models/{model_name}_confusion_matrix.png")
    plt.close()
    print(f"Saved confusion matrix for {model_name}")


def run_shap_analysis(model, X_val, feature_names, 
model_name="xgboost"):
    print(f"\nRunning SHAP analysis for {model_name}...")

    explainer  = shap.TreeExplainer(model)
    shap_vals  = explainer.shap_values(X_val[:500])
    explanation = explainer(X_val[:500])

    # summary plot
    plt.figure()
    shap.summary_plot(shap_vals, X_val[:500],
                      feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig("models/shap_summary.png")
    plt.close()

    # waterfall plot for first sample
    plt.figure()
    shap.plots.waterfall(explanation[0], show=False)
    plt.tight_layout()
    plt.savefig("models/shap_waterfall.png")
    plt.close()
    print("Saved shap_waterfall.png")

    # force plot for first sample
    shap.initjs()
    force = shap.force_plot(
        explainer.expected_value[0] if 
isinstance(explainer.expected_value, list) else 
explainer.expected_value,
        shap_vals[0][0] if isinstance(shap_vals, list) else 
shap_vals[0],
        X_val[0],
        feature_names=feature_names,
        show=False,
        matplotlib=True
    )
    plt.tight_layout()
    plt.savefig("models/shap_force.png", bbox_inches="tight")
    plt.close()
    print("Saved shap_force.png")

    if isinstance(shap_vals, list):
        sv = np.abs(shap_vals[0])
    else:
        sv = np.abs(shap_vals)

    importance = pd.Series(sv.mean(axis=0), index=feature_names)
    top8 = importance.nlargest(8)
    print(f"\nTop 8 SHAP features:\n{top8}")
    top8.to_csv("models/shap_top8_features.csv")

def run_evaluation(X_val, y_val, feature_names):
    input_dim  = X_val.shape[1]
    models     = load_models(input_dim)
    class_names = ["No Interaction", "Minor", "Moderate", "Major", "Contraindicated"]

    results = []
    for name, model in models.items():
        if name == "pytorch_mlp":
            continue
        res = evaluate_sklearn_model(model, X_val, y_val, name)
        results.append(res)
        plot_confusion_matrix(model, X_val, y_val, name, class_names)

    run_shap_analysis(models["xgboost"], X_val, feature_names)

    df_results = pd.DataFrame(results)
    df_results.to_csv("models/evaluation_results.csv", index=False)
    print("\nEvaluation results saved.")
    return df_results


if __name__ == "__main__":
    from preprocess import load_and_preprocess
    X, y, le, scaler, features = load_and_preprocess()
    from sklearn.model_selection import train_test_split
    _, X_val, _, y_val = train_test_split(X, y, test_size=0.2,
                                           random_state=42, stratify=y)
    run_evaluation(X_val, y_val, features)
