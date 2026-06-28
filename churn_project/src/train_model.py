
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (accuracy_score, f1_score, cohen_kappa_score,
                             make_scorer, classification_report, confusion_matrix)

import churn_lib as cl
import model_utils as mu

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    raw = pd.read_csv(os.path.join(HERE, "data", "train.csv"))
    ref = cl.get_reference_date(raw)

    # Drop invalid-target rows, then clean + engineer.
    raw = raw[raw[cl.TARGET] != -1].copy()
    df = cl.prepare(raw, reference_date=ref)

    X = df[cl.NUMERIC_FEATURES + cl.CATEGORICAL_FEATURES]
    y = df[cl.TARGET]

    # --- Headline metrics via 5-fold cross-validation (robust, not luck-of-split) ---
    print("Running 5-fold cross-validation for headline metrics...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    qwk_scorer = make_scorer(cohen_kappa_score, weights="quadratic")
    f1m_scorer = make_scorer(f1_score, average="macro")
    cv_acc = cross_val_score(mu.build_pipeline(), X, y, cv=cv, scoring="accuracy")
    cv_f1m = cross_val_score(mu.build_pipeline(), X, y, cv=cv, scoring=f1m_scorer)
    cv_qwk = cross_val_score(mu.build_pipeline(), X, y, cv=cv, scoring=qwk_scorer)
    print(f"\n=== 5-fold CV performance (headline numbers) ===")
    print(f"Accuracy           : {cv_acc.mean():.4f} (+/- {cv_acc.std():.4f})")
    print(f"Macro-F1           : {cv_f1m.mean():.4f} (+/- {cv_f1m.std():.4f})")
    print(f"Quadratic W. Kappa : {cv_qwk.mean():.4f} (+/- {cv_qwk.std():.4f})")

    # --- Detailed per-class diagnostic on a single held-out split ---
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)

    pipe = mu.build_pipeline()
    print("\nFitting one holdout split for the per-class diagnostic...")
    pipe.fit(X_tr, y_tr)

    pred = pipe.predict(X_val)
    acc = accuracy_score(y_val, pred)
    f1m = f1_score(y_val, pred, average="macro")
    qwk = cohen_kappa_score(y_val, pred, weights="quadratic")
    print(f"\n=== Holdout split (diagnostic detail) ===")
    print(f"Accuracy : {acc:.4f}   Macro-F1 : {f1m:.4f}   QWK : {qwk:.4f}")
    print("\nPer-class report:\n", classification_report(y_val, pred))
    print("Confusion matrix (rows=true, cols=pred):\n",
          confusion_matrix(y_val, pred))

    # Refit on ALL data for the final shipped model.
    print("\nRefitting on full dataset for production model...")
    pipe.fit(X, y)

    os.makedirs(os.path.join(HERE, "models"), exist_ok=True)
    joblib.dump(pipe, os.path.join(HERE, "models", "churn_model.joblib"))
    meta = {
        "reference_date": str(ref.date()),
        "numeric_features": cl.NUMERIC_FEATURES,
        "categorical_features": cl.CATEGORICAL_FEATURES,
        "classes": [int(c) for c in pipe.named_steps["clf"].classes_],
        "cv_accuracy": round(cv_acc.mean(), 4),
        "cv_macro_f1": round(cv_f1m.mean(), 4),
        "cv_quadratic_weighted_kappa": round(cv_qwk.mean(), 4),
        "holdout_accuracy": round(acc, 4),
        "holdout_macro_f1": round(f1m, 4),
        "holdout_qwk": round(qwk, 4),
    }
    json.dump(meta, open(os.path.join(HERE, "models", "meta.json"), "w"), indent=2)

    # Low-risk baselines for the per-customer explanations in the app.
    baselines = mu.compute_baselines(pipe, df)
    json.dump(baselines, open(os.path.join(HERE, "models", "baselines.json"), "w"),
              indent=2, default=str)
    print("Saved models/churn_model.joblib, meta.json, baselines.json")

    # Score the official test set.
    test = pd.read_csv(os.path.join(HERE, "data", "test.csv"))
    tprep = cl.prepare(test, reference_date=ref)
    Xt = tprep[cl.NUMERIC_FEATURES + cl.CATEGORICAL_FEATURES]
    proba = pipe.predict_proba(Xt)
    score = mu.risk_score_from_proba(proba, pipe.named_steps["clf"].classes_)
    out = pd.DataFrame({
        "customer_id": test["customer_id"],
        # Official integer churn_risk_score on the 1-5 scale:
        "churn_risk_score": pipe.predict(Xt),
        # Continuous 1-5 score for finer ranking within a band:
        "churn_risk_score_precise": np.round(score, 2),
        "risk_tier": [mu.risk_tier(s) for s in score],
    })
    out.to_csv(os.path.join(HERE, "data", "test_predictions.csv"), index=False)
    print("Saved data/test_predictions.csv")


if __name__ == "__main__":
    main()
