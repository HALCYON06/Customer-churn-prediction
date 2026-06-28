"""
model_utils.py -- shared model construction, scoring, and explanation logic.

Used by BOTH train_model.py and the Gradio app so the math is identical
everywhere. Key idea: the model predicts probabilities over the ordinal classes
1..5; we collapse those into a single, smooth "Churn Risk Score" on the 1-5 scale.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier

import churn_lib as cl


def build_pipeline() -> Pipeline:
    """Cleaning -> impute -> encode -> gradient-boosted classifier, as ONE object.

    Persisting the whole pipeline means the app loads a single file and never
    has to re-implement preprocessing (no train/serve skew).
    """
    numeric = cl.NUMERIC_FEATURES
    categorical = cl.CATEGORICAL_FEATURES

    num_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        # handle_unknown=ignore -> a never-before-seen category at predict time
        # becomes all-zeros instead of crashing the app.
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    pre = ColumnTransformer([
        ("num", num_pipe, numeric),
        ("cat", cat_pipe, categorical),
    ])
    # Hyperparameters explored via RandomizedSearchCV (25 candidates, 4-fold CV).
    # The search's own "best score" (macro-F1 0.774) did NOT hold up under an
    # independent 5-fold CV re-check -- it came back at 0.7695, same as the
    # untuned defaults (this is "search optimism": picking the max over many
    # noisy CV estimates is itself biased upward). Kept anyway because it is
    # not worse and mildly regularizes (shallower trees, larger leaf size).
    clf = HistGradientBoostingClassifier(
        max_iter=700, learning_rate=0.1, max_depth=10, max_leaf_nodes=15,
        min_samples_leaf=40, l2_regularization=1.0, random_state=42,
        early_stopping=True, validation_fraction=0.1, n_iter_no_change=25,
    )
    return Pipeline([("pre", pre), ("clf", clf)])


def risk_score_from_proba(proba: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """Collapse class probabilities into a continuous score on the 1-5 scale.

    Score = expected risk level = sum_k k * P(k)  (ranges 1.0 .. 5.0).
    A customer certain to be class 5 scores 5.0; certain class 1 scores 1.0.
    This keeps the original 1-5 scale of churn_risk_score but is smoother than a
    hard integer label, so customers can still be ranked within a band.
    Round it for the official integer label.
    """
    classes = np.asarray(classes, dtype=float)
    return (proba * classes).sum(axis=1)


def risk_tier(score: float) -> str:
    """Map a 1-5 score to a business-facing tier."""
    if score < 2.5:  return "Low"
    if score < 3.5:  return "Medium"
    if score < 4.5:  return "High"
    return "Critical"


def compute_baselines(pipeline, df_prepared: pd.DataFrame) -> dict:
    """Find the lowest-risk value for each feature (used for explanations).

    Categorical -> the level with the lowest mean predicted risk.
    Numeric     -> the 10th or 90th percentile, whichever lowers risk (per the
                   sign of its correlation with the score).
    """
    X = df_prepared[cl.NUMERIC_FEATURES + cl.CATEGORICAL_FEATURES]
    classes = pipeline.named_steps["clf"].classes_
    s = risk_score_from_proba(pipeline.predict_proba(X), classes)
    tmp = df_prepared.copy(); tmp["_score"] = s
    baselines = {}
    for c in cl.CATEGORICAL_FEATURES:
        baselines[c] = tmp.groupby(c)["_score"].mean().idxmin()
    for c in cl.NUMERIC_FEATURES:
        filled = tmp[c].fillna(tmp[c].median())
        corr = np.corrcoef(filled, tmp["_score"])[0, 1]
        pct = 10 if corr > 0 else 90
        baselines[c] = float(np.nanpercentile(tmp[c], pct))
    return baselines


def local_reasons(pipeline, row_df: pd.DataFrame, baselines: dict, top_n: int = 4):
    """Explain ONE customer's score without external libraries (no SHAP needed).

    Local counterfactual analysis: for each feature, replace the customer's
    value with the empirically LOWEST-RISK value for that feature (`baselines`,
    precomputed at train time -- e.g. membership -> Premium, feedback ->
    positive) and measure how much the risk score drops. Features whose best-case
    swap drops risk most are this customer's top risk drivers.
    Returns (base_score, [(feature, risk_points_attributable), ...]).
    """
    classes = pipeline.named_steps["clf"].classes_
    base_score = risk_score_from_proba(pipeline.predict_proba(row_df), classes)[0]

    contribs = []
    for feat in cl.NUMERIC_FEATURES + cl.CATEGORICAL_FEATURES:
        if feat not in baselines:
            continue
        probe = row_df.copy()
        probe[feat] = baselines[feat]
        s = risk_score_from_proba(pipeline.predict_proba(probe), classes)[0]
        contribs.append((feat, base_score - s))  # positive => pushes risk UP

    contribs.sort(key=lambda x: x[1], reverse=True)
    # Keep drivers worth >= 0.1 of a risk point (on the 1-5 scale).
    return base_score, [c for c in contribs if c[1] > 0.1][:top_n]
