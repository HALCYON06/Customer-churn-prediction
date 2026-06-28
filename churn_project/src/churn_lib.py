"""
churn_lib.py
============
Single source of truth for cleaning + feature engineering.

Why this file exists:
The SAME transformations must run when we (a) train the model and (b) score a
live customer in the web app. If they ever drift apart you get "train/serve
skew" -- the model sees differently-shaped data at prediction time and silently
gives garbage. By importing these functions everywhere, that can't happen.

Two stages, kept deliberately separate:
  1. sanitize_junk(df)      -> turn placeholders/impossible values into real NaN
  2. engineer_features(df)  -> derive model-ready columns (tenure, login hour...)

Imputation (filling NaN) and encoding happen LATER, inside the scikit-learn
Pipeline, so that the fill values are learned from training data only.
"""
import numpy as np
import pandas as pd

# --- column roles -----------------------------------------------------------
TARGET = "churn_risk_score"
ID_COL = "customer_id"

# Columns we drop for modelling (identifiers / free text / leakage-prone).
DROP_COLS = [
    "customer_id", "Name", "security_no", "referral_id",
    "joining_date",      # replaced by engineered 'tenure_days'
    "last_visit_time",   # replaced by engineered 'last_visit_hour'
]

NUMERIC_FEATURES = [
    "age", "days_since_last_login", "avg_time_spent", "avg_transaction_value",
    "avg_frequency_login_days", "points_in_wallet",
    "tenure_days", "last_visit_hour",   # engineered
]

CATEGORICAL_FEATURES = [
    "gender", "region_category", "membership_category",
    "joined_through_referral", "preferred_offer_types", "medium_of_operation",
    "internet_option", "used_special_discount", "offer_application_preference",
    "past_complaint", "complaint_status", "feedback",
    "was_referred",   # engineered
]

# Tokens that mean "missing / unknown" but are disguised as real values.
JUNK_TOKENS = {"?", "Error", "Unknown", "xxxxxxxx", "NA", "N/A", "nan", ""}


def sanitize_junk(df: pd.DataFrame) -> pd.DataFrame:
    """Convert every disguised-missing value and impossible number into NaN.

    Pure & idempotent: safe to call on the train set, the test set, or a single
    live customer row. It never drops rows and never imputes -- it only marks
    bad cells as NaN so later steps can handle them consistently.
    """
    df = df.copy()

    # 1) avg_frequency_login_days arrives as TEXT because of the "Error" token.
    if "avg_frequency_login_days" in df:
        df["avg_frequency_login_days"] = pd.to_numeric(
            df["avg_frequency_login_days"], errors="coerce")

    # 2) Replace junk string tokens across all object/text columns.
    obj_cols = df.select_dtypes(include="object").columns
    for c in obj_cols:
        df[c] = df[c].apply(
            lambda v: np.nan if (isinstance(v, str) and v.strip() in JUNK_TOKENS) else v)

    # 3) Impossible negatives -> NaN.
    #    days_since_last_login uses -999 as a sentinel; any negative is invalid.
    if "days_since_last_login" in df:
        df.loc[df["days_since_last_login"] < 0, "days_since_last_login"] = np.nan
    #    Time spent, wallet points, frequency, age cannot be negative.
    for c in ["avg_time_spent", "points_in_wallet", "avg_transaction_value",
              "age", "avg_frequency_login_days"]:
        if c in df:
            df.loc[df[c] < 0, c] = np.nan

    return df


def engineer_features(df: pd.DataFrame, reference_date: pd.Timestamp = None) -> pd.DataFrame:
    """Derive model-ready columns. `reference_date` is the data snapshot date
    (the latest joining_date in TRAIN) and is saved so the app reuses the exact
    same value -- otherwise tenure would shift every day."""
    df = df.copy()

    # was_referred: a clean Yes/No from the messy referral_id column.
    if "referral_id" in df:
        df["was_referred"] = np.where(
            df["referral_id"].astype(str).str.strip().isin(["xxxxxxxx", "", "nan"]),
            "No", "Yes")

    # tenure_days: how long the customer has been with us.
    if "joining_date" in df:
        jd = pd.to_datetime(df["joining_date"], errors="coerce")
        ref = reference_date if reference_date is not None else jd.max()
        df["tenure_days"] = (ref - jd).dt.days

    # last_visit_hour: hour of day of last visit (0-23) from "HH:MM:SS".
    if "last_visit_time" in df:
        df["last_visit_hour"] = pd.to_datetime(
            df["last_visit_time"], format="%H:%M:%S", errors="coerce").dt.hour

    return df


def prepare(df: pd.DataFrame, reference_date: pd.Timestamp = None) -> pd.DataFrame:
    """Full pipeline up to (not including) imputation/encoding."""
    return engineer_features(sanitize_junk(df), reference_date=reference_date)


def get_reference_date(train_df: pd.DataFrame) -> pd.Timestamp:
    """The snapshot date = latest joining date in the training data."""
    return pd.to_datetime(train_df["joining_date"], errors="coerce").max()
