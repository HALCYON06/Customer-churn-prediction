"""
eda.py -- Exploratory Data Analysis on the cleaned training data.

Run:  python3 src/eda.py
Outputs:
  reports/eda_report.md          (statistics, churn drivers, in words)
  reports/figures/*.png          (charts, if matplotlib/seaborn installed)

It degrades gracefully: if plotting libraries are missing, you still get the
full text report.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
import numpy as np
import pandas as pd
import churn_lib as cl

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(HERE, "reports", "figures")
os.makedirs(FIG, exist_ok=True)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style="whitegrid")
    PLOT = True
except Exception as e:
    PLOT = False
    print(f"[warn] plotting libs not available ({e}); text report only.")


def main():
    df = pd.read_csv(os.path.join(HERE, "data", "train_clean.csv"))
    T = cl.TARGET
    # Binary "high risk" view (4-5) for clearer driver analysis
    df["high_risk"] = (df[T] >= 4).astype(int)

    out = ["# Exploratory Data Analysis\n"]
    out.append(f"Rows: **{len(df):,}**")
    out.append("\n## 1. Target distribution (churn_risk_score 1-5)\n")
    vc = df[T].value_counts().sort_index()
    out.append("| score | count | share |\n|---|---|---|")
    for k, v in vc.items():
        out.append(f"| {k} | {v:,} | {v/len(df)*100:.1f}% |")
    out.append(f"\n- High risk (4-5): **{df['high_risk'].mean()*100:.1f}%** of customers")

    # Numeric drivers: mean by high_risk
    out.append("\n## 2. Numeric features: mean for High-risk vs Low-risk\n")
    out.append("| feature | low-risk mean | high-risk mean | gap |\n|---|---|---|---|")
    for c in cl.NUMERIC_FEATURES:
        lo = df.loc[df.high_risk == 0, c].mean()
        hi = df.loc[df.high_risk == 1, c].mean()
        out.append(f"| {c} | {lo:,.1f} | {hi:,.1f} | {hi-lo:+,.1f} |")

    # Categorical drivers: high-risk rate by category
    out.append("\n## 3. Categorical features: high-risk rate by level\n")
    for c in cl.CATEGORICAL_FEATURES:
        rates = df.groupby(c)["high_risk"].mean().sort_values(ascending=False)
        out.append(f"\n**{c}**")
        for lvl, r in rates.items():
            out.append(f"- {lvl}: {r*100:.1f}% high-risk")

    # Correlations among numerics
    corr = df[cl.NUMERIC_FEATURES + ["high_risk"]].corr()["high_risk"].drop("high_risk")
    out.append("\n## 4. Correlation of numeric features with high-risk\n")
    out.append("| feature | corr |\n|---|---|")
    for c, v in corr.sort_values(key=abs, ascending=False).items():
        out.append(f"| {c} | {v:+.3f} |")

    with open(os.path.join(HERE, "reports", "eda_report.md"), "w") as f:
        f.write("\n".join(out))
    print("Wrote reports/eda_report.md")

    if not PLOT:
        return

    # ---- Figures ----
    # 1. Target distribution
    plt.figure(figsize=(6, 4))
    sns.countplot(x=T, data=df, palette="viridis")
    plt.title("Churn risk score distribution"); plt.tight_layout()
    plt.savefig(f"{FIG}/01_target_distribution.png", dpi=110); plt.close()

    # 2. Membership category vs high risk
    plt.figure(figsize=(8, 4))
    order = df.groupby("membership_category")["high_risk"].mean().sort_values().index
    sns.barplot(x="membership_category", y="high_risk", data=df, order=order, palette="rocket")
    plt.xticks(rotation=30, ha="right"); plt.ylabel("high-risk rate")
    plt.title("High-risk rate by membership category"); plt.tight_layout()
    plt.savefig(f"{FIG}/02_membership_vs_risk.png", dpi=110); plt.close()

    # 3. Correlation heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(df[cl.NUMERIC_FEATURES + ["high_risk"]].corr(), annot=True,
                fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Numeric correlation heatmap"); plt.tight_layout()
    plt.savefig(f"{FIG}/03_corr_heatmap.png", dpi=110); plt.close()

    # 4. points_in_wallet distribution by risk
    plt.figure(figsize=(7, 4))
    sns.kdeplot(data=df, x="points_in_wallet", hue="high_risk", fill=True, common_norm=False)
    plt.title("points_in_wallet by risk group"); plt.tight_layout()
    plt.savefig(f"{FIG}/04_points_by_risk.png", dpi=110); plt.close()

    # 5. feedback vs risk
    plt.figure(figsize=(9, 4))
    order = df.groupby("feedback")["high_risk"].mean().sort_values().index
    sns.barplot(x="feedback", y="high_risk", data=df, order=order, palette="mako")
    plt.xticks(rotation=40, ha="right"); plt.ylabel("high-risk rate")
    plt.title("High-risk rate by feedback"); plt.tight_layout()
    plt.savefig(f"{FIG}/05_feedback_vs_risk.png", dpi=110); plt.close()
    print(f"Wrote figures to {FIG}")


if __name__ == "__main__":
    main()
