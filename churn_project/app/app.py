
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import numpy as np
import pandas as pd
import joblib
import gradio as gr

import churn_lib as cl
import model_utils as mu

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Human-readable names + retention advice for the top-driver explanations.
DRIVER_ADVICE = {
    "membership_category": "Upsell to a higher membership tier (Premium/Platinum almost never churn).",
    "feedback": "Address the negative feedback theme directly; route to customer success.",
    "points_in_wallet": "Credit loyalty points / wallet balance to re-engage.",
    "avg_transaction_value": "Offer targeted high-value bundles or personalised discounts.",
    "avg_frequency_login_days": "Re-engagement campaign: nudges, reminders, fresh content.",
    "days_since_last_login": "Win-back email — the customer has been inactive recently.",
    "past_complaint": "Proactively follow up on the unresolved complaint.",
    "complaint_status": "Escalate and resolve the open complaint.",
    "was_referred": "Strengthen onboarding for referred customers.",
    "tenure_days": "Tenure-based loyalty reward.",
    "avg_time_spent": "Improve engagement / content relevance.",
}

TIER_COLOR = {"Low": "#1a9850", "Medium": "#f6c000", "High": "#fb8c00", "Critical": "#d73027"}

# ----------------------------- load artifacts (once) ------------------------
pipe = joblib.load(os.path.join(HERE, "models", "churn_model.joblib"))
meta = json.load(open(os.path.join(HERE, "models", "meta.json")))
surv = json.load(open(os.path.join(HERE, "models", "survival_baseline.json")))
baselines = json.load(open(os.path.join(HERE, "models", "baselines.json")))
ref_date = pd.Timestamp(meta["reference_date"])
classes = np.array(meta["classes"])


def gauge_html(score, tier, color):
    # score is on the 1-5 scale; fill the bar as a fraction of that range.
    pct = max(0, min(100, (score - 1) / 4 * 100))
    return f"""
    <div style="font-family:sans-serif;padding:6px 2px">
      <div style="font-size:54px;font-weight:800;color:{color};line-height:1">{score:.2f}<span style="font-size:22px;color:#888">/5</span></div>
      <div style="font-size:20px;font-weight:700;color:{color};margin:2px 0 8px">{tier} risk</div>
      <div style="background:#eee;border-radius:8px;height:16px;width:100%;overflow:hidden">
        <div style="height:16px;width:{pct}%;background:{color};border-radius:8px"></div>
      </div>
    </div>"""


def build_row(age, gender, region_category, membership_category, tenure_years,
              joined_through_referral, preferred_offer_types, medium_of_operation,
              internet_option, days_since_last_login, last_visit_hour, avg_time_spent,
              avg_transaction_value, avg_frequency_login_days, points_in_wallet,
              used_special_discount, offer_application_preference, past_complaint,
              complaint_status, feedback):
    """Assemble a single RAW-style row, then run the SAME prepare() as training."""
    joining_date = ref_date - pd.Timedelta(days=int(tenure_years * 365))
    raw = pd.DataFrame([{
        "customer_id": "LIVE", "Name": "LIVE", "security_no": "X",
        "age": age, "gender": gender, "region_category": region_category,
        "membership_category": membership_category,
        "joining_date": joining_date.strftime("%Y-%m-%d"),
        "joined_through_referral": joined_through_referral,
        "referral_id": "CID00001" if joined_through_referral == "Yes" else "xxxxxxxx",
        "preferred_offer_types": preferred_offer_types,
        "medium_of_operation": medium_of_operation,
        "internet_option": internet_option,
        "last_visit_time": f"{int(last_visit_hour):02d}:00:00",
        "days_since_last_login": days_since_last_login,
        "avg_time_spent": avg_time_spent,
        "avg_transaction_value": avg_transaction_value,
        "avg_frequency_login_days": avg_frequency_login_days,
        "points_in_wallet": points_in_wallet,
        "used_special_discount": used_special_discount,
        "offer_application_preference": offer_application_preference,
        "past_complaint": past_complaint, "complaint_status": complaint_status,
        "feedback": feedback,
    }])
    prepped = cl.prepare(raw, reference_date=ref_date)
    return prepped[cl.NUMERIC_FEATURES + cl.CATEGORICAL_FEATURES]


def score_customer(age, gender, region_category, membership_category, tenure_years,
                   joined_through_referral, preferred_offer_types, medium_of_operation,
                   internet_option, days_since_last_login, last_visit_hour, avg_time_spent,
                   avg_transaction_value, avg_frequency_login_days, points_in_wallet,
                   used_special_discount, offer_application_preference, past_complaint,
                   complaint_status, feedback):
    X = build_row(age, gender, region_category, membership_category, tenure_years,
                  joined_through_referral, preferred_offer_types, medium_of_operation,
                  internet_option, days_since_last_login, last_visit_hour, avg_time_spent,
                  avg_transaction_value, avg_frequency_login_days, points_in_wallet,
                  used_special_discount, offer_application_preference, past_complaint,
                  complaint_status, feedback)

    proba = pipe.predict_proba(X)[0]
    score = float(mu.risk_score_from_proba(proba.reshape(1, -1), classes)[0])
    tier = mu.risk_tier(score)
    color = TIER_COLOR[tier]
    pred_class = int(classes[np.argmax(proba)])
    confidence = float(proba.max())

    # Probability across risk classes -> gr.Label dict.
    prob_dict = {f"Class {int(c)}": float(p) for c, p in zip(classes, proba)}

    # Business impact: risk-weighted annual value of this customer.
    risk_frac = (score - 1) / 4.0
    est_annual_value = avg_transaction_value * max(1, 30.0 / max(avg_frequency_login_days, 1))
    revenue_at_risk = risk_frac * est_annual_value

    # Expected timeline from the survival baselines.
    sb = surv.get(tier, {})
    md = sb.get("median_days_to_churn")
    ttc = f"~{md/30:.0f} months" if md else "Stable (tier rarely reaches 50% churn)"
    tier_rate = f"{sb.get('event_rate', 0)*100:.0f}%"

    summary_md = (
        f"### Business impact & timeline\n"
        f"| Metric | Value |\n|---|---|\n"
        f"| Predicted risk class (1–5) | **{pred_class}** |\n"
        f"| Model confidence | {confidence*100:.0f}% |\n"
        f"| Revenue at risk | **{revenue_at_risk:,.0f}** |\n"
        f"| Est. annual value | {est_annual_value:,.0f} |\n"
        f"| Est. time to churn | {ttc} |\n"
        f"| Tier churn rate | {tier_rate} |\n"
    )

    # Top risk drivers (local counterfactual explanations).
    base_score, reasons = mu.local_reasons(pipe, X, baselines, top_n=4)
    if not reasons:
        drivers_md = "### 🔎 Why this score\n✅ No strong risk drivers — this looks like a healthy customer."
    else:
        lines = ["### 🔎 Why this score — top risk drivers"]
        for feat, pts in reasons:
            advice = DRIVER_ADVICE.get(feat, "Review this factor.")
            lines.append(f"- **{feat.replace('_',' ').title()}** · +{pts:.2f} risk points (of 5)  \n"
                         f"  ↳ *Recommended action:* {advice}")
        drivers_md = "\n".join(lines)

    return gauge_html(score, tier, color), prob_dict, summary_md, drivers_md


# ------------------------------ Gradio UI -----------------------------------
FOOTER = (
    f"Model: gradient-boosted classifier (tuned) · 5-fold CV accuracy "
    f"{meta.get('cv_accuracy','?')}, macro-F1 {meta.get('cv_macro_f1','?')}, "
    f"quadratic weighted kappa {meta.get('cv_quadratic_weighted_kappa','?')}."
)

with gr.Blocks(title="Churn Risk Scorer", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🛡️ Customer Churn Risk Scorer")
    gr.Markdown("Enter a customer's details to get their churn risk score, the drivers "
                "behind it, and a recommended retention action.")

    with gr.Row():
        # ----------------------- inputs (left) -----------------------
        with gr.Column(scale=1):
            gr.Markdown("### Customer details")
            age = gr.Slider(10, 80, value=37, step=1, label="Age")
            gender = gr.Dropdown(["F", "M"], value="F", label="Gender")
            region_category = gr.Dropdown(["Town", "City", "Village"], value="Town", label="Region")
            membership_category = gr.Dropdown(
                ["No Membership", "Basic Membership", "Silver Membership",
                 "Gold Membership", "Premium Membership", "Platinum Membership"],
                value="No Membership", label="Membership")
            tenure_years = gr.Slider(0.0, 3.0, value=1.5, step=0.1, label="Years as a customer")
            joined_through_referral = gr.Dropdown(["No", "Yes"], value="No", label="Joined through referral?")
            preferred_offer_types = gr.Dropdown(
                ["Gift Vouchers/Coupons", "Credit/Debit Card Offers", "Without Offers"],
                value="Gift Vouchers/Coupons", label="Preferred offers")
            medium_of_operation = gr.Dropdown(["Desktop", "Smartphone", "Both"], value="Desktop", label="Medium")
            internet_option = gr.Dropdown(["Wi-Fi", "Mobile_Data", "Fiber_Optic"], value="Wi-Fi", label="Internet")
            days_since_last_login = gr.Slider(0, 30, value=12, step=1, label="Days since last login")
            last_visit_hour = gr.Slider(0, 23, value=12, step=1, label="Last visit hour (0-23)")
            avg_time_spent = gr.Number(value=290.0, label="Avg time spent (mins)")
            avg_transaction_value = gr.Number(value=30000.0, label="Avg transaction value")
            avg_frequency_login_days = gr.Number(value=16.0, label="Avg login frequency (days)")
            points_in_wallet = gr.Number(value=700.0, label="Points in wallet")
            used_special_discount = gr.Dropdown(["Yes", "No"], value="Yes", label="Used special discount?")
            offer_application_preference = gr.Dropdown(["Yes", "No"], value="Yes", label="Applies offers?")
            past_complaint = gr.Dropdown(["No", "Yes"], value="No", label="Past complaint?")
            complaint_status = gr.Dropdown(
                ["Not Applicable", "Unsolved", "Solved", "Solved in Follow-up", "No Information Available"],
                value="Not Applicable", label="Complaint status")
            feedback = gr.Dropdown(
                ["Poor Product Quality", "Poor Customer Service", "Poor Website", "Too many ads",
                 "No reason specified", "Reasonable Price", "Quality Customer Care",
                 "Products always in Stock", "User Friendly Website"],
                value="No reason specified", label="Feedback")
            go = gr.Button("Score customer", variant="primary")

        # ----------------------- outputs (right) -----------------------
        with gr.Column(scale=1):
            gr.Markdown("### Churn risk")
            gauge_out = gr.HTML()
            prob_out = gr.Label(label="Probability across risk classes", num_top_classes=5)
            summary_out = gr.Markdown()
            drivers_out = gr.Markdown()

    gr.Markdown(
        "**How to read the output** — Score 1–5: higher = more likely to churn. "
        "Tiers: Low (<2.5) / Medium (2.5–3.5) / High (3.5–4.5) / Critical (≥4.5). "
        "Drivers are the factors pushing *this* customer's risk up. Revenue at risk = "
        "risk fraction × estimated annual value. Time-to-churn comes from the survival model."
    )
    gr.Markdown(FOOTER)

    inputs = [age, gender, region_category, membership_category, tenure_years,
              joined_through_referral, preferred_offer_types, medium_of_operation,
              internet_option, days_since_last_login, last_visit_hour, avg_time_spent,
              avg_transaction_value, avg_frequency_login_days, points_in_wallet,
              used_special_discount, offer_application_preference, past_complaint,
              complaint_status, feedback]
    outputs = [gauge_out, prob_out, summary_out, drivers_out]
    go.click(fn=score_customer, inputs=inputs, outputs=outputs)


if __name__ == "__main__":
    demo.launch()
