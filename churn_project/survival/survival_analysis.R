# ============================================================================
# survival_analysis.R
# Customer churn as a survival-analysis problem (medical-style).
#
# Analogy:
#   patient            -> customer
#   time-to-event      -> tenure_days (how long they've been with us)
#   death/event = 1    -> customer churned (high churn risk, score 4-5)
#   censored = 0       -> still a healthy, low-risk customer at observation end
#
# RUN IT WITH (from the churn_project folder):
#   Rscript survival/survival_analysis.R
#
# Outputs go to survival/outputs/
# ============================================================================

# --- packages ---------------------------------------------------------------
need <- c("survival", "survminer", "ggplot2", "dplyr", "readr")
miss <- need[!sapply(need, requireNamespace, quietly = TRUE)]
if (length(miss) > 0) {
  stop(paste0("Missing R packages: ", paste(miss, collapse = ", "),
              "\nInstall with: install.packages(c(",
              paste(sprintf('\"%s\"', miss), collapse = ", "), "))"))
}
suppressMessages({
  library(survival); library(survminer)
  library(ggplot2);  library(dplyr); library(readr)
})

# --- locate paths relative to this script -----------------------------------
args <- commandArgs(trailingOnly = FALSE)
script_path <- sub("^--file=", "", args[grep("^--file=", args)])
base_dir <- if (length(script_path) > 0) dirname(normalizePath(script_path)) else "survival"
out_dir  <- file.path(base_dir, "outputs")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

data_path <- file.path(base_dir, "survival_data.csv")
df <- read_csv(data_path, show_col_types = FALSE)

# --- light cleaning for modelling -------------------------------------------
df <- df %>%
  mutate(
    membership_category = factor(membership_category),
    gender = factor(gender),
    region_category = factor(region_category),
    feedback_negative = factor(feedback_negative, labels = c("Positive", "Negative")),
    past_complaint = factor(past_complaint, labels = c("No", "Yes"))
  )

sink(file.path(out_dir, "survival_summary.txt"))
cat("=========================================================\n")
cat(" CUSTOMER CHURN SURVIVAL ANALYSIS\n")
cat("=========================================================\n")
cat(sprintf("Customers: %d   Events (churn): %d   Event rate: %.1f%%\n",
            nrow(df), sum(df$event), 100 * mean(df$event)))
cat(sprintf("Tenure days  -- min %d, median %d, max %d\n\n",
            min(df$time), median(df$time), max(df$time)))

# --- 1. Overall Kaplan-Meier survival curve ---------------------------------
surv_obj <- Surv(time = df$time, event = df$event)
km_all <- survfit(surv_obj ~ 1, data = df)
cat("---- 1. Overall Kaplan-Meier ----\n")
print(summary(km_all, times = c(90, 180, 365, 730, 1095)))

p1 <- ggsurvplot(km_all, data = df, conf.int = TRUE,
                 palette = "#2c7fb8", ggtheme = theme_minimal(),
                 title = "Overall customer 'survival' (retention) curve",
                 xlab = "Tenure (days)", ylab = "Probability of still being retained")
ggsave(file.path(out_dir, "km_overall.png"), p1$plot, width = 8, height = 5, dpi = 110)

# --- 2. KM stratified by membership category + log-rank test ----------------
km_mem <- survfit(surv_obj ~ membership_category, data = df)
lr_mem <- survdiff(surv_obj ~ membership_category, data = df)
cat("\n---- 2. Log-rank test: membership_category ----\n")
print(lr_mem)

p2 <- ggsurvplot(km_mem, data = df, conf.int = FALSE, pval = TRUE,
                 legend = "right", ggtheme = theme_minimal(),
                 title = "Retention by membership category",
                 xlab = "Tenure (days)", ylab = "Retention probability")
ggsave(file.path(out_dir, "km_membership.png"), p2$plot, width = 9, height = 5.5, dpi = 110)

# --- 3. KM stratified by feedback sentiment ---------------------------------
km_fb <- survfit(surv_obj ~ feedback_negative, data = df)
cat("\n---- 3. Log-rank test: feedback sentiment ----\n")
print(survdiff(surv_obj ~ feedback_negative, data = df))
p3 <- ggsurvplot(km_fb, data = df, conf.int = TRUE, pval = TRUE,
                 palette = c("#1a9850", "#d73027"), ggtheme = theme_minimal(),
                 title = "Retention by feedback sentiment",
                 xlab = "Tenure (days)", ylab = "Retention probability")
ggsave(file.path(out_dir, "km_feedback.png"), p3$plot, width = 8, height = 5, dpi = 110)

# --- 4. Cox proportional hazards model --------------------------------------
# Hazard ratio > 1  => increases churn hazard; < 1 => protective.
#
# NOTE on stratification: membership_category and feedback_negative each have
# a group with an EXACT 0% or 100% event rate in this data (e.g. Platinum/
# Premium members never churn; positive feedback never churns). A Cox model
# cannot estimate a finite coefficient for a perfectly-separating predictor --
# the partial likelihood is maximised by sending that coefficient to infinity,
# which makes the information matrix singular (this is what crashed cox.zph).
# Their effect is already fully captured above via the Kaplan-Meier curves and
# log-rank tests (sections 2-3), so here we STRATIFY by them instead of
# estimating a coefficient: this lets each combination have its own baseline
# hazard while we estimate clean, finite hazard ratios for the remaining,
# non-deterministic predictors.
cox <- coxph(surv_obj ~ strata(membership_category) + strata(feedback_negative) +
               points_in_wallet + avg_transaction_value +
               avg_frequency_login_days + past_complaint + age + gender,
             data = df)
cat("\n---- 4. Cox proportional hazards model ----\n")
cat("(membership_category and feedback_negative are stratified, not estimated,\n")
cat(" because each has a group with an exact 0%/100% event rate -- see note in script)\n\n")
print(summary(cox))

# Forest plot of hazard ratios
tryCatch({
  pcox <- ggforest(cox, data = as.data.frame(df))
  ggsave(file.path(out_dir, "cox_forest.png"), pcox, width = 9, height = 7, dpi = 110)
}, error = function(e) cat("ggforest skipped:", conditionMessage(e), "\n"))

# --- 5. Proportional hazards assumption check -------------------------------
cat("\n---- 5. Proportional hazards assumption (cox.zph) ----\n")
zph <- cox.zph(cox)
print(zph)

sink()
cat("\nDone. See survival/outputs/ for survival_summary.txt and PNG plots.\n")
