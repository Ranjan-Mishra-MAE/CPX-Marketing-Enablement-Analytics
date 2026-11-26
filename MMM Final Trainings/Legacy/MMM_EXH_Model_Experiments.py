# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# DBTITLE 1,EXH MMM Model Experiments
# MAGIC %md
# MAGIC # EXH MMM — Model Experiments
# MAGIC **Purpose:** Test alternative model specifications before importing the winner into `MMM_EXH_Final_All_Charts_Tables`.
# MAGIC
# MAGIC ### Experiment Plan
# MAGIC 1. **Prior Sensitivity Re-check** — Vary `ALPHA_MU_PRIOR` and `MU_GAMMA_PRIOR` across a grid; track Val MAPE, R², and CI width.
# MAGIC 2. **Weibull Adstock** — Replace geometric decay with Weibull (delayed peak) for Awareness combos; compare fit.
# MAGIC 3. **Time-Varying Coefficients** — Allow `gamma` to drift over time (random-walk or regime-switch); assess whether 2024→2025 shift is captured.
# MAGIC 4. **Expanded Training** — Retrain on train+val (150 weeks), validate on current test (15 weeks); tighten CIs.
# MAGIC
# MAGIC ### Key References
# MAGIC - Source data: `/Volumes/data_exploration_hub/marketing_hub/model_data/Final_Data/EXH_ModelData.xlsx`
# MAGIC - Baseline artifacts: `/Volumes/data_exploration_hub/marketing_hub/model_data/Final_Data/MMM_EXH_Outcomes_v2/`
# MAGIC - Current model: 4 chains × 2000 draws, target_accept=0.99, geometric adstock, Hill (Consideration) / exponential (Awareness, Conversion)
# MAGIC - Current accuracy: Train MAPE 19.5%, Val MAPE 23.2%, Train R² 0.630, Val R² 0.481

# COMMAND ----------

# DBTITLE 1,Install & Imports
# MAGIC %pip install pymc arviz numpyro jax jaxlib pytensor scikit-learn openpyxl netcdf4 h5netcdf scipy xarray --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Imports & Data Loading
import os, pickle, warnings
import numpy as np
import pandas as pd
import pytensor
import pytensor.tensor as pt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error, r2_score
import pymc as pm
import arviz as az
import pymc.sampling.jax as pmjax
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=FutureWarning)

# ── Paths ─────────────────────────────────────────────────────────────────────
EXH_FILE_PATH     = "/Volumes/data_exploration_hub/marketing_hub/model_data/Final_Data/EXH_ModelData.xlsx"
EXH_ARTIFACT_PATH = "/Volumes/data_exploration_hub/marketing_hub/model_data/Final_Data/MMM_EXH_Outcomes_v2/"
EXH_EXPERIMENT_PATH = "/Volumes/data_exploration_hub/marketing_hub/model_data/Final_Data/MMM_EXH_Experiments/"
os.makedirs(EXH_EXPERIMENT_PATH, exist_ok=True)

print(f"PyMC  : {pm.__version__}")
print(f"ArviZ : {az.__version__}")

# COMMAND ----------

# DBTITLE 1,Load Data & Reproduce Current Split
# ── Load & engineer features (identical to main notebook) ─────────────────────
df = pd.read_excel(EXH_FILE_PATH, sheet_name=0)
df = df.sort_values("week_date").reset_index(drop=True)
df["stat_holiday_in_week"] = (df["stat_holiday_in_week"] == "Y").astype(int)
df["is_long_weekend"]      = (df["is_long_weekend"]      == "Y").astype(int)
for s in ["spring", "summer", "fall"]:
    df[f"season_{s}"] = (df["season"] == s).astype(int)
df["is_2024"]    = (df["week_date"].dt.year == 2024).astype(int)
df["is_2025"]    = (df["week_date"].dt.year >= 2025).astype(int)
df["week_index"] = np.arange(len(df)) / len(df)

target_col = "total_weekly_attendance"
date_col   = "week_date"

# Current split
n       = len(df)
n_test  = 15
n_val   = 26
n_train = n - n_val - n_test

train_df = df.iloc[:n_train].copy()
val_df   = df.iloc[n_train:n_train+n_val].copy()
test_df  = df.iloc[n_train+n_val:].copy()

print(f"Data: {n} weeks | Train: {n_train} | Val: {n_val} | Test: {n_test}")
print(f"Date range: {df[date_col].min().date()} → {df[date_col].max().date()}")
print(f"\nExpanded split (for Experiment 4):")
print(f"  New Train: {n_train + n_val} weeks (current train+val)")
print(f"  New Val:   {n_test} weeks (current test)")

# COMMAND ----------

# DBTITLE 1,Load Current Model Artifacts (Baseline Reference)
# Load baseline model artifacts for comparison
with open(f"{EXH_ARTIFACT_PATH}response_params.pkl", "rb") as f:
    response_params_baseline = pickle.load(f)
with open(f"{EXH_ARTIFACT_PATH}model_meta.pkl", "rb") as f:
    model_meta = pickle.load(f)

ACTIVE_COMBOS     = model_meta["ACTIVE_COMBOS"]
FUNNEL_COMBOS     = model_meta["FUNNEL_COMBOS"]
LOB_GROUP_MAP     = model_meta["LOB_GROUP_MAP"]
baseline_features = model_meta["baseline_features"]

print(f"Active combos: {len(ACTIVE_COMBOS)}")
print(f"Baseline features: {len(baseline_features)}")
print(f"\nCurrent priors:")
print(f"  ALPHA_MU_PRIOR:")
print(f"    Awareness    : Beta(7,2) → mean 0.78")
print(f"    Consideration: Beta(3,3) → mean 0.50")
print(f"    Conversion   : Beta(2,5) → mean 0.29")
print(f"  MU_GAMMA_PRIOR:")
print(f"    Awareness=0.5, Consideration=0.3, Conversion=0.6")
print(f"\nResponse params (current baseline):")
for c, p in response_params_baseline.items():
    print(f"  {c:<40} α={p['alpha']:.3f}  γ={p['gamma']:.3f}  sat={p['saturation']}")

# COMMAND ----------

# DBTITLE 1,Feature Selection, Combo Aggregation & Scaling
# ━━ FEATURE SELECTION, COMBO AGGREGATION & SCALING ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Reproduces the main notebook's cells 7–12 setup.

# ── Feature Selection ─────────────────────────────────────────────────────────
DROP_FEATURES = [
    "national_max_temp", "national_min_temp", "national_temp_anomaly",
    "action_intensity", "adaptation_intensity", "adventure_intensity",
    "animated_intensity", "biopic_intensity", "comedy_intensity",
    "crime_intensity", "documentary_intensity", "drama_intensity",
    "family_intensity", "fantasy_intensity", "foreign_intensity",
    "horror_intensity", "musical_intensity", "mystery_intensity",
    "period_intensity", "romance_intensity", "sci_fi_intensity",
    "sequel_intensity", "superhero_intensity", "thriller_intensity",
    "war_intensity", "western_intensity",
]

# Baseline features = all numeric columns except target, date, media, and dropped
all_cols    = df.columns.tolist()
target_idx  = all_cols.index(target_col)
media_start = all_cols.index("Awareness_Cineclub_Digital")
media_cols  = all_cols[media_start:target_idx]

# Use baseline_features from model_meta, filtering to columns that exist in df
# (some engineered cols in the main notebook may not be reproduced here)
baseline_features = [c for c in baseline_features if c in df.columns]
print(f"Baseline features (from model_meta, filtered to existing): {len(baseline_features)}")
print(f"  {baseline_features}")

# ── LOB Grouping & Combo Aggregation ───────────────────────────────────────
LOB_GROUPS = {
    "Premium_Brand" : ["Premium Experiences - AVX", "Premium Experiences - VIP", "F&B", "EXH - Brand"],
    "Film_Marketing": ["Premium Experiences - IMAX", "Film Marketing - Inform", "Performance - Convert & Upgrade"],
    "Prepaid"       : ["Prepaid - B2B", "Prepaid - B2C"],
    "Cineclub"      : ["Cineclub"],
    "GE"            : ["G&E"],
}

FUNNELS = ["Awareness", "Consideration", "Conversion"]

def parse_media_col(col):
    parts = col.split("_")
    funnel = parts[0].capitalize()
    channel = parts[-1]
    lob = "_".join(parts[1:-1])
    return funnel, lob, channel

media_meta = pd.DataFrame(
    [parse_media_col(c) + (c,) for c in media_cols],
    columns=["funnel", "lob", "channel", "col_name"],
)

# Create funnel x LOB group columns
for funnel in FUNNELS:
    for lob_group, lobs in LOB_GROUPS.items():
        col_name = f"{funnel}__{lob_group}"
        matching = media_meta.loc[
            (media_meta["funnel"] == funnel) & (media_meta["lob"].isin(lobs)),
            "col_name",
        ].tolist()
        df[col_name] = df[matching].sum(axis=1) if matching else 0.0

# ITER2 combo mapping & aggregation
ITER2_COMBOS = {
    "Awareness__Premium_Brand"     : ["Awareness__Premium_Brand"],
    "Awareness__Other"             : ["Awareness__Cineclub", "Awareness__Film_Marketing"],
    "Consideration__Premium_Brand" : ["Consideration__Premium_Brand"],
    "Consideration__GE"            : ["Consideration__GE"],
    "Consideration__Prepaid"       : ["Consideration__Prepaid"],
    "Consideration__Other"         : ["Consideration__Film_Marketing", "Consideration__Cineclub"],
    "Conversion__Film_Marketing"   : ["Conversion__Film_Marketing"],
    "Conversion__Premium_Brand"    : ["Conversion__Premium_Brand"],
    "Conversion__Cineclub"         : ["Conversion__Cineclub"],
    "Conversion__Other"            : ["Conversion__Prepaid", "Conversion__GE"],
}

DROP_SPARSE = ["Awareness__Other", "Consideration__Other", "Conversion__Other"]

for combo_name, source_cols in ITER2_COMBOS.items():
    df[combo_name] = df[source_cols].sum(axis=1)

ACTIVE_COMBOS = [c for c in ITER2_COMBOS if c not in DROP_SPARSE]
FUNNEL_COMBOS = {
    funnel: [c for c in ACTIVE_COMBOS if c.startswith(funnel)]
    for funnel in FUNNELS
}

print(f"Active combos: {len(ACTIVE_COMBOS)}")
for f, combos in FUNNEL_COMBOS.items():
    print(f"  {f}: {combos}")

# ── Scaling ─────────────────────────────────────────────────────────────────
train_df = df.iloc[:n_train].copy()
val_df   = df.iloc[n_train:n_train+n_val].copy()
test_df  = df.iloc[n_train+n_val:].copy()

scaler_X = StandardScaler()
X_train  = scaler_X.fit_transform(train_df[baseline_features].values)
X_val    = scaler_X.transform(val_df[baseline_features].values)
X_test   = scaler_X.transform(test_df[baseline_features].values)

scaler_y = StandardScaler()
y_train  = scaler_y.fit_transform(train_df[[target_col]].values).ravel()
y_val    = scaler_y.transform(val_df[[target_col]].values).ravel()
y_test   = scaler_y.transform(test_df[[target_col]].values).ravel()

scale_factor = float(scaler_y.scale_[0])
print(f"\ny scale factor: {scale_factor:,.0f}")

# ── Media Spend Scaling (0–1 per combo, using train max) ────────────────────
spend_scaled = {}
spend_train_max = {}

for combo in ACTIVE_COMBOS:
    m = float(train_df[combo].max())
    spend_train_max[combo] = m
    spend_scaled[combo] = {
        "train":    train_df[combo].values.astype(np.float64) / m,
        "val":      val_df[combo].values.astype(np.float64) / m,
        "test":     test_df[combo].values.astype(np.float64) / m,
        "trainval": np.concatenate([train_df[combo].values, val_df[combo].values]).astype(np.float64) / m,
    }

print(f"Spend scaled for {len(spend_scaled)} combos")
print("\nSetup complete — ready for experiments.")

# COMMAND ----------

# DBTITLE 1,Experiment 1 — Prior Sensitivity
# MAGIC %md
# MAGIC ---
# MAGIC ## Experiment 1: Prior Sensitivity Analysis
# MAGIC Vary alpha and gamma priors across a grid. For each configuration, train with fewer draws (500 tune + 500 draw, 2 chains) and compare Val MAPE / R² / CI width.

# COMMAND ----------

# DBTITLE 1,Prior Sensitivity Grid
# ── Prior grid to test ────────────────────────────────────────────────────────
# Each entry: (label, ALPHA_MU_PRIOR, MU_GAMMA_PRIOR)
PRIOR_GRID = [
    ("baseline", 
     {"Awareness": {"alpha":7,"beta":2}, "Consideration":{"alpha":3,"beta":3}, "Conversion":{"alpha":2,"beta":5}},
     {"Awareness":0.5, "Consideration":0.3, "Conversion":0.6}),
    ("flatter_alpha",
     {"Awareness": {"alpha":4,"beta":2}, "Consideration":{"alpha":2,"beta":2}, "Conversion":{"alpha":2,"beta":3}},
     {"Awareness":0.5, "Consideration":0.3, "Conversion":0.6}),
    ("tighter_gamma",
     {"Awareness": {"alpha":7,"beta":2}, "Consideration":{"alpha":3,"beta":3}, "Conversion":{"alpha":2,"beta":5}},
     {"Awareness":0.3, "Consideration":0.2, "Conversion":0.4}),
    ("wider_gamma",
     {"Awareness": {"alpha":7,"beta":2}, "Consideration":{"alpha":3,"beta":3}, "Conversion":{"alpha":2,"beta":5}},
     {"Awareness":0.8, "Consideration":0.5, "Conversion":1.0}),
]

print(f"Prior configurations to test: {len(PRIOR_GRID)}")
for label, ap, gp in PRIOR_GRID:
    print(f"  {label}:")
    for f in ['Awareness','Consideration','Conversion']:
        a, b = ap[f]['alpha'], ap[f]['beta']
        print(f"    {f}: α~Beta({a},{b}) mean={a/(a+b):.2f}  γ~HalfNorm(σ={gp[f]})")

# COMMAND ----------

# DBTITLE 1,Experiment 1 — Run Prior Sensitivity Grid
# ━━ EXPERIMENT 1: PRIOR SENSITIVITY — Full Model Loop ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Quick screening: 2 chains × 500 draws + 500 tune per configuration.
# Evaluates Val MAPE, R², and CI width for +10% budget scenario.

import time

# ── Response transformation functions (PyTensor) ───────────────────────────
def geometric_adstock(x, alpha):
    x_t = pt.as_tensor_variable(x.astype(np.float64))
    def step(x_curr, adstock_prev, alpha):
        return x_curr + alpha * adstock_prev
    adstocked, _ = pytensor.scan(
        fn=step, sequences=[x_t],
        outputs_info=[pt.as_tensor_variable(np.float64(0.0))],
        non_sequences=[alpha], strict=True,
    )
    return adstocked

def hill_saturation(x, k, alpha_s):
    x_safe = pt.maximum(x, 1e-6)
    return (x_safe ** alpha_s) / (x_safe ** alpha_s + k ** alpha_s)

def logistic_saturation(x, lam):
    return 1 - pt.exp(-lam * x)

# ── Model builder parameterized by priors ─────────────────────────────────
def build_model(alpha_mu_prior, mu_gamma_prior, X_obs, y_obs, spend_data):
    """
    Build the mixed-hierarchical MMM model with given priors.
    spend_data: dict of combo -> array (scaled 0-1 for training period)
    """
    with pm.Model() as model:
        intercept = pm.Normal("intercept", mu=0, sigma=1)
        beta      = pm.Normal("beta", mu=0, sigma=1, shape=X_obs.shape[1])
        sigma     = pm.HalfNormal("sigma", sigma=1)

        mu = intercept + pm.math.dot(X_obs, beta)

        for funnel, combos in FUNNEL_COMBOS.items():
            ap = alpha_mu_prior[funnel]

            # Funnel-level hyperpriors
            mu_alpha_f    = pm.Beta(f"mu_alpha_{funnel}", alpha=ap["alpha"], beta=ap["beta"])
            kappa_alpha_f = pm.HalfNormal(f"kappa_alpha_{funnel}", sigma=2)
            mu_gamma_f    = pm.HalfNormal(f"mu_gamma_{funnel}", sigma=mu_gamma_prior[funnel])

            if funnel == "Consideration":
                mu_k_f       = pm.HalfNormal("mu_k_Consideration", sigma=0.5)
                mu_alpha_s_f = pm.HalfNormal("mu_alpha_s_Consideration", sigma=1.5)

            for combo in combos:
                alpha_c = pm.Beta(
                    f"alpha_{combo}",
                    alpha=mu_alpha_f * kappa_alpha_f,
                    beta=(1 - mu_alpha_f) * kappa_alpha_f,
                )
                gamma_c = pm.HalfNormal(f"gamma_{combo}", sigma=mu_gamma_f)
                adstocked = geometric_adstock(spend_data[combo], alpha_c)

                if funnel == "Consideration":
                    k_c       = pm.HalfNormal(f"k_{combo}", sigma=mu_k_f)
                    alpha_s_c = pm.HalfNormal(f"alpha_s_{combo}", sigma=mu_alpha_s_f)
                    saturated = hill_saturation(adstocked, k_c, alpha_s_c)
                else:
                    lam_c = pm.HalfNormal(f"lam_{combo}", sigma=1)
                    saturated = logistic_saturation(adstocked, lam_c)

                mu = mu + gamma_c * saturated

        pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y_obs)
    return model

# ── Prediction function (posterior mean) ─────────────────────────────────
def predict_attendance(trace, X_data, spend_data_period, n_obs):
    """
    Compute point predictions using posterior means.
    Returns predictions in ORIGINAL scale (attendance).
    """
    post = trace.posterior
    intercept_mean = float(post["intercept"].values.ravel().mean())
    beta_mean = post["beta"].values.reshape(-1, len(baseline_features)).mean(axis=0)

    mu_pred = intercept_mean + X_data @ beta_mean

    for funnel, combos in FUNNEL_COMBOS.items():
        for combo in combos:
            alpha_mean = float(post[f"alpha_{combo}"].values.ravel().mean())
            gamma_mean = float(post[f"gamma_{combo}"].values.ravel().mean())
            x_spend = spend_data_period[combo]

            # Numpy adstock
            adstocked = np.zeros(len(x_spend))
            adstocked[0] = x_spend[0]
            for t in range(1, len(x_spend)):
                adstocked[t] = x_spend[t] + alpha_mean * adstocked[t - 1]

            # Saturation
            if funnel == "Consideration":
                k_mean  = float(post[f"k_{combo}"].values.ravel().mean())
                as_mean = float(post[f"alpha_s_{combo}"].values.ravel().mean())
                x_safe  = np.maximum(adstocked, 1e-6)
                sat = (x_safe ** as_mean) / (x_safe ** as_mean + k_mean ** as_mean)
            else:
                lam_mean = float(post[f"lam_{combo}"].values.ravel().mean())
                sat = 1 - np.exp(-lam_mean * adstocked)

            mu_pred += gamma_mean * sat

    # Inverse-transform to original scale
    pred_original = mu_pred * scale_factor + scaler_y.mean_[0]
    return pred_original

# ── CI Width for +10% budget scenario ─────────────────────────────────────
def compute_ci_width(trace, ci=0.89):
    """
    Compute CI width for a +10% proportional budget uplift.
    Uses posterior draws of gamma/alpha/saturation params.
    Returns: (point_lift_pct, ci_width_pct)
    """
    post = trace.posterior
    n_draws = min(200, post.dims["draw"] * post.dims["chain"])
    all_draws = np.random.choice(
        post.dims["draw"] * post.dims["chain"], size=n_draws, replace=False
    )

    def combo_contrib_draws(combo, annual_spend, draws_idx):
        """Vectorized contribution across posterior draws."""
        alpha_arr = post[f"alpha_{combo}"].values.ravel()[draws_idx]
        gamma_arr = post[f"gamma_{combo}"].values.ravel()[draws_idx]
        alpha_arr = np.minimum(alpha_arr, 0.999)
        mx = spend_train_max[combo]
        s = (annual_spend / 52) / mx
        ad = s / (1 - alpha_arr)

        funnel = combo.split("__")[0]
        if funnel == "Consideration":
            k_arr  = post[f"k_{combo}"].values.ravel()[draws_idx]
            as_arr = post[f"alpha_s_{combo}"].values.ravel()[draws_idx]
            sat = (ad ** as_arr) / (ad ** as_arr + k_arr ** as_arr)
        else:
            lam_arr = post[f"lam_{combo}"].values.ravel()[draws_idx]
            sat = 1 - np.exp(-lam_arr * ad)

        return gamma_arr * sat * scale_factor * 52

    # Baseline spend (2025 annual from train data)
    baseline_spend = {}
    _df_2025 = df[df["week_date"].dt.year == 2025]
    for combo in ACTIVE_COMBOS:
        baseline_spend[combo] = float(_df_2025[combo].sum())

    # Compute baseline and +10% draws
    base_draws = np.zeros(n_draws)
    scen_draws = np.zeros(n_draws)
    for combo in ACTIVE_COMBOS:
        base_draws += combo_contrib_draws(combo, baseline_spend[combo], all_draws)
        scen_draws += combo_contrib_draws(combo, baseline_spend[combo] * 1.1, all_draws)

    lift_pct_draws = (scen_draws - base_draws) / base_draws * 100
    lo_q, hi_q = (1 - ci) / 2, 1 - (1 - ci) / 2
    ci_lo = float(np.quantile(lift_pct_draws, lo_q))
    ci_hi = float(np.quantile(lift_pct_draws, hi_q))
    point_pct = float(np.median(lift_pct_draws))

    return point_pct, ci_hi - ci_lo

# ━━ RUN THE GRID ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
results = []

for label, alpha_prior, gamma_prior in PRIOR_GRID:
    print(f"\n{'='*70}")
    print(f"  CONFIG: {label}")
    print(f"{'='*70}")

    t0 = time.time()

    # Build model on training data
    spend_train_data = {c: spend_scaled[c]["train"] for c in ACTIVE_COMBOS}
    model = build_model(alpha_prior, gamma_prior, X_train, y_train, spend_train_data)

    # Sample (quick: 2 chains, 500+500)
    with model:
        trace = pmjax.sample_numpyro_nuts(
            draws=500, tune=500, chains=2,
            target_accept=0.95, random_seed=42,
        )

    elapsed = time.time() - t0
    print(f"  Sampled in {elapsed:.0f}s")

    # ── Convergence quick check ───────────────────────────────────────────
    summ = az.summary(trace, var_names=["intercept", "sigma"])
    max_rhat = float(summ["r_hat"].max())
    divergences = int(trace.sample_stats.diverging.sum().item())
    print(f"  R-hat (intercept/sigma): {max_rhat:.3f} | Divergences: {divergences}")

    # ── Predict on train+val ───────────────────────────────────────────────
    X_trainval = np.vstack([X_train, X_val])
    spend_trainval = {c: spend_scaled[c]["trainval"] for c in ACTIVE_COMBOS}
    pred_all = predict_attendance(trace, X_trainval, spend_trainval, n_train + n_val)

    actual_train = train_df[target_col].values
    actual_val   = val_df[target_col].values
    pred_train   = pred_all[:n_train]
    pred_val     = pred_all[n_train:]

    mape_train = float(np.mean(np.abs(actual_train - pred_train) / actual_train) * 100)
    mape_val   = float(np.mean(np.abs(actual_val - pred_val) / actual_val) * 100)
    r2_train   = float(1 - ((actual_train - pred_train)**2).sum() / ((actual_train - actual_train.mean())**2).sum())
    r2_val     = float(1 - ((actual_val - pred_val)**2).sum() / ((actual_val - actual_val.mean())**2).sum())

    print(f"  Train MAPE: {mape_train:.1f}%  R²: {r2_train:.3f}")
    print(f"  Val   MAPE: {mape_val:.1f}%  R²: {r2_val:.3f}")

    # ── CI width for +10% scenario ────────────────────────────────────────
    point_lift, ci_width = compute_ci_width(trace)
    print(f"  +10% scenario: lift {point_lift:.1f}% | 89% CI width: {ci_width:.1f}pp")

    results.append({
        "Config": label,
        "Train MAPE": f"{mape_train:.1f}%",
        "Val MAPE": f"{mape_val:.1f}%",
        "Train R²": f"{r2_train:.3f}",
        "Val R²": f"{r2_val:.3f}",
        "+10% Lift": f"{point_lift:.1f}%",
        "CI Width": f"{ci_width:.1f}pp",
        "Divergences": divergences,
        "Time (s)": f"{elapsed:.0f}",
    })

# ── Results Table ───────────────────────────────────────────────────────────
results_df = pd.DataFrame(results)
print(f"\n{'='*70}")
print("  PRIOR SENSITIVITY RESULTS COMPARISON")
print(f"{'='*70}")
display(results_df)

# Save for later reference
results_df.to_csv(f"{EXH_EXPERIMENT_PATH}prior_sensitivity_results.csv", index=False)
print(f"\nResults saved → {EXH_EXPERIMENT_PATH}prior_sensitivity_results.csv")

# COMMAND ----------

# DBTITLE 1,Experiment 2 — Weibull Adstock
# MAGIC %md
# MAGIC ---
# MAGIC ## Experiment 2: Weibull Adstock
# MAGIC Replace geometric decay (`adstocked[t] = x[t] + α·adstocked[t-1]`) with Weibull PDF kernel that allows a delayed peak:
# MAGIC
# MAGIC ```
# MAGIC weights[k] = (shape/scale) * (k/scale)^(shape-1) * exp(-(k/scale)^shape)
# MAGIC adstocked[t] = Σ_{k=0}^{L} weights[k] * x[t-k]
# MAGIC ```
# MAGIC
# MAGIC Test on Awareness combos (where brand campaigns likely have delayed peak response). Compare MAPE, posterior gamma, and contribution estimates.

# COMMAND ----------

# DBTITLE 1,Experiment 2 — Weibull Adstock Implementation
# ━━ EXPERIMENT 2: WEIBULL ADSTOCK ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Replace geometric decay with Weibull kernel for selected funnels.
#
# Weibull weight formula (unnormalized):
#   w(k) = (k)^(shape-1) * exp(-(k/scale)^shape)   for k = 1..L
#   w(0) = 1.0 (instantaneous contribution)
#   Then normalize: weights = w / sum(w)
#
# Properties:
#   shape = 1 → exponential decay (≈ geometric)
#   shape > 1 → delayed peak at k ≈ scale * ((shape-1)/shape)^(1/shape)
#   shape < 1 → heavy-tail, front-loaded
#
# For Awareness (brand), expect shape > 1: campaigns build over 2-4 weeks.
# For Conversion (performance), expect shape ≈ 1: immediate response.

MAX_LAG = 13  # 13-week maximum carryover window

# ── Precompute lag matrices (numpy, fixed data) ──────────────────────────────
def make_lag_matrix(x, L):
    """Create (T, L+1) lag matrix. Column k = x shifted by k weeks."""
    T = len(x)
    x_padded = np.concatenate([np.zeros(L), x])
    lag_mat = np.zeros((T, L + 1))
    for k in range(L + 1):
        lag_mat[:, k] = x_padded[L - k : L - k + T]
    return lag_mat

# Build lag matrices for all combos and splits
lag_matrices = {}
for combo in ACTIVE_COMBOS:
    lag_matrices[combo] = {
        "train":    make_lag_matrix(spend_scaled[combo]["train"], MAX_LAG),
        "val":      make_lag_matrix(spend_scaled[combo]["val"], MAX_LAG),
        "trainval": make_lag_matrix(spend_scaled[combo]["trainval"], MAX_LAG),
    }

print(f"Lag matrices built: {len(lag_matrices)} combos × {MAX_LAG+1} lags")
print(f"Shape (train): {lag_matrices[ACTIVE_COMBOS[0]]['train'].shape}")

# ── Weibull adstock function (PyTensor, no scan needed!) ─────────────────────
def weibull_adstock_pt(lag_matrix_np, shape, scale, L=MAX_LAG):
    """
    Compute Weibull-adstocked signal using precomputed lag matrix.
    lag_matrix_np: (T, L+1) numpy array
    shape, scale: PyTensor scalar random variables
    Returns: (T,) symbolic adstocked signal
    """
    # Lag indices: 0, 1, 2, ..., L
    lags = pt.arange(0, L + 1).astype("float64")
    
    # Weights: w(0)=1, w(k) = k^(shape-1) * exp(-(k/scale)^shape) for k>=1
    # Use lags+0.5 to smoothly handle k=0 (avoids 0^(shape-1) singularity)
    lags_shifted = lags + 0.5
    raw_weights = (lags_shifted ** (shape - 1)) * pt.exp(-((lags_shifted / scale) ** shape))
    
    # Normalize to sum to 1
    weights = raw_weights / raw_weights.sum()
    
    # Convolve: (T, L+1) @ (L+1,) → (T,)
    lag_mat_t = pt.as_tensor_variable(lag_matrix_np)
    return pt.dot(lag_mat_t, weights)

# ── Numpy equivalent for prediction ─────────────────────────────────────────
def weibull_adstock_np(lag_matrix_np, shape_val, scale_val, L=MAX_LAG):
    """Numpy version for posterior-mean predictions."""
    lags = np.arange(0, L + 1).astype(np.float64)
    lags_shifted = lags + 0.5
    raw_weights = (lags_shifted ** (shape_val - 1)) * np.exp(-((lags_shifted / scale_val) ** shape_val))
    weights = raw_weights / raw_weights.sum()
    return lag_matrix_np @ weights

print("\n── Weibull weight profiles for reference ──")
for s_val, sc_val in [(1.0, 3.0), (1.5, 3.0), (2.0, 4.0), (2.5, 5.0)]:
    lags = np.arange(0, MAX_LAG + 1) + 0.5
    w = (lags ** (s_val - 1)) * np.exp(-((lags / sc_val) ** s_val))
    w /= w.sum()
    peak_lag = np.argmax(w)
    print(f"  shape={s_val:.1f}, scale={sc_val:.1f} → peak at lag {peak_lag}, "
          f"top-3 weights: {w[:6].round(3)}")

print("\nReady for Weibull model build.")

# COMMAND ----------

# DBTITLE 1,Experiment 2 — Weibull Model Build & Sample
import time

# ── Re-define shared response functions (in case cell 9 kernel state is lost) ──
def geometric_adstock(x, alpha):
    x_t = pt.as_tensor_variable(x.astype(np.float64))
    def step(x_curr, adstock_prev, alpha):
        return x_curr + alpha * adstock_prev
    adstocked, _ = pytensor.scan(
        fn=step, sequences=[x_t],
        outputs_info=[pt.as_tensor_variable(np.float64(0.0))],
        non_sequences=[alpha], strict=True,
    )
    return adstocked

def hill_saturation(x, k, alpha_s):
    x_safe = pt.maximum(x, 1e-6)
    return (x_safe ** alpha_s) / (x_safe ** alpha_s + k ** alpha_s)

def logistic_saturation(x, lam):
    return 1 - pt.exp(-lam * x)

# ━━ BUILD WEIBULL MODEL ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Configuration:
#   - Awareness: Weibull adstock (test delayed peak hypothesis)
#   - Consideration: Weibull adstock (may also have delayed peak via brand consideration)
#   - Conversion: Geometric adstock (performance media = immediate response)

def build_weibull_model(X_obs, y_obs, spend_lag_mats, spend_data_geo):
    """
    Hybrid model: Weibull for Awareness+Consideration, Geometric for Conversion.
    spend_lag_mats: dict combo -> (T, L+1) lag matrix (for Weibull combos)
    spend_data_geo: dict combo -> (T,) scaled spend (for Geometric combos)
    """
    with pm.Model() as model:
        intercept = pm.Normal("intercept", mu=0, sigma=1)
        beta      = pm.Normal("beta", mu=0, sigma=1, shape=X_obs.shape[1])
        sigma     = pm.HalfNormal("sigma", sigma=1)

        mu = intercept + pm.math.dot(X_obs, beta)

        # ─── AWARENESS: Weibull adstock + logistic saturation ───────────────
        # Hyperpriors
        mu_shape_aw = pm.HalfNormal("mu_shape_Awareness", sigma=1.5)  # expect >1
        mu_scale_aw = pm.HalfNormal("mu_scale_Awareness", sigma=4.0)  # weeks to peak
        mu_gamma_aw = pm.HalfNormal("mu_gamma_Awareness", sigma=0.5)

        for combo in FUNNEL_COMBOS["Awareness"]:
            shape_c = pm.HalfNormal(f"shape_{combo}", sigma=mu_shape_aw)
            scale_c = pm.HalfNormal(f"scale_{combo}", sigma=mu_scale_aw)
            gamma_c = pm.HalfNormal(f"gamma_{combo}", sigma=mu_gamma_aw)
            lam_c   = pm.HalfNormal(f"lam_{combo}", sigma=1)

            adstocked = weibull_adstock_pt(spend_lag_mats[combo], shape_c, scale_c)
            saturated = logistic_saturation(adstocked, lam_c)
            mu = mu + gamma_c * saturated

        # ─── CONSIDERATION: Weibull adstock + Hill saturation ──────────────
        mu_shape_co = pm.HalfNormal("mu_shape_Consideration", sigma=1.0)
        mu_scale_co = pm.HalfNormal("mu_scale_Consideration", sigma=3.0)
        mu_gamma_co = pm.HalfNormal("mu_gamma_Consideration", sigma=0.3)
        mu_k_co     = pm.HalfNormal("mu_k_Consideration", sigma=0.5)
        mu_alpha_s_co = pm.HalfNormal("mu_alpha_s_Consideration", sigma=1.5)

        for combo in FUNNEL_COMBOS["Consideration"]:
            shape_c   = pm.HalfNormal(f"shape_{combo}", sigma=mu_shape_co)
            scale_c   = pm.HalfNormal(f"scale_{combo}", sigma=mu_scale_co)
            gamma_c   = pm.HalfNormal(f"gamma_{combo}", sigma=mu_gamma_co)
            k_c       = pm.HalfNormal(f"k_{combo}", sigma=mu_k_co)
            alpha_s_c = pm.HalfNormal(f"alpha_s_{combo}", sigma=mu_alpha_s_co)

            adstocked = weibull_adstock_pt(spend_lag_mats[combo], shape_c, scale_c)
            saturated = hill_saturation(adstocked, k_c, alpha_s_c)
            mu = mu + gamma_c * saturated

        # ─── CONVERSION: Geometric adstock + logistic saturation ───────────
        # Keep geometric for Conversion (immediate response hypothesis)
        mu_alpha_cv = pm.Beta("mu_alpha_Conversion", alpha=2, beta=5)  # mean 0.29
        kappa_cv    = pm.HalfNormal("kappa_alpha_Conversion", sigma=2)
        mu_gamma_cv = pm.HalfNormal("mu_gamma_Conversion", sigma=0.6)

        for combo in FUNNEL_COMBOS["Conversion"]:
            alpha_c = pm.Beta(
                f"alpha_{combo}",
                alpha=mu_alpha_cv * kappa_cv,
                beta=(1 - mu_alpha_cv) * kappa_cv,
            )
            gamma_c = pm.HalfNormal(f"gamma_{combo}", sigma=mu_gamma_cv)
            lam_c   = pm.HalfNormal(f"lam_{combo}", sigma=1)

            adstocked = geometric_adstock(spend_data_geo[combo], alpha_c)
            saturated = logistic_saturation(adstocked, lam_c)
            mu = mu + gamma_c * saturated

        pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y_obs)
    return model

# ── Prepare inputs ───────────────────────────────────────────────────────────
spend_lag_train = {c: lag_matrices[c]["train"] for c in ACTIVE_COMBOS
                   if not c.startswith("Conversion")}
spend_geo_train = {c: spend_scaled[c]["train"] for c in FUNNEL_COMBOS["Conversion"]}

print("Building Weibull hybrid model...")
t0 = time.time()
weibull_model = build_weibull_model(X_train, y_train, spend_lag_train, spend_geo_train)
print(f"Model built in {time.time()-t0:.1f}s")
print(f"Free parameters: {weibull_model.rvs_to_values.keys().__len__()}")

# ── Sample (quick: 2 chains × 500) ──────────────────────────────────────────
print("\nSampling (2 chains × 500 draws, 500 tune)...")
t0 = time.time()
with weibull_model:
    trace_weibull = pmjax.sample_numpyro_nuts(
        draws=500, tune=500, chains=2,
        target_accept=0.95, random_seed=123,
    )
elapsed_wb = time.time() - t0
print(f"Sampled in {elapsed_wb:.0f}s")

# ── Convergence ──────────────────────────────────────────────────────────────
divergences_wb = int(trace_weibull.sample_stats.diverging.sum().item())
summ_wb = az.summary(trace_weibull, var_names=["intercept", "sigma"])
max_rhat_wb = float(summ_wb["r_hat"].max())
print(f"R-hat (intercept/sigma): {max_rhat_wb:.3f} | Divergences: {divergences_wb}")

# ── Inspect Weibull shape/scale posteriors ───────────────────────────────────
print("\n── Weibull Shape & Scale Posteriors ──")
for combo in ACTIVE_COMBOS:
    if combo.startswith("Conversion"):
        continue
    post = trace_weibull.posterior
    shape_vals = post[f"shape_{combo}"].values.ravel()
    scale_vals = post[f"scale_{combo}"].values.ravel()
    peak_lag = scale_vals * ((shape_vals - 1) / shape_vals) ** (1 / shape_vals)
    peak_lag = np.where(shape_vals > 1, peak_lag, 0)  # peak=0 if shape<=1
    print(f"  {combo}:")
    print(f"    shape: {shape_vals.mean():.2f} ± {shape_vals.std():.2f}  "
          f"(median {np.median(shape_vals):.2f})")
    print(f"    scale: {scale_vals.mean():.2f} ± {scale_vals.std():.2f}")
    print(f"    implied peak lag: {peak_lag.mean():.1f} weeks "
          f"(95% of draws: {np.percentile(peak_lag, 5):.1f}–{np.percentile(peak_lag, 95):.1f})")

# COMMAND ----------

# DBTITLE 1,Experiment 2 — Weibull vs Geometric Comparison
# ━━ PREDICT & COMPARE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def predict_weibull(trace, X_data, lag_mats_period, spend_geo_period):
    """
    Predict using Weibull model posteriors.
    lag_mats_period: dict combo -> (T, L+1) lag matrix
    spend_geo_period: dict combo -> (T,) spend for geometric combos
    """
    post = trace.posterior
    intercept_mean = float(post["intercept"].values.ravel().mean())
    beta_mean = post["beta"].values.reshape(-1, len(baseline_features)).mean(axis=0)

    mu_pred = intercept_mean + X_data @ beta_mean

    # Awareness + Consideration: Weibull
    for funnel in ["Awareness", "Consideration"]:
        for combo in FUNNEL_COMBOS[funnel]:
            shape_mean = float(post[f"shape_{combo}"].values.ravel().mean())
            scale_mean = float(post[f"scale_{combo}"].values.ravel().mean())
            gamma_mean = float(post[f"gamma_{combo}"].values.ravel().mean())

            adstocked = weibull_adstock_np(lag_mats_period[combo], shape_mean, scale_mean)

            if funnel == "Consideration":
                k_mean  = float(post[f"k_{combo}"].values.ravel().mean())
                as_mean = float(post[f"alpha_s_{combo}"].values.ravel().mean())
                x_safe  = np.maximum(adstocked, 1e-6)
                sat = (x_safe ** as_mean) / (x_safe ** as_mean + k_mean ** as_mean)
            else:
                lam_mean = float(post[f"lam_{combo}"].values.ravel().mean())
                sat = 1 - np.exp(-lam_mean * adstocked)

            mu_pred += gamma_mean * sat

    # Conversion: Geometric
    for combo in FUNNEL_COMBOS["Conversion"]:
        alpha_mean = float(post[f"alpha_{combo}"].values.ravel().mean())
        gamma_mean = float(post[f"gamma_{combo}"].values.ravel().mean())
        lam_mean   = float(post[f"lam_{combo}"].values.ravel().mean())
        x_spend = spend_geo_period[combo]

        adstocked = np.zeros(len(x_spend))
        adstocked[0] = x_spend[0]
        for t in range(1, len(x_spend)):
            adstocked[t] = x_spend[t] + alpha_mean * adstocked[t - 1]

        sat = 1 - np.exp(-lam_mean * adstocked)
        mu_pred += gamma_mean * sat

    return mu_pred * scale_factor + scaler_y.mean_[0]

# ── Predict on Train + Val ───────────────────────────────────────────────────
X_trainval = np.vstack([X_train, X_val])

# Weibull lag matrices for trainval
lag_mats_trainval = {c: lag_matrices[c]["trainval"] for c in ACTIVE_COMBOS
                     if not c.startswith("Conversion")}
spend_geo_trainval = {c: spend_scaled[c]["trainval"] for c in FUNNEL_COMBOS["Conversion"]}

pred_wb = predict_weibull(trace_weibull, X_trainval, lag_mats_trainval, spend_geo_trainval)

actual_train = train_df[target_col].values
actual_val   = val_df[target_col].values
pred_wb_train = pred_wb[:n_train]
pred_wb_val   = pred_wb[n_train:]

mape_train_wb = float(np.mean(np.abs(actual_train - pred_wb_train) / actual_train) * 100)
mape_val_wb   = float(np.mean(np.abs(actual_val - pred_wb_val) / actual_val) * 100)
r2_train_wb   = float(1 - ((actual_train - pred_wb_train)**2).sum() / ((actual_train - actual_train.mean())**2).sum())
r2_val_wb     = float(1 - ((actual_val - pred_wb_val)**2).sum() / ((actual_val - actual_val.mean())**2).sum())

print("━━ EXPERIMENT 2: WEIBULL vs GEOMETRIC COMPARISON ━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print(f"\n{'Metric':<20} {'Geometric (baseline)':<25} {'Weibull (hybrid)':<25}")
print(f"{'─'*70}")
print(f"{'Train MAPE':<20} {'20.0%':<25} {mape_train_wb:.1f}%")
print(f"{'Val MAPE':<20} {'22.5%':<25} {mape_val_wb:.1f}%")
print(f"{'Train R²':<20} {'0.607':<25} {r2_train_wb:.3f}")
print(f"{'Val R²':<20} {'0.499':<25} {r2_val_wb:.3f}")
print(f"{'Divergences':<20} {'0':<25} {divergences_wb}")
print(f"{'Time':<20} {'240s':<25} {elapsed_wb:.0f}s")

# ── Interpretation ───────────────────────────────────────────────────────────
print("\n── Interpretation ──")
if mape_val_wb < 22.5:
    print(f"  ✓ Weibull IMPROVES Val MAPE by {22.5 - mape_val_wb:.1f}pp")
elif mape_val_wb > 23.5:
    print(f"  ✗ Weibull WORSENS Val MAPE by {mape_val_wb - 22.5:.1f}pp")
else:
    print(f"  ≈ Weibull Val MAPE similar to baseline (Δ={mape_val_wb - 22.5:.1f}pp)")

print("\n── Key question: Does Awareness show shape > 1 (delayed peak)?")
for combo in FUNNEL_COMBOS["Awareness"]:
    shape_vals = trace_weibull.posterior[f"shape_{combo}"].values.ravel()
    pct_delayed = (shape_vals > 1).mean() * 100
    print(f"  {combo}: {pct_delayed:.0f}% of posterior draws have shape > 1 (delayed peak)")
    if pct_delayed > 80:
        print(f"    → STRONG evidence for delayed peak in brand Awareness")
    elif pct_delayed > 50:
        print(f"    → Moderate evidence for delayed peak")
    else:
        print(f"    → Weak evidence; geometric may suffice")

# Save experiment results
weibull_results = {
    "mape_train": mape_train_wb, "mape_val": mape_val_wb,
    "r2_train": r2_train_wb, "r2_val": r2_val_wb,
    "divergences": divergences_wb, "elapsed": elapsed_wb,
}
with open(f"{EXH_EXPERIMENT_PATH}weibull_results.pkl", "wb") as f:
    pickle.dump({"trace": trace_weibull, "metrics": weibull_results}, f)
print(f"\nResults saved → {EXH_EXPERIMENT_PATH}weibull_results.pkl")

# COMMAND ----------

# DBTITLE 1,Experiment 3 — Time-Varying Coefficients
# MAGIC %md
# MAGIC ---
# MAGIC ## Experiment 3: Time-Varying Coefficients
# MAGIC Allow `gamma` to evolve over time via:
# MAGIC - **Option A**: Random walk (`gamma[t] ~ N(gamma[t-1], σ_drift)`)
# MAGIC - **Option B**: Regime-switch (2023 vs 2024 vs 2025 eras with different gamma levels)
# MAGIC
# MAGIC Motivation: media effectiveness may have shifted as audiences recovered post-COVID and as CPMs inflated.

# COMMAND ----------

# DBTITLE 1,Experiment 3 — Time-Varying Coefficients (Regime Switch)
# ━━ EXPERIMENT 3: TIME-VARYING COEFFICIENTS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# Option B: REGIME SWITCH
# Allow gamma to differ across eras (2023, 2024, 2025).
# Parameterization:
#   gamma_effective[combo, t] = gamma_base[combo] * exp(delta[funnel, era])
#
# Where delta[funnel, era] is a funnel-level shift (hierarchical — shared
# across combos in the same funnel, keeping parameter count low).
#   - delta = 0 means same as 2023 (reference era)
#   - delta > 0 means media MORE effective in that era
#   - delta < 0 means media LESS effective
#
# Total new parameters: 3 funnels × 2 eras = 6 (very parsimonious)

import time

# ── Era indicators for training period ─────────────────────────────────────────
train_years = train_df["week_date"].dt.year.values
val_years   = val_df["week_date"].dt.year.values

# Binary era dummies (reference = 2023)
era_2024_train = (train_years == 2024).astype(np.float64)
era_2025_train = (train_years >= 2025).astype(np.float64)
era_2024_val   = (val_years == 2024).astype(np.float64)
era_2025_val   = (val_years >= 2025).astype(np.float64)

# Combined trainval
era_2024_trainval = np.concatenate([era_2024_train, era_2024_val])
era_2025_trainval = np.concatenate([era_2025_train, era_2025_val])

print(f"Training era composition:")
print(f"  2023: {(train_years == 2023).sum()} weeks")
print(f"  2024: {(train_years == 2024).sum()} weeks")
print(f"  2025+: {(train_years >= 2025).sum()} weeks")
print(f"\nValidation era: {val_years[0]}–{val_years[-1]} "
      f"(2024: {era_2024_val.sum():.0f}w, 2025: {era_2025_val.sum():.0f}w)")

# ── Build Regime-Switch Model ───────────────────────────────────────────────
def build_regime_model(X_obs, y_obs, spend_data, era_24, era_25):
    """
    Geometric adstock model with era-varying gamma.
    gamma_eff = gamma_base * exp(delta_24 * era_24 + delta_25 * era_25)
    """
    with pm.Model() as model:
        intercept = pm.Normal("intercept", mu=0, sigma=1)
        beta      = pm.Normal("beta", mu=0, sigma=1, shape=X_obs.shape[1])
        sigma     = pm.HalfNormal("sigma", sigma=1)

        mu = intercept + pm.math.dot(X_obs, beta)

        # Era indicators as tensors
        era_24_t = pt.as_tensor_variable(era_24)
        era_25_t = pt.as_tensor_variable(era_25)

        for funnel, combos in FUNNEL_COMBOS.items():
            # Standard funnel priors (same as baseline)
            if funnel == "Awareness":
                ap_a, ap_b = 7, 2
            elif funnel == "Consideration":
                ap_a, ap_b = 3, 3
            else:
                ap_a, ap_b = 2, 5
            mu_gamma_sigma = {"Awareness": 0.5, "Consideration": 0.3, "Conversion": 0.6}[funnel]

            mu_alpha_f    = pm.Beta(f"mu_alpha_{funnel}", alpha=ap_a, beta=ap_b)
            kappa_alpha_f = pm.HalfNormal(f"kappa_alpha_{funnel}", sigma=2)
            mu_gamma_f    = pm.HalfNormal(f"mu_gamma_{funnel}", sigma=mu_gamma_sigma)

            # TIME-VARYING: funnel-level era shifts
            # Normal(0, 0.3) → exp(±0.3) = ±30% shift in effectiveness
            delta_24_f = pm.Normal(f"delta_2024_{funnel}", mu=0, sigma=0.3)
            delta_25_f = pm.Normal(f"delta_2025_{funnel}", mu=0, sigma=0.3)

            if funnel == "Consideration":
                mu_k_f       = pm.HalfNormal("mu_k_Consideration", sigma=0.5)
                mu_alpha_s_f = pm.HalfNormal("mu_alpha_s_Consideration", sigma=1.5)

            for combo in combos:
                alpha_c = pm.Beta(
                    f"alpha_{combo}",
                    alpha=mu_alpha_f * kappa_alpha_f,
                    beta=(1 - mu_alpha_f) * kappa_alpha_f,
                )
                gamma_base = pm.HalfNormal(f"gamma_{combo}", sigma=mu_gamma_f)
                adstocked = geometric_adstock(spend_data[combo], alpha_c)

                if funnel == "Consideration":
                    k_c       = pm.HalfNormal(f"k_{combo}", sigma=mu_k_f)
                    alpha_s_c = pm.HalfNormal(f"alpha_s_{combo}", sigma=mu_alpha_s_f)
                    saturated = hill_saturation(adstocked, k_c, alpha_s_c)
                else:
                    lam_c = pm.HalfNormal(f"lam_{combo}", sigma=1)
                    saturated = logistic_saturation(adstocked, lam_c)

                # Apply era-varying multiplier
                era_multiplier = pt.exp(delta_24_f * era_24_t + delta_25_f * era_25_t)
                mu = mu + gamma_base * era_multiplier * saturated

        pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y_obs)
    return model

# ── Build & Sample ───────────────────────────────────────────────────────────
spend_train_data = {c: spend_scaled[c]["train"] for c in ACTIVE_COMBOS}

print("\nBuilding regime-switch model...")
t0 = time.time()
regime_model = build_regime_model(X_train, y_train, spend_train_data, era_2024_train, era_2025_train)
print(f"Model built in {time.time()-t0:.1f}s")
print(f"Free parameters: {len(regime_model.rvs_to_values)}")

print("\nSampling (2 chains × 500 draws, 500 tune)...")
t0 = time.time()
with regime_model:
    trace_regime = pmjax.sample_numpyro_nuts(
        draws=500, tune=500, chains=2,
        target_accept=0.95, random_seed=42,
    )
elapsed_regime = time.time() - t0
print(f"Sampled in {elapsed_regime:.0f}s")

# ── Convergence ──────────────────────────────────────────────────────────────
divergences_reg = int(trace_regime.sample_stats.diverging.sum().item())
summ_reg = az.summary(trace_regime, var_names=["intercept", "sigma"])
max_rhat_reg = float(summ_reg["r_hat"].max())
print(f"R-hat (intercept/sigma): {max_rhat_reg:.3f} | Divergences: {divergences_reg}")

# ── Era shift posteriors (KEY RESULT) ───────────────────────────────────────
print("\n━━ ERA SHIFT POSTERIORS (exp(delta) = multiplier vs 2023) ━━")
print(f"{'Funnel':<20} {'2024 vs 2023':<25} {'2025 vs 2023':<25} {'Interpretation'}")
print(f"{'─'*90}")

for funnel in ["Awareness", "Consideration", "Conversion"]:
    post = trace_regime.posterior
    d24 = post[f"delta_2024_{funnel}"].values.ravel()
    d25 = post[f"delta_2025_{funnel}"].values.ravel()
    
    mult_24 = np.exp(d24)
    mult_25 = np.exp(d25)
    
    # 89% CI
    ci24 = f"{mult_24.mean():.2f}× [{np.percentile(mult_24,5.5):.2f}, {np.percentile(mult_24,94.5):.2f}]"
    ci25 = f"{mult_25.mean():.2f}× [{np.percentile(mult_25,5.5):.2f}, {np.percentile(mult_25,94.5):.2f}]"
    
    # Interpretation
    if mult_24.mean() > 1.1 or mult_25.mean() > 1.1:
        interp = "↑ INCREASING effectiveness"
    elif mult_24.mean() < 0.9 or mult_25.mean() < 0.9:
        interp = "↓ DECREASING effectiveness"
    else:
        interp = "≈ STABLE (no drift)"
    
    print(f"{funnel:<20} {ci24:<25} {ci25:<25} {interp}")

# Probability of meaningful shift (>10% change)
print("\n── Probability of >10% shift from 2023 ──")
for funnel in ["Awareness", "Consideration", "Conversion"]:
    d24 = trace_regime.posterior[f"delta_2024_{funnel}"].values.ravel()
    d25 = trace_regime.posterior[f"delta_2025_{funnel}"].values.ravel()
    p_up_24   = (np.exp(d24) > 1.1).mean() * 100
    p_down_24 = (np.exp(d24) < 0.9).mean() * 100
    p_up_25   = (np.exp(d25) > 1.1).mean() * 100
    p_down_25 = (np.exp(d25) < 0.9).mean() * 100
    print(f"  {funnel}: 2024 [↑{p_up_24:.0f}% | ↓{p_down_24:.0f}%]  "
          f"2025 [↑{p_up_25:.0f}% | ↓{p_down_25:.0f}%]")

# COMMAND ----------

# DBTITLE 1,Experiment 3 — Regime Switch vs Baseline Comparison
# ━━ PREDICT & COMPARE REGIME MODEL ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def predict_regime(trace, X_data, spend_data_period, era_24, era_25):
    """
    Predict using regime-switch model posteriors.
    """
    post = trace.posterior
    intercept_mean = float(post["intercept"].values.ravel().mean())
    beta_mean = post["beta"].values.reshape(-1, len(baseline_features)).mean(axis=0)

    mu_pred = intercept_mean + X_data @ beta_mean

    for funnel, combos in FUNNEL_COMBOS.items():
        delta_24_mean = float(post[f"delta_2024_{funnel}"].values.ravel().mean())
        delta_25_mean = float(post[f"delta_2025_{funnel}"].values.ravel().mean())
        era_mult = np.exp(delta_24_mean * era_24 + delta_25_mean * era_25)

        for combo in combos:
            alpha_mean = float(post[f"alpha_{combo}"].values.ravel().mean())
            gamma_mean = float(post[f"gamma_{combo}"].values.ravel().mean())
            x_spend = spend_data_period[combo]

            # Numpy adstock
            adstocked = np.zeros(len(x_spend))
            adstocked[0] = x_spend[0]
            for t in range(1, len(x_spend)):
                adstocked[t] = x_spend[t] + alpha_mean * adstocked[t - 1]

            # Saturation
            if funnel == "Consideration":
                k_mean  = float(post[f"k_{combo}"].values.ravel().mean())
                as_mean = float(post[f"alpha_s_{combo}"].values.ravel().mean())
                x_safe  = np.maximum(adstocked, 1e-6)
                sat = (x_safe ** as_mean) / (x_safe ** as_mean + k_mean ** as_mean)
            else:
                lam_mean = float(post[f"lam_{combo}"].values.ravel().mean())
                sat = 1 - np.exp(-lam_mean * adstocked)

            mu_pred += gamma_mean * era_mult * sat

    return mu_pred * scale_factor + scaler_y.mean_[0]

# ── Predict on Train + Val ───────────────────────────────────────────────────
X_trainval = np.vstack([X_train, X_val])
spend_trainval = {c: spend_scaled[c]["trainval"] for c in ACTIVE_COMBOS}

pred_reg = predict_regime(
    trace_regime, X_trainval, spend_trainval,
    era_2024_trainval, era_2025_trainval
)

actual_train = train_df[target_col].values
actual_val   = val_df[target_col].values
pred_reg_train = pred_reg[:n_train]
pred_reg_val   = pred_reg[n_train:]

mape_train_reg = float(np.mean(np.abs(actual_train - pred_reg_train) / actual_train) * 100)
mape_val_reg   = float(np.mean(np.abs(actual_val - pred_reg_val) / actual_val) * 100)
r2_train_reg   = float(1 - ((actual_train - pred_reg_train)**2).sum() / ((actual_train - actual_train.mean())**2).sum())
r2_val_reg     = float(1 - ((actual_val - pred_reg_val)**2).sum() / ((actual_val - actual_val.mean())**2).sum())

print("━━ EXPERIMENT 3: REGIME SWITCH vs BASELINE COMPARISON ━━━━━━━━━━━━━━━━━━━━━━")
print(f"\n{'Metric':<20} {'Geometric (static)':<25} {'Regime Switch':<25} {'Delta'}")
print(f"{'─'*85}")
print(f"{'Train MAPE':<20} {'20.0%':<25} {mape_train_reg:.1f}%{'':<17} {mape_train_reg-20.0:+.1f}pp")
print(f"{'Val MAPE':<20} {'22.5%':<25} {mape_val_reg:.1f}%{'':<17} {mape_val_reg-22.5:+.1f}pp")
print(f"{'Train R²':<20} {'0.607':<25} {r2_train_reg:.3f}{'':<18} {r2_train_reg-0.607:+.3f}")
print(f"{'Val R²':<20} {'0.499':<25} {r2_val_reg:.3f}{'':<18} {r2_val_reg-0.499:+.3f}")
print(f"{'Divergences':<20} {'0':<25} {divergences_reg}")
print(f"{'Time':<20} {'240s':<25} {elapsed_regime:.0f}s")
print(f"{'Extra params':<20} {'0':<25} 6 (3 funnels × 2 eras)")

# ── Interpretation ───────────────────────────────────────────────────────────
print("\n── Verdict ──")
if mape_val_reg < 21.5:
    print("  ✓ STRONG IMPROVEMENT — Time-varying model significantly better on validation")
    print("  → Recommend adopting regime switch in production model")
elif mape_val_reg < 22.0:
    print("  ✓ Moderate improvement — Time-varying model slightly better")
    print("  → Consider adopting if era shifts are interpretable")
elif mape_val_reg < 23.0:
    print("  ≈ No meaningful improvement — Media effectiveness appears stable across eras")
    print("  → Keep static gamma (simpler model wins)")
else:
    print("  ✗ Worse fit — Extra parameters overfit without improving generalization")
    print("  → Static gamma confirmed as better choice")

# Save
with open(f"{EXH_EXPERIMENT_PATH}regime_results.pkl", "wb") as f:
    pickle.dump({"trace": trace_regime, "metrics": {
        "mape_train": mape_train_reg, "mape_val": mape_val_reg,
        "r2_train": r2_train_reg, "r2_val": r2_val_reg,
        "divergences": divergences_reg,
    }}, f)
print(f"\nResults saved → {EXH_EXPERIMENT_PATH}regime_results.pkl")

# COMMAND ----------

# DBTITLE 1,Experiment 4 — Expanded Training
# MAGIC %md
# MAGIC ---
# MAGIC ## Experiment 4: Expanded Training (Train+Val → New Train)
# MAGIC Retrain the winning model spec on 150 weeks (current train+val), validate on 15-week test set.
# MAGIC - Expected: tighter posteriors → narrower CIs
# MAGIC - This becomes the production model if test MAPE is acceptable

# COMMAND ----------

# DBTITLE 1,Experiment 4 — Expanded Training (150w Train, 15w Test)
# ━━ EXPERIMENT 4: EXPANDED TRAINING ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# New split:
#   Train:  150 weeks (current train + current val) = weeks 1–150
#   Test:    15 weeks (current test)                = weeks 151–165
#
# Uses the SAME model spec as baseline (confirmed optimal by Exp 1–3):
#   - Geometric adstock (Exp 2 rejected Weibull)
#   - Static gamma (Exp 3 rejected time-varying)
#   - Baseline priors: Beta(7,2)/Beta(3,3)/Beta(2,5) (Exp 1 confirmed)
#
# Quick screen first (2 chains × 500), then compare to current baseline.
# If test MAPE is acceptable, this becomes the production model.

import time

# ── Re-define build_model (cell 9 kernel state was lost after restart) ────────
def build_model(alpha_mu_prior, mu_gamma_prior, X_obs, y_obs, spend_data):
    """Build the mixed-hierarchical MMM model with given priors."""
    with pm.Model() as model:
        intercept = pm.Normal("intercept", mu=0, sigma=1)
        beta      = pm.Normal("beta", mu=0, sigma=1, shape=X_obs.shape[1])
        sigma     = pm.HalfNormal("sigma", sigma=1)
        mu = intercept + pm.math.dot(X_obs, beta)

        for funnel, combos in FUNNEL_COMBOS.items():
            ap = alpha_mu_prior[funnel]
            mu_alpha_f    = pm.Beta(f"mu_alpha_{funnel}", alpha=ap["alpha"], beta=ap["beta"])
            kappa_alpha_f = pm.HalfNormal(f"kappa_alpha_{funnel}", sigma=2)
            mu_gamma_f    = pm.HalfNormal(f"mu_gamma_{funnel}", sigma=mu_gamma_prior[funnel])

            if funnel == "Consideration":
                mu_k_f       = pm.HalfNormal("mu_k_Consideration", sigma=0.5)
                mu_alpha_s_f = pm.HalfNormal("mu_alpha_s_Consideration", sigma=1.5)

            for combo in combos:
                alpha_c = pm.Beta(
                    f"alpha_{combo}",
                    alpha=mu_alpha_f * kappa_alpha_f,
                    beta=(1 - mu_alpha_f) * kappa_alpha_f,
                )
                gamma_c = pm.HalfNormal(f"gamma_{combo}", sigma=mu_gamma_f)
                adstocked = geometric_adstock(spend_data[combo], alpha_c)

                if funnel == "Consideration":
                    k_c       = pm.HalfNormal(f"k_{combo}", sigma=mu_k_f)
                    alpha_s_c = pm.HalfNormal(f"alpha_s_{combo}", sigma=mu_alpha_s_f)
                    saturated = hill_saturation(adstocked, k_c, alpha_s_c)
                else:
                    lam_c = pm.HalfNormal(f"lam_{combo}", sigma=1)
                    saturated = logistic_saturation(adstocked, lam_c)

                mu = mu + gamma_c * saturated

        pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y_obs)
    return model

# ── New Split & Scaling ──────────────────────────────────────────────────────
n_new_train = n_train + n_val  # 150 weeks
n_new_test  = n_test            # 15 weeks

new_train_df = df.iloc[:n_new_train].copy()
new_test_df  = df.iloc[n_new_train:].copy()

print(f"Expanded split:")
print(f"  New Train: {n_new_train} weeks ({new_train_df['week_date'].min().date()} → {new_train_df['week_date'].max().date()})")
print(f"  New Test:  {n_new_test} weeks ({new_test_df['week_date'].min().date()} → {new_test_df['week_date'].max().date()})")

# Re-scale with new training set
from sklearn.preprocessing import StandardScaler

scaler_X_new = StandardScaler()
X_new_train  = scaler_X_new.fit_transform(new_train_df[baseline_features].values)
X_new_test   = scaler_X_new.transform(new_test_df[baseline_features].values)

scaler_y_new = StandardScaler()
y_new_train  = scaler_y_new.fit_transform(new_train_df[[target_col]].values).ravel()
y_new_test   = scaler_y_new.transform(new_test_df[[target_col]].values).ravel()

scale_factor_new = float(scaler_y_new.scale_[0])
print(f"  y scale factor: {scale_factor_new:,.0f} (was {scale_factor:,.0f})")

# Re-scale media spend (new train max)
spend_scaled_new = {}
spend_train_max_new = {}

for combo in ACTIVE_COMBOS:
    m = float(new_train_df[combo].max())
    spend_train_max_new[combo] = m
    spend_scaled_new[combo] = {
        "train": new_train_df[combo].values.astype(np.float64) / m,
        "test":  new_test_df[combo].values.astype(np.float64) / m,
    }

# ── Build Model (same baseline spec) ───────────────────────────────────────
alpha_mu_baseline = {
    "Awareness":     {"alpha": 7, "beta": 2},
    "Consideration": {"alpha": 3, "beta": 3},
    "Conversion":    {"alpha": 2, "beta": 5},
}
mu_gamma_baseline = {"Awareness": 0.5, "Consideration": 0.3, "Conversion": 0.6}

spend_new_train_data = {c: spend_scaled_new[c]["train"] for c in ACTIVE_COMBOS}

print("\nBuilding expanded model (same spec as baseline)...")
t0 = time.time()
expanded_model = build_model(alpha_mu_baseline, mu_gamma_baseline,
                              X_new_train, y_new_train, spend_new_train_data)
print(f"Model built in {time.time()-t0:.1f}s")

# ── Sample: production quality (4 chains × 2000 draws, target_accept=0.99) ────
# This matches the main notebook's sampling config for a fair comparison.
print("\nSampling (4 chains × 2000 draws, 2000 tune, target_accept=0.99)...")
print("  (This is a production-quality run — expect ~10–15 min)")
t0 = time.time()
with expanded_model:
    trace_expanded = pmjax.sample_numpyro_nuts(
        draws=2000, tune=2000, chains=4,
        target_accept=0.99, random_seed=42,
    )
elapsed_exp = time.time() - t0
print(f"Sampled in {elapsed_exp:.0f}s")

# ── Convergence ──────────────────────────────────────────────────────────────
divergences_exp = int(trace_expanded.sample_stats.diverging.sum().item())
var_names_check = [v for v in ["intercept", "sigma"] + 
                   [f"alpha_{c}" for c in ACTIVE_COMBOS] +
                   [f"gamma_{c}" for c in ACTIVE_COMBOS]]
summ_exp = az.summary(trace_expanded, var_names=var_names_check)
max_rhat_exp = float(summ_exp["r_hat"].max())
min_ess_exp  = int(summ_exp["ess_bulk"].min())
print(f"\nConvergence:")
print(f"  Max R-hat: {max_rhat_exp:.4f} (threshold: <1.01)")
print(f"  Min ESS:   {min_ess_exp} (threshold: >400)")
print(f"  Divergences: {divergences_exp}")

# ── Predict on NEW Train + NEW Test ────────────────────────────────────────
def predict_expanded(trace, X_data, spend_period):
    """Predict using expanded model."""
    post = trace.posterior
    intercept_mean = float(post["intercept"].values.ravel().mean())
    beta_mean = post["beta"].values.reshape(-1, len(baseline_features)).mean(axis=0)
    mu_pred = intercept_mean + X_data @ beta_mean

    for funnel, combos in FUNNEL_COMBOS.items():
        for combo in combos:
            alpha_mean = float(post[f"alpha_{combo}"].values.ravel().mean())
            gamma_mean = float(post[f"gamma_{combo}"].values.ravel().mean())
            x_spend = spend_period[combo]

            adstocked = np.zeros(len(x_spend))
            adstocked[0] = x_spend[0]
            for t in range(1, len(x_spend)):
                adstocked[t] = x_spend[t] + alpha_mean * adstocked[t - 1]

            if funnel == "Consideration":
                k_mean  = float(post[f"k_{combo}"].values.ravel().mean())
                as_mean = float(post[f"alpha_s_{combo}"].values.ravel().mean())
                x_safe  = np.maximum(adstocked, 1e-6)
                sat = (x_safe ** as_mean) / (x_safe ** as_mean + k_mean ** as_mean)
            else:
                lam_mean = float(post[f"lam_{combo}"].values.ravel().mean())
                sat = 1 - np.exp(-lam_mean * adstocked)

            mu_pred += gamma_mean * sat

    return mu_pred * scale_factor_new + scaler_y_new.mean_[0]

# Full dataset prediction
X_all = np.vstack([X_new_train, X_new_test])
spend_all = {}
for combo in ACTIVE_COMBOS:
    m = spend_train_max_new[combo]
    spend_all[combo] = df[combo].values.astype(np.float64) / m

pred_all = predict_expanded(trace_expanded, X_all, spend_all)

actual_new_train = new_train_df[target_col].values
actual_new_test  = new_test_df[target_col].values
pred_new_train   = pred_all[:n_new_train]
pred_new_test    = pred_all[n_new_train:]

mape_train_exp = float(np.mean(np.abs(actual_new_train - pred_new_train) / actual_new_train) * 100)
mape_test_exp  = float(np.mean(np.abs(actual_new_test - pred_new_test) / actual_new_test) * 100)
r2_train_exp   = float(1 - ((actual_new_train - pred_new_train)**2).sum() / ((actual_new_train - actual_new_train.mean())**2).sum())
r2_test_exp    = float(1 - ((actual_new_test - pred_new_test)**2).sum() / ((actual_new_test - actual_new_test.mean())**2).sum())

# ── CI Width for +10% scenario ───────────────────────────────────────────────
def compute_ci_expanded(trace, ci=0.89):
    """CI width for +10% budget (using expanded model's posteriors)."""
    post = trace.posterior
    n_draws = min(400, post.dims["draw"] * post.dims["chain"])
    all_draws = np.random.choice(
        post.dims["draw"] * post.dims["chain"], size=n_draws, replace=False
    )

    def combo_contrib_draws(combo, annual_spend, draws_idx):
        alpha_arr = post[f"alpha_{combo}"].values.ravel()[draws_idx]
        gamma_arr = post[f"gamma_{combo}"].values.ravel()[draws_idx]
        alpha_arr = np.minimum(alpha_arr, 0.999)
        mx = spend_train_max_new[combo]
        s = (annual_spend / 52) / mx
        ad = s / (1 - alpha_arr)

        funnel = combo.split("__")[0]
        if funnel == "Consideration":
            k_arr  = post[f"k_{combo}"].values.ravel()[draws_idx]
            as_arr = post[f"alpha_s_{combo}"].values.ravel()[draws_idx]
            sat = (ad ** as_arr) / (ad ** as_arr + k_arr ** as_arr)
        else:
            lam_arr = post[f"lam_{combo}"].values.ravel()[draws_idx]
            sat = 1 - np.exp(-lam_arr * ad)

        return gamma_arr * sat * scale_factor_new * 52

    # Use 2025 spend as baseline
    _df_2025 = df[df["week_date"].dt.year == 2025]
    baseline_spend = {c: float(_df_2025[c].sum()) for c in ACTIVE_COMBOS}

    base_draws = np.zeros(n_draws)
    scen_draws = np.zeros(n_draws)
    for combo in ACTIVE_COMBOS:
        base_draws += combo_contrib_draws(combo, baseline_spend[combo], all_draws)
        scen_draws += combo_contrib_draws(combo, baseline_spend[combo] * 1.1, all_draws)

    lift_pct_draws = (scen_draws - base_draws) / base_draws * 100
    lo_q, hi_q = (1 - ci) / 2, 1 - (1 - ci) / 2
    ci_lo = float(np.quantile(lift_pct_draws, lo_q))
    ci_hi = float(np.quantile(lift_pct_draws, hi_q))
    point_pct = float(np.median(lift_pct_draws))
    return point_pct, ci_hi - ci_lo, ci_lo, ci_hi

point_lift_exp, ci_width_exp, ci_lo_exp, ci_hi_exp = compute_ci_expanded(trace_expanded)

# ━━ FINAL COMPARISON TABLE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n" + "━" * 80)
print("  EXPERIMENT 4: EXPANDED TRAINING — FINAL RESULTS")
print("━" * 80)
print(f"\n{'Metric':<25} {'Old (124w train)':<25} {'New (150w train)':<25} {'Delta'}")
print(f"{'─'*90}")
print(f"{'Train MAPE':<25} {'20.0% (124w)':<25} {mape_train_exp:.1f}% (150w){'':<8} {mape_train_exp-20.0:+.1f}pp")
print(f"{'Holdout MAPE':<25} {'22.5% (26w val)':<25} {mape_test_exp:.1f}% (15w test){'':<4} {mape_test_exp-22.5:+.1f}pp")
print(f"{'Train R²':<25} {'0.607':<25} {r2_train_exp:.3f}{'':<18} {r2_train_exp-0.607:+.3f}")
print(f"{'Holdout R²':<25} {'0.499':<25} {r2_test_exp:.3f}{'':<18} {r2_test_exp-0.499:+.3f}")
print(f"{'+10% CI width':<25} {'6.1pp':<25} {ci_width_exp:.1f}pp{'':<16} {ci_width_exp-6.1:+.1f}pp")
print(f"{'+10% lift [89% CI]':<25} {'4.4% [1.3%,7.4%]':<25} {point_lift_exp:.1f}% [{ci_lo_exp:.1f}%,{ci_hi_exp:.1f}%]")
print(f"{'Divergences':<25} {'0':<25} {divergences_exp}")
print(f"{'Max R-hat':<25} {'1.010':<25} {max_rhat_exp:.4f}")
print(f"{'Min ESS':<25} {'431':<25} {min_ess_exp}")
print(f"{'Sampling':<25} {'4×2000, 124w':<25} 4×2000, 150w")
print(f"{'Time':<25} {'~15 min':<25} {elapsed_exp:.0f}s")

# ── Verdict ─────────────────────────────────────────────────────────────────
print(f"\n{'='*80}")
print("  VERDICT")
print(f"{'='*80}")
if ci_width_exp < 5.0:
    ci_verdict = "✓ TIGHT (< 5pp)"
elif ci_width_exp < 6.0:
    ci_verdict = "✓ Improved (< 6pp)"
else:
    ci_verdict = "✗ Still wide (≥ 6pp)"

if mape_test_exp < 22.0:
    fit_verdict = "✓ EXCELLENT test fit"
elif mape_test_exp < 25.0:
    fit_verdict = "✓ Acceptable test fit"
else:
    fit_verdict = "⚠ Test fit needs review"

print(f"  CI width: {ci_verdict}")
print(f"  Test fit: {fit_verdict}")
print(f"  Convergence: {'PASS' if max_rhat_exp < 1.01 and divergences_exp <= 2 else 'NEEDS REVIEW'}")

if mape_test_exp < 25.0 and ci_width_exp < 6.0 and max_rhat_exp < 1.01:
    print("\n  → RECOMMENDATION: Adopt expanded model as production baseline.")
    print("    Update main notebook with 150w training split.")
else:
    print("\n  → Review needed before adopting.")

# ── Save production artifacts ────────────────────────────────────────────────
with open(f"{EXH_EXPERIMENT_PATH}expanded_model_results.pkl", "wb") as f:
    pickle.dump({
        "trace": trace_expanded,
        "metrics": {
            "mape_train": mape_train_exp, "mape_test": mape_test_exp,
            "r2_train": r2_train_exp, "r2_test": r2_test_exp,
            "ci_width": ci_width_exp, "point_lift": point_lift_exp,
            "ci_lo": ci_lo_exp, "ci_hi": ci_hi_exp,
            "divergences": divergences_exp, "max_rhat": max_rhat_exp,
            "min_ess": min_ess_exp,
        },
        "scalers": {"scaler_X": scaler_X_new, "scaler_y": scaler_y_new},
        "spend_train_max": spend_train_max_new,
        "split": {"n_train": n_new_train, "n_test": n_new_test},
    }, f)
print(f"\nArtifacts saved → {EXH_EXPERIMENT_PATH}expanded_model_results.pkl")
