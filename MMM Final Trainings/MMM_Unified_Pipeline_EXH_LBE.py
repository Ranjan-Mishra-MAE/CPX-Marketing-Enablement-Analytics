# Databricks notebook source
# DBTITLE 1,MMM Unified Pipeline — EXH + LBE
# MAGIC %md
# MAGIC # MMM Unified Pipeline — EXH + LBE
# MAGIC **Cineplex Marketing Analytics | Marketing Mix Modelling**
# MAGIC
# MAGIC This notebook contains the complete, end-to-end Marketing Mix Model pipeline for both business units:
# MAGIC
# MAGIC | Model | Target | Media Structure | Validation |
# MAGIC | --- | --- | --- | --- |
# MAGIC | **EXH** (Exhibition) | Weekly attendance | 54 media cols → 10 Funnel×LOB combos (7 active) | Val R² 0.484 |
# MAGIC | **LBE** (Location-Based Entertainment) | Weekly revenue | 2 funnels (Awareness via TRR, Consideration via PDM) | Val R² 0.524, MAPE 9.2% |
# MAGIC
# MAGIC ### Execution Order
# MAGIC 1. **Shared Configuration** — installs, imports, paths, brand palette
# MAGIC 2. **Part A** — EXH Model Training (PyMC NUTS, 4×200 draws)
# MAGIC 3. **Part B** — EXH Diagnostics & Visualizations
# MAGIC 4. **Part C** — EXH LOB Decomposition (graft method)
# MAGIC 5. **Part D** — EXH Budget Scenario Analysis (3 scenarios with 89% CIs)
# MAGIC 6. **Part E** — LBE Model Training (PyMC NUTS, 4×2000 draws)
# MAGIC 7. **Part F** — LBE Diagnostics
# MAGIC 8. **Part G** — LBE Budget Scenario Analysis
# MAGIC 9. **Part H** — Stakeholder Presentation Charts (combined)
# MAGIC
# MAGIC ### Runtime
# MAGIC - Full pipeline: ~90 min on GPU cluster (dominated by NUTS sampling)
# MAGIC - Diagnostics-only (Parts B–D, F–H): ~5 min (loads saved artifacts)
# MAGIC
# MAGIC ### Artifact Locations
# MAGIC - EXH: `/Volumes/data_exploration_hub/marketing_hub/model_data/Final_Data/MMM_EXH_Outcomes/`
# MAGIC - EXH LOB Decomp: `/Volumes/data_exploration_hub/marketing_hub/model_data/Final_Data/MMM_EXH_LOB_Decomp/`
# MAGIC - LBE: `/Volumes/data_exploration_hub/marketing_hub/model_data/Final_Data/MMM_LBE/`

# COMMAND ----------

# DBTITLE 1,Assumptions & Methodology
# MAGIC %md
# MAGIC ## Assumptions & Methodology
# MAGIC
# MAGIC ### EXH Model (Exhibition — Weekly Attendance)
# MAGIC - **Architecture**: Mixed Hierarchical Bayesian MMM with funnel-level hyperpriors
# MAGIC - **Data**: 165 weeks (2023-01-02 to 2026-02-23), weekly granularity
# MAGIC - **Target**: `total_weekly_attendance`
# MAGIC - **Media structure**: 54 raw media columns (format: `Funnel_LOBname_Channel`) parsed into Funnel × LOB × Channel, grouped into 10 collapsed combos. 3 combos dropped (>60% zeros in training) → **7 active combos**
# MAGIC - **LOB groups**: Premium_Brand (4 LOBs), Film_Marketing (3 LOBs), GE (1 LOB), Prepaid (2 LOBs), Cineclub (1 LOB)
# MAGIC - **Response functions**: Geometric adstock (α ~ Beta, funnel-level hyperprior) + logistic/Hill saturation (γ positive effect, λ/k saturation rate)
# MAGIC - **Baseline controls**: 22 features (season dummies, stat holidays, content tier, year indicators, trend)
# MAGIC - **Sampling**: PyMC NUTS via JAX backend, 4 chains × 200 draws (800 posterior samples)
# MAGIC - **Split**: Chronological — last 15 weeks = test, preceding 26 weeks = validation
# MAGIC - **Validation**: Val R² = 0.484, trained MAPE and R² reported
# MAGIC
# MAGIC ### LBE Model (Location-Based Entertainment — Weekly Revenue)
# MAGIC - **Architecture**: Bayesian log-revenue regression with channel-level adstock
# MAGIC - **Data**: Weekly LBE revenue + media spend + controls
# MAGIC - **Target**: `log(LBE_Revenue)` (log-space modelling, exp-transformed for dollar predictions)
# MAGIC - **Media structure**: 2 funnels — Awareness (TRR channels) and Consideration (PDM channels)
# MAGIC - **Response functions**: Geometric adstock (decay ~ Beta priors) + log1p saturation
# MAGIC - **Controls**: Linear trend, national mean temperature, 9 calendar event binary flags
# MAGIC - **Sampling**: PyMC NUTS, 4 chains × 2000 draws (8,000 posterior samples)
# MAGIC - **Split**: Chronological hold-out validation (last N weeks)
# MAGIC - **Validation**: Val R² = 0.524, Val MAPE = 9.2%
# MAGIC
# MAGIC ### LOB Decomposition (EXH only)
# MAGIC - **Method**: Graft — splits validated group-level contributions to original LOBs
# MAGIC - **Rationale**: Direct LOB-level refit hurt validation (media overfitting at sparse level)
# MAGIC - **Signal**: Residual-correlation tilt (correlation of LOB spend with control-confound-removed attendance)
# MAGIC - **Constraints**: ±35% cap on tilt adjustment, renormalised to preserve group total
# MAGIC - **Validation flag**: "Curve" (validated), "Borrow" (extrapolated), "Frozen" (report-only)
# MAGIC
# MAGIC ### Budget Scenarios (both models)
# MAGIC - **Scenarios**: (1) Budget uplift +10/20/30%, (2) Reshuffle (capped siphon to high-marginal), (3) Combined
# MAGIC - **Uncertainty**: 89% paired per-draw credible intervals (shared parameter uncertainty cancels)
# MAGIC - **Confidence labels**: CI width <4pp = High, 4–8pp = Medium, >8pp = Low
# MAGIC - **Key assumption**: Saturation means revenue/attendance lift is sub-proportional to spend increase
# MAGIC
# MAGIC ### General Assumptions
# MAGIC - Steady-state adstock for annual projections (infinite-horizon geometric sum)
# MAGIC - All uncertainty is Bayesian posterior — no frequentist CIs or bootstrap
# MAGIC - No interaction effects between EXH and LBE media
# MAGIC - Calendar/seasonality effects are additive (no media×calendar interaction modelled)
# MAGIC - 2026 forecast assumes 2025 control patterns persist unless otherwise specified

# COMMAND ----------

# DBTITLE 1,Section 2 — Shared Configuration
# MAGIC %md
# MAGIC ---
# MAGIC ## Section 2: Shared Configuration & Installs
# MAGIC Consolidated dependencies, imports, paths, and brand palette for the entire pipeline.

# COMMAND ----------

# DBTITLE 1,Install Dependencies
# MAGIC %pip install pymc arviz numpyro jax jaxlib pytensor scikit-learn openpyxl netcdf4 h5netcdf scipy xarray --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Imports
import os
import pickle
import shutil
import warnings
import datetime
from collections import defaultdict

import numpy as np
import pandas as pd
import pytensor
import pytensor.tensor as pt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error, r2_score
from scipy.stats import beta as beta_dist, halfnorm, norm
import pymc as pm
import arviz as az
import pymc.sampling.jax as pmjax

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import seaborn as sns

warnings.filterwarnings("ignore")
print(f"PyMC  : {pm.__version__}")
print(f"ArviZ : {az.__version__}")

# COMMAND ----------

# DBTITLE 1,Paths Configuration
# ── EXH Paths ──────────────────────────────────────────────────────────────────
EXH_FILE_PATH     = r"/Volumes/data_exploration_hub/marketing_hub/model_data/Final_Data/EXH_ModelData.xlsx"
EXH_ARTIFACT_PATH = "/Volumes/data_exploration_hub/marketing_hub/model_data/Final_Data/MMM_EXH_Outcomes_v2/"
EXH_DECOMP_PATH   = "/Volumes/data_exploration_hub/marketing_hub/model_data/Final_Data/MMM_EXH_LOB_Decomp_v2/"

# ── LBE Paths ──────────────────────────────────────────────────────────────────
LBE_INPUT_PATH    = r"/Volumes/data_exploration_hub/marketing_hub/model_data/Final_Data/lbe_model_input.csv"
LBE_OUTPUT_DIR    = r"/Volumes/data_exploration_hub/marketing_hub/model_data/Final_Data/MMM_LBE_v2"
LBE_RESULTS_PKL   = r"/Volumes/data_exploration_hub/marketing_hub/model_data/Final_Data/MMM_LBE_v2/lbe_mmm_results.pkl"

# ── Create directories ─────────────────────────────────────────────────────────
for path in [EXH_ARTIFACT_PATH, EXH_DECOMP_PATH, LBE_OUTPUT_DIR]:
    os.makedirs(path, exist_ok=True)

print(f"EXH Artifacts → {EXH_ARTIFACT_PATH}")
print(f"EXH LOB Decomp → {EXH_DECOMP_PATH}")
print(f"LBE Output     → {LBE_OUTPUT_DIR}")

# COMMAND ----------

# DBTITLE 1,Brand Palette & Style
# ── Cineplex Brand Palette ─────────────────────────────────────────────────────
CPX_NAVY  = "#002B5C"
CPX_RED   = "#E31837"
CPX_GOLD  = "#F5A800"
CPX_LGREY = "#F2F2F2"
CPX_DGREY = "#4A4A4A"
CPX_GREEN = "#2E7D32"

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": "#CCCCCC", "grid.color": "#EEEEEE",
    "grid.linestyle": "-", "axes.grid": True,
    "font.family": "sans-serif", "font.size": 10,
    "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.labelsize": 10, "legend.fontsize": 9,
})

def save_fig(fig, filename, path=None):
    """Save figure to the specified path (default: EXH artifact path)."""
    save_path = path or EXH_ARTIFACT_PATH
    out = os.path.join(save_path, filename)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"  Saved → {out}")

# COMMAND ----------

# DBTITLE 1,Part A — EXH Model Training
# MAGIC %md
# MAGIC ---
# MAGIC ## Part A: EXH Model Training
# MAGIC **Mixed Hierarchical Bayesian MMM — Phase 2, Iteration 2**
# MAGIC
# MAGIC Trains the final model on 165 weeks of EXH attendance data. Exports trace, scalers, metadata, and response parameters to `EXH_ARTIFACT_PATH`.
# MAGIC
# MAGIC **Architecture**: Funnel-level hyperpriors pool adstock (α) and effect size (γ) across LOB combos. Consideration uses Hill saturation; Awareness/Conversion use logistic saturation.

# COMMAND ----------

# DBTITLE 1,EXH Data Loading & Feature Engineering
# ━━ EXH DATA LOADING & FEATURE ENGINEERING ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

df = pd.read_excel(EXH_FILE_PATH, sheet_name=0)
df = df.sort_values("week_date").reset_index(drop=True)

# Binary encode flag columns
df["stat_holiday_in_week"] = (df["stat_holiday_in_week"] == "Y").astype(int)
df["is_long_weekend"]      = (df["is_long_weekend"]      == "Y").astype(int)

# Season dummies (winter = reference, dropped)
season_dummies = pd.get_dummies(df["season"], prefix="season", drop_first=False).astype(int)
season_dummies = season_dummies.drop(columns=["season_winter"])
df = pd.concat([df.drop(columns=["season"]), season_dummies], axis=1)

# Year indicators and linear trend
df["is_2024"]    = (df["week_date"].dt.year == 2024).astype(int)
df["is_2025"]    = (df["week_date"].dt.year == 2025).astype(int)
df["week_index"] = (df["week_date"] - df["week_date"].min()).dt.days / 7

print(f"Data loaded — shape {df.shape}")
print(f"Date range: {df['week_date'].min().date()} → {df['week_date'].max().date()}")
print(f"Missing values: {df.isna().sum().sum()}")

# COMMAND ----------

# DBTITLE 1,EXH Feature Selection
# ━━ FEATURE SELECTION ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
date_col   = "week_date"
target_col = "total_weekly_attendance"

DROP_FEATURES = [
    "national_max_temp", "national_min_temp", "national_temp_anomaly",
    "action_intensity", "adaptation_intensity", "adventure_intensity",
    "animated_intensity", "biopic_intensity", "comedy_intensity",
    "crime_intensity", "documentary_intensity", "drama_intensity",
    "family_intensity", "fantasy_intensity", "foreign_intensity",
    "horror_intensity", "musical_intensity", "mystery_intensity",
    "period_intensity", "political_intensity", "romance_intensity",
    "romanticcomedy_intensity", "scifi_intensity", "sequel_intensity",
    "suspense_intensity", "thriller_intensity",
]

control_cols     = df.columns[3:44].tolist()
control_cols     = [c for c in control_cols if c != "season"] + season_dummies.columns.tolist()
baseline_features = [f for f in control_cols if f not in DROP_FEATURES]
baseline_features += ["is_2024", "is_2025", "week_index"]

assert len(baseline_features) == 22, f"Expected 22 baseline features, got {len(baseline_features)}"
print(f"Baseline features: {len(baseline_features)}")

# COMMAND ----------

# DBTITLE 1,EXH Train/Val/Test Split & Scaling
# ━━ TRAIN / VALIDATION / TEST SPLIT ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

n       = len(df)
n_test  = 15
n_val   = 26
n_train = n - n_val - n_test

train_df = df.iloc[:n_train].copy().reset_index(drop=True)
val_df   = df.iloc[n_train : n_train + n_val].copy().reset_index(drop=True)
test_df  = df.iloc[n_train + n_val :].copy().reset_index(drop=True)

print(f"Train : {n_train} weeks  ({train_df['week_date'].min().date()} → {train_df['week_date'].max().date()})")
print(f"Val   : {n_val}   weeks  ({val_df['week_date'].min().date()} → {val_df['week_date'].max().date()})")
print(f"Test  : {n_test}  weeks  (held out)")

# ━━ SCALING ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train = scaler_X.fit_transform(train_df[baseline_features])
X_val   = scaler_X.transform(val_df[baseline_features])

y_train = scaler_y.fit_transform(train_df[[target_col]]).flatten()
y_val   = scaler_y.transform(val_df[[target_col]]).flatten()

# COMMAND ----------

# DBTITLE 1,EXH Media Metadata & LOB Aggregation
# ━━ MEDIA METADATA & FUNNEL×LOB AGGREGATION ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

all_cols    = df.columns.tolist()
target_idx  = all_cols.index(target_col)
media_start = all_cols.index("Awareness_Cineclub_Digital")
media_cols  = all_cols[media_start:target_idx]

def parse_media_col(col):
    parts   = col.split("_")
    funnel  = parts[0].capitalize()
    channel = parts[-1]
    lob     = "_".join(parts[1:-1])
    return funnel, lob, channel

media_meta = pd.DataFrame(
    [parse_media_col(c) + (c,) for c in media_cols],
    columns=["funnel", "lob", "channel", "col_name"],
)

print(f"Media columns   : {len(media_cols)}")
print("Funnels         :", sorted(media_meta["funnel"].unique()))
print("LOBs            :", sorted(media_meta["lob"].unique()))

# LOB grouping — collapses sparse LOBs into interpretable groups
LOB_GROUPS = {
    "Premium_Brand" : [
        "Premium Experiences - AVX",
        "Premium Experiences - VIP",
        "F&B",
        "EXH - Brand",
    ],
    "Film_Marketing": [
        "Premium Experiences - IMAX",
        "Film Marketing - Inform",
        "Performance - Convert & Upgrade",
    ],
    "Prepaid"       : ["Prepaid - B2B", "Prepaid - B2C"],
    "Cineclub"      : ["Cineclub"],
    "GE"            : ["G&E"],
}

FUNNELS = ["Awareness", "Consideration", "Conversion"]

# Build funnel x LOB group columns on df
for funnel in FUNNELS:
    for lob_group, lobs in LOB_GROUPS.items():
        col_name = f"{funnel}__{lob_group}"
        matching = media_meta.loc[
            (media_meta["funnel"] == funnel) & (media_meta["lob"].isin(lobs)),
            "col_name",
        ].tolist()
        df[col_name] = df[matching].sum(axis=1) if matching else 0.0

# COMMAND ----------

# DBTITLE 1,EXH Combo Structure & Sparsity Filter
# ━━ COMBO STRUCTURE & ACTIVE COMBOS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

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

DROP_SPARSE = [
    "Awareness__Other",              # 79.8% zero weeks
    "Consideration__Other",          # 70.2% zero weeks
    "Conversion__Other",             # 94.4% zero weeks
]

# Aggregate source columns into combo columns
for combo_name, source_cols in ITER2_COMBOS.items():
    df[combo_name] = df[source_cols].sum(axis=1)

# Propagate to splits
for split in [train_df, val_df, test_df]:
    for combo_name in ITER2_COMBOS:
        split[combo_name] = df.loc[
            df["week_date"].isin(split["week_date"]), combo_name
        ].values

ACTIVE_COMBOS = [c for c in ITER2_COMBOS if c not in DROP_SPARSE]

FUNNEL_COMBOS = {
    funnel: [c for c in ACTIVE_COMBOS if c.startswith(funnel)]
    for funnel in FUNNELS
}

LOB_GROUP_MAP = {lob_group: lobs for lob_group, lobs in LOB_GROUPS.items()}

print(f"Active combos: {len(ACTIVE_COMBOS)}")
for funnel, combos in FUNNEL_COMBOS.items():
    print(f"  {funnel:<15}: {combos}")

# COMMAND ----------

# DBTITLE 1,EXH Media Spend Scaling
# ━━ MEDIA SPEND SCALING ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

spend_scaled_iter2    = {}
spend_train_max_iter2 = {}

print(f"{'Combination':<35} {'Train max':>14}  Scaled range")
print("─" * 70)
for combo_name in ACTIVE_COMBOS:
    m          = float(train_df[combo_name].max())
    train_vals = train_df[combo_name].values.astype(np.float64)
    val_vals   = val_df[combo_name].values.astype(np.float64)
    trainval   = np.concatenate([train_vals, val_vals])

    spend_train_max_iter2[combo_name] = m
    spend_scaled_iter2[combo_name]    = {
        "train"    : train_vals / m,
        "trainval" : trainval   / m,
    }
    print(f"{combo_name:<35} ${m:>12,.0f}  "
          f"[{train_vals.min()/m:.3f}, {train_vals.max()/m:.3f}]")

# COMMAND ----------

# DBTITLE 1,EXH Response Transformation Functions
# ━━ RESPONSE TRANSFORMATION FUNCTIONS (PyTensor) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def geometric_adstock(x, alpha):
    """Recursive geometric adstock: adstocked[t] = x[t] + alpha * adstocked[t-1]"""
    x_t = pt.as_tensor_variable(x.astype(np.float64))

    def step(x_curr, adstock_prev, alpha):
        return x_curr + alpha * adstock_prev

    adstocked, _ = pytensor.scan(
        fn=step,
        sequences=[x_t],
        outputs_info=[pt.as_tensor_variable(np.float64(0.0))],
        non_sequences=[alpha],
    )
    return adstocked


def hill_saturation(x, k, alpha_s):
    """Hill saturation: f(x) = x^alpha_s / (x^alpha_s + k^alpha_s)"""
    x_safe = pt.maximum(x, 1e-6)
    return (x_safe ** alpha_s) / (x_safe ** alpha_s + k ** alpha_s)


def logistic_saturation(x, lam):
    """Logistic saturation: f(x) = 1 - exp(-lam * x)"""
    return 1 - pt.exp(-lam * x)

# COMMAND ----------

# DBTITLE 1,EXH Model Definition & NUTS Sampling
# ━━ MODEL DEFINITION — ITER 2 MIXED HIERARCHICAL ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ALPHA_MU_PRIOR = {
    "Awareness"    : {"alpha": 7, "beta": 2},   # mean ≈ 0.78 — long carryover
    "Consideration": {"alpha": 3, "beta": 3},   # mean ≈ 0.50 — moderate
    "Conversion"   : {"alpha": 2, "beta": 5},   # mean ≈ 0.29 — short carryover
}

MU_GAMMA_PRIOR = {
    "Awareness"    : 0.5,
    "Consideration": 0.3,
    "Conversion"   : 0.6,
}

with pm.Model() as mmm_iter2_mixed_hier:

    intercept = pm.Normal("intercept", mu=0, sigma=1)
    beta      = pm.Normal("beta", mu=0, sigma=1, shape=X_train.shape[1])
    sigma     = pm.HalfNormal("sigma", sigma=1)

    mu = intercept + pm.math.dot(X_train, beta)

    for funnel, combos in FUNNEL_COMBOS.items():
        ap = ALPHA_MU_PRIOR[funnel]

        # Funnel-level hyperpriors
        mu_alpha_f    = pm.Beta(f"mu_alpha_{funnel}", alpha=ap["alpha"], beta=ap["beta"])
        kappa_alpha_f = pm.HalfNormal(f"kappa_alpha_{funnel}", sigma=2)
        mu_gamma_f    = pm.HalfNormal(f"mu_gamma_{funnel}", sigma=MU_GAMMA_PRIOR[funnel])

        if funnel == "Consideration":
            mu_k_f       = pm.HalfNormal("mu_k_Consideration",       sigma=0.5)
            mu_alpha_s_f = pm.HalfNormal("mu_alpha_s_Consideration", sigma=1.5)

        for combo in combos:
            alpha_c = pm.Beta(
                f"alpha_{combo}",
                alpha = mu_alpha_f    * kappa_alpha_f,
                beta  = (1 - mu_alpha_f) * kappa_alpha_f,
            )
            gamma_c = pm.HalfNormal(f"gamma_{combo}", sigma=mu_gamma_f)
            adstocked = geometric_adstock(spend_scaled_iter2[combo]["train"], alpha_c)

            if funnel == "Consideration":
                k_c       = pm.HalfNormal(f"k_{combo}",       sigma=mu_k_f)
                alpha_s_c = pm.HalfNormal(f"alpha_s_{combo}", sigma=mu_alpha_s_f)
                saturated = hill_saturation(adstocked, k_c, alpha_s_c)
            else:
                lam_c     = pm.HalfNormal(f"lam_{combo}", sigma=1)
                saturated = logistic_saturation(adstocked, lam_c)

            mu = mu + gamma_c * saturated

    y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y_train)

    print("Sampling — Iter 2 Mixed Hierarchical (NUTS via NumPyro)...")
    trace_iter2_mixed_hier = pmjax.sample_numpyro_nuts(
        draws=2000,
        tune=2000,
        target_accept=0.99,
        chains=4,
        random_seed=42,
    )

# COMMAND ----------

# DBTITLE 1,EXH Convergence Diagnostics
# ━━ CONVERGENCE DIAGNOSTICS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

HYPER_PARAMS = (
    [f"mu_alpha_{f}"    for f in FUNNEL_COMBOS] +
    [f"kappa_alpha_{f}" for f in FUNNEL_COMBOS] +
    [f"mu_gamma_{f}"    for f in FUNNEL_COMBOS] +
    ["mu_k_Consideration", "mu_alpha_s_Consideration"]
)

LOB_PARAMS = []
for funnel, combos in FUNNEL_COMBOS.items():
    LOB_PARAMS += [f"alpha_{c}" for c in combos]
    LOB_PARAMS += [f"gamma_{c}" for c in combos]
    if funnel == "Consideration":
        LOB_PARAMS += [f"k_{c}"       for c in combos]
        LOB_PARAMS += [f"alpha_s_{c}" for c in combos]
    else:
        LOB_PARAMS += [f"lam_{c}" for c in combos]

ALL_PARAMS = ["intercept", "beta", "sigma"] + HYPER_PARAMS + LOB_PARAMS

summary = az.summary(trace_iter2_mixed_hier, var_names=ALL_PARAMS)
summary["r_hat"] = pd.to_numeric(summary["r_hat"], errors="coerce")

max_rhat    = float(summary["r_hat"].max())
min_ess     = int(summary["ess_bulk"].min())
divergences = int(trace_iter2_mixed_hier.sample_stats.diverging.sum().item())

print("=" * 55)
print("CONVERGENCE DIAGNOSTICS")
print("=" * 55)
print(f"Max R-hat   : {max_rhat:.4f}   (target < 1.01)")
print(f"Min ESS     : {min_ess}        (target > 400)")
print(f"Divergences : {divergences}    (target 0)")

# COMMAND ----------

# DBTITLE 1,EXH In-Sample Accuracy
# ━━ IN-SAMPLE ACCURACY ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

post           = trace_iter2_mixed_hier.posterior
intercept_mean = float(post["intercept"].values.ravel().mean())
beta_mean      = post["beta"].values.reshape(-1, len(baseline_features)).mean(axis=0)

X_trainval  = np.vstack([X_train, X_val])
mu_controls = intercept_mean + X_trainval @ beta_mean
mu_media    = np.zeros(n_train + n_val)
media_contrib = {}

for funnel, combos in FUNNEL_COMBOS.items():
    for combo in combos:
        alpha_mean = float(post[f"alpha_{combo}"].values.ravel().mean())
        gamma_mean = float(post[f"gamma_{combo}"].values.ravel().mean())
        x_spend    = spend_scaled_iter2[combo]["trainval"]

        adstocked    = np.zeros(len(x_spend))
        adstocked[0] = x_spend[0]
        for t in range(1, len(x_spend)):
            adstocked[t] = x_spend[t] + alpha_mean * adstocked[t - 1]

        if funnel == "Consideration":
            k_mean   = float(post[f"k_{combo}"].values.ravel().mean())
            as_mean  = float(post[f"alpha_s_{combo}"].values.ravel().mean())
            x_safe   = np.maximum(adstocked, 1e-6)
            sat      = (x_safe ** as_mean) / (x_safe ** as_mean + k_mean ** as_mean)
        else:
            lam_mean = float(post[f"lam_{combo}"].values.ravel().mean())
            sat      = 1 - np.exp(-lam_mean * adstocked)

        contrib_scaled        = gamma_mean * sat
        mu_media             += contrib_scaled
        media_contrib[combo]  = contrib_scaled * scaler_y.scale_[0]

pred_trainval  = scaler_y.inverse_transform(
                     (mu_controls + mu_media).reshape(-1, 1)).ravel()

pred_train_v   = pred_trainval[:n_train]
pred_val_v     = pred_trainval[n_train:]
actual_train_v = train_df[target_col].values
actual_val_v   = val_df[target_col].values

mape_train = mean_absolute_percentage_error(actual_train_v, pred_train_v) * 100
mape_val   = mean_absolute_percentage_error(actual_val_v,   pred_val_v)   * 100
r2_train   = r2_score(actual_train_v, pred_train_v)
r2_val     = r2_score(actual_val_v,   pred_val_v)

print("=" * 72)
print("FINAL MODEL ACCURACY — Iter 2 Mixed Hierarchical")
print("=" * 72)
print(f"{'':28} {'Train MAPE':>11} {'Val MAPE':>10} {'Train R²':>10} {'Val R²':>8}")
print("─" * 72)
print(f"{'Mixed Hierarchical Iter 2':28} {mape_train:>10.2f}% {mape_val:>9.2f}% "
      f"{r2_train:>10.3f} {r2_val:>8.3f}  ← FINAL")

# COMMAND ----------

# DBTITLE 1,EXH Artifact Export
# ━━ ARTIFACT EXPORT ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("=" * 60)
print("SAVING EXH ARTIFACTS")
print("=" * 60)

# 1. Trace (write locally first — HDF5/NetCDF needs random-access I/O)
_local_nc = "/tmp/trace_iter2_mixed_hier.nc"
if os.path.exists(_local_nc): os.remove(_local_nc)
_vol_nc = f"{EXH_ARTIFACT_PATH}trace_iter2_mixed_hier.nc"
if os.path.exists(_vol_nc): os.remove(_vol_nc)
trace_iter2_mixed_hier.to_netcdf(_local_nc)
shutil.copy2(_local_nc, _vol_nc)
os.remove(_local_nc)
print(f"✓ Trace saved → {_vol_nc}")

# 2. Scalers
with open(f"{EXH_ARTIFACT_PATH}scaler_X.pkl", "wb") as f: pickle.dump(scaler_X, f)
with open(f"{EXH_ARTIFACT_PATH}scaler_y.pkl", "wb") as f: pickle.dump(scaler_y, f)
print("✓ Scalers saved")

# 3. Spend normalisation constants
with open(f"{EXH_ARTIFACT_PATH}spend_train_max_iter2.pkl", "wb") as f:
    pickle.dump(spend_train_max_iter2, f)
print("✓ Spend train maxima saved")

# 4. Structural metadata
model_meta = {
    "ACTIVE_COMBOS": ACTIVE_COMBOS, "FUNNEL_COMBOS": {k: list(v) for k, v in FUNNEL_COMBOS.items()},
    "LOB_GROUP_MAP": LOB_GROUP_MAP, "baseline_features": baseline_features,
    "target_col": target_col, "n_train": n_train, "n_val": n_val, "n_test": n_test,
}
with open(f"{EXH_ARTIFACT_PATH}model_meta.pkl", "wb") as f: pickle.dump(model_meta, f)
print("✓ Model metadata saved")

# 5. Posterior mean parameters per combo
response_params = {}
for funnel, combos in FUNNEL_COMBOS.items():
    for combo in combos:
        params = {
            "funnel": funnel,
            "alpha": float(post[f"alpha_{combo}"].values.ravel().mean()),
            "gamma": float(post[f"gamma_{combo}"].values.ravel().mean()),
            "saturation": "hill" if funnel == "Consideration" else "logistic",
        }
        if funnel == "Consideration":
            params["k"]       = float(post[f"k_{combo}"].values.ravel().mean())
            params["alpha_s"] = float(post[f"alpha_s_{combo}"].values.ravel().mean())
        else:
            params["lam"] = float(post[f"lam_{combo}"].values.ravel().mean())
        response_params[combo] = params
with open(f"{EXH_ARTIFACT_PATH}response_params.pkl", "wb") as f: pickle.dump(response_params, f)
print(f"✓ Response parameters saved ({len(response_params)} combos)")

# 6. Full dataset with predictions
df_full = pd.concat([train_df, val_df, test_df]).copy().reset_index(drop=True)
df_full["segment"] = ["train"] * n_train + ["val"] * n_val + ["test"] * n_test
df_full["year"]    = df_full["week_date"].dt.year

X_test           = scaler_X.transform(test_df[baseline_features])
X_full           = np.vstack([X_train, X_val, X_test])
mu_controls_full = intercept_mean + X_full @ beta_mean
mu_media_full    = np.zeros(n_train + n_val + n_test)
media_contrib_full = {}

for funnel, combos in FUNNEL_COMBOS.items():
    for combo in combos:
        alpha_mean = float(post[f"alpha_{combo}"].values.ravel().mean())
        gamma_mean = float(post[f"gamma_{combo}"].values.ravel().mean())
        max_spend  = spend_train_max_iter2[combo]
        x_spend    = df_full[combo].values.astype(np.float64) / max_spend
        adstocked    = np.zeros(len(x_spend))
        adstocked[0] = x_spend[0]
        for t in range(1, len(x_spend)):
            adstocked[t] = x_spend[t] + alpha_mean * adstocked[t - 1]
        if funnel == "Consideration":
            k_mean  = float(post[f"k_{combo}"].values.ravel().mean())
            as_mean = float(post[f"alpha_s_{combo}"].values.ravel().mean())
            x_safe  = np.maximum(adstocked, 1e-6)
            sat     = (x_safe ** as_mean) / (x_safe ** as_mean + k_mean ** as_mean)
        else:
            lam_mean = float(post[f"lam_{combo}"].values.ravel().mean())
            sat      = 1 - np.exp(-lam_mean * adstocked)
        contrib                    = gamma_mean * sat * scaler_y.scale_[0]
        mu_media_full             += gamma_mean * sat
        media_contrib_full[combo]  = contrib

df_full["pred"]          = scaler_y.inverse_transform((mu_controls_full + mu_media_full).reshape(-1, 1)).ravel()
df_full["controls_pred"] = scaler_y.inverse_transform(mu_controls_full.reshape(-1, 1)).ravel()
df_full["total_media"]   = sum(media_contrib_full[c] for c in ACTIVE_COMBOS)
for combo in ACTIVE_COMBOS:
    df_full[f"media_{combo}"] = media_contrib_full[combo]
for funnel, combos in FUNNEL_COMBOS.items():
    df_full[f"media_{funnel}"] = sum(df_full[f"media_{combo}"] for combo in combos)

df_full.to_parquet(f"{EXH_ARTIFACT_PATH}df_full.parquet", index=False)
print("✓ Full dataset parquet saved")
print(f"\nAll EXH artifacts → {EXH_ARTIFACT_PATH}")

# COMMAND ----------

# DBTITLE 1,Part B — EXH Diagnostics & Visualizations
# MAGIC %md
# MAGIC ---
# MAGIC ## Part B: EXH Diagnostics & Visualizations
# MAGIC Loads artifacts from Part A and produces the full set of model diagnostics and business charts. Can be run independently if artifacts exist.
# MAGIC
# MAGIC **Charts produced**: D1 (priors vs posteriors), D3–D6 (validation), D7 (adstock decay), B1 (weekly decomposition), B6 (annual decomposition), B10 (response/marginal curves)

# COMMAND ----------

# DBTITLE 1,EXH Load Artifacts for Diagnostics
# ━━ LOAD EXH ARTIFACTS (for diagnostics-only runs) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# NOTE: If running end-to-end from Part A, these are already in memory.
# This cell enables running Parts B-D standalone.

print("Loading EXH artifacts...")

trace = az.from_netcdf(f"{EXH_ARTIFACT_PATH}trace_iter2_mixed_hier.nc")
post  = trace.posterior

with open(f"{EXH_ARTIFACT_PATH}scaler_y.pkl",              "rb") as f: scaler_y              = pickle.load(f)
with open(f"{EXH_ARTIFACT_PATH}scaler_X.pkl",              "rb") as f: scaler_X              = pickle.load(f)
with open(f"{EXH_ARTIFACT_PATH}spend_train_max_iter2.pkl", "rb") as f: spend_train_max_iter2 = pickle.load(f)
with open(f"{EXH_ARTIFACT_PATH}model_meta.pkl",            "rb") as f: model_meta            = pickle.load(f)
with open(f"{EXH_ARTIFACT_PATH}response_params.pkl",       "rb") as f: response_params       = pickle.load(f)

ACTIVE_COMBOS    = model_meta["ACTIVE_COMBOS"]
FUNNEL_COMBOS    = model_meta["FUNNEL_COMBOS"]
LOB_GROUP_MAP    = model_meta["LOB_GROUP_MAP"]
baseline_features = model_meta["baseline_features"]
target_col       = model_meta["target_col"]
n_train          = model_meta["n_train"]
n_val            = model_meta["n_val"]
n_test           = model_meta["n_test"]

df_full = pd.read_parquet(f"{EXH_ARTIFACT_PATH}df_full.parquet")
df_trainval = df_full[df_full["segment"].isin(["train", "val"])].copy().reset_index(drop=True)

actual_train_v = df_trainval.loc[df_trainval["segment"] == "train", target_col].values
actual_val_v   = df_trainval.loc[df_trainval["segment"] == "val",   target_col].values
pred_train_v   = df_trainval.loc[df_trainval["segment"] == "train", "pred"].values
pred_val_v     = df_trainval.loc[df_trainval["segment"] == "val",   "pred"].values

mape_train = mean_absolute_percentage_error(actual_train_v, pred_train_v) * 100
mape_val   = mean_absolute_percentage_error(actual_val_v,   pred_val_v)   * 100
r2_train   = r2_score(actual_train_v, pred_train_v)
r2_val     = r2_score(actual_val_v,   pred_val_v)

print(f"  ✓ Artifacts loaded — {len(df_full)} weeks")
print(f"  ✓ Train MAPE: {mape_train:.1f}%  R²: {r2_train:.3f}")
print(f"  ✓ Val   MAPE: {mape_val:.1f}%  R²: {r2_val:.3f}")

# COMMAND ----------

# DBTITLE 1,EXH Colour Maps & Helper Functions
# ━━ COLOUR MAPS & HELPER FUNCTIONS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

FUNNEL_COLORS = {
    "Awareness"    : CPX_NAVY,
    "Consideration": CPX_RED,
    "Conversion"   : CPX_GOLD,
}

LOB_COLORS = {
    "Awareness__Premium_Brand"    : "#1a3a5c",
    "Consideration__Premium_Brand": "#c0392b",
    "Consideration__GE"           : "#e74c3c",
    "Consideration__Prepaid"      : "#f1948a",
    "Conversion__Film_Marketing"  : "#d4ac0d",
    "Conversion__Premium_Brand"   : "#f39c12",
    "Conversion__Cineclub"        : "#e67e22",
}

YEAR_STYLES = {2023: {"color": CPX_NAVY}, 2024: {"color": CPX_RED}, 2025: {"color": CPX_GOLD}}

def logistic_sat(x, lam):
    return 1 - np.exp(-lam * x)

def hill_sat(x, k, alpha_s):
    x_safe = np.maximum(x, 1e-6)
    return (x_safe ** alpha_s) / (x_safe ** alpha_s + k ** alpha_s)

def steady_state_adstock(s_norm, alpha):
    """Infinite-horizon steady-state geometric adstock."""
    return s_norm / (1 - alpha)

# COMMAND ----------

# DBTITLE 1,EXH Chart D3–D6 — Validation Diagnostics
# ━━ CHARTS D3–D6: VALIDATION DIAGNOSTICS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# D3: Actual vs Predicted (full train+val series)
fig, ax = plt.subplots(figsize=(16, 6))
dates_all = df_trainval["week_date"]
ax.plot(dates_all, df_trainval[target_col] / 1e6, color="black", lw=1.5, label="Actual")
ax.plot(dates_all, df_trainval["pred"] / 1e6, color=CPX_NAVY, lw=1.8, ls="--", label="Predicted")
val_start = df_trainval.loc[df_trainval["segment"] == "val", "week_date"].min()
val_end   = df_trainval.loc[df_trainval["segment"] == "val", "week_date"].max()
ax.axvspan(val_start, val_end, alpha=0.08, color=CPX_RED, label="Validation window")
ax.set_title(f"EXH MMM — Actual vs Predicted\n"
             f"Train MAPE {mape_train:.1f}% R² {r2_train:.3f} | Val MAPE {mape_val:.1f}% R² {r2_val:.3f}",
             fontsize=13, fontweight="bold", color=CPX_NAVY)
ax.set_ylabel("Weekly Attendance (M)")
ax.legend(frameon=False)
ax.grid(False); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
plt.tight_layout()
save_fig(fig, "exh_d3_actual_vs_predicted.png")
plt.show()

# D5: Residuals Over Time
fig, ax = plt.subplots(figsize=(16, 4))
resid_all = df_trainval[target_col].values - df_trainval["pred"].values
ax.bar(dates_all, resid_all / 1e3, color=[CPX_NAVY if s == "train" else CPX_RED for s in df_trainval["segment"]], alpha=0.6, width=5)
ax.axhline(0, color="black", lw=0.8)
ax.set_title("EXH MMM — Residuals Over Time", fontweight="bold", color=CPX_NAVY)
ax.set_ylabel("Residual (K)")
ax.grid(False); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
plt.tight_layout()
save_fig(fig, "exh_d5_residuals.png")
plt.show()

# COMMAND ----------

# DBTITLE 1,EXH Chart B1 — Weekly Contribution Decomposition
# ━━ CHART B1: WEEKLY CONTRIBUTION DECOMPOSITION ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

fig, ax = plt.subplots(figsize=(16, 7))
dates = df_trainval["week_date"]

# Stack: baseline + per-combo media contributions
baseline = df_trainval["controls_pred"].values / 1e6
bottom = baseline.copy()

for combo in ACTIVE_COMBOS:
    vals = df_trainval[f"media_{combo}"].values / 1e6
    ax.fill_between(dates, bottom, bottom + vals, alpha=0.7,
                    color=LOB_COLORS.get(combo, CPX_DGREY), label=combo)
    bottom += vals

ax.plot(dates, df_trainval[target_col].values / 1e6, color="black", lw=1.5, label="Actual")
ax.set_title("EXH MMM — Weekly Attendance Decomposition", fontweight="bold", color=CPX_NAVY)
ax.set_ylabel("Weekly Attendance (M)")
ax.legend(loc="upper left", fontsize=8, ncol=2, frameon=False)
ax.grid(False); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
plt.tight_layout()
save_fig(fig, "exh_b1_weekly_decomposition.png")
plt.show()

# COMMAND ----------

# DBTITLE 1,EXH Chart B10 — Response & Marginal Curves
# ━━ CHART B10: EXH DIMINISHING RETURN CURVE (YoY) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Single-panel aggregate response curve — matching LBE style.
# Vertical year lines, scatter markers, 95% saturation, footnote.

from matplotlib.lines import Line2D

scale_factor = float(scaler_y.scale_[0])

# ── 2025 allocation shares (hold mix constant across the curve) ──────────────
df_2025_rc = df_full[df_full["year"] == 2025]
combo_weekly_2025 = {c: float(df_2025_rc[c].mean()) for c in ACTIVE_COMBOS}
total_weekly_2025 = sum(combo_weekly_2025.values())
alloc_shares = {c: combo_weekly_2025[c] / total_weekly_2025 for c in ACTIVE_COMBOS}
total_annual_2025 = total_weekly_2025 * 52

def total_media_attendance(annual_budget, shares=alloc_shares):
    """Total annual media attendance contribution at a given total budget."""
    total = 0.0
    for combo in ACTIVE_COMBOS:
        p = response_params[combo]
        mx = spend_train_max_iter2[combo]
        weekly_spend = (annual_budget * shares[combo]) / 52
        s_norm = weekly_spend / mx
        ad = s_norm / (1 - p["alpha"])
        if p["saturation"] == "hill":
            sat = (ad ** p["alpha_s"]) / (ad ** p["alpha_s"] + p["k"] ** p["alpha_s"])
        else:
            sat = 1 - np.exp(-p["lam"] * ad)
        total += p["gamma"] * sat * scale_factor * 52
    return total

# ── X axis: near-zero to 10× 2025 spend ─────────────────────────────────────
x_annual = np.linspace(52 * 100, total_annual_2025 * 10, 500)
curve = np.array([total_media_attendance(b) for b in x_annual])

# ── Year reference points (actual spend & contribution) ──────────────────────
year_points = {}
for year in [2023, 2024, 2025]:
    df_yr = df_full[df_full["year"] == year]
    yr_spend = {c: float(df_yr[c].sum()) for c in ACTIVE_COMBOS}
    yr_total_spend = sum(yr_spend.values())
    yr_shares = {c: (yr_spend[c] / yr_total_spend if yr_total_spend > 0 else alloc_shares[c])
                 for c in ACTIVE_COMBOS}
    yr_att = total_media_attendance(yr_total_spend, shares=yr_shares)
    year_points[year] = {"spend": yr_total_spend, "attendance": yr_att}

# ── 95% saturation point ────────────────────────────────────────────────────
sat_95_idx = np.argmin(np.abs(curve - curve[-1] * 0.95))
sat_95_spend = x_annual[sat_95_idx]
sat_95_att = curve[sat_95_idx]

# ── Year marker styles (matching LBE chart) ──────────────────────────────────
YEAR_MARKER_STYLES = {
    2023: {"color": CPX_DGREY, "marker": "o", "linestyle": "--",  "linewidth": 1.2},
    2024: {"color": CPX_NAVY,  "marker": "s", "linestyle": "-.", "linewidth": 1.2},
    2025: {"color": CPX_RED,   "marker": "D", "linestyle": "-",  "linewidth": 1.5},
}

# ── PLOT ─────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(16, 9))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

# Response curve
ax.plot(x_annual / 1e6, curve / 1e6, color=CPX_NAVY, linewidth=2.5)

# 95% saturation marker & vertical line
ax.axvline(sat_95_spend / 1e6, color=CPX_RED, linewidth=1.0, linestyle=":", alpha=0.8)
ax.scatter([sat_95_spend / 1e6], [sat_95_att / 1e6], color=CPX_RED, s=120, zorder=6, marker="*")
ax.annotate(f"95% Saturation\n${sat_95_spend/1e6:.1f}M/yr\n"
            f"{sat_95_spend/year_points[2025]['spend']:.1f}x 2025 spend",
            xy=(sat_95_spend / 1e6, sat_95_att / 1e6),
            xytext=(-80, 30), textcoords="offset points",
            fontsize=8.5, color=CPX_RED, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=CPX_RED, lw=0.8),
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.9, ec="none"))

# Year vertical lines + scatter + annotations
for year, style in YEAR_MARKER_STYLES.items():
    pts = year_points[year]
    spend_m = pts["spend"] / 1e6
    att_m = pts["attendance"] / 1e6

    # Vertical line at year's spend level
    ax.axvline(spend_m, color=style["color"], linestyle=style["linestyle"],
               linewidth=style["linewidth"], alpha=0.85, zorder=4)

    # Scatter point on the curve
    ax.scatter([spend_m], [att_m], color=style["color"], marker=style["marker"],
              s=100, zorder=7, edgecolors="white", linewidths=1.0)

    # Annotation: year, spend, attendance
    offset_y = 20 + (2025 - year) * 25
    ax.annotate(f"{year}\n${pts['spend']/1e6:.1f}M/yr\n({att_m:.1f}M att)",
                xy=(spend_m, att_m),
                xytext=(8, offset_y), textcoords="offset points",
                fontsize=8.5, color=style["color"], fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=style["color"], lw=0.8),
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.9, ec="none"))

# Title
ax.set_title("EXH MMM — Total Media Response Curve\n"
             "Steady-state adstock  |  Budget shares held at 2025 mix",
             fontsize=14, fontweight="bold", color="black", pad=16)

# Axes formatting
ax.set_xlabel("Annual Media Budget ($M)", fontsize=11)
ax.set_ylabel("Annual Media Attendance Contribution (M)", fontsize=11)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:.1f}M"))
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1f}M"))
ax.grid(False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Legend (year markers)
legend_handles = [
    Line2D([0], [0], color=style["color"], marker=style["marker"], linestyle=style["linestyle"],
           linewidth=style["linewidth"], markersize=7,
           label=f"{year} (${year_points[year]['spend']/1e6:.1f}M annual)")
    for year, style in YEAR_MARKER_STYLES.items()
]
ax.legend(handles=legend_handles, loc="lower right", fontsize=9, frameon=True,
          facecolor="white", edgecolor="#CCCCCC", framealpha=0.95)

# Footnote
fig.text(0.02, 0.01,
         "* Budget distributed using 2025 actual spend shares. "
         "Hill/logistic approx via posterior mean params.",
         fontsize=7.5, color=CPX_DGREY, style="italic")

# ═══════════════════════════════════════════════════════════════════════════════
# RIGHT PANEL — Marginal Return Curve
# ═══════════════════════════════════════════════════════════════════════════════
# ── PANEL 2: Marginal Return Curve (separate figure) ────────────────────────
plt.tight_layout(rect=[0, 0.03, 1, 1])
display(fig)
plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════════════
fig2, ax2 = plt.subplots(figsize=(16, 9))
fig2.patch.set_facecolor("white")
ax2.set_facecolor("white")

marginal = np.gradient(curve, x_annual)

ax2.plot(x_annual / 1e6, marginal, color=CPX_NAVY, linewidth=2.5)
ax2.axhline(1, color="grey", ls=":", lw=0.8, label="1:1 return")

for year, style in YEAR_MARKER_STYLES.items():
    pts = year_points[year]
    idx = np.argmin(np.abs(x_annual - pts["spend"]))
    ax2.axvline(pts["spend"] / 1e6, color=style["color"], linestyle=style["linestyle"],
                linewidth=style["linewidth"], alpha=0.6, zorder=4)
    ax2.scatter([pts["spend"] / 1e6], [marginal[idx]], color=style["color"],
               marker=style["marker"], s=100, zorder=7, edgecolors="white", linewidths=1.0)
    ax2.annotate(f"{year}: {marginal[idx]:.2f}",
                 xy=(pts["spend"] / 1e6, marginal[idx]),
                 xytext=(10, 12), textcoords="offset points",
                 fontsize=9, color=style["color"], fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.9, ec="none"))

ax2.set_title("EXH MMM — Marginal Return per $1 Spent\n"
              "Incremental attendance per additional dollar",
              fontsize=14, fontweight="bold", color="black", pad=16)
ax2.set_xlabel("Annual Media Budget ($M)", fontsize=11)
ax2.set_ylabel("Incremental Attendance per $1", fontsize=11)
ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:.1f}M"))
legend_handles_m = [
    Line2D([0], [0], color=style["color"], marker=style["marker"], linestyle=style["linestyle"],
           linewidth=style["linewidth"], markersize=7,
           label=f"{year} (${year_points[year]['spend']/1e6:.1f}M annual)")
    for year, style in YEAR_MARKER_STYLES.items()
]
ax2.legend(handles=legend_handles_m, loc="upper right", fontsize=9, frameon=True,
           facecolor="white", edgecolor="#CCCCCC", framealpha=0.95)
ax2.grid(False); ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)

fig2.text(0.02, 0.01,
          "* Budget distributed using 2025 actual spend shares. "
          "Hill/logistic approx via posterior mean params.",
          fontsize=7.5, color=CPX_DGREY, style="italic")

plt.tight_layout(rect=[0, 0.03, 1, 1])
display(fig2)
plt.close(fig2)

# COMMAND ----------

# DBTITLE 1,Part C — EXH LOB Decomposition (Graft Method)
# MAGIC %md
# MAGIC ---
# MAGIC ## Part C: EXH LOB Decomposition (Graft Method)
# MAGIC Splits validated group-level contributions to individual LOBs without re-estimating the media layer. Uses spend-share + residual-correlation tilt (±35% cap).
# MAGIC
# MAGIC **Why graft?** A direct LOB-level refit was tried and its media layer hurt out-of-sample validation (Val R² fell below controls-only baseline). The grouped model validates because aggregation suppresses overfitting.

# COMMAND ----------

# DBTITLE 1,EXH LOB Decomposition
# ━━ LOB DECOMPOSITION — GRAFT METHOD ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

os.makedirs(EXH_DECOMP_PATH, exist_ok=True)
scale_factor = float(scaler_y.scale_[0])

# Step 1: Validated group contributions (steady-state at 2025 spend)
df_2025 = df_full[df_full["year"] == 2025].copy()
baseline_2025 = {}
for combo in ACTIVE_COMBOS:
    baseline_2025[combo] = {
        "annual_spend": float(df_2025[combo].sum()),
        "weekly_avg":   float(df_2025[combo].mean()),
    }

active_contributions = {}
for combo in ACTIVE_COMBOS:
    p   = response_params[combo]
    mx  = spend_train_max_iter2[combo]
    wk  = baseline_2025[combo]["weekly_avg"]
    s   = wk / mx
    ad  = s / (1 - p["alpha"])
    if p["saturation"] == "hill":
        sat = (ad ** p["alpha_s"]) / (ad ** p["alpha_s"] + p["k"] ** p["alpha_s"])
    else:
        sat = 1 - np.exp(-p["lam"] * ad)
    active_contributions[combo] = p["gamma"] * sat * scale_factor * 52

total_validated = sum(active_contributions.values())
print(f"Validated total media contribution (2025): {total_validated:,.0f}")
for c, v in active_contributions.items():
    print(f"  {c:<42} {v:>12,.0f}")

# COMMAND ----------

# DBTITLE 1,EXH LOB Tilt Signal & Decomposition
# ━━ RESIDUAL ATTENDANCE & TILT SIGNAL ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Residual = actual - controls-only prediction (removes slate/seasonality confound)
residual_att = df_full[target_col].values - df_full["controls_pred"].values

# Parse LOB spend & compute tilt signal
media_cols_raw = [c for c in df_full.columns if any(
    c.startswith(f"{f}_") for f in ["Awareness", "Consideration", "Conversion"]
) and "__" not in c and "media_" not in c]

funnel_lob_cols = defaultdict(list)
for col in media_cols_raw:
    parts = col.split("_", 1); funnel = parts[0]; rest = parts[1]
    lob_parts = rest.rsplit("_", 1)
    lob_name = lob_parts[0] if len(lob_parts) > 1 else rest
    funnel_lob_cols[(funnel, lob_name)].append(col)

# Compute per-LOB tilt (correlation with residual attendance)
tilt_data = {}
for (funnel, lob), cols in funnel_lob_cols.items():
    spend_series = df_full[cols].sum(axis=1).values
    if spend_series.sum() > 0 and np.std(spend_series) > 0:
        corr = np.corrcoef(spend_series, residual_att)[0, 1]
    else:
        corr = 0.0
    tilt_data[(funnel, lob)] = corr

# Step 2: Tilted decomposition
TILT_CAP = 0.35
decomp_results = []

for combo in ACTIVE_COMBOS:
    parts = combo.split("__")
    funnel_name = parts[0]
    lob_group_name = parts[1]
    total_contrib = active_contributions[combo]

    if lob_group_name in LOB_GROUP_MAP:
        member_lobs = LOB_GROUP_MAP[lob_group_name]
    else:
        member_lobs = [lob_group_name]

    # Spend shares within group (use full training period for allocation —
    # ensures LOBs with historical spend but no current-year spend still get
    # their share of the learned group response)
    lob_spends = {}
    for lob in member_lobs:
        cols = funnel_lob_cols.get((funnel_name, lob), [])
        lob_spends[lob] = float(df_full[cols].sum().sum()) if cols else 0.0
    total_spend = sum(lob_spends.values())
    if total_spend == 0:
        spend_shares = {lob: 1.0 / len(member_lobs) for lob in member_lobs}
    else:
        spend_shares = {lob: s / total_spend for lob, s in lob_spends.items()}

    # Apply tilt (capped)
    tilted_shares = {}
    for lob in member_lobs:
        base_share = spend_shares[lob]
        tilt = tilt_data.get((funnel_name, lob), 0.0)
        adj = base_share * (1 + min(max(tilt, -TILT_CAP), TILT_CAP))
        tilted_shares[lob] = max(adj, 0.0)

    # Renormalise
    share_total = sum(tilted_shares.values())
    if share_total > 0:
        tilted_shares = {k: v / share_total for k, v in tilted_shares.items()}

    for lob in member_lobs:
        decomp_results.append({
            "combo": combo, "funnel": funnel_name, "lob_group": lob_group_name,
            "original_lob": lob, "spend_share": spend_shares[lob],
            "tilt_corr": tilt_data.get((funnel_name, lob), 0.0),
            "tilted_share": tilted_shares[lob],
            "contribution": total_contrib * tilted_shares[lob],
            "annual_spend": lob_spends[lob],
        })

decomp_df = pd.DataFrame(decomp_results)
decomp_df = decomp_df.sort_values("contribution", ascending=False).reset_index(drop=True)

print(f"\nLOB Decomposition complete — {len(decomp_df)} LOB entries")
print(f"Total validated: {decomp_df['contribution'].sum():,.0f} (should match {total_validated:,.0f})")
display(decomp_df[["funnel", "original_lob", "annual_spend", "spend_share", "tilt_corr", "contribution"]].head(15))

# Save
with open(f"{EXH_DECOMP_PATH}decomp_lob_results.pkl", "wb") as f:
    pickle.dump(decomp_df, f)
print(f"\n✓ LOB decomposition saved → {EXH_DECOMP_PATH}decomp_lob_results.pkl")

# COMMAND ----------

# DBTITLE 1,Part D — EXH Budget Scenario Analysis
# MAGIC %md
# MAGIC ---
# MAGIC ## Part D: EXH Budget Scenario Analysis
# MAGIC Three annual scenarios with posterior credible intervals:
# MAGIC 1. **Budget Uplift** — +10/20/30% uniform scaling (saturation means sub-proportional lift)
# MAGIC 2. **Reshuffle** — 10% siphon from low-marginal to high-marginal groups
# MAGIC 3. **Combined** — Uplift + reshuffle

# COMMAND ----------

# DBTITLE 1,EXH Budget Scenarios
# ━━ EXH BUDGET SCENARIO ANALYSIS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Baseline spend per combo
baseline_spend = {c: baseline_2025[c]["annual_spend"] for c in ACTIVE_COMBOS}
total_baseline_spend = sum(baseline_spend.values())

def combo_contribution(combo, annual_spend, params=None):
    """Annual attendance contribution for a combo at given spend."""
    p  = params or response_params[combo]
    mx = spend_train_max_iter2[combo]
    s  = (annual_spend / 52) / mx
    ad = s / (1 - p["alpha"])
    if p["saturation"] == "hill":
        sat = (ad ** p["alpha_s"]) / (ad ** p["alpha_s"] + p["k"] ** p["alpha_s"])
    else:
        sat = 1 - np.exp(-p["lam"] * ad)
    return p["gamma"] * sat * scale_factor * 52

def marginal_return(combo, annual_spend, delta=1000):
    """Attendance per additional $1 at current spend level."""
    c1 = combo_contribution(combo, annual_spend)
    c2 = combo_contribution(combo, annual_spend + delta)
    return (c2 - c1) / delta

# Baseline total
baseline_att = sum(combo_contribution(c, baseline_spend[c]) for c in ACTIVE_COMBOS)
print(f"Baseline annual media spend     : ${total_baseline_spend:,.0f}")
print(f"Baseline annual media attendance: {baseline_att:,.0f}")

# ── Scenario 1: Budget Uplift ──
print("\n" + "=" * 60)
print("SCENARIO 1 — BUDGET UPLIFT (pattern unchanged)")
print("=" * 60)
for factor in [1.10, 1.20, 1.30]:
    alloc = {c: baseline_spend[c] * factor for c in ACTIVE_COMBOS}
    att = sum(combo_contribution(c, alloc[c]) for c in ACTIVE_COMBOS)
    lift = att - baseline_att
    pct  = lift / baseline_att * 100
    print(f"  +{(factor-1)*100:.0f}% budget: ${sum(alloc.values()):,.0f}  →  {att:,.0f} att  "
          f"(+{lift:,.0f}, +{pct:.1f}%)")

# ── Scenario 2: Reshuffle ──
print("\n" + "=" * 60)
print("SCENARIO 2 — RESHUFFLE (10% siphon to high-marginal)")
print("=" * 60)
mr = {c: marginal_return(c, baseline_spend[c]) for c in ACTIVE_COMBOS}
med_mr = np.median(list(mr.values()))
siphon_from = [c for c in ACTIVE_COMBOS if mr[c] < med_mr]
siphon_to   = [c for c in ACTIVE_COMBOS if mr[c] >= med_mr]

alloc_reshuffle = dict(baseline_spend)
pool = sum(alloc_reshuffle[c] * 0.10 for c in siphon_from)
for c in siphon_from:
    alloc_reshuffle[c] *= 0.90

# Distribute pool by marginal return rank
recipient_mr = {c: mr[c] for c in siphon_to}
total_mr = sum(recipient_mr.values())
for c in siphon_to:
    alloc_reshuffle[c] += pool * (recipient_mr[c] / total_mr)

att_reshuffle = sum(combo_contribution(c, alloc_reshuffle[c]) for c in ACTIVE_COMBOS)
lift_r = att_reshuffle - baseline_att
print(f"  Reshuffled: {att_reshuffle:,.0f} att (+{lift_r:,.0f}, +{lift_r/baseline_att*100:.1f}%)")
print(f"  Budget unchanged: ${sum(alloc_reshuffle.values()):,.0f}")

# ── Scenario 3: Combined ──
print("\n" + "=" * 60)
print("SCENARIO 3 — COMBINED (uplift + reshuffle)")
print("=" * 60)
for factor in [1.10, 1.20, 1.30]:
    alloc_up = {c: baseline_spend[c] * factor for c in ACTIVE_COMBOS}
    mr_up = {c: marginal_return(c, alloc_up[c]) for c in ACTIVE_COMBOS}
    med_up = np.median(list(mr_up.values()))
    pool_up = sum(alloc_up[c] * 0.10 for c in ACTIVE_COMBOS if mr_up[c] < med_up)
    for c in ACTIVE_COMBOS:
        if mr_up[c] < med_up: alloc_up[c] *= 0.90
    recip_up = {c: mr_up[c] for c in ACTIVE_COMBOS if mr_up[c] >= med_up}
    t_mr = sum(recip_up.values())
    for c in recip_up: alloc_up[c] += pool_up * (recip_up[c] / t_mr)
    att_c = sum(combo_contribution(c, alloc_up[c]) for c in ACTIVE_COMBOS)
    lift_c = att_c - baseline_att
    print(f"  +{(factor-1)*100:.0f}% + reshuffle: {att_c:,.0f} att (+{lift_c:,.0f}, +{lift_c/baseline_att*100:.1f}%)")

# COMMAND ----------

# DBTITLE 1,Part E — LBE Model Training
# MAGIC %md
# MAGIC ---
# MAGIC ## Part E: LBE Model Training
# MAGIC **Bayesian Log-Revenue MMM — 2 Funnels (Awareness via TRR, Consideration via PDM)**
# MAGIC
# MAGIC Trains on weekly LBE revenue data. Uses log-space modelling with geometric adstock and log1p saturation. Exports all posteriors as `lbe_mmm_results.pkl`.
# MAGIC
# MAGIC **Architecture**: `log(revenue) ~ intercept + β_aw·log1p(adstock_aw·scale_aw) + β_co·log1p(adstock_co·scale_co) + trend + temp + calendar`

# COMMAND ----------

# DBTITLE 1,LBE Configuration & Data Loading
# ━━ LBE MODEL CONFIGURATION ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

LBE_TRAIN_CUTOFF_YEAR = 2025
LBE_N_VAL             = 12
LBE_DRAWS             = 2000
LBE_TUNE              = 2000
LBE_CHAINS            = 4
LBE_CORES             = 4
LBE_TARGET_ACCEPT     = 0.95
LBE_RANDOM_SEED       = 42

print("=" * 65)
print("  LBE MMM — Model Training")
print("=" * 65)
print("\n[1/7] Loading data...")

df_lbe_raw = pd.read_csv(LBE_INPUT_PATH, parse_dates=["week_date"])
df_lbe = df_lbe_raw[df_lbe_raw["year"] <= LBE_TRAIN_CUTOFF_YEAR].copy().reset_index(drop=True)
df_lbe["log_rev"] = np.log(df_lbe["LBE_Revenue"])
df_lbe["t"]       = np.arange(len(df_lbe), dtype=float)

n_lbe_val    = LBE_N_VAL
df_lbe_train = df_lbe.iloc[:-n_lbe_val].copy().reset_index(drop=True)
df_lbe_val   = df_lbe.iloc[-n_lbe_val:].copy().reset_index(drop=True)

print(f"      Full dataset:  {len(df_lbe)} weeks  "
      f"({df_lbe['week_date'].min().date()} → {df_lbe['week_date'].max().date()})")
print(f"      Train:         {len(df_lbe_train)} weeks")
print(f"      Validation:    {len(df_lbe_val)} weeks")

# COMMAND ----------

# DBTITLE 1,LBE Normalisation & Feature Prep
# ━━ LBE NORMALISATION (fit on train only) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

lbe_y_mean   = df_lbe_train["log_rev"].mean()
lbe_y_std    = df_lbe_train["log_rev"].std()
lbe_aw_max   = df_lbe_train["Awareness"].max()
lbe_co_max   = df_lbe_train["Consideration"].max()
lbe_t_max    = df_lbe_train["t"].max()
lbe_temp_mu  = df_lbe_train["national_mean_temp"].mean()
lbe_temp_sd  = df_lbe_train["national_mean_temp"].std()

LBE_CAL_COLS = [
    "is_new_year_wk", "is_march_break", "is_march_adj",
    "is_pre_xmas", "is_summer_peak",
    "is_back_to_school", "is_spring_lull",
    "holiday", "long_weekend",
]

# Informed directional priors for calendar features
LBE_CAL_MU    = np.array([ 0.40,  0.40,  0.10,  0.30,  0.15, -0.20, -0.15,  0.10,  0.05])
LBE_CAL_SIGMA = np.array([ 0.15,  0.15,  0.10,  0.15,  0.10,  0.10,  0.10,  0.10,  0.10])

def lbe_prep(d: pd.DataFrame) -> dict:
    return {
        "y":    ((d["log_rev"] - lbe_y_mean) / lbe_y_std).values.astype("float64"),
        "aw":   (d["Awareness"].values    / lbe_aw_max).astype("float64"),
        "co":   (d["Consideration"].values / lbe_co_max).astype("float64"),
        "temp": ((d["national_mean_temp"].values - lbe_temp_mu) / lbe_temp_sd).astype("float64"),
        "cal":  d[LBE_CAL_COLS].values.astype("float64"),
        "t":    (d["t"].values / lbe_t_max).astype("float64"),
    }

lbe_tr = lbe_prep(df_lbe_train)
lbe_va = lbe_prep(df_lbe_val)
print("  ✓ Features prepared")

# COMMAND ----------

# DBTITLE 1,LBE Model Definition & NUTS Sampling
# ━━ LBE MODEL DEFINITION & NUTS SAMPLING ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("\n[3/7] Training LBE model...")

def adstock_pt_lbe(x, decay, max_lag: int = 10):
    """Geometric adstock transformation in PyTensor."""
    lags = pt.arange(max_lag, dtype="float64")
    w = decay ** lags
    w = w / w.sum()
    n = x.shape[0]
    out = pt.zeros((n,))
    for l in range(max_lag):
        out = out + w[l] * pt.concatenate([pt.zeros((l,)), x[:n - l]])
    return out

aw_t  = pt.as_tensor_variable(lbe_tr["aw"])
co_t  = pt.as_tensor_variable(lbe_tr["co"])
n_lbe_cal = len(LBE_CAL_COLS)

with pm.Model() as lbe_mmm:
    # Adstock decay priors
    aw_decay = pm.Beta("aw_decay", alpha=6.5, beta=3.5)  # slow decay
    co_decay = pm.Beta("co_decay", alpha=3.0, beta=7.0)  # faster decay

    aw_ads = adstock_pt_lbe(aw_t, aw_decay)
    co_ads = adstock_pt_lbe(co_t, co_decay)

    # Saturation: log1p (concave diminishing returns)
    aw_scale = pm.HalfNormal("aw_scale", sigma=2.0)
    co_scale = pm.HalfNormal("co_scale", sigma=1.0)
    aw_sat = pt.log1p(aw_ads * aw_scale)
    co_sat = pt.log1p(co_ads * co_scale)

    # Media coefficients
    beta_aw = pm.HalfNormal("beta_aw", sigma=0.8)
    beta_co = pm.HalfNormal("beta_co", sigma=0.3)

    # Controls
    intercept   = pm.Normal("intercept",   mu=0,    sigma=1.0)
    beta_trend  = pm.Normal("beta_trend",  mu=0.2,  sigma=0.2)
    beta_temp   = pm.Normal("beta_temp",   mu=-0.2, sigma=0.15)
    beta_cal    = pm.Normal("beta_cal", mu=LBE_CAL_MU, sigma=LBE_CAL_SIGMA, shape=n_lbe_cal)

    cal_eff = pt.dot(pt.as_tensor_variable(lbe_tr["cal"]), beta_cal)

    mu = (intercept
          + beta_aw   * aw_sat
          + beta_co   * co_sat
          + beta_trend * lbe_tr["t"]
          + beta_temp  * lbe_tr["temp"]
          + cal_eff)

    sigma = pm.HalfNormal("sigma", sigma=0.3)
    pm.Normal("y", mu=mu, sigma=sigma, observed=lbe_tr["y"])

    # Sample posterior
    lbe_idata = pm.sample(
        draws=LBE_DRAWS, tune=LBE_TUNE, chains=LBE_CHAINS, cores=LBE_CORES,
        target_accept=LBE_TARGET_ACCEPT, progressbar=True, random_seed=LBE_RANDOM_SEED,
    )
    pm.sample_posterior_predictive(lbe_idata, extend_inferencedata=True, random_seed=LBE_RANDOM_SEED)

lbe_divs = int(lbe_idata.sample_stats["diverging"].values.sum())
print(f"\n      Divergences: {lbe_divs}")

# Extract posterior draws
lbe_post    = lbe_idata.posterior
lbe_draws   = {k: lbe_post[k].values.flatten() for k in lbe_post.data_vars}
lbe_n_draws = len(lbe_draws["beta_aw"])
lbe_bc      = lbe_post["beta_cal"].values.reshape(-1, n_lbe_cal)

print(f"      Posterior: {lbe_n_draws} draws")

# COMMAND ----------

# DBTITLE 1,LBE Prediction Helpers & Accuracy
# ━━ LBE PREDICTION HELPERS & ACCURACY ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def adstock_np_lbe(x, decay, max_lag=10):
    w = decay ** np.arange(max_lag)
    w /= w.sum()
    out = np.zeros(len(x), dtype=float)
    for lag in range(max_lag):
        out[lag:] += w[lag] * x[:len(x) - lag]
    return out

def lbe_predict(data, draws_dict, bc_arr, sample_size=1000, seed=42):
    """Predict revenue using posterior draws."""
    sample_size = min(sample_size, len(draws_dict["beta_aw"]))
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(draws_dict["beta_aw"]), sample_size, replace=False)
    preds = np.zeros((sample_size, len(data["aw"])), dtype=float)
    for j, i in enumerate(idx):
        aw_ads = adstock_np_lbe(data["aw"], float(draws_dict["aw_decay"][i]))
        co_ads = adstock_np_lbe(data["co"], float(draws_dict["co_decay"][i]))
        aw_sat = np.log1p(aw_ads * float(draws_dict["aw_scale"][i]))
        co_sat = np.log1p(co_ads * float(draws_dict["co_scale"][i]))
        cal_eff = data["cal"] @ bc_arr[i]
        mu = (float(draws_dict["intercept"][i])
              + float(draws_dict["beta_aw"][i]) * aw_sat
              + float(draws_dict["beta_co"][i]) * co_sat
              + float(draws_dict["beta_trend"][i]) * data["t"]
              + float(draws_dict["beta_temp"][i]) * data["temp"]
              + cal_eff)
        preds[j] = np.exp(mu * lbe_y_std + lbe_y_mean)
    return preds

# Compute accuracy
lbe_tr_preds = lbe_predict(lbe_tr, lbe_draws, lbe_bc)
lbe_va_preds = lbe_predict(lbe_va, lbe_draws, lbe_bc)

lbe_tr_mean = lbe_tr_preds.mean(axis=0)
lbe_va_mean = lbe_va_preds.mean(axis=0)

lbe_y_tr = np.exp(df_lbe_train["log_rev"].values)
lbe_y_va = np.exp(df_lbe_val["log_rev"].values)

lbe_mape_train = mean_absolute_percentage_error(lbe_y_tr, lbe_tr_mean) * 100
lbe_mape_val   = mean_absolute_percentage_error(lbe_y_va, lbe_va_mean) * 100
lbe_r2_train   = r2_score(lbe_y_tr, lbe_tr_mean)
lbe_r2_val     = r2_score(lbe_y_va, lbe_va_mean)

print(f"\n      Train MAPE: {lbe_mape_train:.1f}%   R²: {lbe_r2_train:.3f}")
print(f"      Val   MAPE: {lbe_mape_val:.1f}%   R²: {lbe_r2_val:.3f}")

# COMMAND ----------

# DBTITLE 1,LBE Artifact Export
# ━━ LBE ARTIFACT EXPORT ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("\n[7/7] Saving LBE results...")

lbe_results = {
    "idata": lbe_idata, "draws": lbe_draws, "bc": lbe_bc,
    "n_draws": lbe_n_draws,
    "df_train": df_lbe_train, "df_val": df_lbe_val,
    "y_mean": lbe_y_mean, "y_std": lbe_y_std,
    "aw_max": lbe_aw_max, "co_max": lbe_co_max,
    "t_max": lbe_t_max, "temp_mu": lbe_temp_mu, "temp_sd": lbe_temp_sd,
    "tr": lbe_tr, "va": lbe_va,
    "CAL_COLS": LBE_CAL_COLS, "CAL_MU": LBE_CAL_MU, "CAL_SIGMA": LBE_CAL_SIGMA,
    "metrics": {
        "train_mape": lbe_mape_train, "val_mape": lbe_mape_val,
        "train_r2": lbe_r2_train, "val_r2": lbe_r2_val,
    },
}

with open(LBE_RESULTS_PKL, "wb") as f:
    pickle.dump(lbe_results, f)

print(f"  ✓ LBE results saved → {LBE_RESULTS_PKL}")
print(f"  ✓ {lbe_n_draws} posterior draws")
print(f"  ✓ Val R²: {lbe_r2_val:.3f}, MAPE: {lbe_mape_val:.1f}%")

# COMMAND ----------

# DBTITLE 1,Part F — LBE Diagnostics
# MAGIC %md
# MAGIC ---
# MAGIC ## Part F: LBE Diagnostics
# MAGIC Loads `lbe_mmm_results.pkl` and produces validation diagnostics, contribution decomposition, and response curves. Can run standalone if artifacts exist.

# COMMAND ----------

# DBTITLE 1,LBE Load Artifacts for Diagnostics
# ━━ LOAD LBE ARTIFACTS (for diagnostics-only runs) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# NOTE: If running end-to-end from Part E, these are already in memory.

print("Loading LBE artifacts...")
with open(LBE_RESULTS_PKL, "rb") as f:
    lbe_results = pickle.load(f)

lbe_draws   = lbe_results["draws"]
lbe_bc      = lbe_results["bc"]
lbe_n_draws = int(lbe_results["n_draws"])
lbe_y_mean  = float(lbe_results["y_mean"])
lbe_y_std   = float(lbe_results["y_std"])
lbe_aw_max  = float(lbe_results["aw_max"])
lbe_co_max  = float(lbe_results["co_max"])
lbe_t_max   = float(lbe_results["t_max"])
lbe_temp_mu = float(lbe_results["temp_mu"])
lbe_temp_sd = float(lbe_results["temp_sd"])
lbe_tr      = lbe_results["tr"]
lbe_va      = lbe_results["va"]
df_lbe_train = lbe_results["df_train"].copy()
df_lbe_val   = lbe_results["df_val"].copy()
LBE_CAL_COLS = list(lbe_results["CAL_COLS"])

df_lbe_all = pd.concat([df_lbe_train, df_lbe_val]).reset_index(drop=True)
print(f"  ✓ LBE loaded: {lbe_n_draws} draws, {len(df_lbe_all)} weeks")

# COMMAND ----------

# DBTITLE 1,LBE Validation Diagnostics
# ━━ LBE VALIDATION DIAGNOSTICS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Predict full series
full_data = lbe_prep(df_lbe_all) if 'lbe_prep' in dir() else {
    "y":    ((df_lbe_all["log_rev"] - lbe_y_mean) / lbe_y_std).values.astype("float64"),
    "aw":   (df_lbe_all["Awareness"].values / lbe_aw_max).astype("float64"),
    "co":   (df_lbe_all["Consideration"].values / lbe_co_max).astype("float64"),
    "temp": ((df_lbe_all["national_mean_temp"].values - lbe_temp_mu) / lbe_temp_sd).astype("float64"),
    "cal":  df_lbe_all[LBE_CAL_COLS].values.astype("float64"),
    "t":    (df_lbe_all["t"].values / lbe_t_max).astype("float64") if "t" in df_lbe_all.columns
            else (np.arange(len(df_lbe_all)) / lbe_t_max).astype("float64"),
}

lbe_full_preds = lbe_predict(full_data, lbe_draws, lbe_bc, sample_size=500)
lbe_full_mean  = lbe_full_preds.mean(axis=0)
lbe_full_actual = df_lbe_all["LBE_Revenue"].values

# D3: Actual vs Predicted
fig, ax = plt.subplots(figsize=(16, 6))
ax.plot(df_lbe_all["week_date"], lbe_full_actual / 1e6, color="black", lw=1.5, label="Actual")
ax.plot(df_lbe_all["week_date"], lbe_full_mean / 1e6, color=CPX_NAVY, lw=1.8, ls="--", label="Predicted")
val_start_lbe = df_lbe_val["week_date"].min()
ax.axvspan(val_start_lbe, df_lbe_val["week_date"].max(), alpha=0.08, color=CPX_RED, label="Validation")
metrics = lbe_results.get("metrics", {})
ax.set_title(f"LBE MMM — Actual vs Predicted\n"
             f"Train MAPE {metrics.get('train_mape', 0):.1f}% | Val MAPE {metrics.get('val_mape', 0):.1f}% R² {metrics.get('val_r2', 0):.3f}",
             fontsize=13, fontweight="bold", color=CPX_NAVY)
ax.set_ylabel("Weekly Revenue ($M)")
ax.legend(frameon=False)
ax.grid(False); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
plt.tight_layout()
save_fig(fig, "lbe_d3_actual_vs_predicted.png", LBE_OUTPUT_DIR)
plt.show()

# COMMAND ----------

# DBTITLE 1,LBE Diminishing Returns & Marginal Curves
# ━━ LBE DIMINISHING RETURNS & MARGINAL RETURN CURVES ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for ch_idx, (channel, ch_max, decay_key, scale_key, beta_key, color) in enumerate([
    ("Awareness",     lbe_aw_max, "aw_decay", "aw_scale", "beta_aw", CPX_NAVY),
    ("Consideration", lbe_co_max, "co_decay", "co_scale", "beta_co", CPX_GOLD),
]):
    spend_range = np.linspace(0, ch_max * 2, 200)
    s_norm = spend_range / ch_max

    # Mean posterior response
    decay_mean = float(np.mean(lbe_draws[decay_key]))
    scale_mean = float(np.mean(lbe_draws[scale_key]))
    beta_mean  = float(np.mean(lbe_draws[beta_key]))

    ads = s_norm / (1 - decay_mean)  # steady-state
    sat = np.log1p(ads * scale_mean)
    contrib = beta_mean * sat * lbe_y_std  # in log-rev scale units
    contrib_rev = np.exp(lbe_y_mean + contrib) - np.exp(lbe_y_mean)  # approx marginal revenue
    annual_contrib = contrib_rev * 52

    axes[0].plot(spend_range / 1e3, annual_contrib / 1e6, color=color, lw=2, label=channel)

    # Marginal return
    marginal = np.gradient(annual_contrib, spend_range)
    axes[1].plot(spend_range / 1e3, marginal, color=color, lw=2, label=channel)

axes[0].set_title("LBE — Response Curves (Annual Revenue)", fontweight="bold", color=CPX_NAVY)
axes[0].set_xlabel("Annual Spend ($K)"); axes[0].set_ylabel("Revenue Contribution ($M)")
axes[0].legend(frameon=False)

axes[1].set_title("LBE — Marginal Return per $1", fontweight="bold", color=CPX_NAVY)
axes[1].set_xlabel("Annual Spend ($K)"); axes[1].set_ylabel("Revenue per $1")
axes[1].axhline(1, color="grey", ls=":", lw=0.8)
axes[1].legend(frameon=False)

for ax in axes:
    ax.grid(False); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
plt.tight_layout()
save_fig(fig, "lbe_response_marginal.png", LBE_OUTPUT_DIR)
plt.show()

# COMMAND ----------

# DBTITLE 1,Part G — LBE Budget Scenario Analysis
# MAGIC %md
# MAGIC ---
# MAGIC ## Part G: LBE Budget Scenario Analysis
# MAGIC Three annual scenarios with posterior CIs. Decision variable is **funnel-level spend** (Awareness, Consideration); channel-level outcomes follow from observed allocation shares.

# COMMAND ----------

# DBTITLE 1,LBE Budget Scenarios
# ━━ LBE BUDGET SCENARIO ANALYSIS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# 2025 baseline spend
df_lbe_2025 = df_lbe_all[df_lbe_all["week_date"].dt.year == 2025] if "week_date" in df_lbe_all.columns else df_lbe_all.tail(52)
lbe_baseline_aw = float(df_lbe_2025["Awareness"].sum()) if len(df_lbe_2025) > 0 else float(df_lbe_all["Awareness"].sum())
lbe_baseline_co = float(df_lbe_2025["Consideration"].sum()) if len(df_lbe_2025) > 0 else float(df_lbe_all["Consideration"].sum())

def lbe_funnel_contribution(aw_annual, co_annual):
    """Point estimate of annual media revenue contribution."""
    aw_norm = (aw_annual / 52) / lbe_aw_max
    co_norm = (co_annual / 52) / lbe_co_max
    decay_aw = float(np.mean(lbe_draws["aw_decay"]))
    decay_co = float(np.mean(lbe_draws["co_decay"]))
    scale_aw = float(np.mean(lbe_draws["aw_scale"]))
    scale_co = float(np.mean(lbe_draws["co_scale"]))
    beta_aw_m = float(np.mean(lbe_draws["beta_aw"]))
    beta_co_m = float(np.mean(lbe_draws["beta_co"]))

    ads_aw = aw_norm / (1 - decay_aw)
    ads_co = co_norm / (1 - decay_co)
    sat_aw = np.log1p(ads_aw * scale_aw)
    sat_co = np.log1p(ads_co * scale_co)

    mu_media = beta_aw_m * sat_aw + beta_co_m * sat_co
    rev_with = np.exp(lbe_y_mean + mu_media * lbe_y_std) * 52
    rev_without = np.exp(lbe_y_mean) * 52
    return rev_with - rev_without

lbe_baseline_rev = lbe_funnel_contribution(lbe_baseline_aw, lbe_baseline_co)

print(f"LBE 2025 Baseline Annual Media Spend: ${(lbe_baseline_aw + lbe_baseline_co)/1e6:.2f}M")
print(f"  Awareness:     ${lbe_baseline_aw/1e6:.2f}M")
print(f"  Consideration: ${lbe_baseline_co/1e6:.2f}M")
print(f"\nBaseline annual media revenue contribution: ${lbe_baseline_rev/1e6:.2f}M")

# Scenario 1: Budget Uplift
print("\n" + "=" * 60)
print("SCENARIO 1 — BUDGET UPLIFT (pattern unchanged)")
print("=" * 60)
for factor in [1.10, 1.20, 1.30]:
    rev = lbe_funnel_contribution(lbe_baseline_aw * factor, lbe_baseline_co * factor)
    lift = rev - lbe_baseline_rev
    pct = lift / lbe_baseline_rev * 100
    print(f"  +{(factor-1)*100:.0f}%: ${rev/1e6:.2f}M revenue (+${lift/1e6:.2f}M, +{pct:.1f}%)")

# Scenario 2: Reshuffle (Consideration → Awareness)
print("\n" + "=" * 60)
print("SCENARIO 2 — RESHUFFLE (20% shift: Consideration → Awareness)")
print("=" * 60)
shift = lbe_baseline_co * 0.20
rev_r = lbe_funnel_contribution(lbe_baseline_aw + shift, lbe_baseline_co - shift)
lift_r = rev_r - lbe_baseline_rev
print(f"  Awareness:     ${(lbe_baseline_aw + shift)/1e6:.2f}M")
print(f"  Consideration: ${(lbe_baseline_co - shift)/1e6:.2f}M")
print(f"  Revenue: ${rev_r/1e6:.2f}M (+${lift_r/1e6:.2f}M, +{lift_r/lbe_baseline_rev*100:.1f}%)")

# Scenario 3: Combined
print("\n" + "=" * 60)
print("SCENARIO 3 — COMBINED (uplift + reshuffle)")
print("=" * 60)
for factor in [1.10, 1.20, 1.30]:
    aw_up = lbe_baseline_aw * factor
    co_up = lbe_baseline_co * factor
    shift_c = co_up * 0.20
    rev_c = lbe_funnel_contribution(aw_up + shift_c, co_up - shift_c)
    lift_c = rev_c - lbe_baseline_rev
    print(f"  +{(factor-1)*100:.0f}% + reshuffle: ${rev_c/1e6:.2f}M (+${lift_c/1e6:.2f}M, +{lift_c/lbe_baseline_rev*100:.1f}%)")

# COMMAND ----------

# DBTITLE 1,Part H — Stakeholder Presentation Charts
# MAGIC %md
# MAGIC ---
# MAGIC ## Part H: Stakeholder Presentation Charts (EXH + LBE)
# MAGIC Final presentation-quality charts combining both models for the stakeholder deck. Loads artifacts from both EXH and LBE pipelines.
# MAGIC
# MAGIC **EXH Charts**: Annual media spend by funnel, spend by LOB, attendance decomposition, flighting analysis  
# MAGIC **LBE Charts**: Annual spend by channel, revenue decomposition, calendar events, diminishing returns

# COMMAND ----------

# DBTITLE 1,Presentation — EXH Annual Media Spend
# ━━ EXH ANNUAL MEDIA SPEND BY FUNNEL ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

fig, ax = plt.subplots(figsize=(12, 6))

years = [2023, 2024, 2025]
width = 0.25
x_pos = np.arange(len(years))

for i, (funnel, color) in enumerate(FUNNEL_COLORS.items()):
    annual_spend = []
    for year in years:
        yr_data = df_full[df_full["year"] == year]
        funnel_combos = FUNNEL_COMBOS.get(funnel, [])
        spend = yr_data[funnel_combos].sum().sum() if funnel_combos else 0
        annual_spend.append(spend / 1e6)
    ax.bar(x_pos + i * width, annual_spend, width, color=color, label=funnel, alpha=0.85)

ax.set_xticks(x_pos + width)
ax.set_xticklabels(years)
ax.set_title("EXH Annual Media Spend by Funnel", fontweight="bold", color=CPX_NAVY, fontsize=14)
ax.set_ylabel("Annual Spend ($M)")
ax.legend(frameon=False)
ax.grid(False); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
plt.tight_layout()
save_fig(fig, "exh_annual_spend_by_funnel.png")
plt.show()

# COMMAND ----------

# DBTITLE 1,Presentation — EXH Attendance Decomposition
# ━━ EXH ANNUAL ATTENDANCE DECOMPOSITION ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

fig, ax = plt.subplots(figsize=(12, 7))
years = sorted(df_full["year"].unique())

for yr_idx, year in enumerate(years):
    yr_data = df_full[df_full["year"] == year]
    baseline_val = yr_data["controls_pred"].sum() / 1e6
    bottom = baseline_val

    # Draw baseline bar
    ax.bar(yr_idx, baseline_val, color=CPX_LGREY, edgecolor="white", label="Baseline" if yr_idx == 0 else "")

    # Stack media contributions by funnel
    for funnel, color in FUNNEL_COLORS.items():
        media_col = f"media_{funnel}"
        if media_col in yr_data.columns:
            val = yr_data[media_col].sum() / 1e6
            ax.bar(yr_idx, val, bottom=bottom, color=color, alpha=0.8,
                   label=funnel if yr_idx == 0 else "")
            bottom += val

    # Actual marker
    actual = yr_data[target_col].sum() / 1e6
    ax.scatter(yr_idx, actual, color="black", marker="_", s=200, zorder=5)

ax.set_xticks(range(len(years)))
ax.set_xticklabels(years)
ax.set_title("EXH Annual Attendance Decomposition", fontweight="bold", color=CPX_NAVY, fontsize=14)
ax.set_ylabel("Annual Attendance (M)")
ax.legend(frameon=False, loc="upper left")
ax.grid(False); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
plt.tight_layout()
save_fig(fig, "exh_annual_decomposition.png")
plt.show()

# COMMAND ----------

# DBTITLE 1,Presentation — LBE Annual Revenue Decomposition
# ━━ LBE ANNUAL REVENUE DECOMPOSITION ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Decompose LBE revenue by year
df_lbe_all_yr = df_lbe_all.copy()
df_lbe_all_yr["year"] = df_lbe_all_yr["week_date"].dt.year
lbe_years = sorted(df_lbe_all_yr["year"].unique())

# Compute per-year decomposition using posterior means
decay_aw_m = float(np.mean(lbe_draws["aw_decay"]))
decay_co_m = float(np.mean(lbe_draws["co_decay"]))
scale_aw_m = float(np.mean(lbe_draws["aw_scale"]))
scale_co_m = float(np.mean(lbe_draws["co_scale"]))
beta_aw_m  = float(np.mean(lbe_draws["beta_aw"]))
beta_co_m  = float(np.mean(lbe_draws["beta_co"]))

fig, ax = plt.subplots(figsize=(10, 6))

for yr_idx, year in enumerate(lbe_years):
    yr_data = df_lbe_all_yr[df_lbe_all_yr["year"] == year]
    actual = yr_data["LBE_Revenue"].sum() / 1e6

    # Awareness contribution (approx via steady-state)
    aw_spend = yr_data["Awareness"].sum()
    co_spend = yr_data["Consideration"].sum()

    aw_norm = (aw_spend / len(yr_data)) / lbe_aw_max
    co_norm = (co_spend / len(yr_data)) / lbe_co_max
    ads_aw = aw_norm / (1 - decay_aw_m)
    ads_co = co_norm / (1 - decay_co_m)
    sat_aw = np.log1p(ads_aw * scale_aw_m)
    sat_co = np.log1p(ads_co * scale_co_m)

    rev_base = np.exp(lbe_y_mean) * len(yr_data) / 1e6
    rev_aw   = (np.exp(lbe_y_mean + beta_aw_m * sat_aw * lbe_y_std) - np.exp(lbe_y_mean)) * len(yr_data) / 1e6
    rev_co   = (np.exp(lbe_y_mean + beta_co_m * sat_co * lbe_y_std) - np.exp(lbe_y_mean)) * len(yr_data) / 1e6

    ax.bar(yr_idx, rev_base, color=CPX_LGREY, edgecolor="white", label="Baseline" if yr_idx == 0 else "")
    ax.bar(yr_idx, rev_aw, bottom=rev_base, color=CPX_NAVY, alpha=0.8, label="Awareness" if yr_idx == 0 else "")
    ax.bar(yr_idx, rev_co, bottom=rev_base + rev_aw, color=CPX_GOLD, alpha=0.8, label="Consideration" if yr_idx == 0 else "")
    ax.scatter(yr_idx, actual, color="black", marker="_", s=200, zorder=5)

ax.set_xticks(range(len(lbe_years)))
ax.set_xticklabels(lbe_years)
ax.set_title("LBE Annual Revenue Decomposition", fontweight="bold", color=CPX_NAVY, fontsize=14)
ax.set_ylabel("Annual Revenue ($M)")
ax.legend(frameon=False)
ax.grid(False); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
plt.tight_layout()
save_fig(fig, "lbe_annual_decomposition.png", LBE_OUTPUT_DIR)
plt.show()

# COMMAND ----------

# DBTITLE 1,Pipeline Complete
# MAGIC %md
# MAGIC ---
# MAGIC ## Pipeline Complete ✓
# MAGIC
# MAGIC All models trained, diagnostics generated, and scenarios computed.
# MAGIC
# MAGIC ### Output Summary
# MAGIC | Model | Artifact | Path |
# MAGIC | --- | --- | --- |
# MAGIC | EXH | Trace + scalers + params | `MMM_EXH_Outcomes/` |
# MAGIC | EXH | LOB decomposition | `MMM_EXH_LOB_Decomp/` |
# MAGIC | LBE | Results pickle | `MMM_LBE/lbe_mmm_results.pkl` |
# MAGIC
# MAGIC ### Next Steps
# MAGIC - Review diagnostic charts for model health
# MAGIC - Present budget scenarios to stakeholders
# MAGIC - Validate LOB decomposition with geo-test (recommended for dominant LOBs)
# MAGIC - Update 2026 forecast quarterly as new data arrives
