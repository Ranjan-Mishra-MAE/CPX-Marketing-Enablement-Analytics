# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# DBTITLE 1,MMM EXH — All Charts & Tables
# MAGIC %md
# MAGIC # EXH Marketing Mix Model — All Charts & Tables
# MAGIC **Cineplex Marketing Analytics | Complete End-to-End Pipeline**
# MAGIC
# MAGIC This notebook is the single-source-of-truth for the EXH (Exhibition) Marketing Mix Model. It contains ALL stages from training through stakeholder deliverables, with no redundancies.
# MAGIC
# MAGIC ### Sections
# MAGIC 1. Setup & Configuration (installs, imports, paths, palette)
# MAGIC 2. Data Loading & Feature Engineering
# MAGIC 3. Train/Val/Test Split & Scaling
# MAGIC 4. Media Metadata & Funnel×LOB Aggregation
# MAGIC 5. Model Definition & NUTS Sampling
# MAGIC 6. Convergence Diagnostics & Accuracy
# MAGIC 7. Artifact Export
# MAGIC 8. Model Diagnostics Charts (D3, D5, B1)
# MAGIC 9. Response Curves — Diminishing Returns (YoY) & Marginal Return
# MAGIC 10. Presentation Charts (Spend, Attendance, Features, Correlations, Flighting)
# MAGIC 11. LOB Decomposition (Tilt Signal, Tables, Per-LOB Charts)
# MAGIC 12. Budget Scenarios (S1/S2/S3 with Posterior CIs)
# MAGIC 13. Stakeholder Deliverables (2026 Projections, Summary Tables, Excel)
# MAGIC
# MAGIC ### Key Paths
# MAGIC - **Input**: `EXH_ModelData.xlsx`
# MAGIC - **Outputs**: `MMM_EXH_Outcomes_v2/` (trace, scalers, response params, df_full)
# MAGIC - **LOB Decomp**: `MMM_EXH_LOB_Decomp_v2/` (decomposition results)
# MAGIC
# MAGIC ### Runtime
# MAGIC - Full pipeline (with NUTS): ~60–90 min on GPU
# MAGIC - Diagnostics-only (skip cell 5–7): ~5 min (loads saved artifacts)

# COMMAND ----------

# DBTITLE 1,Install Dependencies
# MAGIC %pip install pymc arviz numpyro jax jaxlib pytensor scikit-learn openpyxl netcdf4 h5netcdf scipy xarray --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Imports & Configuration
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
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patheffects as pe
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import seaborn as sns

warnings.filterwarnings("ignore")
print(f"PyMC  : {pm.__version__}")
print(f"ArviZ : {az.__version__}")

# COMMAND ----------

# DBTITLE 1,Paths & Brand Palette
# ── Artifact Paths ──────────────────────────────────────────────────────────────
EXH_FILE_PATH     = "/Volumes/data_exploration_hub/marketing_hub/model_data/Final_Data/EXH_ModelData.xlsx"
EXH_ARTIFACT_PATH = "/Volumes/data_exploration_hub/marketing_hub/model_data/Final_Data/MMM_EXH_Outcomes_v2/"
EXH_DECOMP_PATH   = "/Volumes/data_exploration_hub/marketing_hub/model_data/Final_Data/MMM_EXH_LOB_Decomp_v2/"
os.makedirs(EXH_ARTIFACT_PATH, exist_ok=True)
os.makedirs(EXH_DECOMP_PATH, exist_ok=True)

# ── Cineplex Brand Palette ────────────────────────────────────────────────────────
CPX_NAVY  = "#002B5C"
CPX_RED   = "#E31837"
CPX_GOLD  = "#F5A800"
CPX_LGREY = "#F2F2F2"
CPX_DGREY = "#4A4A4A"
CPX_MGREY = "#8A8A8A"
CPX_GREEN = "#2E7D32"
CPX_BLUE  = "#0857C3"
WHITE     = "#FFFFFF"

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

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": "#CCCCCC", "grid.color": "#EEEEEE",
    "grid.linestyle": "--", "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 120,
})

def save_fig(fig, fname, path=EXH_ARTIFACT_PATH):
    """Save figure to artifacts volume."""
    out = os.path.join(path, fname)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"  Saved \u2192 {out}")

print(f"EXH Artifacts \u2192 {EXH_ARTIFACT_PATH}")
print(f"EXH LOB Decomp \u2192 {EXH_DECOMP_PATH}")

# COMMAND ----------

# DBTITLE 1,Section 2 — Data Loading
# MAGIC %md
# MAGIC ---
# MAGIC ## Section 2: Data Loading & Feature Engineering
# MAGIC Loads the EXH weekly dataset, encodes binary columns, creates season dummies, year indicators, and a linear trend.

# COMMAND ----------

# DBTITLE 1,EXH Data Loading & Feature Engineering
# ━━ EXH DATA LOADING & FEATURE ENGINEERING ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

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

print(f"Data loaded \u2014 shape {df.shape}")
print(f"Date range: {df['week_date'].min().date()} \u2192 {df['week_date'].max().date()}")
print(f"Missing values: {df.isna().sum().sum()}")

# COMMAND ----------

# DBTITLE 1,EXH Feature Selection
# ━━ FEATURE SELECTION ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
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

print(f"Train : {n_train} weeks  ({train_df['week_date'].min().date()} \u2192 {train_df['week_date'].max().date()})")
print(f"Val   : {n_val}   weeks  ({val_df['week_date'].min().date()} \u2192 {val_df['week_date'].max().date()})")
print(f"Test  : {n_test}  weeks  (held out)")

# ━━ SCALING ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train = scaler_X.fit_transform(train_df[baseline_features])
X_val   = scaler_X.transform(val_df[baseline_features])

y_train = scaler_y.fit_transform(train_df[[target_col]]).flatten()
y_val   = scaler_y.transform(val_df[[target_col]]).flatten()

# COMMAND ----------

# DBTITLE 1,Section 4 — Media Structure
# MAGIC %md
# MAGIC ---
# MAGIC ## Section 4: Media Metadata & Model Training
# MAGIC 54 raw media columns parsed into Funnel×LOB combos, grouped into 10 combinations. 3 sparse combos dropped → 7 active.

# COMMAND ----------

# DBTITLE 1,EXH Media Metadata & LOB Aggregation
# ━━ MEDIA METADATA & FUNNEL×LOB AGGREGATION ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

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
    "Premium_Brand" : ["Premium Experiences - AVX", "Premium Experiences - VIP", "F&B", "EXH - Brand"],
    "Film_Marketing": ["Premium Experiences - IMAX", "Film Marketing - Inform", "Performance - Convert & Upgrade"],
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

# DBTITLE 1,EXH Combo Structure & Active Combos
# ━━ COMBO STRUCTURE & ACTIVE COMBOS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

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
# ━━ MEDIA SPEND SCALING ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

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
# ━━ RESPONSE TRANSFORMATION FUNCTIONS (PyTensor) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

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
# ━━ ARTIFACT EXPORT ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("=" * 60)
print("SAVING EXH ARTIFACTS")
print("=" * 60)

# 1. Trace
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

# DBTITLE 1,Section 8 — Diagnostics Charts
# MAGIC %md
# MAGIC ---
# MAGIC ## Section 8: Model Diagnostics Charts
# MAGIC D3 (Actual vs Predicted), D5 (Residuals), B1 (Weekly Contribution Decomposition)

# COMMAND ----------

# DBTITLE 1,EXH Chart D3 — Actual vs Predicted
# ━━ D3: ACTUAL VS PREDICTED ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

df_trainval = df_full[df_full["segment"].isin(["train", "val"])].copy().reset_index(drop=True)

fig, ax = plt.subplots(figsize=(16, 6))
dates_all = df_trainval["week_date"]
ax.plot(dates_all, df_trainval[target_col] / 1e6, color="black", lw=1.5, label="Actual")
ax.plot(dates_all, df_trainval["pred"] / 1e6, color=CPX_NAVY, lw=1.8, ls="--", label="Predicted")
val_start = df_trainval.loc[df_trainval["segment"] == "val", "week_date"].min()
val_end   = df_trainval.loc[df_trainval["segment"] == "val", "week_date"].max()
ax.axvspan(val_start, val_end, alpha=0.08, color=CPX_RED, label="Validation window")
ax.set_title(f"EXH MMM \u2014 Actual vs Predicted\n"
             f"Train MAPE {mape_train:.1f}% R\u00b2 {r2_train:.3f} | Val MAPE {mape_val:.1f}% R\u00b2 {r2_val:.3f}",
             fontsize=13, fontweight="bold", color=CPX_NAVY)
ax.set_ylabel("Weekly Attendance (M)")
ax.legend(frameon=False)
ax.grid(False)
plt.tight_layout()
save_fig(fig, "exh_d3_actual_vs_predicted.png")
display(fig); plt.close(fig)

# COMMAND ----------

# DBTITLE 1,EXH Chart D5 — Residuals Over Time
# ━━ D5: RESIDUALS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

fig, ax = plt.subplots(figsize=(16, 4))
resid_all = df_trainval[target_col].values - df_trainval["pred"].values
ax.bar(df_trainval["week_date"], resid_all / 1e3,
       color=[CPX_NAVY if s == "train" else CPX_RED for s in df_trainval["segment"]], alpha=0.6, width=5)
ax.axhline(0, color="black", lw=0.8)
ax.set_title("EXH MMM \u2014 Residuals Over Time", fontweight="bold", color=CPX_NAVY)
ax.set_ylabel("Residual (K)")
ax.grid(False)
plt.tight_layout()
save_fig(fig, "exh_d5_residuals.png")
display(fig); plt.close(fig)

# COMMAND ----------

# DBTITLE 1,EXH Chart B1 — Weekly Contribution Decomposition
# ━━ B1: WEEKLY CONTRIBUTION DECOMPOSITION ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

fig, ax = plt.subplots(figsize=(16, 7))
dates = df_trainval["week_date"]

baseline = df_trainval["controls_pred"].values / 1e6
bottom = baseline.copy()

for combo in ACTIVE_COMBOS:
    vals = df_trainval[f"media_{combo}"].values / 1e6
    ax.fill_between(dates, bottom, bottom + vals, alpha=0.7,
                    color=LOB_COLORS.get(combo, CPX_DGREY), label=combo)
    bottom += vals

ax.plot(dates, df_trainval[target_col].values / 1e6, color="black", lw=1.5, label="Actual")
ax.set_title("EXH MMM \u2014 Weekly Attendance Decomposition", fontweight="bold", color=CPX_NAVY)
ax.set_ylabel("Weekly Attendance (M)")
ax.legend(loc="upper left", fontsize=8, ncol=2, frameon=False)
ax.grid(False)
plt.tight_layout()
save_fig(fig, "exh_b1_weekly_decomposition.png")
display(fig); plt.close(fig)

# COMMAND ----------

# DBTITLE 1,Section 9 — Response Curves
# MAGIC %md
# MAGIC ---
# MAGIC ## Section 9: Diminishing Return Curve (YoY) & Marginal Return
# MAGIC Aggregate response curve showing total media attendance vs total budget, with 2023/2024/2025 actual spend points. Budget shares held at 2025 mix.

# COMMAND ----------

# DBTITLE 1,Load Saved Artifacts (standalone from here)
# ━━ LOAD SAVED ARTIFACTS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# This cell allows Sections 9+ to run standalone without re-training.
# After Cell 17 (artifact export) restarts Python, all variables are lost.
# Re-load everything from saved .pkl / .parquet / .nc files.

import os, pickle, shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import arviz as az

# ── Paths ─────────────────────────────────────────────────────────────────────────────
EXH_ARTIFACT_PATH = "/Volumes/data_exploration_hub/marketing_hub/model_data/Final_Data/MMM_EXH_Outcomes_v2/"
EXH_DECOMP_PATH   = "/Volumes/data_exploration_hub/marketing_hub/model_data/Final_Data/MMM_EXH_LOB_Decomp_v2/"

# ── Brand Palette ─────────────────────────────────────────────────────────────────────
CPX_NAVY  = "#002B5C"
CPX_RED   = "#E31837"
CPX_GOLD  = "#F5A800"
CPX_GREEN = "#2E7D32"
CPX_LGREY = "#CCCCCC"
CPX_DGREY = "#4A4A4A"

def save_fig(fig, fname):
    fig.savefig(os.path.join(EXH_ARTIFACT_PATH, fname), dpi=150, bbox_inches="tight", facecolor="white")
    print(f"  Saved \u2192 {EXH_ARTIFACT_PATH}{fname}")

# ── Load Artifacts ────────────────────────────────────────────────────────────────────
with open(f"{EXH_ARTIFACT_PATH}response_params.pkl",       "rb") as f: response_params       = pickle.load(f)
with open(f"{EXH_ARTIFACT_PATH}spend_train_max_iter2.pkl", "rb") as f: spend_train_max_iter2 = pickle.load(f)
with open(f"{EXH_ARTIFACT_PATH}model_meta.pkl",            "rb") as f: model_meta            = pickle.load(f)
with open(f"{EXH_ARTIFACT_PATH}scaler_y.pkl",              "rb") as f: scaler_y              = pickle.load(f)
with open(f"{EXH_ARTIFACT_PATH}scaler_X.pkl",              "rb") as f: scaler_X              = pickle.load(f)

df_full = pd.read_parquet(f"{EXH_ARTIFACT_PATH}df_full.parquet")
df_full["year"] = df_full["week_date"].dt.year

trace_iter2_mixed_hier = az.from_netcdf(f"{EXH_ARTIFACT_PATH}trace_iter2_mixed_hier.nc")

# ── Unpack model_meta ─────────────────────────────────────────────────────────────────
ACTIVE_COMBOS  = model_meta["ACTIVE_COMBOS"]
FUNNEL_COMBOS  = model_meta["FUNNEL_COMBOS"]
LOB_GROUP_MAP  = model_meta["LOB_GROUP_MAP"]
target_col     = model_meta["target_col"]
baseline_features = model_meta["baseline_features"]

scale_factor = float(scaler_y.scale_[0])

# ── Rebuild train/val segments & accuracy metrics ─────────────────────────────────────
df_trainval = df_full[df_full["segment"].isin(["train", "val"])].copy().reset_index(drop=True)

_train = df_trainval[df_trainval["segment"] == "train"]
_val   = df_trainval[df_trainval["segment"] == "val"]
mape_train = float(np.mean(np.abs(_train[target_col] - _train["pred"]) / _train[target_col]) * 100)
mape_val   = float(np.mean(np.abs(_val[target_col] - _val["pred"]) / _val[target_col]) * 100)
_ss_res_t = ((_train[target_col] - _train["pred"])**2).sum()
_ss_tot_t = ((_train[target_col] - _train[target_col].mean())**2).sum()
r2_train  = 1 - _ss_res_t / _ss_tot_t
_ss_res_v = ((_val[target_col] - _val["pred"])**2).sum()
_ss_tot_v = ((_val[target_col] - _val[target_col].mean())**2).sum()
r2_val    = 1 - _ss_res_v / _ss_tot_v

# ── LOB Colours ───────────────────────────────────────────────────────────────────────
LOB_COLORS = {
    "Awareness__Premium_Brand":     CPX_NAVY,
    "Consideration__Premium_Brand": CPX_RED,
    "Consideration__GE":            CPX_GOLD,
    "Consideration__Prepaid":       CPX_GREEN,
    "Conversion__Film_Marketing":   "#0857C3",
    "Conversion__Premium_Brand":    "#F5A800",
    "Conversion__Cineclub":         "#8B5CF6",
}

print(f"\u2713 Artifacts loaded from {EXH_ARTIFACT_PATH}")
print(f"  df_full: {df_full.shape}, date range {df_full['week_date'].min().date()} \u2192 {df_full['week_date'].max().date()}")
print(f"  Active combos: {len(ACTIVE_COMBOS)}, Response params: {len(response_params)}")
print(f"  Train MAPE {mape_train:.1f}% R\u00b2 {r2_train:.3f} | Val MAPE {mape_val:.1f}% R\u00b2 {r2_val:.3f}")
print(f"  Trace loaded: {trace_iter2_mixed_hier.posterior.dims}")

# COMMAND ----------

# DBTITLE 1,EXH B10 — Diminishing Return Curve (YoY)
# ━━ B10: EXH DIMINISHING RETURN CURVE (YoY) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

scale_factor = float(scaler_y.scale_[0])

# 2025 allocation shares (hold mix constant across the curve)
df_2025_rc = df_full[df_full["year"] == 2025]
combo_weekly_2025 = {c: float(df_2025_rc[c].mean()) for c in ACTIVE_COMBOS}
total_weekly_2025 = sum(combo_weekly_2025.values())
alloc_shares = {c: combo_weekly_2025[c] / total_weekly_2025 for c in ACTIVE_COMBOS}
total_annual_2025 = total_weekly_2025 * 52

def total_media_attendance(annual_budget, shares=alloc_shares):
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

# X axis: near-zero to 10\u00d7 2025 spend
x_annual = np.linspace(52 * 100, total_annual_2025 * 10, 500)
curve = np.array([total_media_attendance(b) for b in x_annual])

# Year reference points
year_points = {}
for year in [2023, 2024, 2025]:
    df_yr = df_full[df_full["year"] == year]
    yr_spend = {c: float(df_yr[c].sum()) for c in ACTIVE_COMBOS}
    yr_active_spend = sum(yr_spend.values())
    # Include DROP_SPARSE combos so the label reflects actual total annual spend
    _sparse = [c for c in ["Awareness__Other", "Consideration__Other", "Conversion__Other"] if c in df_yr.columns]
    yr_total_spend = yr_active_spend + float(df_yr[_sparse].sum().sum()) if _sparse else yr_active_spend
    yr_shares = {c: (yr_spend[c] / yr_active_spend if yr_active_spend > 0 else alloc_shares[c])
                 for c in ACTIVE_COMBOS}
    yr_att = total_media_attendance(yr_total_spend, shares=yr_shares)
    year_points[year] = {"spend": yr_total_spend, "attendance": yr_att}

YEAR_MARKER_STYLES = {
    2023: {"color": CPX_DGREY, "marker": "o", "linestyle": "--",  "linewidth": 1.2},
    2024: {"color": CPX_NAVY,  "marker": "s", "linestyle": "-.", "linewidth": 1.2},
    2025: {"color": CPX_RED,   "marker": "D", "linestyle": "-",  "linewidth": 1.5},
}

# Response Curve 
fig, ax = plt.subplots(figsize=(16, 9))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")
ax.plot(x_annual / 1e6, curve / 1e6, color=CPX_NAVY, linewidth=2.5)

# ── Marginal ROI = 1 Efficiency Breakeven ($15M) ────────────────────────────────────
roi_threshold_m = 14.4
roi_att_idx = np.argmin(np.abs(x_annual / 1e6 - roi_threshold_m))
roi_att_m = curve[roi_att_idx] / 1e6
ax.axvline(roi_threshold_m, color=CPX_GOLD, linewidth=1.8, linestyle="--", alpha=0.9, zorder=5)
ax.scatter([roi_threshold_m], [roi_att_m], color=CPX_GOLD, s=120, zorder=6, marker="*",
           edgecolors="white", linewidths=0.8)
ax.annotate(
    "Efficiency Breakeven\nMarginal Return = \$1 per \$1\n\$14.4M / yr",
    xy=(roi_threshold_m, roi_att_m),
    xytext=(-12, 38), textcoords="offset points",
    fontsize=10, color=CPX_GOLD, fontweight="bold", ha="right",
    arrowprops=dict(arrowstyle="->", color=CPX_GOLD, lw=0.8),
    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.9, ec="none"),
)

for year, style in YEAR_MARKER_STYLES.items():
    pts = year_points[year]
    spend_m = pts["spend"] / 1e6
    att_m = pts["attendance"] / 1e6
    ax.axvline(spend_m, color=style["color"], linestyle=style["linestyle"],
               linewidth=style["linewidth"], alpha=0.85, zorder=4)
    ax.scatter([spend_m], [att_m], color=style["color"], marker=style["marker"],
              s=100, zorder=7, edgecolors="white", linewidths=1.0)


ax.set_title("EXH MMM \u2014 Total Media Response Curve\n"
             "Steady-state adstock  |  Budget shares held at 2025 mix",
             fontsize=14, fontweight="bold", color="black", pad=16)
ax.set_xlabel("Annual Media Budget ($M)", fontsize=14, fontweight="bold")
ax.set_ylabel("Annual Media Attendance Contribution (M)", fontsize=14, fontweight="bold")
ax.xaxis.set_major_locator(plt.MultipleLocator(2))
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:.0f}M"))
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1f}M"))
plt.setp(ax.get_xticklabels(), fontsize=12, fontweight="bold")
plt.setp(ax.get_yticklabels(), fontsize=12, fontweight="bold")
ax.grid(False)

legend_handles = [
    Line2D([0], [0], color=style["color"], marker=style["marker"], linestyle=style["linestyle"],
           linewidth=style["linewidth"], markersize=7,
           label=f"{year} (${year_points[year]['spend']/1e6:.1f}M annual)")
    for year, style in YEAR_MARKER_STYLES.items()
]
ax.legend(handles=legend_handles, loc="lower right", fontsize=13, frameon=True,
          facecolor="white", edgecolor="#CCCCCC", framealpha=0.95,
          markerscale=1.6, borderpad=1.0, labelspacing=0.8, handlelength=2.8,
          title="Annual Spend Reference", title_fontsize=12)

fig.text(0.02, 0.01, "* Budget distributed using 2025 actual spend shares. Hill/logistic approx via posterior mean params.",
         fontsize=7.5, color=CPX_DGREY, style="italic")
plt.tight_layout(rect=[0, 0.03, 1, 1])
save_fig(fig, "exh_b10_response_yoy.png")
display(fig); plt.close(fig)

# COMMAND ----------

# DBTITLE 1,2027 Budget Planning Meeting Revision
# ━━ B10: EXH DIMINISHING RETURN CURVE (YoY) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

scale_factor = float(scaler_y.scale_[0])

# 2025 allocation shares (hold mix constant across the curve)
df_2025_rc = df_full[df_full["year"] == 2025]
combo_weekly_2025 = {c: float(df_2025_rc[c].mean()) for c in ACTIVE_COMBOS}
total_weekly_2025 = sum(combo_weekly_2025.values())
alloc_shares = {c: combo_weekly_2025[c] / total_weekly_2025 for c in ACTIVE_COMBOS}
total_annual_2025 = total_weekly_2025 * 52

def total_media_attendance(annual_budget, shares=alloc_shares):
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

# X axis: near-zero to 10× 2025 spend
x_annual = np.linspace(52 * 100, total_annual_2025 * 10, 500)
curve = np.array([total_media_attendance(b) for b in x_annual])

# Year reference points
year_points = {}
for year in [2023, 2024, 2025]:
    df_yr = df_full[df_full["year"] == year]
    yr_spend = {c: float(df_yr[c].sum()) for c in ACTIVE_COMBOS}
    yr_active_spend = sum(yr_spend.values())
    # Include DROP_SPARSE combos so the label reflects actual total annual spend
    _sparse = [c for c in ["Awareness__Other", "Consideration__Other", "Conversion__Other"] if c in df_yr.columns]
    yr_total_spend = yr_active_spend + float(df_yr[_sparse].sum().sum()) if _sparse else yr_active_spend
    yr_shares = {c: (yr_spend[c] / yr_active_spend if yr_active_spend > 0 else alloc_shares[c])
                 for c in ACTIVE_COMBOS}
    yr_att = total_media_attendance(yr_total_spend, shares=yr_shares)
    year_points[year] = {"spend": yr_total_spend, "attendance": yr_att}

YEAR_MARKER_STYLES = {
    2023: {"color": CPX_DGREY, "marker": "o", "linestyle": "--",  "linewidth": 1.2},
    2024: {"color": CPX_NAVY,  "marker": "s", "linestyle": "-.", "linewidth": 1.2},
    2025: {"color": CPX_RED,   "marker": "D", "linestyle": "-",  "linewidth": 1.5},
}

# ── 2026 Budget / 2026+30% / 2027 (Budget) scenarios (2025 mix) ─────────────────────
budget_2026 = 4_786_700
budget_2027 = 5_070_000
for yr, b in {"2026 Budget": budget_2026, "2026+30%": budget_2026 * 1.30, "2027 Budget": budget_2027}.items():
    year_points[yr] = {"spend": b, "attendance": total_media_attendance(b)}
YEAR_MARKER_STYLES["2026 Budget"] = {"color": "#2E8B57", "marker": "^", "linestyle": ":", "linewidth": 1.5}
YEAR_MARKER_STYLES["2026+30%"] = {"color": "#7B3FA0", "marker": "v", "linestyle": ":", "linewidth": 1.5}
YEAR_MARKER_STYLES["2027 Budget"] = {"color": "#CC5500", "marker": "P", "linestyle": ":", "linewidth": 1.5}

for yr in ["2026 Budget", "2026+30%", "2027 Budget"]:
    print(f"{yr}: ${year_points[yr]['spend']/1e6:.2f}M -> {year_points[yr]['attendance']/1e6:.2f}M")
print(f"2026+30% vs 2026 Budget lift: "
      f"{(year_points['2026+30%']['attendance'] - year_points['2026 Budget']['attendance'])/1e6:.2f}M")

# Response Curve 
fig, ax = plt.subplots(figsize=(16, 9))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")
ax.plot(x_annual / 1e6, curve / 1e6, color=CPX_NAVY, linewidth=2.5)

# ── Marginal ROI = 1 Efficiency Breakeven ($15M) ────────────────────────────────────
roi_threshold_m = 14.4
roi_att_idx = np.argmin(np.abs(x_annual / 1e6 - roi_threshold_m))
roi_att_m = curve[roi_att_idx] / 1e6
ax.axvline(roi_threshold_m, color=CPX_GOLD, linewidth=1.8, linestyle="--", alpha=0.9, zorder=5)
ax.scatter([roi_threshold_m], [roi_att_m], color=CPX_GOLD, s=120, zorder=6, marker="*",
           edgecolors="white", linewidths=0.8)
ax.annotate(
    "Efficiency Breakeven\nMarginal Return = \$1 per \$1\n\$14.4M / yr",
    xy=(roi_threshold_m, roi_att_m),
    xytext=(-12, 38), textcoords="offset points",
    fontsize=10, color=CPX_GOLD, fontweight="bold", ha="right",
    arrowprops=dict(arrowstyle="->", color=CPX_GOLD, lw=0.8),
    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.9, ec="none"),
)

for year, style in YEAR_MARKER_STYLES.items():
    pts = year_points[year]
    spend_m = pts["spend"] / 1e6
    att_m = pts["attendance"] / 1e6
    ax.axvline(spend_m, color=style["color"], linestyle=style["linestyle"],
               linewidth=style["linewidth"], alpha=0.85, zorder=4)
    ax.scatter([spend_m], [att_m], color=style["color"], marker=style["marker"],
              s=100, zorder=7, edgecolors="white", linewidths=1.0)


ax.set_title("EXH MMM — Total Media Response Curve\n"
             "Steady-state adstock  |  Budget shares held at 2025 mix",
             fontsize=14, fontweight="bold", color="black", pad=16)
ax.set_xlabel("Annual Media Budget ($M)", fontsize=14, fontweight="bold")
ax.set_ylabel("Annual Media Attendance Contribution (M)", fontsize=14, fontweight="bold")
ax.xaxis.set_major_locator(plt.MultipleLocator(2))
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:.0f}M"))
ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=12))
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1f}M"))
plt.setp(ax.get_xticklabels(), fontsize=12, fontweight="bold")
plt.setp(ax.get_yticklabels(), fontsize=12, fontweight="bold")
ax.grid(False)

legend_handles = [
    Line2D([0], [0], color=style["color"], marker=style["marker"], linestyle=style["linestyle"],
           linewidth=style["linewidth"], markersize=7,
           label=f"{year} (${year_points[year]['spend']/1e6:.1f}M annual)")
    for year, style in YEAR_MARKER_STYLES.items()
]
ax.legend(handles=legend_handles, loc="lower right", fontsize=13, frameon=True,
          facecolor="white", edgecolor="#CCCCCC", framealpha=0.95,
          markerscale=1.6, borderpad=1.0, labelspacing=0.8, handlelength=2.8,
          title="Annual Spend Reference", title_fontsize=12)

fig.text(0.02, 0.01, "* Budget distributed using 2025 actual spend shares. Hill/logistic approx via posterior mean params.",
         fontsize=7.5, color=CPX_DGREY, style="italic")
plt.tight_layout(rect=[0, 0.03, 1, 1])
save_fig(fig, "exh_b10_response_yoy.png")
display(fig); plt.close(fig)

# COMMAND ----------

# DBTITLE 1,EXH B10b — Marginal Return per $1
# ━━ B10b: MARGINAL RETURN CURVE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

fig2, ax2 = plt.subplots(figsize=(16, 9))
fig2.patch.set_facecolor("white")
ax2.set_facecolor("white")

marginal = np.gradient(curve, x_annual)
ax2.plot(x_annual / 1e6, marginal, color=CPX_NAVY, linewidth=2.5)
# ── Inefficient spending zone: red band where marginal return falls below $1/$1 ──
ax2.axhspan(0, 1, alpha=0.12, color=CPX_RED, zorder=0)
ax2.axhline(1, color=CPX_RED, lw=1.8, alpha=0.85, zorder=6)
ax2.text(
    x_annual[-1] / 1e6 * 0.60, 0.5,
    "Inefficient Spending Zone\nMarginal Return < \$1 per \$1",
    fontsize=10, color=CPX_RED, fontweight="bold", alpha=0.9,
    va="center", ha="center",
    bbox=dict(boxstyle="round,pad=0.35", fc="white", alpha=0.8, ec=CPX_RED, lw=0.6),
)

# ── ROI = 1 crossover: vertical drop to x-axis ───────────────────────────────────────
_roi1_idx = np.argmin(np.abs(marginal - 1.0))
_roi1_x_m = x_annual[_roi1_idx] / 1e6
_roi1_y = marginal[_roi1_idx]
ax2.vlines(_roi1_x_m, ymin=0, ymax=_roi1_y, color=CPX_GOLD, linewidth=2.2,
           linestyle="--", alpha=0.9, zorder=5)
ax2.scatter([_roi1_x_m], [_roi1_y], color=CPX_GOLD, s=140, zorder=8,
            marker="*", edgecolors="white", linewidths=0.8)
ax2.annotate(f"${_roi1_x_m:.1f}M",
             xy=(_roi1_x_m, 0), xytext=(5, 6), textcoords="offset points",
             fontsize=10, color=CPX_GOLD, fontweight="bold", ha="left")

for year, style in YEAR_MARKER_STYLES.items():
    pts = year_points[year]
    idx = np.argmin(np.abs(x_annual - pts["spend"]))
    ax2.axvline(pts["spend"] / 1e6, color=style["color"], linestyle=style["linestyle"],
                linewidth=2.2, alpha=0.85, zorder=4)
    ax2.scatter([pts["spend"] / 1e6], [marginal[idx]], color=style["color"],
               marker=style["marker"], s=100, zorder=7, edgecolors="white", linewidths=1.0)


ax2.set_title("EXH MMM \u2014 Marginal Return per $1 Spent\n"
              "Incremental attendance per additional dollar",
              fontsize=14, fontweight="bold", color="black", pad=16)
ax2.set_xlabel("Annual Media Budget ($M)", fontsize=14, fontweight="bold")
ax2.set_ylabel("Incremental Attendance per $1", fontsize=14, fontweight="bold")
ax2.xaxis.set_major_locator(plt.MultipleLocator(2))
ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:.0f}M"))
plt.setp(ax2.get_xticklabels(), fontsize=12, fontweight="bold")
plt.setp(ax2.get_yticklabels(), fontsize=12, fontweight="bold")

legend_handles_m = [
    Line2D([0], [0], color=style["color"], marker=style["marker"], linestyle=style["linestyle"],
           linewidth=style["linewidth"], markersize=7,
           label=f"{year} (${year_points[year]['spend']/1e6:.1f}M annual)")
    for year, style in YEAR_MARKER_STYLES.items()
]
ax2.legend(handles=legend_handles_m, loc="upper right", fontsize=13, frameon=True,
           facecolor="white", edgecolor="#CCCCCC", framealpha=0.95,
           markerscale=1.6, borderpad=1.0, labelspacing=0.8, handlelength=2.8,
           title="Annual Spend Reference", title_fontsize=12)
ax2.set_ylim(bottom=0)
ax2.grid(False)

fig2.text(0.02, 0.01, "* Budget distributed using 2025 actual spend shares. Hill/logistic approx via posterior mean params.",
          fontsize=7.5, color=CPX_DGREY, style="italic")
plt.tight_layout(rect=[0, 0.03, 1, 1])
save_fig(fig2, "exh_b10b_marginal_return.png")
display(fig2); plt.close(fig2)

# COMMAND ----------

# DBTITLE 1,Section 10 — Presentation Charts
# MAGIC %md
# MAGIC ---
# MAGIC ## Section 10: Presentation Charts
# MAGIC Annual media spend by funnel, attendance breakdown, feature correlations, annual decomposition, tentpole/content tier analysis, LOB spend & contribution, flighting.

# COMMAND ----------

# DBTITLE 1,EXH Annual Media Spend by Funnel
# ━━ EXH ANNUAL MEDIA SPEND BY FUNNEL (2023–2025) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Compute funnel spend directly from df_full media columns

_funnels_list = ["Awareness", "Consideration", "Conversion"]
_media_cols_all = [c for c in df_full.columns if any(c.startswith(f + "_") for f in _funnels_list)
                   and "__" not in c and "media_" not in c]

awareness, consideration, conversion = [], [], []
for year in [2023, 2024, 2025]:
    yr = df_full[df_full["year"] == year]
    awareness.append(yr[[c for c in _media_cols_all if c.startswith("Awareness_")]].sum().sum() / 1e6)
    consideration.append(yr[[c for c in _media_cols_all if c.startswith("Consideration_")]].sum().sum() / 1e6)
    conversion.append(yr[[c for c in _media_cols_all if c.startswith("Conversion_")]].sum().sum() / 1e6)

years = ['2023', '2024', '2025']

fig, ax = plt.subplots(figsize=(11, 9))

x     = np.arange(len(years))
width = 0.55

b1 = ax.bar(x, awareness,     width, label="Awareness",     color=CPX_NAVY)
b2 = ax.bar(x, consideration, width, label="Consideration", color=CPX_RED,
            bottom=awareness)
b3 = ax.bar(x, conversion,    width, label="Conversion",    color=CPX_GOLD,
            bottom=[a + c for a, c in zip(awareness, consideration)])

def label_segments(bars, bottoms):
    for i, bar in enumerate(bars):
        h = bar.get_height()
        b = bottoms[i]
        if h >= 0.1:
            ax.text(bar.get_x() + bar.get_width() / 2, b + h / 2,
                    f"${h:.2f}M", ha="center", va="center",
                    fontsize=14, fontweight="bold", color="white")

label_segments(b1, [0]*3)
label_segments(b2, awareness)
label_segments(b3, [a+c for a, c in zip(awareness, consideration)])

totals = [a+c+cv for a,c,cv in zip(awareness, consideration, conversion)]
for i, t in enumerate(totals):
    ax.text(i, t + 0.05, f"${t:.2f}M", ha="center", fontsize=14, fontweight="bold", color=CPX_NAVY)

ax.set_xticks(x)
ax.set_xticklabels(years, fontsize=14, fontweight="bold")
plt.setp(ax.get_yticklabels(), fontsize=12, fontweight="bold")
ax.set_ylabel("Annual Media Spend ($M)", fontsize=14, fontweight="bold")
ax.set_title("EXH — Annual Media Spend by Funnel\n2023–2025",
             fontsize=14, fontweight="bold", color=CPX_NAVY)
ax.legend(loc="upper left", frameon=False, fontsize=12)
ax.set_ylim(0, max(totals) * 1.15)
ax.grid(False)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
plt.tight_layout()
save_fig(fig, "exh_annual_spend_by_funnel.png")
display(fig); plt.close(fig)

# ── Data Table ────────────────────────────────────────────────────────────────
spend_table = pd.DataFrame({
    "Year": years,
    "Awareness ($M)": [round(v, 3) for v in awareness],
    "Consideration ($M)": [round(v, 3) for v in consideration],
    "Conversion ($M)": [round(v, 3) for v in conversion],
    "Total ($M)": [round(t, 3) for t in totals],
})
print(f"{'═'*60}")
print(f"  EXH Annual Media Spend by Funnel (2023–2025)")
print(f"{'═'*60}")
display(spend_table)

# COMMAND ----------

# DBTITLE 1,EXH Spend by Original LOB Per Year
# ━━ EXH SPEND BY ORIGINAL LOB — STACKED BY FUNNEL — ONE PLOT PER YEAR ━━━━━━━━━━━━━━━
import matplotlib.ticker as mticker

# ── Parse Media Columns ─────────────────────────────────────────────────────────────
funnels_chart = ["Awareness", "Consideration", "Conversion"]
media_cols_chart = [c for c in df_full.columns if any(c.startswith(f + "_") for f in funnels_chart)
                   and "__" not in c and "media_" not in c]

def parse_col(col):
    for funnel in funnels_chart:
        if col.startswith(funnel + "_"):
            rest    = col[len(funnel) + 1:]
            channel = rest.rsplit("_", 1)[-1]
            lob     = rest.rsplit("_", 1)[0]
            return funnel, lob, channel
    return None, None, None

meta_chart = pd.DataFrame(
    [parse_col(c) + (c,) for c in media_cols_chart],
    columns=["Funnel", "LOB", "Channel", "col_name"]
).dropna(subset=["Funnel"])

# ── Aggregate Spend ───────────────────────────────────────────────────────────────
rows_spend = []
for _, row in meta_chart.iterrows():
    for year in [2023, 2024, 2025]:
        spend = df_full.loc[df_full["year"] == year, row["col_name"]].sum()
        rows_spend.append({
            "Funnel" : row["Funnel"],
            "LOB"    : row["LOB"],
            "Year"   : year,
            "Spend_M": round(spend / 1e6, 3)
        })

spend_df = (
    pd.DataFrame(rows_spend)
    .groupby(["Funnel", "LOB", "Year"])["Spend_M"]
    .sum()
    .reset_index()
)
spend_df = spend_df[spend_df["Spend_M"] > 0.001].copy()

# ── Funnel Colours ────────────────────────────────────────────────────────────────
funnel_colors = {
    "Awareness"    : CPX_NAVY,
    "Consideration": CPX_RED,
    "Conversion"   : CPX_GOLD,
}

# ── One Plot Per Year (vertical stacked bars) ───────────────────────────────────
for year in [2023, 2024, 2025]:
    ydata = spend_df[spend_df["Year"] == year].copy()

    lob_totals = ydata.groupby("LOB")["Spend_M"].sum().sort_values(ascending=False)
    lobs       = lob_totals.index.tolist()
    x          = np.arange(len(lobs))

    fig, ax = plt.subplots(figsize=(14, 8))
    fig.patch.set_facecolor("white")

    bottoms = np.zeros(len(lobs))

    for funnel in funnels_chart:
        heights = []
        for lob in lobs:
            val = ydata.loc[
                (ydata["Funnel"] == funnel) & (ydata["LOB"] == lob), "Spend_M"
            ]
            heights.append(float(val.values[0]) if len(val) > 0 else 0.0)

        bars = ax.bar(
            x, heights,
            bottom=bottoms,
            label=funnel,
            color=funnel_colors[funnel],
            width=0.55,
            edgecolor="white",
            linewidth=0.5
        )

        # Segment labels
        for i, bar in enumerate(bars):
            h = bar.get_height()
            if h > 0.03:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bottoms[i] + h / 2,
                    f"${h:.2f}M",
                    ha="center", va="center",
                    fontsize=9, fontweight="bold", color="white"
                )
        bottoms += np.array(heights)

    # Total labels on top
    for i, total in enumerate(bottoms):
        if total > 0:
            ax.text(
                x[i], total + 0.01,
                f"${total:.2f}M",
                ha="center", va="bottom",
                fontsize=11, fontweight="bold", color="black"
            )

    ax.set_xticks(x)
    ax.set_xticklabels(lobs, rotation=35, ha="right", fontsize=12, fontweight="bold")
    ax.set_ylabel("Annual Spend ($M)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Line of Business", fontsize=14, fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:.2f}M"))
    plt.setp(ax.get_yticklabels(), fontsize=12, fontweight="bold")
    ax.set_title(f"EXH Annual Media Spend by LOB & Funnel — {year}",
                 fontsize=13, fontweight="bold", color=CPX_NAVY)
    ax.legend(title="Funnel", fontsize=9, title_fontsize=10, loc="upper right")
    ax.grid(False)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    save_fig(fig, f"exh_spend_by_lob_{year}.png")
    display(fig); plt.close(fig)

# ── Data Tables ───────────────────────────────────────────────────────────────────
for year in [2023, 2024, 2025]:
    pivot = (
        spend_df[spend_df["Year"] == year]
        .pivot_table(index="LOB", columns="Funnel", values="Spend_M", aggfunc="sum", fill_value=0)
        .reindex(columns=["Awareness", "Consideration", "Conversion"]).fillna(0)
    )
    pivot = pivot * 100  # $10K units
    pivot["Total"] = pivot.sum(axis=1)
    pivot = pivot.sort_values("Total", ascending=False).round(2).reset_index()
    print(f"\n{'═'*60}")
    print(f"  EXH Annual Media Spend by LOB & Funnel — {year} (in $10K)")
    print(f"{'═'*60}")
    display(pivot)

# COMMAND ----------

# DBTITLE 1,EXH Attendance Breakdown by Year
# ━━ EXH ATTENDANCE BREAKDOWN BY YEAR ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

att_df = (
    df_full.groupby("year")[target_col]
    .sum()
    .reset_index()
    .rename(columns={target_col: "Attendance"})
)
att_df = att_df[att_df["year"].isin([2023, 2024, 2025])].copy()
att_df["Attendance_M"] = att_df["Attendance"] / 1e6

# Add 2026 forecast (hard-coded per planning)
forecast_row = pd.DataFrame([
    {"year": 2026, "Attendance": 46_400_000, "Attendance_M": 46.4},
    {"year": 2027, "Attendance": 47_400_000, "Attendance_M": 47.4},
])
att_df = pd.concat([att_df, forecast_row], ignore_index=True)
att_df["YoY_pct"] = att_df["Attendance_M"].pct_change() * 100

fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(att_df))
bars = ax.bar(x, att_df["Attendance_M"], width=0.5, color="black", edgecolor="white")

for i, bar in enumerate(bars):
    v = att_df["Attendance_M"].iloc[i]
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.5,
            f"{v:.1f}M", ha="center", fontsize=13, fontweight="bold", color=CPX_NAVY)
    yoy = att_df["YoY_pct"].iloc[i]
    if pd.notna(yoy):
        color = CPX_GREEN if yoy > 0 else CPX_RED
        ax.text(bar.get_x() + bar.get_width()/2, v + 3.5,
                f"{yoy:+.1f}%", ha="center", fontsize=12, color=color, fontweight="bold")

labels = [str(int(y)) + (" (F)" if y >= 2026 else "") for y in att_df["year"]]
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=12, fontweight="bold")
ax.set_ylabel("Annual Attendance (M)", fontsize=14, fontweight="bold")
plt.setp(ax.get_yticklabels(), fontsize=12, fontweight="bold")
ax.set_title("EXH — Total Annual Attendance\n2023–2027 (Forecast)",
             fontsize=13, fontweight="bold", color=CPX_NAVY)
ax.set_ylim(0, att_df["Attendance_M"].max() * 1.4)
ax.grid(False)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
plt.tight_layout()
save_fig(fig, "exh_attendance_by_year.png")
display(fig); plt.close(fig)

# Data Table
att_table = att_df[["year", "Attendance_M", "YoY_pct"]].copy()
att_table.columns = ["Year", "Total Attendance (M)", "YoY Change (%)"]
att_table["Total Attendance (M)"] = att_table["Total Attendance (M)"].round(2)
att_table["YoY Change (%)"] = att_table["YoY Change (%)"].round(1)
print(f"{'═'*60}")
print(f"  EXH Total Annual Attendance (2023–2027F)")
print(f"{'═'*60}")
display(att_table)

# COMMAND ----------

# DBTITLE 1,EXH Feature Correlations with Attendance
# ━━ FEATURE CORRELATIONS WITH ATTENDANCE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
import matplotlib.patches as mpatches

feature_groups = {
    "Seasonality": ["season_spring", "season_summer", "season_fall", "is_2024", "is_2025", "week_index"],
    "Calendar": ["stat_holiday_in_week", "is_long_weekend", "days_to_christmas"],
    "Content": ["active_films", "cat1_share", "cat2_share", "cat3_share", "cat4_share", "cat5_share"],
    "Weather": ["national_mean_temp", "national_rain_mm", "national_snow_cm", "bad_weather_flag"],
    "Economic": ["headline_cpi"],
}

display_names = {
    "season_spring": "Spring Season", "season_summer": "Summer Season",
    "season_fall": "Fall Season", "is_2024": "Year — 2024",
    "is_2025": "Year — 2025", "week_index": "Linear Trend",
    "stat_holiday_in_week": "Stat Holiday", "is_long_weekend": "Long Weekend",
    "days_to_christmas": "Days to Christmas", "active_films": "Active Films",
    "cat1_share": "Category 1 Share", "cat2_share": "Category 2 Share",
    "cat3_share": "Category 3 Share", "cat4_share": "Category 4 Share",
    "cat5_share": "Category 5 Share", "national_mean_temp": "Mean Temperature",
    "national_rain_mm": "Rainfall", "national_snow_cm": "Snowfall",
    "bad_weather_flag": "Bad Weather", "headline_cpi": "CPI",
}

feature_order = []
for group, features in feature_groups.items():
    feature_order.extend(features)

# Correlation with target
corr = df_full[feature_order + [target_col]].corr()[target_col].drop(target_col)
corr.index = [display_names.get(f, f) for f in corr.index]

group_colors = {"Seasonality": CPX_NAVY, "Calendar": CPX_RED, "Content": CPX_GOLD,
                "Weather": CPX_GREEN, "Economic": CPX_DGREY}
bar_colors = []
for group, features in feature_groups.items():
    bar_colors.extend([group_colors[group]] * len(features))

fig, ax = plt.subplots(figsize=(12, 8))
ax.barh(corr.index, corr.values, color=bar_colors, alpha=0.85)
ax.axvline(0, color="black", lw=0.8)
ax.set_xlabel("Pearson Correlation with Weekly Attendance", fontsize=14, fontweight="bold")
ax.set_title("EXH — Baseline Feature Correlations with Attendance",
             fontsize=13, fontweight="bold", color=CPX_NAVY)
plt.setp(ax.get_xticklabels(), fontsize=11, fontweight="bold")
plt.setp(ax.get_yticklabels(), fontsize=11, fontweight="bold")
ax.grid(False)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

legend_handles = [mpatches.Patch(color=c, label=g) for g, c in group_colors.items()]
ax.legend(handles=legend_handles, loc="lower right", frameon=False, fontsize=9)
plt.tight_layout()
save_fig(fig, "exh_feature_correlations.png")
display(fig); plt.close(fig)

# Data table
rows_feat = []
for group, features in feature_groups.items():
    for f in features:
        disp = display_names.get(f, f)
        rows_feat.append({"Category": group, "Feature": disp, "Correlation": round(corr.get(disp, 0), 3)})
print(f"{'═'*60}\n  EXH Model — Feature Categories & Correlations\n{'═'*60}")
display(pd.DataFrame(rows_feat))

# COMMAND ----------

# DBTITLE 1,EXH Annual Attendance Decomposition (2023-2026F)
# ── EXH ANNUAL ATTENDANCE DECOMPOSITION BY FUNNEL + 2026 FORECAST ────────────
# Uses: df_full, FUNNEL_COMBOS, target_col, CPX_*, save_fig from cell 23.
import matplotlib.patheffects as pe
import matplotlib.ticker as mticker

CPX_MGREY = "#9AA0A6"

# ── Aggregate Annual by Funnel ────────────────────────────────────────────────
agg_dict = {target_col: "sum"}
for funnel in FUNNEL_COMBOS:
    agg_dict[f"media_{funnel}"] = "sum"

annual      = df_full[df_full["year"] != 2026].groupby("year").agg(agg_dict).round(0)
years_avail = sorted(annual.index.tolist())
actual_vals = annual.loc[years_avail, target_col].values

awareness_vals     = annual.loc[years_avail, "media_Awareness"].values
consideration_vals = annual.loc[years_avail, "media_Consideration"].values
conversion_vals    = annual.loc[years_avail, "media_Conversion"].values
baseline_vals      = actual_vals - awareness_vals - consideration_vals - conversion_vals

# ── 2026 Forecast ─────────────────────────────────────────────────────────────
forecast_2026    = 46_384_441

# Compute actual media share from model decomposition (not hard-coded)
media_2025_total = awareness_vals[-1] + consideration_vals[-1] + conversion_vals[-1]
media_share_2025 = media_2025_total / actual_vals[-1]  # derived from model output
aw_share         = awareness_vals[-1]     / media_2025_total
csd_share        = consideration_vals[-1] / media_2025_total
cvs_share        = conversion_vals[-1]    / media_2025_total

media_2026_total = forecast_2026 * media_share_2025
aw_2026          = media_2026_total * aw_share
csd_2026         = media_2026_total * csd_share
cvs_2026         = media_2026_total * cvs_share
baseline_2026    = forecast_2026 - media_2026_total

forecast_2027    = 47_400_000
media_2027_total = forecast_2027 * media_share_2025
aw_2027          = media_2027_total * aw_share
csd_2027         = media_2027_total * csd_share
cvs_2027         = media_2027_total * cvs_share
baseline_2027    = forecast_2027 - media_2027_total

# ── Combine Actuals + Forecast ────────────────────────────────────────────────
all_years      = years_avail + [2026, 2027]
all_actual     = list(actual_vals) + [forecast_2026, forecast_2027]
all_baseline   = list(baseline_vals) + [baseline_2026, baseline_2027]
all_awareness  = list(awareness_vals) + [aw_2026, aw_2027]
all_consid     = list(consideration_vals) + [csd_2026, csd_2027]
all_conversion = list(conversion_vals) + [cvs_2026, cvs_2027]

# ── Helper ────────────────────────────────────────────────────────────────────
def bar_label(ax, x_pos, bottom, height, pct, color, min_height=0.5):
    if abs(height) / 1e6 < min_height:
        return
    light_colors = {CPX_LGREY, CPX_GOLD}
    txt_color = "black" if color in light_colors else "white"
    ax.text(
        x_pos,
        (bottom + height / 2) / 1e6,
        f"{pct:.0f}%",
        ha="center",
        va="center",
        fontsize=18,
        fontweight="bold",
        color=txt_color,
        path_effects=[pe.withStroke(
            linewidth=2,
            foreground="white" if txt_color == "black" else "black"
        )],
    )

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(16, 9))
fig.patch.set_facecolor("white")

x = np.arange(len(all_years))
w = 0.5
alpha = [0.9, 0.9, 0.9, 0.5, 0.5]  # 2026/2027 bars slightly transparent to indicate forecast

for i, year in enumerate(all_years):
    a = alpha[i]
    b_base = all_baseline[i]
    b_aw   = all_awareness[i]
    b_csd  = all_consid[i]
    b_cvs  = all_conversion[i]
    total  = all_actual[i]

    ax.bar(x[i], b_base / 1e6,
           width=w, color=CPX_LGREY, alpha=a,
           edgecolor="white", linewidth=0.5,
           label="Baseline (non-media)" if i == 0 else "_nolegend_")
    ax.bar(x[i], b_aw / 1e6,
           width=w, bottom=b_base / 1e6,
           color=CPX_NAVY, alpha=a,
           edgecolor="white", linewidth=0.5,
           label="Awareness" if i == 0 else "_nolegend_")
    ax.bar(x[i], b_csd / 1e6,
           width=w, bottom=(b_base + b_aw) / 1e6,
           color=CPX_RED, alpha=a,
           edgecolor="white", linewidth=0.5,
           label="Consideration" if i == 0 else "_nolegend_")
    ax.bar(x[i], b_cvs / 1e6,
           width=w, bottom=(b_base + b_aw + b_csd) / 1e6,
           color=CPX_GOLD, alpha=a,
           edgecolor="white", linewidth=0.5,
           label="Conversion" if i == 0 else "_nolegend_")

    bar_label(ax, x[i], 0,                     b_base, b_base / total * 100, CPX_LGREY)
    bar_label(ax, x[i], b_base,                b_aw,   b_aw   / total * 100, CPX_NAVY, 1.5)
    bar_label(ax, x[i], b_base + b_aw,         b_csd,  b_csd  / total * 100, CPX_RED,  1.5)
    bar_label(ax, x[i], b_base + b_aw + b_csd, b_cvs,  b_cvs  / total * 100, CPX_GOLD, 1.5)

    label = f"{total/1e6:.2f}M" + (" (F)" if year >= 2026 else "")
    ax.text(x[i], total / 1e6 + 0.3, label,
            ha="center", va="bottom",
            fontsize=16, fontweight="bold",
            color=CPX_RED if year == 2026 else CPX_DGREY)

# Actual dots (2023-2025 only)
ax.scatter(x[:3], np.array(actual_vals) / 1e6,
           color="black", zorder=5, s=80,
           marker="D", label="Actual Attendance")

# Forecast marker
ax.scatter(x[3:], [forecast_2026 / 1e6, forecast_2027 / 1e6],
           color=CPX_RED, zorder=5, s=80,
           marker="D", label="2026/2027 Forecast")

ax.axvline(x=2.75, color=CPX_MGREY, linewidth=1, linestyle="--")
ax.text(2.78, 2, "Forecast →",
        fontsize=9, color=CPX_MGREY, fontstyle="italic")

ax.set_xticks(x)
ax.set_xticklabels(all_years, fontsize=16, fontweight="bold")
plt.setp(ax.get_yticklabels(), fontsize=16, fontweight="bold", color="black")
ax.set_ylabel("Annual Attendance (M)", fontsize=16, fontweight="bold")
ax.set_xlabel("Year", fontsize=16, fontweight="bold")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}M"))
ax.set_title("EXH Annual Attendance Decomposition by Funnel (2023–2027F)", loc="center",
             fontsize=18, fontweight="bold", color=CPX_NAVY)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=3, fontsize=16, frameon=False)
ax.grid(False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
save_fig(fig, "exh_annual_decomposition.png")
display(fig)
plt.close(fig)

# ── Data table retained for downstream export ────────────────────────────────
decomp_annual_table = pd.DataFrame({
    "Year": [str(y) + (" (F)" if y >= 2026 else "") for y in all_years],
    "Baseline (M)": [round(v / 1e6, 2) for v in all_baseline],
    "Awareness (M)": [round(v / 1e6, 2) for v in all_awareness],
    "Consideration (M)": [round(v / 1e6, 2) for v in all_consid],
    "Conversion (M)": [round(v / 1e6, 2) for v in all_conversion],
    "Media Total (M)": [round((a + c + cv) / 1e6, 2) for a, c, cv in zip(all_awareness, all_consid, all_conversion)],
    "Total (M)": [round(v / 1e6, 2) for v in all_actual],
    "Media %": [round((a + c + cv) / t * 100, 1) for a, c, cv, t in zip(all_awareness, all_consid, all_conversion, all_actual)],
})
print(f"{'═'*60}\n  EXH Attendance Decomposition by Funnel (2023-2026F)\n{'═'*60}")
display(decomp_annual_table)

# COMMAND ----------

# DBTITLE 1,EXH Funnel Efficiency — Spend, Contribution & Att/$ by Year
# ━━ EXH FUNNEL EFFICIENCY — SPEND, CONTRIBUTION & ATT/$ BY YEAR ━━━━━━━━━━━━━━━━━━━━
import matplotlib.ticker as mticker
import matplotlib.patheffects as pe

funnels_eff = ["Awareness", "Consideration", "Conversion"]
years_eff   = [2023, 2024, 2025]

# Raw spend columns (original media, no 'media_' prefix, no '__')
_spend_cols_all = [c for c in df_full.columns if any(c.startswith(f + "_") for f in funnels_eff)
                   and "__" not in c and "media_" not in c]

# ── Build efficiency DataFrame ─────────────────────────────────────────────────
eff_rows = []
for year in years_eff:
    yr = df_full[df_full["year"] == year]
    for funnel in funnels_eff:
        f_spend_cols = [c for c in _spend_cols_all if c.startswith(funnel + "_")]
        spend  = yr[f_spend_cols].sum().sum()
        contrib = yr[f"media_{funnel}"].sum()
        eff_rows.append({
            "Year": year,
            "Funnel": funnel,
            "Spend_M": spend / 1e6,
            "Contribution_M": contrib / 1e6,
            "Att_per_Dollar": contrib / spend if spend > 0 else 0
        })

eff_df = pd.DataFrame(eff_rows)

# ── Funnel colours & styling ──────────────────────────────────────────────────
funnel_colors_eff = {"Awareness": CPX_NAVY, "Consideration": CPX_RED, "Conversion": CPX_GOLD}
funnel_markers    = {"Awareness": "o", "Consideration": "s", "Conversion": "D"}

# ── Figure: 1×3 subplots (one per funnel) ─────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 7), sharey=False)
fig.patch.set_facecolor("white")

for idx, funnel in enumerate(funnels_eff):
    ax  = axes[idx]
    ax2 = ax.twinx()
    fdata = eff_df[eff_df["Funnel"] == funnel]
    color = funnel_colors_eff[funnel]
    x = np.arange(len(years_eff))
    w = 0.32

    # Bars: Spend (left) and Contribution (right)
    bars_sp = ax.bar(x - w/2, fdata["Spend_M"], width=w,
                     color=color, alpha=0.35, edgecolor=color, linewidth=1.2,
                     label="Spend ($M)")
    bars_ct = ax.bar(x + w/2, fdata["Contribution_M"], width=w,
                     color=color, alpha=0.85, edgecolor="white", linewidth=0.5,
                     label="Contribution (M att)")

    # Spend labels
    for i, bar in enumerate(bars_sp):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.05,
                f"${h:.2f}M", ha="center", va="bottom",
                fontsize=9, color=color, fontweight="bold")

    # Contribution labels
    for i, bar in enumerate(bars_ct):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.05,
                f"{h:.2f}M", ha="center", va="bottom",
                fontsize=9, color="black", fontweight="bold")

    # Line: Att/$ on secondary axis
    att_vals = fdata["Att_per_Dollar"].values
    line = ax2.plot(x, att_vals, color="black", linewidth=2.5, zorder=5,
                    marker=funnel_markers[funnel], markersize=10,
                    markerfacecolor="white", markeredgecolor="black", markeredgewidth=2)

    # Att/$ labels with white outline for readability
    for i, val in enumerate(att_vals):
        ax2.annotate(
            f"{val:.2f}",
            xy=(x[i], val), xytext=(0, 12),
            textcoords="offset points", ha="center",
            fontsize=11, fontweight="bold", color="black",
            path_effects=[pe.withStroke(linewidth=3, foreground="white")]
        )

    # Formatting
    ax.set_xticks(x)
    ax.set_xticklabels(years_eff, fontsize=12, fontweight="bold")
    ax.set_ylabel("$M Spend / M Attendance" if idx == 0 else "", fontsize=13, fontweight="bold")
    ax2.set_ylabel("Att / $" if idx == 2 else "", fontsize=13, fontweight="bold", color="black")
    ax.set_title(funnel, fontsize=14, fontweight="bold", color=color, pad=12)
    ax.set_ylim(0, max(fdata["Contribution_M"].max(), fdata["Spend_M"].max()) * 1.35)
    ax2.set_ylim(0, eff_df["Att_per_Dollar"].max() * 1.4)
    ax.grid(False)
    ax2.grid(False)
    plt.setp(ax.get_yticklabels(), fontsize=10, fontweight="bold")
    plt.setp(ax2.get_yticklabels(), fontsize=10, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)

    if idx == 0:
        ax.legend(loc="upper left", fontsize=8, frameon=False)
        # Add att/$ legend entry
        ax2.plot([], [], color="black", linewidth=2.5, marker="o",
                 markerfacecolor="white", markeredgecolor="black",
                 label="Att/$ Efficiency")
        ax2.legend(loc="upper right", fontsize=8, frameon=False)

fig.suptitle("EXH — Funnel Efficiency: Spend, Contribution & Att/$ (2023–2025)",
             fontsize=14, fontweight="bold", color=CPX_NAVY, y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.94])
save_fig(fig, "exh_funnel_efficiency_by_year.png")
display(fig)
plt.close(fig)

# ── Summary Data Table ────────────────────────────────────────────────────────
eff_table = eff_df.copy()
eff_table["Spend ($M)"]       = eff_table["Spend_M"].round(3)
eff_table["Contribution (M)"] = eff_table["Contribution_M"].round(3)
eff_table["Att/$"]            = eff_table["Att_per_Dollar"].round(2)
eff_table["YoY Eff Δ"]       = eff_table.groupby("Funnel")["Att_per_Dollar"].pct_change().round(3) * 100

eff_display = eff_table[["Year", "Funnel", "Spend ($M)", "Contribution (M)", "Att/$", "YoY Eff Δ"]].copy()
eff_display["YoY Eff Δ"] = eff_display["YoY Eff Δ"].apply(lambda v: f"{v:+.1f}%" if pd.notna(v) else "—")

print(f"{'═'*70}")
print(f"  EXH Funnel Efficiency Summary (2023–2025)")
print(f"{'═'*70}")
print(f"  Att/$ = Attendance driven per dollar of media spend")
print(f"{'─'*70}")
display(eff_display)

# COMMAND ----------

# DBTITLE 1,EXH Funnel Planning Parameters — Adstock & Lead Time
# ━━ EXH FUNNEL ADSTOCK DECAY CURVES — CARRY-OVER & PLANNING LEAD TIME ━━━━━━━━━━━━
# Shows how each funnel's media impact decays week-over-week after activation
import matplotlib.patheffects as pe

# Adstock decay parameters (absolute alpha = carryover rate per week)
decay_params = {
    "Awareness":     {"alpha": 0.75, "color": CPX_NAVY,  "carryover": "4–5 wks", "lead": "3–4 wks before"},
    "Consideration": {"alpha": 0.47, "color": CPX_RED,   "carryover": "2–3 wks", "lead": "2–3 wks before"},
    "Conversion":    {"alpha": 0.21, "color": CPX_GOLD,  "carryover": "~1 wk",   "lead": "Same week"},
}

weeks = np.arange(0, 9, 0.1)  # 0 to 8 weeks, smooth curve
weeks_int = np.arange(0, 9)    # integer weeks for markers

fig, ax = plt.subplots(figsize=(14, 7))
fig.patch.set_facecolor("white")

for funnel, params in decay_params.items():
    alpha = params["alpha"]
    color = params["color"]

    # Decay curve: effect remaining = alpha^t (impulse response)
    decay_smooth = alpha ** weeks * 100
    decay_int    = alpha ** weeks_int * 100

    # Main curve
    ax.plot(weeks, decay_smooth, color=color, linewidth=3, label=funnel, zorder=3)
    ax.scatter(weeks_int, decay_int, color=color, s=60, zorder=4,
              edgecolors="white", linewidths=1.2)

    # Shade the carry-over window
    if funnel == "Awareness":
        shade_end = 5
    elif funnel == "Consideration":
        shade_end = 3
    else:
        shade_end = 1

    shade_x = weeks[weeks <= shade_end]
    shade_y = alpha ** shade_x * 100
    ax.fill_between(shade_x, 0, shade_y, color=color, alpha=0.08)

    # Annotate the carry-over window on the curve
    annot_x = shade_end
    annot_y = alpha ** annot_x * 100
    ax.annotate(
        f"α = {alpha:.2f}\n{params['carryover']} carry-over\nPlan: {params['lead']}",
        xy=(annot_x, annot_y),
        xytext=(annot_x + 0.8, annot_y + 8),
        fontsize=9, fontweight="bold", color=color,
        arrowprops=dict(arrowstyle="->", color=color, lw=1.5),
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor=color, alpha=0.9),
        path_effects=[pe.withStroke(linewidth=0, foreground="white")]
    )

# Reference lines
ax.axhline(50, color=CPX_LGREY, linewidth=1, linestyle="--", zorder=1)
ax.text(8.1, 50, "50%", fontsize=10, fontweight="bold", color=CPX_DGREY, va="center")
ax.axhline(10, color=CPX_LGREY, linewidth=1, linestyle=":", zorder=1)
ax.text(8.1, 10, "10%", fontsize=10, fontweight="bold", color=CPX_DGREY, va="center")

# Formatting
ax.set_xlabel("Weeks After Media Activation", fontsize=14, fontweight="bold", labelpad=10)
ax.set_ylabel("Remaining Effect (%)", fontsize=14, fontweight="bold", labelpad=10)
ax.set_xlim(-0.2, 8.5)
ax.set_ylim(0, 105)
ax.set_xticks(range(9))
ax.set_xticklabels([f"Wk {i}" for i in range(9)], fontsize=12, fontweight="bold")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
plt.setp(ax.get_yticklabels(), fontsize=12, fontweight="bold")
ax.grid(False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(loc="upper right", fontsize=11, frameon=True, framealpha=0.9,
          edgecolor=CPX_LGREY)

ax.set_title("EXH — Adstock Decay by Funnel: How Quickly Does Media Impact Fade?",
             fontsize=14, fontweight="bold", color=CPX_NAVY, pad=15)
ax.text(0.5, -0.14,
        "Higher α = slower decay = longer carry-over.  "
        "Shaded area = effective carry-over window.  "
        "Planning implication: activate upper-funnel media weeks ahead of target period.",
        transform=ax.transAxes, ha="center", fontsize=9, color=CPX_DGREY, style="italic")

plt.tight_layout()
save_fig(fig, "exh_funnel_adstock_decay_curves.png")
display(fig)
plt.close(fig)

# COMMAND ----------

# DBTITLE 1,Section 11 — LOB Decomposition
# MAGIC %md
# MAGIC ---
# MAGIC ## Section 11: LOB Decomposition (Graft Method)
# MAGIC Allocate validated group-level contributions to individual LOBs via spend share + residual-correlation tilt signal. Unvalidated LOBs extrapolated via group-average efficiency, flagged for geo-test.

# COMMAND ----------

# DBTITLE 1,EXH LOB Decomposition — Group Contributions & Baseline
# ━━ LOB DECOMPOSITION: GROUP CONTRIBUTIONS & BASELINE 2025 ━━━━━━━━━━━━━━━━━━━━━━
from collections import defaultdict

scale_factor = float(scaler_y.scale_[0])

# Build baseline_2025 from df_full (avg weekly spend per combo in 2025)
_funnels_all = ["Awareness", "Consideration", "Conversion"]
_media_cols_raw = [c for c in df_full.columns if any(c.startswith(f"{f}_") for f in _funnels_all)
                   and "__" not in c and "media_" not in c]

def _combo_cols(combo):
    funnel, group = combo.split("__", 1)
    lobs = LOB_GROUP_MAP.get(group, [])
    return [c for c in _media_cols_raw if c.startswith(f"{funnel}_")
            and any(lob in c for lob in lobs)]

baseline_2025 = {}
_df_2025 = df_full[df_full["year"] == 2025]
for combo in ACTIVE_COMBOS:
    cols = _combo_cols(combo)
    weekly_spend = _df_2025[cols].sum(axis=1)
    annual_spend = float(weekly_spend.sum())
    weeks_active = int((weekly_spend > 0).sum())
    weekly_avg   = annual_spend / 52
    baseline_2025[combo] = {"annual_spend": annual_spend, "weekly_avg": weekly_avg, "weeks_active": weeks_active}

# Validated group contributions at 2025 steady-state spend
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

# ── Jensen's correction: scale steady-state to actual model decomposition ────
# The response curve at uniform spend (annual/52) overstates contributions because
# actual spend is variable and the curve is concave (diminishing returns).
# Scale per-funnel so that group totals match the actual week-by-week output in df_full.
_actual_funnel_contrib = {
    "Awareness":     float(_df_2025["media_Awareness"].sum()),
    "Consideration": float(_df_2025["media_Consideration"].sum()),
    "Conversion":    float(_df_2025["media_Conversion"].sum()),
}
print(f"\nJensen's correction (steady-state → actual week-by-week):")
for funnel in _funnels_all:
    ss_funnel = sum(active_contributions[c] for c in ACTIVE_COMBOS if c.startswith(funnel))
    actual_f  = _actual_funnel_contrib[funnel]
    if ss_funnel > 0:
        ratio = actual_f / ss_funnel
        for c in ACTIVE_COMBOS:
            if c.startswith(funnel):
                active_contributions[c] *= ratio
        print(f"  {funnel:<15} SS={ss_funnel:>12,.0f} → Actual={actual_f:>12,.0f}  (×{ratio:.3f})")

total_validated = sum(active_contributions.values())

# Group-average efficiency (att-per-$) for extrapolating dropped LOBs
group_avg_rate = {}
for combo in ACTIVE_COMBOS:
    sp = baseline_2025[combo]["annual_spend"]
    group_avg_rate[combo.split("__")[1]] = active_contributions[combo] / sp if sp > 0 else 0

funnel_median_rate = {}
for funnel in _funnels_all:
    rates = [active_contributions[c] / baseline_2025[c]["annual_spend"]
             for c in ACTIVE_COMBOS if c.startswith(funnel) and baseline_2025[c]["annual_spend"] > 0]
    funnel_median_rate[funnel] = float(np.median(rates)) if rates else 0.0

print(f"Validated total media contribution (2025): {total_validated:,.0f}")
for c in ACTIVE_COMBOS:
    print(f"  {c:<40} {active_contributions[c]:>12,.0f}")
print(f"\nFunnel-average att-per-$ (for extrapolating dropped LOBs):")
for f, r in funnel_median_rate.items():
    print(f"  {f:<15} {r:.2f}")

# COMMAND ----------

# DBTITLE 1,EXH LOB Tilt Signal & Tilted Decomposition
# ━━ RESIDUAL ATTENDANCE & TILT SIGNAL ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

control_pred = df_full["controls_pred"].values
residual_att = df_full[target_col].values - control_pred
print(f"Residual attendance: mean {residual_att.mean():,.0f} | std {residual_att.std():,.0f}")

# Parse LOB spend & compute per-LOB tilt signal
funnel_lob_cols = defaultdict(list)
for col in _media_cols_raw:
    parts = col.split("_", 1); funnel = parts[0]; rest = parts[1]
    funnel_lob_cols[(funnel, rest[:rest.rfind("_")])].append(col)

lob_spend_full, lob_tilt_corr, lob_spend_2025 = {}, {}, {}
_df_2025_dc = df_full[df_full["year"] == 2025]          # 2025 slice for decomp anchoring
for (funnel, lob), cols in funnel_lob_cols.items():
    weekly = df_full[cols].sum(axis=1).values.astype(float)
    lob_spend_full[(funnel, lob)] = float(df_full[cols].sum().sum())
    lob_spend_2025[(funnel, lob)] = float(_df_2025_dc[cols].sum().sum())
    active = weekly > 0
    if active.sum() >= 5 and np.std(weekly[active]) > 0 and np.std(residual_att[active]) > 0:
        lob_tilt_corr[(funnel, lob)] = float(np.corrcoef(weekly[active], residual_att[active])[0, 1])
    else:
        lob_tilt_corr[(funnel, lob)] = 0.0

print(f"\nParsed {len(funnel_lob_cols)} (funnel, LOB) cells")
print("\nTop residual-attendance correlations (tilt signal):")
for k in sorted(lob_tilt_corr, key=lambda x: -abs(lob_tilt_corr[x]))[:10]:
    print(f"  {k[0]:<14} {k[1]:<34} r={lob_tilt_corr[k]:+.2f}  ${lob_spend_full[k]:>10,.0f}")

# ━━ TILTED DECOMPOSITION ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TILT_CAP = 0.35

def lob_display_name(raw):
    return {"EXH__Brand":"EXH - Brand","FandB":"F&B","GandE":"G&E",
            "Film_Marketing__Inform":"Film Marketing - Inform",
            "Performance__Convert_and_Upgrade":"Performance - Convert & Upgrade",
            "Premium_Experiences":"Premium Experiences - AVX",
            "Premium_Experiences__AVX":"Premium Experiences - AVX",
            "Premium_Experiences__IMAX":"Premium Experiences - IMAX",
            "Premium_Experiences__VIP":"Premium Experiences - VIP",
            "Prepaid__B2B":"Prepaid - B2B","Prepaid__B2C":"Prepaid - B2C",
            "Cineclub":"Cineclub"}.get(raw, raw)

LOB_TREAT = {"Film Marketing - Inform": "borrow", "Performance - Convert & Upgrade": "borrow"}

def get_parent_combo(funnel, lob_raw):
    for group_name, lob_list in LOB_GROUP_MAP.items():
        if lob_raw in lob_list:
            combo = f"{funnel}__{group_name}"
            if combo in ACTIVE_COMBOS:
                return combo, group_name, "validated"
            return None, group_name, "dropped"
    return None, None, "no_group"

combo_members = defaultdict(list)
extrapolate_cells = []
for (funnel, lob_raw), spend in lob_spend_full.items():
    if spend <= 0: continue
    # Use 2025-only spend so graft shares and Att per $1 are consistent with
    # cell 35's Jensen's correction (which also anchors to 2025 df_full values).
    s25 = lob_spend_2025.get((funnel, lob_raw), 0.0)
    combo, group, routing = get_parent_combo(funnel, lob_display_name(lob_raw))
    if routing == "validated":
        combo_members[combo].append((funnel, lob_raw, s25))
    else:
        extrapolate_cells.append((funnel, lob_raw, s25))

rows_decomp = []
for combo, members in combo_members.items():
    group_total_spend  = sum(s for _, _, s in members)
    group_contribution = active_contributions[combo]
    base = {(f, l): (s / group_total_spend if group_total_spend > 0 else 1/len(members))
            for f, l, s in members}
    tilt_factor = {}
    for f, l, s in members:
        disp = lob_display_name(l)
        treat = LOB_TREAT.get(disp, "curve")
        if treat == "curve":
            r = lob_tilt_corr.get((f, l), 0.0)
            tilt_factor[(f, l)] = 1.0 + TILT_CAP * max(-1.0, min(1.0, r))
        else:
            tilt_factor[(f, l)] = 1.0
    raw_w = {k: base[k] * tilt_factor[k] for k in base}
    wsum  = sum(raw_w.values())
    shares = {k: raw_w[k] / wsum for k in raw_w}
    for f, l, s in members:
        disp = lob_display_name(l)
        treat = LOB_TREAT.get(disp, "curve")
        share = shares[(f, l)]
        contrib = group_contribution * share
        rows_decomp.append({
            "Funnel": f, "LOB": disp, "Parent Combo": combo,
            "Status": "Decomposed (validated)", "Treatment": treat,
            "Reallocatable": "YES" if treat == "curve" else "directional",
            "Annual Spend": round(s, 0),
            "Spend Share (grp)": round(base[(f, l)] * 100, 1),
            "Tilt r": round(lob_tilt_corr.get((f, l), 0.0), 2) if treat == "curve" else None,
            "Final Share (grp)": round(share * 100, 1),
            "Annual Contrib": round(contrib, 0),
            "Att per $1": round(contrib / s, 2) if s > 0 else None,
        })

# Extrapolated (dropped) LOBs
for f, l, s in extrapolate_cells:
    disp = lob_display_name(l)
    _, grp, _ = get_parent_combo(f, disp)
    if grp is not None and grp in group_avg_rate and group_avg_rate[grp] > 0:
        rate = group_avg_rate[grp]
    else:
        rate = funnel_median_rate.get(f, 0.0)
    contrib = s * rate
    rows_decomp.append({
        "Funnel": f, "LOB": disp, "Parent Combo": f"{f}__Other",
        "Status": "Extrapolated ⚠ (not validated)", "Treatment": "extrapolated",
        "Reallocatable": "NO — geo-test",
        "Annual Spend": round(s, 0), "Spend Share (grp)": None,
        "Tilt r": None, "Final Share (grp)": None,
        "Annual Contrib": round(contrib, 0),
        "Att per $1": round(rate, 2),
    })

decomp = pd.DataFrame(rows_decomp)
validated_contrib = decomp.loc[decomp["Status"].str.startswith("Decomposed"), "Annual Contrib"].sum()
extrapolated_contrib = decomp.loc[decomp["Status"].str.startswith("Extrapolated"), "Annual Contrib"].sum()

print(f"\nValidated decomposed total : {validated_contrib:,.0f}  (should match {total_validated:,.0f})")
print(f"Extrapolated (dropped) add : {extrapolated_contrib:,.0f}")
print(f"Planning total (val+extrap): {validated_contrib + extrapolated_contrib:,.0f}")
print(f"\nValidated sum-back diff: {validated_contrib - total_validated:,.0f}  (≈0 expected)")

# Save decomp artifact
reallocatable_lobs = set(decomp.loc[decomp["Reallocatable"] == "YES", "LOB"].unique())
decomp_artifact = {"decomp_table": decomp, "reallocatable_lobs": list(reallocatable_lobs)}
with open(f"{EXH_DECOMP_PATH}lob_decomposition.pkl", "wb") as f:
    pickle.dump(decomp_artifact, f)
with open(f"{EXH_DECOMP_PATH}decomp_lob_results.pkl", "wb") as f:
    pickle.dump(decomp_artifact, f)
print(f"\n✓ LOB decomposition saved → {EXH_DECOMP_PATH}")

# Display summary
print(f"\n{'═'*70}\n  LOB Decomposition Summary\n{'═'*70}")
display(decomp[["Funnel","LOB","Status","Annual Spend","Final Share (grp)","Annual Contrib","Att per $1"]]
        .sort_values("Annual Contrib", ascending=False))

# COMMAND ----------

# DBTITLE 1,Per-LOB Annual Spend & Attendance Contribution
# ━━ Per-LOB Annual Spend & Attendance Contribution — grouped bars by year ━━━━━━━━━━
# One chart per LOB. For each year: stacked spend bar + attendance contribution bar side by side.
# Unvalidated/geo-test LOBs included with funnel-avg att-per-$1 as directional proxy.

import matplotlib.ticker as mticker
import matplotlib.patches as mpatches

FUNNEL_COLOR = {"Awareness": CPX_NAVY, "Consideration": CPX_RED, "Conversion": CPX_GOLD}
FUNNELS      = ["Awareness", "Consideration", "Conversion"]
OUT_PATH     = EXH_DECOMP_PATH

# ── 1. Validated att-per-$1 from decomp ─────────────────────────────────────────
decomp_valid = decomp[decomp["Status"].str.startswith("Decomposed")].copy()

fl_rate = {}
for (funnel, lob), grp in decomp_valid.groupby(["Funnel", "LOB"]):
    valid = grp[grp["Att per $1"].notna() & (grp["Annual Spend"] > 0)]
    if valid.empty:
        fl_rate[(funnel, lob)] = 0.0
    else:
        fl_rate[(funnel, lob)] = (
            (valid["Att per $1"] * valid["Annual Spend"]).sum() / valid["Annual Spend"].sum()
        )

# ── 2. Year-specific funnel att/$ from actual model decomposition ────────────
years_avail = sorted(y for y in df_full["year"].dropna().unique().astype(int) if y < 2026)

_spend_cols_all_37 = [c for c in df_full.columns if any(c.startswith(f"{f}_") for f in FUNNELS)
                      and "__" not in c and "media_" not in c]

# ── Pre-compute annual aggregates once ────────────────────────────────────────
# Replaces O(n) df_full scans inside every loop with O(1) index lookups.
_df_grp_yr       = df_full[df_full["year"].isin(years_avail)].groupby("year", observed=True)
_spend_by_yr     = _df_grp_yr[_spend_cols_all_37].sum()          # index = year
_media_by_yr     = _df_grp_yr[[f"media_{f}" for f in FUNNELS]].sum()
_actual_att_by_yr = _df_grp_yr[target_col].sum()                 # total attendance per year

funnel_yr_rate = {}  # funnel → {year: att/$}
funnel_rate_2025 = {}
for funnel in FUNNELS:
    funnel_yr_rate[funnel] = {}
    f_spend_cols = [c for c in _spend_cols_all_37 if c.startswith(funnel + "_")]
    for yr in years_avail:
        spend  = float(_spend_by_yr.loc[yr, f_spend_cols].sum())
        contrib = float(_media_by_yr.loc[yr, f"media_{funnel}"])
        funnel_yr_rate[funnel][yr] = contrib / spend if spend > 0 else 0.0
    funnel_rate_2025[funnel] = funnel_yr_rate[funnel].get(2025, 1.0)

# Fallback: funnel-average rate (static, for LOBs with no validated decomp)
funnel_avg_rate_chart = {}
for funnel in FUNNELS:
    rates = [v for (f, _), v in fl_rate.items() if f == funnel and v > 0]
    funnel_avg_rate_chart[funnel] = float(np.mean(rates)) if rates else 0.0

directional_lobs = set()
for (funnel, lob_raw), cols in funnel_lob_cols.items():
    disp = lob_display_name(lob_raw)
    key  = (funnel, disp)
    if key not in fl_rate:
        has_spend = any(
            float(_spend_by_yr.loc[yr, cols].sum()) > 0
            for yr in years_avail
        )
        if has_spend:
            fl_rate[key] = funnel_avg_rate_chart.get(funnel, 0.0)
            directional_lobs.add(key)

# ── 3. Annual spend per (funnel, LOB, year) — all LOBs ───────────────────────
fl_yr_spend = {}
for (funnel, lob_raw), cols in funnel_lob_cols.items():
    disp = lob_display_name(lob_raw)
    key  = (funnel, disp)
    if key not in fl_rate:
        continue
    if key not in fl_yr_spend:
        fl_yr_spend[key] = {yr: 0.0 for yr in years_avail}
    for yr in years_avail:
        fl_yr_spend[key][yr] += float(_spend_by_yr.loc[yr, cols].sum())

# ── 4. Unique LOBs ───────────────────────────────────────────────────────────
all_lobs = sorted({lob for (_, lob) in fl_rate})

# ── 4b. Full reconciliation: anchor ALL LOBs to df_full funnel totals ─────────────
# For each (funnel, year) both validated AND directional LOBs are allocated
# proportionally so their combined total = df_full["media_f"].sum() for that
# year — reconciling exactly with cell 31 across all years.
# Weight per LOB = spend × rate (validated: LOB-specific fl_rate;
#                                directional: funnel-avg yr rate)
_actual_funnel_yr = {
    yr: {f: float(_media_by_yr.loc[yr, f"media_{f}"]) for f in FUNNELS}
    for yr in years_avail
}
_wgt_sum_yr = {yr: {f: 0.0 for f in FUNNELS} for yr in years_avail}
for (f, lob) in fl_rate:
    for yr in years_avail:
        spend_yr = fl_yr_spend.get((f, lob), {}).get(yr, 0.0)
        if spend_yr <= 0:
            continue
        if (f, lob) in directional_lobs:
            r = funnel_yr_rate[f].get(yr, 0.0)
        else:
            lob_r = fl_rate[(f, lob)]
            # LOBs inactive in 2025 have fl_rate=0 (no validated 2025 rate).
            # Fall back to funnel average so their historical spend still carries weight
            # in the proportional allocation — otherwise their spend is silently zeroed out.
            r = lob_r if lob_r > 0 else funnel_avg_rate_chart.get(f, 0.0)
        _wgt_sum_yr[yr][f] += spend_yr * r
_alloc_factor_yr = {
    (f, yr): (_actual_funnel_yr[yr][f] / _wgt_sum_yr[yr][f]
              if _wgt_sum_yr[yr][f] > 0 else 1.0)
    for yr in years_avail for f in FUNNELS
}
print("Reconciliation allocation factors (≈1.0 → LOB rates well-calibrated):")
for yr in years_avail:
    row = "  ".join(f"{f[:3]} {yr}: ×{_alloc_factor_yr[(f, yr)]:.4f}" for f in FUNNELS)
    print(f"  {row}")
# Verify: print implied funnel totals vs cell 31 actuals
print("\nReconciliation check — sum of all LOB contrib vs cell 31 df_full totals:")
for yr in years_avail:
    for f in FUNNELS:
        implied = _wgt_sum_yr[yr][f] * _alloc_factor_yr[(f, yr)]
        actual  = _actual_funnel_yr[yr][f]
        print(f"  {f:<15} {yr}: implied={implied:>12,.0f}  actual={actual:>12,.0f}  "
              f"diff={implied-actual:+.0f}")

# ── Module-level reconciled rate — callable by all downstream cells ────────────────
def _lob_rate_yr(funnel, lob, yr):
    """Att/$ for (funnel, lob, yr), anchored to df_full[\'media_f\'] for every year.
    Validated LOBs: fl_rate (or funnel-avg fallback for LOBs inactive in 2025).
    Directional LOBs: funnel_yr_rate.
    Both multiplied by _alloc_factor_yr so all LOBs sum exactly to cell 31 actuals."""
    if (funnel, lob) in directional_lobs:
        base = funnel_yr_rate[funnel].get(yr, 0.0)
    else:
        lob_r = fl_rate.get((funnel, lob), 0.0)
        base = lob_r if lob_r > 0 else funnel_avg_rate_chart.get(funnel, 0.0)
    return base * _alloc_factor_yr.get((funnel, yr), 1.0)

# ── 5. One chart per LOB ─────────────────────────────────────────────────────
n_yrs = len(years_avail)
bar_w = 0.35
gap   = 0.15
group = bar_w * 2 + gap + 0.2

for lob in all_lobs:
    active_funnels = [f for f in FUNNELS if (f, lob) in fl_rate and
                      any(fl_yr_spend.get((f, lob), {}).get(yr, 0) > 0 for yr in years_avail)]
    if not active_funnels:
        continue

    is_directional = any((f, lob) in directional_lobs for f in active_funnels)

    spend_by_f = {f: np.array([fl_yr_spend.get((f, lob), {}).get(yr, 0) / 1e3
                               for yr in years_avail]) for f in active_funnels}

    contrib_by_f = {f: np.array([fl_yr_spend.get((f, lob), {}).get(yr, 0) * _lob_rate_yr(f, lob, yr) / 1e6
                                 for yr in years_avail]) for f in active_funnels}

    total_spend_yr   = sum(spend_by_f[f]  for f in active_funnels)
    total_contrib_yr = sum(contrib_by_f[f] for f in active_funnels)

    x_spend   = np.array([i * group for i in range(n_yrs)])
    x_contrib = x_spend + bar_w + gap

    fig, ax_s = plt.subplots(figsize=(11, 9))
    ax_c = ax_s.twinx()

    # Stacked spend bars (left axis)
    bottoms_s = np.zeros(n_yrs)
    for f in active_funnels:
        ax_s.bar(x_spend, spend_by_f[f], bar_w, bottom=bottoms_s,
                 color=FUNNEL_COLOR[f], alpha=0.9, edgecolor="white", lw=0.4)
        bottoms_s += spend_by_f[f]

    # Stacked contribution bars (right axis) — hatched
    bottoms_c = np.zeros(n_yrs)
    for f in active_funnels:
        ax_c.bar(x_contrib, contrib_by_f[f], bar_w, bottom=bottoms_c,
                 color=FUNNEL_COLOR[f],
                 alpha=0.35 if is_directional else 0.55,
                 edgecolor="white", lw=0.4, hatch="///")
        bottoms_c += contrib_by_f[f]

    # Value labels on top of bars
    for i in range(n_yrs):
        if bottoms_s[i] > 0:
            ax_s.annotate(f"${bottoms_s[i]:.0f}K",
                          xy=(x_spend[i], bottoms_s[i]),
                          xytext=(0, 4), textcoords="offset points",
                          ha="center", fontsize=7.5, color=CPX_DGREY, fontweight="bold")
        if bottoms_c[i] > 0:
            ax_c.annotate(f"{bottoms_c[i]:.2f}M",
                          xy=(x_contrib[i], bottoms_c[i]),
                          xytext=(0, 4), textcoords="offset points",
                          ha="center", fontsize=7.5, color=CPX_DGREY, fontweight="bold")

    # X-axis
    x_mid = (x_spend + x_contrib) / 2
    ax_s.set_xticks(x_mid)
    ax_s.set_xticklabels([str(y) for y in years_avail], fontsize=11, fontweight="bold", color=CPX_DGREY)
    ax_s.set_xlim(-0.3, x_contrib[-1] + bar_w + 0.08)

    # Axis formatting
    ax_s.set_ylabel("Annual Media Spend ($K)", fontsize=12, fontweight="bold", color=CPX_DGREY, labelpad=8)
    ax_s.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:.0f}K"))
    ax_s.set_ylim(bottom=0, top=max(bottoms_s.max(), 0.01) * 1.3)

    ax_c.set_ylabel("Media Attendance Contribution (M)", fontsize=12, fontweight="bold", color=CPX_DGREY, labelpad=8)
    ax_c.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.2f}M"))
    ax_c.set_ylim(bottom=0, top=max(bottoms_c.max(), 0.01) * 1.3)

    for ax in (ax_s, ax_c):
        ax.spines["top"].set_visible(False)
        ax.tick_params(axis="y", labelsize=10)
    # Right axis: force spine visible, ticks outward (default twinx can point inward)
    ax_c.spines["right"].set_visible(True)
    ax_c.spines["right"].set_linewidth(0.8)
    ax_c.tick_params(axis="y", direction="out", length=5, width=0.8)
    plt.setp(ax_s.get_yticklabels(), fontweight="bold")
    plt.setp(ax_c.get_yticklabels(), fontweight="bold")

    ax_s.set_axisbelow(True)
    ax_s.grid(axis="y", alpha=0.2, color=CPX_LGREY)

    # Legend
    handles = [mpatches.Patch(color=FUNNEL_COLOR[f], label=f) for f in active_funnels]
    handles += [
        mpatches.Patch(facecolor="#999999", alpha=0.9, label="Spend"),
        mpatches.Patch(facecolor="#999999", alpha=0.55, hatch="///", label="Attendance Contrib"),
    ]
    ax_s.legend(handles=handles, fontsize=8,
                loc="upper center",
                ncol=len(handles),
                borderaxespad=0.5, framealpha=0.9, edgecolor=CPX_LGREY)

    ax_s.set_title(
        f"{lob}\nAnnual Spend vs Attendance Contribution",
        fontsize=14, fontweight="bold",
        color="black",
        pad=18
    )

    plt.tight_layout(rect=[0, 0, 0.88, 1])
    save_fig(fig, f"I2_B10_LOB_Spend_{lob.replace(' ','_').replace('-','')}.png")
    display(fig)
    plt.close(fig)

    # ── Small summary table for this LOB (pivoted: funnel × metric as rows, years as columns + Δ YoY) ──
    def _yoy_pct(prev, curr):
        """Compute YoY % change string."""
        if curr is None and prev is None:
            return "–"
        if curr is None:
            return "–"
        if prev is None or prev == 0:
            if curr != 0:
                return "New"
            return "–"
        if curr == 0:
            return "▼ Removed"
        pct = (curr - prev) / abs(prev) * 100
        if pct <= -100:
            return "▼ Removed"
        elif pct < 0:
            return f"▼ {abs(pct):.0f}%"
        elif pct == 0:
            return "▶ 0%"
        else:
            return f"▲ {pct:.0f}%"

    tbl_rows = []
    for f in active_funnels:
        is_dir = (f, lob) in directional_lobs
        status_icon = " ⚠" if is_dir else " ✓"
        funnel_label = f"{f}{status_icon}"
        # Spend row
        spend_row = {"Funnel": funnel_label, "Metric": "Spend ($K)"}
        spend_vals = []
        for yr_idx, yr in enumerate(years_avail):
            val = round(spend_by_f[f][yr_idx], 1)
            spend_row[str(yr)] = val
            spend_vals.append(val)
            if yr_idx > 0:
                spend_row[f"Δ YoY.{yr_idx}"] = _yoy_pct(spend_vals[yr_idx - 1], val)
        tbl_rows.append(spend_row)
        # Contribution row
        contrib_row = {"Funnel": "", "Metric": "Contribution (M)"}
        contrib_vals = []
        for yr_idx, yr in enumerate(years_avail):
            val = round(contrib_by_f[f][yr_idx], 3)
            contrib_row[str(yr)] = val
            contrib_vals.append(val)
            if yr_idx > 0:
                contrib_row[f"Δ YoY.{yr_idx}"] = _yoy_pct(contrib_vals[yr_idx - 1], val)
        tbl_rows.append(contrib_row)
        # Att/$ row
        att_row = {"Funnel": "", "Metric": "Att/$"}
        att_vals = []
        for yr_idx, yr in enumerate(years_avail):
            sp = fl_yr_spend.get((f, lob), {}).get(yr, 0)
            tag = "†" if is_dir else ""
            if sp > 0:
                rate_val = _lob_rate_yr(f, lob, yr)
                att_row[str(yr)] = f"{rate_val:.2f}{tag}"
                att_vals.append(rate_val)
            else:
                att_row[str(yr)] = "—"
                att_vals.append(None)
            if yr_idx > 0:
                att_row[f"Δ YoY.{yr_idx}"] = _yoy_pct(att_vals[yr_idx - 1], att_vals[yr_idx])
        tbl_rows.append(att_row)
    # Total rows
    total_spend_row = {"Funnel": "Total", "Metric": "Spend ($K)"}
    total_contrib_row = {"Funnel": "", "Metric": "Contribution (M)"}
    ts_vals, tc_vals = [], []
    for yr_idx, yr in enumerate(years_avail):
        s_val = round(total_spend_yr[yr_idx], 1)
        c_val = round(total_contrib_yr[yr_idx], 3)
        total_spend_row[str(yr)] = s_val
        total_contrib_row[str(yr)] = c_val
        ts_vals.append(s_val)
        tc_vals.append(c_val)
        if yr_idx > 0:
            total_spend_row[f"Δ YoY.{yr_idx}"] = _yoy_pct(ts_vals[yr_idx - 1], s_val)
            total_contrib_row[f"Δ YoY.{yr_idx}"] = _yoy_pct(tc_vals[yr_idx - 1], c_val)
    tbl_rows.append(total_spend_row)
    tbl_rows.append(total_contrib_row)

    # Build column order: Funnel, Metric, 2023, 2024, Δ YoY, 2025, Δ YoY ...
    col_order = ["Funnel", "Metric"]
    for yr_idx, yr in enumerate(years_avail):
        col_order.append(str(yr))
        if yr_idx > 0:
            col_order.append(f"Δ YoY.{yr_idx}")
    _tbl = pd.DataFrame(tbl_rows)[col_order].astype(str)
    _tbl.columns = [c if not c.startswith("Δ YoY") else "Δ YoY" for c in _tbl.columns]
    display(_tbl)

# COMMAND ----------

# DBTITLE 1,Section 12 — Budget Scenarios
# MAGIC %md
# MAGIC ---
# MAGIC ## Section 12: Budget Scenarios
# MAGIC S1 (proportional uplift +10/20/30%), S2 (reshuffle held constant), S3 (combined). Posterior CI engine provides 89% credible intervals via paired differencing.

# COMMAND ----------

# DBTITLE 1,EXH Budget Scenario Functions & Posterior CI
# ━━ BUDGET SCENARIO FUNCTIONS & POSTERIOR CI ENGINE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━
import arviz as az

def combo_contribution(combo, annual_spend, params=None):
    p  = params or response_params[combo]
    mx = spend_train_max_iter2[combo]
    s  = (annual_spend / 52) / mx
    ad = s / (1 - p["alpha"])
    if p["saturation"] == "hill":
        sat = (ad ** p["alpha_s"]) / (ad ** p["alpha_s"] + p["k"] ** p["alpha_s"])
    else:
        sat = 1 - np.exp(-p["lam"] * ad)
    return p["gamma"] * sat * scale_factor * 52

def marginal_return(combo, annual_spend, delta=5200.0):
    up = combo_contribution(combo, annual_spend + delta)
    dn = combo_contribution(combo, max(0, annual_spend - delta))
    return (up - dn) / (2 * delta)

# Baseline spend from 2025
baseline_spend = {c: baseline_2025[c]["annual_spend"] for c in ACTIVE_COMBOS}
total_baseline_spend = sum(baseline_spend.values())

# Steady-state baseline (used ONLY for computing response-curve deltas)
_SS_BASELINE = sum(combo_contribution(c, baseline_spend[c]) for c in ACTIVE_COMBOS)

# Actual baseline: model's week-by-week decomposition (anchors scenarios to reality)
_df_2025_base = df_full[df_full["year"] == 2025]
baseline_attendance = float(
    _df_2025_base["media_Awareness"].sum()
    + _df_2025_base["media_Consideration"].sum()
    + _df_2025_base["media_Conversion"].sum()
)

# Map combo → has reallocatable LOBs
combo_has_realloc = {}
for combo in ACTIVE_COMBOS:
    members = decomp[(decomp["Parent Combo"] == combo)]
    combo_has_realloc[combo] = bool((members["Reallocatable"] == "YES").any())

print(f"Baseline annual media spend     : ${total_baseline_spend:,.0f}")
print(f"Baseline annual media attendance : {baseline_attendance:,.0f}  (actual model decomposition)")
print(f"  (Steady-state reference: {_SS_BASELINE:,.0f} — used only for delta computation)")
print("\nPer-group marginal return at baseline:")
for c in sorted(ACTIVE_COMBOS, key=lambda x: -marginal_return(x, baseline_spend[x])):
    print(f"  {c:<40} {marginal_return(c, baseline_spend[c]):.2f}  "
          f"{'(can receive)' if combo_has_realloc[c] else '(siphon-only)'}")

# ── Posterior CI Engine ─────────────────────────────────────────────────────────
POST_SCEN = trace_iter2_mixed_hier.posterior

def _flat(name, combo):
    key = f"{name}_{combo}"
    if key in POST_SCEN:
        return POST_SCEN[key].values.ravel()
    return None

N_DRAWS = 800
_probe = None
for c in ACTIVE_COMBOS:
    for nm in ["gamma","lam","alpha"]:
        arr = _flat(nm, c)
        if arr is not None: _probe = arr; break
    if _probe is not None: break
_pool = _probe.shape[0]
DRAW_IDX = np.random.choice(_pool, size=min(N_DRAWS, _pool), replace=False)

_PARAM_CACHE = {}
for c in ACTIVE_COMBOS:
    p = response_params[c]
    def pick(nm, dflt):
        arr = _flat(nm, c)
        return arr[DRAW_IDX] if arr is not None else np.full(len(DRAW_IDX), dflt)
    alpha_arr = pick("alpha", p["alpha"])
    alpha_arr = np.minimum(alpha_arr, 0.999)
    entry = {"alpha": alpha_arr, "gamma": pick("gamma", p["gamma"]),
             "sat": p["saturation"], "mx": spend_train_max_iter2[c]}
    if p["saturation"] == "hill":
        entry["alpha_s"] = pick("alpha_s", p["alpha_s"])
        entry["k"] = pick("k", p["k"])
    else:
        entry["lam"] = pick("lam", p["lam"])
    _PARAM_CACHE[c] = entry

def combo_contribution_draws(combo, annual_spend):
    e = _PARAM_CACHE[combo]
    s  = (annual_spend / 52) / e["mx"]
    ad = s / (1 - e["alpha"])
    if e["sat"] == "hill":
        sat = (ad ** e["alpha_s"]) / (ad ** e["alpha_s"] + e["k"] ** e["alpha_s"])
    else:
        sat = 1 - np.exp(-e["lam"] * ad)
    return e["gamma"] * sat * scale_factor * 52

def total_attendance_draws(spend_alloc):
    tot = np.zeros(len(DRAW_IDX))
    for c, sp in spend_alloc.items():
        tot += combo_contribution_draws(c, sp)
    return tot

BASELINE_DRAWS = total_attendance_draws(baseline_spend)
print(f"\nPosterior ready: {len(DRAW_IDX)} draws cached for {len(ACTIVE_COMBOS)} combos.")
print(f"Baseline attendance: actual {baseline_attendance:,.0f} | SS {_SS_BASELINE:,.0f} | draw mean {BASELINE_DRAWS.mean():,.0f}")
print(f"  Jensen's correction factor: {baseline_attendance/_SS_BASELINE:.3f}×")

# COMMAND ----------

# DBTITLE 1,EXH Budget Scenarios S1/S2/S3
# ━━ BUDGET SCENARIOS: S1, S2, S3 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def lift_with_ci(scenario_alloc, label, ci=0.89):
    # Scenario in steady-state space
    scen_ss  = sum(combo_contribution(c, scenario_alloc[c]) for c in scenario_alloc)
    # Delta from response curves (valid for marginal analysis)
    delta_ss = scen_ss - _SS_BASELINE
    # Anchor to actual baseline (Jensen's-corrected)
    point_att = baseline_attendance + delta_ss
    lift_pt   = delta_ss
    pct_pt    = delta_ss / baseline_attendance * 100

    # Posterior draws — delta-based CI anchored to actual baseline
    scen_draws = total_attendance_draws(scenario_alloc)
    lift_draws = scen_draws - BASELINE_DRAWS
    pct_draws  = lift_draws / baseline_attendance * 100

    lo_q, hi_q = (1-ci)/2, 1-(1-ci)/2
    pct_lo = float(np.quantile(pct_draws, lo_q))
    pct_hi = float(np.quantile(pct_draws, hi_q))
    width  = pct_hi - pct_lo
    conf = "High" if width < 4 else ("Medium" if width < 8 else "Low")
    crosses_zero = (pct_lo <= 0 <= pct_hi)
    return {"label":label, "spend":sum(scenario_alloc.values()),
            "attendance":point_att, "lift":lift_pt, "lift_pct":pct_pt,
            "pct_lo":pct_lo, "pct_hi":pct_hi, "ci_width":width,
            "confidence":conf, "crosses_zero":crosses_zero}

def reshuffle(base_alloc, siphon_cap=0.10):
    alloc = dict(base_alloc)
    mr = {c: marginal_return(c, alloc[c]) for c in ACTIVE_COMBOS}
    med_mr = np.median(list(mr.values()))
    pool = 0.0
    for c in ACTIVE_COMBOS:
        if mr[c] < med_mr:
            take = alloc[c] * siphon_cap
            alloc[c] -= take; pool += take
    receivers = sorted([c for c in ACTIVE_COMBOS if combo_has_realloc[c] and mr[c] >= med_mr],
                       key=lambda x: -mr[x])
    if not receivers:
        receivers = sorted([c for c in ACTIVE_COMBOS if combo_has_realloc[c]], key=lambda x:-mr[x])
    total_mr = sum(mr[c] for c in receivers)
    remaining = pool
    for c in receivers:
        want = pool * (mr[c]/total_mr) if total_mr > 0 else pool/len(receivers)
        cap  = base_alloc[c] * siphon_cap
        give = min(want, cap, remaining)
        alloc[c] += give; remaining -= give
    if remaining > 1e-6:
        siphoned = [c for c in ACTIVE_COMBOS if mr[c] < med_mr]
        for c in siphoned:
            alloc[c] += remaining/len(siphoned)
    return alloc

# ── S1: Budget Uplift ───────────────────────────────────────────────────────────────
s1_results = []
for factor in [1.10, 1.20, 1.30]:
    alloc = {c: baseline_spend[c] * factor for c in ACTIVE_COMBOS}
    r = lift_with_ci(alloc, f"+{round((factor-1)*100)}% budget")
    r["spend_increase_pct"] = (factor-1)*100
    s1_results.append(r)

print("SCENARIO 1 — BUDGET UPLIFT (pattern unchanged)\n")
s1_df = pd.DataFrame([{
    "Scenario": r["label"], "Annual Spend": f"${r['spend']:,.0f}",
    "Attendance": f"{r['attendance']:,.0f}", "Lift %": f"+{r['lift_pct']:.1f}%",
    "89% CI": f"[+{r['pct_lo']:.1f}%, +{r['pct_hi']:.1f}%]", "Confidence": r["confidence"],
} for r in s1_results])
display(s1_df)

# ── S2: Reshuffle ──────────────────────────────────────────────────────────────────
s2_alloc = reshuffle(baseline_spend)
r2 = lift_with_ci(s2_alloc, "Reshuffle (10% cap)")

move_rows = []
for c in ACTIVE_COMBOS:
    d = s2_alloc[c] - baseline_spend[c]
    move_rows.append({"Group":c, "Baseline":f"${baseline_spend[c]:,.0f}",
        "Reshuffled":f"${s2_alloc[c]:,.0f}", "Δ":f"{d:+,.0f}",
        "Δ%":f"{d/baseline_spend[c]*100:+.1f}%",
        "Role":"receiver" if d>1 else ("siphoned" if d<-1 else "—")})
print(f"\nSCENARIO 2 — RESHUFFLE (total spend held constant: ${sum(s2_alloc.values()):,.0f})\n")
display(pd.DataFrame(move_rows))
print(f"Attendance: {r2['attendance']:,.0f}  Lift: +{r2['lift']:,.0f} (+{r2['lift_pct']:.1f}%)")

# ── S3: Combined ───────────────────────────────────────────────────────────────────
s3_results = []
for factor in [1.10, 1.20, 1.30]:
    uplifted = {c: baseline_spend[c]*factor for c in ACTIVE_COMBOS}
    combined = reshuffle(uplifted)
    r = lift_with_ci(combined, f"+{round((factor-1)*100)}% + reshuffle")
    r["spend_increase_pct"] = (factor-1)*100
    s3_results.append(r)

print(f"\nSCENARIO 3 — COMBINED\n")
s3_df = pd.DataFrame([{
    "Scenario": r["label"], "Annual Spend": f"${r['spend']:,.0f}",
    "Attendance": f"{r['attendance']:,.0f}", "Lift %": f"+{r['lift_pct']:.1f}%",
    "89% CI": f"[+{r['pct_lo']:.1f}%, +{r['pct_hi']:.1f}%]", "Confidence": r["confidence"],
} for r in s3_results])
display(s3_df)

# COMMAND ----------

# DBTITLE 1,EXH Scenario Outcome Visuals
# ━━ SCENARIO OUTCOME VISUALS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
import matplotlib.patches as mpatches

fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

# (A) S1: Diminishing returns
ax = axes[0]
sp = [0] + [r["spend_increase_pct"] for r in s1_results]
at = [0] + [r["lift_pct"]           for r in s1_results]
ax.plot(sp, sp, "--", color=CPX_LGREY, lw=1.5, label="1:1 (no saturation)")
ax.plot(sp, at, "-o", color=CPX_NAVY,  lw=2.5, markersize=8, label="Predicted attendance lift")
for r in s1_results:
    ax.errorbar(r["spend_increase_pct"], r["lift_pct"],
                yerr=[[r["lift_pct"]-r["pct_lo"]], [r["pct_hi"]-r["lift_pct"]]],
                color=CPX_NAVY, capsize=5, alpha=0.5)
    ax.annotate(f"+{r['lift_pct']:.1f}%", (r["spend_increase_pct"], r["lift_pct"]),
                textcoords="offset points", xytext=(8, -4),
                fontsize=9, fontweight="bold", color=CPX_NAVY)
ax.set_xlabel("Budget increase (%)", fontsize=13, fontweight="bold"); ax.set_ylabel("Attendance lift (%)", fontsize=13, fontweight="bold")
ax.set_title("S1 — Diminishing Returns on Budget Uplift", fontweight="bold", color=CPX_NAVY)
ax.legend(frameon=False, fontsize=9)
plt.setp(ax.get_yticklabels(), fontsize=11, fontweight="bold")
ax.grid(axis="y", alpha=0.2); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

# (B) All scenarios — bar chart with CI whiskers
ax = axes[1]
all_results = s1_results + [r2] + s3_results
labels  = [r["label"]    for r in all_results]
lifts   = [r["lift_pct"] for r in all_results]
los     = [r["lift_pct"] - r["pct_lo"]  for r in all_results]
his     = [r["pct_hi"]   - r["lift_pct"] for r in all_results]
colors  = [CPX_NAVY]*3 + [CPX_GOLD] + [CPX_GREEN]*3
x       = np.arange(len(labels))
ax.bar(x, lifts, color=colors, alpha=0.88, edgecolor="white")
ax.errorbar(x, lifts, yerr=[los, his], fmt="none", ecolor=CPX_DGREY, capsize=4, alpha=0.6)
for i, r in enumerate(all_results):
    tag = r["confidence"]
    ax.text(i, lifts[i] + his[i] + max(his)*0.12, tag, ha="center", fontsize=8, fontweight="bold", color=CPX_DGREY)
    ax.text(i, lifts[i]/2, f"+{lifts[i]:.1f}%", ha="center", fontsize=8, fontweight="bold", color="white")
ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=10, fontweight="bold")
ax.set_ylabel("Attendance lift (%)", fontsize=13, fontweight="bold")
ax.set_title("All Scenarios — Annual Lift with 89% CI", fontweight="bold", color=CPX_NAVY)
plt.setp(ax.get_yticklabels(), fontsize=11, fontweight="bold")
ax.grid(axis="y", alpha=0.2); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

leg = [mpatches.Patch(facecolor=CPX_NAVY, label="Budget uplift"),
       mpatches.Patch(facecolor=CPX_GOLD, label="Reshuffle"),
       mpatches.Patch(facecolor=CPX_GREEN, label="Combined")]
fig.legend(handles=leg, loc="lower center", ncol=3, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.06))
fig.suptitle("EXH MMM — Budget Scenario Outcomes (2025)", fontsize=13, fontweight="bold", color=CPX_NAVY, y=1.02)
plt.tight_layout()
save_fig(fig, "scenario_outcomes.png")
display(fig); plt.close(fig)

# COMMAND ----------

# DBTITLE 1,Per-Scenario LOB Spend — Baseline vs Scenario
# ── CELL: Per-Scenario LOB Spend — Baseline vs Scenario (graft-split) ─────────
# For each scenario: split group-level spend to LOBs using graft Final Share (grp),
# then plot baseline vs scenario LOB spend side by side, coloured by funnel.

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np
import os

CPX_NAVY  = "#002B5C"
CPX_RED   = "#E31837"
CPX_GOLD  = "#F5A800"
CPX_DGREY = "#4A4A4A"
CPX_LGREY = "#CCCCCC"

FUNNEL_COLORS = {"Awareness": CPX_NAVY, "Consideration": CPX_RED, "Conversion": CPX_GOLD}
OUT_PATH = EXH_ARTIFACT_PATH

# ── Construct funnel_lob_cols from df_full column names ───────────────────────
_CHANNELS = ["Digital", "Social", "SEM"]
_FUNNELS  = ["Awareness", "Consideration", "Conversion"]
funnel_lob_cols = {}
for col in df_full.columns:
    if "__" in col or col.startswith("media_"):
        continue
    for f in _FUNNELS:
        if col.startswith(f + "_"):
            suffix = col[len(f)+1:]  # e.g. "Cineclub_Digital"
            for ch in _CHANNELS:
                if suffix.endswith("_" + ch):
                    lob_raw = suffix[:-(len(ch)+1)]  # e.g. "Cineclub"
                    funnel_lob_cols.setdefault((f, lob_raw), []).append(col)
                    break
            break

def lob_display_name(lob_raw):
    """Identity — raw LOB names already match decomp table display names."""
    return lob_raw

def save_fig(fig, fname):
    fig.savefig(os.path.join(OUT_PATH, fname), dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved: {fname}")

# ── 1. Build LOB share table from graft decomp ───────────────────────────────
decomp_valid = decomp[decomp["Status"].str.startswith("Decomposed")].copy()
decomp_valid["share_frac"] = decomp_valid["Final Share (grp)"] / 100.0

# ── 2. split_to_lobs — validated LOBs only (via graft shares) ────────────────
def split_to_lobs(group_alloc):
    lob_spend = {}
    for _, row in decomp_valid.iterrows():
        combo    = row["Parent Combo"]
        funnel   = row["Funnel"]
        lob      = row["LOB"]
        share    = row["share_frac"]
        group_sp = group_alloc.get(combo, 0.0)
        key      = (funnel, lob)
        lob_spend[key] = lob_spend.get(key, 0.0) + group_sp * share
    return lob_spend

# ── 2b. All-LOB spend — validated via graft + unvalidated held fixed ──────────
years_avail_s = sorted(y for y in df_full["year"].dropna().unique().astype(int) if y < 2026)

def all_lob_spend(group_alloc):
    """
    Returns spend per (funnel, lob) for ALL original LOBs.
    Validated: group_alloc × graft share.
    Unvalidated/geo-test: 2025 actual spend held fixed (scale=1.0).
    """
    # Start with validated split
    lob_spend = split_to_lobs(group_alloc)

    # Add unvalidated LOBs at fixed 2025 spend
    for (funnel, lob_raw), cols in funnel_lob_cols.items():
        disp  = lob_display_name(lob_raw)
        key   = (funnel, disp)
        if key in lob_spend:
            continue  # already covered by graft
        sp_2025 = float(df_full[df_full["year"] == 2025][cols].sum().sum())
        if sp_2025 > 0:
            lob_spend[key] = sp_2025  # held fixed

    return lob_spend

baseline_lob_spend = all_lob_spend(baseline_spend)

# ── 3. Scenarios ──────────────────────────────────────────────────────────────
scenarios = (
    [(r["label"], {c: baseline_spend[c] * (1 + r["spend_increase_pct"]/100)
                   for c in ACTIVE_COMBOS})
     for r in s1_results]
    + [("Reshuffle (10% cap)", s2_alloc)]
    + [(r["label"], {c: reshuffle({c2: baseline_spend[c2] * (1 + r["spend_increase_pct"]/100)
                                   for c2 in ACTIVE_COMBOS})[c]
                    for c in ACTIVE_COMBOS})
       for r in s3_results]
)

# ── 4. LOB ordering — by baseline spend descending ───────────────────────────
# Flag which LOBs are unvalidated (held fixed)
validated_keys = {(row["Funnel"], row["LOB"]) for _, row in decomp_valid.iterrows()}

lob_keys = sorted(baseline_lob_spend.keys(),
                  key=lambda k: -baseline_lob_spend[k])
n_lobs   = len(lob_keys)
x        = np.arange(n_lobs)
bar_w    = 0.35

# ── 5. One chart per scenario ─────────────────────────────────────────────────
for scen_label, group_alloc in scenarios:
    scen_lob_spend = all_lob_spend(group_alloc)

    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.2, color=CPX_LGREY)

    for i, (funnel, lob) in enumerate(lob_keys):
        base_val = baseline_lob_spend.get((funnel, lob), 0.0) / 1e6
        scen_val = scen_lob_spend.get((funnel, lob), 0.0)     / 1e6
        color    = FUNNEL_COLORS[funnel]

        # Baseline bar — full opacity
        ax.bar(x[i] - bar_w / 2, base_val, bar_w * 0.88,
               color=color, alpha=0.9, edgecolor="white", lw=0.4)

        # Scenario bar — hatched to distinguish
        ax.bar(x[i] + bar_w / 2, scen_val, bar_w * 0.88,
               color=color, alpha=0.5, edgecolor="white", lw=0.4, hatch="///")

        # Delta label above the taller bar
        delta = scen_val - base_val
        top   = max(base_val, scen_val)
        is_validated = (funnel, lob) in validated_keys
        if abs(delta) > 0.005:
            ax.annotate(f"{delta:+.2f}M",
                        xy=(x[i] + bar_w / 2, top),
                        xytext=(0, 4), textcoords="offset points",
                        ha="center", fontsize=9, fontweight="bold",
                        color=CPX_NAVY if delta > 0 else CPX_RED)
        elif not is_validated:
            ax.annotate("— fixed",
                        xy=(x[i], top),
                        xytext=(0, 4), textcoords="offset points",
                        ha="center", fontsize=7, color=CPX_LGREY,
                        fontweight="bold")
        # Attendance lift label — top right of each chart
        lift_r = next(r for r in (s1_results + [r2] + s3_results) if r["label"] == scen_label)
        lift_text = (
            f"Attendance Lift: +{lift_r['lift_pct']:.1f}%\n"
            f"89% CI: [+{lift_r['pct_lo']:.1f}%, +{lift_r['pct_hi']:.1f}%]\n"
            f"Confidence: {lift_r['confidence']}\n"
            f"— = spend held fixed (geo-test or unvalidated LOB)"
        )

    ax.text(
        0.99, 0.72, lift_text,
        transform=ax.transAxes,
        ha="right", va="top",
        fontsize=11, color=CPX_DGREY, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  edgecolor=CPX_LGREY, alpha=0.95),
    )

    # X-axis — LOB labels
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{lob}\n({fun[:3]})" for fun, lob in lob_keys],
        rotation=45, ha="right", fontsize=10, fontweight="bold", color=CPX_DGREY
    )

    ax.set_ylabel("Annual Media Spend ($M)", fontsize=13, fontweight="bold", color=CPX_DGREY)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:.1f}M"))
    ax.tick_params(axis="x", labelsize=11)
    ax.set_ylim(bottom=0, top=ax.get_ylim()[1] * 1.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", labelsize=11)
    plt.setp(ax.get_yticklabels(), fontweight="bold")

    # Legend: funnel colours + baseline vs scenario
    funnel_patches = [mpatches.Patch(color=FUNNEL_COLORS[f], label=f) for f in FUNNEL_COLORS]
    style_patches  = [
        mpatches.Patch(facecolor=CPX_DGREY, alpha=0.9, label="Baseline 2025"),
        mpatches.Patch(facecolor=CPX_DGREY, alpha=0.5, hatch="///", label="Scenario"),
    ]
    ax.legend(handles=funnel_patches + style_patches,
              fontsize=10, loc="upper right",
              framealpha=0.95, edgecolor=CPX_LGREY)

    ax.set_title(
        f"LOB Spend — Baseline vs {scen_label}\nSplit via graft tilt shares  |  Δ labels in $M",
        fontsize=13, fontweight="bold", color=CPX_NAVY, pad=10
    )

    fig.text(0.99, 0.01,
             "Cineplex Marketing Analytics  |  EXH MMM Phase 2 Iter 2",
             ha="right", va="bottom", fontsize=6, color=CPX_LGREY)

    plt.tight_layout()

    safe = scen_label.replace(" ", "_").replace("%", "pct").replace("+", "plus").replace("/", "_")
    save_fig(fig, f"I2_B10_LOB_Spend_{safe}.png")
    display(fig)
    plt.close(fig)

# COMMAND ----------

# DBTITLE 1,Annual Media Contribution by Original LOB (per year)
# ── CELL: Annual Media Contribution by Original LOB — grouped by year, sourced from decomp ──
# Replicates the grouped-model chart using decomp-derived contributions.
# One bar per (funnel, LOB) per year, coloured by funnel.
# Unvalidated/geo-test LOBs included with funnel-avg att-per-$1 as directional proxy (hatched).

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import numpy as np
import os

CPX_NAVY  = "#002B5C"
CPX_RED   = "#E31837"
CPX_GOLD  = "#F5A800"
CPX_DGREY = "#4A4A4A"
CPX_LGREY = "#CCCCCC"

FUNNEL_COLORS = {"Awareness": CPX_NAVY, "Consideration": CPX_RED, "Conversion": CPX_GOLD}
FUNNELS       = ["Awareness", "Consideration", "Conversion"]

def save_fig(fig, fname):
    fig.savefig(os.path.join(OUT_PATH, fname), dpi=150, bbox_inches="tight")
    print(f"Saved: {fname}")

# ── 1. Validated att-per-$1 from decomp ────────────────────────────────────────
decomp_valid = decomp[decomp["Status"].str.startswith("Decomposed")].copy()

fl_rate = {}
for (funnel, lob), grp in decomp_valid.groupby(["Funnel", "LOB"]):
    valid = grp[grp["Att per $1"].notna() & (grp["Annual Spend"] > 0)]
    if valid.empty:
        fl_rate[(funnel, lob)] = 0.0
    else:
        fl_rate[(funnel, lob)] = (
            (valid["Att per $1"] * valid["Annual Spend"]).sum() / valid["Annual Spend"].sum()
        )

# ── 2. Fallback: funnel-average rate for unvalidated/geo-test LOBs ───────────
funnel_avg_rate = {}
for funnel in FUNNELS:
    rates = [v for (f, _), v in fl_rate.items() if f == funnel and v > 0]
    funnel_avg_rate[funnel] = float(np.mean(rates)) if rates else 0.0

years_avail = sorted(y for y in df_full["year"].dropna().unique().astype(int) if y < 2026)

directional_lobs = set()
for (funnel, lob_raw), cols in funnel_lob_cols.items():
    disp = lob_display_name(lob_raw)
    key  = (funnel, disp)
    if key not in fl_rate:
        has_spend = any(
            float(_spend_by_yr.loc[yr, cols].sum()) > 0
            for yr in years_avail
        )
        if has_spend:
            fl_rate[key] = funnel_avg_rate.get(funnel, 0.0)
            directional_lobs.add(key)

# ── 3. Annual spend per (funnel, LOB, year) — all LOBs ───────────────────
fl_yr_spend = {}
for (funnel, lob_raw), cols in funnel_lob_cols.items():
    disp = lob_display_name(lob_raw)
    key  = (funnel, disp)
    if key not in fl_rate:
        continue
    if key not in fl_yr_spend:
        fl_yr_spend[key] = {yr: 0.0 for yr in years_avail}
    for yr in years_avail:
        fl_yr_spend[key][yr] += float(_spend_by_yr.loc[yr, cols].sum())

# ── 4. Compute year_att_structure (needed for annotation box) ───────────────
year_att_structure = {}
for yr in years_avail:
    actual   = float(_actual_att_by_yr.loc[yr]) if yr in _actual_att_by_yr.index else 0.0
    media    = float(_media_by_yr.loc[yr, [f"media_{f}" for f in FUNNELS]].sum())
    controls = actual - media

    validated_sum = 0.0
    directional_sum = 0.0
    for (funnel, lob) in fl_rate:
        spend = fl_yr_spend.get((funnel, lob), {}).get(yr, 0.0)
        contrib = spend * _lob_rate_yr(funnel, lob, yr)
        if (funnel, lob) in directional_lobs:
            directional_sum += contrib
        else:
            validated_sum += contrib

    year_att_structure[yr] = {
        "actual": actual,
        "controls": controls,
        "media": media,
        "validated": validated_sum,
        "directional_graft": directional_sum,
    }

print(f"Year attendance structure computed for: {years_avail}")
for yr, s in year_att_structure.items():
    print(f"  {yr}: actual={s['actual']/1e6:.2f}M, media={s['media']/1e6:.2f}M, "
          f"validated={s['validated']/1e6:.2f}M, directional={s['directional_graft']/1e6:.2f}M")

# ── 5. Ordered LOB list — validated first, then directional ──────────────────
ORIG_LOBS = [
    {"funnel": f, "lob": lob,
     "color": FUNNEL_COLORS[f],
     "directional": (f, lob) in directional_lobs}
    for f in FUNNELS
    for lob in sorted({l for (fn, l) in fl_rate if fn == f})
]

# ── 6. One chart per year ────────────────────────────────────────────────────
n_lobs  = len(ORIG_LOBS)
bar_w   = 0.7
x_pos   = np.arange(n_lobs)

for yr in years_avail:
    fig, ax = plt.subplots(figsize=(18, 9))
    legend_handles = {f: None for f in FUNNELS}
    legend_handle_dir = None

    for l_idx, lob_item in enumerate(ORIG_LOBS):
        f, lob = lob_item["funnel"], lob_item["lob"]
        is_dir = lob_item["directional"]
        val    = fl_yr_spend.get((f, lob), {}).get(yr, 0.0) * _lob_rate_yr(f, lob, yr)

        bars = ax.bar(x_pos[l_idx], val / 1e6, width=bar_w * 0.85,
                      color=lob_item["color"],
                      alpha=0.4 if is_dir else 0.85,
                      hatch="///" if is_dir else None,
                      edgecolor="white", linewidth=0.4)

        if legend_handles[f] is None:
            legend_handles[f] = bars
        if is_dir and legend_handle_dir is None:
            legend_handle_dir = bars

        # Value label
        if val / 1e6 > 0.02:
            ax.text(x_pos[l_idx], val / 1e6 + 0.01,
                    f"{val/1e6:.2f}M~" if is_dir else f"{val/1e6:.2f}M",
                    ha="center", va="bottom", fontsize=11,
                    color=CPX_RED if is_dir else CPX_DGREY,
                    fontweight="bold")

    # X-axis LOB labels
    lob_labels = [
        li["lob"] + (" ⚠" if li["directional"] else "")
        for li in ORIG_LOBS
    ]
    ax.set_xticks(x_pos)
    ax.set_xticklabels(lob_labels, rotation=45, ha="right", fontsize=12,
                       fontweight="bold", color=CPX_DGREY)

    # Axes & formatting
    ax.set_ylabel("Media Attendance Contribution (M)", fontsize=14, fontweight="bold", color=CPX_DGREY)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.2f}M"))
    ax.set_ylim(0, ax.get_ylim()[1] * 1.25)
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.2, color=CPX_LGREY)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.setp(ax.get_yticklabels(), fontsize=12, fontweight="bold")

    ax.set_title(
        f"Annual Media Contribution by Original LOB — {yr}\n"
        "Media effect only  |  ⚠ Hatched = directional, funnel-avg rate applied",
        fontsize=15, fontweight="bold", color=CPX_NAVY,
    )

    # Legend
    leg_handles = [
        mpatches.Patch(facecolor=FUNNEL_COLORS[f], alpha=0.85, label=f)
        for f in FUNNELS
        if any(li["funnel"] == f for li in ORIG_LOBS
               if fl_yr_spend.get((li["funnel"], li["lob"]), {}).get(yr, 0.0) > 0)
    ]
    leg_labels = [p.get_label() for p in leg_handles]

    if any(li["directional"] for li in ORIG_LOBS):
        leg_handles.append(mpatches.Patch(facecolor=CPX_DGREY, alpha=0.4, hatch="///",
                                          label="Directional ⚠ (unvalidated)"))
        leg_labels.append("Directional ⚠ (unvalidated)")

    ax.legend(handles=leg_handles, labels=leg_labels,
              fontsize=12, framealpha=0.9, edgecolor=CPX_LGREY,
              loc="upper right")

    fig.text(0.99, 0.01,
             "Cineplex Marketing Analytics  |  EXH MMM Phase 2 Iter 2",
             ha="right", va="bottom", fontsize=6, color=CPX_LGREY)

    # ── Attendance structure annotation ──────────────────────────────────────
    s = year_att_structure[yr]
    ann_text = (
        f"Total Attendance:                    {s['actual']/1e6:.2f}M\n"
        f"Baseline (non-media):                {s['controls']/1e6:.2f}M  "
        f"({s['controls']/s['actual']*100:.1f}% of total)\n"
        f"Media (model-validated total):       {s['media']/1e6:.2f}M  "
        f"({s['media']/s['actual']*100:.1f}% of total)\n"
        f"  of which — validated LOBs:         {s['validated']/1e6:.2f}M  "
        f"({s['validated']/s['media']*100:.1f}% of media)\n"
        f"  of which — directional LOBs ⚠:    {s['directional_graft']/1e6:.2f}M  "
        f"({s['directional_graft']/s['media']*100:.1f}% of media)"
    )
    ax.text(
        0.01, 0.97, ann_text,
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=11, color=CPX_DGREY, fontweight="bold",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  edgecolor=CPX_LGREY, alpha=0.95),
    )

    plt.tight_layout()
    save_fig(fig, f"I2_B9_Annual_OrigLOB_{yr}_Decomp.png")
    display(fig)
    plt.close(fig)

    # ── Data Table for this year ───────────────────────────────────────────
    tbl_rows = []
    for li in ORIG_LOBS:
        f_name, lob_name = li["funnel"], li["lob"]
        spend = fl_yr_spend.get((f_name, lob_name), {}).get(yr, 0.0)
        rate  = _lob_rate_yr(f_name, lob_name, yr)
        contrib = spend * rate
        if spend > 0 or contrib > 0:
            tbl_rows.append({
                "Funnel": f_name,
                "LOB": lob_name + (" ⚠" if li["directional"] else ""),
                "Spend ($K)": f"{spend/1e3:.1f}",
                "Att per $1": f"{rate:.2f}",
                "Contribution (M)": f"{contrib/1e6:.3f}",
                "Status": "Directional" if li["directional"] else "Validated",
            })
    _yr_tbl = pd.DataFrame(tbl_rows)
    print(f"\n{'='*60}")
    print(f"  {yr} — Media Contribution by Original LOB")
    print(f"{'='*60}")
    display(_yr_tbl)

# COMMAND ----------

# DBTITLE 1,Annual Media Contribution SHARE by Original LOB (per year)
# ── CELL: Annual Media Contribution SHARE by Original LOB — per year ─────────────
# Duplicate of the absolute-contribution chart, but Y-axis shows each LOB's
# % share of total model media attribution for that year instead of hard M values.
# Re-uses all variables from cell 43: fl_rate, fl_yr_spend, directional_lobs,
# year_att_structure, ORIG_LOBS, _lob_rate_yr, _alloc_factor_yr.

n_lobs = len(ORIG_LOBS)
bar_w  = 0.7
x_pos  = np.arange(n_lobs)

for yr in years_avail:
    total_media_yr = year_att_structure[yr]["media"]  # total model media attribution for year

    fig, ax = plt.subplots(figsize=(18, 9))
    legend_handles    = {f: None for f in FUNNELS}
    legend_handle_dir = None

    for l_idx, lob_item in enumerate(ORIG_LOBS):
        f, lob = lob_item["funnel"], lob_item["lob"]
        is_dir  = lob_item["directional"]
        _contrib = fl_yr_spend.get((f, lob), {}).get(yr, 0.0) * _lob_rate_yr(f, lob, yr)
        val      = (_contrib / total_media_yr * 100) if total_media_yr > 0 else 0.0

        bars = ax.bar(x_pos[l_idx], val, width=bar_w * 0.85,
                      color=lob_item["color"],
                      alpha=0.4 if is_dir else 0.85,
                      hatch="///" if is_dir else None,
                      edgecolor="white", linewidth=0.4)

        if legend_handles[f] is None:
            legend_handles[f] = bars
        if is_dir and legend_handle_dir is None:
            legend_handle_dir = bars

        # Value label — only show if share is meaningful
        if val > 0.4:
            ax.text(x_pos[l_idx], val + ax.get_ylim()[1] * 0.01,
                    f"{val:.1f}%~" if is_dir else f"{val:.1f}%",
                    ha="center", va="bottom", fontsize=11,
                    color=CPX_RED if is_dir else CPX_DGREY,
                    fontweight="bold")

    # X-axis LOB labels
    lob_labels = [li["lob"] + (" \u26a0" if li["directional"] else "") for li in ORIG_LOBS]
    ax.set_xticks(x_pos)
    ax.set_xticklabels(lob_labels, rotation=45, ha="right",
                       fontsize=12, fontweight="bold", color=CPX_DGREY)

    # Axes & formatting
    ax.set_ylabel("Share of Total Media Contribution (%)",
                  fontsize=14, fontweight="bold", color=CPX_DGREY)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.1f}%"))
    _ylim_top = max(ax.get_ylim()[1], 5) * 1.28
    ax.set_ylim(0, _ylim_top)
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.2, color=CPX_LGREY)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.setp(ax.get_yticklabels(), fontsize=12, fontweight="bold")

    ax.set_title(
        f"Share of Annual Media Contribution by Original LOB \u2014 {yr}\n"
        "% of total model media attribution  |  \u26a0 Hatched = directional, funnel-avg rate applied",
        fontsize=15, fontweight="bold", color=CPX_NAVY,
    )

    # Legend
    leg_handles = [
        mpatches.Patch(facecolor=FUNNEL_COLORS[f], alpha=0.85, label=f)
        for f in FUNNELS
        if any(li["funnel"] == f for li in ORIG_LOBS
               if fl_yr_spend.get((li["funnel"], li["lob"]), {}).get(yr, 0.0) > 0)
    ]
    if any(li["directional"] for li in ORIG_LOBS):
        leg_handles.append(mpatches.Patch(facecolor=CPX_DGREY, alpha=0.4, hatch="///",
                                          label="Directional \u26a0 (unvalidated)"))
    ax.legend(handles=leg_handles, labels=[p.get_label() for p in leg_handles],
              fontsize=12, framealpha=0.9, edgecolor=CPX_LGREY, loc="upper right")

    fig.text(0.99, 0.01,
             "Cineplex Marketing Analytics  |  EXH MMM Phase 2 Iter 2",
             ha="right", va="bottom", fontsize=6, color=CPX_LGREY)

    # ── Annotation box: absolute totals for reference ─────────────────────────
    s = year_att_structure[yr]
    ann_text = (
        f"Total Attendance:              {s['actual']/1e6:.2f}M\n"
        f"Baseline (non-media):          {s['controls']/1e6:.2f}M  "
        f"({s['controls']/s['actual']*100:.1f}% of total)\n"
        f"Media (model total):           {s['media']/1e6:.2f}M  "
        f"({s['media']/s['actual']*100:.1f}% of total)\n"
        f"  Validated LOBs:              {s['validated']/1e6:.2f}M  "
        f"({s['validated']/s['media']*100:.1f}% of media)\n"
        f"  Directional LOBs \u26a0:         {s['directional_graft']/1e6:.2f}M  "
        f"({s['directional_graft']/s['media']*100:.1f}% of media)"
    )
    ax.text(0.01, 0.97, ann_text, transform=ax.transAxes,
            ha="left", va="top", fontsize=11, color=CPX_DGREY, fontweight="bold",
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor=CPX_LGREY, alpha=0.95))

    plt.tight_layout()
    save_fig(fig, f"I2_B9_Annual_OrigLOB_{yr}_Share.png")
    display(fig)
    plt.close(fig)

    # ── Data table: share % per LOB ────────────────────────────────────────────
    tbl_rows = []
    for li in ORIG_LOBS:
        f_name, lob_name = li["funnel"], li["lob"]
        spend        = fl_yr_spend.get((f_name, lob_name), {}).get(yr, 0.0)
        rate         = _lob_rate_yr(f_name, lob_name, yr)
        contrib_abs  = spend * rate
        share_pct    = (contrib_abs / total_media_yr * 100) if total_media_yr > 0 else 0.0
        is_dir       = li["directional"]
        if spend > 0 or share_pct > 0:
            tbl_rows.append({
                "Funnel":     f_name,
                "LOB":        lob_name + (" \u26a0" if is_dir else ""),
                "Spend ($K)": f"{spend/1e3:.1f}",
                "Att per $1": f"{rate:.2f}" + ("\u2020" if is_dir else ""),
                "Share (%)": f"{share_pct:.1f}%",
                "Status":     "Directional" if is_dir else "Validated",
            })
    tbl_rows_sorted = sorted(tbl_rows, key=lambda r: -float(r["Share (%)"].replace("%", "")))
    print(f"\n{'='*60}")
    print(f"  {yr} \u2014 Media Contribution Share by Original LOB")
    print(f"{'='*60}")
    display(pd.DataFrame(tbl_rows_sorted))

# COMMAND ----------

# DBTITLE 1,Section 13 — Stakeholder Deliverables
# MAGIC %md
# MAGIC ---
# MAGIC ## Section 13: Stakeholder Deliverables
# MAGIC Master summary CSV, scenario comparison table, and Excel workbook export for stakeholder distribution.

# COMMAND ----------

# DBTITLE 1,EXH Master Summary & CSV Export
# ━━ MASTER SUMMARY & CSV EXPORT ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Combine all scenario results into one master table
all_scenario_rows = []
all_scenario_rows.append({"Scenario": "Baseline (2025)", "Annual Spend": f"${total_baseline_spend:,.0f}",
    "Attendance": f"{baseline_attendance:,.0f}",
    "Total Attendance": f"{float(df_full[df_full['year']==2025][target_col].sum()):,.0f}",
    "Lift": "—", "Lift %": "—",
    "89% CI": "—", "Confidence": "—"})
_total_2025 = float(df_full[df_full['year']==2025][target_col].sum())
for r in s1_results:
    all_scenario_rows.append({"Scenario": f"S1: {r['label']}", "Annual Spend": f"${r['spend']:,.0f}",
        "Attendance": f"{r['attendance']:,.0f}",
        "Total Attendance": f"{_total_2025 + r['lift']:,.0f}",
        "Lift": f"+{r['lift']:,.0f}",
        "Lift %": f"+{r['lift_pct']:.1f}%",
        "89% CI": f"[+{r['pct_lo']:.1f}%, +{r['pct_hi']:.1f}%]", "Confidence": r["confidence"]})
all_scenario_rows.append({"Scenario": f"S2: {r2['label']}", "Annual Spend": f"${r2['spend']:,.0f}",
    "Attendance": f"{r2['attendance']:,.0f}",
    "Total Attendance": f"{_total_2025 + r2['lift']:,.0f}",
    "Lift": f"+{r2['lift']:,.0f}",
    "Lift %": f"+{r2['lift_pct']:.1f}%",
    "89% CI": f"[+{r2['pct_lo']:.1f}%, +{r2['pct_hi']:.1f}%]", "Confidence": r2["confidence"]})
for r in s3_results:
    all_scenario_rows.append({"Scenario": f"S3: {r['label']}", "Annual Spend": f"${r['spend']:,.0f}",
        "Attendance": f"{r['attendance']:,.0f}",
        "Total Attendance": f"{_total_2025 + r['lift']:,.0f}",
        "Lift": f"+{r['lift']:,.0f}",
        "Lift %": f"+{r['lift_pct']:.1f}%",
        "89% CI": f"[+{r['pct_lo']:.1f}%, +{r['pct_hi']:.1f}%]", "Confidence": r["confidence"]})

master_df = pd.DataFrame(all_scenario_rows)
print(f"{'═'*75}")
print(f"  EXH MMM — MASTER SCENARIO SUMMARY")
print(f"{'═'*75}")
display(master_df)

# Save CSV
csv_path = f"{EXH_DECOMP_PATH}scenario_master_summary.csv"
master_df.to_csv(csv_path, index=False)
print(f"\n✓ Master summary CSV → {csv_path}")

# COMMAND ----------

# DBTITLE 1,EXH Final Scenario Charts — Media Lift & Stacked Attendance
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np

CPX_NAVY  = "#002B5C"
CPX_RED   = "#E31837"
CPX_GOLD  = "#F5A800"
CPX_GREEN = "#2E8B57"
CPX_GREY  = "#9AA0A6"
CPX_DGREY = "#4A4A4A"
CPX_LGREY = "#CCCCCC"

# ── Build final_df from scenario results (cells 37–38) ────────────────────────
# actual_total = 2025 total attendance; media_base = media-attributed portion
actual_total = float(df_full[df_full["year"] == 2025][target_col].sum())
media_base   = baseline_attendance  # 10,258,273 (from cell 33/37)

DECOMP_PATH = EXH_DECOMP_PATH

rows = []
for r in s1_results:
    rows.append({
        "Scenario": r["label"], "Type": "budget",
        "Confidence": r["confidence"],
        "Media Lift %": r["lift_pct"],
        "Media CI Low %": r["pct_lo"],
        "Media CI High %": r["pct_hi"],
        "Admissions Gained": r["lift"],
        "Gained CI Low": r["pct_lo"] / 100 * baseline_attendance,
        "Gained CI High": r["pct_hi"] / 100 * baseline_attendance,
    })
rows.append({
    "Scenario": r2["label"], "Type": "reshuffle",
    "Confidence": r2["confidence"],
    "Media Lift %": r2["lift_pct"],
    "Media CI Low %": r2["pct_lo"],
    "Media CI High %": r2["pct_hi"],
    "Admissions Gained": r2["lift"],
    "Gained CI Low": r2["pct_lo"] / 100 * baseline_attendance,
    "Gained CI High": r2["pct_hi"] / 100 * baseline_attendance,
})
for r in s3_results:
    rows.append({
        "Scenario": r["label"], "Type": "combined",
        "Confidence": r["confidence"],
        "Media Lift %": r["lift_pct"],
        "Media CI Low %": r["pct_lo"],
        "Media CI High %": r["pct_hi"],
        "Admissions Gained": r["lift"],
        "Gained CI Low": r["pct_lo"] / 100 * baseline_attendance,
        "Gained CI High": r["pct_hi"] / 100 * baseline_attendance,
    })

final_df = pd.DataFrame(rows)
print(f"final_df built: {len(final_df)} scenarios")
display(final_df[["Scenario","Type","Media Lift %","Confidence"]].to_string(index=False))

# ── Chart helpers ─────────────────────────────────────────────────────────────
def bar_color(row):
    if "crosses" in str(row["Confidence"]).lower(): return CPX_GREY
    return {"budget": CPX_NAVY, "combined": CPX_GREEN, "reshuffle": CPX_GOLD}.get(row["Type"], CPX_GREY)

colors = [bar_color(r) for _, r in final_df.iterrows()]
x      = np.arange(len(final_df))
labels = final_df["Scenario"].tolist()
m_pt   = final_df["Media Lift %"].values
m_lo   = (final_df["Media Lift %"] - final_df["Media CI Low %"]).values
m_hi   = (final_df["Media CI High %"] - final_df["Media Lift %"]).values

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Media-Attributable Lift %
# ═════════════════════════════════════════════════════════════════════════════
fig1, ax = plt.subplots(figsize=(16, 9))
ax.set_axisbelow(True)
ax.grid(axis="y", alpha=0.2, color=CPX_LGREY)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.bar(x, m_pt, color=colors, alpha=0.88, edgecolor="white", width=0.6)
ax.errorbar(x, m_pt, yerr=[m_lo, m_hi],
            fmt="none", ecolor=CPX_DGREY, capsize=5, alpha=0.7, lw=1.5)

for i, (_, r) in enumerate(final_df.iterrows()):
    ax.text(i, m_pt[i] * 0.5, f"+{m_pt[i]:.1f}%",
            ha="center", va="center", fontsize=10,
            fontweight="bold", color="white")
    ax.text(i, m_pt[i] + m_hi[i] + 0.3,
            f"[+{r['Media CI Low %']:.1f}%, +{r['Media CI High %']:.1f}%]",
            ha="center", va="bottom", fontsize=8,
            fontweight="bold", color=CPX_DGREY)

ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=11, fontweight="bold", color=CPX_DGREY)
ax.set_ylabel("2025 Lift on media-attributable attendance (%)", fontsize=13, fontweight="bold", color=CPX_DGREY)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
ax.set_ylim(0, max(m_pt + m_hi) * 1.45)
ax.tick_params(axis="y", labelsize=11, labelcolor=CPX_DGREY)
plt.setp(ax.get_yticklabels(), fontweight="bold")
ax.set_title("2025 Media-Attributable Attendance Lift (By Scenario)",
             fontweight="bold", color=CPX_NAVY, fontsize=13, pad=10)

leg1 = [
    mpatches.Patch(facecolor=CPX_NAVY,  label="Budget uplift"),
    mpatches.Patch(facecolor=CPX_GREEN, label="Combined"),
    # mpatches.Patch(facecolor=CPX_GOLD,  label="Reshuffle"),
    mpatches.Patch(facecolor=CPX_GREY,  label="Directional (CI crosses 0)"),
]
ax.legend(handles=leg1, fontsize=9, loc="upper left",
          framealpha=0.95, edgecolor=CPX_LGREY)

# fig1.text(0.99, 0.01,
#           "Cineplex Marketing Analytics  |  EXH MMM Phase 2 Iter 2",
#           ha="right", va="bottom", fontsize=6, color=CPX_LGREY)

plt.tight_layout()
fig1.savefig(f"{DECOMP_PATH}FINAL_media_lift.png",
             dpi=160, bbox_inches="tight", facecolor="white")
print("✓ Saved FINAL_media_lift.png")
display(fig1); plt.close(fig1)

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Stacked absolute attendance: baseline + media + scenario lift
# ═════════════════════════════════════════════════════════════════════════════
non_media_base = actual_total - media_base   # baseline / non-media component

fig2, ax = plt.subplots(figsize=(16, 9))
ax.set_axisbelow(True)
ax.grid(axis="y", alpha=0.2, color=CPX_LGREY)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

bar_w = 0.6
admissions_gained = final_df["Admissions Gained"].values
gained_lo = (final_df["Admissions Gained"] - final_df["Gained CI Low"]).values
gained_hi = (final_df["Gained CI High"]    - final_df["Admissions Gained"]).values

# Layer 1 — non-media baseline (grey)
ax.bar(x, non_media_base / 1e6, bar_w,
       color=CPX_GREY, alpha=0.6, edgecolor="white", lw=0.4, label="Baseline / Non-Media")

# Layer 2 — current media-attributed attendance (Navy)
ax.bar(x, media_base / 1e6, bar_w,
       bottom=non_media_base / 1e6,
       color=CPX_NAVY, alpha=0.85, edgecolor="white", lw=0.4, label="Current Media Attribution")

# Layer 3 — scenario lift (CPX Gold, consistent across all scenarios)
ax.bar(x, admissions_gained / 1e6, bar_w,
       bottom=(non_media_base + media_base) / 1e6,
       color=CPX_GOLD, alpha=0.9, edgecolor="white", lw=0.4, label="Scenario Lift")

# CI whiskers on top of scenario lift
tops = (non_media_base + media_base + admissions_gained) / 1e6
ax.errorbar(x, tops, yerr=[gained_lo / 1e6, gained_hi / 1e6],
            fmt="none", ecolor=CPX_DGREY, capsize=5, alpha=0.7, lw=1.5)

# Labels on each layer
for i, (_, r) in enumerate(final_df.iterrows()):
    # Non-media label (inside grey bar)
    ax.text(i, (non_media_base / 2) / 1e6,
            f"{non_media_base/1e6:.1f}M",
            ha="center", va="center", fontsize=8,
            fontweight="bold", color="white")

    # Media label (inside navy bar)
    ax.text(i, (non_media_base + media_base / 2) / 1e6,
            f"{media_base/1e6:.1f}M",
            ha="center", va="center", fontsize=8,
            fontweight="bold", color="white")

    # Scenario lift label (inside gold bar) — only if segment tall enough
    if admissions_gained[i] / 1e6 > 0.1:
        ax.text(i, (non_media_base + media_base + admissions_gained[i] / 2) / 1e6,
                f"+{admissions_gained[i]/1e6:.2f}M",
                ha="center", va="center", fontsize=8,
                fontweight="bold", color=CPX_NAVY)

    # CI label above whisker
    ax.text(i, tops[i] + gained_hi[i] / 1e6 + 0.1,
            f"[{r['Gained CI Low']:+,.0f}, {r['Gained CI High']:+,.0f}]",
            ha="center", va="bottom", fontsize=7.5,
            fontweight="bold", color=CPX_DGREY)

ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=11, fontweight="bold", color=CPX_DGREY)
ax.set_ylabel("Annual Attendance (M)", fontsize=13, fontweight="bold", color=CPX_DGREY)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}M"))
ax.set_ylim(0, (actual_total + max(admissions_gained) + max(gained_hi)) / 1e6 * 1.15)
ax.tick_params(axis="y", labelsize=11, labelcolor=CPX_DGREY)
plt.setp(ax.get_yticklabels(), fontweight="bold")
ax.set_title(
    f"Total Attendance by Scenario  |  Total Attendance (2025) {actual_total/1e6:.1f}M  →  Baseline + Media + Scenario Lift",
    fontweight="bold", color=CPX_NAVY, fontsize=13, pad=10
)

ax.legend(fontsize=9, loc="upper left", framealpha=0.95, edgecolor=CPX_LGREY)

fig2.text(0.99, 0.01,
          "Cineplex Marketing Analytics  |  EXH MMM Phase 2 Iter 2",
          ha="right", va="bottom", fontsize=6, color=CPX_LGREY)

plt.tight_layout()
fig2.savefig(f"{DECOMP_PATH}FINAL_attendance_stacked.png",
             dpi=160, bbox_inches="tight", facecolor="white")
print("✓ Saved FINAL_attendance_stacked.png")
display(fig2); plt.close(fig2)

# ── Attendance Decomposition Table (2023–2025) ─────────────────────────────────
_years_tbl = [2023, 2024, 2025]
_tbl_rows = []
for yr in _years_tbl:
    yr_df = df_full[df_full["year"] == yr]
    total = float(yr_df[target_col].sum())
    media_aw = float(yr_df["media_Awareness"].sum())
    media_co = float(yr_df["media_Consideration"].sum())
    media_cv = float(yr_df["media_Conversion"].sum())
    media_total = media_aw + media_co + media_cv
    baseline = total - media_total
    _tbl_rows.append({
        "Year": yr,
        "Total Attendance": f"{total/1e6:.1f}M",
        "Baseline (Non-Media)": f"{baseline/1e6:.1f}M ({baseline/total*100:.0f}%)",
        "Total Media Attributed": f"{media_total/1e6:.1f}M ({media_total/total*100:.0f}%)",
        "— Awareness": f"{media_aw/1e6:.1f}M ({media_aw/total*100:.0f}%)",
        "— Consideration": f"{media_co/1e6:.1f}M ({media_co/total*100:.0f}%)",
        "— Conversion": f"{media_cv/1e6:.1f}M ({media_cv/total*100:.0f}%)",
    })

_att_decomp_tbl = pd.DataFrame(_tbl_rows).set_index("Year").T.reset_index()
_att_decomp_tbl = _att_decomp_tbl.rename(columns={"index": "Metric"})
print(f"\n{'='*60}")
print(f"  Annual Attendance Decomposition by Funnel")
print(f"{'='*60}")
display(_att_decomp_tbl)

# COMMAND ----------

# DBTITLE 1,EXH 2026 Forecast — Scenario Lift over Forecasted Attendance
# ━━ 2026 FORECAST: SCENARIO LIFT OVER FORECASTED ATTENDANCE ━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2026 forecast = 46.38M (externally imposed), media share = 24.1% (from 2025 model)

FORECAST_2026 = 46_380_000
media_share_2025 = media_base / actual_total  # ~24.1%
media_base_2026  = FORECAST_2026 * media_share_2025
non_media_2026   = FORECAST_2026 - media_base_2026

print(f"2026 Forecast Breakdown:")
print(f"  Total:     {FORECAST_2026/1e6:.2f}M")
print(f"  Media:     {media_base_2026/1e6:.2f}M ({media_share_2025*100:.1f}%)")
print(f"  Non-Media: {non_media_2026/1e6:.2f}M")

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Media-Attributable Lift % (relative to 2026 media base)
# ═══════════════════════════════════════════════════════════════════════════
# Lift % recalculated relative to 2026 media base (11.19M vs 10.26M in 2025)
m_pt_2026 = final_df["Admissions Gained"].values / media_base_2026 * 100
m_lo_2026 = final_df["Gained CI Low"].values / media_base_2026 * 100
m_hi_2026 = final_df["Gained CI High"].values / media_base_2026 * 100
m_lo_err  = (m_pt_2026 - m_lo_2026)
m_hi_err  = (m_hi_2026 - m_pt_2026)

def bar_color(row):
    if "crosses" in str(row["Confidence"]).lower(): return CPX_GREY
    return {"budget": CPX_NAVY, "combined": CPX_GREEN, "reshuffle": CPX_GOLD}.get(row["Type"], CPX_GREY)

colors = [bar_color(r) for _, r in final_df.iterrows()]
labels = final_df["Scenario"].tolist()
x = np.arange(len(final_df))

fig1, ax = plt.subplots(figsize=(16, 9))
ax.set_axisbelow(True)
ax.grid(axis="y", alpha=0.2, color=CPX_LGREY)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.bar(x, m_pt_2026, color=colors, alpha=0.88, edgecolor="white", width=0.6)
ax.errorbar(x, m_pt_2026, yerr=[m_lo_err, m_hi_err],
            fmt="none", ecolor=CPX_DGREY, capsize=5, alpha=0.7, lw=1.5)

for i, (_, r) in enumerate(final_df.iterrows()):
    ax.text(i, m_pt_2026[i] * 0.5, f"+{m_pt_2026[i]:.1f}%",
            ha="center", va="center", fontsize=10,
            fontweight="bold", color="white")
    ax.text(i, m_pt_2026[i] + m_hi_err[i] + 0.3,
            f"[+{m_lo_2026[i]:.1f}%, +{m_hi_2026[i]:.1f}%]",
            ha="center", va="bottom", fontsize=8,
            fontweight="bold", color=CPX_DGREY)

ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=11, fontweight="bold", color=CPX_DGREY)
ax.set_ylabel("Lift on 2026 media-attributable attendance (%)", fontsize=13, fontweight="bold", color=CPX_DGREY)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
ax.set_ylim(0, max(m_pt_2026 + m_hi_err) * 1.45)
ax.tick_params(axis="y", labelsize=11, labelcolor=CPX_DGREY)
plt.setp(ax.get_yticklabels(), fontweight="bold")
ax.set_title(
    f"2026 Media-Attributable Attendance Lift (By Scenario)  |  Media Base: {media_base_2026/1e6:.2f}M",
    fontweight="bold", color=CPX_NAVY, fontsize=13, pad=10
)

leg1 = [
    mpatches.Patch(facecolor=CPX_NAVY,  label="Budget uplift"),
    mpatches.Patch(facecolor=CPX_GREEN, label="Combined"),
    mpatches.Patch(facecolor=CPX_GREY,  label="Directional (CI crosses 0)"),
]
ax.legend(handles=leg1, fontsize=9, loc="upper left",
          framealpha=0.95, edgecolor=CPX_LGREY)

plt.tight_layout()
save_fig(fig1, "FINAL_2026_media_lift.png")
display(fig1); plt.close(fig1)

# ── Data Table: 2026 Scenario Summary ───────────────────────────────────────────
tbl_2026 = pd.DataFrame({
    "Scenario": final_df["Scenario"],
    "Type": final_df["Type"],
    "Media Lift %": [f"+{v:.1f}%" for v in m_pt_2026],
    "89% CI": [f"[+{lo:.1f}%, +{hi:.1f}%]" for lo, hi in zip(m_lo_2026, m_hi_2026)],
    "Admissions Gained": [f"+{v:,.0f}" for v in final_df["Admissions Gained"]],
    "Total 2026 Attendance": [f"{(FORECAST_2026 + g)/1e6:.2f}M" for g in final_df["Admissions Gained"]],
    "Confidence": final_df["Confidence"],
})
print(f"\n{'='*70}")
print(f"  2026 Forecast Scenario Summary (Media Base: {media_base_2026/1e6:.2f}M)")
print(f"{'='*70}")
display(tbl_2026)

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Stacked 2026 Forecast: Non-Media + Media + Scenario Lift
# ═══════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(16, 9))
ax.set_axisbelow(True)
ax.grid(axis="y", alpha=0.2, color=CPX_LGREY)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

bar_w = 0.6
admissions_gained = final_df["Admissions Gained"].values
gained_lo = (final_df["Admissions Gained"] - final_df["Gained CI Low"]).values
gained_hi = (final_df["Gained CI High"]    - final_df["Admissions Gained"]).values
labels = final_df["Scenario"].tolist()
x = np.arange(len(final_df))

# Layer 1 — non-media baseline (grey)
ax.bar(x, non_media_2026 / 1e6, bar_w,
       color=CPX_GREY, alpha=0.6, edgecolor="white", lw=0.4,
       label="Baseline / Non-Media")

# Layer 2 — media-attributed (Navy)
ax.bar(x, media_base_2026 / 1e6, bar_w,
       bottom=non_media_2026 / 1e6,
       color=CPX_NAVY, alpha=0.85, edgecolor="white", lw=0.4,
       label="Current Media Attribution")

# Layer 3 — scenario lift (Gold)
ax.bar(x, admissions_gained / 1e6, bar_w,
       bottom=(non_media_2026 + media_base_2026) / 1e6,
       color=CPX_GOLD, alpha=0.9, edgecolor="white", lw=0.4,
       label="Scenario Lift")

# CI whiskers
tops = (non_media_2026 + media_base_2026 + admissions_gained) / 1e6
ax.errorbar(x, tops, yerr=[gained_lo / 1e6, gained_hi / 1e6],
            fmt="none", ecolor=CPX_DGREY, capsize=5, alpha=0.7, lw=1.5)

# Labels
for i, (_, r) in enumerate(final_df.iterrows()):
    ax.text(i, (non_media_2026 / 2) / 1e6,
            f"{non_media_2026/1e6:.1f}M",
            ha="center", va="center", fontsize=8,
            fontweight="bold", color="white")
    ax.text(i, (non_media_2026 + media_base_2026 / 2) / 1e6,
            f"{media_base_2026/1e6:.1f}M",
            ha="center", va="center", fontsize=8,
            fontweight="bold", color="white")
    if admissions_gained[i] / 1e6 > 0.1:
        ax.text(i, (non_media_2026 + media_base_2026 + admissions_gained[i] / 2) / 1e6,
                f"+{admissions_gained[i]/1e6:.2f}M",
                ha="center", va="center", fontsize=8,
                fontweight="bold", color=CPX_NAVY)
    # CI label above whisker
    ax.text(i, tops[i] + gained_hi[i] / 1e6 + 0.1,
            f"[{r['Gained CI Low']:+,.0f}, {r['Gained CI High']:+,.0f}]",
            ha="center", va="bottom", fontsize=7.5,
            fontweight="bold", color=CPX_DGREY)

    # Total with lift
    total_scen = FORECAST_2026 + admissions_gained[i]
    ax.text(i, tops[i] + gained_hi[i] / 1e6 + 0.7,
            f"{total_scen/1e6:.1f}M",
            ha="center", va="bottom", fontsize=9,
            fontweight="bold", color=CPX_NAVY)

ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=11, fontweight="bold", color=CPX_DGREY)
ax.set_ylabel("Annual Attendance (M)", fontsize=13, fontweight="bold", color=CPX_DGREY)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}M"))
ax.set_ylim(0, (FORECAST_2026 + max(admissions_gained) + max(gained_hi)) / 1e6 * 1.15)
ax.tick_params(axis="y", labelsize=11, labelcolor=CPX_DGREY)
plt.setp(ax.get_yticklabels(), fontweight="bold")
ax.set_title(
    f"2026 Forecasted Attendance by Scenario  |  Total Attendance {FORECAST_2026/1e6:.1f}M  →  Baseline + Media + Scenario Lift",
    fontweight="bold", color=CPX_NAVY, fontsize=13, pad=10
)

ax.legend(fontsize=9, loc="upper left", framealpha=0.95, edgecolor=CPX_LGREY)

fig.text(0.99, 0.01,
         "Cineplex Marketing Analytics  |  EXH MMM Phase 2 Iter 2",
         ha="right", va="bottom", fontsize=6, color=CPX_LGREY)

plt.tight_layout()
save_fig(fig, "FINAL_2026_attendance_stacked.png")
display(fig); plt.close(fig)

# COMMAND ----------

# DBTITLE 1,2026 Forecast — Total Attendance by Scenario (Reference-Line View)
# ── 2027 SCENARIO UPSIDE — Ladder to Best Case (compact) ─────────────────────
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

df_plot = final_df.sort_values("Admissions Gained", ascending=True).reset_index(drop=True)
labels  = df_plot["Scenario"].tolist()
gains   = (df_plot["Admissions Gained"] / 1e6).to_numpy()
lows    = (df_plot["Gained CI Low"]  / 1e6).to_numpy()
highs   = (df_plot["Gained CI High"] / 1e6).to_numpy()
n             = len(df_plot)
ceiling       = gains.max()
FORECAST_2027 = 47_400_000

colors = [bar_color(df_plot.iloc[i]) for i in range(n)]
RECOMMENDED = labels[int(np.argmax(gains))]
for i in range(n):
    if labels[i] == RECOMMENDED:
        colors[i] = CPX_GOLD                             # best case → gold
    elif df_plot.iloc[i]["Type"] == "reshuffle":
        colors[i] = CPX_RED                              # reshuffle → red

fig, ax = plt.subplots(figsize=(13, 4.6))
y = np.arange(n)
x_ci = ceiling * 1.02   # fixed x — all CI labels aligned to same column
for i in range(n):
    is_rec = labels[i] == RECOMMENDED
    c = colors[i]
    ax.barh(y[i], ceiling, height=0.72, color=CPX_LGREY, alpha=0.28,
            edgecolor="none", zorder=1)
    ax.barh(y[i], gains[i], height=0.72, color=c,
            alpha=0.92 if is_rec else 0.8, edgecolor="none", zorder=2)
    ax.text(gains[i] - ceiling*0.012, y[i], f"+{gains[i]:.2f}M",
            ha="right", va="center", fontsize=10 if is_rec else 9,
            fontweight="bold", color="white", zorder=4)
    att_27 = FORECAST_2027 / 1e6 + gains[i]
    ax.text(x_ci, y[i],
            f"[{lows[i]:.2f}–{highs[i]:.2f}]",
            ha="left", va="center", fontsize=9,
            fontweight="bold", color=c, zorder=4, clip_on=False)
    ax.text(x_ci + 0.38, y[i],
            f"2027: {att_27:.1f}M",
            ha="left", va="center", fontsize=7.5,
            fontweight="bold", color=CPX_GREEN, zorder=4, clip_on=False)

ax.axvline(ceiling, color=CPX_RED, lw=1.4, ls="--", alpha=0.8, zorder=5)
ax.text(ceiling, n-0.30, f"Best case +{ceiling:.2f}M",
        ha="center", va="bottom", fontsize=8.5, fontweight="bold",
        color=CPX_RED, zorder=6)

ax.set_yticks(y)
ytick_labels = [f"★ {l}" if l == RECOMMENDED else l for l in labels]
ax.set_yticklabels(ytick_labels, fontsize=9.5, fontweight="bold", color=CPX_DGREY)
ax.tick_params(axis="y", length=0, pad=8)
ax.set_xlim(0, 3.5)
ax.set_ylim(-0.6, n+0.5)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"+{v:.1f}M"))
plt.setp(ax.get_xticklabels(), fontsize=8.5, fontweight="bold", color=CPX_DGREY)
for sp in ("top", "right", "left", "bottom"):
    ax.spines[sp].set_visible(False)
ax.grid(axis="x", alpha=0.14, color=CPX_LGREY)
ax.set_axisbelow(True)
ax.tick_params(axis="x", labelbottom=False, length=0)

leg = [mpatches.Patch(facecolor=CPX_NAVY,  label="Budget increase"),
       mpatches.Patch(facecolor=CPX_GREEN, label="Combined"),
       mpatches.Patch(facecolor=CPX_GOLD,  label="Best case scenario"),
       mpatches.Patch(facecolor=CPX_RED,   label="Reshuffle (10% cap)"),
       mpatches.Patch(facecolor=CPX_LGREY, alpha=0.28, label="Gap to best case"),
       Line2D([0], [0], color=CPX_RED,   lw=1.4, ls="--", alpha=0.8, label="Best case line"),
       Line2D([0], [0], color=CPX_DGREY, lw=1.5, ls="-",  label="[low–high] = 89% CI")]
fig.legend(handles=leg, fontsize=7.5,
           loc="lower center", bbox_to_anchor=(0.5, 0.80),
           ncol=4, frameon=True, framealpha=0.9, edgecolor=CPX_LGREY,
           borderpad=0.5, labelspacing=0.5, handlelength=1.5)

fig.suptitle("2027 Scenario Upside — Climb to Best Case",
             fontsize=13, fontweight="bold", color=CPX_NAVY, y=0.98)
fig.text(0.99, 0.005, "Cineplex Marketing Analytics  |  EXH MMM Phase 2 Iter 2",
         ha="right", va="bottom", fontsize=6, color=CPX_LGREY)

plt.tight_layout(rect=[0, 0, 1, 1])
save_fig(fig, "FINAL_2027_scenario_ladder_compact.png")
display(fig)
plt.close(fig)

# COMMAND ----------

# DBTITLE 1,2025–2027 Attendance Build — Baseline + Media Split (Pre-Scenario)
# ── 2025–2027 ATTENDANCE BUILD — Baseline + Media Split (Pre-Scenario Lift) ─────

_ms = media_base / actual_total          # 2025 media share (~24.1%), stable assumption

_data = [
    {"year": "2025 (Actual)",   "total": actual_total,  "media": media_base},
    {"year": "2026 (Forecast)", "total": FORECAST_2026, "media": FORECAST_2026 * _ms},
    {"year": "2027 (Forecast)", "total": FORECAST_2027, "media": FORECAST_2027 * _ms},
]
_max_tot = max(d["total"] for d in _data) / 1e6

fig, axes = plt.subplots(1, 3, figsize=(22, 2))
fig.subplots_adjust(wspace=0.1)

for ax, d in zip(axes, _data):
    nm  = (d["total"] - d["media"]) / 1e6
    m   =  d["media"] / 1e6
    tot =  d["total"] / 1e6

    ax.barh(0, nm, height=0.32, color=CPX_GREY,  alpha=0.65, edgecolor="none", zorder=2)
    ax.barh(0, m,  height=0.32, color=CPX_NAVY,  alpha=0.85, edgecolor="none", zorder=2, left=nm)

    ax.text(nm * 0.5, 0,
            f"Non-Media\n{nm:.1f}M  ({nm/tot*100:.0f}%)",
            ha="center", va="center", fontsize=12, fontweight="bold", color="black", zorder=3)
    ax.text(nm + m * 0.5, 0,
            f"Media\n{m:.1f}M  ({m/tot*100:.0f}%)",
            ha="center", va="center", fontsize=12, fontweight="bold", color="white", zorder=3)

    ax.set_title(f"{d['year']}\n{tot:.1f}M", fontsize=12, fontweight="bold",
                 color='black', pad=2)
    ax.set_yticks([])
    ax.set_ylim(-0.22, 0.22)
    ax.set_xlim(0, _max_tot * 1.04)
    ax.tick_params(axis="x", labelbottom=False, length=0)
    for sp in ("top", "right", "left", "bottom"):
        ax.spines[sp].set_visible(False)
    ax.grid(axis="x", alpha=0.12, color=CPX_LGREY)
    ax.set_axisbelow(True)

leg = [mpatches.Patch(facecolor=CPX_GREY,  alpha=0.65, label="Baseline / Non-Media"),
       mpatches.Patch(facecolor=CPX_NAVY,  alpha=0.85, label="Media Attribution")]
fig.legend(handles=leg, fontsize=9, loc="lower center",
           bbox_to_anchor=(0.5, -0.04), ncol=2,
           framealpha=0.9, edgecolor=CPX_LGREY, borderpad=0.5)

fig.text(0.99, -0.08, "Cineplex Marketing Analytics  |  EXH MMM Phase 2 Iter 2",
         ha="right", va="bottom", fontsize=6, color=CPX_LGREY)

plt.tight_layout()
save_fig(fig, "FINAL_attendance_build_2025_2027.png")
display(fig)
plt.close(fig)

# COMMAND ----------

# DBTITLE 1,EXH Excel Workbook Export
# ━━ EXCEL WORKBOOK EXPORT ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
import openpyxl

_tmp_excel = "/tmp/EXH_MMM_Results.xlsx"
excel_path = f"{EXH_ARTIFACT_PATH}EXH_MMM_Results.xlsx"

with pd.ExcelWriter(_tmp_excel, engine="openpyxl") as writer:
    # Sheet 1: Scenario Summary
    master_df.to_excel(writer, sheet_name="Scenarios", index=False)

    # Sheet 2: LOB Decomposition
    decomp.to_excel(writer, sheet_name="LOB Decomposition", index=False)

    # Sheet 3: Annual Decomposition
    decomp_annual_table.to_excel(writer, sheet_name="Annual Decomp", index=False)

    # Sheet 4: Response Parameters
    params_df = pd.DataFrame([
        {"Combo": c, **{k: v for k, v in p.items()}} for c, p in response_params.items()
    ])
    params_df.to_excel(writer, sheet_name="Response Params", index=False)

    # Sheet 5: Model Accuracy
    acc_df = pd.DataFrame([{
        "Metric": "Train MAPE", "Value": f"{mape_train:.2f}%"},
        {"Metric": "Val MAPE", "Value": f"{mape_val:.2f}%"},
        {"Metric": "Train R²", "Value": f"{r2_train:.4f}"},
        {"Metric": "Val R²", "Value": f"{r2_val:.4f}"},
    ])
    acc_df.to_excel(writer, sheet_name="Model Accuracy", index=False)

shutil.copy2(_tmp_excel, excel_path)
os.remove(_tmp_excel)
print(f"✓ Excel workbook saved → {excel_path}")
print(f"  Sheets: Scenarios, LOB Decomposition, Annual Decomp, Response Params, Model Accuracy")

# COMMAND ----------

# MAGIC %md
# MAGIC Budget Meeting Scenarios
# MAGIC

# COMMAND ----------

# ━━ 2026F MEDIA-ATTRIBUTABLE ATTENDANCE (2025 pattern, 2026 budget) ━━━━━━━━━━━━━━━

BUDGET_2026 = 4_786_700

hist = df_full[df_full["year"].isin([2024, 2025])]          # assumes df_full is in week order
wk_2025 = hist[hist["year"] == 2025]
scale_2026 = BUDGET_2026 / year_points[2025]["spend"]        # same 2025 total as curve legend

def media_attendance_by_week(spend_df):
    """Weekly media attendance: posterior-mean params, week-by-week carryover."""
    total = np.zeros(len(spend_df))
    for combo in ACTIVE_COMBOS:
        p = response_params[combo]
        s_norm = spend_df[combo].to_numpy() / spend_train_max_iter2[combo]
        ad, carry = np.zeros_like(s_norm), 0.0
        for t, s in enumerate(s_norm):
            carry = s + p["alpha"] * carry
            ad[t] = carry
        if p["saturation"] == "hill":
            sat = ad ** p["alpha_s"] / (ad ** p["alpha_s"] + p["k"] ** p["alpha_s"])
        else:
            sat = 1 - np.exp(-p["lam"] * ad)
        total += p["gamma"] * sat * scale_factor
    return total

# 2024 actual → 2025 actual → 2026 (= 2025 pattern × scale)
run = pd.concat([hist[ACTIVE_COMBOS], wk_2025[ACTIVE_COMBOS] * scale_2026], ignore_index=True)
weekly = media_attendance_by_week(run)

n_24, n_25 = (hist["year"] == 2024).sum(), len(wk_2025)
att_2025 = weekly[n_24:n_24 + n_25].sum()
att_2026 = weekly[n_24 + n_25:].sum()

print(f"Budget scale vs 2025 : {scale_2026 - 1:+.2%}")
print(f"2025 media attendance: {att_2025/1e6:.2f}M  (check vs decomp table: 10.26M)")
print(f"2026 media attendance: {att_2026/1e6:.2f}M")
print(f"Lift                 : {(att_2026 - att_2025)/1e6:+.2f}M ({att_2026/att_2025 - 1:+.2%})")

# COMMAND ----------

# ── EXH ANNUAL ATTENDANCE DECOMPOSITION BY FUNNEL + 2026 FORECAST ────────────
# Uses: df_full, FUNNEL_COMBOS, target_col, CPX_*, save_fig from cell 23.
import matplotlib.patheffects as pe
import matplotlib.ticker as mticker

CPX_MGREY = "#9AA0A6"

# ── Aggregate Annual by Funnel ────────────────────────────────────────────────
agg_dict = {target_col: "sum"}
for funnel in FUNNEL_COMBOS:
    agg_dict[f"media_{funnel}"] = "sum"

annual      = df_full[df_full["year"] != 2026].groupby("year").agg(agg_dict).round(0)
years_avail = sorted(annual.index.tolist())
actual_vals = annual.loc[years_avail, target_col].values

awareness_vals     = annual.loc[years_avail, "media_Awareness"].values
consideration_vals = annual.loc[years_avail, "media_Consideration"].values
conversion_vals    = annual.loc[years_avail, "media_Conversion"].values
baseline_vals      = actual_vals - awareness_vals - consideration_vals - conversion_vals

# ── 2026 / 2027 Forecast (model response params, budget-driven media) ─────────
forecast_2026 = 47_783_007
forecast_2027 = 47_400_000          # unchanged — below 2026F, confirm
BUDGET_2026   = 4_786_700
BUDGET_2027   = BUDGET_2026 * 1.30
FUNNELS       = ["Awareness", "Consideration", "Conversion"]

def media_by_funnel(spend_df):
    """Weekly media attendance per funnel: model response params + week-by-week carryover."""
    out = {f: np.zeros(len(spend_df)) for f in FUNNELS}
    for combo in ACTIVE_COMBOS:
        p = response_params[combo]
        s_norm = spend_df[combo].to_numpy() / spend_train_max_iter2[combo]
        ad, carry = np.zeros_like(s_norm), 0.0
        for t, s in enumerate(s_norm):
            carry = s + p["alpha"] * carry
            ad[t] = carry
        if p["saturation"] == "hill":
            sat = ad ** p["alpha_s"] / (ad ** p["alpha_s"] + p["k"] ** p["alpha_s"])
        else:
            sat = 1 - np.exp(-p["lam"] * ad)
        out[combo.split("__")[0]] += p["gamma"] * sat * scale_factor
    return out

# 2024 actual → 2025 actual → 2026 (2025 pattern × scale) → 2027 (2025 pattern × scale)
hist       = df_full[df_full["year"].isin([2024, 2025])]
wk_2025    = hist[hist["year"] == 2025][ACTIVE_COMBOS]
spend_2025 = year_points[2025]["spend"]
run = pd.concat([hist[ACTIVE_COMBOS],
                 wk_2025 * (BUDGET_2026 / spend_2025),
                 wk_2025 * (BUDGET_2027 / spend_2025)], ignore_index=True)
wk = media_by_funnel(run)

n24, n25 = int((hist["year"] == 2024).sum()), len(wk_2025)
s25 = slice(n24, n24 + n25)
s26 = slice(n24 + n25, n24 + 2 * n25)
s27 = slice(n24 + 2 * n25, None)

# 2026: 2025 media share of the new total × model's budget lift (set budget_lift = 1.0 to ignore budget)
media_2025_total = awareness_vals[-1] + consideration_vals[-1] + conversion_vals[-1]
media_share_2025 = media_2025_total / actual_vals[-1]
replay_2026      = {f: wk[f][s26].sum() for f in FUNNELS}
budget_lift      = sum(replay_2026.values()) / sum(wk[f][s25].sum() for f in FUNNELS)
media_2026_total = forecast_2026 * media_share_2025 * budget_lift
aw_2026, csd_2026, cvs_2026 = (media_2026_total * replay_2026[f] / sum(replay_2026.values()) for f in FUNNELS)
baseline_2026    = forecast_2026 - media_2026_total

aw_2027, csd_2027, cvs_2027 = (wk[f][s27].sum() for f in FUNNELS)
media_2027_total = aw_2027 + csd_2027 + cvs_2027
baseline_2027    = forecast_2027 - media_2027_total

print(f"2025 media share: {media_share_2025:.2%} | budget lift: {budget_lift - 1:+.2%}")
print(f"2026 media: {media_2026_total/1e6:.2f}M ({media_2026_total/forecast_2026:.1%}) | "
      f"baseline: {baseline_2026/1e6:.2f}M")

# ── Combine Actuals + Forecast ────────────────────────────────────────────────
all_years      = years_avail + [2026, 2027]
all_actual     = list(actual_vals) + [forecast_2026, forecast_2027]
all_baseline   = list(baseline_vals) + [baseline_2026, baseline_2027]
all_awareness  = list(awareness_vals) + [aw_2026, aw_2027]
all_consid     = list(consideration_vals) + [csd_2026, csd_2027]
all_conversion = list(conversion_vals) + [cvs_2026, cvs_2027]

# ── Helper ────────────────────────────────────────────────────────────────────
def bar_label(ax, x_pos, bottom, height, pct, color, min_height=0.5):
    if abs(height) / 1e6 < min_height:
        return
    light_colors = {CPX_LGREY, CPX_GOLD}
    txt_color = "black" if color in light_colors else "white"
    ax.text(
        x_pos,
        (bottom + height / 2) / 1e6,
        f"{pct:.0f}%",
        ha="center",
        va="center",
        fontsize=18,
        fontweight="bold",
        color=txt_color,
        path_effects=[pe.withStroke(
            linewidth=2,
            foreground="white" if txt_color == "black" else "black"
        )],
    )

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(16, 9))
fig.patch.set_facecolor("white")

x = np.arange(len(all_years))
w = 0.5
alpha = [0.9, 0.9, 0.9, 0.5, 0.5]  # 2026/2027 bars slightly transparent to indicate forecast

for i, year in enumerate(all_years):
    a = alpha[i]
    b_base = all_baseline[i]
    b_aw   = all_awareness[i]
    b_csd  = all_consid[i]
    b_cvs  = all_conversion[i]
    total  = all_actual[i]

    ax.bar(x[i], b_base / 1e6,
           width=w, color=CPX_LGREY, alpha=a,
           edgecolor="white", linewidth=0.5,
           label="Baseline (non-media)" if i == 0 else "_nolegend_")
    ax.bar(x[i], b_aw / 1e6,
           width=w, bottom=b_base / 1e6,
           color=CPX_NAVY, alpha=a,
           edgecolor="white", linewidth=0.5,
           label="Awareness" if i == 0 else "_nolegend_")
    ax.bar(x[i], b_csd / 1e6,
           width=w, bottom=(b_base + b_aw) / 1e6,
           color=CPX_RED, alpha=a,
           edgecolor="white", linewidth=0.5,
           label="Consideration" if i == 0 else "_nolegend_")
    ax.bar(x[i], b_cvs / 1e6,
           width=w, bottom=(b_base + b_aw + b_csd) / 1e6,
           color=CPX_GOLD, alpha=a,
           edgecolor="white", linewidth=0.5,
           label="Conversion" if i == 0 else "_nolegend_")

    bar_label(ax, x[i], 0,                     b_base, b_base / total * 100, CPX_LGREY)
    bar_label(ax, x[i], b_base,                b_aw,   b_aw   / total * 100, CPX_NAVY, 1.5)
    bar_label(ax, x[i], b_base + b_aw,         b_csd,  b_csd  / total * 100, CPX_RED,  1.5)
    bar_label(ax, x[i], b_base + b_aw + b_csd, b_cvs,  b_cvs  / total * 100, CPX_GOLD, 1.5)

    label = f"{total/1e6:.2f}M" + (" (F)" if year >= 2026 else "")
    ax.text(x[i], total / 1e6 + 0.3, label,
            ha="center", va="bottom",
            fontsize=16, fontweight="bold",
            color=CPX_RED if year == 2026 else CPX_DGREY)

# Actual dots (2023-2025 only)
ax.scatter(x[:3], np.array(actual_vals) / 1e6,
           color="black", zorder=5, s=80,
           marker="D", label="Actual Attendance")

# Forecast marker
ax.scatter(x[3:], [forecast_2026 / 1e6, forecast_2027 / 1e6],
           color=CPX_RED, zorder=5, s=80,
           marker="D", label="2026/2027 Forecast")

ax.axvline(x=2.75, color=CPX_MGREY, linewidth=1, linestyle="--")
ax.text(2.78, 2, "Forecast →",
        fontsize=9, color=CPX_MGREY, fontstyle="italic")

ax.set_xticks(x)
ax.set_xticklabels(all_years, fontsize=16, fontweight="bold")
plt.setp(ax.get_yticklabels(), fontsize=16, fontweight="bold", color="black")
ax.set_ylabel("Annual Attendance (M)", fontsize=16, fontweight="bold")
ax.set_xlabel("Year", fontsize=16, fontweight="bold")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}M"))
ax.set_title("EXH Annual Attendance Decomposition by Funnel (2023–2027F)", loc="center",
             fontsize=18, fontweight="bold", color=CPX_NAVY)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=3, fontsize=16, frameon=False)
ax.grid(False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
save_fig(fig, "exh_annual_decomposition.png")
display(fig)
plt.close(fig)

# ── Data table retained for downstream export ────────────────────────────────
decomp_annual_table = pd.DataFrame({
    "Year": [str(y) + (" (F)" if y >= 2026 else "") for y in all_years],
    "Baseline (M)": [round(v / 1e6, 2) for v in all_baseline],
    "Awareness (M)": [round(v / 1e6, 2) for v in all_awareness],
    "Consideration (M)": [round(v / 1e6, 2) for v in all_consid],
    "Conversion (M)": [round(v / 1e6, 2) for v in all_conversion],
    "Media Total (M)": [round((a + c + cv) / 1e6, 2) for a, c, cv in zip(all_awareness, all_consid, all_conversion)],
    "Total (M)": [round(v / 1e6, 2) for v in all_actual],
    "Media %": [round((a + c + cv) / t * 100, 1) for a, c, cv, t in zip(all_awareness, all_consid, all_conversion, all_actual)],
})
print(f"{'═'*60}\n  EXH Attendance Decomposition by Funnel (2023-2026F)\n{'═'*60}")
display(decomp_annual_table)

# COMMAND ----------

# ── EXH ANNUAL ATTENDANCE DECOMPOSITION BY FUNNEL + 2026/2027 FORECAST ────────
# Uses: df_full, FUNNEL_COMBOS, target_col, CPX_*, save_fig from cell 23.
import matplotlib.patheffects as pe
import matplotlib.ticker as mticker

CPX_MGREY = "#9AA0A6"

# ── Aggregate Annual by Funnel ────────────────────────────────────────────────
agg_dict = {target_col: "sum"}
for funnel in FUNNEL_COMBOS:
    agg_dict[f"media_{funnel}"] = "sum"

annual      = df_full[df_full["year"] != 2026].groupby("year").agg(agg_dict).round(0)
years_avail = sorted(annual.index.tolist())
actual_vals = annual.loc[years_avail, target_col].values

awareness_vals     = annual.loc[years_avail, "media_Awareness"].values
consideration_vals = annual.loc[years_avail, "media_Consideration"].values
conversion_vals    = annual.loc[years_avail, "media_Conversion"].values
baseline_vals      = actual_vals - awareness_vals - consideration_vals - conversion_vals

# ── 2026 / 2027 Forecast (model response params, budget-driven media) ─────────
forecast_2026 = 47_783_007
forecast_2027 = 47_400_000          # unchanged — below 2026F, confirm
BUDGET_2026   = 4_786_700
BUDGET_2027   = 4_965_700
FUNNELS       = ["Awareness", "Consideration", "Conversion"]

def media_by_funnel(spend_df):
    """Weekly media attendance per funnel: model response params + week-by-week carryover."""
    out = {f: np.zeros(len(spend_df)) for f in FUNNELS}
    for combo in ACTIVE_COMBOS:
        p = response_params[combo]
        s_norm = spend_df[combo].to_numpy() / spend_train_max_iter2[combo]
        ad, carry = np.zeros_like(s_norm), 0.0
        for t, s in enumerate(s_norm):
            carry = s + p["alpha"] * carry
            ad[t] = carry
        if p["saturation"] == "hill":
            sat = ad ** p["alpha_s"] / (ad ** p["alpha_s"] + p["k"] ** p["alpha_s"])
        else:
            sat = 1 - np.exp(-p["lam"] * ad)
        out[combo.split("__")[0]] += p["gamma"] * sat * scale_factor
    return out

# 2024 actual → 2025 actual → 2026 (2025 pattern × scale) → 2027 (2025 pattern × scale)
hist       = df_full[df_full["year"].isin([2024, 2025])]
wk_2025    = hist[hist["year"] == 2025][ACTIVE_COMBOS]
spend_2025 = year_points[2025]["spend"]
run = pd.concat([hist[ACTIVE_COMBOS],
                 wk_2025 * (BUDGET_2026 / spend_2025),
                 wk_2025 * (BUDGET_2027 / spend_2025)], ignore_index=True)
wk = media_by_funnel(run)

n24, n25 = int((hist["year"] == 2024).sum()), len(wk_2025)
s25 = slice(n24, n24 + n25)
s26 = slice(n24 + n25, n24 + 2 * n25)
s27 = slice(n24 + 2 * n25, None)

# 2026 & 2027: model-based media attendance directly from response curves (no share/lift scaling)
aw_2026, csd_2026, cvs_2026 = (wk[f][s26].sum() for f in FUNNELS)
media_2026_total = aw_2026 + csd_2026 + cvs_2026
baseline_2026    = forecast_2026 - media_2026_total

aw_2027, csd_2027, cvs_2027 = (wk[f][s27].sum() for f in FUNNELS)
media_2027_total = aw_2027 + csd_2027 + cvs_2027
baseline_2027    = forecast_2027 - media_2027_total

print(f"2026 | Budget ${BUDGET_2026:,.0f} | Media attendance: {media_2026_total/1e6:.2f}M "
      f"({media_2026_total/forecast_2026:.1%}) | Baseline: {baseline_2026/1e6:.2f}M")
print(f"2027 | Budget ${BUDGET_2027:,.0f} | Media attendance: {media_2027_total/1e6:.2f}M "
      f"({media_2027_total/forecast_2027:.1%}) | Baseline: {baseline_2027/1e6:.2f}M")

# ── Combine Actuals + Forecast ────────────────────────────────────────────────
all_years      = years_avail + [2026, 2027]
all_actual     = list(actual_vals) + [forecast_2026, forecast_2027]
all_baseline   = list(baseline_vals) + [baseline_2026, baseline_2027]
all_awareness  = list(awareness_vals) + [aw_2026, aw_2027]
all_consid     = list(consideration_vals) + [csd_2026, csd_2027]
all_conversion = list(conversion_vals) + [cvs_2026, cvs_2027]

# ── Helper ────────────────────────────────────────────────────────────────────
def bar_label(ax, x_pos, bottom, height, pct, color, min_height=0.5):
    if abs(height) / 1e6 < min_height:
        return
    light_colors = {CPX_LGREY, CPX_GOLD}
    txt_color = "black" if color in light_colors else "white"
    ax.text(
        x_pos,
        (bottom + height / 2) / 1e6,
        f"{pct:.0f}%",
        ha="center",
        va="center",
        fontsize=18,
        fontweight="bold",
        color=txt_color,
        path_effects=[pe.withStroke(
            linewidth=2,
            foreground="white" if txt_color == "black" else "black"
        )],
    )

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(16, 9))
fig.patch.set_facecolor("white")

x = np.arange(len(all_years))
w = 0.5
alpha = [0.9, 0.9, 0.9, 0.5, 0.5]  # 2026/2027 bars slightly transparent to indicate forecast

for i, year in enumerate(all_years):
    a = alpha[i]
    b_base = all_baseline[i]
    b_aw   = all_awareness[i]
    b_csd  = all_consid[i]
    b_cvs  = all_conversion[i]
    total  = all_actual[i]

    ax.bar(x[i], b_base / 1e6,
           width=w, color=CPX_LGREY, alpha=a,
           edgecolor="white", linewidth=0.5,
           label="Baseline (non-media)" if i == 0 else "_nolegend_")
    ax.bar(x[i], b_aw / 1e6,
           width=w, bottom=b_base / 1e6,
           color=CPX_NAVY, alpha=a,
           edgecolor="white", linewidth=0.5,
           label="Awareness" if i == 0 else "_nolegend_")
    ax.bar(x[i], b_csd / 1e6,
           width=w, bottom=(b_base + b_aw) / 1e6,
           color=CPX_RED, alpha=a,
           edgecolor="white", linewidth=0.5,
           label="Consideration" if i == 0 else "_nolegend_")
    ax.bar(x[i], b_cvs / 1e6,
           width=w, bottom=(b_base + b_aw + b_csd) / 1e6,
           color=CPX_GOLD, alpha=a,
           edgecolor="white", linewidth=0.5,
           label="Conversion" if i == 0 else "_nolegend_")

    bar_label(ax, x[i], 0,                     b_base, b_base / total * 100, CPX_LGREY)
    bar_label(ax, x[i], b_base,                b_aw,   b_aw   / total * 100, CPX_NAVY, 1.5)
    bar_label(ax, x[i], b_base + b_aw,         b_csd,  b_csd  / total * 100, CPX_RED,  1.5)
    bar_label(ax, x[i], b_base + b_aw + b_csd, b_cvs,  b_cvs  / total * 100, CPX_GOLD, 1.5)

    label = f"{total/1e6:.2f}M" + (" (F)" if year >= 2026 else "")
    ax.text(x[i], total / 1e6 + 0.3, label,
            ha="center", va="bottom",
            fontsize=16, fontweight="bold",
            color=CPX_RED if year == 2026 else CPX_DGREY)

# Actual dots (2023-2025 only)
ax.scatter(x[:3], np.array(actual_vals) / 1e6,
           color="black", zorder=5, s=80,
           marker="D", label="Actual Attendance")

# Forecast marker
ax.scatter(x[3:], [forecast_2026 / 1e6, forecast_2027 / 1e6],
           color=CPX_RED, zorder=5, s=80,
           marker="D", label="2026/2027 Forecast")

ax.axvline(x=2.75, color=CPX_MGREY, linewidth=1, linestyle="--")
ax.text(2.78, 2, "Forecast →",
        fontsize=9, color=CPX_MGREY, fontstyle="italic")

ax.set_xticks(x)
ax.set_xticklabels(all_years, fontsize=16, fontweight="bold")
plt.setp(ax.get_yticklabels(), fontsize=16, fontweight="bold", color="black")
ax.set_ylabel("Annual Attendance (M)", fontsize=16, fontweight="bold")
ax.set_xlabel("Year", fontsize=16, fontweight="bold")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}M"))
ax.set_title("EXH Annual Attendance Decomposition by Funnel (2023–2027F)", loc="center",
             fontsize=18, fontweight="bold", color=CPX_NAVY)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=3, fontsize=16, frameon=False)
ax.grid(False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
save_fig(fig, "exh_annual_decomposition.png")
display(fig)
plt.close(fig)

# ── Data table retained for downstream export ────────────────────────────────
decomp_annual_table = pd.DataFrame({
    "Year": [str(y) + (" (F)" if y >= 2026 else "") for y in all_years],
    "Baseline (M)": [round(v / 1e6, 2) for v in all_baseline],
    "Awareness (M)": [round(v / 1e6, 2) for v in all_awareness],
    "Consideration (M)": [round(v / 1e6, 2) for v in all_consid],
    "Conversion (M)": [round(v / 1e6, 2) for v in all_conversion],
    "Media Total (M)": [round((a + c + cv) / 1e6, 2) for a, c, cv in zip(all_awareness, all_consid, all_conversion)],
    "Total (M)": [round(v / 1e6, 2) for v in all_actual],
    "Media %": [round((a + c + cv) / t * 100, 1) for a, c, cv, t in zip(all_awareness, all_consid, all_conversion, all_actual)],
})
print(f"{'═'*60}\n  EXH Attendance Decomposition by Funnel (2023-2026F)\n{'═'*60}")
display(decomp_annual_table)

# COMMAND ----------

# ━━ B10: EXH DIMINISHING RETURN CURVE (YoY) — carryover-simulation based ──────────

scale_factor = float(scaler_y.scale_[0])

def carryover_media_attendance(annual_budget):
    """Replay 2025 weekly pattern scaled to annual_budget, run through the model's
    carryover + saturation, return total attendance for that budget year."""
    scaled = wk_2025 * (annual_budget / spend_2025)
    run = pd.concat([hist[ACTIVE_COMBOS], scaled], ignore_index=True)
    wk_local = media_by_funnel(run)
    s_local = slice(len(hist), len(hist) + n25)   # the scaled replay, not the embedded 2025 actuals
    return sum(wk_local[f][s_local].sum() for f in FUNNELS)

# X axis: near-zero to $20M (100 points — carryover sim is per-point)
x_annual = np.linspace(52 * 100, 20_000_000, 100)
curve = np.array([carryover_media_attendance(b) for b in x_annual])

# ── Year reference points: spend from actuals, attendance via 2025-mix curve model ──
year_points = {}
for i, year in enumerate(years_avail):  # 2023, 2024, 2025
    df_yr = df_full[df_full["year"] == year]
    yr_spend = {c: float(df_yr[c].sum()) for c in ACTIVE_COMBOS}
    yr_active_spend = sum(yr_spend.values())
    _sparse = [c for c in ["Awareness__Other", "Consideration__Other", "Conversion__Other"] if c in df_yr.columns]
    yr_total_spend = yr_active_spend + float(df_yr[_sparse].sum().sum()) if _sparse else yr_active_spend
    year_points[year] = {"spend": yr_total_spend, "attendance": carryover_media_attendance(yr_total_spend)}

year_points["2026 Budget"] = {"spend": BUDGET_2026, "attendance": media_2026_total}
year_points["2027 Budget"] = {"spend": BUDGET_2027, "attendance": media_2027_total}

YEAR_MARKER_STYLES = {
    2023: {"color": CPX_DGREY, "marker": "o", "linestyle": "--",  "linewidth": 1.2},
    2024: {"color": CPX_NAVY,  "marker": "s", "linestyle": "-.", "linewidth": 1.2},
    2025: {"color": CPX_RED,   "marker": "D", "linestyle": "-",  "linewidth": 1.5},
    "2026 Budget": {"color": "#2E8B57", "marker": "^", "linestyle": ":", "linewidth": 1.5},
    "2027 Budget": {"color": "#CC5500", "marker": "P", "linestyle": ":", "linewidth": 1.5},
}

for yr in ["2026 Budget", "2027 Budget"]:
    print(f"{yr}: ${year_points[yr]['spend']/1e6:.2f}M -> {year_points[yr]['attendance']/1e6:.2f}M")

# Response Curve 
fig, ax = plt.subplots(figsize=(16, 9))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")
ax.plot(x_annual / 1e6, curve / 1e6, color=CPX_NAVY, linewidth=2.5)

# ── Marginal ROI = 1 Efficiency Breakeven ────────────────────────────────────────
marginal_roi = np.gradient(curve, x_annual)
roi_att_idx = np.argmin(np.abs(marginal_roi - 1.0))
roi_threshold_m = x_annual[roi_att_idx] / 1e6
roi_att_m = curve[roi_att_idx] / 1e6
ax.axvline(roi_threshold_m, color=CPX_GOLD, linewidth=1.8, linestyle="--", alpha=0.9, zorder=5)
ax.scatter([roi_threshold_m], [roi_att_m], color=CPX_GOLD, s=120, zorder=6, marker="*",
           edgecolors="white", linewidths=0.8)
ax.annotate(
    f"Efficiency Breakeven\nMarginal Return = \$1 per \$1\n\${roi_threshold_m:.1f}M / yr",
    xy=(roi_threshold_m, roi_att_m),
    xytext=(-12, 38), textcoords="offset points",
    fontsize=10, color=CPX_GOLD, fontweight="bold", ha="right",
    arrowprops=dict(arrowstyle="->", color=CPX_GOLD, lw=0.8),
    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.9, ec="none"),
)

# Fix axis limits BEFORE drawing marker lines, so we know the true bottom edge
ax.set_xlim(0, 20)
y_bottom = ax.get_ylim()[0]

for year, style in YEAR_MARKER_STYLES.items():
    pts = year_points[year]
    spend_m = pts["spend"] / 1e6
    att_m = pts["attendance"] / 1e6

    # Vertical line: from the true bottom of the axis up to the marker
    ax.plot([spend_m, spend_m], [y_bottom, att_m], color=style["color"], linestyle=style["linestyle"],
            linewidth=style["linewidth"], alpha=0.85, zorder=4)
    # Horizontal guideline: y-axis (x=0) to the marker's x position
    ax.plot([0, spend_m], [att_m, att_m], color=style["color"],
            linestyle=style["linestyle"], linewidth=style["linewidth"], alpha=0.6, zorder=3)
    ax.scatter([spend_m], [att_m], color=style["color"], marker=style["marker"],
              s=100, zorder=7, edgecolors="white", linewidths=1.0)


ax.set_title("EXH MMM — Total Media Response Curve\n"
             "Carryover simulation  |  Budget shares held at 2025 mix",
             fontsize=14, fontweight="bold", color="black", pad=16)
ax.set_xlabel("Annual Media Budget ($M)", fontsize=14, fontweight="bold")
ax.set_ylabel("Annual Media Attendance Contribution (M)", fontsize=14, fontweight="bold")
ax.set_ylim(bottom=y_bottom)  # keep the same bottom so lines still reach it after further draws
ax.xaxis.set_major_locator(plt.MultipleLocator(2))
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:.0f}M"))
ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=18))
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1f}M"))
plt.setp(ax.get_xticklabels(), fontsize=12, fontweight="bold")
plt.setp(ax.get_yticklabels(), fontsize=12, fontweight="bold")
ax.grid(False)

legend_handles = [
    Line2D([0], [0], color=style["color"], marker=style["marker"], linestyle=style["linestyle"],
           linewidth=style["linewidth"], markersize=7,
           label=f"{year} (${year_points[year]['spend']/1e6:.1f}M annual)")
    for year, style in YEAR_MARKER_STYLES.items()
]
ax.legend(handles=legend_handles, loc="lower right", fontsize=13, frameon=True,
          facecolor="white", edgecolor="#CCCCCC", framealpha=0.95,
          markerscale=1.6, borderpad=1.0, labelspacing=0.8, handlelength=2.8,
          title="Annual Spend Reference", title_fontsize=12)

fig.text(0.02, 0.01, "* Curve and markers both use carryover-simulation media totals (matches decomposition table).",
         fontsize=7.5, color=CPX_DGREY, style="italic")
plt.tight_layout(rect=[0, 0.03, 1, 1])
save_fig(fig, "exh_b10_response_yoy.png")
display(fig); plt.close(fig)

# COMMAND ----------

# ━━ B11: EXH MARGINAL RETURN CURVE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

marginal_roi = np.gradient(curve, x_annual)

fig, ax = plt.subplots(figsize=(16, 9))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")
ax.plot(x_annual / 1e6, marginal_roi, color=CPX_NAVY, linewidth=2.5)

# Breakeven line at marginal return = 1
ax.axhline(1.0, color=CPX_GOLD, linewidth=1.8, linestyle="--", alpha=0.9, zorder=4)
roi_idx = np.argmin(np.abs(marginal_roi - 1.0))
roi_x_m = x_annual[roi_idx] / 1e6
ax.scatter([roi_x_m], [1.0], color=CPX_GOLD, s=120, zorder=6, marker="*",
           edgecolors="white", linewidths=0.8)
ax.annotate(
    f"Efficiency Breakeven\n\${roi_x_m:.1f}M / yr",
    xy=(roi_x_m, 1.0),
    xytext=(20, 40), textcoords="offset points",
    fontsize=10, color=CPX_GOLD, fontweight="bold", ha="left",
    arrowprops=dict(arrowstyle="->", color=CPX_GOLD, lw=0.8),
    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.9, ec="none"),
)

ax.set_title("EXH MMM — Marginal Return Curve\n"
             "Carryover simulation  |  Budget shares held at 2025 mix",
             fontsize=14, fontweight="bold", color="black", pad=16)
ax.set_xlabel("Annual Media Budget ($M)", fontsize=14, fontweight="bold")
ax.set_ylabel("Marginal Return ($ attendance per $ spend)", fontsize=14, fontweight="bold")
ax.set_xlim(0, 20)
ax.xaxis.set_major_locator(plt.MultipleLocator(2))
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:.0f}M"))
ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=12))
plt.setp(ax.get_xticklabels(), fontsize=12, fontweight="bold")
plt.setp(ax.get_yticklabels(), fontsize=12, fontweight="bold")
ax.grid(False)

fig.text(0.02, 0.01, "* Marginal return = d(attendance)/d(budget), numerically differentiated from the carryover-simulation response curve.",
         fontsize=7.5, color=CPX_DGREY, style="italic")
plt.tight_layout(rect=[0, 0.03, 1, 1])
save_fig(fig, "exh_b11_marginal_return.png")
display(fig); plt.close(fig)
