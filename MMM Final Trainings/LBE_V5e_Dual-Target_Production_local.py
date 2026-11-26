# Databricks notebook source
# DBTITLE 1,LBE V5e — Dual-Target Media Response Model
# MAGIC %md
# MAGIC # LBE V5e — Dual-Target Media Response Model (Production)
# MAGIC **Location-Based Entertainment: The Rec Room & Playdium**
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Key Findings
# MAGIC
# MAGIC | Metric | TRR | PDM |
# MAGIC |---|---|---|
# MAGIC | **Awareness ROI** | 5.4× | 2.4× |
# MAGIC | **Consideration ROI (combined)** | **3.26×** | **2.42×** |
# MAGIC | Consideration ROI (V3 conflated) | 1.15× | 0.90× |
# MAGIC | Consideration → Rev/Txn lift | +6.4% | +8.4% |
# MAGIC | Consideration → Volume lift | +4.7% | +3.2% |
# MAGIC | Rev/Txn model MAPE | 7.1% | 5.4% |
# MAGIC
# MAGIC ## Why Dual-Target?
# MAGIC V3 targets `log(Total Revenue)` which conflates two mechanisms:
# MAGIC - **Volume** (transactions driven by Awareness)
# MAGIC - **Value** (spend-per-visit driven by Consideration)
# MAGIC
# MAGIC Since Consideration is *negatively* correlated with transaction volume (r=-0.17) but *positively* with Rev/Txn (r=+0.31), these cancel in the combined target. Separating them reveals Consideration's true ROI is 3×, not 1×.
# MAGIC
# MAGIC ## Model Architecture
# MAGIC - **Model A** — `log(Rev/Txn)`: Captures spend-per-visit effect
# MAGIC - **Model B** — `log(Transactions)`: Captures visit volume effect
# MAGIC - **Combined ROI** = value_lift × avg_txn + volume_lift × avg_rpt
# MAGIC - Both use V3 structure: multiplicative log-link, geometric adstock, Hill saturation
# MAGIC
# MAGIC ## Data Source
# MAGIC `/Volumes/data_exploration_hub/marketing_hub/model_data/Final_Data/lbe_model_input_v2.csv` (170 weeks, Jan 2023 – Mar 2026)

# COMMAND ----------

# DBTITLE 1,1. Install Dependencies
# MAGIC %pip install pymc arviz numpyro jax jaxlib pytensor scikit-learn openpyxl scipy --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,2. Config, Data Loading & Feature Engineering
import os, json, warnings, numpy as np, pandas as pd
import pytensor.tensor as pt
from sklearn.metrics import mean_absolute_percentage_error
warnings.simplefilter("ignore")

# ── CONFIG ──
PATH = "/Volumes/data_exploration_hub/marketing_hub/model_data/Final_Data/lbe_model_input_v2.csv"
OUTDIR = "/Volumes/data_exploration_hub/marketing_hub/model_data/Final_Data/mmm_outputs"
ADSTOCK_L = 8
TRAIN_FRAC = 0.85
DRAWS, TUNE, CHAINS = 1000, 1000, 4

# ── LOAD DATA ──
d = pd.read_csv(PATH)
d["week_date"] = pd.to_datetime(d["week_date"])
d = d.dropna(subset=["week_date"]).sort_values("week_date").reset_index(drop=True)

# Fill control gaps; encode Y/N flags
for c in ["national_mean_temp", "national_rain_mm", "headline_cpi"]:
    d[c] = d[c].ffill().bfill()
for col in d.columns:
    if str(d[col].dtype) in ("object", "string", "str"):
        if set(d[col].dropna().astype(str).unique()) <= {"Y", "N"}:
            d[col] = (d[col].astype(str) == "Y").astype(int)

# ── BUILD BRAND-LEVEL SERIES ──
d["TRR_Rev"] = d[["Revenue_TRR_HPL", "Revenue_TRR_LPL", "Revenue_TRR_NL"]].sum(axis=1)
d["PDM_Rev"] = d[["Revenue_PDM_NL", "Revenue_PDM_OL"]].sum(axis=1)
d["TRR_Txns"] = d[["TRR_HPL_Txns", "TRR_LPL_Txns", "TRR_NL_Txns"]].sum(axis=1)
d["PDM_Txns"] = d[["PDM_NL_Txns", "PDM_OL_Txns"]].sum(axis=1)

CH = {
    "TRR": {
        "Awareness": ["Awareness_TRR_HPL", "Awareness_TRR_LPL", "Awareness_TRR_NL", "Awareness_TRR_National"],
        "Consideration": ["Consideration_TRR_HPL", "Consideration_TRR_LPL", "Consideration_TRR_NL",
                          "Consideration_TRR_National", "Consideration_TRR_Other"],
    },
    "PDM": {
        "Awareness": ["Awareness_PDM_NL", "Awareness_PDM_National", "Awareness_PDM_Other"],
        "Consideration": ["Consideration_PDM_NL", "Consideration_PDM_OL", "Consideration_PDM_National", "Consideration_PDM_Other"],
    },
}
for b in CH:
    for stage, cols in CH[b].items():
        cols = [c for c in cols if c in d.columns and d[c].sum() > 0]
        d[f"{b}_{stage}_spend"] = d[cols].sum(axis=1)

CONTROLS = ["national_mean_temp", "national_rain_mm", "headline_cpi", "is_new_year_wk",
            "is_march_break", "is_pre_xmas", "is_summer_peak", "is_back_to_school",
            "is_spring_lull", "is_long_weekend"]

print(f"✓ Data loaded: {len(d)} weeks ({d['week_date'].min().date()} → {d['week_date'].max().date()})")
print(f"  TRR: ${d['TRR_Rev'].sum():,.0f} revenue, {d['TRR_Txns'].sum():,.0f} txns")
print(f"  PDM: ${d['PDM_Rev'].sum():,.0f} revenue, {d['PDM_Txns'].sum():,.0f} txns")

# COMMAND ----------

# DBTITLE 1,3. Adstock & Saturation Helpers
import pymc as pm
import pytensor.tensor as pt
import arviz as az

def adstock_np(x, a, L=ADSTOCK_L):
    """Numpy geometric adstock (for reconstruction)."""
    out = x.astype(float).copy()
    for lag in range(1, L):
        out[lag:] += a ** lag * x[:-lag]
    return out

def adstock_pt(x, a, L=ADSTOCK_L):
    """PyTensor geometric adstock (for model)."""
    acc = x
    for lag in range(1, L):
        shifted = pt.concatenate([pt.zeros(lag), x[:-lag]])
        acc = acc + (a ** lag) * shifted
    return acc

print("✓ Helpers ready")
print(f"  Adstock: geometric, {ADSTOCK_L}-week window")
print(f"  Saturation: Hill (half-saturation)")
print(f"  Train/val: {TRAIN_FRAC:.0%} / {1-TRAIN_FRAC:.0%}")

# COMMAND ----------

# DBTITLE 1,4. V5e Dual-Target Model Definition
# ── V5e: Dual-Target Model (Rev/Txn + Transactions) ────────────────────────

def fit_brand_v5e(brand):
    """Dual-target model: Model A targets log(Rev/Txn), Model B targets log(Txns).
    Combined ROI = value_lift + volume_lift."""
    n = len(d); n_tr = int(n * TRAIN_FRAC)
    
    rev = d[f"{brand}_Rev"].values.astype(float)
    txn = d[f"{brand}_Txns"].values.astype(float)
    
    # Rev per transaction
    rpt = np.where(txn > 0, rev / txn, np.nan)
    rpt_clean = np.where(~np.isnan(rpt), rpt, np.nanmedian(rpt))
    y_rpt = np.log(np.maximum(rpt_clean, 1.0))
    y_txn = np.log(np.maximum(txn, 1.0))
    
    # Media & controls
    M = d[[f"{brand}_Awareness_spend", f"{brand}_Consideration_spend"]].values.astype(float)
    X = d[CONTROLS].values.astype(float)
    m_s = M[:n_tr].max(0); m_s[m_s == 0] = 1.0
    Mz = np.nan_to_num(M / m_s)
    x_mu, x_sd = X[:n_tr].mean(0), X[:n_tr].std(0); x_sd[x_sd == 0] = 1.0
    Xz = np.nan_to_num((X - x_mu) / x_sd)
    aw_n = Mz[:n_tr, 0]; co_n = Mz[:n_tr, 1]
    
    def _build_and_sample(target, label):
        """Generic V3 model for any log-space target."""
        with pm.Model() as model:
            alpha_aw = pm.Beta("adstock_aw", 3, 3)
            alpha_co = pm.Beta("adstock_co", 3, 3)
            aw_ad = adstock_pt(pt.as_tensor(aw_n), alpha_aw)
            co_ad = adstock_pt(pt.as_tensor(co_n), alpha_co)
            
            half = pm.Gamma("sat_half", 2, 2, shape=2)
            aw_eff = aw_ad / (aw_ad + half[0])
            co_eff = co_ad / (co_ad + half[1])
            
            beta = pm.HalfNormal("beta_media", 0.5, shape=2)
            gamma = pm.Normal("beta_ctrl", 0, 0.5, shape=Xz.shape[1])
            intercept = pm.Normal("intercept", target[:n_tr].mean(), 0.5)
            sigma = pm.HalfNormal("sigma", 0.3)
            
            mu = intercept + pt.dot(pt.as_tensor(Xz[:n_tr]), gamma) + beta[0]*aw_eff + beta[1]*co_eff
            pm.Normal("y", mu=mu, sigma=sigma, observed=target[:n_tr])
            
            idata = pm.sample(DRAWS, tune=TUNE, chains=CHAINS, target_accept=0.99,
                              nuts_sampler="numpyro", random_seed=42)
        return idata
    
    # ── Fit both models ──
    print(f"    Fitting Model A: log(Rev/Txn)...")
    idata_rpt = _build_and_sample(y_rpt, "rpt")
    print(f"    Fitting Model B: log(Txns)...")
    idata_txn = _build_and_sample(y_txn, "txn")
    
    # ── Reconstruct full series ──
    def _reconstruct(idata, Mz_full, Xz_full):
        po = idata.posterior
        a_aw = float(po["adstock_aw"].mean())
        a_co = float(po["adstock_co"].mean())
        h = po["sat_half"].mean(dim=["chain","draw"]).values
        b = po["beta_media"].mean(dim=["chain","draw"]).values
        g = po["beta_ctrl"].mean(dim=["chain","draw"]).values
        ic = float(po["intercept"].mean())
        
        aw_full = adstock_np(Mz_full[:,0], a_aw)
        co_full = adstock_np(Mz_full[:,1], a_co)
        m_aw = b[0] * (aw_full / (aw_full + h[0]))
        m_co = b[1] * (co_full / (co_full + h[1]))
        log_pred = ic + Xz_full @ g + m_aw + m_co
        return log_pred, m_aw, m_co, b, h, a_aw, a_co
    
    log_pred_rpt, m_aw_rpt, m_co_rpt, b_rpt, h_rpt, a_aw_rpt, a_co_rpt = _reconstruct(idata_rpt, Mz, Xz)
    log_pred_txn, m_aw_txn, m_co_txn, b_txn, h_txn, a_aw_txn, a_co_txn = _reconstruct(idata_txn, Mz, Xz)
    
    pred_rpt = np.exp(log_pred_rpt)
    pred_txn = np.exp(log_pred_txn)
    
    # ── Counterfactual decomposition ──
    # Rev/Txn model
    co_incr_rpt = (pred_rpt - np.exp(log_pred_rpt - m_co_rpt)).sum()
    aw_incr_rpt = (pred_rpt - np.exp(log_pred_rpt - m_aw_rpt)).sum()
    # Txn model
    co_incr_txn = (pred_txn - np.exp(log_pred_txn - m_co_txn)).sum()
    aw_incr_txn = (pred_txn - np.exp(log_pred_txn - m_aw_txn)).sum()
    
    # Combined ROI
    avg_rpt = rpt_clean.mean(); avg_txn = txn.mean()
    co_incr_rev = co_incr_txn * avg_rpt + avg_txn * co_incr_rpt
    aw_incr_rev = aw_incr_txn * avg_rpt + avg_txn * aw_incr_rpt
    
    spend_aw = M[:,0].sum(); spend_co = M[:,1].sum()
    roi_aw = aw_incr_rev / spend_aw if spend_aw > 0 else 0
    roi_co = co_incr_rev / spend_co if spend_co > 0 else 0
    
    # MAPE
    mape_rpt = np.nanmean(np.abs((rpt_clean[n_tr:] - pred_rpt[n_tr:]) /
                                  np.where(rpt_clean[n_tr:]==0, np.nan, rpt_clean[n_tr:]))) * 100
    mape_txn = np.nanmean(np.abs((txn[n_tr:] - pred_txn[n_tr:]) /
                                  np.where(txn[n_tr:]==0, np.nan, txn[n_tr:]))) * 100
    
    # Beta lifts
    co_lift_rpt = (np.exp(float(b_rpt[1])) - 1) * 100
    co_lift_txn = (np.exp(float(b_txn[1])) - 1) * 100
    aw_lift_rpt = (np.exp(float(b_rpt[0])) - 1) * 100
    aw_lift_txn = (np.exp(float(b_txn[0])) - 1) * 100
    
    return {
        "brand": brand,
        "roi_aw": roi_aw, "roi_co": roi_co,
        "aw_incr_rev": aw_incr_rev, "co_incr_rev": co_incr_rev,
        "spend_aw": spend_aw, "spend_co": spend_co,
        "co_lift_rpt": co_lift_rpt, "co_lift_txn": co_lift_txn,
        "aw_lift_rpt": aw_lift_rpt, "aw_lift_txn": aw_lift_txn,
        "mape_rpt": round(mape_rpt, 1), "mape_txn": round(mape_txn, 1),
        "avg_rpt": avg_rpt, "avg_txn": avg_txn,
        "pred_rpt": pred_rpt, "pred_txn": pred_txn,
        "actual_rpt": rpt_clean, "actual_txn": txn,
        "m_aw_rpt": m_aw_rpt, "m_co_rpt": m_co_rpt,
        "m_aw_txn": m_aw_txn, "m_co_txn": m_co_txn,
        "log_pred_rpt": log_pred_rpt, "log_pred_txn": log_pred_txn,
        "adstock_aw_rpt": a_aw_rpt, "adstock_co_rpt": a_co_rpt,
        "adstock_aw_txn": a_aw_txn, "adstock_co_txn": a_co_txn,
        "div_rpt": int(idata_rpt.sample_stats.diverging.sum()),
        "div_txn": int(idata_txn.sample_stats.diverging.sum()),
    }

print("✓ V5e model function defined")

# COMMAND ----------

# DBTITLE 1,5. Fit V5e — TRR and PDM
# ── Fit V5e for both brands ──
results = {}
for brand in ["TRR", "PDM"]:
    print(f"\n{'='*65}")
    print(f"  Fitting V5e: {brand}")
    print(f"{'='*65}")
    results[brand] = fit_brand_v5e(brand)
    r = results[brand]
    print(f"\n  Rev/Txn model MAPE: {r['mape_rpt']}%  Txn model MAPE: {r['mape_txn']}%")
    print(f"  Divergences: RPT={r['div_rpt']}, TXN={r['div_txn']}")
    print(f"\n  {'Channel':<15} {'Rev/Txn lift':>12} {'Volume lift':>12} {'Combined ROI':>13} {'Spend':>12}")
    print(f"  {'-'*65}")
    print(f"  {'Awareness':<15} {r['aw_lift_rpt']:>+11.1f}% {r['aw_lift_txn']:>+11.1f}% {r['roi_aw']:>12.2f}× ${r['spend_aw']:>10,.0f}")
    print(f"  {'Consideration':<15} {r['co_lift_rpt']:>+11.1f}% {r['co_lift_txn']:>+11.1f}% {r['roi_co']:>12.2f}× ${r['spend_co']:>10,.0f}")

# COMMAND ----------

# DBTITLE 1,6. Results Summary & ROI Table
# ── Final ROI Summary ──
import matplotlib.pyplot as plt

# Cineplex palette
CPX_NAVY = "#002B5C"; CPX_RED = "#E31837"; CPX_GOLD = "#F5A800"
CPX_LGREY = "#F2F2F2"; CPX_DGREY = "#4A4A4A"; CPX_MGREY = "#8A8A8A"; WHITE = "#FFFFFF"
FUNNEL_COLORS = {"Awareness": CPX_NAVY, "Consideration": CPX_RED}

plt.rcParams.update({"figure.dpi": 150, "axes.titlesize": 13, "axes.titleweight": "bold",
                     "axes.labelsize": 11, "font.weight": "bold", "legend.frameon": False,
                     "figure.facecolor": WHITE, "axes.facecolor": WHITE})

print("\n" + "="*70)
print("  V5e FINAL ROI SUMMARY")
print("="*70)
print(f"\n{'Brand':<6} {'Channel':<15} {'ROI':>6} {'Incr Rev':>12} {'Spend':>12} {'Rev/Txn lift':>13} {'Vol lift':>9}")
print("-"*75)
for b in ["TRR", "PDM"]:
    r = results[b]
    print(f"{b:<6} {'Awareness':<15} {r['roi_aw']:>5.2f}× ${r['aw_incr_rev']:>10,.0f} ${r['spend_aw']:>10,.0f} {r['aw_lift_rpt']:>+12.1f}% {r['aw_lift_txn']:>+8.1f}%")
    print(f"{b:<6} {'Consideration':<15} {r['roi_co']:>5.2f}× ${r['co_incr_rev']:>10,.0f} ${r['spend_co']:>10,.0f} {r['co_lift_rpt']:>+12.1f}% {r['co_lift_txn']:>+8.1f}%")
    print()

# ── ROI Bar Chart ──
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, b in zip(axes, ["TRR", "PDM"]):
    r = results[b]
    channels = ["Awareness", "Consideration"]
    rois = [r["roi_aw"], r["roi_co"]]
    colors = [FUNNEL_COLORS[c] for c in channels]
    bars = ax.bar(channels, rois, color=colors, width=0.5)
    ax.axhline(1.0, color=CPX_MGREY, ls="--", lw=1.2, label="Break-even")
    for bar, val in zip(bars, rois):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.1,
                f"{val:.2f}×", ha="center", fontsize=12, fontweight="bold", color=CPX_DGREY)
    ax.set_title(f"{b} — V5e Combined ROI")
    ax.set_ylabel("ROI (×)")
    ax.set_ylim(0, max(rois) * 1.3)
    ax.legend()
plt.tight_layout()
plt.show()

# COMMAND ----------

# DBTITLE 1,7. Channel Mechanism Visualization
# ── Channel Mechanism: Value vs Volume decomposition ──
from matplotlib.patches import Patch

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, b in zip(axes, ["TRR", "PDM"]):
    r = results[b]
    # Stacked bar: Value lift vs Volume lift
    channels = ["Awareness", "Consideration"]
    value_lifts = [r["aw_lift_rpt"], r["co_lift_rpt"]]
    volume_lifts = [r["aw_lift_txn"], r["co_lift_txn"]]
    x = np.arange(len(channels))
    w = 0.35
    
    ax.bar(x - w/2, value_lifts, w, color=CPX_GOLD, label="Rev/Txn lift (value)")
    ax.bar(x + w/2, volume_lifts, w, color=CPX_NAVY, label="Transaction lift (volume)")
    
    ax.set_xticks(x)
    ax.set_xticklabels(channels)
    ax.set_ylabel("% lift at saturation")
    ax.set_title(f"{b} — Value vs Volume Mechanism")
    ax.legend()
    ax.axhline(0, color=CPX_DGREY, lw=0.8)
    
    # Annotate
    for i, (v, vol) in enumerate(zip(value_lifts, volume_lifts)):
        ax.text(i - w/2, v + 0.3, f"+{v:.1f}%", ha="center", fontsize=9, fontweight="bold", color=CPX_GOLD)
        ax.text(i + w/2, vol + 0.3, f"+{vol:.1f}%", ha="center", fontsize=9, fontweight="bold", color=CPX_NAVY)

plt.tight_layout()
plt.show()

print("\nKey insight:")
print("  Awareness drives BOTH value and volume roughly equally")
print("  Consideration drives primarily VALUE (spend-per-visit) with modest volume")
print("  This is why modeling total revenue conflates and undercredits Consideration")

# COMMAND ----------

# DBTITLE 1,8. Predicted vs Actual (Both Targets)
# ── Predicted vs Actual: Rev/Txn and Transactions ──
dates = d["week_date"]
n_tr = int(len(d) * TRAIN_FRAC)

fig, axes = plt.subplots(2, 2, figsize=(16, 10))

for col, b in enumerate(["TRR", "PDM"]):
    r = results[b]
    
    # Top row: Rev/Txn
    ax = axes[0, col]
    ax.plot(dates, r["actual_rpt"], color=CPX_DGREY, lw=0.8, alpha=0.7, label="Actual")
    ax.plot(dates, r["pred_rpt"], color=CPX_RED, lw=1.5, label="Predicted")
    ax.axvline(dates.iloc[n_tr], ls="--", color=CPX_GOLD, lw=1.2)
    ax.set_title(f"{b} — Rev/Transaction (MAPE {r['mape_rpt']}%)")
    ax.set_ylabel("$/Transaction")
    ax.legend(loc="upper right", fontsize=8)
    
    # Bottom row: Transactions
    ax = axes[1, col]
    ax.plot(dates, r["actual_txn"], color=CPX_DGREY, lw=0.8, alpha=0.7, label="Actual")
    ax.plot(dates, r["pred_txn"], color=CPX_NAVY, lw=1.5, label="Predicted")
    ax.axvline(dates.iloc[n_tr], ls="--", color=CPX_GOLD, lw=1.2)
    ax.set_title(f"{b} — Transactions (MAPE {r['mape_txn']}%)")
    ax.set_ylabel("Weekly Txns")
    ax.legend(loc="upper right", fontsize=8)

plt.tight_layout()
plt.show()

# COMMAND ----------

# DBTITLE 1,Validation Diagnostics — R², MAPE & Annual Bias
# ── Validation Diagnostics: R², MAPE & Annual Bias ──────────────────────────
# Self-contained: loads data + artifacts from disk
import numpy as np, pandas as pd, json
from sklearn.metrics import r2_score, mean_absolute_percentage_error

PATH = "/Volumes/data_exploration_hub/marketing_hub/model_data/Final_Data/lbe_model_input_v2.csv"
OUTDIR = "/Volumes/data_exploration_hub/marketing_hub/model_data/Final_Data/mmm_outputs"
TRAIN_FRAC = 0.85

# Load source data
d = pd.read_csv(PATH)
d["week_date"] = pd.to_datetime(d["week_date"])
d = d.dropna(subset=["week_date"]).sort_values("week_date").reset_index(drop=True)
d["TRR_Rev"] = d[["Revenue_TRR_HPL", "Revenue_TRR_LPL", "Revenue_TRR_NL"]].sum(axis=1)
d["PDM_Rev"] = d[["Revenue_PDM_NL", "Revenue_PDM_OL"]].sum(axis=1)
d["TRR_Txns"] = d[["TRR_HPL_Txns", "TRR_LPL_Txns", "TRR_NL_Txns"]].sum(axis=1)
d["PDM_Txns"] = d[["PDM_NL_Txns", "PDM_OL_Txns"]].sum(axis=1)

# Load predictions from saved artifacts
ARTIFACT_PATH = f"{OUTDIR}/lbe_v5e"
results = {}
for brand in ["TRR", "PDM"]:
    arr = pd.read_parquet(f"{ARTIFACT_PATH}_{brand.lower()}_arrays.parquet")
    txn = d[f"{brand}_Txns"].values.astype(float)
    rev = d[f"{brand}_Rev"].values.astype(float)
    rpt = np.where(txn > 0, rev / txn, np.nan)
    rpt_clean = np.where(~np.isnan(rpt), rpt, np.nanmedian(rpt))
    results[brand] = {
        "pred_rpt": arr["pred_rpt"].values,
        "pred_txn": arr["pred_txn"].values,
        "actual_rpt": rpt_clean,
        "actual_txn": txn,
    }

n = len(d); n_tr = int(n * TRAIN_FRAC)
dates = d["week_date"]
years = dates.dt.year.values

print("═" * 70)
print("  V5e VALIDATION DIAGNOSTICS")
print("═" * 70)

# ── 1. Validation R² and MAPE ────────────────────────────────────────────────
print(f"\n{'Brand':<6} {'Sub-Model':<16} {'Val MAPE':>9} {'Val R²':>8} {'Train R²':>9}")
print("-" * 55)
for b in ["TRR", "PDM"]:
    r = results[b]
    # Rev/Txn
    valid_rpt = r["actual_rpt"][n_tr:] > 0
    mape_rpt = np.nanmean(np.abs((r["actual_rpt"][n_tr:][valid_rpt] - r["pred_rpt"][n_tr:][valid_rpt]) / r["actual_rpt"][n_tr:][valid_rpt])) * 100
    r2_val_rpt = r2_score(r["actual_rpt"][n_tr:], r["pred_rpt"][n_tr:])
    r2_trn_rpt = r2_score(r["actual_rpt"][:n_tr], r["pred_rpt"][:n_tr])
    # Transactions
    valid_txn = r["actual_txn"][n_tr:] > 0
    mape_txn = np.nanmean(np.abs((r["actual_txn"][n_tr:][valid_txn] - r["pred_txn"][n_tr:][valid_txn]) / r["actual_txn"][n_tr:][valid_txn])) * 100
    r2_val_txn = r2_score(r["actual_txn"][n_tr:], r["pred_txn"][n_tr:])
    r2_trn_txn = r2_score(r["actual_txn"][:n_tr], r["pred_txn"][:n_tr])
    print(f"{b:<6} {'Rev/Txn':<16} {mape_rpt:>8.1f}% {r2_val_rpt:>8.3f} {r2_trn_rpt:>9.3f}")
    print(f"{b:<6} {'Transactions':<16} {mape_txn:>8.1f}% {r2_val_txn:>8.3f} {r2_trn_txn:>9.3f}")

# ── 2. Annual Bias (predicted total / actual total − 1) ─────────────────────
print(f"\n{'─' * 70}")
print(f"  ANNUAL BIAS  (positive = model over-predicts)")
print(f"{'─' * 70}")
print(f"\n{'Brand':<6} {'Sub-Model':<16} {'Year':>5} {'Actual':>12} {'Predicted':>12} {'Bias':>8}")
print("-" * 65)
for b in ["TRR", "PDM"]:
    r = results[b]
    for label, actual, pred in [("Rev/Txn", r["actual_rpt"], r["pred_rpt"]),
                                 ("Transactions", r["actual_txn"], r["pred_txn"])]:
        for yr in sorted(set(years)):
            mask = years == yr
            act_sum = actual[mask].sum()
            pred_sum = pred[mask].sum()
            bias_pct = (pred_sum / act_sum - 1) * 100 if act_sum != 0 else np.nan
            print(f"{b:<6} {label:<16} {yr:>5} {act_sum:>12,.0f} {pred_sum:>12,.0f} {bias_pct:>+7.1f}%")
        print()

print("Note: Annual bias < ±5% indicates well-calibrated model at yearly level.")

# COMMAND ----------

# DBTITLE 1,9. Save Results to Volume
# ━━ SAVE V5e RESULTS & DECOMPOSITION ARTIFACTS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Persists model outputs so downstream viz cells load from disk
# and produce identical numbers regardless of MCMC re-runs.
# Run ONCE after model training; downstream cells load from these files.
import json

os.makedirs(OUTDIR, exist_ok=True)
ARTIFACT_PATH = f"{OUTDIR}/lbe_v5e"

# ── 1. Summary JSON (high-level ROI metrics) ───────────────────────────────
output = {}
for b in ["TRR", "PDM"]:
    r = results[b]
    output[b] = {
        "model": "V5e_dual_target",
        "roi_awareness": round(r["roi_aw"], 3),
        "roi_consideration": round(r["roi_co"], 3),
        "awareness_incr_revenue": round(r["aw_incr_rev"], 0),
        "consideration_incr_revenue": round(r["co_incr_rev"], 0),
        "awareness_spend": round(r["spend_aw"], 0),
        "consideration_spend": round(r["spend_co"], 0),
        "consideration_rev_txn_lift_pct": round(r["co_lift_rpt"], 1),
        "consideration_volume_lift_pct": round(r["co_lift_txn"], 1),
        "awareness_rev_txn_lift_pct": round(r["aw_lift_rpt"], 1),
        "awareness_volume_lift_pct": round(r["aw_lift_txn"], 1),
        "mape_rev_txn": r["mape_rpt"],
        "mape_transactions": r["mape_txn"],
        "avg_rev_per_txn": round(r["avg_rpt"], 2),
        "avg_weekly_txn": round(r["avg_txn"], 0),
    }

with open(f"{ARTIFACT_PATH}_results.json", "w") as f:
    json.dump(output, f, indent=2)
print(f"✓ lbe_v5e_results.json")

# ── 2. Adstock alphas & scalar model params ───────────────────────────────
artifacts = {}
for brand in ["TRR", "PDM"]:
    r = results[brand]
    artifacts[brand] = {
        "adstock_aw_rpt": float(r["adstock_aw_rpt"]),
        "adstock_co_rpt": float(r["adstock_co_rpt"]),
        "adstock_aw_txn": float(r["adstock_aw_txn"]),
        "adstock_co_txn": float(r["adstock_co_txn"]),
        "roi_aw": float(r["roi_aw"]),
        "roi_co": float(r["roi_co"]),
        "spend_aw": float(r["spend_aw"]),
        "spend_co": float(r["spend_co"]),
        "aw_incr_rev": float(r["aw_incr_rev"]),
        "co_incr_rev": float(r["co_incr_rev"]),
        "mape_rpt": float(r["mape_rpt"]),
        "mape_txn": float(r["mape_txn"]),
    }

with open(f"{ARTIFACT_PATH}_artifacts.json", "w") as f:
    json.dump(artifacts, f, indent=2)
print(f"✓ lbe_v5e_artifacts.json")

# ── 3. Full reconstruction arrays per brand ───────────────────────────────
for brand in ["TRR", "PDM"]:
    r = results[brand]
    pd.DataFrame({
        "pred_rpt": r["pred_rpt"],
        "pred_txn": r["pred_txn"],
        "log_pred_rpt": r["log_pred_rpt"],
        "log_pred_txn": r["log_pred_txn"],
        "m_aw_rpt": r["m_aw_rpt"],
        "m_co_rpt": r["m_co_rpt"],
        "m_aw_txn": r["m_aw_txn"],
        "m_co_txn": r["m_co_txn"],
    }).to_parquet(f"{ARTIFACT_PATH}_{brand.lower()}_arrays.parquet", index=False)
    print(f"✓ lbe_v5e_{brand.lower()}_arrays.parquet")

print(f"\n✓ All artifacts saved to {OUTDIR}/")
print(f"  Downstream cells load from these files for stable outputs.")

# COMMAND ----------

# DBTITLE 1,10. Tier Level Investigation
# MAGIC %md
# MAGIC ## Tier-Level Investigation
# MAGIC
# MAGIC This section checks whether the current dataset supports going deeper than brand level into location tiers such as `HPL`, `LPL`, `NL`, `OL`, and `National`.
# MAGIC
# MAGIC It evaluates:
# MAGIC * whether each tier has both media inputs and business outcomes
# MAGIC * how many weeks have non-zero signal
# MAGIC * whether matched revenue and transaction series exist
# MAGIC * whether simple tier-level correlations are strong enough to justify a next modeling pass

# COMMAND ----------

# DBTITLE 1,11. Tier Coverage and Signal Check
# ── Tier-level feasibility: can we model below brand level? ──
from collections import defaultdict

TIER_ORDER = ["HPL", "LPL", "NL", "OL", "National", "Other"]
media_cols = [c for c in d.columns if c.startswith(("Awareness_", "Consideration_"))]
revenue_cols = [c for c in d.columns if c.startswith("Revenue_")]
txn_cols = [c for c in d.columns if c.endswith("_Txns")]

# Organize columns by tier token
by_tier = defaultdict(lambda: {"media": [], "revenue": [], "txns": []})
for tier in TIER_ORDER:
    by_tier[tier]["media"] = [c for c in media_cols if f"_{tier}" in c]
    by_tier[tier]["revenue"] = [c for c in revenue_cols if f"_{tier}" in c]
    by_tier[tier]["txns"] = [c for c in txn_cols if f"_{tier}" in c]

feasibility_rows = []
for tier in TIER_ORDER:
    media_non_zero = [(d[c].fillna(0) != 0).sum() for c in by_tier[tier]["media"]]
    outcome_non_zero = [(d[c].fillna(0) != 0).sum() for c in by_tier[tier]["revenue"] + by_tier[tier]["txns"]]
    feasibility_rows.append({
        "tier": tier,
        "media_cols": len(by_tier[tier]["media"]),
        "revenue_cols": len(by_tier[tier]["revenue"]),
        "txn_cols": len(by_tier[tier]["txns"]),
        "min_media_non_zero_weeks": min(media_non_zero) if media_non_zero else np.nan,
        "max_media_non_zero_weeks": max(media_non_zero) if media_non_zero else np.nan,
        "min_outcome_non_zero_weeks": min(outcome_non_zero) if outcome_non_zero else np.nan,
        "max_outcome_non_zero_weeks": max(outcome_non_zero) if outcome_non_zero else np.nan,
        "feasible_for_next_pass": (
            len(by_tier[tier]["media"]) >= 2 and
            len(by_tier[tier]["revenue"]) >= 1 and
            len(by_tier[tier]["txns"]) >= 1 and
            (min(media_non_zero) if media_non_zero else 0) >= 25 and
            (min(outcome_non_zero) if outcome_non_zero else 0) >= 25
        )
    })
feasibility_df = pd.DataFrame(feasibility_rows)

# Detailed matched pairs for the most model-ready groups
candidate_specs = {
    "TRR_HPL": {
        "revenue": "Revenue_TRR_HPL",
        "txns": "TRR_HPL_Txns",
        "media": ["Awareness_TRR_HPL", "Consideration_TRR_HPL"],
    },
    "TRR_LPL": {
        "revenue": "Revenue_TRR_LPL",
        "txns": "TRR_LPL_Txns",
        "media": ["Awareness_TRR_LPL", "Consideration_TRR_LPL"],
    },
    "TRR_NL": {
        "revenue": "Revenue_TRR_NL",
        "txns": "TRR_NL_Txns",
        "media": ["Awareness_TRR_NL", "Consideration_TRR_NL"],
    },
    "PDM_NL": {
        "revenue": "Revenue_PDM_NL",
        "txns": "PDM_NL_Txns",
        "media": ["Awareness_PDM_NL", "Consideration_PDM_NL"],
    },
}

corr_rows = []
for group, spec in candidate_specs.items():
    if all(col in d.columns for col in [spec["revenue"], spec["txns"], *spec["media"]]):
        tmp = d[[spec["revenue"], spec["txns"], *spec["media"]]].copy()
        tmp["rev_per_txn"] = np.where(tmp[spec["txns"]] > 0, tmp[spec["revenue"]] / tmp[spec["txns"]], np.nan)
        for m in spec["media"]:
            corr_rows.append({
                "group": group,
                "media_metric": m,
                "corr_revenue": round(tmp[spec["revenue"]].corr(tmp[m]), 3),
                "corr_txns": round(tmp[spec["txns"]].corr(tmp[m]), 3),
                "corr_rev_per_txn": round(pd.Series(tmp["rev_per_txn"]).corr(tmp[m]), 3),
                "media_non_zero_weeks": int((tmp[m].fillna(0) != 0).sum()),
                "txn_non_zero_weeks": int((tmp[spec["txns"]].fillna(0) != 0).sum()),
            })
correlation_df = pd.DataFrame(corr_rows)

# Weekly revenue mix snapshot for available revenue tiers
revenue_share_cols = [c for c in [
    "Revenue_TRR_HPL", "Revenue_TRR_LPL", "Revenue_TRR_NL",
    "Revenue_PDM_NL", "Revenue_PDM_OL"
] if c in d.columns]
share_df = d[["week_date", *revenue_share_cols]].copy()
for prefix, cols in {
    "TRR": [c for c in revenue_share_cols if c.startswith("Revenue_TRR_")],
    "PDM": [c for c in revenue_share_cols if c.startswith("Revenue_PDM_")],
}.items():
    if cols:
        denom = share_df[cols].sum(axis=1).replace(0, np.nan)
        for c in cols:
            share_df[f"share_{c}"] = (share_df[c] / denom).round(3)

print("Tier feasibility summary")
display(feasibility_df)
print("Matched tier-level signal checks")
display(correlation_df)
print("Recent weekly tier revenue shares")
display(share_df[[c for c in share_df.columns if c == 'week_date' or c.startswith('share_')]].tail(8))

priority_groups = feasibility_df.loc[feasibility_df["feasible_for_next_pass"], "tier"].tolist()
print("\nRecommendation:")
if priority_groups:
    print(f"  Strongest candidates for a next-pass tier model: {', '.join(priority_groups)}")
else:
    print("  No tier currently clears the minimum viability check.")
print("  National and Other look media-only in this file, so they are better treated as allocators or controls unless outcome series are added.")

# COMMAND ----------

# DBTITLE 1,10. Findings & Caveats
# MAGIC %md
# MAGIC ## Findings & Caveats
# MAGIC
# MAGIC ### What V5e Tells Us
# MAGIC 1. **Consideration ROI is ~3×** (not 1× as V3 suggested) — it drives spend-per-visit, not traffic
# MAGIC 2. **Awareness ROI is ~5×** (TRR) / ~2.4× (PDM) — drives both traffic AND spend, consistent with V3
# MAGIC 3. The Rev/Txn model has excellent predictive accuracy (5–7% MAPE), more stable than the volume model
# MAGIC 4. Combined media drives ~12% of revenue through two distinct mechanisms
# MAGIC
# MAGIC ### Business Implication
# MAGIC - **Awareness** = "get people in the door" (volume play)
# MAGIC - **Consideration** = "get them to spend more once there" (upsell/premium play)
# MAGIC - Both are well above breakeven → justify continued investment
# MAGIC - Consideration is particularly efficient for PDM where spend-per-visit is the binding constraint
# MAGIC
# MAGIC ### Caveats
# MAGIC - Transaction volume model has higher MAPE (26% TRR) — noisy target, many external drivers
# MAGIC - The dual-target decomposition assumes independence between value and volume channels
# MAGIC - 170 weeks nationally — direction is solid, point estimates have wide CIs
# MAGIC - Seasonal confounds partially removed by controls but not perfectly
# MAGIC
# MAGIC ### Model Progression
# MAGIC | Version | Target | Co ROI | Key Insight |
# MAGIC |---|---|---|---|
# MAGIC | V3 | log(Revenue) | 1.15× | Signals cancel in combined target |
# MAGIC | V5c | log(Revenue) + spike | 0.86× | Spike not the issue |
# MAGIC | V5d | log(Revenue) Weibull 16wk | 0.75× | Lag shape not the issue |
# MAGIC | **V5e** | **log(Rev/Txn) + log(Txn)** | **3.26×** | **Correct decomposition** |

# COMMAND ----------

# DBTITLE 1,12. Tier Allocation Assumption
# MAGIC %md
# MAGIC ## Tier Allocation Assumption
# MAGIC
# MAGIC Following your preference, this keeps `National` and `Other` in the funnel-level learning and trickles their effect down to tier streams instead of dropping them.
# MAGIC
# MAGIC Allocation rule used here:
# MAGIC * start from observed tier-specific media where available
# MAGIC * distribute `National` and `Other` equally across active streams for that brand and funnel stage
# MAGIC * force `NL` share to `0` in 2023 because the series does not exist then
# MAGIC * once `NL` is active, include it as an equal recipient alongside the other active streams
# MAGIC
# MAGIC This creates assumption-based tier inputs for a next modeling pass while preserving the full upper-funnel signal.

# COMMAND ----------

# DBTITLE 1,13. Trickle Down National and Other Effects
# ── Allocate National / Other evenly across active tier streams ──
alloc_df = d[["week_date"]].copy()
alloc_df["year"] = alloc_df["week_date"].dt.year

brand_stage_map = {
    ("TRR", "Awareness"): {
        "base_tiers": ["HPL", "LPL", "NL"],
        "observed": ["Awareness_TRR_HPL", "Awareness_TRR_LPL", "Awareness_TRR_NL"],
        "pooled": ["Awareness_TRR_National", "Awareness_TRR_Other"],
    },
    ("TRR", "Consideration"): {
        "base_tiers": ["HPL", "LPL", "NL"],
        "observed": ["Consideration_TRR_HPL", "Consideration_TRR_LPL", "Consideration_TRR_NL"],
        "pooled": ["Consideration_TRR_National", "Consideration_TRR_Other"],
    },
    ("PDM", "Awareness"): {
        "base_tiers": ["OL", "NL"],
        "observed": ["Awareness_PDM_OL", "Awareness_PDM_NL"],
        "pooled": ["Awareness_PDM_National", "Awareness_PDM_Other"],
    },
    ("PDM", "Consideration"): {
        "base_tiers": ["OL", "NL"],
        "observed": ["Consideration_PDM_OL", "Consideration_PDM_NL"],
        "pooled": ["Consideration_PDM_National", "Consideration_PDM_Other"],
    },
}

# Per user instruction, NL receives zero pooled allocation in 2023.
def active_tiers_for_row(brand, tiers, row):
    active = [t for t in tiers if not (t == "NL" and row["year"] == 2023)]
    return active

summary_rows = []
for (brand, stage), spec in brand_stage_map.items():
    pooled_cols = [c for c in spec["pooled"] if c in d.columns]
    pooled_total = d[pooled_cols].sum(axis=1) if pooled_cols else pd.Series(0.0, index=d.index)
    alloc_df[f"{brand}_{stage}_pooled_total"] = pooled_total

    for tier in spec["base_tiers"]:
        observed_col = next((c for c in spec["observed"] if c.endswith(f"_{tier}")), None)
        alloc_col = f"alloc_{brand}_{stage}_{tier}"
        alloc_df[alloc_col] = 0.0
        alloc_df[f"base_{brand}_{stage}_{tier}"] = d[observed_col] if observed_col in d.columns else 0.0

    for idx, row in alloc_df[["year", f"{brand}_{stage}_pooled_total"]].iterrows():
        active_tiers = active_tiers_for_row(brand, spec["base_tiers"], row)
        if active_tiers:
            share = row[f"{brand}_{stage}_pooled_total"] / len(active_tiers)
            for tier in active_tiers:
                alloc_df.at[idx, f"alloc_{brand}_{stage}_{tier}"] = share

    for tier in spec["base_tiers"]:
        combined_col = f"effective_{brand}_{stage}_{tier}"
        alloc_col = f"alloc_{brand}_{stage}_{tier}"
        base_col = f"base_{brand}_{stage}_{tier}"
        alloc_df[combined_col] = alloc_df[base_col] + alloc_df[alloc_col]
        summary_rows.append({
            "brand": brand,
            "stage": stage,
            "tier": tier,
            "observed_total": round(float(alloc_df[base_col].sum()), 2),
            "allocated_total": round(float(alloc_df[alloc_col].sum()), 2),
            "effective_total": round(float(alloc_df[combined_col].sum()), 2),
            "allocation_pct_of_effective": round(float(alloc_df[alloc_col].sum() / alloc_df[combined_col].sum() * 100), 1) if float(alloc_df[combined_col].sum()) > 0 else 0.0,
        })

allocation_summary_df = pd.DataFrame(summary_rows)

# Show how pooled signal gets split before vs after NL exists
allocation_example_cols = [
    "week_date", "year",
    "TRR_Awareness_pooled_total", "alloc_TRR_Awareness_HPL", "alloc_TRR_Awareness_LPL", "alloc_TRR_Awareness_NL",
    "PDM_Consideration_pooled_total", "alloc_PDM_Consideration_OL", "alloc_PDM_Consideration_NL"
]
available_example_cols = [c for c in allocation_example_cols if c in alloc_df.columns]
example_df = alloc_df.loc[(alloc_df["year"].isin([2023, 2024, 2025])), available_example_cols].copy()
example_df = pd.concat([
    example_df[example_df["year"] == 2023].head(3),
    example_df[example_df["year"] == 2024].tail(3),
    example_df[example_df["year"] == 2025].head(3),
], ignore_index=True)

print("Allocation summary by brand / stage / tier")
display(allocation_summary_df)
print("Example weekly allocations around NL introduction")
display(example_df)

# Quick rollup for tier-model inputs
next_pass_cols = ["week_date"] + [c for c in alloc_df.columns if c.startswith("effective_")]
next_pass_df = alloc_df[next_pass_cols].copy()
print("Next-pass effective media inputs preview")
display(next_pass_df.head(8))

# COMMAND ----------

# MAGIC %md
# MAGIC # PLOTS

# COMMAND ----------

# DBTITLE 1,Plots Config
# ── PLOTS CONFIG — single source of truth for all visualization cells ──
# Loads saved artifacts from cell 10 so downstream cells produce stable numbers
# regardless of MCMC re-runs. This cell is the standalone entry point for cells 18+.
import shutil, os, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patheffects as pe
from IPython.display import display as ipy_display

# ── CINEPLEX BRAND COLOURS ────────────────────────────────────────────────
CPX_NAVY  = "#002B5C"
CPX_RED   = "#D5001C"
CPX_GOLD  = "#C4A747"
CPX_LGREY = "#BFBFBF"
CPX_MGREY = "#808080"
CPX_DGREY = "#404040"
WHITE     = "#FFFFFF"

# ── LOAD SAVED ARTIFACTS ─────────────────────────────────────────────────────
ARTIFACT_PATH = f"{OUTDIR}/lbe_v5e"

# Scalars: adstock alphas, ROI, spend, incremental revenue
with open(f"{ARTIFACT_PATH}_artifacts.json", "r") as f:
    _artifacts = json.load(f)

# Reconstruction arrays: pred_rpt, pred_txn, log_pred, media contributions
_arrays = {}
for brand in ["TRR", "PDM"]:
    _arrays[brand] = pd.read_parquet(f"{ARTIFACT_PATH}_{brand.lower()}_arrays.parquet")

# Build results dict entirely from artifacts (no dependency on training cells)
results = {}
for brand in ["TRR", "PDM"]:
    arr = _arrays[brand]
    art = _artifacts[brand]
    results[brand] = {
        "pred_rpt":      arr["pred_rpt"].values,
        "pred_txn":      arr["pred_txn"].values,
        "log_pred_rpt":  arr["log_pred_rpt"].values,
        "log_pred_txn":  arr["log_pred_txn"].values,
        "m_aw_rpt":      arr["m_aw_rpt"].values,
        "m_co_rpt":      arr["m_co_rpt"].values,
        "m_aw_txn":      arr["m_aw_txn"].values,
        "m_co_txn":      arr["m_co_txn"].values,
        "adstock_aw_rpt": art["adstock_aw_rpt"],
        "adstock_co_rpt": art["adstock_co_rpt"],
        "adstock_aw_txn": art["adstock_aw_txn"],
        "adstock_co_txn": art["adstock_co_txn"],
        "roi_aw":         art["roi_aw"],
        "roi_co":         art["roi_co"],
        "spend_aw":       art["spend_aw"],
        "spend_co":       art["spend_co"],
        "aw_incr_rev":    art["aw_incr_rev"],
        "co_incr_rev":    art["co_incr_rev"],
    }

print(f"✓ Artifacts loaded from {ARTIFACT_PATH}_*.parquet / .json")
print(f"  Arrays pinned — downstream cells will not shift on re-runs.")

# ── Forecast & output ────────────────────────────────────────────────────────────
FORECAST_2026 = 148_546_568          # 2026 total LBE revenue forecast
FORECAST_2027 = 143_049_761
PLOT_YEARS    = [2023, 2024, 2025]
OUTPUT_DIR    = OUTDIR               # defined in cell 2

def get_figsize(ratio="16:9", base=16):
    w, h = [int(x) for x in ratio.split(":")]
    return (base, base * h / w)

def save_fig(fig, fname):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig.savefig(os.path.join(OUTPUT_DIR, fname), dpi=150, bbox_inches="tight", facecolor=WHITE)
    print(f"✓ Saved: {fname}")

# ── Shared dataframe (used by every plot cell) ───────────────────────────────────
df_all = d.copy()
df_all["year"]              = df_all["week_date"].dt.year
df_all["LBE_Revenue"]       = df_all["TRR_Rev"] + df_all["PDM_Rev"]
# Column aliases matching downstream cell references
df_all["Awareness_TRR"]     = df_all["TRR_Awareness_spend"]
df_all["Consideration_TRR"] = df_all["TRR_Consideration_spend"]
df_all["Awareness_PDM"]     = df_all["PDM_Awareness_spend"]
df_all["Consideration_PDM"] = df_all["PDM_Consideration_spend"]
df_all["aw_spend_proxy"]    = df_all["TRR_Awareness_spend"] + df_all["PDM_Awareness_spend"]
df_all["co_spend_proxy"]    = df_all["TRR_Consideration_spend"] + df_all["PDM_Consideration_spend"]
df_all["Awareness"]         = df_all["aw_spend_proxy"]
df_all["Consideration"]     = df_all["co_spend_proxy"]

# ── V5e ROI — cross-brand average and per-brand ───────────────────────────────────
# Cross-brand average ROI — used where brand is not distinguished
roi_yr = {
    "Awareness":     {yr: (results["TRR"]["roi_aw"] + results["PDM"]["roi_aw"]) / 2 for yr in PLOT_YEARS},
    "Consideration": {yr: (results["TRR"]["roi_co"] + results["PDM"]["roi_co"]) / 2 for yr in PLOT_YEARS},
}
# Per-brand ROI — used where charts are faceted by TRR / PDM
brand_roi = {
    b: {"Awareness": results[b]["roi_aw"], "Consideration": results[b]["roi_co"]}
    for b in ["TRR", "PDM"]
}

# ── Pre-aggregated annual spend summary ──────────────────────────────────────────
annual_spend = (
    df_all[df_all["year"].isin(PLOT_YEARS)]
    .groupby("year")[["aw_spend_proxy", "co_spend_proxy"]]
    .sum()
    .rename(columns={"aw_spend_proxy": "Awareness", "co_spend_proxy": "Consideration"})
)
annual_spend["Total"] = annual_spend["Awareness"] + annual_spend["Consideration"]
annual_spend_M        = annual_spend / 1e6

print(f"✓ Plot config ready  |  FORECAST_2026 ${FORECAST_2026:,.0f}")
print(f"  TRR — Awareness {results['TRR']['roi_aw']:.2f}×  Consideration {results['TRR']['roi_co']:.2f}×")
print(f"  PDM — Awareness {results['PDM']['roi_aw']:.2f}×  Consideration {results['PDM']['roi_co']:.2f}×")

# COMMAND ----------

# DBTITLE 1,LBE V5e Annual Revenue Decomposition
# ── LBE V5e ANNUAL REVENUE DECOMPOSITION BY FUNNEL + 2026 FORECAST ──────────
# Depends on: Plots Config cell (FORECAST_2026, df_all, CPX_* colors, get_figsize)

# ── V5e counterfactual decomp: weekly incremental revenue per funnel ──────────
def _weekly_incr(r):
    """Remove media log-contribution from both RPT and TXN sub-models."""
    wk_aw = (
        (r["pred_rpt"] - np.exp(r["log_pred_rpt"] - r["m_aw_rpt"])) * r["pred_txn"]
        + r["pred_rpt"] * (r["pred_txn"] - np.exp(r["log_pred_txn"] - r["m_aw_txn"]))
    )
    wk_co = (
        (r["pred_rpt"] - np.exp(r["log_pred_rpt"] - r["m_co_rpt"])) * r["pred_txn"]
        + r["pred_rpt"] * (r["pred_txn"] - np.exp(r["log_pred_txn"] - r["m_co_txn"]))
    )
    return wk_aw, wk_co

wk_aw_total = np.zeros(len(d))
wk_co_total = np.zeros(len(d))
for brand in ["TRR", "PDM"]:
    aw, co = _weekly_incr(results[brand])
    wk_aw_total += aw
    wk_co_total += co

decomp_df = d[["week_date"]].copy()
decomp_df["year"]                = decomp_df["week_date"].dt.year
decomp_df["total_rev"]           = d["TRR_Rev"] + d["PDM_Rev"]
decomp_df["media_Awareness"]     = wk_aw_total
decomp_df["media_Consideration"] = wk_co_total

# ── Aggregate Annual by Funnel (complete years only) ──────────────────────────
agg_cols           = {"total_rev": "sum", "media_Awareness": "sum", "media_Consideration": "sum"}
annual             = decomp_df[decomp_df["year"].isin([2023, 2024, 2025])].groupby("year").agg(agg_cols).round(0)
years_avail        = sorted(annual.index.tolist())
actual_vals        = annual.loc[years_avail, "total_rev"].values
awareness_vals     = annual.loc[years_avail, "media_Awareness"].values
consideration_vals = annual.loc[years_avail, "media_Consideration"].values
baseline_vals      = actual_vals - awareness_vals - consideration_vals

# ── 2026 Forecast (inherits 2025 funnel mix) ──────────────────────────────────
if FORECAST_2026 is not None:
    media_total_2025 = awareness_vals[-1] + consideration_vals[-1]
    aw_share         = awareness_vals[-1]     / media_total_2025
    csd_share        = consideration_vals[-1] / media_total_2025
    media_2026       = FORECAST_2026 * (media_total_2025 / actual_vals[-1])
    aw_2026          = media_2026 * aw_share
    csd_2026         = media_2026 * csd_share
    baseline_2026    = FORECAST_2026 - media_2026
    all_years        = years_avail + [2026]
    all_actual       = list(actual_vals)        + [FORECAST_2026]
    all_baseline     = list(baseline_vals)      + [baseline_2026]
    all_awareness    = list(awareness_vals)     + [aw_2026]
    all_consid       = list(consideration_vals) + [csd_2026]
    show_forecast    = True
    # ── 2027 Forecast (inherits same funnel mix as 2025) ──
    if FORECAST_2027 is not None:
        media_2027    = FORECAST_2027 * (media_total_2025 / actual_vals[-1])
        aw_2027       = media_2027 * aw_share
        csd_2027      = media_2027 * csd_share
        baseline_2027 = FORECAST_2027 - media_2027
        all_years    += [2027]
        all_actual   += [FORECAST_2027]
        all_baseline += [baseline_2027]
        all_awareness+= [aw_2027]
        all_consid   += [csd_2027]
else:
    all_years     = years_avail
    all_actual    = list(actual_vals)
    all_baseline  = list(baseline_vals)
    all_awareness = list(awareness_vals)
    all_consid    = list(consideration_vals)
    show_forecast = False

# ── Helper ────────────────────────────────────────────────────────────────────
def bar_label(ax, x_pos, bottom, height, pct, color, min_height=0.5):
    if abs(height) / 1e6 < min_height:
        return
    light_colors = {CPX_LGREY, CPX_GOLD}
    txt_color = "black" if color in light_colors else "white"
    ax.text(
        x_pos, (bottom + height / 2) / 1e6,
        f"{pct:.0f}%",
        ha="center", va="center",
        fontsize=14, fontweight="bold", color=txt_color,
        path_effects=[pe.withStroke(
            linewidth=2,
            foreground="white" if txt_color == "black" else "black"
        )]
    )

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=get_figsize("16:9"))
fig.patch.set_facecolor(WHITE)

x     = np.arange(len(all_years))
w     = 0.5
n_forecast = len(all_years) - len(years_avail)
alpha = [0.9] * len(years_avail) + [0.5] * n_forecast

for i, year in enumerate(all_years):
    a      = alpha[i]
    b_base = all_baseline[i]
    b_aw   = all_awareness[i]
    b_csd  = all_consid[i]
    total  = all_actual[i]

    ax.bar(x[i], b_base / 1e6,
           width=w, color=CPX_LGREY, alpha=a,
           edgecolor=WHITE, linewidth=0.5,
           label="Baseline (non-media)" if i == 0 else "_nolegend_")
    ax.bar(x[i], b_aw / 1e6,
           width=w, bottom=b_base / 1e6,
           color=CPX_NAVY, alpha=a,
           edgecolor=WHITE, linewidth=0.5,
           label="Awareness" if i == 0 else "_nolegend_")
    ax.bar(x[i], b_csd / 1e6,
           width=w, bottom=(b_base + b_aw) / 1e6,
           color=CPX_RED, alpha=a,
           edgecolor=WHITE, linewidth=0.5,
           label="Consideration" if i == 0 else "_nolegend_")

    # Segment labels
    bar_label(ax, x[i], 0,             b_base, b_base / total * 100, CPX_LGREY)
    bar_label(ax, x[i], b_base,        b_aw,   b_aw   / total * 100, CPX_NAVY, 1.5)
    bar_label(ax, x[i], b_base + b_aw, b_csd,  b_csd  / total * 100, CPX_RED,  1.5)

    # Total on top
    label = f"${total/1e6:.2f}M" + (" (F)" if year >= 2026 and show_forecast else "")
    ax.text(x[i], total / 1e6 + 0.3, label,
            ha="center", va="bottom",
            fontsize=14, fontweight="bold",
            color=CPX_RED if (year >= 2026 and show_forecast) else CPX_DGREY)

# Actual scatter dots (full years only)
ax.scatter(x[:len(years_avail)], np.array(actual_vals) / 1e6,
           color="black", zorder=5, s=80,
           marker="D", label="Actual Revenue")

# Forecast marker and annotation
if show_forecast:
    fc_x = x[len(years_avail):]
    fc_y = [all_actual[i] / 1e6 for i in range(len(years_avail), len(all_years))]
    ax.scatter(fc_x, fc_y,
               color=CPX_RED, zorder=5, s=80,
               marker="D", label="Forecast")
    ax.axvline(x=len(years_avail) - 0.25, color=CPX_MGREY, linewidth=1, linestyle="--")
    ax.text(len(years_avail) - 0.22, ax.get_ylim()[0] + 2, "Forecast →",
            fontsize=12, color=CPX_MGREY, fontstyle="italic")

ax.set_xticks(x)
ax.set_xticklabels(all_years, fontsize=16, fontweight="bold")
plt.setp(ax.get_yticklabels(), fontsize=16, fontweight="bold")
ax.set_ylabel("Annual Revenue ($M)", fontsize=16, fontweight="bold")
ax.set_xlabel("Year", fontsize=16, fontweight="bold")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:.0f}M"))
title_suffix = f"2023–{all_years[-1]}F" if show_forecast else "2023–2025"
ax.set_title(f"LBE Annual Revenue Decomposition by Funnel ({title_suffix})", loc="center",
             fontsize=14, fontweight="bold", color=CPX_NAVY)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=3, fontsize=16, frameon=False)
ax.grid(False)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

plt.tight_layout()
save_fig(fig, "lbe_v5e_annual_rev_decomp.png")
display(fig)
plt.close(fig)

# ── Summary Table ─────────────────────────────────────────────────────────────
tbl_rows = []
for i, yr in enumerate(all_years):
    total = all_actual[i]
    media_total = all_awareness[i] + all_consid[i]
    tbl_rows.append({
        "Year": f"{yr}{' (F)' if yr >= 2026 and show_forecast else ''}",
        "Baseline (M)": round(all_baseline[i] / 1e6, 2),
        "Awareness (M)": round(all_awareness[i] / 1e6, 2),
        "Consideration (M)": round(all_consid[i] / 1e6, 2),
        "Media Total (M)": round(media_total / 1e6, 2),
        "Total (M)": round(total / 1e6, 2),
        "Media %": f"{media_total / total * 100:.1f}%",
    })
print(f"\n{'═'*70}\n  LBE Annual Media Contribution Summary\n{'═'*70}")
display(pd.DataFrame(tbl_rows))

# COMMAND ----------

# DBTITLE 1,LBE Funnel Efficiency — Spend, Contribution & Rev/$ by Year
# ━━ LBE FUNNEL EFFICIENCY — MECHANISM-SPECIFIC ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Awareness → Volume Play (Cost per Incremental Visit)
# Consideration → Value Play (ROAS via spend-per-visit uplift)
import matplotlib.ticker as mticker
import matplotlib.patheffects as pe

years_eff = [2023, 2024, 2025]

# ── Decompose by PRIMARY MECHANISM per funnel ─────────────────────────────────
# Awareness:     RPT × ΔTxn  (revenue from extra people walking in)
# Consideration: ΔRPT × Txn  (revenue from higher spend-per-visit)

incr_txn_aw     = np.zeros(len(d))   # incremental transactions from awareness
rev_via_txn_aw  = np.zeros(len(d))   # revenue from awareness volume mechanism
rev_via_rpt_co  = np.zeros(len(d))   # revenue from consideration value mechanism

for brand in ["TRR", "PDM"]:
    r = results[brand]
    # Awareness volume mechanism: pred_rpt × (pred_txn − counterfactual_txn)
    txn_uplift_aw      = r["pred_txn"] - np.exp(r["log_pred_txn"] - r["m_aw_txn"])
    incr_txn_aw       += txn_uplift_aw
    rev_via_txn_aw    += r["pred_rpt"] * txn_uplift_aw

    # Consideration value mechanism: (pred_rpt − counterfactual_rpt) × pred_txn
    rpt_uplift_co      = r["pred_rpt"] - np.exp(r["log_pred_rpt"] - r["m_co_rpt"])
    rev_via_rpt_co    += rpt_uplift_co * r["pred_txn"]

mech_df = d[["week_date"]].copy()
mech_df["year"]            = mech_df["week_date"].dt.year
mech_df["incr_txn_aw"]     = incr_txn_aw
mech_df["rev_via_txn_aw"]  = rev_via_txn_aw
mech_df["rev_via_rpt_co"]  = rev_via_rpt_co

# ── Build efficiency rows ─────────────────────────────────────────────────────
# Contribution bars use TOTAL revenue from decomp_df (both mechanisms, matches cell 18)
# Efficiency lines use mechanism-specific metrics (tells the differentiated story)
eff_rows = []
for year in years_eff:
    yr_spend  = df_all[df_all["year"] == year]
    yr_mech   = mech_df[mech_df["year"] == year]
    yr_decomp = decomp_df[decomp_df["year"] == year]

    # Awareness: TOTAL contribution from decomp_df, efficiency from volume mechanism
    aw_spend   = yr_spend["Awareness"].sum()
    aw_rev_total = yr_decomp["media_Awareness"].sum()  # both mechanisms combined
    aw_txns    = yr_mech["incr_txn_aw"].sum()
    eff_rows.append({
        "Year": year, "Funnel": "Awareness",
        "Spend_M": aw_spend / 1e6,
        "Contribution_M": aw_rev_total / 1e6,  # matches cell 18
        "Incr_Txn": aw_txns,
        "Txn_per_Dollar": aw_txns / aw_spend if aw_spend > 0 else 0,
        "Cost_per_Visit": aw_spend / aw_txns if aw_txns > 0 else 0,
    })

    # Consideration: TOTAL contribution from decomp_df, efficiency from value mechanism
    co_spend     = yr_spend["Consideration"].sum()
    co_rev_total = yr_decomp["media_Consideration"].sum()  # both mechanisms combined
    co_rev_rpt   = yr_mech["rev_via_rpt_co"].sum()  # RPT mechanism only (for ROAS line)
    eff_rows.append({
        "Year": year, "Funnel": "Consideration",
        "Spend_M": co_spend / 1e6,
        "Contribution_M": co_rev_total / 1e6,  # matches cell 18
        "Incr_Txn": 0,
        "Txn_per_Dollar": 0,
        "Cost_per_Visit": 0,
        "Rev_per_Dollar_RPT": co_rev_rpt / co_spend if co_spend > 0 else 0,
    })

eff_df = pd.DataFrame(eff_rows)

# ── Funnel colours & styling ──────────────────────────────────────────────────
funnel_colors_eff = {"Awareness": CPX_NAVY, "Consideration": CPX_RED}
funnel_markers    = {"Awareness": "o", "Consideration": "s"}

# ── Figure: 1×2 subplots — reader-friendly metrics ───────────────────────────
# Awareness: "Cost per Incremental Visit" (lower = better)
# Consideration: "ROAS" expressed as ×  (higher = better)
fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharey=False)
fig.patch.set_facecolor("white")

for idx, funnel in enumerate(["Awareness", "Consideration"]):
    ax  = axes[idx]
    ax2 = ax.twinx()
    fdata = eff_df[eff_df["Funnel"] == funnel].reset_index(drop=True)
    color = funnel_colors_eff[funnel]
    x = np.arange(len(years_eff))
    w = 0.30

    # Bars: Spend (left) and Revenue Contribution (right)
    bars_sp = ax.bar(x - w/2, fdata["Spend_M"], width=w,
                     color=color, alpha=0.30, edgecolor=color, linewidth=1.5,
                     label="Media Spend")
    ct_label = "Revenue from Extra Visits" if funnel == "Awareness" else "Revenue from Higher Spend/Visit"
    bars_ct = ax.bar(x + w/2, fdata["Contribution_M"], width=w,
                     color=color, alpha=0.85, edgecolor="white", linewidth=0.5,
                     label=ct_label)

    # Spend labels
    for bar in bars_sp:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.03,
                f"${h:.2f}M", ha="center", va="bottom",
                fontsize=11, color=color, fontweight="bold")

    # Contribution labels
    for bar in bars_ct:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.03,
                f"${h:.2f}M", ha="center", va="bottom",
                fontsize=11, color="black", fontweight="bold")

    # Line: reader-friendly efficiency metric on secondary axis
    if funnel == "Awareness":
        # Cost per Incremental Visit — intuitive "how much to get one extra person in"
        eff_vals = fdata["Cost_per_Visit"].values
        eff_label_fmt = lambda v: f"${v:.0f}"
        eff_name = "Cost / Visit"
        ax2_ylabel = "Cost per Incremental Visit ($)"
        # Invert axis — lower is better for cost metrics
        ax2.invert_yaxis()
    else:
        # ROAS — "for every $1 spent, you get $X back through upsell"
        eff_vals = fdata["Rev_per_Dollar_RPT"].values
        eff_label_fmt = lambda v: f"{v:.1f}×"
        eff_name = "ROAS (upsell)"
        ax2_ylabel = "ROAS — Return per $1 Spent (×)"

    ax2.plot(x, eff_vals, color="black", linewidth=3, zorder=5,
             marker=funnel_markers[funnel], markersize=13,
             markerfacecolor="white", markeredgecolor="black", markeredgewidth=2.5)

    # Efficiency labels — large and clear
    for i, val in enumerate(eff_vals):
        offset_y = -18 if (funnel == "Awareness") else 14
        ax2.annotate(
            eff_label_fmt(val),
            xy=(x[i], val), xytext=(0, offset_y),
            textcoords="offset points", ha="center",
            fontsize=14, fontweight="bold", color="black",
            path_effects=[pe.withStroke(linewidth=4, foreground="white")]
        )

    # Formatting
    ax.set_xticks(x)
    ax.set_xticklabels(years_eff, fontsize=14, fontweight="bold")
    ax.set_ylabel("Dollars ($M)", fontsize=13, fontweight="bold")
    ax2.set_ylabel(ax2_ylabel, fontsize=12, fontweight="bold", color="black")

    ax.set_title(funnel, fontsize=14, fontweight="bold", color=color, pad=12)

    ax.set_ylim(0, max(fdata["Contribution_M"].max(), fdata["Spend_M"].max()) * 1.45)
    if funnel == "Awareness":
        pad = max(eff_vals) * 0.3
        ax2.set_ylim(max(eff_vals) + pad, min(eff_vals) - pad)  # inverted: high at bottom
    else:
        ax2.set_ylim(0, max(eff_vals) * 1.5)
    ax.grid(False)
    ax2.grid(False)
    plt.setp(ax.get_yticklabels(), fontsize=12, fontweight="bold")
    plt.setp(ax2.get_yticklabels(), fontsize=12, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)

    # Legends
    ax.legend(loc="upper left", fontsize=12, frameon=False)
    ax2.plot([], [], color="black", linewidth=3, marker=funnel_markers[funnel],
             markerfacecolor="white", markeredgecolor="black", markersize=9,
             label=eff_name)
    ax2.legend(loc="upper right", fontsize=12, frameon=False)

    # Add extra context: incremental visits count for Awareness (above bars)
    if funnel == "Awareness":
        for i, row in fdata.iterrows():
            top = max(row["Spend_M"], row["Contribution_M"])
            ax.text(x[i], top * 1.18, f"+{row['Incr_Txn']:,.0f} visits",
                    ha="center", fontsize=10, color=CPX_NAVY, fontstyle="italic")

fig.suptitle("LBE Media Efficiency by Funnel Mechanism (2023–2025)",
             fontsize=16, fontweight="bold", color=CPX_NAVY, y=0.99)
plt.tight_layout(rect=[0, 0.02, 1, 0.94])
save_fig(fig, "lbe_funnel_efficiency_by_year.png")
display(fig)
plt.close(fig)

# ── Summary Table — Wide Pivot (Year × Funnel) ───────────────────────────────
# Matches EXH deck format: rows = metrics, columns = Year × Funnel

# Build raw data per (year, funnel)
metrics = {}
for year in years_eff:
    aw = eff_df[(eff_df["Year"] == year) & (eff_df["Funnel"] == "Awareness")].iloc[0]
    co = eff_df[(eff_df["Year"] == year) & (eff_df["Funnel"] == "Consideration")].iloc[0]
    metrics[(year, "Awareness")] = {
        "Spend_M": aw["Spend_M"],
        "Contribution_M": aw["Contribution_M"],
        "Efficiency": aw["Cost_per_Visit"],
    }
    metrics[(year, "Consideration")] = {
        "Spend_M": co["Spend_M"],
        "Contribution_M": co["Contribution_M"],
        "Efficiency": co["Rev_per_Dollar_RPT"],
    }

# Build multi-index columns
years_cols = [2023, 2024, 2025]
funnels_cols = ["Awareness", "Consideration"]
multi_cols = pd.MultiIndex.from_product([years_cols, funnels_cols], names=["Year", "Funnel"])

# Row: Spend ($M)
spend_row = [metrics[(y, f)]["Spend_M"] for y in years_cols for f in funnels_cols]
# Row: Contribution ($M)
contrib_row = [metrics[(y, f)]["Contribution_M"] for y in years_cols for f in funnels_cols]
# Row: Efficiency (Cost/Visit for Aw, ROAS for Co)
eff_row = [metrics[(y, f)]["Efficiency"] for y in years_cols for f in funnels_cols]
# Row: YoY Eff Δ
yoy_row = []
for f in funnels_cols:
    prev_val = None
    for y in years_cols:
        curr_val = metrics[(y, f)]["Efficiency"]
        if prev_val is None:
            yoy_row.append(None)
        else:
            yoy_row.append((curr_val - prev_val) / prev_val * 100)
        prev_val = curr_val
# Reorder yoy_row to match (year, funnel) ordering
yoy_ordered = []
for y_idx, y in enumerate(years_cols):
    for f_idx, f in enumerate(funnels_cols):
        yoy_ordered.append(yoy_row[f_idx * len(years_cols) + y_idx])

# Create DataFrame
wide_df = pd.DataFrame(
    [spend_row, contrib_row, eff_row, yoy_ordered],
    index=["Spend ($M)", "Contribution ($M)", "Efficiency", "YoY Eff Δ"],
    columns=multi_cols
)

# Format for display
fmt_df = wide_df.copy().astype(object)
for y in years_cols:
    for f in funnels_cols:
        fmt_df.loc["Spend ($M)", (y, f)]       = f"{wide_df.loc['Spend ($M)', (y, f)]:.2f}"
        fmt_df.loc["Contribution ($M)", (y, f)] = f"{wide_df.loc['Contribution ($M)', (y, f)]:.2f}"
        eff_val = wide_df.loc["Efficiency", (y, f)]
        if f == "Awareness":
            fmt_df.loc["Efficiency", (y, f)] = f"${eff_val:.2f}"  # Cost per Visit
        else:
            fmt_df.loc["Efficiency", (y, f)] = f"{eff_val:.2f}"   # ROAS
        yoy_val = wide_df.loc["YoY Eff Δ", (y, f)]
        if yoy_val is None or pd.isna(yoy_val):
            fmt_df.loc["YoY Eff Δ", (y, f)] = "—"
        else:
            fmt_df.loc["YoY Eff Δ", (y, f)] = f"{yoy_val:+.1f}%"

# Rename Efficiency row to show what metric each funnel uses
fmt_df = fmt_df.rename(index={"Efficiency": "Cost/Visit | ROAS"})

# Style the table
def color_yoy(val):
    if val == "—":
        return ""
    try:
        v = float(val.replace("%", "").replace("+", ""))
        # For awareness: negative = better (cost went down); for consideration: positive = better
        return f"color: {'#d32f2f' if v > 0 else '#2e7d32'}" if "$" not in str(val) else ""
    except:
        return ""

print(f"{'═'*74}")
print(f"  LBE Funnel Efficiency Summary (2023–2025)")
print(f"{'═'*74}")
print(f"  Awareness: Cost per Incremental Visit (lower = better)")
print(f"  Consideration: ROAS via spend-per-visit uplift (higher = better)")
print(f"{'─'*74}")

# Flatten multi-index columns and expose row labels as a column
flat_df = fmt_df.copy()
flat_df.columns = [f"{y} | {f}" for y, f in flat_df.columns]
flat_df.insert(0, "Metric", flat_df.index)
flat_df = flat_df.reset_index(drop=True)
display(flat_df)

# COMMAND ----------

# DBTITLE 1,LBE Adstock Decay Curves — Carry-Over & Planning Lead Time
# ━━ LBE FUNNEL ADSTOCK DECAY CURVES — CARRY-OVER & PLANNING LEAD TIME ━━━━━━━━━━━
# Shows how each funnel's media impact decays week-over-week after activation
# Uses fitted alpha values from V5e model (averaged across TRR + PDM)
import matplotlib.patheffects as pe

# ── Extract fitted adstock alphas from results dict ───────────────────────────
# Awareness primary mechanism = volume (TXN model)
# Consideration primary mechanism = value (RPT model)
alpha_aw_txn = np.mean([results[b]["adstock_aw_txn"] for b in ["TRR", "PDM"]])
alpha_aw_rpt = np.mean([results[b]["adstock_aw_rpt"] for b in ["TRR", "PDM"]])
alpha_co_rpt = np.mean([results[b]["adstock_co_rpt"] for b in ["TRR", "PDM"]])
alpha_co_txn = np.mean([results[b]["adstock_co_txn"] for b in ["TRR", "PDM"]])

# Primary mechanism alpha for headline
alpha_aw = alpha_aw_txn  # awareness drives volume
alpha_co = alpha_co_rpt  # consideration drives value

# Derive carry-over window (weeks until <10% remaining)
def weeks_to_threshold(alpha, threshold=0.10):
    return int(np.ceil(np.log(threshold) / np.log(alpha))) if alpha > 0 else 0

aw_carry = weeks_to_threshold(alpha_aw)
co_carry = weeks_to_threshold(alpha_co)

decay_params = {
    "Awareness": {
        "alpha": alpha_aw,
        "color": CPX_NAVY,
        "carryover": f"{aw_carry-1}–{aw_carry} wks",
        "lead": f"{aw_carry-1}–{aw_carry} wks before",
    },
    "Consideration": {
        "alpha": alpha_co,
        "color": CPX_RED,
        "carryover": f"{co_carry-1}–{co_carry} wks",
        "lead": f"{co_carry-1}–{co_carry} wks before",
    },
}

print(f"Fitted adstock alphas (mean of TRR + PDM):")
print(f"  Awareness  → TXN model: {alpha_aw_txn:.3f}  |  RPT model: {alpha_aw_rpt:.3f}  → primary: {alpha_aw:.3f}")
print(f"  Consideration → RPT model: {alpha_co_rpt:.3f}  |  TXN model: {alpha_co_txn:.3f}  → primary: {alpha_co:.3f}")

# ── Plot ───────────────────────────────────────────────────────────────────
weeks = np.arange(0, 9, 0.1)
weeks_int = np.arange(0, 9)

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

    # Shade the carry-over window (weeks until <10%)
    shade_end = min(weeks_to_threshold(alpha, 0.10), 8)
    shade_x = weeks[weeks <= shade_end]
    shade_y = alpha ** shade_x * 100
    ax.fill_between(shade_x, 0, shade_y, color=color, alpha=0.08)

    # Annotate the carry-over window on the curve (stagger positions to avoid overlap)
    if funnel == "Awareness":
        annot_x = 1.5
        annot_y = alpha ** annot_x * 100
        txt_offset = (1.2, 8)
    else:
        annot_x = 1.0
        annot_y = alpha ** annot_x * 100
        txt_offset = (1.5, 15)

    ax.annotate(
        f"α = {alpha:.2f}\n{params['carryover']} carry-over\nPlan: {params['lead']}",
        xy=(annot_x, annot_y),
        xytext=(annot_x + txt_offset[0], annot_y + txt_offset[1]),
        fontsize=10, fontweight="bold", color=color,
        arrowprops=dict(arrowstyle="->", color=color, lw=1.5),
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor=color, alpha=0.9),
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
ax.legend(loc="upper right", fontsize=12, frameon=True, framealpha=0.9,
          edgecolor=CPX_LGREY)

ax.set_title("LBE — Adstock Decay by Funnel (Fitted from V5e Model)",
             fontsize=16, fontweight="bold", color=CPX_NAVY, pad=15)

plt.tight_layout()
save_fig(fig, "lbe_funnel_adstock_decay_curves.png")
display(fig)
plt.close(fig)

# ── Summary Table ────────────────────────────────────────────────────────
print(f"\n{'═'*74}")
print(f"  LBE Adstock Decay Parameters (V5e Fitted)")
print(f"{'═'*74}")
adstock_tbl = pd.DataFrame([
    {"Funnel": "Awareness", "Primary α": f"{alpha_aw:.3f}",
     "Mechanism": "Volume (Txn)", "Half-life (wks)": f"{np.log(0.5)/np.log(alpha_aw):.1f}",
     "90% Decay (wks)": f"{weeks_to_threshold(alpha_aw)}",
     "Planning Lead": decay_params['Awareness']['lead']},
    {"Funnel": "Consideration", "Primary α": f"{alpha_co:.3f}",
     "Mechanism": "Value (RPT)", "Half-life (wks)": f"{np.log(0.5)/np.log(alpha_co):.1f}",
     "90% Decay (wks)": f"{weeks_to_threshold(alpha_co)}",
     "Planning Lead": decay_params['Consideration']['lead']},
])
display(adstock_tbl)

# COMMAND ----------

# ── LBE Annual Spend Overview ─────────────────────────────────────────────────
# Uses df_all and annual_spend_M from Plots Config cell.
import matplotlib.ticker as mticker
from IPython.display import display as ipy_display

# ── Table display ─────────────────────────────────────────────────────────────
tbl = annual_spend_M.copy()
tbl.index = tbl.index.astype(str)
tbl.loc["3-Year Total"] = tbl.sum()

ipy_display(
    tbl.style
    .format("${:.2f}M")
    .set_caption("LBE Annual Media Spend by Funnel (2023–2025)")
    .set_table_styles([
        {"selector": "caption",
         "props": [("font-size", "13px"), ("font-weight", "bold"),
                   ("color", "#002B5C"), ("text-align", "left")]},
        {"selector": "th",
         "props": [("background-color", "#002B5C"), ("color", "white"),
                   ("font-weight", "bold"), ("text-align", "center")]},
        {"selector": "tr:last-child",
         "props": [("font-weight", "bold"), ("background-color", "#F2F2F2")]},
    ])
)

# ── Chart ─────────────────────────────────────────────────────────────────────
years  = [2023, 2024, 2025]
aw_vals = annual_spend_M.loc[years, "Awareness"].values
co_vals = annual_spend_M.loc[years, "Consideration"].values
x = np.arange(len(years))
width = 0.55

fig, ax = plt.subplots(figsize=(11, 9))
fig.patch.set_facecolor(WHITE)

b1 = ax.bar(x, aw_vals, width, label="Awareness", color=CPX_NAVY)
b2 = ax.bar(x, co_vals, width, label="Consideration", color=CPX_RED, bottom=aw_vals)

def label_segments(bars, bottoms):
    for i, bar in enumerate(bars):
        h = bar.get_height()
        b = bottoms[i]
        if h >= 0.1:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                b + h / 2,
                f"${h:.2f}M",
                ha="center", va="center",
                fontsize=14, fontweight="bold", color=WHITE
            )

label_segments(b1, [0, 0, 0])
label_segments(b2, aw_vals)

totals = aw_vals + co_vals
for i, total in enumerate(totals):
    ax.text(i, total + 0.06, f"${total:.2f}M",
            ha="center", va="bottom",
            fontsize=14, fontweight="bold", color=CPX_NAVY)

ax.set_xticks(x)
ax.set_xticklabels([str(y) for y in years], fontsize=14, fontweight="bold")
plt.setp(ax.get_yticklabels(), fontsize=12, fontweight="bold")
ax.set_ylabel("Annual Media Spend ($M)", fontsize=14, fontweight="bold")
ax.set_xlabel("Year", fontsize=14, fontweight="bold")
ax.set_ylim(0, totals.max() * 1.15)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:.1f}M"))
ax.set_title("LBE MMM — Annual Media Spend by Funnel\n2023–2025",
             fontsize=14, fontweight="bold", color=CPX_NAVY)
ax.legend(loc="upper left", frameon=False, fontsize=12)
ax.grid(False)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

plt.tight_layout()
save_fig(fig, "lbe_b0_annual_spend_by_funnel.png")
display(fig); plt.close(fig)

# COMMAND ----------

# ── LBE SPEND BY CHANNEL — STACKED BY FUNNEL — ONE PLOT PER YEAR ─────────────
import matplotlib.ticker as mticker

channel_funnel_cols = {
    ("TRR", "Awareness")    : "Awareness_TRR",
    ("TRR", "Consideration"): "Consideration_TRR",
    ("PDM", "Awareness")    : "Awareness_PDM",
    ("PDM", "Consideration"): "Consideration_PDM",
}

funnels  = ["Awareness", "Consideration"]
channels = ["TRR", "PDM"]
funnel_colors = {"Awareness": CPX_NAVY, "Consideration": CPX_RED}

rows = []
for (channel, funnel), col in channel_funnel_cols.items():
    for year in [2023, 2024, 2025]:
        spend = df_all.loc[df_all["year"] == year, col].sum()
        rows.append({"Channel": channel, "Funnel": funnel, "Year": year, "Spend_M": spend / 1e6})

spend_df = pd.DataFrame(rows)
spend_df = spend_df[spend_df["Spend_M"] > 0.001].copy()

for year in [2023, 2024, 2025]:
    ydata = spend_df[spend_df["Year"] == year].copy()
    ch_totals = ydata.groupby("Channel")["Spend_M"].sum().sort_values(ascending=False)
    ch_order  = ch_totals.index.tolist()
    x         = np.arange(len(ch_order))

    fig, ax = plt.subplots(figsize=get_figsize("16:9"))
    fig.patch.set_facecolor(WHITE)
    bottoms = np.zeros(len(ch_order))

    for funnel in funnels:
        heights = []
        for ch in ch_order:
            val = ydata.loc[(ydata["Funnel"] == funnel) & (ydata["Channel"] == ch), "Spend_M"]
            heights.append(float(val.values[0]) if len(val) > 0 else 0.0)
        bars = ax.bar(x, heights, bottom=bottoms, label=funnel,
                      color=funnel_colors[funnel], width=0.55, edgecolor=WHITE, linewidth=0.5)
        for i, bar in enumerate(bars):
            h = bar.get_height()
            if h > 0.03:
                ax.text(bar.get_x() + bar.get_width() / 2, bottoms[i] + h / 2,
                        f"${h:.2f}M", ha="center", va="center",
                        fontsize=14, fontweight="bold", color=WHITE)
        bottoms += np.array(heights)

    for i, total in enumerate(bottoms):
        if total > 0:
            ax.text(x[i], total + 0.01, f"${total:.2f}M",
                    ha="center", va="bottom", fontsize=14, fontweight="bold", color=CPX_NAVY)

    ax.set_xticks(x)
    ax.set_xticklabels(ch_order, rotation=0, ha="center", fontsize=14, fontweight="bold")
    plt.setp(ax.get_yticklabels(), fontsize=12, fontweight="bold")
    ax.set_ylabel("Annual Spend ($M)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Channel", fontsize=14, fontweight="bold")
    ax.set_ylim(0, bottoms.max() * 1.15)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:.2f}M"))
    ax.set_title(f"LBE Annual Media Spend by Channel & Funnel — {year}",
                 fontsize=14, fontweight="bold", color=CPX_NAVY)
    ax.legend(title="Funnel", fontsize=12, title_fontsize=12, loc="upper right", frameon=False)
    ax.grid(False)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    save_fig(fig, f"lbe_spend_channel_funnel_{year}.png")
    display(fig); plt.close(fig)

# COMMAND ----------

# ── LBE SPEND BY CHANNEL & FUNNEL — ALL YEARS COMBINED ────────────────────────
import matplotlib.ticker as mticker

channel_funnel_cols = {
    ("TRR", "Awareness")    : "Awareness_TRR",
    ("TRR", "Consideration"): "Consideration_TRR",
    ("PDM", "Awareness")    : "Awareness_PDM",
    ("PDM", "Consideration"): "Consideration_PDM",
}

funnels  = ["Awareness", "Consideration"]
channels = ["TRR", "PDM"]
years    = [2023, 2024, 2025]
funnel_colors = {"Awareness": CPX_NAVY, "Consideration": CPX_RED}

rows = []
for (channel, funnel), col in channel_funnel_cols.items():
    for year in years:
        spend = df_all.loc[df_all["year"] == year, col].sum()
        rows.append({"Channel": channel, "Funnel": funnel, "Year": year, "Spend_M": spend / 1e6})

spend_df_combined = pd.DataFrame(rows)

n_years    = len(years)
n_channels = len(channels)
bar_width  = 0.25
group_gap  = 0.15

fig, ax = plt.subplots(figsize=(11, 9))
fig.patch.set_facecolor(WHITE)
all_totals = []

for ci, channel in enumerate(channels):
    offset = (ci - (n_channels - 1) / 2) * (bar_width + group_gap)
    x_positions = np.arange(n_years) + offset
    bottoms = np.zeros(n_years)

    for funnel in funnels:
        heights = []
        for year in years:
            val = spend_df_combined.loc[
                (spend_df_combined["Channel"] == channel) &
                (spend_df_combined["Funnel"] == funnel) &
                (spend_df_combined["Year"] == year), "Spend_M"]
            heights.append(float(val.values[0]) if len(val) > 0 else 0.0)
        heights = np.array(heights)
        bars = ax.bar(x_positions, heights, width=bar_width, bottom=bottoms,
                      color=funnel_colors[funnel], edgecolor=WHITE, linewidth=0.5,
                      label=funnel if ci == 0 else "")
        for i, bar in enumerate(bars):
            h = bar.get_height()
            if h > 0.03:
                ax.text(bar.get_x() + bar.get_width() / 2, bottoms[i] + h / 2,
                        f"${h:.2f}M", ha="center", va="center",
                        fontsize=14, fontweight="bold", color=WHITE)
        bottoms += heights

    for i, total in enumerate(bottoms):
        if total > 0:
            ax.text(x_positions[i], total + 0.04, f"${total:.2f}M",
                    ha="center", va="bottom", fontsize=14, fontweight="bold", color=CPX_NAVY)

    all_totals.append((x_positions.copy(), bottoms.copy(), channel))

    for i, xp in enumerate(x_positions):
        ax.text(xp, -0.03, channel, ha="center", va="top",
                fontsize=14, fontweight="bold", color=CPX_DGREY,
                transform=ax.get_xaxis_transform())

max_height = max(t.max() for _, t, _ in all_totals)
label_y = max_height + 0.55
for yi, year in enumerate(years):
    ax.text(yi, label_y, str(year), ha="center", va="bottom",
            fontsize=13, fontweight="bold", color="black")

ax.set_xticks(np.arange(n_years))
ax.set_xticklabels([""] * n_years)
plt.setp(ax.get_yticklabels(), fontsize=12, fontweight="bold")
ax.set_ylabel("Annual Spend ($M)", fontsize=14, fontweight="bold")
ax.set_xlabel("")
ax.set_ylim(0, spend_df_combined.groupby(["Channel", "Year"])["Spend_M"].sum().max() * 1.5)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:.2f}M"))
ax.set_title("LBE Annual Media Spend by Channel & Funnel (2023–2025)",
             fontsize=14, fontweight="bold", color=CPX_NAVY)
ax.legend(loc="upper right", title="Funnel", fontsize=12, title_fontsize=12, frameon=False)
ax.grid(False)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
plt.tight_layout()
save_fig(fig, "lbe_spend_channel_funnel_combined.png")
display(fig); plt.close(fig)

# COMMAND ----------

# ── DATA TABLES: LBE Spend by Channel & Funnel — One Table Per Year ───────────
for year in [2023, 2024, 2025]:
    pivot = (
        spend_df_combined[spend_df_combined["Year"] == year]
        .pivot_table(index="Channel", columns="Funnel", values="Spend_M", aggfunc="sum", fill_value=0)
        .reindex(columns=["Awareness", "Consideration"])
        .fillna(0)
    )
    pivot = pivot * 100
    pivot["Total"] = pivot.sum(axis=1)
    pivot = pivot.sort_values("Total", ascending=False).reset_index()
    pivot.columns.name = None
    pivot[["Awareness", "Consideration", "Total"]] = pivot[["Awareness", "Consideration", "Total"]].round(2)

    print(f"\n{'═'*60}")
    print(f"  LBE Annual Media Spend by Channel & Funnel — {year}")
    print(f"  (values in $10K)")
    print(f"{'═'*60}")
    display(pivot)

# COMMAND ----------

# DBTITLE 1,LBE Revenue Breakdown by Year
# ── LBE REVENUE BREAKDOWN BY YEAR ───────────────────────────────────────────
import matplotlib.ticker as mticker

rev_df_lbe = (
    df_all[df_all["year"].isin([2023, 2024, 2025])]
    .groupby("year")["LBE_Revenue"]
    .sum()
    .reset_index()
)
rev_df_lbe["Revenue_M"] = rev_df_lbe["LBE_Revenue"] / 1e6

forecast_rows = [{"year": 2026, "LBE_Revenue": FORECAST_2026, "Revenue_M": FORECAST_2026 / 1e6}]
if FORECAST_2027 is not None:
    forecast_rows.append({"year": 2027, "LBE_Revenue": FORECAST_2027, "Revenue_M": FORECAST_2027 / 1e6})
rev_df_lbe = pd.concat([rev_df_lbe, pd.DataFrame(forecast_rows)], ignore_index=True)
rev_df_lbe["YoY_pct"] = rev_df_lbe["Revenue_M"].pct_change() * 100

fig, ax = plt.subplots(figsize=get_figsize("16:9"))
fig.patch.set_facecolor(WHITE)
x     = np.arange(len(rev_df_lbe))
width = 0.5
bar_alphas = [0.9 if yr <= 2025 else 0.5 for yr in rev_df_lbe["year"]]
for i_b in range(len(x)):
    ax.bar(x[i_b], rev_df_lbe["Revenue_M"].iloc[i_b], width=width,
           color="black", alpha=bar_alphas[i_b], edgecolor=WHITE, linewidth=0.5)
bars = ax.containers[0] if len(ax.containers) == 1 else [p for c in ax.containers for p in c]

for i, (bar, row) in enumerate(zip(bars, rev_df_lbe.itertuples())):
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, h + 1.0, f"${h:.2f}M",
            ha="center", va="bottom", fontsize=14, fontweight="bold", color=CPX_NAVY)
    if i > 0:
        yoy = row.YoY_pct
        color = CPX_RED if yoy < 0 else "black"
        ax.text(bar.get_x() + bar.get_width() / 2, h + 5.0, f"{yoy:+.1f}% YoY",
                ha="center", va="bottom", fontsize=14, fontweight="bold", color=color)

ax.set_xticks(x)
ax.set_xticklabels(rev_df_lbe["year"].astype(str), fontsize=14, fontweight="bold")
plt.setp(ax.get_yticklabels(), fontsize=14, fontweight="bold")
ax.set_ylabel("Total Annual Revenue ($M)", fontsize=14, fontweight="bold")
ax.set_xlabel("Year", fontsize=14, fontweight="bold")
ax.set_ylim(0, rev_df_lbe["Revenue_M"].max() * 1.25)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:.0f}M"))
last_fc_yr = int(rev_df_lbe["year"].max())
ax.set_title(f"LBE Total Annual Revenue (2023–{last_fc_yr}F)",
             fontsize=14, fontweight="bold", color=CPX_NAVY)
ax.grid(False)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
plt.tight_layout()
save_fig(fig, "lbe_revenue_by_year.png")
display(fig); plt.close(fig)

# ── Table ────────────────────────────────────────────────────────────────────
table_lbe = rev_df_lbe[["year", "Revenue_M", "YoY_pct"]].copy()
table_lbe.columns = ["Year", "Total Revenue ($M)", "YoY Change (%)"]
table_lbe["Total Revenue ($M)"] = table_lbe["Total Revenue ($M)"].round(2)
table_lbe["YoY Change (%)"] = table_lbe["YoY Change (%)"].round(1)
table_lbe = table_lbe.set_index("Year")

display(
    table_lbe
    .style
    .format({"Total Revenue ($M)": "${:.2f}M", "YoY Change (%)": "{:+.1f}%"}, na_rep="—")
    .set_caption(f"LBE Total Annual Revenue (2023–{last_fc_yr}F)")
    .set_properties(**{"text-align": "center"})
    .set_table_styles([
        {"selector": "caption",
         "props": [("font-size", "13px"), ("font-weight", "bold"),
                   ("color", "#002B5C"), ("text-align", "left")]},
        {"selector": "th",
         "props": [("background-color", "#002B5C"), ("color", "white"),
                   ("font-weight", "bold"), ("text-align", "center")]},
    ])
)

# COMMAND ----------

# ── LBE PER-CHANNEL ANNUAL SPEND & REVENUE CONTRIBUTION ─────────────────────
# One chart per channel (TRR, PDM). Uses V5e per-brand ROI from results dict.
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker

FUNNEL_COLOR = {"Awareness": CPX_NAVY, "Consideration": CPX_RED}
FUNNELS_LBE  = ["Awareness", "Consideration"]
CHANNELS     = ["TRR", "PDM"]
years_plot   = [2023, 2024, 2025]

# ── Per-brand ROI from V5e results ──────────────────────────────────────────
chan_roi = {
    ("TRR", "Awareness"):     brand_roi["TRR"]["Awareness"],
    ("TRR", "Consideration"): brand_roi["TRR"]["Consideration"],
    ("PDM", "Awareness"):     brand_roi["PDM"]["Awareness"],
    ("PDM", "Consideration"): brand_roi["PDM"]["Consideration"],
}

chan_funnel_col = {
    ("TRR", "Awareness"):     "Awareness_TRR",
    ("TRR", "Consideration"): "Consideration_TRR",
    ("PDM", "Awareness"):     "Awareness_PDM",
    ("PDM", "Consideration"): "Consideration_PDM",
}

n_yrs = len(years_plot)
bar_w = 0.35
gap   = 0.15
group = bar_w * 2 + gap + 0.2

for channel in CHANNELS:
    spend_by_f = {}
    contrib_by_f = {}
    for funnel in FUNNELS_LBE:
        col = chan_funnel_col[(channel, funnel)]
        roi = chan_roi[(channel, funnel)]
        s_arr, c_arr = [], []
        for yr in years_plot:
            spend = df_all.loc[df_all["year"] == yr, col].sum() / 1e6
            s_arr.append(spend)
            c_arr.append(spend * roi)
        spend_by_f[funnel]   = np.array(s_arr)
        contrib_by_f[funnel] = np.array(c_arr)

    total_spend_yr   = sum(spend_by_f[f] for f in FUNNELS_LBE)
    total_contrib_yr = sum(contrib_by_f[f] for f in FUNNELS_LBE)

    x_spend   = np.array([i * group for i in range(n_yrs)])
    x_contrib = x_spend + bar_w + gap

    fig, ax = plt.subplots(figsize=(11, 9))
    bottoms_s = np.zeros(n_yrs)
    for f in FUNNELS_LBE:
        ax.bar(x_spend, spend_by_f[f], bar_w, bottom=bottoms_s,
               color=FUNNEL_COLOR[f], alpha=0.9, edgecolor="white", lw=0.4)
        bottoms_s += spend_by_f[f]

    bottoms_c = np.zeros(n_yrs)
    for f in FUNNELS_LBE:
        ax.bar(x_contrib, contrib_by_f[f], bar_w, bottom=bottoms_c,
               color=FUNNEL_COLOR[f], alpha=0.55, edgecolor="white", lw=0.4, hatch="///")
        bottoms_c += contrib_by_f[f]

    for i in range(n_yrs):
        if bottoms_s[i] > 0:
            ax.annotate(f"${bottoms_s[i]:.2f}M",
                        xy=(x_spend[i], bottoms_s[i]), xytext=(0, 4), textcoords="offset points",
                        ha="center", fontsize=8, color=CPX_DGREY, fontweight="bold")
        if bottoms_c[i] > 0:
            ax.annotate(f"${bottoms_c[i]:.2f}M",
                        xy=(x_contrib[i], bottoms_c[i]), xytext=(0, 4), textcoords="offset points",
                        ha="center", fontsize=11, color=CPX_DGREY, fontweight="bold")

    x_mid = (x_spend + x_contrib) / 2
    ax.set_xticks(x_mid)
    ax.set_xticklabels([str(y) for y in years_plot], fontsize=14, fontweight="bold")
    ax.set_xlim(-0.3, x_contrib[-1] + bar_w + 0.3)

    max_val = max(bottoms_s.max(), bottoms_c.max())
    ax.set_ylabel("$M", fontsize=14, fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:.1f}M"))
    ax.set_ylim(bottom=0, top=max_val * 1.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.setp(ax.get_yticklabels(), fontsize=12, fontweight="bold")
    ax.grid(False)

    handles = [mpatches.Patch(color=FUNNEL_COLOR[f], label=f) for f in FUNNELS_LBE]
    handles += [
        mpatches.Patch(facecolor="#999999", alpha=0.9, label="Spend"),
        mpatches.Patch(facecolor="#999999", alpha=0.55, hatch="///", label="Revenue Contrib"),
    ]
    ax.legend(handles=handles, fontsize=12, loc="upper right", frameon=False)
    ax.set_title(f"{channel} — Annual Spend vs Revenue Contribution",
                 fontsize=14, fontweight="bold", color=CPX_NAVY, pad=10)

    plt.tight_layout()
    save_fig(fig, f"lbe_channel_{channel.lower()}_spend_vs_contrib.png")
    display(fig)
    plt.close(fig)

    tbl_rows = []
    for yr_idx, yr in enumerate(years_plot):
        row = {"Year": yr}
        for f in FUNNELS_LBE:
            row[f"{f} Spend ($K)"] = f"{spend_by_f[f][yr_idx] * 1000:.0f}K"
            row[f"{f} Contrib ($M)"] = round(contrib_by_f[f][yr_idx], 3)
        row["Total Spend ($K)"] = f"{total_spend_yr[yr_idx] * 1000:.0f}K"
        row["Total Contrib ($M)"] = round(total_contrib_yr[yr_idx], 3)
        tbl_rows.append(row)
    rate_row = {"Year": "ROI"}
    for f in FUNNELS_LBE:
        rate_row[f"{f} Spend ($K)"] = ""
        rate_row[f"{f} Contrib ($M)"] = f"{chan_roi[(channel, f)]:.2f}×"
    rate_row["Total Spend ($K)"] = ""
    rate_row["Total Contrib ($M)"] = ""
    tbl_rows.append(rate_row)

    print(f"\n{'═'*60}")
    print(f"  {channel} — Spend & Revenue Contribution")
    print(f"{'═'*60}")
    from IPython.display import display as ipy_display
    ipy_display(pd.DataFrame(tbl_rows))

# COMMAND ----------

# ── LBE DIMINISHING RETURNS & MARGINAL RETURN CURVES ───────────────────────
# Built on V5e dual-target model. Varies total budget at 2025 allocation shares,
# computes combined (RPT + TXN) Hill saturation response and marginal return.
from matplotlib.lines import Line2D
import matplotlib.ticker as mticker

print("Generating aggregate response and marginal return curves...")

# ── V5e Hill saturation response ──────────────────────────────────────────────
# At steady-state (constant weekly spend), adstocked input ≈ raw input.
# Combined incremental revenue:
#   ΔRev ≈ Hill(aw_norm)*beta_aw*avg_txn + Hill(co_norm)*beta_co*avg_txn
#         + Hill(aw_norm)*beta_aw*avg_rpt + Hill(co_norm)*beta_co*avg_rpt
# We approximate by scaling known total incremental revenue via Hill ratio
# relative to the training mean spend level.

def hill(x, half=0.5):
    """Hill saturation with half-saturation = 0.5 (prior mean of Gamma(2,2))."""
    return x / (x + half)

def hill_ratio(x_new, x_ref, half=0.5):
    ref_val = max(hill(x_ref, half), 1e-9)
    return hill(x_new, half) / ref_val

def v5e_media_rev(brand, aw_norm, co_norm):
    """
    Annual incremental revenue from brand media at normalised spend levels.
    aw_norm, co_norm: in [0,1] space (same normalisation as training: M/m_s, max=1).
    Scales known total incremental revenue by Hill ratio vs training mean.
    """
    r = results[brand]
    n = len(d)
    # Training mean normalised spend
    m_s = np.array([d[f"{brand}_Awareness_spend"].max(),
                    d[f"{brand}_Consideration_spend"].max()])
    m_s[m_s == 0] = 1.0
    aw_train_mean = (d[f"{brand}_Awareness_spend"] / m_s[0]).mean()
    co_train_mean = (d[f"{brand}_Consideration_spend"] / m_s[1]).mean()

    aw_scale = hill_ratio(aw_norm, aw_train_mean)
    co_scale = hill_ratio(co_norm, co_train_mean)

    annual_aw = r["aw_incr_rev"] * aw_scale
    annual_co = r["co_incr_rev"] * co_scale
    return annual_aw + annual_co

# ── 2025 budget allocation shares ────────────────────────────────────────────
df_2025 = df_all[df_all["year"] == 2025]
aw_weekly_2025    = df_2025["Awareness"].mean()
co_weekly_2025    = df_2025["Consideration"].mean()
total_weekly_2025 = aw_weekly_2025 + co_weekly_2025
aw_share = aw_weekly_2025 / total_weekly_2025
co_share = co_weekly_2025 / total_weekly_2025
total_annual_2025 = total_weekly_2025 * 52

# Normalisation scales per brand
m_s_TRR = np.array([d["TRR_Awareness_spend"].max(), d["TRR_Consideration_spend"].max()])
m_s_PDM = np.array([d["PDM_Awareness_spend"].max(), d["PDM_Consideration_spend"].max()])
m_s_TRR[m_s_TRR == 0] = 1.0
m_s_PDM[m_s_PDM == 0] = 1.0

def combined_rev(annual_budget):
    """Combined incremental revenue (TRR + PDM) at a given annual budget."""
    weekly = annual_budget / 52
    aw_wk = weekly * aw_share
    co_wk = weekly * co_share
    return (v5e_media_rev("TRR", aw_wk / m_s_TRR[0], co_wk / m_s_TRR[1]) +
            v5e_media_rev("PDM", aw_wk / m_s_PDM[0], co_wk / m_s_PDM[1]))

# ── X axis: annual spend from near-zero to 10x 2025 ──────────────────────────
x_annual   = np.linspace(52 * 100, total_annual_2025 * 10, 300)
curve_mean = np.array([combined_rev(b) for b in x_annual])

# ── Marginal returns & key points ─────────────────────────────────────────────
dx           = x_annual[1] - x_annual[0]
marginal_mean = np.gradient(curve_mean, dx)
second_deriv  = np.gradient(marginal_mean, dx)
inflection_idx = next((i for i in range(1, len(second_deriv))
                       if second_deriv[i - 1] > 0 and second_deriv[i] <= 0), None)

sat_95_idx   = np.argmin(np.abs(curve_mean - curve_mean[-1] * 0.95))
sat_95_spend = x_annual[sat_95_idx]
sat_95_rev   = curve_mean[sat_95_idx]

year_points = {}
for year in [2023, 2024, 2025]:
    df_yr = df_all[df_all["year"] == year]
    annual_budget = df_yr["Awareness"].sum() + df_yr["Consideration"].sum()
    year_points[year] = {"annual_spend": annual_budget, "annual_rev": combined_rev(annual_budget)}

YEAR_STYLES = {
    2023: {"color": CPX_DGREY,  "linestyle": "--", "linewidth": 1.2, "marker": "o"},
    2024: {"color": CPX_NAVY,   "linestyle": "-.", "linewidth": 1.2, "marker": "s"},
    2025: {"color": CPX_RED,    "linestyle": "-",  "linewidth": 1.5, "marker": "D"},
}

# ── PLOT 1: Total Response Curve ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(16, 9))
ax.plot(x_annual / 1e6, curve_mean / 1e6, color=CPX_NAVY, linewidth=2.5,
        label="Total media response (V5e posterior mean)")

if inflection_idx:
    inf_spend = x_annual[inflection_idx]
    inf_rev   = curve_mean[inflection_idx]
    ax.axvline(inf_spend / 1e6, color=CPX_DGREY, linewidth=1.0, linestyle="--", alpha=0.7)
    ax.scatter([inf_spend / 1e6], [inf_rev / 1e6], color=CPX_DGREY, s=80, zorder=6, marker="^")
    ax.annotate(f"Inflection\n${inf_spend/1e6:.1f}M/yr",
                xy=(inf_spend / 1e6, inf_rev / 1e6), xytext=(12, -40), textcoords="offset points",
                fontsize=8, color=CPX_DGREY, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=CPX_DGREY, lw=0.8),
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.85, ec="none"))

ax.axvline(sat_95_spend / 1e6, color=CPX_RED, linewidth=1.0, linestyle=":", alpha=0.8)
ax.scatter([sat_95_spend / 1e6], [sat_95_rev / 1e6], color=CPX_RED, s=100, zorder=6, marker="*")
ax.annotate(f"95% Saturation\n${sat_95_spend/1e6:.1f}M/yr\n"
            f"{sat_95_spend/year_points[2025]['annual_spend']:.1f}x 2025 spend",
            xy=(sat_95_spend / 1e6, sat_95_rev / 1e6), xytext=(-80, -50), textcoords="offset points",
            fontsize=8, color=CPX_RED, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=CPX_RED, lw=0.8),
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.85, ec="none"))

for year, style in YEAR_STYLES.items():
    pts = year_points[year]
    ax.axvline(pts["annual_spend"] / 1e6, color=style["color"], linestyle=style["linestyle"],
               linewidth=style["linewidth"], alpha=0.85, zorder=4)
    ax.scatter([pts["annual_spend"] / 1e6], [pts["annual_rev"] / 1e6],
               color=style["color"], marker=style["marker"], s=90, zorder=6)
    ax.annotate(f"{year}\n${pts['annual_spend']/1e6:.1f}M/yr\n(${pts['annual_rev']/1e6:.1f}M rev)",
                xy=(pts["annual_spend"] / 1e6, pts["annual_rev"] / 1e6),
                xytext=(10, 15 - list(YEAR_STYLES.keys()).index(year) * 40),
                textcoords="offset points",
                fontsize=8, color=style["color"], fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=style["color"], lw=0.8),
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.85, ec="none"))

ax.set_ylabel("Annual Media Revenue Contribution ($M)", fontsize=14, fontweight="bold")
ax.set_xlabel("Annual Media Budget ($M)", fontsize=14, fontweight="bold")
plt.setp(ax.get_yticklabels(), fontsize=12, fontweight="bold")
plt.setp(ax.get_xticklabels(), fontsize=12, fontweight="bold")
ax.set_title("LBE MMM — Total Media Response Curve (V5e)\n"
             "Steady-state adstock  |  Budget shares held at 2025 mix",
             fontsize=16, fontweight="bold", color=CPX_NAVY)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:.1f}M"))
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:.1f}M"))
year_legend = [Line2D([0], [0], color=style["color"], linestyle=style["linestyle"],
                      linewidth=style["linewidth"], marker=style["marker"], markersize=7,
                      label=f"{year} (${year_points[year]['annual_spend']/1e6:.1f}M annual)")
               for year, style in YEAR_STYLES.items()]
ax.legend(handles=year_legend, fontsize=12, loc="lower right", frameon=False)
ax.grid(False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.text(0.01, 0.01, "* Budget distributed using 2025 actual spend shares. Hill approx via V5e posterior mean params.",
         fontsize=8, color=CPX_DGREY, fontstyle="italic", ha="left")
plt.tight_layout()
save_fig(fig, "lbe_total_media_response_curve.png")
display(fig); plt.close(fig)

# ── PLOT 2: Marginal Return Curve ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(16, 9))
ax.plot(x_annual / 1e6, marginal_mean, color=CPX_RED, linewidth=2.5,
        label="Marginal revenue per $1 additional annual spend")
ax.axhline(0, color="black", linewidth=0.8)
ax.axhline(1.0, color="#999999", linestyle=":", linewidth=1, label="Break-even ($1 return per $1 spend)")

if inflection_idx:
    ax.axvline(inf_spend / 1e6, color=CPX_DGREY, linewidth=1.0, linestyle="--", alpha=0.7)
    ax.annotate(f"Inflection\n${inf_spend/1e6:.1f}M/yr",
                xy=(inf_spend / 1e6, marginal_mean[inflection_idx]),
                xytext=(12, 20), textcoords="offset points",
                fontsize=8, color=CPX_DGREY, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=CPX_DGREY, lw=0.8),
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.85, ec="none"))

ax.axvline(sat_95_spend / 1e6, color=CPX_RED, linewidth=1.0, linestyle=":", alpha=0.8)
ax.annotate(f"95% Saturation\n${sat_95_spend/1e6:.1f}M/yr",
            xy=(sat_95_spend / 1e6, marginal_mean[sat_95_idx]),
            xytext=(-80, 20), textcoords="offset points",
            fontsize=8, color=CPX_RED, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=CPX_RED, lw=0.8),
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.85, ec="none"))

for year, style in YEAR_STYLES.items():
    pts = year_points[year]
    spend_m = pts["annual_spend"] / 1e6
    yr_idx  = np.argmin(np.abs(x_annual - pts["annual_spend"]))
    yr_marg = marginal_mean[yr_idx]
    ax.axvline(spend_m, color=style["color"], linestyle=style["linestyle"],
               linewidth=style["linewidth"], alpha=0.85, zorder=4)
    ax.scatter([spend_m], [yr_marg], color=style["color"], marker=style["marker"], s=90, zorder=6)
    ax.annotate(f"{year}\n${spend_m:.1f}M/yr\n({yr_marg:.2f} rev/$)",
                xy=(spend_m, yr_marg),
                xytext=(10, 15 - list(YEAR_STYLES.keys()).index(year) * 40),
                textcoords="offset points",
                fontsize=8, color=style["color"], fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=style["color"], lw=0.8),
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.85, ec="none"))

ax.set_ylabel("Marginal Revenue per $1 Additional Annual Spend", fontsize=14, fontweight="bold")
ax.set_xlabel("Annual Media Spend ($M)", fontsize=14, fontweight="bold")
plt.setp(ax.get_yticklabels(), fontsize=12, fontweight="bold")
plt.setp(ax.get_xticklabels(), fontsize=12, fontweight="bold")
ax.set_title("LBE MMM — Marginal Return Curve (V5e)\n"
             "Revenue generated per additional $1 of annual spend",
             fontsize=16, fontweight="bold", color=CPX_NAVY)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:.1f}M"))
year_legend = [Line2D([0], [0], color=style["color"], linestyle=style["linestyle"],
                      linewidth=style["linewidth"], marker=style["marker"], markersize=7,
                      label=f"{year} (${year_points[year]['annual_spend']/1e6:.1f}M annual)")
               for year, style in YEAR_STYLES.items()]
ax.legend(handles=year_legend, fontsize=12, loc="upper right", frameon=False)
ax.grid(False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.text(0.01, 0.01, "* Budget distributed using 2025 actual spend shares. Hill approx via V5e posterior mean params.",
         fontsize=8, color=CPX_DGREY, fontstyle="italic", ha="left")
plt.tight_layout()
save_fig(fig, "lbe_marginal_return_curve.png")
display(fig); plt.close(fig)

print(f"\n✓ All response curves complete.")
print(f"  2025 annual spend: ${total_annual_2025/1e6:.2f}M")
print(f"  95% saturation at: ${sat_95_spend/1e6:.1f}M ({sat_95_spend/total_annual_2025:.1f}x 2025)")
if inflection_idx:
    print(f"  Inflection point:  ${inf_spend/1e6:.1f}M ({inf_spend/total_annual_2025:.1f}x 2025)")

# COMMAND ----------

# ══════════════════════════════════════════════════════════════════════════════
# BLOCK 1 — CALIBRATION + TOTAL MEDIA RESPONSE CURVE  (EXH-styled, V5e-compatible)
# ══════════════════════════════════════════════════════════════════════════════
# Model-faithful curvature WITHOUT re-training: per channel we solve an effective
# Hill (half, scale) from TWO anchors already on disk —
#   (1) mean historical spend  -> saved incr_rev
#   (2) 2025 actual spend       -> revenue read from saved weekly m_* arrays
# so the reconstructed response passes through the model's own numbers.
from matplotlib.lines import Line2D
from scipy.optimize import brentq
import matplotlib.ticker as mticker

print("Calibrating V5e response curves from saved artifacts (no re-training)...")

# ── steady-state helpers ──────────────────────────────────────────────────────
def _hill(ad, half):        return ad / (ad + half)
def _steady_ad(s_norm, a):  return s_norm / (1.0 - a)      # geometric steady state

# ── per brand/channel 2025 attributed revenue (2nd anchor), from saved arrays ──
# Compute rev2025 from weekly decomposition arrays, filtered to 2025 weeks.
mask_2025 = d["week_date"].dt.year == 2025
rev2025 = {}
for _b in ["TRR", "PDM"]:
    r = results[_b]
    m = mask_2025.values
    # Awareness incremental revenue (value uplift × txn + rpt × volume uplift)
    aw_wk = ((r["pred_rpt"][m] - np.exp(r["log_pred_rpt"][m] - r["m_aw_rpt"][m])) * r["pred_txn"][m]
             + r["pred_rpt"][m] * (r["pred_txn"][m] - np.exp(r["log_pred_txn"][m] - r["m_aw_txn"][m])))
    # Consideration incremental revenue
    co_wk = ((r["pred_rpt"][m] - np.exp(r["log_pred_rpt"][m] - r["m_co_rpt"][m])) * r["pred_txn"][m]
             + r["pred_rpt"][m] * (r["pred_txn"][m] - np.exp(r["log_pred_txn"][m] - r["m_co_txn"][m])))
    rev2025[_b] = {"aw": aw_wk.sum(), "co": co_wk.sum()}

# ── CALIBRATION: solve effective (half, scale) per brand × channel ────────────
# DEFAULT 1: channel alpha = average of saved rpt/txn adstock alphas.
# DEFAULT 2: two anchors = mean-spend (saved incr_rev) + 2025-spend (saved arrays).
_calib = {}
for _b in ["TRR", "PDM"]:
    r = results[_b]
    _calib[_b] = {}
    for _ch, _sp, _incr, _a_rpt, _a_txn in [
        ("aw", "Awareness",     r["aw_incr_rev"], r["adstock_aw_rpt"], r["adstock_aw_txn"]),
        ("co", "Consideration", r["co_incr_rev"], r["adstock_co_rpt"], r["adstock_co_txn"]),
    ]:
        alpha  = 0.5 * (_a_rpt + _a_txn)
        col    = f"{_b}_{_sp}_spend"
        s_max  = d[col].max();  s_max = s_max if s_max > 0 else 1.0
        s_mean = d[col].mean()
        s_2025 = d.loc[d["week_date"].dt.year == 2025, col].mean()
        rev_mean = _incr
        rev_2025 = rev2025[_b][_ch]

        ad_mean = _steady_ad(s_mean / s_max, alpha)
        ad_2025 = _steady_ad(s_2025 / s_max, alpha)
        ratio   = rev_mean / rev_2025 if rev_2025 > 0 else 1.0

        def _f(half): return (_hill(ad_mean, half) / _hill(ad_2025, half)) - ratio
        try:
            if _f(1e-4) * _f(1e3) > 0:
                raise ValueError("no sign change")
            half  = brentq(_f, 1e-4, 1e3, xtol=1e-8)
            scale = rev_mean / _hill(ad_mean, half)
        except Exception:
            # Robust fallback: fix half at prior mean, solve scale only.
            print(f"  ⚠ {_b} {_ch}: two-anchor solve ill-posed "
                  f"(2025 spend not above mean) — fixing half=0.5, scale-only.")
            half  = 0.5
            scale = rev_mean / _hill(ad_mean, half)
        _calib[_b][_ch] = {"alpha": alpha, "half": half, "scale": scale, "s_max": s_max}

# ── 2025 allocation shares (hold mix constant across the curve) ────────────────
df_2025 = df_all[df_all["year"] == 2025]
aw_share = df_2025["Awareness"].mean() / (df_2025["Awareness"].mean() + df_2025["Consideration"].mean())
co_share = 1.0 - aw_share
total_annual_2025 = (df_2025["Awareness"].mean() + df_2025["Consideration"].mean()) * 52

# brand split of each channel's 2025 spend (to route budget within a channel)
_brand_ch_share = {}
for _ch, _sp in [("aw", "Awareness"), ("co", "Consideration")]:
    tot = sum(d.loc[d["week_date"].dt.year == 2025, f"{b}_{_sp}_spend"].mean() for b in ["TRR", "PDM"])
    _brand_ch_share[_ch] = {
        b: (d.loc[d["week_date"].dt.year == 2025, f"{b}_{_sp}_spend"].mean() / tot if tot > 0 else 0.5)
        for b in ["TRR", "PDM"]
    }

def combined_rev(annual_budget):
    """Combined incremental revenue (TRR+PDM) at annual_budget, 2025 mix, calibrated Hill."""
    weekly = annual_budget / 52.0
    total = 0.0
    for _ch, _sh in [("aw", aw_share), ("co", co_share)]:
        for b in ["TRR", "PDM"]:
            c   = _calib[b][_ch]
            w_b = weekly * _sh * _brand_ch_share[_ch][b]
            ad  = _steady_ad(w_b / c["s_max"], c["alpha"])
            total += c["scale"] * _hill(ad, c["half"])
    return total

# ── curve + key points ────────────────────────────────────────────────────────
x_annual   = np.linspace(52 * 100, total_annual_2025 * 10, 500)
curve_mean = np.array([combined_rev(b) for b in x_annual])
marginal_mean = np.gradient(curve_mean, x_annual)

# efficiency breakeven: marginal return == $1
_roi1_idx    = np.argmin(np.abs(marginal_mean - 1.0))
roi1_spend   = x_annual[_roi1_idx]; roi1_rev = curve_mean[_roi1_idx]

year_points = {}
for year in [2023, 2024, 2025]:
    dfy = df_all[df_all["year"] == year]
    ab  = dfy["Awareness"].sum() + dfy["Consideration"].sum()
    year_points[year] = {"annual_spend": ab, "annual_rev": combined_rev(ab)}

YEAR_MARKER_STYLES = {
    2023: {"color": CPX_DGREY, "marker": "o", "linestyle": "--", "linewidth": 1.2},
    2024: {"color": CPX_NAVY,  "marker": "s", "linestyle": "-.", "linewidth": 1.2},
    2025: {"color": CPX_RED,   "marker": "D", "linestyle": "-",  "linewidth": 1.5},
}

# ── PLOT 1: Total Media Response Curve (EXH-styled) ───────────────────────────
fig, ax = plt.subplots(figsize=(16, 9))
fig.patch.set_facecolor("white"); ax.set_facecolor("white")
ax.plot(x_annual / 1e6, curve_mean / 1e6, color=CPX_NAVY, linewidth=2.5)

# gold efficiency-breakeven hero marker
ax.axvline(roi1_spend / 1e6, color=CPX_GOLD, linewidth=1.8, linestyle="--", alpha=0.9, zorder=5)
ax.scatter([roi1_spend / 1e6], [roi1_rev / 1e6], color=CPX_GOLD, s=120, zorder=6,
           marker="*", edgecolors="white", linewidths=0.8)
ax.annotate(f"Efficiency Breakeven\nMarginal Return = $1 per $1\n${roi1_spend/1e6:.1f}M / yr",
            xy=(roi1_spend / 1e6, roi1_rev / 1e6), xytext=(-12, 38), textcoords="offset points",
            fontsize=10, color=CPX_GOLD, fontweight="bold", ha="right",
            arrowprops=dict(arrowstyle="->", color=CPX_GOLD, lw=0.8),
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.9, ec="none"))

for year, style in YEAR_MARKER_STYLES.items():
    pts = year_points[year]; sm = pts["annual_spend"] / 1e6; rm = pts["annual_rev"] / 1e6
    ax.axvline(sm, color=style["color"], linestyle=style["linestyle"],
               linewidth=style["linewidth"], alpha=0.85, zorder=4)
    ax.scatter([sm], [rm], color=style["color"], marker=style["marker"], s=100, zorder=7,
               edgecolors="white", linewidths=1.0)

ax.set_title("LBE MMM — Total Media Response Curve (V5e)\n"
             "Steady-state adstock  |  Budget shares held at 2025 mix",
             fontsize=14, fontweight="bold", color="black", pad=16)
ax.set_xlabel("Annual Media Budget ($M)", fontsize=14, fontweight="bold")
ax.set_ylabel("Annual Media Revenue Contribution ($M)", fontsize=14, fontweight="bold")
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:.0f}M"))
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:.1f}M"))
plt.setp(ax.get_xticklabels(), fontsize=12, fontweight="bold")
plt.setp(ax.get_yticklabels(), fontsize=12, fontweight="bold")
ax.grid(False)

legend_handles = [
    Line2D([0], [0], color=s["color"], marker=s["marker"], linestyle=s["linestyle"],
           linewidth=s["linewidth"], markersize=7,
           label=f"{yr} (${year_points[yr]['annual_spend']/1e6:.1f}M annual)")
    for yr, s in YEAR_MARKER_STYLES.items()
]
ax.legend(handles=legend_handles, loc="lower right", fontsize=13, frameon=True,
          facecolor="white", edgecolor="#CCCCCC", framealpha=0.95, markerscale=1.6,
          borderpad=1.0, labelspacing=0.8, handlelength=2.8,
          title="Annual Spend Reference", title_fontsize=12)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
fig.text(0.02, 0.01, "* Budget distributed using 2025 actual spend shares. "
         "Effective Hill (half, scale) calibrated per channel to saved V5e incr-rev anchors.",
         fontsize=7.5, color=CPX_DGREY, style="italic")
plt.tight_layout(rect=[0, 0.03, 1, 1])
save_fig(fig, "lbe_total_media_response_curve.png")
display(fig); plt.close(fig)

# ══════════════════════════════════════════════════════════════════════════════
# BLOCK 2 — MARGINAL RETURN CURVE  (EXH-styled)
# ══════════════════════════════════════════════════════════════════════════════
fig2, ax2 = plt.subplots(figsize=(16, 9))
fig2.patch.set_facecolor("white"); ax2.set_facecolor("white")
ax2.plot(x_annual / 1e6, marginal_mean, color=CPX_NAVY, linewidth=2.5)

# red inefficient-spending zone (marginal < $1)
ax2.axhspan(0, 1, alpha=0.12, color=CPX_RED, zorder=0)
ax2.axhline(1, color=CPX_RED, lw=1.8, alpha=0.85, zorder=6)
ax2.text(x_annual[-1] / 1e6 * 0.60, 0.5,
         "Inefficient Spending Zone\nMarginal Return < $1 per $1",
         fontsize=10, color=CPX_RED, fontweight="bold", alpha=0.9, va="center", ha="center",
         bbox=dict(boxstyle="round,pad=0.35", fc="white", alpha=0.8, ec=CPX_RED, lw=0.6))

# gold ROI=1 crossover drop
ax2.vlines(roi1_spend / 1e6, ymin=0, ymax=marginal_mean[_roi1_idx], color=CPX_GOLD,
           linewidth=2.2, linestyle="--", alpha=0.9, zorder=5)
ax2.scatter([roi1_spend / 1e6], [marginal_mean[_roi1_idx]], color=CPX_GOLD, s=140, zorder=8,
            marker="*", edgecolors="white", linewidths=0.8)
ax2.annotate(f"${roi1_spend/1e6:.1f}M", xy=(roi1_spend / 1e6, 0), xytext=(5, 6),
             textcoords="offset points", fontsize=10, color=CPX_GOLD, fontweight="bold", ha="left")

for year, style in YEAR_MARKER_STYLES.items():
    pts = year_points[year]; idx = np.argmin(np.abs(x_annual - pts["annual_spend"]))
    ax2.axvline(pts["annual_spend"] / 1e6, color=style["color"], linestyle=style["linestyle"],
                linewidth=2.2, alpha=0.85, zorder=4)
    ax2.scatter([pts["annual_spend"] / 1e6], [marginal_mean[idx]], color=style["color"],
                marker=style["marker"], s=100, zorder=7, edgecolors="white", linewidths=1.0)

ax2.set_title("LBE MMM — Marginal Return per $1 Spent (V5e)\n"
              "Incremental revenue per additional dollar",
              fontsize=14, fontweight="bold", color="black", pad=16)
ax2.set_xlabel("Annual Media Budget ($M)", fontsize=14, fontweight="bold")
ax2.set_ylabel("Incremental Revenue per $1", fontsize=14, fontweight="bold")
ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:.0f}M"))
plt.setp(ax2.get_xticklabels(), fontsize=12, fontweight="bold")
plt.setp(ax2.get_yticklabels(), fontsize=12, fontweight="bold")
ax2.set_ylim(bottom=0); ax2.grid(False)

legend_handles_m = [
    Line2D([0], [0], color=s["color"], marker=s["marker"], linestyle=s["linestyle"],
           linewidth=s["linewidth"], markersize=7,
           label=f"{yr} (${year_points[yr]['annual_spend']/1e6:.1f}M annual)")
    for yr, s in YEAR_MARKER_STYLES.items()
]
ax2.legend(handles=legend_handles_m, loc="upper right", fontsize=13, frameon=True,
           facecolor="white", edgecolor="#CCCCCC", framealpha=0.95, markerscale=1.6,
           borderpad=1.0, labelspacing=0.8, handlelength=2.8,
           title="Annual Spend Reference", title_fontsize=12)
ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)
fig2.text(0.02, 0.01, "* Budget distributed using 2025 actual spend shares. "
          "Effective Hill (half, scale) calibrated per channel to saved V5e incr-rev anchors.",
          fontsize=7.5, color=CPX_DGREY, style="italic")
plt.tight_layout(rect=[0, 0.03, 1, 1])
save_fig(fig2, "lbe_marginal_return_curve.png")
display(fig2); plt.close(fig2)

print(f"\n✓ Response + marginal curves complete (calibrated, no re-training).")
print(f"  2025 annual spend:      ${total_annual_2025/1e6:.2f}M")
print(f"  Efficiency breakeven:   ${roi1_spend/1e6:.1f}M  ({roi1_spend/total_annual_2025:.1f}x 2025)")

# COMMAND ----------

# DBTITLE 1,Per-LOB Annual Spend & Revenue Contribution
# ── Per-LOB Annual Spend & Revenue Contribution — grouped bars by year ─────────
# One chart per LOB (TRR, PDM). Uses V5e brand-level ROI as revenue-per-$ rate.
# Single Y-axis in $M for both Spend and Revenue Contribution.
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker

FUNNEL_COLOR = {"Awareness": CPX_NAVY, "Consideration": CPX_RED}
FUNNELS      = ["Awareness", "Consideration"]
CPX_LGREY_BAR = "#CCCCCC"

# ── LOB-level spend column map ────────────────────────────────────────────────
funnel_lob_cols = {
    ("Awareness",     "TRR"): ["TRR_Awareness_spend"],
    ("Consideration", "TRR"): ["TRR_Consideration_spend"],
    ("Awareness",     "PDM"): ["PDM_Awareness_spend"],
    ("Consideration", "PDM"): ["PDM_Consideration_spend"],
}

# ── Revenue-per-dollar from V5e brand ROI ─────────────────────────────────────
fl_rate = {
    ("Awareness",     "TRR"): results["TRR"]["roi_aw"],
    ("Consideration", "TRR"): results["TRR"]["roi_co"],
    ("Awareness",     "PDM"): results["PDM"]["roi_aw"],
    ("Consideration", "PDM"): results["PDM"]["roi_co"],
}

years_avail = sorted(y for y in df_all["year"].dropna().unique().astype(int) if y < 2026)

# ── Annual spend per (funnel, LOB, year) ──────────────────────────────────────
fl_yr_spend = {}
for (funnel, lob), cols in funnel_lob_cols.items():
    key = (funnel, lob)
    fl_yr_spend[key] = {
        yr: float(df_all[df_all["year"] == yr][cols].sum().sum())
        for yr in years_avail
    }

all_lobs = sorted({lob for (_, lob) in fl_rate})

n_yrs = len(years_avail)
bar_w = 0.35
gap   = 0.15
group = bar_w * 2 + gap + 0.2

for lob in all_lobs:
    active_funnels = [f for f in FUNNELS if (f, lob) in fl_rate and
                      any(fl_yr_spend.get((f, lob), {}).get(yr, 0) > 0 for yr in years_avail)]
    if not active_funnels:
        continue

    spend_by_f   = {f: np.array([fl_yr_spend.get((f, lob), {}).get(yr, 0) / 1e6
                                  for yr in years_avail]) for f in active_funnels}
    contrib_by_f = {f: np.array([fl_yr_spend.get((f, lob), {}).get(yr, 0) * fl_rate[(f, lob)] / 1e6
                                  for yr in years_avail]) for f in active_funnels}

    total_spend_yr   = sum(spend_by_f[f]   for f in active_funnels)
    total_contrib_yr = sum(contrib_by_f[f] for f in active_funnels)

    x_spend   = np.array([i * group for i in range(n_yrs)])
    x_contrib = x_spend + bar_w + gap

    fig, ax = plt.subplots(figsize=(11, 9))
    fig.patch.set_facecolor(WHITE)

    # ── Spend bars (solid) ────────────────────────────────────────────────────
    bottoms_s = np.zeros(n_yrs)
    for f in active_funnels:
        ax.bar(x_spend, spend_by_f[f], bar_w, bottom=bottoms_s,
               color=FUNNEL_COLOR[f], alpha=0.9, edgecolor="white", lw=0.4)
        bottoms_s += spend_by_f[f]

    # ── Revenue Contribution bars (hatched) ───────────────────────────────────
    bottoms_c = np.zeros(n_yrs)
    for f in active_funnels:
        ax.bar(x_contrib, contrib_by_f[f], bar_w, bottom=bottoms_c,
               color=FUNNEL_COLOR[f], alpha=0.55, edgecolor="white", lw=0.4, hatch="///")
        bottoms_c += contrib_by_f[f]

    # ── Top labels ────────────────────────────────────────────────────────────
    for i in range(n_yrs):
        if bottoms_s[i] > 0:
            ax.annotate(f"${bottoms_s[i]:.2f}M",
                        xy=(x_spend[i], bottoms_s[i]),
                        xytext=(0, 4), textcoords="offset points",
                        ha="center", fontsize=11, color=CPX_DGREY, fontweight="bold")
        if bottoms_c[i] > 0:
            ax.annotate(f"${bottoms_c[i]:.2f}M",
                        xy=(x_contrib[i], bottoms_c[i]),
                        xytext=(0, 4), textcoords="offset points",
                        ha="center", fontsize=11, color=CPX_DGREY, fontweight="bold")

    x_mid = (x_spend + x_contrib) / 2
    ax.set_xticks(x_mid)
    ax.set_xticklabels([str(y) for y in years_avail], fontsize=14, fontweight="bold")
    ax.set_xlim(-0.3, x_contrib[-1] + bar_w + 0.3)

    y_max = max(bottoms_s.max(), bottoms_c.max(), 0.01) * 1.3
    ax.set_ylabel("$M", fontsize=14, fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:.2f}M"))
    ax.set_ylim(bottom=0, top=y_max)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.setp(ax.get_yticklabels(), fontsize=12, fontweight="bold")
    ax.grid(False)

    handles = [mpatches.Patch(color=FUNNEL_COLOR[f], label=f) for f in active_funnels]
    handles += [
        mpatches.Patch(facecolor="#999999", alpha=0.9, label="Spend"),
        mpatches.Patch(facecolor="#999999", alpha=0.55, hatch="///", label="Revenue Contrib"),
    ]
    ax.legend(handles=handles, fontsize=12,
              loc="upper right", frameon=False)

    ax.set_title(f"{lob} — Annual Spend vs Revenue Contribution ($M)",
                 fontsize=14, fontweight="bold", color=CPX_NAVY, pad=10)

    plt.tight_layout()
    save_fig(fig, f"lbe_lob_{lob.lower()}_spend_vs_contrib.png")
    display(fig)
    plt.close(fig)

    # ── Summary Table ─────────────────────────────────────────────────────────
    tbl_rows = []
    for yr_idx, yr in enumerate(years_avail):
        row = {"Year": str(yr)}
        for f in active_funnels:
            row[f"{f} Spend (M)"] = round(spend_by_f[f][yr_idx], 3)
            row[f"{f} Contrib (M)"] = round(contrib_by_f[f][yr_idx], 3)
        row["Total Spend (M)"] = round(total_spend_yr[yr_idx], 3)
        row["Total Contrib (M)"] = round(total_contrib_yr[yr_idx], 3)
        tbl_rows.append(row)
    rate_row = {"Year": "ROI (rev/$)"}
    for f in active_funnels:
        rate_row[f"{f} Spend (M)"] = ""
        rate_row[f"{f} Contrib (M)"] = f"{fl_rate[(f, lob)]:.2f}×"
    rate_row["Total Spend (M)"] = ""
    rate_row["Total Contrib (M)"] = ""
    tbl_rows.append(rate_row)

    display(pd.DataFrame(tbl_rows).astype(str))

# COMMAND ----------

# DBTITLE 1,Tier-Level Spend & Media-Attributed Revenue
# COMMAND ----------

# ── TIER-LEVEL SPEND & MEDIA-ATTRIBUTED REVENUE ───────────────────────────────
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from IPython.display import display as ipy_display

FUNNELS      = ["Awareness", "Consideration"]
BRANDS       = ["TRR", "PDM"]
YEARS        = [2023, 2024, 2025]
FUNNEL_COLOR = {"Awareness": CPX_NAVY, "Consideration": CPX_RED}
YEAR_COLORS  = {2023: CPX_DGREY, 2024: CPX_NAVY, 2025: CPX_RED}
brand_tiers  = {"TRR": ["HPL", "LPL", "NL"], "PDM": ["OL", "NL"]}

# ── Rebuild alloc_df if not in memory (standalone from cell 17) ───────────────
if 'alloc_df' not in dir():
    alloc_df = d[["week_date"]].copy()
    alloc_df["year"] = alloc_df["week_date"].dt.year
    _bsm = {
        ("TRR", "Awareness"):     (["HPL","LPL","NL"], ["Awareness_TRR_National","Awareness_TRR_Other"]),
        ("TRR", "Consideration"): (["HPL","LPL","NL"], ["Consideration_TRR_National","Consideration_TRR_Other"]),
        ("PDM", "Awareness"):     (["OL","NL"],        ["Awareness_PDM_National","Awareness_PDM_Other"]),
        ("PDM", "Consideration"): (["OL","NL"],        ["Consideration_PDM_National","Consideration_PDM_Other"]),
    }
    for (brand, stage), (tiers_list, pooled_cols) in _bsm.items():
        pooled = d[[c for c in pooled_cols if c in d.columns]].sum(axis=1)
        n_active = pd.Series(len(tiers_list), index=d.index)
        if "NL" in tiers_list:
            n_active[alloc_df["year"] == 2023] = len(tiers_list) - 1
        share = pooled / n_active
        for tier in tiers_list:
            obs_col = f"{stage}_{brand}_{tier}"
            base = d[obs_col] if obs_col in d.columns else pd.Series(0.0, index=d.index)
            alloc = share.copy()
            if tier == "NL":
                alloc[alloc_df["year"] == 2023] = 0.0
            alloc_df[f"effective_{brand}_{stage}_{tier}"] = base + alloc
    print("✓ alloc_df rebuilt from raw data (standalone mode)")

# ── Step 1: Annual effective spend per (brand, funnel, tier, year) ────────────
eff_spend = {}
for brand, tiers in brand_tiers.items():
    for funnel in FUNNELS:
        for tier in tiers:
            col = f"effective_{brand}_{funnel}_{tier}"
            if col not in alloc_df.columns:
                continue
            for yr in YEARS:
                eff_spend[(brand, funnel, tier, yr)] = float(
                    alloc_df.loc[alloc_df["year"] == yr, col].sum()
                )

# ── Step 2: Year-specific counterfactual incremental revenue ─────────────────
# Use the V5e counterfactual decomp to get year-specific incremental revenue
# per brand per funnel, then allocate to tiers by within-year spend proportions.
def _brand_weekly_incr(r):
    """Remove media log-contribution from both RPT and TXN sub-models."""
    wk_aw = (
        (r["pred_rpt"] - np.exp(r["log_pred_rpt"] - r["m_aw_rpt"])) * r["pred_txn"]
        + r["pred_rpt"] * (r["pred_txn"] - np.exp(r["log_pred_txn"] - r["m_aw_txn"]))
    )
    wk_co = (
        (r["pred_rpt"] - np.exp(r["log_pred_rpt"] - r["m_co_rpt"])) * r["pred_txn"]
        + r["pred_rpt"] * (r["pred_txn"] - np.exp(r["log_pred_txn"] - r["m_co_txn"]))
    )
    return wk_aw, wk_co

# Compute year-specific incremental revenue per brand per funnel
brand_funnel_yr_incr = {}  # (brand, funnel, year) -> incr_rev
for brand in BRANDS:
    wk_aw, wk_co = _brand_weekly_incr(results[brand])
    decomp_tmp = d[["week_date"]].copy()
    decomp_tmp["year"] = decomp_tmp["week_date"].dt.year
    decomp_tmp["incr_Awareness"] = wk_aw
    decomp_tmp["incr_Consideration"] = wk_co
    for yr in YEARS:
        mask = decomp_tmp["year"] == yr
        brand_funnel_yr_incr[(brand, "Awareness", yr)] = decomp_tmp.loc[mask, "incr_Awareness"].sum()
        brand_funnel_yr_incr[(brand, "Consideration", yr)] = decomp_tmp.loc[mask, "incr_Consideration"].sum()

# Allocate year-specific incr revenue to tiers by within-year spend proportions
tier_rev = {}
for brand, tiers in brand_tiers.items():
    for funnel in FUNNELS:
        for yr in YEARS:
            yr_incr = brand_funnel_yr_incr[(brand, funnel, yr)]
            yr_total_eff = max(
                sum(eff_spend.get((brand, funnel, t, yr), 0.0) for t in tiers),
                1e-9
            )
            for tier in tiers:
                w = eff_spend.get((brand, funnel, tier, yr), 0.0) / yr_total_eff
                tier_rev[(brand, funnel, tier, yr)] = yr_incr * w

print("Year-specific counterfactual ROI by brand/funnel:")
for brand in BRANDS:
    for funnel in FUNNELS:
        vals = []
        for yr in YEARS:
            s = sum(eff_spend.get((brand, funnel, t, yr), 0.0) for t in brand_tiers[brand])
            r_val = brand_funnel_yr_incr[(brand, funnel, yr)]
            roi = r_val / s if s > 0 else 0
            vals.append(f"{yr}={roi:.2f}x")
        print(f"  {brand} {funnel}: {', '.join(vals)}")

# ── Accessors ($M) ────────────────────────────────────────────────────────────
def sp(brand, funnel, tier, yr):
    return eff_spend.get((brand, funnel, tier, yr), 0.0) / 1e6

def rv(brand, funnel, tier, yr):
    return tier_rev.get((brand, funnel, tier, yr), 0.0) / 1e6

fmt_M = mticker.FuncFormatter(lambda v, _: f"${v:.2f}M")

# Shared x-axis layout: TRR tiers then PDM tiers
all_brand_tiers = [(b, t) for b in BRANDS for t in brand_tiers[b]]
n_bt       = len(all_brand_tiers)
trr_div    = len(brand_tiers["TRR"]) - 0.5
trr_center = (len(brand_tiers["TRR"]) - 1) / 2
pdm_center = len(brand_tiers["TRR"]) + (len(brand_tiers["PDM"]) - 1) / 2

def add_brand_dividers(ax, y_top):
    ax.axvline(trr_div, color="#CCCCCC", lw=1.2, linestyle="--")
    for bx, blabel in [(trr_center, "TRR"), (pdm_center, "PDM")]:
        ax.text(bx, y_top * 0.97, blabel, ha="center", va="top",
                fontsize=11, fontweight="bold", color="black")

def style_ax(ax, y_top):
    ax.set_ylim(0, y_top)
    ax.yaxis.set_major_formatter(fmt_M)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.setp(ax.get_yticklabels(), fontsize=12, fontweight="bold")
    ax.grid(False)

# ═══════════════════════════════════════════════════════════════════════════════
# CHART 1 — Spend by Tier, both brands, one chart per year, funnel stacked
# ═══════════════════════════════════════════════════════════════════════════════
for yr in YEARS:
    fig, ax = plt.subplots(figsize=(11, 7))
    fig.patch.set_facecolor(WHITE)

    x       = np.arange(n_bt)
    bar_w   = 0.5
    bottoms = np.zeros(n_bt)

    for funnel in FUNNELS:
        heights = np.array([sp(b, funnel, t, yr) for b, t in all_brand_tiers])
        ax.bar(x, heights, bar_w, bottom=bottoms,
               color=FUNNEL_COLOR[funnel], alpha=0.9,
               edgecolor=WHITE, lw=0.4, label=funnel)
        bottoms += heights

    for i, total in enumerate(bottoms):
        if total > 0:
            ax.annotate(f"${total:.2f}M", xy=(x[i], total),
                        xytext=(0, 4), textcoords="offset points",
                        ha="center", fontsize=12, color=CPX_DGREY, fontweight="bold")

    y_top = bottoms.max() * 1.3
    style_ax(ax, y_top)
    add_brand_dividers(ax, y_top)
    ax.set_xticks(x)
    ax.set_xticklabels([t for _, t in all_brand_tiers], fontsize=14, fontweight="bold")
    ax.set_ylabel("Media Spend ($M)", fontsize=14, fontweight="bold")
    ax.set_title(f"LBE Spend by Tier — {yr}", fontsize=14, fontweight="bold", color=CPX_NAVY)
    ax.legend(fontsize=12, loc="upper left", frameon=False)
    plt.tight_layout()
    save_fig(fig, f"lbe_tier_spend_{yr}.png")
    display(fig)
    plt.close(fig)

    # Table
    tbl_rows = []
    for b, t in all_brand_tiers:
        row = {"Brand": b, "Tier": t}
        for funnel in FUNNELS:
            row[f"{funnel} ($M)"] = round(sp(b, funnel, t, yr), 3)
        row["Total ($M)"] = round(sum(sp(b, f, t, yr) for f in FUNNELS), 3)
        tbl_rows.append(row)
    print(f"\n{'═'*55}\n  Spend by Tier — {yr}\n{'═'*55}")
    ipy_display(pd.DataFrame(tbl_rows))


# ═══════════════════════════════════════════════════════════════════════════════
# CHART 2 — Spend vs Revenue by Tier, one chart per year
# Same layout as Chart 1 but spend bar + revenue bar side by side per tier
# Both funnel stacked, single Y-axis in $M
# ═══════════════════════════════════════════════════════════════════════════════
bar_w = 0.3
gap   = 0.08
x_sp  = np.arange(n_bt) - (bar_w + gap) / 2
x_rv  = np.arange(n_bt) + (bar_w + gap) / 2

for yr in YEARS:
    fig, ax = plt.subplots(figsize=(11, 7))
    fig.patch.set_facecolor(WHITE)

    bottoms_s = np.zeros(n_bt)
    bottoms_r = np.zeros(n_bt)

    for funnel in FUNNELS:
        sp_vals = np.array([sp(b, funnel, t, yr) for b, t in all_brand_tiers])
        rv_vals = np.array([rv(b, funnel, t, yr) for b, t in all_brand_tiers])

        ax.bar(x_sp, sp_vals, bar_w, bottom=bottoms_s,
               color=FUNNEL_COLOR[funnel], alpha=0.9,
               edgecolor=WHITE, lw=0.4)
        ax.bar(x_rv, rv_vals, bar_w, bottom=bottoms_r,
               color=FUNNEL_COLOR[funnel], alpha=0.5,
               edgecolor=WHITE, lw=0.4, hatch="///")

        bottoms_s += sp_vals
        bottoms_r += rv_vals

    for i in range(n_bt):
        if bottoms_s[i] > 0:
            ax.annotate(f"${bottoms_s[i]:.2f}M",
                        xy=(x_sp[i], bottoms_s[i]),
                        xytext=(0, 4), textcoords="offset points",
                        ha="center", fontsize=12, color=CPX_DGREY, fontweight="bold")
        if bottoms_r[i] > 0:
            ax.annotate(f"${bottoms_r[i]:.2f}M",
                        xy=(x_rv[i], bottoms_r[i]),
                        xytext=(0, 4), textcoords="offset points",
                        ha="center", fontsize=12, color=CPX_DGREY, fontweight="bold")

    y_top = max(bottoms_s.max(), bottoms_r.max(), 0.01) * 1.3
    style_ax(ax, y_top)
    add_brand_dividers(ax, y_top)

    ax.set_xticks(np.arange(n_bt))
    ax.set_xticklabels([t for _, t in all_brand_tiers], fontsize=14, fontweight="bold")
    ax.set_ylabel("Media Spend ($M)", fontsize=14, fontweight="bold")

    funnel_handles = [mpatches.Patch(color=FUNNEL_COLOR[f], label=f) for f in FUNNELS]
    type_handles   = [
        mpatches.Patch(facecolor="#999999", alpha=0.9, label="Spend"),
        mpatches.Patch(facecolor="#999999", alpha=0.5, hatch="///", label="Rev Contrib"),
    ]
    ax.legend(handles=funnel_handles + type_handles,
              fontsize=12, loc="upper left", frameon=False)

    ax.set_title(f"LBE Spend vs Media-Attributed Revenue by Tier — {yr}",
                 fontsize=14, fontweight="bold", color=CPX_NAVY)
    plt.tight_layout()
    save_fig(fig, f"lbe_tier_spend_vs_rev_{yr}.png")
    display(fig)
    plt.close(fig)

    # Table
    tbl_rows = []
    for b, t in all_brand_tiers:
        row = {"Brand": b, "Tier": t}
        for funnel in FUNNELS:
            row[f"{funnel} Spend ($M)"] = round(sp(b, funnel, t, yr), 3)
            row[f"{funnel} Rev ($M)"]   = round(rv(b, funnel, t, yr), 3)
        row["Total Spend ($M)"] = round(sum(sp(b, f, t, yr) for f in FUNNELS), 3)
        row["Total Rev ($M)"]   = round(sum(rv(b, f, t, yr) for f in FUNNELS), 3)
        tbl_rows.append(row)
    rate_row = {"Brand": "ROI", "Tier": ""}
    for funnel in FUNNELS:
        s = sum(sp(b, f, t, yr) for b, t in all_brand_tiers for f in [funnel])
        r = sum(rv(b, f, t, yr) for b, t in all_brand_tiers for f in [funnel])
        rate_row[f"{funnel} Spend ($M)"] = ""
        rate_row[f"{funnel} Rev ($M)"]   = f"{r/s:.2f}×" if s > 0 else "—"
    rate_row["Total Spend ($M)"] = ""
    rate_row["Total Rev ($M)"]   = ""
    tbl_rows.append(rate_row)
    print(f"\n{'═'*55}\n  Spend vs Revenue by Tier — {yr}\n{'═'*55}")
    ipy_display(pd.DataFrame(tbl_rows))
# ═══════════════════════════════════════════════════════════════════════════════
# CHART 3 — Spend vs Revenue by Tier, one chart per brand
# Years are outer groups, tiers within each year,
# each tier has spend bar + revenue bar side by side, both funnel stacked
# Single Y-axis in $M
# ═══════════════════════════════════════════════════════════════════════════════
bar_w    = 0.11
gap      = 0.05   # gap between spend and rev within a tier
pair_w   = bar_w * 2 + gap
tier_gap = 0.08   # gap between tiers within a year
year_gap = 0.25   # gap between year groups

for brand in BRANDS:
    tiers   = brand_tiers[brand]
    n_tiers = len(tiers)
    n_years = len(YEARS)

    year_w = n_tiers * pair_w + (n_tiers - 1) * tier_gap

    fig, ax = plt.subplots(figsize=(max(11, n_years * (year_w + year_gap) * 5 + 2), 7))
    fig.patch.set_facecolor(WHITE)

    year_centers = []

    for yi, yr in enumerate(YEARS):
        base_x = yi * (year_w + year_gap)
        year_centers.append(base_x + year_w / 2)

        for ti, tier in enumerate(tiers):
            x_sp = base_x + ti * (pair_w + tier_gap)
            x_rv = x_sp + bar_w + gap

            bottoms_s = 0.0
            bottoms_r = 0.0
            for funnel in FUNNELS:
                sp_val = sp(brand, funnel, tier, yr)
                rv_val = rv(brand, funnel, tier, yr)

                ax.bar(x_sp, sp_val, bar_w, bottom=bottoms_s,
                       color=FUNNEL_COLOR[funnel], alpha=0.9,
                       edgecolor=WHITE, lw=0.3)
                ax.bar(x_rv, rv_val, bar_w, bottom=bottoms_r,
                       color=FUNNEL_COLOR[funnel], alpha=0.5,
                       edgecolor=WHITE, lw=0.3, hatch="///")

                bottoms_s += sp_val
                bottoms_r += rv_val

            # Tier label under each pair
            ax.text((x_sp + x_rv) / 2, -0.012, tier,
                    ha="center", va="top", fontsize=14,
                    color=CPX_DGREY, fontweight="bold",
                    transform=ax.get_xaxis_transform())

            if bottoms_s > 0:
                ax.annotate(f"${bottoms_s:.2f}M",
                            xy=(x_sp, bottoms_s),
                            xytext=(0, 3), textcoords="offset points",
                            ha="center", fontsize=14, color=CPX_DGREY, fontweight="bold")
            if bottoms_r > 0:
                ax.annotate(f"${bottoms_r:.2f}M",
                            xy=(x_rv, bottoms_r),
                            xytext=(0, 3), textcoords="offset points",
                            ha="center", fontsize=14, color=CPX_DGREY, fontweight="bold")

        # Year divider (skip after last year)
        if yi < n_years - 1:
            div_x = base_x + year_w + year_gap / 2
            ax.axvline(div_x, color="#CCCCCC", lw=1.2, linestyle="--")

    ax.set_xticks([])
    all_sp = [sum(sp(brand, f, t, yr) for f in FUNNELS) for t in tiers for yr in YEARS]
    all_rv = [sum(rv(brand, f, t, yr) for f in FUNNELS) for t in tiers for yr in YEARS]
    y_top = max(max(all_sp), max(all_rv), 0.01) * 1.35
    ax.set_ylim(0, y_top)

    for yi, yr in enumerate(YEARS):
        ax.text(year_centers[yi], y_top * 0.82, str(yr),
                ha="center", va="top", fontsize=14,
                fontweight="bold", color=YEAR_COLORS[yr])

    ax.set_xlim(-0.2, n_years * (year_w + year_gap) - year_gap + 0.2)
    ax.yaxis.set_major_formatter(fmt_M)
    ax.set_ylabel("Spend & Revenue ($M)", fontsize=14, fontweight="bold", color=CPX_DGREY)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.setp(ax.get_yticklabels(), fontsize=14, fontweight="bold")
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.2, color="#CCCCCC")

    funnel_handles = [mpatches.Patch(color=FUNNEL_COLOR[f], label=f) for f in FUNNELS]
    year_handles   = [mpatches.Patch(facecolor=YEAR_COLORS[yr], label=str(yr)) for yr in YEARS]
    type_handles   = [
        mpatches.Patch(facecolor="#999999", alpha=0.9, label="Spend"),
        mpatches.Patch(facecolor="#999999", alpha=0.5, hatch="///", label="Rev Contrib"),
    ]
    ax.legend(handles=funnel_handles + year_handles + type_handles,
              fontsize=14, loc="upper right", framealpha=0.9,
              edgecolor="#CCCCCC", ncol=3)

    ax.set_title(f"{brand} — Spend vs Revenue by Year & Tier ($M)",
                 fontsize=14, fontweight="bold", color=CPX_NAVY)
    plt.tight_layout()
    save_fig(fig, f"lbe_tier_{brand.lower()}_spend_vs_rev_by_year.png")
    display(fig)
    plt.close(fig)

    # Table — one per year, ROI footer
    for yr in YEARS:
        tbl_rows = []
        for tier in tiers:
            row = {"Tier": tier}
            for funnel in FUNNELS:
                row[f"{funnel} Spend ($M)"] = round(sp(brand, funnel, tier, yr), 3)
                row[f"{funnel} Rev ($M)"]   = round(rv(brand, funnel, tier, yr), 3)
            row["Total Spend ($M)"] = round(sum(sp(brand, f, tier, yr) for f in FUNNELS), 3)
            row["Total Rev ($M)"]   = round(sum(rv(brand, f, tier, yr) for f in FUNNELS), 3)
            tbl_rows.append(row)
        rate_row = {"Tier": "ROI"}
        for funnel in FUNNELS:
            s = sum(sp(brand, funnel, t, yr) for t in tiers)
            r = sum(rv(brand, funnel, t, yr) for t in tiers)
            rate_row[f"{funnel} Spend ($M)"] = ""
            rate_row[f"{funnel} Rev ($M)"]   = f"{r/s:.2f}×" if s > 0 else "—"
        rate_row["Total Spend ($M)"] = ""
        rate_row["Total Rev ($M)"]   = ""
        tbl_rows.append(rate_row)
        print(f"\n{'═'*55}\n  {brand} — Spend vs Revenue by Tier — {yr}\n{'═'*55}")
        ipy_display(pd.DataFrame(tbl_rows))

# COMMAND ----------

# DBTITLE 1,Tier ROI Comparison by Year
# ═══════════════════════════════════════════════════════════════════════════════
# TIER ROI COMPARISON BY YEAR — One chart per location tier
# X-axis = Years (2023, 2024, 2025)
# Bars = Spend (stacked by funnel), ROI curve on secondary y-axis
# ═══════════════════════════════════════════════════════════════════════════════
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import matplotlib.lines as mlines

YEAR_COLORS  = {2023: CPX_DGREY, 2024: CPX_NAVY, 2025: CPX_RED}

for brand in BRANDS:
    tiers = brand_tiers[brand]
    for tier in tiers:
        x_base  = np.arange(len(YEARS))
        bar_w   = 0.55

        fig, ax1 = plt.subplots(figsize=(11, 7))
        fig.patch.set_facecolor(WHITE)

        # ── Stacked bars: Spend by funnel per year ─────────────────────────
        bottoms = np.zeros(len(YEARS))
        for funnel in FUNNELS:
            vals = np.array([sp(brand, funnel, tier, yr) for yr in YEARS])
            ax1.bar(x_base, vals, bar_w, bottom=bottoms,
                    color=FUNNEL_COLOR[funnel], alpha=0.85,
                    edgecolor=WHITE, lw=0.5, label=funnel)
            bottoms += vals

        # Annotate total spend on top of each bar
        for i, total in enumerate(bottoms):
            if total > 0:
                ax1.annotate(f"${total:.2f}M",
                             xy=(x_base[i], total),
                             xytext=(0, 5), textcoords="offset points",
                             ha="center", fontsize=14, color=CPX_DGREY, fontweight="bold")

        # Left y-axis: Spend
        ax1.set_ylabel("Media Spend ($M)", fontsize=14, fontweight="bold", color=CPX_DGREY)
        ax1.set_xticks(x_base)
        ax1.set_xticklabels([str(yr) for yr in YEARS], fontsize=14, fontweight="bold")
        plt.setp(ax1.get_yticklabels(), fontsize=14, fontweight="bold")
        ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:.2f}M"))
        ax1.spines["top"].set_visible(False)
        ax1.set_axisbelow(True)
        ax1.grid(axis="y", alpha=0.15, color="#CCCCCC")
        ax1.set_ylim(0, bottoms.max() * 1.4)

        # ── ROI curve on secondary y-axis ─────────────────────────────────
        ax2 = ax1.twinx()
        roi_vals = []
        for yr in YEARS:
            t_spend = sum(sp(brand, f, tier, yr) for f in FUNNELS)
            t_rev   = sum(rv(brand, f, tier, yr) for f in FUNNELS)
            roi_vals.append(t_rev / t_spend if t_spend > 0 else 0.0)

        ax2.plot(x_base, roi_vals,
                 color="black", linewidth=3, marker="D",
                 markersize=12, markeredgecolor=WHITE, markeredgewidth=2,
                 zorder=5, label="ROI")

        # Annotate ROI on each point
        for i, roi in enumerate(roi_vals):
            if roi > 0:
                ax2.annotate(f"{roi:.2f}×",
                             xy=(x_base[i], roi),
                             xytext=(0, 12), textcoords="offset points",
                             ha="center", fontsize=14, fontweight="bold",
                             color="black")

        ax2.set_ylabel("ROI (Rev / Spend)", fontsize=14, fontweight="bold", color=CPX_DGREY)
        plt.setp(ax2.get_yticklabels(), fontsize=14, fontweight="bold")
        ax2.spines["top"].set_visible(False)
        if max(roi_vals) > 0:
            # Use larger headroom so ROI curve sits well below bar-top annotations
            ax2.set_ylim(0, max(roi_vals) * 2.2)
        else:
            ax2.set_ylim(0, 5)

        # ── Legend ─────────────────────────────────────────────────────────
        funnel_handles = [mpatches.Patch(color=FUNNEL_COLOR[f], label=f"{f} Spend")
                          for f in FUNNELS]
        roi_handle = mlines.Line2D([], [], color="black", linewidth=3,
                                   marker="D", markersize=10, label="ROI")
        ax1.legend(handles=funnel_handles + [roi_handle],
                   fontsize=14, loc="upper left", framealpha=0.9,
                   edgecolor="#CCCCCC")

        ax1.set_title(f"{brand} {tier} — Spend & ROI by Year",
                      fontsize=14, fontweight="bold", color=CPX_NAVY)
        plt.tight_layout()
        save_fig(fig, f"lbe_tier_roi_{brand.lower()}_{tier.lower()}_by_year.png")
        display(fig)
        plt.close(fig)

        # ── Summary table ───────────────────────────────────────────────────
        tbl_rows = []
        for yr in YEARS:
            t_spend = sum(sp(brand, f, tier, yr) for f in FUNNELS)
            t_rev   = sum(rv(brand, f, tier, yr) for f in FUNNELS)
            roi     = t_rev / t_spend if t_spend > 0 else 0.0
            tbl_rows.append({
                "Year": yr,
                "Awareness Spend ($M)": round(sp(brand, "Awareness", tier, yr), 3),
                "Consideration Spend ($M)": round(sp(brand, "Consideration", tier, yr), 3),
                "Total Spend ($M)": round(t_spend, 3),
                "Total Rev ($M)": round(t_rev, 3),
                "ROI": f"{roi:.2f}×" if t_spend > 0 else "—",
            })
        print(f"\n{'═'*60}\n  {brand} {tier} — Spend & ROI by Year\n{'═'*60}")
        ipy_display(pd.DataFrame(tbl_rows))

# COMMAND ----------

# DBTITLE 1,Awareness → Volume & Consideration → Yield Correlation
# ═══════════════════════════════════════════════════════════════════════════════
# WHY TWO MODELS? — Revenue Mechanism Decomposition
# Shows stakeholders how each funnel drives revenue through different paths:
#   Awareness  → gets people in the door (VOLUME = more transactions)
#   Consideration → gets them to spend more (YIELD = higher rev per visit)
# Revenue = Transactions × Rev/Transaction, so both paths lead to $$.
# ═══════════════════════════════════════════════════════════════════════════════
import matplotlib.patches as mpatches

# ── Decompose each funnel’s revenue into Volume vs Yield paths ───────────────
# Volume path: RPT × ΔTXN (more visits at current spend-per-visit)
# Yield path:  ΔRPT × TXN (higher spend per visit with current traffic)
decomp = {}
for brand in BRANDS:
    r = results[brand]
    pred_rpt = r["pred_rpt"]
    pred_txn = r["pred_txn"]
    log_pred_rpt = r["log_pred_rpt"]
    log_pred_txn = r["log_pred_txn"]

    # Awareness decomposition
    aw_volume = (pred_rpt * (pred_txn - np.exp(log_pred_txn - r["m_aw_txn"]))).sum()
    aw_yield  = ((pred_rpt - np.exp(log_pred_rpt - r["m_aw_rpt"])) * pred_txn).sum()
    aw_total  = aw_volume + aw_yield

    # Consideration decomposition
    co_volume = (pred_rpt * (pred_txn - np.exp(log_pred_txn - r["m_co_txn"]))).sum()
    co_yield  = ((pred_rpt - np.exp(log_pred_rpt - r["m_co_rpt"])) * pred_txn).sum()
    co_total  = co_volume + co_yield

    decomp[brand] = {
        "Awareness":     {"Volume": aw_volume, "Yield": aw_yield, "Total": aw_total},
        "Consideration": {"Volume": co_volume, "Yield": co_yield, "Total": co_total},
    }

# ═══ FIGURE 1: Mechanism split — horizontal stacked bars (stakeholder-friendly) ═══
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.patch.set_facecolor(WHITE)

for ci, brand in enumerate(BRANDS):
    ax = axes[ci]
    funnels = ["Awareness", "Consideration"]
    y_pos   = np.arange(len(funnels))
    bar_h   = 0.5

    vol_pcts = []
    yld_pcts = []
    for funnel in funnels:
        total = decomp[brand][funnel]["Total"]
        vol_pct = decomp[brand][funnel]["Volume"] / total * 100 if total > 0 else 0
        yld_pct = decomp[brand][funnel]["Yield"] / total * 100 if total > 0 else 0
        vol_pcts.append(vol_pct)
        yld_pcts.append(yld_pct)

    # Stacked horizontal bars
    bars_vol = ax.barh(y_pos, vol_pcts, bar_h, color=CPX_NAVY, label="Volume Path\n(more visits)")
    bars_yld = ax.barh(y_pos, yld_pcts, bar_h, left=vol_pcts, color=CPX_RED, label="Yield Path\n(higher spend/visit)")

    # Annotate percentages inside bars
    for i in range(len(funnels)):
        if vol_pcts[i] > 15:
            ax.text(vol_pcts[i] / 2, y_pos[i], f"{vol_pcts[i]:.0f}%\nVolume",
                    ha="center", va="center", fontsize=14, fontweight="bold", color=WHITE)
        if yld_pcts[i] > 15:
            ax.text(vol_pcts[i] + yld_pcts[i] / 2, y_pos[i], f"{yld_pcts[i]:.0f}%\nYield",
                    ha="center", va="center", fontsize=14, fontweight="bold", color=WHITE)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(funnels, fontsize=14, fontweight="bold")
    ax.set_xlabel("% of Revenue Contribution", fontsize=14, fontweight="bold")
    ax.set_xlim(0, 100)
    plt.setp(ax.get_xticklabels(), fontsize=14, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title(f"{brand}", fontsize=14, fontweight="bold", color=CPX_NAVY)

# Single figure-level legend below the charts in one row
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, fontsize=14, loc="lower center", ncol=2,
           framealpha=0.9, edgecolor="#CCCCCC", bbox_to_anchor=(0.5, -0.08))

fig.suptitle("How Each Funnel Drives Revenue — Two Different Mechanisms\n"
             "Revenue = Transactions × Rev per Transaction",
             fontsize=14, fontweight="bold", color=CPX_NAVY, y=1.0)
plt.tight_layout()
plt.subplots_adjust(bottom=0.15)
save_fig(fig, "lbe_why_two_models_mechanism_split.png")
display(fig)
plt.close(fig)

# ═══ FIGURE 2: Dollar contribution by mechanism (absolute $M) ═════════════════
fig2, ax2 = plt.subplots(figsize=(12, 7))
fig2.patch.set_facecolor(WHITE)

labels = []
vol_vals = []
yld_vals = []
for brand in BRANDS:
    for funnel in ["Awareness", "Consideration"]:
        labels.append(f"{brand}\n{funnel}")
        vol_vals.append(decomp[brand][funnel]["Volume"] / 1e6)
        yld_vals.append(decomp[brand][funnel]["Yield"] / 1e6)

x = np.arange(len(labels))
bar_w = 0.55

ax2.bar(x, vol_vals, bar_w, color=CPX_NAVY, label="Volume Path (RPT × ΔTxn)")
ax2.bar(x, yld_vals, bar_w, bottom=vol_vals, color=CPX_RED, label="Yield Path (ΔRPT × Txn)")

# Annotate totals
for i in range(len(labels)):
    total = vol_vals[i] + yld_vals[i]
    ax2.annotate(f"${total:.1f}M", xy=(x[i], total),
                 xytext=(0, 5), textcoords="offset points",
                 ha="center", fontsize=14, fontweight="bold", color=CPX_DGREY)
    # Annotate split inside
    pct_vol = vol_vals[i] / total * 100
    pct_yld = yld_vals[i] / total * 100
    if vol_vals[i] > 0.3:
        ax2.text(x[i], vol_vals[i] / 2, f"{pct_vol:.0f}%",
                 ha="center", va="center", fontsize=14, fontweight="bold", color=WHITE)
    if yld_vals[i] > 0.3:
        ax2.text(x[i], vol_vals[i] + yld_vals[i] / 2, f"{pct_yld:.0f}%",
                 ha="center", va="center", fontsize=14, fontweight="bold", color=WHITE)

ax2.set_xticks(x)
ax2.set_xticklabels(labels, fontsize=14, fontweight="bold")
ax2.set_ylabel("Media-Attributed Revenue ($M)", fontsize=14, fontweight="bold")
plt.setp(ax2.get_yticklabels(), fontsize=14, fontweight="bold")
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
ax2.legend(fontsize=14, loc="upper right", framealpha=0.9)
ax2.set_title("Revenue Attribution by Mechanism — Why We Need Two Models",
              fontsize=14, fontweight="bold", color=CPX_NAVY)
ax2.grid(axis="y", alpha=0.15)
plt.tight_layout()
save_fig(fig2, "lbe_why_two_models_dollar_split.png")
display(fig2)
plt.close(fig2)

# ── Summary table for stakeholders ───────────────────────────────────────────
print("\n" + "═"*75)
print("  WHY TWO MODELS? — Summary")
print("═"*75)
rows = []
for brand in BRANDS:
    for funnel in ["Awareness", "Consideration"]:
        d_item = decomp[brand][funnel]
        total = d_item["Total"]
        rows.append({
            "Brand": brand,
            "Funnel": funnel,
            "Total Revenue ($M)": f"${total/1e6:.1f}M",
            "Via Volume (more visits)": f"{d_item['Volume']/total*100:.0f}%" if total > 0 else "—",
            "Via Yield (higher spend)": f"{d_item['Yield']/total*100:.0f}%" if total > 0 else "—",
            "Primary Target": "Transactions" if d_item['Volume'] > d_item['Yield'] else "Rev/Transaction",
        })
ipy_display(pd.DataFrame(rows))

print("\n\U0001F4A1 Key Takeaway for Stakeholders:")
print("   Awareness and Consideration grow revenue through DIFFERENT mechanisms.")
print("   A single-model approach conflates these, understating Consideration’s")
print("   true ROI (≈1× in V3 vs 3× in V5e). The dual-target model separates:")
print("     • Model A: log(Rev/Txn) — how media lifts YIELD (spend per visit)")
print("     • Model B: log(Txn)     — how media lifts VOLUME (foot traffic)")
print("   Then multiplies them back: Revenue = Rev/Txn × Transactions")

# COMMAND ----------

# DBTITLE 1,Log-Space Partial Correlations (matching model specification)
# ═══════════════════════════════════════════════════════════════════════════════
# LOG-SPACE PARTIAL CORRELATIONS
# The V5e model estimates: log(Txn) = baseline + β_aw*f(aw) + controls
#                          log(RPT) = baseline + β_co*f(co) + controls
# So the correct correlation space is log-transformed outcomes.
# This matches what the Bayesian model actually sees.
# ═══════════════════════════════════════════════════════════════════════════════

# Pre-compute LOG-SPACE residualized series for both brands
_series_log = {}
for brand in BRANDS:
    aw_raw = d[f"{brand}_Awareness_spend"].values
    co_raw = d[f"{brand}_Consideration_spend"].values
    txn    = d[f"{brand}_Txns"].values
    rpt    = d[f"{brand}_Rev"].values / np.maximum(txn, 1)

    # Log-transform outcomes (what the model actually estimates)
    log_txn = np.log(np.maximum(txn, 1))
    log_rpt = np.log(np.maximum(rpt, 1))

    # Adstock with model-fitted alphas
    aw_ads = _adstock(aw_raw, results[brand]["adstock_aw_txn"])
    co_ads = _adstock(co_raw, results[brand]["adstock_co_rpt"])

    # Log-transform media (model uses log1p of normalized adstock)
    log_aw = np.log1p(aw_ads)
    log_co = np.log1p(co_ads)

    # Residualize in log-space
    _series_log[brand] = {
        "aw_resid":     _residualize(log_aw, X_ctrl),
        "co_resid":     _residualize(log_co, X_ctrl),
        "txn_resid":    _residualize(log_txn, X_ctrl),
        "rpt_resid":    _residualize(log_rpt, X_ctrl),
    }

# ═══ FIGURE 1: AWARENESS in log-space ══════════════════════════════════════════
fig1, axes1 = plt.subplots(2, 2, figsize=(14, 10))
fig1.patch.set_facecolor(WHITE)

for ci, brand in enumerate(BRANDS):
    s = _series_log[brand]
    # Top: log(Awareness) vs log(Transactions) — expected: strong
    _scatter_partial(axes1[0, ci], s["aw_resid"], s["txn_resid"],
                     f"{brand} — log(Awareness) → log(Transactions)",
                     "log(Awareness) residualized", "log(Txn) residualized",
                     CPX_NAVY, CPX_RED)
    # Bottom: log(Awareness) vs log(RPT) — expected: weak
    _scatter_partial(axes1[1, ci], s["aw_resid"], s["rpt_resid"],
                     f"{brand} — log(Awareness) → log(Rev/Txn)",
                     "log(Awareness) residualized", "log(RPT) residualized",
                     CPX_LGREY, CPX_DGREY)

fig1.suptitle("AWARENESS — Log-Space Partial Correlations (matching V5e model specification)\n"
              "Top: expected target log(Txn)  |  Bottom: non-target log(RPT)",
              fontsize=14, fontweight="bold", color=CPX_NAVY, y=1.02)
plt.tight_layout()
save_fig(fig1, "lbe_awareness_logspace_partial_correlation.png")
display(fig1)
plt.close(fig1)

# ═══ FIGURE 2: CONSIDERATION in log-space ═══════════════════════════════════════
fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))
fig2.patch.set_facecolor(WHITE)

for ci, brand in enumerate(BRANDS):
    s = _series_log[brand]
    # Top: log(Consideration) vs log(RPT) — expected: strong
    _scatter_partial(axes2[0, ci], s["co_resid"], s["rpt_resid"],
                     f"{brand} — log(Consideration) → log(Rev/Txn)",
                     "log(Consideration) residualized", "log(RPT) residualized",
                     CPX_RED, CPX_NAVY)
    # Bottom: log(Consideration) vs log(Txn) — expected: weak
    _scatter_partial(axes2[1, ci], s["co_resid"], s["txn_resid"],
                     f"{brand} — log(Consideration) → log(Transactions)",
                     "log(Consideration) residualized", "log(Txn) residualized",
                     CPX_LGREY, CPX_DGREY)

fig2.suptitle("CONSIDERATION — Log-Space Partial Correlations (matching V5e model specification)\n"
              "Top: expected target log(RPT)  |  Bottom: non-target log(Txn)",
              fontsize=14, fontweight="bold", color=CPX_NAVY, y=1.02)
plt.tight_layout()
save_fig(fig2, "lbe_consideration_logspace_partial_correlation.png")
display(fig2)
plt.close(fig2)

# ── Summary comparison table ────────────────────────────────────────────────────
print("\n" + "═"*75)
print("  COMPARISON: Linear-space vs Log-space partial r")
print("═"*75)
rows = []
for brand in BRANDS:
    s_lin = _series[brand]
    s_log = _series_log[brand]
    # Awareness → Txn
    r_lin, _ = pearsonr(s_lin["aw_resid"], s_lin["txn_resid"])
    r_log, _ = pearsonr(s_log["aw_resid"], s_log["txn_resid"])
    rows.append({"Brand": brand, "Relationship": "Awareness → Txn",
                 "Linear r": f"{r_lin:.3f}", "Log-space r": f"{r_log:.3f}", "Expected": "Strong +"})
    # Awareness → RPT
    r_lin, _ = pearsonr(s_lin["aw_resid"], s_lin["rpt_resid"])
    r_log, _ = pearsonr(s_log["aw_resid"], s_log["rpt_resid"])
    rows.append({"Brand": brand, "Relationship": "Awareness → RPT",
                 "Linear r": f"{r_lin:.3f}", "Log-space r": f"{r_log:.3f}", "Expected": "Weak / 0"})
    # Consideration → RPT
    r_lin, _ = pearsonr(s_lin["co_resid"], s_lin["rpt_resid"])
    r_log, _ = pearsonr(s_log["co_resid"], s_log["rpt_resid"])
    rows.append({"Brand": brand, "Relationship": "Consideration → RPT",
                 "Linear r": f"{r_lin:.3f}", "Log-space r": f"{r_log:.3f}", "Expected": "Strong +"})
    # Consideration → Txn
    r_lin, _ = pearsonr(s_lin["co_resid"], s_lin["txn_resid"])
    r_log, _ = pearsonr(s_log["co_resid"], s_log["txn_resid"])
    rows.append({"Brand": brand, "Relationship": "Consideration → Txn",
                 "Linear r": f"{r_lin:.3f}", "Log-space r": f"{r_log:.3f}", "Expected": "Weak / 0"})

ipy_display(pd.DataFrame(rows))

print("\nKey insight: The V5e model operates in log-space, so log(media) ↔ log(outcome)")
print("captures the multiplicative relationship the Bayesian model estimates.")
print("Linear-space partial r underestimates the signal because it misses the")
print("diminishing-returns (concave) structure the model exploits.")

# COMMAND ----------

# DBTITLE 1,LBE Budget Scenario Dependencies
# ━━ LBE BUDGET SCENARIO — DEPENDENCIES ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Derives an LBE-level single-target response model from V5e dual-target results.
# Calibrated so that at 2025 observed spend, the model exactly reproduces
# V5e total media-attributed revenue ($33.51M).

# ── 1. df_lbe_all: weekly total Awareness & Consideration spend ───────────────
df_lbe_all = df_all[["week_date", "Awareness", "Consideration"]].copy()

# ── 2. Normalization maxes (weekly) ───────────────────────────────────────────
lbe_aw_max = float(df_lbe_all["Awareness"].max())
lbe_co_max = float(df_lbe_all["Consideration"].max())

# ── 3. Log-revenue statistics (LBE = TRR + PDM) ──────────────────────────────
_lbe_weekly_rev = d["TRR_Rev"] + d["PDM_Rev"]
_log_rev = np.log(_lbe_weekly_rev)
lbe_y_mean = float(_log_rev.mean())
lbe_y_std  = float(_log_rev.std())

# ── 4. Adstock decay (revenue-weighted average of TRR & PDM, across sub-models)
_trr_w = (results["TRR"]["aw_incr_rev"] + results["TRR"]["co_incr_rev"])
_pdm_w = (results["PDM"]["aw_incr_rev"] + results["PDM"]["co_incr_rev"])
_w_trr = _trr_w / (_trr_w + _pdm_w)
_w_pdm = _pdm_w / (_trr_w + _pdm_w)

_aw_decay = float(
    _w_trr * (results["TRR"]["adstock_aw_rpt"] + results["TRR"]["adstock_aw_txn"]) / 2
    + _w_pdm * (results["PDM"]["adstock_aw_rpt"] + results["PDM"]["adstock_aw_txn"]) / 2
)
_co_decay = float(
    _w_trr * (results["TRR"]["adstock_co_rpt"] + results["TRR"]["adstock_co_txn"]) / 2
    + _w_pdm * (results["PDM"]["adstock_co_rpt"] + results["PDM"]["adstock_co_txn"]) / 2
)

# ── 5. Calibrate beta & scale so model reproduces V5e totals at 2025 spend ────
_total_aw_incr = results["TRR"]["aw_incr_rev"] + results["PDM"]["aw_incr_rev"]
_total_co_incr = results["TRR"]["co_incr_rev"] + results["PDM"]["co_incr_rev"]
_total_media   = _total_aw_incr + _total_co_incr

_df_2025 = df_lbe_all[df_lbe_all["week_date"].dt.year == 2025]
_aw_annual = float(_df_2025["Awareness"].sum())
_co_annual = float(_df_2025["Consideration"].sum())

_ads_aw = ((_aw_annual / 52) / lbe_aw_max) / (1 - _aw_decay)
_ads_co = ((_co_annual / 52) / lbe_co_max) / (1 - _co_decay)

_aw_scale = 1.0
_co_scale = 1.0

_baseline_rev = np.exp(lbe_y_mean) * 52
_mu_total = np.log(1 + _total_media / _baseline_rev) / lbe_y_std
_mu_aw = _mu_total * (_total_aw_incr / _total_media)
_mu_co = _mu_total * (_total_co_incr / _total_media)

_beta_aw = _mu_aw / np.log1p(_ads_aw * _aw_scale)
_beta_co = _mu_co / np.log1p(_ads_co * _co_scale)

# ── 6. Construct lbe_draws (point-estimate posterior, repeated DRAWS times) ───
_n = DRAWS if "DRAWS" in dir() else 1000
lbe_draws = {
    "aw_decay":  np.full(_n, _aw_decay),
    "co_decay":  np.full(_n, _co_decay),
    "aw_scale":  np.full(_n, _aw_scale),
    "co_scale":  np.full(_n, _co_scale),
    "beta_aw":   np.full(_n, _beta_aw),
    "beta_co":   np.full(_n, _beta_co),
}

print("LBE Budget Scenario — calibrated dependencies")
print(f"  lbe_aw_max:  {lbe_aw_max:>12,.0f}")
print(f"  lbe_co_max:  {lbe_co_max:>12,.0f}")
print(f"  lbe_y_mean:  {lbe_y_mean:>12.6f}")
print(f"  lbe_y_std:   {lbe_y_std:>12.6f}")
print(f"  aw_decay:    {_aw_decay:>12.4f}")
print(f"  co_decay:    {_co_decay:>12.4f}")
print(f"  beta_aw:     {_beta_aw:>12.4f}")
print(f"  beta_co:     {_beta_co:>12.4f}")
print(f"  Draws:       {_n}")
print(f"\n  Verification at 2025 spend:")
_mu_v = _beta_aw * np.log1p(_ads_aw * _aw_scale) + _beta_co * np.log1p(_ads_co * _co_scale)
_rev_v = np.exp(lbe_y_mean + _mu_v * lbe_y_std) * 52 - _baseline_rev
print(f"    Reproduced media contribution: ${_rev_v/1e6:.2f}M (V5e target: ${_total_media/1e6:.2f}M) ✓")

# COMMAND ----------

# DBTITLE 1,LBE Budget Scenario Analysis
# ━━ LBE BUDGET SCENARIO ANALYSIS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# 2025 baseline spend
df_lbe_2025 = df_lbe_all[df_lbe_all["week_date"].dt.year == 2025]
lbe_baseline_aw = float(df_lbe_2025["Awareness"].sum())
lbe_baseline_co = float(df_lbe_2025["Consideration"].sum())

# ── Model uncertainty from V5e MAPE (used for CI estimation) ──────────────────
_mape_rpt_trr = results["TRR"].get("mape_rpt", 7.0)
_mape_txn_trr = results["TRR"].get("mape_txn", 7.0)
_mape_rpt_pdm = results["PDM"].get("mape_rpt", 7.0)
_mape_txn_pdm = results["PDM"].get("mape_txn", 7.0)
# Combined model CV: propagated from sub-model MAPEs
_model_cv = np.sqrt((_mape_rpt_trr**2 + _mape_txn_trr**2
                     + _mape_rpt_pdm**2 + _mape_txn_pdm**2) / 4) / 100

def lbe_funnel_contribution_ci(aw_annual, co_annual, n_sim=2000):
    """Point estimate + 90% CI via model uncertainty propagation."""
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
    rev_point = np.exp(lbe_y_mean + mu_media * lbe_y_std) * 52
    rev_without = np.exp(lbe_y_mean) * 52
    contribution_point = rev_point - rev_without

    # Monte Carlo CI using model uncertainty
    np.random.seed(42)
    noise = np.random.normal(1.0, _model_cv, n_sim)
    contributions_sim = contribution_point * noise
    ci_low  = float(np.percentile(contributions_sim, 5))
    ci_high = float(np.percentile(contributions_sim, 95))

    return contribution_point, ci_low, ci_high

# ── Confidence classification ─────────────────────────────────────────────────
def confidence_label(ci_low, ci_high, point):
    """High / Medium / Low based on CI width relative to point estimate."""
    if point == 0:
        return "Low"
    ci_width_pct = (ci_high - ci_low) / abs(point) * 100
    if ci_width_pct < 20:
        return "High"
    elif ci_width_pct < 35:
        return "Medium"
    else:
        return "Low"

# ── 2025 Actual Total Revenue (TRR + PDM) ───────────────────────────────────
_lbe_rev_2025 = float((d.loc[d["week_date"].dt.year == 2025, "TRR_Rev"].sum()
                       + d.loc[d["week_date"].dt.year == 2025, "PDM_Rev"].sum()))

# ── Baseline media contribution ─────────────────────────────────────────────
lbe_baseline_rev, lbe_base_ci_lo, lbe_base_ci_hi = lbe_funnel_contribution_ci(
    lbe_baseline_aw, lbe_baseline_co
)

# ── Define all scenarios ──────────────────────────────────────────────────────
scenarios = []

# Baseline
scenarios.append({
    "Scenario": "2025 Baseline",
    "Type": "Baseline",
    "aw": lbe_baseline_aw,
    "co": lbe_baseline_co,
})

# Budget uplift scenarios
for pct in [3, 5, 10, 20, 30]:
    factor = 1 + pct / 100
    scenarios.append({
        "Scenario": f"+{pct}% Budget Uplift",
        "Type": "Budget",
        "aw": lbe_baseline_aw * factor,
        "co": lbe_baseline_co * factor,
    })

# Reshuffle scenario (20% shift Consideration -> Awareness)
shift = lbe_baseline_co * 0.20
scenarios.append({
    "Scenario": "20% Shift: Co -> Aw",
    "Type": "Reshuffle",
    "aw": lbe_baseline_aw + shift,
    "co": lbe_baseline_co - shift,
})

# Combined scenarios
for pct in [10, 20, 30]:
    factor = 1 + pct / 100
    aw_up = lbe_baseline_aw * factor
    co_up = lbe_baseline_co * factor
    shift_c = co_up * 0.20
    scenarios.append({
        "Scenario": f"+{pct}% + 20% Shift",
        "Type": "Combined",
        "aw": aw_up + shift_c,
        "co": co_up - shift_c,
    })

# ── Compute results for each scenario ─────────────────────────────────────────
tbl_rows = []
for s in scenarios:
    rev, ci_lo, ci_hi = lbe_funnel_contribution_ci(s["aw"], s["co"])
    lift = rev - lbe_baseline_rev
    lift_pct = (lift / lbe_baseline_rev * 100) if lbe_baseline_rev != 0 else 0.0
    conf = confidence_label(ci_lo, ci_hi, rev)

    # Projected Revenue = Baseline (non-media) + media-attributed revenue
    _baseline_non_media = _lbe_rev_2025 - lbe_baseline_rev
    projected_rev = _baseline_non_media + rev
    ci_lo_proj = _baseline_non_media + ci_lo
    ci_hi_proj = _baseline_non_media + ci_hi

    tbl_rows.append({
        "Scenario":              s["Scenario"],
        "Type":                  s["Type"],
        "2025 Actual Rev ($M)":  round(_lbe_rev_2025 / 1e6, 2),
        "Scenario Lift ($M)":    round(lift / 1e6, 2),
        "Projected Rev ($M)":    round(projected_rev / 1e6, 2),
        "CI Low ($M)":           round(ci_lo_proj / 1e6, 2),
        "CI High ($M)":          round(ci_hi_proj / 1e6, 2),
        "Lift vs Actual %":      f"{lift_pct:+.1f}%",
        "Confidence":            conf,
    })

scenario_df = pd.DataFrame(tbl_rows)
print(f"LBE 2025 Media Budget: ${(lbe_baseline_aw + lbe_baseline_co)/1e6:.2f}M  |  "
      f"Model CV: {_model_cv*100:.1f}%  |  CI: 90% (5th-95th percentile)\n")
display(scenario_df)

# COMMAND ----------

# DBTITLE 1,LBE Scenario Spend & Revenue by Tier
# ━━ LBE SCENARIO: SPEND & REVENUE BY TIER ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# One chart per scenario (from cell 29). Each chart shows:
#   • Spend by tier: 2025 actual (solid) vs scenario (hatched)
#   • Revenue by tier: 2025 actual (solid) vs scenario (hatched)
#   • Delta labels on scenario bars
#   • Title includes total revenue lift & CI
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker

# ── Tier structure ───────────────────────────────────────────────────────────
FUNNELS_T      = ["Awareness", "Consideration"]
BRANDS_T       = ["TRR", "PDM"]
brand_tiers_t  = {"TRR": ["HPL", "LPL", "NL"], "PDM": ["OL", "NL"]}
FUNNEL_COLOR_T = {"Awareness": CPX_NAVY, "Consideration": CPX_RED}

# Build tier list: (funnel, brand, tier)
tier_list = []
for funnel in FUNNELS_T:
    for brand in BRANDS_T:
        for tier in brand_tiers_t[brand]:
            tier_list.append((funnel, brand, tier))

n_tiers_total = len(tier_list)

# ── 2025 actual spend & revenue per tier ────────────────────────────────────
actual_spend_tier = {}
for funnel, brand, tier in tier_list:
    actual_spend_tier[(funnel, brand, tier)] = eff_spend.get((brand, funnel, tier, 2025), 0.0)

# Funnel totals for proportional allocation
actual_aw_total = sum(v for (f, b, t), v in actual_spend_tier.items() if f == "Awareness")
actual_co_total = sum(v for (f, b, t), v in actual_spend_tier.items() if f == "Consideration")

# ── Funnel-level revenue function (for splitting total rev to Aw/Co) ────────
def _funnel_rev_split(aw_annual, co_annual):
    """Returns (aw_contribution, co_contribution) using the model."""
    # Total contribution
    total_rev, _, _ = lbe_funnel_contribution_ci(aw_annual, co_annual)
    # Awareness-only contribution
    aw_only, _, _ = lbe_funnel_contribution_ci(aw_annual, 0.0)
    # Consideration-only contribution
    co_only, _, _ = lbe_funnel_contribution_ci(0.0, co_annual)
    # Proportional split (handles interaction)
    denom = aw_only + co_only if (aw_only + co_only) > 0 else 1.0
    aw_share = aw_only / denom
    co_share = co_only / denom
    return total_rev * aw_share, total_rev * co_share

# ── 2025 actual media-attributed revenue per tier (model-consistent) ───────
# Uses the same funnel-split + spend-weight methodology as scenario allocation
# so that baseline and scenario are on the same basis and lifts sum correctly.
_aw_rev_2025, _co_rev_2025 = _funnel_rev_split(lbe_baseline_aw, lbe_baseline_co)
actual_rev_tier = {}
for funnel, brand, tier in tier_list:
    funnel_spend = actual_aw_total if funnel == "Awareness" else actual_co_total
    if funnel_spend > 0:
        w = actual_spend_tier[(funnel, brand, tier)] / funnel_spend
    else:
        w = 0.0
    funnel_rev = _aw_rev_2025 if funnel == "Awareness" else _co_rev_2025
    actual_rev_tier[(funnel, brand, tier)] = funnel_rev * w

# ── Compute scenario tier allocations ────────────────────────────────────────
def scenario_tier_alloc(aw_annual, co_annual):
    """Allocate scenario spend & revenue to tiers proportionally."""
    # Spend allocation: scale each tier by funnel ratio
    aw_factor = aw_annual / actual_aw_total if actual_aw_total > 0 else 1.0
    co_factor = co_annual / actual_co_total if actual_co_total > 0 else 1.0

    scen_spend = {}
    for funnel, brand, tier in tier_list:
        factor = aw_factor if funnel == "Awareness" else co_factor
        scen_spend[(funnel, brand, tier)] = actual_spend_tier[(funnel, brand, tier)] * factor

    # Revenue allocation: total scenario rev split by funnel, then by tier spend weight
    aw_rev_total, co_rev_total = _funnel_rev_split(aw_annual, co_annual)

    scen_rev = {}
    for funnel, brand, tier in tier_list:
        funnel_spend = sum(scen_spend[(f, b, t)] for f, b, t in tier_list if f == funnel)
        if funnel_spend > 0:
            w = scen_spend[(funnel, brand, tier)] / funnel_spend
        else:
            w = 0.0
        funnel_rev = aw_rev_total if funnel == "Awareness" else co_rev_total
        scen_rev[(funnel, brand, tier)] = funnel_rev * w

    return scen_spend, scen_rev

# ── Plotting ─────────────────────────────────────────────────────────────────
fmt_K = mticker.FuncFormatter(lambda v, _: f"${v/1e3:.0f}K")

for s in scenarios:
    if s["Type"] == "Baseline":
        continue  # skip baseline (no change to show)

    scen_spend, scen_rev = scenario_tier_alloc(s["aw"], s["co"])
    total_rev_scen, ci_lo, ci_hi = lbe_funnel_contribution_ci(s["aw"], s["co"])
    lift = total_rev_scen - lbe_baseline_rev
    lift_pct = lift / lbe_baseline_rev * 100

    # CI as % of baseline
    ci_lo_pct = (ci_lo - lbe_baseline_rev) / lbe_baseline_rev * 100
    ci_hi_pct = (ci_hi - lbe_baseline_rev) / lbe_baseline_rev * 100

    fig, (ax_sp, ax_rv) = plt.subplots(1, 2, figsize=(18, 7))
    fig.patch.set_facecolor(WHITE)

    bar_w = 0.35
    x = np.arange(n_tiers_total)

    # ── LEFT PANEL: Spend by Tier (actual vs scenario) ─────────────────────
    actual_vals_sp = np.array([actual_spend_tier[(f, b, t)] for f, b, t in tier_list])
    scen_vals_sp   = np.array([scen_spend[(f, b, t)] for f, b, t in tier_list])
    colors         = [FUNNEL_COLOR_T[f] for f, b, t in tier_list]

    for i in range(n_tiers_total):
        ax_sp.bar(x[i] - bar_w/2, actual_vals_sp[i], bar_w,
                  color=colors[i], alpha=0.9, edgecolor=WHITE, lw=0.4)
        ax_sp.bar(x[i] + bar_w/2, scen_vals_sp[i], bar_w,
                  color=colors[i], alpha=0.45, edgecolor=colors[i],
                  lw=0.6, hatch="///")

    # Spend delta labels
    for i in range(n_tiers_total):
        delta = scen_vals_sp[i] - actual_vals_sp[i]
        if abs(delta) > 0:
            sign = "+" if delta >= 0 else ""
            lbl = f"{sign}${delta/1e3:.0f}K" if abs(delta) < 1e6 else f"{sign}${delta/1e6:.1f}M"
            ax_sp.annotate(lbl, xy=(x[i] + bar_w/2, scen_vals_sp[i]),
                           xytext=(0, 4), textcoords="offset points",
                           ha="center", fontsize=7.5, fontweight="bold",
                           color=CPX_NAVY if delta >= 0 else CPX_RED)

    xlabels = [f"{b}\n{t}" for f, b, t in tier_list]
    ax_sp.set_xticks(x)
    ax_sp.set_xticklabels(xlabels, fontsize=10, fontweight="bold")
    ax_sp.set_ylabel("Spend ($K)", fontsize=12, fontweight="bold")
    ax_sp.yaxis.set_major_formatter(fmt_K)
    ax_sp.spines["top"].set_visible(False)
    ax_sp.spines["right"].set_visible(False)
    plt.setp(ax_sp.get_yticklabels(), fontsize=10, fontweight="bold")
    ax_sp.grid(False)
    n_aw = sum(1 for f, _, _ in tier_list if f == "Awareness")
    ax_sp.axvline(n_aw - 0.5, color="#CCCCCC", lw=1.2, linestyle="--")
    y_top_sp = max(actual_vals_sp.max(), scen_vals_sp.max()) * 1.25
    ax_sp.set_ylim(0, y_top_sp)
    ax_sp.text(n_aw / 2 - 0.5, y_top_sp * 0.96, "Awareness",
               ha="center", fontsize=10, fontweight="bold", color=CPX_NAVY)
    ax_sp.text(n_aw + (n_tiers_total - n_aw) / 2 - 0.5, y_top_sp * 0.96, "Consideration",
               ha="center", fontsize=10, fontweight="bold", color=CPX_RED)
    ax_sp.set_title("Spend by Tier", fontsize=13, fontweight="bold", color=CPX_NAVY, pad=8)

    # ── RIGHT PANEL: Total Revenue Lift by Tier ──────────────────────────────
    # Shows how the total revenue lift distributes across tiers
    lift_per_tier = np.array([
        scen_rev[(f, b, t)] - actual_rev_tier[(f, b, t)]
        for f, b, t in tier_list
    ])
    total_lift_check = lift_per_tier.sum()

    for i in range(n_tiers_total):
        bar_color = colors[i]
        ax_rv.bar(x[i], lift_per_tier[i], bar_w * 1.5,
                  color=bar_color,
                  alpha=0.75 if lift_per_tier[i] >= 0 else 0.5,
                  edgecolor=bar_color, lw=0.6)

    # Value labels on each bar
    for i in range(n_tiers_total):
        val = lift_per_tier[i]
        sign = "+" if val >= 0 else ""
        lbl = f"{sign}${val/1e6:.2f}M"
        y_pos = val if val >= 0 else val
        va = "bottom" if val >= 0 else "top"
        ax_rv.annotate(lbl, xy=(x[i], y_pos),
                       xytext=(0, 4 if val >= 0 else -4),
                       textcoords="offset points",
                       ha="center", va=va, fontsize=7.5, fontweight="bold",
                       color=CPX_NAVY if val >= 0 else CPX_RED)

    ax_rv.set_xticks(x)
    ax_rv.set_xticklabels(xlabels, fontsize=10, fontweight="bold")
    ax_rv.set_ylabel("Revenue Lift ($M)", fontsize=12, fontweight="bold")
    ax_rv.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v/1e6:.2f}M"))
    ax_rv.spines["top"].set_visible(False)
    ax_rv.spines["right"].set_visible(False)
    plt.setp(ax_rv.get_yticklabels(), fontsize=10, fontweight="bold")
    ax_rv.grid(False)
    ax_rv.axhline(0, color="#666666", lw=0.8)
    ax_rv.axvline(n_aw - 0.5, color="#CCCCCC", lw=1.2, linestyle="--")
    y_max_rv = max(abs(lift_per_tier.max()), abs(lift_per_tier.min()), 0.01) * 1.4
    ax_rv.set_ylim(-y_max_rv if lift_per_tier.min() < 0 else 0, y_max_rv)
    ax_rv.text(n_aw / 2 - 0.5, y_max_rv * 0.92, "Awareness",
               ha="center", fontsize=10, fontweight="bold", color=CPX_NAVY)
    ax_rv.text(n_aw + (n_tiers_total - n_aw) / 2 - 0.5, y_max_rv * 0.92, "Consideration",
               ha="center", fontsize=10, fontweight="bold", color=CPX_RED)
    ax_rv.set_title(f"Total Revenue Lift by Tier (Total: +${total_lift_check/1e6:.2f}M)",
                    fontsize=13, fontweight="bold", color=CPX_NAVY, pad=8)

    # Legend
    legend_handles = [
        mpatches.Patch(facecolor=CPX_NAVY, alpha=0.9, label="Awareness"),
        mpatches.Patch(facecolor=CPX_RED, alpha=0.9, label="Consideration"),
        mpatches.Patch(facecolor="#666666", alpha=0.9, label="2025 Actual (solid)"),
        mpatches.Patch(facecolor="#666666", alpha=0.45, hatch="///", label="Scenario (hatched)"),
    ]
    fig.legend(handles=legend_handles, fontsize=11, loc="upper right",
              bbox_to_anchor=(0.98, 0.98), frameon=False)

    fig.suptitle(
        f"LBE MMM \u2014 {s['Scenario']}\n"
        f"2025 Actual vs Scenario  |  Rev lift: +${lift/1e6:.2f}M ({lift_pct:+.1f}%)  "
        f"|  90% CI: [{ci_lo_pct:+.1f}%, {ci_hi_pct:+.1f}%]",
        fontsize=14, fontweight="bold", color=CPX_NAVY, y=1.02
    )

    plt.tight_layout()
    save_fig(fig, f"lbe_scenario_{s['Scenario'].lower().replace(' ', '_').replace('%','pct')}.png")
    display(fig)
    plt.close(fig)

    # ── Table: Spend & Revenue Lift by Tier ───────────────────────────────────
    tbl = []
    for f, b, t in tier_list:
        act_sp = actual_spend_tier[(f, b, t)]
        scn_sp = scen_spend[(f, b, t)]
        act_rv = actual_rev_tier[(f, b, t)]
        scn_rv_val = scen_rev[(f, b, t)]
        tbl.append({
            "Funnel": f,
            "Brand": b,
            "Tier": t,
            "2025 Spend ($K)": round(act_sp / 1e3, 1),
            "Scenario Spend ($K)": round(scn_sp / 1e3, 1),
            "Spend Δ ($K)": round((scn_sp - act_sp) / 1e3, 1),
            "2025 Media Rev ($M)": round(act_rv / 1e6, 3),
            "Scenario Media Rev ($M)": round(scn_rv_val / 1e6, 3),
            "Rev Lift ($M)": round((scn_rv_val - act_rv) / 1e6, 3),
        })
    # Totals row
    tbl.append({
        "Funnel": "", "Brand": "", "Tier": "TOTAL",
        "2025 Spend ($K)": round(sum(actual_spend_tier.values()) / 1e3, 1),
        "Scenario Spend ($K)": round(sum(scen_spend.values()) / 1e3, 1),
        "Spend Δ ($K)": round((sum(scen_spend.values()) - sum(actual_spend_tier.values())) / 1e3, 1),
        "2025 Media Rev ($M)": round(sum(actual_rev_tier.values()) / 1e6, 2),
        "Scenario Media Rev ($M)": round(sum(scen_rev.values()) / 1e6, 2),
        "Rev Lift ($M)": round(total_lift_check / 1e6, 2),
    })
    print(f"\n{s['Scenario']} — Tier Breakdown")
    display(pd.DataFrame(tbl).astype(str))

# COMMAND ----------

# DBTITLE 1,Budget Scenario Over 2025 Revenue
# ━━ BUDGET SCENARIO OVER 2025 REVENUE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker

# ── Data from scenario_df (cell 29) + additional 3%/5% scenarios ─────────────
import re as _re
forecast_2027_M = FORECAST_2027 / 1e6  # 2026 forecasted revenue

# Add +3% and +5% budget scenarios
_scen_extended = scenario_df[scenario_df["Type"] != "Baseline"].copy()
for _pct in [3, 5]:
    _factor = 1 + _pct / 100
    _aw_s = lbe_baseline_aw * _factor
    _co_s = lbe_baseline_co * _factor
    _rev, _ci_lo, _ci_hi = lbe_funnel_contribution_ci(_aw_s, _co_s)
    _lift = (_rev - lbe_baseline_rev) / 1e6
    _actual_rev_M = _lbe_rev_2025 / 1e6
    _proj = _actual_rev_M + _lift
    _ci_lo_M = _actual_rev_M + (_ci_lo - lbe_baseline_rev) / 1e6
    _ci_hi_M = _actual_rev_M + (_ci_hi - lbe_baseline_rev) / 1e6
    _new_row = pd.DataFrame([{
        "Scenario": f"+{_pct}% Budget Uplift",
        "Type": "Budget",
        "2025 Actual Rev ($M)": _actual_rev_M,
        "Scenario Lift ($M)": _lift,
        "Projected Rev ($M)": _proj,
        "CI Low ($M)": _ci_lo_M,
        "CI High ($M)": _ci_hi_M,
        "Lift vs Actual %": f"+{_lift / _actual_rev_M * 100:.1f}%",
        "Confidence": "Medium",
    }])
    _scen_extended = pd.concat([_scen_extended, _new_row], ignore_index=True)

# Assumption: the 2026 forecast includes 2025-level media contribution.
# Each scenario's LIFT over baseline is additive to the 2026 forecast.
# Projected 2026 Revenue = 2026 Forecast + Scenario Lift
scen_labels = []
scen_proj_rev = []  # projected 2026 revenue under each scenario
scen_lift = []      # lift over 2026 forecast
scen_lift_pct = []  # lift as % of forecast

for _, row in _scen_extended.iterrows():
    lift_M = row["Scenario Lift ($M)"]
    proj_2027 = forecast_2027_M + lift_M
    # Clean label: remove Co->Aw, rename standalone shift to "Reshuffle Only"
    lbl = _re.sub(r'\d+%\s*Shift', 'Reshuffle', row["Scenario"])
    lbl = _re.sub(r'[:\s]*Co\s*[->\u2192]+\s*Aw', '', lbl).strip()
    lbl = "Reshuffle Only" if lbl == "Reshuffle" else lbl
    scen_labels.append(lbl)
    scen_proj_rev.append(proj_2027)
    scen_lift.append(lift_M)
    scen_lift_pct.append(f"{lift_M / forecast_2027_M * 100:+.1f}%")

n_scen = len(scen_labels)
best_idx = np.argmax(scen_proj_rev)

# ── Bar Chart ───────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 7))
fig.patch.set_facecolor(WHITE)

x = np.arange(n_scen)
bar_w = 0.6

# Colors by scenario type: Budget=Navy, Reshuffle=Gold, Combined=Green
_type_color = {"Budget": CPX_NAVY, "Reshuffle": "#D4A017", "Combined": "#1B8A6B"}
bar_colors = []
for _, row in _scen_extended.iterrows():
    bar_colors.append(_type_color.get(row["Type"], CPX_NAVY))

ax.bar(x, scen_proj_rev, bar_w, color=bar_colors, alpha=0.85,
       edgecolor=WHITE, lw=0.5)

# 2026 Forecast baseline dashed line
ax.axhline(forecast_2027_M, color=CPX_RED, linewidth=2, linestyle="--", zorder=3)
ax.text(n_scen - 0.5, forecast_2027_M - 0.4,
        f"2027 Forecast: ${forecast_2027_M:.1f}M",
        ha="right", va="top", fontsize=9, fontstyle="italic",
        fontweight="bold", color=CPX_RED)

# Value labels on top of bars
for i in range(n_scen):
    rev = scen_proj_rev[i]
    lift_val = scen_lift[i]
    ax.text(x[i], rev + 0.15, f"${rev:.2f}M",
            ha="center", va="bottom", fontsize=12, fontweight="bold", color="black")
    sign = "+" if lift_val >= 0 else ""
    ax.text(x[i], rev + 0.7, f"({sign}${lift_val:.2f}M)",
            ha="center", va="bottom", fontsize=10,
            color="#1B8A6B" if lift_val > 0 else CPX_RED)

# Formatting
ax.set_xticks(x)
ax.set_xticklabels(scen_labels, fontsize=11, fontweight="bold", rotation=15, ha="right")
plt.setp(ax.get_yticklabels(), fontsize=12, fontweight="bold")
ax.set_ylabel("Projected Revenue ($M)", fontsize=14, fontweight="bold")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:.0f}M"))
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(False)
y_min = min(min(scen_proj_rev), forecast_2027_M) - 2
y_max = max(max(scen_proj_rev), forecast_2027_M) + 2.5
ax.set_ylim(y_min, y_max)

ax.plot([], [], color=CPX_RED, linewidth=2, linestyle="--", label="2026 Forecast (baseline)")
ax.legend(fontsize=12, loc="lower left", frameon=False)

# Title & subtitle
best_name = scen_labels[best_idx]
best_lift_val = scen_lift[best_idx]
best_pct = scen_lift_pct[best_idx]
ax.set_title(
    f"BUDGET SCENARIO OVER 2026 FORECASTED REVENUE\n"
    f"A {best_name} delivers the highest projected 2026 revenue "
    f"(+${best_lift_val:.2f}M over forecast, {best_pct})",
    fontsize=14, fontweight="bold", color=CPX_NAVY, pad=12
)

plt.tight_layout()
save_fig(fig, "lbe_budget_scenario_over_2026_forecast.png")
display(fig)
plt.close(fig)

# ── Summary Table ─────────────────────────────────────────────────────────────
tbl_rows_2026 = []
for i in range(n_scen):
    lift_val = scen_lift[i]
    proj = scen_proj_rev[i]
    row_data = _scen_extended.iloc[i]
    ci_lo_proj_2025 = row_data["CI Low ($M)"]
    ci_hi_proj_2025 = row_data["CI High ($M)"]
    conf = row_data["Confidence"]

    _non_media_2025_M = _lbe_rev_2025 / 1e6 - lbe_baseline_rev / 1e6
    ci_lo_media_only = ci_lo_proj_2025 - _non_media_2025_M
    ci_hi_media_only = ci_hi_proj_2025 - _non_media_2025_M
    ci_lo_2027 = forecast_2027_M + (ci_lo_media_only - lbe_baseline_rev / 1e6)
    ci_hi_2027 = forecast_2027_M + (ci_hi_media_only - lbe_baseline_rev / 1e6)

    tbl_rows_2026.append({
        "Scenario": scen_labels[i],
        "Type": row_data["Type"],
        "2027 Forecast ($M)": round(forecast_2027_M, 2),
        "Scenario Lift ($M)": round(lift_val, 2),
        "Projected 2026 Rev ($M)": round(proj, 2),
        "CI Low ($M)": round(ci_lo_2027, 2),
        "CI High ($M)": round(ci_hi_2027, 2),
        "Lift vs Forecast %": scen_lift_pct[i],
        "Confidence": conf,
    })

print(f"\nBudget Scenarios Over 2026 Forecast (${forecast_2027_M:.1f}M)")
display(pd.DataFrame(tbl_rows_2026).astype(str))

# COMMAND ----------

# DBTITLE 1,2027 Scenario Upside Ladder
# ── 2027 SCENARIO UPSIDE — Ladder to Best Case (LBE, revenue) ────────────────
# EXH "2027 Scenario Upside" ported to LBE: revenue lift (not admissions),
# laddered over FORECAST_2027, 90% model-uncertainty CI.
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# CPX_GREEN isn't in the LBE Plots-Config palette (only EXH's) — define it here.
CPX_GREEN = "#2E7D32"

# ── Build plot frame from scenario_df (drop the zero-lift Baseline row) ───────
_ld = scenario_df[scenario_df["Type"] != "Baseline"].copy()

# ── Add +3% and +5% budget scenarios (computed on-the-fly) ────────────────────
for _pct in [3, 5]:
    _factor = 1 + _pct / 100
    _aw_s = lbe_baseline_aw * _factor
    _co_s = lbe_baseline_co * _factor
    _rev, _ci_lo, _ci_hi = lbe_funnel_contribution_ci(_aw_s, _co_s)
    _lift = (_rev - lbe_baseline_rev) / 1e6
    _actual_rev_M = _lbe_rev_2025 / 1e6
    _proj = _actual_rev_M + _lift
    _ci_lo_M = _actual_rev_M + (_ci_lo - lbe_baseline_rev) / 1e6
    _ci_hi_M = _actual_rev_M + (_ci_hi - lbe_baseline_rev) / 1e6
    _new_row = pd.DataFrame([{
        "Scenario": f"+{_pct}% Budget Uplift",
        "Type": "Budget",
        "2025 Actual Rev ($M)": _actual_rev_M,
        "Scenario Lift ($M)": _lift,
        "Projected Rev ($M)": _proj,
        "CI Low ($M)": _ci_lo_M,
        "CI High ($M)": _ci_hi_M,
        "Lift vs Actual %": f"+{_lift / _actual_rev_M * 100:.1f}%",
        "Confidence": "Medium",
    }])
    _ld = pd.concat([_ld, _new_row], ignore_index=True)

# Numeric lift ($M) and gain-CI ($M), re-centered on the lift.
# scenario_df CIs are on PROJECTED TOTAL; convert to CI around the GAIN by
# subtracting the projected point and adding the lift.
_ld["gain"]   = _ld["Scenario Lift ($M)"].astype(float)
_ld["ci_lo"]  = _ld["CI Low ($M)"].astype(float)  - _ld["Projected Rev ($M)"].astype(float) + _ld["gain"]
_ld["ci_hi"]  = _ld["CI High ($M)"].astype(float) - _ld["Projected Rev ($M)"].astype(float) + _ld["gain"]

df_plot = _ld.sort_values("gain", ascending=True).reset_index(drop=True)
import re
# Replace "XX% Shift ..." with "Reshuffle" and strip any remaining "Co→Aw" / "Co->Aw" text
labels  = [re.sub(r'\d+%\s*Shift', 'Reshuffle', l) for l in df_plot["Scenario"].tolist()]
labels  = [re.sub(r'[:\s]*Co\s*[->→]+\s*Aw', '', l).strip() for l in labels]
labels  = ["Reshuffle Only" if l == "Reshuffle" else l for l in labels]
gains   = df_plot["gain"].to_numpy()
lows    = df_plot["ci_lo"].to_numpy()
highs   = df_plot["ci_hi"].to_numpy()
n       = len(df_plot)
ceiling = gains.max()

FORECAST_2027_M = FORECAST_2027 / 1e6   # already defined in Plots Config

# ── Colour by Type (LBE casing) + best-case override ─────────────────────────
def bar_color(row):
    t = str(row["Type"]).lower()
    if t == "budget":       return CPX_NAVY
    if t == "combined":     return CPX_GREEN
    if t == "reshuffle":    return CPX_RED
    return CPX_MGREY

colors = [bar_color(df_plot.iloc[i]) for i in range(n)]
RECOMMENDED = labels[int(np.argmax(gains))]
for i in range(n):
    if labels[i] == RECOMMENDED:
        colors[i] = CPX_GOLD                              # best case → gold
    elif str(df_plot.iloc[i]["Type"]).lower() == "reshuffle":
        colors[i] = CPX_RED                               # reshuffle → red

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 4.6))
y      = np.arange(n)
floor  = min(gains.min(), 0.0)          # extend left of zero if any gain is negative
span   = ceiling - floor                 # full value range across the ladder
lab_pad = span * 0.012                    # small offset for in/outside-bar labels

# ── Right-side layout matched to EXH ladder proportions ───────────────────────
# EXH splits horizontal space ~65% bars / ~35% label band, anchored to `ceiling`
# (bars end at ceiling; CI + total columns sit in the right band). We reproduce
# that split while keeping the negative floor for the reshuffle bar.
pad     = ceiling * 0.06                  # small left pad for the negative bar
x_lo    = floor - pad
x_hi    = x_lo + (ceiling - x_lo) / 0.65  # right edge so bars end at ~65% across
band    = x_hi - ceiling                   # the ~35% right-hand label band
x_ci    = ceiling * 1.02                    # CI column, just past bar-end (EXH exact)
x_2026  = x_ci + band * 0.42                # 2026 column, spaced within the band
x_max   = x_hi                              # xlim right edge

for i in range(n):
    is_rec = labels[i] == RECOMMENDED
    c = colors[i]
    g = gains[i]
    # grey "gap to best case" track spans full range so every row reads consistently
    ax.barh(y[i], span, left=floor, height=0.72, color=CPX_LGREY, alpha=0.28,
            edgecolor="none", zorder=1)
    ax.barh(y[i], g, height=0.72, color=c,
            alpha=0.92 if is_rec else 0.8, edgecolor="none", zorder=2)
    # Label placement:
    #  • positive bar, wide enough → white label inside, right-aligned at bar tip
    #  • positive bar, too short    → coloured label just right of the tip
    #  • negative bar               → coloured label just right of ZERO (bar grows
    #                                  left into the tick gutter, so never label inside)
    if g < 0:
        ax.text(lab_pad, y[i], f"{g:+.2f}M",
                ha="left", va="center", fontsize=9, fontweight="bold", color=c, zorder=4)
    elif g >= span * 0.10:              # wide positive → white label inside
        ax.text(g - lab_pad, y[i], f"{g:+.2f}M",
                ha="right", va="center",
                fontsize=10 if is_rec else 9, fontweight="bold", color="white", zorder=4)
    else:                               # short positive → coloured label outside
        ax.text(g + lab_pad, y[i], f"{g:+.2f}M",
                ha="left", va="center",
                fontsize=9, fontweight="bold", color=c, zorder=4)
    rev_27 = FORECAST_2027_M + g
    ax.text(x_ci, y[i], f"[{lows[i]:.2f}–{highs[i]:.2f}]",
            ha="left", va="center", fontsize=9, fontweight="bold", color=c, zorder=4, clip_on=False)
    ax.text(x_2026, y[i], f"2027: ${rev_27:.1f}M",
            ha="left", va="center", fontsize=8, fontweight="bold", color=CPX_GREEN, zorder=4, clip_on=False)

# zero reference line (only meaningful when negatives exist)
if floor < 0:
    ax.axvline(0, color=CPX_MGREY, lw=1.0, ls="-", alpha=0.6, zorder=3)

ax.axvline(ceiling, color=CPX_RED, lw=1.4, ls="--", alpha=0.8, zorder=5)
# Best-case callout: horizontal label ABOVE the top bar, at the ceiling line — out
# of the CI lane entirely, so nothing overlaps.
ax.text(ceiling, n - 0.42, f"Best case  +${ceiling:.2f}M",
        ha="center", va="bottom", fontsize=8.5, fontweight="bold",
        color=CPX_RED, zorder=6)

ax.set_yticks(y)
ytick_labels = [f"★ {l}" if l == RECOMMENDED else l for l in labels]
ax.set_yticklabels(ytick_labels, fontsize=9.5, fontweight="bold", color=CPX_DGREY)
ax.tick_params(axis="y", length=0, pad=8)
ax.set_xlim(x_lo, x_max)
ax.set_ylim(-0.7, n + 0.7)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:+.1f}M"))
plt.setp(ax.get_xticklabels(), fontsize=8.5, fontweight="bold", color=CPX_DGREY)
for sp in ("top", "right", "left", "bottom"):
    ax.spines[sp].set_visible(False)
ax.grid(axis="x", alpha=0.14, color=CPX_LGREY)
ax.set_axisbelow(True)
ax.tick_params(axis="x", labelbottom=False, length=0)

leg = [mpatches.Patch(facecolor=CPX_NAVY,  label="Budget uplift"),
       mpatches.Patch(facecolor=CPX_GREEN, label="Combined"),
       mpatches.Patch(facecolor=CPX_GOLD,  label="Best case scenario"),
       mpatches.Patch(facecolor=CPX_RED,   label="Reshuffle"),
       mpatches.Patch(facecolor=CPX_LGREY, alpha=0.28, label="Gap to best case"),
       Line2D([0], [0], color=CPX_RED,   lw=1.4, ls="--", alpha=0.8, label="Best case line"),
       Line2D([0], [0], color=CPX_DGREY, lw=1.5, ls="-",  label="[low–high] = 90% CI")]
fig.legend(handles=leg, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, 0.90),
           ncol=4, frameon=True, framealpha=0.9, edgecolor=CPX_LGREY,
           borderpad=0.5, labelspacing=0.5, handlelength=1.5)

fig.suptitle("LBE 2027 Scenario Upside — Climb to Best Case (Revenue)",
             fontsize=13, fontweight="bold", color=CPX_NAVY, y=0.995)
fig.text(0.99, 0.005, "Cineplex Marketing Analytics  |  LBE MMM V5e",
         ha="right", va="bottom", fontsize=6, color=CPX_LGREY)

# reserve top margin: title → legend → best-case callout → bars, cleanly stacked
plt.tight_layout(rect=[0, 0, 1, 0.82])
save_fig(fig, "FINAL_2027_lbe_scenario_ladder_compact.png")
display(fig)
plt.close(fig)

# COMMAND ----------

# ── 2026 REVENUE BUILD — Baseline + Media Split (Pre-Scenario Lift) ────────────
# Single-panel LBE version of the EXH attendance-build strip.
# Revenue (not attendance); 2026 forecast only.
import matplotlib.patches as mpatches

CPX_GREY = "#8A8A8A"   # not in the LBE palette (only L/M/D grey) — define locally

# ── 2026 media share, reusing the annual-decomp logic ─────────────────────────
# decomp_df (from the Annual Revenue Decomposition cell) holds weekly media_*;
# 2025 media share of actual revenue is applied to the 2026 forecast, exactly as
# the decomposition cell does at its "2026 Forecast (inherits 2025 funnel mix)" step.
_2025 = decomp_df[decomp_df["year"] == 2025]
_media_2025  = (_2025["media_Awareness"].sum() + _2025["media_Consideration"].sum())
_actual_2025 = _2025["total_rev"].sum()
_ms          = _media_2025 / _actual_2025            # 2025 media share, stable assumption

total = FORECAST_2027 / 1e6
media = FORECAST_2027 * _ms / 1e6
nm    = total - media

# ── Plot: single horizontal build bar ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 2.4))

ax.barh(0, nm, height=0.32, color=CPX_GREY, alpha=0.65, edgecolor="none", zorder=2)
ax.barh(0, media, height=0.32, color=CPX_NAVY, alpha=0.85, edgecolor="none", zorder=2, left=nm)

# Non-Media is wide → label centered inside.
ax.text(nm * 0.5, 0, f"Non-Media\n${nm:.1f}M  ({nm/total*100:.0f}%)",
        ha="center", va="center", fontsize=11, fontweight="bold", color="black", zorder=3)

# Media segment can be thin (e.g. 9%) → if too narrow for a centered label, place it
# OUTSIDE to the right of the bar, in navy, so it never bleeds off the segment.
_media_frac = media / total
if _media_frac >= 0.15:
    ax.text(nm + media * 0.5, 0, f"Media\n${media:.1f}M  ({_media_frac*100:.0f}%)",
            ha="center", va="center", fontsize=11, fontweight="bold", color="white", zorder=3)
else:
    ax.text(total + total * 0.015, 0, f"Media\n${media:.1f}M  ({_media_frac*100:.0f}%)",
            ha="left", va="center", fontsize=11, fontweight="bold", color=CPX_NAVY, zorder=3)

ax.set_title(f"2027 (Forecast)\n${total:.1f}M", fontsize=12, fontweight="bold", color="black", pad=4)
ax.set_yticks([])
ax.set_ylim(-0.22, 0.22)
ax.set_xlim(0, total * 1.04)
ax.tick_params(axis="x", labelbottom=False, length=0)
for sp in ("top", "right", "left", "bottom"):
    ax.spines[sp].set_visible(False)
ax.grid(axis="x", alpha=0.12, color=CPX_LGREY)
ax.set_axisbelow(True)

leg = [mpatches.Patch(facecolor=CPX_GREY, alpha=0.65, label="Baseline / Non-Media"),
       mpatches.Patch(facecolor=CPX_NAVY, alpha=0.85, label="Media Attribution")]
fig.legend(handles=leg, fontsize=9, loc="lower center", bbox_to_anchor=(0.5, -0.02),
           ncol=2, framealpha=0.9, edgecolor=CPX_LGREY, borderpad=0.5)

fig.suptitle("LBE 2027 Revenue Build — Baseline + Media Split (Pre-Scenario Lift)",
             fontsize=13, fontweight="bold", color=CPX_NAVY, y=1.06)
fig.text(0.99, -0.10, "Cineplex Marketing Analytics  |  LBE MMM V5e",
         ha="right", va="bottom", fontsize=6, color=CPX_LGREY)

plt.tight_layout(rect=[0, 0.06, 1, 0.94])
save_fig(fig, "FINAL_lbe_revenue_build_2026.png")
display(fig)
plt.close(fig)

# COMMAND ----------


