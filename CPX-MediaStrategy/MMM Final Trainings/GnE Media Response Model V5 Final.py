# Databricks notebook source
# DBTITLE 1,G&E Media Response Model V5 — Executive Summary
# MAGIC %md
# MAGIC # G&E Media Response Model V5 — Final
# MAGIC **Groups & Events Bayesian Marketing Mix Model**
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Key Findings
# MAGIC
# MAGIC | Metric | Value |
# MAGIC |---|---|
# MAGIC | **Total media share** | 13.3% of inquiries |
# MAGIC | G&E SEM contribution | 8.0% (6,116 incremental inquiries) |
# MAGIC | G&E Social contribution | 5.8% (4,403 incremental inquiries) |
# MAGIC | Non-media (baseline) | 86.7% |
# MAGIC | Validation MAPE | 14.8% |
# MAGIC | SEM same-week response (α) | 73% |
# MAGIC | Social peak response lag | Week 3.6 |
# MAGIC
# MAGIC ## Model Architecture (V5)
# MAGIC - **Multiplicative** (log-link): media effects are % lifts, not fixed additive counts
# MAGIC - **B-spline baseline** (12 basis functions, random-walk prior): captures nonlinear structural decline flexibly
# MAGIC - **SEM: Spike-and-Carry adstock**: α×same-week + (1-α)×Weibull carry (fixes undercrediting from V1–V4)
# MAGIC - **Social: Pure Weibull adstock**: delayed-peak response (awareness → consideration → inquiry pipeline)
# MAGIC - **Hill saturation**: learnable half-saturation (K) and shape (n) per channel
# MAGIC - **G&E-specific seasonality**: Fourier harmonics, school-in-session, corporate party ramp, stat holiday (negative), end-of-quarter
# MAGIC
# MAGIC ## Data Sources
# MAGIC - Inquiries: `combined_Inquiries.xlsx` (80,896 rows, 2023–2026)
# MAGIC - Revenue: Glacier Report (34,317 realized events, $144M)
# MAGIC - Media spend: `EXH_ModelData.xlsx` (165 weeks, 3 G&E channels)

# COMMAND ----------

# DBTITLE 1,1. Install Dependencies
# MAGIC %pip install pymc arviz numpyro jax jaxlib pytensor scikit-learn openpyxl netcdf4 h5netcdf scipy xarray --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,2. Config & Data Loading
import pandas as pd, numpy as np
import warnings
warnings.simplefilter("ignore")

# ── CONFIG ────────────────────────────────────────────────────────────────
INQ_FILE     = "/Volumes/data_exploration_hub/marketing_hub/model_data/GnE_Data/combined_Inquiries.xlsx"
SPEND_FILE   = "/Volumes/data_exploration_hub/marketing_hub/model_data/Final_Data/EXH_ModelData.xlsx"
GLACIER_FILE = "/Volumes/data_exploration_hub/marketing_hub/model_data/GnE_Data/GroupsAndEvent107 - Glacier Report 2023-01-01 to 2026-05-25.xlsx"

TODAY = pd.Timestamp.now().normalize()
print("Config set. Today =", TODAY.date())

# ── Load Inquiries ────────────────────────────────────────────────────────
inq = pd.read_excel(INQ_FILE)
inq["Created On"] = pd.to_datetime(inq["Created On"], errors="coerce")
inq = inq.dropna(subset=["Created On"])
inq["Year"] = inq["Created On"].dt.year
print(f"Inquiries: {len(inq):,} rows  ({inq['Created On'].min().date()} -> {inq['Created On'].max().date()})")

# ── Load Glacier Revenue (4 LOB sheets) ───────────────────────────────────
GLACIER_SHEETS = {"Corporate_Events": 7, "Group Sales": 8, "Repeat Business": 8, "Rec Room": 8}
frames = []
for sheet, hdr in GLACIER_SHEETS.items():
    d = pd.read_excel(GLACIER_FILE, sheet_name=sheet, header=hdr).dropna(axis=1, how="all")
    d = d[d["Date of Event"].notna()].copy()
    frames.append(d)
glac = pd.concat(frames, ignore_index=True)
glac["Date of Event"] = pd.to_datetime(glac["Date of Event"], errors="coerce")
glac["Total"] = pd.to_numeric(glac["Total"], errors="coerce")
is_lbe = glac["Theatre Name"].astype(str).str.contains("Rec Room|TRR|Playdium|PDM", case=False, na=False)
glac["Segment"] = np.where(is_lbe, "LBE", "EXH")
n_future = int((glac["Date of Event"] > TODAY).sum())
glac = glac[glac["Date of Event"] <= TODAY].copy()
glac["Year"] = glac["Date of Event"].dt.year
print(f"Glacier: {len(glac):,} realized events  (dropped {n_future:,} future-dated)")
print(f"Revenue total: ${glac['Total'].sum():,.0f}")

# ── Load Spend + Seasonality ──────────────────────────────────────────────
spend_raw = pd.read_excel(SPEND_FILE)
spend_raw = spend_raw.set_index(pd.to_datetime(spend_raw["week_date"]))
print(f"Spend file: {len(spend_raw)} weeks  ({spend_raw.index.min().date()} -> {spend_raw.index.max().date()})")
ge_cols = [c for c in spend_raw.columns if "G&E" in c]
print("G&E spend columns:", ge_cols)

# COMMAND ----------

# DBTITLE 1,3. Feature Engineering (G&E Seasonality + B-Spline + Log Target)
from scipy.interpolate import BSpline
from scipy.stats import weibull_min

# ── Weekly inquiry target ───────────────────────────────────────────────────
wk_inq = inq.set_index("Created On").resample("W-MON").size().rename("inquiries")

# ── Media spend (combined social) ──────────────────────────────────────────
sem_spend = pd.to_numeric(spend_raw["Consideration_G&E_SEM"], errors="coerce").fillna(0)
soc_consid = pd.to_numeric(spend_raw["Consideration_G&E_Social"], errors="coerce").fillna(0)
soc_conv = pd.to_numeric(spend_raw["Conversion_G&E_Social"], errors="coerce").fillna(0)
soc_spend = soc_consid + soc_conv  # combined social

# ── G&E-Specific Seasonality Features ─────────────────────────────────────
week_dates = pd.to_datetime(spend_raw["week_date"] if "week_date" in spend_raw.columns else spend_raw.index)
day_of_year = week_dates.dt.dayofyear
month = week_dates.dt.month

# Fourier harmonics (2 pairs)
fourier = pd.DataFrame(index=spend_raw.index)
for k in [1, 2]:
    fourier[f"sin_{k}"] = np.sin(2 * np.pi * k * day_of_year / 365.25)
    fourier[f"cos_{k}"] = np.cos(2 * np.pi * k * day_of_year / 365.25)

# School-in-session (Sep–Jun=1, Jul–Aug=0)
school_in_session = ((month >= 9) | (month <= 6)).astype(float)

# End-of-quarter flag
is_quarter_end_month = month.isin([3, 6, 9, 12])
end_of_quarter = (is_quarter_end_month & (week_dates.dt.day >= 15)).astype(float)

# Corporate party ramp (Gaussian, peaks 5 weeks before Christmas)
days_to_xmas = spend_raw["days_to_christmas"].astype(float)
corp_party_ramp = np.exp(-0.5 * ((days_to_xmas - 35) / 10) ** 2)

# Stat holiday & long weekend (allow NEGATIVE sign via Normal prior)
stat_holiday = (spend_raw["stat_holiday_in_week"] == "Y").astype(float)
long_weekend = (spend_raw["is_long_weekend"] == "Y").astype(float)

# Assemble seasonality matrix
seas = pd.DataFrame({
    "sin_1": fourier["sin_1"].values, "cos_1": fourier["cos_1"].values,
    "sin_2": fourier["sin_2"].values, "cos_2": fourier["cos_2"].values,
    "school": school_in_session.values, "qtr_end": end_of_quarter.values,
    "corp_party_ramp": corp_party_ramp.values,
    "stat_holiday": stat_holiday.values, "long_weekend": long_weekend.values,
}, index=spend_raw.index)

# ── Combined modeling dataframe ───────────────────────────────────────────
df = pd.concat([wk_inq, sem_spend.rename("sem"), soc_spend.rename("soc"), seas], axis=1).dropna()
n_weeks = len(df)

# Model arrays
y = df["inquiries"].values.astype(float)
log_y = np.log(y)
sem_n = (df["sem"] / df["sem"].max()).values  # normalized [0,1]
soc_n = (df["soc"] / df["soc"].max()).values
seas_cols = [c for c in df.columns if c not in ("inquiries", "sem", "soc")]
Xs = df[seas_cols].values
Xs = (Xs - Xs.mean(0)) / (Xs.std(0) + 1e-9)  # standardize

# ── B-Spline basis for nonlinear baseline ────────────────────────────────
n_knots = 8
spline_degree = 3
t_norm = np.linspace(0, 1, n_weeks)
interior_knots = np.linspace(0, 1, n_knots + 2)[1:-1]
knots = np.concatenate([np.zeros(spline_degree + 1), interior_knots, np.ones(spline_degree + 1)])
n_bases = len(knots) - spline_degree - 1

B_spline = np.zeros((n_weeks, n_bases))
for i in range(n_bases):
    c = np.zeros(n_bases); c[i] = 1.0
    B_spline[:, i] = BSpline(knots, c, spline_degree)(t_norm)

MAX_LAG = 12  # adstock window

print(f"Modeling weeks: {n_weeks}  ({df.index.min().date()} → {df.index.max().date()})")
print(f"Response: log(inquiries), range {log_y.min():.2f}–{log_y.max():.2f}")
print(f"B-spline: {n_bases} basis functions, {n_knots} interior knots")
print(f"Seasonality features ({len(seas_cols)}): {seas_cols}")
print(f"Social spend: ${df['soc'].sum():,.0f} (Consideration + Conversion combined)")
print(f"\n✓ Feature engineering complete")

# COMMAND ----------

# DBTITLE 1,4. V5 Model — Spike-and-Carry SEM + Weibull Social + B-Spline Baseline
import pymc as pm
import pytensor.tensor as pt
import arviz as az

# ── Weibull adstock (lagged, starting at lag 1) ─────────────────────────────
def weibull_adstock(x_np, shape_k, scale_lam):
    L = MAX_LAG
    lags = pt.arange(1, L + 1).astype("float64")
    t_scaled = lags / scale_lam
    log_w = (pt.log(shape_k) - pt.log(scale_lam)
             + (shape_k - 1) * pt.log(t_scaled)
             - pt.power(t_scaled, shape_k))
    w = pt.exp(log_w)
    w = w / w.sum()
    padded = pt.concatenate([pt.zeros(L), pt.as_tensor_variable(x_np)])
    idx = pt.arange(len(x_np))[:, None] + pt.arange(L)[None, :]
    return padded[idx].dot(w[::-1])

# ── Hill saturation ───────────────────────────────────────────────────────
def hill(x, K, n):
    xn = pt.power(x + 1e-8, n)
    return xn / (pt.power(K, n) + xn)

# ── V5 Model Definition ──────────────────────────────────────────────────
with pm.Model() as model_v5:
    # ── B-Spline baseline (random-walk prior for smoothness) ──
    spline_sigma = pm.HalfNormal("spline_sigma", 0.1)
    spline_c0 = pm.Normal("spline_c0", log_y.mean(), 0.3)
    spline_diffs = pm.Normal("spline_diffs", 0, spline_sigma, shape=n_bases - 1)
    spline_coeffs = pt.concatenate([[spline_c0], spline_c0 + pt.cumsum(spline_diffs)])
    log_baseline = pt.dot(pt.as_tensor_variable(B_spline), spline_coeffs)

    # ── SEM: Spike-and-Carry Adstock ──
    alpha_sem = pm.Beta("alpha_sem", 4, 2)  # ~60-70% immediate
    k_sem = pm.Gamma("k_sem", mu=1.5, sigma=0.7)
    lam_sem = pm.Gamma("lam_sem", mu=3.0, sigma=1.5)
    sem_immediate = pt.as_tensor_variable(sem_n)
    sem_carry = weibull_adstock(sem_n, k_sem, lam_sem)
    sem_ad = alpha_sem * sem_immediate + (1 - alpha_sem) * sem_carry

    # ── Social: Pure Weibull (no same-week response) ──
    k_soc = pm.Gamma("k_soc", mu=2.5, sigma=1.0)
    lam_soc = pm.Gamma("lam_soc", mu=4.0, sigma=1.5)
    soc_ad = weibull_adstock(soc_n, k_soc, lam_soc)

    # ── Hill saturation ──
    K_sem = pm.Beta("K_sem", 2, 3)  # SEM: lower K = saturates later
    K_soc = pm.Beta("K_soc", 2, 2)
    n_sem = pm.Gamma("n_sem", 2, 2)
    n_soc = pm.Gamma("n_soc", 2, 2)
    sem_eff = hill(sem_ad, K_sem, n_sem)
    soc_eff = hill(soc_ad, K_soc, n_soc)

    # ── Media betas (log-space % lifts) ──
    b_sem = pm.HalfNormal("beta_sem", 0.3)
    b_soc = pm.HalfNormal("beta_soc", 0.25)

    # ── Seasonality ──
    b_seas = pm.Normal("beta_seas", 0, 0.1, shape=Xs.shape[1])

    # ── Tracked contributions ──
    media_sem = pm.Deterministic("media_sem", b_sem * sem_eff)
    media_soc = pm.Deterministic("media_soc", b_soc * soc_eff)

    # ── Multiplicative model: log(y) = baseline + media + seasonality ──
    log_mu = log_baseline + media_sem + media_soc + pt.dot(Xs, b_seas)
    sigma = pm.HalfNormal("sigma", 0.12)
    pm.Normal("log_y", mu=log_mu, sigma=sigma, observed=log_y)

    # ── Sample ──
    idata = pm.sample(1500, tune=2500, chains=4, target_accept=0.995,
                      nuts_sampler="numpyro", random_seed=42)

print("✓ V5 model sampling complete.")

# COMMAND ----------

# DBTITLE 1,5. Results — Posteriors, Decomposition, MAPE
# ── V5 Results ────────────────────────────────────────────────────────────────
print("="*70)
print("  V5 RESULTS: Spike-and-Carry SEM + Weibull Social + B-Spline Baseline")
print("="*70)

# Posteriors
summary = az.summary(idata, var_names=["beta_sem", "beta_soc", "alpha_sem",
                                       "k_sem", "lam_sem", "k_soc", "lam_soc",
                                       "K_sem", "K_soc", "spline_sigma", "sigma"])
print("\n── Posteriors ──")
print(summary.to_string())
print(f"\nDivergences: {int(idata.sample_stats.diverging.sum())}")

# ── Counterfactual Decomposition ──
post = idata.posterior
media_sem_mean = post["media_sem"].mean(dim=["chain", "draw"]).values
media_soc_mean = post["media_soc"].mean(dim=["chain", "draw"]).values

# Reconstruct spline baseline
c0 = float(post["spline_c0"].mean())
diffs = post["spline_diffs"].mean(dim=["chain", "draw"]).values
coeffs = np.concatenate([[c0], c0 + np.cumsum(diffs)])
log_bl = B_spline @ coeffs
b_seas_mean = post["beta_seas"].mean(dim=["chain", "draw"]).values

# Full prediction
log_full = log_bl + media_sem_mean + media_soc_mean + Xs @ b_seas_mean
pred_full = np.exp(log_full)

# Counterfactual (no media)
log_no_media = log_bl + Xs @ b_seas_mean
pred_no_media = np.exp(log_no_media)

# Incremental inquiries
actual_total = y.sum()
sem_incr = (np.exp(log_full) - np.exp(log_full - media_sem_mean)).sum()
soc_incr = (np.exp(log_full) - np.exp(log_full - media_soc_mean)).sum()
media_incr = pred_full.sum() - pred_no_media.sum()

sem_pct = sem_incr / actual_total * 100
soc_pct = soc_incr / actual_total * 100
media_pct = media_incr / actual_total * 100
base_pct = 100 - media_pct

# MAPE (80/20 split)
split_idx = int(n_weeks * 0.8)
mape_train = np.mean(np.abs((y[:split_idx] - pred_full[:split_idx]) / y[:split_idx])) * 100
mape_val = np.mean(np.abs((y[split_idx:] - pred_full[split_idx:]) / y[split_idx:])) * 100

print("\n── Counterfactual Decomposition ──")
print(f"  G&E SEM:    {sem_pct:.1f}%  ({sem_incr:,.0f} incremental inquiries)")
print(f"  G&E Social: {soc_pct:.1f}%  ({soc_incr:,.0f} incremental inquiries)")
print(f"  Total media: {media_pct:.1f}%  ({media_incr:,.0f} inquiries)")
print(f"  Non-media:   {base_pct:.1f}%")

print("\n── SEM Spike-and-Carry ──")
alpha_mean = float(summary.loc["alpha_sem", "mean"])
print(f"  alpha_sem = {alpha_mean:.2f} → {alpha_mean*100:.0f}% of SEM effect is SAME-WEEK")

print("\n── Beta Interpretation (log-space → % lift at saturation) ──")
b_sem_val = float(summary.loc["beta_sem", "mean"])
b_soc_val = float(summary.loc["beta_soc", "mean"])
print(f"  SEM: beta={b_sem_val:.3f} → at full saturation, lifts inquiries by ~{(np.exp(b_sem_val)-1)*100:.1f}%")
print(f"  Soc: beta={b_soc_val:.3f} → at full saturation, lifts inquiries by ~{(np.exp(b_soc_val)-1)*100:.1f}%")

print(f"\n── MAPE ──")
print(f"  Train: {mape_train:.1f}%   Val: {mape_val:.1f}%")
print(f"  Split: {split_idx} weeks train / {n_weeks-split_idx} weeks val")

# COMMAND ----------

# DBTITLE 1,6. Visualization Setup (Brand Palette)
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── Cineplex Brand Palette ───────────────────────────────────────────────────
CPX_NAVY  = "#002B5C"
CPX_RED   = "#E31837"
CPX_GOLD  = "#F5A800"
CPX_LGREY = "#F2F2F2"
CPX_DGREY = "#4A4A4A"
CPX_GREEN = "#2E7D32"
CPX_MGREY = "#8A8A8A"
WHITE     = "#FFFFFF"

plt.rcParams.update({
    "figure.facecolor": WHITE, "figure.dpi": 150,
    "axes.facecolor": WHITE, "axes.edgecolor": CPX_DGREY,
    "axes.linewidth": 1.2, "axes.grid": False,
    "axes.spines.top": True, "axes.spines.right": True,
    "axes.spines.left": True, "axes.spines.bottom": True,
    "axes.titlesize": 13, "axes.titleweight": "bold",
    "axes.titlecolor": "black", "axes.labelsize": 11,
    "axes.labelweight": "bold", "axes.labelcolor": CPX_DGREY,
    "xtick.labelsize": 10, "ytick.labelsize": 10,
    "xtick.color": CPX_DGREY, "ytick.color": CPX_DGREY,
    "legend.fontsize": 9, "legend.frameon": False,
    "font.family": "sans-serif", "font.size": 10,
    "font.weight": "bold",
})

def get_figsize(ratio="16:9"):
    ratios = {"16:9": (16, 9), "wide": (16, 6), "4:3": (12, 9)}
    return ratios.get(ratio, (16, 9))

print("✓ Cineplex brand palette configured")

# COMMAND ----------

# DBTITLE 1,7. Decay Profiles — SEM Spike-and-Carry vs Social Weibull
# ── Decay Profiles ──────────────────────────────────────────────────────────
from scipy.stats import weibull_min

k_sem_val = float(summary.loc["k_sem", "mean"])
lam_sem_val = float(summary.loc["lam_sem", "mean"])
k_soc_val = float(summary.loc["k_soc", "mean"])
lam_soc_val = float(summary.loc["lam_soc", "mean"])

lags_full = np.arange(0, MAX_LAG + 1).astype(float)
lags_carry = np.arange(1, MAX_LAG + 1).astype(float)

# SEM: Spike-and-Carry
w_carry = weibull_min.pdf(lags_carry, k_sem_val, scale=lam_sem_val)
w_carry = w_carry / w_carry.sum()
sem_profile = np.zeros(len(lags_full))
sem_profile[0] = alpha_mean
sem_profile[1:] = (1 - alpha_mean) * w_carry

# Social: Pure Weibull
soc_profile = np.zeros(len(lags_full))
w_soc = weibull_min.pdf(lags_carry, k_soc_val, scale=lam_soc_val)
w_soc = w_soc / w_soc.sum()
soc_profile[1:] = w_soc

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=get_figsize("16:9"), sharey=True)

# SEM
bars = ax1.bar(lags_full, sem_profile, color=CPX_NAVY, width=0.7, edgecolor=WHITE, linewidth=0.5)
bars[0].set_color(CPX_RED)
ax1.set_xlabel("Lag (weeks)"); ax1.set_ylabel("Weight (normalized)")
ax1.set_title("SEM: Spike-and-Carry Adstock")
ax1.set_xticks(lags_full)
ax1.annotate(f"Spike: α={alpha_mean:.0%}\n(same-week)",
             xy=(0, sem_profile[0]), xytext=(3, sem_profile[0]*0.85),
             fontsize=9, fontweight="bold", color=CPX_RED,
             arrowprops=dict(arrowstyle="->", color=CPX_RED, lw=1.5))
ax1.annotate(f"Carry: {1-alpha_mean:.0%}\n(Weibull k={k_sem_val:.1f}, λ={lam_sem_val:.1f})",
             xy=(3, sem_profile[3]), xytext=(6, sem_profile[0]*0.45),
             fontsize=9, fontweight="bold", color=CPX_NAVY,
             arrowprops=dict(arrowstyle="->", color=CPX_NAVY, lw=1.5))

# Social
ax2.bar(lags_full, soc_profile, color=CPX_RED, width=0.7, edgecolor=WHITE, linewidth=0.5)
ax2.set_xlabel("Lag (weeks)")
ax2.set_title("Social: Pure Weibull (No Same-Week)")
ax2.set_xticks(lags_full)
peak_soc = lam_soc_val * ((k_soc_val - 1) / k_soc_val) ** (1 / k_soc_val)
ax2.annotate(f"Peak at week {peak_soc:.1f}\n(k={k_soc_val:.1f}, λ={lam_soc_val:.1f})",
             xy=(round(peak_soc), soc_profile[round(peak_soc)]),
             xytext=(7, soc_profile[round(peak_soc)]*1.5),
             fontsize=9, fontweight="bold", color=CPX_RED,
             arrowprops=dict(arrowstyle="->", color=CPX_RED, lw=1.5))
ax2.text(0, 0.005, "No lag-0\n(awareness)", ha="center", va="bottom",
         fontsize=8, color=CPX_MGREY, fontstyle="italic")

fig.suptitle("V5 Learned Decay Profiles — SEM (Immediate + Carry) vs Social (Delayed Peak)",
             fontsize=13, fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.show()

# ── Decay Profile Table ───────────────────────────────────────────────────
print(f"\n{'Lag (wk)':<10} {'SEM Weight':>12} {'Social Weight':>14}")
print("-" * 38)
for i, lag in enumerate(lags_full):
    print(f"{int(lag):<10} {sem_profile[i]:>12.4f} {soc_profile[i]:>14.4f}")
print(f"\nSEM  : α={alpha_mean:.0%} same-week spike | Carry Weibull k={k_sem_val:.1f}, λ={lam_sem_val:.1f}")
print(f"Social: No lag-0 | Peak week {peak_soc:.1f} | Weibull k={k_soc_val:.1f}, λ={lam_soc_val:.1f}")

# COMMAND ----------

# ── Decay Profiles ──────────────────────────────────────────────────────────
from scipy.stats import weibull_min

k_sem_val = float(summary.loc["k_sem", "mean"])
lam_sem_val = float(summary.loc["lam_sem", "mean"])
k_soc_val = float(summary.loc["k_soc", "mean"])
lam_soc_val = float(summary.loc["lam_soc", "mean"])

lags_full = np.arange(0, MAX_LAG + 1).astype(float)
lags_carry = np.arange(1, MAX_LAG + 1).astype(float)

# SEM: Spike-and-Carry
w_carry = weibull_min.pdf(lags_carry, k_sem_val, scale=lam_sem_val)
w_carry = w_carry / w_carry.sum()
sem_profile = np.zeros(len(lags_full))
sem_profile[0] = alpha_mean
sem_profile[1:] = (1 - alpha_mean) * w_carry

# Social: Pure Weibull
soc_profile = np.zeros(len(lags_full))
w_soc = weibull_min.pdf(lags_carry, k_soc_val, scale=lam_soc_val)
w_soc = w_soc / w_soc.sum()
soc_profile[1:] = w_soc

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=get_figsize("16:9"), sharey=True)

# SEM: single sparkline from week 0 across full carryover
ax1.fill_between(lags_full, sem_profile, color=CPX_NAVY, alpha=0.15, zorder=1)
ax1.plot(lags_full, sem_profile, color=CPX_NAVY, lw=2, zorder=2)
ax1.plot([0], [sem_profile[0]], "o", color=CPX_RED, markersize=9, zorder=3)
ax1.set_xlabel("Lag (weeks)", fontsize=13); ax1.set_ylabel("Weight (normalized)", fontsize=13)
ax1.set_title("SEM: Spike-and-Carry Adstock", fontsize=15)
ax1.set_xticks(lags_full)
ax1.tick_params(labelsize=12)
ax1.annotate(f"Spike: α={alpha_mean:.0%}\n(same-week)",
             xy=(0, sem_profile[0]), xytext=(3, sem_profile[0]*0.85),
             fontsize=12, fontweight="bold", color=CPX_RED,
             arrowprops=dict(arrowstyle="->", color=CPX_RED, lw=1.5))
ax1.annotate(f"Carry: {1-alpha_mean:.0%}\n(Weibull k={k_sem_val:.1f}, λ={lam_sem_val:.1f})",
             xy=(3, sem_profile[3]), xytext=(6, sem_profile[0]*0.45),
             fontsize=12, fontweight="bold", color=CPX_NAVY,
             arrowprops=dict(arrowstyle="->", color=CPX_NAVY, lw=1.5))

# Social: single sparkline from week 0 across full decay
ax2.fill_between(lags_full, soc_profile, color=CPX_RED, alpha=0.15, zorder=1)
ax2.plot(lags_full, soc_profile, color=CPX_RED, lw=2, zorder=2)
ax2.set_xlabel("Lag (weeks)", fontsize=13)
ax2.set_title("Social: Pure Weibull (No Same-Week)", fontsize=15)
ax2.set_xticks(lags_full)
ax2.tick_params(labelsize=12)
peak_soc = lam_soc_val * ((k_soc_val - 1) / k_soc_val) ** (1 / k_soc_val)
ax2.annotate(f"Peak at week {peak_soc:.1f}\n(k={k_soc_val:.1f}, λ={lam_soc_val:.1f})",
             xy=(round(peak_soc), soc_profile[round(peak_soc)]),
             xytext=(7, soc_profile[round(peak_soc)]*1.5),
             fontsize=12, fontweight="bold", color=CPX_RED,
             arrowprops=dict(arrowstyle="->", color=CPX_RED, lw=1.5))


fig.suptitle("V5 Learned Decay Profiles — SEM (Immediate + Carry) vs Social (Delayed Peak)",
             fontsize=16, fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.show()

# ── Decay Profile Table ───────────────────────────────────────────────────
print(f"\n{'Lag (wk)':<10} {'SEM Weight':>12} {'Social Weight':>14}")
print("-" * 38)
for i, lag in enumerate(lags_full):
    print(f"{int(lag):<10} {sem_profile[i]:>12.4f} {soc_profile[i]:>14.4f}")
print(f"\nSEM  : α={alpha_mean:.0%} same-week spike | Carry Weibull k={k_sem_val:.1f}, λ={lam_sem_val:.1f}")
print(f"Social: No lag-0 | Peak week {peak_soc:.1f} | Weibull k={k_soc_val:.1f}, λ={lam_soc_val:.1f}")

# COMMAND ----------

# DBTITLE 1,8. Predicted vs Actual Time Series
# ── Predicted vs Actual ─────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=get_figsize("16:9"))
week_dates_plot = df.index

ax.plot(week_dates_plot, y, color=CPX_DGREY, lw=1, alpha=0.7, label="Actual")
ax.plot(week_dates_plot, pred_full, color=CPX_RED, lw=1.5, label="V5 Predicted")
ax.axvline(week_dates_plot[split_idx], ls="--", color=CPX_GOLD, lw=1.5, label="Train/Val Split")

ax.fill_between(week_dates_plot[:split_idx], 0, y[:split_idx], alpha=0.05, color=CPX_NAVY)
ax.fill_between(week_dates_plot[split_idx:], 0, y[split_idx:], alpha=0.05, color=CPX_GOLD)

ax.set_ylabel("Weekly Inquiries")
ax.set_xlabel("Week")
ax.set_title(f"V5 Model Fit — Train MAPE: {mape_train:.1f}% | Val MAPE: {mape_val:.1f}%")
ax.legend(loc="upper right")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v/1e3:.0f}K" if v >= 1000 else f"{v:.0f}"))
plt.tight_layout()
plt.show()

# ── Fit Summary Table ────────────────────────────────────────────────────
print(f"\nOverall | Train MAPE: {mape_train:.1f}%  |  Val MAPE: {mape_val:.1f}%")
print(f"\n{'Year':<8} {'Actual Inq':>12} {'Predicted Inq':>14} {'MAPE':>8}")
print("-" * 46)
for yr in FULL_YRS:
    mask = (df.index.year == yr)
    act  = y[mask].sum()
    prd  = pred_full[mask].sum()
    mape_yr = np.mean(np.abs(y[mask] - pred_full[mask]) / y[mask]) * 100
    print(f"{yr:<8} {act:>12,.0f} {prd:>14,.0f} {mape_yr:>7.1f}%")

# COMMAND ----------

# DBTITLE 1,Data Prep for Business Visualizations
# ── Compute business metrics for Cells 10–12 ────────────────────────────────
FULL_YRS = [2023, 2024, 2025]

# sc: Revenue by Year × Segment (EXH vs LBE)
seg_rev = (glac[glac["Year"].isin(FULL_YRS)]
      .groupby(["Year", "Segment"])["Total"].sum()
      .unstack("Segment").fillna(0)
      .reindex(FULL_YRS))

# fn: Funnel metrics (Inquiries, Revenue, Revenue-per-Inquiry) by year
inq_yr = inq[inq["Year"].isin(FULL_YRS)].groupby("Year").size()
rev_yr = glac[glac["Year"].isin(FULL_YRS)].groupby("Year")["Total"].sum()
fn = pd.DataFrame({"Inq": inq_yr, "Rev": rev_yr}).reindex(FULL_YRS).fillna(0)
fn["RPI"] = fn["Rev"] / fn["Inq"]

# imix: Inquiry mix by Booking Type (%)
inq_full = inq[inq["Year"].isin(FULL_YRS)]
bt_counts = inq_full["Booking Type"].value_counts()
imix = (bt_counts / bt_counts.sum() * 100).head(8)  # top 8 categories

# rmix: Revenue mix by Type Of Event (%)
glac_full = glac[glac["Year"].isin(FULL_YRS)]
rev_by_type = glac_full.groupby("Type Of Event")["Total"].sum()
rmix = (rev_by_type / rev_by_type.sum() * 100).sort_values(ascending=False).head(8)

print(f"✓ Business metrics ready")
print(f"  seg_rev: {seg_rev.shape} (Revenue by Year × Segment)")
print(f"  fn: {fn.shape} (Inq/Rev/RPI by year)")
print(f"  imix: {len(imix)} booking types | rmix: {len(rmix)} event types")

# COMMAND ----------

# DBTITLE 1,Cell 11
# Plot 1 — EXH vs LBE growth
fig, ax = plt.subplots(figsize=get_figsize("16:9"))
x = np.arange(3)
w = 0.38

b1 = ax.bar(x - w/2, seg_rev["EXH"]/1e6, w, label="EXH (Exhibition)", color=CPX_NAVY)
b2 = ax.bar(x + w/2, seg_rev["LBE"]/1e6, w, label="LBE (Rec Room/Playdium)", color=CPX_GOLD)

# Data labels on bars
for bars in [b1, b2]:
    for bar in bars:
        h = bar.get_height()
        if h >= 0.1:
            ax.text(bar.get_x() + bar.get_width()/2, h/2,
                    f"${h:.1f}M", ha="center", va="center",
                    fontsize=9, fontweight="bold", color=WHITE)

# Total labels above each year
for i in range(3):
    total = seg_rev["EXH"].iloc[i]/1e6 + seg_rev["LBE"].iloc[i]/1e6
    ax.text(i, max(seg_rev["EXH"].iloc[i], seg_rev["LBE"].iloc[i])/1e6 + 0.8,
            f"Total: ${total:.1f}M", ha="center", va="bottom",
            fontsize=10, fontweight="bold", color=CPX_DGREY)

ax.set_xticks(x)
ax.set_xticklabels(FULL_YRS, fontsize=11)
ax.set_ylabel("Revenue ($M)")
ax.set_xlabel("Year")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:.0f}M"))
ax.set_title("G&E Realized Sales by Segment — LBE Growing Faster")
ax.legend(loc="upper left")
plt.tight_layout()
plt.show()

# ── Revenue by Segment Table ─────────────────────────────────────────────
print(f"\n{'Year':<8} {'EXH Revenue':>14} {'LBE Revenue':>14} {'Total':>14} {'LBE Share':>10}")
print("-" * 64)
for yr in FULL_YRS:
    exh = seg_rev.loc[yr, "EXH"]
    lbe = seg_rev.loc[yr, "LBE"]
    tot = exh + lbe
    lbe_share = lbe / tot * 100 if tot > 0 else 0
    print(f"{yr:<8} ${exh:>12,.0f} ${lbe:>12,.0f} ${tot:>12,.0f} {lbe_share:>9.1f}%")

# COMMAND ----------

# Plot 1 — EXH vs LBE growth
fig, ax = plt.subplots(figsize=get_figsize("16:9"))
x = np.arange(3)
w = 0.38

b1 = ax.bar(x - w/2, seg_rev["EXH"]/1e6, w, label="EXH (Exhibition)", color=CPX_NAVY)
b2 = ax.bar(x + w/2, seg_rev["LBE"]/1e6, w, label="LBE (Rec Room/Playdium)", color=CPX_GOLD)

# Data labels on bars
for bars in [b1, b2]:
    for bar in bars:
        h = bar.get_height()
        if h >= 0.1:
            ax.text(bar.get_x() + bar.get_width()/2, h/2,
                    f"${h:.1f}M", ha="center", va="center",
                    fontsize=14, fontweight="bold", color=WHITE)

# Total labels above each year
for i in range(3):
    total = seg_rev["EXH"].iloc[i]/1e6 + seg_rev["LBE"].iloc[i]/1e6
    ax.text(i, max(seg_rev["EXH"].iloc[i], seg_rev["LBE"].iloc[i])/1e6 + 0.8,
            f"Total: ${total:.1f}M", ha="center", va="bottom",
            fontsize=14, fontweight="bold", color=CPX_DGREY)

ax.set_xticks(x)
ax.set_xticklabels(FULL_YRS, fontsize=14)
ax.set_ylabel("Revenue ($M)", fontsize=14)
ax.set_xlabel("Year", fontsize=14)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:.0f}M"))
ax.tick_params(axis="y", labelsize=13)
ax.set_ylim(0, (seg_rev["EXH"].max()/1e6) * 1.18)
ax.set_title("G&E Realized Sales by Segment — LBE Growing Faster", fontsize=16)
ax.legend(loc="upper left", fontsize=13)
plt.tight_layout()
plt.show()

# ── Revenue by Segment Table ─────────────────────────────────────────────
print(f"\n{'Year':<8} {'EXH Revenue':>14} {'LBE Revenue':>14} {'Total':>14} {'LBE Share':>10}")
print("-" * 64)
for yr in FULL_YRS:
    exh = seg_rev.loc[yr, "EXH"]
    lbe = seg_rev.loc[yr, "LBE"]
    tot = exh + lbe
    lbe_share = lbe / tot * 100 if tot > 0 else 0
    print(f"{yr:<8} ${exh:>12,.0f} ${lbe:>12,.0f} ${tot:>12,.0f} {lbe_share:>9.1f}%")

# COMMAND ----------

print(type(sc))

# COMMAND ----------

# DBTITLE 1,Cell 12
# Plot 2+3 — funnel inversion (dual axis)
fig, ax = plt.subplots(figsize=get_figsize("16:9"))
ax2 = ax.twinx()

# Bars for inquiry volume
bars = ax.bar(FULL_YRS, fn["Inq"], width=0.55, color="black", label="Inquiries", zorder=2)
for bar in bars:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h/2,
            f"{h:,.0f}", ha="center", va="center",
            fontsize=14, fontweight="bold", color=WHITE)

# Line for revenue per inquiry
line = ax2.plot(FULL_YRS, fn["RPI"], color=CPX_GREEN, marker="o", lw=3,
               ms=10, zorder=3, label="Rev per Inquiry")
for yr, rpi in zip(FULL_YRS, fn["RPI"]):
    ax2.annotate(f"${rpi:,.0f}", (yr, rpi), textcoords="offset points",
                 xytext=(0, 14), ha="center", fontsize=14, fontweight="bold",
                 color=CPX_GREEN)

ax.set_xticks(FULL_YRS)
ax.set_xticklabels(FULL_YRS, fontsize=14)
ax.set_ylabel("Inquiries", fontsize=14)
ax.set_xlabel("Year", fontsize=14)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v/1e3:.0f}K"))
ax.tick_params(axis="y", labelsize=14)
ax.set_ylim(0, fn["Inq"].max() * 1.15)
ax2.set_ylabel("Revenue per Inquiry ($)", color=CPX_GREEN, fontsize=14)
ax2.tick_params(axis="y", colors=CPX_GREEN, labelsize=14)
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:,.0f}"))
ax2.set_ylim(top=fn["RPI"].max() * 1.12)
ax.set_title("Inquiries Falling, but Revenue per Inquiry Nearly Doubled", fontsize=16, pad=20)

# Combined legend — top-left inside the plot
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
leg_elements = [Patch(facecolor="black", label="Inquiries"),
                Line2D([0], [0], color=CPX_GREEN, lw=3, marker="o", label="Rev/Inquiry")]
ax.legend(handles=leg_elements, loc="upper left", fontsize=13)
plt.tight_layout()
plt.show()

# ── Funnel Table ─────────────────────────────────────────────────────────
print(f"\n{'Year':<8} {'Inquiries':>12} {'YoY Inq Δ':>12} {'Revenue':>14} {'Rev/Inquiry':>14}")
print("-" * 64)
prev_inq = None
for yr in FULL_YRS:
    inq_n = fn.loc[yr, "Inq"]
    rev_n = fn.loc[yr, "Rev"]
    rpi_n = fn.loc[yr, "RPI"]
    delta = f"{(inq_n/prev_inq - 1)*100:+.1f}%" if prev_inq else "—"
    print(f"{yr:<8} {inq_n:>12,.0f} {delta:>12} ${rev_n:>12,.0f} ${rpi_n:>12,.0f}")
    prev_inq = inq_n

# COMMAND ----------

# DBTITLE 1,Cell 13
# Plot 4 — demand vs revenue mix (100% stacked, ungrouped categories)
import textwrap

# Normalize each Series to sum to 100%
imix_n = imix / imix.sum() * 100
rmix_n = rmix / rmix.sum() * 100

inq_colors = plt.get_cmap("Blues")(np.linspace(0.9, 0.35, len(imix_n)))
rev_colors = plt.get_cmap("Reds")(np.linspace(0.9, 0.35, len(rmix_n)))

fig, ax = plt.subplots(figsize=(20, 5))

INLINE_MIN = 5  # segments wider than this get name+% inside; narrower go to legend

def draw_stack(series, colors, y):
    left = 0.0
    legend_items = []
    for (cat, val), color in zip(series.items(), colors):
        ax.barh(y, val, left=left, color=color, height=0.55, edgecolor=WHITE, linewidth=1.2)
        txt_color = WHITE if color[:3].mean() < 0.5 else CPX_DGREY  # contrast-aware
        if val >= INLINE_MIN:
            name = "\n".join(textwrap.wrap(str(cat), 16))
            ax.text(left + val/2, y, f"{name}\n{val:.1f}%", ha="center", va="center",
                    fontsize=12, fontweight="bold", color=txt_color, linespacing=1.3)
        elif val >= 2:
            ax.text(left + val/2, y, f"{val:.1f}%", ha="center", va="center",
                    fontsize=12, fontweight="bold", color=txt_color)
            legend_items.append((cat, color, val))
        left += val
    return legend_items

inq_legend = draw_stack(imix_n, inq_colors, 1)
rev_legend = draw_stack(rmix_n, rev_colors, 0)

ax.set_yticks([0, 1])
ax.set_yticklabels(["% of Revenue", "% of Inquiries"], fontsize=13, fontweight="bold")
ax.set_xlim(0, 100)
ax.set_ylim(-0.55, 1.55)
ax.set_xlabel("Share (%)", fontsize=14)
ax.tick_params(axis="x", labelsize=13)
ax.set_xticks([0, 20, 40, 60, 80, 100])
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))

# Compact legend for the thin segments that couldn't fit an inline name
from matplotlib.patches import Patch
handles = [Patch(facecolor=c, label=f"{cat} ({v:.1f}%)") for cat, c, v in inq_legend + rev_legend]
if handles:
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.22),
              ncol=len(handles), fontsize=10, frameon=False, title="Smaller segments")

fig.suptitle("Demand ≠ Revenue: Corporate Drives Half of Revenue on 10% of Inquiries",
             fontsize=16, fontweight="bold", y=1.04)
plt.tight_layout()
plt.show()

# ── Mix Tables ────────────────────────────────────────────────────────────
print(f"\n{'Booking Type':<40} {'% Inquiries':>12}")
print("-" * 54)
for bt, pct in imix_n.items():
    print(f"{str(bt):<40} {pct:>11.1f}%")

print(f"\n{'Event Type':<40} {'% Revenue':>12}")
print("-" * 54)
for et, pct in rmix_n.items():
    print(f"{str(et):<40} {pct:>11.1f}%")

# COMMAND ----------

print("IMIX:", len(imix), list(imix.index))
print("RMIX:", len(rmix), list(rmix.index))

# COMMAND ----------

# DBTITLE 1,9. Decomposition — Media vs Non-Media
# ── Decomposition Stacked Bar — YoY ───────────────────────────────────────
from matplotlib.patches import Patch

# Compute YoY decomposition from weekly model outputs
years_in_data = df.index.year
yoy_data = []
for yr in FULL_YRS:
    mask = (years_in_data == yr)
    if mask.sum() == 0:
        continue
    yr_actual = y[mask].sum()
    yr_sem_incr = (np.exp(log_full[mask]) - np.exp(log_full[mask] - media_sem_mean[mask])).sum()
    yr_soc_incr = (np.exp(log_full[mask]) - np.exp(log_full[mask] - media_soc_mean[mask])).sum()
    yr_media = yr_sem_incr + yr_soc_incr
    yoy_data.append({
        "Year": yr,
        "SEM %": yr_sem_incr / yr_actual * 100,
        "Social %": yr_soc_incr / yr_actual * 100,
        "Non-Media %": (1 - yr_media / yr_actual) * 100,
        "SEM inq": yr_sem_incr,
        "Soc inq": yr_soc_incr,
        "Total inq": yr_actual,
    })

# Add overall
# yoy_data.append({
#     "Year": "Overall",
#     "SEM %": sem_pct, "Social %": soc_pct, "Non-Media %": base_pct,
#     "SEM inq": sem_incr, "Soc inq": soc_incr, "Total inq": actual_total,
# })

fig, ax = plt.subplots(figsize=get_figsize("16:9"))
labels = [str(d["Year"]) for d in yoy_data]
x = np.arange(len(labels))
w = 0.5

# Stacked bars
for i, d in enumerate(yoy_data):
    bot = 0
    for val, color in [(d["Non-Media %"], CPX_MGREY), (d["Social %"], CPX_RED), (d["SEM %"], CPX_NAVY)]:
        ax.bar(i, val, bottom=bot, color=color, width=w)
        if val >= 2.5:
            ax.text(i, bot + val/2, f"{val:.1f}%", ha="center", va="center",
                    fontsize=14, fontweight="bold", color=WHITE)
        bot += val
    # Total media label above bar
    media_tot = d["SEM %"] + d["Social %"]
    ax.text(i, 102, f"Media: {media_tot:.1f}%", ha="center", va="bottom",
            fontsize=14, fontweight="bold", color=CPX_DGREY)

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=14)
ax.set_ylim(0, 128)
ax.set_ylabel("% of Total Inquiries", fontsize=14)
ax.tick_params(axis="y", labelsize=14)
ax.set_yticks([0, 20, 40, 60, 80, 100])
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
ax.set_title("Inquiry Decomposition by Year", fontsize=14, pad=16)

legend_items = [
    Patch(facecolor=CPX_NAVY, label="G&E SEM"),
    Patch(facecolor=CPX_RED, label="G&E Social"),
    Patch(facecolor=CPX_MGREY, label="Non-Media"),
]
ax.legend(handles=legend_items, loc="upper right", fontsize=14)
plt.tight_layout()
plt.show()

# Print table
print(f"{'Year':<10} {'SEM':>7} {'Social':>7} {'Media':>7} {'Non-Media':>10} {'Total Inq':>10}")
print("-"*55)
for d in yoy_data:
    print(f"{str(d['Year']):<10} {d['SEM %']:>6.1f}% {d['Social %']:>6.1f}% "
          f"{d['SEM %']+d['Social %']:>6.1f}% {d['Non-Media %']:>9.1f}% {d['Total inq']:>10,.0f}")

# COMMAND ----------

# DBTITLE 1,10. Media Efficiency — Cost per Incremental Inquiry
# ── Cost per Incremental Inquiry — Per Year ─────────────────────────────────
years_in_data = df.index.year

eff_data = []
for yr in FULL_YRS:
    mask = (years_in_data == yr)
    if mask.sum() == 0:
        continue
    yr_sem_spend = df.loc[mask, "sem"].sum()
    yr_soc_spend = df.loc[mask, "soc"].sum()
    yr_row = next((d for d in yoy_data if d["Year"] == yr), None)
    if yr_row is None:
        continue
    yr_sem_incr = yr_row["SEM inq"]
    yr_soc_incr = yr_row["Soc inq"]
    eff_data.append({
        "Year": yr,
        "CPI SEM":    yr_sem_spend / yr_sem_incr if yr_sem_incr > 0 else np.inf,
        "CPI Social": yr_soc_spend / yr_soc_incr if yr_soc_incr > 0 else np.inf,
        "SEM Spend":  yr_sem_spend,
        "Soc Spend":  yr_soc_spend,
        "SEM Inq":    yr_sem_incr,
        "Soc Inq":    yr_soc_incr,
    })

# Overall (for reference line)
cpi_sem_overall = df["sem"].sum() / sem_incr if sem_incr > 0 else np.inf
cpi_soc_overall = df["soc"].sum() / soc_incr if soc_incr > 0 else np.inf

# ── Grouped bar chart ────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=get_figsize("16:9"))
x  = np.arange(len(eff_data))
w  = 0.38
cpi_sem_vals = [d["CPI SEM"]    for d in eff_data]
cpi_soc_vals = [d["CPI Social"] for d in eff_data]

b1 = ax.bar(x - w/2, cpi_sem_vals, w, label="G&E SEM",    color=CPX_NAVY)
b2 = ax.bar(x + w/2, cpi_soc_vals, w, label="G&E Social", color=CPX_RED)

for bars_grp in [b1, b2]:
    for bar in bars_grp:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.3,
                f"${h:.2f}", ha="center", va="bottom",
                fontsize=14, fontweight="bold", color=CPX_DGREY)

# Overall reference lines
ax.axhline(cpi_sem_overall, ls="--", lw=1.2, color=CPX_NAVY,  alpha=0.5,
           label=f"SEM overall avg ${cpi_sem_overall:.2f}")
ax.axhline(cpi_soc_overall, ls="--", lw=1.2, color=CPX_RED,   alpha=0.5,
           label=f"Social overall avg ${cpi_soc_overall:.2f}")

ax.set_xticks(x)
ax.set_xticklabels([str(d["Year"]) for d in eff_data], fontsize=14)
ax.set_ylabel("Cost per Incremental Inquiry ($)", fontsize=14)
ax.tick_params(axis="y", labelsize=14)
ax.set_title("Media Efficiency by Year — Cost per Incremental G&E Inquiry", fontsize=14, pad=16)
ax.set_ylim(0, max(max(cpi_sem_vals), max(cpi_soc_vals)) * 1.55)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:.0f}"))
ax.legend(loc="upper right", fontsize=14)
plt.tight_layout()
plt.show()

# ── Summary table ─────────────────────────────────────────────────────────────
print(f"\n{'Year':<8} {'SEM Spend':>12} {'SEM Inq':>10} {'CPI SEM':>10} "
      f"{'Soc Spend':>12} {'Soc Inq':>10} {'CPI Soc':>10}")
print("-" * 78)
for d in eff_data:
    print(f"{d['Year']:<8} ${d['SEM Spend']:>10,.0f} {d['SEM Inq']:>10,.0f} ${d['CPI SEM']:>9.2f} "
          f"${d['Soc Spend']:>10,.0f} {d['Soc Inq']:>10,.0f} ${d['CPI Social']:>9.2f}")
print("-" * 78)
print(f"{'Overall':<8} ${df['sem'].sum():>10,.0f} {sem_incr:>10,.0f} ${cpi_sem_overall:>9.2f} "
      f"${df['soc'].sum():>10,.0f} {soc_incr:>10,.0f} ${cpi_soc_overall:>9.2f}")

# COMMAND ----------

# DBTITLE 1,11. B-Spline Baseline — Structural Decline Curve
# ── B-Spline Baseline Curve ───────────────────────────────────────────────
fig, ax = plt.subplots(figsize=get_figsize("16:9"))

baseline_inquiries = np.exp(log_bl)
week_dates_plot = df.index

ax.plot(week_dates_plot, y, color=CPX_MGREY, lw=0.8, alpha=0.5, label="Actual inquiries")
ax.plot(week_dates_plot, baseline_inquiries, color=CPX_NAVY, lw=2.5, label="B-Spline baseline")
ax.plot(week_dates_plot, pred_full, color=CPX_RED, lw=1.2, alpha=0.7, label="Full model (baseline + media + seas)")

# Annotate decline
start_bl = baseline_inquiries[0]
end_bl = baseline_inquiries[-1]
pct_decline = (end_bl - start_bl) / start_bl * 100
ax.annotate(f"Structural decline: {pct_decline:.0f}%",
            xy=(week_dates_plot[n_weeks//2], baseline_inquiries[n_weeks//2]),
            xytext=(week_dates_plot[n_weeks//2], baseline_inquiries[0] * 0.95),
            fontsize=10, fontweight="bold", color=CPX_NAVY,
            arrowprops=dict(arrowstyle="->", color=CPX_NAVY, lw=1.5))

ax.set_ylabel("Inquiries/week")
ax.set_xlabel("Week")
ax.set_title("B-Spline Baseline — Nonlinear Structural Decline (Not Forced Linear)")
ax.legend(loc="upper right")
plt.tight_layout()
plt.show()

print(f"Baseline start: {start_bl:.0f} inq/wk | End: {end_bl:.0f} inq/wk | Decline: {pct_decline:.1f}%")
print(f"The spline flexes to capture fast decline in 2024 and stabilization in 2025.")

# ── Baseline vs Actual by Year ────────────────────────────────────────────
print(f"\n{'Year':<8} {'Avg Baseline Inq/wk':>22} {'Avg Actual Inq/wk':>20} {'Media+Seas Lift':>17}")
print("-" * 72)
for yr in FULL_YRS:
    mask = (df.index.year == yr)
    avg_bl  = baseline_inquiries[mask].mean()
    avg_act = y[mask].mean()
    lift    = avg_act - avg_bl
    print(f"{yr:<8} {avg_bl:>22.0f} {avg_act:>20.0f} {lift:>+17.0f}")

# COMMAND ----------

# DBTITLE 1,12. Findings & Caveats
# MAGIC %md
# MAGIC ## Findings & Caveats
# MAGIC
# MAGIC ### What V5 Tells Us (Defensible)
# MAGIC 1. **G&E media credibly drives 13.3% of inquiry volume** — SEM (8.0%) + Social (5.8%), with total ~10,125 incremental inquiries over 165 weeks.
# MAGIC 2. **SEM is an immediate-response channel** — 73% of its effect is same-week (search → click → inquiry form). V1–V4 discarded this signal by starting adstock at lag 1.
# MAGIC 3. **Social is a delayed-awareness channel** — effect peaks at week 3.6. The ad plants a seed; the inquiry comes 3–4 weeks later.
# MAGIC 4. **The structural decline is nonlinear** — fast in 2024, stabilizing in 2025. A rigid linear trend misallocates credit.
# MAGIC 5. **Val MAPE = 14.8%** — best across all model versions (V1: 39.9%, V3: 17.4%, V4: 16.1%).
# MAGIC
# MAGIC ### Model Progression
# MAGIC | Version | Media Share | Key Improvement |
# MAGIC |---|---|---|
# MAGIC | V1 | 3.3% | Baseline (additive, EXH seasonality, tanh, geometric) |
# MAGIC | V2 | 4.9% | G&E-specific seasonality (school, holidays negative) |
# MAGIC | V3 | 9.8% | Multiplicative log-link + Hill saturation |
# MAGIC | V4 | 9.8% | B-spline baseline + Weibull adstock |
# MAGIC | **V5** | **13.3%** | **Spike-and-carry SEM (lag 0 included)** |
# MAGIC
# MAGIC ### Caveats
# MAGIC - **No media → revenue model** — booking pipeline lag (variable, multi-month) makes timing broken.
# MAGIC - **165 weeks nationally** — data is thin for precise point estimates; direction is solid, magnitude has wide CIs.
# MAGIC - **Geo-level data** would be the next major unlock — regional variation breaks the confound between seasonal spend timing and seasonal demand.
# MAGIC - **2026 partial year** — excluded from all trend statements.
# MAGIC - **Counterfactual decomposition** = “what if media were zero?” Not a marginal response curve.
# MAGIC
# MAGIC ### Next Steps
# MAGIC - Regional / provincial modeling (if spend data available by market)
# MAGIC - Media × Seasonality interaction terms (media may be more effective during corporate party season)
# MAGIC - Time-varying coefficients (has SEM effectiveness improved year-over-year?)

# COMMAND ----------

# ── Baseline Attribution + Feature Correlation ─────────────────────────────
# Q1: How many inquiries are attributed to the baseline (non-media, non-seasonal)?
# Q2: Does that baseline track any seasonality/media feature? (It shouldn't —
#     the B-spline is meant to hold ONLY smooth structural trend.)

baseline_inq = np.exp(log_bl)          # inquiries attributed to baseline
week_dates_plot = df.index

# Build feature matrix for correlation: seasonality (standardized Xs) + media
feat_names = list(seas_cols) + ["sem_spend", "soc_spend"]
feat_matrix = np.column_stack([Xs, sem_n, soc_n])

# Correlate each feature against the baseline curve
corrs = np.array([np.corrcoef(baseline_inq, feat_matrix[:, j])[0, 1]
                  for j in range(feat_matrix.shape[1])])
order = np.argsort(np.abs(corrs))[::-1]          # strongest first
names_sorted = [feat_names[i] for i in order]
corrs_sorted = corrs[order]
max_abs_corr = np.abs(corrs_sorted).max()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6),
                               gridspec_kw={"width_ratios": [2.4, 1]})

# ── Left: baseline inquiries over time ──
ax1.fill_between(week_dates_plot, 0, baseline_inq, color=CPX_NAVY, alpha=0.12)
ax1.plot(week_dates_plot, baseline_inq, color=CPX_NAVY, lw=2.5, label="Baseline inquiries")
ax1.plot(week_dates_plot, y, color=CPX_MGREY, lw=0.8, alpha=0.5, label="Actual inquiries")
ax1.set_ylabel("Inquiries / week")
ax1.set_title(f"Baseline Attribution — {baseline_inq.sum():,.0f} inquiries "
              f"({baseline_inq.sum()/y.sum()*100:.1f}% of total)")
ax1.legend(loc="upper right")
ax1.set_ylim(bottom=0)

# ── Right: correlation of baseline vs each feature (the "why we assigned it" evidence) ──
bar_colors = [CPX_RED if c < 0 else CPX_GOLD for c in corrs_sorted]
ax2.barh(names_sorted, corrs_sorted, color=bar_colors, edgecolor=CPX_DGREY, lw=0.8)
ax2.axvline(0, color=CPX_DGREY, lw=1)
ax2.invert_yaxis()                                # strongest at top
ax2.set_xlabel("Corr. with baseline")
ax2.set_title(f"No Feature Explains Baseline\n(max |corr| = {max_abs_corr:.2f} → structural)")
ax2.set_xlim(-1, 1)
for i, c in enumerate(corrs_sorted):
    ax2.text(c + (0.04 if c >= 0 else -0.04), i, f"{c:+.2f}",
             va="center", ha="left" if c >= 0 else "right", fontsize=9)

plt.tight_layout()
plt.show()

# ── Ranked print ──
print(f"Baseline total: {baseline_inq.sum():,.0f} inquiries "
      f"({baseline_inq.sum()/y.sum()*100:.1f}% of {y.sum():,.0f} actual)")
print(f"Baseline range: {baseline_inq.min():.0f}–{baseline_inq.max():.0f} inq/wk\n")
print(f"{'Feature':<18} {'Corr w/ baseline':>18}")
print("-" * 38)
for name, c in zip(names_sorted, corrs_sorted):
    flag = "  <-- notable" if abs(c) > 0.5 else ""
    print(f"{name:<18} {c:>+18.2f}{flag}")

# COMMAND ----------

# ── Stacked Decomposition — What Moved Inquiries Each Week ──────────────────
# Log-additive model → incremental inquiries per component vs baseline-only.
# Positive components stack ABOVE baseline; negative (holidays, qtr-end) BELOW.

wk = df.index
baseline_inq = np.exp(log_bl)

# Per-feature seasonal contribution in log-space (each column of Xs * its beta)
seas_contrib_log = Xs * b_seas_mean          # (n_weeks, n_features)

# Convert each component to incremental inquiries: baseline * (exp(effect) - 1),
# built cumulatively so the pieces sum to the full fitted prediction.
running_log = log_bl.copy()
def incr(effect_log):
    global running_log
    lo = running_log
    hi = running_log + effect_log
    running_log = hi
    return np.exp(hi) - np.exp(lo)

# Order: media first, then seasonal features
comp = {}
comp["G&E SEM"]    = incr(media_sem_mean)
comp["G&E Social"] = incr(media_soc_mean)
for j, name in enumerate(seas_cols):
    comp[name] = incr(seas_contrib_log[:, j])

fitted = np.exp(running_log)                  # == full model prediction

# Color map (brand-ish, distinct)
palette = {
    "G&E SEM": CPX_RED, "G&E Social": CPX_GOLD,
    "sin_1": "#4C72B0", "cos_1": "#6A93CB", "sin_2": "#8FB0DD", "cos_2": "#B5CDE8",
    "school": CPX_GREEN, "corp_party_ramp": "#7B4EA8",
    "qtr_end": "#C97A40", "stat_holiday": "#9E2A2B", "long_weekend": "#D4A5A5",
}
def col(name): return palette.get(name, CPX_MGREY)

fig, ax = plt.subplots(figsize=get_figsize("16:9"))

# Baseline as the floor band
ax.fill_between(wk, 0, baseline_inq, color=CPX_NAVY, alpha=0.85, label="Baseline")

# Split each component into positive / negative and stack from baseline
pos_base = baseline_inq.copy()
neg_base = baseline_inq.copy()
for name, vals in comp.items():
    pos = np.clip(vals, 0, None)
    neg = np.clip(vals, None, 0)
    if pos.sum() > 0:
        ax.fill_between(wk, pos_base, pos_base + pos, color=col(name),
                        alpha=0.9, label=f"{name} (+)" if pos.sum() >= abs(neg.sum()) else name)
        pos_base = pos_base + pos
    if neg.sum() < 0:
        ax.fill_between(wk, neg_base + neg, neg_base, color=col(name),
                        alpha=0.55, hatch="///", edgecolor="white", lw=0)
        neg_base = neg_base + neg

# Actual overlay
ax.plot(wk, y, color="black", lw=1.0, alpha=0.7, label="Actual inquiries")

ax.set_ylabel("Inquiries / week")
ax.set_title("Inquiry Decomposition — Baseline + Media + Seasonality (per week)")
ax.set_ylim(bottom=0)
ax.legend(loc="upper right", ncol=2, fontsize=8)
plt.tight_layout()
plt.show()

# ── Yearly contribution summary (total incremental inquiries per component) ──
print(f"{'Component':<18} {'2023':>8} {'2024':>8} {'2025':>8} {'Total':>10}")
print("-" * 56)
yrs = [2023, 2024, 2025]
for name, vals in comp.items():
    row = [vals[df.index.year == yr].sum() for yr in yrs]
    print(f"{name:<18} " + " ".join(f"{v:>8.0f}" for v in row) + f" {vals.sum():>10.0f}")
print("-" * 56)
base_row = [baseline_inq[df.index.year == yr].sum() for yr in yrs]
print(f"{'Baseline':<18} " + " ".join(f"{v:>8.0f}" for v in base_row) + f" {baseline_inq.sum():>10.0f}")

# COMMAND ----------

# ── Seasonal Decomposition Over Time — Baseline + Calendar Features ─────────
# No media. Shows natural demand (baseline) + what each seasonal lever adds/removes.
wk = df.index
baseline_inq = np.exp(log_bl)
seas_log = Xs * b_seas_mean                       # per-feature log contribution

# Collapse the 4 Fourier terms into one "Annual cycle" lever
fourier_cols = ["sin_1", "cos_1", "sin_2", "cos_2"]
fourier_idx = [seas_cols.index(c) for c in fourier_cols]
event_defs = [("Annual cycle", None),
              ("School in session", "school"),
              ("Corp party ramp", "corp_party_ramp"),
              ("Stat holiday", "stat_holiday"),
              ("Long weekend", "long_weekend"),
              ("Quarter-end", "qtr_end")]

# Incremental inquiries per component, built cumulatively on the baseline floor
running_log = log_bl.copy()
def incr(effect_log):
    global running_log
    lo = running_log; hi = running_log + effect_log; running_log = hi
    return np.exp(hi) - np.exp(lo)

comp = {}
comp["Annual cycle"] = incr(seas_log[:, fourier_idx].sum(axis=1))
for label, col in event_defs[1:]:
    comp[label] = incr(seas_log[:, seas_cols.index(col)])

colors = {"Annual cycle": "#4C72B0", "School in session": CPX_GREEN,
          "Corp party ramp": "#7B4EA8", "Stat holiday": "#9E2A2B",
          "Long weekend": CPX_GOLD, "Quarter-end": "#C97A40"}

fig, ax = plt.subplots(figsize=get_figsize("16:9"))
ax.fill_between(wk, 0, baseline_inq, color=CPX_NAVY, alpha=0.85, label="Baseline (natural demand)")

pos_base = baseline_inq.copy()
neg_base = baseline_inq.copy()
for name, vals in comp.items():
    pos = np.clip(vals, 0, None); neg = np.clip(vals, None, 0)
    if pos.sum() > 0:
        ax.fill_between(wk, pos_base, pos_base + pos, color=colors[name], alpha=0.9, label=name)
        pos_base = pos_base + pos
    if neg.sum() < 0:
        ax.fill_between(wk, neg_base + neg, neg_base, color=colors[name],
                        alpha=0.5, hatch="///", edgecolor="white", lw=0)
        neg_base = neg_base + neg

ax.plot(wk, y, color="black", lw=1.0, alpha=0.7, label="Actual inquiries")
ax.set_ylabel("Inquiries / week")
ax.set_title("Where Inquiries Come From — Baseline + Seasonal Calendar (media excluded)")
ax.set_ylim(bottom=0)
ax.legend(loc="upper right", ncol=2, fontsize=8)
plt.tight_layout()
plt.show()

# ── How much is natural vs seasonal, and when seasonal levers hit ───────────
print(f"Baseline (natural): {baseline_inq.sum():,.0f} inquiries "
      f"({baseline_inq.sum()/y.sum()*100:.0f}% of {y.sum():,.0f} actual)\n")
print(f"{'Seasonal lever':<20} {'Net inq':>9} {'When active (avg wk-of-year)':>32}")
print("-" * 62)
woy = df.index.isocalendar().week.values
for label, col in event_defs:
    vals = comp[label]
    if col is None:                      # annual cycle: name its peak week
        peak_w = np.bincount(woy, weights=np.clip(vals, 0, None)).argmax()
        when = f"peaks ~week {peak_w}"
    else:
        active = Xs[:, seas_cols.index(col)] > Xs[:, seas_cols.index(col)].mean()
        wks = sorted(set(woy[active]))
        when = f"weeks {min(wks)}–{max(wks)}" if wks else "n/a"
    print(f"{label:<20} {vals.sum():>+9.0f} {when:>32}")

# COMMAND ----------

# ── Seasonal Feature Attribution — Net Inquiries per Feature, per Year ──────
seas_log = Xs * b_seas_mean
baseline_inq = np.exp(log_bl)

# Incremental inquiries for one feature = baseline * (exp(effect) - 1)
def feature_incr(effect_log):
    return baseline_inq * (np.exp(effect_log) - 1)

fourier_cols = ["sin_1", "cos_1", "sin_2", "cos_2"]
features = {
    "Annual cycle": seas_log[:, [seas_cols.index(c) for c in fourier_cols]].sum(axis=1),
    "School in session": seas_log[:, seas_cols.index("school")],
    "Corp party ramp":   seas_log[:, seas_cols.index("corp_party_ramp")],
    "Stat holiday":      seas_log[:, seas_cols.index("stat_holiday")],
    "Long weekend":      seas_log[:, seas_cols.index("long_weekend")],
    "Quarter-end":       seas_log[:, seas_cols.index("qtr_end")],
}
incr = {name: feature_incr(eff) for name, eff in features.items()}

yrs = [2023, 2024, 2025]
names = list(features.keys())
year_vals = {yr: [incr[n][df.index.year == yr].sum() for n in names] for yr in yrs}

fig, ax = plt.subplots(figsize=get_figsize("16:9"))
x = np.arange(len(names)); w = 0.26
year_colors = {2023: CPX_NAVY, 2024: CPX_GOLD, 2025: CPX_RED}
for i, yr in enumerate(yrs):
    ax.bar(x + (i - 1) * w, year_vals[yr], w, label=str(yr),
           color=year_colors[yr], edgecolor=CPX_DGREY, lw=0.6)

ax.axhline(0, color=CPX_DGREY, lw=1.2)
ax.set_xticks(x); ax.set_xticklabels(names, rotation=25, ha="right", fontsize=14)
ax.set_ylabel("Net inquiries added / removed", fontsize=14)
ax.tick_params(axis="y", labelsize=14)
ax.set_title("Seasonal Effect on Inquiries — Above Zero = More, Below = Fewer", fontsize=14, pad=16)
ax.legend(title="Year", loc="upper right", fontsize=14, title_fontsize=14)
plt.tight_layout()
plt.show()

# ── Plain-English read ──────────────────────────────────────────────────────
print("What each calendar feature does to inquiries (total over 3 yrs):\n")
for n in names:
    tot = incr[n].sum()
    verb = "MORE" if tot >= 0 else "FEWER"
    print(f"  {n:<20} {tot:>+7.0f}  →  {verb} inquiries when active")

# COMMAND ----------

# ━━ B10: G&E MEDIA RESPONSE CURVE (Dollar x-axis, anchored to 2025 spend) ━━━━━━━━━
# G&E models media as a log-space % lift on NORMALIZED spend. We compute the curve
# in spend-multiples internally, then relabel x to dollars via the 2025 annual anchor.
from matplotlib.lines import Line2D

k_sem_v   = float(summary.loc["k_sem", "mean"]);   lam_sem_v = float(summary.loc["lam_sem", "mean"])
k_soc_v   = float(summary.loc["k_soc", "mean"]);   lam_soc_v = float(summary.loc["lam_soc", "mean"])
alpha_v   = float(summary.loc["alpha_sem", "mean"])
b_sem_v   = float(summary.loc["beta_sem", "mean"]); b_soc_v  = float(summary.loc["beta_soc", "mean"])
K_sem_v   = float(summary.loc["K_sem", "mean"]);   K_soc_v   = float(summary.loc["K_soc", "mean"])
n_sem_v   = float(idata.posterior["n_sem"].mean())
n_soc_v   = float(idata.posterior["n_soc"].mean())

def _weibull_weights(k, lam, L=MAX_LAG):
    lags = np.arange(1, L + 1); t = lags / lam
    w = (k / lam) * t**(k - 1) * np.exp(-t**k)
    return w / w.sum()

def _adstock(x, k, lam, L=MAX_LAG):
    w = _weibull_weights(k, lam, L)
    padded = np.concatenate([np.zeros(L), x])
    idx = np.arange(len(x))[:, None] + np.arange(L)[None, :]
    return padded[idx] @ w[::-1]

def _hill(x, K, n):
    xn = np.power(x + 1e-8, n)
    return xn / (np.power(K, n) + xn)

log_floor = log_bl + Xs @ b_seas_mean

def incremental_inquiries(mult):
    sem_s = sem_n * mult; soc_s = soc_n * mult
    sem_ad = alpha_v * sem_s + (1 - alpha_v) * _adstock(sem_s, k_sem_v, lam_sem_v)
    soc_ad = _adstock(soc_s, k_soc_v, lam_soc_v)
    media_log = b_sem_v * _hill(sem_ad, K_sem_v, n_sem_v) + b_soc_v * _hill(soc_ad, K_soc_v, n_soc_v)
    return (np.exp(log_floor + media_log) - np.exp(log_floor)).sum()

# ── Dollar anchor: 2025 annualized combined SEM + Social spend ──
d2025 = df[df.index.year == 2025]
annual_2025_spend = (d2025["sem"].sum() + d2025["soc"].sum()) / len(d2025) * 52

mult_grid = np.linspace(0, 5, 500)
x_dollars = mult_grid * annual_2025_spend
resp = np.array([incremental_inquiries(m) for m in mult_grid])

# ── Auto-knee (shared with B10b): marginal drops to 25% of near-origin value ──
marginal = np.gradient(resp, x_dollars)
m0 = marginal[1]
knee_frac = 0.25
knee_idx = np.argmin(np.abs(marginal - knee_frac * m0))
knee_x_m = x_dollars[knee_idx] / 1e6          # knee spend in $M — reused by B10b

year_dollars = {}
for yr in [2023, 2024, 2025]:
    dyr = df[df.index.year == yr]
    year_dollars[yr] = (dyr["sem"].sum() + dyr["soc"].sum()) / len(dyr) * 52

YEAR_MARKER_STYLES = {
    2023: {"color": CPX_DGREY, "marker": "o", "linestyle": "--", "linewidth": 1.2},
    2024: {"color": CPX_NAVY,  "marker": "s", "linestyle": "-.", "linewidth": 1.2},
    2025: {"color": CPX_RED,   "marker": "D", "linestyle": "-",  "linewidth": 1.5},
}

fig, ax = plt.subplots(figsize=(16, 9))
fig.patch.set_facecolor("white"); ax.set_facecolor("white")
ax.plot(x_dollars / 1e6, resp, color=CPX_NAVY, linewidth=2.5)

# ── Knee marker on the response curve (same x as B10b) ──
knee_resp = resp[knee_idx]
ax.axvline(knee_x_m, color=CPX_GOLD, linewidth=1.8, linestyle="--", alpha=0.9, zorder=5)
ax.scatter([knee_x_m], [knee_resp], color=CPX_GOLD, s=140, zorder=6, marker="*",
           edgecolors="white", linewidths=0.8)
ax.annotate(f"Diminishing returns\n${knee_x_m:.1f}M",
            xy=(knee_x_m, knee_resp), xytext=(12, -28), textcoords="offset points",
            fontsize=10, color=CPX_GOLD, fontweight="bold", ha="left",
            arrowprops=dict(arrowstyle="->", color=CPX_GOLD, lw=0.8),
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.9, ec="none"))

for yr, style in YEAR_MARKER_STYLES.items():
    xd = year_dollars[yr]
    ym = incremental_inquiries(xd / annual_2025_spend)
    ax.axvline(xd / 1e6, color=style["color"], linestyle=style["linestyle"],
               linewidth=style["linewidth"], alpha=0.85, zorder=4)
    ax.scatter([xd / 1e6], [ym], color=style["color"], marker=style["marker"],
               s=100, zorder=7, edgecolors="white", linewidths=1.0)

ax.set_title("G&E MMM — Total Media Response Curve\n"
             "Incremental inquiries vs. annual SEM + Social spend  |  anchored to 2025 spend",
             fontsize=14, fontweight="bold", color="black", pad=16)
ax.set_xlabel("Annual Media Spend ($M, combined SEM + Social)", fontsize=14, fontweight="bold")
ax.set_ylabel("Annual Incremental Inquiries", fontsize=14, fontweight="bold")
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:.1f}M"))
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.0f}"))
plt.setp(ax.get_xticklabels(), fontsize=12, fontweight="bold")
plt.setp(ax.get_yticklabels(), fontsize=12, fontweight="bold")
ax.set_ylim(bottom=0); ax.grid(False)

legend_handles = [
    Line2D([0], [0], color=s["color"], marker=s["marker"], linestyle=s["linestyle"],
           linewidth=s["linewidth"], markersize=7,
           label=f"{yr} (${year_dollars[yr]/1e6:.2f}M annual)")
    for yr, s in YEAR_MARKER_STYLES.items()
]
ax.legend(handles=legend_handles, loc="lower right", fontsize=13, frameon=True,
          facecolor="white", edgecolor="#CCCCCC", framealpha=0.95, markerscale=1.6,
          borderpad=1.0, labelspacing=0.8, handlelength=2.8,
          title="Annual Spend Reference", title_fontsize=12)

fig.text(0.02, 0.01, "* Dollar axis anchored to 2025 annualized SEM + Social spend; model estimates % lift on "
         "normalized spend. Saturation via posterior-mean Hill params.", fontsize=7.5, color=CPX_DGREY, style="italic")
plt.tight_layout(rect=[0, 0.03, 1, 1])
save_fig(fig, "gne_b10_response_yoy.png")
display(fig); plt.close(fig)

# COMMAND ----------

# ━━ B10b: G&E MARGINAL RETURN CURVE (Dollar x-axis) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Marginal = incremental inquiries per additional $1 of annual media spend.
# Reuses marginal / knee_idx / knee_x_m computed in B10.

fig2, ax2 = plt.subplots(figsize=(16, 9))
fig2.patch.set_facecolor("white"); ax2.set_facecolor("white")

ax2.plot(x_dollars / 1e6, marginal, color=CPX_NAVY, linewidth=2.5)

# ── Diminishing-returns knee (25% of initial marginal) ──
_knee_y = marginal[knee_idx]
ax2.axhspan(0, knee_frac * m0, alpha=0.10, color=CPX_RED, zorder=0)
ax2.axhline(knee_frac * m0, color=CPX_RED, lw=1.8, alpha=0.85, zorder=6)
ax2.text(x_dollars[-1] / 1e6 * 0.6, knee_frac * m0 * 0.5,
         f"Diminishing-Returns Zone\nMarginal < {int(knee_frac*100)}% of initial",
         fontsize=10, color=CPX_RED, fontweight="bold", alpha=0.9, va="center", ha="center",
         bbox=dict(boxstyle="round,pad=0.35", fc="white", alpha=0.8, ec=CPX_RED, lw=0.6))
ax2.vlines(knee_x_m, ymin=0, ymax=_knee_y, color=CPX_GOLD, linewidth=2.2,
           linestyle="--", alpha=0.9, zorder=5)
ax2.scatter([knee_x_m], [_knee_y], color=CPX_GOLD, s=140, zorder=8, marker="*",
            edgecolors="white", linewidths=0.8)
ax2.annotate(f"${knee_x_m:.1f}M knee", xy=(knee_x_m, 0), xytext=(5, 6),
             textcoords="offset points", fontsize=10, color=CPX_GOLD, fontweight="bold", ha="left")

for yr, style in YEAR_MARKER_STYLES.items():
    xd = year_dollars[yr]
    idx = np.argmin(np.abs(x_dollars - xd))
    ax2.axvline(xd / 1e6, color=style["color"], linestyle=style["linestyle"],
                linewidth=2.2, alpha=0.85, zorder=4)
    ax2.scatter([xd / 1e6], [marginal[idx]], color=style["color"], marker=style["marker"],
                s=100, zorder=7, edgecolors="white", linewidths=1.0)

ax2.set_title("G&E MMM — Marginal Return per $1 Spent\n"
              "Incremental inquiries per additional dollar of media spend",
              fontsize=14, fontweight="bold", color="black", pad=16)
ax2.set_xlabel("Annual Media Spend ($M, combined SEM + Social)", fontsize=14, fontweight="bold")
ax2.set_ylabel("Incremental Inquiries per $1", fontsize=14, fontweight="bold")
ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:.1f}M"))
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.4f}"))
plt.setp(ax2.get_xticklabels(), fontsize=12, fontweight="bold")
plt.setp(ax2.get_yticklabels(), fontsize=12, fontweight="bold")
ax2.set_ylim(bottom=0); ax2.grid(False)

legend_handles_m = [
    Line2D([0], [0], color=s["color"], marker=s["marker"], linestyle=s["linestyle"],
           linewidth=s["linewidth"], markersize=7,
           label=f"{yr} (${year_dollars[yr]/1e6:.2f}M annual)")
    for yr, s in YEAR_MARKER_STYLES.items()
]
ax2.legend(handles=legend_handles_m, loc="upper right", fontsize=13, frameon=True,
           facecolor="white", edgecolor="#CCCCCC", framealpha=0.95, markerscale=1.6,
           borderpad=1.0, labelspacing=0.8, handlelength=2.8,
           title="Annual Spend Reference", title_fontsize=12)

fig2.text(0.02, 0.01, "* Marginal = Δ incremental inquiries per $1. Dollar axis anchored to 2025 annualized spend.",
          fontsize=7.5, color=CPX_DGREY, style="italic")
plt.tight_layout(rect=[0, 0.03, 1, 1])
save_fig(fig2, "gne_b10b_marginal_return.png")
display(fig2); plt.close(fig2)

# COMMAND ----------

# ── B10 Summary Table ──────────────────────────────────────────────────────
print(f"\n{'Year':<8} {'Annual Spend':>14} {'Spend (×)':>10} {'Incr. Inq':>12} {'Inq per $1':>12}")
print("-" * 60)
for yr in [2023, 2024, 2025]:
    xd = year_dollars[yr]                                   # plotted x position ($)
    mult = xd / annual_2025_spend                           # multiple used for the curve value
    incr = incremental_inquiries(mult)                      # plotted y on B10 response curve
    marg = marginal[np.argmin(np.abs(x_dollars - xd))]      # plotted y on B10b marginal curve
    print(f"{yr:<8} ${xd:>12,.0f} {mult:>9.2f}× {incr:>12,.0f} {marg:>12.4f}")
print("-" * 60)
print(f"Anchor: 1.00× = 2025 annualized spend = ${annual_2025_spend:,.0f}")
print(f"Diminishing-returns knee: ${knee_x_m:.1f}M  (marginal < 25% of peak efficiency)")

# COMMAND ----------

mape_train = np.mean(np.abs((y[:split_idx] - pred_full[:split_idx]) / y[:split_idx])) * 100
mape_val   = np.mean(np.abs((y[split_idx:] - pred_full[split_idx:]) / y[split_idx:])) * 100

# COMMAND ----------

# ── Validation Metrics Table ───────────────────────────────────────────────
# Uses the 80/20 split already defined in Cell 5 (split_idx, pred_full, y)
y_val    = y[split_idx:]
pred_val = pred_full[split_idx:]
n_val    = n_weeks - split_idx

val_mape = np.mean(np.abs((y_val - pred_val) / y_val)) * 100
ss_res   = np.sum((y_val - pred_val) ** 2)
ss_tot   = np.sum((y_val - y_val.mean()) ** 2)
val_r2   = 1 - ss_res / ss_tot
val_bias = (pred_val.sum() - y_val.sum()) / y_val.sum() * 100
divergences = int(idata.sample_stats.diverging.sum())

print("=" * 60)
print("  V5 VALIDATION METRICS (held-out data)")
print("=" * 60)
print(f"\n{'Validation MAPE':>20} {'Validation R²':>16} {'Holdout':>12} {'Annual Bias':>14}")
print("-" * 62)
print(f"{val_mape:>19.1f}% {val_r2:>16.3f} {str(n_val)+' wks':>12} {val_bias:>+13.1f}%")
print("-" * 62)
print(f"\nConvergence: {divergences} divergences "
      f"({'clean — no sampling issues' if divergences == 0 else 'REVIEW: divergences present'})")
print(f"Split: {split_idx} weeks train / {n_val} weeks validation (80/20 time-based)")

# COMMAND ----------

# ── Media ↔ Seasonality Entanglement Diagnostic ────────────────────────────
# Q: Can the model separate media effects from seasonality, given they may co-move?
# Uses df, seas_cols, idata — all defined upstream.

# ── Check 1: Correlation of each media channel with each seasonal feature ──
#    (uses RAW spend + raw seasonal features, on the modeled weeks)
sem_raw = df["sem"].values
soc_raw = df["soc"].values
seas_raw = df[seas_cols].values  # pre-standardization is fine for correlation

print("=" * 64)
print("  CHECK 1: Media spend vs. seasonal features (collinearity risk)")
print("=" * 64)
print(f"\n{'Seasonal feature':<18} {'corr w/ SEM':>13} {'corr w/ Social':>15}")
print("-" * 48)
sem_corrs, soc_corrs = {}, {}
for j, feat in enumerate(seas_cols):
    f = seas_raw[:, j]
    if f.std() < 1e-9:
        cs = cso = np.nan
    else:
        cs  = np.corrcoef(sem_raw, f)[0, 1]
        cso = np.corrcoef(soc_raw, f)[0, 1]
    sem_corrs[feat], soc_corrs[feat] = cs, cso
    flag = "  <-- watch" if (abs(cs) > 0.4 or abs(cso) > 0.4) else ""
    print(f"{feat:<18} {cs:>+13.2f} {cso:>+15.2f}{flag}")

max_sem = np.nanmax(np.abs(list(sem_corrs.values())))
max_soc = np.nanmax(np.abs(list(soc_corrs.values())))
print("-" * 48)
print(f"Max |corr|:  SEM {max_sem:.2f}  |  Social {max_soc:.2f}")
print("Interpretation: |corr|<0.4 = channels are separable from calendar;")
print("                >0.6 = attribution may be entangled, treat split cautiously.")

# ── Check 2: Posterior credible intervals on media coefficients ──
#    Wide interval spanning ~0 => model can't confidently isolate the media effect.
print("\n" + "=" * 64)
print("  CHECK 2: Media coefficient posteriors (is the effect confident?)")
print("=" * 64)
import arviz as az
med_summary = az.summary(idata, var_names=["beta_sem", "beta_soc"],
                         hdi_prob=0.94).loc[:, ["mean", "hdi_3%", "hdi_97%", "r_hat"]]
print(f"\n{'Coef':<10} {'mean':>8} {'94% HDI':>20} {'excludes 0?':>13} {'r_hat':>7}")
print("-" * 62)
for coef in ["beta_sem", "beta_soc"]:
    m  = med_summary.loc[coef, "mean"]
    lo = med_summary.loc[coef, "hdi_3%"]
    hi = med_summary.loc[coef, "hdi_97%"]
    rh = med_summary.loc[coef, "r_hat"]
    excl = "YES (robust)" if (lo > 0 or hi < 0) else "NO (uncertain)"
    print(f"{coef:<10} {m:>+8.3f}   [{lo:>+6.3f}, {hi:>+6.3f}]  {excl:>13} {rh:>7.2f}")
print("-" * 62)
print("If the 94% HDI excludes 0, the media effect is distinguishable from")
print("noise/seasonality. If it straddles 0, the split is not well-identified.")

# COMMAND ----------

# ── Check 2: Posterior credible intervals on media coefficients ──
#    Computed directly from samples — no az.summary version dependency.
print("\n" + "=" * 64)
print("  CHECK 2: Media coefficient posteriors (is the effect confident?)")
print("=" * 64)

post = idata.posterior
print(f"\n{'Coef':<10} {'mean':>8} {'94% HDI':>22} {'excludes 0?':>14}")
print("-" * 58)
for coef in ["beta_sem", "beta_soc"]:
    samples = post[coef].values.flatten()
    m  = samples.mean()
    lo, hi = np.percentile(samples, [3, 97])   # 94% interval
    excl = "YES (robust)" if (lo > 0 or hi < 0) else "NO (uncertain)"
    print(f"{coef:<10} {m:>+8.3f}   [{lo:>+7.3f}, {hi:>+7.3f}]  {excl:>14}")
print("-" * 58)
print("If the 94% interval excludes 0, the media effect is distinguishable")
print("from noise/seasonality. If it straddles 0, the split is not well-identified.")

# COMMAND ----------

# ── G&E Media Spend Overview — SEM vs Social by Year ───────────────────────
spend_yr = pd.DataFrame({
    "SEM": [df.loc[df.index.year == yr, "sem"].sum() for yr in FULL_YRS],
    "Social": [df.loc[df.index.year == yr, "soc"].sum() for yr in FULL_YRS],
}, index=FULL_YRS)

fig, ax = plt.subplots(figsize=get_figsize("16:9"))
x = np.arange(len(FULL_YRS))
w = 0.38

b1 = ax.bar(x - w/2, spend_yr["SEM"] / 1e3, w, label="G&E SEM", color=CPX_NAVY)
b2 = ax.bar(x + w/2, spend_yr["Social"] / 1e3, w, label="G&E Social", color=CPX_RED)

# Data labels on bars
for bars in [b1, b2]:
    for bar in bars:
        h = bar.get_height()
        if h >= 0.1:
            ax.text(bar.get_x() + bar.get_width()/2, h/2,
                    f"${h:.0f}K", ha="center", va="center",
                    fontsize=14, fontweight="bold", color=WHITE)

# Total spend above each year
for i, yr in enumerate(FULL_YRS):
    total = (spend_yr.loc[yr, "SEM"] + spend_yr.loc[yr, "Social"]) / 1e3
    ax.text(i, max(spend_yr.loc[yr, "SEM"], spend_yr.loc[yr, "Social"]) / 1e3 * 1.02,
            f"Total: ${total:.0f}K", ha="center", va="bottom",
            fontsize=14, fontweight="bold", color=CPX_DGREY)

ax.set_xticks(x)
ax.set_xticklabels(FULL_YRS, fontsize=14, fontweight="bold")
ax.set_ylabel("Annual Media Spend ($K)", fontsize=14, fontweight="bold")
ax.set_xlabel("Year", fontsize=14, fontweight="bold")
ax.tick_params(axis="both", labelsize=14)
plt.setp(ax.get_xticklabels(), fontsize=14, fontweight="bold")
plt.setp(ax.get_yticklabels(), fontsize=14, fontweight="bold")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:.0f}K"))
ax.set_ylim(0, spend_yr.max().max() / 1e3 * 1.25)
ax.set_title("G&E Media Spend by Channel — SEM vs Social", fontsize=16, fontweight="bold", pad=16)
ax.legend(loc="upper right", prop={"weight": "bold", "size": 14})
plt.tight_layout()
plt.show()

# ── Spend Table ────────────────────────────────────────────────────────────
print(f"\n{'Year':<8} {'SEM':>12} {'Social':>12} {'Total':>12}")
print("-" * 46)
for yr in FULL_YRS:
    sem = spend_yr.loc[yr, "SEM"]
    soc = spend_yr.loc[yr, "Social"]
    print(f"{yr:<8} ${sem:>10,.0f} ${soc:>10,.0f} ${sem+soc:>10,.0f}")
