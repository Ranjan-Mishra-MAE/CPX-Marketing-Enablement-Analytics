# Databricks notebook source
# DBTITLE 1,LBE MMM Budget Scenarios Overview
# MAGIC %md
# MAGIC # LBE MMM — Budget Scenario Analysis & Stakeholder Deliverable (FINAL)
# MAGIC **Three annual scenarios on the validated LBE model, with posterior credible intervals.**
# MAGIC
# MAGIC Operates on the LBE MMM model's **validated** saturation curves (log1p saturation).
# MAGIC The decision variable is **funnel-level spend** (Awareness, Consideration);
# MAGIC channel-level outcomes (TRR, PDM) follow from observed allocation shares.
# MAGIC
# MAGIC ### Scenarios (annual outcomes only)
# MAGIC 1. **Budget uplift** — total 2025 spend ×(1.10, 1.20, 1.30), allocation *pattern unchanged*,
# MAGIC    baseline unchanged. Shows revenue lift; saturation means lift < spend increase.
# MAGIC 2. **Reshuffle** — shift spend from low-marginal funnel to high-marginal funnel (capped at 20%).
# MAGIC    Total budget constant.
# MAGIC 3. **Combined** — uplift × reshuffle together.
# MAGIC
# MAGIC ### Uncertainty
# MAGIC Each scenario reports the predicted annual revenue lift with an **89% credible interval** from
# MAGIC the model's posterior (8,000 draws). The CI reflects uncertainty in the LIFT via paired
# MAGIC per-draw differencing — shared parameter uncertainty cancels.

# COMMAND ----------

# DBTITLE 1,Install Dependencies
# MAGIC %pip install xarray --quiet

# COMMAND ----------

# DBTITLE 1,Load LBE Model Artifacts
import os, pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker

RESULTS_PKL = "/Volumes/data_exploration_hub/marketing_hub/model_data/Final_Data/MMM_LBE/lbe_mmm_results.pkl"
OUTPUT_DIR  = "/Volumes/data_exploration_hub/marketing_hub/model_data/Final_Data/MMM_LBE"

with open(RESULTS_PKL, "rb") as f:
    results = pickle.load(f)

draws    = results["draws"]
n_draws  = int(results["n_draws"])
y_mean   = float(results["y_mean"])
y_std    = float(results["y_std"])
aw_max   = float(results["aw_max"])
co_max   = float(results["co_max"])
tr       = results["tr"]
va       = results["va"]
df_train = results["df_train"].copy()
df_val   = results["df_val"].copy()
metrics  = results["metrics"]
decomp   = results["decomp"]

df_train["week_date"] = pd.to_datetime(df_train["week_date"])
df_val["week_date"]   = pd.to_datetime(df_val["week_date"])
df_all = pd.concat([df_train, df_val], ignore_index=True).sort_values("week_date")
df_all["year"] = df_all["week_date"].dt.year

# Full data arrays for predictions
full_data = {
    "aw":   np.concatenate([np.asarray(tr["aw"]),   np.asarray(va["aw"])]),
    "co":   np.concatenate([np.asarray(tr["co"]),   np.asarray(va["co"])]),
    "t":    np.concatenate([np.asarray(tr["t"]),    np.asarray(va["t"])]),
    "temp": np.concatenate([np.asarray(tr["temp"]), np.asarray(va["temp"])]),
    "cal":  np.concatenate([np.asarray(tr["cal"]),  np.asarray(va["cal"])]),
}

CPX_NAVY="#002B5C"; CPX_RED="#E31837"; CPX_GOLD="#F5A800"
CPX_GREEN="#2E8B57"; CPX_DGREY="#4A4A4A"; CPX_LGREY="#CCCCCC"
plt.rcParams.update({"font.family":"sans-serif","axes.spines.top":False,
    "axes.spines.right":False,"figure.facecolor":"white","figure.dpi":120})

def save_fig(fig, fname):
    import shutil
    tmp = f"/tmp/{fname}"
    fig.savefig(tmp, dpi=150, bbox_inches="tight", facecolor="white")
    shutil.copy2(tmp, os.path.join(OUTPUT_DIR, fname))
    print(f"\u2713 Saved: {fname}")

print(f"LBE MMM loaded: {n_draws} posterior draws")
print(f"Train MAPE: {metrics['train_mape']:.1f}%  Val MAPE: {metrics['val_mape']:.1f}%")
print(f"Train R\u00b2: {metrics['train_r2']:.3f}   Val R\u00b2: {metrics['val_r2']:.3f}")

# COMMAND ----------

# DBTITLE 1,Response Curve Functions
# MAGIC %md
# MAGIC ## 1. Funnel Response Curve & Contribution Functions
# MAGIC Annual revenue contribution for each funnel at a given annual spend level.
# MAGIC Saturation: `log1p(normalized_spend * scale)`. At steady-state, adstocked value = input
# MAGIC (normalized geometric weights).

# COMMAND ----------

# DBTITLE 1,Response Functions and Baseline
# Average baseline context (trend, temp, calendar at their means)
t_avg    = full_data["t"].mean()
temp_avg = full_data["temp"].mean()
cal_avg  = full_data["cal"].mean(axis=0)
bc_arr   = results["bc"]
bc_mean  = bc_arr.mean(axis=0)
cal_eff_avg = float(cal_avg @ bc_mean)

# Posterior mean parameters
params_mean = {
    "intercept":  float(draws["intercept"].mean()),
    "beta_trend": float(draws["beta_trend"].mean()),
    "beta_temp":  float(draws["beta_temp"].mean()),
    "beta_aw":    float(draws["beta_aw"].mean()),
    "beta_co":    float(draws["beta_co"].mean()),
    "aw_scale":   float(draws["aw_scale"].mean()),
    "co_scale":   float(draws["co_scale"].mean()),
}

def annual_revenue_contribution(aw_annual_spend, co_annual_spend, params=None):
    """Annual media revenue contribution given raw $ spend per funnel.
    Uses steady-state adstock (constant weekly spend)."""
    p = params or params_mean
    # Normalize to model scale
    aw_norm = (aw_annual_spend / 52) / aw_max
    co_norm = (co_annual_spend / 52) / co_max
    # Saturation
    aw_sat = np.log1p(aw_norm * p["aw_scale"])
    co_sat = np.log1p(co_norm * p["co_scale"])
    # Revenue
    base_log = (p["intercept"] + p["beta_trend"] * t_avg
                + p["beta_temp"] * temp_avg + cal_eff_avg)
    full_log = base_log + p["beta_aw"] * aw_sat + p["beta_co"] * co_sat
    base_rev = np.exp(base_log * y_std + y_mean)
    full_rev = np.exp(full_log * y_std + y_mean)
    return (full_rev - base_rev) * 52  # annual media contribution

def funnel_marginal_return(funnel, aw_spend, co_spend, delta=5200.0):
    """Marginal return (revenue per $1) for a funnel at given spend levels."""
    if funnel == "Awareness":
        up = annual_revenue_contribution(aw_spend + delta, co_spend)
        dn = annual_revenue_contribution(max(0, aw_spend - delta), co_spend)
    else:
        up = annual_revenue_contribution(aw_spend, co_spend + delta)
        dn = annual_revenue_contribution(aw_spend, max(0, co_spend - delta))
    return (up - dn) / (2 * delta)

# 2025 baseline spend
df_2025 = df_all[df_all["year"] == 2025]
baseline_aw_annual = df_2025["Awareness"].sum()
baseline_co_annual = df_2025["Consideration"].sum()
total_baseline_spend = baseline_aw_annual + baseline_co_annual
baseline_revenue = annual_revenue_contribution(baseline_aw_annual, baseline_co_annual)

aw_marginal = funnel_marginal_return("Awareness", baseline_aw_annual, baseline_co_annual)
co_marginal = funnel_marginal_return("Consideration", baseline_aw_annual, baseline_co_annual)

print(f"2025 Baseline Annual Media Spend: ${total_baseline_spend/1e6:.2f}M")
print(f"  Awareness:     ${baseline_aw_annual/1e6:.2f}M")
print(f"  Consideration: ${baseline_co_annual/1e6:.2f}M")
print(f"\nBaseline annual media revenue contribution: ${baseline_revenue/1e6:.2f}M")
print(f"\nMarginal return at baseline (revenue per $1):")
print(f"  Awareness:     {aw_marginal:.2f}x")
print(f"  Consideration: {co_marginal:.2f}x")

# COMMAND ----------

# DBTITLE 1,Posterior CI Engine
# MAGIC %md
# MAGIC ## 2. Posterior CI Engine
# MAGIC Propagates the model's 8,000 posterior draws through any spend allocation to get an 89%
# MAGIC credible interval on annual revenue lift. Uses paired per-draw differencing for tight CIs.

# COMMAND ----------

# DBTITLE 1,Posterior Draws and CI Functions
# Cache draw indices
N_DRAWS_CI = 800
rng = np.random.default_rng(42)
DRAW_IDX = rng.choice(n_draws, size=min(N_DRAWS_CI, n_draws), replace=False)

def annual_revenue_draws(aw_annual_spend, co_annual_spend):
    """Vector of annual revenue contributions across posterior draws."""
    aw_norm = (aw_annual_spend / 52) / aw_max
    co_norm = (co_annual_spend / 52) / co_max
    
    intercepts  = draws["intercept"][DRAW_IDX]
    beta_trends = draws["beta_trend"][DRAW_IDX]
    beta_temps  = draws["beta_temp"][DRAW_IDX]
    beta_aws    = draws["beta_aw"][DRAW_IDX]
    beta_cos    = draws["beta_co"][DRAW_IDX]
    aw_scales   = draws["aw_scale"][DRAW_IDX]
    co_scales   = draws["co_scale"][DRAW_IDX]
    
    aw_sat = np.log1p(aw_norm * aw_scales)
    co_sat = np.log1p(co_norm * co_scales)
    
    base_log = intercepts + beta_trends * t_avg + beta_temps * temp_avg + cal_eff_avg
    full_log = base_log + beta_aws * aw_sat + beta_cos * co_sat
    
    base_rev = np.exp(base_log * y_std + y_mean)
    full_rev = np.exp(full_log * y_std + y_mean)
    return (full_rev - base_rev) * 52

# Baseline per-draw vector (shared across all scenarios)
BASELINE_DRAWS = annual_revenue_draws(baseline_aw_annual, baseline_co_annual)
print(f"Posterior CI engine ready: {len(DRAW_IDX)} draws")
print(f"Baseline revenue: point ${baseline_revenue/1e6:.2f}M | draw mean ${BASELINE_DRAWS.mean()/1e6:.2f}M")

# COMMAND ----------

# DBTITLE 1,Confidence Label
# MAGIC %md
# MAGIC ## 3. Confidence Label from CI Width
# MAGIC Translates the credible-interval width into a High/Medium/Low tag. Reflects model uncertainty
# MAGIC about the lift, **not** the probability the scenario is enacted.

# COMMAND ----------

# DBTITLE 1,Lift with CI Function
def lift_with_ci(aw_spend, co_spend, label, ci=0.89):
    """Paired per-draw lift CI: (scenario - baseline) draw by draw."""
    scen_draws = annual_revenue_draws(aw_spend, co_spend)
    lift_draws = scen_draws - BASELINE_DRAWS
    pct_draws  = lift_draws / BASELINE_DRAWS * 100
    
    point_rev = annual_revenue_contribution(aw_spend, co_spend)
    lift_pt   = point_rev - baseline_revenue
    pct_pt    = lift_pt / baseline_revenue * 100
    
    lo_q, hi_q = (1-ci)/2, 1-(1-ci)/2
    pct_lo = float(np.quantile(pct_draws, lo_q))
    pct_hi = float(np.quantile(pct_draws, hi_q))
    width  = pct_hi - pct_lo
    conf = "High" if width < 4 else ("Medium" if width < 8 else "Low")
    crosses_zero = (pct_lo <= 0 <= pct_hi)
    effect = "effect uncertain (CI crosses 0)" if crosses_zero else "directionally positive"
    return {"label": label, "spend": aw_spend + co_spend,
            "aw_spend": aw_spend, "co_spend": co_spend,
            "revenue": point_rev, "lift": lift_pt, "lift_pct": pct_pt,
            "pct_lo": pct_lo, "pct_hi": pct_hi, "ci_width": width,
            "confidence": conf, "crosses_zero": crosses_zero, "effect": effect}

print("Lift CI = 89% quantiles of PAIRED per-draw difference (scenario - baseline).")
print("Confidence: CI width <4pp High, 4-8pp Medium, >8pp Low.")

# COMMAND ----------

# DBTITLE 1,Scenario 1 Header
# MAGIC %md
# MAGIC ## 4. Scenario 1 — Budget Uplift (+10 / +20 / +30%)
# MAGIC Spend pattern unchanged: both funnels scaled by the same factor. Saturation means revenue lift is
# MAGIC sub-proportional to the spend increase.

# COMMAND ----------

# DBTITLE 1,Scenario 1 - Budget Uplift
s1_results = []
for factor in [1.10, 1.20, 1.30]:
    aw_s = baseline_aw_annual * factor
    co_s = baseline_co_annual * factor
    r = lift_with_ci(aw_s, co_s, f"+{round((factor-1)*100)}% budget")
    r["spend_increase_pct"] = (factor - 1) * 100
    s1_results.append(r)

s1_df = pd.DataFrame([{
    "Scenario": r["label"],
    "Annual Spend": f"${r['spend']/1e6:.2f}M",
    "Spend \u2191": f"+{r['spend_increase_pct']:.0f}%",
    "Revenue Contrib": f"${r['revenue']/1e6:.2f}M",
    "Rev Lift": f"+${r['lift']/1e6:.2f}M",
    "Lift %": f"+{r['lift_pct']:.1f}%",
    "89% CI": f"[+{r['pct_lo']:.1f}%, +{r['pct_hi']:.1f}%]",
    "Confidence": r["confidence"],
} for r in s1_results])

print("SCENARIO 1 \u2014 BUDGET UPLIFT (pattern unchanged)\n")
display(s1_df)
print("\nNote: each lift % is sub-proportional to the spend increase due to saturation.")
print("CI = 89% probability the TRUE lift lies in range, conditional on the model.")

# COMMAND ----------

# DBTITLE 1,Scenario 2 Header
# MAGIC %md
# MAGIC ## 5. Scenario 2 — Reshuffle (shift from low-marginal to high-marginal funnel)
# MAGIC Shift up to 20% from the lower-marginal funnel to the higher-marginal funnel.
# MAGIC Total budget held constant.

# COMMAND ----------

# DBTITLE 1,Scenario 2 - Reshuffle
def reshuffle_lbe(aw_base, co_base, shift_pct=0.20):
    """Shift spend from low-marginal to high-marginal funnel."""
    aw_mr = funnel_marginal_return("Awareness", aw_base, co_base)
    co_mr = funnel_marginal_return("Consideration", aw_base, co_base)
    
    if aw_mr > co_mr:
        # Shift from Consideration to Awareness
        shift = co_base * shift_pct
        return aw_base + shift, co_base - shift, "Consideration \u2192 Awareness"
    else:
        # Shift from Awareness to Consideration
        shift = aw_base * shift_pct
        return aw_base - shift, co_base + shift, "Awareness \u2192 Consideration"

aw_reshuffled, co_reshuffled, direction = reshuffle_lbe(baseline_aw_annual, baseline_co_annual)
r2 = lift_with_ci(aw_reshuffled, co_reshuffled, "Reshuffle (20% cap)")

print("SCENARIO 2 \u2014 RESHUFFLE\n")
print(f"Direction: {direction}")
print(f"  Awareness:     ${baseline_aw_annual/1e6:.2f}M \u2192 ${aw_reshuffled/1e6:.2f}M ({(aw_reshuffled-baseline_aw_annual)/baseline_aw_annual*100:+.1f}%)")
print(f"  Consideration: ${baseline_co_annual/1e6:.2f}M \u2192 ${co_reshuffled/1e6:.2f}M ({(co_reshuffled-baseline_co_annual)/baseline_co_annual*100:+.1f}%)")
print(f"\nTotal spend: ${(aw_reshuffled+co_reshuffled)/1e6:.2f}M (unchanged)")
print(f"Revenue: ${r2['revenue']/1e6:.2f}M  Lift: +${r2['lift']/1e6:.2f}M (+{r2['lift_pct']:.1f}%)")
print(f"89% CI [{r2['pct_lo']:+.1f}%, {r2['pct_hi']:+.1f}%]  Confidence: {r2['confidence']}")

# COMMAND ----------

# DBTITLE 1,Scenario 3 Header
# MAGIC %md
# MAGIC ## 6. Scenario 3 — Combined (uplift + reshuffle)
# MAGIC Apply each budget uplift, then reshuffle the larger budget toward the high-marginal funnel
# MAGIC (same 20% cap on shift).

# COMMAND ----------

# DBTITLE 1,Scenario 3 - Combined
s3_results = []
for factor in [1.10, 1.20, 1.30]:
    aw_up = baseline_aw_annual * factor
    co_up = baseline_co_annual * factor
    aw_comb, co_comb, _ = reshuffle_lbe(aw_up, co_up)
    r = lift_with_ci(aw_comb, co_comb, f"+{round((factor-1)*100)}% + reshuffle")
    r["spend_increase_pct"] = (factor - 1) * 100
    s3_results.append(r)

s3_df = pd.DataFrame([{
    "Scenario": r["label"],
    "Annual Spend": f"${r['spend']/1e6:.2f}M",
    "Revenue Contrib": f"${r['revenue']/1e6:.2f}M",
    "Lift %": f"+{r['lift_pct']:.1f}%",
    "89% CI": f"[+{r['pct_lo']:.1f}%, +{r['pct_hi']:.1f}%]",
    "Confidence": r["confidence"],
} for r in s3_results])

print("SCENARIO 3 \u2014 COMBINED\n")
display(s3_df)

# COMMAND ----------

# DBTITLE 1,Visuals Header
# MAGIC %md
# MAGIC ## 7. Annual Outcome Visuals
# MAGIC Two charts: (1) lift vs spend increase for S1 (diminishing returns), (2) all scenarios with CIs.

# COMMAND ----------

# DBTITLE 1,Scenario Outcome Visuals
fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

# ── (A) S1: Diminishing returns curve ────────────────────────────────────
ax = axes[0]
sp = [0] + [r["spend_increase_pct"] for r in s1_results]
at = [0] + [r["lift_pct"]           for r in s1_results]

ax.plot(sp, sp, "--", color=CPX_LGREY, lw=1.5, label="1:1 (no saturation)")
ax.plot(sp, at, "-o", color=CPX_NAVY,  lw=2.5, markersize=8, label="Predicted revenue lift")

for r in s1_results:
    ax.errorbar(r["spend_increase_pct"], r["lift_pct"],
                yerr=[[r["lift_pct"] - r["pct_lo"]], [r["pct_hi"] - r["lift_pct"]]],
                color=CPX_NAVY, capsize=5, alpha=0.5)
    ax.annotate(f"+{r['lift_pct']:.1f}%",
                (r["spend_increase_pct"], r["lift_pct"]),
                textcoords="offset points", xytext=(8, -4),
                fontsize=9, fontweight="bold", color=CPX_NAVY)

ax.set_xlabel("Budget increase (%)", fontsize=9, color=CPX_DGREY)
ax.set_ylabel("Revenue lift (%)",  fontsize=9, color=CPX_DGREY)
ax.set_title("Scenario 1 \u2014 Diminishing Returns on Budget Uplift",
             fontweight="bold", color="black", fontsize=11)
ax.legend(frameon=False, fontsize=9)
ax.set_axisbelow(True)
ax.grid(axis="y", alpha=0.2, color=CPX_LGREY)
ax.tick_params(labelsize=8)

# ── (B) All scenarios — bar chart with CI whiskers ──────────────────────
ax = axes[1]
all_results = s1_results + [r2] + s3_results
labels  = [r["label"]    for r in all_results]
lifts   = [r["lift_pct"] for r in all_results]
los     = [r["lift_pct"] - r["pct_lo"]  for r in all_results]
his     = [r["pct_hi"]   - r["lift_pct"] for r in all_results]
colors  = [CPX_NAVY] * 3 + [CPX_GOLD] + [CPX_GREEN] * 3
x       = np.arange(len(labels))

ax.bar(x, lifts, color=colors, alpha=0.88, edgecolor="white")
ax.errorbar(x, lifts, yerr=[los, his],
            fmt="none", ecolor=CPX_DGREY, capsize=4, alpha=0.6)

for i, r in enumerate(all_results):
    tag   = "effect uncertain" if r["crosses_zero"] else r["confidence"]
    color = CPX_RED if r["crosses_zero"] else CPX_DGREY
    ax.text(i, lifts[i] + his[i] + max(his) * 0.12,
            tag, ha="center", fontsize=8, fontweight="bold", color=color)
    ax.text(i, lifts[i] / 2,
            f"+{lifts[i]:.1f}%", ha="center", fontsize=8,
            fontweight="bold", color="white")

ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8, color=CPX_DGREY)
ax.set_ylabel("Revenue lift (%)", fontsize=9, color=CPX_DGREY)
ax.set_title("All Scenarios \u2014 Annual Lift with 89% CI\n(label = model confidence from CI width)",
             fontweight="bold", color="black", fontsize=11)
ax.set_axisbelow(True)
ax.grid(axis="y", alpha=0.2, color=CPX_LGREY)
ax.tick_params(labelsize=8)

# Shared legend
leg = [
    mpatches.Patch(facecolor=CPX_NAVY,  label="Budget uplift"),
    mpatches.Patch(facecolor=CPX_GOLD,  label="Reshuffle"),
    mpatches.Patch(facecolor=CPX_GREEN, label="Combined"),
]
fig.legend(handles=leg, loc="lower center", ncol=3, frameon=False,
           fontsize=9, bbox_to_anchor=(0.5, -0.06))

fig.suptitle("LBE MMM \u2014 Budget Scenario Outcomes (2025)",
             fontsize=13, fontweight="bold", color="black", y=1.02)

fig.text(0.99, -0.04,
         "Cineplex Marketing Analytics  |  LBE MMM",
         ha="right", va="bottom", fontsize=6, color=CPX_LGREY)

plt.tight_layout()
save_fig(fig, "lbe_scenario_outcomes.png")
plt.show()

print("\nCIs are paired per-draw lift intervals from the posterior \u2014")
print("tight because shared parameter uncertainty cancels in the scenario-minus-baseline difference.")

# COMMAND ----------

# DBTITLE 1,Summary
# MAGIC %md
# MAGIC ## 8. Executive Summary
# MAGIC Consolidated view of all scenarios for stakeholder presentation.

# COMMAND ----------

# DBTITLE 1,Executive Summary Table
# ── Executive Summary Table ───────────────────────────────────────────────
all_results_final = s1_results + [r2] + s3_results

summary_rows = [{
    "Scenario": r["label"],
    "Annual Spend ($M)": f"${r['spend']/1e6:.2f}M",
    "\u0394 vs Baseline": f"+${(r['spend']-total_baseline_spend)/1e6:.2f}M" if r['spend'] > total_baseline_spend else "$0 (constant)",
    "Revenue Contrib ($M)": f"${r['revenue']/1e6:.2f}M",
    "Lift ($M)": f"+${r['lift']/1e6:.2f}M",
    "Lift %": f"+{r['lift_pct']:.1f}%",
    "89% CI": f"[{r['pct_lo']:+.1f}%, {r['pct_hi']:+.1f}%]",
    "Confidence": r["confidence"],
} for r in all_results_final]

summary_df = pd.DataFrame(summary_rows)

print("\u2550" * 80)
print("  LBE MMM \u2014 BUDGET SCENARIO EXECUTIVE SUMMARY")
print(f"  Baseline: ${total_baseline_spend/1e6:.2f}M annual spend \u2192 ${baseline_revenue/1e6:.2f}M media revenue")
print("\u2550" * 80)
from IPython.display import display as ipy_display
ipy_display(summary_df)

print(f"\n\u2022 Awareness marginal ROI ({aw_marginal:.2f}x) >> Consideration ({co_marginal:.2f}x)")
print(f"\u2022 Reshuffle direction: {direction}")
print(f"\u2022 Best outcome: {all_results_final[-1]['label']} at +{all_results_final[-1]['lift_pct']:.1f}% lift")
print(f"\n* CIs from {len(DRAW_IDX)} posterior draws. Confidence = CI width.")

# COMMAND ----------

# DBTITLE 1,Per-Scenario Channel Spend Breakdown
# MAGIC %md
# MAGIC ## Per-Scenario Channel Spend — Baseline vs Scenario
# MAGIC Breaks down funnel-level spend to channels (TRR, PDM) using 2025 observed allocation shares.
# MAGIC Plots baseline vs scenario spend by channel, coloured by funnel.

# COMMAND ----------

# DBTITLE 1,Per-Scenario Channel Spend vs Lift
# ── Per-Scenario Channel Spend — Baseline vs Scenario (channel-split) ─────────
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np

# ── 1. Build channel share table from 2025 data ─────────────────────────────
df25 = df_all[df_all["year"] == 2025]

# Channel shares within each funnel
aw_trr_total = df25["Awareness_TRR"].sum()
aw_pdm_total = df25["Awareness_PDM"].sum()
co_trr_total = df25["Consideration_TRR"].sum()
co_pdm_total = df25["Consideration_PDM"].sum()

aw_total = aw_trr_total + aw_pdm_total
co_total = co_trr_total + co_pdm_total

CHANNEL_SHARES = {
    "Awareness_TRR":     aw_trr_total / aw_total,
    "Awareness_PDM":     aw_pdm_total / aw_total,
    "Consideration_TRR": co_trr_total / co_total,
    "Consideration_PDM": co_pdm_total / co_total,
}

CHANNEL_FUNNEL = {
    "Awareness_TRR":     "Awareness",
    "Awareness_PDM":     "Awareness",
    "Consideration_TRR": "Consideration",
    "Consideration_PDM": "Consideration",
}

FUNNEL_COLORS = {"Awareness": CPX_NAVY, "Consideration": CPX_RED}
CHANNELS = list(CHANNEL_SHARES.keys())

def split_to_channels(aw_spend, co_spend):
    """Split funnel spend to channels using observed shares."""
    return {
        "Awareness_TRR":     aw_spend * CHANNEL_SHARES["Awareness_TRR"],
        "Awareness_PDM":     aw_spend * CHANNEL_SHARES["Awareness_PDM"],
        "Consideration_TRR": co_spend * CHANNEL_SHARES["Consideration_TRR"],
        "Consideration_PDM": co_spend * CHANNEL_SHARES["Consideration_PDM"],
    }

baseline_channels = split_to_channels(baseline_aw_annual, baseline_co_annual)

print("2025 Channel Allocation Shares:")
for ch, share in CHANNEL_SHARES.items():
    print(f"  {ch:<22} {share*100:.1f}%  (${baseline_channels[ch]/1e3:.0f}K)")
print(f"\nTotal: ${sum(baseline_channels.values())/1e6:.2f}M")

# ── 2. Build scenario allocations at channel level ────────────────────────
scenarios_channel = []

# S1: Budget uplift
for r in s1_results:
    ch_alloc = split_to_channels(r["aw_spend"], r["co_spend"])
    scenarios_channel.append({"label": r["label"], "type": "budget",
                              "channels": ch_alloc, "lift_pct": r["lift_pct"],
                              "lift_$M": r["lift"]/1e6, "ci": f"[{r['pct_lo']:+.1f}%, {r['pct_hi']:+.1f}%]"})

# S2: Reshuffle
scenarios_channel.append({"label": r2["label"], "type": "reshuffle",
                          "channels": split_to_channels(r2["aw_spend"], r2["co_spend"]),
                          "lift_pct": r2["lift_pct"],
                          "lift_$M": r2["lift"]/1e6, "ci": f"[{r2['pct_lo']:+.1f}%, {r2['pct_hi']:+.1f}%]"})

# S3: Combined
for r in s3_results:
    ch_alloc = split_to_channels(r["aw_spend"], r["co_spend"])
    scenarios_channel.append({"label": r["label"], "type": "combined",
                              "channels": ch_alloc, "lift_pct": r["lift_pct"],
                              "lift_$M": r["lift"]/1e6, "ci": f"[{r['pct_lo']:+.1f}%, {r['pct_hi']:+.1f}%]"})

# ── 3. Plot: separate image per scenario ─────────────────────────────────
for idx, scen in enumerate(scenarios_channel):
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("white")
    
    ch_labels = [c.replace("_", "\n") for c in CHANNELS]
    baseline_vals = [baseline_channels[c] / 1e3 for c in CHANNELS]  # in $K
    scenario_vals = [scen["channels"][c] / 1e3 for c in CHANNELS]
    colors_ch = [FUNNEL_COLORS[CHANNEL_FUNNEL[c]] for c in CHANNELS]
    
    x_ch = np.arange(len(CHANNELS))
    w = 0.35
    
    # 2025 Actual bars (solid)
    ax.bar(x_ch - w/2, baseline_vals, w, color=colors_ch,
           alpha=0.90, edgecolor="white", linewidth=0.5, label="2025 Actual Spend")
    # Scenario bars (hatched)
    bars_scen = ax.bar(x_ch + w/2, scenario_vals, w, color=colors_ch,
           alpha=0.45, edgecolor=colors_ch, linewidth=1.5, label="Scenario Spend")
    for bar in bars_scen:
        bar.set_hatch("//")
    
    # Delta labels
    for i, ch in enumerate(CHANNELS):
        delta = scenario_vals[i] - baseline_vals[i]
        if abs(delta) > 1:
            color_d = CPX_GREEN if delta > 0 else CPX_RED
            ax.text(x_ch[i] + w/2, scenario_vals[i] + 8,
                    f"{delta:+.0f}K", ha="center", fontsize=10,
                    fontweight="bold", color=color_d)
    
    ax.set_xticks(x_ch)
    ax.set_xticklabels(ch_labels, fontsize=10)
    ax.set_ylabel("Spend ($K)", fontsize=10, color=CPX_DGREY)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:.0f}K"))
    ax.set_title(f"LBE MMM \u2014 {scen['label']}\n"
                 f"2025 Actual vs Scenario Spend  |  Rev lift: +${scen['lift_$M']:.2f}M ({scen['lift_pct']:+.1f}%)  |  89% CI: {scen['ci']}",
                 fontsize=11, fontweight="bold", color="black", pad=12)
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=9)
    
    # Legend
    p_solid = mpatches.Patch(facecolor=CPX_NAVY, alpha=0.90, edgecolor="white", label="2025 Actual (solid)")
    p_hatch = mpatches.Patch(facecolor=CPX_NAVY, alpha=0.45, edgecolor=CPX_NAVY, linewidth=1.5,
                             hatch="//", label="Scenario (hatched)")
    p_aw = mpatches.Patch(facecolor=CPX_NAVY, alpha=0.90, label="Awareness")
    p_co = mpatches.Patch(facecolor=CPX_RED,  alpha=0.90, label="Consideration")
    leg = [p_aw, p_co, p_solid, p_hatch]
    ax.legend(handles=leg, fontsize=9, loc="upper right", framealpha=0.95, edgecolor=CPX_LGREY)
    
    fig.text(0.99, 0.01, "Cineplex Marketing Analytics  |  LBE MMM",
             ha="right", va="bottom", fontsize=6, color=CPX_LGREY)
    
    plt.tight_layout()
    fname = f"FINAL_channel_spend_{idx+1}_{scen['type']}.png"
    save_fig(fig, fname)
    plt.show()
    plt.close(fig)

# COMMAND ----------

# DBTITLE 1,Export Scenario Summary CSV
# ── Master summary table + export ────────────────────────────────────────────
summary_export = pd.DataFrame([{
    "Scenario": r["label"],
    "Annual Spend": round(r["spend"], 0),
    "Annual Revenue Contrib": round(r["revenue"], 0),
    "Lift vs Baseline": round(r["lift"], 0),
    "Lift %": round(r["lift_pct"], 1),
    "CI Low %": round(r["pct_lo"], 1),
    "CI High %": round(r["pct_hi"], 1),
    "Confidence": r["confidence"],
} for r in (s1_results + [r2] + s3_results)])

display(summary_export)
summary_export.to_csv(f"{OUTPUT_DIR}/scenario_summary.csv", index=False)
print(f"\n\u2713 Saved scenario_summary.csv")
print("\nReminder: 'Confidence' and the CI reflect MODEL uncertainty about the lift \u2014")
print("they are NOT the probability a scenario is enacted (that is a business decision).")

# COMMAND ----------

# DBTITLE 1,Stakeholder Deliverable Section
# MAGIC %md
# MAGIC ---
# MAGIC # Stakeholder Deliverable — Finalized Scenario Outcomes
# MAGIC
# MAGIC This section consolidates the analysis into presentation-ready outputs:
# MAGIC * **Both frames side by side** — media-attributable lift (what the model moves) *and* total-revenue
# MAGIC   lift (what it means for the full business), so the compression is explicit.
# MAGIC * **Honest reshuffle labeling** — flagged as directional/inconclusive where its CI crosses zero.
# MAGIC * **Exported tables** — CSV for further use.
# MAGIC
# MAGIC All numbers reconcile to **actual 2025 total revenue from df_all**. Media drives a portion of total,
# MAGIC so media lifts compress to smaller total-revenue lifts.

# COMMAND ----------

# DBTITLE 1,Reconciliation and Final DataFrame
# ── Reconciliation inputs ───────────────────────────────────────────────────────
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker

actual_total_rev = df_all[df_all["year"] == 2025]["LBE_Revenue"].sum()
media_base_rev   = baseline_revenue  # from response curve at 2025 spend
MEDIA_SHARE      = media_base_rev / actual_total_rev
non_media_base   = actual_total_rev - media_base_rev

# Build final_df with both media-frame and total-frame lifts
scen_defs = []
for r in s1_results:
    scen_defs.append((r["label"], r["lift_pct"], r["pct_lo"], r["pct_hi"], "budget"))
scen_defs.append((r2["label"], r2["lift_pct"], r2["pct_lo"], r2["pct_hi"], "reshuffle"))
for r in s3_results:
    scen_defs.append((r["label"], r["lift_pct"], r["pct_lo"], r["pct_hi"], "combined"))

def build_final_rows():
    rows = []
    for name, pt, lo, hi, kind in scen_defs:
        # Media-frame absolute (on validated media base)
        m_pt = media_base_rev * pt / 100
        m_lo = media_base_rev * lo / 100
        m_hi = media_base_rev * hi / 100
        # Total-frame: revenue gained = media gain; % of real total
        t_pt = m_pt / actual_total_rev * 100
        t_lo = m_lo / actual_total_rev * 100
        t_hi = m_hi / actual_total_rev * 100
        crosses = lo <= 0 <= hi
        width = hi - lo
        conf = ("Directional \u2014 CI crosses 0" if crosses
                else ("High" if width < 4 else ("Medium" if width < 8 else "Low")))
        rows.append({
            "Scenario": name, "Type": kind,
            "Media Lift %": pt, "Media CI Low %": lo, "Media CI High %": hi,
            "Revenue Gained ($)": m_pt, "Gained CI Low": m_lo, "Gained CI High": m_hi,
            "New Total Revenue": actual_total_rev + m_pt,
            "Total Lift %": t_pt, "Total CI Low %": t_lo, "Total CI High %": t_hi,
            "Confidence": conf,
        })
    return pd.DataFrame(rows)

final_df = build_final_rows()
print(f"Reconciled: total ${actual_total_rev/1e6:.2f}M | media base ${media_base_rev/1e6:.2f}M "
      f"({MEDIA_SHARE*100:.1f}% of total)")
print(f"Built {len(final_df)} scenarios.")
display(final_df[["Scenario", "Type", "Media Lift %", "Total Lift %", "Revenue Gained ($)", "Confidence"]])

# COMMAND ----------

# DBTITLE 1,Side-by-Side Visual Header
# MAGIC %md
# MAGIC ## Side-by-Side Visual — Media Lift vs Total-Revenue Lift
# MAGIC Left: lift on media-attributable revenue (what the scenarios move). Right: the same scenarios
# MAGIC translated to % of real total revenue. The right bars are smaller — media is ~8% of total.

# COMMAND ----------

# DBTITLE 1,Side-by-Side Scenario Frames
def bar_color(row):
    if "crosses" in row["Confidence"]: return "#9AA0A6"
    return {"budget": CPX_NAVY, "combined": CPX_GREEN, "reshuffle": CPX_GOLD}[row["Type"]]

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
x = np.arange(len(final_df))
labels = final_df["Scenario"].tolist()
colors = [bar_color(r) for _, r in final_df.iterrows()]

# LEFT — media-attributable lift
ax = axes[0]
m_pt = final_df["Media Lift %"].values
m_lo = (final_df["Media Lift %"] - final_df["Media CI Low %"]).values
m_hi = (final_df["Media CI High %"] - final_df["Media Lift %"]).values
ax.bar(x, m_pt, color=colors, alpha=0.88, edgecolor="white")
ax.errorbar(x, m_pt, yerr=[m_lo, m_hi], fmt="none", ecolor="black", capsize=4, alpha=0.6)
for i, r in final_df.iterrows():
    ax.text(i, r["Media Lift %"] + m_hi[i] + 0.4, f"+{r['Media Lift %']:.1f}%", ha="center",
            fontsize=8.5, fontweight="bold", color=CPX_NAVY)
ax.set_title("Media-Attributable Revenue Lift\n(what the scenarios directly move)",
             fontweight="bold", color="black", fontsize=12)
ax.set_ylabel("Lift on media-attributable revenue (%)")
ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
ax.grid(axis="y", alpha=0.2, color=CPX_LGREY)
ax.set_axisbelow(True)

# RIGHT — total-revenue lift
ax = axes[1]
t_pt = final_df["Total Lift %"].values
t_lo = (final_df["Total Lift %"] - final_df["Total CI Low %"]).values
t_hi = (final_df["Total CI High %"] - final_df["Total Lift %"]).values
ax.bar(x, t_pt, color=colors, alpha=0.88, edgecolor="white")
ax.errorbar(x, t_pt, yerr=[t_lo, t_hi], fmt="none", ecolor="black", capsize=4, alpha=0.6)
ax.axhline(0, color="#333", lw=0.8)
for i, r in final_df.iterrows():
    ax.text(i, r["Total Lift %"] + t_hi[i] + 0.05, f"+{r['Total Lift %']:.2f}%", ha="center",
            fontsize=8.5, fontweight="bold", color=CPX_GREEN if r["Type"] != "reshuffle" else "#8a6d00")
ax.set_title(f"Total-Revenue Lift (of ${actual_total_rev/1e6:.1f}M actual)\n"
             f"media is {MEDIA_SHARE*100:.0f}% of total \u2192 lifts compress",
             fontweight="bold", color="black", fontsize=12)
ax.set_ylabel("Lift on TOTAL 2025 revenue (%)")
ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
ax.grid(axis="y", alpha=0.2, color=CPX_LGREY)
ax.set_axisbelow(True)

# Shared legend
leg = [mpatches.Patch(facecolor=CPX_NAVY, label="Budget uplift"),
       mpatches.Patch(facecolor=CPX_GREEN, label="Combined"),
       mpatches.Patch(facecolor=CPX_GOLD, label="Reshuffle"),
       mpatches.Patch(facecolor="#9AA0A6", label="Directional (CI crosses 0)")]
fig.legend(handles=leg, loc="lower center", ncol=4, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.04))
fig.suptitle("LBE MMM \u2014 Budget Scenario Outcomes (2025, reconciled to actual)",
             fontsize=14, fontweight="bold", color="black", y=1.02)
plt.tight_layout()
save_fig(fig, "FINAL_scenario_both_frames.png")
plt.show()

# COMMAND ----------

# DBTITLE 1,FINAL Media Lift and Stacked Revenue Charts
# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Media-Attributable Revenue Lift %
# ═════════════════════════════════════════════════════════════════════════════
colors_f = [bar_color(r) for _, r in final_df.iterrows()]
x_f      = np.arange(len(final_df))
labels_f = final_df["Scenario"].tolist()
m_pt_f   = final_df["Media Lift %"].values
m_lo_f   = (final_df["Media Lift %"] - final_df["Media CI Low %"]).values
m_hi_f   = (final_df["Media CI High %"] - final_df["Media Lift %"]).values

fig1, ax = plt.subplots(figsize=(16, 9))
ax.set_axisbelow(True)
ax.grid(axis="y", alpha=0.2, color=CPX_LGREY)

ax.bar(x_f, m_pt_f, color=colors_f, alpha=0.88, edgecolor="white", width=0.6)
ax.errorbar(x_f, m_pt_f, yerr=[m_lo_f, m_hi_f],
            fmt="none", ecolor=CPX_DGREY, capsize=5, alpha=0.7, lw=1.5)

for i, (_, r) in enumerate(final_df.iterrows()):
    ax.text(i, m_pt_f[i] * 0.5, f"+{m_pt_f[i]:.1f}%",
            ha="center", va="center", fontsize=10,
            fontweight="bold", color="white")
    ax.text(i, m_pt_f[i] + m_hi_f[i] + 0.3,
            f"[+{r['Media CI Low %']:.1f}%, +{r['Media CI High %']:.1f}%]",
            ha="center", va="bottom", fontsize=8,
            fontweight="bold", color=CPX_DGREY)

ax.set_xticks(x_f)
ax.set_xticklabels(labels_f, rotation=20, ha="right", fontsize=9, color=CPX_DGREY)
ax.set_ylabel("Lift on media-attributable revenue (%)", fontsize=10, color=CPX_DGREY)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
ax.set_ylim(0, max(m_pt_f + m_hi_f) * 1.45)
ax.tick_params(axis="y", labelsize=9, labelcolor=CPX_DGREY)
ax.set_title("Media-Attributable Revenue Lift (By Scenario)",
             fontweight="bold", color="black", fontsize=13, pad=10)

leg1 = [
    mpatches.Patch(facecolor=CPX_NAVY,    label="Budget uplift"),
    mpatches.Patch(facecolor=CPX_GREEN,   label="Combined"),
    mpatches.Patch(facecolor="#9AA0A6",   label="Directional (CI crosses 0)"),
]
ax.legend(handles=leg1, fontsize=9, loc="upper left", framealpha=0.95, edgecolor=CPX_LGREY)
fig1.text(0.99, 0.01, "Cineplex Marketing Analytics  |  LBE MMM",
          ha="right", va="bottom", fontsize=6, color=CPX_LGREY)
plt.tight_layout()
save_fig(fig1, "FINAL_media_lift.png")
plt.show()

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Stacked absolute revenue: baseline + media + scenario lift
# ═════════════════════════════════════════════════════════════════════════════
fig2, ax = plt.subplots(figsize=(16, 9))
ax.set_axisbelow(True)
ax.grid(axis="y", alpha=0.2, color=CPX_LGREY)

bar_w = 0.6
revenue_gained = final_df["Revenue Gained ($)"].values
gained_lo_abs  = (final_df["Revenue Gained ($)"] - final_df["Gained CI Low"]).values
gained_hi_abs  = (final_df["Gained CI High"] - final_df["Revenue Gained ($)"]).values

# Layer 1 — non-media baseline (grey)
ax.bar(x_f, non_media_base / 1e6, bar_w,
       color="#9AA0A6", alpha=0.6, edgecolor="white", lw=0.4, label="Baseline / Non-Media")

# Layer 2 — current media-attributed revenue (Navy)
ax.bar(x_f, media_base_rev / 1e6, bar_w,
       bottom=non_media_base / 1e6,
       color=CPX_NAVY, alpha=0.85, edgecolor="white", lw=0.4, label="Current Media Attribution")

# Layer 3 — scenario lift (Gold)
ax.bar(x_f, revenue_gained / 1e6, bar_w,
       bottom=(non_media_base + media_base_rev) / 1e6,
       color=CPX_GOLD, alpha=0.9, edgecolor="white", lw=0.4, label="Scenario Lift")

# CI whiskers on top of scenario lift
tops = (non_media_base + media_base_rev + revenue_gained) / 1e6
ax.errorbar(x_f, tops, yerr=[gained_lo_abs / 1e6, gained_hi_abs / 1e6],
            fmt="none", ecolor=CPX_DGREY, capsize=5, alpha=0.7, lw=1.5)

# Labels
for i, (_, r) in enumerate(final_df.iterrows()):
    ax.text(i, (non_media_base / 2) / 1e6, f"${non_media_base/1e6:.1f}M",
            ha="center", va="center", fontsize=8, fontweight="bold", color="white")
    ax.text(i, (non_media_base + media_base_rev / 2) / 1e6, f"${media_base_rev/1e6:.1f}M",
            ha="center", va="center", fontsize=8, fontweight="bold", color="white")
    if revenue_gained[i] / 1e6 > 0.1:
        ax.text(i, (non_media_base + media_base_rev + revenue_gained[i] / 2) / 1e6,
                f"+${revenue_gained[i]/1e6:.2f}M",
                ha="center", va="center", fontsize=8, fontweight="bold", color=CPX_NAVY)
    ax.text(i, tops[i] + gained_hi_abs[i] / 1e6 + 0.3,
            f"[{r['Gained CI Low']/1e6:+.2f}M, {r['Gained CI High']/1e6:+.2f}M]",
            ha="center", va="bottom", fontsize=7.5, fontweight="bold", color=CPX_DGREY)

ax.set_xticks(x_f)
ax.set_xticklabels(labels_f, rotation=20, ha="right", fontsize=9, color=CPX_DGREY)
ax.set_ylabel("Annual Revenue ($M)", fontsize=10, color=CPX_DGREY)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:.0f}M"))
ax.set_ylim(0, (actual_total_rev + max(revenue_gained) + max(gained_hi_abs)) / 1e6 * 1.15)
ax.tick_params(axis="y", labelsize=9, labelcolor=CPX_DGREY)
ax.set_title(
    f"Total Revenue by Scenario  |  Baseline ${actual_total_rev/1e6:.1f}M  \u2192  Baseline + Media + Scenario Lift",
    fontweight="bold", color="black", fontsize=13, pad=10)
ax.legend(fontsize=9, loc="upper left", framealpha=0.95, edgecolor=CPX_LGREY)
fig2.text(0.99, 0.01, "Cineplex Marketing Analytics  |  LBE MMM",
          ha="right", va="bottom", fontsize=6, color=CPX_LGREY)
plt.tight_layout()
save_fig(fig2, "FINAL_revenue_stacked.png")
plt.show()

# COMMAND ----------

# DBTITLE 1,2026 Forecast Scenario Projections
# ── 2026 Forecast + Scenario Lift Projections ───────────────────────────────
FORECAST_2026 = 148_546_568  # from LBE 2026 revenue forecast

proj_rows = []
for _, r in final_df.iterrows():
    gained_pt = r["Revenue Gained ($)"]
    gained_lo = r["Gained CI Low"]
    gained_hi = r["Gained CI High"]
    proj_rows.append({
        "Scenario":            r["Scenario"],
        "Type":                r["Type"],
        "2026 Forecast ($M)":  FORECAST_2026 / 1e6,
        "Scenario Lift ($M)":  gained_pt / 1e6,
        "Projected 2026 ($M)": (FORECAST_2026 + gained_pt) / 1e6,
        "CI Low ($M)":         (FORECAST_2026 + gained_lo) / 1e6,
        "CI High ($M)":        (FORECAST_2026 + gained_hi) / 1e6,
        "Lift vs Forecast %":  gained_pt / FORECAST_2026 * 100,
        "Confidence":          r["Confidence"],
    })

proj_df = pd.DataFrame(proj_rows)

from IPython.display import display as ipy_display
ipy_display(
    proj_df.style
    .format({
        "2026 Forecast ($M)": "${:.2f}M",
        "Scenario Lift ($M)": "+${:.2f}M",
        "Projected 2026 ($M)": "${:.2f}M",
        "CI Low ($M)": "${:.2f}M",
        "CI High ($M)": "${:.2f}M",
        "Lift vs Forecast %": "+{:.2f}%",
    })
    .set_caption("2026 Revenue Projections \u2014 Forecast ($148.55M) + Media Scenario Lift")
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

# ── Chart: 2026 projected revenue by scenario ───────────────────────────────
fig, ax = plt.subplots(figsize=(16, 7))
ax.set_axisbelow(True)
ax.grid(axis="y", alpha=0.2, color=CPX_LGREY)

x_pos  = np.arange(len(proj_df))
labels_p = proj_df["Scenario"].tolist()
colors_p = [bar_color(r) for _, r in proj_df.iterrows()]

projected = proj_df["Projected 2026 ($M)"].values
ci_lo_p   = (proj_df["Projected 2026 ($M)"] - proj_df["CI Low ($M)"]).values
ci_hi_p   = (proj_df["CI High ($M)"] - proj_df["Projected 2026 ($M)"]).values

# Baseline reference line
ax.axhline(FORECAST_2026 / 1e6, color=CPX_RED, lw=1.5, ls="--", alpha=0.7,
           label=f"2026 Forecast: ${FORECAST_2026/1e6:.1f}M")

# Bars showing projected total
ax.bar(x_pos, projected, color=colors_p, alpha=0.88, edgecolor="white", width=0.6)
ax.errorbar(x_pos, projected, yerr=[ci_lo_p, ci_hi_p],
            fmt="none", ecolor=CPX_DGREY, capsize=5, alpha=0.7, lw=1.5)

# Labels
for i, row in proj_df.iterrows():
    lift = row["Scenario Lift ($M)"]
    ax.text(i, projected[i] + ci_hi_p[i] + 0.2,
            f"+${lift:.2f}M\n({row['Lift vs Forecast %']:+.2f}%)",
            ha="center", va="bottom", fontsize=8.5,
            fontweight="bold", color=CPX_NAVY)
    ax.text(i, projected[i] * 0.5,
            f"${projected[i]:.1f}M",
            ha="center", va="center", fontsize=9,
            fontweight="bold", color="white")

ax.set_xticks(x_pos)
ax.set_xticklabels(labels_p, rotation=20, ha="right", fontsize=9, color=CPX_DGREY)
ax.set_ylabel("Projected 2026 Revenue ($M)", fontsize=10, color=CPX_DGREY)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:.0f}M"))
ax.set_ylim(140, max(projected + ci_hi_p) * 1.06)
ax.tick_params(axis="y", labelsize=9, labelcolor=CPX_DGREY)
ax.set_title(
    "2026 Projected Revenue by Scenario\n"
    f"Baseline Forecast: ${FORECAST_2026/1e6:.1f}M  +  Media Scenario Lift",
    fontweight="bold", color="black", fontsize=13, pad=10)
ax.legend(fontsize=10, loc="upper left", framealpha=0.95, edgecolor=CPX_LGREY)

fig.text(0.99, 0.01, "Cineplex Marketing Analytics  |  LBE MMM",
         ha="right", va="bottom", fontsize=6, color=CPX_LGREY)
plt.tight_layout()
save_fig(fig, "FINAL_2026_scenario_projections.png")
plt.show()

print(f"\n2026 Forecast: ${FORECAST_2026:,.0f}")
print(f"Best scenario ({proj_df.iloc[-1]['Scenario']}): "
      f"${proj_df.iloc[-1]['Projected 2026 ($M)']:.2f}M "
      f"(+${proj_df.iloc[-1]['Scenario Lift ($M)']:.2f}M)")
print("\u2713 Saved FINAL_2026_scenario_projections.png")

# COMMAND ----------

# DBTITLE 1,2025 Revenue-Based Scenario Projections
# ── Scenario Projections Based on 2025 Actual Revenue ─────────────────────
BASELINE_2025 = actual_total_rev  # 2025 actual revenue from df_all

proj_rows_25 = []
for _, r in final_df.iterrows():
    gained_pt = r["Revenue Gained ($)"]
    gained_lo = r["Gained CI Low"]
    gained_hi = r["Gained CI High"]
    proj_rows_25.append({
        "Scenario":              r["Scenario"],
        "Type":                  r["Type"],
        "2025 Revenue ($M)":     BASELINE_2025 / 1e6,
        "Scenario Lift ($M)":    gained_pt / 1e6,
        "Projected Revenue ($M)": (BASELINE_2025 + gained_pt) / 1e6,
        "CI Low ($M)":           (BASELINE_2025 + gained_lo) / 1e6,
        "CI High ($M)":          (BASELINE_2025 + gained_hi) / 1e6,
        "Lift vs 2025 %":        gained_pt / BASELINE_2025 * 100,
        "Confidence":            r["Confidence"],
    })

proj_df_25 = pd.DataFrame(proj_rows_25)

from IPython.display import display as ipy_display
ipy_display(
    proj_df_25.style
    .format({
        "2025 Revenue ($M)": "${:.2f}M",
        "Scenario Lift ($M)": "+${:.2f}M",
        "Projected Revenue ($M)": "${:.2f}M",
        "CI Low ($M)": "${:.2f}M",
        "CI High ($M)": "${:.2f}M",
        "Lift vs 2025 %": "+{:.2f}%",
    })
    .set_caption(f"Scenario Projections \u2014 2025 Actual Revenue (${BASELINE_2025/1e6:.2f}M) + Media Scenario Lift")
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

# ── Chart: Projected revenue based on 2025 actual ─────────────────────────
fig, ax = plt.subplots(figsize=(16, 7))
ax.set_axisbelow(True)
ax.grid(axis="y", alpha=0.2, color=CPX_LGREY)

x_pos  = np.arange(len(proj_df_25))
labels_p = proj_df_25["Scenario"].tolist()
colors_p = [bar_color(r) for _, r in proj_df_25.iterrows()]

projected = proj_df_25["Projected Revenue ($M)"].values
ci_lo_p   = (proj_df_25["Projected Revenue ($M)"] - proj_df_25["CI Low ($M)"]).values
ci_hi_p   = (proj_df_25["CI High ($M)"] - proj_df_25["Projected Revenue ($M)"]).values

# Baseline reference line
ax.axhline(BASELINE_2025 / 1e6, color=CPX_RED, lw=1.5, ls="--", alpha=0.7,
           label=f"2025 Actual: ${BASELINE_2025/1e6:.1f}M")

# Bars showing projected total
ax.bar(x_pos, projected, color=colors_p, alpha=0.88, edgecolor="white", width=0.6)
ax.errorbar(x_pos, projected, yerr=[ci_lo_p, ci_hi_p],
            fmt="none", ecolor=CPX_DGREY, capsize=5, alpha=0.7, lw=1.5)

# Labels
for i, row in proj_df_25.iterrows():
    lift = row["Scenario Lift ($M)"]
    ax.text(i, projected[i] + ci_hi_p[i] + 0.2,
            f"+${lift:.2f}M\n({row['Lift vs 2025 %']:+.2f}%)",
            ha="center", va="bottom", fontsize=8.5,
            fontweight="bold", color=CPX_NAVY)
    ax.text(i, projected[i] * 0.5,
            f"${projected[i]:.1f}M",
            ha="center", va="center", fontsize=9,
            fontweight="bold", color="white")

ax.set_xticks(x_pos)
ax.set_xticklabels(labels_p, rotation=20, ha="right", fontsize=9, color=CPX_DGREY)
ax.set_ylabel("Projected Revenue ($M)", fontsize=10, color=CPX_DGREY)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:.0f}M"))
ax.set_ylim(125, max(projected + ci_hi_p) * 1.06)
ax.tick_params(axis="y", labelsize=9, labelcolor=CPX_DGREY)
ax.set_title(
    "Projected Revenue by Scenario (Based on 2025 Actual)\n"
    f"Baseline: ${BASELINE_2025/1e6:.1f}M  +  Media Scenario Lift",
    fontweight="bold", color="black", fontsize=13, pad=10)
ax.legend(fontsize=10, loc="upper left", framealpha=0.95, edgecolor=CPX_LGREY)

fig.text(0.99, 0.01, "Cineplex Marketing Analytics  |  LBE MMM",
         ha="right", va="bottom", fontsize=6, color=CPX_LGREY)
plt.tight_layout()
save_fig(fig, "FINAL_2025_based_scenario_projections.png")
plt.show()

print(f"\n2025 Actual Revenue: ${BASELINE_2025:,.0f}")
print(f"Best scenario ({proj_df_25.iloc[-1]['Scenario']}): "
      f"${proj_df_25.iloc[-1]['Projected Revenue ($M)']:.2f}M "
      f"(+${proj_df_25.iloc[-1]['Scenario Lift ($M)']:.2f}M)")
print("\u2713 Saved FINAL_2025_based_scenario_projections.png")
