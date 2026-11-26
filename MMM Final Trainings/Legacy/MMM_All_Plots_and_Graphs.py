# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# DBTITLE 1,MMM Plots & Graphs — EXH + LBE (All Charts)
# MAGIC %md
# MAGIC # MMM Plots & Graphs — EXH + LBE (All Charts)
# MAGIC **Cineplex Marketing Analytics | Visualization-Only Notebook**
# MAGIC
# MAGIC This notebook generates ALL presentation and diagnostic charts for both the EXH and LBE Marketing Mix Models. It loads pre-computed artifacts only — no model training.
# MAGIC
# MAGIC ### EXH Charts
# MAGIC - Annual media spend by funnel & LOB
# MAGIC - Attendance breakdown & decomposition (2023–2026F)
# MAGIC - Model features & correlations
# MAGIC - Tentpole & content tier analysis
# MAGIC - Per-LOB spend & attendance contribution
# MAGIC - Flighting analysis (channel & LOB level)
# MAGIC - Diagnostics: D1 (prior/posterior), D3–D6 (validation), D7 (adstock), B1/B6 (decomposition), B10 (response curves)
# MAGIC
# MAGIC ### LBE Charts
# MAGIC - Annual spend by channel & funnel
# MAGIC - Revenue breakdown & calendar events
# MAGIC - Diminishing returns & marginal curves
# MAGIC - Flighting analysis
# MAGIC - Diagnostics: Actual vs predicted, prior/posterior, adstock/calendar effects, annual attribution
# MAGIC

# COMMAND ----------

# DBTITLE 1,Install Dependencies
# MAGIC
# MAGIC %pip install arviz netcdf4 h5netcdf scipy openpyxl xarray --quiet
# MAGIC

# COMMAND ----------

# DBTITLE 1,Imports

import os
import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import arviz as az
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patheffects as pe
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import beta as beta_dist, halfnorm, norm
import seaborn as sns


# COMMAND ----------

# DBTITLE 1,Paths & Brand Palette

# ── Artifact Paths ──────────────────────────────────────────────────────────────
EXH_PATH      = "/Volumes/data_exploration_hub/marketing_hub/model_data/Final_Data/MMM_EXH_Outcomes_v2/"
EXH_DECOMP    = "/Volumes/data_exploration_hub/marketing_hub/model_data/Final_Data/MMM_EXH_LOB_Decomp_v2/"
LBE_PATH      = "/Volumes/data_exploration_hub/marketing_hub/model_data/Final_Data/MMM_LBE_v2/"
EXH_DATA_PATH = "/Volumes/data_exploration_hub/marketing_hub/model_data/Final_Data/EXH_ModelData.xlsx"
LBE_DATA_PATH = "/Volumes/data_exploration_hub/marketing_hub/model_data/Final_Data/lbe_model_input.csv"

# ── Cineplex Brand Palette ─────────────────────────────────────────────────────
CPX_NAVY  = "#002B5C"
CPX_RED   = "#E31837"
CPX_GOLD  = "#F5A800"
CPX_LGREY = "#F2F2F2"
CPX_DGREY = "#4A4A4A"
CPX_GREEN = "#2E7D32"
CPX_MGREY = "#8A8A8A"
WHITE     = "#FFFFFF"

FUNNEL_COLORS = {
    "Awareness"    : CPX_NAVY,
    "Consideration": CPX_RED,
    "Conversion"   : CPX_GOLD,
}

YEAR_COLORS = {2023: CPX_NAVY, 2024: CPX_RED, 2025: CPX_GOLD}

# ── Aspect Ratio Presets ──────────────────────────────────────────────────────
RATIOS = {
    "16:9"   : (16, 9),
    "8:9"    : (8, 9),
    "16:4.5" : (16, 4.5),
    "4:3"    : (12, 9),
    "1:1"    : (9, 9),
    "3:4"    : (9, 12),
}

def get_figsize(ratio="16:9"):
    if ratio in RATIOS:
        return RATIOS[ratio]
    raise ValueError(f"Unknown ratio '{ratio}'. Available: {list(RATIOS.keys())}")

# ── Matplotlib Style ──────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": WHITE, "figure.dpi": 150,
    "axes.facecolor": WHITE, "axes.edgecolor": CPX_DGREY,
    "axes.linewidth": 1.2, "axes.grid": False,
    "axes.spines.top": True, "axes.spines.right": True,
    "axes.spines.left": True, "axes.spines.bottom": True,
    "axes.titlesize": 16, "axes.titleweight": "bold",
    "axes.titlecolor": "black", "axes.labelsize": 12,
    "axes.labelweight": "bold", "axes.labelcolor": CPX_DGREY,
    "xtick.labelsize": 11, "ytick.labelsize": 11,
    "xtick.color": CPX_DGREY, "ytick.color": CPX_DGREY,
    "xtick.major.width": 1.2, "ytick.major.width": 1.2,
    "xtick.labelcolor": CPX_DGREY, "ytick.labelcolor": CPX_DGREY,
    "legend.fontsize": 10, "legend.frameon": False,
    "font.family": "sans-serif", "font.size": 11,
    "font.weight": "bold", "axes.titlelocation": "center",
})

def save_fig(fig, filename, path=EXH_PATH):
    """Save figure to specified path."""
    out = os.path.join(path, filename)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"  Saved → {out}")

print("✓ Paths & palette configured")


# COMMAND ----------

# DBTITLE 1,Load EXH Artifacts

# ━━ LOAD EXH ARTIFACTS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("Loading EXH artifacts...")

trace = az.from_netcdf(f"{EXH_PATH}trace_iter2_mixed_hier.nc")
post  = trace.posterior

with open(f"{EXH_PATH}scaler_y.pkl", "rb") as f:
    scaler_y = pickle.load(f)
with open(f"{EXH_PATH}scaler_X.pkl", "rb") as f:
    scaler_X = pickle.load(f)
with open(f"{EXH_PATH}spend_train_max_iter2.pkl", "rb") as f:
    spend_train_max_iter2 = pickle.load(f)
with open(f"{EXH_PATH}model_meta.pkl", "rb") as f:
    model_meta = pickle.load(f)
with open(f"{EXH_PATH}response_params.pkl", "rb") as f:
    response_params = pickle.load(f)

ACTIVE_COMBOS     = model_meta["ACTIVE_COMBOS"]
FUNNEL_COMBOS     = model_meta["FUNNEL_COMBOS"]
LOB_GROUP_MAP     = model_meta["LOB_GROUP_MAP"]
baseline_features = model_meta["baseline_features"]
target_col        = model_meta["target_col"]
n_train           = model_meta["n_train"]
n_val             = model_meta["n_val"]
n_test            = model_meta["n_test"]

df_full = pd.read_parquet(f"{EXH_PATH}df_full.parquet")
df_full["year"] = df_full["week_date"].dt.year

# Also load the original Excel for spend-level charts
df_exh = pd.read_excel(EXH_DATA_PATH, sheet_name=0)
df_exh["year"] = pd.to_datetime(df_exh["week_date"]).dt.year

# Decomp results
if os.path.exists(f"{EXH_DECOMP}decomp_lob_results.pkl"):
    with open(f"{EXH_DECOMP}decomp_lob_results.pkl", "rb") as f:
        decomp_df = pickle.load(f)
    print(f"  ✓ LOB decomposition loaded ({len(decomp_df)} entries)")
else:
    decomp_df = None
    print("  ⚠ LOB decomposition not found")

print(f"  ✓ EXH artifacts loaded — {len(df_full)} weeks, {len(ACTIVE_COMBOS)} active combos")
print(f"  ✓ Trace: {post.dims['chain']} chains × {post.dims['draw']} draws")


# COMMAND ----------

# DBTITLE 1,Load LBE Artifacts

# # ━━ LOAD LBE ARTIFACTS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# print("Loading LBE artifacts...")

# with open(f"{LBE_PATH}lbe_mmm_results.pkl", "rb") as f:
#     results = pickle.load(f)

# idata       = results["idata"]
# draws       = results["draws"]
# bc          = results["bc"]
# n_draws     = int(results["n_draws"])
# tr          = results["tr"]
# va          = results["va"]
# df_lbe_train = results["df_train"].copy()
# df_lbe_val   = results["df_val"].copy()
# y_mean      = float(results["y_mean"])
# y_std       = float(results["y_std"])
# aw_max      = float(results["aw_max"])
# co_max      = float(results["co_max"])
# t_max       = float(results["t_max"])
# temp_mu     = float(results["temp_mu"])
# temp_sd     = float(results["temp_sd"])
# cal_cols    = list(results["CAL_COLS"])

# CAL_MU      = np.asarray(results["CAL_MU"]).ravel()
# CAL_SIGMA   = np.asarray(results["CAL_SIGMA"]).ravel()

# df_lbe_all = pd.concat([df_lbe_train, df_lbe_val]).reset_index(drop=True)
# df_lbe_all["year"] = df_lbe_all["week_date"].dt.year

# # Also load raw input CSV for spend-level charts
# df_lbe_raw = pd.read_csv(LBE_DATA_PATH, parse_dates=["week_date"])
# df_lbe_raw["year"] = df_lbe_raw["week_date"].dt.year

# # Prediction helpers
# def adstock_np(x, decay, max_lag=10):
#     w = decay ** np.arange(max_lag)
#     w = w / w.sum()
#     out = np.zeros(len(x), dtype=float)
#     for lag in range(max_lag):
#         out[lag:] += w[lag] * x[:len(x) - lag]
#     return out

# def predict_revenue(data, sample_size=1000, seed=42):
#     sample_size = min(sample_size, n_draws)
#     rng = np.random.default_rng(seed)
#     idx = rng.choice(n_draws, sample_size, replace=False)
#     preds = np.zeros((sample_size, len(data["aw"])), dtype=float)
#     for j, i in enumerate(idx):
#         aw_ads = adstock_np(np.asarray(data["aw"]), float(draws["aw_decay"][i]))
#         co_ads = adstock_np(np.asarray(data["co"]), float(draws["co_decay"][i]))
#         aw_sat = np.log1p(aw_ads * float(draws["aw_scale"][i]))
#         co_sat = np.log1p(co_ads * float(draws["co_scale"][i]))
#         cal_eff = np.asarray(data["cal"]) @ bc[i]
#         mu = (float(draws["intercept"][i])
#               + float(draws["beta_aw"][i]) * aw_sat
#               + float(draws["beta_co"][i]) * co_sat
#               + float(draws["beta_trend"][i]) * data["t"]
#               + float(draws["beta_temp"][i]) * data["temp"]
#               + cal_eff)
#         preds[j] = np.exp(mu * y_std + y_mean)
#     return preds

# metrics = results.get("metrics", {})
# print(f"  ✓ LBE loaded: {n_draws} draws, {len(df_lbe_all)} weeks")
# print(f"  ✓ Val R²: {metrics.get('val_r2', 'N/A'):.3f}, MAPE: {metrics.get('val_mape', 'N/A'):.1f}%")


# COMMAND ----------

# DBTITLE 1,EXH Section Header
# MAGIC
# MAGIC %md
# MAGIC ---
# MAGIC ## EXH Presentation Charts
# MAGIC

# COMMAND ----------

# DBTITLE 1,EXH Annual Media Spend by Funnel
# ── EXH ANNUAL MEDIA SPEND BY FUNNEL (2023–2025) ─────────────────────────────
fig, ax = plt.subplots(figsize=(11, 9))

# Data
years         = ['2023', '2024', '2025']
awareness     = [0.81, 0.94, 1.05]
consideration = [0.98, 0.87, 1.08]
conversion    = [1.96, 1.89, 1.75]

x     = np.arange(len(years))
width = 0.55

# Stacked bars
b1 = ax.bar(x, awareness,     width, label="Awareness",     color=CPX_NAVY)
b2 = ax.bar(x, consideration, width, label="Consideration", color=CPX_RED,
            bottom=awareness)
b3 = ax.bar(x, conversion,    width, label="Conversion",    color=CPX_GOLD,
            bottom=[a + c for a, c in zip(awareness, consideration)])

# ── Segment labels ────────────────────────────────────────────────────────────
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
                fontsize=10, fontweight="bold", color=WHITE
            )

bottom_c = [a + c for a, c in zip(awareness, consideration)]

label_segments(b1, [0, 0, 0])
label_segments(b2, awareness)
label_segments(b3, bottom_c)

# ── Total labels on top ───────────────────────────────────────────────────────
totals = [a + c + cv for a, c, cv in zip(awareness, consideration, conversion)]
for i, total in enumerate(totals):
    ax.text(
        i, total + 0.06,
        f"${total:.2f}M",
        ha="center", va="bottom",
        fontsize=12, fontweight="bold", color=CPX_DGREY
    )

# ── Styling ───────────────────────────────────────────────────────────────────
ax.set_xticks(x)
ax.set_xticklabels(years, fontsize=12)
ax.set_ylabel("Annual Media Spend ($M)")
ax.set_xlabel("Year")
ax.set_ylim(0, 5.5)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:.1f}M"))
ax.set_title("EXH Annual Media Spend by Funnel (2023–2025)")
ax.legend(loc="upper right")

plt.tight_layout()
plt.show()


# COMMAND ----------

# DBTITLE 1,EXH Spend by Original LOB Per Year
# ── EXH SPEND BY ORIGINAL LOB — STACKED BY FUNNEL — ONE PLOT PER YEAR ─────────
import pandas as pd
import numpy as np
import matplotlib.ticker as mticker

# ── Load Data ─────────────────────────────────────────────────────────────────
FILE_PATH = "/Volumes/data_exploration_hub/marketing_hub/model_data/Final_Data/EXH_ModelData.xlsx"

df = pd.read_excel(FILE_PATH, sheet_name=0)
df["year"] = pd.to_datetime(df["week_date"]).dt.year

# ── Parse Media Columns ───────────────────────────────────────────────────────
funnels    = ["Awareness", "Consideration", "Conversion"]
media_cols = [c for c in df.columns if any(c.startswith(f + "_") for f in funnels)]

def parse_col(col):
    for funnel in funnels:
        if col.startswith(funnel + "_"):
            rest    = col[len(funnel) + 1:]
            channel = rest.rsplit("_", 1)[-1]
            lob     = rest.rsplit("_", 1)[0]
            return funnel, lob, channel
    return None, None, None

meta = pd.DataFrame(
    [parse_col(c) + (c,) for c in media_cols],
    columns=["Funnel", "LOB", "Channel", "col_name"]
).dropna(subset=["Funnel"])

# ── Aggregate Spend ───────────────────────────────────────────────────────────
rows = []
for _, row in meta.iterrows():
    for year in [2023, 2024, 2025]:
        spend = df.loc[df["year"] == year, row["col_name"]].sum()
        rows.append({
            "Funnel" : row["Funnel"],
            "LOB"    : row["LOB"],
            "Year"   : year,
            "Spend_M": round(spend / 1e6, 3)
        })

spend_df = (
    pd.DataFrame(rows)
    .groupby(["Funnel", "LOB", "Year"])["Spend_M"]
    .sum()
    .reset_index()
)
spend_df = spend_df[spend_df["Spend_M"] > 0.001].copy()

# ── Funnel Colours ────────────────────────────────────────────────────────────
funnel_colors = {
    "Awareness"    : CPX_NAVY,
    "Consideration": CPX_RED,
    "Conversion"   : CPX_GOLD,
}

# ── One Plot Per Year ─────────────────────────────────────────────────────────
for year in [2023, 2024, 2025]:
    ydata = spend_df[spend_df["Year"] == year].copy()

    lob_totals = ydata.groupby("LOB")["Spend_M"].sum().sort_values(ascending=False)
    lobs       = lob_totals.index.tolist()
    x          = np.arange(len(lobs))

    fig, ax = plt.subplots(figsize=get_figsize("16:9"))
    fig.patch.set_facecolor(WHITE)

    bottoms = np.zeros(len(lobs))

    for funnel in funnels:
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
            edgecolor=WHITE,
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
                    fontsize=8, fontweight="bold", color=WHITE
                )
        bottoms += np.array(heights)

    # Total labels on top
    for i, total in enumerate(bottoms):
        if total > 0:
            ax.text(
                x[i], total + 0.01,
                f"${total:.2f}M",
                ha="center", va="bottom",
                fontsize=9, fontweight="bold", color=CPX_DGREY
            )

    ax.set_xticks(x)
    ax.set_xticklabels(lobs, rotation=35, ha="right", fontsize=10)
    ax.set_ylabel("Annual Spend ($M)")
    ax.set_xlabel("Line of Business")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:.2f}M"))
    ax.set_title(f"EXH Annual Media Spend by LOB & Funnel — {year}", loc="center")
    ax.legend(
        title="Funnel", fontsize=9, title_fontsize=10,
        loc="upper right"
    )

    plt.tight_layout()
    plt.show()
    print(f"✓ Saved: EXH_Spend_LOB_Funnel_{year}.png")


# COMMAND ----------

# DBTITLE 1,EXH Spend by LOB & Funnel — Data Tables
# ── DATA TABLES: EXH Spend by LOB & Funnel — One Table Per Year ──────────────
# Values in $10K units (e.g., 31.5 = $315,000)
for year in [2023, 2024, 2025]:
    pivot = (
        spend_df[spend_df["Year"] == year]
        .pivot_table(index="LOB", columns="Funnel", values="Spend_M", aggfunc="sum", fill_value=0)
        .reindex(columns=["Awareness", "Consideration", "Conversion"])
        .fillna(0)
    )
    # Convert from $M to $10K (multiply by 100)
    pivot = pivot * 100
    pivot["Total"] = pivot.sum(axis=1)
    pivot = pivot.sort_values("Total", ascending=False)
    pivot = pivot.reset_index()
    pivot.columns.name = None

    # Round to 2 decimal places
    num_cols = ["Awareness", "Consideration", "Conversion", "Total"]
    pivot[num_cols] = pivot[num_cols].round(2)

    print(f"\n{'═'*60}")
    print(f"  EXH Annual Media Spend by LOB & Funnel — {year}")
    print(f"  (values in $10K)")
    print(f"{'═'*60}")
    display(pivot)


# COMMAND ----------

# DBTITLE 1,EXH Attendance Breakdown by Year
# ── EXH ATTENDANCE BREAKDOWN BY YEAR ─────────────────────────────────────────
# (Uses df, numpy, pandas, matplotlib from earlier cells)

# ── Aggregate Annual Attendance ───────────────────────────────────────────────
att_df = (
    df.groupby("year")["total_weekly_attendance"]
    .sum()
    .reset_index()
    .rename(columns={"total_weekly_attendance": "Attendance"})
)
att_df = att_df[att_df["year"].isin([2023, 2024, 2025])].copy()
att_df["Attendance_M"] = att_df["Attendance"] / 1e6

# Add 2026 forecast (hard-coded)
forecast_row = pd.DataFrame([{"year": 2026, "Attendance": 46_400_000, "Attendance_M": 46.4}])
att_df = pd.concat([att_df, forecast_row], ignore_index=True)
att_df["YoY_pct"] = att_df["Attendance_M"].pct_change() * 100

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=get_figsize("16:9"))
fig.patch.set_facecolor(WHITE)

x     = np.arange(len(att_df))
width = 0.5

bars = ax.bar(
    x,
    att_df["Attendance_M"],
    width=width,
    color="black",
    edgecolor=WHITE,
    linewidth=0.5
)

# Value labels on top
for i, (bar, row) in enumerate(zip(bars, att_df.itertuples())):
    h = bar.get_height()
    # Attendance value label
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        h + 0.3,
        f"{h:.2f}M",
        ha="center", va="bottom",
        fontsize=12, fontweight="bold", color=CPX_DGREY
    )
    # YoY change label (stacked above the value)
    if i > 0:
        yoy = row.YoY_pct
        color = CPX_RED if yoy < 0 else "black"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + 1.8,
            f"{yoy:+.1f}% YoY",
            ha="center", va="bottom",
            fontsize=10, fontweight="bold", color=color
        )

ax.set_xticks(x)
ax.set_xticklabels(att_df["year"].astype(str), fontsize=12)
ax.set_ylabel("Total Annual Attendance (M)")
ax.set_xlabel("Year")
ax.set_ylim(0, att_df["Attendance_M"].max() * 1.25)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}M"))
ax.set_title("EXH Total Annual Attendance (2023–2026F)", loc="center")

plt.tight_layout()
plt.show()

# ── Table ─────────────────────────────────────────────────────────────────────
table = att_df[["year", "Attendance_M", "YoY_pct"]].copy()
table.columns = ["Year", "Total Attendance (M)", "YoY Change (%)"]
table["Total Attendance (M)"] = table["Total Attendance (M)"].round(2)
table["YoY Change (%)"] = table["YoY Change (%)"].round(1)
table = table.set_index("Year")
table.loc["—", :] = ""

display(
    table
    .style
    .format({"Total Attendance (M)": "{:.2f}M", "YoY Change (%)": "{:+.1f}%"}, na_rep="—")
    .set_caption("EXH Total Annual Attendance (2023–2026F)")
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

# DBTITLE 1,EXH Attendance Data Table
# ── EXH Attendance Data Table ────────────────────────────────────────────────
att_table = att_df[["year", "Attendance_M", "YoY_pct"]].copy()
att_table.columns = ["Year", "Total Attendance (M)", "YoY Change (%)"]
att_table["Total Attendance (M)"] = att_table["Total Attendance (M)"].round(2)
att_table["YoY Change (%)"] = att_table["YoY Change (%)"].round(1)

print(f"{'═'*60}")
print(f"  EXH Total Annual Attendance (2023–2026F)")
print(f"{'═'*60}")
display(att_table)


# COMMAND ----------

# DBTITLE 1,EXH Model Features & Correlations
# ── EXH MODEL FEATURES & CORRELATIONS ────────────────────────────────────────
# (Uses df, numpy, pandas, matplotlib from earlier cells)
import matplotlib.patches as mpatches

# ── Feature Groups ────────────────────────────────────────────────────────────
feature_groups = {
    "Seasonality": [
        "season_spring", "season_summer", "season_fall",
        "is_2024", "is_2025", "week_index"
    ],
    "Calendar": [
        "stat_holiday_in_week", "is_long_weekend",
        "days_to_christmas"
    ],
    "Content": [
        "active_films", "cat1_share", "cat2_share",
        "cat3_share", "cat4_share", "cat5_share"
    ],
    "Weather": [
        "national_mean_temp", "national_rain_mm",
        "national_snow_cm", "bad_weather_flag"
    ],
    "Economic": [
        "headline_cpi"
    ],
}

# Flatten to ordered list
feature_order = []
for group, features in feature_groups.items():
    feature_order.extend(features)

# ── Compute Correlations ──────────────────────────────────────────────────────
target = "total_weekly_attendance"

# Encode binary columns — re-encode from raw if corrupted (zero variance)
if df["stat_holiday_in_week"].nunique() <= 1 or df["is_long_weekend"].nunique() <= 1:
    raw = pd.read_excel(FILE_PATH, sheet_name=0)
    df["stat_holiday_in_week"] = (raw["stat_holiday_in_week"] == "Y").astype(int)
    df["is_long_weekend"]      = (raw["is_long_weekend"]      == "Y").astype(int)
elif df["stat_holiday_in_week"].dtype == "object":
    df["stat_holiday_in_week"] = (df["stat_holiday_in_week"] == "Y").astype(int)
    df["is_long_weekend"]      = (df["is_long_weekend"]      == "Y").astype(int)

# Season dummies (only add if not already present)
if "season_spring" not in df.columns:
    season_dummies = pd.get_dummies(df["season"], prefix="season", drop_first=False).astype(int)
    df = pd.concat([df, season_dummies], axis=1)

# Year indicators & trend (only add if not already present)
if "is_2024" not in df.columns:
    df["is_2024"] = (df["year"] == 2024).astype(int)
if "is_2025" not in df.columns:
    df["is_2025"] = (df["year"] == 2025).astype(int)
if "week_index" not in df.columns:
    df["week_index"] = (pd.to_datetime(df["week_date"]) - pd.to_datetime(df["week_date"]).min()).dt.days / 7

# Keep only features that exist in df
valid_features = [f for f in feature_order if f in df.columns]

# Deduplicate columns before computing correlation
df_corr = df[valid_features + [target]].loc[:, ~df[valid_features + [target]].columns.duplicated()]
corr = (
    df_corr
    .corr()[target]
    .drop(target)
    .dropna()
)

# ── Feature Display Names ─────────────────────────────────────────────────────
display_names = {
    "season_spring"         : "Spring Season",
    "season_summer"         : "Summer Season",
    "season_fall"           : "Fall Season",
    "is_2024"               : "Year — 2024",
    "is_2025"               : "Year — 2025",
    "week_index"            : "Linear Trend",
    "stat_holiday_in_week"  : "Stat Holiday",
    "is_long_weekend"       : "Long Weekend",
    "days_to_christmas"     : "Days to Christmas",
    "active_films"          : "Active Films",
    "cat1_share"            : "Category 1 Share",
    "cat2_share"            : "Category 2 Share",
    "cat3_share"            : "Category 3 Share",
    "cat4_share"            : "Category 4 Share",
    "cat5_share"            : "Category 5 Share",
    "national_mean_temp"    : "Mean Temperature",
    "national_rain_mm"      : "Rainfall (mm)",
    "national_snow_cm"      : "Snowfall (cm)",
    "bad_weather_flag"      : "Bad Weather Flag",
    "headline_cpi"          : "Headline CPI",
}

corr.index = [display_names.get(f, f) for f in corr.index]
corr = corr.sort_values()

# ── Group Colours ─────────────────────────────────────────────────────────────
group_colors = {
    "Seasonality": CPX_NAVY,
    "Calendar"   : CPX_GOLD,
    "Content"    : "#2E8B57",
    "Weather"    : "#5B8DB8",
    "Economic"   : CPX_RED,
}

# Map each feature to its group colour
feature_color_map = {}
for group, features in feature_groups.items():
    for f in features:
        disp_name = display_names.get(f, f)
        feature_color_map[disp_name] = group_colors[group]

bar_colors = [
    (CPX_RED if v < 0 else feature_color_map.get(f, CPX_NAVY))
    for f, v in zip(corr.index, corr.values)
]

# ── Plot: Correlation Bar Chart (full width) ─────────────────────────────────
fig, ax = plt.subplots(figsize=get_figsize("16:9"))
fig.patch.set_facecolor(WHITE)

y = np.arange(len(corr))
bars = ax.barh(y, corr.values, color=bar_colors, height=0.6, edgecolor=WHITE)

# Value labels
for i, (val, bar) in enumerate(zip(corr.values, bars)):
    xpos = val + 0.005 if val >= 0 else val - 0.005
    ha   = "left" if val >= 0 else "right"
    ax.text(xpos, bar.get_y() + bar.get_height() / 2,
            f"{val:+.2f}",
            ha=ha, va="center",
            fontsize=8, fontweight="bold", color=CPX_DGREY)

ax.axvline(0, color=CPX_DGREY, linewidth=0.8, linestyle="--")
ax.set_yticks(y)
ax.set_yticklabels(corr.index, fontsize=9)
ax.set_xlabel("Pearson Correlation with Attendance")
ax.set_title("EXH Model — Feature Correlations with Weekly Attendance", loc="center")

# Legend
legend_patches = [
    mpatches.Patch(color=c, label=g)
    for g, c in group_colors.items()
]
ax.legend(handles=legend_patches, loc="lower right", fontsize=8, title="Category")

plt.tight_layout()
plt.show()


# COMMAND ----------

# DBTITLE 1,EXH Model Feature Categories Table
# ── Feature Categories & Correlations Table ────────────────────────────────
from IPython.display import display as ipy_display

rows = []
for group, features in feature_groups.items():
    for f in features:
        if f in corr.index or display_names.get(f, f) in corr.index:
            disp = display_names.get(f, f)
            corr_val = corr.get(disp, None)
            rows.append({
                "Category": group,
                "Feature": disp,
                "Correlation": round(corr_val, 3) if corr_val is not None else None
            })

feature_table = pd.DataFrame(rows)

print(f"{'═'*60}")
print(f"  EXH Model — Feature Categories & Correlations")
print(f"{'═'*60}")
ipy_display(feature_table)


# COMMAND ----------

# DBTITLE 1,EXH Annual Attendance Decomposition (2023-2026F)
# ── EXH ANNUAL ATTENDANCE DECOMPOSITION BY FUNNEL + 2026 FORECAST ────────────
# (Uses pd, np, plt, mticker from earlier cells)
import matplotlib.patheffects as pe
import pickle
from IPython.display import display as ipy_display

ARTIFACT_PATH = "/Volumes/data_exploration_hub/marketing_hub/model_data/Final_Data/MMM_EXH_Outcomes_v2/"

# ── Load Artifacts ────────────────────────────────────────────────────────────
with open(f"{ARTIFACT_PATH}model_meta.pkl", "rb") as f: model_meta = pickle.load(f)

ACTIVE_COMBOS = model_meta["ACTIVE_COMBOS"]
FUNNEL_COMBOS = model_meta["FUNNEL_COMBOS"]
target_col    = model_meta["target_col"]

df_full = pd.read_parquet(f"{ARTIFACT_PATH}df_full.parquet")
df_full["year"] = df_full["week_date"].dt.year

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
forecast_2026      = 46_384_441
media_share_2025   = 0.23

# Apply 2025 funnel mix to 2026 media total
media_2025_total   = awareness_vals[-1] + consideration_vals[-1] + conversion_vals[-1]
aw_share           = awareness_vals[-1]     / media_2025_total
csd_share          = consideration_vals[-1] / media_2025_total
cvs_share          = conversion_vals[-1]    / media_2025_total

media_2026_total   = forecast_2026 * media_share_2025
aw_2026            = media_2026_total * aw_share
csd_2026           = media_2026_total * csd_share
cvs_2026           = media_2026_total * cvs_share
baseline_2026      = forecast_2026 - media_2026_total

# ── Combine Actuals + Forecast ────────────────────────────────────────────────
all_years       = years_avail + [2026]
all_actual      = list(actual_vals)      + [forecast_2026]
all_baseline    = list(baseline_vals)    + [baseline_2026]
all_awareness   = list(awareness_vals)   + [aw_2026]
all_consid      = list(consideration_vals) + [csd_2026]
all_conversion  = list(conversion_vals)  + [cvs_2026]

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
        fontsize=9, fontweight="bold", color=txt_color,
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
alpha = [0.9, 0.9, 0.9, 0.5]  # 2026 bar slightly transparent to indicate forecast

for i, year in enumerate(all_years):
    a = alpha[i]
    b_base = all_baseline[i]
    b_aw   = all_awareness[i]
    b_csd  = all_consid[i]
    b_cvs  = all_conversion[i]
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
    ax.bar(x[i], b_cvs / 1e6,
           width=w, bottom=(b_base + b_aw + b_csd) / 1e6,
           color=CPX_GOLD, alpha=a,
           edgecolor=WHITE, linewidth=0.5,
           label="Conversion" if i == 0 else "_nolegend_")

    # Segment labels
    bar_label(ax, x[i], 0,                       b_base, b_base / total * 100, CPX_LGREY)
    bar_label(ax, x[i], b_base,                  b_aw,   b_aw   / total * 100, CPX_NAVY, 1.5)
    bar_label(ax, x[i], b_base + b_aw,           b_csd,  b_csd  / total * 100, CPX_RED,  1.5)
    bar_label(ax, x[i], b_base + b_aw + b_csd,   b_cvs,  b_cvs  / total * 100, CPX_GOLD, 1.5)

    # Total on top
    label = f"{total/1e6:.2f}M" + (" (F)" if year == 2026 else "")
    ax.text(x[i], total / 1e6 + 0.3, label,
            ha="center", va="bottom",
            fontsize=10, fontweight="bold",
            color=CPX_RED if year == 2026 else CPX_DGREY)

# Actual dots (2023-2025 only)
ax.scatter(x[:3], np.array(actual_vals) / 1e6,
           color="black", zorder=5, s=80,
           marker="D", label="Actual Attendance")

# Forecast marker
ax.scatter(x[3], forecast_2026 / 1e6,
           color=CPX_RED, zorder=5, s=80,
           marker="D", label="2026 Forecast")

# Forecast annotation
ax.axvline(x=2.75, color=CPX_MGREY, linewidth=1, linestyle="--")
ax.text(2.78, 2, "Forecast →",
        fontsize=9, color=CPX_MGREY, fontstyle="italic")

ax.set_xticks(x)
ax.set_xticklabels(all_years, fontsize=12)
ax.set_ylabel("Annual Attendance (M)")
ax.set_xlabel("Year")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}M"))
ax.set_title("EXH Annual Attendance Decomposition by Funnel (2023–2026F)", loc="center")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=3, fontsize=9, frameon=False)

plt.tight_layout()
plt.show()


# COMMAND ----------

# DBTITLE 1,EXH Attendance Decomposition Data Table
# ── EXH Attendance Decomposition Table ────────────────────────────────────
from IPython.display import display as ipy_display

decomp_table = pd.DataFrame({
    "Year"              : [str(y) + (" (F)" if y == 2026 else "") for y in all_years],
    "Baseline (M)"      : [round(v / 1e6, 2) for v in all_baseline],
    "Awareness (M)"     : [round(v / 1e6, 2) for v in all_awareness],
    "Consideration (M)" : [round(v / 1e6, 2) for v in all_consid],
    "Conversion (M)"    : [round(v / 1e6, 2) for v in all_conversion],
    "Media Total (M)"   : [round((a + c + cv) / 1e6, 2)
                           for a, c, cv in zip(all_awareness, all_consid, all_conversion)],
    "Total (M)"         : [round(v / 1e6, 2) for v in all_actual],
    "Media %"           : [round((a + c + cv) / t * 100, 1)
                           for a, c, cv, t in zip(all_awareness, all_consid,
                                                   all_conversion, all_actual)],
})

print(f"{chr(9552)*60}")
print(f"  EXH Attendance Decomposition by Funnel (2023-2026F)")
print(f"{chr(9552)*60}")
ipy_display(decomp_table)


# COMMAND ----------

# DBTITLE 1,EXH Media Contribution by Tentpole & Content Tier
# ── EXH MEDIA CONTRIBUTION BY TENTPOLE & CONTENT TIER ────────────────────────
# (Uses pd, np, pickle from earlier cells)
from IPython.display import display as ipy_display

ARTIFACT_PATH = "/Volumes/data_exploration_hub/marketing_hub/model_data/Final_Data/MMM_EXH_Outcomes_v2/"

# ── Load Artifacts ────────────────────────────────────────────────────────────
with open(f"{ARTIFACT_PATH}model_meta.pkl", "rb") as f: model_meta = pickle.load(f)

FUNNEL_COMBOS = model_meta["FUNNEL_COMBOS"]
target_col    = model_meta["target_col"]

df_full = pd.read_parquet(f"{ARTIFACT_PATH}df_full.parquet")
df_full["year"] = df_full["week_date"].dt.year

# ── Total Media Column ────────────────────────────────────────────────────────
funnel_media_cols = [f"media_{f}" for f in FUNNEL_COMBOS if f"media_{f}" in df_full.columns]
df_full["total_media"] = df_full[funnel_media_cols].sum(axis=1)

# ── Restrict to 2023-2025 ─────────────────────────────────────────────────────
df = df_full[df_full["year"].isin([2023, 2024, 2025])].copy()

# ── TENTPOLE CLASSIFICATION ───────────────────────────────────────────────────
# Tentpole = week has a cat1 (top tier) film
df["is_tentpole"] = df["cat1_share"] > 0

# ── CONTENT TIER CLASSIFICATION ──────────────────────────────────────────────
# Based on cat1_share thresholds
def content_tier(row):
    if row["cat1_share"] >= 0.3:
        return "High Content"
    elif row["cat1_share"] > 0:
        return "Medium Content"
    else:
        return "Low Content"

df["content_tier"] = df.apply(content_tier, axis=1)

# ── TABLE 1: TENTPOLE BY YEAR ─────────────────────────────────────────────────
print("=" * 75)
print("TABLE 1 — MEDIA CONTRIBUTION BY TENTPOLE STATUS AND YEAR")
print("=" * 75)

rows = []
for year in [2023, 2024, 2025]:
    yr          = df[df["year"] == year]
    tent        = yr[yr["is_tentpole"]]
    non_tent    = yr[~yr["is_tentpole"]]
    actual      = yr[target_col].sum()
    media       = yr["total_media"].sum()

    rows.append({
        "Year"              : year,
        "Actual Attendance" : f"{actual/1e6:.2f}M",
        "Media Contribution": f"{media/1e6:.2f}M",
        "Media %"           : f"{media/actual*100:.1f}%",
        "Tentpole Weeks"    : f"{len(tent)} ({len(tent)/len(yr)*100:.1f}%)",
        "Non-Tentpole Weeks": f"{len(non_tent)} ({len(non_tent)/len(yr)*100:.1f}%)",
    })

# Total row
total_actual = df[target_col].sum()
total_media  = df["total_media"].sum()
tent_all     = df[df["is_tentpole"]]
non_tent_all = df[~df["is_tentpole"]]
rows.append({
    "Year"              : "Total",
    "Actual Attendance" : f"{total_actual/1e6:.2f}M",
    "Media Contribution": f"{total_media/1e6:.2f}M",
    "Media %"           : f"{total_media/total_actual*100:.1f}%",
    "Tentpole Weeks"    : f"{len(tent_all)} ({len(tent_all)/len(df)*100:.1f}%)",
    "Non-Tentpole Weeks": f"{len(non_tent_all)} ({len(non_tent_all)/len(df)*100:.1f}%)",
})

tbl1 = pd.DataFrame(rows).set_index("Year")
ipy_display(
    tbl1
    .style
    .set_caption("EXH Media Contribution by Tentpole Status (2023–2025)")
    .set_properties(**{"text-align": "center"})
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

# ── TABLE 2: CONTENT TIER ─────────────────────────────────────────────────────
print("\n" + "=" * 75)
print("TABLE 2 — MEDIA CONTRIBUTION BY CONTENT TIER (2023–2025 COMBINED)")
print("=" * 75)

tier_rows = []
for tier in ["High Content", "Medium Content", "Low Content"]:
    t       = df[df["content_tier"] == tier]
    actual  = t[target_col].sum()
    media   = t["total_media"].sum()
    tier_rows.append({
        "Content Tier"      : tier,
        "Weeks"             : len(t),
        "Actual Attendance" : f"{actual/1e6:.2f}M",
        "Media Contribution": f"{media/1e6:.2f}M",
        "Media %"           : f"{media/actual*100:.1f}%",
        "Avg Weekly Media %" : f"{(t['total_media']/t[target_col]*100).mean():.1f}%",
    })

tbl2 = pd.DataFrame(tier_rows).set_index("Content Tier")
ipy_display(
    tbl2
    .style
    .set_caption("EXH Media Contribution by Content Tier (2023–2025)")
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

# DBTITLE 1,Content Tier by Cat1 Film Count
# ── CONTENT TIER BASED ON CAT1 FILM COUNT ────────────────────────────────────
df["cat1_count"] = (df["cat1_share"] * df["active_films"]).round(1)

def content_tier(count):
    if count >= 3:
        return "High Content"
    elif count >= 1:
        return "Medium Content"
    else:
        return "Low Content"

df["content_tier"] = df["cat1_count"].apply(content_tier)

# ── TABLE: CONTENT TIER ───────────────────────────────────────────────────────
tier_rows = []
for tier in ["High Content", "Medium Content", "Low Content"]:
    t      = df[df["content_tier"] == tier]
    actual = t[target_col].sum()
    media  = t["total_media"].sum()
    tier_rows.append({
        "Content Tier"       : tier,
        "Weeks"              : len(t),
        "Cat1 Films"         : f"{t['cat1_count'].mean():.1f} avg",
        "Actual Attendance"  : f"{actual/1e6:.2f}M",
        "Media Contribution" : f"{media/1e6:.2f}M",
        "Media %"            : f"{media/actual*100:.1f}%",
        "Avg Weekly Media %"  : f"{(t['total_media']/t[target_col]*100).mean():.1f}%",
    })

tbl2 = pd.DataFrame(tier_rows).set_index("Content Tier")
from IPython.display import display as ipy_display
ipy_display(
    tbl2
    .style
    .set_caption("EXH Media Contribution by Content Tier (2023–2025)")
    .set_properties(**{"text-align": "center"})
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


# COMMAND ----------

# DBTITLE 1,EXH Statistical Clarity Tentpole and Content Tier
# ── STATISTICAL CLARITY — TENTPOLE & CONTENT TIER ────────────────────────────
from scipy import stats
import numpy as np

# ── Weekly media % per observation ───────────────────────────────────────────
df["weekly_media_pct"] = df["total_media"] / df[target_col] * 100

# ── 1. TENTPOLE vs NON-TENTPOLE ───────────────────────────────────────────────
tent     = df[df["is_tentpole"]]["weekly_media_pct"]
non_tent = df[~df["is_tentpole"]]["weekly_media_pct"]

t_stat, p_val = stats.ttest_ind(tent, non_tent)
cohen_d = (non_tent.mean() - tent.mean()) / np.sqrt(
    ((len(tent)-1)*tent.std()**2 + (len(non_tent)-1)*non_tent.std()**2) /
    (len(tent) + len(non_tent) - 2)
)

print("=" * 65)
print("1. TENTPOLE vs NON-TENTPOLE — Weekly Media %")
print("=" * 65)
print(f"  Tentpole weeks     : n={len(tent):>3}  mean={tent.mean():.1f}%  std={tent.std():.1f}%")
print(f"  Non-tentpole weeks : n={len(non_tent):>3}  mean={non_tent.mean():.1f}%  std={non_tent.std():.1f}%")
print(f"  Difference         : {non_tent.mean() - tent.mean():+.1f}pp")
print(f"  t-statistic        : {t_stat:.3f}")
print(f"  p-value            : {p_val:.4f}")
print(f"  Cohen's d          : {cohen_d:.3f}")
print(f"  Statistically significant (p<0.05): {'YES' if p_val < 0.05 else 'NO'}")

# ── 2. CONTENT TIER — ANOVA + PAIRWISE ───────────────────────────────────────
high   = df[df["content_tier"] == "High Content"]["weekly_media_pct"]
medium = df[df["content_tier"] == "Medium Content"]["weekly_media_pct"]
low    = df[df["content_tier"] == "Low Content"]["weekly_media_pct"]

f_stat, p_anova = stats.f_oneway(high, medium, low)

print("\n" + "=" * 65)
print("2. CONTENT TIER — Weekly Media % (One-Way ANOVA)")
print("=" * 65)
print(f"  High Content   : n={len(high):>3}  mean={high.mean():.1f}%  std={high.std():.1f}%")
print(f"  Medium Content : n={len(medium):>3}  mean={medium.mean():.1f}%  std={medium.std():.1f}%")
print(f"  Low Content    : n={len(low):>3}  mean={low.mean():.1f}%  std={low.std():.1f}%")
print(f"  F-statistic    : {f_stat:.3f}")
print(f"  p-value        : {p_anova:.4f}")
print(f"  Statistically significant (p<0.05): {'YES' if p_anova < 0.05 else 'NO'}")

# Pairwise t-tests
print("\n  Pairwise comparisons:")
pairs = [
    ("High vs Medium", high, medium),
    ("High vs Low",    high, low),
    ("Medium vs Low",  medium, low),
]
for label, g1, g2 in pairs:
    t, p = stats.ttest_ind(g1, g2)
    d = (g2.mean() - g1.mean()) / np.sqrt(
        ((len(g1)-1)*g1.std()**2 + (len(g2)-1)*g2.std()**2) /
        (len(g1) + len(g2) - 2)
    )
    print(f"  {label:<20} diff={g2.mean()-g1.mean():+.1f}pp  "
          f"p={p:.4f}  Cohen's d={d:.3f}  "
          f"{'*' if p < 0.05 else 'ns'}")


# COMMAND ----------

# DBTITLE 1,EXH Tentpole Definition Cat1 Film Opens
# ── TENTPOLE = WEEK A NEW CAT1 FILM OPENS ────────────────────────────────────
from scipy import stats
import numpy as np

df = df.sort_values("week_date").reset_index(drop=True)
df["cat1_count"]  = (df["cat1_share"] * df["active_films"]).round(1)
df["cat1_change"] = df["cat1_count"].diff()
df["is_tentpole"] = df["cat1_change"] > 0
df["weekly_media_pct"] = df["total_media"] / df[target_col] * 100

# ── TABLE: TENTPOLE BY YEAR ───────────────────────────────────────────────────
rows = []
for year in [2023, 2024, 2025]:
    yr       = df[df["year"] == year]
    tent     = yr[yr["is_tentpole"]]
    non_tent = yr[~yr["is_tentpole"]]
    actual   = yr[target_col].sum()
    media    = yr["total_media"].sum()
    rows.append({
        "Year"               : year,
        "Actual Attendance"  : f"{actual/1e6:.2f}M",
        "Media Contribution" : f"{media/1e6:.2f}M",
        "Media %"            : f"{media/actual*100:.1f}%",
        "Tentpole Weeks"     : f"{len(tent)} ({len(tent)/len(yr)*100:.1f}%)",
        "Non-Tentpole Weeks" : f"{len(non_tent)} ({len(non_tent)/len(yr)*100:.1f}%)",
        "Tent Media %"       : f"{tent['total_media'].sum()/tent[target_col].sum()*100:.1f}%",
        "Non-Tent Media %"   : f"{non_tent['total_media'].sum()/non_tent[target_col].sum()*100:.1f}%",
    })

tent_all     = df[df["is_tentpole"]]
non_tent_all = df[~df["is_tentpole"]]
total_actual = df[target_col].sum()
total_media  = df["total_media"].sum()

rows.append({
    "Year"               : "Total",
    "Actual Attendance"  : f"{total_actual/1e6:.2f}M",
    "Media Contribution" : f"{total_media/1e6:.2f}M",
    "Media %"            : f"{total_media/total_actual*100:.1f}%",
    "Tentpole Weeks"     : f"{len(tent_all)} ({len(tent_all)/len(df)*100:.1f}%)",
    "Non-Tentpole Weeks" : f"{len(non_tent_all)} ({len(non_tent_all)/len(df)*100:.1f}%)",
    "Tent Media %"       : f"{tent_all['total_media'].sum()/tent_all[target_col].sum()*100:.1f}%",
    "Non-Tent Media %"   : f"{non_tent_all['total_media'].sum()/non_tent_all[target_col].sum()*100:.1f}%",
})

tbl1 = pd.DataFrame(rows).set_index("Year")
from IPython.display import display as ipy_display

# replace the display() call with:
ipy_display(
    tbl1
    .style
    .set_caption("EXH Media Contribution — Tentpole Opening Weeks vs Non-Tentpole (2023–2025)")
    .set_properties(**{"text-align": "center"})
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

# ── STATISTICAL TEST ──────────────────────────────────────────────────────────
tent_pct     = tent_all["weekly_media_pct"]
non_tent_pct = non_tent_all["weekly_media_pct"]

t_stat, p_val = stats.ttest_ind(tent_pct, non_tent_pct)
cohen_d = (non_tent_pct.mean() - tent_pct.mean()) / np.sqrt(
    ((len(tent_pct)-1)*tent_pct.std()**2 + (len(non_tent_pct)-1)*non_tent_pct.std()**2) /
    (len(tent_pct) + len(non_tent_pct) - 2)
)

print("=" * 65)
print("STATISTICAL TEST — Tentpole Opening vs Non-Tentpole")
print("=" * 65)
print(f"  Tentpole opening weeks : n={len(tent_pct):>3}  mean={tent_pct.mean():.1f}%  std={tent_pct.std():.1f}%")
print(f"  Non-tentpole weeks     : n={len(non_tent_pct):>3}  mean={non_tent_pct.mean():.1f}%  std={non_tent_pct.std():.1f}%")
print(f"  Difference             : {non_tent_pct.mean() - tent_pct.mean():+.1f}pp")
print(f"  t-statistic            : {t_stat:.3f}")
print(f"  p-value                : {p_val:.4f}")
print(f"  Cohen's d              : {cohen_d:.3f}")
print(f"  Significant (p<0.05)   : {'YES' if p_val < 0.05 else 'NO'}")


# COMMAND ----------

# DBTITLE 1,Correlation Analysis Media vs Key Drivers
# ── CORRELATION ANALYSIS — MEDIA % vs KEY DRIVERS ────────────────────────────
import pandas as pd
import numpy as np
from scipy import stats

# Variables to test against weekly media %
candidates = {
    "Active Films"          : "active_films",
    "Cat1 Count"            : "cat1_count",
    "Cat2 Share"            : "cat2_share",
    "Cat3 Share"            : "cat3_share",
    "Mean Temperature"      : "national_mean_temp",
    "Rainfall"              : "national_rain_mm",
    "Snowfall"              : "national_snow_cm",
    "Bad Weather"           : "bad_weather_flag",
    "Stat Holiday"          : "stat_holiday_in_week",
    "Long Weekend"          : "is_long_weekend",
    "Days to Christmas"     : "days_to_christmas",
    "Awareness Spend"       : "media_Awareness",
    "Consideration Spend"   : "media_Consideration",
    "Conversion Spend"      : "media_Conversion",
}

rows = []
for label, col in candidates.items():
    if col not in df.columns:
        continue
    x = pd.to_numeric(df[col], errors="coerce")
    y = df["weekly_media_pct"]
    mask = x.notna() & y.notna()
    r, p = stats.pearsonr(x[mask], y[mask])
    rows.append({
        "Variable"   : label,
        "Pearson r"  : round(r, 3),
        "p-value"    : round(p, 4),
        "Significant": "YES" if p < 0.05 else "NO"
    })

corr_df = (
    pd.DataFrame(rows)
    .set_index("Variable")
    .sort_values("Pearson r", key=abs, ascending=False)
)

print(corr_df.to_string())


# COMMAND ----------

# DBTITLE 1,Exploratory Correlation Analysis
# ── EXPLORATORY CORRELATION ANALYSIS — ALL DIRECTIONS ────────────────────────
import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

# ── Rebuild season from week_date if missing ──────────────────────────────────
def assign_season(month):
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Spring"
    elif month in [6, 7, 8]:
        return "Summer"
    else:
        return "Fall"

df["month"]      = df["week_date"].dt.month
df["season"]     = df["month"].apply(assign_season)
df["week_of_year"] = df["week_date"].dt.isocalendar().week.astype(int)

# ── 1. FUNNEL EFFICIENCY OVER TIME ────────────────────────────────────────────
print("=" * 65)
print("1. FUNNEL SPEND vs MEDIA % — BY YEAR")
print("=" * 65)
for year in [2023, 2024, 2025]:
    yr = df[df["year"] == year]
    print(f"\n  {year}:")
    for label, col in [("Awareness",     "media_Awareness"),
                        ("Consideration", "media_Consideration"),
                        ("Conversion",    "media_Conversion")]:
        if col not in yr.columns:
            continue
        r, p = stats.pearsonr(yr[col], yr["weekly_media_pct"])
        print(f"    {label:<15} r={r:+.3f}  p={p:.4f}  {'*' if p < 0.05 else 'ns'}")

# ── 2. SEASONAL MEDIA EFFICIENCY ─────────────────────────────────────────────
print("\n" + "=" * 65)
print("2. MEDIA % BY SEASON")
print("=" * 65)
for season in ["Winter", "Spring", "Summer", "Fall"]:
    s = df[df["season"] == season]
    print(f"  {season:<10} n={len(s):>3}  "
          f"mean={s['weekly_media_pct'].mean():.1f}%  "
          f"std={s['weekly_media_pct'].std():.1f}%  "
          f"median={s['weekly_media_pct'].median():.1f}%")

groups = [df[df["season"] == s]["weekly_media_pct"]
          for s in ["Winter", "Spring", "Summer", "Fall"]]
f, p = stats.f_oneway(*groups)
print(f"\n  ANOVA: F={f:.3f}  p={p:.4f}  "
      f"{'SIGNIFICANT' if p < 0.05 else 'not significant'}")

# Pairwise
seasons = ["Winter", "Spring", "Summer", "Fall"]
print("\n  Pairwise t-tests:")
for i, s1 in enumerate(seasons):
    for s2 in seasons[i+1:]:
        g1 = df[df["season"] == s1]["weekly_media_pct"]
        g2 = df[df["season"] == s2]["weekly_media_pct"]
        t, p = stats.ttest_ind(g1, g2)
        print(f"    {s1} vs {s2:<8} "
              f"diff={g2.mean()-g1.mean():+.1f}pp  "
              f"p={p:.4f}  {'*' if p < 0.05 else 'ns'}")

# ── 3. WEEK-OF-YEAR MEDIA EFFICIENCY ─────────────────────────────────────────
print("\n" + "=" * 65)
print("3. WEEK-OF-YEAR — TOP 10 & BOTTOM 10 BY AVG MEDIA %")
print("=" * 65)
woy = (
    df.groupby("week_of_year")["weekly_media_pct"]
    .agg(["mean", "std", "count"])
    .round(2)
    .rename(columns={"mean": "Avg Media %", "std": "Std", "count": "n"})
)
print("\n  TOP 10 weeks (highest media %):")
print(woy.sort_values("Avg Media %", ascending=False).head(10).to_string())
print("\n  BOTTOM 10 weeks (lowest media %):")
print(woy.sort_values("Avg Media %").head(10).to_string())

# ── 4. SPEND CONCENTRATION ────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("4. MEDIA CONTRIBUTION CONCENTRATION")
print("=" * 65)
df_sorted = df.sort_values("total_media", ascending=False).copy()
df_sorted["cumulative_media"] = df_sorted["total_media"].cumsum()
df_sorted["cumulative_pct"]   = (df_sorted["cumulative_media"] /
                                  df_sorted["total_media"].sum() * 100)
for pct in [50, 80, 90]:
    n_weeks = (df_sorted["cumulative_pct"] <= pct).sum() + 1
    print(f"  {pct}% of media contribution comes from top {n_weeks} weeks "
          f"({n_weeks/len(df)*100:.1f}% of all weeks)")

# ── 5. GENRE INTENSITY vs MEDIA % ────────────────────────────────────────────
print("\n" + "=" * 65)
print("5. GENRE INTENSITY vs MEDIA %")
print("=" * 65)
genre_cols = [c for c in df.columns if c.endswith("_intensity")]
genre_rows = []
for col in genre_cols:
    r, p = stats.pearsonr(df[col], df["weekly_media_pct"])
    genre_rows.append({
        "Genre"      : col.replace("_intensity", "").title(),
        "Pearson r"  : round(r, 3),
        "p-value"    : round(p, 4),
        "Significant": "YES" if p < 0.05 else "NO"
    })
genre_df = (
    pd.DataFrame(genre_rows)
    .set_index("Genre")
    .sort_values("Pearson r", key=abs, ascending=False)
)
print(genre_df.to_string())

# ── 6. CROSS-FUNNEL INTERACTION ───────────────────────────────────────────────
print("\n" + "=" * 65)
print("6. CROSS-FUNNEL INTERACTION")
print("=" * 65)
df["aw_x_csd"]  = df["media_Awareness"]     * df["media_Consideration"]
df["aw_x_cvs"]  = df["media_Awareness"]     * df["media_Conversion"]
df["csd_x_cvs"] = df["media_Consideration"] * df["media_Conversion"]

for label, col in [("Awareness × Consideration", "aw_x_csd"),
                    ("Awareness × Conversion",    "aw_x_cvs"),
                    ("Consideration × Conversion","csd_x_cvs")]:
    r, p = stats.pearsonr(df[col], df["weekly_media_pct"])
    print(f"  {label:<30} r={r:+.3f}  p={p:.4f}  "
          f"{'*' if p < 0.05 else 'ns'}")


# COMMAND ----------

# DBTITLE 1,Setup LOB Spend and Contribution Charts
# ── SETUP — Load all dependencies for LOB SpendContrib charts ────────────────
import os
import pickle
import numpy as np
import pandas as pd
from collections import defaultdict

GROUPED_PATH = "/Volumes/data_exploration_hub/marketing_hub/model_data/Final_Data/MMM_EXH_Outcomes_v2/"
DECOMP_PATH  = "/Volumes/data_exploration_hub/marketing_hub/model_data/Final_Data/MMM_EXH_LOB_Decomp_v2/"
OUT_PATH     = "/Volumes/data_exploration_hub/marketing_hub/model_data/Final_Data/MMM_EXH_LOB_Decomp_v2/"

# ── Load artifacts ────────────────────────────────────────────────────────────
with open(f"{GROUPED_PATH}model_meta.pkl",            "rb") as f: model_meta            = pickle.load(f)
with open(f"{GROUPED_PATH}response_params.pkl",       "rb") as f: response_params       = pickle.load(f)
with open(f"{GROUPED_PATH}spend_train_max_iter2.pkl", "rb") as f: spend_train_max_iter2 = pickle.load(f)
with open(f"{DECOMP_PATH}decomp_lob_results.pkl",      "rb") as f: decomp_artifact       = pickle.load(f)

ACTIVE_COMBOS = model_meta["ACTIVE_COMBOS"]
FUNNEL_COMBOS = model_meta["FUNNEL_COMBOS"]
LOB_GROUP_MAP = model_meta["LOB_GROUP_MAP"]
target_col    = model_meta["target_col"]

df_full = pd.read_parquet(f"{GROUPED_PATH}df_full.parquet")
df_full["year"] = df_full["week_date"].dt.year

# ── Load decomp table ─────────────────────────────────────────────────────────
# v2 format: DataFrame directly (not dict); adapt column names for compatibility
if isinstance(decomp_artifact, dict):
    decomp = decomp_artifact["decomp_table"]
else:
    decomp = decomp_artifact.rename(columns={"funnel": "Funnel", "original_lob": "LOB"})
    decomp["Status"] = "Decomposed"
    # Add derived columns expected by downstream chart cells
    decomp["Annual Spend"] = decomp["annual_spend"]
    decomp["Total Contrib (M)"] = decomp["contribution"] / 1e6
    decomp["Total Spend ($10K)"] = decomp["annual_spend"] / 1e4
    decomp["Att per $1"] = decomp["contribution"] / decomp["annual_spend"].replace(0, np.nan)

# ── Parse funnel_lob_cols from raw media columns ──────────────────────────────
FUNNELS    = ["Awareness", "Consideration", "Conversion"]
media_cols = [
    c for c in df_full.columns
    if any(c.startswith(f + "_") for f in FUNNELS)
    and "__" not in c
    and "media_" not in c
]

funnel_lob_cols = defaultdict(list)
for col in media_cols:
    for funnel in FUNNELS:
        if col.startswith(funnel + "_"):
            rest    = col[len(funnel) + 1:]
            channel = rest.rsplit("_", 1)[-1]
            lob_raw = rest.rsplit("_", 1)[0]
            funnel_lob_cols[(funnel, lob_raw)].append(col)
            break

# ── LOB display name mapper ───────────────────────────────────────────────────
def lob_display_name(raw):
    mapping = {
        "EXH - Brand"                        : "EXH - Brand",
        "F&B"                                : "F&B",
        "G&E"                                : "G&E",
        "Film Marketing - Inform"            : "Film Marketing - Inform",
        "Performance - Convert & Upgrade"    : "Performance - Convert & Upgrade",
        "Premium Experiences - AVX"          : "Premium Experiences - AVX",
        "Premium Experiences - IMAX"         : "Premium Experiences - IMAX",
        "Premium Experiences - VIP"          : "Premium Experiences - VIP",
        "Prepaid - B2B"                      : "Prepaid - B2B",
        "Prepaid - B2C"                      : "Prepaid - B2C",
        "Cineclub"                           : "Cineclub",
    }
    return mapping.get(raw, raw)

print("✓ All dependencies loaded — ready to run LOB SpendContrib charts.")
print(f"  Decomp LOBs     : {decomp['LOB'].nunique()}")
print(f"  df_full rows    : {len(df_full)}")
print(f"  funnel_lob_cols : {len(funnel_lob_cols)} cells")
print(f"  OUT_PATH        : {OUT_PATH}")


# COMMAND ----------

# DBTITLE 1,Per-LOB Annual Spend & Attendance Contribution
# ── CELL: Per-LOB Annual Spend & Attendance Contribution — grouped bars by year ──
# One chart per LOB. For each year: stacked spend bar + attendance contribution bar side by side.
# Unvalidated/geo-test LOBs included with funnel-avg att-per-$1 as directional proxy.
# Funnel colours: Awareness=#002B5C, Consideration=#E31837, Conversion=#F5A800

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

FUNNEL_COLOR = {"Awareness": CPX_NAVY, "Consideration": CPX_RED, "Conversion": CPX_GOLD}
FUNNELS      = ["Awareness", "Consideration", "Conversion"]

def save_fig(fig, fname):
    fig.savefig(os.path.join(OUT_PATH, fname), dpi=150, bbox_inches="tight")
    print(f"Saved: {fname}")

# ── 1. Validated att-per-$1 from decomp ──────────────────────────────────────
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
            float(df_full[df_full["year"] == yr][cols].sum().sum()) > 0
            for yr in years_avail
        )
        if has_spend:
            fl_rate[key] = funnel_avg_rate.get(funnel, 0.0)
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
        fl_yr_spend[key][yr] += float(df_full[df_full["year"] == yr][cols].sum().sum())

# ── 4. Unique LOBs ────────────────────────────────────────────────────────────
all_lobs = sorted({lob for (_, lob) in fl_rate})

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

    spend_by_f   = {f: np.array([fl_yr_spend.get((f, lob), {}).get(yr, 0) / 1e3
                                  for yr in years_avail]) for f in active_funnels}
    contrib_by_f = {f: np.array([fl_yr_spend.get((f, lob), {}).get(yr, 0) * fl_rate[(f, lob)] / 1e6
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
    ax_s.set_xticklabels([str(y) for y in years_avail], fontsize=10, color=CPX_DGREY)
    ax_s.set_xlim(-0.3, x_contrib[-1] + bar_w + 0.3)

    # Axis formatting
    ax_s.set_ylabel("Annual Media Spend ($K)", fontsize=9, color=CPX_DGREY)
    ax_s.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:.0f}K"))
    ax_s.set_ylim(bottom=0, top=max(bottoms_s.max(), 0.01) * 1.3)

    ax_c.set_ylabel("Media Attendance Contribution (M)", fontsize=9, color=CPX_DGREY)
    ax_c.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.2f}M"))
    ax_c.set_ylim(bottom=0, top=max(bottoms_c.max(), 0.01) * 1.3)

    for ax in (ax_s, ax_c):
        ax.spines["top"].set_visible(False)
        ax.tick_params(axis="y", labelsize=8)

    ax_s.set_axisbelow(True)
    ax_s.grid(axis="y", alpha=0.2, color=CPX_LGREY)

    # Legend
    handles = [mpatches.Patch(color=FUNNEL_COLOR[f], label=f) for f in active_funnels]
    handles += [
        mpatches.Patch(facecolor="#999999", alpha=0.9, label="Spend"),
        mpatches.Patch(facecolor="#999999", alpha=0.55, hatch="///", label="Attendance Contrib"),
    ]
    ax_s.legend(handles=handles, fontsize=8,
                loc="upper left", bbox_to_anchor=(1.18, 1),
                borderaxespad=0, framealpha=0.9, edgecolor=CPX_LGREY)

    ax_s.set_title(
        f"{lob}\nAnnual Spend vs Attendance Contribution",
        fontsize=11, fontweight="bold",
        color="black",
        pad=10
    )

    plt.tight_layout()
    plt.subplots_adjust(right=0.68)
    plt.show()
    plt.close(fig)

    # ── Small summary table for this LOB ──────────────────────────────────────
    tbl_rows = []
    for yr_idx, yr in enumerate(years_avail):
        row = {"Year": yr}
        for f in active_funnels:
            row[f"{f} Spend ($K)"] = round(spend_by_f[f][yr_idx], 1)
            row[f"{f} Contrib (M)"] = round(contrib_by_f[f][yr_idx], 3)
        row["Total Spend ($K)"] = round(total_spend_yr[yr_idx], 1)
        row["Total Contrib (M)"] = round(total_contrib_yr[yr_idx], 3)
        tbl_rows.append(row)
    # Add att-per-$1 rates as footer info
    rate_row = {"Year": "Att/$1"}
    for f in active_funnels:
        rate_row[f"{f} Spend ($K)"] = ""
        rate_row[f"{f} Contrib (M)"] = f"{fl_rate[(f, lob)]:.2f}"
    rate_row["Total Spend ($K)"] = ""
    rate_row["Total Contrib (M)"] = "" 
    tbl_rows.append(rate_row)

    from IPython.display import display as ipy_display
    ipy_display(pd.DataFrame(tbl_rows))


# COMMAND ----------

# DBTITLE 1,EXH Flighting Analysis
# ── EXH FLIGHTING ANALYSIS — FOOLPROOF STANDALONE BLOCK ──────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import pickle
import arviz as az
from collections import defaultdict
from IPython.display import display as ipy_display

# ── Paths ─────────────────────────────────────────────────────────────────────
ARTIFACT_PATH = "/Volumes/data_exploration_hub/marketing_hub/model_data/Final_Data/MMM_EXH_Outcomes_v2/"
DECOMP_PATH   = "/Volumes/data_exploration_hub/marketing_hub/model_data/Final_Data/MMM_EXH_LOB_Decomp_v2/"
OUT_PATH      = "/Volumes/data_exploration_hub/marketing_hub/model_data/Final_Data/MMM_EXH_LOB_Decomp_v2/"

# ── Brand colours ─────────────────────────────────────────────────────────────
CPX_NAVY  = "#002B5C"
CPX_RED   = "#E31837"
CPX_GOLD  = "#F5A800"
CPX_DGREY = "#4A4A4A"
CPX_MGREY = "#8A8A8A"
CPX_LGREY = "#F2F2F2"
CPX_GREEN = "#2E7D32"
CPX_MID   = "#B0BEC5"
WHITE     = "#FFFFFF"

FUNNELS = ["Awareness", "Consideration", "Conversion"]
FUNNEL_COLORS = {
    "Awareness"    : CPX_NAVY,
    "Consideration": CPX_RED,
    "Conversion"   : CPX_GOLD,
}

# ── Load artifacts ────────────────────────────────────────────────────────────
with open(f"{ARTIFACT_PATH}model_meta.pkl", "rb") as f:
    model_meta = pickle.load(f)

target_col    = model_meta["target_col"]
ACTIVE_COMBOS = model_meta["ACTIVE_COMBOS"]

# Load trace for posterior decay
trace = az.from_netcdf(f"{ARTIFACT_PATH}trace_iter2_mixed_hier.nc")
post  = trace.posterior

# Load data
df_full = pd.read_parquet(f"{ARTIFACT_PATH}df_full.parquet")
df_full["year"] = df_full["week_date"].dt.year
df_fl = df_full[df_full["year"].isin([2023, 2024, 2025])].copy()
yrs   = [2023, 2024, 2025]

# ── LOB to combo alpha mapping ────────────────────────────────────────────────
# Each original LOB inherits its parent combo's posterior alpha
with open(f"{ARTIFACT_PATH}response_params.pkl", "rb") as f:
    response_params = pickle.load(f)

LOB_GROUP_MAP = model_meta["LOB_GROUP_MAP"]

# Invert: LOB -> combo name
lob_to_combo = {}
for combo, lobs in LOB_GROUP_MAP.items():
    for lob in lobs:
        lob_to_combo[lob] = combo

# ── Optimal lag range — combo-specific ───────────────────────────────────────
def optimal_lag_range(alpha):
    """
    Derive pre-peak spend window from adstock alpha.
    Uses the week range where cumulative carryover is between 20% and 70% of total.
    Fast decay (alpha<0.3): immediate effect — lag 0–1 wks.
    Moderate (0.3–0.5)    : lag 1–3 wks.
    Slow (alpha>0.5)       : lag 3–6 wks.
    """
    if alpha >= 1.0:
        return 1, 8
    total = 1 / (1 - alpha)
    cum   = 0.0
    lmin, lmax = None, None
    for w in range(1, 25):
        cum += alpha ** w
        pct  = cum / total
        if lmin is None and pct >= 0.20:
            lmin = w
        if lmin is not None and lmax is None and pct >= 0.70:
            lmax = w
        if lmin and lmax:
            break
    lmin = lmin or 1
    lmax = lmax or (lmin + 2)
    # Fast decay — Conversion combos
    if alpha < 0.30:
        return 1, 2
    return lmin, lmax

# ── Identify raw LOB spend columns ───────────────────────────────────────────
# Format: Funnel_LOB_Channel (single underscore, no media_ prefix, no __ combos)
media_cols = [
    c for c in df_full.columns
    if any(c.startswith(f"{f}_") for f in FUNNELS)
    and "__" not in c
    and not c.startswith("media_")
]

print(f"\nMedia columns found: {len(media_cols)}")

# ── Build funnel-level spend on df_fl ─────────────────────────────────────────
spend_cols_by_funnel = {
    f: [c for c in media_cols if c.startswith(f"{f}_")]
    for f in FUNNELS
}
for f in FUNNELS:
    cols = spend_cols_by_funnel[f]
    df_fl[f"spend_{f}"] = df_fl[cols].sum(axis=1) if cols else 0.0
    print(f"  {f:<15} {len(cols)} cols  "
          f"total 3yr={df_fl['spend_'+f].sum()/1e6:.2f}M")

df_fl["woy"] = df_fl["week_date"].dt.isocalendar().week.astype(int)

# ── Parse individual LOB columns ──────────────────────────────────────────
funnel_lob_cols = defaultdict(list)
for col in media_cols:
    for funnel in FUNNELS:
        if col.startswith(f"{funnel}_"):
            rest    = col[len(funnel) + 1:]
            lob_raw = rest.rsplit("_", 1)[0]
            funnel_lob_cols[(funnel, lob_raw)].append(col)
            break

print(f"Parsed {len(funnel_lob_cols)} (funnel, LOB) combinations")

# Build LOB-level alpha from response_params
lob_alpha = {}
for (funnel, lob_raw), cols in funnel_lob_cols.items():
    lob_label = lob_raw.strip()
    combo     = lob_to_combo.get(lob_label)
    if combo is None:
        continue
    rp_key = f"{funnel}__{combo}"
    if rp_key in response_params:
        lob_alpha[(funnel, lob_label)] = response_params[rp_key]["alpha"]

print("\nLOB-level alpha values:")
for (funnel, lob), alpha in sorted(lob_alpha.items()):
    print(f"  {funnel:<15} {lob:<35} alpha={alpha:.4f}")

# Funnel-level decay = mean of LOB alphas per funnel
funnel_decay = {}
for f in FUNNELS:
    alphas = [v for (fn, _), v in lob_alpha.items() if fn == f]
    funnel_decay[f] = float(np.mean(alphas)) if alphas else 0.5

print("\nFunnel mean alpha:")
for f in FUNNELS:
    print(f"  {f:<15} alpha={funnel_decay[f]:.4f}")

# Per-funnel lags
funnel_lags = {f: optimal_lag_range(funnel_decay[f]) for f in FUNNELS}

print("\nFunnel lag windows:")
for f in FUNNELS:
    print(f"  {f:<15} lag={funnel_lags[f][0]}-{funnel_lags[f][1]} wks")

# Per-LOB lags for zone classification
lob_lags = {}
for (funnel, lob), alpha in lob_alpha.items():
    lob_lags[(funnel, lob)] = optimal_lag_range(alpha)

print("\nLOB-level lag windows:")
for (funnel, lob), (lmin, lmax) in sorted(lob_lags.items()):
    print(f"  {funnel:<15} {lob:<35} "
          f"alpha={lob_alpha[(funnel,lob)]:.4f}  lag={lmin}-{lmax} wks")

woy_avg = (
    df_fl.groupby("woy")
    .agg(
        attendance          =(target_col,            "mean"),
        spend_Awareness     =("spend_Awareness",      "mean"),
        spend_Consideration =("spend_Consideration",  "mean"),
        spend_Conversion    =("spend_Conversion",     "mean"),
    )
    .reset_index()
)
wks = woy_avg["woy"].values

# ── Derive peak weeks empirically ─────────────────────────────────────────────
# Peaks: New Year (1), Easter (14), Summer (30), Christmas (52)
PEAK_WKS = [1, 14, 30, 52]

# Troughs: post-New Year lull (2-6) and back-to-school (37-46)
TROUGH_WKS = list(range(2, 7)) + list(range(37, 47))

print(f"EXH Peak weeks   : {PEAK_WKS}")
print(f"EXH Trough weeks : {TROUGH_WKS}")

# Verify against data
print("\nPeak week attendance:")
print(woy_avg[woy_avg["woy"].isin(PEAK_WKS)][["woy","attendance"]]
      .assign(attendance=lambda x: (x["attendance"]/1e6).round(3))
      .to_string())

print("\nTrough week attendance (avg):")
trough_avg = woy_avg[woy_avg["woy"].isin(TROUGH_WKS)]["attendance"].mean()
print(f"  Mean trough attendance: {trough_avg/1e6:.3f}M")
print(f"  Overall mean          : {woy_avg['attendance'].mean()/1e6:.3f}M")

# ── Derive trough weeks statistically ─────────────────────────────────────────


# ── Optimal week sets per funnel ──────────────────────────────────────────────
def get_optimal_weeks(lag_min, lag_max):
    wks_set = set()
    for p in PEAK_WKS:
        for lag in range(lag_min, lag_max + 1):
            w = p - lag
            if w <= 0: w += 52
            wks_set.add(w)
    return sorted(wks_set)

funnel_opt_wks = {f: get_optimal_weeks(*funnel_lags[f]) for f in FUNNELS}

# ── Year-by-year flighting alignment ──────────────────────────────────────────
print("\n" + "=" * 65)
print("FLIGHTING ZONE % BY FUNNEL AND YEAR")
print("=" * 65)

for funnel in FUNNELS:
    spend_col = f"spend_{funnel}"
    opt_wks   = funnel_opt_wks[funnel]
    print(f"\n{funnel}  "
          f"(decay={funnel_decay[funnel]:.3f}  "
          f"lag={funnel_lags[funnel][0]}–{funnel_lags[funnel][1]} wks)")
    print(f"  {'Year':<6} {'Optimal':>9} {'Trough':>9} "
          f"{'Peak(late)':>12} {'Neutral':>9}")
    print("  " + "-" * 45)

    for year in yrs:
        yr     = df_fl[df_fl["year"] == year]
        woy_yr = yr.groupby("woy")[spend_col].mean().reset_index()
        total  = woy_yr[spend_col].sum()
        if total == 0:
            print(f"  {year}  — no spend")
            continue
        sp_opt    = woy_yr[woy_yr["woy"].isin(opt_wks)][spend_col].sum()
        sp_trough = woy_yr[woy_yr["woy"].isin(TROUGH_WKS)][spend_col].sum()
        sp_peak   = woy_yr[woy_yr["woy"].isin(PEAK_WKS)][spend_col].sum()
        sp_other  = total - sp_opt - sp_trough - sp_peak
        print(f"  {year:<6} "
              f"{sp_opt/total*100:>8.1f}% "
              f"{sp_trough/total*100:>8.1f}% "
              f"{sp_peak/total*100:>11.1f}% "
              f"{sp_other/total*100:>8.1f}%")

# ── Confidence tier map ───────────────────────────────────────────────────────
CONFIDENCE_TIERS = {
    # Geo-test
    ("Conversion",    "Film Marketing - Inform")         : "Low — geo-test",
    ("Conversion",    "Cineclub")                        : "Low — geo-test",
    ("Awareness",     "Cineclub")                        : "Low — geo-test",
    ("Consideration", "Film Marketing - Inform")         : "Low — geo-test",
    ("Consideration", "Cineclub")                        : "Low — geo-test",
    # Awareness
    ("Awareness",     "EXH - Brand")                    : "Medium",
    ("Awareness",     "Premium Experiences - AVX")      : "Medium",
    ("Awareness",     "F&B")                            : "Low",
    ("Awareness",     "Film Marketing - Inform")        : "Low",
    ("Awareness",     "Performance - Convert & Upgrade"): "Low",
    ("Awareness",     "Premium Experiences - VIP")      : "Low",
    # Consideration
    ("Consideration", "EXH - Brand")                    : "Low",
    ("Consideration", "G&E")                            : "Low",
    ("Consideration", "Premium Experiences - AVX")      : "Low",
    ("Consideration", "Prepaid - B2B")                  : "Low",
    ("Consideration", "Prepaid - B2C")                  : "Low",
    ("Consideration", "F&B")                            : "Low",
    ("Consideration", "Premium Experiences - IMAX")     : "Low",
    ("Consideration", "Premium Experiences - VIP")      : "Low",
    # Conversion — validated
    ("Conversion",    "EXH - Brand")                    : "High",
    ("Conversion",    "Performance - Convert & Upgrade") : "High",
    ("Conversion",    "Premium Experiences - IMAX")     : "High",
    ("Conversion",    "Premium Experiences - AVX")      : "Medium",
    ("Conversion",    "Premium Experiences - VIP")      : "Medium",
    ("Conversion",    "Prepaid - B2B")                  : "Medium",
    ("Conversion",    "F&B")                            : "Low",
    ("Conversion",    "G&E")                            : "Low",
}

# ── Parse individual LOB columns ──────────────────────────────────────────────
funnel_lob_cols = defaultdict(list)
for col in media_cols:
    for funnel in FUNNELS:
        if col.startswith(f"{funnel}_"):
            rest    = col[len(funnel) + 1:]
            lob_raw = rest.rsplit("_", 1)[0]
            funnel_lob_cols[(funnel, lob_raw)].append(col)
            break

print(f"\nParsed {len(funnel_lob_cols)} (funnel, LOB) combinations")

# ── LOB flighting zone breakdown ──────────────────────────────────────────────
lob_rows = []
for (funnel, lob_raw), cols in funnel_lob_cols.items():
    if not cols:
        continue
    lob_label  = lob_raw.strip()
    confidence = CONFIDENCE_TIERS.get((funnel, lob_label), "Medium")

    # Use LOB-specific lag if available, fall back to funnel-level
    lag_min, lag_max = lob_lags.get(
        (funnel, lob_label),
        funnel_lags[funnel]
    )
    opt_wks = get_optimal_weeks(lag_min, lag_max)

    woy_lob = (
        df_fl
        .assign(woy=df_fl["week_date"].dt.isocalendar().week.astype(int))
        .groupby("woy")[cols]
        .mean()
        .sum(axis=1)
        .reset_index()
        .rename(columns={0: "spend"})
    )
    total = woy_lob["spend"].sum()
    if total == 0:
        continue

    sp_opt = sp_peak = sp_trough = sp_other = 0.0
    for row in woy_lob.itertuples():
        w, s = row.woy, row.spend
        if   w in set(opt_wks):    sp_opt    += s
        elif w in set(PEAK_WKS):   sp_peak   += s
        elif w in set(TROUGH_WKS): sp_trough += s
        else:                      sp_other  += s

    alpha_val = lob_alpha.get((funnel, lob_label), funnel_decay[funnel])

    lob_rows.append({
        "Funnel"               : funnel,
        "LOB"                  : lob_label,
        "Confidence"           : confidence,
        "Alpha (decay)"        : round(alpha_val, 4),
        "Optimal lag (wks)"    : f"{lag_min}-{lag_max}",
        "Optimal %"            : round(sp_opt    / total * 100, 1),
        "Neutral %"            : round(sp_other  / total * 100, 1),
        "Peak wk % (too late)" : round(sp_peak   / total * 100, 1),
        "Trough %"             : round(sp_trough / total * 100, 1),
        "Avg wkly spend $K"    : round(woy_lob["spend"].mean() / 1e3, 2),
    })

lob_df = (
    pd.DataFrame(lob_rows)
    .sort_values("Optimal %", ascending=False)
    .reset_index(drop=True)
)
lob_df["Annual Spend $K"] = lob_df["Avg wkly spend $K"] * 52

print(f"\nLOB rows built: {len(lob_df)}")

ipy_display(
    lob_df
    .style
    .set_caption("EXH LOB Spend Zone Breakdown — Empirically Derived Windows")
    .set_properties(**{"text-align": "center"})
    .map(lambda v: "color: red" if isinstance(v, str) and "geo-test" in v else "")
    .set_table_styles([
        {"selector": "caption",
         "props": [("font-size", "13px"), ("font-weight", "bold"),
                   ("color", "#002B5C"), ("text-align", "left")]},
        {"selector": "th",
         "props": [("background-color", "#002B5C"), ("color", "white"),
                   ("font-weight", "bold"), ("text-align", "center")]},
    ])
)

# ── Chart ─────────────────────────────────────────────────────────────────────
MIN_AVG_SPEND_K = 1.0
plot_df = (
    lob_df[lob_df["Avg wkly spend $K"] >= MIN_AVG_SPEND_K]
    .iloc[::-1]
    .reset_index(drop=True)
)
plot_df["label"] = plot_df["Funnel"] + " | " + plot_df["LOB"]

# Compute annual $ in each zone for stacked bars
plot_df["Optimal $K"]    = plot_df["Annual Spend $K"] * plot_df["Optimal %"] / 100
plot_df["Neutral $K"]    = plot_df["Annual Spend $K"] * plot_df["Neutral %"] / 100
plot_df["Peak wk $K"]    = plot_df["Annual Spend $K"] * plot_df["Peak wk % (too late)"] / 100
plot_df["Trough $K"]     = plot_df["Annual Spend $K"] * plot_df["Trough %"] / 100

n = len(plot_df)
y = np.arange(n)

fig, ax = plt.subplots(figsize=(16, 9))
fig.patch.set_facecolor(WHITE)
lefts = np.zeros(n)

zone_config = [
    ("Optimal",            "Optimal $K",   CPX_GREEN),
    ("Neutral",            "Neutral $K",   CPX_MID),
    ("Peak wk (too late)", "Peak wk $K",   CPX_GOLD),
    ("Trough",             "Trough $K",    CPX_RED),
]

for label, col_key, color in zone_config:
    vals = plot_df[col_key].values
    ax.barh(y, vals, left=lefts, height=0.6,
            color=color, alpha=0.88, label=label)
    for i, (v, l) in enumerate(zip(vals, lefts)):
        if v >= 20:
            txt_color = CPX_DGREY if color in (CPX_MID, CPX_GOLD) else WHITE
            ax.text(l + v / 2, i, f"${v:.0f}K",
                    ha="center", va="center",
                    fontsize=8.5, fontweight="bold", color=txt_color)
    lefts += vals

# Right axis — total annual + confidence tier
max_bar = lefts.max()
for i, row in plot_df.iterrows():
    conf_color = (
        CPX_RED   if "Low"    in row["Confidence"] else
        CPX_GOLD  if "Medium" in row["Confidence"] else
        CPX_GREEN
    )
    ax.text(
        max_bar * 1.03, i,
        f"Total: ${row['Annual Spend $K']:.0f}K/yr  [{row['Confidence']}]",
        va="center", fontsize=7.5, color=conf_color
    )

ax.set_yticks(y)
ax.set_yticklabels(
    [f"{row['Funnel']} | {row['LOB']}" for _, row in plot_df.iterrows()],
    fontsize=9
)
ax.set_xlabel("Annual Spend ($K)",
              fontsize=12, fontweight="bold", color=CPX_DGREY)
ax.set_xlim(0, max_bar * 1.45)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:.0f}K"))
ax.set_title(
    "EXH MMM — LOB Spend Zone Breakdown (Annual $)\n"
    "Sorted by % in optimal pre-peak window  |  "
    "Right axis = total annual spend & confidence tier",
    fontsize=14, fontweight="bold", color="black", loc="center"
)
ax.legend(loc="lower right", fontsize=10, frameon=True, framealpha=0.9)
ax.grid(False)
for spine in ["top", "right", "left", "bottom"]:
    ax.spines[spine].set_visible(True)
    ax.spines[spine].set_color(CPX_DGREY)

fig.text(
    0.5, -0.01,
    f"Peak weeks derived empirically from WOY attendance data  |  "
    f"Trough = post-NYE (wk 2-6) + back-to-school (wk 37-46)  |  "
    f"Optimal windows from posterior mean adstock decay (mu_alpha) per funnel",
    ha="center", fontsize=7.5, color=CPX_MGREY, style="italic"
)

plt.tight_layout()
plt.savefig(
    f"{OUT_PATH}EXH_flighting_lob_zone_breakdown_v2.png",
    dpi=150, bbox_inches="tight"
)
plt.show()
print("✓ Saved: EXH_flighting_lob_zone_breakdown_v2.png")


# COMMAND ----------

# DBTITLE 1,EXH LOB Flighting Zone Breakdown Per Year
# ── EXH LOB FLIGHTING ZONE BREAKDOWN — PER YEAR (2023, 2024, 2025) ───────────
# One horizontal stacked bar chart per year showing LOB Spend Zone Breakdown.
# Depends on: df_fl, media_cols, FUNNELS, PEAK_WKS, TROUGH_WKS,
#             funnel_lob_cols, lob_alpha, lob_lags, funnel_decay, funnel_lags,
#             CONFIDENCE_TIERS, get_optimal_weeks from Cell 31.

MIN_AVG_SPEND_K_YR = 1.0  # filter out LOBs below this threshold

for year in [2023, 2024, 2025]:
    yr_data = df_fl[df_fl["year"] == year].copy()
    yr_data["woy"] = yr_data["week_date"].dt.isocalendar().week.astype(int)

    lob_rows_yr = []
    for (funnel, lob_raw), cols in funnel_lob_cols.items():
        if not cols:
            continue
        lob_label  = lob_raw.strip()
        confidence = CONFIDENCE_TIERS.get((funnel, lob_label), "Medium")

        # LOB-specific or funnel-level lag
        lag_min, lag_max = lob_lags.get(
            (funnel, lob_label),
            funnel_lags[funnel]
        )
        opt_wks = get_optimal_weeks(lag_min, lag_max)

        woy_lob = (
            yr_data
            .groupby("woy")[cols]
            .mean()
            .sum(axis=1)
            .reset_index()
            .rename(columns={0: "spend"})
        )
        total = woy_lob["spend"].sum()
        if total == 0:
            continue

        sp_opt = sp_peak = sp_trough = sp_other = 0.0
        for row in woy_lob.itertuples():
            w, s = row.woy, row.spend
            if   w in set(opt_wks):    sp_opt    += s
            elif w in set(PEAK_WKS):   sp_peak   += s
            elif w in set(TROUGH_WKS): sp_trough += s
            else:                      sp_other  += s

        alpha_val = lob_alpha.get((funnel, lob_label), funnel_decay[funnel])
        n_weeks   = yr_data["woy"].nunique()
        annual_spend_k = yr_data[cols].sum().sum() / 1e3

        lob_rows_yr.append({
            "Funnel"               : funnel,
            "LOB"                  : lob_label,
            "Confidence"           : confidence,
            "Alpha (decay)"        : round(alpha_val, 4),
            "Optimal lag (wks)"    : f"{lag_min}-{lag_max}",
            "Optimal %"            : round(sp_opt    / total * 100, 1),
            "Neutral %"            : round(sp_other  / total * 100, 1),
            "Peak wk % (too late)" : round(sp_peak   / total * 100, 1),
            "Trough %"             : round(sp_trough / total * 100, 1),
            "Annual Spend $K"      : round(annual_spend_k, 1),
            "Avg wkly spend $K"    : round((annual_spend_k / n_weeks) if n_weeks > 0 else 0, 2),
        })

    lob_df_yr = (
        pd.DataFrame(lob_rows_yr)
        .sort_values("Optimal %", ascending=False)
        .reset_index(drop=True)
    )

    # ── Styled table ──────────────────────────────────────────────────────────
    ipy_display(
        lob_df_yr
        .style
        .set_caption(f"EXH LOB Spend Zone Breakdown — {year}")
        .set_properties(**{"text-align": "center"})
        .map(lambda v: "color: red" if isinstance(v, str) and "geo-test" in v else "")
        .set_table_styles([
            {"selector": "caption",
             "props": [("font-size", "13px"), ("font-weight", "bold"),
                       ("color", "#002B5C"), ("text-align", "left")]},
            {"selector": "th",
             "props": [("background-color", "#002B5C"), ("color", "white"),
                       ("font-weight", "bold"), ("text-align", "center")]},
        ])
    )

    # ── Horizontal Stacked Bar Chart ──────────────────────────────────────────
    plot_df_yr = (
        lob_df_yr[lob_df_yr["Avg wkly spend $K"] >= MIN_AVG_SPEND_K_YR]
        .iloc[::-1]
        .reset_index(drop=True)
    )

    if plot_df_yr.empty:
        print(f"  {year}: No LOBs above spend threshold — skipping chart.")
        continue

    plot_df_yr["label"]      = plot_df_yr["Funnel"] + " | " + plot_df_yr["LOB"]
    plot_df_yr["Optimal $K"] = plot_df_yr["Annual Spend $K"] * plot_df_yr["Optimal %"] / 100
    plot_df_yr["Neutral $K"] = plot_df_yr["Annual Spend $K"] * plot_df_yr["Neutral %"] / 100
    plot_df_yr["Peak wk $K"] = plot_df_yr["Annual Spend $K"] * plot_df_yr["Peak wk % (too late)"] / 100
    plot_df_yr["Trough $K"]  = plot_df_yr["Annual Spend $K"] * plot_df_yr["Trough %"] / 100

    n = len(plot_df_yr)
    y_pos = np.arange(n)

    fig, ax = plt.subplots(figsize=(16, max(9, n * 0.45)))
    fig.patch.set_facecolor(WHITE)
    lefts = np.zeros(n)

    zone_config = [
        ("Optimal",            "Optimal $K",   CPX_GREEN),
        ("Neutral",            "Neutral $K",   CPX_MID),
        ("Peak wk (too late)", "Peak wk $K",   CPX_GOLD),
        ("Trough",             "Trough $K",    CPX_RED),
    ]

    for label, col_key, color in zone_config:
        vals = plot_df_yr[col_key].values
        ax.barh(y_pos, vals, left=lefts, height=0.6,
                color=color, alpha=0.88, label=label)
        for i, (v, l) in enumerate(zip(vals, lefts)):
            if v >= 20:
                txt_color = CPX_DGREY if color in (CPX_MID, CPX_GOLD) else WHITE
                ax.text(l + v / 2, i, f"${v:.0f}K",
                        ha="center", va="center",
                        fontsize=8.5, fontweight="bold", color=txt_color)
        lefts += vals

    # Right axis — total + confidence tier
    max_bar = lefts.max()
    for i, row in plot_df_yr.iterrows():
        conf_color = (
            CPX_RED   if "Low"    in row["Confidence"] else
            CPX_GOLD  if "Medium" in row["Confidence"] else
            CPX_GREEN
        )
        ax.text(
            max_bar * 1.03, i,
            f"Total: ${row['Annual Spend $K']:.0f}K  [{row['Confidence']}]",
            va="center", fontsize=7.5, color=conf_color
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(
        [f"{row['Funnel']} | {row['LOB']}" for _, row in plot_df_yr.iterrows()],
        fontsize=9
    )
    ax.set_xlabel("Annual Spend ($K)",
                  fontsize=12, fontweight="bold", color=CPX_DGREY)
    ax.set_xlim(0, max_bar * 1.45)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:.0f}K"))
    ax.set_title(
        f"EXH MMM — LOB Spend Zone Breakdown — {year}\n"
        "Sorted by % in optimal pre-peak window  |  "
        "Right axis = total spend & confidence tier",
        fontsize=14, fontweight="bold", color="black", loc="center"
    )
    ax.legend(loc="lower right", fontsize=10, frameon=True, framealpha=0.9)
    ax.grid(False)
    for spine in ["top", "right", "left", "bottom"]:
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_color(CPX_DGREY)

    fig.text(
        0.5, -0.01,
        f"Peak weeks: New Year (wk 1), Easter (wk 14), "
        f"Summer Peak (wk 30), Christmas (wk 52)  |  "
        f"Trough = post-NYE (wk 2-6) + Back to School (wk 37-46)  |  "
        f"Optimal windows from posterior adstock decay per LOB",
        ha="center", fontsize=7.5, color=CPX_MGREY, style="italic"
    )

    plt.tight_layout()
    plt.savefig(
        f"{OUT_PATH}EXH_flighting_lob_zone_{year}.png",
        dpi=150, bbox_inches="tight"
    )
    plt.show()
    print(f"\u2713 Saved: EXH_flighting_lob_zone_{year}.png")

print("\n\u2713 Per-year EXH LOB flighting analysis complete.")


# COMMAND ----------

# DBTITLE 1,LBE Annual Spend Overview
# ── LBE Annual Spend Overview — Standalone Block ─────────────────────────────
# Loads from lbe_mmm_results.pkl. Drop into LBE Diagnostics notebook.

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shutil
import os

RESULTS_PKL = "/Volumes/data_exploration_hub/marketing_hub/model_data/Final_Data/MMM_LBE_v2/lbe_mmm_results.pkl"
OUTPUT_DIR  = "/Volumes/data_exploration_hub/marketing_hub/model_data/Final_Data/MMM_LBE"

CPX_NAVY  = "#002B5C"
CPX_RED   = "#E31837"
CPX_GOLD  = "#F5A800"
CPX_DGREY = "#4A4A4A"
CPX_LGREY = "#F0F2F5"
WHITE     = "#FFFFFF"

def save_fig(fig, fname):
    tmp = f"/tmp/{fname}"
    fig.savefig(tmp, dpi=150, bbox_inches="tight", facecolor=WHITE)
    shutil.copy2(tmp, os.path.join(OUTPUT_DIR, fname))
    print(f"✓ Saved: {fname}")

# ── Load ──────────────────────────────────────────────────────────────────────
with open(RESULTS_PKL, "rb") as f:
    results = pickle.load(f)

df_train = results["df_train"].copy()
df_val   = results["df_val"].copy()
aw_max   = float(results["aw_max"])
co_max   = float(results["co_max"])

df_train["week_date"] = pd.to_datetime(df_train["week_date"])
df_val["week_date"]   = pd.to_datetime(df_val["week_date"])
df_all = pd.concat([df_train, df_val], ignore_index=True).sort_values("week_date")

# ── Recover raw spend from normalized adstocked inputs ────────────────────────
# tr["aw"] and tr["co"] are adstocked+normalized — we need the raw spend cols
# Raw spend columns: check what's available
spend_cols = [c for c in df_all.columns if any(
    c.lower().startswith(f) for f in ["awareness", "consideration", "aw_", "co_"]
) and "adstock" not in c.lower()]

print("Candidate spend cols:", spend_cols)

# If raw spend cols exist, use them directly
# Otherwise reconstruct from normalized adstocked via aw_max/co_max as proxy
tr_data = results["tr"]
va_data = results["va"]

aw_raw_train = np.asarray(tr_data["aw"]) * aw_max
aw_raw_val   = np.asarray(va_data["aw"]) * aw_max
co_raw_train = np.asarray(tr_data["co"]) * co_max
co_raw_val   = np.asarray(va_data["co"]) * co_max

df_train["aw_spend_proxy"] = aw_raw_train
df_train["co_spend_proxy"] = co_raw_train
df_val["aw_spend_proxy"]   = aw_raw_val
df_val["co_spend_proxy"]   = co_raw_val

df_all = pd.concat([df_train, df_val], ignore_index=True).sort_values("week_date")
df_all["year"] = df_all["week_date"].dt.year

# ── Annual spend aggregation ──────────────────────────────────────────────────
annual_spend = (
    df_all[df_all["year"].isin([2023, 2024, 2025])]
    .groupby("year")[["aw_spend_proxy", "co_spend_proxy"]]
    .sum()
    .rename(columns={"aw_spend_proxy": "Awareness", "co_spend_proxy": "Consideration"})
)
annual_spend["Total"] = annual_spend["Awareness"] + annual_spend["Consideration"]
annual_spend_M = annual_spend / 1e6

print("\nLBE Annual Spend ($M):")
print(annual_spend_M.round(2))

# ── Table display ─────────────────────────────────────────────────────────────
from IPython.display import display as ipy_display

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

# Stacked bars
b1 = ax.bar(x, aw_vals, width, label="Awareness", color=CPX_NAVY)
b2 = ax.bar(x, co_vals, width, label="Consideration", color=CPX_RED,
            bottom=aw_vals)

# ── Segment labels ────────────────────────────────────────────────────────────
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
                fontsize=10, fontweight="bold", color=WHITE
            )

label_segments(b1, [0, 0, 0])
label_segments(b2, aw_vals)

# ── Total labels on top ───────────────────────────────────────────────────────
totals = aw_vals + co_vals
for i, total in enumerate(totals):
    ax.text(
        i, total + 0.06,
        f"${total:.2f}M",
        ha="center", va="bottom",
        fontsize=12, fontweight="bold", color=CPX_DGREY
    )

# ── Styling ───────────────────────────────────────────────────────────────────
ax.set_xticks(x)
ax.set_xticklabels([str(y) for y in years], fontsize=12)
ax.set_ylabel("Annual Media Spend ($M)")
ax.set_xlabel("Year")
ax.set_ylim(0, totals.max() * 1.4)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:.1f}M"))
ax.set_title("LBE MMM — Annual Media Spend by Funnel (2023–2025)")
ax.legend(loc="upper right")

plt.tight_layout()
save_fig(fig, "lbe_b0_annual_spend_by_funnel.png")
plt.show()


# COMMAND ----------

# DBTITLE 1,LBE Spend by Channel & Funnel Per Year
# ── LBE SPEND BY CHANNEL — STACKED BY FUNNEL — ONE PLOT PER YEAR ─────────────
# (Uses df_all, annual data from cell 3)
import matplotlib.ticker as mticker

# ── Channel × Funnel mapping ──────────────────────────────────────────────────
channel_funnel_cols = {
    ("TRR", "Awareness")    : "Awareness_TRR",
    ("TRR", "Consideration"): "Consideration_TRR",
    ("PDM", "Awareness")    : "Awareness_PDM",
    ("PDM", "Consideration"): "Consideration_PDM",
}

funnels  = ["Awareness", "Consideration"]
channels = ["TRR", "PDM"]

funnel_colors = {
    "Awareness"    : CPX_NAVY,
    "Consideration": CPX_RED,
}

# ── Aggregate Spend by Channel, Funnel, Year ──────────────────────────────────
rows = []
for (channel, funnel), col in channel_funnel_cols.items():
    for year in [2023, 2024, 2025]:
        spend = df_all.loc[df_all["year"] == year, col].sum()
        rows.append({
            "Channel": channel,
            "Funnel" : funnel,
            "Year"   : year,
            "Spend_M": spend / 1e6,
        })

spend_df = pd.DataFrame(rows)
spend_df = spend_df[spend_df["Spend_M"] > 0.001].copy()

# ── One Plot Per Year ─────────────────────────────────────────────────────────
for year in [2023, 2024, 2025]:
    ydata = spend_df[spend_df["Year"] == year].copy()

    # Sort channels by total spend descending
    ch_totals = ydata.groupby("Channel")["Spend_M"].sum().sort_values(ascending=False)
    ch_order  = ch_totals.index.tolist()
    x         = np.arange(len(ch_order))

    fig, ax = plt.subplots(figsize=get_figsize("16:9"))
    fig.patch.set_facecolor(WHITE)

    bottoms = np.zeros(len(ch_order))

    for funnel in funnels:
        heights = []
        for ch in ch_order:
            val = ydata.loc[
                (ydata["Funnel"] == funnel) & (ydata["Channel"] == ch), "Spend_M"
            ]
            heights.append(float(val.values[0]) if len(val) > 0 else 0.0)

        bars = ax.bar(
            x, heights,
            bottom=bottoms,
            label=funnel,
            color=funnel_colors[funnel],
            width=0.55,
            edgecolor=WHITE,
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
                    fontsize=10, fontweight="bold", color=WHITE
                )
        bottoms += np.array(heights)

    # Total labels on top
    for i, total in enumerate(bottoms):
        if total > 0:
            ax.text(
                x[i], total + 0.01,
                f"${total:.2f}M",
                ha="center", va="bottom",
                fontsize=12, fontweight="bold", color=CPX_DGREY
            )

    ax.set_xticks(x)
    ax.set_xticklabels(ch_order, rotation=0, ha="center", fontsize=12)
    ax.set_ylabel("Annual Spend ($M)")
    ax.set_xlabel("Channel")
    ax.set_ylim(0, bottoms.max() * 1.4)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:.2f}M"))
    ax.set_title(f"LBE Annual Media Spend by Channel & Funnel — {year}", loc="center")
    ax.legend(
        title="Funnel", fontsize=9, title_fontsize=10,
        loc="upper right"
    )

    plt.tight_layout()
    save_fig(fig, f"lbe_spend_channel_funnel_{year}.png")
    plt.show()


# COMMAND ----------

# DBTITLE 1,LBE Spend by Channel & Funnel Combined
# ── LBE SPEND BY CHANNEL & FUNNEL — ALL YEARS COMBINED ────────────────────────
import matplotlib.ticker as mticker

# ── Build spend matrix: Channel × Funnel × Year ──────────────────────────────
channel_funnel_cols = {
    ("TRR", "Awareness")    : "Awareness_TRR",
    ("TRR", "Consideration"): "Consideration_TRR",
    ("PDM", "Awareness")    : "Awareness_PDM",
    ("PDM", "Consideration"): "Consideration_PDM",
}

funnels  = ["Awareness", "Consideration"]
channels = ["TRR", "PDM"]
years    = [2023, 2024, 2025]

funnel_colors = {
    "Awareness"    : CPX_NAVY,
    "Consideration": CPX_RED,
}

rows = []
for (channel, funnel), col in channel_funnel_cols.items():
    for year in years:
        spend = df_all.loc[df_all["year"] == year, col].sum()
        rows.append({"Channel": channel, "Funnel": funnel, "Year": year, "Spend_M": spend / 1e6})

spend_df_combined = pd.DataFrame(rows)

# ── Layout: Years on x-axis, channels grouped within each year ────────────────
n_years    = len(years)
n_channels = len(channels)
bar_width  = 0.25
group_gap  = 0.15

fig, ax = plt.subplots(figsize=(11, 9))
fig.patch.set_facecolor(WHITE)

all_totals = []  # collect (x_positions, totals, channel) for uniform labels

for ci, channel in enumerate(channels):
    # Position offset for this channel within each year group
    offset = (ci - (n_channels - 1) / 2) * (bar_width + group_gap)
    x_positions = np.arange(n_years) + offset

    bottoms = np.zeros(n_years)

    for funnel in funnels:
        heights = []
        for year in years:
            val = spend_df_combined.loc[
                (spend_df_combined["Channel"] == channel) &
                (spend_df_combined["Funnel"] == funnel) &
                (spend_df_combined["Year"] == year),
                "Spend_M"
            ]
            heights.append(float(val.values[0]) if len(val) > 0 else 0.0)

        heights = np.array(heights)

        bars = ax.bar(
            x_positions, heights,
            width=bar_width,
            bottom=bottoms,
            color=funnel_colors[funnel],
            edgecolor=WHITE,
            linewidth=0.5,
            label=funnel if ci == 0 else "",
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
                    fontsize=9, fontweight="bold", color=WHITE
                )

        bottoms += heights

    # Total labels just above each bar
    for i, total in enumerate(bottoms):
        if total > 0:
            ax.text(
                x_positions[i], total + 0.04,
                f"${total:.2f}M",
                ha="center", va="bottom",
                fontsize=10, fontweight="bold", color=CPX_DGREY
            )

    # Track max heights for year label placement
    all_totals.append((x_positions.copy(), bottoms.copy(), channel))

    # Channel label close to x-axis
    for i, xp in enumerate(x_positions):
        ax.text(
            xp, -0.03,
            channel,
            ha="center", va="top",
            fontsize=9, fontweight="bold", color=CPX_DGREY,
            transform=ax.get_xaxis_transform()
        )

# ── Uniform year labels above all bars ────────────────────────────────────────
max_height = max(t.max() for _, t, _ in all_totals)
label_y = max_height + 0.55  # uniform y with padding above tallest bar

for yi, year in enumerate(years):
    ax.text(
        yi, label_y,
        str(year),
        ha="center", va="bottom",
        fontsize=13, fontweight="bold", color="black"
    )

# ── Styling ───────────────────────────────────────────────────────────────────
ax.set_xticks(np.arange(n_years))
ax.set_xticklabels([""] * n_years)  # hide x-tick labels (years shown above bars)
ax.set_ylabel("Annual Spend ($M)")
ax.set_xlabel("")
ax.set_ylim(0, spend_df_combined.groupby(["Channel", "Year"])["Spend_M"].sum().max() * 1.5)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:.2f}M"))
ax.set_title("LBE Annual Media Spend by Channel & Funnel (2023–2025)", loc="center")
ax.legend(loc="upper right", title="Funnel", fontsize=9, title_fontsize=10)

plt.tight_layout()
save_fig(fig, "lbe_spend_channel_funnel_combined.png")
plt.show()


# COMMAND ----------

# DBTITLE 1,LBE Spend by Channel & Funnel Data Tables
# ── DATA TABLES: LBE Spend by Channel & Funnel — One Table Per Year ───────────
# Values in $10K units (e.g., 31.5 = $315,000)
for year in [2023, 2024, 2025]:
    pivot = (
        spend_df_combined[spend_df_combined["Year"] == year]
        .pivot_table(index="Channel", columns="Funnel", values="Spend_M", aggfunc="sum", fill_value=0)
        .reindex(columns=["Awareness", "Consideration"])
        .fillna(0)
    )
    # Convert from $M to $10K (multiply by 100)
    pivot = pivot * 100
    pivot["Total"] = pivot.sum(axis=1)
    pivot = pivot.sort_values("Total", ascending=False)
    pivot = pivot.reset_index()
    pivot.columns.name = None

    # Round to 2 decimal places
    num_cols = ["Awareness", "Consideration", "Total"]
    pivot[num_cols] = pivot[num_cols].round(2)

    print(f"\n{'\u2550'*60}")
    print(f"  LBE Annual Media Spend by Channel & Funnel \u2014 {year}")
    print(f"  (values in $10K)")
    print(f"{'\u2550'*60}")
    display(pivot)


# COMMAND ----------

# DBTITLE 1,LBE Revenue Breakdown by Year
# ── LBE REVENUE BREAKDOWN BY YEAR ───────────────────────────────────────────
# (Uses df_all from cell 3)
import matplotlib.ticker as mticker

# ── Aggregate Annual Revenue ────────────────────────────────────────────────
rev_df_lbe = (
    df_all[df_all["year"].isin([2023, 2024, 2025])]
    .groupby("year")["LBE_Revenue"]
    .sum()
    .reset_index()
)
rev_df_lbe["Revenue_M"] = rev_df_lbe["LBE_Revenue"] / 1e6

# Add 2026 forecast (hard-coded)
forecast_2026 = 148_546_568
forecast_row = pd.DataFrame([{"year": 2026, "LBE_Revenue": forecast_2026, "Revenue_M": forecast_2026 / 1e6}])
rev_df_lbe = pd.concat([rev_df_lbe, forecast_row], ignore_index=True)
rev_df_lbe["YoY_pct"] = rev_df_lbe["Revenue_M"].pct_change() * 100

# ── Plot ─────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=get_figsize("16:9"))
fig.patch.set_facecolor(WHITE)

x     = np.arange(len(rev_df_lbe))
width = 0.5

bars = ax.bar(
    x,
    rev_df_lbe["Revenue_M"],
    width=width,
    color="black",
    edgecolor=WHITE,
    linewidth=0.5
)

# Value labels on top
for i, (bar, row) in enumerate(zip(bars, rev_df_lbe.itertuples())):
    h = bar.get_height()
    # Revenue value label
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        h + 1.0,
        f"${h:.2f}M",
        ha="center", va="bottom",
        fontsize=12, fontweight="bold", color=CPX_DGREY
    )
    # YoY change label (stacked above the value)
    if i > 0:
        yoy = row.YoY_pct
        color = CPX_RED if yoy < 0 else "black"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + 5.0,
            f"{yoy:+.1f}% YoY",
            ha="center", va="bottom",
            fontsize=10, fontweight="bold", color=color
        )

ax.set_xticks(x)
ax.set_xticklabels(rev_df_lbe["year"].astype(str), fontsize=12)
ax.set_ylabel("Total Annual Revenue ($M)")
ax.set_xlabel("Year")
ax.set_ylim(0, rev_df_lbe["Revenue_M"].max() * 1.25)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:.0f}M"))
ax.set_title("LBE Total Annual Revenue (2023–2026F)", loc="center")

plt.tight_layout()
save_fig(fig, "lbe_revenue_by_year.png")
plt.show()

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
    .set_caption("LBE Total Annual Revenue (2023–2026F)")
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

# DBTITLE 1,LBE Revenue Data Table
# ── LBE Revenue Data Table ─────────────────────────────────────────────────
rev_table_lbe = rev_df_lbe[["year", "Revenue_M", "YoY_pct"]].copy()
rev_table_lbe.columns = ["Year", "Total Revenue ($M)", "YoY Change (%)"]
rev_table_lbe["Total Revenue ($M)"] = rev_table_lbe["Total Revenue ($M)"].round(2)
rev_table_lbe["YoY Change (%)"] = rev_table_lbe["YoY Change (%)"].round(1)

print(f"{'\u2550'*60}")
print(f"  LBE Total Annual Revenue (2023\u20132026F)")
print(f"{'\u2550'*60}")
from IPython.display import display as ipy_display
ipy_display(rev_table_lbe)


# COMMAND ----------

# DBTITLE 1,LBE Calendar Event Revenue Contribution
# ── LBE CALENDAR EVENT REVENUE CONTRIBUTION ────────────────────────────────
import matplotlib.patheffects as pe

# ── Extract model components ────────────────────────────────────────────────
draws    = results["draws"]
bc       = results["bc"]
n_draws  = int(results["n_draws"])
y_mean   = float(results["y_mean"])
y_std    = float(results["y_std"])
cal_cols = list(results["CAL_COLS"])
tr       = results["tr"]
va       = results["va"]

cal_display = [
    "New Year Wk", "March Break", "March Adj", "Pre-Xmas",
    "Summer Peak", "Back to School", "Spring Lull", "Holiday", "Long Weekend"
]

# ── Prediction helpers ─────────────────────────────────────────────────────
def adstock_np(x, decay, max_lag=10):
    w = decay ** np.arange(max_lag)
    w = w / w.sum()
    out = np.zeros(len(x), dtype=float)
    for lag in range(max_lag):
        out[lag:] += w[lag] * x[:len(x) - lag]
    return out

def predict_revenue(data, sample_size=1000, seed=42):
    sample_size = min(sample_size, n_draws)
    rng = np.random.default_rng(seed)
    idx = rng.choice(n_draws, sample_size, replace=False)
    preds = np.zeros((sample_size, len(data["aw"])), dtype=float)
    for j, i in enumerate(idx):
        aw_ads = adstock_np(np.asarray(data["aw"]), float(draws["aw_decay"][i]))
        co_ads = adstock_np(np.asarray(data["co"]), float(draws["co_decay"][i]))
        aw_sat = np.log1p(aw_ads * float(draws["aw_scale"][i]))
        co_sat = np.log1p(co_ads * float(draws["co_scale"][i]))
        cal_eff = np.asarray(data["cal"]) @ bc[i]
        mu = (
            float(draws["intercept"][i])
            + float(draws["beta_aw"][i]) * aw_sat
            + float(draws["beta_co"][i]) * co_sat
            + float(draws["beta_trend"][i]) * np.asarray(data["t"])
            + float(draws["beta_temp"][i]) * np.asarray(data["temp"])
            + cal_eff
        )
        preds[j] = np.exp(mu * y_std + y_mean)
    return preds

# ── Compute full_mean (posterior mean prediction per week) ────────────────
full_data = {
    "aw":   np.concatenate([np.asarray(tr["aw"]),   np.asarray(va["aw"])]),
    "co":   np.concatenate([np.asarray(tr["co"]),   np.asarray(va["co"])]),
    "temp": np.concatenate([np.asarray(tr["temp"]), np.asarray(va["temp"])]),
    "cal":  np.concatenate([np.asarray(tr["cal"]),  np.asarray(va["cal"])]),
    "t":    np.concatenate([np.asarray(tr["t"]),    np.asarray(va["t"])]),
}
full_preds = predict_revenue(full_data, sample_size=1000, seed=44)
full_mean  = full_preds.mean(axis=0)

# ── Counterfactual lift per calendar event ─────────────────────────────────
df_all["year"] = df_all["week_date"].dt.year

cal_lift_m = np.zeros((n_draws, len(cal_cols)))

for j, col in enumerate(cal_cols):
    event_mask  = df_all[col].values.astype(float)
    event_pred  = (full_mean * event_mask).sum()
    bc_j        = bc[:, j]
    lift_factor = 1.0 - np.exp(-bc_j * y_std)
    cal_lift_m[:, j] = event_pred * lift_factor / 1e6

cal_lift_mean = cal_lift_m.mean(axis=0)
cal_lift_lo   = np.percentile(cal_lift_m,  2.5, axis=0)
cal_lift_hi   = np.percentile(cal_lift_m, 97.5, axis=0)

event_weeks_n  = np.array([int(df_all[col].sum()) for col in cal_cols])
wkly_lift_mean = np.where(event_weeks_n > 0, cal_lift_mean / event_weeks_n, 0.0)

# Per-year breakdown
years_uniq = sorted(df_all["year"].unique())
yr_lift_mean = {}
for yr in years_uniq:
    yr_mask = (df_all["year"] == yr).values.astype(float)
    yr_row  = np.zeros(len(cal_cols))
    for j, col in enumerate(cal_cols):
        ev_mask_yr  = df_all[col].values.astype(float) * yr_mask
        ev_pred_yr  = (full_mean * ev_mask_yr).sum()
        yr_row[j]   = ev_pred_yr * (1 - np.exp(-bc[:, j] * y_std)).mean() / 1e6
    yr_lift_mean[yr] = yr_row

# ── Summary Table ───────────────────────────────────────────────────────────
cal_summary = pd.DataFrame({
    "Calendar Event":     cal_display,
    "Weeks Active":       event_weeks_n,
    "Total Lift $M":      np.round(cal_lift_mean, 2),
    "95% CI Lo $M":       np.round(cal_lift_lo,   2),
    "95% CI Hi $M":       np.round(cal_lift_hi,   2),
    "Avg per Week $M":    np.round(wkly_lift_mean, 2),
    **{f"{yr} $M": np.round(yr_lift_mean[yr], 2) for yr in years_uniq},
}).sort_values("Total Lift $M", ascending=False, key=abs).reset_index(drop=True)

print("Calendar event revenue contribution (3-year total, 95% CI from 8,000 posterior draws):")
display(cal_summary)

# ── Sort for charts ─────────────────────────────────────────────────────────
sort_idx  = np.argsort(cal_lift_mean)[::-1]
s_display = [cal_display[i]   for i in sort_idx]
s_means   = cal_lift_mean[sort_idx]
s_lo      = cal_lift_lo[sort_idx]
s_hi      = cal_lift_hi[sort_idx]
s_wkly    = wkly_lift_mean[sort_idx]
s_colors  = [CPX_NAVY if m >= 0 else CPX_RED for m in s_means]

x = np.arange(len(cal_cols))

# ── Chart 1: 3-Year Total Contribution with 95% CI ─────────────────────
fig, ax = plt.subplots(figsize=(16, 9))
bars = ax.bar(x, s_means, color=s_colors, alpha=0.85, width=0.6)
ax.errorbar(
    x, s_means,
    yerr=[s_means - s_lo, s_hi - s_means],
    fmt="none", color=CPX_DGREY, capsize=6, lw=1.8,
)
for bar, m, hi_v, lo_v in zip(bars, s_means, s_hi, s_lo):
    y_pos = hi_v + 0.05 if m >= 0 else lo_v - 0.15
    ax.text(
        bar.get_x() + bar.get_width() / 2, y_pos,
        f"${m:+.1f}M",
        ha="center", va="bottom" if m >= 0 else "top",
        fontsize=10, fontweight="bold",
        color=CPX_NAVY if m >= 0 else CPX_RED,
    )
ax.axhline(0, color="black", lw=0.9)
ax.set_xticks(x)
ax.set_xticklabels(s_display, rotation=30, ha="right", fontsize=11)
ax.set_ylabel("Revenue Contribution ($M, 3-year total)", fontsize=12)
ax.set_title(
    "LBE MMM \u2014 Calendar Event Revenue Contribution (3-Year Total)\n"
    "Bars = posterior mean  |  Error bars = 95% CI  |  Sorted by magnitude",
    fontsize=15, fontweight="bold", color="black",
)
ax.grid(False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
save_fig(fig, "lbe_calendar_revenue_contribution_total.png")
plt.show()

# ── Chart 2: Average Weekly Lift per Event ──────────────────────────────
fig, ax = plt.subplots(figsize=(16, 9))
bars2 = ax.bar(x, s_wkly, color=s_colors, alpha=0.85, width=0.6)
for bar, m in zip(bars2, s_wkly):
    if abs(m) > 0.01:
        y_pos = m + (0.005 if m >= 0 else -0.015)
        ax.text(
            bar.get_x() + bar.get_width() / 2, y_pos,
            f"${m * 1e3:.0f}K/wk",
            ha="center", va="bottom" if m >= 0 else "top",
            fontsize=10, fontweight="bold",
            color=CPX_NAVY if m >= 0 else CPX_RED,
        )
ax.axhline(0, color="black", lw=0.9)
ax.set_xticks(x)
ax.set_xticklabels(s_display, rotation=30, ha="right", fontsize=11)
ax.set_ylabel("Average Revenue Lift per Active Week ($M)", fontsize=12)
ax.set_title(
    "LBE MMM \u2014 Calendar Event Average Weekly Lift\n"
    "Total lift \u00f7 weeks active  |  Sorted by 3-year magnitude",
    fontsize=15, fontweight="bold", color="black",
)
ax.grid(False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
save_fig(fig, "lbe_calendar_revenue_contribution_weekly.png")
plt.show()


# COMMAND ----------

# DBTITLE 1,LBE Per-Channel Annual Spend & Revenue Contribution
# ── LBE PER-CHANNEL ANNUAL SPEND & REVENUE CONTRIBUTION ─────────────────────
# One chart per channel (TRR, PDM). For each year: stacked spend bar + revenue
# contribution bar side by side. Uses per-year funnel ROI from model decomp.
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker

FUNNEL_COLOR = {"Awareness": CPX_NAVY, "Consideration": CPX_RED}
FUNNELS_LBE  = ["Awareness", "Consideration"]
CHANNELS     = ["TRR", "PDM"]
years_plot   = [2023, 2024, 2025]

# ── Per-year ROI from decomp (compute from posterior if not pre-saved) ─────
if "decomp" in results:
    decomp_lbe = results["decomp"]
    aw_roi_yr = {d["year"]: d["aw_roi"] for d in decomp_lbe}
    co_roi_yr = {d["year"]: d["co_roi"] for d in decomp_lbe}
else:
    # Estimate ROI from posterior means: revenue_per_dollar
    aw_roi_yr = {yr: float(np.mean(draws["beta_aw"])) * y_std for yr in years_plot}
    co_roi_yr = {yr: float(np.mean(draws["beta_co"])) * y_std for yr in years_plot}
roi_yr = {"Awareness": aw_roi_yr, "Consideration": co_roi_yr}

# ── Channel × Funnel spend columns ─────────────────────────────────────────
chan_funnel_col = {
    ("TRR", "Awareness"):     "Awareness_TRR",
    ("TRR", "Consideration"): "Consideration_TRR",
    ("PDM", "Awareness"):     "Awareness_PDM",
    ("PDM", "Consideration"): "Consideration_PDM",
}

# ── One chart per channel ───────────────────────────────────────────────────
n_yrs = len(years_plot)
bar_w = 0.35
gap   = 0.15
group = bar_w * 2 + gap + 0.2

for channel in CHANNELS:
    # Compute spend and contribution per funnel per year
    spend_by_f = {}
    contrib_by_f = {}
    for funnel in FUNNELS_LBE:
        col = chan_funnel_col[(channel, funnel)]
        s_arr = []
        c_arr = []
        for yr in years_plot:
            spend = df_all.loc[df_all["year"] == yr, col].sum() / 1e6
            rev_contrib = spend * roi_yr[funnel].get(yr, 0)
            s_arr.append(spend)
            c_arr.append(rev_contrib)
        spend_by_f[funnel]   = np.array(s_arr)
        contrib_by_f[funnel] = np.array(c_arr)

    total_spend_yr   = sum(spend_by_f[f] for f in FUNNELS_LBE)
    total_contrib_yr = sum(contrib_by_f[f] for f in FUNNELS_LBE)

    x_spend   = np.array([i * group for i in range(n_yrs)])
    x_contrib = x_spend + bar_w + gap

    fig, ax = plt.subplots(figsize=(11, 9))

    # Stacked spend bars
    bottoms_s = np.zeros(n_yrs)
    for f in FUNNELS_LBE:
        ax.bar(x_spend, spend_by_f[f], bar_w, bottom=bottoms_s,
               color=FUNNEL_COLOR[f], alpha=0.9, edgecolor="white", lw=0.4)
        bottoms_s += spend_by_f[f]

    # Stacked contribution bars — hatched
    bottoms_c = np.zeros(n_yrs)
    for f in FUNNELS_LBE:
        ax.bar(x_contrib, contrib_by_f[f], bar_w, bottom=bottoms_c,
               color=FUNNEL_COLOR[f],
               alpha=0.55, edgecolor="white", lw=0.4, hatch="///")
        bottoms_c += contrib_by_f[f]

    # Value labels on top of bars
    for i in range(n_yrs):
        if bottoms_s[i] > 0:
            ax.annotate(f"${bottoms_s[i]:.2f}M",
                        xy=(x_spend[i], bottoms_s[i]),
                        xytext=(0, 4), textcoords="offset points",
                        ha="center", fontsize=8, color=CPX_DGREY, fontweight="bold")
        if bottoms_c[i] > 0:
            ax.annotate(f"${bottoms_c[i]:.2f}M",
                        xy=(x_contrib[i], bottoms_c[i]),
                        xytext=(0, 4), textcoords="offset points",
                        ha="center", fontsize=8, color=CPX_DGREY, fontweight="bold")

    # X-axis
    x_mid = (x_spend + x_contrib) / 2
    ax.set_xticks(x_mid)
    ax.set_xticklabels([str(y) for y in years_plot], fontsize=11, color=CPX_DGREY)
    ax.set_xlim(-0.3, x_contrib[-1] + bar_w + 0.3)

    # Single shared y-axis
    max_val = max(bottoms_s.max(), bottoms_c.max())
    ax.set_ylabel("$M", fontsize=10, color=CPX_DGREY)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:.1f}M"))
    ax.set_ylim(bottom=0, top=max_val * 1.25)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", labelsize=9)
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.2, color="#CCCCCC")

    # Legend
    handles = [mpatches.Patch(color=FUNNEL_COLOR[f], label=f) for f in FUNNELS_LBE]
    handles += [
        mpatches.Patch(facecolor="#999999", alpha=0.9, label="Spend"),
        mpatches.Patch(facecolor="#999999", alpha=0.55, hatch="///", label="Revenue Contrib"),
    ]
    ax.legend(handles=handles, fontsize=9,
              loc="upper right", framealpha=0.9, edgecolor="#CCCCCC")

    ax.set_title(
        f"{channel}\nAnnual Spend vs Revenue Contribution",
        fontsize=13, fontweight="bold", color="black", pad=10
    )

    plt.tight_layout()
    save_fig(fig, f"lbe_channel_{channel.lower()}_spend_vs_contrib.png")
    plt.show()
    plt.close(fig)

    # ── Summary table for this channel ────────────────────────────────────────
    tbl_rows = []
    for yr_idx, yr in enumerate(years_plot):
        row = {"Year": yr}
        for f in FUNNELS_LBE:
            row[f"{f} Spend ($K)"] = f"{spend_by_f[f][yr_idx] * 1000:.0f}K"
            row[f"{f} Contrib ($M)"] = round(contrib_by_f[f][yr_idx], 3)
        row["Total Spend ($K)"] = f"{total_spend_yr[yr_idx] * 1000:.0f}K"
        row["Total Contrib ($M)"] = round(total_contrib_yr[yr_idx], 3)
        tbl_rows.append(row)
    # ROI footer
    rate_row = {"Year": "ROI"}
    for f in FUNNELS_LBE:
        rate_row[f"{f} Spend ($K)"] = ""
        rate_row[f"{f} Contrib ($M)"] = f"{np.mean([roi_yr[f][yr] for yr in years_plot]):.2f}x"
    rate_row["Total Spend ($K)"] = ""
    rate_row["Total Contrib ($M)"] = ""
    tbl_rows.append(rate_row)

    print(f"\n{'\u2550'*60}")
    print(f"  {channel} \u2014 Spend & Revenue Contribution")
    print(f"{'\u2550'*60}")
    from IPython.display import display as ipy_display
    ipy_display(pd.DataFrame(tbl_rows))


# COMMAND ----------

# DBTITLE 1,LBE Diminishing Returns & Marginal Return Curves
# ── LBE DIMINISHING RETURNS & MARGINAL RETURN CURVES ───────────────────────
# Adapted from EXH logic: varies total budget (maintaining 2025 allocation
# shares), computes steady-state response curve and marginal return.
from matplotlib.lines import Line2D
import matplotlib.ticker as mticker

print("Generating aggregate response and marginal return curves...")

# ── Model components from cell 10 (already loaded) ──────────────────────
draws    = results["draws"]
n_draws  = int(results["n_draws"])
y_mean_v = float(results["y_mean"])
y_std_v  = float(results["y_std"])
aw_max_v = float(results["aw_max"])
co_max_v = float(results["co_max"])

# Average baseline context (trend, temp, calendar at their means)
tr_data = results["tr"]
va_data = results["va"]
full_data_rc = {
    "aw":   np.concatenate([np.asarray(tr_data["aw"]),   np.asarray(va_data["aw"])]),
    "co":   np.concatenate([np.asarray(tr_data["co"]),   np.asarray(va_data["co"])]),
    "t":    np.concatenate([np.asarray(tr_data["t"]),    np.asarray(va_data["t"])]),
    "temp": np.concatenate([np.asarray(tr_data["temp"]), np.asarray(va_data["temp"])]),
    "cal":  np.concatenate([np.asarray(tr_data["cal"]),  np.asarray(va_data["cal"])]),
}

t_avg    = full_data_rc["t"].mean()
temp_avg = full_data_rc["temp"].mean()
cal_avg  = full_data_rc["cal"].mean(axis=0)
bc_arr   = results["bc"]
bc_mean  = bc_arr.mean(axis=0)
cal_eff_avg = float(cal_avg @ bc_mean)


def total_media_revenue(weekly_aw_norm, weekly_co_norm, params):
    """
    Estimate weekly media-driven revenue given normalized spend levels.
    At steady-state with constant weekly spend, adstocked value = input
    (since geometric weights are normalized).
    """
    aw_sat = np.log1p(weekly_aw_norm * params["aw_scale"])
    co_sat = np.log1p(weekly_co_norm * params["co_scale"])
    base_log = (params["intercept"]
                + params["beta_trend"] * t_avg
                + params["beta_temp"] * temp_avg
                + cal_eff_avg)
    full_log = base_log + params["beta_aw"] * aw_sat + params["beta_co"] * co_sat
    base_rev = np.exp(base_log * y_std_v + y_mean_v)
    full_rev = np.exp(full_log * y_std_v + y_mean_v)
    return (full_rev - base_rev) * 52  # annual media contribution


# ── Posterior mean parameters ──────────────────────────────────────────────
params_mean = {
    "intercept":  float(draws["intercept"].mean()),
    "beta_trend": float(draws["beta_trend"].mean()),
    "beta_temp":  float(draws["beta_temp"].mean()),
    "beta_aw":    float(draws["beta_aw"].mean()),
    "beta_co":    float(draws["beta_co"].mean()),
    "aw_scale":   float(draws["aw_scale"].mean()),
    "co_scale":   float(draws["co_scale"].mean()),
}

# ── 2025 budget shares (hold allocation mix constant) ────────────────────
df_2025 = df_all[df_all["year"] == 2025]
aw_weekly_2025 = df_2025["Awareness"].mean()   # raw $ per week
co_weekly_2025 = df_2025["Consideration"].mean()
total_weekly_2025 = aw_weekly_2025 + co_weekly_2025
aw_share = aw_weekly_2025 / total_weekly_2025
co_share = co_weekly_2025 / total_weekly_2025
total_annual_2025 = total_weekly_2025 * 52

# ── X axis: annual spend from near-zero to 10x 2025 ──────────────────────
x_annual = np.linspace(52 * 100, total_annual_2025 * 10, 300)

curve_mean = np.array([
    total_media_revenue(
        (b / 52) * aw_share / aw_max_v,   # normalized aw spend
        (b / 52) * co_share / co_max_v,   # normalized co spend
        params_mean
    )
    for b in x_annual
])

# ── 89% CI from 100 posterior samples ────────────────────────────────────
n_samples  = 100
sample_idx = np.linspace(0, n_draws - 1, n_samples, dtype=int)

curve_samples = []
for si in sample_idx:
    p = {
        "intercept":  float(draws["intercept"][si]),
        "beta_trend": float(draws["beta_trend"][si]),
        "beta_temp":  float(draws["beta_temp"][si]),
        "beta_aw":    float(draws["beta_aw"][si]),
        "beta_co":    float(draws["beta_co"][si]),
        "aw_scale":   float(draws["aw_scale"][si]),
        "co_scale":   float(draws["co_scale"][si]),
    }
    curve_samples.append(np.array([
        total_media_revenue(
            (b / 52) * aw_share / aw_max_v,
            (b / 52) * co_share / co_max_v,
            p
        )
        for b in x_annual
    ]))

curve_samples = np.array(curve_samples)
curve_lo = np.quantile(curve_samples, 0.055, axis=0)
curve_hi = np.quantile(curve_samples, 0.945, axis=0)

# ── Year reference points ─────────────────────────────────────────────────
year_points = {}
for year in [2023, 2024, 2025]:
    df_yr = df_all[df_all["year"] == year]
    annual_spend = (df_yr["Awareness"].sum() + df_yr["Consideration"].sum())
    aw_wk = df_yr["Awareness"].mean() / aw_max_v
    co_wk = df_yr["Consideration"].mean() / co_max_v
    annual_rev = total_media_revenue(aw_wk, co_wk, params_mean)
    year_points[year] = {"annual_spend": annual_spend, "annual_rev": annual_rev}

YEAR_STYLES = {
    2023: {"color": CPX_DGREY,  "linestyle": "--", "linewidth": 1.2, "marker": "o"},
    2024: {"color": CPX_NAVY,   "linestyle": "-.", "linewidth": 1.2, "marker": "s"},
    2025: {"color": CPX_RED,    "linestyle": "-",  "linewidth": 1.5, "marker": "D"},
}

# ── Marginal returns & key points ────────────────────────────────────────
dx = x_annual[1] - x_annual[0]
marginal_mean = np.gradient(curve_mean, dx)
second_deriv  = np.gradient(marginal_mean, dx)
inflection_idx = next((i for i in range(1, len(second_deriv))
                       if second_deriv[i - 1] > 0 and second_deriv[i] <= 0), None)

# 95% saturation point
sat_95_idx   = np.argmin(np.abs(curve_mean - curve_mean[-1] * 0.95))
sat_95_spend = x_annual[sat_95_idx]
sat_95_rev   = curve_mean[sat_95_idx]

# ── PLOT 1: Total Response Curve (Diminishing Returns) ───────────────────
fig, ax = plt.subplots(figsize=(16, 9))
ax.plot(x_annual / 1e6, curve_mean / 1e6, color=CPX_NAVY, linewidth=2.5,
        label="Total media response (posterior mean)")
ax.fill_between(x_annual / 1e6, curve_lo / 1e6, curve_hi / 1e6,
                color=CPX_NAVY, alpha=0.12, label="89% credible interval")

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

ax.set_ylabel("Annual Media Revenue Contribution ($M)", fontsize=11)
ax.set_xlabel("Annual Media Budget ($M)", fontsize=11)
ax.set_title("LBE MMM \u2014 Total Media Response Curve\n"
             "Steady-state adstock  |  Budget shares held at 2025 mix",
             fontsize=14, fontweight="bold", color="black")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:.1f}M"))
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:.1f}M"))

year_legend = [Line2D([0], [0], color=style["color"], linestyle=style["linestyle"],
                      linewidth=style["linewidth"], marker=style["marker"], markersize=7,
                      label=f"{year} (${year_points[year]['annual_spend']/1e6:.1f}M annual)")
               for year, style in YEAR_STYLES.items()]
ax.legend(handles=year_legend, fontsize=9, loc="lower right", framealpha=0.9, edgecolor=CPX_LGREY)
ax.grid(False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.text(0.01, 0.01, "* Budget distributed using 2025 actual spend shares.",
         fontsize=7, color=CPX_DGREY, fontstyle="italic", ha="left")
plt.tight_layout()
save_fig(fig, "lbe_total_media_response_curve.png")
plt.show()

# ── PLOT 2: Marginal Return Curve ───────────────────────────────────────
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
    ax.scatter([spend_m], [yr_marg], color=style["color"],
               marker=style["marker"], s=90, zorder=6)
    ax.annotate(f"{year}\n${spend_m:.1f}M/yr\n({yr_marg:.2f} rev/$)",
                xy=(spend_m, yr_marg),
                xytext=(10, 15 - list(YEAR_STYLES.keys()).index(year) * 40),
                textcoords="offset points",
                fontsize=8, color=style["color"], fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=style["color"], lw=0.8),
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.85, ec="none"))

ax.set_ylabel("Marginal Revenue per $1 Additional Annual Spend", fontsize=11)
ax.set_xlabel("Annual Media Spend ($M)", fontsize=11)
ax.set_title("LBE MMM \u2014 Marginal Return Curve\n"
             "Revenue generated per additional $1 of annual spend",
             fontsize=14, fontweight="bold", color="black")
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:.1f}M"))

year_legend = [Line2D([0], [0], color=style["color"], linestyle=style["linestyle"],
                      linewidth=style["linewidth"], marker=style["marker"], markersize=7,
                      label=f"{year} (${year_points[year]['annual_spend']/1e6:.1f}M annual)")
               for year, style in YEAR_STYLES.items()]
ax.legend(handles=year_legend, fontsize=9, loc="upper right", framealpha=0.9, edgecolor=CPX_LGREY)
ax.grid(False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.text(0.01, 0.01, "* Budget distributed using 2025 actual spend shares.",
         fontsize=7, color=CPX_DGREY, fontstyle="italic", ha="left")
plt.tight_layout()
save_fig(fig, "lbe_marginal_return_curve.png")
plt.show()

print(f"\n\u2713 All response curves complete.")
print(f"  2025 annual spend: ${total_annual_2025/1e6:.2f}M")
print(f"  95% saturation at: ${sat_95_spend/1e6:.1f}M ({sat_95_spend/total_annual_2025:.1f}x 2025)")
if inflection_idx:
    print(f"  Inflection point:  ${inf_spend/1e6:.1f}M ({inf_spend/total_annual_2025:.1f}x 2025)")


# COMMAND ----------

# DBTITLE 1,LBE Flighting Analysis
# ── LBE FLIGHTING ANALYSIS — FOOLPROOF STANDALONE BLOCK ──────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import pickle, os, shutil
from IPython.display import display as ipy_display

# ── Paths ─────────────────────────────────────────────────────────────────────
RESULTS_PKL = "/Volumes/data_exploration_hub/marketing_hub/model_data/Final_Data/MMM_LBE_v2/lbe_mmm_results.pkl"
OUTPUT_DIR  = "/Volumes/data_exploration_hub/marketing_hub/model_data/Final_Data/MMM_LBE"

# ── Brand colours ─────────────────────────────────────────────────────────────
CPX_NAVY  = "#002B5C"
CPX_RED   = "#E31837"
CPX_GOLD  = "#F5A800"
CPX_DGREY = "#4A4A4A"
CPX_MGREY = "#8A8A8A"
CPX_LGREY = "#F2F2F2"
CPX_GREEN = "#2E7D32"
CPX_MID   = "#B0BEC5"
WHITE     = "#FFFFFF"

def save_fig(fig, fname):
    tmp = f"/tmp/{fname}"
    fig.savefig(tmp, dpi=150, bbox_inches="tight", facecolor="white")
    shutil.copy2(tmp, os.path.join(OUTPUT_DIR, fname))
    print(f"\u2713 Saved: {fname}")

# ── Load LBE model artifacts ──────────────────────────────────────────────────
with open(RESULTS_PKL, "rb") as f:
    results = pickle.load(f)

draws    = results["draws"]
df_train = results["df_train"].copy()
df_val   = results["df_val"].copy()
df_train["week_date"] = pd.to_datetime(df_train["week_date"])
df_val["week_date"]   = pd.to_datetime(df_val["week_date"])
df_all = pd.concat([df_train, df_val], ignore_index=True).sort_values("week_date")
df_all["year"] = df_all["week_date"].dt.year

print(f"LBE data loaded: {len(df_all)} weeks ({df_all['year'].min()}\u2013{df_all['year'].max()})")

# ── Model-derived decay parameters from posterior ─────────────────────────────
AW_DECAY_MEAN = float(np.asarray(draws["aw_decay"]).mean())
CO_DECAY_MEAN = float(np.asarray(draws["co_decay"]).mean())

def optimal_lag_range(alpha):
    """Derive pre-peak spend window from adstock alpha."""
    if alpha >= 1.0:
        return 1, 8
    total = 1 / (1 - alpha)
    cum, lmin, lmax = 0.0, None, None
    for w in range(1, 25):
        cum += alpha ** w
        pct = cum / total
        if lmin is None and pct >= 0.20:
            lmin = w
        if lmin is not None and lmax is None and pct >= 0.70:
            lmax = w
        if lmin and lmax:
            break
    lmin = lmin or 1
    lmax = lmax or (lmin + 2)
    if alpha < 0.30:
        return 1, 2
    return lmin, lmax

AW_LAG_MIN, AW_LAG_MAX = optimal_lag_range(AW_DECAY_MEAN)
CO_LAG_MIN, CO_LAG_MAX = optimal_lag_range(CO_DECAY_MEAN)

print(f"\nPosterior mean Awareness decay:     {AW_DECAY_MEAN:.4f}  lag={AW_LAG_MIN}\u2013{AW_LAG_MAX} wks")
print(f"Posterior mean Consideration decay:  {CO_DECAY_MEAN:.4f}  lag={CO_LAG_MIN}\u2013{CO_LAG_MAX} wks")

# ── Channel-level spend totals ────────────────────────────────────────────────
FUNNELS = ["Awareness", "Consideration"]
for f in FUNNELS:
    trr = df_all[f"{f}_TRR"].sum() / 1e6
    pdm = df_all[f"{f}_PDM"].sum() / 1e6
    print(f"  {f:<15} TRR=${trr:.2f}M  PDM=${pdm:.2f}M  Total=${trr+pdm:.2f}M")

# ── Weekly-of-year aggregation ────────────────────────────────────────────────
df_flight = df_all.copy()
df_flight["woy"] = df_flight["week_date"].dt.isocalendar().week.astype(int)

woy_avg = (
    df_flight.groupby("woy")
    .agg(
        revenue       = ("LBE_Revenue",   "mean"),
        awareness     = ("Awareness",     "mean"),
        consideration = ("Consideration", "mean"),
    )
    .reset_index()
)
wks = woy_avg["woy"].values

# ── Calendar zone definitions (from LBE model calendar features) ──────────────
# Peaks: New Year (wk 1), March Break (wk 11), Summer Peak (wk 33), Pre-Xmas (wks 49-50)
PEAK_WKS = [1, 11, 33, 49, 50]

# Troughs: Spring Lull (wks 19-23), Back to School (wks 36-41)
TROUGH_WKS = list(range(19, 24)) + list(range(36, 42))

# Optimal Awareness windows: LAG_MIN to LAG_MAX weeks BEFORE each peak
optimal_aw_wks = set()
for p in PEAK_WKS:
    for lag in range(AW_LAG_MIN, AW_LAG_MAX + 1):
        w = p - lag
        if w <= 0:
            w += 52
        optimal_aw_wks.add(w)
optimal_aw_wks = sorted(optimal_aw_wks)

# Visual spans for shading
OPT_SPANS    = [(5, 7), (27, 29), (43, 48)]
TROUGH_SPANS = [(19, 23), (36, 41)]
LAG_ARROWS   = [(6, 11, f"\u007e{AW_LAG_MIN} wk lag"), (28, 33, f"\u007e{AW_LAG_MIN} wk lag"),
                (45, 50, f"\u007e{AW_LAG_MIN} wk lag"), (48, 1, "wraps \u2192\nJan")]
peak_labels  = {1: "New Year\nWeek", 11: "March\nBreak", 33: "Summer\nPeak", 49: "Pre-\nXmas"}

# ── Spend zone breakdown ──────────────────────────────────────────────────────
aw_total  = woy_avg["awareness"].sum()
aw_opt    = woy_avg.loc[woy_avg["woy"].isin(optimal_aw_wks), "awareness"].sum()
aw_trough = woy_avg.loc[woy_avg["woy"].isin(TROUGH_WKS),    "awareness"].sum()
aw_late   = woy_avg.loc[woy_avg["woy"].isin(PEAK_WKS),      "awareness"].sum()
aw_other  = aw_total - aw_opt - aw_trough - aw_late

co_total  = woy_avg["consideration"].sum()
co_opt    = woy_avg.loc[woy_avg["woy"].isin(optimal_aw_wks), "consideration"].sum()
co_trough = woy_avg.loc[woy_avg["woy"].isin(TROUGH_WKS),    "consideration"].sum()
co_late   = woy_avg.loc[woy_avg["woy"].isin(PEAK_WKS),      "consideration"].sum()
co_other  = co_total - co_opt - co_trough - co_late

co_arr    = woy_avg["consideration"].values
rv_arr    = woy_avg["revenue"].values
corr_lag0 = np.corrcoef(co_arr, rv_arr)[0, 1]
corr_lag4 = np.corrcoef(np.roll(co_arr, 4), rv_arr)[0, 1]

print(f"\nAwareness zone breakdown:")
print(f"  Optimal windows : {aw_opt/aw_total*100:.1f}%")
print(f"  Trough periods  : {aw_trough/aw_total*100:.1f}%")
print(f"  Peak week itself: {aw_late/aw_total*100:.1f}%")
print(f"  Neutral weeks   : {aw_other/aw_total*100:.1f}%")

print(f"\nConsideration zone breakdown:")
print(f"  Optimal windows : {co_opt/co_total*100:.1f}%")
print(f"  Trough periods  : {co_trough/co_total*100:.1f}%")
print(f"  Peak week itself: {co_late/co_total*100:.1f}%")
print(f"  Neutral weeks   : {co_other/co_total*100:.1f}%")

# ── Year-by-year flighting alignment ──────────────────────────────────────────
yrs = sorted(df_flight["year"].unique())
print("\n" + "=" * 65)
print("FLIGHTING ZONE % BY FUNNEL AND YEAR")
print("=" * 65)

for funnel in FUNNELS:
    col = funnel
    print(f"\n{funnel}  (decay={AW_DECAY_MEAN if funnel=='Awareness' else CO_DECAY_MEAN:.4f})")
    print(f"  {'Year':<6} {'Optimal':>9} {'Trough':>9} {'Peak(late)':>12} {'Neutral':>9}")
    print("  " + "-" * 45)
    for year in yrs:
        yr_data = df_flight[df_flight["year"] == year]
        woy_yr  = yr_data.groupby("woy")[col].mean().reset_index()
        total   = woy_yr[col].sum()
        if total == 0:
            print(f"  {year}  \u2014 no spend")
            continue
        sp_opt    = woy_yr[woy_yr["woy"].isin(optimal_aw_wks)][col].sum()
        sp_trough = woy_yr[woy_yr["woy"].isin(TROUGH_WKS)][col].sum()
        sp_peak   = woy_yr[woy_yr["woy"].isin(PEAK_WKS)][col].sum()
        sp_other  = total - sp_opt - sp_trough - sp_peak
        print(f"  {year:<6} {sp_opt/total*100:>8.1f}% {sp_trough/total*100:>8.1f}% "
              f"{sp_peak/total*100:>11.1f}% {sp_other/total*100:>8.1f}%")

# ── Channel-level zone breakdown (per Funnel × Channel) ──────────────────────
CHANNELS = ["TRR", "PDM"]

chan_rows = []
for funnel in FUNNELS:
    lag_min = AW_LAG_MIN if funnel == "Awareness" else CO_LAG_MIN
    lag_max = AW_LAG_MAX if funnel == "Awareness" else CO_LAG_MAX
    decay   = AW_DECAY_MEAN if funnel == "Awareness" else CO_DECAY_MEAN

    # Optimal weeks for this funnel
    opt_wks_f = set()
    for p in PEAK_WKS:
        for lag in range(lag_min, lag_max + 1):
            w = p - lag
            if w <= 0: w += 52
            opt_wks_f.add(w)
    opt_wks_f = sorted(opt_wks_f)

    for channel in CHANNELS:
        col = f"{funnel}_{channel}"
        woy_ch = df_flight.groupby("woy")[col].mean().reset_index()
        total  = woy_ch[col].sum()
        if total == 0:
            continue

        sp_opt    = woy_ch[woy_ch["woy"].isin(opt_wks_f)][col].sum()
        sp_trough = woy_ch[woy_ch["woy"].isin(TROUGH_WKS)][col].sum()
        sp_peak   = woy_ch[woy_ch["woy"].isin(PEAK_WKS)][col].sum()
        sp_other  = total - sp_opt - sp_trough - sp_peak

        n_years = df_flight["year"].nunique()
        annual_spend_k = (df_flight[col].sum() / n_years) / 1e3

        chan_rows.append({
            "Funnel"               : funnel,
            "Channel"              : channel,
            "Alpha (decay)"        : round(decay, 4),
            "Optimal lag (wks)"    : f"{lag_min}-{lag_max}",
            "Optimal %"            : round(sp_opt    / total * 100, 1),
            "Neutral %"            : round(sp_other  / total * 100, 1),
            "Peak wk % (too late)" : round(sp_peak   / total * 100, 1),
            "Trough %"             : round(sp_trough / total * 100, 1),
            "Annual Spend $K"      : round(annual_spend_k, 1),
        })

chan_df = pd.DataFrame(chan_rows).sort_values("Optimal %", ascending=False).reset_index(drop=True)

print(f"\nChannel zone breakdown rows: {len(chan_df)}")

# ── Styled dataframe display (like EXH) ──────────────────────────────────────
ipy_display(
    chan_df
    .style
    .set_caption("LBE Channel Spend Zone Breakdown \u2014 Empirically Derived Windows")
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

# ══════════════════════════════════════════════════════════════════════════════
# CHART: Horizontal Stacked Bar — Channel Spend Zone Breakdown (EXH style)
# ══════════════════════════════════════════════════════════════════════════════
# Compute annual $ in each zone for stacked bars
plot_df = chan_df.iloc[::-1].reset_index(drop=True)
plot_df["label"] = plot_df["Funnel"] + " | " + plot_df["Channel"]
plot_df["Optimal $K"]    = plot_df["Annual Spend $K"] * plot_df["Optimal %"] / 100
plot_df["Neutral $K"]    = plot_df["Annual Spend $K"] * plot_df["Neutral %"] / 100
plot_df["Peak wk $K"]    = plot_df["Annual Spend $K"] * plot_df["Peak wk % (too late)"] / 100
plot_df["Trough $K"]     = plot_df["Annual Spend $K"] * plot_df["Trough %"] / 100

n = len(plot_df)
y = np.arange(n)

fig, ax = plt.subplots(figsize=(16, 9))
fig.patch.set_facecolor(WHITE)
lefts = np.zeros(n)

zone_config = [
    ("Optimal",            "Optimal $K",   CPX_GREEN),
    ("Neutral",            "Neutral $K",   CPX_MID),
    ("Peak wk (too late)", "Peak wk $K",   CPX_GOLD),
    ("Trough",             "Trough $K",    CPX_RED),
]

for label, col_key, color in zone_config:
    vals = plot_df[col_key].values
    ax.barh(y, vals, left=lefts, height=0.6,
            color=color, alpha=0.88, label=label)
    for i, (v, l) in enumerate(zip(vals, lefts)):
        if v >= 20:
            txt_color = CPX_DGREY if color in (CPX_MID, CPX_GOLD) else WHITE
            ax.text(l + v / 2, i, f"${v:.0f}K",
                    ha="center", va="center",
                    fontsize=8.5, fontweight="bold", color=txt_color)
    lefts += vals

# Right axis — total annual + decay info
max_bar = lefts.max()
for i, row in plot_df.iterrows():
    ax.text(
        max_bar * 1.03, i,
        f"Total: ${row['Annual Spend $K']:.0f}K/yr  "
        f"[\u03b1={row['Alpha (decay)']:.3f}  lag={row['Optimal lag (wks)']}]",
        va="center", fontsize=8, color=CPX_DGREY
    )

ax.set_yticks(y)
ax.set_yticklabels(
    [f"{row['Funnel']} | {row['Channel']}" for _, row in plot_df.iterrows()],
    fontsize=10
)
ax.set_xlabel("Annual Spend ($K)",
              fontsize=12, fontweight="bold", color=CPX_DGREY)
ax.set_xlim(0, max_bar * 1.45)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:.0f}K"))
ax.set_title(
    "LBE MMM \u2014 Channel Spend Zone Breakdown (Annual $)\n"
    "Sorted by % in optimal pre-peak window  |  "
    "Right axis = total annual spend & adstock decay",
    fontsize=14, fontweight="bold", color="black", loc="center"
)
ax.legend(loc="lower right", fontsize=10, frameon=True, framealpha=0.9)
ax.grid(False)
for spine in ["top", "right", "left", "bottom"]:
    ax.spines[spine].set_visible(True)
    ax.spines[spine].set_color(CPX_DGREY)

fig.text(
    0.5, -0.01,
    f"Peak weeks from LBE calendar features: New Year (wk 1), March Break (wk 11), "
    f"Summer Peak (wk 33), Pre-Xmas (wks 49-50)  |  "
    f"Trough = Spring Lull (wk 19-23) + Back to School (wk 36-41)  |  "
    f"Optimal windows from posterior mean adstock decay per funnel",
    ha="center", fontsize=7.5, color=CPX_MGREY, style="italic"
)

plt.tight_layout()
save_fig(fig, "lbe_flighting_channel_zone_breakdown.png")
plt.show()

print("\u2713 LBE flighting analysis complete.")


# COMMAND ----------

# DBTITLE 1,LBE Flighting Zone Breakdown Per Year
# ── LBE FLIGHTING ZONE BREAKDOWN — PER YEAR (2023, 2024, 2025) ───────────────
# One horizontal stacked bar chart per year showing Channel Spend Zone Breakdown
# Depends on: df_flight, PEAK_WKS, TROUGH_WKS, AW_DECAY_MEAN, CO_DECAY_MEAN,
#             AW_LAG_MIN/MAX, CO_LAG_MIN/MAX from the preceding LBE Flighting cell.

import matplotlib.ticker as mticker
import matplotlib.patches as mpatches

FUNNELS_FL  = ["Awareness", "Consideration"]
CHANNELS_FL = ["TRR", "PDM"]

for year in [2023, 2024, 2025]:
    yr_data = df_flight[df_flight["year"] == year].copy()
    yr_data["woy"] = yr_data["week_date"].dt.isocalendar().week.astype(int)

    chan_rows_yr = []
    for funnel in FUNNELS_FL:
        lag_min = AW_LAG_MIN if funnel == "Awareness" else CO_LAG_MIN
        lag_max = AW_LAG_MAX if funnel == "Awareness" else CO_LAG_MAX
        decay   = AW_DECAY_MEAN if funnel == "Awareness" else CO_DECAY_MEAN

        # Optimal weeks for this funnel
        opt_wks_f = set()
        for p in PEAK_WKS:
            for lag in range(lag_min, lag_max + 1):
                w = p - lag
                if w <= 0:
                    w += 52
                opt_wks_f.add(w)
        opt_wks_f = sorted(opt_wks_f)

        for channel in CHANNELS_FL:
            col = f"{funnel}_{channel}"
            woy_ch = yr_data.groupby("woy")[col].mean().reset_index()
            total  = woy_ch[col].sum()
            if total == 0:
                continue

            sp_opt    = woy_ch[woy_ch["woy"].isin(opt_wks_f)][col].sum()
            sp_trough = woy_ch[woy_ch["woy"].isin(TROUGH_WKS)][col].sum()
            sp_peak   = woy_ch[woy_ch["woy"].isin(PEAK_WKS)][col].sum()
            sp_other  = total - sp_opt - sp_trough - sp_peak

            annual_spend_k = yr_data[col].sum() / 1e3

            chan_rows_yr.append({
                "Funnel"               : funnel,
                "Channel"              : channel,
                "Alpha (decay)"        : round(decay, 4),
                "Optimal lag (wks)"    : f"{lag_min}-{lag_max}",
                "Optimal %"            : round(sp_opt    / total * 100, 1),
                "Neutral %"            : round(sp_other  / total * 100, 1),
                "Peak wk % (too late)" : round(sp_peak   / total * 100, 1),
                "Trough %"             : round(sp_trough / total * 100, 1),
                "Annual Spend $K"      : round(annual_spend_k, 1),
            })

    chan_df_yr = pd.DataFrame(chan_rows_yr).sort_values("Optimal %", ascending=False).reset_index(drop=True)

    # ── Styled table ──────────────────────────────────────────────────────────
    ipy_display(
        chan_df_yr
        .style
        .set_caption(f"LBE Channel Spend Zone Breakdown — {year}")
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

    # ── Horizontal Stacked Bar Chart ──────────────────────────────────────────
    plot_df = chan_df_yr.iloc[::-1].reset_index(drop=True)
    plot_df["label"] = plot_df["Funnel"] + " | " + plot_df["Channel"]
    plot_df["Optimal $K"]  = plot_df["Annual Spend $K"] * plot_df["Optimal %"] / 100
    plot_df["Neutral $K"]  = plot_df["Annual Spend $K"] * plot_df["Neutral %"] / 100
    plot_df["Peak wk $K"]  = plot_df["Annual Spend $K"] * plot_df["Peak wk % (too late)"] / 100
    plot_df["Trough $K"]   = plot_df["Annual Spend $K"] * plot_df["Trough %"] / 100

    n = len(plot_df)
    y = np.arange(n)

    fig, ax = plt.subplots(figsize=(16, 9))
    fig.patch.set_facecolor(WHITE)
    lefts = np.zeros(n)

    zone_config = [
        ("Optimal",            "Optimal $K",   CPX_GREEN),
        ("Neutral",            "Neutral $K",   CPX_MID),
        ("Peak wk (too late)", "Peak wk $K",   CPX_GOLD),
        ("Trough",             "Trough $K",    CPX_RED),
    ]

    for label, col_key, color in zone_config:
        vals = plot_df[col_key].values
        ax.barh(y, vals, left=lefts, height=0.6,
                color=color, alpha=0.88, label=label)
        for i, (v, l) in enumerate(zip(vals, lefts)):
            if v >= 20:
                txt_color = CPX_DGREY if color in (CPX_MID, CPX_GOLD) else WHITE
                ax.text(l + v / 2, i, f"${v:.0f}K",
                        ha="center", va="center",
                        fontsize=8.5, fontweight="bold", color=txt_color)
        lefts += vals

    # Right axis — total + decay info
    max_bar = lefts.max()
    for i, row in plot_df.iterrows():
        ax.text(
            max_bar * 1.03, i,
            f"Total: ${row['Annual Spend $K']:.0f}K  "
            f"[\u03b1={row['Alpha (decay)']:.3f}  lag={row['Optimal lag (wks)']}]",
            va="center", fontsize=8, color=CPX_DGREY
        )

    ax.set_yticks(y)
    ax.set_yticklabels(
        [f"{row['Funnel']} | {row['Channel']}" for _, row in plot_df.iterrows()],
        fontsize=10
    )
    ax.set_xlabel("Annual Spend ($K)",
                  fontsize=12, fontweight="bold", color=CPX_DGREY)
    ax.set_xlim(0, max_bar * 1.45)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:.0f}K"))
    ax.set_title(
        f"LBE MMM — Channel Spend Zone Breakdown — {year}\n"
        "Sorted by % in optimal pre-peak window  |  "
        "Right axis = total spend & adstock decay",
        fontsize=14, fontweight="bold", color="black", loc="center"
    )
    ax.legend(loc="lower right", fontsize=10, frameon=True, framealpha=0.9)
    ax.grid(False)
    for spine in ["top", "right", "left", "bottom"]:
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_color(CPX_DGREY)

    fig.text(
        0.5, -0.01,
        f"Peak weeks: New Year (wk 1), March Break (wk 11), "
        f"Summer Peak (wk 33), Pre-Xmas (wks 49-50)  |  "
        f"Trough = Spring Lull (wk 19-23) + Back to School (wk 36-41)  |  "
        f"Optimal windows from posterior mean adstock decay",
        ha="center", fontsize=7.5, color=CPX_MGREY, style="italic"
    )

    plt.tight_layout()
    save_fig(fig, f"lbe_flighting_channel_zone_{year}.png")
    plt.show()

print("\u2713 Per-year LBE flighting analysis complete.")

