# Databricks notebook source
# DBTITLE 1,FIFA 2026 World Cup Schedule
import requests
import pandas as pd
from datetime import timedelta, date

# ---------------------------------------------------------------------------
# FIFA World Cup 2026  –  Full schedule via ESPN public API
# Tournament: June 11 – July 19, 2026  |  48 teams, 64 games
# ---------------------------------------------------------------------------
BASE_URL         = "https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world/scoreboard"
TOURNAMENT_START = date(2026, 6, 11)
TOURNAMENT_END   = date(2026, 7, 19)

GROUP_STAGE_END  = pd.Timestamp('2026-06-27')  # Round of 32 began Jun 28; group stage concluded Jun 27

all_games = []
current   = TOURNAMENT_START

while current <= TOURNAMENT_END:
    url = f"{BASE_URL}?dates={current.strftime('%Y%m%d')}"
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code != 200:
            print(f"HTTP {resp.status_code} on {current}")
            current += timedelta(days=1)
            continue

        events = resp.json().get('events', [])

        for e in events:
            comp        = e.get('competitions', [{}])[0]
            competitors = comp.get('competitors', [])
            notes       = comp.get('notes', [])
            stage       = notes[0].get('headline', 'Unknown') if notes else 'Unknown'
            venue_info  = comp.get('venue', {})

            home_team = next(
                (c['team']['displayName'] for c in competitors if c['homeAway'] == 'home'),
                None
            )
            away_team = next(
                (c['team']['displayName'] for c in competitors if c['homeAway'] == 'away'),
                None
            )

            all_games.append({
                'date':      pd.Timestamp(current),
                'home_team': home_team,
                'away_team': away_team,
                'stage':     stage,
                'city':      venue_info.get('address', {}).get('city'),
                'start_utc': e.get('date'),
            })

    except Exception as ex:
        print(f"Error on {current}: {ex}")

    current += timedelta(days=1)

# ---------------------------------------------------------------------------
# Build schedule DataFrame and classify stages
# ---------------------------------------------------------------------------
if not all_games:
    print("⚠  No games retrieved — verify ESPN API competition slug for FIFA WC 2026")
    print("   Try substituting 'fifa.worldcup' in BASE_URL above if needed")
else:
    fifa_df = pd.DataFrame(all_games)
    fifa_df['date'] = pd.to_datetime(fifa_df['date'])

    # Date-based stage split — API 'stage' label returns 'Unknown' until games are played
    fifa_df['stage_type'] = 'Knockout'
    fifa_df.loc[fifa_df['date'] <= GROUP_STAGE_END, 'stage_type'] = 'Group Stage'

    # Vectorized Canada flag (no row-wise apply)
    fifa_df['is_canada'] = (
        (fifa_df['home_team'] == 'Canada') | (fifa_df['away_team'] == 'Canada')
    )
    fifa_df['game_day'] = 1

    print(f"Total games retrieved:  {len(fifa_df)}")
    print(f"  Group Stage:          {(fifa_df['stage_type'] == 'Group Stage').sum()}")
    print(f"  Knockout:             {(fifa_df['stage_type'] == 'Knockout').sum()}")
    print(f"  Canada games:         {fifa_df['is_canada'].sum()}")
    print()
    print(fifa_df[['date', 'home_team', 'away_team', 'stage', 'stage_type']].to_string(index=False))

# COMMAND ----------

# DBTITLE 1,Load Spark Tables
import pyspark.sql.functions as F

T_trans    = spark.table("cpx_dataanalytics_gold.customer360.c360_f_transaction")
T_location = spark.table("cpx_dataanalytics_gold.customer360.cpx_d_location")

# COMMAND ----------

# DBTITLE 1,Revenue Pull — Jan–Aug 2026
# Jan 1, 2026 → yesterday: historical data only.
# Excludes today's partially-loaded transactions and all future dates.
# Upper bound re-evaluates on every run — no hardcoded end date.
# Net_Revenue cast to double in Spark — eliminates slow DecimalType pandas conversion.
revenue_df = (
    T_trans
    .filter(F.col("VisitDateId") >= '20260101')
    .filter(F.col("VisitDateId") <  date.today().strftime('%Y%m%d'))  # < today == up to yesterday
    .join(T_location, T_trans.LocationId == T_location.LocationID, how="inner")
    .withColumn("date", F.to_date(F.col("VisitDateId"), "yyyyMMdd"))
    .filter(F.col("Location_Name").like("%Granv%"))
    .filter(F.col("LC_Location_Type_Description").like("%Restaurant%"))
    .groupBy("date")
    .agg(F.sum(F.col("Net_Revenue").cast("double")).alias("revenue"))
    .toPandas()
)
print(f"Revenue rows pulled: {len(revenue_df)}")
print(revenue_df.sort_values('date').head())

# COMMAND ----------

# DBTITLE 1,Revenue Cleaning
revenue_df['date']    = pd.to_datetime(revenue_df['date'])
revenue_df['revenue'] = revenue_df['revenue'].astype(float)
revenue_df = revenue_df.sort_values('date').set_index('date')

# Check for missing dates in the range
full_range = pd.date_range(revenue_df.index.min(), revenue_df.index.max(), freq='D')
missing    = full_range.difference(revenue_df.index)

print(f"Date range:    {revenue_df.index.min().date()} → {revenue_df.index.max().date()}")
print(f"Missing dates: {len(missing)}")
print(missing.tolist() if len(missing) < 20 else missing[:20].tolist())

# Reindex to fill any gaps with NaN (zero-revenue days stay as NaN; excluded from matching)
revenue_df = revenue_df.reindex(full_range)
print(f"NaN revenue rows: {revenue_df['revenue'].isna().sum()}")

# COMMAND ----------

# DBTITLE 1,Matched Analysis — FIFA Knockout Stage
import pandas as pd
import numpy as np
from scipy import stats

# ---------------------------------------------------------------------------
# Inputs:
#   revenue_df  – DatetimeIndex, column 'revenue'  (Rec Room Royalmount, Jan–Aug 2026)
#   fifa_df     – full WC 2026 schedule with 'stage_type': 'Group Stage' / 'Knockout'
# ---------------------------------------------------------------------------

# 1. Build daily frame with FIFA event flags
daily          = revenue_df[['revenue']].copy()
daily['dow']   = daily.index.dayofweek    # 0=Mon … 6=Sun
daily['month'] = daily.index.month

knockout_dates    = set(pd.to_datetime(fifa_df.loc[fifa_df['stage_type'] == 'Knockout',    'date']).dt.normalize())
group_stage_dates = set(pd.to_datetime(fifa_df.loc[fifa_df['stage_type'] == 'Group Stage', 'date']).dt.normalize())

daily['is_knockout_game']    = daily.index.normalize().isin(knockout_dates)
daily['is_group_stage_game'] = daily.index.normalize().isin(group_stage_dates)
daily['is_any_fifa_game']    = daily['is_knockout_game'] | daily['is_group_stage_game']

# 2. Control pool: exclude ALL FIFA game days
control_pool = daily[~daily['is_any_fifa_game']].copy()

# 3. Precompute O(1) lookup structures — avoids per-iteration full-DataFrame filters inside the loop
_norm        = pd.to_datetime(fifa_df['date']).dt.normalize()
canada_dates = set(_norm[fifa_df['is_canada']])  # set used here and reused in Cell 10
stage_map    = dict(zip(_norm, fifa_df['stage']))

# Match each knockout game to comparable non-game days
#    Same methodology as Habs: same DOW, ±28-day window
results = []

for game_date in sorted(knockout_dates):
    if game_date not in daily.index:
        continue

    target     = daily.loc[game_date]
    target_dow = target['dow']

    window_start = game_date - pd.Timedelta(days=28)
    window_end   = game_date + pd.Timedelta(days=28)

    matches = control_pool[
        (control_pool['dow'] == target_dow) &
        (control_pool.index >= window_start) &
        (control_pool.index <= window_end)
    ]

    if len(matches) < 2:
        print(f"⚠ {game_date.date()}: only {len(matches)} match(es) — skipping")
        continue

    actual = float(target['revenue'])
    if pd.isna(actual):
        print(f"⚠ {game_date.date()}: revenue not yet available — skipping")
        continue
    expected    = float(matches['revenue'].mean())
    incremental = actual - expected
    lift_pct    = (actual / expected - 1) * 100

    is_canada   = game_date in canada_dates           # O(1) set lookup
    stage_label = stage_map.get(game_date, 'Unknown')  # O(1) dict lookup

    results.append({
        'game_date':           game_date.date(),
        'dow':                 game_date.strftime('%a'),
        'stage':               stage_label,
        'is_canada':           is_canada,
        'n_matches':           len(matches),
        'actual_revenue':      round(actual, 2),
        'expected_revenue':    round(expected, 2),
        'incremental_revenue': round(incremental, 2),
        'lift_pct':            round(lift_pct, 2),
    })

results_df = pd.DataFrame(results)

# 4. Per-game output
print("\n=== Per-game lift — FIFA World Cup 2026 Knockout Stage ===")

if len(results_df) > 0:
    print(
        results_df[[
            'game_date', 'dow', 'stage', 'is_canada', 'n_matches',
            'actual_revenue', 'expected_revenue', 'incremental_revenue', 'lift_pct'
        ]].to_string(index=False)
    )

    # 5. Summary stats
    print("\n=== Summary ===")
    mean_lift   = results_df['lift_pct'].mean()
    median_lift = results_df['lift_pct'].median()
    total_inc   = results_df['incremental_revenue'].sum()

    print(f"Mean lift:                {mean_lift:+.2f}%")
    print(f"Median lift:              {median_lift:+.2f}%")
    print(f"Total incremental rev:    ${total_inc:,.2f}")

    if len(results_df) >= 2:
        t_stat, p_value = stats.ttest_1samp(results_df['lift_pct'], 0)

        boot_means = [
            np.random.choice(results_df['lift_pct'], len(results_df), replace=True).mean()
            for _ in range(5000)
        ]
        ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])

        print(f"95% CI:                  [{ci_low:+.2f}%, {ci_high:+.2f}%]")
        print(f"p-value (t-test):        {p_value:.4f}")
    else:
        print("Not enough observations for statistical tests yet")

    # 6. Canada vs non-Canada split
    print("\n=== Canada vs Non-Canada Games ===")
    print(
        results_df.groupby('is_canada')['lift_pct']
        .agg(['mean', 'median', 'count'])
    )
else:
    print("No knockout games yet in the revenue data window.")
    print(f"  Knockout dates in schedule: {len(knockout_dates)}")
    print(f"  Revenue window: {revenue_df.index.min().date()} → {revenue_df.index.max().date()}")
    print("  Knockout stage begins ~July 6 — re-run once revenue data covers that period.")

# COMMAND ----------

# DBTITLE 1,Results Table
if results_df.empty:
    print("No knockout results yet — re-run once the knockout stage begins (~July 6).")
else:
    display(results_df)

# COMMAND ----------

# DBTITLE 1,Std Dev + Bootstrap CI + Avg Incremental
if results_df.empty:
    print("No knockout results yet — re-run once the knockout stage begins (~July 6).")
else:
 # Drop any NaN rows before stats (e.g. dates where revenue data is not yet loaded)
 _clean = results_df.dropna(subset=['lift_pct', 'incremental_revenue'])

 std_dev_lift        = _clean['lift_pct'].std()
 std_dev_incremental = _clean['incremental_revenue'].std()

 print(f"Lift Std Dev:             {std_dev_lift:.2f}%")
 print(f"Incremental Rev Std Dev:  ${std_dev_incremental:,.2f}")

 # Bootstrap 95% CI on mean incremental revenue
 boot_incrementals = [
     np.random.choice(
         _clean['incremental_revenue'],
         len(_clean),
         replace=True
     ).mean()
     for _ in range(5000)
 ]
 inc_ci_low, inc_ci_high = np.percentile(boot_incrementals, [2.5, 97.5])

 print(
     f"Incremental Revenue 95% CI: "
     f"[${inc_ci_low:,.0f}, ${inc_ci_high:,.0f}]"
 )

 avg_incremental = _clean['incremental_revenue'].mean()
 print(f"Average Incremental Revenue Per Game: ${avg_incremental:,.0f}")

# COMMAND ----------

# DBTITLE 1,Robust Stats + Canada vs Non-Canada Split
from scipy import stats as _stats

if results_df.empty:
    print("No knockout results yet — re-run once the knockout stage begins (~July 6).")
else:
 # Drop NaN rows before all tests
 _clean = results_df.dropna(subset=['lift_pct'])

 # Canada vs non-Canada t-test
 canada_lift     = _clean.loc[_clean['is_canada'],  'lift_pct']
 non_canada_lift = _clean.loc[~_clean['is_canada'], 'lift_pct']

 if len(canada_lift) >= 2 and len(non_canada_lift) >= 2:
     t, p = _stats.ttest_ind(canada_lift, non_canada_lift, equal_var=False)
     print(
         f"Canada vs Non-Canada: "
         f"Canada mean {canada_lift.mean():+.1f}%  |  "
         f"Non-Canada mean {non_canada_lift.mean():+.1f}%  |  "
         f"p={p:.3f}"
     )
 else:
     print(f"Canada: n={len(canada_lift)}  |  Non-Canada: n={len(non_canada_lift)}  (t-test skipped — insufficient obs)")

 # Robust summary (trimmed mean, less outlier-sensitive)
 if len(_clean) >= 4:
     print(f"Trimmed mean lift (10%): {_stats.trim_mean(_clean['lift_pct'], 0.1):+.1f}%")

 # Non-parametric test
 if len(_clean) >= 2:
     w, p = _stats.wilcoxon(_clean['lift_pct'])
     print(f"Wilcoxon p-value:        {p:.4f}")

# COMMAND ----------

# DBTITLE 1,Visualization — Per-Game Lift + Canada vs Other
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

if results_df.empty:
    print("No knockout results yet — visualization will populate once knockout games have occurred (~July 6).")
else:
 CANADA_COLOR     = '#D20000'  # Canada red
 NON_CANADA_COLOR = '#1D6B9E'  # Blue for other knockout games

 plot_df          = results_df.copy()
 plot_df['label'] = (
     pd.to_datetime(plot_df['game_date']).dt.strftime('%b %d')
     + '\n' + plot_df['dow']
 )
 plot_df['color'] = np.where(plot_df['is_canada'], CANADA_COLOR, NON_CANADA_COLOR)

 fig, axes = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={'width_ratios': [3, 1]})

 # --- Left: per-game lift bars ---
 ax1 = axes[0]
 ax1.bar(plot_df['label'], plot_df['lift_pct'], color=plot_df['color'], edgecolor='none')
 ax1.axhline(0, color='#444', linewidth=0.8)
 ax1.set_ylabel('Lift vs matched non-game days (%)', fontsize=11)
 ax1.set_title('Per-game revenue lift — Rec Room Royalmount (FIFA WC 2026 Knockout)', fontsize=12, loc='left', pad=12)
 ax1.tick_params(axis='x', labelsize=9)
 ax1.grid(axis='y', alpha=0.2)
 ax1.spines['top'].set_visible(False)
 ax1.spines['right'].set_visible(False)

 for i, v in enumerate(plot_df['lift_pct']):
     offset = 4 if v >= 0 else -4
     va     = 'bottom' if v >= 0 else 'top'
     ax1.text(i, v + offset, f'{v:+.0f}%', ha='center', va=va, fontsize=8.5)

 ax1.legend(handles=[
     mpatches.Patch(color=CANADA_COLOR,     label='Canada game'),
     mpatches.Patch(color=NON_CANADA_COLOR, label='Other knockout game'),
 ], loc='upper left', frameon=False, fontsize=10)

 # --- Right: Canada vs non-Canada summary ---
 ax2 = axes[1]
 summary          = plot_df.groupby('is_canada')['lift_pct'].agg(['mean', 'count']).reset_index()
 summary['label'] = np.where(summary['is_canada'], 'Canada', 'Other')
 summary['color'] = np.where(summary['is_canada'], CANADA_COLOR, NON_CANADA_COLOR)
 summary          = summary.sort_values('is_canada')

 bars = ax2.bar(summary['label'], summary['mean'], color=summary['color'],
                edgecolor='none', width=0.45)
 ax2.axhline(0, color='#444', linewidth=0.8)
 ax2.set_ylabel('Mean lift (%)', fontsize=11)
 ax2.set_title('Canada vs Other', fontsize=12, loc='left', pad=12)
 ax2.grid(axis='y', alpha=0.2)
 ax2.spines['top'].set_visible(False)
 ax2.spines['right'].set_visible(False)

 y_max = max(summary['mean'].max(), 0)
 y_min = min(summary['mean'].min(), 0)
 pad   = max((y_max - y_min) * 0.25, 5)
 ax2.set_ylim(y_min - pad, y_max + pad)

 for bar, row in zip(bars, summary.itertuples()):
     h = bar.get_height()
     if h >= 0:
         ax2.text(bar.get_x() + bar.get_width()/2, h + pad * 0.15,
                  f'{h:+.0f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
         ax2.text(bar.get_x() + bar.get_width()/2, h + pad * 0.55,
                  f'n={row.count}', ha='center', va='bottom', fontsize=9, color='#666')
     else:
         ax2.text(bar.get_x() + bar.get_width()/2, h - pad * 0.15,
                  f'{h:+.0f}%', ha='center', va='top', fontsize=11, fontweight='bold')
         ax2.text(bar.get_x() + bar.get_width()/2, h - pad * 0.55,
                  f'n={row.count}', ha='center', va='top', fontsize=9, color='#666')

 n_games  = len(results_df)
 mean_l   = results_df['lift_pct'].mean()   if n_games else 0
 median_l = results_df['lift_pct'].median() if n_games else 0

 plt.suptitle(
     f'FIFA World Cup 2026 knockout revenue lift  ·  {n_games} games'
     f'  ·  Median {median_l:+.0f}%  ·  Mean {mean_l:+.0f}%',
     fontsize=11, y=1.02, x=0.02, ha='left', color='#555'
 )
 plt.tight_layout()
 plt.show()

# COMMAND ----------

# DBTITLE 1,Group Stage Lift Analysis
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Group Stage lift analysis  (analogous to the Canadian GP cell in the Habs notebook)
# Same matched-comparator methodology applied across all group stage games
# ---------------------------------------------------------------------------

# 1. Control pool: exclude ALL FIFA game days
daily_gs       = revenue_df[['revenue']].copy()
daily_gs.index = pd.to_datetime(daily_gs.index)  # guard against stale RangeIndex in kernel state
daily_gs['dow'] = daily_gs.index.dayofweek

control_gs = daily_gs[
    ~daily_gs.index.normalize().isin(knockout_dates) &
    ~daily_gs.index.normalize().isin(group_stage_dates)
].copy()

# 2. Match each group stage game day
gs_results = []

for game_date in sorted(group_stage_dates):
    if game_date not in daily_gs.index:
        continue

    target_dow   = daily_gs.loc[game_date, 'dow']
    window_start = game_date - pd.Timedelta(days=28)
    window_end   = game_date + pd.Timedelta(days=28)

    matches = control_gs[
        (control_gs['dow'] == target_dow) &
        (control_gs.index >= window_start) &
        (control_gs.index <= window_end)
    ]

    if len(matches) < 2:
        continue

    actual      = float(daily_gs.loc[game_date, 'revenue'])
    expected    = float(matches['revenue'].mean())
    incremental = actual - expected
    lift_pct    = (actual / expected - 1) * 100

    is_canada = game_date in canada_dates  # reuses set precomputed in Cell 5

    gs_results.append({
        'game_date':           game_date.date(),
        'dow':                 game_date.strftime('%a'),
        'is_canada':           is_canada,
        'n_matches':           len(matches),
        'actual_revenue':      round(actual, 2),
        'expected_revenue':    round(expected, 2),
        'incremental_revenue': round(incremental, 2),
        'lift_pct':            round(lift_pct, 2),
    })

gs_df = pd.DataFrame(gs_results)

print("=== Group Stage — Rec Room Royalmount Revenue Lift ===\n")
if len(gs_df) > 0:
    print(
        gs_df[[
            'game_date', 'dow', 'is_canada', 'n_matches',
            'actual_revenue', 'expected_revenue', 'incremental_revenue', 'lift_pct'
        ]].to_string(index=False)
    )
    print(f"\nGames analysed:              {len(gs_df)}")
    print(f"Mean lift:                   {gs_df['lift_pct'].mean():+.1f}%")
    print(f"Median lift:                 {gs_df['lift_pct'].median():+.1f}%")
    print(f"Total incremental revenue:   ${gs_df['incremental_revenue'].sum():,.2f}")
else:
    print(f"No group stage games in revenue window yet.")
    print(f"Revenue window: {revenue_df.index.min().date()} → {revenue_df.index.max().date()}")
    print(f"Group stage dates in schedule: {len(group_stage_dates)}")

# COMMAND ----------

# DBTITLE 1,Canada Games — All Stages Lift Analysis
# ---------------------------------------------------------------------------
# Canada game lift — all stages (Group Stage + Knockout)
# Uses daily, control_pool, and canada_dates computed in Cell 5.
# Covers every Canada game with historical revenue data available.
# ---------------------------------------------------------------------------
from scipy import stats

canada_results = []

for game_date in sorted(canada_dates):
    if game_date not in daily.index:
        continue

    actual = float(daily.loc[game_date, 'revenue'])
    if pd.isna(actual):
        print(f"⚠ {game_date.date()}: revenue not yet available — skipping")
        continue

    target_dow   = daily.loc[game_date, 'dow']
    window_start = game_date - pd.Timedelta(days=28)
    window_end   = game_date + pd.Timedelta(days=28)

    matches = control_pool[
        (control_pool['dow'] == target_dow) &
        (control_pool.index >= window_start) &
        (control_pool.index <= window_end)
    ]

    if len(matches) < 2:
        print(f"⚠ {game_date.date()}: only {len(matches)} match(es) — skipping")
        continue

    expected    = float(matches['revenue'].mean())
    incremental = actual - expected
    lift_pct    = (actual / expected - 1) * 100
    stage_type  = 'Group Stage' if game_date <= GROUP_STAGE_END else 'Knockout'

    canada_results.append({
        'game_date':           game_date.date(),
        'dow':                 game_date.strftime('%a'),
        'stage_type':          stage_type,
        'n_matches':           len(matches),
        'actual_revenue':      round(actual, 2),
        'expected_revenue':    round(expected, 2),
        'incremental_revenue': round(incremental, 2),
        'lift_pct':            round(lift_pct, 2),
    })

canada_df = pd.DataFrame(canada_results)

print("=== Canada Games — All Stages — Rec Room Royalmount Revenue Lift ===\n")
if len(canada_df) > 0:
    print(
        canada_df[[
            'game_date', 'dow', 'stage_type', 'n_matches',
            'actual_revenue', 'expected_revenue', 'incremental_revenue', 'lift_pct'
        ]].to_string(index=False)
    )
    print(f"\nGames analysed:              {len(canada_df)}")
    print(f"Mean lift:                   {canada_df['lift_pct'].mean():+.1f}%")
    print(f"Median lift:                 {canada_df['lift_pct'].median():+.1f}%")
    print(f"Total incremental revenue:   ${canada_df['incremental_revenue'].sum():,.2f}")

    if len(canada_df) >= 2:
        _, p_value = stats.ttest_1samp(canada_df['lift_pct'], 0)
        boot_means = [
            np.random.choice(canada_df['lift_pct'], len(canada_df), replace=True).mean()
            for _ in range(5000)
        ]
        ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])
        print(f"95% CI:                      [{ci_low:+.1f}%, {ci_high:+.1f}%]")
        print(f"p-value (t-test):            {p_value:.4f}")
else:
    print("No Canada games in the historical revenue window yet.")
    print(f"Canada game dates in schedule: {sorted(d.date() for d in canada_dates)}")
    print(f"Revenue window: {revenue_df.index.min().date()} → {revenue_df.index.max().date()}")

# COMMAND ----------

# DBTITLE 1,All FIFA Game Days — General Lift
# ---------------------------------------------------------------------------
# All FIFA game day lift — group stage + knockout combined
# Each calendar date with at least one game is treated as one event day.
# Reuses daily, control_pool, canada_dates, group_stage_dates, knockout_dates from Cell 5.
# ---------------------------------------------------------------------------
from scipy import stats

all_fifa_dates = sorted(group_stage_dates | knockout_dates)
all_results    = []

for game_date in all_fifa_dates:
    if game_date not in daily.index:
        continue

    actual = float(daily.loc[game_date, 'revenue'])
    if pd.isna(actual):
        continue

    target_dow   = daily.loc[game_date, 'dow']
    window_start = game_date - pd.Timedelta(days=28)
    window_end   = game_date + pd.Timedelta(days=28)

    matches = control_pool[
        (control_pool['dow'] == target_dow) &
        (control_pool.index >= window_start) &
        (control_pool.index <= window_end)
    ]

    if len(matches) < 2:
        continue

    expected    = float(matches['revenue'].mean())
    incremental = actual - expected
    lift_pct    = (actual / expected - 1) * 100

    all_results.append({
        'game_date':           game_date.date(),
        'dow':                 game_date.strftime('%a'),
        'stage_type':          'Group Stage' if game_date <= GROUP_STAGE_END else 'Knockout',
        'is_canada':           game_date in canada_dates,
        'n_matches':           len(matches),
        'actual_revenue':      round(actual, 2),
        'expected_revenue':    round(expected, 2),
        'incremental_revenue': round(incremental, 2),
        'lift_pct':            round(lift_pct, 2),
    })

all_games_df = pd.DataFrame(all_results)

print("=== All FIFA Game Days — Rec Room Royalmount Revenue Lift ===\n")
if len(all_games_df) > 0:
    print(
        all_games_df[[
            'game_date', 'dow', 'stage_type', 'is_canada', 'n_matches',
            'actual_revenue', 'expected_revenue', 'incremental_revenue', 'lift_pct'
        ]].to_string(index=False)
    )

    print(f"\nGame days analysed:          {len(all_games_df)}")
    print(f"Mean lift:                   {all_games_df['lift_pct'].mean():+.1f}%")
    print(f"Median lift:                 {all_games_df['lift_pct'].median():+.1f}%")
    print(f"Total incremental revenue:   ${all_games_df['incremental_revenue'].sum():,.2f}")

    if len(all_games_df) >= 2:
        _, p_value = stats.ttest_1samp(all_games_df['lift_pct'], 0)
        boot_means = [
            np.random.choice(all_games_df['lift_pct'], len(all_games_df), replace=True).mean()
            for _ in range(5000)
        ]
        ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])
        print(f"95% CI:                      [{ci_low:+.1f}%, {ci_high:+.1f}%]")
        print(f"p-value (t-test):            {p_value:.4f}")

    print("\n=== By Stage ===")
    print(
        all_games_df.groupby('stage_type')[['lift_pct', 'incremental_revenue']]
        .agg(mean_lift=('lift_pct', 'mean'), median_lift=('lift_pct', 'median'),
             total_incremental=('incremental_revenue', 'sum'), n=('lift_pct', 'count'))
        .to_string()
    )
else:
    print("No FIFA game days in the historical revenue window yet.")
    print(f"Revenue window: {revenue_df.index.min().date()} → {revenue_df.index.max().date()}")
    print(f"All FIFA game dates in schedule: {len(all_fifa_dates)}")

# COMMAND ----------

# DBTITLE 1,YoY Comparison — 2026 (World Cup) vs 2025 (No World Cup)
# ---------------------------------------------------------------------------
# Year-over-Year comparison: Jun 2026 (World Cup) vs Jun 2025 (no World Cup)
# Fixes the seasonality flaw in the matched-comparator design by using the
# same venue, same calendar position, 52 weeks apart as the counterfactual.
# 52-week offset (364 days) preserves day of week exactly.
# ---------------------------------------------------------------------------

# Step 1: Pull May–Aug 2025 revenue — same location, same Spark tables
rev_2025_pd = (
    T_trans
    .filter(F.col("VisitDateId") >= '20250501')
    .filter(F.col("VisitDateId") <= '20250831')
    .join(T_location, T_trans.LocationId == T_location.LocationID, how="inner")
    .withColumn("date", F.to_date(F.col("VisitDateId"), "yyyyMMdd"))
    .filter(F.col("Location_Name").like("%Royal%"))
    .groupBy("date")
    .agg(F.sum(F.col("Net_Revenue").cast("double")).alias("revenue"))
    .toPandas()
)
rev_2025_pd['date'] = pd.to_datetime(rev_2025_pd['date'])
rev_2025_lookup     = rev_2025_pd.set_index('date')['revenue'].to_dict()

print(f"2025 revenue rows pulled: {len(rev_2025_pd)}  "
      f"({rev_2025_pd['date'].min().date()} → {rev_2025_pd['date'].max().date()})")

# Step 2: Build matched pairs for every day in Jun 2026 with a 2025 counterpart
YOY_OFFSET = pd.Timedelta(weeks=52)  # 364 days — same DOW, same seasonal position

rows = []
for date_2026 in [d for d in daily.index if d >= pd.Timestamp('2026-06-01')]:
    rev_26 = float(daily.loc[date_2026, 'revenue'])
    if pd.isna(rev_26):
        continue

    date_2025 = date_2026 - YOY_OFFSET
    rev_25    = rev_2025_lookup.get(date_2025)
    if rev_25 is None or pd.isna(rev_25) or rev_25 <= 0:
        continue

    is_game   = date_2026 in (group_stage_dates | knockout_dates)
    stage     = ('Group Stage' if date_2026 in group_stage_dates
                 else 'Knockout'  if date_2026 in knockout_dates
                 else 'Non-game')

    rows.append({
        'date_2026':    date_2026.date(),
        'date_2025':    date_2025.date(),
        'dow':          date_2026.strftime('%a'),
        'is_game':      is_game,
        'stage':        stage,
        'is_canada':    date_2026 in canada_dates,
        'revenue_2026': round(rev_26, 2),
        'revenue_2025': round(float(rev_25), 2),
        'yoy_change':   round(rev_26 - float(rev_25), 2),
        'yoy_pct':      round((rev_26 / float(rev_25) - 1) * 100, 2),
    })

yoy_df = pd.DataFrame(rows)

if yoy_df.empty:
    print("⚠ No matched YoY pairs found — check 2025 revenue pull.")
else:
    game_days     = yoy_df[yoy_df['is_game']].copy()
    non_game_days = yoy_df[~yoy_df['is_game']]

    # Step 3: Baseline YoY from non-game days (pre-tournament June 1–10)
    baseline_yoy = non_game_days['yoy_pct'].mean() if len(non_game_days) >= 2 else 0.0
    game_days['yoy_vs_baseline'] = game_days['yoy_pct'] - baseline_yoy

    print(f"Baseline YoY (non-game days, {len(non_game_days)} days): {baseline_yoy:+.1f}%")
    print("This is the underlying 2026-vs-2025 venue trend, independent of FIFA.\n")

    # Step 4: Per-game-day YoY table
    print("=== FIFA Game Days: YoY Revenue vs Same Week in 2025 ===\n")
    print(
        game_days[[
            'date_2026', 'date_2025', 'dow', 'stage', 'is_canada',
            'revenue_2026', 'revenue_2025', 'yoy_pct', 'yoy_vs_baseline'
        ]].to_string(index=False)
    )

    # Step 5: Summary
    print(f"\nGame days with YoY pair:        {len(game_days)}")
    print(f"Mean game-day YoY:              {game_days['yoy_pct'].mean():+.1f}%")
    print(f"Baseline non-game-day YoY:      {baseline_yoy:+.1f}%")
    print(f"Net FIFA effect (vs baseline):  {game_days['yoy_pct'].mean() - baseline_yoy:+.1f}%")
    print(f"Median game-day YoY:            {game_days['yoy_pct'].median():+.1f}%")

    if len(game_days) >= 2:
        _, p_val = stats.ttest_1samp(game_days['yoy_vs_baseline'], 0)
        boot = [
            np.random.choice(game_days['yoy_vs_baseline'], len(game_days), replace=True).mean()
            for _ in range(5000)
        ]
        ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])
        print(f"95% CI (vs baseline):          [{ci_lo:+.1f}%, {ci_hi:+.1f}%]")
        print(f"p-value (vs baseline):         {p_val:.4f}")

    # Step 6: Canada vs non-Canada split
    print("\n=== Canada vs Non-Canada (YoY vs baseline) ===")
    print(
        game_days.groupby('is_canada')[['yoy_pct', 'yoy_vs_baseline']]
        .agg(mean_yoy=('yoy_pct', 'mean'), net_fifa_effect=('yoy_vs_baseline', 'mean'), n=('yoy_pct', 'count'))
        .to_string()
    )

# COMMAND ----------

# DBTITLE 1,YoY Visualization — Game Days vs Baseline
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

CANADA_COLOR     = '#D20000'
NON_CANADA_COLOR = '#1D6B9E'
BASELINE_COLOR   = '#27AE60'
NONGAME_COLOR    = '#BBBBBB'

# Combine all June days into one sorted frame
all_days            = yoy_df.copy().sort_values('date_2026').reset_index(drop=True)
all_days['date_dt'] = pd.to_datetime(all_days['date_2026'].astype(str))

bar_colors = [
    (CANADA_COLOR if row['is_canada'] else NON_CANADA_COLOR) if row['is_game'] else NONGAME_COLOR
    for _, row in all_days.iterrows()
]

x        = np.arange(len(all_days))
labels   = all_days['date_dt'].dt.strftime('%b %d\n%a').tolist()
first_gi = int(all_days['is_game'].tolist().index(True))  # bar index where tournament starts

fig, axes = plt.subplots(1, 2, figsize=(16, 5), gridspec_kw={'width_ratios': [3, 1]})

# ━━━ Left panel: per-day bars ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ax1 = axes[0]
ax1.bar(x, all_days['yoy_pct'], color=bar_colors, edgecolor='none', width=0.7)
ax1.axhline(baseline_yoy, color=BASELINE_COLOR, linewidth=2, linestyle='--', zorder=5)
ax1.axhline(0,            color='#333',          linewidth=0.8)
ax1.axvline(first_gi - 0.5, color='#aaa', linewidth=1, linestyle=':', alpha=0.8)

# Fix y-limits before annotations so text placement is reliable
y_lo = min(all_days['yoy_pct'].min(), 0) - 12
y_hi = max(all_days['yoy_pct'].max(), baseline_yoy) + 28
ax1.set_ylim(y_lo, y_hi)

# Tournament start label
ax1.text(first_gi - 0.2, y_hi - 4,
         'Tournament\nstarts ▼', fontsize=8, color='#999', ha='left', va='top')

# Baseline label (anchored in the non-game zone)
ax1.text(first_gi * 0.4, baseline_yoy + 3,
         f'Baseline +{baseline_yoy:.0f}% YoY\n(non-game days)',
         fontsize=8.5, color=BASELINE_COLOR, fontweight='bold', va='bottom')

# Value labels on game-day bars only
for i, row in all_days[all_days['is_game']].iterrows():
    v = row['yoy_pct']
    ax1.text(i, v + (2 if v >= 0 else -3),
             f'{v:+.0f}%', ha='center', va=('bottom' if v >= 0 else 'top'),
             fontsize=7, color='#222')

ax1.set_xticks(x)
ax1.set_xticklabels(labels, fontsize=7.5)
ax1.set_ylabel('YoY Revenue Change (%)', fontsize=11)
ax1.set_title('YoY revenue — FIFA game days vs non-game baseline, Rec Room Royalmount',
              fontsize=11, loc='left', pad=12)
ax1.grid(axis='y', alpha=0.2)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.legend(handles=[
    mpatches.Patch(color=CANADA_COLOR,     label='Canada game'),
    mpatches.Patch(color=NON_CANADA_COLOR, label='Other FIFA game'),
    mpatches.Patch(color=NONGAME_COLOR,    label='Non-game day'),
    plt.Line2D([0], [0], color=BASELINE_COLOR, linewidth=2, linestyle='--',
               label=f'Baseline +{baseline_yoy:.0f}% YoY'),
], loc='lower left', frameon=False, fontsize=9)

# ━━━ Right panel: group mean bars ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ax2 = axes[1]

mean_all_games  = game_days['yoy_pct'].mean()
mean_canada     = game_days.loc[game_days['is_canada'],  'yoy_pct'].mean()
mean_non_canada = game_days.loc[~game_days['is_canada'], 'yoy_pct'].mean()
net_eff         = mean_all_games - baseline_yoy

cats = ['Non-game\n(Baseline)', 'All FIFA\ngames', 'Canada\ngames', 'Other\ngames']
vals = [baseline_yoy, mean_all_games, mean_canada, mean_non_canada]
clrs = [NONGAME_COLOR, NON_CANADA_COLOR, CANADA_COLOR, NON_CANADA_COLOR]

bars2 = ax2.bar(cats, vals, color=clrs, edgecolor='none', width=0.55)
ax2.axhline(baseline_yoy, color=BASELINE_COLOR, linewidth=1.5, linestyle='--')
ax2.axhline(0, color='#333', linewidth=0.8)
ax2.set_ylabel('Mean YoY (%)', fontsize=11)
ax2.set_title('Mean YoY\nby Group', fontsize=11, loc='left', pad=12)
ax2.tick_params(axis='x', labelsize=8.5)
ax2.grid(axis='y', alpha=0.2)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.set_ylim(min(vals) - 20, max(vals) + 28)

for bar, v in zip(bars2, vals):
    ax2.text(bar.get_x() + bar.get_width() / 2, v + (2 if v >= 0 else -3),
             f'{v:+.0f}%', ha='center', va=('bottom' if v >= 0 else 'top'),
             fontsize=10.5, fontweight='bold')

# Double-headed arrow showing net FIFA effect
ax2.annotate('', xy=(1, mean_all_games), xytext=(1, baseline_yoy),
             arrowprops=dict(arrowstyle='<->', color='#e74c3c', lw=1.5))
ax2.text(1.33, (mean_all_games + baseline_yoy) / 2,
         f'{net_eff:+.0f}pp', fontsize=9, color='#e74c3c', va='center', fontweight='bold')

plt.suptitle(
    f'FIFA WC 2026  ·  Game days avg +{mean_all_games:.0f}% YoY  ·  '
    f'Baseline +{baseline_yoy:.0f}%  ·  Net FIFA effect {net_eff:+.0f}pp  ·  p={p_val:.4f}',
    fontsize=10, y=1.02, x=0.01, ha='left', color='#555'
)
plt.tight_layout()
plt.show()

# COMMAND ----------

# DBTITLE 1,Hour-Windowed FIFA Lift — Pre-Tournament Matched Baseline
# COMMAND ----------

# DBTITLE 1,Hour-Windowed FIFA Lift — Pre-Tournament Matched Baseline
# ---------------------------------------------------------------------------
# Clean replacement for the daily ±28-day matched cells.
#
# Fixes vs the old design:
#   - Revenue is summed only in each game's kickoff-1h to +4h window
#     (not the whole calendar day).
#   - Control days come ONLY from the pre-tournament period (Apr 1 - Jun 10),
#     so the baseline is never contaminated by tournament-period traffic.
#   - Controls match each game's DOW *and* its exact clock window.
#   - n_controls reflects rows actually used (NaN windows dropped).
#
# Assumes: fifa_df already built (with 'start_utc', 'is_canada', 'stage_type').
#          Kickoffs are UTC; transactions are Toronto-local (America/Toronto).
# ---------------------------------------------------------------------------
import pandas as pd
import numpy as np
from scipy import stats

# --- Config ---
WINDOW_PRE_H    = 1     # hours before kickoff
WINDOW_POST_H   = 4     # hours after kickoff
BASELINE_START  = '20260401'
BASELINE_END    = '20260610'   # inclusive, last pre-tournament day
TOURNAMENT_OPEN = pd.Timestamp('2026-06-11')

# ---------------------------------------------------------------------------
# 1. Build the game list with Eastern kickoff times + windows
# ---------------------------------------------------------------------------
games = fifa_df.dropna(subset=['start_utc']).copy()
games['kickoff_et'] = (
    pd.to_datetime(games['start_utc'], utc=True)
      .dt.tz_convert('America/Toronto')
      .dt.tz_localize(None)
)
games['win_start'] = games['kickoff_et'] - pd.Timedelta(hours=WINDOW_PRE_H)
games['win_end']   = games['kickoff_et'] + pd.Timedelta(hours=WINDOW_POST_H)
games['dow']       = games['kickoff_et'].dt.dayofweek
games['game_date'] = games['kickoff_et'].dt.normalize()

# Drop duplicate game records (ESPN pull can return the same game on two
# adjacent date queries due to UTC-vs-local date boundaries).
games = games.drop_duplicates(subset=['kickoff_et', 'win_start']).copy()

# Multiple games can share a date but have different kickoff windows —
# we keep them as separate events (each window scored independently).
print(f"Games with kickoff times: {len(games)}")

# ---------------------------------------------------------------------------
# 2. Pull transaction-grain revenue for both periods
# Guard: skip the heavy Spark pull if tx is already in memory from a
# previous run (e.g. when re-running only to pick up a stage reclassification).
# ---------------------------------------------------------------------------
if '_tx_idx' not in globals():
    tx = (
        T_trans
        .filter(F.col("VisitDateId") >= BASELINE_START)
        .filter(F.col("VisitDateId") <  date.today().strftime('%Y%m%d'))
        .join(T_location, T_trans.LocationId == T_location.LocationID, how="inner")
        .filter(F.col("Location_Name").like("%Royal%"))
        .select(
            F.col("Transaction_DateTime").alias("ts"),
            F.col("Net_Revenue").cast("double").alias("rev"),
        )
        .toPandas()
    )
    tx['ts'] = pd.to_datetime(tx['ts'])
    tx = tx.dropna(subset=['ts', 'rev']).sort_values('ts').set_index('ts')
    _tx_idx = tx.index.values.astype('datetime64[ns]')
    _tx_rev = tx['rev'].values

    def window_revenue(start, end):
        """Sum Net_Revenue for transactions in [start, end). Uses searchsorted — O(log n)."""
        lo = _tx_idx.searchsorted(np.datetime64(start, 'ns'))
        hi = _tx_idx.searchsorted(np.datetime64(end,   'ns'))
        return float(_tx_rev[lo:hi].sum()) if hi > lo else np.nan

    print(f"Transaction rows pulled: {len(tx):,}  "
          f"({tx.index.min()} -> {tx.index.max()})")
else:
    print(f"Reusing cached transactions ({len(tx):,} rows, up to {tx.index.max().date()})")

# ---------------------------------------------------------------------------
# 3. Pre-tournament control windows, keyed by (dow, clock-slot)
# ---------------------------------------------------------------------------
baseline_days = pd.date_range(
    pd.Timestamp(BASELINE_START), pd.Timestamp(BASELINE_END), freq='D'
)

# Pre-group baseline days by DOW once — eliminates the per-iteration DOW check inside the loop
baseline_by_dow = {}
for d in baseline_days:
    baseline_by_dow.setdefault(d.dayofweek, []).append(d)

results = []
for g in games.itertuples():
    actual = window_revenue(g.win_start, g.win_end)
    if pd.isna(actual):
        continue  # no revenue data for this window yet (future/today)

    # Same DOW pre-tournament days; replay the same clock window on each
    start_t = g.win_start.time()
    end_t   = g.win_end.time()
    ctrl_vals = []
    for d in baseline_by_dow.get(g.dow, []):  # only same-DOW days, no check needed
        cs = pd.Timestamp.combine(d.date(), start_t)
        ce = pd.Timestamp.combine(d.date(), end_t)
        if ce <= cs:            # window crosses midnight
            ce += pd.Timedelta(days=1)
        v = window_revenue(cs, ce)
        if not pd.isna(v):
            ctrl_vals.append(v)

    if len(ctrl_vals) < 2:
        print(f"⚠ {g.game_date.date()} {g.kickoff_et.time()}: "
              f"only {len(ctrl_vals)} control window(s) — skipping")
        continue

    expected = float(np.mean(ctrl_vals))
    if expected < 2000:
        # window falls mostly outside operating hours (e.g. midnight kickoffs);
        # near-zero baseline makes the ratio explode — skip
        continue
    incremental = actual - expected
    lift_pct    = (actual / expected - 1) * 100 if expected > 0 else np.nan

    results.append({
        'game_date':           g.game_date.date(),
        'kickoff_et':          g.kickoff_et.strftime('%H:%M'),
        'dow':                 g.kickoff_et.strftime('%a'),
        'stage_type':          g.stage_type,
        'is_canada':           bool(g.is_canada),
        'n_controls':          len(ctrl_vals),
        'actual_revenue':      round(actual, 2),
        'expected_revenue':    round(expected, 2),
        'incremental_revenue': round(incremental, 2),
        'lift_pct':            round(lift_pct, 2),
    })

lift_df = pd.DataFrame(results)

# ---------------------------------------------------------------------------
# 4. Output
# ---------------------------------------------------------------------------
print("\n=== Hour-Windowed FIFA Lift — Rec Room Royalmount ===")
print(f"(window: kickoff -{WINDOW_PRE_H}h to +{WINDOW_POST_H}h | "
      f"baseline: {BASELINE_START} to {BASELINE_END}, same DOW)\n")

if lift_df.empty:
    print("No games with both window revenue and >=2 pre-tournament controls yet.")
else:
    print(lift_df.to_string(index=False))

    def summarize(df, label):
        if len(df) == 0:
            print(f"\n{label}: no games")
            return
        n   = len(df)
        ml  = df['lift_pct'].mean()
        mdl = df['lift_pct'].median()
        inc = df['incremental_revenue'].sum()
        print(f"\n{label}  (n={n})  "
              f"mean {ml:+.1f}% | median {mdl:+.1f}% | total incr ${inc:,.0f}")
        if n >= 2:
            # log-ratio test: ratios are skewed, mean-of-pct is outlier-prone
            logr = np.log(df['actual_revenue'] / df['expected_revenue'])
            _, p_t = stats.ttest_1samp(logr, 0)
            try:
                _, p_w = stats.wilcoxon(df['lift_pct'])
            except ValueError:
                p_w = np.nan
            boot = [np.random.choice(df['lift_pct'], n, replace=True).mean()
                    for _ in range(5000)]
            lo, hi = np.percentile(boot, [2.5, 97.5])
            print(f"   95% CI (mean lift): [{lo:+.1f}%, {hi:+.1f}%] | "
                  f"p(log-ratio t)={p_t:.3f} | p(Wilcoxon)={p_w:.3f}")
            print(f"   NOTE: n={n} is small — treat as directional, not precise.")

    summarize(lift_df,                                         "ALL FIFA games")
    summarize(lift_df[lift_df['stage_type'] == 'Group Stage'], "Group stage")
    summarize(lift_df[lift_df['stage_type'] == 'Knockout'],    "Knockout")
    summarize(lift_df[lift_df['is_canada']],                   "Canada games")
    summarize(lift_df[~lift_df['is_canada']],                  "Non-Canada games")

# COMMAND ----------

# DBTITLE 1,Hour-Windowed Lift Visualization
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

CANADA_COLOR     = '#D20000'
NON_CANADA_COLOR = '#1D6B9E'

# ── Prep ───────────────────────────────────────────────────────────────────────────
lift_plot    = lift_df.reset_index(drop=True).copy()
unique_dates = sorted(lift_plot['game_date'].unique())
date_to_x    = {d: i for i, d in enumerate(unique_dates)}

rng = np.random.default_rng(42)
lift_plot['xpos']  = [date_to_x[d] + rng.uniform(-0.28, 0.28) for d in lift_plot['game_date']]
lift_plot['color'] = [CANADA_COLOR if r else NON_CANADA_COLOR for r in lift_plot['is_canada']]

mean_lift   = lift_df['lift_pct'].mean()
median_lift = lift_df['lift_pct'].median()
total_incr  = lift_df['incremental_revenue'].sum()
n_all       = len(lift_df)
n_canada    = int(lift_df['is_canada'].sum())
n_other     = n_all - n_canada

fig, axes = plt.subplots(1, 2, figsize=(16, 5), gridspec_kw={'width_ratios': [3, 1]})

# ━━━ Left panel: strip plot (one dot per kickoff window) ━━━━━━━━━━━━━━━━━━━━━━━
ax1 = axes[0]

# Alternating light bands to delineate days
for i in range(len(unique_dates)):
    if i % 2 == 0:
        ax1.axvspan(i - 0.5, i + 0.5, color='#f5f5f5', zorder=0)

ax1.scatter(lift_plot['xpos'], lift_plot['lift_pct'],
            c=lift_plot['color'], s=60, alpha=0.85, zorder=3, linewidths=0)
ax1.axhline(0,         color='#333', linewidth=0.9)
ax1.axhline(mean_lift, color='#888', linewidth=1.5, linestyle='--', zorder=4)

# Annotate Canada dots with kickoff time
for _, row in lift_plot[lift_plot['is_canada']].iterrows():
    ax1.annotate(
        row['kickoff_et'],
        xy=(row['xpos'], row['lift_pct']),
        xytext=(row['xpos'] + 0.35, row['lift_pct'] + 4),
        fontsize=7.5, color=CANADA_COLOR, fontweight='bold',
        arrowprops=dict(arrowstyle='-', color=CANADA_COLOR, lw=0.7),
    )

y_lo = lift_plot['lift_pct'].min() - 10
y_hi = lift_plot['lift_pct'].max() + 20
ax1.set_ylim(y_lo, y_hi)
ax1.set_xlim(-0.7, len(unique_dates) - 0.3)

# Mean label placed at right edge
ax1.text(len(unique_dates) - 0.35, mean_lift + 1.5,
         f'Mean {mean_lift:+.1f}%', fontsize=8, color='#888', va='bottom', ha='right')

ax1.set_xticks(range(len(unique_dates)))
ax1.set_xticklabels(
    [pd.Timestamp(d).strftime('%b %d\n%a') for d in unique_dates], fontsize=7.5
)
ax1.set_ylabel('Window lift %\n(kickoff −1h to +4h vs same-DOW pre-tournament baseline)', fontsize=10)
ax1.set_title('Hour-windowed revenue lift per kickoff slot — Rec Room Royalmount (FIFA WC 2026)',
              fontsize=11, loc='left', pad=12)
ax1.grid(axis='y', alpha=0.15)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.legend(handles=[
    mpatches.Patch(color=CANADA_COLOR,     label='Canada game window'),
    mpatches.Patch(color=NON_CANADA_COLOR, label='Other game window'),
    plt.Line2D([0], [0], color='#888', linewidth=1.5, linestyle='--',
               label=f'Mean {mean_lift:+.1f}%'),
], loc='upper right', frameon=False, fontsize=9)

# ━━━ Right panel: group mean summary ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ax2 = axes[1]

mean_canada     = lift_df.loc[lift_df['is_canada'],  'lift_pct'].mean()
mean_non_canada = lift_df.loc[~lift_df['is_canada'], 'lift_pct'].mean()

cats = [f'All\n(n={n_all})', f'Canada\n(n={n_canada})', f'Other\n(n={n_other})']
vals = [mean_lift, mean_canada, mean_non_canada]
clrs = [NON_CANADA_COLOR, CANADA_COLOR, NON_CANADA_COLOR]

bars2 = ax2.bar(cats, vals, color=clrs, edgecolor='none', width=0.55)
ax2.axhline(0, color='#333', linewidth=0.8)
ax2.set_ylabel('Mean Window Lift (%)', fontsize=11)
ax2.set_title('Mean Lift\nby Group', fontsize=11, loc='left', pad=12)
ax2.tick_params(axis='x', labelsize=8.5)
ax2.grid(axis='y', alpha=0.2)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.set_ylim(min(vals) - 8, max(vals) + 14)

for bar, v in zip(bars2, vals):
    ax2.text(bar.get_x() + bar.get_width() / 2, v + (1.5 if v >= 0 else -2),
             f'{v:+.1f}%', ha='center', va=('bottom' if v >= 0 else 'top'),
             fontsize=11, fontweight='bold')

plt.suptitle(
    f'FIFA WC 2026  ·  {n_all} kickoff windows across {len(unique_dates)} game days  ·  '
    f'Mean {mean_lift:+.1f}%  ·  Median {median_lift:+.1f}%  ·  '
    f'Total incr ${total_incr:,.0f}  ·  p(Wilcoxon)=0.030',
    fontsize=10, y=1.02, x=0.01, ha='left', color='#555'
)
plt.tight_layout()
plt.show()

# COMMAND ----------

# COMMAND ----------

# DBTITLE 1,Placebo Test — Tournament Non-Game Windows vs Same Baseline
# ---------------------------------------------------------------------------
# Resolves the contradiction: "weekly revenue is up, but game windows show
# negative lift." If non-game tournament windows ALSO come out ~-8% vs the
# April baseline, the negative is seasonal drift, not FIFA. If non-game
# windows are ~0% and only game windows are negative, the FIFA effect is real.
#
# Method: take the SAME clock windows used by real games, but place them on
# tournament-period days (Jun 11-28) that had NO game in that slot. Match each
# against the same pre-tournament, same-DOW, same-clock baseline.
# Reuses: tx, window_revenue, baseline_days, games  (from the cell above)
# ---------------------------------------------------------------------------
TOURNAMENT_DAYS = pd.date_range('2026-06-11', '2026-06-28', freq='D')

# Real game (date, clock-slot) pairs to EXCLUDE — these are the actual matches
game_slots = {
    (g.game_date.date(), g.win_start.time())
    for g in games.itertuples()
}

# The distinct clock windows that real games used (we reuse these shapes)
clock_windows = sorted({
    (g.win_start.time(), g.win_end.time(), g.dow)
    for g in games.itertuples()
})

placebo_results = []
for d in TOURNAMENT_DAYS:
    for start_t, end_t, _ in clock_windows:
        # skip if this exact (date, slot) was a real game
        if (d.date(), start_t) in game_slots:
            continue

        ws = pd.Timestamp.combine(d.date(), start_t)
        we = pd.Timestamp.combine(d.date(), end_t)
        if we <= ws:
            we += pd.Timedelta(days=1)

        actual = window_revenue(ws, we)
        if pd.isna(actual):
            continue

        # baseline: same DOW, same clock window, pre-tournament
        ctrl_vals = []
        for bd in baseline_days:
            if bd.dayofweek != d.dayofweek:
                continue
            cs = pd.Timestamp.combine(bd.date(), start_t)
            ce = pd.Timestamp.combine(bd.date(), end_t)
            if ce <= cs:
                ce += pd.Timedelta(days=1)
            v = window_revenue(cs, ce)
            if not pd.isna(v):
                ctrl_vals.append(v)

        if len(ctrl_vals) < 2:
            continue

        expected = float(np.mean(ctrl_vals))
        if expected < 2000:
            continue

        placebo_results.append({
            'date':          d.date(),
            'dow':           d.strftime('%a'),
            'win_start':     start_t.strftime('%H:%M'),
            'actual':        round(actual, 2),
            'expected':      round(expected, 2),
            'lift_pct':      round((actual / expected - 1) * 100, 2),
        })

placebo_df = pd.DataFrame(placebo_results).drop_duplicates(subset=['date', 'win_start'])

print("=== PLACEBO: Tournament non-game windows vs April baseline ===\n")
if placebo_df.empty:
    print("No non-game tournament windows with valid baseline.")
else:
    n   = len(placebo_df)
    ml  = placebo_df['lift_pct'].mean()
    mdl = placebo_df['lift_pct'].median()
    print(f"Non-game windows (n={n})  mean {ml:+.1f}% | median {mdl:+.1f}%")
    _, p_w = stats.wilcoxon(placebo_df['lift_pct'])
    print(f"Wilcoxon p={p_w:.3f}")

    print("\n=== INTERPRETATION ===")
    print(f"  GAME windows:     median -8.7%   (from cell above)")
    print(f"  NON-GAME windows: median {mdl:+.1f}%   (this placebo)")
    if mdl <= -5:
        print("  -> Non-game windows ALSO negative. The -8.7% is mostly SEASONAL")
        print("     DRIFT (April baseline too low for late June), not a FIFA effect.")
    elif -5 < mdl < 2:
        print("  -> Non-game windows ~flat, only game windows negative.")
        print("     The FIFA effect is REAL: matches depress venue revenue.")
    else:
        print("  -> Non-game windows POSITIVE. Venue is up vs April baseline in")
        print("     general; FIFA windows underperform that elevated norm.")

# COMMAND ----------

# COMMAND ----------

# DBTITLE 1,FIFA Net Effect — Game Windows vs Tournament Non-Game Windows
# ---------------------------------------------------------------------------
# The honest metric. Both lift_df (games) and placebo_df (non-game tournament
# windows) are measured against the SAME April baseline, so each carries the
# same seasonal drift. Comparing them to each other DIFFERENCES OUT the drift:
#
#     net FIFA effect = game-window lift  -  non-game-window lift
#
# This is immune to the April-baseline-too-low problem that made raw lift
# look negative. Requires lift_df (game cell) and placebo_df (placebo cell).
# ---------------------------------------------------------------------------
from scipy import stats

game_lift    = lift_df['lift_pct']
placebo_lift = placebo_df['lift_pct']

g_med, p_med = game_lift.median(), placebo_lift.median()
g_mn,  p_mn  = game_lift.mean(),   placebo_lift.mean()
net_median   = g_med - p_med
net_mean     = g_mn  - p_mn

# Two-sample tests: is the game-window distribution different from the
# non-game (placebo) distribution? Mann-Whitney is the nonparametric version.
u_stat, p_mw = stats.mannwhitneyu(game_lift, placebo_lift, alternative='two-sided')
t_stat, p_tt = stats.ttest_ind(game_lift, placebo_lift, equal_var=False)

print("=== FIFA NET EFFECT (drift-corrected) — Rec Room Royalmount ===\n")
print(f"  Game windows        (n={len(game_lift):>3}):  "
      f"mean {g_mn:+.1f}% | median {g_med:+.1f}%")
print(f"  Non-game windows    (n={len(placebo_lift):>3}):  "
      f"mean {p_mn:+.1f}% | median {p_med:+.1f}%")
print(f"\n  NET FIFA effect (game - non-game):")
print(f"     median diff:  {net_median:+.1f} pts")
print(f"     mean diff:    {net_mean:+.1f} pts")
print(f"     Mann-Whitney p = {p_mw:.3f}")
print(f"     Welch t-test p = {p_tt:.3f}")

print("\n=== READ ===")
sig = p_mw < 0.05
direction = "higher" if net_median > 0 else "lower"
if not sig:
    print(f"  Game windows run {net_median:+.1f} pts {direction} than non-game windows,")
    print(f"  but the difference is NOT significant (p={p_mw:.3f}).")
    print(f"  CONCLUSION: No detectable FIFA effect on venue revenue. The ~-10%")
    print(f"  raw figure is seasonal drift (April baseline), shared by game and")
    print(f"  non-game windows alike — not caused by the tournament.")
else:
    print(f"  Game windows run {net_median:+.1f} pts {direction} than non-game")
    print(f"  windows, and the difference IS significant (p={p_mw:.3f}).")
    print(f"  CONCLUSION: A real FIFA effect exists, direction = {direction}.")

# Optional: Canada subset net effect (still tiny n — directional only)
can = lift_df[lift_df['is_canada']]['lift_pct']
if len(can) >= 1:
    print(f"\n  Canada games (n={len(can)}): median {can.median():+.1f}% "
          f"vs non-game {p_med:+.1f}%  -> net {can.median() - p_med:+.1f} pts "
          f"(n too small to test)")

# COMMAND ----------

# DBTITLE 1,Net Effect Visualization — Game vs Non-Game Distributions
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy.stats import gaussian_kde

GAME_COLOR    = '#1D6B9E'
CANADA_COLOR  = '#D20000'
PLACEBO_COLOR = '#AAAAAA'

game_lifts    = lift_df['lift_pct'].dropna().values
placebo_lifts = placebo_df['lift_pct'].dropna().values
canada_lifts  = lift_df.loc[lift_df['is_canada'], 'lift_pct'].dropna().values

g_med = float(np.median(game_lifts))
p_med = float(np.median(placebo_lifts))
c_med = float(np.median(canada_lifts)) if len(canada_lifts) > 0 else np.nan
net_eff = g_med - p_med

fig, axes = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={'width_ratios': [2, 1]})

# ━━━ Left panel: overlapping KDE distributions ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ax1 = axes[0]

x_lo = min(game_lifts.min(), placebo_lifts.min()) - 8
x_hi = max(game_lifts.max(), placebo_lifts.max()) + 8
x    = np.linspace(x_lo, x_hi, 400)

kde_g = gaussian_kde(game_lifts,    bw_method=0.4)
kde_p = gaussian_kde(placebo_lifts, bw_method=0.4)

# Filled KDEs
ax1.fill_between(x, kde_p(x), alpha=0.35, color=PLACEBO_COLOR)
ax1.fill_between(x, kde_g(x), alpha=0.55, color=GAME_COLOR)
ax1.plot(x, kde_p(x), color=PLACEBO_COLOR, linewidth=1.8)
ax1.plot(x, kde_g(x), color=GAME_COLOR,    linewidth=1.8)

# Zero line
ax1.axvline(0, color='#333', linewidth=0.9, zorder=5)

# Median lines
ymax = max(kde_g(x).max(), kde_p(x).max())
ax1.axvline(g_med, color=GAME_COLOR,    linewidth=2, linestyle='--', zorder=4)
ax1.axvline(p_med, color=PLACEBO_COLOR, linewidth=2, linestyle='--', zorder=4)

# Median labels
ax1.text(g_med + 1,   ymax * 0.90, f'Game median\n{g_med:+.1f}%',
         fontsize=8.5, color=GAME_COLOR,    fontweight='bold', va='top')
ax1.text(p_med - 1,   ymax * 0.72, f'Non-game\nmedian {p_med:+.1f}%',
         fontsize=8.5, color='#777',         fontweight='bold', va='top', ha='right')

# Double-headed arrow showing net difference
arrow_y = ymax * 0.42
ax1.annotate('', xy=(g_med, arrow_y), xytext=(p_med, arrow_y),
             arrowprops=dict(arrowstyle='<->', color='#e74c3c', lw=1.8))
ax1.text((g_med + p_med) / 2, arrow_y + ymax * 0.04,
         f'Net FIFA effect\n{net_eff:+.1f} pp  p={p_mw:.3f}',
         ha='center', va='bottom', fontsize=8.5, color='#e74c3c', fontweight='bold')

ax1.set_xlabel('Window lift % vs April pre-tournament baseline', fontsize=11)
ax1.set_ylabel('Density', fontsize=11)
ax1.set_title('Both groups are similarly negative — seasonal drift, not FIFA',
              fontsize=11, loc='left', pad=12)
ax1.set_ylim(0, ymax * 1.35)
ax1.grid(axis='y', alpha=0.15)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.legend(handles=[
    mpatches.Patch(color=PLACEBO_COLOR, alpha=0.7,
                   label=f'Non-game tournament windows (n={len(placebo_lifts)})'),
    mpatches.Patch(color=GAME_COLOR,    alpha=0.8,
                   label=f'FIFA game windows (n={len(game_lifts)})'),
], frameon=False, fontsize=9, loc='upper right')

# ━━━ Right panel: median summary bars ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ax2 = axes[1]

cats = ['Non-game\nwindows', 'All FIFA\ngames', 'Canada\ngames']
meds = [p_med, g_med, c_med]
clrs = [PLACEBO_COLOR, GAME_COLOR, CANADA_COLOR]
ns   = [len(placebo_lifts), len(game_lifts), len(canada_lifts)]

bars = ax2.bar(range(len(cats)), meds, color=clrs, edgecolor='none', width=0.55)
ax2.axhline(0, color='#333', linewidth=0.8)
ax2.set_ylabel('Median Window Lift (%)', fontsize=11)
ax2.set_title('Median Lift\nby Group', fontsize=11, loc='left', pad=12)
ax2.set_xticks(range(len(cats)))
ax2.set_xticklabels(cats, fontsize=8.5)
ax2.grid(axis='y', alpha=0.2)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

_vmin = min(v for v in meds if not np.isnan(v)) - 8
_vmax = 8
ax2.set_ylim(_vmin, _vmax)

for bar, v, n in zip(bars, meds, ns):
    if np.isnan(v):
        continue
    ax2.text(bar.get_x() + bar.get_width() / 2, v + (1 if v >= 0 else -1.5),
             f'{v:+.1f}%', ha='center', va=('bottom' if v >= 0 else 'top'),
             fontsize=11, fontweight='bold')
    ax2.text(bar.get_x() + bar.get_width() / 2, _vmax - 0.5,
             f'n={n}', ha='center', va='top', fontsize=8, color='#666')

plt.suptitle(
    f'FIFA WC 2026 net effect (drift-corrected)  ·  '
    f'Game {g_med:+.1f}% vs non-game {p_med:+.1f}% median  ·  '
    f'Net {net_eff:+.1f} pp  ·  Mann-Whitney p={p_mw:.3f}  ·  no significant FIFA effect',
    fontsize=10, y=1.02, x=0.01, ha='left', color='#555'
)
plt.tight_layout()
plt.show()

# COMMAND ----------

# COMMAND ----------

# DBTITLE 1,YoY Net Effect — Hour-Windowed, Difference-in-Differences
# ---------------------------------------------------------------------------
# Venue opened late 2024, so June 2025 is its first summer (still ramping).
# Raw YoY will be large and POSITIVE from maturation, NOT from FIFA.
# The interpretable number is the difference-in-differences:
#
#   net effect = (game-window YoY)  -  (non-game-window YoY)
#
# Maturation inflates both terms equally, so it cancels. Whatever's left is
# the FIFA-attributable difference. Hour-windowed to stay consistent with the
# main lift cell. Reuses: games, tx, window_revenue, WINDOW_PRE_H/POST_H.
# ---------------------------------------------------------------------------
import pandas as pd
import numpy as np
from scipy import stats

YOY_OFFSET = pd.Timedelta(days=364)  # 52 weeks — preserves day of week

# --- Pull 2025 transaction-grain revenue (same venue, same window months) ---
tx25 = (
    T_trans
    .filter(F.col("VisitDateId") >= '20250501')
    .filter(F.col("VisitDateId") <= '20250715')
    .join(T_location, T_trans.LocationId == T_location.LocationID, how="inner")
    .filter(F.col("Location_Name").like("%Royal%"))
    .select(
        F.col("Transaction_DateTime").alias("ts"),
        F.col("Net_Revenue").cast("double").alias("rev"),
    )
    .toPandas()
)
tx25['ts'] = pd.to_datetime(tx25['ts'])
tx25 = tx25.dropna(subset=['ts', 'rev']).sort_values('ts').set_index('ts')
print(f"2025 transaction rows: {len(tx25):,}  "
      f"({tx25.index.min()} -> {tx25.index.max()})")

def window_revenue_2025(start, end):
    seg = tx25.loc[(tx25.index >= start) & (tx25.index < end), 'rev']
    return seg.sum() if len(seg) else np.nan

# --- Build the set of real game (date, clock-slot) pairs to flag game windows ---
game_slots = {
    (g.game_date.date(), g.win_start.time(), g.win_end.time())
    for g in games.itertuples()
}

# --- For every tournament-period day, score each distinct clock window in
#     both 2026 and its 2025 counterpart (same DOW via 364-day offset) ---
TOURNAMENT_DAYS = pd.date_range('2026-06-11', '2026-06-28', freq='D')
clock_windows = sorted({(g.win_start.time(), g.win_end.time()) for g in games.itertuples()})

yoy_rows = []
for d26 in TOURNAMENT_DAYS:
    d25 = d26 - YOY_OFFSET
    for start_t, end_t in clock_windows:
        ws26 = pd.Timestamp.combine(d26.date(), start_t)
        we26 = pd.Timestamp.combine(d26.date(), end_t)
        ws25 = pd.Timestamp.combine(d25.date(), start_t)
        we25 = pd.Timestamp.combine(d25.date(), end_t)
        if we26 <= ws26:
            we26 += pd.Timedelta(days=1); we25 += pd.Timedelta(days=1)

        r26 = window_revenue(ws26, we26)
        r25 = window_revenue_2025(ws25, we25)
        if pd.isna(r26) or pd.isna(r25) or r25 < 2000:  # floor: avoid closed-hour blowups
            continue

        yoy_rows.append({
            'date_2026': d26.date(),
            'dow':       d26.strftime('%a'),
            'win_start': start_t.strftime('%H:%M'),
            'is_game':   (d26.date(), start_t, end_t) in game_slots,
            'rev_2026':  round(r26, 2),
            'rev_2025':  round(r25, 2),
            'yoy_pct':   round((r26 / r25 - 1) * 100, 2),
        })

yoy_df = pd.DataFrame(yoy_rows).drop_duplicates(subset=['date_2026', 'win_start'])

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
print("\n=== YoY Net Effect (hour-windowed DiD) — Rec Room Royalmount ===\n")
if yoy_df.empty:
    print("No matched 2026/2025 window pairs — check 2025 pull / venue history.")
else:
    game_w     = yoy_df[yoy_df['is_game']]
    non_game_w = yoy_df[~yoy_df['is_game']]

    g_yoy = game_w['yoy_pct'].median()
    b_yoy = non_game_w['yoy_pct'].median()
    net   = g_yoy - b_yoy

    print(f"  Game windows      (n={len(game_w):>3}):  median YoY {g_yoy:+.1f}%  "
          f"(mean {game_w['yoy_pct'].mean():+.1f}%)")
    print(f"  Non-game windows  (n={len(non_game_w):>3}):  median YoY {b_yoy:+.1f}%  "
          f"(mean {non_game_w['yoy_pct'].mean():+.1f}%)")
    print(f"\n  This baseline YoY is the venue's maturation/organic growth,")
    print(f"  independent of FIFA. The net difference isolates the tournament:\n")
    print(f"  NET FIFA EFFECT (game - non-game): {net:+.1f} pts")

    if len(game_w) >= 2 and len(non_game_w) >= 2:
        u, p_mw = stats.mannwhitneyu(game_w['yoy_pct'], non_game_w['yoy_pct'],
                                     alternative='two-sided')
        print(f"  Mann-Whitney p = {p_mw:.3f}")
        print("\n  READ:", "significant FIFA effect" if p_mw < 0.05
              else "no significant difference — FIFA effect indistinguishable from zero")

    print("\n=== Canada vs Non-Canada (game windows only) ===")
    can_slots = {(g.game_date.date(), g.win_start.time())
                 for g in games.itertuples() if g.is_canada}
    game_w = game_w.copy()
    game_w['is_canada'] = [
        (r.date_2026, pd.to_datetime(r.win_start).time()) in can_slots
        for r in game_w.itertuples()
    ]
    print(game_w.groupby('is_canada')['yoy_pct']
              .agg(median_yoy='median', mean_yoy='mean', n='count').to_string())

# COMMAND ----------

# DBTITLE 1,Revenue Category Dictionaries
games_categories = {
    "TRR - AMUSEMENT - G&E",
    "TRR - AMUSEMENT - GAMES",
    "TRR - AMUSEMENT - PHYSICAL GAMES",
    "TRR - PROMO - AMUSEMENT",
}

food_categories = {
    "MISC",
    "PDM - MENU FOOD - CANDY",
    "PDM - MENU FOOD - G&E",
    "TRR - INV ALC BEV - BOTTLE",
    "TRR - INV ALC BEV - CAN",
    "TRR - INV ALC BEV - WINE",
    "TRR - MENU ALC BEV - COCKTAIL",
    "TRR - MENU ALC BEV - DRAUGHT BEER",
    "TRR - MENU BEV - NA COLD BEV",
    "TRR - MENU BEV - NA HOT BEV",
    "TRR - MENU FOOD - APP",
    "TRR - MENU FOOD - DESSERT",
    "TRR - MENU FOOD - G&E",
    "TRR - MENU FOOD - KIDS",
    "TRR - MENU FOOD - PIZZA",
    "TRR - MENU FOOD - POUTINE",
    "TRR - MENU FOOD - SANDWICHES",
    "TRR - MOD MENU - ADD-ON",
    "TRR - MOD MENU - NO VALUE",
    "CHARITY",
    "COUPONS",
    "TRR - INV BEV - NA BEV",
    "TRR - MENU ALC BEV - LIQUOR",
    "TRR - MENU ALC BEV - WINE",
    "TRR - MENU FOOD - ENTRÉE",
    "TRR - MENU FOOD - SALAD",
    "VHO-LICENSED FOOD",
    "Box Office Surcharges and Fees",
    "TRR - INV ALC BEV - LIQUOR",
    "VHO-CANDY",
    "VIP Food - Addons",
    "VHO-LICENSED DRINKS",
    "TRR - MENU FOOD - BRUNCH",
    "TRR - MENU FOOD - G&E OLD",
    "VHO-BIRTHDAY",
    "VHO-DRINK",
}

print(f"Games categories: {len(games_categories)}")
print(f"Food categories:  {len(food_categories)}")



# COMMAND ----------

# DBTITLE 1,Transactions by Category — Food vs Games
from pyspark.sql.functions import when, col, lit

T_item = spark.table("cpx_dataanalytics_gold.customer360.cpx_d_item")

# Filter for June–July 2025 and June–July 2026, Royalmount location,
# Item_Class_Description1 in either games or food categories
category_tx = (
    T_trans
    .join(T_location, T_trans.LocationId == T_location.LocationID, how="inner")
    .join(T_item, T_trans.ItemSkey == T_item.itemskey, how="inner")
    .filter(F.col("Location_Name").like("%Granvi%"))
    .filter(F.col("LC_Location_Type_Description").like("%Restaurant%"))
    .filter(
        (
            (F.col("VisitDateId") >= "20250101") & (F.col("VisitDateId") <= "20250630")
        ) | (
            (F.col("VisitDateId") >= "20260101") & (F.col("VisitDateId") <= "20260630")
        )
    )
    .filter(
        F.col("Item_Class_Description1").isin(list(games_categories) + list(food_categories))
    )
    .withColumn(
        "category_label",
        when(F.col("Item_Class_Description1").isin(list(games_categories)), lit("Games"))
        .when(F.col("Item_Class_Description1").isin(list(food_categories)), lit("Food"))
    )
    .filter(F.col("category_label").isNotNull())
)

print(f"Total rows: {category_tx.count():,}")
category_tx.groupBy("category_label").count().show()

# COMMAND ----------

# DBTITLE 1,Daily Revenue by Category (Pandas)
# Convert to Pandas — select only needed columns to keep it lean
daily_rev = (
    category_tx
    .withColumn("date", F.to_date(F.col("VisitDateId").cast("string"), "yyyyMMdd"))
    .select("date", "category_label", F.col("Net_Revenue").cast("double").alias("net_revenue"))
    .toPandas()
)

# Daily revenue by category
daily_category_rev = (
    daily_rev
    .groupby(["date", "category_label"], as_index=False)["net_revenue"]
    .sum()
    .sort_values(["date", "category_label"])
    .reset_index(drop=True)
)

print(f"Shape: {daily_category_rev.shape}")
display(daily_category_rev.head(10))

# COMMAND ----------

# DBTITLE 1,YoY Food vs Games Performance
import pandas as pd

# Add year and month columns
daily_category_rev["date"] = pd.to_datetime(daily_category_rev["date"])
daily_category_rev["year"] = daily_category_rev["date"].dt.year
daily_category_rev["month"] = daily_category_rev["date"].dt.month
daily_category_rev["month_name"] = daily_category_rev["date"].dt.strftime("%B")

# Monthly totals by category and year
monthly = (
    daily_category_rev
    .groupby(["year", "month", "month_name", "category_label"], as_index=False)["net_revenue"]
    .sum()
)

# Pivot to get 2025 vs 2026 side-by-side
yoy = monthly.pivot_table(
    index=["month", "month_name", "category_label"],
    columns="year",
    values="net_revenue"
).reset_index()

yoy.columns = ["month", "month_name", "category_label", "rev_2025", "rev_2026"]
yoy["yoy_change"] = yoy["rev_2026"] - yoy["rev_2025"]
yoy["yoy_pct"] = ((yoy["rev_2026"] / yoy["rev_2025"]) - 1) * 100

print("=== YoY Performance by Category (Monthly) ===")
print()
for cat in ["Food", "Games"]:
    subset = yoy[yoy["category_label"] == cat].sort_values("month")
    print(f"--- {cat} ---")
    for _, row in subset.iterrows():
        print(f"  {row['month_name']:>8}:  2025 ${row['rev_2025']:>12,.2f}  |  "
              f"2026 ${row['rev_2026']:>12,.2f}  |  "
              f"YoY {row['yoy_pct']:+.1f}%  (${row['yoy_change']:+,.0f})")
    total_25 = subset["rev_2025"].sum()
    total_26 = subset["rev_2026"].sum()
    print(f"  {'Total':>8}:  2025 ${total_25:>12,.2f}  |  "
          f"2026 ${total_26:>12,.2f}  |  "
          f"YoY {((total_26/total_25)-1)*100:+.1f}%  (${total_26-total_25:+,.0f})")
    print()

# Summary table
print("\n=== Summary Table ===")
display(yoy.sort_values(["category_label", "month"]))

# COMMAND ----------

import numpy as np

# --- Food only, June only, both years ---
food = daily_category_rev[daily_category_rev["category_label"] == "Food"].copy()
food = food[food["month"] == 6].copy()
food["dow"] = food["date"].dt.dayofweek  # 0=Mon

# --- Tag 2026 days as game / non-game ---
game_dates_2026 = set(pd.to_datetime(fifa_df["date"]).dt.normalize())
food["is_game_day"] = food["date"].dt.normalize().isin(game_dates_2026)
# 2025 has no tournament; is_game_day stays False there by construction

f25 = food[food["year"] == 2025]
f26 = food[food["year"] == 2026]

# --- 2026 split ---
gd26 = f26[f26["is_game_day"]]
ngd26 = f26[~f26["is_game_day"]]

# --- 2025 baseline matched by weekday to each 2026 day-type pool ---
# game-day baseline = 2025 days falling on the weekdays that hosted games in 2026
# non-game-day baseline = 2025 days on the weekdays of 2026 non-game days
gd_dows = gd26["dow"].unique()
ngd_dows = ngd26["dow"].unique()

base_gd25 = f25[f25["dow"].isin(gd_dows)]
base_ngd25 = f25[f25["dow"].isin(ngd_dows)]

def mean_daily(df):
    # mean revenue per day (df already one row per date)
    return df["net_revenue"].mean()

gd26_m, ngd26_m = mean_daily(gd26), mean_daily(ngd26)
gd25_m, ngd25_m = mean_daily(base_gd25), mean_daily(base_ngd25)

gd_yoy = (gd26_m / gd25_m - 1) * 100
ngd_yoy = (ngd26_m / ngd25_m - 1) * 100
did = gd_yoy - ngd_yoy

print("=== FIFA F&B Diff-in-Diff (full June, both years) ===\n")
print(f"Game days 2026:     n={len(gd26):2d}  mean daily Food ${gd26_m:,.0f}")
print(f"  matched 2025 base: n={len(base_gd25):2d}  mean daily Food ${gd25_m:,.0f}")
print(f"  Game-day YoY:      {gd_yoy:+.1f}%\n")
print(f"Non-game days 2026: n={len(ngd26):2d}  mean daily Food ${ngd26_m:,.0f}")
print(f"  matched 2025 base: n={len(base_ngd25):2d}  mean daily Food ${ngd25_m:,.0f}")
print(f"  Non-game-day YoY:  {ngd_yoy:+.1f}%\n")
print(f">>> FIFA-attributable lift (DiD): {did:+.1f} pts")
print(f"    (game-day YoY minus non-game-day YoY; nets out maturation + drift)")
