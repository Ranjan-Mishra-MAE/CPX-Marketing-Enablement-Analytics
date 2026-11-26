# MMM Pipeline Flow — EXH + LBE

> **Folder**: `MMM Final Trainings`  
> **Last updated**: 2026-06-11  
> **Models**: EXH Phase 2 Iter 2 (Mixed Hierarchical) | LBE (Log-Revenue Bayesian)

---

## Combined Pipeline Overview

```
╔═══════════════════════════════════════════════════════════════════════════╗
║                  EXH PIPELINE                    LBE PIPELINE          ║
╠═════════════════════════════════════╬═════════════════════════════════════╣
║  1. EXH Model Training            ║  L1. LBE Model Training           ║
║      ▼                            ║      ▼                            ║
║  2. EXH Diagnostics  (parallel)   ║  L2. LBE Diagnostics  (parallel)  ║
║  3. EXH LOB Decomposition         ║  L3. LBE Budget Scenarios         ║
║      ▼                            ║                                   ║
║  4. EXH Budget Scenarios          ║                                   ║
╚═════════════════╬═══════════════════╬═══════════════════╬═════════════════╝
                  └───────────────────┬───────────────────┘
                                    ▼
              ┌───────────────────────────────────────┐
              │  5. Presentation Charts              │
              │  Plots and Graphs                     │
              │  (EXH cells 3–22 | LBE cells 23–33)  │
              └───────────────────────────────────────┘
```

---

## EXH Pipeline — Execution Order

```
┌─────────────────────────────────────────────────────────────────────┐
│  1. Model Training                                                  │
│     EXH MMM Phase 2 Iter 2 Model Training                          │
│     (35 cells · ~45 min on GPU cluster)                             │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
               ┌───────────────┴───────────────┐
               ▼                               ▼
┌──────────────────────────────┐  ┌──────────────────────────────────┐
│  2. Diagnostics              │  │  3. LOB Decomposition            │
│  EXH MMM Phase 2 Iter 2     │  │  EXH_MMM_LOB_Decomposition_      │
│  Diagnostics                 │  │  Graft_2                         │
│  (37 cells)                  │  │  (24 cells)                      │
└──────────────┬───────────────┘  └───────────────┬──────────────────┘
               │                                  │
               └───────────────┬──────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  4. Budget Scenarios                                                │
│     EXH_MMM_Budget_Scenarios_FINAL                                  │
│     (37 cells)                                                      │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  5. Plots and Graphs                                                │
│     (33 cells · EXH + LBE presentation charts)                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Artifact Lineage

| Artifact | Written By | Read By |
| --- | --- | --- |
| `trace_iter2_mixed_hier.nc` | 1. Training | 2. Diagnostics, 4. Budget Scenarios |
| `scaler_X.pkl` | 1. Training | 2. Diagnostics |
| `scaler_y.pkl` | 1. Training | 2. Diagnostics |
| `spend_train_max_iter2.pkl` | 1. Training | 2, 3, 4 |
| `response_params.pkl` | 1. Training | 2, 3, 4 |
| `model_meta.pkl` | 1. Training | 2, 3, 4, 5 |
| `df_full.parquet` | 1. Training | 2, 3, 4, 5 |
| `decomp_lob_results.pkl` | 3. LOB Decomp | 4. Budget Scenarios, 5. Plots |
| `lbe_mmm_results.pkl` | (external) | 5. Plots |

**Storage paths:**
- EXH Outcomes: `/Volumes/data_exploration_hub/marketing_hub/model_data/Final_Data/MMM_EXH_Outcomes/`
- LOB Decomp: `/Volumes/data_exploration_hub/marketing_hub/model_data/Final_Data/MMM_EXH_LOB_Decomp/`
- LBE Model: `/Volumes/data_exploration_hub/marketing_hub/model_data/Final_Data/MMM_LBE/`

---

## Notebook 1: EXH MMM Phase 2 Iter 2 Model Training

**Purpose**: Trains the Mixed Hierarchical Bayesian MMM and exports all artifacts.  
**Runtime**: ~45 min (NUTS sampling with JAX backend)  
**Input**: `EXH_ModelData.xlsx` (165 weeks, 106 columns)

### Cell Flow

| # | Title | Description |
| --- | --- | --- |
| 1 | Overview (markdown) | Script purpose, artifacts manifest |
| 2 | Install Dependencies | pymc, arviz, numpyro, jax, scikit-learn |
| 3 | Imports | Core library imports |
| 4-5 | Configuration | Paths, artifact directory setup |
| 6-7 | Data Loading & Feature Engineering | Load Excel, encode flags, season dummies, trend |
| 8-9 | Feature Selection | Drop 26 multicollinear features, retain 22 baseline |
| 10-11 | Train/Val/Test Split | Chronological: train / 26-wk val / 15-wk test |
| 12-13 | Scaling | StandardScaler on X and y (fit on train only) |
| 14-17 | Media Metadata & LOB Aggregation | Parse 54 media cols into Funnel x LOB x Channel, group into 10 combos |
| 18-19 | Media Spend Scaling | Normalize spend by train-period max per combo |
| 20 | Sparsity Audit | Identify combos with >60% zeros |
| 21-22 | Response Functions | Adstock (geometric), saturation (logistic + Hill) |
| 23-24 | Model Definition & NUTS Sampling | PyMC model with hierarchical priors, 4 chains x 200 draws |
| 25-26 | Convergence Diagnostics | R-hat, ESS, divergence checks |
| 27-28 | In-Sample Accuracy | MAPE, R-squared on train |
| 29-35 | Artifact Export | Save trace, scalers, metadata, response params, df_full.parquet |

---

## Notebook 2: EXH MMM Phase 2 Iter 2 Diagnostics

**Purpose**: Produces all model diagnostic charts and business decomposition visualizations.  
**Input**: All artifacts from Notebook 1  
**Output**: PNG charts saved to artifact path

### Cell Flow

| # | Title | Description |
| --- | --- | --- |
| 1 | Overview (markdown) | Chart manifest |
| 2 | Install Dependencies | arviz, scipy, matplotlib, netcdf4 |
| 3 | Imports | Core library imports |
| 4-5 | Configuration | Paths, brand palette, save_fig helper |
| 6-7 | Trace Inspection | Load trace, check tree depth & leapfrog stats |
| 8-9 | Load Artifacts | Trace, scalers, metadata, parquet |
| 10-11 | Colour Maps & Constants | Brand palette, LOB colors, year styles, priors |
| 12-13 | Helper Functions | Saturation (logistic/Hill), adstock (steady-state) |
| 14-15 | Original LOB Attribution | Split combo contributions back to individual LOBs |
| 16-17 | Chart D1 | Hyperparameter prior vs posterior (3x4 grid) |
| 18-19 | Chart D7 | Adstock decay curves with 95% CI by funnel |
| 20-24 | Charts D3-D6 | Actual vs predicted, validation detail, residuals, scatter |
| 25-26 | Chart B1 | Weekly contribution decomposition (stacked area) |
| 27-28 | Chart B6 | Annual attendance decomposition by funnel |
| 29-30 | Monthly LOB Charts | Monthly aggregation and LOB-level charts |
| 31-34 | Charts B10a/b | Aggregate response curve & marginal return curve |
| 35-37 | Flighting Analysis | Media flighting analysis + LOB spend zone breakdown |

---

## Notebook 3: EXH_MMM_LOB_Decomposition_Graft_2

**Purpose**: Decomposes group-level validated contributions to individual LOBs using spend-share + tilt signal.  
**Input**: Grouped model artifacts (response_params, spend_train_max, model_meta, df_full)  
**Output**: LOB-level decomposition table with confidence flags

### Cell Flow

| # | Title | Description |
| --- | --- | --- |
| 1 | Overview (markdown) | Method description (graft approach rationale) |
| 2 | Load Artifacts & Configuration | Load response_params, spend_max, model_meta, df_full |
| 3-4 | Validated Group Contributions | Steady-state contribution per combo at 2025 spend |
| 5-6 | Reconciliation Audit | Check if `__Other` LOBs are mapping bugs vs genuinely dropped |
| 7-8 | Residual Attendance | Compute control-confound-removed residual attendance |
| 9-10 | Parse LOB Spend & Tilt Signal | Per-LOB correlation with residual attendance |
| 11-12 | Tilted Decomposition | Split combo contributions by spend share + tilt (±35% cap) |
| 13-14 | Per-LOB Output Table | Final table: contribution, confidence, reallocation eligibility |
| 15-16 | Export Results | Save decomp to pickle and display summary |
| 17-20 | Annual Spend & Contribution Charts | Per-LOB grouped bar charts by year |
| 21-24 | Flighting Zone Breakdown | LOB spend zone analysis |

---

## Notebook 4: EXH_MMM_Budget_Scenarios_FINAL

**Purpose**: Three budget scenarios with posterior credible intervals for stakeholder decisions.  
**Input**: Grouped model trace + LOB decomposition tilt weights  
**Output**: Scenario comparison tables, waterfall charts, 2026 forecasts

### Cell Flow

| # | Title | Description |
| --- | --- | --- |
| 1 | Overview (markdown) | Scenario definitions (A/B/C) |
| 2 | Install Dependencies | arviz, h5netcdf, matplotlib |
| 3 | Load Artifacts | response_params, spend_max, trace, tilt weights |
| 4-5 | Group Response Curve | `combo_contribution()` function + marginal return calc |
| 6-7 | Posterior CI Engine | Propagate posterior through spend allocations → 89% CI |
| 8-9 | Confidence Labels | CI width → High/Medium/Low confidence tag |
| 10-11 | Scenario 1: Budget Uplift | +10/20/30% uniform scaling (sub-proportional lift) |
| 12-13 | Scenario 2: Reshuffle | 10% siphon from low-marginal → high-marginal groups |
| 14-15 | Scenario 3: Combined | Uplift + reshuffle |
| 16-20 | Scenario Comparison Charts | Waterfall, bar charts, scenario table |
| 21-25 | LOB-Level Projections | Disaggregate group scenarios to LOBs via tilt weights |
| 26-29 | 2026 Forecast & Decomposition | Annual attendance decomposition with forecast |
| 30-37 | Stakeholder Deliverable | Final summary tables, export to volume |

---

## Notebook 5: Plots and Graphs

**Purpose**: Final presentation-quality charts for both EXH and LBE models (stakeholder deck).  
**Input**: Artifacts from EXH pipeline + LBE model results  
**Output**: Branded PNG charts saved to respective output directories

### Cell Flow

| # | Title | Description |
| --- | --- | --- |
| 1-2 | Install Packages | openpyxl, xarray, arviz, netcdf4 |
| 3 | Theme & Shared Configuration | CPX brand colours, aspect ratios |
| **EXH Section (Cells 4-22)** | | |
| 4 | EXH Annual Media Spend by Funnel | Stacked bar chart (2023-2025) |
| 5-6 | EXH Spend by Original LOB | Per-year LOB spend + data tables |
| 7-8 | EXH Attendance | Annual attendance breakdown + table |
| 9-10 | EXH Model Features | Correlations & feature categories |
| 11-12 | EXH Attendance Decomposition | Funnel decomposition + 2026 forecast |
| 13-16 | EXH Tentpole & Content Tier | Media contribution by content calendar |
| 17-18 | Correlation Analysis | Media % vs key drivers |
| 19-20 | LOB Spend & Contribution | Per-LOB annual spend/contribution charts |
| 21-22 | EXH Flighting Analysis | Foolproof standalone + per-year breakdown |
| **LBE Section (Cells 23-33)** | | |
| 23 | LBE Annual Spend Overview | Load lbe_mmm_results.pkl, annual spend chart |
| 24-26 | LBE Spend by Channel | Per-year + combined + data tables |
| 27-28 | LBE Revenue Breakdown | Annual revenue + YoY table |
| 29 | LBE Calendar Event Revenue | Revenue lift per calendar event |
| 30 | LBE Per-Channel Spend & Revenue | TRR/PDM spend vs contribution |
| 31 | LBE Diminishing Returns | Response curve + marginal return |
| 32-33 | LBE Flighting Analysis | Standalone + per-year breakdown |

---

## Suggested Notebook Renaming

For clarity, prefix with execution order:

| Current Name | Suggested Name |
| --- | --- |
| EXH MMM Phase 2 Iter 2 Model Training | `1_EXH_Model_Training` |
| EXH MMM Phase 2 Iter 2 Diagnostics | `2_EXH_Diagnostics` |
| EXH_MMM_LOB_Decomposition_Graft_2 | `3_EXH_LOB_Decomposition` |
| EXH_MMM_Budget_Scenarios_FINAL | `4_EXH_Budget_Scenarios` |
| Plots and Graphs | `5_Presentation_Charts` |

---

---

## LBE Pipeline

### Execution Order

```
┌─────────────────────────────────────────────────────────────────────┐
│  L1. LBE Model Training                                             │
│      MMM_LBE_Model_Training                                         │
│      (18 cells · training + inline diagnostics)                      │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
               ┌───────────────┴───────────────┐
               ▼                               ▼
┌──────────────────────────────┐  ┌──────────────────────────────────┐
│  L2. LBE Diagnostics         │  │  L3. LBE Budget Scenarios         │
│  LBE MMM Diagnostics From    │  │  LBE_MMM_Budget_Scenarios_FINAL   │
│  Saved Exports               │  │  (29 cells)                       │
│  (15 cells)                  │  │                                   │
└──────────────┬───────────────┘  └───────────────┬──────────────────┘
               │                                  │
               └───────────────┬──────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  5. Presentation Charts (LBE section, cells 23–33)                  │
│     Plots and Graphs                                                │
└─────────────────────────────────────────────────────────────────────┘
```

### LBE Artifact Lineage

| Artifact | Written By | Read By |
| --- | --- | --- |
| `lbe_model_input.csv` | (external/prep) | L1. Training |
| `lbe_mmm_results.pkl` | L1. Training | L2. Diagnostics, L3. Budget Scenarios, 5. Plots |
| Diagnostic PNGs | L1. Training, L2. Diagnostics | (presentation) |

**Storage path:** `/Volumes/data_exploration_hub/marketing_hub/model_data/Final_Data/MMM_LBE/`

---

### Notebook L1: MMM_LBE_Model_Training

**Purpose**: Trains the LBE Bayesian MMM (log-revenue model with geometric adstock + log1p saturation) and exports all posteriors as a single pickle.  
**Runtime**: ~30 min (NUTS sampling with JAX backend)  
**Input**: `lbe_model_input.csv` (weekly LBE revenue + media spend + controls)  
**Val R-squared**: 0.524 | Val MAPE: 9.2%

#### Cell Flow

| # | Title | Description |
| --- | --- | --- |
| 1 | Configuration | File paths, output dir, hyperparameters |
| 2 | Install Dependencies | pymc, arviz, numpyro, jax |
| 3 | Imports | Core library imports |
| 4 | Brand Palette & Style | CPX colours, matplotlib rcParams |
| 5 | Data Loading & Prep | Load CSV, log-transform revenue, train/val split |
| 6 | Normalisation | Fit scalers on train: y_mean/std, aw_max, co_max, temp |
| 7 | EDA Plot | 2x2 grid: revenue by year, spend over time, distributions |
| 8 | Model Definition & Training | PyMC model, NUTS sampling (4 chains x 2000 draws) |
| 9 | Prediction Helpers | adstock_np, predict_revenue functions, accuracy metrics |
| 10 | Diagnostic Plots | Prior vs posterior, fit quality |
| 11 | Display Plots | Show saved PNGs inline |
| 12 | Residual Diagnostics | Residuals, Q-Q, autocorrelation |
| 13 | Calendar Effects | Posterior calendar coefficient bars |
| 14-16 | Business Charts | Decomposition, spend contribution, response curves |
| 17-18 | Artifact Export | Save `lbe_mmm_results.pkl` with all posteriors |

---

### Notebook L2: LBE MMM Diagnostics From Saved Exports

**Purpose**: Recreates EXH-style diagnostics for LBE using only the saved `lbe_mmm_results.pkl`. No retraining.  
**Input**: `lbe_mmm_results.pkl`  
**Output**: Full diagnostic chart suite saved to `exports_only_diagnostics/`

#### Cell Flow

| # | Title | Description |
| --- | --- | --- |
| 1 | Overview (markdown) | Purpose and scope |
| 2 | Install Dependencies | xarray, arviz, scipy |
| 3 | Imports & Paths | Load libraries, set output directory |
| 4 | Load Saved Exports | Unpack results pickle (draws, bc, scalers, data) |
| 5 | Prediction & Attribution Helpers | adstock_np, predict_revenue, annual decomposition |
| 6 | Posterior & Adstock Diagnostics | Prior vs posterior overlays, decay curves |
| 7 | Validation Diagnostics | D3 actual vs predicted, D4 validation window, D5 residuals |
| 8 | Business Outcome Diagnostics | Weekly decomposition stacked area |
| 9 | B6 Annual Revenue Attribution | Annual stacked bars by funnel contribution |
| 10 | Diminishing Returns & Marginal Curves | Response curve + marginal return per $1 |
| 11 | Annual Bias Check | Year-level prediction accuracy |
| 12 | Baseline Feature Correlation Matrix | Heatmap of control variables |
| 13 | Calendar Effects & Adstock Carryover | Event coefficients + decay half-life |
| 14 | Calendar Event Revenue Contribution | Counterfactual lift per event (posterior draws) |
| 15 | Media Flighting Analysis | Spend zone breakdown (optimal/neutral/peak/trough) |

---

### Notebook L3: LBE_MMM_Budget_Scenarios_FINAL

**Purpose**: Three budget scenarios with posterior CIs for stakeholder decisions. Funnel-level spend optimization.  
**Input**: `lbe_mmm_results.pkl`  
**Output**: Scenario comparison tables, waterfall charts

#### Cell Flow

| # | Title | Description |
| --- | --- | --- |
| 1 | Overview (markdown) | Scenario definitions |
| 2 | Install Dependencies | xarray |
| 3 | Load LBE Model Artifacts | Unpack results, compute baselines |
| 4-5 | Response Curve Functions | `funnel_contribution()`, marginal return calc |
| 6-7 | Posterior CI Engine | 800 draws cached, `annual_revenue_draws()` |
| 8-9 | Confidence Labels & Lift CI | CI width → High/Medium/Low tag |
| 10-11 | Scenario 1: Budget Uplift | +10/20/30% uniform scaling |
| 12-13 | Scenario 2: Reshuffle | Shift 20% from Consideration → Awareness |
| 14-15 | Scenario 3: Combined | Uplift + reshuffle |
| 16-20 | Scenario Comparison Charts | Waterfall, bar charts, summary table |
| 21-25 | Channel-Level Projections | TRR/PDM disaggregation from funnel scenarios |
| 26-29 | 2026 Forecast | Projected revenue under each scenario |

---

### LBE Notebook Locations (current)

| Notebook | Current Path |
| --- | --- |
| MMM_LBE_Model_Training | `/Users/ranjan.mishra@cineplex.com/MMM_LBE_Model_Training` |
| LBE MMM Diagnostics From Saved Exports | `/Users/ranjan.mishra@cineplex.com/LBE MMM Diagnostics From Saved Exports` |
| LBE_MMM_Budget_Scenarios_FINAL | `/Users/ranjan.mishra@cineplex.com/LBE_MMM_Budget_Scenarios_FINAL` |

### Suggested Consolidation

Move LBE notebooks into the `MMM Final Trainings` folder and prefix:

| Current Name | Suggested Name (in MMM Final Trainings/) |
| --- | --- |
| MMM_LBE_Model_Training | `6_LBE_Model_Training` |
| LBE MMM Diagnostics From Saved Exports | `7_LBE_Diagnostics` |
| LBE_MMM_Budget_Scenarios_FINAL | `8_LBE_Budget_Scenarios` |

---

## Key Dependencies & Assumptions

- **Python env**: PyMC 6.x, JAX backend, ArviZ 1.x, scikit-learn
- **Compute**: GPU cluster recommended for Notebook 1 (NUTS sampling); CPU sufficient for 2-5
- **Data source**: `EXH_ModelData.xlsx` (165 weeks, 2023-01 to 2026-02)
- **Validation strategy**: Chronological hold-out (26-week val, 15-week test)
- **Model**: Mixed Hierarchical Bayesian with geometric adstock + logistic/Hill saturation
- **Val R-squared**: 0.484 (grouped model)
- **LOB decomposition**: Graft approach (spend-share + residual-correlation tilt, ±35% cap)
