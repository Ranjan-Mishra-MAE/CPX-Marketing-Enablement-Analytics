---
name: cineclub-coding
description: CineClub coding rules for the dashboard production scripts and any CineClub analysis — membership flows (sign-ups, exits, upgrades, renewals, dunning, lapsed, balance, gifts), RFM tagging, tickets, concessions, visits, demographics, loader pattern, known inconsistencies and validation checks. Use for any CineClub SQL in this repo.
---

# CineClub — AI-Assisted Coding Instructions

Nov 26, 2026 · Ranjan · built from `CPX-CineClub/CineClub Dashboard Scripts/` (Production + Legacy), the CPX Coding Agent project instructions and `CPX_Business_Analytics_Rulebook.md`

---

## 0. How to use this file

- This file **adds** CineClub-specific rules on top of the `cpx-analytics-coding` skill (workflow, validation, optimization rules) and the `cpx-business-rulebook` skill. It does not replace them.
- **Precedence** (highest first):
  1. Rules the owner confirms in chat
  2. Owner-confirmed rulebook rules (type *Owner*)
  3. **This file's "Production standard" rules** (sections 3–4) — they describe what the live dashboard actually computes
  4. Other rulebook rules
  5. Legacy notebooks / ad-hoc scripts
  6. Agent inference (always label it)
- **Production scripts are the reference implementation.** For any CineClub dashboard metric, copy the logic from `Production/` — never from `Legacy/` (hard-coded dates, older date logic) or MOM notebooks (archived Chargebee logic).
- Where production scripts disagree with each other or with the rulebook, they are listed in **section 6**. Do not silently "fix" them. Keep the production behaviour, flag the difference, and ask.
- Delivery defaults for this repo: Databricks SQL (not PySpark) for exploratory work; cell-by-cell code; ask only for the minimum data needed to check results; keep validation to the crucial checks.

---

## 1. Dashboard map

All output tables live in `data_exploration_hub.marketing_hub`. All sources are in `cpx_dataanalytics_gold` unless noted.

| Output table | Production script | Grain | Sources | Load readiness gate |
| --- | --- | --- | --- | --- |
| `cc_daily_signups` | Daily Signups Backfill (Daily Production) | snapshot_date × rfm_category (11 rows/day) | subscription events, snapshot, gifting, RFM | day ≤ min(max snapshot, max RFM, yesterday) |
| `cc_monthly_signups` | Monthly Signups Backfill (Monthly Production) | month_start × rfm_category (11 rows/month) | same | last_day(month) ≤ min(max snapshot, max RFM, yesterday); supports rebuild mode |
| `mkt_dashboard_demographics_monthly` | Demographics Back Fill (Monthly Production) | report_month × metric_type (3 rows/month) | events, snapshot, cpx_d_customer, transactions, RFM | same as monthly signups |
| `mkt_dashboard_visitation_behavior` | Visitation Behaviour Backfill (Monthly Production) | period_end_date × period_type × venue_type × rfm_segment | transactions, location, date, item, RFM | RFM snapshot for **period_end + 1** must exist; period_end ≤ yesterday |
| `mkt_dashboard_tickets_monthly` | Cineclub Ticket Table (Monthly Production) | report_month × population_type (3 rows/month) | transactions, location, date, session, ticket type, events, snapshot | last_day(month) ≤ min(max snapshot, yesterday). Starts 2025-01-01 if empty |
| `mkt_dashboard_concessions_monthly` | Cineclub - Concession Behaviour Table (Monthly Production) | report_month × population_type (3 rows/month) | transactions, location, item | last_day(month) ≤ yesterday |
| `mkt_dashboard_top_films_monthly` | Top Films (Monthly Production) | report_month × rank (5 rows/month) | transactions, location, film | last_day(month) ≤ yesterday |

Legacy (`Legacy/`) holds the hard-coded, single-period "Verified" versions these were built from, sample-data notebooks, and the Guest Services tables (`GS_CC_Members`, `GS_CC_Transactions` — see rulebook BR-051, DQ-15).

---

## 2. Database rules

### 2.1 Source tables and keys

| Table | Key / grain | CineClub fields used | Notes |
| --- | --- | --- | --- |
| `marketing.mkt_subscription_events` | one row per event | cc_user_profile_id, event_type, plan_id, reporting_datetime, record_occurred_at, changed, changed_from, changed_to, current_term_start_datetime, current_term_end_datetime, subtotal | **Official source for all membership flows** (BR-027). Never use the legacy name `marketing.subscription_events` |
| `marketing.mkt_subscriber_profile_snapshot` | cc_user_profile_id × snapshot_date_id | cde_id, plan_id, renewal_status, recognitions_left, recognitions_left_10_99, recognitions_redeemed | Daily member status. Never use legacy `marketing.subscriber_profile_snapshot` |
| `cpx_dataanalytics_silver.cineclub.gifting` | one row per gift code | purchaser_id, redeemer_id, is_redeemed, purchase_date, redemption_date | **Silver** catalog. Timestamps are UTC |
| `customer360.c360_f_rfm_model` | CDE_ID × SnapshotDateId | Segment | Standard RFM (BR-034) |
| `customer360.c360_f_transaction` | one row per line item | CDE_ID, VisitDateId, LocationId, ItemSkey, TicketTypeSkey, sessionskey, film_id, Transaction_Number, Quantity, Net_Revenue, CineClubFLAG, SCENEFLAG, TicketFLAG, ConcessionFLAG, CineClub_Discount_Type_ID | Refunds are negative Quantity |
| `customer360.cpx_d_customer` | one row per cde_id (dedupe in code) | cc_user_profile_id, cde_id, gender_text, age_cohort_id, province | |
| `customer360.cpx_d_location / cpx_d_date / cpx_d_item / cpx_d_session / cpx_d_ticket_type / cpx_d_film` | see rulebook §2 | | |

### 2.2 Identity bridging

- **Subscription world** is keyed by `cc_user_profile_id`. **Transaction world** is keyed by `CDE_ID`.
- The only bridge used in production is the **snapshot**: `snapshot.cc_user_profile_id → snapshot.cde_id`.
- `snapshot.cde_id` is stored as a string; transaction/RFM `CDE_ID` is numeric. Production casts explicitly:
  - RFM join: `CAST(snapshot.cde_id AS INT) = rfm.CDE_ID`
  - Customer join from transactions: `CAST(c.cde_id AS STRING) = p.cde_id`
  Keep these casts. Do not rely on implicit casting in new code.
- Gift ids (`purchaser_id`, `redeemer_id`) are `cc_user_profile_id` values.
- **CDE lookup with fallback** (used for RFM tagging of flows): take the cde_id from the period-end snapshot; for members missing there (e.g. lapsed), fall back to the prior period-end snapshot.

```sql
cde_lookup AS (
    SELECT cc_user_profile_id, cde_id FROM cde_lookup_eom
    UNION ALL
    SELECT p.cc_user_profile_id, p.cde_id
    FROM cde_lookup_prior_eom p
    LEFT ANTI JOIN cde_lookup_eom t ON t.cc_user_profile_id = p.cc_user_profile_id
)
```

### 2.3 Date and type handling

| Field | How production treats it |
| --- | --- |
| `reporting_datetime` (events) | Event date. Compared to `'yyyy-MM-dd'` strings or DATEs (`= v_day_str`, `BETWEEN month_start AND month_end`) |
| `record_occurred_at` (events) | Unix epoch seconds. Use `CAST(from_unixtime(record_occurred_at) AS DATE)`. **Only used for upgrades initiated** |
| `current_term_start/end_datetime` | Timestamps; cast to DATE for term-window tests |
| `snapshot_date_id` | yyyyMMdd. Production compares it both as string (`= date_format(d,'yyyyMMdd')`) and as INT (`BETWEEN CAST(... AS INT)`). Keep whichever the script you are editing uses; in new code pick one and say so |
| `SnapshotDateId` (RFM) | yyyyMMdd, compared to `date_format(...)` |
| `VisitDateId` | yyyyMMdd int. Either join `cpx_d_date` (`d.datefull BETWEEN ...`) or filter `VisitDateId BETWEEN CAST(date_format(start,'yyyyMMdd') AS INT) AND ...` |
| Gift `purchase_date`, `redemption_date` | UTC → `CAST(from_utc_timestamp(CAST(x AS TIMESTAMP), 'America/New_York') AS DATE)` before any date filter (BR-032) |

- **All dashboard behaviour tables date on `VisitDateId` (visit month)**, including tickets. The legacy ticket table used `TransactionDateId`; production deliberately moved to `VisitDateId`. Do not switch back without the owner.
- Snapshot history **starts 2025-06-18**, and **20251031 is missing**. Any month-end snapshot logic must handle a missing snapshot (see CC-30).

### 2.4 Plans and statuses

- Plans: `monthly1`, `annual1`, `gift-plan` (BR-021). Display mapping: Monthly / Annual / Gift Plan.
- Active statuses: `Active`, `Active_Non_Renewing`, `Active_Annual_Upgrade` (correct spelling — never `Active_Annnual_Upgrade`, DQ-11).
- Rulebook standard for snapshot-active is `renewal_status LIKE 'Active%'` (BR-022). Production dashboard scripts use the explicit `IN ('Active','Active_Non_Renewing','Active_Annual_Upgrade')`. Keep the form the script already uses.

---

## 3. Business rules — membership flows (Production standard)

These are exactly what `cc_daily_signups` / `cc_monthly_signups` compute. Daily = single `reporting_datetime` day; monthly = `BETWEEN month_start AND month_end`. All counts are `COUNT(DISTINCT cc_user_profile_id)`.

| ID | Metric | Production logic |
| --- | --- | --- |
| CC-01 | First-timer (member attribute) | Member with exactly **one** `subscription_created` event ever (all plans, all time; `cc_user_profile_id IS NOT NULL`) |
| CC-02 | Monthly sign-ups | `subscription_created`, `plan_id='monthly1'` in period |
| CC-03 | Annual sign-ups | `subscription_created`, `plan_id='annual1'` in period, **minus** members in `upgrades_took_effect` for the same period (DQ-10) |
| CC-04 | First-time / returning split | First-time = sign-up ∩ first-timers. Returning = sign-up `LEFT ANTI JOIN` first-timers |
| CC-05 | Returning-from breakdown | `prev_plan_id = LAG(plan_id) OVER (PARTITION BY cc_user_profile_id ORDER BY reporting_datetime)` across all `subscription_created` events; bucket on prev_plan_id = monthly1 / annual1 / gift-plan |
| CC-06 | Exits | `subscription_cancelled`, by plan, `changed IS NOT TRUE`, `cc_user_profile_id IS NOT NULL` |
| CC-07 | Upgrades initiated | `subscription_changed`, `changed = TRUE`, `changed_from != changed_to`, `changed_to='annual1'`, dated by `from_unixtime(record_occurred_at)` (BR-056) |
| CC-08 | Upgrades took effect | `subscription_cancelled`, `plan_id='monthly1'`, `changed = TRUE`, dated by `reporting_datetime` |
| CC-09 | Annual renewed | `payment_succeeded` on `annual1` in period; member's annual start = MIN(first annual `subscription_created`, first upgrade-took-effect); `DATEDIFF(payment_date, start) >= 360` and `MOD(DATEDIFF, 365) <= 5 OR >= 360`; excludes the period's annual sign-ups, annual exits and upgrades took effect (BR-028). Monthly version anchors each payment on its own date |
| CC-10 | Monthly dunning | Latest monthly1 payment event (as of period end) is `payment_failed`, dated within **15** days of period end, failure date = term start date, no later successful term start |
| CC-11 | Annual dunning | Same as CC-10 on `annual1` with **14** days (open conflict CF-15 — keep 15/14 until the owner's revised script arrives) |
| CC-12 | Annual lapsed | In prior snapshot with `plan_id='annual1'` (**no status filter**), not in current snapshot `annual1` nor annual dunning, and no non-upgrade annual `subscription_cancelled` to date |
| CC-13 | Gift lapsed | In prior snapshot `gift-plan` with an active status, not active gift-plan in current snapshot, no non-upgrade gift `subscription_cancelled` to date |
| CC-14 | Active balance by plan | Monthly: `payment_succeeded` monthly1 whose term covers period end. Annual / gift: `subscription_created/renewed/resumed/started` whose term covers period end (BR-030) |
| CC-15 | Total active balance | `UNION` (distinct) of monthly active + annual active + gift active + monthly dunning + annual dunning. **Never sum the bucket counts** — in-flight upgrades sit in two buckets |
| CC-16 | Net increase | `signups − exits` per plan. Does not include upgrades, lapsed or dunning |
| CC-17 | Gift purchasers / redeemers | Distinct purchaser_id (purchase date) / redeemer_id with `is_redeemed = TRUE` (redemption date), ET-converted |
| CC-18 | Gift net new vs returning | Net new = not in the **prior** period-end snapshot (any plan, any status). Returning-from = that snapshot's `plan_id` |

"Prior snapshot" = previous day (daily) or prior month-end snapshot (monthly). "Current snapshot" = the day itself / month-end snapshot.

### 3.1 RFM tagging of flows

| ID | Rule |
| --- | --- |
| CC-20 | RFM segment = `c360_f_rfm_model.Segment` at `SnapshotDateId` = the day (daily) or `month_end` (monthly, demographics) |
| CC-21 | Member → cde_id via the snapshot lookup with prior-snapshot fallback (§2.2). `COALESCE(Segment,'Dormant')` |
| CC-22 | Fixed category list, exact strings: `All, Top 3 RFM, Champions, Potential Loyalist, Value Conscious Loyals, Promising, At Risk, New, Fleeting, High Value Lapsed, Dormant` |
| CC-23 | Per metric: count `All` and each named non-Dormant segment once; `Top 3 RFM` = SUM(Champions, Potential Loyalist, Value Conscious Loyals); `Dormant` = All − SUM(named segments). Members with no lookup row at all (e.g. gift purchasers who were never members) therefore land in Dormant |
| CC-24 | Output is zero-filled: one LEFT JOIN per metric onto `categories`, `COALESCE(cnt, 0)` |
| CC-25 | Row order: All, Top 3 RFM, Champions, Potential Loyalist, Value Conscious Loyals, Promising, At Risk, New, Fleeting, High Value Lapsed, Dormant |

---

## 4. Business rules — behaviour tables (Production standard)

### 4.1 Populations

| ID | Population | Rule | Used in |
| --- | --- | --- | --- |
| CC-40 | CC | Line with `CineClubFLAG='Yes'` (and `CDE_ID IS NOT NULL` where member counts matter) | tickets, concessions, films, visitation |
| CC-41 | Scene (non-CC) | `SCENEFLAG='Yes' AND CineClubFLAG IS DISTINCT FROM 'Yes'` | tickets, concessions, films |
| CC-42 | Scene (visitation table) | `SCENEFLAG='Yes'` — **includes CC members** (see CD-05) | visitation only |
| CC-43 | Total_Biz | All lines in scope, **no CDE_ID filter** (anonymous buyers included) | tickets, concessions, films |
| CC-44 | Theatre scope | `LC_Location_Type_Description = 'Cineplex Theatre'` | all EXH logic |
| CC-45 | LBE scope | `LC_Location_Type_Description LIKE 'Restaurant%' AND LC_Trade_Brand_Description IN ('Rec Room','Playdium')` (BR-012) | concessions, visitation |
| CC-46 | Active CC base (demographics) | `ptd_balance` = CC-15 union at month end, joined to `cpx_d_customer` on cc_user_profile_id. Its cde_ids (via the month-end snapshot) are what CC-47 excludes | demographics |
| CC-47 | Scene balance (demographics) | Distinct CDE_ID with any `SCENEFLAG='Yes'` line, `VisitDateId` from `add_months(month_end,-12)` to month_end, any venue, minus active CC cde_ids | demographics |

### 4.2 Visits

| ID | Rule | Where |
| --- | --- | --- |
| CC-50 | **Ticket table visits** (MOM-matched): any theatre line (tickets + concessions), netted per `cde_id × date × location`, kept if `SUM(Quantity) > 0 AND cde_id > 0`; visits = `COUNT(DISTINCT cde_id, datefull)`; members = `COUNT(DISTINCT cde_id)`. CC and Scene come from this grain; Total_Biz comes from ticket lines | tickets |
| CC-51 | **Visitation table visits**: grain `cde_id × date` (no location); `is_visit = SUM(Quantity) > 0`; total visits = SUM(is_visit); distinct visitors = members with is_visit = 1 | visitation (EXH) |
| CC-52 | Per-member averages (visitation, EXH) use only members with ≥ 1 visit in the period (`member_has_visit`) in both numerator and denominator | visitation |
| CC-53 | Concession incidence = visit-days with `concession_qty > 0` ÷ visit-days × 100 | visitation |
| CC-54 | LBE visits = count of `cde_id × date` rows with any line (not netted) | visitation (LBE) |
| CC-55 | Visitation periods: monthly (month), quarterly (3 months ending period end), ytd, p12m (12 months ending period end), ptd (from **2021-08-01**). RFM at `SnapshotDateId = period_end + 1` | visitation |

### 4.3 Tickets

| ID | Rule |
| --- | --- |
| CC-60 | Tickets = `SUM(Quantity)` on `TicketFLAG='Yes'` lines (refunds net out naturally) |
| CC-61 | Format waterfall on session: Auditorium 2 AVX, 3 IMAX, 4 VIP, 5/6 ALT, 7 FourDX, 8 ScreenX; else Performance_Experience_ID 2 ThreeD; else Regular (BR-038/T-05) |
| CC-62 | DBox is counted **independently** (`Is_DBOXflag='Yes'`), outside the waterfall — it overlaps other formats |
| CC-63 | `tickets_upgrades` = all non-Regular tickets (ALT and 3D included) — see CD-07 |
| CC-64 | Paid = `paid_attendance_inflag='Yes'`; free = `IS DISTINCT FROM 'Yes'` (NULL = free) |
| CC-65 | Discount types: 6/8 = redeemed member ticket ($9.99 / $10.99); 5/7 = companion ($9.99 / $10.99); 1 = member, no discount (BR-018). **Any CC ticket logic must include 7 and 8** (BR-020, DQ-18) |
| CC-66 | $9.99/$10.99 shares are CC-only; set NULL (not 0) for Scene/Total_Biz. Use `NULLIF(denominator, 0)` |
| CC-67 | Tickets credited = distinct (cc_user_profile_id, current_term_start_datetime, subtotal) from `payment_succeeded` or `subscription_renewed` with term start in the month, all plans. Tier by `subtotal`: $9.99 = 9.99, 109.89, 119.88; $10.99 = 10.99, 120.89. Gift-plan / null subtotal count in the total only |
| CC-68 | Tickets in market = SUM(`recognitions_left`) [$9.99] + SUM(COALESCE(`recognitions_left_10_99`,0)) [$10.99] on the exact month-end snapshot |

### 4.4 Concessions and LBE spend

| ID | Rule |
| --- | --- |
| CC-70 | EXH concession lines = theatre, `ConcessionFLAG='Yes'`, category not NULL |
| CC-71 | Concession category (`cpx_d_item`), first match wins: **popcorn** (`%POPCORN%`, `%CORN%UPSIZE%`, `%UPSIZE%CORN%`, not `%CORN DOG%`) → **fountain_drinks** (`%DRINK%CUP%`, `%VIP%DRINK%`, `%UPSIZE%FOUNTAIN%`) → **candy** (class `VHO-CANDY`, `CANDY`, `SMALL CANDY`, excluding `DELIVERY%`, `VHO DELIVERY%`, `SNACK%`) → **hot_foods** (`%NACHO%`, `%HOT DOG%`) → **merch** (6 merch classes) → NULL |
| CC-72 | Per category: revenue = SUM(Net_Revenue); transactions = COUNT(DISTINCT Transaction_Number); quantity = SUM(Quantity); members = COUNT(DISTINCT CDE_ID) |
| CC-73 | LBE gaming / F&B = `Item_Class_Description1` lists in the production scripts. Copy the list from the script you are editing — the two scripts differ on G&E (CD-04) |
| CC-74 | CC concession discount $ = `Net_Spend / 4` for discount types 2, 3 (BR-019) — not currently in the dashboard; use this if asked |

### 4.5 Demographics

| ID | Rule |
| --- | --- |
| CC-80 | Three metric_types: `program_monthly_signups` (all `subscription_created`, any plan, in month), `ptd_balance` (CC-46), `scene_balance` (CC-47) |
| CC-81 | Gender: Male / Female / other (other = anything else or NULL) |
| CC-82 | Age from `cpx_d_customer.age_cohort_id`: 1 = 14–18, 2 = 19–24, 3 = 25–34, 4 = 35–49, 5 = 50–64, 6 = 65+, anything else = unknown |
| CC-83 | Province `UPPER(TRIM())`: ON, BC, AB, QC, MB, SK, ATL (NB, NS, PE, PEI, NL), other |
| CC-84 | RFM: CC populations via CC-21; Scene via direct `CDE_ID` join; Dormant = no match |

### 4.6 Top films

| ID | Rule |
| --- | --- |
| CC-90 | Grouped by `cpx_d_film.title_name` (rolls formats/languages together) |
| CC-91 | Ranked by **CC tickets** with ROW_NUMBER; top 5; Scene and Total_Biz shown for the same titles |

---

## 5. Production loader pattern

Every production script follows the same SQL-scripting pattern. New dashboard tables must too.

```sql
BEGIN
  DECLARE v_month_start DATE;
  DECLARE v_end         DATE;

  -- 1. Resume after the last loaded period
  SET v_month_start = (SELECT add_months(MAX(report_month), 1) FROM <target>);

  -- 2. Readiness gate: every source the query needs must be loaded
  SET v_end = (SELECT LEAST(
      (SELECT to_date(CAST(MAX(snapshot_date_id) AS STRING), 'yyyyMMdd') FROM <snapshot>),
      (SELECT to_date(CAST(MAX(SnapshotDateId)  AS STRING), 'yyyyMMdd') FROM <rfm>),
      date_sub(current_date(), 1)));

  -- 3. Loop closed periods only; delete-then-insert makes each period idempotent
  WHILE last_day(v_month_start) <= v_end DO
    DELETE FROM <target> WHERE report_month = v_month_start;
    INSERT INTO <target> (<explicit column list in SELECT order>)
    WITH ... SELECT ...;
    SET v_month_start = add_months(v_month_start, 1);
  END WHILE;
END;
```

Rules:

- **Always** DELETE the period before INSERT (re-runs must not duplicate).
- **Always** name the INSERT column list in the SELECT's order (the ticket table's SELECT order differs from its table order).
- The gate must include every source with lag: snapshot and RFM for anything RFM- or status-based; transactions-only tables can gate on yesterday.
- Parameterise only the period; never hard-code dates inside the CTEs (that is what `Legacy/` did).
- Rebuild mode (monthly signups): `v_rebuild_from` / `v_rebuild_to` as month starts; leave NULL for append.
- Missing snapshots: follow the monthly signups fallback — latest snapshot within the last 3 days of the month (EOM) and within the 4 days before month start (prior EOM); return NULL (not 0) for metrics that depend on a missing snapshot.

---

## 6. Known inconsistencies (flag, don't fix silently)

These exist in the current production scripts. When you touch one of these scripts, or reuse its logic elsewhere, mention the relevant item and ask before changing it.

| ID | Where | Issue | Impact |
| --- | --- | --- | --- |
| CD-01 | All theatre bases (tickets, concessions, films, visitation) | `RecRoomFLAG='No' AND STOREFLAG='No'` not applied, though BR-011 (Owner) requires it | Possible non-theatre lines in theatre metrics |
| CD-02 | Visit key | Ticket table nets per cde × date × **location** then counts cde × date; visitation table nets per cde × date. Rulebook KPI-08 = distinct cde × date × location | Visit counts differ across tables and from KPI-08 |
| CD-03 | `cde_id > 0` | Only in the ticket table's member grain; not in visitation, concessions, demographics | Small CC/Scene member-count differences |
| CD-04 | LBE gaming | Concessions table includes `TRR - AMUSEMENT - G&E` in gaming; visitation excludes it (matches BR-042) | Gaming revenue not comparable across the two tables |
| CD-05 | Scene population | Visitation uses SCENEFLAG = Yes (CC included); other tables use non-CC Scene | Scene figures not comparable across tables |
| CD-06 | RFM date | Signups/demographics use the period-end SnapshotDateId; visitation uses period end **+ 1 day** | Segment mixes differ slightly |
| CD-07 | `tickets_upgrades` | Counts every non-Regular ticket incl. ALT and 3D; KPI-19 premium = auditorium 2, 3, 4, 7, 8 only, ALT separate | Not a "premium" measure — don't relabel it as one |
| CD-08 | Top-3 label | Signups: `Top 3 RFM`; visitation: `Top 3` | Join/filter mismatch in Power BI or downstream SQL |
| CD-09 | Demographics sign-ups | `program_monthly_signups` doesn't exclude upgrade completions logged as annual `subscription_created` (CC-03 does) | Demographic sign-up base > dashboard sign-ups |
| CD-10 | Demographics load | No DELETE before INSERT | Re-running a month duplicates rows |
| CD-11 | Demographics join | `cpx_d_customer` joined on cc_user_profile_id / cde_id without dedupe (DQ-05) | Possible fan-out |
| CD-12 | Demographics age | Uses `age_cohort_id`; MOM/rulebook computes age from `date_of_birth_masked` (BR-041) | Bin counts can differ from MOM |
| CD-13 | Scene balance window | `add_months(month_end,-12)` to month_end inclusive = 12 months + 1 day, any venue (incl. LBE) | Slightly wider than P12M theatre |
| CD-14 | Ticket type join | `cpx_d_ticket_type` not deduped on tickettypeskey (DQ-05) | Possible ticket inflation if the dimension has duplicates |
| CD-15 | Tickets in market | No test-account ceiling on recognitions_left (BR-025); exact month-end snapshot only, no fallback (Oct 2025 snapshot missing) | Test accounts included; Oct 2025 in-market = NULL |
| CD-16 | Concession transactions | `COUNT(DISTINCT Transaction_Number)` without location/date — numbers may repeat across locations | Possible under-count of transactions |
| CD-17 | Concessions script | `active_cc_members` CTE is defined but never used (dead code) | None — mention, don't delete |
| CD-18 | First-timers | Defined over **all-time** events, not as of the period | Backfills can reclassify past "first-time" sign-ups as "returning" |
| CD-19 | Active status form | Production uses `IN (3 statuses)`; BR-022 uses `LIKE 'Active%'` | Same today; diverges if a new Active_* status appears |
| CD-20 | Annual lapsed | Prior-snapshot annual base has no status filter; gift lapsed does | Asymmetric definitions |
| CD-21 | Dunning | 15 days monthly vs 14 days annual (CF-15, open) | Keep until owner provides revised script |
| CD-22 | snapshot_date_id type | Compared as string in some scripts, INT in others | Works via implicit cast; keep consistent within a script |

---

## 7. CineClub validation checklist

Use the general validation rules first. Then run the checks below that apply. Expected values are identities, not numbers — never invent expected numbers.

**Signups (daily or monthly output)**

```sql
-- 1. 11 rows per period, no duplicates
SELECT month_start, COUNT(*) AS rows, COUNT(DISTINCT rfm_category) AS cats
FROM data_exploration_hub.marketing_hub.cc_monthly_signups
GROUP BY month_start HAVING COUNT(*) <> 11 OR COUNT(DISTINCT rfm_category) <> 11;

-- 2. All = sum of named segments + Dormant; Top 3 = sum of its three
SELECT month_start,
       MAX(CASE WHEN rfm_category='All' THEN annual_signups END) AS all_cnt,
       SUM(CASE WHEN rfm_category NOT IN ('All','Top 3 RFM') THEN annual_signups END) AS seg_sum,
       MAX(CASE WHEN rfm_category='Top 3 RFM' THEN annual_signups END) AS top3,
       SUM(CASE WHEN rfm_category IN ('Champions','Potential Loyalist','Value Conscious Loyals') THEN annual_signups END) AS top3_sum
FROM data_exploration_hub.marketing_hub.cc_monthly_signups
GROUP BY month_start
HAVING all_cnt <> seg_sum OR top3 <> top3_sum;

-- 3. Total balance never exceeds the sum of its buckets; Dormant never negative
SELECT month_start, rfm_category
FROM data_exploration_hub.marketing_hub.cc_monthly_signups
WHERE total_active_balance > monthly_active_balance + annual_active_balance + gift_active_balance
                             + monthly_members_dunning + annual_dunning
   OR monthly_signups < 0 OR annual_signups < 0 OR total_active_balance < 0;

-- 4. Split identities
SELECT month_start FROM data_exploration_hub.marketing_hub.cc_monthly_signups
WHERE rfm_category='All'
  AND (monthly_signups <> monthly_first_time_signups + monthly_returning_signups
    OR annual_signups  <> annual_first_time_signups  + annual_returning_signups);
```

- Returning-from buckets can sum to **less** than returning sign-ups (prev_plan_id NULL or another plan) — check the gap, don't force it to zero.
- Daily flow counts summed over a month can exceed the monthly table (a member can sign up twice). Compare daily vs monthly only for event metrics, and expect daily sum ≥ monthly.
- Balance at month end in `cc_monthly_signups` should equal `cc_daily_signups` on the last day of the month (same logic, same date) — good reconciliation.

**Tickets**

```sql
SELECT report_month, population_type
FROM data_exploration_hub.marketing_hub.mkt_dashboard_tickets_monthly
WHERE tickets_regular + tickets_upgrades <> tickets_free + tickets_paid            -- both = total tickets
   OR tickets_upgrades <> tickets_avx + tickets_imax + tickets_vip + tickets_alt
                          + tickets_fourdx + tickets_screenx + tickets_threed      -- DBox excluded by design
   OR (population_type='CC' AND ABS(COALESCE(share_9_99_Free,0) + COALESCE(share_10_99_Free,0) - 1) > 0.0001
       AND tickets_free_redeemed > 0);
```

**Other tables**

- Concessions / tickets / films: 3 population rows (5 rank rows for films) per month; CC ≤ Total_Biz and Scene ≤ Total_Biz for every additive column.
- Visitation: for each period_end, ptd ≥ p12m ≥ quarterly ≥ monthly on distinct visitors and total visits (ytd sits between monthly and p12m).
- Any new member-level join: compare `COUNT(*)` vs `COUNT(DISTINCT key)` before and after.
- Any new CC ticket logic covering dates ≥ 2026-06-15: confirm discount types 7 and 8 return rows.

---

## 8. Quick do / don't

**Do**
- Start from the production script for the metric; change only what was asked.
- Use `mkt_subscription_events` for flows and the snapshot for status, recognitions and cde_id.
- Convert gift timestamps to America/New_York.
- Exclude `upgrades_took_effect` from annual sign-ups; use `changed IS NOT TRUE` for exits.
- Union member lists before counting totals (CC-15).
- Return NULL, not 0, for CC-only fields on non-CC rows and for metrics with a missing snapshot.
- Match RFM labels against the table's own `Segment` values (BR-034).

**Don't**
- Use the Chargebee export, MOM sign-up logic or `Legacy/` scripts as the source of truth.
- Use `marketing.subscription_events` / `marketing.subscriber_profile_snapshot` (pre-rename).
- Filter member tickets with `IN (5,6)` only.
- Sum per-plan balances into a total.
- Count `subscription_cancelled` with `changed = TRUE` as an exit.
- Hard-code report dates inside CTEs, or drop the DELETE step from a loader.
- Change the visit key, Scene definition or G&E handling in one table to "match" another without the owner's say-so (section 6).

---

## 9. Open questions for the owner

1. Should theatre bases add `RecRoomFLAG='No' AND STOREFLAG='No'` (CD-01)?
2. One visit key for all dashboard tables (CD-02), or keep each as is?
3. G&E in LBE gaming: exclude in the concessions table to match BR-042 (CD-04)?
4. Scene in the visitation table: switch to non-CC Scene (CD-05)?
5. Tickets credited (CC-67): annual members — should each payment count once, or should monthly recognition credits be counted?
6. Revised dunning / sign-up scripts (CF-15).
