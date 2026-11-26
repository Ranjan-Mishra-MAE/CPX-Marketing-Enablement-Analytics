# CPX Marketing Enablement — Business & Analytics Rulebook

Sep 27, 2026 · updated Oct 21, 2026 · Ranjan

## 1. Executive Summary

The repo measures three things: CineClub subscription health, guest visitation and spend across Exhibition (EXH) and LBE (The Rec Room, Playdium), and marketing effectiveness (offers, email tests, MMM). Almost everything is built on one fact table, `c360_f_transaction`, joined to date, location, item, film, session and ticket-type dimensions, plus two CineClub tables: `mkt_subscription_events` and `mkt_subscriber_profile_snapshot`.

**Core KPIs.** Sign-ups, first-time vs returning members, exits, upgrades, ending active balance, visits, tickets, ticket and concession spend, member savings, redemption rate, ticket liability, and LBE Food vs Games revenue.

**Common methods.** Descriptive cohorts; pre/post and YoY comparisons; detrended counterfactuals and difference-in-differences with a control category; RFM segmentation; Bayesian MMM (PyMC/NumPyro) with adstock and saturation; K-means film clustering; break-even offer modelling.

**The most important recurring rules:**

- A **visit** is one CDE_ID × date × location with net quantity above zero (refunds netted out).
- **Theatre scope** is `LC_Location_Type_Description = 'Cineplex Theatre'` with RecRoomFLAG and STOREFLAG = No. **LBE** is Rec Room + Playdium combined.
- **Membership flows** (sign-ups, exits, upgrades, balance) come from `mkt_subscription_events` using the YTD SQL logic; the Chargebee-export logic is archived.
- **Plans** are `monthly1`, `annual1` and `gift-plan`. Active status is read as `renewal_status LIKE 'Active%'`.
- **Discount types** in `CineClub_Discount_Type_ID`: 2 and 3 = 20% off concessions and games; 5 and 6 = $9.99 companion and redeemed tickets; 7 and 8 = $10.99 companion and redeemed tickets (from the 2026-06-15 price change).
- **Dates:** `Transaction_DateTime` for financial analysis, `VisitDateId` for behavioural analysis.

**Biggest risks found:**

- **Older notebooks use non-standard logic** now superseded by owner decisions: Chargebee sign-ups, alternative upgrade and visit definitions, ALT counted as regular, G&E classes in Food/Games, legacy (non-`mkt_`) subscription table names.
- **Post-price-change gaps:** code that filters member tickets with discount types 5 and 6 misses types 7 and 8 from 2026-06-15.
- **The renamed status** `Active_Annnual_Upgrade` → `Active_Annual_Upgrade` in about 10 older notebooks.

**Coverage.** All non-MOM code files (71 of the repo's 91) have been read in full (last pass 2026-09-27); the monthly MOM report notebooks were not re-reviewed beyond the Aug 2026 version. MMM parameters are intentionally excluded because the repo version is not final. Rules the owner confirmed on 2026-09-27 are typed **Owner**; the rest are **Implemented** unless a post-mortem document states them (**Doc**).

## 2. Data Model

Everything hangs off `customer360.c360_f_transaction`, joined left to dimensions by surrogate keys. CineClub status comes from the two `marketing` subscription tables, which link to transactions through `cde_id` and to each other through `cc_user_profile_id`. Where a grain is not stated in code, it is marked *inferred*.

**Tables** (the `cpx_dataanalytics_gold` catalog unless noted)

| Table | Alias in code | Purpose | Grain | Key fields used |
| --- | --- | --- | --- | --- |
| customer360.c360_f_transaction | T_trans | Box office, concession, LBE food/games line items | One row per line item (confirmed by owner); refunds appear as negative Quantity | CDE_ID, VisitDateId (yyyyMMdd int), Transaction_DateTime, Transaction_Number, LocationId, ItemSkey, TicketTypeSkey, sessionskey, film_id, SalesChannelSkey, Quantity, Net_Revenue, Net_Spend, SCENE_Points_Earned/Redeemed, CineClub_Discount_Type_ID, TicketFLAG, ConcessionFLAG, CineClubFLAG, SCENEFLAG, RecRoomFLAG, STOREFLAG, COTFLAG, CDE_Visit_ID_MNL |
| customer360.cpx_d_date | T_date | Calendar | One row per date | dateid, datefull |
| customer360.cpx_d_location | T_locations | Venue attributes | One row per location | LocationID, LC_Location_Type_Description, LC_Trade_Brand_Description, Location_Name, Province_CD, Opening_Date |
| customer360.cpx_d_item | T_items | Product catalogue | One row per itemskey | itemskey, Item_ID, Item_Description, Item_Class_Description1, EIS_Sub_Group_Description |
| customer360.cpx_d_film | T_films | Film master | One row per film_id | film_id, title_id, title_name, theatrical_release_date, language |
| customer360.film_composite_genres | T_genre | Genre hierarchy | One row per film_id (*inferred*) | film_id, genre_hierarchy_1 |
| customer360.cpx_d_session | T_sessions | Showtime / auditorium | One row per sessionskey | Auditorium_Experience_ID, Performance_Experience_ID, Performance_Type_ID, Is_DBOXflag, format flags |
| customer360.cpx_d_performance_type | T_perf | Performance sub-category (e.g. Film Presentation vs event types); gives the ALT breakdown | One row per performance_type_id | performance_type_id, performance_type; joined via cpx_d_session.Performance_Type_ID |
| customer360.cpx_d_ticket_type | T_ticket | Ticket type attributes | One row per tickettypeskey | ticket_type_id, ticket_type_description_english, report_code1, paid_attendance_inflag, seniorflag, childflag |
| customer360.cpx_d_sales_channel | T_sales | Sales channel | One row per saleschannelskey | sales_channel |
| customer360.cpx_d_customer | T_demo | Customer demographics | One row per cde_id (deduped in code) | cde_id, cc_user_profile_id, gender_text, date_of_birth_masked (year), province |
| customer360.c360_f_rfm_model | — | Standard RFM segment | CDE_ID × SnapshotDateId | CDE_ID, Segment, SnapshotDateId |
| marketing.mkt_subscription_events | T_events | Chargebee subscription events; the official source for membership flows. Current name (renamed with an `mkt_` prefix); `marketing.subscription_events` is the pre-rename legacy name still used in CineClub Rewind (SQL) and Black Friday Deal 2025 | One row per event | cc_user_profile_id, event_type, plan_id, reporting_datetime, record_occurred_at, changed, changed_from, changed_to, current_term_start/end_datetime |
| marketing.mkt_subscriber_profile_snapshot | T_subscriber | Daily CineClub member status. Current name; `marketing.subscriber_profile_snapshot` is the pre-rename legacy name | cc_user_profile_id × snapshot_date_id (yyyyMMdd string) | cde_id, plan_id, renewal_status, recognitions_left/redeemed, Member_Priced_Recognitions_Redeemed, renewal_datetime, annual_current_term_end_datetime, activated_at_datetime |
| marketing.cpx_d_customer_email_address | T_emails | Emails per customer | Several rows per CDE_ID possible | CDE_ID, Email_Address |
| marketing.mkt_adobe_message_feedback | — | Adobe email sends | One row per message send | journey_id, journey_action_id, feedback_status, message_profile_id, Email_Address |
| marketing.mkt_adobe_email_tracking | — | Opens and clicks | One row per interaction | message_profile_id, interaction_type, timestamp |
| cpx_dataanalytics_silver.cineclub.gifting | — | Gift memberships | One row per gift code | id (code), purchaser_id, redeemer_id, redeemer_plan_id, is_redeemed, purchase_date, redemption_date (UTC) |
| marketing.mkt_tbltransvoucher, mkt_tblvouchertype | — | Voucher redemption by BIN (e.g. 7146, 7132), matched to transactions by location + transaction number + visit date | One row per voucher scan | voucher_type_id, n_voucher_code, l_location_id, trans_c_lgn_number, session_dtm_realshow |
| data_exploration_hub.marketing_hub.mkt_dashboard_concessions_monthly | — | Pre-aggregated monthly concession revenue by category | One row per report_month × population_type | concession_{popcorn,fountain_drinks,candy,hot_foods,merch}_{revenue,transactions,quantity,members} |
| Excel: General Admit Prices.xlsx (Sheet3) | df_ga_xlsx | GA ticket price by location and year, used for member savings | One row per LocationID × year (melted from wide) | LocationId, visit_year, ga_ticket_price |
| Excel: Circuit Pricing.xlsx | df_pricing | A second, differently shaped GA price source also used for savings; Theatre_ID is a comma-separated list exploded to one row per theatre | One row per theatre after explode | Theatre_ID, General (price), Child, Senior, Cineplex_Tuesday |
| Excel: Subscriptions <Mon YYYY>.xlsx | df_main | Monthly Chargebee export; **archived** for sign-up reporting (events table takes precedence) | One row per subscription | customers.id, subscriptions.plan_id/status/created_at/cancelled_at |

**Standard joins**

| Rule | From → To | Key | Type | Used in |
| --- | --- | --- | --- | --- |
| BR-001 | Transaction → Location | LocationId = LocationID | Left (inner in LBE notebooks) | ~110 joins, all domains |
| BR-002 | Transaction → Date | VisitDateId = dateid | Left | All visit logic |
| BR-003 | Transaction → Ticket type | TicketTypeSkey = tickettypeskey | Left | ~50 joins: MOM, Wrapped, formats, savings |
| BR-004 | Transaction → Item | ItemSkey = itemskey | Left / inner | Concessions, LBE Food/Games, Wrapped, gamebands |
| BR-005 | Transaction → Session | sessionskey | Left | Premium/format classification |
| BR-006 | Transaction → Film | film_id | Inner / left | Film history, film clustering, GS dashboard |
| BR-007 | Snapshot ↔ Transactions | snapshot.cde_id = trans.CDE_ID | Inner | Member behaviour |
| BR-008 | Events / Chargebee / gifting → Snapshot or Customer | customer_id, purchaser_id or redeemer_id = cc_user_profile_id | Inner | Demographics, gifts |
| BR-009 | Email sends → Customer | Email_Address → CDE_ID (lower-cased in the GS dashboard) | Left | Email tests, cross-visitation |

## 3. Business Rules

BR-001 to BR-009 are the joins in section 2. The rest are listed below.

**Type codes:** Owner = confirmed by the repo owner (2026-09-27 unless another date is given); Impl = implemented in code; Doc = confirmed by an offer post-mortem document; Conv = analytical convention; Tech = technical rule; Assump = assumption; AS = analysis-specific.

**Confidence:** H = consistent across 5 or more scripts or owner-confirmed; M = 2–4 scripts, or one canonical script; L = a single script.

**Precedence:** when the owner supplies script logic for an analysis, follow it exactly, even where it differs from the standards in this book. Flag the differences; don't apply them. Owner-specified analyses are recorded here with their exceptions (e.g. KPI-29).

| Rule ID | Category | Rule | Type | Source | Conf. |
| --- | --- | --- | --- | --- | --- |
| BR-010 | Filtering | Theatre scope = `LC_Location_Type_Description = 'Cineplex Theatre'` | Impl | MOM series, Wrapped, Incrementality, Cross-Visitation SQL | H |
| BR-011 | Filtering | Every theatre base also requires `RecRoomFLAG = 'No'` and `STOREFLAG = 'No'`, so nothing non-theatre leaks into theatre analysis | Owner | MOM (df_trans_main), Wrapped; missing in the CineClub ticket base (update before reuse); not applied in KPI-29 by owner choice | H |
| BR-012 | Filtering | LBE = Rec Room + Playdium combined. Use `LC_Location_Type_Description LIKE 'Restaurant%'` when brand-level detail is not needed; add `LC_Trade_Brand_Description IN ('Rec Room','Playdium')` for a brand split | Owner | F&B Trend, Scene Redemption, Summer Pass Final; other notebooks use non-standard filters | H |
| BR-013 | Population | Identified customer = `CDE_ID IS NOT NULL` and `cde_id > 0` | Tech | MOM, Wrapped, Incrementality, CPP | H |
| BR-014 | Aggregation | Refunds are netted by summing Quantity to the visit grain and keeping `Net_Quantity > 0` | Tech | MOM, Wrapped, ticket-format tables | H |
| BR-015 | Population | CineClub line = `CineClubFLAG = 'Yes'`. Scene+ line = `SCENEFLAG = 'Yes'`. Scene+ non-CineClub = SCENEFLAG Yes and CineClubFLAG IS DISTINCT FROM 'Yes' | Impl | MOM, YTD Visitation, CPP, CPA/CPR | H |
| BR-016 | Data Definition | Ticket line = `TicketFLAG = 'Yes'`; concession line = `ConcessionFLAG = 'Yes'` | Impl | ~600 uses across the repo | H |
| BR-017 | Filtering | Paid attendance = ticket lines with `paid_attendance_inflag = 'Yes'` | Impl | MOM Aug 2026, cell 36 | L |
| BR-018 | Data Definition | `CineClub_Discount_Type_ID`: 0 non-member; 1 member, no discount; 2 20% off concessions; 3 20% off games; 4 10% off rentals/purchases; 5 companion ticket at $9.99; 6 redeemed member ticket valued at $9.99 (ties to `recognitions_redeemed`); 7 companion ticket at $10.99; 8 redeemed member ticket valued at $10.99 | Owner | Owner comment; GS Dashboard labels 0–6 | H |
| BR-019 | Business Logic | CineClub concession discount $ = `Net_Spend / 4` for discount types 2 and 3 (Net_Spend avoids over-counting Scene+ point redemptions) | Owner | MOM, P&L concession trend | H |
| BR-020 | KPI | $9.99 member tickets = discount types 5 and 6; $10.99 member tickets = types 7 and 8. Analyses covering dates from 2026-06-15 onward must include 7 and 8 | Owner | MOM series "9.99 Tickets" (5, 6 only) | H |
| BR-021 | Data Definition | Plans: `monthly1` = Monthly, `annual1` = Annual, `gift-plan` = Gift Plan | Impl | GS Dashboard, MOM, YTD SQL | H |
| BR-022 | Population | Active member (snapshot) = `renewal_status LIKE 'Active%'`, which covers Active, Active_Non_Renewing and Active_Annual_Upgrade | Impl | Incrementality, Liability, Redemption Rate, Rewind, CPA/CPR | H |
| BR-023 | Population | *Archived:* Active member (Chargebee export) = status Active or Non Renewing. Replaced by the events-table rules BR-027 to BR-031 | Owner | MOM series | — |
| BR-024 | Data Quality | Customer `AzqBunUAsPVB41nKT` is always excluded from the Chargebee export: it is an artifact row that causes a data-type error | Owner | 13 MOM notebooks | H |
| BR-025 | Data Quality | Test/dummy accounts are excluded with a date-appropriate `recognitions_left` ceiling. CineClub launched August 2021 and a member earns at most 1 recognition per month, so the maximum legitimate balance ≈ months elapsed since 2021-08 at the analysis date. Fixed values seen in code (54, 80, 100) are snapshots of that moving ceiling, not standard constants | Owner | Wrapped, Liability, Redemption Rate | H |
| BR-026 | Time Logic | Snapshot choice varies: a hard-coded `snapshot_date_id`, `MAX(snapshot_date_id)` (GS dashboard, Lifetime Savings), yesterday (RFM Segmentation), or period end (YTD SQL) | Conv | Repo-wide | M |
| BR-027 | Business Logic | Membership flows (sign-ups, first-time, returning, exits, upgrades, renewals, balance, lapsed, dunning) come from `mkt_subscription_events` using the YTD SQL logic. A `subscription_cancelled` with `changed = TRUE` is an upgrade completing, not an exit | Owner | YTD Signups SQL, CPA/CPR series | H |
| BR-028 | KPI | Annual renewal = an `annual1` payment_succeeded at least 360 days after start and within 5 days of a 365-day anniversary. Excludes the period's sign-ups, exits and upgrade completions | Impl | YTD Signups SQL, CPA/CPR series | M |
| BR-029 | Population | Dunning = latest payment event is `payment_failed`, dated on the term start, within 15 days (monthly) or 14 days (annual) of period end, with no later successful term | Impl | YTD Signups SQL (revised script pending, CF-15) | M |
| BR-030 | Population | Active balance (events) = monthly `payment_succeeded` whose term covers period end; annual/gift created, renewed, resumed or started event whose term covers period end | Impl | YTD Signups SQL ("verified daily script") | H |
| BR-031 | Population | Lapsed = active at the prior period end; not active or dunning at period end; no cancellation event to date | Impl | YTD Signups SQL | M |
| BR-032 | Time Logic | Gift purchase and redemption timestamps are converted UTC → America/New_York before dating | Tech | YTD Signups SQL, CPA/CPR series | H |
| BR-033 | Population | A gift purchaser or redeemer is net new if absent from the prior-period-end snapshot | Impl | YTD Signups SQL, CPA/CPR series | M |
| BR-034 | Segmentation | RFM segment comes from `c360_f_rfm_model` at the period-end snapshot. No match = Dormant. Top 3 RFM = Champions + Potential Loyalist + Value Conscious Loyals. Always match labels against the table's own `Segment` column values, never hand-typed strings. The hand-coded RFM in Incrementality Final was a one-off and is not standard | Owner | YTD Signups SQL, RFM Segmentation, Ethnic Analysis | H |
| BR-035 | Business Logic | Members earn 1 free ticket per month, from the month after activation to 1 month before renewal | Impl (stated in notebook) | Ticket Liability Analysis | M |
| BR-036 | Transformation | Effective renewal date = `annual_current_term_end_datetime` for (a) gift-plan Active_Non_Renewing with a past renewal date, or (b) annual1 whose term end is more than 11 months past renewal; otherwise `renewal_datetime` | Impl | Ticket Liability Analysis | L |
| BR-037 | Business Logic | A gift is pending (not yet applied) if redeemed outside the 12 months before the effective renewal date | Impl | Ticket Liability Analysis | L |
| BR-038 | Segmentation | Ticket format comes from session `Auditorium_Experience_ID` (and 3D from `Performance_Experience_ID = 2`). ALT (IDs 5, 6) is its own category for special screenings and events, neither premium nor regular. ALT sub-categories = `cpx_d_performance_type.performance_type` (via session Performance_Type_ID); within ALT, format comes from the first session flag set to Yes, in priority order: 3D, 4DX, 70mm, AVX, ScreenX, DBox, Dolby Atmos, DS, HFR, IMAX, IMAX VR, Non-IMAX, Clubhouse, else Regular | Owner | MOM, Incrementality, Q4 Offer, CPA/CPR, Event Cinema | H |
| BR-039 | Transformation | Province: NB, NL, NS, PE and PEI → ATL. Territories, US states and 'WRONG' → NULL, excluded from the denominator | Impl | MOM series | H |
| BR-040 | Filtering | Gender share uses Male and Female only; other values are dropped from the denominator | Impl | MOM series | H |
| BR-041 | Transformation | Age = current year − `date_of_birth_masked`; under 14 → NULL, excluded from the denominator | Impl | MOM series, Cross-Visitation | H |
| BR-042 | Segmentation | LBE Food vs Games is set by `Item_Class_Description1` lists; exclude any class reflecting G&E or COMP items | Owner | F&B Trend (closest to the rule); FIFA, PDM, Minions include G&E (non-standard) | H |
| BR-043 | Assumption | A redeemed Scene+ point is worth $0.00942 to the business; each free LBE game costs up to $7 | Assump | LBE Scene Redemption Offer | L |
| BR-044 | Business Logic | Large popcorn price $9.49, cost $0.67; Grande nachos price $7.99, cost $3.42. Black Friday popcorn cost per redemption $6.37 | Doc | Exclusive Monthly Offers post-mortem; CPA/CPR Analysis 1 | M |
| BR-045 | Population | Offer eligibility counts monthly members who click upgrade before the cut-off, even if annual billing starts later | Doc | Black Friday, Winter Holiday, Exclusive Offers post-mortems | M |
| BR-046 | Business Logic | Kernel counts: small 300, regular 400, large 500; upsizes use the larger size | AS | CineClub Wrapped 2025 | L |
| BR-047 | Segmentation | "Top 10% visitor" badge = Total_Visits > 21 (one-off Wrapped requirement, not a standard rule) | AS | CineClub Wrapped 2025 | L |
| BR-048 | Segmentation | Concession badge = concession visits ÷ total visits ≥ 0.75 | AS | CineClub Wrapped 2025 | L |
| BR-049 | Filtering | Film clustering keeps films with ticket sales after 2022-12-01, average line revenue under $100 and attendance above 0 | AS | Film Classification Model | L |
| BR-050 | Population | Email cross-visitation: exclude members who redeemed Scene+ points on the qualifying ticket transaction. Cohort = opener vs non-opener | AS | LBE Final (EXH x LBE) SQL | L |
| BR-051 | Population | GS dashboard members = latest snapshot, `recognitions_redeemed > 0`, joined inner to email; transactions after 2021-01-01, tickets only | Impl | GS Dashboard Table Definitions | M |
| BR-052 | Data Quality | Chargebee export: dedupe on 7 columns; `days_in_program` = cancelled_at − created_at, or −9999 if either is null | Tech | MOM series (archived source) | H |
| BR-053 | Business Logic | Price change on 2026-06-15: monthly members move to the $10.99 plan at their next renewal; existing annual members stay at $9.99 | Owner | Owner comment | H |
| BR-054 | Business Logic | Ticket burn order: $9.99 recognitions are used first, then $10.99 once no $9.99 recognitions remain | Owner | Owner comment | H |
| BR-055 | Business Logic | Annual members who stacked a gift before 2026-06-15 keep $9.99 benefits (recognitions and companion tickets) for 1 year from the day after their annual term ends | Owner | Owner comment | H |
| BR-056 | Business Logic | Upgrade to annual = `subscription_changed` with `changed = TRUE`, `changed_from != changed_to`, `changed_to = 'annual1'`, dated by `record_occurred_at` | Owner | YTD Signups SQL | H |
| BR-057 | Business Logic | Voucher redemptions are matched to transactions on location + transaction number + visit date, deduped per transaction, and capped at 3 per member. Vouchers can be passed on, so direct CDE matching undercounts (~48% in Black Friday 2025); total valid redemptions are allocated to segments proportionally by segment size | AS | CPA/CPR Analysis 1 | L |
| BR-058 | Segmentation | CPA vs CPR cost buckets: CPA = first-time, returning, upgraded, net-new gift purchasers and redeemers; CPR = renewals | AS | CPA/CPR series | L |
| BR-061 | Segmentation | Scene vs Connect: Scene line = `SCENEFLAG = 'Yes'`; Connect line = `SCENEFLAG = 'No'`, i.e. an identified guest (`CDE_ID IS NOT NULL`) who didn't scan Scene+. The split is per line, not per guest, so a guest can appear in both groups. NULL SCENEFLAG lines are in neither group. CineClub members stay inside Scene (no CineClub split) | Owner (2026-10-21) | Scene vs Connect spend script | H |

## 4. KPI Dictionary

The standard definition is the one confirmed by the owner or most consistently implemented. Non-standard variants are kept for reference.

| KPI | Standard definition and formula | Source / filters | Variants / notes |
| --- | --- | --- | --- |
| KPI-01 Sign-ups | Distinct cc_user_profile_id with `subscription_created` in the period, by plan_id. Annual excludes upgrade completions (BR-027) | YTD SQL (standard, owner-confirmed) | Archived: count of export rows with `created_at` in the month (MOM) |
| KPI-02 First-time sign-up | Member with exactly one `subscription_created` event ever | YTD SQL | Archived: no earlier `created_at` in the export; MOM "Net New" = one export row only |
| KPI-03 Returning sign-up | Sign-up by a member with more than one created event; prev_plan_id = LAG(plan_id) over created events | YTD SQL | Archived: plan of the latest earlier export record |
| KPI-04 Exits | Distinct members with `subscription_cancelled` and `changed IS NOT TRUE` in the period | YTD SQL | Archived: export rows with `cancelled_at` in the month |
| KPI-05 Net exits | Not needed under the standard: upgrade cancellations are already removed from KPI-04 | YTD SQL | Archived MOM formula: distinct exits − exits now holding active annual1 |
| KPI-06 Upgrades to annual | BR-056 | YTD SQL (standard, owner-confirmed) | Older offer notebooks use non-standard versions |
| KPI-07 Ending active balance | Term-window event logic (BR-030) plus dunning (BR-029) at period end | YTD SQL | Snapshot `renewal_status LIKE 'Active%'`; archived: export status Active or Non Renewing |
| KPI-08 Visits | Distinct CDE_ID × date × LocationId with net quantity > 0 (owner-confirmed key) | MOM, Wrapped, YTD Visitation, PDM, CPP, Tim Hortons RU2W | Non-standard: countDistinct(datefull), CDE_Visit_ID_MNL, CDE_ID\|VisitDateId. KPI-29 uses distinct VisitDateId per guest by owner choice — keep it there |
| KPI-09 Tickets | SUM(Quantity) where TicketFLAG = Yes. Refunds are always included — a refund is a negative quantity that cancels its original transaction | MOM, Rewind, Incrementality | Incrementality Revision's row-count version needs fixing |
| KPI-10 Paid tickets | SUM(Quantity) where TicketFLAG = Yes and paid_attendance_inflag = Yes | MOM | — |
| KPI-11 Active members (behavioural) | Members with ≥1 visit in the period | MOM, YTD | Differs from KPI-07, which is subscription status |
| KPI-12 Avg spend per active member | Total ticket (or concession) Net_Revenue ÷ members with ≥1 visit | MOM, CPA/CPR | — |
| KPI-13 Concession transactions (CC vs non-CC) | Per transaction number: CC = any line with CineClubFLAG and ConcessionFLAG; non-CC = has a concession line and no CC line | MOM concession trend | Discount type 2 and CineClubFLAG = Yes are interchangeable for CC concession lines (owner-confirmed) |
| KPI-14 CC discount $ | Net_Spend (types 2, 3) ÷ 4 | MOM P&L trend (owner-confirmed) | Net_Revenue × 0.25 (Savings, Rewind) is non-standard |
| KPI-15 Member savings | Tickets × (GA price for location and year − ticket value) + 25% of type-2 concession revenue + 25% of type-3 gaming revenue + online booking savings ($1.50 flat per member-priced or free ticket, per the Wrapped notebook — owner-confirmed standard). Ticket value = $9.99 for types 5 and 6, $10.99 for types 7 and 8 | Lifetime Savings, Rewind, CC Savings May | Existing code covers 5 and 6 only and hard-codes $9.99. Two GA price sources (General Admit Prices.xlsx vs Circuit Pricing.xlsx) are not yet reconciled. Lifetime Savings' "$1 per online CC ticket visit" is superseded |
| KPI-16 Free ticket redemption rate | Tickets redeemed ÷ (redeemed + recognitions_left) | Redemption Rate (2027 Budget) | Should split $9.99 and $10.99 recognitions after 2026-06-15 |
| KPI-17 Ticket liability | Current holdings (SUM recognitions_left, below the test-account ceiling) + forward liability (MAX(0, CEIL(months_between(effective_renewal, snapshot) − 1))) + pending gifts | Ticket Liability Analysis | — |
| KPI-18 Ticket format share | Net tickets by Final_Format ÷ total net tickets × 100; unique buyers by format | MOM | — |
| KPI-19 Premium uptake | Premium tickets ÷ (regular + premium). Premium = auditorium IDs 2, 3, 4, 7, 8; ALT (5, 6) reported separately | Q4 Offer, Incrementality; ALT rule owner-confirmed | CPA/CPR counts anything not Regular (incl. ALT, 3D, D-BOX) as premium — non-standard |
| KPI-20 Revenue per visit | Net_Revenue ÷ visits | Incrementality | — |
| KPI-21 LBE Food / Games revenue | SUM(Net_Revenue) by category label, per venue and period; G&E and COMP classes excluded | F&B Trend | FIFA, PDM, Minions include G&E (non-standard) |
| KPI-22 Scene+ penetration (LBE) | Identified (non-null CDE_ID) item rows or revenue ÷ all rows or revenue | F&B Trend | Identified visits are treated as a lower bound on footfall |
| KPI-23 Offer cost (voucher) | Opportunity cost mixes redeemers' P12M behaviour: 0 purchases = incremental revenue + cost; 1 = cost of 2nd item; 2+ = forgone revenue | Doc: Exclusive Offers post-mortem | CPA/CPR: scaled redemptions × cost per redemption ÷ segment population |
| KPI-24 Scene redemption break-even | points × $0.00942 ≥ free games × $7; max games = floor(band points × 0.00942 ÷ 7) | LBE Scene Redemption Offer | — |
| KPI-25 Retention after offer | Offer cohort members with `renewal_status LIKE 'Active%'` on a later snapshot ÷ cohort size | CPA/CPR Analysis 3 | — |
| KPI-26 CPP (concessions per patron) | Total concession Net_Revenue ÷ total ticket Quantity, by population (CC / Scene / total business) | CPP Deep Dive series | Also reported as concession attach rate: visits (or ticket-buying patrons) with a concession line ÷ all visits (or patrons) |
| KPI-29 Scene vs Connect guest value (visits × avg spend) | Per year × group (Scene / Connect, BR-061) × category (box office / concessions): guests = COUNT(DISTINCT CDE_ID); visits = sum over guests of COUNT(DISTINCT VisitDateId); revenue = SUM(Net_Revenue); visits per guest = visits ÷ guests; revenue per visit = revenue ÷ visits; revenue per guest = revenue ÷ guests (= visits per guest × revenue per visit) | `LC_Location_Type_Description = 'Cineplex Theatre'`; `CDE_ID IS NOT NULL`; box office = `TicketFLAG = 'Yes'` and `paid_attendance_inflag = 'Yes'` (ticket type left-joined); concessions = `ConcessionFLAG = 'Yes'`, no paid filter; window TL-25. Owner-confirmed script, 2026-10-21 | Owner-specified exceptions — keep as is: visits are distinct dates per guest, not KPI-08; no refund netting (BR-014); no RecRoomFLAG/STOREFLAG filter (BR-011); no `CDE_ID > 0` (BR-013); ticket type not deduped (DQ-05); dated by VisitDateId, not Transaction_DateTime (TL-04). Box office and concessions each have their own guest base, so their per-guest figures don't add. Connect covers identified non-Scene guests only; anonymous guests (no CDE_ID) aren't measured |

## 5. Population Definitions

The first seven rows are reused across analyses and count as standard. Rows marked **AS** are analysis-specific and should not be reused without checking.

| Population | Inclusion | Exclusion | Time window | Source |
| --- | --- | --- | --- | --- |
| P-01 Active CineClub member (snapshot) | `renewal_status LIKE 'Active%'` on the chosen snapshot_date_id | Test accounts via the date-based recognitions_left ceiling (BR-025) | Single snapshot date | Incrementality, Liability, Rewind, Email Test |
| P-02 Active CineClub member (export) — archived | status Active or Non Renewing | Customer AzqBunUAsPVB41nKT | Export date | MOM |
| P-03 Program-to-date (PTD) sign-ups | Every export customer_id, inner-joined to cpx_d_customer on cc_user_profile_id | Customers without a CDE match | All history | MOM |
| P-04 Month sign-ups cohort | Export rows created in MOM_month, deduped by customer_id, left-joined to demographics | — | Report month | MOM |
| P-05 Scene+ members (demographics) | Any SCENEFLAG = Yes transaction ever, joined to cpx_d_customer | — | All history | MOM |
| P-06 Scene+ visitors (current year) | SCENEFLAG Yes, RecRoomFLAG No, STOREFLAG No, VisitDateId LIKE '2026%' | — | Hard-coded year | MOM Aug 2026 |
| P-07 Scene+ non-CineClub | SCENEFLAG Yes and CineClubFLAG != Yes, theatre, identified | — | Period | MOM, YTD, CPP |
| P-08 Wrapped 2025 primary (AS) | Snapshot 2025-12-04: (annual1 or gift-plan with Active or Active_Non_Renewing) or (monthly1 with the annual-upgrade status) | recognitions_left ≥ 54 | 2025-01-01 to 2025-12-04 | Wrapped V5, All |
| P-09 Wrapped 2025 monthly (AS) | Snapshot 2025-12-04: monthly1 with Active or Active_Non_Renewing | recognitions_left ≥ 54 | Same | Wrapped V5 |
| P-10 Liability base (AS) | Snapshot 2026-07-15, Active%, plan matches `%annual%` or `%gift%` | No recognitions_left filter | Snapshot → effective renewal | Ticket Liability |
| P-11 Redemption-rate base (AS) | Snapshot 2026-07-15, 0 < recognitions_left < 100 | Test accounts ≥ 100 | P12M: 2025-07-15 to 2026-07-14 | Redemption Rate (2027 Budget) |
| P-12 5+ unused tickets email test (AS) | Snapshot 2026-05-13, recognitions_left ≥ 5, Active% or Cancelled%. Four journeys: Active/Cancelled × visited/not visited in P12M. Treatment = sends with feedback_status 'sent' (10% of audience); control = eligible and not sent | — | 30 days post-send (May 13 – Jun 13) | 5+ Unused Tickets Email Test |
| P-13 Minions × LBE cross-visitation (AS) | Scene+ ticket buyers of title_id 38680, 2026-07-02 to 2026-08-21. Opener = any open of the journey's sends; control = non-opener | Members who redeemed Scene+ points on the qualifying transaction | LBE visits up to 2026-09-11; "within 3 weeks" check | LBE Final SQL, Minions Post Analysis |
| P-14 Gift purchasers / redeemers | Distinct purchaser_id or redeemer_id (is_redeemed = TRUE) with an ET-converted date in the period | — | Year-to-date | YTD SQL |
| P-15 Tim Hortons gift redeemers (AS) | Listed gift codes joined to gifting on code = id; redeemer → snapshot → cde_id | — | ±365 days around each member's redemption date | Tim Hortons Incrementality, RU2W Winners |
| P-16 PDM Vaughan incumbents (AS) | Identified customers at Playdium Orion Gate, Whitby and Fairview; Vaughan's post-opening customers used for transference | Vaughan itself from detrended revenue (no pre-history) | Post = 2026-06-17 to 2026-07-26; opening date read from cpx_d_location | PDM Vaughan Impact |
| P-17 Offer eligibility (Doc) | New or returning annual joiners; active monthly members upgrading, including those still upgrading; Winter Holiday adds gift purchasers | None. The Winter Holiday 30-day cancel rule was waived by the stakeholder | Black Friday 2025-11-28 to 12-01; Winter Holiday 2025-12-11 to 12-26 | Offer post-mortem docs |
| P-18 Current / Former / Never CC (AS) | Current = latest snapshot Active%; Former = any CineClubFLAG transaction but not current; Never = no CC transaction ever | — | Transactions from 2021-08-01 | Lifetime Savings Send Out |
| P-21 Scene / Connect guests (AS) | Scene = guests with ≥1 `SCENEFLAG = 'Yes'` line; Connect = guests with ≥1 `SCENEFLAG = 'No'` line (BR-061). Theatre (`LC_Location_Type_Description = 'Cineplex Theatre'`), `CDE_ID IS NOT NULL`, counted within each category (box office or concessions) | NULL SCENEFLAG lines; anonymous guests | Per year, TL-25 | Scene vs Connect spend script (owner, 2026-10-21) |

## 6. Transformation Dictionary

| ID | Input | Logic | Output | Purpose | Source |
| --- | --- | --- | --- | --- | --- |
| T-01 | Chargebee export datetimes (`dd-MMM-yyyy HH:mm`) | Parse in pandas → string → Spark `to_timestamp`; rename dotted columns | created_at, cancelled_at, current_term_start/end | Arrow-safe load of the monthly export (archived source) | MOM |
| T-02 | created_at, cancelled_at, current_term_end | `date_format(..., 'yyyy-MM')` | *_year_month | Month bucketing (archived source) | MOM |
| T-03 | Transaction lines | groupBy(cde_id, datefull, LocationId) → SUM(Quantity), plus ticket/concession quantity and spend via conditional sums; keep Net_Quantity > 0 | Visit-grain table | Standard visit key, refund-netted | MOM `build_visit_grain_base`, Wrapped |
| T-04 | Ticket lines | groupBy(cde_id, datefull, LocationId, sessionskey, TicketTypeSkey) → Net_Ticket_Qty > 0 | Ticket-event grain | Format classification | MOM |
| T-05 | Session Auditorium_Experience_ID, Performance_Experience_ID | Map 2 AVX, 3 IMAX, 4 VIP, 5–6 ALT (special screenings and events), 7 4DX, 8 ScreenX; else Performance = 2 → 3D; else Regular | Final_Format | Ticket format mix | MOM, CPA/CPR |
| T-06 | date_of_birth_masked | `year(current_date) − dob`, then binned (section 7) | age, age_bin | Demographics | MOM |
| T-07 | province | upper(trim), then ATL collapse and invalid → NULL | Province | Demographics | MOM |
| T-08 | Visits per customer and year | Cross join the cohort with the years in `var_visit_year` (2019, 2025); zero-fill missing | total_visits_year including 0 | Visit distribution with non-visitors | MOM |
| T-09 | Transaction lines | groupBy(CDE_ID, LocationId, datefull, Transaction_Number); MAX of flag indicators | Has_Concession, Is_CineClub_Transaction | Transaction-level concession incidence | MOM concession trend |
| T-10 | Email table | lower(Email_Address); row_number per CDE_ID (ordered by CDE_ID only) → Primary and Alternate | Primary_Email, Alternate_Email, Search_Column | Guest Services lookup | GS Dashboard |
| T-11 | plan_id codes | monthly1 → Monthly, annual1 → Annual, gift-plan → Gift Plan | Display plan | Reporting | GS Dashboard |
| T-12 | CineClub_Discount_Type_ID | 0–8 → text labels (BR-018); the GS dashboard maps only 0–6 today | Discount_Type | Reporting | GS Dashboard |
| T-13 | Subscription events | LAG(plan_id) over reporting_datetime per member | prev_plan_id | Returning-from breakdown | YTD SQL |
| T-14 | Snapshot dates | renewal_datetime corrected to annual_current_term_end_datetime (BR-036) | effective_renewal_datetime | Liability projection | Ticket Liability |
| T-15 | Recency, visits, revenue per visit | R, F and M scores 1–4 concatenated as a string → segment map. One-off; the standard is the gold RFM table (BR-034) | RFM_Score, Segment | Member value segmentation (AS) | Incrementality Final |
| T-16 | Item_Class_Description1 | Membership in games or food lists, G&E and COMP classes excluded | category_label (Games / Food) | LBE revenue split | F&B Trend, FIFA, PDM |
| T-17 | Item_Description | LIKE patterns for POPCORN × SMALL / REGULAR / LARGE, VHO DRINK sizes, upsizes; kernel weights | Popcorn and drink counts, Total_Kernels | Wrapped stats | Wrapped 2025 |
| T-18 | total_attendance, avg_ticket_revenue per film | VectorAssembler → MinMaxScaler → K-means | cluster → category label | Film tiering | Film Classification |
| T-19 | Gift timestamps | from_utc_timestamp(…, 'America/New_York') → date | Local purchase and redemption date | Correct month bucketing | YTD SQL |
| T-20 | Gameband Item_Description | Static mapping of POS tag names to price bins (section 7) | price_bin | Gameband tier analysis | $25 Gameband, Summer Pass Final |
| T-21 | Concession line items | Exclude Item_Description LIKE '%Delivery%', '%BP TAG%', 'VHO MISC%', 'VHO XSCAPE%'; exclude 'CAKE CUTTING FEE', 'HOST FEE', '<Unknown>', ''; exclude EIS_Sub_Group_Description LIKE '%Merchandise%' | "Real" member spend | Quarterly savings base (AS) | CineClub Rewind (SQL) |
| T-22 | Candy items | Item_Class_Description1 IN ('VHO-CANDY','CANDY','SMALL CANDY'), excluding DELIVERY%, VHO DELIVERY%, SNACK%; '% SUP %' in the name = Large | Candy size, modal price | Candy analysis (AS) | CineClub 5 Year Anniversary |

## 7. Segmentation & Bucketing

All boundaries are exactly as implemented. Each boundary is inclusive unless stated otherwise.

**Age bins** (MOM series; age = current year − birth year)

| Bin | 14–18 | 19–24 | 25–34 | 35–49 | 50–64 | 65+ | NULL |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Rule | ≥14 | ≥19 | ≥25 | ≥35 | ≥50 | ≥65 | <14 or missing (excluded from %) |

**Annual visit bins** (MOM series; a customer × year grid including zero-visit years)

| Bin | 0 Visits | 1 Visit | 2–3 Visits | 4–6 Visits | 7–12 Visits | Over 12 Visits |
| --- | --- | --- | --- | --- | --- | --- |
| Rule | = 0 | = 1 | 2 to 3 | 4 to 6 | 7 to 12 | ≥ 13 |

**RFM** — the standard segment comes from `c360_f_rfm_model`, with Dormant for members with no match (BR-034). Always read and match against the exact strings in the table's own `Segment` column; notebooks have used inconsistent spellings ('At Risk' vs 'At_Risk', 'Value Conscious Loyal' vs 'Loyals') that a filter can silently miss. The hand-coded RFM below was a one-off (Incrementality Final) and is kept for reference only.

| Score | R (days since last visit) | F (visits) | M (revenue per visit) |
| --- | --- | --- | --- |
| 4 | < 45 | > 6 | ≥ $40 |
| 3 | 45–89 | 4–6 | $30–39.99 |
| 2 | 90–179 | 2–3 | $20–29.99 |
| 1 | ≥ 180 | ≤ 1 | < $20 |

| Segment (one-off map) | Scores |
| --- | --- |
| Champions | 443, 444 |
| Potential Loyalist | 232–234, 242–244, 332–334, 342–344, 432–434, 442 |
| Value Conscious Loyals | 231, 241, 331, 341, 431, 441 |
| Promising | 221–224, 321–324, 421–424 |
| New | 311–314, 411–414 |
| At Risk | 121–124, 131–134, 141–144 |
| Fleeting | 111, 112, 211–214 |
| High Value Lapsed | 113, 114 |
| NA | Anything else |

**Ticket format** (MOM `add_ticket_format`)

| Auditorium_Experience_ID | 2 | 3 | 4 | 5, 6 | 7 | 8 | Other, with Performance_Experience_ID = 2 | Other |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Label | AVX | IMAX | VIP | ALT | FourDX | ScreenX | ThreeD | Regular |

Premium = Auditorium IDs 2, 3, 4, 7, 8; regular = 1. ALT (5, 6) is a separate category for special screenings and events — neither premium nor regular (owner-confirmed). Incrementality and Q4 Offer still count ALT as regular and need updating before reuse.

**recognitions_left buckets** (CC Ticket Redemption Stats) — two versions exist; both are acceptable (owner-confirmed):

- **Version A:** 0 · 1 · 2–3 · 4–5 · "6 to 7" · 8–10 · 10+ (> 10).
- **Version B:** 0 through 9 individually, then 10+ (≥ 10).

**Gameband price tiers** (tag bins mapped from POS item names)

| Bin | Items |
| --- | --- |
| $5 | TAG $5 - 20 CR |
| $10 | BP TAG+$10 - 52 CR; PREM TAG+$10 - 54 CR; TAG $10 - 40 CR |
| $15 | BP TAG+$15 - 72 CR; PREM TAG+$15 - 74 CR; TAG $15 - 60 CR |
| $25 | BP TAG+$25 - 172 CR; PREM TAG+$25 - 174 CR; TAG $25 - 160 CR |
| $35 | BP TAG+$35 - 257 CR; PREM TAG+$35 - 257 CR; PREM TAG+$35 - 259 CR; TAG $35 - 245 CR |
| $50 | BP TAG+$50 - 377 CR; PREM TAG+$50 - 379 CR; TAG $50 - 365 CR |
| $75 | BP TAG+$75 - 577 CR; PREM TAG+$75 - 579 CR; TAG $75 - 565 CR |
| $100 | BP TAG+$100 - 802 CR; PREM TAG+$100 - 804 CR; PROMO BP TAG $100 - 1002 CR; TAG $100 - 790 CR |
| $129 | BP TAG+$129 - 1107 CR; PREM TAG+$129 - 1109 CR; TAG $129 - 1095 CR |

Some analyses (Summer Pass Final) group $10/$15 and $100/$129 together.

**Other segmentations**

| Dimension | Values | Source |
| --- | --- | --- |
| Province | ON, QC, BC, AB, MB, SK, ATL (NB, NL, NS, PE, PEI); everything else NULL | MOM |
| Gender | Male, Female only | MOM |
| LBE brand | Rec Room (TRR), Playdium (PDM), from `LC_Trade_Brand_Description`; LBE = both combined | LBE notebooks |
| PDM venues | Vaughan, Orion Gate, Whitby, Fairview | PDM Vaughan |
| Cohort | CineClub vs Scene+ (non-CineClub) | MOM, YTD, CPP |
| Member ticket price tier | $9.99 (discount types 5, 6) vs $10.99 (7, 8), from 2026-06-15 | Owner (BR-053) |
| Film category | K-means clusters labelled in order: Blockbuster, Mainstream, Limited Run, Niche, Micro Release, Specialty, Archive, Other | Film Classification |
| Wrapped badges (AS) | Top 10% visitor = visits > 21; Concession badge = ratio ≥ 0.75 | Wrapped 2025 |

## 8. Time & Date Rules

Most windows are calendar-based and hard-coded per run.

| ID | Rule | Start | End | Inclusivity | Source |
| --- | --- | --- | --- | --- | --- |
| TL-01 | Report month = `MOM_month` ('yyyy-MM') | First of month | `last_day(start)` | Both inclusive | MOM |
| TL-02 | MOM snapshot = first day of the following month ('20260901' for Aug 2026) | — | — | Point in time | MOM |
| TL-03 | Default activity date = `VisitDateId` → `cpx_d_date.datefull` | — | — | — | Repo-wide |
| TL-04 | Standard by analysis type (owner-confirmed): `Transaction_DateTime` for financial or revenue analysis (a ticket bought now for a future show counts when paid); `VisitDateId` for behavioural or visitation analysis | — | — | — | Owner |
| TL-05 | YTD = Jan 1 of the ref year through the last day of the ref month; baseline = prior Dec 31 snapshot | trunc(ref, 'YEAR') | last_day(ref) | Inclusive | YTD SQL |
| TL-06 | Half-year "YTD" (Jan 1 – Jun 30) is an intentional ad-hoc view: the stakeholder reports YTD at H1 and H2 end each year | Jan 1 | Jun 30 (H1) or Dec 31 (H2) | Inclusive | MOM Aug 2026 (owner-confirmed) |
| TL-07 | PTD visit window | 2021-01-01 | Last day of the report month | Inclusive | MOM |
| TL-08 | Visit-distribution years = `var_visit_year` = [2019, 2025] | — | — | Calendar years | MOM |
| TL-09 | Current-year filters via string match, e.g. `VisitDateId LIKE '2026%'` | — | — | Calendar year or month | MOM |
| TL-10 | P12M for the redemption rate | 2025-07-15 | 2026-07-14 | Inclusive; rolling from the snapshot | Redemption Rate |
| TL-11 | Gift dates converted UTC → America/New_York before bucketing | — | — | — | YTD SQL |
| TL-12 | Age uses `current_date()`, not the snapshot or report date | — | — | — | MOM |
| TL-13 | Offer windows are parameters set per offer (e.g. Black Friday 2025-11-28 to 12-01; Winter Holiday 2025-12-11 to 12-26) | — | — | Per offer | Post-mortems + notebooks |
| TL-14 | Tim Hortons pre/post = ±365 days around each member's redemption date | redemption − 365 | redemption + 365 | Member-relative | Tim Hortons Incrementality, RU2W |
| TL-15 | FIFA tournament 2026-06-11 to 07-19. Counterfactual base = June 2025; trend = Jan–May YoY | — | — | Inclusive | TRR FIFA, Royalmount FIFA |
| TL-16 | PDM Vaughan post window 2026-06-17 to 07-26; opening date from `cpx_d_location.Opening_Date` | — | — | Inclusive | PDM Vaughan |
| TL-17 | LBE trend window: 2025-01-01 through yesterday | 20250101 | yesterday | Inclusive | F&B Trend, FIFA |
| TL-18 | Scene redemption base: Aug 2024 – Jul 2026, 24 full months | — | — | Month-level | LBE Scene Redemption |
| TL-19 | Gameband forecast: Oct 2024 → Oct 2025 YoY applied to Oct 2025 to forecast Oct 2026. $25 gameband pre/post = ±50 days around launch (2026-03-13), with prior-year same windows | — | — | Full October / ±50 days | Gameband notebooks |
| TL-20 | Summer Pass: offer period May 18 – Sep 7, pre-period Apr 13 – May 17, compared 2025 vs 2026 | — | — | Inclusive | Summer Pass Final |
| TL-21 | Subscription price change effective 2026-06-15; discount types 7 and 8 appear only from then on | 2026-06-15 | — | — | Owner (BR-053) |
| TL-22 | Film attendance: tickets after 2022-12-01 | 2022-12-02 | Today | Exclusive start | Film Classification |
| TL-23 | Dunning look-back: 15 days (monthly) or 14 days (annual) before period end | — | — | Inclusive | YTD SQL (revision pending) |
| TL-24 | CPA/CPR offer-cohort behaviour: pre = 2025-01-01 to 08-01, post = 2026-01-01 to 08-01; retention checked on the 2026-08-01 snapshot | — | — | Inclusive | CPA/CPR series |
| TL-25 | Scene vs Connect window: Jan 1 – Oct 20 of each year (2025 and 2026) on `VisitDateId`, so both years cover the same days | Jan 1 | Oct 20 | Inclusive (`between`) | Scene vs Connect spend script (owner) |

## 9. Analytical Methodologies

The repo repeats five approaches: cohort description, pre/post and YoY comparison, detrended counterfactual with a control category, test vs holdout, and Bayesian MMM. Only the email holdout (M-05) uses a randomised control.

| ID | Method | Purpose | Inputs and target | Approach | Validation | Known limitations | Scripts |
| --- | --- | --- | --- | --- | --- | --- | --- |
| M-01 | Monthly membership reporting | Sign-ups, exits, balance, demographics, visitation | Membership flows from the events table (M-02); snapshot and transactions for demographics and visits | Descriptive counts and shares by plan, cohort and bin | None in code | The Chargebee-export sign-up logic in MOM notebooks is archived | MOM series |
| M-02 | Membership flow (standard) | Event-based sign-ups, returning-from, upgrades, renewals, lapsed, dunning, by RFM | Subscription events, snapshot, gifting, RFM table | CTE pipeline; Dormant = All − sum of named segments | "Verified daily script" | Hard-coded ref date | YTD Signups Results, Anniversary Signups |
| M-03 | Member value / incrementality | Compare CineClub vs non-members and pre/post behaviour | Transactions, RFM scores, snapshot recognitions | Segment × year KPIs (visits, tickets, premium share, revenue) | None | Descriptive; no matched control | Incrementality Final, Revision, SQL Code, Q4 Offer |
| M-04 | Pre/post around a member event | Behaviour change after a gift redemption | Gift codes → gifting → snapshot → transactions | ±365 days around each member's own redemption date | None | No control group | Tim Hortons Incrementality, RU2W Winners |
| M-05 | Email test vs holdout | Lift from a reminder email | Adobe send list, snapshot at 2026-05-13 | Treatment = sent (10%); control = eligible and not sent; compare ticket use, churn, reactivation within 30 days | Holdout design | Control selection method not stated | 5+ Unused Tickets Email Test |
| M-06 | Opener vs non-opener | LBE cross-visits after a film email | Adobe sends and opens, Minions ticket buyers | Cohort by open status; visits within 3 weeks; P12M LBE history segments | None | Opening is self-selected, so lift is likely biased upward | LBE Final SQL, Minions Post Analysis |
| M-07 | Detrended counterfactual | Event lift for LBE F&B (FIFA) | Monthly Food/Games Net_Revenue per venue | CF = base 2025 × (1 + own Jan–May YoY trend); incremental = actual − CF; Games as a "clean" control if within ±10% of its own trend | Control check | Assumes the Jan–May trend holds | TRR FIFA Final |
| M-08 | Pre-trend-adjusted DiD | Same question, different estimator | Food and Games YoY in window vs pre-period | Lift = (Food − Games YoY in window) − (Food − Games YoY pre-period) | Games contamination check | Assumes Games unaffected | TRR FIFA Final |
| M-09 | Game-window vs placebo | Hour-level FIFA lift at Royalmount | Transactions by clock window | Same clock windows on non-game days by weekday → placebo distribution; Mann-Whitney test | Placebo test | Small n per match | TRR Royalmount FIFA.py |
| M-10 | Cannibalization | New venue impact on incumbents | Playdium visits and revenue | Transference in 3 tiers; visit shift vs two baselines; revenue detrended as in M-07 | Dual baseline | Identified customers only | PDM Vaughan Impact |
| M-11 | Break-even offer design | Free games per Scene+ gameband | Redemption history, member base | Break-even: points × $0.00942 ≥ games × $7; redeemer projection = OLS trend + calendar-month seasonal factor over 24 months | None | Every free game assumed a lost sale | LBE Scene Redemption Offer |
| M-12 | YoY projection | October gameband forecast | Oct 2024 and Oct 2025 by price tier and province | Oct-26 = Oct-25 × (1 + YoY); new tiers or markets get the national average YoY | None | Assumes growth repeats | Gameband (Scene Tiers) |
| M-13 | Liability projection | Outstanding and future free tickets | Snapshot, gifting | Holdings + forward credits + pending gifts (KPI-17) | Stacked-gift correction checked against subscription_changed events | Needs a $9.99 / $10.99 split after 2026-06-15 | Ticket Liability, Redemption Rate |
| M-14 | Offer post-mortem | Offer uptake, cost and sign-up mix | Events, snapshot, transactions, vouchers | Sign-ups by type; P12M visits; voucher opportunity cost (KPI-23); CPA/CPR with proportional redemption allocation (BR-057); pre/post behaviour and retention | Compared with prior-year windows | Older notebooks use non-standard upgrade logic | Black Friday, CPA/CPR series, Winter Holiday, Cravings, Q4 Offer |
| M-15 | Marketing mix modelling (broad rules only) | Media effect on EXH attendance | Weekly attendance target; spend named funnel_LOB_channel; controls for seasonality, holidays and trend | Spend grouped by funnel (Awareness, Consideration, Conversion) and LOB; adstock + saturation; chronological split with no shuffling; scalers fit on train only | Convergence and hold-out fit checks | Repo version is not final; parameters not recorded here | MMM notebooks |
| M-16 | Clustering | Film tiers | Film attendance and average line revenue | K-means, see section 10 | Silhouette and elbow method | Only 2 features | Film Classification |
| M-17 | Concession productivity | CPP and attach rate by population | Ticket and concession lines, theatre only, from 2024-01-01 to the start of the current month | Monthly CPP by CC / Scene / total; visit-level attach rate; share of CC concession visits using discount 2/3 | None | — | CPP Deep Dive series |

## 10. Model Specifications

Film classification is the only model specified here. The MMM in the repo is not the final version, so its parameters are left out; its broad rules are in M-15.

**Film Classification** (`Film Classification Model.ipynb`)

| Item | Specification |
| --- | --- |
| Unit | film_id (joined to film and genre, reported by title) |
| Features | total_attendance = SUM(Quantity); avg_ticket_revenue = AVG(Net_Revenue) per ticket *line*, not per ticket |
| Filters | Tickets with VisitDateId > 20221201; avg_ticket_revenue < 100; attendance > 0 |
| Scaling | MinMaxScaler |
| Model | Spark K-means, k-means\|\| initialisation, maxIter 100, tol 1e-6, seed 42 |
| k selection | k from 3 to 8 (the comment says 2 to 8); OPTIMAL_K = best silhouette, compared with the elbow method |
| Labels | Clusters mapped to the fixed list: Blockbuster, Mainstream, Limited Run, Niche, … |

## 11. Data Quality Rules

DQ-01 to DQ-11 are rules applied in code. DQ-12 onward are defects found in code; each needs a fix before the notebook is reused.

| ID | Area | Rule or finding | Source |
| --- | --- | --- | --- |
| DQ-01 | Nulls | Drop transactions with null CDE_ID for member-level metrics; LBE notebooks treat identified visits as a lower bound | MOM, F&B Trend, PDM |
| DQ-02 | Nulls | Missing counts after a left join are zero-filled; zero-visit years are added explicitly | MOM |
| DQ-03 | Nulls | NULL age, province or gender is excluded from the share denominator | MOM |
| DQ-04 | Refunds | Refunds are netted before counting (Net_Quantity > 0) | MOM, Wrapped |
| DQ-05 | Duplicates | Chargebee export deduped on 7 columns; demographics deduped by cde_id or customer_id; ticket-type dimension deduped on tickettypeskey before joining | MOM, Lifetime Savings |
| DQ-06 | Duplicates | Email opens deduped so any open across sends makes an opener; email → CDE join checked for fan-out | LBE Final SQL, 5+ Unused Email Test |
| DQ-07 | Test accounts | Excluded by the date-based recognitions_left ceiling (BR-025); the Chargebee artifact row AzqBunUAsPVB41nKT is dropped | MOM, Liability, Wrapped |
| DQ-08 | Outliers | Film average line revenue ≥ $100 excluded | Film Classification |
| DQ-09 | Stale fields | renewal_datetime is stale for stacked gifts; corrected with annual_current_term_end_datetime | Ticket Liability |
| DQ-10 | Event leakage | An upgrade completion can be logged as `subscription_created` (annual1); excluded via upgrades_took_effect | YTD SQL, CPA/CPR |
| DQ-11 | Renamed value | The status value was renamed from `Active_Annnual_Upgrade` to `Active_Annual_Upgrade`. Use the corrected spelling; older notebooks (Wrapped, Rewind, Winter Holiday, Cravings, Unknown 1) must be updated before reuse | Owner |
| DQ-12 | Defect | A timestamp compared with date strings using `between` drops the last day of the window | Black Friday, CPA/CPR |
| DQ-13 | Defect | Wrapped Scene+ points window runs to 2026-12-04; every other Wrapped window ends 2025-12-04 | Wrapped V5 |
| DQ-14 | Defect | Bucket "6 to 7" is coded as BETWEEN 5 AND 7 (label only; 5 is caught earlier) | CC Ticket Redemption Stats |
| DQ-15 | Defect | The email pivot orders row_number by CDE_ID itself, so which email becomes Primary is arbitrary | GS Dashboard |
| DQ-16 | Defect | Two back-to-back cells define "CC concession revenue" under the same alias (results equal by design, but duplicated code) | MOM concession trend |
| DQ-17 | Defect | F&B Trend says its category lists are "identical to the FIFA notebook" but they differ; F&B Trend matches the owner rule | F&B Trend vs TRR FIFA |
| DQ-18 | Defect | Code filtering member tickets with `isin(5, 6)` misses types 7 and 8 for dates from 2026-06-15; the GS dashboard labels only 0–6 | MOM, Savings, Rewind, Redemption Rate, GS Dashboard, CPA/CPR |
| DQ-19 | Defect | Event Cinema swaps the labels for types 5 and 6 by mistake (BR-018 is correct). Its CineClub_Discount = round(Mode_Ticket_Price − 9.99) × redemptions rounds to whole dollars and hard-codes 9.99 | Event Cinema |
| DQ-20 | Defect | Wrapped - Employees loads the employee CDE-ID CSV into df_employee_pd but builds df_employee from df_pricing_pd (copy-paste error), so the employee population never loads | Wrapped - Employees |
| DQ-21 | Defect | Both the correct and misspelled upgrade status values are filtered in different cells of the same notebook | Unknown 1 |
| DQ-22 | Defect | Ticket savings (GA − 9.99) × qty has no floor at 0 in the Former/Never-CC calculation, so it can go negative | Lifetime Savings Send Out |
| DQ-23 | Defect | Legacy table names `marketing.subscriber_profile_snapshot` / `subscription_events` (pre-`mkt_` rename) | CineClub Rewind (SQL), Black Friday Deal 2025 |

## 12. Rule Conflict Register

One conflict remains open, pending a revised script from the owner. The owner resolved the rest on 2026-09-27; their standards now live in the rules and KPIs above.

| ID | Rule | Implementation A | Implementation B | Impact | Next step |
| --- | --- | --- | --- | --- | --- |
| CF-15 | Dunning look-back | Monthly: 15 days | Annual: 14 days | Small count differences | Owner will provide revised dunning and sign-up scripts |

## 13. Assumptions

**Explicit** (stated in code comments, markdown or post-mortem docs)

| ID | Assumption | Source |
| --- | --- | --- |
| A-01 | A redeemed Scene+ point is worth $0.00942; every free LBE game is a lost sale worth up to $7 | LBE Scene Redemption Offer |
| A-02 | Large popcorn $9.49 price / $0.67 cost; nachos $7.99 / $3.42; all popcorn assumed large for voucher costing | Exclusive Offers post-mortem |
| A-03 | Members earn 1 free ticket per month, from the month after activation to 1 month before renewal | Ticket Liability |
| A-04 | CineClub launched Aug 2021; maximum legitimate balance grows by 1 per month since launch | Owner, Ticket Liability |
| A-05 | Chargebee auto-stacks a gift only within 12 months before renewal | Ticket Liability |
| A-06 | The Jan–May YoY trend carries into the event window (FIFA, PDM) | TRR FIFA, PDM Vaughan |
| A-07 | Games revenue is unaffected by FIFA, so it can serve as the control; within ±10% counts as "clean" | TRR FIFA |
| A-08 | The Oct 2024 → Oct 2025 growth repeats in Oct 2026; new tiers or markets take the national average YoY | Gameband (Scene Tiers) |
| A-09 | Identified (CDE) visits are a lower bound on footfall | F&B Trend, PDM Vaughan |
| A-10 | Ticket savings = GA price − ticket value ($9.99 for types 5/6, $10.99 for 7/8); GA defaults to the ticket value (zero saving) when missing | Lifetime Savings; owner-confirmed values |
| A-11 | Vouchers are passed along, so direct CDE matching undercounts redemptions; proportional allocation by segment size corrects for it | CPA/CPR Analysis 1 |

**Inferred** (implied by the code but not stated)

| ID | Assumption | Basis |
| --- | --- | --- |
| A-12 | The ÷ 4 discount formula assumes Net_Spend is already 80% of list price (20% off) | Consistent with the "25% of discounted revenue" comment in Savings |
| A-13 | Non-openers are a fair control for openers | Cohort design in LBE Final SQL |
| A-14 | A customer with exactly one created event has never been a member before | First-time logic |
| A-15 | `current_date()` is close enough to the report date for age | Age calculation |
| A-16 | A missing RFM match means the member is Dormant | COALESCE in YTD SQL |
| A-17 | Visits at two locations on the same day count as two visits (owner-confirmed key); the MOM history count still collapses them to one | Mixed visit keys in older notebooks |

## 14. Open Questions

- [ ] Revised dunning and sign-up scripts (CF-15) — to be provided by the owner.
- [ ] Which GA price source is standard for savings: General Admit Prices.xlsx or Circuit Pricing.xlsx?
