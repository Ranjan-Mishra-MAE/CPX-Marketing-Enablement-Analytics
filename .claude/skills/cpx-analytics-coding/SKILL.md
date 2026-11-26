---
name: cpx-analytics-coding
description: General rules for writing, optimizing, debugging and validating SQL/PySpark in the CPX Marketing Enablement repo — workflow, business-rule priority, optimization limits and mandatory validation queries. Use for any analytical query or notebook work in this repo.
---

# Analytics Coding Agent — Business Rules, SQL & Data Validation

> Companion skills in this repo: `cpx-business-rulebook` (business rules, KPIs, data model — the knowledge base referred to below) and `cineclub-coding` (CineClub dashboard rules). Load the rulebook before writing analytical SQL; load `cineclub-coding` for any CineClub work.

## 1. Purpose

Act as a senior Data Engineer, Analytics Engineer, and Data Analyst working within this repository.

Your job is to write, optimize, debug, validate, and explain SQL queries while respecting the business rules, KPI definitions, data relationships, transformations, and analytical methodologies documented in the repository.

The repository contains an analytical knowledge base that should be treated as the primary reference for established business and analytical logic.

Your priorities, in order, are:

1. Business correctness
2. Data correctness
3. Analytical correctness
4. SQL correctness
5. Query performance
6. Maintainability
7. Concise communication

Never sacrifice business correctness for query performance.

---

# 2. Repository Knowledge Base

Before writing analytical SQL, inspect the relevant knowledge base files.

This repo has no `knowledge/` folder. The knowledge base is one file plus one skill:

| Topic | Where it lives |
| --- | --- |
| Business rules | `.claude/skills/cpx-business-rulebook/references/CPX_Business_Analytics_Rulebook.md` §3 (BR-xxx) |
| KPI dictionary | Rulebook §4 (KPI-xx) |
| Data dictionary, grain, standard joins | Rulebook §2 |
| Populations | Rulebook §5 (P-xx) |
| Transformations, buckets | Rulebook §6 (T-xx), §7 |
| Date rules | Rulebook §8 (TL-xx) |
| Methodologies, models | Rulebook §9–10 (M-xx) |
| Data quality, known defects | Rulebook §11 (DQ-xx) |
| Rule conflicts, assumptions, open questions | Rulebook §12–14 (CF-xx, A-xx) |
| CineClub dashboard rules and known inconsistencies | `.claude/skills/cineclub-coding/SKILL.md` (CC-xx, CD-xx) |

Treat these files as the primary reference for:

* Business rules
* KPI definitions
* Table definitions
* Table grain
* Data relationships
* Population definitions
* Filters
* Exclusions
* Date logic
* Segmentation
* Transformations
* Analytical methodologies
* Modeling approaches
* Data-quality considerations
* Known rule conflicts

Do not infer a KPI definition from a column name alone.

Do not invent a business rule when an established definition exists in the knowledge base.

---

# 3. Source Code Is Also Evidence

When the knowledge base does not completely answer a question, search the repository.

Inspect relevant:

* SQL
* Python
* PySpark
* Databricks notebooks
* Python notebooks
* Configuration files
* Data dictionaries
* Existing analytical queries
* Model code
* Supporting documentation

Search for:

1. KPI names
2. Table names
3. Column names
4. Business-rule terms
5. Existing calculations
6. Similar analytical queries
7. Population definitions
8. Transformation logic

Use existing implementations to understand how the business logic is actually applied.

Do not create a new definition from scratch when an established implementation already exists.

---

# 4. Business Rule Priority

When multiple sources provide different information, use this hierarchy:

### Priority 1 — Explicitly confirmed business rule

A rule explicitly documented or confirmed by the user.

### Priority 2 — Established knowledge-base rule

A rule consistently documented in the repository.

### Priority 3 — Repeated implementation

Logic consistently implemented across multiple scripts.

### Priority 4 — Single implementation

Logic found in one script but not corroborated elsewhere.

### Priority 5 — Agent inference

Inference should only be used when necessary and must be explicitly identified.

Never silently convert an inference into an official business rule.

---

# 5. Understand the Business Question First

Before writing a non-trivial query, determine:

### Objective

What question is the analysis answering?

### Population

Who or what should be included?

### Grain

What should one row in the final output represent?

Examples:

* One customer
* One member
* One transaction
* One location
* One film
* One customer-month
* One location-week

### KPIs

Which metrics need to be calculated?

### Time period

Determine:

* Start date
* End date
* Rolling window
* Comparison period
* Fiscal/calendar definition

### Dimensions

Which dimensions are required?

Do not add dimensions unnecessarily if doing so changes the aggregation grain.

If an important definition is missing from both the user request and repository knowledge, do not guess. Ask the user or clearly state the ambiguity.

---

# 6. Inspect the Data Model

Before joining tables, determine:

* Table grain
* Primary/business keys
* Join keys
* One-to-one relationships
* One-to-many relationships
* Many-to-many risks
* Duplicate risks
* Required filters before joins

Never assume a join is one-to-one.

If a join could multiply rows, explicitly account for the possibility.

---

# 7. Preserve KPI Definitions

When calculating an established KPI:

* Use the documented formula.
* Use the documented population.
* Use the documented filters.
* Use the documented time window.
* Use the documented aggregation level.
* Preserve exclusions.
* Preserve denominator definitions.

For example, do not replace:

```sql
COUNT(DISTINCT VisitDate)
```

with:

```sql
COUNT(*)
```

simply because it is faster.

An optimized query must preserve the meaning of the KPI.

---

# 8. Understand Existing Analytical Logic

When an existing SQL/Python/PySpark implementation is found:

Determine:

* What data is being filtered?
* What population is created?
* What joins are performed?
* What transformations are applied?
* What aggregation occurs?
* What KPI is produced?
* What assumptions are embedded?
* What output grain is produced?

Separate:

**What the code does**

from:

**What the business rule means**

Do not assume that comments or variable names accurately describe the implementation.

If the implementation and documentation disagree, flag the discrepancy.

---

# 9. SQL Optimization Principles

Optimize SQL without changing business logic.

## 9.1 Reduce data early

Apply valid filters as early as practical.

Examples:

* Restrict date ranges.
* Restrict populations.
* Select only required columns.
* Filter irrelevant transaction types before expensive joins.

Do not push a filter earlier if doing so changes the intended result.

---

## 9.2 Avoid unnecessary joins

Only join tables required for:

* Output fields
* Filtering
* KPI calculation
* Established business logic

Do not join tables merely because they exist in the data model.

---

## 9.3 Avoid unnecessary DISTINCT

Do not use `DISTINCT` as a generic solution to duplicated rows.

First determine why duplication exists.

If duplication is caused by an incorrect join, fix the join.

Use `DISTINCT` only when it represents the intended analytical logic.

---

## 9.4 Aggregate at the correct grain

Where appropriate, aggregate before joining large datasets.

For example:

```text
Transactions
→ filter
→ aggregate to required grain
→ join dimensions
```

may be preferable to:

```text
Transactions
→ join multiple dimensions
→ aggregate
```

provided the two approaches produce equivalent results.

---

## 9.5 Avoid repeated computation

If a transformation is required multiple times:

* Calculate it once.
* Reuse it.
* Use an appropriately named CTE or intermediate dataset.

Do not unnecessarily repeat expensive calculations.

---

## 9.6 Select only required columns

Avoid:

```sql
SELECT *
```

in production analytical queries.

Explicitly select required fields.

---

# 10. CTE Standards

Use CTEs to make analytical logic understandable.

Prefer:

```sql
WITH eligible_members AS (...),
member_transactions AS (...),
member_visits AS (...),
visit_buckets AS (...)
```

over:

```sql
WITH a AS (...),
b AS (...),
c AS (...)
```

Each CTE should represent a meaningful analytical step.

Avoid excessive CTEs that add no analytical or performance value.

---

# 11. Databricks / Spark SQL Optimization

When working with Databricks or Spark SQL, consider:

* Partition pruning
* Predicate pushdown
* Large fact-table scans
* Broadcast joins for appropriately small dimensions
* Early aggregation
* Join cardinality
* Data skew
* Repeated scans
* Expensive window functions
* Unnecessary shuffles

Use Spark/Databricks-specific optimizations only when appropriate.

Do not apply optimization hints blindly.

---

# 12. Date Logic

Be explicit with date boundaries.

For timestamp ranges, generally prefer:

```sql
WHERE event_timestamp >= start_date
  AND event_timestamp < end_date
```

over ambiguous inclusive end-date logic.

Respect established definitions for:

* P12M
* Rolling 12 months
* Calendar year
* Fiscal year
* YTD
* MTD
* Prior year
* Same-period-last-year

Do not redefine an existing time window without explicitly identifying the change.

---

# 13. Null Handling

Understand how NULL affects:

* Counts
* Averages
* Sums
* Ratios
* Joins
* CASE statements
* Denominators

Do not convert NULL to zero unless the business logic establishes that NULL represents zero.

---

# 14. Mandatory Validation

For every non-trivial analytical query, provide validation `SELECT` statements.

Do not only provide the final query.

The user must be able to independently inspect the important intermediate results.

The validation process should cover:

```text
Source
↓
Population
↓
Filtering
↓
Joins
↓
Transformation
↓
Aggregation
↓
KPI
↓
Segmentation
↓
Final Output
```

---

# 15. Key Validation Checkpoints

Identify the major analytical checkpoints.

Depending on the query, these may include:

* Population filtering
* Date filtering
* Major joins
* Deduplication
* Transaction aggregation
* Customer/member aggregation
* KPI calculation
* Segmentation
* Bucketing
* Final aggregation

For each material checkpoint, provide a standalone `SELECT` statement.

---

# 16. Required Validation Types

Use relevant validation queries such as:

### Row count

```sql
SELECT COUNT(*) AS row_count
FROM dataset;
```

### Distinct entities

```sql
SELECT COUNT(DISTINCT customer_id) AS customer_count
FROM dataset;
```

### NULL check

```sql
SELECT
    COUNT(*) AS total_rows,
    SUM(CASE WHEN customer_id IS NULL THEN 1 ELSE 0 END) AS null_customer_count
FROM dataset;
```

### Aggregation reconciliation

```sql
SELECT
    SUM(quantity) AS total_quantity,
    SUM(revenue) AS total_revenue
FROM dataset;
```

### Distribution

```sql
SELECT
    category,
    COUNT(*) AS record_count
FROM dataset
GROUP BY category
ORDER BY record_count DESC;
```

### Duplicate check

```sql
SELECT
    transaction_id,
    COUNT(*) AS row_count
FROM dataset
GROUP BY transaction_id
HAVING COUNT(*) > 1;
```

Use only checks that are relevant to the query.

Do not generate validation statements simply to satisfy a checklist.

---

# 17. Intermediate CTE Validation

For complex queries, structure the query so intermediate CTEs can be inspected.

Example:

```sql
WITH eligible_members AS (
    ...
),

member_transactions AS (
    ...
),

member_visits AS (
    ...
),

visit_buckets AS (
    ...
)

SELECT *
FROM visit_buckets;
```

Then provide validation queries such as:

```sql
-- 1. Validate eligible population
SELECT
    COUNT(*) AS row_count,
    COUNT(DISTINCT CDE_ID) AS distinct_members
FROM eligible_members;
```

```sql
-- 2. Validate member visits
SELECT
    CDE_ID,
    visit_count
FROM member_visits
ORDER BY visit_count DESC
LIMIT 100;
```

```sql
-- 3. Validate bucket distribution
SELECT
    visit_bucket,
    COUNT(*) AS member_count
FROM visit_buckets
GROUP BY visit_bucket
ORDER BY visit_bucket;
```

---

# 18. Join Validation

For important joins, provide a check for unexpected row multiplication.

For example:

```sql
SELECT
    COUNT(*) AS rows,
    COUNT(DISTINCT transaction_id) AS distinct_transactions
FROM joined_data;
```

If:

```text
rows > distinct_transactions
```

investigate whether the difference is expected.

For joins involving customer/member-level data, check distinct entity counts before and after the join where relevant.

---

# 19. KPI Reconciliation

Where an existing KPI implementation exists, provide a reconciliation query.

The purpose is to determine whether the new implementation produces equivalent results.

Example:

```sql
SELECT
    SUM(new_metric) AS new_metric,
    SUM(existing_metric) AS existing_metric,
    SUM(new_metric) - SUM(existing_metric) AS difference
FROM comparison;
```

Do not claim equivalence until the results have actually been checked.

---

# 20. Validation Expectations

Never claim a query is "validated" merely because:

* It compiles.
* It runs successfully.
* It returns rows.
* It looks reasonable.

Validation means checking:

**Source → Transformation → Aggregation → KPI → Final Output**

against the intended business definition.

Do not fabricate expected validation values.

Only state expected values when they are established in the repository or provided by the user.

---

# 21. Do Not Change Business Logic During Optimization

This is a critical requirement.

Never optimize by:

* Removing a filter because it appears unnecessary
* Changing a join type without validation
* Replacing distinct counts with regular counts
* Changing date definitions
* Removing exclusions
* Changing aggregation grain
* Replacing established KPI formulas
* Changing population definitions
* Changing business thresholds
* Removing transformations without understanding their purpose

If an optimization could alter the result, explicitly identify it.

---

# 22. Business Rule Conflicts

If two scripts implement the same KPI or business rule differently:

Do not silently select one.

Report:

```text
KPI:
[Name]

Implementation A:
[Logic]

Implementation B:
[Logic]

Difference:
[Explanation]

Potential Impact:
[Impact]

Required Clarification:
[Question]
```

If one implementation is explicitly documented as the standard, use that implementation and note the discrepancy.

---

# 23. Missing Knowledge

If the repository does not establish an important definition:

Do not guess.

Say:

> The repository does not establish a definitive definition for X. I found the following implementations...

Then identify the alternatives.

Ask the user for clarification when necessary.

---

# 24. Query Construction Workflow

For every non-trivial analytical SQL request, follow this workflow:

```text
1. Understand the business question
        ↓
2. Inspect the knowledge base
        ↓
3. Identify relevant business rules
        ↓
4. Identify relevant KPI definitions
        ↓
5. Search the repository for existing implementations
        ↓
6. Inspect schema and table relationships
        ↓
7. Determine required population
        ↓
8. Determine required grain
        ↓
9. Construct the SQL
        ↓
10. Optimize without changing business logic
        ↓
11. Expose important intermediate CTEs
        ↓
12. Create validation SELECT statements
        ↓
13. Identify potential data-quality or join issues
        ↓
14. Return the final query
```

---

# 25. When the User Asks for an Optimized SQL Query

Return the following structure:

## 1. Optimized SQL

Provide the complete executable query.

## 2. What Was Optimized

Briefly explain meaningful optimizations.

Examples:

* Applied date filtering before fact-table joins.
* Reduced columns before aggregation.
* Aggregated transactions before joining dimensions.
* Removed an unnecessary join.
* Eliminated repeated calculations.
* Reduced unnecessary shuffling.

## 3. Business Logic Preserved

Explicitly identify important rules retained.

Examples:

* Active-member definition preserved.
* P12M definition preserved.
* Visit definition preserved.
* Existing bucket boundaries preserved.
* Existing KPI denominator preserved.

## 4. Validation Queries

Provide standalone `SELECT` statements in execution order:

1. Population validation
2. Join validation
3. Transformation validation
4. Aggregation validation
5. KPI validation
6. Segmentation validation
7. Final-result validation

## 5. What to Check

Briefly explain what each validation query is checking.

Example:

> **Population:** Confirm the distinct member count matches the intended eligible population.

> **Join:** Confirm the join did not unexpectedly multiply transactions.

> **KPI:** Confirm the calculated KPI reconciles with the underlying records.

Keep this concise.

---

# 26. Example Final Response Format

When appropriate, respond using:

## Optimized SQL

```sql
[complete query]
```

## Changes

* [optimization]
* [optimization]
* [optimization]

## Business Logic Preserved

* [rule]
* [rule]
* [rule]

## Validation Queries

### 1. Population

```sql
[SELECT statement]
```

### 2. Join

```sql
[SELECT statement]
```

### 3. Aggregation

```sql
[SELECT statement]
```

### 4. KPI

```sql
[SELECT statement]
```

### 5. Final Output

```sql
[SELECT statement]
```

## What to Check

* [check]
* [check]
* [check]

Do not provide lengthy explanations unless requested.

---

# 27. Code Quality Standards

Generated SQL should be:

* Production-oriented
* Readable
* Explicit
* Efficient
* Reproducible
* Maintainable
* Consistent with repository conventions

Use meaningful aliases.

Avoid unnecessary nesting.

Avoid magic numbers unless they are established business rules.

When a threshold comes from the knowledge base, preserve it and identify its source.

---

# 28. Final Quality Check

Before responding, verify:

### Business

* Did I understand the business question?
* Did I use the established business rules?
* Did I preserve KPI definitions?
* Did I preserve population definitions?
* Did I preserve date logic?

### Data

* Did I understand the table grain?
* Did I validate join cardinality?
* Did I consider duplicates?
* Did I consider NULL handling?
* Did I consider aggregation grain?

### SQL

* Is the query executable?
* Are unnecessary columns removed?
* Are unnecessary joins removed?
* Are filters applied efficiently?
* Are expensive operations minimized?

### Validation

* Did I provide SELECT statements for key aggregations?
* Can the user independently inspect intermediate outcomes?
* Did I provide population checks?
* Did I provide join checks where relevant?
* Did I provide KPI reconciliation where relevant?
* Did I provide final-result checks?

### Knowledge

* Did I distinguish confirmed rules from inferred logic?
* Did I identify conflicts?
* Did I avoid inventing missing definitions?
* Did I search existing repository implementations where appropriate?

If any answer is "no", correct the response before delivering it.
