---
name: cpx-business-rulebook
description: CPX business and analytics rulebook — data model, standard joins, business rules (BR-xxx), KPI definitions, populations, transformations, date rules, data-quality defects and open conflicts for EXH, LBE, CineClub and Scene+. Use before writing or reviewing any analytical SQL or notebook in this repo.
---

# CPX Business & Analytics Rulebook

The full rulebook is in `references/CPX_Business_Analytics_Rulebook.md`. Read the sections relevant to the task before writing SQL.

| Need | Section |
| --- | --- |
| Tables, grain, keys, standard joins (BR-001–009) | §2 Data Model |
| Business rules (BR-010+) with type and confidence | §3 |
| KPI definitions (KPI-01+) and non-standard variants | §4 |
| Population definitions (P-xx) | §5 |
| Transformations (T-xx) | §6 |
| Age, visit, RFM, format, gameband buckets | §7 |
| Date windows and time rules (TL-xx) | §8 |
| Methodologies (M-xx), film model | §9–10 |
| Data-quality rules and known defects (DQ-xx) | §11 |
| Open conflicts (CF-xx), assumptions, open questions | §12–14 |

How to apply it:

- Rules typed **Owner** override everything except what the owner says in chat.
- When the owner supplies script logic for an analysis, follow it exactly; flag differences from the rulebook, don't apply them.
- Before reusing an older notebook, check §11 for defects listed against it.
- Never invent a rule the rulebook doesn't establish — say it's missing and ask.
- Open conflicts (§12) are not settled: CF-15 (dunning look-back) and CF-16 (three annual renewal-rate definitions). Ask which to use before computing either.
- The Scene vs Connect script behind BR-061 / KPI-29 was supplied in chat and is not in the repo; the rulebook entries are its only record.
- For CineClub dashboard work, also load the `cineclub-coding` skill; for workflow and validation format, `cpx-analytics-coding`.
