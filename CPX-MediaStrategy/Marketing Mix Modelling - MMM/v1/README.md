# Data Requirements

## Adobe Analytics Inputs
- Website visits (Cineplex, The Rec Room, Playdium)
## Marketing Department Inputs
- SEM spend
- Meta spend
- Datorama spend (social and programmatic)
- Email sends
- PR spend
- Mass offer calendar
## SQL Server Production Inputs
- Attendance and revenue
- Cineclub signups
- Weekly film releases

# Data Dictionary

## Exhibition
| **Variable Name** | **Data Type** | **Description** |
|--|--|--|
| `week` | datetime64 | Calendar week starting Friday |
| `summer` | int64 | Indicator of whether the month is July/August or not|
| `holiday` | int64 | Indicator of whether the week includes a federal holiday |
| `releases` | int64 | Number of films released in the week |
| `mass_exh` | int64 | Marketing spend on Exhibition LOB mass offers |
| `mass_film` | int64 | Marketing spend on Film LOB mass offers |
| `mass_store` | int64 | Marketing spend on Cineplex Store LOB mass offers |
| `sem_[lob]` | float64 | Media spend on the SEM channel within `lob` |
| `meta_[lob]` | float64 | Media spend on the Meta channel within `lob` |
| `programmatic_[lob]` | float64 | Media spend on the Programmatic channel within `lob` |
| `social_[lob]` | float64 | Media spend on the Social channel within `lob` |
| `pr_spend` | float64 | Marketing spend on PR campaigns |
| `website_visits` | int64 | Number of views received across all websites (Cineplex, Rec Room, Playdium) |
| `cineclub_signups` | int64 | Number of Cineclub signups received across all plans (monthly, annual, gift) |
| `attendance` | int64 | Box office attendance across all Cineplex locations |
---
## LBE
| **Variable Name** | **Data Type** | **Description** |
|--|--|--|
| `week` | datetime64 | Calendar week starting Friday |
| `summer` | int64 | Indicator of whether the month is July/August or not|
| `holiday` | int64 | Indicator of whether the week includes a federal holiday |
| `canada` | float64 | Media spend on nationwide campaigns |
| `hpl` | float64 | Media spend on high-performing-location campaigns |
| `hpl_lpl` | float64 | Media spend on a combination of high- and low-performing-location campaigns |
| `lpl` | float64 | Media spend on low-performing-location campaigns |
| `nl` | float64 | Media spend on new location campaigns |
| `pdm` | float64 | Media spend on Playdium campaigns |
| `pr_nl` | float64 | Marketing spend on PR new location campaigns |
| `pr_pdm` | float64 | Marketing spend on PR Playdium campaigns |
| `transactions` | int64| Number of customer transactions made across all Rec Room and Playdium locations |
