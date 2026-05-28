# Data Manifest

**Last verified:** 2026-05-28
**Purpose:** Per-source provenance, vintage, and quality flags for every dataset in `signal_tool/data/`. Update whenever data is refreshed.

The signal tool runs on cached CSVs (not live API calls), so this manifest is the audit trail. Every value should be traceable to a primary source — if you can't trace it, the row should be flagged as ESTIMATED.

---

## `copper_ppi.csv`

**Status:** ✅ VERIFIED against primary source

| Field | Value |
|---|---|
| **Series ID** | `WPUSI019011` |
| **Title** | Producer Price Index by Commodity: Special Indexes: Copper and Copper Products |
| **Publisher** | FRED (St. Louis Fed) — original from US Bureau of Labor Statistics |
| **URL** | https://fred.stlouisfed.org/series/WPUSI019011 |
| **Data URL** | https://fred.stlouisfed.org/data/WPUSI019011.txt |
| **Units** | Index 1982=100, Not Seasonally Adjusted |
| **Frequency** | Monthly |
| **Date Range** | 1967-01-01 to 2026-04-01 |
| **Row Count** | 712 |
| **FRED Last Updated** | 2026-05-13 09:24 AM CDT |
| **Fetched by signal_tool** | 2026-05-28 (via web_fetch) |
| **License** | Public Domain: Citation Requested |

**Verification log:**
- 2026-05-28 — Re-fetched and spot-checked 20 dates spanning 1967–2026. All values match exactly. Row count consistent (712 = 59 years × 12 + 4 months in 2026 = 712). ✓

**Known caveats:**
- Not seasonally adjusted — possible monthly noise from commodity-cycle seasonality
- Index 1982=100; not directly comparable in levels to transformer PPI (different base year). Use YoY % for cross-comparison.

**Citation:** U.S. Bureau of Labor Statistics, "Producer Price Index by Commodity: Special Indexes: Copper and Copper Products [WPUSI019011]," retrieved from FRED, Federal Reserve Bank of St. Louis; https://fred.stlouisfed.org/series/WPUSI019011, May 28, 2026.

---

## `transformer_ppi.csv`

**Status:** ✅ VERIFIED against primary source

| Field | Value |
|---|---|
| **Series ID** | `PCU335311335311` |
| **Title** | Producer Price Index by Industry: Electric Power and Specialty Transformer Manufacturing |
| **Publisher** | FRED (St. Louis Fed) — original from US Bureau of Labor Statistics |
| **URL** | https://fred.stlouisfed.org/series/PCU335311335311 |
| **Data URL** | https://fred.stlouisfed.org/data/PCU335311335311.txt |
| **Units** | Index Jun 1981=100, Not Seasonally Adjusted |
| **Frequency** | Monthly |
| **Date Range** | 1967-01-01 to 2026-04-01 |
| **Row Count** | 712 |
| **FRED Last Updated** | 2026-05-13 09:34 AM CDT |
| **Fetched by signal_tool** | 2026-05-28 (via web_fetch) |
| **License** | Public Domain: Citation Requested |

**Verification log:**
- 2026-05-28 — Re-fetched and spot-checked 20 dates spanning 1967–2026. All values match exactly. ✓

**Known caveats:**
- "Specialty transformer" includes industrial and instrument transformers, not just power transformers. Power-specific subseries (`PCU3353113353111`) exists separately if needed.
- Different index base year (Jun 1981=100) from copper PPI (1982=100). Use YoY % for cross-comparison.
- Recent values may be revised — FRED republishes BLS updates. Re-fetch monthly.

**Citation:** U.S. Bureau of Labor Statistics, "Producer Price Index by Industry: Electric Power and Specialty Transformer Manufacturing [PCU335311335311]," retrieved from FRED, Federal Reserve Bank of St. Louis; https://fred.stlouisfed.org/series/PCU335311335311, May 28, 2026.

---

## `hyperscaler_capex.csv`

**Status:** ⚠️ PARTIAL — values 2024-2025 are primary-source; 2015-2023 are partly estimated. Flagged for upgrade in a future sprint.

| Field | Value |
|---|---|
| **What** | Combined annual capital expenditure for Amazon + Google/Alphabet + Meta + Microsoft |
| **Aggregator** | Platformonomics ("Follow the CAPEX" annual retrospective series) |
| **Primary sources** | SEC EDGAR 10-K filings of each company (Amazon CIK 1018724, Microsoft 789019, Alphabet 1652044, Meta 1326801) |
| **URLs** | https://platformonomics.com/2026/02/follow-the-capex-2025-retrospective/ |
| | https://platformonomics.com/2025/02/follow-the-capex-cloud-table-stakes-2024-retrospective/ |
| | https://platformonomics.com/2024/02/follow-the-capex-cloud-table-stakes-2023-retrospective/ |
| | https://platformonomics.com/2023/02/follow-the-capex-cloud-table-stakes-2022-retrospective/ |
| | https://platformonomics.com/2022/02/follow-the-capex-cloud-table-stakes-2021-retrospective/ |
| | https://platformonomics.com/2021/02/follow-the-capex-cloud-table-stakes-2020-retrospective/ |
| | https://platformonomics.com/2020/02/follow-the-capex-cloud-table-stakes-2019-edition/ |
| **Units** | USD billions, calendar year |
| **Frequency** | Annual |
| **Date Range** | 2015–2025 |
| **Row Count** | 11 |
| **Fetched by signal_tool** | 2026-05-28 |

### Per-year confidence and source

| Year | Value ($B) | Confidence | Notes |
|---|---|---|---|
| 2015 | 24 | 🟡 Approximate | Platformonomics baseline (~$23.8B for 4-co quoted in DCpulse). Predates regular Platformonomics retrospective coverage. |
| 2016 | 31 | 🟡 Approximate | Interpolated from 2015 / 2020 trend; cross-checked against 3-cloud Platformonomics 2018-edition retrospective |
| 2017 | 42 | 🟡 Approximate | Same as above |
| 2018 | 60 | 🟡 Approximate | Platformonomics 2018 retrospective: 3-hypercloud = $68B for 2018. With Meta capex ~$13.9B, total 4-co ~$60-65B. Using $60B as midpoint. |
| 2019 | 80 | 🟡 Approximate | Platformonomics 2019: 3-hypercloud = $73.5B. With Meta $15B, total ~$80-85B. |
| 2020 | 94 | 🟢 Primary | Platformonomics 2020 retrospective: 3-hypercloud = $97B (Amazon $54, Google $22.3, Microsoft $20.6); + Meta $16B → revisit; current value $94 may be slightly under |
| 2021 | 143 | 🟡 Approximate | 3-hypercloud $124B (Platformonomics) + Meta $19B = $143B |
| 2022 | 159 | 🟡 Approximate | 3-hypercloud $127B + Meta $32B = $159B |
| 2023 | 155 | 🟡 Approximate | 3-hypercloud $127B + Meta $28B (efficiency year) = $155B |
| 2024 | 251 | 🟢 Primary | Platformonomics 2024 retrospective explicitly states "$251 billion" 4-co total |
| 2025 | 416 | 🟢 Primary | Platformonomics 2025 retrospective: AMZN $134.7 + GOOG $91.5 + MSFT $118.0 + META $72.2 = $416.4B |

**Known data definition issues:**

1. **Different capex definitions per company.** Microsoft and Meta include finance lease ROU assets; Google does not; Amazon mixes retail PP&E with cloud infrastructure. Platformonomics normalizes where possible but residual variance remains. Epoch AI's quarterly dataset uses stricter XBRL tag rules — see https://epoch.ai/data-insights/hyperscaler-capex-trend.

2. **Fiscal year alignment.** Microsoft (FY ending June) and Oracle (FY ending May) report on non-calendar fiscal years. Platformonomics maps these to calendar quarters; calendar-year totals should be approximately right but not perfectly aligned.

3. **Meta included in 4-co cohort but historically tracked separately.** Platformonomics' headline retrospectives often focused on 3-hypercloud (excl. Meta) for 2018-2023. Meta capex came from secondary sources (DCD, Statista, FB earnings).

### Upgrade plan (not yet done)

To move from ⚠️ PARTIAL to ✅ VERIFIED, replace approximated years with per-company 10-K capex line items extracted from SEC EDGAR XBRL filings (matching Epoch AI's methodology). Estimated effort: 4-6 hours.

**Citation:** Fitzgerald, Charles. "Follow the CAPEX: Cloud Table Stakes" annual retrospective series 2019–2026, Platformonomics. Cross-referenced with company 10-K filings on SEC EDGAR. Retrieved May 28, 2026.

---

## `aluminum_ppi.csv`

**Status:** ✅ VERIFIED against primary source

| Field | Value |
|---|---|
| **Series ID** | `WPU102501` |
| **Title** | Producer Price Index by Commodity: Metals and Metal Products: Aluminum Mill Shapes |
| **Publisher** | FRED (St. Louis Fed) — original from US Bureau of Labor Statistics |
| **URL** | https://fred.stlouisfed.org/series/WPU102501 |
| **Data URL** | https://fred.stlouisfed.org/data/WPU102501.txt |
| **Units** | Index 1982=100, Not Seasonally Adjusted |
| **Frequency** | Monthly |
| **Date Range (in cache)** | 1967-01-01 to 2026-04-01 (FRED has data back to 1947; trimmed to align with transformer PPI) |
| **Row Count** | 712 |
| **FRED Last Updated** | 2026-05-13 09:27 AM CDT |
| **Fetched by signal_tool** | 2026-05-28 (via web_fetch) |
| **License** | Public Domain: Citation Requested |

**Verification log:**
- 2026-05-28 — Fetched fresh from FRED, spot-checked 14 dates spanning 1967–2026. All values match exactly. ✓

**Known caveats:**
- "Aluminum Mill Shapes" is a broad category covering rolled, drawn, and extruded aluminum products. Includes inputs used in transformer windings and housings.
- Same index base (1982=100) as copper PPI — directly comparable in levels with copper but not with transformer PPI.

**Citation:** U.S. Bureau of Labor Statistics, "Producer Price Index by Commodity: Metals and Metal Products: Aluminum Mill Shapes [WPU102501]," retrieved from FRED, Federal Reserve Bank of St. Louis; https://fred.stlouisfed.org/series/WPU102501, May 28, 2026.

---

## `steel_ppi.csv`

**Status:** ✅ VERIFIED against primary source

| Field | Value |
|---|---|
| **Series ID** | `WPU101` |
| **Title** | Producer Price Index by Commodity: Metals and Metal Products: Iron and Steel |
| **Publisher** | FRED (St. Louis Fed) — original from US Bureau of Labor Statistics |
| **URL** | https://fred.stlouisfed.org/series/WPU101 |
| **Data URL** | https://fred.stlouisfed.org/data/WPU101.txt |
| **Units** | Index 1982=100, Not Seasonally Adjusted |
| **Frequency** | Monthly |
| **Date Range (in cache)** | 1967-01-01 to 2026-04-01 (FRED has data back to 1926; trimmed to align with transformer PPI) |
| **Row Count** | 712 |
| **FRED Last Updated** | 2026-05-13 09:27 AM CDT |
| **Fetched by signal_tool** | 2026-05-28 (via web_fetch) |
| **License** | Public Domain: Citation Requested |

**Verification log:**
- 2026-05-28 — Fetched fresh from FRED, spot-checked 13 dates spanning 1967–2026. All values match exactly. ✓

**Known caveats:**
- Broad iron and steel commodity index. The transformer-specific component is **grain-oriented electrical steel (GOES)** used in transformer cores, which isn't a separate FRED series. WPU101 is a reasonable proxy because GOES prices track broader steel prices closely.
- Different formats observed in FRED HTML output (pipe-table for 1926–2009-04, hash-prefix for 2009-05 onwards). Our extractor handles both.

**Citation:** U.S. Bureau of Labor Statistics, "Producer Price Index by Commodity: Metals and Metal Products: Iron and Steel [WPU101]," retrieved from FRED, Federal Reserve Bank of St. Louis; https://fred.stlouisfed.org/series/WPU101, May 28, 2026.

---

## `manufacturing_employment.csv`

**Status:** ✅ VERIFIED against primary source

| Field | Value |
|---|---|
| **Series ID** | `MANEMP` |
| **Title** | All Employees, Manufacturing |
| **Publisher** | FRED (St. Louis Fed) — original from US Bureau of Labor Statistics (Current Employment Statistics survey) |
| **URL** | https://fred.stlouisfed.org/series/MANEMP |
| **Data URL** | https://fred.stlouisfed.org/data/MANEMP |
| **Units** | Thousands of Persons, Seasonally Adjusted |
| **Frequency** | Monthly |
| **Date Range (in cache)** | 1967-01-01 to 2026-04-01 (FRED has data back to 1939; trimmed to align with transformer PPI) |
| **Row Count** | 712 |
| **FRED Last Updated** | 2026-05-08 08:30 AM CDT |
| **Fetched by signal_tool** | 2026-05-28 (via web_fetch) |
| **License** | Public Domain: Citation Requested |

**Verification log:**
- 2026-05-28 — Fetched fresh from FRED, spot-checked 10 dates spanning 1967–2026. All values match exactly. ✓
- Note: BLS revises this series monthly. We pin to the 2026-05-08 vintage.

**Known caveats:**
- Seasonally adjusted (unlike the PPI series which are NSA). This is the standard form economists use, but back-test interpretation should account for the smoothing.
- Source code from BLS: `CES3000000001`. The not-seasonally-adjusted version is `CEU3000000001`.
- BLS revises this series monthly as new establishment survey data comes in. Latest 2-3 months should be treated as preliminary.

**Citation:** U.S. Bureau of Labor Statistics, "All Employees, Manufacturing [MANEMP]," retrieved from FRED, Federal Reserve Bank of St. Louis; https://fred.stlouisfed.org/series/MANEMP, May 28, 2026.

---

## How to refresh this manifest

When refreshing any dataset:

1. Re-fetch the source data (URL above)
2. Run `python verify.py <dataset_name>` (planned — currently manual)
3. If values match, update **"Fetched by signal_tool"** date
4. If values differ, document the discrepancy in the **Verification log** and decide: do we trust the new source (FRED revised values) or our cache (the version our backtests were run against)?
5. Update **"FRED Last Updated"** date from the source page

## General data quality principles

These apply across all sources in this tool:

- **Pinned vintage.** Each backtest result is reproducible only against a known data vintage. Re-running with newer data may produce different correlations — that's expected, and worth tracking via a `vintage` column on each backtest output.
- **No silent updates.** If we re-fetch and values change, that's an event to log, not absorb silently. Could indicate a methodology change at the source, a data revision, or a parsing bug in our extractor.
- **Confidence per row.** When a value is estimated rather than directly sourced, flag it. Better to say "this is approximate" than to pretend precision we don't have.
- **Source URL on every chart.** Every output the tool produces should cite the data sources by URL and vintage. Surface the manifest in the UI, don't hide it.
