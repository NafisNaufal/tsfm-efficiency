# PRISMA 2020 counts - title/abstract stage

```mermaid
flowchart TD
  I["Records identified from databases (n = 1356)<br/>arxiv n = 391; openalex n = 407; s2 n = 558"]
  R["Records removed before screening:<br/>duplicate records (n = 438)"]
  S["Records screened (n = 918)"]
  X["Records excluded (n = 557)<br/>E1 Publication type (editorial, comment, erratum, front matter, call for papers, dataset-only): 18<br/>E2 Not about forecasting/filtering/predicting time series or dynamical systems: 330<br/>E3 No pretrained foundation model and no in-context transformer forecaster: 138<br/>E4 None of oracle / classical / naive-on-unpredictable / context-length / mechanism analysis: 71"]
  F["Reports sought for retrieval (n = 361)"]
  I --> R
  I --> S
  S --> X
  S --> F
```

```json
{
 "stage": "ta",
 "tool": "sr-screener 1.0.0",
 "complete": true,
 "pending_records": 0,
 "state_problems": [],
 "identified_by_database": {
  "arxiv": 391,
  "openalex": 407,
  "s2": 558
 },
 "identified_total": 1356,
 "duplicates_removed": 438,
 "records_screened": 918,
 "records_to_screen": 918,
 "records_excluded": 557,
 "excluded_by_code": {
  "E1": 18,
  "E2": 330,
  "E3": 138,
  "E4": 71
 },
 "reports_sought_for_retrieval": 361,
 "advanced_include": 162,
 "advanced_unclear": 199,
 "advanced_outside_allowed_languages": 0,
 "agreement": {
  "n": 918,
  "both_advance": 298,
  "both_exclude": 575,
  "a_only_advance": 21,
  "b_only_advance": 24,
  "observed_agreement": 0.951,
  "kappa": 0.892,
  "pabak": 0.902,
  "conflicts": 45,
  "resolved_by": {
   "A+B": 848,
   "ADJ": 44,
   "QC": 26
  }
 },
 "decided_by": {
  "A+B": 848,
  "ADJ": 44,
  "QC": 26
 }
}
```
