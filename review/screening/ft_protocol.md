# Screening protocol: full-text stage
Version 1.0 (full-text). Derived from the confirmed title/abstract protocol v1.0. Confirmed by the review author on 2026-11-23. The author's words were: "confirm and do fully autonomous systematic review between agents". The protocol was frozen before any record from the database export was screened.

Review (systematic review of methods studies): "Statistical efficiency of in-context (zero-shot) forecasting by pretrained time-series foundation models and in-context transformers".

Question: what evidence exists on how well pretrained time-series foundation models, and transformers that forecast in-context, perform as statisticians would measure it? Specifically:
- against a known optimal (oracle or Bayes) predictor;
- against classical estimators fitted to the same context;
- as the context length changes;
- on unpredictable series.

And what is known about the forecasting rule these models implement?

## Eligibility (full protocol)
- **Population (the models studied).** At least one of:
  - (a) a pretrained time-series foundation model used for forecasting, meaning a model pretrained on many series and then applied to new series without task-specific training. Examples: Chronos, Chronos-Bolt, Chronos-2, TimesFM, Moirai, Lag-Llama, Timer, TimeGPT, Toto, TiRex, Sundial, TTM, Time-MoE, Moment, or a large language model used as a time-series forecaster;
  - (b) a transformer or attention model trained to forecast, filter or predict a time series or dynamical system in-context. "In-context" means the model infers the process from the sequence it is given, without weight updates.
- **Exposure.** Forecasting, filtering or prediction of future values of a time series or dynamical system from a context window.
- **Comparator or analysis.** At least one of:
  - (i) an oracle, Bayes-optimal or true-parameter predictor on data with a known generating process;
  - (ii) classical statistical estimators (AR, ARIMA, ETS, GARCH, least squares, Kalman filter, system identification) used as comparators;
  - (iii) naive, random-walk, zero or no-change forecasts on series that are hard or impossible to predict (for example asset returns, random walks, white noise);
  - (iv) analysis of how performance changes with context or window length;
  - (v) theoretical or mechanistic analysis of the algorithm or forecasting rule implemented in-context. Examples: proofs that a transformer implements least squares, gradient descent or a Kalman filter; probes, Jacobians, impulse responses, filters.
- **Outcome.** Forecast accuracy, excess risk, calibration or uncertainty quality, or a characterisation of the implemented forecasting rule.
- **Design.** Empirical evaluation or theoretical analysis, published 2019–2026. Peer-reviewed papers and preprints are both eligible.
- **Full-text criteria (apply now).** The study must actually report results (not only propose) for at least one comparator/analysis (i)-(v) applied to a population (a) or (b) model. "Mentions" in related work or future work do not count. A survey or review is excluded at full text with E1, unless it contains its own new empirical or theoretical analysis meeting (i)-(v).
- **Reading material.** You are given text extracted from the PDF (figures omitted, tables as plain text, page markers "=== PAGE n ==="). Judge what the text reports; cite the page marker in "where".

## "Comparator or analysis" (interpret broadly at screening)
- **Counts:**
  - any mention of comparing a foundation model or in-context transformer with ARIMA, ETS, GARCH, AR, statistical baselines, naive or random-walk forecasts;
  - any study of context or lookback length;
  - any theory of what the model computes in-context;
  - synthetic data with a known generating law.
- **Does not count on its own:** a new foundation model paper that only reports benchmark leaderboard scores against other deep models.
  - Exception: if that abstract also mentions classical baselines, context-length analysis or synthetic known processes, treat it as "unclear".

## Decisions
- "include" (code INC): population (a) or (b) is met AND the report presents results for at least one of (i)-(v).
- "unclear" (code UNC): only when information needed to decide is missing from the report itself (for example the extracted text is truncated or unreadable, or the model type is never specified). This is awaiting classification, not a hedge.
- "exclude": give ONE code, the first criterion that clearly fails, checked in this order:
  - E1 publication type or not a primary study: editorial, comment, erratum, front matter, call for papers, dataset/software-only record, thesis listing without content; surveys/reviews without their own new analysis meeting (i)-(v).
  - E2 not time series: not about forecasting, filtering or predicting time series or dynamical systems.
  - E3 population: no pretrained time-series foundation model (or LLM used as forecaster) is evaluated, and no in-context transformer forecaster is studied; or the foundation model is used only for classification, anomaly detection, imputation or feature extraction.
  - E4 comparator/analysis: population met, but no results for any of (i) oracle/Bayes/true-parameter predictor on known processes, (ii) classical statistical estimators as comparators, (iii) naive/random-walk/zero forecasts on hard-to-predict series, (iv) performance as a function of context/window length, (v) theory or mechanistic analysis of the in-context forecasting rule.
  - E9 other reason (explain).

## Principles
- **Decide on the report.** At full text, decide include or exclude whenever the report allows it; "unclear" is only for information missing from the report.
- **Decisive:** most reports are clearly classifiable from the text; use one code and cite the page.
- Judge the content, not the language. Use the publication type in the record header for E1.

## Amendment log
| Date | Change | Reason | Records to re-screen |
|------|--------|--------|----------------------|
