# Screening protocol: title/abstract stage
Version 1.0. Confirmed by the review author on 2026-11-23. The author's words were: "confirm and do fully autonomous systematic review between agents". The protocol was frozen before any record from the database export was screened.

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
- **Full-text-only criteria.** These must NOT be used to exclude at the title/abstract stage: exact baselines used; whether synthetic data with a known process was used; which context lengths were studied.

## "Comparator or analysis" (interpret broadly at screening)
- **Counts:**
  - any mention of comparing a foundation model or in-context transformer with ARIMA, ETS, GARCH, AR, statistical baselines, naive or random-walk forecasts;
  - any study of context or lookback length;
  - any theory of what the model computes in-context;
  - synthetic data with a known generating law.
- **Does not count on its own:** a new foundation model paper that only reports benchmark leaderboard scores against other deep models.
  - Exception: if that abstract also mentions classical baselines, context-length analysis or synthetic known processes, treat it as "unclear".

## Decisions
- **"include" (code INC):** the population (a or b) is clearly met AND at least one comparator/analysis (i)–(v) is clearly indicated in the title or abstract.
- **"unclear" (code UNC):** the population is plausibly met, but the title/abstract does not show whether a comparator or analysis (i)–(v) is present. For example:
  - (a) a foundation-model evaluation in a domain, such as finance or energy, with "baselines" unspecified;
  - (b) a new pretrained forecasting model whose abstract mentions synthetic data or zero-shot evaluation without saying what it is compared against;
  - (c) no abstract, but the title suggests a foundation model or in-context forecasting study.
- **"exclude":** give ONE code, the first criterion that clearly fails, checked in this order.
  - **E1 publication type:** editorial, comment, erratum, book front matter, thesis listing without content, call for papers, dataset-only description. Surveys and reviews are NOT excluded by E1: mark them "unclear", because their reference lists are used for citation chasing.
  - **E2 not time series:** the record is not about forecasting, filtering or predicting time series or dynamical systems. Examples: image models, NLP-only in-context learning, tabular classification.
  - **E3 population:** about time series, but no pretrained foundation model and no in-context transformer forecaster. Examples: a task-specific LSTM or a classical model only, or a foundation model used only for classification, anomaly detection or imputation and not for forecasting.
  - **E4 comparator/analysis:** population met, but the abstract clearly shows the study only compares deep models with each other on benchmarks, with none of (i)–(v). This includes applications reporting accuracy of a fine-tuned model with no classical, naive or oracle comparison, and no context-length or mechanism analysis.
  - **E9 other reason** (explain).

## Principles
- **Sensitive:** missing an eligible study is worse than an extra full-text check. When in doubt between include/unclear and exclude, choose "unclear".
- **Decisive:** most records are clearly irrelevant, for example LLM text tasks or medical prediction with task-specific models. Use "unclear" only when the population is plausibly met.
- Judge the content, not the language. Use the publication type in the record header for E1.

## Amendment log
| Date | Change | Reason | Records to re-screen |
|------|--------|--------|----------------------|
