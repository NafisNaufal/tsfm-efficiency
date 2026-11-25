# How statistically efficient is in-context forecasting?

Code, analysis plan and result tables for an oracle-relative evaluation of zero-shot time-series foundation models (TSFMs).

Authors: Nafis Naufal Rahman and Taqiyudin Miftah Adn (Universitas Brawijaya). The paper is in preparation for ESANN; a link will be added here.

## What the study does
- **Data.** Ten synthetic families with known parameters:
  - AR(1), AR(3), ARMA(1,1), seasonal AR (s = 12);
  - local level, 2-regime Markov-switching AR;
  - GARCH(1,1) with Gaussian and with Student-t innovations;
  - random walk, white noise.
- **Sample.** N = 2000 series per family, with random parameters for every series.
- **Contexts.** T ∈ {16, 32, …, 2048}. Every context is a suffix of the same series with the same forecast origin, so all comparisons are paired.
- **Methods compared:**
  - the Bayes oracle (true parameters);
  - a correctly specified MLE fitted to the same context;
  - AutoARIMA, AutoETS and naive;
  - eight pretrained TSFMs used zero-shot: Chronos-Bolt-S/B, Chronos-2, TimesFM-2.5, TimesFM-3, TiRex, Moirai-2.0-R-small and Toto-Open-Base-1.0.
- **Primary estimand.** Excess risk ER = E[(f − o)²] / E[Var(y | context)], where o is the oracle conditional mean. This equals the relative MSE excess, (MSE_f − MSE_o)/MSE_o.
- **Hypotheses.** Crossover against the MLE (H1), saturation of the learning-curve slope (H2) and spurious predictability under martingale nulls (H3). They are tested with a joint paired bootstrap (B = 10⁴) and Holm correction.
- **Exploratory.** Finite-difference Jacobian ("effective filter") analysis.

## Main results (`results/main/`)
| file | content |
|---|---|
| `summary.csv` | ER (primary and realized-loss estimators, 95% CIs), pinball-loss ratio, 80% coverage and QLIKE for every family × T × method × horizon |
| `re.csv` | relative efficiency ER_TSFM / ER_MLE at h = 1 |
| `H1.csv`, `H2.csv`, `H3.csv` | confirmatory tests for the 8 pre-specified models (one Holm family per hypothesis) |
| `*_toto_median.csv` | exploratory: Toto scored by its predictive median (deviation D5, post hoc) |
| `../jacobian/` | effective-filter metrics at ε = 1e-3 and 1e-2, plus notes (TiRex excluded as non-smooth, with the reason) |

## Analysis plan
`ANALYSIS_PLAN.md` was frozen before any main-run forecast was produced. Deviations D1–D5 are logged in it with dates and reasons; D5 was made after inspecting main results and is labelled exploratory.

The plan was written with an AI research assistant and was not registered on a public registry, so we call it a *pre-specified analysis plan*, not a pre-registration. SHA-256 of the frozen version: `dc76521814c3d0847185fc44bbbcedcd746071a0a61cbc9c738d48c3ebc8b57c`. The repository copy includes the later deviation rows.

## Reproducing
1. Create the three Python environments from `envs/`:
   - main: chronos-forecasting, timesfm, tirex-ts, statsmodels, statsforecast, arch;
   - Moirai: uni2ts;
   - Toto: toto-ts.
2. Set the paths in `src/run.py` (`TSFMS` maps each model to its environment).
3. Run:

```bash
python src/run.py --stage data      --tag main --N 2000
python src/run.py --stage oracle    --tag main --procs 64
python src/run.py --stage baselines --tag main --procs 64
python src/run.py --stage tsfm      --tag main --batch 128     # GPU; retries automatically on CUDA OOM
python src/analyze.py    --tag main --B 10000
python src/hypotheses.py --tag main --B 10000
python src/jacobian.py build|cpu|tsfm|analyze --eps 1e-3        # exploratory filter analysis
python src/figures.py    --tag main
```

Raw forecasts (about 2.8 GB) are not included; the commands above regenerate them.

Original run environment: one NVIDIA A100-SXM4-80GB and an AMD EPYC 7742. Hugging Face revisions:

| model | revision |
|---|---|
| chronos-bolt-small | 772f3d25 |
| chronos-bolt-base | 5d9f166d |
| chronos-2 | 29ec3766 |
| timesfm-2.5-200m-pytorch | 1d952420 |
| timesfm-3.0-pytorch | 43046b85 |
| TiRex | 43ebaaff |
| moirai-2.0-R-small | 30f43ff0 |
| Toto-Open-Base-1.0 | 0411ceb2 |

Toto samples are Monte Carlo draws and are not bit-reproducible.

## Literature review (`review/`)
The literature review was a systematic review with dual blinded screening and adjudication:
- 918 records were screened on title and abstract;
- 249 full texts were assessed;
- 174 studies were included.
- Inter-reviewer agreement at full text: κ = 0.906.

This folder contains:
- the protocols, PRISMA counts, the search and filter scripts, and `decision_log.csv` (bibliographic metadata and decisions only);
- no abstracts and no full texts of other authors' papers, which are not redistributed.

## Use of AI
The study design, code, experiments, literature screening and analysis were carried out with an AI research assistant (Claude, Anthropic), working under the authors' direction on the authors' hardware. Screening decisions were made by AI reviewer agents under a documented protocol.

## License
MIT (see `LICENSE`).
