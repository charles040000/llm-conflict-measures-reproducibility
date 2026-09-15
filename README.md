# LLM-Generated Conflict Measures: Reproducibility Repository

This repository contains the code and compact data needed to reproduce the empirical analysis of LLM-generated news covariates used in the thesis. The application studies whether the share of rebel-related articles reporting direct conflict relevance, main-story physical violence, or main-story fatalities is associated with actor-linked UCDP fatalities in the following two-week period.

The main regression sample contains 360 actor-period observations for Hamas, the Taliban, and the FARC family from 2020 through 2025. The repository includes deterministic preprocessing code, prompt files, validation data, model-ready panels, model implementations, tidy results, and the code used to generate the figures. The thesis LaTeX source is intentionally excluded.

The final labeled analytical corpus contains 116,900 articles after excluding 272 articles that still match more than one analysis actor. The three focal actors account for 115,142 articles: 88,554 for Hamas, 23,641 for the Taliban, and 2,947 for the FARC family.

## Repository layout

```text
analysis/       panel construction, validation, regressions, corrections, and figures
pipeline/       GDELT retrieval, scraping, cleaning, actor matching, and translation
labeling/       OpenAI labeling and manual-labeling tools
prompts/        finalized system and article prompts
data/
  processed/    compact article labels and model-ready panels
  reference/    actor names, aliases, and grouping rules
  validation/   the two released manual-validation files
  external/     instructions for large or restricted source data
results/        tidy model outputs used by the figure script
figures/        PDF and PNG versions of the generated figures
```

## Setup

Python 3.10 or 3.11 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The complete acquisition pipeline has additional dependencies:

```bash
python -m pip install -r requirements-pipeline.txt
```

All scripts resolve paths from the repository root. They do not require the repository to be placed in a particular directory.

## Quick reproduction

The included compact data and archived stochastic model output are sufficient to rebuild the validation statistics, naive regressions, descriptive analysis, and all figures:

```bash
python reproduce.py quick
```

The main deterministic steps can also be run separately:

```bash
python analysis/run_validation_analysis.py
python analysis/run_baseline_models.py
python analysis/run_taliban_descriptive.py
python analysis/generate_figures.py
```

The baseline command should report 360 observations and the following raw-share coefficients, subject only to ordinary floating-point differences:

| Measure | Estimate | HAC SE |
|---|---:|---:|
| Direct conflict relevance | 0.963 | 0.627 |
| Main-story physical violence | 1.030 | 0.424 |
| Main-story fatalities | 0.199 | 0.385 |

Check the released sample, validation rates, privacy exclusions, and path portability with:

```bash
python verify_release.py
```

## Full model reproduction

To rerun the validation bootstrap, joint Hamiltonian Monte Carlo fits, integrated maximum-likelihood models, timing diagnostic, and robustness models:

```bash
python reproduce.py full
```

The full workflow uses four HMC chains with 1,000 warmup and 2,000 retained draws per chain for each primary label. Runtime depends heavily on the local JAX installation and available CPU cores. Posterior draws will not be bit-for-bit identical across hardware, but the fixed seeds and reported convergence diagnostics make the results directly comparable.

Individual model entry points are:

```bash
python analysis/run_calibration_models.py
python analysis/run_joint_beta_model.py
python analysis/build_estimator_comparison.py
python analysis/run_timing_diagnostic.py
python analysis/run_robustness_models.py
python analysis/run_joint_prior_sensitivity.py
```

The preferred joint model uses a beta distribution for actor-period latent shares and follows the benchmark `Gamma(1,10)` prior for the residual standard deviation. Because that prior places substantial weight below the residual scale observed in this application, `results/calibration/sigma_prior_sensitivity/` reports comparisons with `HalfNormal(1)` and `Gamma(2,2)` alternatives.

## Validation data

`data/validation/` intentionally contains two CSV files:

- `manual_validation.csv` contains 173 unique manually reviewed articles and corresponding LLM labels. The Boolean `in_three_actor_calibration` identifies the 159 observations used to estimate the pooled classification rates for the three analysis actors.
- `inter_annotator.csv` contains 50 independently coded article pairs. Annotators are anonymized as `coder_1` and `coder_2`.

The three primary manual variables are complete. The validation analysis treats manually assigned labels as the empirical reference labels, while inter-annotator agreement documents the limitation that manual coding is itself imperfect for complex categories.

## Included data and excluded source material

The compact article file, `data/processed/article_labels_compact.csv`, contains IDs, actor assignments, dates, weights, and LLM labels for the three analysis actors. It deliberately excludes full article text, URLs, API responses, credentials, and token metadata. This keeps the release small and avoids redistributing copyrighted news text.

The included model-ready panels reproduce the published analysis without access to raw news text. Rebuilding the panels from the beginning additionally requires:

- UCDP GED version 26.1 as `data/external/GEDEvent_v26_1.csv`;
- GDELT BigQuery access and Google credentials for new event retrieval;
- access to the original article URLs for scraping;
- an OpenAI API key for regenerating labels;
- a Google Earth Engine project for rebuilding the exploratory VIIRS outcomes.

See `data/external/README.md` for the expected external inputs.

## Full acquisition and labeling pipeline

The numbered scripts in `pipeline/` document the complete construction order:

```bash
python pipeline/01_build_actor_terms.py
python pipeline/02_query_gdelt_events.py
python pipeline/03_query_gdelt_mentions.py
python pipeline/04_scrape_articles.py
python pipeline/05_clean_articles.py
python pipeline/06_deduplicate_articles.py
python pipeline/07_create_strict_actor_matches.py
python pipeline/08_attach_actor_metadata.py
python pipeline/09_create_manual_actor_mapping.py
python pipeline/10_detect_article_languages.py
python pipeline/11_prepare_translation_input.py
python pipeline/12_translate_non_english_articles.py
python pipeline/13_join_translated_article_texts.py
```

Copy `config/example.env` to `.env` or export the variables in the shell. Credentials are never read from a tracked file unless the user explicitly places one at `config/service_account_key.json`.

The finalized labeling input can then be processed with:

```bash
export OPENAI_API_KEY=...
python labeling/label_articles.py --articles data/external/llm_input_articles.csv
python labeling/merge_label_outputs.py
```

The historical production model identifier was `gpt-5.4-nano`. A different available model can be supplied with `--model` or `OPENAI_MODEL`. Strict JSON-schema output and the prompt files in `prompts/` preserve the coding structure.

## Rebuilding panels and satellite extensions

With UCDP GED available locally, rebuild the panel with:

```bash
python analysis/build_actor_month_panel.py \
  --article-labels data/processed/article_labels_from_api.csv
python analysis/run_temporal_robustness.py
```

The exploratory VIIRS data can be rebuilt after authenticating Google Earth Engine:

```bash
python analysis/build_viirs_nightlight_outcome.py --ee-project YOUR_PROJECT
python analysis/build_viirs_thermal_outcome.py --ee-project YOUR_PROJECT
python analysis/run_viirs_nightlight_models.py
python analysis/run_viirs_thermal_models.py
```

The checked-in VIIRS panels allow the model stage to be rerun without Earth Engine.

## Reproducibility boundaries

The compact release reproduces the reported statistical analysis but not a frozen copy of the external news universe. GDELT records, source websites, model availability, and external datasets can change. A fresh end-to-end acquisition run can therefore differ from the archived 2020-2025 corpus even when the code and prompts are unchanged.

No API keys, Google credentials are included.
