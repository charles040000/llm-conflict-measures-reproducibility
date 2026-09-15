# Data inventory

## `processed/`

- `article_labels_compact.csv`: article IDs, actor assignments, dates, weights, and LLM labels for the three analysis actors. Full text is excluded.
- `actor_period_panel.csv`: final two-week panel, including raw and directly calibrated shares.
- `monthly_model_panel.csv`: monthly three-actor panel used for descriptive and VIIRS merges.
- `actor_month_panel.csv`: broader monthly actor panel used to show article-count concentration.
- `temporal_windows/`: one-week, two-week, four-week, and calendar-month model panels.
- `viirs_nightlight_panel.csv` and `viirs_thermal_panel.csv`: model-ready satellite extensions.

## `reference/`

Actor names, aliases, UCDP identifiers, and the FARC-family grouping rules used by the pipeline.

## `validation/`

Exactly two source files are released:

- `manual_validation.csv`: 173 unique manual-reference observations; `in_three_actor_calibration` selects the final 159-observation pooled calibration sample.
- `inter_annotator.csv`: 50 anonymized coder pairs.

Derived agreement tables and classification rates are written to `results/validation/`; they are not additional source-label files.

`CHECKSUMS.sha256` records SHA-256 hashes for the released processed and validation CSVs. From the repository root, verify them with `shasum -a 256 -c data/CHECKSUMS.sha256`.
