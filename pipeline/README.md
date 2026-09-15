# Data pipeline

Run the numbered scripts from the repository root. Steps 1-5 use `data/raw/`; subsequent steps write cleaned products to `data/processed/`, reports to `data/reports/`, and translation files to `data/translation/`.

The default GDELT query is deliberately in sample mode. Set a complete date window and disable sample mode only after checking BigQuery dry-run estimates:

```bash
export GDELT_START_DATE=20200101
export GDELT_END_DATE=20251231
export GDELT_SAMPLE_MODE=false
```

Authentication uses `GOOGLE_APPLICATION_CREDENTIALS` when set and otherwise Google Application Default Credentials. Never commit the service-account file.

The scraper accesses third-party publisher websites. Reproduction therefore depends on current site availability, robots policies, page structure, and legal access conditions. The checked-in compact data should be used when the goal is to reproduce the statistical results rather than recollect the corpus.
