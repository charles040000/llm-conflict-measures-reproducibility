# External inputs

Large, restricted, or living source data are not bundled with this repository.

Place UCDP GED version 26.1 at:

```text
data/external/GEDEvent_v26_1.csv
```

To regenerate LLM labels, prepare the full article-level labeling input at:

```text
data/external/llm_input_articles.csv
```

That file is expected to contain the source metadata and cleaned or translated article text consumed by `labeling/label_articles.py`. It is excluded because it contains redistributed news text.

GDELT query and scrape outputs are written under `data/raw/`. They can be reconstructed with the numbered scripts in `pipeline/`, subject to current BigQuery records and source-website availability.

The checked-in VIIRS panels can be analyzed directly. Re-extraction requires a Google Earth Engine project and the UCDP GED file because UCDP activity regions define the satellite footprints.
