#!/usr/bin/env python3
"""Join chunked API label CSVs back to the article-actor metadata."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTICLES = ROOT / "data" / "external" / "llm_input_articles.csv"
DEFAULT_LABEL_DIR = ROOT / "results" / "labels"
DEFAULT_OUTPUT = ROOT / "data" / "processed" / "article_labels_from_api.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--articles", type=Path, default=DEFAULT_ARTICLES)
    parser.add_argument("--label-dir", type=Path, default=DEFAULT_LABEL_DIR)
    parser.add_argument("--pattern", default="llm_labels_*.csv")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    paths = sorted(args.label_dir.expanduser().resolve().glob(args.pattern))
    if not paths:
        raise SystemExit(f"No label files matching {args.pattern!r} in {args.label_dir}")
    labels = pd.concat(
        [pd.read_csv(path, dtype={"GlobalEventID": str}, low_memory=False) for path in paths],
        ignore_index=True,
    )
    if "error" in labels.columns:
        labels = labels.loc[labels["error"].fillna("").astype(str).eq("")].copy()
    labels = labels.drop_duplicates("GlobalEventID", keep="last")

    articles = pd.read_csv(
        args.articles.expanduser().resolve(),
        dtype={"GlobalEventID": str},
        low_memory=False,
    )
    overlapping = [column for column in labels.columns if column in articles.columns and column != "GlobalEventID"]
    merged = articles.drop(columns=overlapping).merge(
        labels,
        on="GlobalEventID",
        how="inner",
        validate="many_to_one",
    )
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output, index=False)
    print(f"Merged {labels['GlobalEventID'].nunique():,} labeled articles into {len(merged):,} article-actor rows")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
