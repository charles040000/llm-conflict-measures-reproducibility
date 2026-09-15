#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import pandas as pd
from openai import AuthenticationError, BadRequestError, OpenAI

from prompt_runtime import (
    ARTICLE_COLUMNS,
    LABEL_FIELDS,
    ROOT_DIR,
    append_jsonl,
    build_user_prompt,
    call_model,
    read_table,
    resolve_path,
)


DEFAULT_ARTICLES_PATH = ROOT_DIR / "data" / "external" / "llm_input_articles.csv"
DEFAULT_SYSTEM_PROMPT = ROOT_DIR / "prompts" / "system_prompt.txt"
DEFAULT_USER_PROMPT = ROOT_DIR / "prompts" / "user_prompt.txt"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "results" / "labels"


def safe_model_name(model: str) -> str:
    return model.replace("/", "_").replace(":", "_")


def output_paths(output_dir: Path, model: str, year: str, chunk_index: int | None) -> tuple[Path, Path]:
    safe_model = safe_model_name(model)
    chunk_suffix = f"_chunk_{chunk_index:03d}" if chunk_index is not None else ""
    jsonl = output_dir / f"llm_labels_{year}_{safe_model}{chunk_suffix}.jsonl"
    csv = output_dir / f"llm_labels_{year}_{safe_model}{chunk_suffix}.csv"
    return jsonl, csv


def existing_prediction_paths(output_dir: Path, model: str, year: str) -> list[Path]:
    safe_model = safe_model_name(model)
    return sorted(output_dir.glob(f"llm_labels_{year}_{safe_model}*.jsonl"))


def filter_rows(df: pd.DataFrame, year: str) -> pd.DataFrame:
    if "SQLDATE" in df.columns:
        mask = df["SQLDATE"].fillna("").astype(str).str.startswith(year)
    else:
        mask = pd.Series(False, index=df.index)
    if "source_month" in df.columns:
        mask |= df["source_month"].fillna("").astype(str).str.startswith(f"{year}_")
    return df[mask].copy().sort_values(["source_month", "GlobalEventID"], kind="stable")


def trim_article_text(row: pd.Series, max_chars: int | None) -> pd.Series:
    if max_chars is None or max_chars <= 0:
        return row
    row = row.copy()
    for col in ["article_text_for_llm", "article_text_translated", "article_text_original", "text_cleaned", "text"]:
        text = row.get(col, "")
        if pd.notna(text) and str(text):
            row[col] = str(text)[:max_chars]
            break
    return row


def predictions_to_csv(predictions_path: Path, csv_path: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    with predictions_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            row = {
                "GlobalEventID": str(record.get("GlobalEventID", "")),
                "model": record.get("model", ""),
                "error": record.get("error", ""),
            }
            prediction = record.get("prediction", {})
            for field in LABEL_FIELDS:
                row[field] = prediction.get(field, "")
            rows.append(row)
    out = pd.DataFrame(rows)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(csv_path, index=False)
    return out


def load_existing_article_predictions(paths: list[Path]) -> set[str]:
    done: set[str] = set()
    for path in paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                if str(record.get("error", "") or "").strip():
                    continue
                done.add(str(record.get("GlobalEventID", "")))
    return done


def print_distribution(df: pd.DataFrame) -> None:
    if df.empty:
        return
    ok = df[df["error"].eq("")] if "error" in df.columns else df
    print(f"Successful labels: {len(ok):,}")
    for col in ["conflict_relevance", "event_context", "event_modality", "matched_actor_role"]:
        if col in ok.columns:
            print(f"\n{col}")
            print(ok[col].fillna("").astype(str).value_counts().to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description="Label a year/chunk of deduped articles with the OpenAI prompt.")
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-5.4-nano"))
    parser.add_argument("--articles", type=Path, default=DEFAULT_ARTICLES_PATH)
    parser.add_argument("--year", required=True)
    parser.add_argument("--chunk-size", type=int, default=100)
    parser.add_argument("--chunk-index", type=int, default=None, help="Zero-based chunk index.")
    parser.add_argument("--start", type=int, default=None, help="Zero-based start row after year filtering.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--text-max-chars", type=int, default=None, help="Optional text truncation. Omit for full article text.")
    parser.add_argument("--system-prompt", type=Path, default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--user-prompt", type=Path, default=DEFAULT_USER_PROMPT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--max-output-tokens", type=int, default=1800)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    articles_path = resolve_path(args.articles)
    system_prompt_path = resolve_path(args.system_prompt)
    user_prompt_path = resolve_path(args.user_prompt)
    output_dir = resolve_path(args.output_dir)
    predictions_path, csv_path = output_paths(output_dir, args.model, args.year, args.chunk_index)

    articles = read_table(articles_path, ARTICLE_COLUMNS + ["SQLDATE"]).fillna("")
    articles["GlobalEventID"] = articles["GlobalEventID"].astype(str)
    rows = filter_rows(articles, args.year).reset_index(drop=True)
    total_year_rows = len(rows)

    if args.chunk_index is not None:
        start = args.chunk_index * args.chunk_size
        limit = args.chunk_size
    else:
        start = args.start or 0
        limit = args.limit
    rows = rows.iloc[start : start + limit if limit is not None else None].copy()
    if rows.empty:
        raise SystemExit(f"No rows selected for year={args.year}, start={start}, limit={limit}.")

    system_prompt = system_prompt_path.read_text(encoding="utf-8")
    user_template = user_prompt_path.read_text(encoding="utf-8")

    first_row = trim_article_text(rows.iloc[0], args.text_max_chars)
    first_prompt = build_user_prompt(user_template, first_row)
    if args.dry_run:
        print(f"Year rows: {total_year_rows:,}")
        print(f"Selected rows: {len(rows):,}")
        print(f"Output JSONL: {predictions_path}")
        print("\nFirst user prompt preview:")
        print(first_prompt[:4000])
        return

    if args.overwrite and predictions_path.exists():
        predictions_path.unlink()
    existing_paths = existing_prediction_paths(output_dir, args.model, args.year)
    done = load_existing_article_predictions(existing_paths)
    selected_done = int(rows["GlobalEventID"].isin(done).sum())
    if selected_done:
        print(f"Skipping {selected_done:,} selected row(s) already present in existing {args.year} outputs.")
    client = OpenAI()

    attempted = 0
    for row in rows.itertuples(index=False):
        row_s = trim_article_text(pd.Series(row._asdict()), args.text_max_chars)
        eid = str(row_s["GlobalEventID"])
        if eid in done:
            continue
        user_prompt = build_user_prompt(user_template, row_s)
        try:
            prediction, meta = call_model(client, args.model, system_prompt, user_prompt, args.max_output_tokens)
            record = {
                "GlobalEventID": eid,
                "model": args.model,
                "prediction": prediction,
                "response_id": meta.get("response_id", ""),
                "usage": meta.get("usage", {}),
            }
        except (AuthenticationError, BadRequestError) as exc:
            raise SystemExit(f"Stopping after API configuration error for {eid}: {exc}") from exc
        except Exception as exc:
            error_text = repr(exc)
            if "insufficient_quota" in error_text or "You exceeded your current quota" in error_text:
                raise SystemExit(f"Stopping after quota error for {eid}: {exc}") from exc
            record = {
                "GlobalEventID": eid,
                "model": args.model,
                "error": error_text,
            }
        append_jsonl(predictions_path, record)
        attempted += 1
        print(f"{attempted:>4} / {len(rows):<4} {eid}")
        if args.sleep:
            time.sleep(args.sleep)

    out = predictions_to_csv(predictions_path, csv_path)
    print(f"Year rows: {total_year_rows:,}")
    print(f"Selected rows: {len(rows):,}")
    print(f"Wrote predictions: {predictions_path}")
    print(f"Wrote CSV: {csv_path}")
    print_distribution(out)


if __name__ == "__main__":
    main()
