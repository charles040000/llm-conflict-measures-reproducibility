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


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_LABELS_PATH = ROOT_DIR / "data" / "validation" / "manual_validation.csv"
DEFAULT_ARTICLES_PATH = ROOT_DIR / "data" / "external" / "llm_input_articles.csv"
DEFAULT_SYSTEM_PROMPT = ROOT_DIR / "prompts" / "system_prompt.txt"
DEFAULT_USER_PROMPT = ROOT_DIR / "prompts" / "user_prompt.txt"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "results" / "labeling_evaluation"

LABEL_FIELDS = [
    "conflict_relevance",
    "event_context",
    "event_modality",
    "matched_actor_role",
    "event_time_relation",
    "any_matched_actor_mentioned",
    "any_physical_violence_mentioned",
    "any_fatalities_mentioned",
    "any_injuries_mentioned",
    "fatality_count_anywhere_num",
    "injury_count_anywhere_num",
    "any_weapon_mentioned",
    "weapon_type_anywhere_text",
    "physical_violence_occurred",
    "fatalities_present",
    "fatality_count_reported_text",
    "fatality_count_reported_num",
    "injuries_present",
    "injury_count_reported_text",
    "injury_count_reported_num",
    "weapon_type_text",
    "weapon_use_class",
    "article_location_country_text",
    "location_matches_actor_country",
    "multiple_events_mentioned",
    "main_story_confidence",
    "article_language",
    "annotator_notes",
    "flagged",
]

CATEGORICAL_EVAL_FIELDS = [
    "conflict_relevance",
    "event_context",
    "event_modality",
    "matched_actor_role",
    "event_time_relation",
    "any_matched_actor_mentioned",
    "any_physical_violence_mentioned",
    "any_fatalities_mentioned",
    "any_injuries_mentioned",
    "any_weapon_mentioned",
    "physical_violence_occurred",
    "fatalities_present",
    "injuries_present",
    "weapon_use_class",
    "location_matches_actor_country",
    "multiple_events_mentioned",
    "main_story_confidence",
]

ARTICLE_COLUMNS = [
    "GlobalEventID",
    "Actor1Name",
    "Actor2Name",
    "Actor1CountryCode",
    "Actor2CountryCode",
    "ActionGeo_CountryCode",
    "raw_actor_alias_hit_side",
    "raw_actor_alias_terms_in_text",
    "raw_actor_alias_hit_records_json",
    "matched_actor_sides",
    "matched_aliases",
    "matched_gdelt_actor",
    "possible_rebel_names",
    "source_month",
    "final_url",
    "used_url",
    "article_text_for_llm",
    "article_text_translated",
    "article_text_original",
    "text_cleaned",
    "text",
]


def resolve_path(path: str | Path) -> Path:
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = ROOT_DIR / p
    return p


def prefer_fresh_parquet(path: Path) -> Path:
    parquet_path = path.with_suffix(".parquet")
    if (
        path.suffix == ".csv"
        and parquet_path.exists()
        and (not path.exists() or parquet_path.stat().st_mtime_ns >= path.stat().st_mtime_ns)
    ):
        return parquet_path
    return path


def read_table(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    path = prefer_fresh_parquet(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path, columns=columns)
    return pd.read_csv(path, low_memory=False, usecols=(lambda c: c in set(columns)) if columns else None)


def safe_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value)


def normalize_value(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.lower() in {"nan", "none"}:
        return ""
    if text.endswith(".0") and text[:-2].isdigit():
        return text[:-2]
    return text


def parse_records(value: Any) -> list[dict[str, Any]]:
    raw = safe_text(value).strip()
    if not raw:
        return []
    try:
        records = json.loads(raw)
    except json.JSONDecodeError:
        return []
    return records if isinstance(records, list) else []


def matched_actor_names(row: pd.Series) -> str:
    names: list[str] = []
    for record in parse_records(row.get("raw_actor_alias_hit_records_json", "")):
        name = safe_text(record.get("source_actor_name")).strip()
        if name and name not in names:
            names.append(name)
    if names:
        return "; ".join(names)

    possible_names = safe_text(row.get("possible_rebel_names")).strip()
    if possible_names:
        return possible_names

    side = safe_text(row.get("raw_actor_alias_hit_side")).strip()
    if side == "actor1":
        return safe_text(row.get("Actor1Name")).strip()
    if side == "actor2":
        return safe_text(row.get("Actor2Name")).strip()
    matched_gdelt_actor = safe_text(row.get("matched_gdelt_actor")).strip()
    if matched_gdelt_actor:
        return matched_gdelt_actor
    return ""


def normalize_actor_side(value: Any) -> str:
    text = safe_text(value).strip()
    if ";" in text:
        text = text.split(";", 1)[0].strip()
    lowered = text.lower()
    if lowered in {"actor1name", "actor1", "actor_1"}:
        return "actor1"
    if lowered in {"actor2name", "actor2", "actor_2"}:
        return "actor2"
    return text


def prompt_format_value(row: pd.Series, key: str) -> str:
    if key == "matched_actor_names":
        return matched_actor_names(row)
    if key == "matched_actor_side":
        return normalize_actor_side(row.get("raw_actor_alias_hit_side") or row.get("matched_actor_sides"))
    if key == "matched_actor_alias_terms_in_text":
        return safe_text(row.get("raw_actor_alias_terms_in_text") or row.get("matched_aliases")).strip()
    if key == "article_text":
        return safe_text(
            row.get("article_text_for_llm")
            or row.get("article_text_translated")
            or row.get("article_text_original")
            or row.get("text_cleaned")
            or row.get("text")
        ).strip()
    if key == "final_url":
        return safe_text(row.get("final_url") or row.get("used_url")).strip()
    return safe_text(row.get(key)).strip()


def build_user_prompt(template: str, row: pd.Series) -> str:
    values = {
        "GlobalEventID": prompt_format_value(row, "GlobalEventID"),
        "matched_actor_names": prompt_format_value(row, "matched_actor_names"),
        "matched_actor_side": prompt_format_value(row, "matched_actor_side"),
        "matched_actor_alias_terms_in_text": prompt_format_value(row, "matched_actor_alias_terms_in_text"),
        "Actor1Name": prompt_format_value(row, "Actor1Name"),
        "Actor2Name": prompt_format_value(row, "Actor2Name"),
        "Actor1CountryCode": prompt_format_value(row, "Actor1CountryCode"),
        "Actor2CountryCode": prompt_format_value(row, "Actor2CountryCode"),
        "ActionGeo_CountryCode": prompt_format_value(row, "ActionGeo_CountryCode"),
        "source_month": prompt_format_value(row, "source_month"),
        "final_url": prompt_format_value(row, "final_url"),
        "article_text": prompt_format_value(row, "article_text"),
    }
    return template.format(**values)


def response_schema() -> dict[str, Any]:
    properties: dict[str, Any] = {}
    for field in LABEL_FIELDS:
        if field == "flagged":
            properties[field] = {"type": "boolean"}
        else:
            properties[field] = {"type": "string"}
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
        "required": LABEL_FIELDS,
    }


def output_text(response: Any) -> str:
    text = getattr(response, "output_text", None)
    if text:
        return text
    parts: list[str] = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            value = getattr(content, "text", None)
            if value:
                parts.append(value)
    return "\n".join(parts)


def call_model(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_output_tokens: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "article_labeling_v2",
                "schema": response_schema(),
                "strict": True,
            }
        },
        max_output_tokens=max_output_tokens,
    )
    raw_text = output_text(response)
    parsed = json.loads(raw_text)
    meta = {
        "response_id": getattr(response, "id", ""),
        "model": model,
        "usage": getattr(response, "usage", None).model_dump() if getattr(response, "usage", None) else {},
    }
    return parsed, meta


def load_existing_predictions(path: Path) -> set[tuple[str, str]]:
    if not path.exists():
        return set()
    done: set[tuple[str, str]] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if "error" in record or "prediction" not in record:
                continue
            done.add((str(record["GlobalEventID"]), str(record["annotator_id"])))
    return done


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def prepare_rows(labels_path: Path, articles_path: Path, annotator: str | None) -> pd.DataFrame:
    labels = pd.read_csv(labels_path, dtype={"GlobalEventID": str}, low_memory=False).fillna("")
    if annotator:
        labels = labels[labels["annotator_id"].astype(str).eq(annotator)].copy()
    article_columns = [col for col in ARTICLE_COLUMNS]
    articles = read_table(articles_path, article_columns).fillna("")
    articles["GlobalEventID"] = articles["GlobalEventID"].astype(str)
    rows = labels.merge(articles, on="GlobalEventID", how="left", validate="many_to_one")
    rows["_has_article_text"] = rows.apply(lambda row: bool(prompt_format_value(row, "article_text")), axis=1)
    return rows


def build_comparison(predictions_path: Path, labels_path: Path, output_path: Path, annotator: str | None) -> pd.DataFrame:
    labels = pd.read_csv(labels_path, dtype={"GlobalEventID": str}, low_memory=False).fillna("")
    if annotator:
        labels = labels[labels["annotator_id"].astype(str).eq(annotator)].copy()

    prediction_rows: list[dict[str, Any]] = []
    with predictions_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if "error" in record:
                continue
            pred = record.get("prediction", {})
            row = {
                "GlobalEventID": str(record["GlobalEventID"]),
                "annotator_id": str(record["annotator_id"]),
                "model": record.get("model", ""),
            }
            for field in LABEL_FIELDS:
                row[f"llm_{field}"] = pred.get(field, "")
            prediction_rows.append(row)

    predictions = pd.DataFrame(prediction_rows)
    if predictions.empty:
        raise SystemExit(f"No successful predictions found in {predictions_path}")
    comparison = labels.merge(predictions, on=["GlobalEventID", "annotator_id"], how="inner")
    for field in CATEGORICAL_EVAL_FIELDS:
        manual = comparison[field].map(normalize_value)
        llm = comparison[f"llm_{field}"].map(normalize_value)
        comparison[f"{field}_match"] = manual.eq(llm)
    match_cols = [f"{field}_match" for field in CATEGORICAL_EVAL_FIELDS]
    comparison["categorical_match_count"] = comparison[match_cols].sum(axis=1)
    comparison["categorical_match_share"] = comparison["categorical_match_count"] / len(match_cols)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output_path, index=False)
    return comparison


def print_summary(comparison: pd.DataFrame) -> None:
    if comparison.empty:
        print("No comparison rows written.")
        return
    print(f"Compared {len(comparison):,} rows.")
    for field in CATEGORICAL_EVAL_FIELDS:
        match_col = f"{field}_match"
        if match_col in comparison.columns:
            print(f"{field}: {comparison[match_col].mean():.1%} exact match")
    print(f"Mean categorical row match share: {comparison['categorical_match_share'].mean():.1%}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the manual labeling prompt on manually labeled articles and compare outputs.")
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-5.4-nano"))
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS_PATH)
    parser.add_argument("--articles", type=Path, default=DEFAULT_ARTICLES_PATH)
    parser.add_argument("--system-prompt", type=Path, default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--user-prompt", type=Path, default=DEFAULT_USER_PROMPT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--annotator", default=None, help="Optional annotator_id filter, for example Carlo.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sleep", type=float, default=0.0, help="Seconds to sleep between API calls.")
    parser.add_argument("--max-output-tokens", type=int, default=1800)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print the first formatted prompt and exit without API calls.")
    args = parser.parse_args()

    labels_path = resolve_path(args.labels)
    articles_path = resolve_path(args.articles)
    system_prompt_path = resolve_path(args.system_prompt)
    user_prompt_path = resolve_path(args.user_prompt)
    output_dir = resolve_path(args.output_dir)
    safe_model = args.model.replace("/", "_").replace(":", "_")
    suffix = f"_{args.annotator}" if args.annotator else ""
    predictions_path = output_dir / f"manual_prompt_predictions_{safe_model}{suffix}.jsonl"
    comparison_path = output_dir / f"manual_prompt_comparison_{safe_model}{suffix}.csv"

    rows = prepare_rows(labels_path, articles_path, args.annotator)
    rows = rows[rows["_has_article_text"]].copy()
    if args.limit is not None:
        rows = rows.head(args.limit).copy()
    if rows.empty:
        raise SystemExit("No labeled rows with article text found.")

    system_prompt = system_prompt_path.read_text(encoding="utf-8")
    user_template = user_prompt_path.read_text(encoding="utf-8")

    first_prompt = build_user_prompt(user_template, rows.iloc[0])
    if args.dry_run:
        print("SYSTEM PROMPT:")
        print(system_prompt[:2000])
        print("\nUSER PROMPT:")
        print(first_prompt[:4000])
        print(f"\nRows ready: {len(rows):,}")
        return

    if args.overwrite and predictions_path.exists():
        predictions_path.unlink()
    done = load_existing_predictions(predictions_path)
    if done:
        print(f"Resuming from {predictions_path}: skipping {len(done):,} successful prediction(s).")
    client = OpenAI()

    for row_index, row in enumerate(rows.itertuples(index=False), start=1):
        row_s = pd.Series(row._asdict())
        key = (str(row_s["GlobalEventID"]), str(row_s["annotator_id"]))
        if key in done:
            continue
        user_prompt = build_user_prompt(user_template, row_s)
        try:
            prediction, meta = call_model(client, args.model, system_prompt, user_prompt, args.max_output_tokens)
            record = {
                "GlobalEventID": key[0],
                "annotator_id": key[1],
                "model": args.model,
                "prediction": prediction,
                "response_id": meta.get("response_id", ""),
                "usage": meta.get("usage", {}),
            }
        except (AuthenticationError, BadRequestError) as exc:
            raise SystemExit(f"Stopping after API configuration error for {key[0]}: {exc}") from exc
        except Exception as exc:
            record = {
                "GlobalEventID": key[0],
                "annotator_id": key[1],
                "model": args.model,
                "error": repr(exc),
            }
        append_jsonl(predictions_path, record)
        print(f"{row_index:>4} / {len(rows):<4} {key[0]} {key[1]}")
        if args.sleep:
            time.sleep(args.sleep)

    comparison = build_comparison(predictions_path, labels_path, comparison_path, args.annotator)
    print(f"Wrote predictions: {predictions_path}")
    print(f"Wrote comparison: {comparison_path}")
    print_summary(comparison)


if __name__ == "__main__":
    main()
