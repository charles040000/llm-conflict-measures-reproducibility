from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data/processed"

DEFAULT_EVENT_GROUPS = DATA_DIR / "datasets/02b_strict_rebel_actor_event_groups_high_confidence_with_language.csv"
DEFAULT_ARTICLES = DATA_DIR / "datasets/02b_articles_deduped.csv"
DEFAULT_TRANSLATIONS = REPO_ROOT / "data/translation/translated_non_english_articles.csv"
DEFAULT_OUT = DATA_DIR / "datasets/03_llm_labeling_input_high_confidence_with_translations.csv"
DEFAULT_ARTICLE_OUT = DATA_DIR / "datasets/03_llm_labeling_input_high_confidence_articles_with_translations.csv"
DEFAULT_SUMMARY = REPO_ROOT / "data/reports/translation_join_summary.txt"


ARTICLE_USECOLS = [
    "GlobalEventID",
    "Actor1Name",
    "Actor2Name",
    "Actor1CountryCode",
    "Actor2CountryCode",
    "NumArticles",
    "NumMentions",
    "NumSources",
    "match_sources",
    "text_len",
    "word_count",
    "dup_group_id",
    "dup_group_size",
    "quality_bad_score",
    "quality_bad_reason",
    "text_cleaned",
]

TRANSLATION_USECOLS = [
    "GlobalEventID",
    "translation_status",
    "translation_error",
    "translation_model",
    "source_chars",
    "text_en",
]


def nonempty_text(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().ne("")


def clean_item(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def join_unique(values: pd.Series, max_items: int = 100) -> str:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        for part in clean_item(value).split(";"):
            item = part.strip()
            if not item or item.lower() == "nan" or item in seen:
                continue
            seen.add(item)
            out.append(item)
    out = sorted(out)
    if len(out) <= max_items:
        return "; ".join(out)
    return "; ".join(out[:max_items]) + f"; ... (+{len(out) - max_items})"


def collapse_to_article_level(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse event-actor rows to one row per GlobalEventID."""
    join_cols = [
        "matched_gdelt_actor",
        "matched_actor_sides",
        "matched_aliases",
        "possible_rebel_ids",
        "possible_rebel_names",
        "metadata_locations",
        "metadata_gwno_locs",
        "metadata_regions",
        "metadata_government_sides",
        "metadata_conflict_ids",
        "metadata_dyad_ids",
        "metadata_information_available",
        "metadata_armament_origins",
    ]
    join_cols = [col for col in join_cols if col in df.columns]

    first_cols = [col for col in df.columns if col not in {"GlobalEventID", *join_cols}]
    aggregations: dict[str, object] = {col: "first" for col in first_cols}
    aggregations.update({col: join_unique for col in join_cols})

    article = (
        df.sort_values(["GlobalEventID", "matched_gdelt_actor"])
        .groupby("GlobalEventID", as_index=False, dropna=False)
        .agg(aggregations)
    )
    article["matched_actor_count"] = (
        df.groupby("GlobalEventID")["matched_gdelt_actor"]
        .nunique()
        .reindex(article["GlobalEventID"])
        .to_numpy()
        if "matched_gdelt_actor" in df.columns
        else 0
    )

    preferred_front = [
        "GlobalEventID",
        "matched_gdelt_actor",
        "matched_actor_count",
        "SQLDATE",
        "event_month",
        "source_month",
        "final_url",
        "domain",
        "detected_language",
        "language_name",
        "is_english",
        "translation_status",
        "llm_text_source",
        "has_llm_text",
        "article_text_for_llm",
        "article_text_original",
        "article_text_translated",
    ]
    ordered_cols = [col for col in preferred_front if col in article.columns] + [
        col for col in article.columns if col not in preferred_front
    ]
    return article[ordered_cols]


def run(
    event_groups_path: Path,
    articles_path: Path,
    translations_path: Path,
    out_path: Path,
    article_out_path: Path,
    summary_path: Path,
    drop_unknown_language: bool,
) -> None:
    groups = pd.read_csv(event_groups_path, low_memory=False)
    articles = pd.read_csv(articles_path, usecols=ARTICLE_USECOLS, low_memory=False)
    translations = pd.read_csv(translations_path, usecols=TRANSLATION_USECOLS, low_memory=False)

    articles = articles.drop_duplicates("GlobalEventID").copy()
    translations = translations.drop_duplicates("GlobalEventID").copy()

    articles = articles.rename(columns={"text_cleaned": "article_text_original"})
    translations = translations.rename(columns={"text_en": "article_text_translated"})

    df = groups.merge(articles, on="GlobalEventID", how="left", validate="many_to_one")
    df = df.merge(translations, on="GlobalEventID", how="left", validate="many_to_one")

    is_english = df["detected_language"].fillna("").astype(str).eq("en")
    translated_ok = (
        df["translation_status"].fillna("").astype(str).eq("translated")
        & nonempty_text(df["article_text_translated"])
    )

    df["article_text_for_llm"] = df["article_text_original"].fillna("").astype(str)
    df.loc[translated_ok, "article_text_for_llm"] = df.loc[
        translated_ok, "article_text_translated"
    ].astype(str)

    df["llm_text_source"] = "original_non_english_untranslated"
    df.loc[is_english, "llm_text_source"] = "original_english"
    df.loc[translated_ok, "llm_text_source"] = "translated_to_english"
    df.loc[
        ~is_english & df["translation_status"].fillna("").astype(str).eq("failed"),
        "llm_text_source",
    ] = "translation_failed_original_text"
    df.loc[
        ~is_english & df["detected_language"].fillna("").astype(str).isin(["", "unknown"]),
        "llm_text_source",
    ] = "unknown_language_original_text"

    df["has_llm_text"] = nonempty_text(df["article_text_for_llm"])

    preferred_front = [
        "GlobalEventID",
        "matched_gdelt_actor",
        "SQLDATE",
        "event_month",
        "source_month",
        "final_url",
        "domain",
        "detected_language",
        "language_name",
        "is_english",
        "translation_status",
        "llm_text_source",
        "has_llm_text",
        "article_text_for_llm",
        "article_text_original",
        "article_text_translated",
    ]
    ordered_cols = [col for col in preferred_front if col in df.columns] + [
        col for col in df.columns if col not in preferred_front
    ]
    df = df[ordered_cols]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    article_df = collapse_to_article_level(df)
    article_rows_before_unknown_drop = len(article_df)
    if drop_unknown_language:
        article_df = article_df[
            ~article_df["llm_text_source"].eq("unknown_language_original_text")
        ].copy()
    article_unknown_rows_dropped = article_rows_before_unknown_drop - len(article_df)
    article_out_path.parent.mkdir(parents=True, exist_ok=True)
    article_df.to_csv(article_out_path, index=False)

    source_counts = df["llm_text_source"].value_counts(dropna=False)
    translation_counts = df["translation_status"].fillna("not_queued").value_counts(dropna=False)
    missing_original = int((~nonempty_text(df["article_text_original"])).sum())
    missing_llm_text = int((~df["has_llm_text"]).sum())

    lines = [
        "Translation join summary",
        f"Event groups input: {event_groups_path}",
        f"Deduped articles input: {articles_path}",
        f"Translations input: {translations_path}",
        "",
        f"Output rows: {len(df):,}",
        f"Unique articles: {df['GlobalEventID'].nunique():,}",
        f"Unique matched actors: {df['matched_gdelt_actor'].nunique():,}",
        f"Article-level output rows: {len(article_df):,}",
        f"Article-level unique articles: {article_df['GlobalEventID'].nunique():,}",
        f"Article-level unknown-language rows dropped: {article_unknown_rows_dropped:,}",
        f"Missing original article text rows: {missing_original:,}",
        f"Missing final LLM text rows: {missing_llm_text:,}",
        "",
        "LLM text source counts",
        source_counts.to_string(),
        "",
        "Translation status counts after join",
        translation_counts.to_string(),
        "",
        "Outputs",
        f"Event-actor level output: {out_path}",
        f"Article-level LLM output: {article_out_path}",
    ]
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Join translated non-English text into the high-confidence LLM input dataset."
    )
    parser.add_argument("--event_groups", type=Path, default=DEFAULT_EVENT_GROUPS)
    parser.add_argument("--articles", type=Path, default=DEFAULT_ARTICLES)
    parser.add_argument("--translations", type=Path, default=DEFAULT_TRANSLATIONS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--article_out", type=Path, default=DEFAULT_ARTICLE_OUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument(
        "--keep_unknown_language",
        action="store_true",
        help="Keep rows with unknown language in the article-level LLM output.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(
        event_groups_path=args.event_groups,
        articles_path=args.articles,
        translations_path=args.translations,
        out_path=args.out,
        article_out_path=args.article_out,
        summary_path=args.summary,
        drop_unknown_language=not args.keep_unknown_language,
    )


if __name__ == "__main__":
    main()
