from __future__ import annotations

import argparse
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "data/processed/datasets/02b_articles_deduped.csv"
DEFAULT_TERMS = REPO_ROOT / "data/reference/cleaned_actor_terms.csv"
DEFAULT_MATCHES_OUT = REPO_ROOT / "data/processed/datasets/02b_strict_rebel_actor_location_matches.csv"
DEFAULT_ARTICLES_OUT = REPO_ROOT / "data/processed/datasets/02b_strict_rebel_actor_articles.csv"
DEFAULT_ACTOR_SUMMARY_OUT = REPO_ROOT / "data/reports/strict_rebel_actor_distribution.csv"
DEFAULT_LOCATION_SUMMARY_OUT = REPO_ROOT / "data/reports/strict_rebel_actor_location_distribution.csv"
DEFAULT_HIGH_CONF_OUT = REPO_ROOT / "data/processed/datasets/02b_strict_rebel_actor_location_matches_high_confidence.csv"
DEFAULT_REPORT_OUT = REPO_ROOT / "data/reports/strict_rebel_actor_location_summary.txt"


KEEP_ARTICLE_COLUMNS = [
    "GlobalEventID",
    "SQLDATE",
    "DATEADDED",
    "Actor1Name",
    "Actor2Name",
    "Actor1CountryCode",
    "Actor2CountryCode",
    "ActionGeo_CountryCode",
    "EventCode",
    "EventBaseCode",
    "EventRootCode",
    "QuadClass",
    "NumArticles",
    "NumMentions",
    "NumSources",
    "AvgTone",
    "GoldsteinScale",
    "match_sources",
    "final_url",
    "domain",
    "text_len",
    "word_count",
    "source_month",
    "dup_group_id",
    "dup_group_size",
    "quality_bad_score",
    "quality_bad_reason",
]


# These are exact aliases in the reference table, but they are too broad,
# geographic, or source-artifact-like for direct location attachment without
# manual review. The strict dataset keeps them and flags them; the high
# confidence subset excludes them.
SUSPICIOUS_ALIAS_NORMS = {
    "estados unidos",
    "liberation army",
    "opposition coalition",
    "palestine",
    "republic of croatia",
    "republic of georgia",
    "tibet",
}

SUSPICIOUS_SOURCE_COLUMNS = {
    "NameAlliance",
}


def clean_value(value: Any) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def strip_diacritics(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def normalize_phrase(value: Any) -> str:
    value = strip_diacritics(clean_value(value).lower())
    value = value.replace("&", " and ")
    value = re.sub(r"['’`´]", "", value)
    value = re.sub(r"[^a-z0-9]+", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    value = re.sub(r"^(the|a|an)\s+", "", value)
    return value


def compact_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", normalize_phrase(value))


def acronym_key(value: Any) -> str:
    phrase = normalize_phrase(value)
    parts = phrase.split()
    if len(parts) >= 2 and all(len(part) == 1 for part in parts):
        return "".join(parts)
    return ""


def bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def load_terms(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Actor terms file not found: {path}")
    terms = pd.read_csv(path, low_memory=False)

    required = {"term", "term_source_dataset", "source_actor_id", "source_actor_name"}
    missing = required - set(terms.columns)
    if missing:
        raise ValueError(f"Missing expected columns in {path}: {sorted(missing)}")

    if "use_for_matching" in terms.columns:
        terms = terms[terms["use_for_matching"].map(bool_value)].copy()

    terms = terms[terms["term_source_dataset"].astype(str).isin(["RAD", "UCDP_ACTOR"])].copy()
    terms["alias"] = terms["term"].map(clean_value)
    terms["alias_norm"] = terms["alias"].map(normalize_phrase)
    terms["alias_compact"] = terms["alias"].map(compact_key)
    terms = terms[terms["alias_norm"].ne("")].copy()

    cols = [
        "alias",
        "alias_norm",
        "alias_compact",
        "term_source_dataset",
        "term_source_column",
        "source_actor_id",
        "source_actor_name",
        "source_org",
    ]
    for col in cols:
        if col not in terms.columns:
            terms[col] = ""
    terms = terms[cols].drop_duplicates().reset_index(drop=True)
    return terms


def build_lookup(terms: pd.DataFrame) -> dict[str, list[dict[str, Any]]]:
    lookup: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen: set[tuple[str, str, str, str]] = set()

    for rec in terms.to_dict("records"):
        source_id = clean_value(rec.get("source_actor_id"))
        alias = clean_value(rec.get("alias"))
        source_dataset = clean_value(rec.get("term_source_dataset"))

        for match_key_type, match_key in [
            ("phrase", clean_value(rec.get("alias_norm"))),
            ("compact", clean_value(rec.get("alias_compact"))),
        ]:
            if not match_key:
                continue
            dedup_key = (match_key_type, match_key, source_id, alias)
            if dedup_key in seen:
                continue
            seen.add(dedup_key)
            item = dict(rec)
            item["match_key_type"] = match_key_type
            item["match_key"] = match_key
            lookup[match_key].append(item)

    return dict(lookup)


def actor_match_keys(actor: str) -> list[tuple[str, str]]:
    phrase = normalize_phrase(actor)
    compact = compact_key(actor)
    acronym = acronym_key(actor)

    keys: list[tuple[str, str]] = []
    if phrase:
        keys.append(("phrase", phrase))
    if compact and compact != phrase.replace(" ", ""):
        keys.append(("compact", compact))
    elif compact and any(ch in actor for ch in ".'’`´-_/ "):
        keys.append(("compact", compact))
    if acronym and acronym != compact:
        keys.append(("acronym", acronym))

    out: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for key_type, key in keys:
        item = (key_type, key)
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def find_actor_matches(actor: str, lookup: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()

    for actor_key_type, key in actor_match_keys(actor):
        for rec in lookup.get(key, []):
            source_id = clean_value(rec.get("source_actor_id"))
            source_name = clean_value(rec.get("source_actor_name"))
            alias = clean_value(rec.get("alias"))
            source_dataset = clean_value(rec.get("term_source_dataset"))
            dedup_key = (source_id, source_name, alias, source_dataset)
            if dedup_key in seen:
                continue
            seen.add(dedup_key)
            item = dict(rec)
            item["actor_match_key_type"] = actor_key_type
            item["actor_match_key"] = key
            matches.append(item)

    return matches


def create_outputs(
    input_path: Path,
    terms_path: Path,
    matches_out: Path,
    articles_out: Path,
    actor_summary_out: Path,
    location_summary_out: Path,
    high_conf_out: Path,
    report_out: Path,
) -> None:
    df = pd.read_csv(input_path, low_memory=False)
    terms = load_terms(terms_path)
    lookup = build_lookup(terms)

    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        for side_num in (1, 2):
            actor = clean_value(row.get(f"Actor{side_num}Name"))
            if not actor:
                continue
            side_matches = find_actor_matches(actor, lookup)
            if not side_matches:
                continue

            other_num = 2 if side_num == 1 else 1
            base = {col: row.get(col) for col in KEEP_ARTICLE_COLUMNS if col in df.columns}
            for match in side_matches:
                out = dict(base)
                out.update(
                    {
                        "matched_actor_side": f"Actor{side_num}Name",
                        "matched_gdelt_actor": actor,
                        "matched_gdelt_actor_norm": normalize_phrase(actor),
                        "matched_gdelt_actor_compact": compact_key(actor),
                        "other_actor": clean_value(row.get(f"Actor{other_num}Name")),
                        "other_actor_country_code": row.get(f"Actor{other_num}CountryCode"),
                        "matched_actor_country_code": row.get(f"Actor{side_num}CountryCode"),
                        "matched_alias": clean_value(match.get("alias")),
                        "matched_alias_norm": clean_value(match.get("alias_norm")),
                        "matched_alias_compact": clean_value(match.get("alias_compact")),
                        "matched_alias_source_dataset": clean_value(match.get("term_source_dataset")),
                        "matched_alias_source_column": clean_value(match.get("term_source_column")),
                        "rebel_source_actor_id": clean_value(match.get("source_actor_id")),
                        "rebel_source_actor_name": clean_value(match.get("source_actor_name")),
                        "rebel_source_org": clean_value(match.get("source_org")),
                        "match_key_type": clean_value(match.get("actor_match_key_type")),
                        "match_key": clean_value(match.get("actor_match_key")),
                    }
                )
                rows.append(out)

    matches = pd.DataFrame(rows)
    if len(matches):
        alias_actor_counts = matches.groupby("matched_alias_norm")["rebel_source_actor_id"].transform("nunique")
        matches["alias_maps_to_multiple_rebel_ids"] = alias_actor_counts.gt(1)
        matches["alias_is_suspicious_broad_or_geographic"] = matches["matched_alias_norm"].isin(SUSPICIOUS_ALIAS_NORMS)
        matches["alias_source_column_needs_review"] = matches["matched_alias_source_column"].isin(SUSPICIOUS_SOURCE_COLUMNS)

        def review_flag(row: pd.Series) -> str:
            flags: list[str] = []
            if bool(row["alias_is_suspicious_broad_or_geographic"]):
                flags.append("suspicious_broad_or_geographic_alias")
            if bool(row["alias_source_column_needs_review"]):
                flags.append("alliance_alias")
            if bool(row["alias_maps_to_multiple_rebel_ids"]):
                flags.append("alias_maps_to_multiple_rebel_ids")
            return ";".join(flags) if flags else "clean_exact_alias"

        matches["alias_review_flag"] = matches.apply(review_flag, axis=1)
        matches["recommended_high_confidence"] = (
            ~matches["alias_is_suspicious_broad_or_geographic"]
            & ~matches["alias_source_column_needs_review"]
        )
    else:
        matches["alias_maps_to_multiple_rebel_ids"] = []
        matches["alias_is_suspicious_broad_or_geographic"] = []
        matches["alias_source_column_needs_review"] = []
        matches["alias_review_flag"] = []
        matches["recommended_high_confidence"] = []

    matches_out.parent.mkdir(parents=True, exist_ok=True)
    matches.to_csv(matches_out, index=False)

    high_conf = matches[matches["recommended_high_confidence"]].copy() if len(matches) else matches.copy()
    high_conf_out.parent.mkdir(parents=True, exist_ok=True)
    high_conf.to_csv(high_conf_out, index=False)

    article_ids = sorted(matches["GlobalEventID"].dropna().unique()) if len(matches) else []
    articles = df[df["GlobalEventID"].isin(article_ids)].copy() if len(article_ids) else df.iloc[0:0].copy()
    articles_out.parent.mkdir(parents=True, exist_ok=True)
    articles.to_csv(articles_out, index=False)

    if len(matches):
        actor_summary = (
            matches.groupby(["rebel_source_actor_id", "rebel_source_actor_name"], dropna=False)
            .agg(
                match_rows=("GlobalEventID", "size"),
                unique_events=("GlobalEventID", "nunique"),
                unique_event_locations=("ActionGeo_CountryCode", "nunique"),
                actor1_matches=("matched_actor_side", lambda s: int((s == "Actor1Name").sum())),
                actor2_matches=("matched_actor_side", lambda s: int((s == "Actor2Name").sum())),
                matched_gdelt_actors=("matched_gdelt_actor", lambda s: "; ".join(sorted(set(map(str, s)))[:20])),
                matched_aliases=("matched_alias", lambda s: "; ".join(sorted(set(map(str, s)))[:20])),
                source_datasets=("matched_alias_source_dataset", lambda s: "; ".join(sorted(set(map(str, s))))),
                alias_review_flags=("alias_review_flag", lambda s: "; ".join(sorted(set(map(str, s))))),
                recommended_high_confidence_rows=("recommended_high_confidence", "sum"),
                top_event_location=("ActionGeo_CountryCode", lambda s: s.fillna("").replace("", pd.NA).value_counts().index[0] if s.fillna("").replace("", pd.NA).notna().any() else ""),
            )
            .reset_index()
        )
        total_unique_events = matches["GlobalEventID"].nunique()
        actor_summary["unique_event_share"] = actor_summary["unique_events"] / total_unique_events
        actor_summary = actor_summary.sort_values(["unique_events", "match_rows"], ascending=False)

        loc_summary = (
            matches.groupby(
                ["rebel_source_actor_id", "rebel_source_actor_name", "ActionGeo_CountryCode"],
                dropna=False,
            )
            .agg(
                match_rows=("GlobalEventID", "size"),
                unique_events=("GlobalEventID", "nunique"),
                recommended_high_confidence_rows=("recommended_high_confidence", "sum"),
                matched_gdelt_actors=("matched_gdelt_actor", lambda s: "; ".join(sorted(set(map(str, s)))[:15])),
            )
            .reset_index()
            .sort_values(["unique_events", "match_rows"], ascending=False)
        )
    else:
        actor_summary = pd.DataFrame()
        loc_summary = pd.DataFrame()

    actor_summary_out.parent.mkdir(parents=True, exist_ok=True)
    actor_summary.to_csv(actor_summary_out, index=False)
    location_summary_out.parent.mkdir(parents=True, exist_ok=True)
    loc_summary.to_csv(location_summary_out, index=False)

    obvious_false_positive_actors = ["TRADE UNION", "THE EUROPEAN UNION", "EUROPEAN UNION", "SECURITY FORCE", "WEBSITE", "OIL TANKER"]
    false_positive_hits = {}
    if len(matches):
        for actor in obvious_false_positive_actors:
            false_positive_hits[actor] = int(matches["matched_gdelt_actor"].astype(str).str.upper().eq(actor).sum())

    lines = [
        "Strict normalized rebel actor-location match summary",
        f"Input dataset: {input_path}",
        f"Input rows/articles: {len(df):,}",
        f"Terms reference: {terms_path}",
        f"Reference aliases loaded: {len(terms):,}",
        "",
        "Match rule",
        "A GDELT Actor1Name/Actor2Name is matched only when the full actor label equals a full RAD/UCDP alias after conservative normalization.",
        "Normalization lowercases, strips diacritics, removes apostrophes/punctuation, collapses whitespace, and also compares compact acronym-like keys.",
        "Substring-only matches are not accepted.",
        "",
        "Output counts",
        f"Actor-location match rows: {len(matches):,}",
        f"Unique matched articles/events: {matches['GlobalEventID'].nunique() if len(matches) else 0:,}",
        f"Unique rebel source actor IDs: {actor_summary['rebel_source_actor_id'].nunique() if len(actor_summary) else 0:,}",
        f"Articles with >=1 strict rebel actor match: {len(articles):,}",
        f"Recommended high-confidence match rows: {len(high_conf):,}",
        f"Recommended high-confidence unique events: {high_conf['GlobalEventID'].nunique() if len(high_conf) else 0:,}",
        "",
        "False-positive spot checks; expected to be 0 under strict matching",
    ]
    for actor, count in false_positive_hits.items():
        lines.append(f"{actor}: {count}")

    if len(actor_summary):
        lines.extend(
            [
                "",
                "Rows by alias_review_flag",
                matches["alias_review_flag"].value_counts().to_string(),
                "",
                "Top 30 rebel actors by unique_events",
                actor_summary.head(30).to_string(index=False),
            ]
        )
    if len(loc_summary):
        lines.extend(
            [
                "",
                "Top 30 rebel actor-location pairs by unique_events",
                loc_summary.head(30).to_string(index=False),
            ]
        )
    lines.extend(
        [
            "",
            "Outputs",
            f"Actor-location match rows: {matches_out}",
            f"Recommended high-confidence match rows: {high_conf_out}",
            f"Article subset with at least one strict match: {articles_out}",
            f"Actor distribution: {actor_summary_out}",
            f"Actor-location distribution: {location_summary_out}",
        ]
    )
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create strict rebel actor-location matches from deduped articles.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--terms", type=Path, default=DEFAULT_TERMS)
    parser.add_argument("--matches_out", type=Path, default=DEFAULT_MATCHES_OUT)
    parser.add_argument("--articles_out", type=Path, default=DEFAULT_ARTICLES_OUT)
    parser.add_argument("--actor_summary_out", type=Path, default=DEFAULT_ACTOR_SUMMARY_OUT)
    parser.add_argument("--location_summary_out", type=Path, default=DEFAULT_LOCATION_SUMMARY_OUT)
    parser.add_argument("--high_conf_out", type=Path, default=DEFAULT_HIGH_CONF_OUT)
    parser.add_argument("--report_out", type=Path, default=DEFAULT_REPORT_OUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    create_outputs(
        input_path=args.input,
        terms_path=args.terms,
        matches_out=args.matches_out,
        articles_out=args.articles_out,
        actor_summary_out=args.actor_summary_out,
        location_summary_out=args.location_summary_out,
        high_conf_out=args.high_conf_out,
        report_out=args.report_out,
    )


if __name__ == "__main__":
    main()
