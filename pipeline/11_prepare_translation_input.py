from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data/processed"

DEFAULT_ARTICLES = DATA_DIR / "datasets/02b_articles_deduped.csv"
DEFAULT_LANG = REPO_ROOT / "data/reports/article_language_detection.csv"
DEFAULT_GROUPS = DATA_DIR / "datasets/02b_strict_rebel_actor_event_groups_high_confidence_with_language.csv"
DEFAULT_MAPPING = REPO_ROOT / "data/reports/proposed_rebel_actor_manual_mapping.csv"
DEFAULT_OUT_DIR = REPO_ROOT / "data/translation"


def prepare_package(
    articles_path: Path,
    language_path: Path,
    groups_path: Path,
    mapping_path: Path,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    langs = pd.read_csv(language_path, low_memory=False)
    if "in_high_conf_rebel_actor_dataset" in langs.columns:
        langs = langs[langs["in_high_conf_rebel_actor_dataset"].fillna(False).astype(bool)].copy()

    non_english = langs[
        langs["detected_language"].fillna("").astype(str).ne("en")
        & langs["detected_language"].fillna("").astype(str).ne("unknown")
        & langs["detected_language"].fillna("").astype(str).ne("")
    ].copy()

    ids = set(non_english["GlobalEventID"].dropna().unique())
    articles = pd.read_csv(
        articles_path,
        usecols=["GlobalEventID", "SQLDATE", "source_month", "domain", "final_url", "text_cleaned"],
        low_memory=False,
    )
    articles = articles[articles["GlobalEventID"].isin(ids)].drop_duplicates("GlobalEventID").copy()
    out = articles.merge(
        non_english[
            [
                "GlobalEventID",
                "detected_language",
                "language_name",
                "language_probability",
                "language_candidates",
            ]
        ],
        on="GlobalEventID",
        how="inner",
    )
    out["text_cleaned"] = out["text_cleaned"].fillna("").astype(str)
    out["is_canonical"] = True
    out["translate_queue"] = True
    out = out[
        [
            "GlobalEventID",
            "SQLDATE",
            "source_month",
            "domain",
            "final_url",
            "detected_language",
            "language_name",
            "language_probability",
            "language_candidates",
            "is_canonical",
            "translate_queue",
            "text_cleaned",
        ]
    ].sort_values(["detected_language", "GlobalEventID"])

    input_out = out_dir / "non_english_articles_to_translate.csv"
    out.to_csv(input_out, index=False)

    if groups_path.exists():
        pd.read_csv(groups_path, low_memory=False).to_csv(
            out_dir / "event_groups_high_confidence_with_language.csv",
            index=False,
        )
    if mapping_path.exists():
        pd.read_csv(mapping_path, low_memory=False).to_csv(
            out_dir / "proposed_rebel_actor_manual_mapping.csv",
            index=False,
        )

    summary = [
        "Server translation package",
        f"Input article file: {articles_path}",
        f"Language file: {language_path}",
        f"Output directory: {out_dir}",
        "",
        f"Non-English known-language articles queued: {len(out):,}",
        f"Unique GlobalEventID queued: {out['GlobalEventID'].nunique():,}",
        f"English rows included: {int(out['detected_language'].eq('en').sum()):,}",
        f"Unknown-language rows included: {int(out['detected_language'].eq('unknown').sum()):,}",
        "",
        "Queued language counts",
        out["detected_language"].value_counts().to_string(),
        "",
        "Files to transfer to server",
        str(input_out),
        str(out_dir / "event_groups_high_confidence_with_language.csv"),
        str(out_dir / "proposed_rebel_actor_manual_mapping.csv"),
        str(REPO_ROOT / "pipeline/12_translate_non_english_articles.py"),
        str(out_dir / "requirements_translation.txt"),
    ]
    (out_dir / "translation_package_summary.txt").write_text("\n".join(summary), encoding="utf-8")
    (out_dir / "requirements_translation.txt").write_text(
        "pandas\n"
        "deep-translator\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare non-English matched articles for server-side translation.")
    parser.add_argument("--articles", type=Path, default=DEFAULT_ARTICLES)
    parser.add_argument("--language", type=Path, default=DEFAULT_LANG)
    parser.add_argument("--groups", type=Path, default=DEFAULT_GROUPS)
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prepare_package(
        articles_path=args.articles,
        language_path=args.language,
        groups_path=args.groups,
        mapping_path=args.mapping,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
