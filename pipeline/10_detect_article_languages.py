from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import pandas as pd
from langdetect import DetectorFactory, detect_langs
from langdetect.lang_detect_exception import LangDetectException


DetectorFactory.seed = 0

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTICLES = REPO_ROOT / "data/processed/datasets/02b_articles_deduped.csv"
DEFAULT_HIGH_CONF_GROUPS = REPO_ROOT / "data/processed/datasets/02b_strict_rebel_actor_event_groups_high_confidence.csv"
DEFAULT_HIGH_CONF_GROUPS_WITH_LANGUAGE = REPO_ROOT / "data/processed/datasets/02b_strict_rebel_actor_event_groups_high_confidence_with_language.csv"
DEFAULT_OUT = REPO_ROOT / "data/reports/article_language_detection.csv"
DEFAULT_SUMMARY_OUT = REPO_ROOT / "data/reports/article_language_distribution.csv"
DEFAULT_REPORT_OUT = REPO_ROOT / "data/reports/article_language_detection_summary.txt"


LANG_NAMES = {
    "af": "Afrikaans",
    "ar": "Arabic",
    "bg": "Bulgarian",
    "ca": "Catalan",
    "cs": "Czech",
    "cy": "Welsh",
    "da": "Danish",
    "de": "German",
    "el": "Greek",
    "en": "English",
    "es": "Spanish",
    "et": "Estonian",
    "fa": "Persian",
    "fi": "Finnish",
    "fr": "French",
    "he": "Hebrew",
    "hi": "Hindi",
    "hr": "Croatian",
    "hu": "Hungarian",
    "id": "Indonesian",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "lt": "Lithuanian",
    "lv": "Latvian",
    "mk": "Macedonian",
    "nl": "Dutch",
    "no": "Norwegian",
    "pl": "Polish",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "sk": "Slovak",
    "sl": "Slovenian",
    "so": "Somali",
    "sq": "Albanian",
    "sv": "Swedish",
    "sw": "Swahili",
    "ta": "Tamil",
    "te": "Telugu",
    "th": "Thai",
    "tl": "Tagalog",
    "tr": "Turkish",
    "uk": "Ukrainian",
    "ur": "Urdu",
    "vi": "Vietnamese",
    "zh-cn": "Chinese",
    "zh-tw": "Chinese",
    "unknown": "Unknown",
}


def clean_text(value: Any, max_chars: int) -> str:
    if pd.isna(value):
        return ""
    text = re.sub(r"\s+", " ", str(value)).strip()
    return text[:max_chars]


def detect_one(text: str) -> tuple[str, float, str]:
    if not text or len(text) < 40:
        return "unknown", 0.0, ""
    try:
        langs = detect_langs(text)
    except LangDetectException:
        return "unknown", 0.0, ""
    if not langs:
        return "unknown", 0.0, ""
    top = langs[0]
    return top.lang, float(top.prob), ";".join(f"{item.lang}:{item.prob:.3f}" for item in langs[:3])


def summarize(df: pd.DataFrame, subset_name: str) -> pd.DataFrame:
    total = len(df)
    out = (
        df.groupby("detected_language", dropna=False)
        .agg(
            articles=("GlobalEventID", "nunique"),
            avg_probability=("language_probability", "mean"),
        )
        .reset_index()
        .sort_values("articles", ascending=False)
    )
    out["subset"] = subset_name
    out["language_name"] = out["detected_language"].map(lambda x: LANG_NAMES.get(str(x), str(x)))
    out["share"] = out["articles"] / total if total else 0
    out["share_pct"] = out["share"] * 100
    return out[["subset", "detected_language", "language_name", "articles", "share", "share_pct", "avg_probability"]]


def run(
    articles_path: Path,
    high_conf_groups_path: Path,
    out_path: Path,
    summary_out: Path,
    report_out: Path,
    high_conf_groups_with_language_out: Path,
    max_chars: int,
) -> None:
    articles = pd.read_csv(
        articles_path,
        usecols=["GlobalEventID", "SQLDATE", "source_month", "domain", "final_url", "text"],
        low_memory=False,
    )
    articles = articles.drop_duplicates("GlobalEventID").reset_index(drop=True)

    detections = []
    for text in articles["text"].map(lambda x: clean_text(x, max_chars=max_chars)):
        detections.append(detect_one(text))

    articles["detected_language"] = [x[0] for x in detections]
    articles["language_probability"] = [x[1] for x in detections]
    articles["language_candidates"] = [x[2] for x in detections]
    articles["is_english"] = articles["detected_language"].eq("en")

    if high_conf_groups_path.exists():
        high_conf_ids = set(
            pd.read_csv(high_conf_groups_path, usecols=["GlobalEventID"], low_memory=False)["GlobalEventID"]
            .dropna()
            .unique()
        )
    else:
        high_conf_ids = set()

    articles["in_high_conf_rebel_actor_dataset"] = articles["GlobalEventID"].isin(high_conf_ids)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    articles.drop(columns=["text"]).to_csv(out_path, index=False)

    if high_conf_groups_path.exists():
        groups = pd.read_csv(high_conf_groups_path, low_memory=False)
        language_cols = [
            "GlobalEventID",
            "detected_language",
            "language_name",
            "language_probability",
            "language_candidates",
            "is_english",
        ]
        groups = groups.merge(
            articles[language_cols],
            on="GlobalEventID",
            how="left",
        )
        high_conf_groups_with_language_out.parent.mkdir(parents=True, exist_ok=True)
        groups.to_csv(high_conf_groups_with_language_out, index=False)

    full_summary = summarize(articles, "full_deduped_02b")
    high_conf = articles[articles["in_high_conf_rebel_actor_dataset"]].copy()
    high_conf_summary = summarize(high_conf, "strict_rebel_actor_high_conf_unique_articles")
    summary = pd.concat([full_summary, high_conf_summary], ignore_index=True)
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_out, index=False)

    def block(name: str, frame: pd.DataFrame) -> list[str]:
        total = len(frame)
        english = int(frame["is_english"].sum())
        non_english = total - english
        lines = [
            name,
            f"Total articles: {total:,}",
            f"English: {english:,} ({english / total * 100:.2f}%)" if total else "English: 0",
            f"Non-English/unknown: {non_english:,} ({non_english / total * 100:.2f}%)" if total else "Non-English/unknown: 0",
            "",
            "Language distribution",
            summarize(frame, name)[["detected_language", "language_name", "articles", "share_pct"]]
            .to_string(index=False, formatters={"share_pct": "{:.2f}".format}),
        ]
        return lines

    lines = [
        "Article language detection summary",
        f"Detector: langdetect, deterministic seed=0",
        f"Input articles: {articles_path}",
        f"Text characters used per article: first {max_chars:,}",
        "",
    ]
    lines.extend(block("full_deduped_02b", articles))
    lines.extend(["", ""])
    lines.extend(block("strict_rebel_actor_high_conf_unique_articles", high_conf))
    lines.extend(
        [
            "",
            "Outputs",
            f"Per-article language detection: {out_path}",
            f"High-confidence event-actor groups with language: {high_conf_groups_with_language_out}",
            f"Language distribution summary: {summary_out}",
        ]
    )
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect article languages for deduped and strict matched article sets.")
    parser.add_argument("--articles", type=Path, default=DEFAULT_ARTICLES)
    parser.add_argument("--high_conf_groups", type=Path, default=DEFAULT_HIGH_CONF_GROUPS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--summary_out", type=Path, default=DEFAULT_SUMMARY_OUT)
    parser.add_argument("--report_out", type=Path, default=DEFAULT_REPORT_OUT)
    parser.add_argument("--high_conf_groups_with_language_out", type=Path, default=DEFAULT_HIGH_CONF_GROUPS_WITH_LANGUAGE)
    parser.add_argument("--max_chars", type=int, default=3000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(
        articles_path=args.articles,
        high_conf_groups_path=args.high_conf_groups,
        out_path=args.out,
        summary_out=args.summary_out,
        report_out=args.report_out,
        high_conf_groups_with_language_out=args.high_conf_groups_with_language_out,
        max_chars=args.max_chars,
    )


if __name__ == "__main__":
    main()
