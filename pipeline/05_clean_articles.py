"""
05_clean_articles.py

Language-agnostic boilerplate removal for raw scraped article text.

Adds a `text_cleaned` column to 02_articles_raw.csv (in-place).
All downstream steps use `text_cleaned` rather than `text`.

What is removed
---------------
Sentence-level fragments matching any of:
  - Email addresses
  - Phone numbers
  - Bare URLs
  - Timestamp / view-count strings (e.g. "15:53 • 2322 просмотра")
  - Copyright and rights-reserved lines
  - Tag / share / subscribe navigation lines
  - Any fragment where fewer than 40% of characters are alphabetic
    (catches emoji-heavy lines, symbol separators, etc.)

The original `text` column is never modified.

Usage
-----
python pipeline/05_clean_articles.py
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import re
import os
import time
import unicodedata
from pathlib import Path

os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import pandas as pd

BASE_DIR   = Path(__file__).resolve().parents[1]
DATA_ROOT  = Path(os.getenv("GDELT_DATA_ROOT", str(BASE_DIR / "data" / "raw")))
INPUT_FILE = DATA_ROOT / "datasets" / "02_articles_raw.csv"
SUMMARIES_DIR = DATA_ROOT / "summaries"
SUMMARIES_DIR.mkdir(parents=True, exist_ok=True)
OUT_SUMMARY = SUMMARIES_DIR / "cleaning_summary.txt"


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return int(value)


CLEAN_WORKERS = max(1, _env_int("GDELT_CLEAN_WORKERS", min(os.cpu_count() or 1, 8)))
CLEAN_CHUNKSIZE = max(1, _env_int("GDELT_CLEAN_CHUNKSIZE", 100))


def _process_pool_kwargs() -> dict:
    method = os.getenv("GDELT_PROCESS_START_METHOD")
    if method is None:
        method = "fork" if os.name == "posix" else ""
    return {"mp_context": mp.get_context(method)} if method else {}

# ---------------------------------------------------------------------------
# Boilerplate patterns
# ---------------------------------------------------------------------------

_BOILERPLATE_PATTERNS: list[re.Pattern] = [
    # Email addresses
    re.compile(r"\S+@\S+\.\S+"),
    # Phone numbers
    re.compile(r"\+?\d[\d\s\-\(\)\.]{6,}"),
    # Bare URLs
    re.compile(r"https?://\S+"),
    # Lines with no word characters at all
    re.compile(r"^[^\w]*$"),
    # Timestamp / view-count strings e.g. "15:53 • 2322 views"
    re.compile(r"\d{1,2}:\d{2}\s*[•·]\s*\d+"),
    # Copyright lines
    re.compile(r"©|\bAll rights reserved\b|\bCopyright\b", re.IGNORECASE),
    # Navigation / social share lines
    re.compile(
        r"^(Tags|Share|Follow|Subscribe|Newsletter|Related|Read more|"
        r"See also|More on|Click here)",
        re.IGNORECASE,
    ),
    # Reading time e.g. "3 min read" / "5 minute read"
    re.compile(r"^\d+\s+min(ute)?\s+read$", re.IGNORECASE),
    # Social follow prompts
    re.compile(r"\b(follow us on|like us on|find us on|join us on)\b", re.IGNORECASE),
    # Comment count artifacts e.g. "47 comments" / "Leave a comment"
    re.compile(r"^\d+\s+comments?$", re.IGNORECASE),
    re.compile(r"^leave a comment", re.IGNORECASE),
    # Advertisement / sponsored markers
    re.compile(r"^(advertisement|sponsored|promoted content|paid content)$", re.IGNORECASE),
    # Pagination e.g. "Page 1 of 3"
    re.compile(r"^page\s+\d+\s+of\s+\d+", re.IGNORECASE),
    # Cookie consent fragments
    re.compile(
        r"\b(we use cookies|cookie policy|gdpr|privacy choices|privacy dashboard|"
        r"manage privacy settings|datenschutzeinstellungen|datenschutz-dashboard|"
        r"alle akzeptieren|alle ablehnen|cookie-einstellungen)\b",
        re.IGNORECASE,
    ),
    # Paywall / subscription prompts
    re.compile(r"\b(subscribe to (read|continue|access)|subscribers only|premium content)\b", re.IGNORECASE),
    re.compile(
        r"\b(log in the star|digital access|cancel anytime|"
        r"already a subscriber|unlimited access with perks|"
        r"become a member|read all member content|get exclusive in-depth reports|"
        r"connect with israel,\s*right from your home|lift up the voice of truth and hope)\b",
        re.IGNORECASE,
    ),
    re.compile(r"^\s*the times of israel community\.?\s*$", re.IGNORECASE),
    # Publisher/legal boilerplate
    re.compile(
        r"\b(this copy is for your personal,\s*non-commercial use only|"
        r"for non-personal use or to order multiple copies|"
        r"a previous version of this article reported|"
        r"we are happy to clarify this and apologise for the error|"
        r"except for the headline,\s*this story has not been edited)\b",
        re.IGNORECASE,
    ),
    # Syndication/photo/chrome fragments observed in scraped GDELT articles.
    re.compile(r"\bread full story\b", re.IGNORECASE),
    re.compile(r"^\s*photo\s*:", re.IGNORECASE),
    re.compile(r"\b(see more daily mail on google|save us as a preferred source)\b", re.IGNORECASE),
    # Generic page chrome
    re.compile(r"\b(more top stories|featured videos|what do you want to cook today)\b", re.IGNORECASE),
]

_TRASH_PAGE_PATTERNS: list[re.Pattern] = [
    re.compile(r"\b(yahoo is part of the yahoo family of brands|ihre privatsph(?:ä|ae)re ist uns wichtig)\b", re.IGNORECASE),
    re.compile(r"\b(consent\.yahoo\.com|privacy choices|privacy dashboard|datenschutzeinstellungen)\b", re.IGNORECASE),
    re.compile(r"\b(page not found|nothing found|couldn'?t be found|404 ooops|access denied|enable javascript)\b", re.IGNORECASE),
]

_MOJIBAKE_MARKERS = set("ÃÂâ€š„™œØÙÐÑ")
_SEVERE_MOJIBAKE_MARKERS = set("ØÙÐÑ")
_REPLACEMENT_CHAR = "\ufffd"
_MOJIBAKE_SEQUENCE_PATTERN = re.compile(r"Ã.|Â.|â[\u0080-\u009f€]|[ØÙÐÑ]")


def _marker_count(text: str, markers: set[str]) -> int:
    return sum(ch in markers for ch in text)


def looks_like_mojibake(text: str) -> bool:
    if not text:
        return False
    marker_count = _marker_count(text, _MOJIBAKE_MARKERS)
    severe_count = _marker_count(text, _SEVERE_MOJIBAKE_MARKERS)
    sequence_count = len(_MOJIBAKE_SEQUENCE_PATTERN.findall(text))
    text_len = max(len(text), 1)
    return (
        (sequence_count >= 3 and marker_count >= 3)
        or
        (marker_count >= 20 and marker_count / text_len >= 0.02)
        or (severe_count >= 8 and severe_count / text_len >= 0.01)
    )


def looks_like_trash_page(text: str) -> bool:
    if not text:
        return False
    return any(pattern.search(text) for pattern in _TRASH_PAGE_PATTERNS)


def _corruption_score(text: str) -> tuple[int, int, int]:
    return (
        _marker_count(text, _MOJIBAKE_MARKERS),
        text.count(_REPLACEMENT_CHAR) * 5,
        sum(1 for ch in text if unicodedata.category(ch)[0] == "C"),
    )


def repair_mojibake(text: str) -> str:
    """
    Repair common UTF-8-as-Latin-1/Windows-1252 mojibake. If too many bytes
    were already lost, return an empty string so downstream quality filtering
    can drop the row instead of translating garbage.
    """
    if not looks_like_mojibake(text):
        return text

    candidates = [text]
    for encoding in ("latin1", "cp1252"):
        try:
            candidates.append(text.encode(encoding).decode("utf-8"))
        except UnicodeError:
            candidates.append(text.encode(encoding, errors="ignore").decode("utf-8", errors="replace"))

    best = min(candidates, key=_corruption_score)
    if best is text:
        return ""

    replacement_count = best.count(_REPLACEMENT_CHAR)
    if replacement_count > 20 or replacement_count / max(len(best), 1) > 0.02:
        return ""
    if looks_like_mojibake(best):
        return ""
    return best.replace(_REPLACEMENT_CHAR, "")


def _is_boilerplate(fragment: str) -> bool:
    for pattern in _BOILERPLATE_PATTERNS:
        if pattern.search(fragment):
            return True
    letters = sum(1 for c in fragment if c.isalpha())
    if len(fragment) > 0 and letters / len(fragment) < 0.40:
        return True
    return False


def clean_article(text: str) -> str:
    """
    Remove boilerplate sentence fragments from article text.
    Returns cleaned text, or original text if nothing survives filtering.
    """
    if not text or not text.strip():
        return text

    text = repair_mojibake(text)
    if not text.strip():
        return ""

    if looks_like_trash_page(text):
        return ""

    # Primary split: sentence-ending punctuation
    parts = re.split(r"(?<=[.!?…])\s+", text)
    kept = [s.strip() for s in parts if len(s.strip()) >= 20 and not _is_boilerplate(s.strip())]

    # Fallback: newline-based split for texts without sentence punctuation
    if len(kept) <= 1:
        kept = [
            s.strip() for s in text.split("\n")
            if len(s.strip()) >= 20 and not _is_boilerplate(s.strip())
        ]

    return " ".join(kept) if kept else text


def clean_article_value(value) -> str:
    if pd.isna(value):
        return ""
    return clean_article(str(value))


def clean_articles_parallel(values: list) -> list[str]:
    if CLEAN_WORKERS <= 1 or len(values) <= 1:
        return [clean_article_value(value) for value in values]

    try:
        with ProcessPoolExecutor(max_workers=CLEAN_WORKERS, **_process_pool_kwargs()) as executor:
            return list(executor.map(clean_article_value, values, chunksize=CLEAN_CHUNKSIZE))
    except PermissionError as exc:
        print(f"Cleaning multiprocessing unavailable ({exc}); falling back to 1 worker", flush=True)
        return [clean_article_value(value) for value in values]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    start_time = time.time()

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE, low_memory=False)
    print(f"Loaded {len(df)} articles from {INPUT_FILE}")

    if "text" not in df.columns:
        raise ValueError("Column 'text' not found in input CSV")

    print(f"Cleaning workers: {CLEAN_WORKERS}")
    print(f"Cleaning chunksize: {CLEAN_CHUNKSIZE}")
    raw_text = df["text"].fillna("").astype(str)
    df["cleaning_flag_mojibake"] = raw_text.map(looks_like_mojibake)
    df["cleaning_flag_trash_page"] = raw_text.map(looks_like_trash_page)
    df["text_cleaned"] = clean_articles_parallel(df["text"].tolist())
    df["cleaning_flag_blank_after_clean"] = df["text_cleaned"].fillna("").astype(str).str.len().eq(0)

    # Report how much was removed on average
    orig_len  = df["text"].str.len().mean()
    clean_len = df["text_cleaned"].str.len().mean()
    reduction = 1 - clean_len / max(orig_len, 1)

    df.to_csv(INPUT_FILE, index=False, encoding="utf-8")
    elapsed = time.time() - start_time

    lines = [
        "ARTICLE CLEANING SUMMARY",
        "",
        f"Input file          : {INPUT_FILE}",
        f"Rows                : {len(df)}",
        f"Cleaning workers    : {CLEAN_WORKERS}",
        f"Cleaning chunksize  : {CLEAN_CHUNKSIZE}",
        f"Avg original length : {orig_len:.0f} chars",
        f"Avg cleaned length  : {clean_len:.0f} chars",
        f"Avg reduction       : {reduction*100:.1f}%",
        f"Mojibake flagged    : {int(df['cleaning_flag_mojibake'].sum())}",
        f"Trash pages flagged : {int(df['cleaning_flag_trash_page'].sum())}",
        f"Blank after clean   : {int(df['cleaning_flag_blank_after_clean'].sum())}",
        f"Elapsed seconds     : {elapsed:.2f}",
        "",
        f"Written to          : {INPUT_FILE}",
        f"Saved summary to    : {OUT_SUMMARY}",
    ]
    summary_text = "\n".join(lines)
    print(summary_text)

    with open(OUT_SUMMARY, "w", encoding="utf-8") as f:
        f.write(summary_text)


if __name__ == "__main__":
    main()
