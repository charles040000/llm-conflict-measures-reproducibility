from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRANSLATION_DIR = REPO_ROOT / "data" / "translation"
from deep_translator import GoogleTranslator


GOOGLE_LANG_MAP = {
    "he": "iw",
    "zh-cn": "zh-CN",
    "zh-tw": "zh-TW",
    "zh": "zh-CN",
}


def clean_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def split_chunks(text: str, max_chars: int) -> list[str]:
    chunks: list[str] = []
    current = ""

    def flush() -> None:
        nonlocal current
        if current:
            chunks.append(current)
            current = ""

    def add_piece(piece: str) -> None:
        nonlocal current
        piece = piece.strip()
        if not piece:
            return
        if len(piece) > max_chars:
            flush()
            words = piece.split()
            word_chunk = ""
            for word in words:
                if len(word) > max_chars:
                    if word_chunk:
                        chunks.append(word_chunk)
                        word_chunk = ""
                    for start in range(0, len(word), max_chars):
                        chunks.append(word[start:start + max_chars])
                    continue
                candidate = f"{word_chunk} {word}".strip() if word_chunk else word
                if len(candidate) > max_chars:
                    chunks.append(word_chunk)
                    word_chunk = word
                else:
                    word_chunk = candidate
            if word_chunk:
                chunks.append(word_chunk)
            return

        candidate = f"{current} {piece}".strip() if current else piece
        if len(candidate) > max_chars:
            flush()
            current = piece
        else:
            current = candidate

    for sentence in re.split(r"(?<=[.!?…])\s+", text):
        add_piece(sentence)
    flush()
    return chunks


def translate_text(
    text: str,
    src_lang: str,
    chunk_chars: int,
    sleep_seconds: float,
    retries: int,
) -> str:
    google_src = GOOGLE_LANG_MAP.get(src_lang.lower(), src_lang.lower())
    parts: list[str] = []
    for chunk in split_chunks(text, chunk_chars):
        last_error: Exception | None = None
        for attempt in range(1, retries + 2):
            try:
                translated = GoogleTranslator(source=google_src, target="en").translate(chunk)
                if not translated:
                    raise RuntimeError("empty translation response")
                parts.append(translated)
                time.sleep(sleep_seconds)
                break
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt <= retries:
                    time.sleep(max(sleep_seconds, 0.5) * attempt)
                    continue
                raise RuntimeError(
                    f"Google translation failed after {retries + 1} attempts "
                    f"(lang={src_lang}, google_lang={google_src}, chunk_len={len(chunk)}): {last_error}"
                ) from last_error
    return " ".join(parts)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def run(
    input_csv: Path,
    output_csv: Path,
    checkpoint_csv: Path,
    progress_jsonl: Path,
    sleep_seconds: float,
    retries: int,
    chunk_chars: int,
    checkpoint_every: int,
    limit: int,
) -> None:
    df = pd.read_csv(input_csv, low_memory=False)
    if "detected_language" not in df.columns:
        raise ValueError("Input must contain detected_language")
    if "text_cleaned" not in df.columns:
        raise ValueError("Input must contain text_cleaned")

    # Hard guard: never send English/unknown rows to Google.
    queue_mask = (
        df["detected_language"].fillna("").astype(str).ne("en")
        & df["detected_language"].fillna("").astype(str).ne("unknown")
        & df["detected_language"].fillna("").astype(str).ne("")
    )
    df = df[queue_mask].copy().reset_index(drop=True)

    if output_csv.exists():
        prev = pd.read_csv(output_csv, low_memory=False)
        done_ids = set(prev.loc[prev["translation_status"].isin(["translated"]), "GlobalEventID"])
        if done_ids:
            print(f"Resume mode: found {len(done_ids):,} already translated rows in {output_csv}")
            df = df[~df["GlobalEventID"].isin(done_ids)].copy().reset_index(drop=True)
            merged = prev.copy()
        else:
            merged = pd.DataFrame()
    else:
        merged = pd.DataFrame()

    if limit > 0:
        df = df.head(limit).copy()

    print(f"Rows queued for this run: {len(df):,}", flush=True)
    print("Queued languages:", flush=True)
    print(df["detected_language"].value_counts().to_string(), flush=True)

    rows: list[dict[str, Any]] = []
    start_time = time.time()
    for pos, (_, row) in enumerate(df.iterrows(), start=1):
        source = clean_text(row["text_cleaned"])
        lang = clean_text(row["detected_language"]).lower()
        out = row.to_dict()
        out["source_chars"] = len(source)
        out["translation_status"] = ""
        out["translation_error"] = ""
        out["text_en"] = ""
        out["translation_model"] = "google_translate"

        try:
            translated = translate_text(
                source,
                src_lang=lang,
                chunk_chars=chunk_chars,
                sleep_seconds=sleep_seconds,
                retries=retries,
            )
            out["text_en"] = translated
            out["translation_status"] = "translated"
        except Exception as exc:  # noqa: BLE001
            out["translation_status"] = "failed"
            out["translation_error"] = str(exc)

        rows.append(out)
        append_jsonl(
            progress_jsonl,
            {
                "GlobalEventID": out.get("GlobalEventID"),
                "pos": pos,
                "queued_rows": len(df),
                "detected_language": lang,
                "status": out["translation_status"],
                "error": out["translation_error"],
                "elapsed_seconds": round(time.time() - start_time, 2),
            },
        )

        if pos % checkpoint_every == 0:
            current = pd.DataFrame(rows)
            combined = pd.concat([merged, current], ignore_index=True) if len(merged) else current
            combined.to_csv(checkpoint_csv, index=False)
            combined.to_csv(output_csv, index=False)
            print(
                f"[checkpoint] {pos:,}/{len(df):,} queued rows processed; "
                f"translated={int(current['translation_status'].eq('translated').sum()):,}; "
                f"failed={int(current['translation_status'].eq('failed').sum()):,}",
                flush=True,
            )

    current = pd.DataFrame(rows)
    combined = pd.concat([merged, current], ignore_index=True) if len(merged) else current
    combined.to_csv(checkpoint_csv, index=False)
    combined.to_csv(output_csv, index=False)
    print("", flush=True)
    print("Done.", flush=True)
    print(f"Output: {output_csv}", flush=True)
    print(combined["translation_status"].value_counts(dropna=False).to_string(), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Translate known non-English matched articles to English.")
    parser.add_argument("--input_csv", type=Path, default=DEFAULT_TRANSLATION_DIR / "non_english_articles_to_translate.csv")
    parser.add_argument("--output_csv", type=Path, default=DEFAULT_TRANSLATION_DIR / "translated_non_english_articles.csv")
    parser.add_argument("--checkpoint_csv", type=Path, default=DEFAULT_TRANSLATION_DIR / "translated_non_english_articles.checkpoint.csv")
    parser.add_argument("--progress_jsonl", type=Path, default=DEFAULT_TRANSLATION_DIR / "translation_progress.jsonl")
    parser.add_argument("--sleep", type=float, default=0.5)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--chunk_chars", type=int, default=2500)
    parser.add_argument("--checkpoint_every", type=int, default=100)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        checkpoint_csv=args.checkpoint_csv,
        progress_jsonl=args.progress_jsonl,
        sleep_seconds=args.sleep,
        retries=args.retries,
        chunk_chars=args.chunk_chars,
        checkpoint_every=args.checkpoint_every,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
