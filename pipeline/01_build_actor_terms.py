"""
01_build_actor_terms.py

Builds an auditable actor-term lexicon for the UCDP-augmented GDELT pipeline.

Sources
-------
1. RAD group dataset 1.0
   Used to preserve continuity with the original RAD-based thesis pipeline.

2. UCDP Actor Dataset v25.1
   Used to expand the non-state actor lexicon beyond RAD's temporal coverage.
   By default, only Org == 1 actors are included. In the UCDP actor codebook,
   Org == 1 marks formally organized armed actors.

Outputs
-------
data/reference/cleaned_actor_terms.csv
    One term per row with source provenance and matching flags.

data/reference/cleaned_rebel_terms.csv
    Backward-compatible copy for older scripts that expect this filename.

data/reference/cleaned_rebel_data.csv
    RAD legacy actor table, retained for compatibility.
"""

from __future__ import annotations

import hashlib
import os
import re
import unicodedata
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.getenv("GDELT_DATA_ROOT", str(BASE_DIR / "data/reference")))

RAD_GROUP_PATH = BASE_DIR / "data/reference/RAD_group_dataset_1_0.csv"
UCDP_ACTOR_PATH = BASE_DIR / "data/reference/Actor_v26_1.csv"

OUT_DIR = DATA_ROOT
OUT_TERMS = OUT_DIR / "cleaned_actor_terms.csv"
OUT_TERMS_LEGACY = OUT_DIR / "cleaned_rebel_terms.csv"
OUT_RAD_LEGACY = OUT_DIR / "cleaned_rebel_data.csv"
OUT_SUMMARY = OUT_DIR / "actor_terms_summary.txt"


def _env_set(name: str, default: set[str]) -> set[str]:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    return {x.strip() for x in raw.split(",") if x.strip()}


UCDP_ORG_VALUES = _env_set("UCDP_ACTOR_ORG_VALUES", {"1"})


def _normalize_text(value: object) -> str:
    s = "" if pd.isna(value) else str(value)
    s = unicodedata.normalize("NFKC", s)
    s = s.strip()
    s = re.sub(r"[‘’“”]", "", s)
    s = re.sub(r"[\"']", "", s)
    s = re.sub(r"\s+", " ", s)
    return s


def _clean_term(value: object) -> str:
    s = _normalize_text(value)
    s = re.sub(r"[\[\]{}<>]", " ", s)
    s = re.sub(r"\s+", " ", s).strip(" -")
    return s


def _split_terms(value: object) -> list[str]:
    s = _clean_term(value)
    if not s:
        return []

    # RAD and UCDP often encode aliases as comma/semicolon/slash-separated
    # strings. Avoid splitting on hyphen because many group names contain it.
    parts = re.split(r"[;,/|:]|\baka\b|\ba\.k\.a\.\b", s, flags=re.IGNORECASE)
    out = [_clean_term(p) for p in parts]
    return [p for p in out if p]


def _term_id(dataset: str, source_id: str, source_column: str, term_lower: str) -> str:
    raw = f"{dataset}||{source_id}||{source_column}||{term_lower}".encode("utf-8")
    return hashlib.sha1(raw).hexdigest()


BLACKLIST = {
    "",
    "the",
    "group",
    "groups",
    "army",
    "armed group",
    "armed groups",
    "militia",
    "militias",
    "rebels",
    "rebel group",
    "rebel groups",
    "insurgents",
    "insurgent group",
    "opposition",
    "opposition group",
    "military faction",
    "multiple groups",
    "government",
    "government forces",
    "forces",
    "security forces",
    "police",
    "civilians",
    "unknown",
    "none",
}


def _is_useful_term(term: str) -> bool:
    t = term.lower().strip()
    if t in BLACKLIST:
        return False
    if len(t) < 3:
        return False
    if t.startswith("government of "):
        return False
    if re.fullmatch(r"\d+", t):
        return False
    return True


def _quality_flags(term: str) -> dict[str, object]:
    is_single = not bool(re.search(r"\s", term))
    term_len = len(term)
    has_digit = bool(re.search(r"\d", term))

    # Short acronyms are noisy but sometimes substantive (e.g. M23, ELN).
    # Keep them available, but expose the flag for auditing.
    is_short_single = is_single and term_len <= 4
    is_high_noise = is_single and term_len <= 2 and not has_digit

    return {
        "is_single_token": is_single,
        "term_len": term_len,
        "is_short_single_token": is_short_single,
        "is_high_noise_term": is_high_noise,
        "use_for_matching": not is_high_noise and _is_useful_term(term),
    }


def _append_term(
    records: list[dict[str, object]],
    *,
    dataset: str,
    source_id: object,
    source_name: object,
    source_column: str,
    source_org: object = "",
    term: object,
) -> None:
    for raw_term in _split_terms(term):
        cleaned = _clean_term(raw_term)
        if not _is_useful_term(cleaned):
            continue

        term_lower = cleaned.lower()
        flags = _quality_flags(cleaned)

        records.append({
            "term_id": _term_id(dataset, str(source_id), source_column, term_lower),
            "term": cleaned,
            "term_lower": term_lower,
            "term_source_dataset": dataset,
            "term_source_column": source_column,
            "source_actor_id": "" if pd.isna(source_id) else str(source_id),
            "source_actor_name": _normalize_text(source_name),
            "source_org": "" if pd.isna(source_org) else str(source_org),
            **flags,
        })


def build_rad_terms(records: list[dict[str, object]]) -> pd.DataFrame:
    if not RAD_GROUP_PATH.exists():
        raise FileNotFoundError(f"RAD file not found: {RAD_GROUP_PATH}")

    rad = pd.read_csv(RAD_GROUP_PATH, low_memory=False)
    rad_legacy = rad.rename(columns={
        "Group_UCDP_ID": "NSA_UCDP_ID",
        "Group_UCDP_Name_short": "NSA_UCDP_Name_short",
        "Group_UCDP_Name_long": "NSA_Name_long",
    })[["NSA_UCDP_ID", "NSA_UCDP_Name_short", "NSA_Name_long"]].copy()

    OUT_RAD_LEGACY.parent.mkdir(parents=True, exist_ok=True)
    rad_legacy.to_csv(OUT_RAD_LEGACY, index=False)

    for _, row in rad.iterrows():
        source_id = row.get("Group_UCDP_ID", "")
        source_name = row.get("Group_UCDP_Name_short", "")
        for col in ["Group_UCDP_Name_short", "Group_UCDP_Name_long"]:
            _append_term(
                records,
                dataset="RAD",
                source_id=source_id,
                source_name=source_name,
                source_column=col,
                term=row.get(col, ""),
            )

    return rad


def build_ucdp_terms(records: list[dict[str, object]]) -> pd.DataFrame:
    if not UCDP_ACTOR_PATH.exists():
        raise FileNotFoundError(f"UCDP actor file not found: {UCDP_ACTOR_PATH}")

    ucdp = pd.read_csv(UCDP_ACTOR_PATH, low_memory=False)
    org = ucdp.get("Org", pd.Series("", index=ucdp.index)).fillna("").astype(str)
    ucdp = ucdp[org.isin(UCDP_ORG_VALUES)].copy()

    source_cols = [
        "NameData",
        "NameOrig",
        "NameOrigFull",
        "NameOrigFullEng",
        "NewName",
        "NewNameFullMotherTongue",
        "NewNameFullEng",
        "NamePrev",
        "NameSplitTemp",
        "NameAlliance",
        "GroupName",
    ]
    source_cols = [c for c in source_cols if c in ucdp.columns]

    for _, row in ucdp.iterrows():
        source_id = row.get("ActorId", "")
        source_name = row.get("NameData", "") or row.get("NameOrig", "")
        source_org = row.get("Org", "")
        for col in source_cols:
            _append_term(
                records,
                dataset="UCDP_ACTOR",
                source_id=source_id,
                source_name=source_name,
                source_column=col,
                source_org=source_org,
                term=row.get(col, ""),
            )

    return ucdp


def main() -> None:
    print("Executing UCDP-augmented actor term builder")
    print("Base dir:", BASE_DIR)
    print("Data root:", DATA_ROOT)
    print("UCDP Org values included:", ", ".join(sorted(UCDP_ORG_VALUES)))

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, object]] = []
    rad = build_rad_terms(records)
    ucdp = build_ucdp_terms(records)

    terms = pd.DataFrame(records)
    if terms.empty:
        raise RuntimeError("No actor terms were built")

    terms = terms.drop_duplicates(
        subset=["term_source_dataset", "source_actor_id", "term_lower", "term_source_column"]
    ).reset_index(drop=True)

    # A compact duplicate view is useful for diagnosing terms shared by RAD and UCDP.
    term_sources = (
        terms.groupby("term_lower")["term_source_dataset"]
        .apply(lambda s: ";".join(sorted(set(s))))
        .rename("term_source_datasets")
        .reset_index()
    )
    terms = terms.merge(term_sources, on="term_lower", how="left")

    terms.to_csv(OUT_TERMS, index=False)
    terms.to_csv(OUT_TERMS_LEGACY, index=False)

    lines = [
        "Actor term summary",
        f"RAD group rows: {len(rad)}",
        f"UCDP actor rows after Org filter: {len(ucdp)}",
        f"Total term rows: {len(terms)}",
        f"Unique term_lower values: {terms['term_lower'].nunique()}",
        f"Rows used for matching: {int(terms['use_for_matching'].sum())}",
        "",
        "Rows by source dataset",
        terms["term_source_dataset"].value_counts().to_string(),
        "",
        "Top source overlaps",
        terms["term_source_datasets"].value_counts().head(20).to_string(),
    ]
    summary = "\n".join(lines)
    OUT_SUMMARY.write_text(summary + "\n", encoding="utf-8")

    print(summary)
    print("Saved terms:", OUT_TERMS)
    print("Saved legacy terms:", OUT_TERMS_LEGACY)
    print("Saved RAD legacy data:", OUT_RAD_LEGACY)


if __name__ == "__main__":
    main()
