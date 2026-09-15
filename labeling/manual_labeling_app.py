"""
article_labeling_v2.py

Focused Streamlit UI for the high-precision, full-text article-labeling schema.

Usage
-----
streamlit run 09_ui/article_labeling_v2.py

Optional environment overrides
------------------------------
ARTICLE_LABELING_V2_CSV_PATH
ARTICLE_LABELING_V2_LABELS_PATH
ARTICLE_LABELING_V2_DEFAULT_ANNOTATOR
"""

from __future__ import annotations

import os
import json
import random
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components


BASE_DIR = Path(__file__).resolve().parents[1]

DEFAULT_CSV_PATH = (
    BASE_DIR
    / "data"
    / "external"
    / "manual_labeling_sample.csv"
)
DEFAULT_LABELS_PATH = (
    BASE_DIR
    / "results"
    / "manual_labels_working.csv"
)

LABEL_COLUMNS_V2 = [
    "GlobalEventID",
    "annotator_id",
    "split",
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
    "annotator_notes",
    "flagged",
]

DEPRECATED_LABEL_COLUMNS = [
    "fatality_count_anywhere_text",
    "injury_count_anywhere_text",
    "location_locality_text",
    "location_country_text",
    "location_certainty",
    "location_region_pre_llm",
    "location_region_llm_check",
]

COUNT_TEXT_COLUMNS = [
    "fatality_count_anywhere_num",
    "injury_count_anywhere_num",
    "fatality_count_reported_num",
    "injury_count_reported_num",
]

EVENT_CONTEXT_OPTIONS = ["", "POL", "CBT", "CIV", "ORG", "OPR", "CRM", "PRO", "not_applicable", "unclear"]
EVENT_MODALITY_OPTIONS = ["", "realized", "threatened", "not_applicable", "unclear"]
YES_NO_OPTIONS = ["", "yes", "no", "not_applicable", "unclear"]
MENTION_OPTIONS = ["", "yes", "no", "unclear"]
WEAPON_OPTIONS = ["", "none_used", "salw", "heavy", "mixed", "not_applicable", "unclear"]
RELEVANCE_OPTIONS = ["", "direct", "background_or_context", "irrelevant", "unclear"]
ACTOR_ROLE_OPTIONS = ["", "perpetrator", "target", "both", "mentioned_only", "unclear", "not_applicable"]
TIME_RELATION_OPTIONS = ["", "current_or_recent", "historical_background", "future_or_planned", "unclear"]
LOCATION_MATCH_OPTIONS = ["", "yes", "no", "ambiguous", "not_reported", "unclear"]
CONFIDENCE_OPTIONS = ["", "high", "medium", "low", "unclear"]
MULTIPLE_OPTIONS = ["", "yes", "no", "unclear"]
IRR_OVERLAP_SHARE = 0.10

IRRELEVANT_MAIN_STORY_DEFAULTS = {
    "event_context": "not_applicable",
    "event_modality": "not_applicable",
    "matched_actor_role": "not_applicable",
    "event_time_relation": "unclear",
    "physical_violence_occurred": "not_applicable",
    "fatalities_present": "not_applicable",
    "fatality_count_reported_text": "",
    "fatality_count_reported_num": "",
    "injuries_present": "not_applicable",
    "injury_count_reported_text": "",
    "injury_count_reported_num": "",
    "weapon_type_text": "",
    "weapon_use_class": "not_applicable",
    "article_location_country_text": "",
    "location_matches_actor_country": "not_reported",
    "multiple_events_mentioned": "unclear",
    "main_story_confidence": "unclear",
}


def resolve_path(raw: str | Path) -> Path:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = BASE_DIR / path
    return path


def prefer_fresh_parquet(path: Path) -> Path:
    parquet_path = path.with_suffix(".parquet")
    if (
        path.suffix == ".csv"
        and parquet_path.exists()
        and (not path.exists() or parquet_path.stat().st_mtime_ns >= path.stat().st_mtime_ns)
    ):
        return parquet_path
    return path


CSV_PATH = prefer_fresh_parquet(resolve_path(os.getenv("ARTICLE_LABELING_V2_CSV_PATH", str(DEFAULT_CSV_PATH))))
LABELS_PATH = resolve_path(os.getenv("ARTICLE_LABELING_V2_LABELS_PATH", str(DEFAULT_LABELS_PATH)))
DEFAULT_ANNOTATOR = os.getenv("ARTICLE_LABELING_V2_DEFAULT_ANNOTATOR", "annotator_1")


st.set_page_config(
    page_title="Article Labeling v2",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(ttl=30, show_spinner=False)
def load_articles(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix == ".parquet":
        df = pd.read_parquet(p)
    else:
        df = pd.read_csv(p, low_memory=False)
    df["GlobalEventID"] = df["GlobalEventID"].astype(str)
    return df


def clean_count_text(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.lower() in {"nan", "none"}:
        return ""
    if text.endswith(".0") and text[:-2].isdigit():
        return text[:-2]
    return text


@st.cache_data(ttl=10, show_spinner=False)
def load_labels_cached(path: str, mtime_ns: int | None) -> pd.DataFrame:
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return pd.DataFrame(columns=LABEL_COLUMNS_V2)
    df = pd.read_csv(p, dtype={"GlobalEventID": str}, low_memory=False)
    if "article_location_country_text" not in df.columns and "location_country_text" in df.columns:
        df["article_location_country_text"] = df["location_country_text"]
    df = df.drop(columns=DEPRECATED_LABEL_COLUMNS, errors="ignore")
    for col in LABEL_COLUMNS_V2:
        if col not in df.columns:
            df[col] = ""
    for col in COUNT_TEXT_COLUMNS:
        if col in df.columns:
            df[col] = df[col].map(clean_count_text)
    return df[LABEL_COLUMNS_V2 + [c for c in df.columns if c not in LABEL_COLUMNS_V2]]


def load_labels(path: Path = LABELS_PATH) -> pd.DataFrame:
    mtime_ns = path.stat().st_mtime_ns if path.exists() else None
    return load_labels_cached(str(path), mtime_ns)


def save_label(record: dict, path: Path = LABELS_PATH) -> None:
    labels = load_labels(path).copy()
    eid = str(record["GlobalEventID"])
    annotator_id = str(record["annotator_id"])
    if len(labels):
        labels = labels[
            ~(
                labels["GlobalEventID"].astype(str).eq(eid)
                & labels["annotator_id"].astype(str).eq(annotator_id)
            )
        ].copy()
    labels = pd.concat([labels, pd.DataFrame([record])], ignore_index=True)
    labels = labels.drop(columns=DEPRECATED_LABEL_COLUMNS, errors="ignore")
    for col in LABEL_COLUMNS_V2:
        if col not in labels.columns:
            labels[col] = ""
    for col in COUNT_TEXT_COLUMNS:
        if col in labels.columns:
            labels[col] = labels[col].map(clean_count_text)
    labels = labels[LABEL_COLUMNS_V2 + [c for c in labels.columns if c not in LABEL_COLUMNS_V2]]
    path.parent.mkdir(parents=True, exist_ok=True)
    labels.to_csv(path, index=False)
    load_labels_cached.clear()


def delete_label(eid: str, annotator_id: str, path: Path = LABELS_PATH) -> None:
    labels = load_labels(path).copy()
    if len(labels):
        labels = labels[
            ~(
                labels["GlobalEventID"].astype(str).eq(str(eid))
                & labels["annotator_id"].astype(str).eq(str(annotator_id))
            )
        ].copy()
    path.parent.mkdir(parents=True, exist_ok=True)
    labels.to_csv(path, index=False)
    load_labels_cached.clear()


def save_labels_dataframe(labels: pd.DataFrame, path: Path = LABELS_PATH) -> None:
    labels = labels.copy()
    labels = labels.drop(columns=["_delete"] + DEPRECATED_LABEL_COLUMNS, errors="ignore")
    for col in LABEL_COLUMNS_V2:
        if col not in labels.columns:
            labels[col] = ""
    labels = labels[LABEL_COLUMNS_V2 + [c for c in labels.columns if c not in LABEL_COLUMNS_V2]]
    labels = labels.fillna("")
    path.parent.mkdir(parents=True, exist_ok=True)
    labels.to_csv(path, index=False)
    load_labels_cached.clear()


def safe_get(row: pd.Series, col: str, default: str = "") -> str:
    return str(row[col]) if col in row.index and pd.notna(row[col]) else default


def first_existing_text(row: pd.Series, columns: list[str]) -> str:
    for col in columns:
        value = safe_get(row, col)
        if value:
            return value
    return ""


def option_index(options: list[str], value: object) -> int:
    if pd.isna(value):
        return 0
    value_str = str(value)
    return options.index(value_str) if value_str in options else 0


def bool_value(value: object) -> bool:
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def parse_records_json(value: object) -> list[dict]:
    if pd.isna(value):
        return []
    raw = str(value).strip()
    if not raw:
        return []
    try:
        records = json.loads(raw)
    except json.JSONDecodeError:
        return []
    return records if isinstance(records, list) else []


def record_count(value: object) -> int:
    return len(parse_records_json(value))


def matched_actor_summary(row: pd.Series) -> dict[str, object]:
    records_json = first_existing_text(
        row,
        ["matched_actor_records_json", "raw_actor_alias_hit_records_json"],
    )
    records = parse_records_json(records_json)
    actor_names = []
    actor_ids = []
    for record in records:
        if not isinstance(record, dict):
            continue
        name = str(record.get("source_actor_name", "")).strip()
        actor_id = str(record.get("source_actor_id", "")).strip()
        if name and name not in actor_names:
            actor_names.append(name)
        if actor_id and actor_id not in actor_ids:
            actor_ids.append(actor_id)

    side = first_existing_text(row, ["matched_actor_side", "raw_actor_alias_hit_side"])
    gdelt_actor = ""
    if side == "actor1":
        gdelt_actor = safe_get(row, "Actor1Name")
    elif side == "actor2":
        gdelt_actor = safe_get(row, "Actor2Name")

    return {
        "ucdp_actor": "; ".join(actor_names),
        "ucdp_actor_id": "; ".join(actor_ids),
        "gdelt_side": side,
        "gdelt_actor": gdelt_actor,
        "alias_terms": first_existing_text(row, ["matched_actor_alias_terms_in_text", "raw_actor_alias_terms_in_text"]),
        "match_sources": safe_get(row, "match_sources"),
        "records_json": records_json,
        "records": records,
    }


def matched_actor_records_table(records: list[dict]) -> pd.DataFrame:
    rows = []
    for record in records:
        if not isinstance(record, dict):
            continue
        rows.append(
            {
                "source_actor_name": str(record.get("source_actor_name", "")).strip(),
                "source_actor_id": str(record.get("source_actor_id", "")).strip(),
                "matched_side": str(record.get("matched_side", "")).strip(),
                "gdelt_actor_name": str(record.get("gdelt_actor_name", "")).strip(),
                "matched_term": str(record.get("term", "")).strip(),
                "source_dataset": str(record.get("source_dataset", "")).strip(),
            }
        )
    return pd.DataFrame(rows)


def compact_metadata(row: pd.Series, columns: list[str]) -> pd.DataFrame:
    data = {col: safe_get(row, col) for col in columns if col in row.index}
    return pd.DataFrame.from_dict(data, orient="index", columns=["value"])


def article_selector_label(row: pd.Series, idx: int, labeled_ids: set[str], other_ids: set[str]) -> str:
    eid = safe_get(row, "GlobalEventID", str(idx))
    a1 = safe_get(row, "Actor1Name")[:18]
    a2 = safe_get(row, "Actor2Name")[:18]
    side = first_existing_text(row, ["matched_actor_side", "raw_actor_alias_hit_side"])
    month = safe_get(row, "source_month") or safe_get(row, "SQLDATE")[:6]
    lang = safe_get(row, "lang_detected")
    done = "[x] " if eid in labeled_ids else ""
    other = " [2nd]" if eid in other_ids else ""
    return f"{done}[{idx}] ID={eid} | {month} | {a1} / {a2} | side={side} | lang={lang}{other}"


def text_area_height(text: str) -> int:
    soft_wrap_lines = sum(max(1, (len(line) + 105) // 106) for line in text.splitlines() or [""])
    return max(180, min(520, 42 + soft_wrap_lines * 30))


if not CSV_PATH.exists():
    st.error(f"Article table not found: `{CSV_PATH}`")
    st.stop()

df = load_articles(str(CSV_PATH))
labels_df = load_labels(LABELS_PATH)

st.sidebar.title("Article Labeling v2")
current_annotator = st.sidebar.text_input("Annotator name", value=DEFAULT_ANNOTATOR, key="annotator").strip()
if not current_annotator:
    st.sidebar.error("Enter an annotator name before labeling.")
    st.stop()
st.sidebar.caption(f"Articles: {CSV_PATH}")
st.sidebar.caption(f"Labels: {LABELS_PATH}")

if st.sidebar.button("Refresh", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

st.sidebar.divider()
st.sidebar.title("Filters")

langs = sorted(df["lang_detected"].dropna().astype(str).unique().tolist()) if "lang_detected" in df.columns else []
selected_langs = st.sidebar.multiselect("Language", langs, default=langs)

months = sorted(df["source_month"].dropna().astype(str).unique().tolist()) if "source_month" in df.columns else []
selected_months = st.sidebar.multiselect("Month", months, default=months)

show_quality_bad = st.sidebar.checkbox("Include quality-bad articles", value=True)
only_single_actor = st.sidebar.checkbox("Single matched source actor only", value=False)
only_alias_in_text = st.sidebar.checkbox("Actor alias present in text", value=True)
show_mode = st.sidebar.radio("Label status", ["Unlabeled first", "Unlabeled only", "Labeled only", "All"], horizontal=False)
irr_only = st.sidebar.checkbox("Only articles needing my 2nd label", value=False)

my_labels_df = labels_df[labels_df["annotator_id"].astype(str).eq(current_annotator)] if len(labels_df) else labels_df
labeled_ids = set(my_labels_df["GlobalEventID"].astype(str).tolist()) if len(my_labels_df) else set()
all_labeled_ids = set(labels_df["GlobalEventID"].astype(str).tolist()) if len(labels_df) else set()
other_annotator_labels_df = (
    labels_df[~labels_df["annotator_id"].astype(str).eq(current_annotator)]
    if len(labels_df)
    else labels_df
)
other_labeled_ids = (
    set(other_annotator_labels_df["GlobalEventID"].astype(str).tolist())
    if len(other_annotator_labels_df)
    else set()
)
my_overlap_ids = labeled_ids & other_labeled_ids
current_overlap_count = len(my_overlap_ids)
max_overlap_after_next = int((len(labeled_ids) + 1) * IRR_OVERLAP_SHARE)
allow_next_overlap = current_overlap_count < max_overlap_after_next

mask = pd.Series(True, index=df.index)
if selected_langs and "lang_detected" in df.columns:
    mask &= df["lang_detected"].astype(str).isin(selected_langs)
if selected_months and "source_month" in df.columns:
    mask &= df["source_month"].astype(str).isin(selected_months)
if not show_quality_bad and "quality_bad_score" in df.columns:
    mask &= pd.to_numeric(df["quality_bad_score"], errors="coerce").fillna(0).eq(0)
if only_single_actor and "is_single_matched_source_actor" in df.columns:
    mask &= df["is_single_matched_source_actor"].astype(str).str.lower().isin(["true", "1", "yes"])
elif only_single_actor and "raw_actor_alias_hit_records_json" in df.columns:
    mask &= df["raw_actor_alias_hit_records_json"].map(record_count).eq(1)
if only_alias_in_text:
    if "has_actor_alias_in_text" in df.columns:
        mask &= df["has_actor_alias_in_text"].astype(str).str.lower().isin(["true", "1", "yes"])
    elif "raw_actor_alias_terms_in_text" in df.columns:
        mask &= df["raw_actor_alias_terms_in_text"].fillna("").astype(str).str.len().gt(0)

filtered = df[mask].copy()
filtered["_labeled_by_me"] = filtered["GlobalEventID"].isin(labeled_ids)
filtered["_labeled_by_others"] = filtered["GlobalEventID"].isin(other_labeled_ids)
normal_unlabeled = filtered[~filtered["_labeled_by_me"]].copy()
if not allow_next_overlap:
    normal_unlabeled = normal_unlabeled[~normal_unlabeled["_labeled_by_others"]].copy()
else:
    normal_unlabeled = pd.concat(
        [
            normal_unlabeled[~normal_unlabeled["_labeled_by_others"]],
            normal_unlabeled[normal_unlabeled["_labeled_by_others"]],
        ],
        ignore_index=False,
    )

if irr_only:
    if allow_next_overlap:
        pool = filtered[filtered["_labeled_by_others"] & ~filtered["_labeled_by_me"]].reset_index(drop=True)
    else:
        pool = filtered.iloc[0:0].copy().reset_index(drop=True)
elif show_mode == "Unlabeled only":
    pool = normal_unlabeled.reset_index(drop=True)
elif show_mode == "Labeled only":
    pool = filtered[filtered["_labeled_by_me"]].reset_index(drop=True)
elif show_mode == "All":
    pool = filtered.reset_index(drop=True)
else:
    pool = normal_unlabeled.reset_index(drop=True)

st.sidebar.markdown(f"**{len(pool)} / {len(df)} articles shown**")
st.sidebar.markdown(f"**{len(labeled_ids)} labeled by me**")
st.sidebar.markdown(f"**{current_overlap_count} overlap labels by me**")
st.sidebar.caption(
    f"Normal queue overlap cap: {int(IRR_OVERLAP_SHARE * 100)}%; "
    f"{'overlap allowed' if allow_next_overlap else 'fresh articles only'}."
)

st.title("Article Labeling v2")
st.caption("Full-text labels with conflict relevance, actor role, time relation, and location audit.")

metrics = st.columns(6)
metrics[0].metric("Articles", len(df))
metrics[1].metric("Filtered", len(pool))
metrics[2].metric("My labeled", len(labeled_ids))
metrics[3].metric("All labeled", len(all_labeled_ids))
metrics[4].metric("Flagged", int(my_labels_df["flagged"].map(bool_value).sum()) if len(my_labels_df) else 0)
metrics[5].metric("My overlap", current_overlap_count)

st.divider()

if len(pool) == 0:
    st.info("No articles match the current filters.")
    st.stop()

unlabeled_pool_indices = [
    i for i in range(len(pool))
    if safe_get(pool.iloc[i], "GlobalEventID") not in labeled_ids
]

pending_selector_idx = st.session_state.pop("pending_article_selector_v2", None)
if pending_selector_idx in range(len(pool)):
    st.session_state["article_selector_v2"] = pending_selector_idx

selector_col, random_col = st.columns([5, 1])
with random_col:
    if st.button("Random", use_container_width=True, disabled=len(unlabeled_pool_indices) == 0):
        st.session_state["article_selector_v2"] = random.choice(unlabeled_pool_indices)

with selector_col:
    selected_idx = st.selectbox(
        "Select article",
        options=list(range(len(pool))),
        format_func=lambda i: article_selector_label(pool.iloc[i], i, labeled_ids, other_labeled_ids),
        key="article_selector_v2",
    )

row = pool.iloc[selected_idx]
eid = safe_get(row, "GlobalEventID")
match_summary = matched_actor_summary(row)

existing = {}
if eid in labeled_ids:
    existing = my_labels_df[my_labels_df["GlobalEventID"].astype(str).eq(eid)].iloc[0].to_dict()
    info_col, delete_col = st.columns([5, 1])
    info_col.info("You already labeled this article. You can update it below.")
    if delete_col.button("Delete label", type="secondary"):
        delete_label(eid, current_annotator)
        st.rerun()
elif eid in other_labeled_ids:
    n_others = int(labels_df[labels_df["GlobalEventID"].astype(str).eq(eid)]["annotator_id"].nunique()) if len(labels_df) else 0
    st.info(f"Labeled by {n_others} other annotator(s). Your label will be added for IRR.")

st.divider()
st.subheader("Labeling Workbench")

article_text = first_existing_text(
    row,
    [
        "article_text_for_llm",
        "article_text_translated",
        "article_text_original",
        "article_text",
        "text_en",
        "text_cleaned",
        "text",
    ],
)
url = safe_get(row, "final_url") or safe_get(row, "SOURCEURL") or safe_get(row, "used_url")
label_widget_prefix = f"label_v2_{current_annotator}_{eid}"
pre_cols = [
    "GlobalEventID",
    "SQLDATE",
    "source_month",
    "Actor1Name",
    "Actor2Name",
    "Actor1CountryCode",
    "Actor2CountryCode",
    "ActionGeo_CountryCode",
    "EventCode",
    "EventRootCode",
    "match_sources",
    "match_rad_term",
    "match_ucdp_actor_term",
    "match_gdelt_reb_type",
    "match_rebel_term",
    "match_rebel_type",
    "matched_actor_side",
    "raw_actor_alias_hit_side",
    "n_matched_source_actors",
    "is_single_matched_source_actor",
    "actor1_rad_ucdp_alias_terms",
    "actor2_rad_ucdp_alias_terms",
    "matched_actor_alias_terms_in_text",
    "raw_actor_alias_terms_in_text",
    "has_actor_alias_in_text",
    "has_actor1_signal",
    "has_actor2_signal",
    "has_actor_signal",
    "has_rebel_keyword",
    "domain",
    "final_url",
]

article_block = st.container(border=True)
with article_block:
    header_col, source_col = st.columns([1, 3])
    header_col.markdown("#### Article")
    header_col.caption(f"{len(article_text)} chars")
    if url:
        source_col.caption(f"Source: {url}")

    st.markdown("##### Matched actor for role coding")
    match_col1, match_col2, match_col3, match_col4 = st.columns([2, 1.2, 1.2, 1.6])
    match_col1.markdown("**UCDP actor**")
    match_col1.write(match_summary["ucdp_actor"] or "not parsed")
    if match_summary["ucdp_actor_id"]:
        match_col1.caption(f"ActorId: {match_summary['ucdp_actor_id']}")
    match_col2.markdown("**GDELT side**")
    match_col2.write(match_summary["gdelt_side"] or "not parsed")
    match_col3.markdown("**GDELT actor**")
    match_col3.write(match_summary["gdelt_actor"] or "not parsed")
    match_col4.markdown("**Alias in text**")
    match_col4.write(match_summary["alias_terms"] or "none")

    match_records_df = matched_actor_records_table(match_summary["records"])
    if len(match_records_df):
        st.dataframe(match_records_df, use_container_width=True, hide_index=True, height=96)
    else:
        st.caption(f"Match sources: {match_summary['match_sources'] or 'none'}")

    st.text_area(
        "Article text",
        article_text,
        height=text_area_height(article_text),
        label_visibility="collapsed",
        key=f"{label_widget_prefix}_article_text",
    )

label_block = st.container(border=True)
with label_block:
    st.markdown("#### Labels")
    st.caption("Article-wide mentions ignore the main-story rule and ask whether the item appears anywhere in the article text.")
    mention_col1, mention_col2, mention_col3, mention_col4 = st.columns(4)
    with mention_col1:
        any_matched_actor_mentioned = st.selectbox(
            "Any matched actor mentioned",
            MENTION_OPTIONS,
            index=option_index(MENTION_OPTIONS, existing.get("any_matched_actor_mentioned", "")),
            key=f"{label_widget_prefix}_any_matched_actor_mentioned",
        )
    with mention_col2:
        any_physical_violence_mentioned = st.selectbox(
            "Any physical violence mentioned",
            MENTION_OPTIONS,
            index=option_index(MENTION_OPTIONS, existing.get("any_physical_violence_mentioned", "")),
            key=f"{label_widget_prefix}_any_physical_violence_mentioned",
        )
    with mention_col3:
        any_fatalities_mentioned = st.selectbox(
            "Any fatalities mentioned",
            MENTION_OPTIONS,
            index=option_index(MENTION_OPTIONS, existing.get("any_fatalities_mentioned", "")),
            key=f"{label_widget_prefix}_any_fatalities_mentioned",
        )
    with mention_col4:
        any_injuries_mentioned = st.selectbox(
            "Any injuries mentioned",
            MENTION_OPTIONS,
            index=option_index(MENTION_OPTIONS, existing.get("any_injuries_mentioned", "")),
            key=f"{label_widget_prefix}_any_injuries_mentioned",
        )
    mention_detail_col1, mention_detail_col2, mention_detail_col3 = st.columns(3)
    with mention_detail_col1:
        fatality_count_anywhere_num = st.text_input(
            "Fatality count anywhere numeric",
            value=existing.get("fatality_count_anywhere_num", ""),
            placeholder="manual sum of numeric fatality mentions",
            key=f"{label_widget_prefix}_fatality_count_anywhere_num",
        )
    with mention_detail_col2:
        injury_count_anywhere_num = st.text_input(
            "Injury count anywhere numeric",
            value=existing.get("injury_count_anywhere_num", ""),
            placeholder="manual sum of numeric injury mentions",
            key=f"{label_widget_prefix}_injury_count_anywhere_num",
        )
    with mention_detail_col3:
        any_weapon_mentioned = st.selectbox(
            "Any weapon mentioned",
            MENTION_OPTIONS,
            index=option_index(MENTION_OPTIONS, existing.get("any_weapon_mentioned", "")),
            key=f"{label_widget_prefix}_any_weapon_mentioned",
        )
        weapon_type_anywhere_text = st.text_input(
            "Weapon type anywhere text",
            value=existing.get("weapon_type_anywhere_text", ""),
            placeholder="optional: weapons mentioned anywhere",
            key=f"{label_widget_prefix}_weapon_type_anywhere_text",
        )
    st.markdown("##### Main story labels")
    label_col1, label_col2, label_col3 = st.columns(3)
    with label_col1:
        conflict_relevance = st.selectbox(
            "A. Conflict relevance",
            RELEVANCE_OPTIONS,
            index=option_index(RELEVANCE_OPTIONS, existing.get("conflict_relevance", "")),
            key=f"{label_widget_prefix}_conflict_relevance",
        )
        event_context = st.selectbox(
            "B. Event context",
            EVENT_CONTEXT_OPTIONS,
            index=option_index(EVENT_CONTEXT_OPTIONS, existing.get("event_context", "")),
            key=f"{label_widget_prefix}_event_context",
        )
        event_modality = st.selectbox(
            "C. Event modality",
            EVENT_MODALITY_OPTIONS,
            index=option_index(EVENT_MODALITY_OPTIONS, existing.get("event_modality", "")),
            key=f"{label_widget_prefix}_event_modality",
        )
        matched_actor_role = st.selectbox(
            "D. Matched actor role",
            ACTOR_ROLE_OPTIONS,
            index=option_index(ACTOR_ROLE_OPTIONS, existing.get("matched_actor_role", "")),
            key=f"{label_widget_prefix}_matched_actor_role",
        )
        event_time_relation = st.selectbox(
            "E. Event time relation",
            TIME_RELATION_OPTIONS,
            index=option_index(TIME_RELATION_OPTIONS, existing.get("event_time_relation", "")),
            key=f"{label_widget_prefix}_event_time_relation",
        )
        physical_violence_occurred = st.selectbox(
            "F. Physical violence occurred",
            YES_NO_OPTIONS,
            index=option_index(YES_NO_OPTIONS, existing.get("physical_violence_occurred", "")),
            key=f"{label_widget_prefix}_physical_violence_occurred",
        )
    with label_col2:
        fatalities_present = st.selectbox(
            "G. Fatalities present",
            YES_NO_OPTIONS,
            index=option_index(YES_NO_OPTIONS, existing.get("fatalities_present", "")),
            key=f"{label_widget_prefix}_fatalities_present",
        )
        fatality_count_text = st.text_input(
            "G. Fatality count text",
            value=existing.get("fatality_count_reported_text", ""),
            placeholder="e.g. 25, at least five, dozens, unclear",
            key=f"{label_widget_prefix}_fatality_count_reported_text",
        )
        fatality_count_num = st.text_input(
            "G. Fatality count numeric",
            value=existing.get("fatality_count_reported_num", ""),
            placeholder="e.g. 25",
            key=f"{label_widget_prefix}_fatality_count_reported_num",
        )
        injuries_present = st.selectbox(
            "H. Injuries present",
            YES_NO_OPTIONS,
            index=option_index(YES_NO_OPTIONS, existing.get("injuries_present", "")),
            key=f"{label_widget_prefix}_injuries_present",
        )
        injury_count_text = st.text_input(
            "H. Injury count text",
            value=existing.get("injury_count_reported_text", ""),
            placeholder="e.g. 34, several, unclear",
            key=f"{label_widget_prefix}_injury_count_reported_text",
        )
        injury_count_num = st.text_input(
            "H. Injury count numeric",
            value=existing.get("injury_count_reported_num", ""),
            placeholder="e.g. 34",
            key=f"{label_widget_prefix}_injury_count_reported_num",
        )
        weapon_type_text = st.text_input(
            "I. Weapon type text",
            value=existing.get("weapon_type_text", ""),
            placeholder="e.g. AK-47; mortar fire; suicide bomb",
            key=f"{label_widget_prefix}_weapon_type_text",
        )
        weapon_use_class = st.selectbox(
            "I. Weapon use class",
            WEAPON_OPTIONS,
            index=option_index(WEAPON_OPTIONS, existing.get("weapon_use_class", "")),
            key=f"{label_widget_prefix}_weapon_use_class",
        )
    with label_col3:
        st.caption("Location is assigned later from the matched rebel actor; these fields only audit that assumption.")
        article_location_country_text = st.text_input(
            "J. Article location country",
            value=existing.get("article_location_country_text", ""),
            placeholder="optional: country/countries reported by article",
            key=f"{label_widget_prefix}_article_location_country_text",
        )
        location_matches_actor_country = st.selectbox(
            "J. Location matches actor country",
            LOCATION_MATCH_OPTIONS,
            index=option_index(LOCATION_MATCH_OPTIONS, existing.get("location_matches_actor_country", "")),
            key=f"{label_widget_prefix}_location_matches_actor_country",
        )
        multiple_events_mentioned = st.selectbox(
            "K. Multiple events mentioned",
            MULTIPLE_OPTIONS,
            index=option_index(MULTIPLE_OPTIONS, existing.get("multiple_events_mentioned", "")),
            key=f"{label_widget_prefix}_multiple_events_mentioned",
        )
        main_story_confidence = st.selectbox(
            "L. Main story confidence",
            CONFIDENCE_OPTIONS,
            index=option_index(CONFIDENCE_OPTIONS, existing.get("main_story_confidence", "")),
            key=f"{label_widget_prefix}_main_story_confidence",
        )
    notes_col, save_col = st.columns([2, 1])
    with notes_col:
        annotator_notes = st.text_area(
            "Annotator notes",
            value=existing.get("annotator_notes", ""),
            height=100,
            key=f"{label_widget_prefix}_annotator_notes",
        )
    with save_col:
        flagged = st.checkbox(
            "Flag as uncertain / needs review",
            value=bool_value(existing.get("flagged", False)),
            key=f"{label_widget_prefix}_flagged",
        )
        st.caption(f"Saving to `{LABELS_PATH}`")
        save_clicked = st.button(
            "Save label",
            type="primary",
            use_container_width=True,
            key=f"{label_widget_prefix}_save",
        )

details_col1, details_col2 = st.columns(2)
with details_col1.expander("Pre-LLM Rebel Match", expanded=False):
    st.dataframe(compact_metadata(row, pre_cols), use_container_width=True, height=320)
with details_col2.expander("Matched actor records", expanded=False):
    match_records_df = matched_actor_records_table(match_summary["records"])
    if len(match_records_df):
        st.dataframe(match_records_df, use_container_width=True, hide_index=True, height=220)
    else:
        st.info("No parsed matched actor records available for this article.")

if save_clicked:
    record = {
        "GlobalEventID": eid,
        "annotator_id": current_annotator,
        "split": "",
        "conflict_relevance": conflict_relevance,
        "event_context": event_context,
        "event_modality": event_modality,
        "matched_actor_role": matched_actor_role,
        "event_time_relation": event_time_relation,
        "any_matched_actor_mentioned": any_matched_actor_mentioned,
        "any_physical_violence_mentioned": any_physical_violence_mentioned,
        "any_fatalities_mentioned": any_fatalities_mentioned,
        "any_injuries_mentioned": any_injuries_mentioned,
        "fatality_count_anywhere_num": fatality_count_anywhere_num,
        "injury_count_anywhere_num": injury_count_anywhere_num,
        "any_weapon_mentioned": any_weapon_mentioned,
        "weapon_type_anywhere_text": weapon_type_anywhere_text,
        "physical_violence_occurred": physical_violence_occurred,
        "fatalities_present": fatalities_present,
        "fatality_count_reported_text": fatality_count_text,
        "fatality_count_reported_num": fatality_count_num,
        "injuries_present": injuries_present,
        "injury_count_reported_text": injury_count_text,
        "injury_count_reported_num": injury_count_num,
        "weapon_type_text": weapon_type_text,
        "weapon_use_class": weapon_use_class,
        "article_location_country_text": article_location_country_text,
        "location_matches_actor_country": location_matches_actor_country,
        "multiple_events_mentioned": multiple_events_mentioned,
        "main_story_confidence": main_story_confidence,
        "annotator_notes": annotator_notes,
        "flagged": flagged,
    }
    if conflict_relevance == "irrelevant":
        record.update(IRRELEVANT_MAIN_STORY_DEFAULTS)
    save_label(record)
    next_labeled_count = len(labeled_ids) + 1
    next_overlap_count = current_overlap_count + (1 if eid in other_labeled_ids else 0)
    max_overlap_after_following_label = int((next_labeled_count + 1) * IRR_OVERLAP_SHARE)
    allow_following_overlap = next_overlap_count < max_overlap_after_following_label
    normal_queue_mode = (not irr_only) and show_mode in {"Unlabeled first", "Unlabeled only"}
    next_candidates = [
        i for i in range(len(pool))
        if safe_get(pool.iloc[i], "GlobalEventID") not in labeled_ids
        and safe_get(pool.iloc[i], "GlobalEventID") != eid
        and (
            not normal_queue_mode
            or allow_following_overlap
            or not bool(pool.iloc[i].get("_labeled_by_others", False))
        )
    ]
    if next_candidates:
        st.session_state["pending_article_selector_v2"] = random.choice(next_candidates)
    st.session_state["scroll_top_v2"] = True
    st.rerun()

if st.session_state.pop("scroll_top_v2", False):
    components.html("<script>window.parent.document.querySelector('.main').scrollTo(0,0);</script>", height=0)

st.divider()
st.subheader("Saved labels")
labels_latest = load_labels(LABELS_PATH)
if len(labels_latest):
    editable_labels = labels_latest.fillna("").copy()
    if "flagged" in editable_labels.columns:
        editable_labels["flagged"] = editable_labels["flagged"].map(bool_value)
    editable_labels.insert(0, "_delete", False)
    edited_labels = st.data_editor(
        editable_labels,
        use_container_width=True,
        hide_index=True,
        height=360,
        key="saved_labels_editor_v2",
        column_config={
            "_delete": st.column_config.CheckboxColumn("Delete", help="Mark row for deletion."),
            "conflict_relevance": st.column_config.SelectboxColumn("conflict_relevance", options=RELEVANCE_OPTIONS),
            "event_context": st.column_config.SelectboxColumn("event_context", options=EVENT_CONTEXT_OPTIONS),
            "event_modality": st.column_config.SelectboxColumn("event_modality", options=EVENT_MODALITY_OPTIONS),
            "matched_actor_role": st.column_config.SelectboxColumn("matched_actor_role", options=ACTOR_ROLE_OPTIONS),
            "event_time_relation": st.column_config.SelectboxColumn("event_time_relation", options=TIME_RELATION_OPTIONS),
            "any_matched_actor_mentioned": st.column_config.SelectboxColumn(
                "any_matched_actor_mentioned",
                options=MENTION_OPTIONS,
            ),
            "any_physical_violence_mentioned": st.column_config.SelectboxColumn(
                "any_physical_violence_mentioned",
                options=MENTION_OPTIONS,
            ),
            "any_fatalities_mentioned": st.column_config.SelectboxColumn(
                "any_fatalities_mentioned",
                options=MENTION_OPTIONS,
            ),
            "any_injuries_mentioned": st.column_config.SelectboxColumn(
                "any_injuries_mentioned",
                options=MENTION_OPTIONS,
            ),
            "any_weapon_mentioned": st.column_config.SelectboxColumn(
                "any_weapon_mentioned",
                options=MENTION_OPTIONS,
            ),
            "fatality_count_anywhere_num": st.column_config.TextColumn("fatality_count_anywhere_num"),
            "injury_count_anywhere_num": st.column_config.TextColumn("injury_count_anywhere_num"),
            "physical_violence_occurred": st.column_config.SelectboxColumn("physical_violence_occurred", options=YES_NO_OPTIONS),
            "fatalities_present": st.column_config.SelectboxColumn("fatalities_present", options=YES_NO_OPTIONS),
            "fatality_count_reported_num": st.column_config.TextColumn("fatality_count_reported_num"),
            "injuries_present": st.column_config.SelectboxColumn("injuries_present", options=YES_NO_OPTIONS),
            "injury_count_reported_num": st.column_config.TextColumn("injury_count_reported_num"),
            "weapon_use_class": st.column_config.SelectboxColumn("weapon_use_class", options=WEAPON_OPTIONS),
            "location_matches_actor_country": st.column_config.SelectboxColumn(
                "location_matches_actor_country",
                options=LOCATION_MATCH_OPTIONS,
            ),
            "multiple_events_mentioned": st.column_config.SelectboxColumn(
                "multiple_events_mentioned",
                options=MULTIPLE_OPTIONS,
            ),
            "main_story_confidence": st.column_config.SelectboxColumn(
                "main_story_confidence",
                options=CONFIDENCE_OPTIONS,
            ),
            "flagged": st.column_config.CheckboxColumn("flagged"),
        },
    )
    selected_delete_count = int(edited_labels["_delete"].fillna(False).sum()) if "_delete" in edited_labels else 0
    duplicate_keys = edited_labels.drop(columns=["_delete"], errors="ignore").duplicated(
        subset=["GlobalEventID", "annotator_id"],
        keep=False,
    )
    if duplicate_keys.any():
        st.warning("Duplicate GlobalEventID + annotator_id rows found. Saving table edits will keep duplicates.")

    save_table_col, delete_table_col, download_col = st.columns([1, 1, 2])
    with save_table_col:
        if st.button("Save table edits", use_container_width=True):
            save_labels_dataframe(edited_labels)
            st.success("Saved table edits.")
            st.rerun()
    with delete_table_col:
        if st.button(
            f"Delete selected ({selected_delete_count})",
            use_container_width=True,
            disabled=selected_delete_count == 0,
        ):
            kept_labels = edited_labels[~edited_labels["_delete"].fillna(False)].copy()
            save_labels_dataframe(kept_labels)
            st.success(f"Deleted {selected_delete_count} label row(s).")
            st.rerun()
    with download_col:
        st.download_button(
            "Download v2 labels CSV",
            data=labels_latest.to_csv(index=False).encode("utf-8"),
            file_name="manual_labels_full_v2.csv",
            mime="text/csv",
            use_container_width=True,
        )
else:
    st.info("No v2 labels saved yet.")
