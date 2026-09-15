from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MATCHES = REPO_ROOT / "data/processed/datasets/02b_strict_rebel_actor_location_matches_high_confidence.csv"
DEFAULT_ACTOR_META = REPO_ROOT / "data/reference/Actor_v26_1.csv"
DEFAULT_RAD_GROUPS = REPO_ROOT / "data/reference/RAD_group_dataset_1_0.csv"
DEFAULT_REBEL_GROUPS = REPO_ROOT / "data/reference/Rebel_groups.csv"

DEFAULT_EVENT_GROUP_OUT = REPO_ROOT / "data/processed/datasets/02b_strict_rebel_actor_event_groups_high_confidence.csv"
DEFAULT_GROUP_META_OUT = REPO_ROOT / "data/reports/strict_rebel_actor_group_metadata.csv"
DEFAULT_REPORT_OUT = REPO_ROOT / "data/reports/strict_rebel_actor_group_metadata_summary.txt"


EVENT_BASE_COLS = [
    "GlobalEventID",
    "SQLDATE",
    "DATEADDED",
    "ActionGeo_CountryCode",
    "EventCode",
    "EventBaseCode",
    "EventRootCode",
    "QuadClass",
    "AvgTone",
    "GoldsteinScale",
    "final_url",
    "domain",
    "source_month",
]


def clean(value: Any) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def id_key(value: Any) -> str:
    value = clean(value)
    if value.endswith(".0"):
        value = value[:-2]
    return value


def join_unique(values: pd.Series, max_items: int = 30) -> str:
    items = []
    seen = set()
    for value in values:
        item = clean(value)
        if not item or item.lower() == "nan" or item in seen:
            continue
        seen.add(item)
        items.append(item)
    items = sorted(items)
    if len(items) <= max_items:
        return "; ".join(items)
    return "; ".join(items[:max_items]) + f"; ... (+{len(items) - max_items})"


def join_unique_ids(values: pd.Series, max_items: int = 30) -> str:
    return join_unique(values.map(id_key), max_items=max_items)


def read_optional(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


def apply_metadata_corrections(actor: pd.DataFrame) -> pd.DataFrame:
    actor = actor.copy()
    taliban_mask = actor["source_actor_id_key"].eq("8079")
    for col in ["Location", "GWNOLoc", "Region"]:
        if col in actor.columns:
            actor.loc[taliban_mask, col] = ""
    return actor


def build_metadata(
    matches_path: Path,
    actor_meta_path: Path,
    rad_groups_path: Path,
    rebel_groups_path: Path,
    event_group_out: Path,
    group_meta_out: Path,
    report_out: Path,
) -> None:
    matches = pd.read_csv(matches_path, low_memory=False)
    matches["source_actor_id_key"] = matches["rebel_source_actor_id"].map(id_key)

    actor = read_optional(actor_meta_path)
    if len(actor):
        actor["source_actor_id_key"] = actor["ActorId"].map(id_key)
        actor_keep = [
            "source_actor_id_key",
            "Location",
            "GWNOLoc",
            "Region",
            "ConflictId",
            "DyadId",
            "NSID",
            "NSCoalition",
            "NSCoalitionID",
            "Splinter",
            "NamePrev",
            "ActorIdPrev",
            "Alliance",
            "NameAlliance",
            "JoinGroup",
            "GroupName",
            "Org",
        ]
        actor = actor[[c for c in actor_keep if c in actor.columns]].drop_duplicates("source_actor_id_key")
        actor = apply_metadata_corrections(actor)

    rad = read_optional(rad_groups_path)
    if len(rad):
        rad["source_actor_id_key"] = rad["Group_UCDP_ID"].map(id_key)
        rad_keep = [
            "source_actor_id_key",
            "Group_UCDP_Earliest_year",
            "Group_UCDP_Latest_year",
            "Group_Information_available",
            "Group_Smallarms",
            "Group_Lightweapons",
            "Group_MCW",
            "Group_Explosives",
            "Group_Other",
            "Group_Evidenceentries",
            "Group_Evidenceentries_qnty",
            "Group_Potential_armament_origins",
        ]
        rad = rad[[c for c in rad_keep if c in rad.columns]].drop_duplicates("source_actor_id_key")

    rebel = read_optional(rebel_groups_path)
    if len(rebel):
        rebel["source_actor_id_key"] = rebel["NSA_UCDP_ID"].map(id_key)
        rebel_keep = [
            "source_actor_id_key",
            "Government_side",
            "Earliest_year_of_NSA",
            "Latest_year_of_NSA",
            "UCDP_Conflict_ID",
            "UCDP_Dyad_ID",
            "Government_UCDP_ID",
            "Total_battle_deaths_of_country",
            "Total_battle_deaths_of_NSA",
            "Total_bd_of_country_over_median",
            "NSA_Entries",
        ]
        rebel = rebel[[c for c in rebel_keep if c in rebel.columns]].drop_duplicates("source_actor_id_key")

    enriched = matches.copy()
    if len(actor):
        enriched = enriched.merge(actor, on="source_actor_id_key", how="left")
    if len(rad):
        enriched = enriched.merge(rad, on="source_actor_id_key", how="left")
    if len(rebel):
        enriched = enriched.merge(rebel, on="source_actor_id_key", how="left")

    group_keys = ["GlobalEventID", "matched_gdelt_actor"]
    event_group_aggs: dict[str, tuple[str, Any]] = {
        "matched_actor_sides": ("matched_actor_side", join_unique),
        "matched_aliases": ("matched_alias", join_unique),
        "possible_rebel_ids": ("rebel_source_actor_id", join_unique_ids),
        "possible_rebel_names": ("rebel_source_actor_name", join_unique),
        "possible_rebel_count": ("rebel_source_actor_id", lambda s: s.map(id_key).nunique()),
        "metadata_locations": ("Location", join_unique),
        "metadata_gwno_locs": ("GWNOLoc", join_unique),
        "metadata_regions": ("Region", join_unique),
        "metadata_government_sides": ("Government_side", join_unique),
        "metadata_conflict_ids": ("ConflictId", join_unique),
        "metadata_dyad_ids": ("DyadId", join_unique),
        "metadata_rad_earliest_year": ("Group_UCDP_Earliest_year", "min"),
        "metadata_rad_latest_year": ("Group_UCDP_Latest_year", "max"),
        "metadata_nsa_earliest_year": ("Earliest_year_of_NSA", "min"),
        "metadata_nsa_latest_year": ("Latest_year_of_NSA", "max"),
        "metadata_information_available": ("Group_Information_available", join_unique),
        "metadata_armament_origins": ("Group_Potential_armament_origins", join_unique),
    }
    available_event_aggs = {k: v for k, v in event_group_aggs.items() if v[0] in enriched.columns}
    first_cols = [c for c in EVENT_BASE_COLS if c in enriched.columns]
    event_groups = (
        enriched.groupby(group_keys, dropna=False)
        .agg(
            **{col: (col, "first") for col in first_cols if col not in group_keys},
            **available_event_aggs,
        )
        .reset_index()
    )
    event_groups["event_month"] = event_groups["SQLDATE"].astype(str).str.slice(0, 6)
    event_group_out.parent.mkdir(parents=True, exist_ok=True)
    event_groups.to_csv(event_group_out, index=False)

    group_meta_aggs: dict[str, tuple[str, Any]] = {
        "unique_events": ("GlobalEventID", "nunique"),
        "match_rows_before_grouping": ("GlobalEventID", "size"),
        "event_countries": ("ActionGeo_CountryCode", join_unique),
        "top_event_country": ("ActionGeo_CountryCode", lambda s: s.fillna("").replace("", pd.NA).value_counts().index[0] if s.fillna("").replace("", pd.NA).notna().any() else ""),
        "possible_rebel_ids": ("rebel_source_actor_id", join_unique_ids),
        "possible_rebel_names": ("rebel_source_actor_name", join_unique),
        "possible_rebel_count": ("rebel_source_actor_id", lambda s: s.map(id_key).nunique()),
        "matched_aliases": ("matched_alias", join_unique),
        "metadata_locations": ("Location", join_unique),
        "metadata_gwno_locs": ("GWNOLoc", join_unique),
        "metadata_regions": ("Region", join_unique),
        "metadata_government_sides": ("Government_side", join_unique),
        "metadata_conflict_ids": ("ConflictId", join_unique),
        "metadata_dyad_ids": ("DyadId", join_unique),
        "metadata_rad_earliest_year": ("Group_UCDP_Earliest_year", "min"),
        "metadata_rad_latest_year": ("Group_UCDP_Latest_year", "max"),
        "metadata_nsa_earliest_year": ("Earliest_year_of_NSA", "min"),
        "metadata_nsa_latest_year": ("Latest_year_of_NSA", "max"),
        "metadata_information_available": ("Group_Information_available", join_unique),
        "metadata_armament_origins": ("Group_Potential_armament_origins", join_unique),
    }
    available_group_aggs = {k: v for k, v in group_meta_aggs.items() if v[0] in enriched.columns}
    group_meta = (
        enriched.groupby("matched_gdelt_actor", dropna=False)
        .agg(**available_group_aggs)
        .reset_index()
        .sort_values(["unique_events", "match_rows_before_grouping"], ascending=False)
    )
    group_meta_out.parent.mkdir(parents=True, exist_ok=True)
    group_meta.to_csv(group_meta_out, index=False)

    report_lines = [
        "Rebel actor group metadata summary",
        f"Input matches: {matches_path}",
        f"Input match rows: {len(matches):,}",
        f"Event-group rows written: {len(event_groups):,}",
        f"Unique matched actor groups: {group_meta['matched_gdelt_actor'].nunique():,}",
        "",
        "Interpretation",
        "Rows are grouped by GlobalEventID + matched_gdelt_actor. Ambiguous aliases keep all possible rebel IDs/names and metadata in semicolon-separated support columns.",
        "Use event-group rows for aggregation to avoid double-counting aliases that map to multiple splinters.",
        "",
        "Top actor groups",
        group_meta.head(35).to_string(index=False),
        "",
        "Outputs",
        f"Event actor groups: {event_group_out}",
        f"Actor group metadata: {group_meta_out}",
    ]
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text("\n".join(report_lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collapse strict actor matches to actor groups and attach rebel metadata.")
    parser.add_argument("--matches", type=Path, default=DEFAULT_MATCHES)
    parser.add_argument("--actor_meta", type=Path, default=DEFAULT_ACTOR_META)
    parser.add_argument("--rad_groups", type=Path, default=DEFAULT_RAD_GROUPS)
    parser.add_argument("--rebel_groups", type=Path, default=DEFAULT_REBEL_GROUPS)
    parser.add_argument("--event_group_out", type=Path, default=DEFAULT_EVENT_GROUP_OUT)
    parser.add_argument("--group_meta_out", type=Path, default=DEFAULT_GROUP_META_OUT)
    parser.add_argument("--report_out", type=Path, default=DEFAULT_REPORT_OUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_metadata(
        matches_path=args.matches,
        actor_meta_path=args.actor_meta,
        rad_groups_path=args.rad_groups,
        rebel_groups_path=args.rebel_groups,
        event_group_out=args.event_group_out,
        group_meta_out=args.group_meta_out,
        report_out=args.report_out,
    )


if __name__ == "__main__":
    main()
