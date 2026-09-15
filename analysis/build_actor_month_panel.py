#!/usr/bin/env python3
"""Build an actor-month rolling-risk-set panel.

The LLM labels are used only to construct actor-month regressors. Geography and
outcomes are constructed from UCDP GED and, only when an actor has no recent GED
country in the lookback window, UCDP actor metadata.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_ARTICLE_LABELS = ROOT_DIR / "data" / "processed" / "article_labels_compact.csv"
DEFAULT_GED = ROOT_DIR / "data" / "external" / "GEDEvent_v26_1.csv"
DEFAULT_ACTORS = ROOT_DIR / "data" / "reference" / "Actor_v26_1.csv"
DEFAULT_ACTOR_GROUPS = ROOT_DIR / "data" / "reference" / "actor_group_crosswalk.csv"
DEFAULT_OUTPUT = ROOT_DIR / "data" / "processed" / "actor_month_panel_rebuilt.csv"

LABEL_FIELDS = [
    "conflict_relevance",
    "event_context",
    "event_modality",
    "physical_violence_occurred",
    "fatalities_present",
    "any_fatalities_mentioned",
    "any_physical_violence_mentioned",
    "any_weapon_mentioned",
]

BINARY_SHARE_SPECS = [
    ("conflict_relevance", "direct"),
    ("event_modality", "realized"),
    ("physical_violence_occurred", "yes"),
    ("fatalities_present", "yes"),
    ("any_fatalities_mentioned", "yes"),
    ("any_physical_violence_mentioned", "yes"),
    ("any_weapon_mentioned", "yes"),
]

ARTICLE_USECOLS = [
    "GlobalEventID",
    "date",
    "SQLDATE",
    "month",
    "actor_key",
    "ucdp_actor_id_norm",
    "matched_actor_id_norm",
    "matched_actor_id",
    "actor_name",
    "matched_actor_name",
    "article_actor_weight",
    *LABEL_FIELDS,
]

GED_USECOLS = [
    "date_start",
    "side_a_new_id",
    "side_a",
    "side_b_new_id",
    "side_b",
    "country",
    "best",
]

ACTOR_USECOLS = [
    "ActorId",
    "NameData",
    "NameOrigFullEng",
    "Location",
    "Region",
]

GROUP_USECOLS = [
    "analysis_actor_key",
    "analysis_actor_name",
    "representative_ucdp_actor_id",
    "constituent_ucdp_actor_id",
    "constituent_actor_name",
]


def resolve_path(path: str | Path) -> Path:
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = ROOT_DIR / p
    return p


def norm_actor_id(value: object) -> str:
    raw = "" if pd.isna(value) else str(value).strip()
    if not raw:
        return ""
    try:
        return str(int(float(raw)))
    except ValueError:
        return raw


def slug(value: object) -> str:
    raw = "" if pd.isna(value) else str(value).strip().lower()
    raw = re.sub(r"[^a-z0-9]+", "_", raw).strip("_")
    return raw or "blank"


def norm_label(value: object) -> str:
    return "" if pd.isna(value) else str(value).strip().lower()


def clean_country(value: object) -> str:
    raw = "" if pd.isna(value) else str(value).strip()
    raw = re.sub(r"\s+", " ", raw)
    return raw


def split_metadata_countries(value: object) -> list[str]:
    raw = "" if pd.isna(value) else str(value).strip()
    if not raw:
        return []
    countries = [clean_country(part) for part in re.split(r"\s*,\s*", raw) if clean_country(part)]
    return list(dict.fromkeys(countries))


def period_ordinal(period: pd.Period) -> int:
    return period.year * 12 + period.month


def read_existing_columns(path: Path) -> list[str]:
    return pd.read_csv(path, nrows=0).columns.tolist()


def load_actor_groups(path: Path | None) -> tuple[dict[str, dict[str, object]], dict[str, list[str]], list[str]]:
    if path is None or not path.exists():
        return {}, {}, []
    header = read_existing_columns(path)
    missing = set(GROUP_USECOLS) - set(header)
    if missing:
        raise SystemExit(f"{path} is missing required columns: {sorted(missing)}")
    groups = pd.read_csv(path, usecols=GROUP_USECOLS, dtype=str, low_memory=False).fillna("")
    groups["constituent_ucdp_actor_id"] = groups["constituent_ucdp_actor_id"].map(norm_actor_id)
    groups["representative_ucdp_actor_id"] = groups["representative_ucdp_actor_id"].map(norm_actor_id)
    groups = groups[groups["analysis_actor_key"].str.strip().ne("") & groups["constituent_ucdp_actor_id"].ne("")]

    id_to_group: dict[str, dict[str, object]] = {}
    group_constituents: dict[str, list[str]] = {}
    group_notes: list[str] = []
    for group_key, group in groups.groupby("analysis_actor_key", dropna=False):
        constituents = sorted(group["constituent_ucdp_actor_id"].dropna().astype(str).unique().tolist())
        representative = first_nonempty(group["representative_ucdp_actor_id"])
        if not representative:
            representative = constituents[0] if constituents else ""
        analysis_name = str(first_nonempty(group["analysis_actor_name"]) or group_key)
        group_constituents[str(group_key)] = constituents
        group_notes.append(f"{group_key}: {analysis_name} = {', '.join(constituents)}")
        for actor_id in constituents:
            id_to_group[actor_id] = {
                "analysis_actor_key": str(group_key),
                "analysis_actor_name": analysis_name,
                "representative_ucdp_actor_id": representative,
                "constituent_ucdp_actor_ids": "; ".join(constituents),
            }
    return id_to_group, group_constituents, group_notes


def first_nonempty(series: pd.Series) -> object:
    for value in series:
        if not pd.isna(value) and str(value).strip():
            return value
    return ""


def parse_article_month(df: pd.DataFrame) -> pd.Series:
    if "date" in df.columns:
        parsed = pd.to_datetime(df["date"], errors="coerce")
    else:
        parsed = pd.Series(pd.NaT, index=df.index)
    if parsed.isna().any() and "SQLDATE" in df.columns:
        sql_dates = pd.to_datetime(df["SQLDATE"].astype(str), format="%Y%m%d", errors="coerce")
        parsed = parsed.fillna(sql_dates)
    if parsed.isna().any() and "month" in df.columns:
        month_dates = pd.to_datetime(df["month"].astype(str) + "-01", errors="coerce")
        parsed = parsed.fillna(month_dates)
    return parsed.dt.to_period("M").astype(str)


def load_article_actor_labels(
    path: Path,
    start_month: str,
    end_month: str,
    actor_group_map: dict[str, dict[str, object]] | None = None,
    exclude_multi_actor_articles: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    actor_group_map = actor_group_map or {}
    header = read_existing_columns(path)
    usecols = [col for col in ARTICLE_USECOLS if col in header]
    missing = {"GlobalEventID", "date", "month"} - set(usecols)
    if "GlobalEventID" in missing:
        raise SystemExit(f"{path} must contain GlobalEventID")

    df = pd.read_csv(path, usecols=usecols, dtype={"GlobalEventID": str}, low_memory=False)
    df["GlobalEventID"] = df["GlobalEventID"].astype(str)
    df["month"] = parse_article_month(df)

    if "ucdp_actor_id_norm" not in df.columns:
        if "matched_actor_id_norm" in df.columns:
            df["ucdp_actor_id_norm"] = df["matched_actor_id_norm"]
        elif "matched_actor_id" in df.columns:
            df["ucdp_actor_id_norm"] = df["matched_actor_id"]
        else:
            raise SystemExit(f"{path} must contain a matched UCDP actor ID column")
    df["ucdp_actor_id_norm"] = df["ucdp_actor_id_norm"].map(norm_actor_id)
    df["original_ucdp_actor_id_norm"] = df["ucdp_actor_id_norm"]

    if "actor_key" not in df.columns:
        df["actor_key"] = df["ucdp_actor_id_norm"]
    df["actor_key"] = df["actor_key"].map(norm_actor_id)
    df.loc[df["actor_key"].eq(""), "actor_key"] = df.loc[df["actor_key"].eq(""), "ucdp_actor_id_norm"]
    df["original_actor_key"] = df["actor_key"]

    if "actor_name" not in df.columns:
        df["actor_name"] = df.get("matched_actor_name", "")
    df["actor_name"] = df["actor_name"].fillna("").astype(str).str.strip()
    df["original_actor_name"] = df["actor_name"]

    if "article_actor_weight" in df.columns:
        df["article_actor_weight"] = pd.to_numeric(df["article_actor_weight"], errors="coerce")
    else:
        df["article_actor_weight"] = np.nan

    # The input file is actor-country expanded. Collapse back to one original
    # article-actor row before computing X_{g,t}; country columns are ignored.
    # Actor-group mapping is applied after this step so grouped actor weights
    # can be summed correctly.
    sort_cols = ["GlobalEventID", "original_actor_key"]
    original_article_actor = df.sort_values(sort_cols).drop_duplicates(sort_cols, keep="first").copy()
    original_article_actor = original_article_actor[
        original_article_actor["month"].between(start_month, end_month)
        & original_article_actor["original_actor_key"].ne("")
        & original_article_actor["original_ucdp_actor_id_norm"].ne("")
    ].copy()

    actor_counts = original_article_actor.groupby("GlobalEventID", dropna=False)["original_actor_key"].transform("nunique")
    fallback_weight = np.divide(
        1.0,
        actor_counts.to_numpy(dtype=float),
        out=np.zeros(len(original_article_actor), dtype=float),
        where=actor_counts.to_numpy(dtype=float) > 0,
    )
    fallback_weight = pd.Series(fallback_weight, index=original_article_actor.index)
    weight = pd.to_numeric(original_article_actor["article_actor_weight"], errors="coerce")
    original_article_actor["article_actor_weight"] = weight.where(weight.gt(0), fallback_weight).fillna(fallback_weight)

    original_article_actor["constituent_ucdp_actor_ids"] = original_article_actor["original_ucdp_actor_id_norm"]
    original_article_actor["actor_group_applied"] = False
    for actor_id, group_info in actor_group_map.items():
        mask = original_article_actor["original_ucdp_actor_id_norm"].eq(actor_id)
        if not mask.any():
            continue
        original_article_actor.loc[mask, "actor_key"] = str(group_info["analysis_actor_key"])
        original_article_actor.loc[mask, "actor_name"] = str(group_info["analysis_actor_name"])
        original_article_actor.loc[mask, "ucdp_actor_id_norm"] = str(group_info["representative_ucdp_actor_id"])
        original_article_actor.loc[mask, "constituent_ucdp_actor_ids"] = str(group_info["constituent_ucdp_actor_ids"])
        original_article_actor.loc[mask, "actor_group_applied"] = True

    label_cols = [col for col in LABEL_FIELDS if col in original_article_actor.columns]
    first_cols = [
        "month",
        "ucdp_actor_id_norm",
        "actor_name",
        "constituent_ucdp_actor_ids",
        "actor_group_applied",
        *label_cols,
    ]
    agg_spec = {
        "article_actor_weight": "sum",
        "original_actor_key": lambda s: "; ".join(sorted(set(str(v) for v in s if str(v).strip()))),
        "original_ucdp_actor_id_norm": lambda s: "; ".join(sorted(set(str(v) for v in s if str(v).strip()))),
        "original_actor_name": lambda s: "; ".join(sorted(set(str(v) for v in s if str(v).strip()))),
    }
    agg_spec.update({col: "first" for col in first_cols})
    article_actor = (
        original_article_actor.groupby(["GlobalEventID", "actor_key"], dropna=False)
        .agg(agg_spec)
        .reset_index()
        .rename(
            columns={
                "original_actor_key": "source_actor_keys",
                "original_ucdp_actor_id_norm": "source_ucdp_actor_ids",
                "original_actor_name": "source_actor_names",
            }
        )
    )
    analysis_actor_counts = article_actor.groupby("GlobalEventID", dropna=False)["actor_key"].transform("nunique")
    excluded_article_ids = article_actor.loc[analysis_actor_counts.gt(1), "GlobalEventID"].drop_duplicates()
    article_diagnostics: dict[str, object] = {
        "n_unique_articles_before_remaining_multi_actor_exclusion": int(article_actor["GlobalEventID"].nunique()),
        "n_article_analysis_actor_rows_before_remaining_multi_actor_exclusion": int(len(article_actor)),
        "n_remaining_multi_actor_articles_after_grouping": int(excluded_article_ids.nunique()),
        "n_remaining_multi_actor_article_rows_after_grouping": int(analysis_actor_counts.gt(1).sum()),
        "exclude_remaining_multi_actor_articles": bool(exclude_multi_actor_articles),
    }
    if exclude_multi_actor_articles and not excluded_article_ids.empty:
        article_actor = article_actor.loc[analysis_actor_counts.eq(1)].copy()
    article_diagnostics.update(
        {
            "n_unique_articles_used_for_llm_actor_month": int(article_actor["GlobalEventID"].nunique()),
            "n_article_analysis_actor_rows_used_for_llm_actor_month": int(len(article_actor)),
            "n_excluded_remaining_multi_actor_articles": int(excluded_article_ids.nunique())
            if exclude_multi_actor_articles
            else 0,
        }
    )

    actor_dim = (
        article_actor.groupby(["actor_key", "ucdp_actor_id_norm", "constituent_ucdp_actor_ids"], dropna=False)
        .agg(actor_name=("actor_name", first_nonempty))
        .reset_index()
        .sort_values(["actor_key", "ucdp_actor_id_norm"])
        .reset_index(drop=True)
    )
    return article_actor, actor_dim, article_diagnostics


def weighted_positive_count(df: pd.DataFrame, field: str, positive_value: str) -> float:
    if field not in df.columns:
        return 0.0
    return float(df.loc[df[field].map(norm_label).eq(positive_value), "article_actor_weight"].sum())


def aggregate_llm_actor_month(article_actor: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["actor_key", "ucdp_actor_id_norm", "actor_name", "month"]
    base = (
        article_actor.groupby(group_cols, dropna=False)
        .agg(
            n_articles=("article_actor_weight", "sum"),
            n_unique_articles_raw=("GlobalEventID", "nunique"),
            n_article_actor_rows_raw=("GlobalEventID", "size"),
            mean_article_actor_weight=("article_actor_weight", "mean"),
        )
        .reset_index()
    )
    pieces = [base]

    for field, positive in BINARY_SHARE_SPECS:
        if field not in article_actor.columns:
            continue
        counts = (
            article_actor.assign(_is_positive=article_actor[field].map(norm_label).eq(positive).astype(float))
            .assign(_weighted_positive=lambda x: x["_is_positive"] * x["article_actor_weight"])
            .groupby(group_cols, dropna=False)["_weighted_positive"]
            .sum()
            .reset_index(name=f"n_{field}_{slug(positive)}")
        )
        pieces.append(counts)

    if "event_context" in article_actor.columns:
        tmp = article_actor.copy()
        tmp["_event_context_slug"] = tmp["event_context"].map(slug)
        counts = (
            tmp.groupby(group_cols + ["_event_context_slug"], dropna=False)["article_actor_weight"]
            .sum()
            .unstack(fill_value=0)
            .reset_index()
        )
        counts.columns = [
            *group_cols,
            *[f"n_event_context_{str(col)}" for col in counts.columns[len(group_cols) :]],
        ]
        pieces.append(counts)

    panel = pieces[0]
    for piece in pieces[1:]:
        panel = panel.merge(piece, on=group_cols, how="left")
    panel = panel.fillna(0)

    n_cols = [col for col in panel.columns if col.startswith("n_") and col not in {"n_unique_articles_raw", "n_article_actor_rows_raw"}]
    denom = panel["n_articles"].replace(0, np.nan)
    for col in n_cols:
        if col == "n_articles":
            continue
        panel[f"share_{col[2:]}"] = panel[col] / denom
        panel[f"log1p_{col}"] = np.log1p(pd.to_numeric(panel[col], errors="coerce").fillna(0))
    panel["log1p_n_articles"] = np.log1p(pd.to_numeric(panel["n_articles"], errors="coerce").fillna(0))
    return panel.fillna(0)


def load_actor_metadata(path: Path) -> pd.DataFrame:
    actor_meta = pd.read_csv(path, usecols=lambda c: c in set(ACTOR_USECOLS), encoding="latin1", dtype=str, low_memory=False)
    actor_meta["ucdp_actor_id_norm"] = actor_meta["ActorId"].map(norm_actor_id)
    actor_meta["metadata_countries"] = actor_meta["Location"].map(split_metadata_countries)
    return actor_meta[
        [
            "ucdp_actor_id_norm",
            "NameData",
            "NameOrigFullEng",
            "Location",
            "Region",
            "metadata_countries",
        ]
    ].drop_duplicates("ucdp_actor_id_norm")


def load_ged(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.Period]:
    ged = pd.read_csv(path, usecols=lambda c: c in set(GED_USECOLS), low_memory=False)
    ged["date"] = pd.to_datetime(ged["date_start"], errors="coerce")
    ged = ged[ged["date"].notna()].copy()
    ged["month"] = ged["date"].dt.to_period("M")
    ged["country"] = ged["country"].map(clean_country)
    ged = ged[ged["country"].ne("")]
    ged["best"] = pd.to_numeric(ged["best"], errors="coerce").fillna(0)

    country_month = (
        ged.groupby(["month", "country"], dropna=False)["best"]
        .sum()
        .reset_index(name="ucdp_best_fatalities_country_month")
    )

    base_cols = ["month", "country"]
    side_a = ged[base_cols + ["side_a_new_id", "side_a"]].rename(
        columns={"side_a_new_id": "ucdp_actor_id_norm", "side_a": "ged_actor_name"}
    )
    side_b = ged[base_cols + ["side_b_new_id", "side_b"]].rename(
        columns={"side_b_new_id": "ucdp_actor_id_norm", "side_b": "ged_actor_name"}
    )
    actor_country_month = pd.concat([side_a, side_b], ignore_index=True)
    actor_country_month["ucdp_actor_id_norm"] = actor_country_month["ucdp_actor_id_norm"].map(norm_actor_id)
    actor_country_month = actor_country_month[
        actor_country_month["ucdp_actor_id_norm"].ne("")
        & actor_country_month["country"].ne("")
    ].drop_duplicates(["ucdp_actor_id_norm", "country", "month"])

    return actor_country_month, country_month, ged["month"].max()


def make_actor_history(actor_country_month: pd.DataFrame) -> dict[str, list[tuple[int, str]]]:
    history: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for row in actor_country_month.itertuples(index=False):
        history[str(row.ucdp_actor_id_norm)].append((period_ordinal(row.month), str(row.country)))
    for actor_id in history:
        history[actor_id].sort(key=lambda item: item[0])
    return history


def split_actor_ids(value: object) -> list[str]:
    raw = "" if pd.isna(value) else str(value).strip()
    if not raw:
        return []
    return [norm_actor_id(part) for part in re.split(r"\s*;\s*", raw) if norm_actor_id(part)]


def rolling_countries(
    actor_ids: Iterable[str],
    month: pd.Period,
    lookback_months: int,
    history: dict[str, list[tuple[int, str]]],
) -> list[str]:
    current = period_ordinal(month)
    start = current - lookback_months
    countries = {
        country
        for actor_id in actor_ids
        for event_ord, country in history.get(actor_id, [])
        if start <= event_ord <= current
    }
    return sorted(countries)


def make_country_month_lookup(country_month: pd.DataFrame) -> dict[tuple[int, str], float]:
    return {
        (period_ordinal(row.month), str(row.country)): float(row.ucdp_best_fatalities_country_month)
        for row in country_month.itertuples(index=False)
    }


def fatalities_for_countries(
    countries: Iterable[str],
    month: pd.Period,
    lookup: dict[tuple[int, str], float],
) -> float:
    ord_month = period_ordinal(month)
    return float(sum(lookup.get((ord_month, country), 0.0) for country in countries))


def balance_actor_month_panel(
    llm_actor_month: pd.DataFrame,
    actor_dim: pd.DataFrame,
    actor_meta: pd.DataFrame,
    actor_history: dict[str, list[tuple[int, str]]],
    country_month_lookup: dict[tuple[int, str], float],
    ged_max_month: pd.Period,
    start_month: str,
    end_month: str,
    lookback_months: int,
) -> pd.DataFrame:
    periods = pd.period_range(start=start_month, end=end_month, freq="M")
    grid = actor_dim.merge(pd.DataFrame({"month": periods.astype(str)}), how="cross")
    out = grid.merge(llm_actor_month, on=["actor_key", "ucdp_actor_id_norm", "actor_name", "month"], how="left")

    numeric_cols = out.select_dtypes(include=["number"]).columns.tolist()
    out[numeric_cols] = out[numeric_cols].fillna(0)
    for col in out.columns:
        if col.startswith(("n_", "share_", "log1p_", "mean_")):
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0)

    actor_meta_map = actor_meta.set_index("ucdp_actor_id_norm")["metadata_countries"].to_dict()
    risk_countries_col: list[str] = []
    risk_source_col: list[str] = []
    area_size_col: list[int] = []
    current_y_col: list[float] = []
    next_y_col: list[float] = []
    next_available_col: list[bool] = []
    empty_no_metadata_col: list[bool] = []

    max_ord = period_ordinal(ged_max_month)
    for row in out.itertuples(index=False):
        actor_ids = split_actor_ids(getattr(row, "constituent_ucdp_actor_ids", ""))
        if not actor_ids:
            actor_ids = [str(row.ucdp_actor_id_norm)]
        month = pd.Period(str(row.month), freq="M")
        countries = rolling_countries(actor_ids, month, lookback_months, actor_history)
        source = f"ged_rolling_{lookback_months}m"
        if not countries:
            metadata_countries = {
                country
                for actor_id in actor_ids
                for country in (actor_meta_map.get(actor_id, []) or [])
                if country
            }
            countries = sorted(metadata_countries)
            source = "ucdp_actor_metadata_fallback" if countries else "empty_no_metadata"

        area_size = len(countries)
        next_month = month + 1
        next_available = period_ordinal(next_month) <= max_ord

        risk_countries_col.append("; ".join(countries))
        risk_source_col.append(source)
        area_size_col.append(area_size)
        current_y_col.append(fatalities_for_countries(countries, month, country_month_lookup))
        next_y_col.append(
            fatalities_for_countries(countries, next_month, country_month_lookup) if next_available else np.nan
        )
        next_available_col.append(bool(next_available))
        empty_no_metadata_col.append(source == "empty_no_metadata")

    out[f"rolling_risk_countries_{lookback_months}m"] = risk_countries_col
    out["risk_set_source"] = risk_source_col
    out["area_size"] = area_size_col
    out["log1p_area_size"] = np.log1p(out["area_size"])
    out["risk_set_empty_no_metadata"] = empty_no_metadata_col
    out["next_month_outcome_available"] = next_available_col
    out[f"y_ucdp_best_fatalities_current_{lookback_months}m"] = current_y_col
    out[f"y_ucdp_best_fatalities_next_{lookback_months}m"] = next_y_col
    out[f"log1p_y_ucdp_best_fatalities_current_{lookback_months}m"] = np.log1p(
        pd.to_numeric(out[f"y_ucdp_best_fatalities_current_{lookback_months}m"], errors="coerce").fillna(0)
    )
    out[f"log1p_y_ucdp_best_fatalities_next_{lookback_months}m"] = np.log1p(
        pd.to_numeric(out[f"y_ucdp_best_fatalities_next_{lookback_months}m"], errors="coerce")
    )

    periods_out = pd.PeriodIndex(out["month"].astype(str), freq="M")
    out["period_start"] = periods_out.start_time
    out["year"] = periods_out.year
    out["month_of_year"] = periods_out.month
    out["period_index"] = np.arange(len(periods), dtype=float).take(periods_out.month - periods[0].month + 12 * (periods_out.year - periods[0].year))
    out["unit_id"] = out["actor_key"].astype(str)

    front_cols = [
        "actor_key",
        "ucdp_actor_id_norm",
        "constituent_ucdp_actor_ids",
        "actor_name",
        "unit_id",
        "month",
        "period_start",
        "year",
        "month_of_year",
        "period_index",
        f"rolling_risk_countries_{lookback_months}m",
        "risk_set_source",
        "area_size",
        "log1p_area_size",
        "risk_set_empty_no_metadata",
        "next_month_outcome_available",
        f"y_ucdp_best_fatalities_current_{lookback_months}m",
        f"y_ucdp_best_fatalities_next_{lookback_months}m",
        f"log1p_y_ucdp_best_fatalities_current_{lookback_months}m",
        f"log1p_y_ucdp_best_fatalities_next_{lookback_months}m",
    ]
    other_cols = [col for col in out.columns if col not in front_cols]
    return out[front_cols + other_cols].sort_values(["actor_name", "month", "actor_key"]).reset_index(drop=True)


def write_metadata(
    panel: pd.DataFrame,
    output_path: Path,
    article_labels_path: Path,
    ged_path: Path,
    actor_path: Path,
    actor_groups_path: Path | None,
    actor_group_notes: list[str],
    article_diagnostics: dict[str, object],
    start_month: str,
    end_month: str,
    lookback_months: int,
    ged_max_month: pd.Period,
) -> None:
    source_counts = panel["risk_set_source"].value_counts(dropna=False).to_dict()
    y_next = f"y_ucdp_best_fatalities_next_{lookback_months}m"
    metadata = {
        "panel_path": str(output_path),
        "article_labels_path": str(article_labels_path),
        "ged_path": str(ged_path),
        "actor_metadata_path": str(actor_path),
        "actor_groups_path": str(actor_groups_path) if actor_groups_path else "",
        "actor_groups": actor_group_notes,
        "article_actor_grouping_diagnostics": article_diagnostics,
        "analysis_start_month": start_month,
        "analysis_end_month": end_month,
        "lookback_months": lookback_months,
        "rolling_window_rule": f"R_g,t uses UCDP GED countries with actor involvement from month t-{lookback_months} through t, inclusive.",
        "geography_rule": "LLM article-location labels and article-country panel assignments are not used. Geography is from UCDP GED side_a/side_b actor involvement; grouped analysis actors use the union of constituent UCDP actor IDs; empty rolling sets use UCDP Actor_v26_1 Location metadata when available.",
        "remaining_multi_actor_rule": "After applying actor-group crosswalks, articles still matched to more than one analysis actor are excluded from LLM actor-month aggregation.",
        "empty_no_metadata_rule": "If both rolling GED history and UCDP actor metadata are empty, area_size is 0 and fatalities are the empty-set sum, i.e. 0 when the outcome month is observed.",
        "outcome_rule": "Y is total UCDP GED best fatalities in all countries in the predetermined risk set, not actor-linked fatalities.",
        "right_censor_rule": f"Next-month outcomes after the GED maximum month ({ged_max_month}) are set to missing.",
        "n_rows": int(len(panel)),
        "n_actors": int(panel["actor_key"].nunique()),
        "n_months": int(panel["month"].nunique()),
        "n_nonempty_risk_set_rows": int(panel["area_size"].gt(0).sum()),
        "risk_set_source_counts": {str(k): int(v) for k, v in source_counts.items()},
        "n_missing_next_outcome": int(panel[y_next].isna().sum()),
        "created_columns": panel.columns.tolist(),
    }
    json_path = output_path.with_name(output_path.stem + "_metadata.json")
    json_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    md_path = output_path.with_name(output_path.stem + "_metadata.md")
    md_lines = [
        "# Actor-Month Rolling Risk Panel Metadata",
        "",
        f"- Panel: `{output_path}`",
        f"- Analysis months: `{start_month}` through `{end_month}`",
        f"- Lookback: `{lookback_months}` months, using months `t-{lookback_months}` through `t` inclusive.",
        "- LLM role: article labels are aggregated to actor-month regressors only.",
        "- Geography: UCDP GED actor-country event history; UCDP actor metadata fallback for empty rolling windows.",
        "- Actor groups: grouped analysis actors use all listed constituent UCDP actor IDs for GED risk-set construction.",
        "- Remaining multi-actor rule: after grouping, articles still matched to more than one analysis actor are excluded from the LLM aggregation.",
        "- Outcome: total UCDP GED `best` fatalities in the predetermined country risk set.",
        "- Empty no-metadata rows: area size 0; observed outcome months use the empty-set sum of 0.",
        f"- Right censoring: next-month outcomes after `{ged_max_month}` are missing.",
        "",
        "## Counts",
        "",
        f"- Rows: {len(panel):,}",
        f"- Actors: {panel['actor_key'].nunique():,}",
        f"- Months: {panel['month'].nunique():,}",
        f"- Non-empty risk-set rows: {panel['area_size'].gt(0).sum():,}",
        "",
        "## Risk-Set Sources",
        "",
    ]
    for source, count in source_counts.items():
        md_lines.append(f"- `{source}`: {count:,}")
    if actor_group_notes:
        md_lines.extend(["", "## Actor Groups", ""])
        for note in actor_group_notes:
            md_lines.append(f"- {note}")
    md_lines.extend(["", "## Article Grouping Diagnostics", ""])
    for key, value in article_diagnostics.items():
        md_lines.append(f"- `{key}`: {value}")
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--article-labels", type=Path, default=DEFAULT_ARTICLE_LABELS)
    parser.add_argument("--ged", type=Path, default=DEFAULT_GED)
    parser.add_argument("--actors", type=Path, default=DEFAULT_ACTORS)
    parser.add_argument(
        "--actor-groups",
        type=Path,
        default=DEFAULT_ACTOR_GROUPS,
        help="Optional analysis-actor crosswalk. Constituent UCDP IDs are unioned for GED risk sets.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--start-month", default="2020-01")
    parser.add_argument("--end-month", default="2025-12")
    parser.add_argument("--lookback-months", type=int, default=24)
    parser.add_argument(
        "--keep-multi-actor-articles",
        action="store_true",
        help="Do not drop articles that still map to multiple analysis actors after crosswalk grouping.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    article_labels_path = resolve_path(args.article_labels)
    ged_path = resolve_path(args.ged)
    actor_path = resolve_path(args.actors)
    actor_groups_path = resolve_path(args.actor_groups) if args.actor_groups else None
    output_path = resolve_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    actor_group_map, _, actor_group_notes = load_actor_groups(actor_groups_path)
    article_actor, actor_dim, article_diagnostics = load_article_actor_labels(
        article_labels_path,
        args.start_month,
        args.end_month,
        actor_group_map=actor_group_map,
        exclude_multi_actor_articles=not args.keep_multi_actor_articles,
    )
    llm_actor_month = aggregate_llm_actor_month(article_actor)
    actor_meta = load_actor_metadata(actor_path)
    actor_country_month, country_month, ged_max_month = load_ged(ged_path)
    actor_history = make_actor_history(actor_country_month)
    country_month_lookup = make_country_month_lookup(country_month)

    panel = balance_actor_month_panel(
        llm_actor_month=llm_actor_month,
        actor_dim=actor_dim,
        actor_meta=actor_meta,
        actor_history=actor_history,
        country_month_lookup=country_month_lookup,
        ged_max_month=ged_max_month,
        start_month=args.start_month,
        end_month=args.end_month,
        lookback_months=args.lookback_months,
    )
    panel.to_csv(output_path, index=False)
    write_metadata(
        panel=panel,
        output_path=output_path,
        article_labels_path=article_labels_path,
        ged_path=ged_path,
        actor_path=actor_path,
        actor_groups_path=actor_groups_path,
        actor_group_notes=actor_group_notes,
        article_diagnostics=article_diagnostics,
        start_month=args.start_month,
        end_month=args.end_month,
        lookback_months=args.lookback_months,
        ged_max_month=ged_max_month,
    )

    y_next = f"y_ucdp_best_fatalities_next_{args.lookback_months}m"
    print(f"Wrote {output_path} ({len(panel):,} rows)")
    print(f"Actors: {panel['actor_key'].nunique():,}")
    print(f"Months: {panel['month'].nunique():,}")
    print(f"Non-empty risk sets: {panel['area_size'].gt(0).sum():,}")
    print(f"Missing next-month outcomes: {panel[y_next].isna().sum():,}")
    print("Article grouping diagnostics:")
    for key, value in article_diagnostics.items():
        print(f"  {key}: {value}")
    print("Risk-set sources:")
    print(panel["risk_set_source"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
