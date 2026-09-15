#!/usr/bin/env python3
"""Construct actor-month nighttime-light outcomes from VIIRS.

The actor geography is fixed at predictor month t. For each actor-month, the
script finds all country--adm_1 regions containing a UCDP event involving the
actor during t-11,...,t. Google Earth Engine is then used to summarize VIIRS
radiance in that same footprint in months t and t+1.

The script can be run in two stages:

1. ``--prepare-only`` writes the auditable UCDP risk-set table without using
   Earth Engine.
2. With ``--ee-project``, it extracts VIIRS data and writes the final panel.

Earth Engine access is free for eligible research use but requires a Google
Cloud project registered for Earth Engine and local authentication.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import urllib.error
import urllib.request
import time
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GED = ROOT / "data" / "external" / "GEDEvent_v26_1.csv"
DEFAULT_GROUPS = ROOT / "data" / "reference" / "actor_group_crosswalk.csv"
DEFAULT_OUTPUT_DIR = ROOT / "results" / "robustness" / "viirs_nightlight_data"

VIIRS_COLLECTION = "NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG"
ADMIN1_COLLECTION = "FAO/GAUL/2015/level1"
TAKEOVER_DATE = pd.Timestamp("2021-08-15")

ACTOR_NAMES = {
    "209": "Hamas",
    "303": "Taliban",
    "group:farc_bloc": "FARC family",
}


def resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else ROOT / value


def norm_id(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").astype("Int64")
    return values.astype(str).replace("<NA>", "")


def load_farc_ids(path: Path) -> set[str]:
    groups = pd.read_csv(path, dtype=str, low_memory=False).fillna("")
    mask = groups["analysis_actor_key"].eq("group:farc_bloc")
    return set(groups.loc[mask, "constituent_ucdp_actor_id"].map(str))


def load_actor_events(ged_path: Path, groups_path: Path) -> pd.DataFrame:
    columns = [
        "id",
        "date_start",
        "side_a_new_id",
        "side_b_new_id",
        "country",
        "adm_1",
        "latitude",
        "longitude",
        "where_prec",
    ]
    ged = pd.read_csv(ged_path, usecols=columns, low_memory=False)
    ged["event_date"] = pd.to_datetime(ged["date_start"], errors="coerce")
    ged["event_month"] = ged["event_date"].dt.to_period("M")
    ged["side_a_id"] = norm_id(ged["side_a_new_id"])
    ged["side_b_id"] = norm_id(ged["side_b_new_id"])
    ged["latitude"] = pd.to_numeric(ged["latitude"], errors="coerce")
    ged["longitude"] = pd.to_numeric(ged["longitude"], errors="coerce")
    ged["where_prec"] = pd.to_numeric(ged["where_prec"], errors="coerce")

    farc_ids = load_farc_ids(groups_path)
    frames: list[pd.DataFrame] = []

    def append_actor(mask: pd.Series, actor_key: str) -> None:
        selected = ged.loc[mask].copy()
        selected["actor_key"] = actor_key
        selected["actor_name"] = ACTOR_NAMES[actor_key]
        frames.append(selected)

    append_actor(ged["side_a_id"].eq("209") | ged["side_b_id"].eq("209"), "209")

    taliban = (
        (
            ged["event_date"].lt(TAKEOVER_DATE)
            & (ged["side_a_id"].eq("303") | ged["side_b_id"].eq("303"))
        )
        | (
            ged["event_date"].ge(TAKEOVER_DATE)
            & (ged["side_a_id"].eq("130") | ged["side_b_id"].eq("130"))
        )
    )
    append_actor(taliban, "303")
    append_actor(
        ged["side_a_id"].isin(farc_ids) | ged["side_b_id"].isin(farc_ids),
        "group:farc_bloc",
    )

    events = pd.concat(frames, ignore_index=True)
    events = events.drop_duplicates(["actor_key", "id"])
    usable = (
        events["event_month"].notna()
        & events["country"].fillna("").astype(str).str.strip().ne("")
        & events["adm_1"].fillna("").astype(str).str.strip().ne("")
        & events["latitude"].between(-90, 90)
        & events["longitude"].between(-180, 180)
    )
    return events.loc[usable].copy()


def representative_regions(window: pd.DataFrame) -> pd.DataFrame:
    """Choose one observed point for every UCDP country--adm_1 region."""
    ordered = window.sort_values(
        ["country", "adm_1", "where_prec", "event_date"],
        na_position="last",
    )
    return ordered.drop_duplicates(["country", "adm_1"], keep="first")


def build_risk_sets(
    events: pd.DataFrame,
    start_month: str,
    end_month: str,
    lookback_months: int,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    start = pd.Period(start_month, freq="M")
    end = pd.Period(end_month, freq="M")
    months = pd.period_range(start, end, freq="M")
    rows: list[dict[str, object]] = []
    geojson_features: list[dict[str, object]] = []

    for actor_key, actor_name in ACTOR_NAMES.items():
        actor_events = events.loc[events["actor_key"].eq(actor_key)]
        for month in months:
            window_start = month - (lookback_months - 1)
            window = actor_events.loc[
                actor_events["event_month"].between(window_start, month)
            ]
            regions = representative_regions(window)
            region_labels = sorted(
                f"{country}::{adm1}"
                for country, adm1 in regions[["country", "adm_1"]].itertuples(index=False)
            )
            row = {
                "actor_key": actor_key,
                "actor_name": actor_name,
                "predictor_month": str(month),
                "outcome_month": str(month + 1),
                "lookback_months": lookback_months,
                "risk_event_count": int(len(window)),
                "risk_region_count": int(len(regions)),
                "risk_regions": "; ".join(region_labels),
            }
            rows.append(row)

            if regions.empty:
                continue
            coordinates = [
                [float(lon), float(lat)]
                for lon, lat in regions[["longitude", "latitude"]].itertuples(index=False)
            ]
            geojson_features.append(
                {
                    "type": "Feature",
                    "geometry": {"type": "MultiPoint", "coordinates": coordinates},
                    "properties": row,
                }
            )

    return pd.DataFrame(rows), geojson_features


def export_with_earth_engine(
    features: list[dict[str, object]],
    project: str,
    output_csv: Path,
    min_cloudfree_observations: int,
    scale: int,
) -> None:
    try:
        import ee  # type: ignore
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "earthengine-api is required. Install it with "
            "`python -m pip install earthengine-api`, then run "
            "`earthengine authenticate`."
        ) from exc

    ee.Initialize(project=project)
    gaul = ee.FeatureCollection(ADMIN1_COLLECTION)
    viirs = ee.ImageCollection(VIIRS_COLLECTION)

    ee_features = []
    for feature in features:
        points = ee.Geometry(feature["geometry"])
        admin_geometry = gaul.filterBounds(points).geometry()
        # Keep verbose region labels in the local audit table. Sending those
        # repeated strings to Earth Engine can exceed its request-description
        # limit and is unnecessary for joining the extracted values back.
        properties = feature["properties"]
        ee_properties = {
            key: properties[key]
            for key in (
                "actor_key",
                "actor_name",
                "predictor_month",
                "outcome_month",
                "lookback_months",
                "risk_event_count",
                "risk_region_count",
            )
        }
        ee_features.append(ee.Feature(admin_geometry, ee_properties))
    combined_reducer = (
        ee.Reducer.mean()
        .combine(ee.Reducer.median(), sharedInputs=True)
        .combine(ee.Reducer.stdDev(), sharedInputs=True)
        .combine(ee.Reducer.count(), sharedInputs=True)
    )

    def summarize(feature: object) -> object:
        feature = ee.Feature(feature)
        geometry = feature.geometry()
        date_t = ee.Date.parse("YYYY-MM", ee.String(feature.get("predictor_month")))
        date_t1 = ee.Date.parse("YYYY-MM", ee.String(feature.get("outcome_month")))
        image_t = ee.Image(viirs.filterDate(date_t, date_t.advance(1, "month")).first())
        image_t1 = ee.Image(viirs.filterDate(date_t1, date_t1.advance(1, "month")).first())
        coverage_t = image_t.select("cf_cvg")
        coverage_t1 = image_t1.select("cf_cvg")
        valid_t = coverage_t.gte(min_cloudfree_observations)
        valid_t1 = coverage_t1.gte(min_cloudfree_observations)
        common_valid = valid_t.And(valid_t1)
        radiance_t = (
            image_t.select("avg_rad").max(0).updateMask(common_valid).rename("radiance_t")
        )
        radiance_t1 = (
            image_t1.select("avg_rad").max(0).updateMask(common_valid).rename("radiance_t1")
        )
        summary_image = (
            radiance_t
            .addBands(radiance_t1)
            .addBands(valid_t.unmask(0).toFloat().rename("valid_t"))
            .addBands(valid_t1.unmask(0).toFloat().rename("valid_t1"))
            .addBands(common_valid.unmask(0).toFloat().rename("common_valid"))
            .addBands(coverage_t.updateMask(common_valid).rename("coverage_t"))
            .addBands(coverage_t1.updateMask(common_valid).rename("coverage_t1"))
            .addBands(radiance_t.gt(1).toFloat().rename("lit_t"))
            .addBands(radiance_t1.gt(1).toFloat().rename("lit_t1"))
        )
        stats = summary_image.reduceRegion(
            reducer=combined_reducer,
            geometry=geometry,
            scale=scale,
            maxPixels=1_000_000_000,
            bestEffort=True,
            tileScale=4,
        )
        metrics = ee.Dictionary(
            {
                "ntl_mean_t": stats.get("radiance_t_mean"),
                "ntl_mean_t1": stats.get("radiance_t1_mean"),
                "ntl_median_t": stats.get("radiance_t_median"),
                "ntl_median_t1": stats.get("radiance_t1_median"),
                "ntl_sd_t": stats.get("radiance_t_stdDev"),
                "ntl_sd_t1": stats.get("radiance_t1_stdDev"),
                "ntl_valid_pixel_count_t": stats.get("radiance_t_count"),
                "ntl_valid_pixel_count_t1": stats.get("radiance_t1_count"),
                "ntl_valid_area_fraction_t": stats.get("valid_t_mean"),
                "ntl_valid_area_fraction_t1": stats.get("valid_t1_mean"),
                "ntl_common_valid_area_fraction": stats.get("common_valid_mean"),
                "ntl_cloudfree_observations_mean_t": stats.get("coverage_t_mean"),
                "ntl_cloudfree_observations_mean_t1": stats.get("coverage_t1_mean"),
                "ntl_lit_fraction_t": stats.get("lit_t_mean"),
                "ntl_lit_fraction_t1": stats.get("lit_t1_mean"),
            }
        )
        mean_t = ee.Number(metrics.get("ntl_mean_t"))
        mean_t1 = ee.Number(metrics.get("ntl_mean_t1"))
        median_t = ee.Number(metrics.get("ntl_median_t"))
        median_t1 = ee.Number(metrics.get("ntl_median_t1"))
        lit_t = ee.Number(metrics.get("ntl_lit_fraction_t"))
        lit_t1 = ee.Number(metrics.get("ntl_lit_fraction_t1"))

        derived = ee.Dictionary(
            {
                "risk_area_km2": geometry.area(1).divide(1_000_000),
                "ntl_mean_change": mean_t1.subtract(mean_t),
                "ntl_log_mean_change": mean_t1.add(1).log().subtract(mean_t.add(1).log()),
                "ntl_median_change": median_t1.subtract(median_t),
                "ntl_log_median_change": median_t1.add(1).log().subtract(median_t.add(1).log()),
                "ntl_lit_fraction_change": lit_t1.subtract(lit_t),
            }
        )
        return feature.set(metrics).set(derived).setGeometry(None)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []
    chunk_size = 2
    total_chunks = math.ceil(len(ee_features) / chunk_size)
    for chunk_number, start in enumerate(range(0, len(ee_features), chunk_size), start=1):
        chunk = ee.FeatureCollection(ee_features[start : start + chunk_size])
        result = chunk.map(summarize)
        url = result.getDownloadURL(filetype="CSV")
        request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        payload: bytes | None = None
        for attempt in range(1, 6):
            try:
                with urllib.request.urlopen(request, timeout=600) as response:
                    payload = response.read()
                break
            except urllib.error.HTTPError as exc:
                if exc.code != 429 or attempt == 5:
                    raise
                delay = 5 * (2 ** (attempt - 1))
                print(
                    f"Earth Engine throttled chunk {chunk_number}; retrying in {delay}s",
                    flush=True,
                )
                time.sleep(delay)
        if payload is None:
            raise RuntimeError(f"No response payload for Earth Engine chunk {chunk_number}")
        frames.append(pd.read_csv(io.BytesIO(payload), dtype={"actor_key": str}))
        print(f"Downloaded Earth Engine chunk {chunk_number}/{total_chunks}", flush=True)

    if not frames:
        raise SystemExit("Earth Engine returned no VIIRS rows")
    pd.concat(frames, ignore_index=True).to_csv(output_csv, index=False)


def merge_risk_metadata(risk_sets: pd.DataFrame, viirs_csv: Path, output_csv: Path) -> pd.DataFrame:
    viirs = pd.read_csv(viirs_csv, dtype={"actor_key": str})
    key = ["actor_key", "predictor_month", "outcome_month"]
    duplicate_metadata = [
        column for column in risk_sets.columns if column in viirs.columns and column not in key
    ]
    viirs = viirs.drop(columns=duplicate_metadata, errors="ignore")
    result = risk_sets.merge(viirs, on=key, how="left", validate="one_to_one")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_csv, index=False)
    return result


def write_metadata(
    path: Path,
    args: argparse.Namespace,
    risk_sets: pd.DataFrame,
    final: pd.DataFrame | None,
) -> None:
    metadata = {
        "ged": str(resolve(args.ged)),
        "actor_groups": str(resolve(args.actor_groups)),
        "start_month": args.start_month,
        "end_month": args.end_month,
        "lookback_months": args.lookback_months,
        "viirs_collection": VIIRS_COLLECTION,
        "admin1_collection": ADMIN1_COLLECTION,
        "min_cloudfree_observations": args.min_cloudfree_observations,
        "scale_meters": args.scale,
        "risk_set_rows": int(len(risk_sets)),
        "nonempty_risk_set_rows": int(risk_sets["risk_region_count"].gt(0).sum()),
        "final_rows": None if final is None else int(len(final)),
        "primary_satellite_outcome": "ntl_log_mean_change",
        "outcome_definition": (
            "log(1 + mean VIIRS radiance in t+1) - "
            "log(1 + mean VIIRS radiance in t), evaluated over the same "
            "UCDP-defined rolling admin1 footprint fixed at t"
        ),
    }
    path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ged", default=str(DEFAULT_GED))
    parser.add_argument("--actor-groups", default=str(DEFAULT_GROUPS))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--start-month", default="2020-01")
    parser.add_argument("--end-month", default="2025-11")
    parser.add_argument("--lookback-months", type=int, default=12)
    parser.add_argument("--min-cloudfree-observations", type=int, default=1)
    parser.add_argument(
        "--scale",
        type=int,
        default=2000,
        help=(
            "Earth Engine scale in metres for regional summary statistics. "
            "The source VIIRS product remains at its native resolution."
        ),
    )
    parser.add_argument("--ee-project", default="")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument(
        "--viirs-csv",
        default="",
        help="Merge an already exported Earth Engine CSV instead of querying Earth Engine.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.lookback_months < 1:
        raise SystemExit("--lookback-months must be positive")

    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    risk_csv = output_dir / "actor_month_viirs_risk_sets.csv"
    risk_geojson = output_dir / "actor_month_viirs_risk_set_points.geojson"
    raw_viirs_csv = output_dir / "actor_month_viirs_earth_engine_raw.csv"
    final_csv = output_dir / "actor_month_viirs_outcomes.csv"
    metadata_json = output_dir / "actor_month_viirs_metadata.json"

    events = load_actor_events(resolve(args.ged), resolve(args.actor_groups))
    risk_sets, features = build_risk_sets(
        events,
        args.start_month,
        args.end_month,
        args.lookback_months,
    )
    risk_sets.to_csv(risk_csv, index=False)
    risk_geojson.write_text(
        json.dumps({"type": "FeatureCollection", "features": features}, indent=2) + "\n",
        encoding="utf-8",
    )

    if args.prepare_only:
        write_metadata(metadata_json, args, risk_sets, None)
        print(f"Wrote {risk_csv}")
        print(f"Wrote {risk_geojson}")
        return

    if args.viirs_csv:
        source_csv = resolve(args.viirs_csv)
    else:
        if not args.ee_project:
            raise SystemExit("Provide --ee-project, --viirs-csv, or use --prepare-only")
        export_with_earth_engine(
            features,
            args.ee_project,
            raw_viirs_csv,
            args.min_cloudfree_observations,
            args.scale,
        )
        source_csv = raw_viirs_csv

    final = merge_risk_metadata(risk_sets, source_csv, final_csv)
    write_metadata(metadata_json, args, risk_sets, final)
    print(f"Wrote {final_csv} ({len(final):,} rows)")
    print(f"Wrote {metadata_json}")


if __name__ == "__main__":
    main()
