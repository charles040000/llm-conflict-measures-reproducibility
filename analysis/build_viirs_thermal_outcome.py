#!/usr/bin/env python3
"""Construct actor-month VIIRS thermal-anomaly outcomes.

For each actor and predictor month t, the geographic footprint is the union of
country--adm_1 regions containing a UCDP event involving that actor during the
preceding 12 months, including t. Daily NASA VNP14A1 thermal anomalies are
aggregated over that fixed footprint for t and t+1.

The primary outcome is the log next-month rate of nominal- or high-confidence
fire pixel-days per 1,000 square kilometres and calendar day. Raw counts,
nighttime detections, high-confidence detections, unique detected pixels, fire
radiative power, current-month values, and log changes are retained.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

from build_viirs_nightlight_outcome import (
    ADMIN1_COLLECTION,
    DEFAULT_GED,
    DEFAULT_GROUPS,
    build_risk_sets,
    load_actor_events,
    resolve,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "results" / "robustness" / "viirs_thermal_data"
FIRE_COLLECTION = "NASA/VIIRS/002/VNP14A1"


def build_ee_features(features: list[dict[str, object]], gaul: object, ee: object) -> list[object]:
    result = []
    for feature in features:
        points = ee.Geometry(feature["geometry"])
        geometry = gaul.filterBounds(points).geometry()
        properties = feature["properties"]
        compact = {
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
        result.append(ee.Feature(geometry, compact))
    return result


def export_fire_metrics(
    features: list[dict[str, object]],
    project: str,
    output_csv: Path,
    scale: int,
) -> None:
    try:
        import ee  # type: ignore
    except ModuleNotFoundError as exc:
        raise SystemExit("Install earthengine-api and authenticate Earth Engine first") from exc

    ee.Initialize(project=project)
    gaul = ee.FeatureCollection(ADMIN1_COLLECTION)
    fires = ee.ImageCollection(FIRE_COLLECTION)
    ee_features = build_ee_features(features, gaul, ee)
    reducer = ee.Reducer.sum().combine(ee.Reducer.mean(), sharedInputs=True)

    def daily_bands(image: object) -> object:
        image = ee.Image(image)
        fire_mask = image.select("FireMask")
        qa = image.select("QA")
        all_fire = fire_mask.gte(7)
        nominal_high = fire_mask.gte(8)
        high = fire_mask.eq(9)
        night = qa.bitwiseAnd(1 << 4).eq(0)
        valid_land = fire_mask.gte(5)
        frp = image.select("MaxFRP").max(0)
        return (
            all_fire.toFloat().rename("all_fire")
            .addBands(nominal_high.toFloat().rename("nominal_high"))
            .addBands(high.toFloat().rename("high"))
            .addBands(nominal_high.And(night).toFloat().rename("night_nominal_high"))
            .addBands(frp.updateMask(nominal_high).unmask(0).rename("frp_nominal_high"))
            .addBands(valid_land.toFloat().rename("valid_land"))
        ).copyProperties(image, ["system:time_start"])

    def summarize_period(month_text: object, geometry: object, suffix: str) -> object:
        start = ee.Date.parse("YYYY-MM", ee.String(month_text))
        end = start.advance(1, "month")
        collection = fires.filterDate(start, end).map(daily_bands)
        days = end.difference(start, "day")
        summed = collection.select(
            [
                "all_fire",
                "nominal_high",
                "high",
                "night_nominal_high",
                "frp_nominal_high",
                "valid_land",
            ]
        ).sum()
        unique = (
            collection.select(["nominal_high", "high"])
            .max()
            .rename(["unique_nominal_high", "unique_high"])
        )
        stats = summed.addBands(unique).reduceRegion(
            reducer=reducer,
            geometry=geometry,
            scale=scale,
            maxPixels=1_000_000_000,
            bestEffort=True,
            tileScale=4,
        )
        return ee.Dictionary(
            {
                f"fire_days_in_period_{suffix}": days,
                f"fire_all_pixel_days_{suffix}": stats.get("all_fire_sum"),
                f"fire_nominal_high_pixel_days_{suffix}": stats.get("nominal_high_sum"),
                f"fire_high_pixel_days_{suffix}": stats.get("high_sum"),
                f"fire_night_nominal_high_pixel_days_{suffix}": stats.get(
                    "night_nominal_high_sum"
                ),
                f"fire_nominal_high_frp_sum_mw_{suffix}": stats.get(
                    "frp_nominal_high_sum"
                ),
                f"fire_unique_nominal_high_pixels_{suffix}": stats.get(
                    "unique_nominal_high_sum"
                ),
                f"fire_unique_high_pixels_{suffix}": stats.get("unique_high_sum"),
                f"fire_valid_land_observations_mean_{suffix}": stats.get(
                    "valid_land_mean"
                ),
            }
        )

    def summarize(feature: object) -> object:
        feature = ee.Feature(feature)
        geometry = feature.geometry()
        area_km2 = geometry.area(1).divide(1_000_000)
        current = summarize_period(feature.get("predictor_month"), geometry, "t")
        following = summarize_period(feature.get("outcome_month"), geometry, "t1")
        metrics = ee.Dictionary(current).combine(following)
        days_t = ee.Number(metrics.get("fire_days_in_period_t"))
        days_t1 = ee.Number(metrics.get("fire_days_in_period_t1"))

        derived = {"fire_risk_area_km2": area_km2}
        for base in (
            "fire_all_pixel_days",
            "fire_nominal_high_pixel_days",
            "fire_high_pixel_days",
            "fire_night_nominal_high_pixel_days",
            "fire_nominal_high_frp_sum_mw",
        ):
            value_t = ee.Number(metrics.get(f"{base}_t"))
            value_t1 = ee.Number(metrics.get(f"{base}_t1"))
            daily_t = value_t.divide(days_t)
            daily_t1 = value_t1.divide(days_t1)
            rate_t = daily_t.divide(area_km2).multiply(1000)
            rate_t1 = daily_t1.divide(area_km2).multiply(1000)
            derived[f"{base}_daily_mean_t"] = daily_t
            derived[f"{base}_daily_mean_t1"] = daily_t1
            derived[f"{base}_daily_rate_per_1000km2_t"] = rate_t
            derived[f"{base}_daily_rate_per_1000km2_t1"] = rate_t1
            derived[f"{base}_log_rate_t"] = rate_t.add(1).log()
            derived[f"{base}_log_rate_t1"] = rate_t1.add(1).log()
            derived[f"{base}_log_rate_change"] = rate_t1.add(1).log().subtract(
                rate_t.add(1).log()
            )

        return feature.set(metrics).set(derived).setGeometry(None)

    frames: list[pd.DataFrame] = []
    chunk_size = 2
    total_chunks = math.ceil(len(ee_features) / chunk_size)
    for chunk_number, start in enumerate(range(0, len(ee_features), chunk_size), start=1):
        result = ee.FeatureCollection(ee_features[start : start + chunk_size]).map(summarize)
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
            raise RuntimeError(f"No response for Earth Engine chunk {chunk_number}")
        frames.append(pd.read_csv(io.BytesIO(payload), dtype={"actor_key": str}))
        print(f"Downloaded fire chunk {chunk_number}/{total_chunks}", flush=True)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(frames, ignore_index=True).to_csv(output_csv, index=False)


def merge_outputs(risk_sets: pd.DataFrame, raw_csv: Path, final_csv: Path) -> pd.DataFrame:
    raw = pd.read_csv(raw_csv, dtype={"actor_key": str})
    keys = ["actor_key", "predictor_month", "outcome_month"]
    duplicates = [column for column in risk_sets if column in raw and column not in keys]
    raw = raw.drop(columns=duplicates, errors="ignore")
    result = risk_sets.merge(raw, on=keys, how="left", validate="one_to_one")
    result.to_csv(final_csv, index=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ged", default=str(DEFAULT_GED))
    parser.add_argument("--actor-groups", default=str(DEFAULT_GROUPS))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--start-month", default="2020-01")
    parser.add_argument("--end-month", default="2025-11")
    parser.add_argument("--lookback-months", type=int, default=12)
    parser.add_argument("--scale", type=int, default=1000)
    parser.add_argument("--ee-project", default="")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()

    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    risk_csv = output_dir / "actor_month_fire_risk_sets.csv"
    risk_geojson = output_dir / "actor_month_fire_risk_set_points.geojson"
    raw_csv = output_dir / "actor_month_viirs_fire_earth_engine_raw.csv"
    final_csv = output_dir / "actor_month_viirs_fire_outcomes.csv"
    metadata_json = output_dir / "actor_month_viirs_fire_metadata.json"

    events = load_actor_events(resolve(args.ged), resolve(args.actor_groups))
    risk_sets, features = build_risk_sets(
        events, args.start_month, args.end_month, args.lookback_months
    )
    risk_sets.to_csv(risk_csv, index=False)
    risk_geojson.write_text(
        json.dumps({"type": "FeatureCollection", "features": features}, indent=2) + "\n",
        encoding="utf-8",
    )
    if args.prepare_only:
        print(f"Wrote {risk_csv}")
        return
    if not args.ee_project:
        raise SystemExit("Provide --ee-project or use --prepare-only")

    export_fire_metrics(features, args.ee_project, raw_csv, args.scale)
    final = merge_outputs(risk_sets, raw_csv, final_csv)
    metadata = {
        "ged": str(resolve(args.ged)),
        "actor_groups": str(resolve(args.actor_groups)),
        "start_month": args.start_month,
        "end_month": args.end_month,
        "lookback_months": args.lookback_months,
        "fire_collection": FIRE_COLLECTION,
        "admin1_collection": ADMIN1_COLLECTION,
        "scale_meters": args.scale,
        "rows": int(len(final)),
        "nonempty_risk_sets": int(risk_sets["risk_region_count"].gt(0).sum()),
        "primary_outcome": "fire_nominal_high_pixel_days_log_rate_t1",
        "primary_definition": (
            "log(1 + nominal/high-confidence thermal-anomaly pixel-days per "
            "calendar day and 1,000 km2 in t+1)"
        ),
    }
    metadata_json.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {final_csv} ({len(final):,} rows)")


if __name__ == "__main__":
    main()
