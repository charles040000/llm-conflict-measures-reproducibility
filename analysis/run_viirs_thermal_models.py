#!/usr/bin/env python3
"""Fit pooled monthly models for the VIIRS thermal-anomaly outcome."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_viirs_nightlight_models import (
    benjamini_hochberg,
    normal_p_value,
    panel_newey_west,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PANEL = ROOT / "data" / "processed" / "viirs_thermal_panel.csv"
DEFAULT_OUTPUT = ROOT / "results" / "robustness" / "viirs_thermal"
OUTCOME = "fire_nominal_high_pixel_days_log_rate_t1"
CURRENT = "fire_nominal_high_pixel_days_log_rate_t"

SIGNALS = {
    "direct_conflict_relevance": "share_conflict_relevance_direct",
    "realized_event": "share_event_modality_realized",
    "fatalities_mentioned": "share_any_fatalities_mentioned_yes",
    "physical_violence_mentioned": "share_any_physical_violence_mentioned_yes",
    "weapon_mentioned": "share_any_weapon_mentioned_yes",
}

BASE_CONTROLS = [
    CURRENT,
    "fire_log_rate_lag1",
    "fire_log_rate_lag2",
    "log1p_n_articles",
    "log1p_fire_risk_area_km2",
    "time_years",
]


def resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else ROOT / value


def prepare_panel(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path, dtype={"actor_key": str})
    data["period_start"] = pd.to_datetime(data["predictor_month"] + "-01")
    data = data.sort_values(["actor_key", "period_start"]).reset_index(drop=True)
    data["fire_log_rate_lag1"] = data.groupby("actor_key", sort=False)[CURRENT].shift(1)
    data["fire_log_rate_lag2"] = data.groupby("actor_key", sort=False)[CURRENT].shift(2)
    data["log1p_fire_risk_area_km2"] = np.log1p(
        pd.to_numeric(data["fire_risk_area_km2"], errors="coerce")
    )
    month_number = data["period_start"].dt.to_period("M").astype(int)
    data["time_years"] = (month_number - month_number.min()) / 12.0
    data["actor_season"] = (
        data["actor_key"].astype(str) + "_m" + data["period_start"].dt.month.astype(str)
    )
    return data


def build_design(data: pd.DataFrame, signal: str | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    numeric = list(BASE_CONTROLS)
    if signal is not None:
        numeric.append(signal)
    needed = [OUTCOME, "actor_key", "actor_name", "period_start", "actor_season", *numeric]
    frame = data[needed].replace([np.inf, -np.inf], np.nan).copy()
    for column in [OUTCOME, *numeric]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=[OUTCOME, *numeric]).copy()

    continuous = frame[numeric].astype(float)
    continuous = continuous - continuous.mean(axis=0)
    seasonal_fe = pd.get_dummies(
        frame["actor_season"], prefix="actor_month", drop_first=True, dtype=float
    )
    design = pd.concat(
        [pd.Series(1.0, index=frame.index, name="Intercept"), continuous, seasonal_fe],
        axis=1,
    )
    keep = design.std(axis=0).gt(1e-12)
    keep.loc["Intercept"] = True
    return frame, design.loc[:, keep]


def fit_model(
    data: pd.DataFrame,
    model_id: str,
    signal: str | None,
    hac_lags: int,
) -> tuple[pd.DataFrame, dict[str, object], pd.DataFrame]:
    frame, design = build_design(data, signal)
    x = design.to_numpy(dtype=float)
    y = frame[OUTCOME].to_numpy(dtype=float)
    beta = np.linalg.pinv(x.T @ x) @ x.T @ y
    fitted = x @ beta
    residuals = y - fitted
    covariance = panel_newey_west(
        x,
        residuals,
        frame["actor_key"].to_numpy(),
        frame["period_start"].to_numpy(),
        hac_lags,
    )
    se = np.sqrt(np.maximum(np.diag(covariance), 0))
    statistic = beta / np.where(se > 0, se, np.nan)
    coefficients = pd.DataFrame(
        {
            "model_id": model_id,
            "term": design.columns,
            "estimate": beta,
            "hac_se": se,
            "z": statistic,
            "p_value": [normal_p_value(value) for value in statistic],
            "ci_low": beta - 1.96 * se,
            "ci_high": beta + 1.96 * se,
            "is_llm_signal": [column == signal for column in design.columns],
        }
    )
    sse = float(residuals @ residuals)
    sst = float(((y - y.mean()) ** 2).sum())
    actor_ar1 = []
    residual_frame = frame[["actor_key", "actor_name", "period_start"]].copy()
    residual_frame["model_id"] = model_id
    residual_frame["observed"] = y
    residual_frame["fitted"] = fitted
    residual_frame["residual"] = residuals
    for _, group in residual_frame.groupby("actor_key"):
        actor_ar1.append(group.sort_values("period_start")["residual"].autocorr(lag=1))
    diagnostics = {
        "model_id": model_id,
        "signal": signal or "",
        "n": int(len(frame)),
        "actors": int(frame["actor_key"].nunique()),
        "parameters": int(x.shape[1]),
        "rank": int(np.linalg.matrix_rank(x)),
        "condition_number": float(np.linalg.cond(x)),
        "r2": 1 - sse / sst,
        "adjusted_r2": 1 - (sse / (len(y) - x.shape[1])) / (sst / (len(y) - 1)),
        "rmse": float(np.sqrt(np.mean(residuals**2))),
        "mean_actor_residual_ar1": float(np.nanmean(actor_ar1)),
        "hac_lags": int(hac_lags),
        "seasonal_effects": "actor_by_month_of_year",
    }
    return coefficients, diagnostics, residual_frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", default=str(DEFAULT_PANEL))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--hac-lags", type=int, default=3)
    args = parser.parse_args()
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    panel = prepare_panel(resolve(args.panel))

    coefficient_frames = []
    diagnostics_rows = []
    residual_frames = []
    specifications: list[tuple[str, str | None]] = [("controls_only", None)]
    specifications.extend(SIGNALS.items())
    for model_id, signal in specifications:
        coefficients, diagnostics, residuals = fit_model(
            panel, model_id, signal, args.hac_lags
        )
        coefficient_frames.append(coefficients)
        diagnostics_rows.append(diagnostics)
        residual_frames.append(residuals)

    coefficients = pd.concat(coefficient_frames, ignore_index=True)
    diagnostics = pd.DataFrame(diagnostics_rows)
    residuals = pd.concat(residual_frames, ignore_index=True)
    main = coefficients.loc[coefficients["is_llm_signal"]].copy()
    main["fdr_p_value"] = benjamini_hochberg(main["p_value"])
    main["effect_for_10pp_share_change"] = 0.1 * main["estimate"]
    controls = coefficients.loc[
        coefficients["model_id"].eq("controls_only")
        & coefficients["term"].isin(BASE_CONTROLS)
    ].copy()

    coefficients.to_csv(output_dir / "monthly_fire_all_coefficients.csv", index=False)
    diagnostics.to_csv(output_dir / "monthly_fire_model_diagnostics.csv", index=False)
    residuals.to_csv(output_dir / "monthly_fire_model_residuals.csv", index=False)
    main.to_csv(output_dir / "monthly_fire_llm_results.csv", index=False)
    controls.to_csv(output_dir / "monthly_fire_controls_only_coefficients.csv", index=False)
    metadata = {
        "outcome": OUTCOME,
        "outcome_definition": (
            "log(1 + nominal/high-confidence VIIRS fire pixel-days per "
            "calendar day and 1,000 km2 in t+1)"
        ),
        "controls": BASE_CONTROLS,
        "fixed_effects": "actor-specific month-of-year effects",
        "inference": f"within-actor Newey-West HAC, {args.hac_lags} lags",
    }
    (output_dir / "monthly_fire_model_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(diagnostics[["model_id", "n", "r2", "adjusted_r2", "rmse", "mean_actor_residual_ar1"]].to_string(index=False))
    print("\nLLM signals")
    print(main[["model_id", "estimate", "hac_se", "p_value", "fdr_p_value", "ci_low", "ci_high"]].to_string(index=False))


if __name__ == "__main__":
    main()
