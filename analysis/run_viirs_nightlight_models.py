#!/usr/bin/env python3
"""Fit pooled monthly models for the satellite-derived VIIRS outcome.

The dependent variable is the next-month change in log mean nighttime
radiance over an actor's UCDP-defined rolling activity footprint. LLM shares
are entered one at a time against an identical dynamic control specification.
Inference uses a Newey-West covariance estimator that permits serial
correlation within each actor series.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PANEL = ROOT / "data" / "processed" / "viirs_nightlight_panel.csv"
DEFAULT_OUTPUT = ROOT / "results" / "robustness" / "viirs_nightlight"
OUTCOME = "ntl_log_mean_change"

SIGNALS = {
    "direct_conflict_relevance": "share_conflict_relevance_direct",
    "realized_event": "share_event_modality_realized",
    "fatalities_mentioned": "share_any_fatalities_mentioned_yes",
    "physical_violence_mentioned": "share_any_physical_violence_mentioned_yes",
    "weapon_mentioned": "share_any_weapon_mentioned_yes",
}

BASE_CONTROLS = [
    "log_ntl_mean_t",
    "ntl_change_lag1",
    "ntl_change_lag2",
    "log1p_n_articles",
    "log1p_risk_area_km2",
    "time_years",
]


def resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else ROOT / value


def normal_p_value(statistic: float) -> float:
    if not math.isfinite(statistic):
        return float("nan")
    return math.erfc(abs(statistic) / math.sqrt(2.0))


def prepare_panel(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path, dtype={"actor_key": str})
    data["period_start"] = pd.to_datetime(data["predictor_month"] + "-01")
    data = data.sort_values(["actor_key", "period_start"]).reset_index(drop=True)
    data["log_ntl_mean_t"] = np.log1p(pd.to_numeric(data["ntl_mean_t"], errors="coerce"))
    data["log1p_risk_area_km2"] = np.log1p(
        pd.to_numeric(data["risk_area_km2"], errors="coerce")
    )
    data["time_years"] = (
        data["period_start"].dt.to_period("M").astype(int)
        - data["period_start"].dt.to_period("M").astype(int).min()
    ) / 12.0
    data["month_of_year"] = data["period_start"].dt.month.astype(str)
    data["ntl_change_lag1"] = data.groupby("actor_key", sort=False)[OUTCOME].shift(1)
    data["ntl_change_lag2"] = data.groupby("actor_key", sort=False)[OUTCOME].shift(2)
    return data


def build_design(data: pd.DataFrame, signal: str | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    numeric = [*BASE_CONTROLS]
    if signal is not None:
        numeric.append(signal)
    needed = [
        OUTCOME,
        "actor_key",
        "actor_name",
        "period_start",
        "month_of_year",
        *numeric,
    ]
    frame = data[needed].replace([np.inf, -np.inf], np.nan).copy()
    for column in [OUTCOME, *numeric]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=[OUTCOME, *numeric]).copy()

    design = frame[numeric].astype(float)
    design = design - design.mean(axis=0)
    actor_fe = pd.get_dummies(frame["actor_key"], prefix="actor", drop_first=True, dtype=float)
    month_fe = pd.get_dummies(
        frame["month_of_year"], prefix="month", drop_first=True, dtype=float
    )
    design = pd.concat(
        [pd.Series(1.0, index=frame.index, name="Intercept"), design, actor_fe, month_fe],
        axis=1,
    )
    keep = design.std(axis=0).gt(1e-12)
    keep.loc["Intercept"] = True
    design = design.loc[:, keep]
    return frame, design


def panel_newey_west(
    x: np.ndarray,
    residuals: np.ndarray,
    actor: np.ndarray,
    dates: np.ndarray,
    max_lag: int,
) -> np.ndarray:
    """Bartlett-kernel HAC covariance with dependence within actor only."""
    n, k = x.shape
    bread = np.linalg.pinv(x.T @ x)
    meat = np.zeros((k, k), dtype=float)
    for actor_value in pd.unique(actor):
        indices = np.flatnonzero(actor == actor_value)
        indices = indices[np.argsort(dates[indices])]
        scores = x[indices] * residuals[indices, None]
        month_numbers = pd.PeriodIndex(pd.to_datetime(dates[indices]), freq="M").asi8
        position_by_month = {int(month): position for position, month in enumerate(month_numbers)}
        meat += scores.T @ scores
        for lag in range(1, max_lag + 1):
            weight = 1.0 - lag / (max_lag + 1.0)
            pairs = [
                (position, position_by_month[int(month - lag)])
                for position, month in enumerate(month_numbers)
                if int(month - lag) in position_by_month
            ]
            if not pairs:
                continue
            current = np.array([pair[0] for pair in pairs], dtype=int)
            previous = np.array([pair[1] for pair in pairs], dtype=int)
            gamma = scores[current].T @ scores[previous]
            meat += weight * (gamma + gamma.T)
    correction = n / max(n - k, 1)
    return correction * bread @ meat @ bread


def fit_model(data: pd.DataFrame, model_id: str, signal: str | None, max_lag: int) -> tuple[pd.DataFrame, dict[str, object], pd.DataFrame]:
    frame, design = build_design(data, signal)
    columns = design.columns.tolist()
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
        max_lag,
    )
    se = np.sqrt(np.maximum(np.diag(covariance), 0))
    statistic = beta / np.where(se > 0, se, np.nan)
    coefficients = pd.DataFrame(
        {
            "model_id": model_id,
            "term": columns,
            "estimate": beta,
            "hac_se": se,
            "z": statistic,
            "p_value": [normal_p_value(value) for value in statistic],
            "ci_low": beta - 1.96 * se,
            "ci_high": beta + 1.96 * se,
            "is_llm_signal": [column == signal for column in columns],
        }
    )
    sse = float(residuals @ residuals)
    centered = y - y.mean()
    sst = float(centered @ centered)
    residual_frame = frame[["actor_key", "actor_name", "period_start"]].copy()
    residual_frame["model_id"] = model_id
    residual_frame["observed"] = y
    residual_frame["fitted"] = fitted
    residual_frame["residual"] = residuals
    actor_ar1 = []
    for _, group in residual_frame.groupby("actor_key"):
        values = group.sort_values("period_start")["residual"]
        actor_ar1.append(values.autocorr(lag=1))
    diagnostics = {
        "model_id": model_id,
        "signal": signal or "",
        "n": int(len(frame)),
        "actors": int(frame["actor_key"].nunique()),
        "parameters": int(len(columns)),
        "rank": int(np.linalg.matrix_rank(x)),
        "condition_number": float(np.linalg.cond(x)),
        "r2": 1.0 - sse / sst if sst > 0 else float("nan"),
        "adjusted_r2": 1.0 - (sse / max(len(y) - len(columns), 1)) / (sst / max(len(y) - 1, 1))
        if sst > 0
        else float("nan"),
        "rmse": float(np.sqrt(np.mean(residuals**2))),
        "mean_actor_residual_ar1": float(np.nanmean(actor_ar1)),
        "hac_lags": int(max_lag),
        "se_type": "panel_Newey_West_Bartlett_within_actor",
    }
    return coefficients, diagnostics, residual_frame


def benjamini_hochberg(p_values: pd.Series) -> pd.Series:
    values = pd.to_numeric(p_values, errors="coerce").to_numpy(dtype=float)
    result = np.full(len(values), np.nan)
    valid = np.flatnonzero(np.isfinite(values))
    if not len(valid):
        return pd.Series(result, index=p_values.index)
    order = valid[np.argsort(values[valid])]
    ranked = values[order] * len(valid) / np.arange(1, len(valid) + 1)
    adjusted = np.minimum.accumulate(ranked[::-1])[::-1]
    result[order] = np.minimum(adjusted, 1.0)
    return pd.Series(result, index=p_values.index)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", default=str(DEFAULT_PANEL))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--hac-lags", type=int, default=3)
    args = parser.parse_args()
    if args.hac_lags < 0:
        raise SystemExit("--hac-lags must be non-negative")

    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    panel = prepare_panel(resolve(args.panel))
    all_coefficients: list[pd.DataFrame] = []
    all_diagnostics: list[dict[str, object]] = []
    all_residuals: list[pd.DataFrame] = []

    model_specs: list[tuple[str, str | None]] = [("controls_only", None)]
    model_specs.extend((name, column) for name, column in SIGNALS.items())
    for model_id, signal in model_specs:
        coefficients, diagnostics, residuals = fit_model(
            panel, model_id, signal, args.hac_lags
        )
        all_coefficients.append(coefficients)
        all_diagnostics.append(diagnostics)
        all_residuals.append(residuals)

    coefficients = pd.concat(all_coefficients, ignore_index=True)
    diagnostics = pd.DataFrame(all_diagnostics)
    residuals = pd.concat(all_residuals, ignore_index=True)
    main = coefficients.loc[coefficients["is_llm_signal"]].copy()
    main["fdr_p_value"] = benjamini_hochberg(main["p_value"])
    main["effect_for_10pp_share_change"] = 0.1 * main["estimate"]

    coefficients.to_csv(output_dir / "monthly_viirs_all_coefficients.csv", index=False)
    diagnostics.to_csv(output_dir / "monthly_viirs_model_diagnostics.csv", index=False)
    residuals.to_csv(output_dir / "monthly_viirs_model_residuals.csv", index=False)
    main.to_csv(output_dir / "monthly_viirs_llm_results.csv", index=False)
    metadata = {
        "panel": str(resolve(args.panel)),
        "outcome": OUTCOME,
        "signals": SIGNALS,
        "controls": BASE_CONTROLS,
        "fixed_effects": ["actor", "month_of_year"],
        "hac_lags": args.hac_lags,
        "inference": "Panel Newey-West with Bartlett weights within actor",
    }
    (output_dir / "monthly_viirs_model_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(main[["model_id", "term", "estimate", "hac_se", "p_value", "fdr_p_value", "ci_low", "ci_high"]].to_string(index=False))
    print("\nDiagnostics")
    print(diagnostics[["model_id", "n", "r2", "adjusted_r2", "rmse", "mean_actor_residual_ar1"]].to_string(index=False))


if __name__ == "__main__":
    main()
