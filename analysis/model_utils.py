#!/usr/bin/env python3
"""Shared OLS, panel-HAC, calibration, BCA, and BCM utilities."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PANEL = ROOT / "data" / "processed" / "actor_period_panel.csv"
DEFAULT_VALIDATION = ROOT / "data" / "validation" / "manual_validation.csv"
DEFAULT_OUTPUT = ROOT / "results" / "baseline"

OUTCOME = "log1p_y_ucdp_actor_fatalities_next"
BASE_CONTROLS = [
    "log1p_y_ucdp_actor_fatalities",
    "log1p_y_ucdp_actor_fatalities_lag1",
    "log1p_y_ucdp_actor_fatalities_lag2",
    "log1p_n_articles",
    "time_years",
    "taliban_outcome_transition_aug2021",
    "taliban_outcome_post_takeover",
    "hamas_outcome_post_oct2023",
]
ACTOR_KEYS = ["209", "303", "group:farc_bloc"]
CONTEXT_CLASSES = ["cbt", "civ", "pol", "other"]
CONTEXT_THREE_CLASSES = ["violent", "pol", "other"]


@dataclass(frozen=True)
class BinarySpec:
    key: str
    manual_field: str
    llm_field: str
    positive: str

    @property
    def raw_column(self) -> str:
        return f"share_{self.key}_raw"

    @property
    def calibrated_column(self) -> str:
        return f"share_{self.key}_calibrated"


BINARY_SPECS = [
    BinarySpec("direct_conflict_relevance", "conflict_relevance_manual", "conflict_relevance_llm", "direct"),
    BinarySpec("physical_violence", "physical_violence_occurred_manual", "physical_violence_occurred_llm", "yes"),
    BinarySpec("fatalities_present", "fatalities_present_manual", "fatalities_present_llm", "yes"),
]


def resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else ROOT / value


def norm(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().str.lower()


def normal_p_value(statistic: float) -> float:
    if not math.isfinite(statistic):
        return float("nan")
    return math.erfc(abs(statistic) / math.sqrt(2.0))


def fdr_bh(values: pd.Series) -> pd.Series:
    p = pd.to_numeric(values, errors="coerce").to_numpy(float)
    result = np.full(len(p), np.nan)
    valid = np.flatnonzero(np.isfinite(p))
    if not len(valid):
        return pd.Series(result, index=values.index)
    order = valid[np.argsort(p[valid])]
    adjusted = p[order] * len(valid) / np.arange(1, len(valid) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result[order] = np.minimum(adjusted, 1)
    return pd.Series(result, index=values.index)


def estimate_binary_calibration(validation: pd.DataFrame, spec: BinarySpec) -> dict[str, object]:
    manual = norm(validation[spec.manual_field]).eq(spec.positive).astype(int)
    llm = norm(validation[spec.llm_field]).eq(spec.positive).astype(int)
    counts = pd.crosstab(manual, llm).reindex(index=[0, 1], columns=[0, 1], fill_value=0)
    matrix = counts.div(counts.sum(axis=1), axis=0).to_numpy(float)
    specificity = float(matrix[0, 0])
    false_positive_rate = float(matrix[0, 1])
    sensitivity = float(matrix[1, 1])
    denominator = sensitivity - false_positive_rate
    if denominator <= 0:
        raise ValueError(f"Non-informative binary calibration for {spec.key}")
    return {
        "key": spec.key,
        "n": int(len(validation)),
        "counts": counts,
        "matrix": matrix,
        "specificity": specificity,
        "false_positive_rate": false_positive_rate,
        "sensitivity": sensitivity,
        "denominator": denominator,
        "condition_number": float(np.linalg.cond(matrix.T)),
    }


def collapse_context(series: pd.Series) -> pd.Series:
    values = norm(series)
    return values.where(values.isin(CONTEXT_CLASSES[:-1]), "other")


def estimate_context_calibration(validation: pd.DataFrame) -> dict[str, object]:
    manual = collapse_context(validation["event_context_manual"])
    llm = collapse_context(validation["event_context_llm"])
    counts = pd.crosstab(manual, llm).reindex(
        index=CONTEXT_CLASSES, columns=CONTEXT_CLASSES, fill_value=0
    )
    matrix = counts.div(counts.sum(axis=1), axis=0).to_numpy(float)
    transposed = matrix.T
    rank = int(np.linalg.matrix_rank(transposed))
    if rank < len(CONTEXT_CLASSES):
        raise ValueError("Collapsed context confusion matrix is singular")
    return {
        "counts": counts,
        "matrix": matrix,
        "inverse_transposed": np.linalg.inv(transposed),
        "rank": rank,
        "condition_number": float(np.linalg.cond(transposed)),
    }


def collapse_context_three(series: pd.Series) -> pd.Series:
    values = collapse_context(series)
    return values.replace({"cbt": "violent", "civ": "violent"})


def estimate_context_three_calibration(validation: pd.DataFrame) -> dict[str, object]:
    manual = collapse_context_three(validation["event_context_manual"])
    llm = collapse_context_three(validation["event_context_llm"])
    counts = pd.crosstab(manual, llm).reindex(
        index=CONTEXT_THREE_CLASSES,
        columns=CONTEXT_THREE_CLASSES,
        fill_value=0,
    )
    matrix = counts.div(counts.sum(axis=1), axis=0).to_numpy(float)
    transposed = matrix.T
    rank = int(np.linalg.matrix_rank(transposed))
    if rank < len(CONTEXT_THREE_CLASSES):
        raise ValueError("Collapsed three-class context confusion matrix is singular")
    return {
        "counts": counts,
        "matrix": matrix,
        "inverse_transposed": np.linalg.inv(transposed),
        "rank": rank,
        "condition_number": float(np.linalg.cond(transposed)),
    }


def apply_calibration(
    panel: pd.DataFrame,
    binary_rates: dict[str, dict[str, object]],
    context: dict[str, object],
    context_three: dict[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    result = panel.copy()
    diagnostics: list[dict[str, object]] = []
    for spec in BINARY_SPECS:
        rate = binary_rates[spec.key]
        unbounded = (
            pd.to_numeric(result[spec.raw_column], errors="coerce")
            - float(rate["false_positive_rate"])
        ) / float(rate["denominator"])
        result[spec.calibrated_column] = unbounded.clip(0, 1)
        raw_share = pd.to_numeric(result[spec.raw_column], errors="coerce")
        diagnostics.append(
            {
                "variable": spec.key,
                "calibration": "binary_inverse_confusion",
                "rows": int(unbounded.notna().sum()),
                "rows_clipped_low": int(unbounded.lt(0).sum()),
                "rows_clipped_high": int(unbounded.gt(1).sum()),
                "share_rows_clipped": float((unbounded.lt(0) | unbounded.gt(1)).mean()),
                "mean_l1_adjustment": float(
                    np.nanmean(np.abs(result[spec.calibrated_column] - raw_share))
                ),
            }
        )

    raw_columns = [f"share_context_{category}_raw" for category in CONTEXT_CLASSES]
    raw = result[raw_columns].to_numpy(float)
    corrected_unbounded = raw @ np.asarray(context["inverse_transposed"], dtype=float).T
    negative = corrected_unbounded < 0
    corrected = np.maximum(corrected_unbounded, 0)
    totals = corrected.sum(axis=1, keepdims=True)
    corrected = np.divide(
        corrected,
        totals,
        out=np.full_like(corrected, np.nan),
        where=totals > 0,
    )
    for position, category in enumerate(CONTEXT_CLASSES):
        result[f"share_context_{category}_calibrated"] = corrected[:, position]
    diagnostics.append(
        {
            "variable": "event_context_4class",
            "calibration": "multiclass_inverse_confusion_clip_renormalize",
            "rows": int(len(result)),
            "rows_clipped_low": int(negative.any(axis=1).sum()),
            "rows_clipped_high": 0,
            "share_rows_clipped": float(negative.any(axis=1).mean()),
            "mean_l1_adjustment": float(np.nanmean(np.abs(corrected - raw).sum(axis=1))),
        }
    )

    result["share_context_three_violent_raw"] = (
        pd.to_numeric(result["share_context_cbt_raw"], errors="coerce")
        + pd.to_numeric(result["share_context_civ_raw"], errors="coerce")
    )
    result["share_context_three_pol_raw"] = pd.to_numeric(
        result["share_context_pol_raw"], errors="coerce"
    )
    result["share_context_three_other_raw"] = pd.to_numeric(
        result["share_context_other_raw"], errors="coerce"
    )
    raw_three_columns = [
        f"share_context_three_{category}_raw" for category in CONTEXT_THREE_CLASSES
    ]
    raw_three = result[raw_three_columns].to_numpy(float)
    corrected_three_unbounded = (
        raw_three @ np.asarray(context_three["inverse_transposed"], dtype=float).T
    )
    negative_three = corrected_three_unbounded < 0
    corrected_three = np.maximum(corrected_three_unbounded, 0)
    totals_three = corrected_three.sum(axis=1, keepdims=True)
    corrected_three = np.divide(
        corrected_three,
        totals_three,
        out=np.full_like(corrected_three, np.nan),
        where=totals_three > 0,
    )
    for position, category in enumerate(CONTEXT_THREE_CLASSES):
        result[f"share_context_three_{category}_calibrated"] = corrected_three[:, position]
    diagnostics.append(
        {
            "variable": "event_context_3class",
            "calibration": "multiclass_inverse_confusion_clip_renormalize",
            "rows": int(len(result)),
            "rows_clipped_low": int(negative_three.any(axis=1).sum()),
            "rows_clipped_high": 0,
            "share_rows_clipped": float(negative_three.any(axis=1).mean()),
            "mean_l1_adjustment": float(
                np.nanmean(np.abs(corrected_three - raw_three).sum(axis=1))
            ),
        }
    )
    return result, pd.DataFrame(diagnostics)


def base_sample(panel: pd.DataFrame, minimum_articles: int) -> pd.DataFrame:
    needed = [OUTCOME, *BASE_CONTROLS]
    result = panel.loc[pd.to_numeric(panel["n_articles"], errors="coerce").ge(minimum_articles)].copy()
    for column in needed:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    return result.dropna(subset=needed).sort_values(["actor_key", "period_start"]).copy()


def design_matrix(
    frame: pd.DataFrame,
    signals: list[str],
    month_effects: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    columns: list[tuple[str, np.ndarray]] = [(name, frame[name].to_numpy(float)) for name in signals]
    columns.extend((name, frame[name].to_numpy(float)) for name in BASE_CONTROLS)
    columns.append(("Intercept", np.ones(len(frame))))
    columns.append(("actor_Taliban", frame["actor_key"].eq("303").to_numpy(float)))
    columns.append(("actor_FARC_family", frame["actor_key"].eq("group:farc_bloc").to_numpy(float)))
    if month_effects:
        for month in range(2, 13):
            columns.append(
                (f"month_of_year_{month}", frame["month_of_year"].eq(month).to_numpy(float))
            )
    names = [name for name, _ in columns]
    x = np.column_stack([values for _, values in columns])
    keep = np.nanstd(x, axis=0) > 1e-12
    keep[names.index("Intercept")] = True
    names = [name for name, use in zip(names, keep) if use]
    x = x[:, keep]
    y = frame[OUTCOME].to_numpy(float)
    return y, x, names


def panel_hac(
    x: np.ndarray,
    residuals: np.ndarray,
    actors: np.ndarray,
    periods: np.ndarray,
    max_lag: int,
) -> np.ndarray:
    n, k = x.shape
    bread = np.linalg.pinv(x.T @ x)
    meat = np.zeros((k, k))
    for actor in pd.unique(actors):
        indices = np.flatnonzero(actors == actor)
        indices = indices[np.argsort(periods[indices])]
        scores = x[indices] * residuals[indices, None]
        actor_periods = periods[indices].astype(int)
        position = {period: row for row, period in enumerate(actor_periods)}
        meat += scores.T @ scores
        for lag in range(1, max_lag + 1):
            weight = 1 - lag / (max_lag + 1)
            pairs = [
                (row, position[period - lag])
                for row, period in enumerate(actor_periods)
                if period - lag in position
            ]
            if pairs:
                current = np.array([pair[0] for pair in pairs])
                previous = np.array([pair[1] for pair in pairs])
                gamma = scores[current].T @ scores[previous]
                meat += weight * (gamma + gamma.T)
    return (n / max(n - k, 1)) * bread @ meat @ bread


def fit_model(
    sample: pd.DataFrame,
    model_id: str,
    signals: list[str],
    hac_lags: int,
    month_effects: bool = False,
) -> dict[str, object]:
    frame = sample.dropna(subset=signals).copy()
    y, x, names = design_matrix(frame, signals, month_effects=month_effects)
    beta = np.linalg.pinv(x.T @ x) @ x.T @ y
    fitted = x @ beta
    residuals = y - fitted
    covariance = panel_hac(
        x,
        residuals,
        frame["actor_key"].to_numpy(),
        frame["period_index"].to_numpy(int),
        hac_lags,
    )
    se = np.sqrt(np.maximum(np.diag(covariance), 0))
    statistic = beta / np.where(se > 0, se, np.nan)
    sse = float(residuals @ residuals)
    sst = float(((y - y.mean()) ** 2).sum())
    actor_acf = []
    for actor in pd.unique(frame["actor_key"]):
        actor_resid = pd.Series(
            residuals[frame["actor_key"].to_numpy() == actor],
            index=frame.loc[frame["actor_key"].eq(actor), "period_index"],
        ).sort_index()
        pairs = pd.concat([actor_resid, actor_resid.shift(1)], axis=1).dropna()
        actor_acf.append(pairs.iloc[:, 0].corr(pairs.iloc[:, 1]) if len(pairs) > 2 else np.nan)
    coefficients = pd.DataFrame(
        {
            "model_id": model_id,
            "term": names,
            "estimate": beta,
            "hac_se": se,
            "z": statistic,
            "p_value": [normal_p_value(value) for value in statistic],
            "ci_low": beta - 1.96 * se,
            "ci_high": beta + 1.96 * se,
            "is_signal": [name in signals for name in names],
        }
    )
    diagnostics = {
        "model_id": model_id,
        "n": int(len(frame)),
        "actors": int(frame["actor_key"].nunique()),
        "parameters": int(len(names)),
        "rank": int(np.linalg.matrix_rank(x)),
        "condition_number": float(np.linalg.cond(x)),
        "r2": 1 - sse / sst if sst > 0 else np.nan,
        "adjusted_r2": 1 - (sse / max(len(y) - len(names), 1)) / (sst / max(len(y) - 1, 1)),
        "rmse": float(np.sqrt(np.mean(residuals**2))),
        "mean_actor_residual_ar1": float(np.nanmean(actor_acf)),
        "hac_lags": int(hac_lags),
        "month_effects": bool(month_effects),
    }
    signal_positions = [names.index(signal) for signal in signals]
    if signal_positions:
        restriction_beta = beta[signal_positions]
        restriction_cov = covariance[np.ix_(signal_positions, signal_positions)]
        wald = float(restriction_beta @ np.linalg.pinv(restriction_cov) @ restriction_beta)
        wald_df = len(signal_positions)
        diagnostics["signal_wald_statistic"] = wald
        diagnostics["signal_wald_df"] = wald_df
        diagnostics["signal_wald_p"] = float(chi2.sf(wald, wald_df))
    else:
        diagnostics["signal_wald_statistic"] = np.nan
        diagnostics["signal_wald_df"] = 0
        diagnostics["signal_wald_p"] = np.nan
    return {
        "frame": frame,
        "y": y,
        "x": x,
        "names": names,
        "beta": beta,
        "covariance": covariance,
        "coefficients": coefficients,
        "diagnostics": diagnostics,
        "fitted": fitted,
        "residuals": residuals,
        "signals": signals,
    }


def signal_vifs(sample: pd.DataFrame, signals: list[str]) -> pd.DataFrame:
    frame = sample.dropna(subset=signals).copy()
    _, full_x, names = design_matrix(frame, signals)
    rows = []
    for signal in signals:
        position = names.index(signal)
        y = full_x[:, position]
        other = np.delete(full_x, position, axis=1)
        fitted = other @ (np.linalg.pinv(other.T @ other) @ other.T @ y)
        sst = float(((y - y.mean()) ** 2).sum())
        r2 = 1 - float(((y - fitted) ** 2).sum()) / sst if sst > 0 else np.nan
        rows.append(
            {
                "signal": signal,
                "auxiliary_r2": r2,
                "vif": 1 / (1 - r2) if math.isfinite(r2) and r2 < 1 else np.inf,
            }
        )
    return pd.DataFrame(rows)


def corrected_estimator(
    fit: dict[str, object],
    omega: np.ndarray,
    signal_names: list[str],
    cell_sizes: np.ndarray,
) -> dict[str, object]:
    n = len(fit["y"])
    names = list(fit["names"])
    q_hat = np.asarray(fit["x"]).T @ np.asarray(fit["x"]) / n
    block = np.zeros_like(q_hat)
    positions = [names.index(name) for name in signal_names]
    block[np.ix_(positions, positions)] = omega
    kappa = float(np.sum(1 / cell_sizes) / math.sqrt(n))
    adjustment = (kappa / math.sqrt(n)) * np.linalg.pinv(q_hat) @ block
    identity = np.eye(len(names))
    additive_beta = (identity + adjustment) @ np.asarray(fit["beta"])
    base = identity - adjustment
    maximum_eigenvalue = float(np.max(np.abs(np.linalg.eigvals(adjustment))))
    bcm_valid = maximum_eigenvalue < 1 and abs(np.linalg.det(base)) > 1e-12
    multiplicative_beta = np.linalg.inv(base) @ np.asarray(fit["beta"]) if bcm_valid else None
    return {
        "kappa": kappa,
        "omega": omega,
        "adjustment": adjustment,
        "maximum_eigenvalue": maximum_eigenvalue,
        "additive_beta": additive_beta,
        "multiplicative_beta": multiplicative_beta,
        "bcm_valid": bcm_valid,
    }


def binary_index_omega(
    calibrated_share: np.ndarray,
    sensitivity: float,
    false_positive_rate: float,
) -> np.ndarray:
    b = np.array(
        [
            [sensitivity, 1 - sensitivity],
            [false_positive_rate, 1 - false_positive_rate],
        ]
    )
    selector = np.array([[1.0, 0.0]])
    w = np.column_stack([calibrated_share, 1 - calibrated_share])
    left = np.linalg.pinv(b @ b.T) @ b
    omega = (
        selector
        @ left
        @ np.diag(b.T @ w.mean(axis=0))
        @ b.T
        @ left.T
        @ selector.T
        - np.mean(calibrated_share**2)
    )
    return np.array([[max(float(omega.item()), 0)]])


def multiclass_index_omega(calibrated: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    b = np.asarray(matrix, dtype=float)
    left = np.linalg.pinv(b @ b.T) @ b
    full = (
        left
        @ np.diag(b.T @ calibrated.mean(axis=0))
        @ b.T
        @ left.T
        - np.mean(np.einsum("ni,nj->nij", calibrated, calibrated), axis=0)
    )
    included = calibrated.shape[1] - 1
    selected = (full[:included, :included] + full[:included, :included].T) / 2
    eigenvalues, eigenvectors = np.linalg.eigh(selected)
    return eigenvectors @ np.diag(np.maximum(eigenvalues, 0)) @ eigenvectors.T


def correction_rows(
    fit: dict[str, object],
    correction: dict[str, object],
    estimator: str,
) -> pd.DataFrame:
    beta = (
        np.asarray(correction["additive_beta"])
        if estimator == "bca"
        else np.asarray(correction["multiplicative_beta"])
    )
    covariance = np.asarray(fit["covariance"])
    se = np.sqrt(np.maximum(np.diag(covariance), 0))
    rows = []
    for signal in fit["signals"]:
        position = list(fit["names"]).index(signal)
        estimate = float(beta[position])
        standard_error = float(se[position])
        z = estimate / standard_error if standard_error > 0 else np.nan
        rows.append(
            {
                "model_id": str(fit["diagnostics"]["model_id"]),
                "estimator": estimator,
                "term": signal,
                "estimate": estimate,
                "hac_se": standard_error,
                "z": z,
                "p_value": normal_p_value(z),
                "ci_low": estimate - 1.96 * standard_error,
                "ci_high": estimate + 1.96 * standard_error,
                "kappa_hat": correction["kappa"],
                "maximum_adjustment_eigenvalue": correction["maximum_eigenvalue"],
                "correction_stable": bool(correction["maximum_eigenvalue"] < 1),
                "bcm_valid": bool(correction["bcm_valid"]),
            }
        )
    return pd.DataFrame(rows)


def joint_state_diagnostic(validation: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    manual_bits = []
    llm_bits = []
    for spec in BINARY_SPECS:
        manual_bits.append(norm(validation[spec.manual_field]).eq(spec.positive).astype(int).astype(str))
        llm_bits.append(norm(validation[spec.llm_field]).eq(spec.positive).astype(int).astype(str))
    manual = manual_bits[0] + manual_bits[1] + manual_bits[2]
    llm = llm_bits[0] + llm_bits[1] + llm_bits[2]
    states = [f"{index:03b}" for index in range(8)]
    counts = pd.crosstab(manual, llm).reindex(index=states, columns=states, fill_value=0)
    matrix = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0).to_numpy()
    rank = int(np.linalg.matrix_rank(matrix.T))
    return counts, {
        "states": states,
        "rank": rank,
        "required_rank": 8,
        "condition_number": float(np.linalg.cond(matrix.T)),
        "invertible": bool(rank == 8),
        "manual_state_counts": manual.value_counts().sort_index().to_dict(),
        "llm_state_counts": llm.value_counts().sort_index().to_dict(),
    }


def fixed_design(frame: pd.DataFrame, signals: list[str]) -> np.ndarray:
    columns = [frame[name].to_numpy(float) for name in signals]
    columns.extend(frame[name].to_numpy(float) for name in BASE_CONTROLS)
    columns.extend(
        [
            np.ones(len(frame)),
            frame["actor_key"].eq("303").to_numpy(float),
            frame["actor_key"].eq("group:farc_bloc").to_numpy(float),
        ]
    )
    return np.column_stack(columns)


def rolling_predictions(
    sample: pd.DataFrame,
    specifications: dict[str, list[str]],
    start_test: str = "2022-01",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    months = sorted(month for month in sample["month"].unique() if month >= start_test)
    for model_id, signals in specifications.items():
        model_data = sample.dropna(subset=signals).copy()
        for month in months:
            train = model_data.loc[model_data["month"].lt(month)]
            test = model_data.loc[model_data["month"].eq(month)]
            if len(train) < 60 or test.empty:
                continue
            x_train = fixed_design(train, signals)
            x_test = fixed_design(test, signals)
            beta = np.linalg.pinv(x_train.T @ x_train) @ x_train.T @ train[OUTCOME].to_numpy(float)
            predictions = x_test @ beta
            for row, prediction in zip(test.itertuples(index=False), predictions):
                rows.append(
                    {
                        "model_id": model_id,
                        "test_month": month,
                        "actor_key": row.actor_key,
                        "actor_name": row.actor_name,
                        "observed": getattr(row, OUTCOME),
                        "predicted": float(prediction),
                    }
                )
    predictions = pd.DataFrame(rows)
    predictions["error"] = predictions["observed"] - predictions["predicted"]
    summary = (
        predictions.groupby("model_id", as_index=False)
        .agg(
            n_predictions=("error", "size"),
            rmse=("error", lambda x: float(np.sqrt(np.mean(np.square(x))))),
            mae=("error", lambda x: float(np.mean(np.abs(x)))),
        )
    )
    controls_rmse = float(summary.loc[summary["model_id"].eq("controls_only"), "rmse"].iloc[0])
    summary["rmse_change_vs_controls"] = summary["rmse"] - controls_rmse
    summary["rmse_change_percent"] = 100 * (summary["rmse"] / controls_rmse - 1)
    return predictions, summary.sort_values("rmse").reset_index(drop=True)


def rolling_prediction_uncertainty(
    predictions: pd.DataFrame,
    repetitions: int,
    seed: int,
) -> pd.DataFrame:
    controls = predictions.loc[predictions["model_id"].eq("controls_only"), [
        "test_month", "actor_key", "error"
    ]].rename(columns={"error": "control_error"})
    rng = np.random.default_rng(seed)
    rows = []
    for model_id, group in predictions.loc[~predictions["model_id"].eq("controls_only")].groupby("model_id"):
        merged = group.merge(controls, on=["test_month", "actor_key"], how="inner")
        monthly_sums = (
            merged.assign(
                model_squared_error=np.square(merged["error"]),
                control_squared_error=np.square(merged["control_error"]),
            )
            .groupby("test_month", as_index=False)
            .agg(
                n=("error", "size"),
                model_sse=("model_squared_error", "sum"),
                control_sse=("control_squared_error", "sum"),
            )
        )
        months = monthly_sums["test_month"].to_numpy()
        observed_delta = float(
            np.sqrt(np.mean(merged["error"] ** 2))
            - np.sqrt(np.mean(merged["control_error"] ** 2))
        )
        sampled_indices = rng.integers(
            0,
            len(months),
            size=(repetitions, len(months)),
        )
        n_values = monthly_sums["n"].to_numpy(float)
        model_sse_values = monthly_sums["model_sse"].to_numpy(float)
        control_sse_values = monthly_sums["control_sse"].to_numpy(float)
        sampled_n = n_values[sampled_indices].sum(axis=1)
        draws_array = (
            np.sqrt(model_sse_values[sampled_indices].sum(axis=1) / sampled_n)
            - np.sqrt(control_sse_values[sampled_indices].sum(axis=1) / sampled_n)
        )
        rows.append(
            {
                "model_id": model_id,
                "n_predictions": int(len(merged)),
                "n_test_months": int(len(months)),
                "observed_rmse_difference": observed_delta,
                "bootstrap_ci_low": float(np.quantile(draws_array, 0.025)),
                "bootstrap_ci_high": float(np.quantile(draws_array, 0.975)),
                "probability_rmse_improvement": float(np.mean(draws_array < 0)),
            }
        )
    return pd.DataFrame(rows).sort_values("observed_rmse_difference")


def bootstrap_validation_uncertainty(
    panel: pd.DataFrame,
    validation: pd.DataFrame,
    minimum_articles: int,
    hac_lags: int,
    repetitions: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    draw_rows: list[dict[str, object]] = []
    for draw in range(repetitions):
        sampled = validation.iloc[rng.integers(0, len(validation), len(validation))].reset_index(drop=True)
        try:
            binary_rates = {
                spec.key: estimate_binary_calibration(sampled, spec) for spec in BINARY_SPECS
            }
            context = estimate_context_calibration(sampled)
            context_three = estimate_context_three_calibration(sampled)
            calibrated, _ = apply_calibration(
                panel,
                binary_rates,
                context,
                context_three,
            )
            sample = base_sample(calibrated, minimum_articles)
            model_specs: list[tuple[str, list[str]]] = [
                ("joint_binary_calibrated", [spec.calibrated_column for spec in BINARY_SPECS]),
                ("context_calibrated", [f"share_context_{category}_calibrated" for category in CONTEXT_CLASSES[:-1]]),
                (
                    "context_three_calibrated",
                    [
                        f"share_context_three_{category}_calibrated"
                        for category in CONTEXT_THREE_CLASSES[:-1]
                    ],
                ),
            ]
            model_specs.extend(
                (f"single_{spec.key}_calibrated", [spec.calibrated_column])
                for spec in BINARY_SPECS
            )
            for model_id, signals in model_specs:
                fit = fit_model(sample, model_id, signals, hac_lags)
                for signal in signals:
                    row = fit["coefficients"].loc[fit["coefficients"]["term"].eq(signal)].iloc[0]
                    draw_rows.append(
                        {
                            "draw": draw,
                            "model_id": model_id,
                            "estimator": "calibrated",
                            "term": signal,
                            "estimate": row["estimate"],
                            "point_hac_se": row["hac_se"],
                            "maximum_adjustment_eigenvalue": np.nan,
                        }
                    )
                if model_id.startswith("single_"):
                    spec = next(item for item in BINARY_SPECS if item.calibrated_column == signals[0])
                    rate = binary_rates[spec.key]
                    omega = binary_index_omega(
                        fit["frame"][spec.calibrated_column].to_numpy(float),
                        float(rate["sensitivity"]),
                        float(rate["false_positive_rate"]),
                    )
                    correction = corrected_estimator(
                        fit,
                        omega,
                        signals,
                        fit["frame"]["n_articles"].to_numpy(float),
                    )
                    for estimator, beta in [
                        ("bca", correction["additive_beta"]),
                        ("bcm", correction["multiplicative_beta"]),
                    ]:
                        if beta is None:
                            continue
                        position = list(fit["names"]).index(signals[0])
                        draw_rows.append(
                            {
                                "draw": draw,
                                "model_id": model_id,
                                "estimator": estimator,
                                "term": signals[0],
                                "estimate": float(np.asarray(beta)[position]),
                                "point_hac_se": float(
                                    np.sqrt(max(np.asarray(fit["covariance"])[position, position], 0))
                                ),
                                "maximum_adjustment_eigenvalue": correction["maximum_eigenvalue"],
                            }
                        )

                if model_id == "context_calibrated":
                    full = fit["frame"][
                        [f"share_context_{category}_calibrated" for category in CONTEXT_CLASSES]
                    ].to_numpy(float)
                    omega = multiclass_index_omega(full, np.asarray(context["matrix"]))
                    correction = corrected_estimator(
                        fit,
                        omega,
                        signals,
                        fit["frame"]["n_articles"].to_numpy(float),
                    )
                    for estimator, beta in [
                        ("bca", correction["additive_beta"]),
                        ("bcm", correction["multiplicative_beta"]),
                    ]:
                        if beta is None:
                            continue
                        for signal in signals:
                            position = list(fit["names"]).index(signal)
                            draw_rows.append(
                                {
                                    "draw": draw,
                                    "model_id": model_id,
                                    "estimator": estimator,
                                    "term": signal,
                                    "estimate": float(np.asarray(beta)[position]),
                                    "point_hac_se": float(
                                        np.sqrt(max(np.asarray(fit["covariance"])[position, position], 0))
                                    ),
                                    "maximum_adjustment_eigenvalue": correction["maximum_eigenvalue"],
                                }
                            )

                if model_id == "context_three_calibrated":
                    full = fit["frame"][
                        [
                            f"share_context_three_{category}_calibrated"
                            for category in CONTEXT_THREE_CLASSES
                        ]
                    ].to_numpy(float)
                    omega = multiclass_index_omega(
                        full,
                        np.asarray(context_three["matrix"]),
                    )
                    correction = corrected_estimator(
                        fit,
                        omega,
                        signals,
                        fit["frame"]["n_articles"].to_numpy(float),
                    )
                    for estimator, beta in [
                        ("bca", correction["additive_beta"]),
                        ("bcm", correction["multiplicative_beta"]),
                    ]:
                        if beta is None:
                            continue
                        for signal in signals:
                            position = list(fit["names"]).index(signal)
                            draw_rows.append(
                                {
                                    "draw": draw,
                                    "model_id": model_id,
                                    "estimator": estimator,
                                    "term": signal,
                                    "estimate": float(np.asarray(beta)[position]),
                                    "point_hac_se": float(
                                        np.sqrt(
                                            max(
                                                np.asarray(fit["covariance"])[position, position],
                                                0,
                                            )
                                        )
                                    ),
                                    "maximum_adjustment_eigenvalue": correction[
                                        "maximum_eigenvalue"
                                    ],
                                }
                            )
        except (ValueError, np.linalg.LinAlgError):
            continue
    draws = pd.DataFrame(draw_rows)
    summary_rows = []
    for (model_id, estimator, term), group in draws.groupby(["model_id", "estimator", "term"]):
        validation_se = float(group["estimate"].std(ddof=1))
        point_se = float(group["point_hac_se"].median())
        point = float(group["estimate"].median())
        combined_se = math.sqrt(point_se**2 + validation_se**2)
        z = point / combined_se if combined_se > 0 else np.nan
        adjustment_values = pd.to_numeric(
            group["maximum_adjustment_eigenvalue"], errors="coerce"
        ).dropna()
        summary_rows.append(
            {
                "model_id": model_id,
                "estimator": estimator,
                "term": term,
                "successful_draws": int(group["draw"].nunique()),
                "median_estimate": point,
                "validation_se": validation_se,
                "point_hac_se_median": point_se,
                "combined_se": combined_se,
                "combined_p_value": normal_p_value(z),
                "validation_percentile_low": float(group["estimate"].quantile(0.025)),
                "validation_percentile_high": float(group["estimate"].quantile(0.975)),
                "median_maximum_adjustment_eigenvalue": (
                    float(adjustment_values.median()) if not adjustment_values.empty else np.nan
                ),
                "share_draws_with_unstable_adjustment": (
                    float(adjustment_values.ge(1).mean()) if not adjustment_values.empty else np.nan
                ),
            }
        )
    return draws, pd.DataFrame(summary_rows)


def run_threshold_robustness(panel: pd.DataFrame, hac_lags: int) -> pd.DataFrame:
    rows = []
    specs = {
        "joint_binary_raw": [spec.raw_column for spec in BINARY_SPECS],
        "joint_binary_calibrated": [spec.calibrated_column for spec in BINARY_SPECS],
        "context_raw": [f"share_context_{category}_raw" for category in CONTEXT_CLASSES[:-1]],
        "context_calibrated": [f"share_context_{category}_calibrated" for category in CONTEXT_CLASSES[:-1]],
        "context_three_raw": [
            f"share_context_three_{category}_raw"
            for category in CONTEXT_THREE_CLASSES[:-1]
        ],
        "context_three_calibrated": [
            f"share_context_three_{category}_calibrated"
            for category in CONTEXT_THREE_CLASSES[:-1]
        ],
    }
    for threshold in (1, 10, 20, 40):
        sample = base_sample(panel, threshold)
        for model_id, signals in specs.items():
            fit = fit_model(sample, model_id, signals, hac_lags)
            for _, coefficient in fit["coefficients"].loc[fit["coefficients"]["is_signal"]].iterrows():
                rows.append(
                    {
                        "minimum_articles": threshold,
                        "model_id": model_id,
                        "term": coefficient["term"],
                        "estimate": coefficient["estimate"],
                        "hac_se": coefficient["hac_se"],
                        "p_value": coefficient["p_value"],
                        "n": fit["diagnostics"]["n"],
                        "r2": fit["diagnostics"]["r2"],
                        "joint_wald_p": fit["diagnostics"]["signal_wald_p"],
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", default=str(DEFAULT_PANEL))
    parser.add_argument("--validation", default=str(DEFAULT_VALIDATION))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--minimum-articles", type=int, default=20)
    parser.add_argument("--hac-lags", type=int, default=6)
    parser.add_argument("--bootstrap-repetitions", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260902)
    args = parser.parse_args()
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    panel = pd.read_csv(resolve(args.panel), dtype={"actor_key": str})
    validation = pd.read_csv(resolve(args.validation), dtype={"GlobalEventID": str}, low_memory=False)
    if "in_three_actor_calibration" in validation:
        flag = validation["in_three_actor_calibration"].astype(str).str.lower()
        validation = validation.loc[flag.isin({"true", "1"})].copy()

    binary_rates = {spec.key: estimate_binary_calibration(validation, spec) for spec in BINARY_SPECS}
    context = estimate_context_calibration(validation)
    context_three = estimate_context_three_calibration(validation)
    calibrated_panel, calibration_diagnostics = apply_calibration(
        panel,
        binary_rates,
        context,
        context_three,
    )
    calibrated_panel.to_csv(output_dir / "main_actor_month_panel_calibrated.csv", index=False)
    calibration_diagnostics.to_csv(output_dir / "calibration_clipping_diagnostics.csv", index=False)

    binary_rate_rows = []
    binary_count_rows = []
    for spec in BINARY_SPECS:
        rate = binary_rates[spec.key]
        binary_rate_rows.append(
            {
                "variable": spec.key,
                "validation_n": rate["n"],
                "sensitivity": rate["sensitivity"],
                "specificity": rate["specificity"],
                "false_positive_rate": rate["false_positive_rate"],
                "denominator": rate["denominator"],
                "condition_number": rate["condition_number"],
            }
        )
        counts = rate["counts"]
        for true_value in (0, 1):
            for predicted_value in (0, 1):
                binary_count_rows.append(
                    {
                        "variable": spec.key,
                        "manual_binary": true_value,
                        "llm_binary": predicted_value,
                        "count": int(counts.loc[true_value, predicted_value]),
                    }
                )
    pd.DataFrame(binary_rate_rows).to_csv(output_dir / "binary_validation_rates.csv", index=False)
    pd.DataFrame(binary_count_rows).to_csv(output_dir / "binary_validation_counts.csv", index=False)
    context["counts"].to_csv(output_dir / "context_validation_counts.csv")
    pd.DataFrame(context["matrix"], index=CONTEXT_CLASSES, columns=CONTEXT_CLASSES).to_csv(
        output_dir / "context_validation_matrix.csv"
    )
    context_three["counts"].to_csv(output_dir / "context_three_validation_counts.csv")
    pd.DataFrame(
        context_three["matrix"],
        index=CONTEXT_THREE_CLASSES,
        columns=CONTEXT_THREE_CLASSES,
    ).to_csv(output_dir / "context_three_validation_matrix.csv")

    joint_counts, joint_diagnostic = joint_state_diagnostic(validation)
    joint_counts.to_csv(output_dir / "joint_binary_state_validation_counts.csv")
    (output_dir / "joint_binary_state_diagnostic.json").write_text(
        json.dumps(joint_diagnostic, indent=2) + "\n", encoding="utf-8"
    )

    sample = base_sample(calibrated_panel, args.minimum_articles)
    raw_binary = [spec.raw_column for spec in BINARY_SPECS]
    calibrated_binary = [spec.calibrated_column for spec in BINARY_SPECS]
    raw_context = [f"share_context_{category}_raw" for category in CONTEXT_CLASSES[:-1]]
    calibrated_context = [f"share_context_{category}_calibrated" for category in CONTEXT_CLASSES[:-1]]
    raw_context_three = [
        f"share_context_three_{category}_raw"
        for category in CONTEXT_THREE_CLASSES[:-1]
    ]
    calibrated_context_three = [
        f"share_context_three_{category}_calibrated"
        for category in CONTEXT_THREE_CLASSES[:-1]
    ]
    specifications: list[tuple[str, list[str]]] = [("controls_only", [])]
    for spec in BINARY_SPECS:
        specifications.append((f"single_{spec.key}_raw", [spec.raw_column]))
        specifications.append((f"single_{spec.key}_calibrated", [spec.calibrated_column]))
    specifications.extend(
        [
            ("joint_binary_raw", raw_binary),
            ("joint_binary_calibrated", calibrated_binary),
            ("context_raw", raw_context),
            ("context_calibrated", calibrated_context),
            ("context_three_raw", raw_context_three),
            ("context_three_calibrated", calibrated_context_three),
        ]
    )

    vif_frames = []
    for model_id, signals in [
        ("joint_binary_raw", raw_binary),
        ("joint_binary_calibrated", calibrated_binary),
        ("context_raw", raw_context),
        ("context_calibrated", calibrated_context),
        ("context_three_raw", raw_context_three),
        ("context_three_calibrated", calibrated_context_three),
    ]:
        values = signal_vifs(sample, signals)
        values.insert(0, "model_id", model_id)
        vif_frames.append(values)
    vifs = pd.concat(vif_frames, ignore_index=True)
    combined_raw = [*raw_binary, *raw_context]
    combined_calibrated = [*calibrated_binary, *calibrated_context]
    combined_vif_raw = signal_vifs(sample, combined_raw)
    combined_vif_calibrated = signal_vifs(sample, combined_calibrated)
    combined_vif_raw.insert(0, "model_id", "combined_raw")
    combined_vif_calibrated.insert(0, "model_id", "combined_calibrated")
    vifs = pd.concat([vifs, combined_vif_raw, combined_vif_calibrated], ignore_index=True)
    vifs.to_csv(output_dir / "signal_vif_diagnostics.csv", index=False)
    combined_diagnostics_acceptable = bool(
        np.isfinite(vifs.loc[vifs["model_id"].str.startswith("combined"), "vif"]).all()
        and vifs.loc[vifs["model_id"].str.startswith("combined"), "vif"].max() < 10
    )
    if combined_diagnostics_acceptable:
        specifications.extend(
            [
                ("combined_raw", combined_raw),
                ("combined_calibrated", combined_calibrated),
            ]
        )

    fits: dict[str, dict[str, object]] = {}
    coefficient_frames = []
    diagnostics_rows = []
    for model_id, signals in specifications:
        fit = fit_model(sample, model_id, signals, args.hac_lags)
        fits[model_id] = fit
        coefficients = fit["coefficients"].copy()
        coefficients["estimator"] = "raw" if model_id.endswith("_raw") else "calibrated" if model_id.endswith("_calibrated") else "naive"
        coefficient_frames.append(coefficients)
        diagnostics_rows.append(fit["diagnostics"])

    corrected_rows = []
    for spec in BINARY_SPECS:
        fit = fits[f"single_{spec.key}_calibrated"]
        model_data = fit["frame"]
        rate = binary_rates[spec.key]
        omega = binary_index_omega(
            model_data[spec.calibrated_column].to_numpy(float),
            float(rate["sensitivity"]),
            float(rate["false_positive_rate"]),
        )
        correction = corrected_estimator(
            fit,
            omega,
            [spec.calibrated_column],
            model_data["n_articles"].to_numpy(float),
        )
        corrected_rows.append(correction_rows(fit, correction, "bca"))
        if correction["bcm_valid"]:
            corrected_rows.append(correction_rows(fit, correction, "bcm"))

    context_fit = fits["context_calibrated"]
    context_data = context_fit["frame"]
    context_full = context_data[
        [f"share_context_{category}_calibrated" for category in CONTEXT_CLASSES]
    ].to_numpy(float)
    omega_context = multiclass_index_omega(context_full, np.asarray(context["matrix"]))
    context_correction = corrected_estimator(
        context_fit,
        omega_context,
        calibrated_context,
        context_data["n_articles"].to_numpy(float),
    )
    corrected_rows.append(correction_rows(context_fit, context_correction, "bca"))
    if context_correction["bcm_valid"]:
        corrected_rows.append(correction_rows(context_fit, context_correction, "bcm"))

    context_three_fit = fits["context_three_calibrated"]
    context_three_data = context_three_fit["frame"]
    context_three_full = context_three_data[
        [
            f"share_context_three_{category}_calibrated"
            for category in CONTEXT_THREE_CLASSES
        ]
    ].to_numpy(float)
    omega_context_three = multiclass_index_omega(
        context_three_full,
        np.asarray(context_three["matrix"]),
    )
    context_three_correction = corrected_estimator(
        context_three_fit,
        omega_context_three,
        calibrated_context_three,
        context_three_data["n_articles"].to_numpy(float),
    )
    corrected_rows.append(
        correction_rows(context_three_fit, context_three_correction, "bca")
    )
    if context_three_correction["bcm_valid"]:
        corrected_rows.append(
            correction_rows(context_three_fit, context_three_correction, "bcm")
        )

    coefficients = pd.concat(coefficient_frames, ignore_index=True)
    corrected = pd.concat(corrected_rows, ignore_index=True)
    diagnostics = pd.DataFrame(diagnostics_rows)
    signal_coefficients = coefficients.loc[coefficients["is_signal"]].copy()
    signal_coefficients["fdr_p_value_within_estimator"] = signal_coefficients.groupby("estimator")["p_value"].transform(fdr_bh)
    coefficients.to_csv(output_dir / "all_model_coefficients.csv", index=False)
    signal_coefficients.to_csv(output_dir / "signal_coefficients.csv", index=False)
    corrected.to_csv(output_dir / "battaglia_corrected_coefficients.csv", index=False)
    diagnostics.to_csv(output_dir / "model_diagnostics.csv", index=False)

    correlation_columns = [*combined_raw, *combined_calibrated]
    sample[correlation_columns].corr().to_csv(output_dir / "signal_correlations.csv")

    prediction_specs = {model_id: signals for model_id, signals in specifications}
    predictions, prediction_summary = rolling_predictions(sample, prediction_specs)
    predictions.to_csv(output_dir / "rolling_predictions.csv", index=False)
    prediction_summary.to_csv(output_dir / "rolling_prediction_summary.csv", index=False)
    prediction_uncertainty = rolling_prediction_uncertainty(
        predictions, repetitions=2000, seed=args.seed + 1
    )
    prediction_uncertainty.to_csv(
        output_dir / "rolling_prediction_uncertainty.csv", index=False
    )

    bootstrap_draws, bootstrap_summary = bootstrap_validation_uncertainty(
        panel,
        validation,
        args.minimum_articles,
        args.hac_lags,
        args.bootstrap_repetitions,
        args.seed,
    )
    bootstrap_draws.to_csv(output_dir / "validation_bootstrap_draws.csv", index=False)
    bootstrap_summary.to_csv(output_dir / "validation_bootstrap_summary.csv", index=False)
    threshold_robustness = run_threshold_robustness(calibrated_panel, args.hac_lags)
    threshold_robustness.to_csv(output_dir / "minimum_article_threshold_robustness.csv", index=False)

    fixed_effect_rows = []
    for month_effects in (False, True):
        for model_id, signals in [
            ("joint_binary_calibrated", calibrated_binary),
            ("context_calibrated", calibrated_context),
            ("context_three_calibrated", calibrated_context_three),
        ]:
            fit = fit_model(
                sample,
                model_id + ("_month_fe" if month_effects else "_baseline"),
                signals,
                args.hac_lags,
                month_effects=month_effects,
            )
            for _, row in fit["coefficients"].loc[fit["coefficients"]["is_signal"]].iterrows():
                fixed_effect_rows.append(
                    {
                        "month_of_year_effects": month_effects,
                        "model_id": model_id,
                        "term": row["term"],
                        "estimate": row["estimate"],
                        "hac_se": row["hac_se"],
                        "p_value": row["p_value"],
                        "joint_wald_p": fit["diagnostics"]["signal_wald_p"],
                        "n": fit["diagnostics"]["n"],
                        "r2": fit["diagnostics"]["r2"],
                    }
                )
    pd.DataFrame(fixed_effect_rows).to_csv(
        output_dir / "month_effect_robustness.csv", index=False
    )

    metadata = {
        "panel": str(resolve(args.panel)),
        "validation": str(resolve(args.validation)),
        "outcome": OUTCOME,
        "minimum_articles": args.minimum_articles,
        "base_controls": BASE_CONTROLS,
        "fixed_effects": ["actor"],
        "hac_lags": args.hac_lags,
        "binary_rule": "positive class versus all other retained outputs",
        "context_classes": CONTEXT_CLASSES,
        "context_reference": "other",
        "context_three_classes": CONTEXT_THREE_CLASSES,
        "context_three_reference": "other",
        "context_three_condition_number": context_three["condition_number"],
        "combined_model_pre_estimation_vif_acceptable": combined_diagnostics_acceptable,
        "joint_binary_exact_correction_feasible": joint_diagnostic["invertible"],
        "joint_binary_exact_correction_note": (
            "An exact multinomial correction would require an invertible 8-state "
            "manual-to-LLM confusion matrix for the three overlapping binary labels."
        ),
        "bootstrap_repetitions_requested": args.bootstrap_repetitions,
        "bootstrap_successful_context_draws": int(
            bootstrap_draws.loc[bootstrap_draws["model_id"].eq("context_calibrated"), "draw"].nunique()
        ),
    }
    (output_dir / "analysis_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    print("\nModel diagnostics")
    print(
        diagnostics[
            ["model_id", "n", "r2", "adjusted_r2", "rmse", "mean_actor_residual_ar1", "signal_wald_p"]
        ].to_string(index=False)
    )
    print("\nSignal coefficients")
    print(
        signal_coefficients[
            ["model_id", "term", "estimate", "hac_se", "p_value", "fdr_p_value_within_estimator"]
        ].to_string(index=False)
    )
    print("\nBattaglia corrections")
    print(corrected[["model_id", "estimator", "term", "estimate", "hac_se", "p_value"]].to_string(index=False))
    print("\nRolling prediction")
    print(prediction_summary.to_string(index=False))
    print("\nValidation bootstrap")
    print(bootstrap_summary.to_string(index=False))


if __name__ == "__main__":
    main()
