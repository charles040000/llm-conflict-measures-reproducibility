#!/usr/bin/env python3
"""Compare temporal windows with a common approximately 12-week history."""

from __future__ import annotations

import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import build_actor_month_panel as actor_builder  # noqa: E402
import model_utils as main_models  # noqa: E402
import temporal_aggregation_utils as comparison  # noqa: E402


OUTPUT = ROOT / "results" / "robustness" / "temporal_rebuilt"
HISTORY_START = comparison.ORIGIN - pd.Timedelta(weeks=12)


def apply_constant_history(panel: pd.DataFrame, weeks: int | None) -> pd.DataFrame:
    """Replace period lags with three non-overlapping, roughly four-week bins."""
    result = panel.sort_values(["actor_key", "period_start"]).copy()
    periods_per_bin = {1: 4, 2: 2, 4: 1, None: 1}[weeks]
    grouped = result.groupby("actor_key", sort=False)["y_ucdp_actor_fatalities"]
    targets = [
        "log1p_y_ucdp_actor_fatalities",
        "log1p_y_ucdp_actor_fatalities_lag1",
        "log1p_y_ucdp_actor_fatalities_lag2",
    ]
    for bin_index, target in enumerate(targets):
        first_shift = bin_index * periods_per_bin
        shifted = [
            grouped.shift(first_shift + offset)
            for offset in range(periods_per_bin)
        ]
        total = pd.concat(shifted, axis=1).sum(
            axis=1,
            min_count=periods_per_bin,
        )
        result[target] = np.log1p(total)
    return result


def bootstrap_physical_validation(
    panel: pd.DataFrame,
    validation: pd.DataFrame,
    minimum_articles: int,
    hac_lags: int,
    repetitions: int = 1000,
    seed: int = 9621,
) -> pd.DataFrame:
    """Propagate validation-rate uncertainty for the physical-violence model."""
    spec = next(item for item in comparison.SPECS if item.key == "physical_violence")
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float | int | str]] = []
    for draw in range(repetitions):
        sampled = validation.iloc[
            rng.integers(0, len(validation), size=len(validation))
        ].reset_index(drop=True)
        try:
            rate = main_models.estimate_binary_calibration(sampled, spec)
            calibrated = panel.copy()
            calibrated[spec.calibrated_column] = (
                (
                    calibrated[spec.raw_column]
                    - float(rate["false_positive_rate"])
                )
                / float(rate["denominator"])
            ).clip(0, 1)
            sample = main_models.base_sample(calibrated, minimum_articles)
            fit = main_models.fit_model(
                sample,
                "physical_violence_validation_bootstrap",
                [spec.calibrated_column],
                hac_lags,
            )
            position = list(fit["names"]).index(spec.calibrated_column)
            coefficient = fit["coefficients"].set_index("term").loc[
                spec.calibrated_column
            ]
            point_se = float(coefficient["hac_se"])
            rows.append(
                {
                    "draw": draw,
                    "estimator": "calibrated",
                    "estimate": float(coefficient["estimate"]),
                    "point_hac_se": point_se,
                }
            )
            omega = main_models.binary_index_omega(
                fit["frame"][spec.calibrated_column].to_numpy(float),
                float(rate["sensitivity"]),
                float(rate["false_positive_rate"]),
            )
            correction = main_models.corrected_estimator(
                fit,
                omega,
                [spec.calibrated_column],
                fit["frame"]["n_articles"].to_numpy(float),
            )
            for estimator, beta in [
                ("bca", correction["additive_beta"]),
                ("bcm", correction["multiplicative_beta"]),
            ]:
                if beta is not None:
                    rows.append(
                        {
                            "draw": draw,
                            "estimator": estimator,
                            "estimate": float(np.asarray(beta)[position]),
                            "point_hac_se": point_se,
                        }
                    )
        except (ValueError, np.linalg.LinAlgError):
            continue

    draws = pd.DataFrame(rows)
    summary = []
    for estimator, group in draws.groupby("estimator"):
        estimate = float(group["estimate"].median())
        validation_se = float(group["estimate"].std(ddof=1))
        point_se = float(group["point_hac_se"].median())
        combined_se = math.sqrt(point_se**2 + validation_se**2)
        summary.append(
            {
                "estimator": estimator,
                "successful_draws": int(group["draw"].nunique()),
                "median_estimate": estimate,
                "validation_se": validation_se,
                "point_hac_se": point_se,
                "combined_se": combined_se,
                "combined_p_value": main_models.normal_p_value(
                    estimate / combined_se
                ),
                "combined_ci_low": estimate - 1.96 * combined_se,
                "combined_ci_high": estimate + 1.96 * combined_se,
            }
        )
    return pd.DataFrame(summary)


def write_summary(
    results: pd.DataFrame,
    samples: pd.DataFrame,
    rolling: pd.DataFrame,
    rolling_uncertainty: pd.DataFrame,
) -> None:
    physical = results.loc[
        results["threshold_rule"].eq("common_min10")
        & results["estimator"].eq("raw")
        & results["variable"].eq("physical_violence")
    ].set_index("window")
    forecast = rolling.loc[
        rolling["threshold_rule"].eq("common_min10")
        & rolling["model_id"].eq("single_physical_violence_raw")
    ].set_index("window")
    forecast_uncertainty = rolling_uncertainty.loc[
        rolling_uncertainty["threshold_rule"].eq("common_min10")
        & rolling_uncertainty["model_id"].eq("single_physical_violence_raw")
    ].set_index("window")

    lines = [
        "# Temporal aggregation with constant conflict-history lookback",
        "",
        "The weekly, two-week, and four-week models use three non-overlapping four-week fatality-history bins covering approximately 12 weeks. The calendar-month model uses the current and two preceding calendar months. All models therefore contain three history coefficients and approximately three months of prior conflict information.",
        "",
        "## Main-story physical violence",
        "",
        "| Window | N | Coefficient | HAC SE | p | Within-window FDR | Across-window FDR | Partial R2 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for window in comparison.WINDOWS:
        row = physical.loc[window]
        lines.append(
            f"| {comparison.WINDOWS[window]['label']} | {int(row['n'])} | "
            f"{row['estimate']:.3f} | {row['standard_error']:.3f} | "
            f"{row['p_value']:.3f} | {row['fdr_p']:.3f} | "
            f"{row['fdr_across_windows']:.3f} | {row['partial_r2']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Expanding-window prediction",
            "",
            "| Window | Forecast N | RMSE change | Percent change | Bootstrap 95% interval |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for window in comparison.WINDOWS:
        row = forecast.loc[window]
        uncertainty = forecast_uncertainty.loc[window]
        lines.append(
            f"| {comparison.WINDOWS[window]['label']} | "
            f"{int(row['n_predictions'])} | {row['rmse_change_vs_controls']:.4f} | "
            f"{row['rmse_change_percent']:.2f}% | "
            f"[{uncertainty['bootstrap_ci_low']:.4f}, "
            f"{uncertainty['bootstrap_ci_high']:.4f}] |"
        )

    lines.extend(
        [
            "",
            "## Sample by actor",
            "",
            "| Window | Actor | Observations | Median articles |",
            "|---|---|---:|---:|",
        ]
    )
    common = samples.loc[samples["threshold_rule"].eq("common_min10")]
    for window in comparison.WINDOWS:
        for _, row in common.loc[common["window"].eq(window)].iterrows():
            lines.append(
                f"| {row['window_label']} | {row['actor_name']} | "
                f"{int(row['observations'])} | {row['median_articles']:.1f} |"
            )
    (OUTPUT / "temporal_aggregation_constant_history.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    comparison.OUTPUT = OUTPUT
    articles = comparison.load_articles()
    _, constituents, _ = actor_builder.load_actor_groups(comparison.GROUPS)
    events = comparison.load_linked_events(
        set(constituents["group:farc_bloc"]),
        event_start=HISTORY_START,
    )
    validation = pd.read_csv(comparison.VALIDATION)
    if "in_three_actor_calibration" in validation:
        flag = validation["in_three_actor_calibration"].astype(str).str.lower()
        validation = validation.loc[flag.isin({"true", "1"})].copy()
    rates = {
        spec.key: main_models.estimate_binary_calibration(validation, spec)
        for spec in comparison.SPECS
    }

    threshold_schemes = {
        "common_min10": {
            "1_week": 10,
            "2_weeks": 10,
            "4_weeks": 10,
            "calendar_month": 10,
        },
        "scaled_block_threshold": {
            "1_week": 10,
            "2_weeks": 20,
            "4_weeks": 40,
            "calendar_month": 10,
        },
    }
    primary_rows: list[dict[str, object]] = []
    secondary_rows: list[dict[str, object]] = []
    sample_frames: list[pd.DataFrame] = []
    rolling_frames: list[pd.DataFrame] = []
    rolling_uncertainty_frames: list[pd.DataFrame] = []
    validation_uncertainty_frames: list[pd.DataFrame] = []

    for window_id, settings in comparison.WINDOWS.items():
        panel = comparison.build_panel(
            articles,
            events,
            settings["weeks"],
            grid_start=HISTORY_START,
        )
        panel = apply_constant_history(panel, settings["weeks"])
        panel = panel.loc[panel["period_start"].ge(comparison.START)].copy()
        panel = comparison.apply_binary_calibration(panel, rates)
        panel.to_csv(OUTPUT / f"panel_{window_id}.csv", index=False)
        for threshold_rule, thresholds in threshold_schemes.items():
            minimum_articles = thresholds[window_id]
            primary, secondary, sample = comparison.fit_rows(
                panel,
                window_id,
                threshold_rule,
                minimum_articles,
                settings["hac_lags"],
                rates,
            )
            primary_rows.extend(primary)
            secondary_rows.extend(secondary)
            sample_frames.append(sample)

            regression_sample = main_models.base_sample(panel, minimum_articles)
            rolling_specs = {"controls_only": []}
            rolling_specs.update(
                {
                    f"single_{spec.key}_raw": [spec.raw_column]
                    for spec in comparison.SPECS
                }
            )
            predictions, rolling_summary = main_models.rolling_predictions(
                regression_sample,
                rolling_specs,
                start_test="2022-01-01",
            )
            rolling_summary.insert(0, "window", window_id)
            rolling_summary.insert(1, "window_label", settings["label"])
            rolling_summary.insert(2, "threshold_rule", threshold_rule)
            rolling_summary.insert(3, "minimum_articles", minimum_articles)
            rolling_frames.append(rolling_summary)

            uncertainty = main_models.rolling_prediction_uncertainty(
                predictions,
                repetitions=1000,
                seed=8731,
            )
            uncertainty.insert(0, "window", window_id)
            uncertainty.insert(1, "window_label", settings["label"])
            uncertainty.insert(2, "threshold_rule", threshold_rule)
            uncertainty.insert(3, "minimum_articles", minimum_articles)
            rolling_uncertainty_frames.append(uncertainty)

            validation_uncertainty = bootstrap_physical_validation(
                panel,
                validation,
                minimum_articles,
                settings["hac_lags"],
            )
            validation_uncertainty.insert(0, "window", window_id)
            validation_uncertainty.insert(1, "window_label", settings["label"])
            validation_uncertainty.insert(2, "threshold_rule", threshold_rule)
            validation_uncertainty.insert(3, "minimum_articles", minimum_articles)
            validation_uncertainty_frames.append(validation_uncertainty)

    primary_results = comparison.add_fdr(pd.DataFrame(primary_rows))
    secondary_results = pd.DataFrame(secondary_rows)
    samples = pd.concat(sample_frames, ignore_index=True)
    rolling_results = pd.concat(rolling_frames, ignore_index=True)
    rolling_uncertainty = pd.concat(rolling_uncertainty_frames, ignore_index=True)
    validation_uncertainty = pd.concat(
        validation_uncertainty_frames,
        ignore_index=True,
    )

    primary_results.to_csv(OUTPUT / "temporal_aggregation_primary_results.csv", index=False)
    secondary_results.to_csv(OUTPUT / "temporal_aggregation_secondary_results.csv", index=False)
    samples.to_csv(OUTPUT / "temporal_aggregation_sample_summary.csv", index=False)
    rolling_results.to_csv(OUTPUT / "temporal_aggregation_rolling_prediction.csv", index=False)
    rolling_uncertainty.to_csv(
        OUTPUT / "temporal_aggregation_rolling_prediction_uncertainty.csv",
        index=False,
    )
    validation_uncertainty.to_csv(
        OUTPUT / "temporal_aggregation_physical_validation_uncertainty.csv",
        index=False,
    )
    comparison.coefficient_plot(primary_results)
    write_summary(primary_results, samples, rolling_results, rolling_uncertainty)
    print((OUTPUT / "temporal_aggregation_constant_history.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
