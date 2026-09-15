#!/usr/bin/env python3
"""Compare weekly, multi-week, and monthly versions of the final baseline model."""

from __future__ import annotations

import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parent))

import build_actor_month_panel as actor_builder  # noqa: E402
import model_utils as main_models  # noqa: E402


ARTICLES = ROOT / "data" / "processed" / "article_labels_compact.csv"
GED = ROOT / "data" / "external" / "GEDEvent_v26_1.csv"
GROUPS = ROOT / "data" / "reference" / "actor_group_crosswalk.csv"
VALIDATION = ROOT / "data" / "validation" / "manual_validation.csv"
OUTPUT = ROOT / "results" / "robustness" / "temporal_rebuilt"

START = pd.Timestamp("2020-01-01")
END_EXCLUSIVE = pd.Timestamp("2026-01-01")
ORIGIN = pd.Timestamp("2019-12-30")
TAKEOVER = pd.Timestamp("2021-08-15")
HAMAS_BREAK = pd.Timestamp("2023-10-07")

ACTORS = {
    "209": "Hamas",
    "303": "Taliban",
    "group:farc_bloc": "FARC family",
}

WINDOWS = {
    "1_week": {"weeks": 1, "hac_lags": 13, "label": "1 week"},
    "2_weeks": {"weeks": 2, "hac_lags": 6, "label": "2 weeks"},
    "4_weeks": {"weeks": 4, "hac_lags": 3, "label": "4 weeks"},
    "calendar_month": {"weeks": None, "hac_lags": 3, "label": "Calendar month"},
}

SPECS = main_models.BINARY_SPECS


def block_start(values: pd.Series, weeks: int | None) -> pd.Series:
    dates = pd.to_datetime(values, errors="coerce")
    if weeks is None:
        return dates.dt.to_period("M").dt.start_time
    width_days = 7 * weeks
    offsets = (dates - ORIGIN).dt.days // width_days
    return ORIGIN + pd.to_timedelta(offsets * width_days, unit="D")


def period_grid(
    weeks: int | None,
    grid_start: pd.Timestamp = START,
) -> pd.DataFrame:
    if weeks is None:
        first_start = grid_start.to_period("M").start_time
        starts = pd.date_range(first_start, END_EXCLUSIVE, freq="MS", inclusive="left")
        ends = starts + pd.offsets.MonthBegin(1)
    else:
        width = pd.Timedelta(weeks=weeks)
        offset = (grid_start - ORIGIN).days // (7 * weeks)
        first_start = ORIGIN + pd.Timedelta(weeks=offset * weeks)
        starts = pd.date_range(
            first_start,
            END_EXCLUSIVE,
            freq=f"{weeks}W-MON",
            inclusive="left",
        )
        ends = starts + width
        complete = ends <= END_EXCLUSIVE
        starts = starts[complete]
        ends = ends[complete]
    return pd.DataFrame(
        {
            "period_start": starts,
            "period_end": ends,
            "period_index": np.arange(len(starts), dtype=int),
        }
    )


def load_articles() -> pd.DataFrame:
    header = pd.read_csv(ARTICLES, nrows=0).columns
    if {"actor_key", "actor_name", "date", "article_actor_weight"}.issubset(header):
        articles = pd.read_csv(ARTICLES, dtype={"GlobalEventID": str, "actor_key": str})
        articles["date"] = pd.to_datetime(articles["date"], errors="coerce")
        articles = articles.loc[
            articles["actor_key"].isin(ACTORS)
            & articles["date"].ge(START)
            & articles["date"].lt(END_EXCLUSIVE)
        ].copy()
        articles["actor_name"] = articles["actor_key"].map(ACTORS)
        return articles
    group_map, _, _ = actor_builder.load_actor_groups(GROUPS)
    articles, _, _ = actor_builder.load_article_actor_labels(
        ARTICLES,
        "2020-01",
        "2025-12",
        actor_group_map=group_map,
        exclude_multi_actor_articles=True,
    )
    dates = pd.read_csv(ARTICLES, usecols=["GlobalEventID", "date"])
    articles["GlobalEventID"] = articles["GlobalEventID"].astype(str)
    dates["GlobalEventID"] = dates["GlobalEventID"].astype(str)
    dates = dates.drop_duplicates("GlobalEventID")
    articles = articles.merge(dates, on="GlobalEventID", how="left", validate="many_to_one")
    articles["date"] = pd.to_datetime(articles["date"], errors="coerce")
    articles = articles.loc[
        articles["actor_key"].isin(ACTORS)
        & articles["date"].ge(START)
        & articles["date"].lt(END_EXCLUSIVE)
    ].copy()
    articles["actor_name"] = articles["actor_key"].map(ACTORS)
    return articles


def load_linked_events(
    farc_ids: set[str],
    event_start: pd.Timestamp = START,
) -> pd.DataFrame:
    columns = ["id", "date_start", "side_a_new_id", "side_b_new_id", "best"]
    events = pd.read_csv(GED, usecols=columns, low_memory=False)
    events["date"] = pd.to_datetime(events["date_start"], errors="coerce")
    events = events.loc[
        events["date"].notna()
        & events["date"].ge(event_start)
        & events["date"].lt(END_EXCLUSIVE)
    ].copy()
    events["side_a"] = events["side_a_new_id"].map(actor_builder.norm_actor_id)
    events["side_b"] = events["side_b_new_id"].map(actor_builder.norm_actor_id)
    events["best"] = pd.to_numeric(events["best"], errors="coerce").fillna(0).clip(lower=0)

    masks = {
        "209": events["side_a"].eq("209") | events["side_b"].eq("209"),
        "303": (
            events["date"].lt(TAKEOVER)
            & (events["side_a"].eq("303") | events["side_b"].eq("303"))
        )
        | (
            events["date"].ge(TAKEOVER)
            & (events["side_a"].eq("130") | events["side_b"].eq("130"))
        ),
        "group:farc_bloc": events["side_a"].isin(farc_ids) | events["side_b"].isin(farc_ids),
    }
    frames = []
    for actor_key, mask in masks.items():
        selected = events.loc[mask, ["id", "date", "best"]].copy()
        selected["actor_key"] = actor_key
        frames.append(selected.drop_duplicates(["actor_key", "id"]))
    return pd.concat(frames, ignore_index=True)


def aggregate_articles(articles: pd.DataFrame, weeks: int | None) -> pd.DataFrame:
    data = articles.copy()
    data["period_start"] = block_start(data["date"], weeks)
    keys = ["actor_key", "actor_name", "period_start"]
    panel = (
        data.groupby(keys, as_index=False)
        .agg(
            n_articles=("article_actor_weight", "sum"),
            n_unique_articles=("GlobalEventID", "nunique"),
        )
    )
    for spec in SPECS:
        article_field = spec.llm_field.removesuffix("_llm")
        positive = (
            data[article_field].fillna("").astype(str).str.strip().str.lower().eq(spec.positive)
        )
        values = data[keys + ["article_actor_weight"]].copy()
        values["positive_weight"] = positive.astype(float) * data["article_actor_weight"]
        counts = (
            values.groupby(keys, as_index=False)["positive_weight"]
            .sum()
            .rename(columns={"positive_weight": f"n_{spec.key}"})
        )
        panel = panel.merge(counts, on=keys, how="left", validate="one_to_one")

    context = data["event_context"].fillna("").astype(str).str.strip().str.lower()
    data["context_three"] = np.select(
        [context.isin(["cbt", "civ"]), context.eq("pol")],
        ["violent", "pol"],
        default="other",
    )
    context_counts = (
        data.groupby(keys + ["context_three"])["article_actor_weight"]
        .sum()
        .unstack(fill_value=0)
        .reindex(columns=["violent", "pol", "other"], fill_value=0)
        .reset_index()
        .rename(columns=lambda value: f"n_context_three_{value}" if value in {"violent", "pol", "other"} else value)
    )
    panel = panel.merge(context_counts, on=keys, how="left", validate="one_to_one")
    return panel


def build_panel(
    articles: pd.DataFrame,
    events: pd.DataFrame,
    weeks: int | None,
    grid_start: pd.Timestamp = START,
) -> pd.DataFrame:
    grid = period_grid(weeks, grid_start=grid_start)
    actor_grid = pd.DataFrame(
        {"actor_key": list(ACTORS), "actor_name": list(ACTORS.values())}
    ).merge(grid, how="cross")

    article_period = aggregate_articles(articles, weeks)
    linked = events.copy()
    linked["period_start"] = block_start(linked["date"], weeks)
    outcomes = (
        linked.groupby(["actor_key", "period_start"], as_index=False)
        .agg(
            y_ucdp_actor_fatalities=("best", "sum"),
            n_ucdp_actor_events=("id", "nunique"),
        )
    )
    panel = actor_grid.merge(
        article_period,
        on=["actor_key", "actor_name", "period_start"],
        how="left",
    ).merge(outcomes, on=["actor_key", "period_start"], how="left")

    count_columns = [column for column in panel if column.startswith("n_")]
    panel[count_columns] = panel[count_columns].fillna(0)
    panel["n_articles"] = panel["n_articles"].fillna(0)
    panel["y_ucdp_actor_fatalities"] = panel["y_ucdp_actor_fatalities"].fillna(0)
    panel["log1p_y_ucdp_actor_fatalities"] = np.log1p(panel["y_ucdp_actor_fatalities"])
    panel["log1p_n_articles"] = np.log1p(panel["n_articles"])

    denominator = panel["n_articles"].replace(0, np.nan)
    for spec in SPECS:
        panel[spec.raw_column] = panel[f"n_{spec.key}"] / denominator
    for category in ("violent", "pol", "other"):
        panel[f"share_context_three_{category}_raw"] = (
            panel[f"n_context_three_{category}"] / denominator
        )

    panel = panel.sort_values(["actor_key", "period_start"]).reset_index(drop=True)
    grouped = panel.groupby("actor_key", sort=False)
    panel[main_models.OUTCOME] = grouped["log1p_y_ucdp_actor_fatalities"].shift(-1)
    panel["y_ucdp_actor_fatalities_next"] = grouped["y_ucdp_actor_fatalities"].shift(-1)
    panel["log1p_y_ucdp_actor_fatalities_lag1"] = grouped[
        "log1p_y_ucdp_actor_fatalities"
    ].shift(1)
    panel["log1p_y_ucdp_actor_fatalities_lag2"] = grouped[
        "log1p_y_ucdp_actor_fatalities"
    ].shift(2)
    panel["outcome_period_start"] = grouped["period_start"].shift(-1)
    panel["outcome_period_end"] = grouped["period_end"].shift(-1)

    if weeks is None:
        panel["time_years"] = panel["period_index"] / 12.0
    else:
        panel["time_years"] = panel["period_index"] * 7.0 * weeks / 365.25
    panel["taliban_outcome_transition_aug2021"] = (
        panel["actor_key"].eq("303")
        & panel["outcome_period_start"].lt(TAKEOVER)
        & panel["outcome_period_end"].gt(TAKEOVER)
    ).astype(float)
    panel["taliban_outcome_post_takeover"] = (
        panel["actor_key"].eq("303")
        & panel["outcome_period_start"].ge(TAKEOVER)
    ).astype(float)
    panel["hamas_outcome_post_oct2023"] = (
        panel["actor_key"].eq("209")
        & panel["outcome_period_end"].gt(HAMAS_BREAK)
    ).astype(float)
    panel["month"] = panel["period_start"].dt.strftime("%Y-%m-%d")
    return panel


def apply_binary_calibration(
    panel: pd.DataFrame,
    rates: dict[str, dict[str, object]],
) -> pd.DataFrame:
    result = panel.copy()
    for spec in SPECS:
        rate = rates[spec.key]
        result[spec.calibrated_column] = (
            (result[spec.raw_column] - float(rate["false_positive_rate"]))
            / float(rate["denominator"])
        ).clip(0, 1)
    return result


def fit_rows(
    panel: pd.DataFrame,
    window_id: str,
    threshold_rule: str,
    minimum_articles: int,
    hac_lags: int,
    rates: dict[str, dict[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]], pd.DataFrame]:
    sample = main_models.base_sample(panel, minimum_articles)
    controls = main_models.fit_model(
        sample,
        f"{window_id}_{threshold_rule}_controls",
        [],
        hac_lags,
    )
    control_sse = float(np.sum(np.square(controls["residuals"])))
    primary_rows: list[dict[str, object]] = []

    for spec in SPECS:
        raw_fit = main_models.fit_model(
            sample,
            f"{window_id}_{threshold_rule}_{spec.key}_raw",
            [spec.raw_column],
            hac_lags,
        )
        calibrated_fit = main_models.fit_model(
            sample,
            f"{window_id}_{threshold_rule}_{spec.key}_calibrated",
            [spec.calibrated_column],
            hac_lags,
        )
        for estimator, fit, term in [
            ("raw", raw_fit, spec.raw_column),
            ("calibrated", calibrated_fit, spec.calibrated_column),
        ]:
            coefficient = fit["coefficients"].set_index("term").loc[term]
            sse = float(np.sum(np.square(fit["residuals"])))
            primary_rows.append(
                {
                    "window": window_id,
                    "window_label": WINDOWS[window_id]["label"],
                    "threshold_rule": threshold_rule,
                    "minimum_articles": minimum_articles,
                    "hac_lags": hac_lags,
                    "variable": spec.key,
                    "estimator": estimator,
                    "estimate": coefficient["estimate"],
                    "standard_error": coefficient["hac_se"],
                    "p_value": coefficient["p_value"],
                    "ci_low": coefficient["ci_low"],
                    "ci_high": coefficient["ci_high"],
                    "effect_10pp_percent": 100 * (math.exp(0.1 * coefficient["estimate"]) - 1),
                    "n": len(fit["frame"]),
                    "r2": fit["diagnostics"]["r2"],
                    "adjusted_r2": fit["diagnostics"]["adjusted_r2"],
                    "rmse": fit["diagnostics"]["rmse"],
                    "partial_r2": (control_sse - sse) / control_sse,
                    "mean_actor_residual_ar1": fit["diagnostics"]["mean_actor_residual_ar1"],
                }
            )

        calibrated_frame = calibrated_fit["frame"]
        rate = rates[spec.key]
        omega = main_models.binary_index_omega(
            calibrated_frame[spec.calibrated_column].to_numpy(float),
            float(rate["sensitivity"]),
            float(rate["false_positive_rate"]),
        )
        correction = main_models.corrected_estimator(
            calibrated_fit,
            omega,
            [spec.calibrated_column],
            calibrated_frame["n_articles"].to_numpy(float),
        )
        position = list(calibrated_fit["names"]).index(spec.calibrated_column)
        standard_error = float(np.sqrt(calibrated_fit["covariance"][position, position]))
        for estimator, beta in [
            ("bca", correction["additive_beta"]),
            ("bcm", correction["multiplicative_beta"]),
        ]:
            if beta is None:
                continue
            estimate = float(beta[position])
            z_value = estimate / standard_error
            primary_rows.append(
                {
                    "window": window_id,
                    "window_label": WINDOWS[window_id]["label"],
                    "threshold_rule": threshold_rule,
                    "minimum_articles": minimum_articles,
                    "hac_lags": hac_lags,
                    "variable": spec.key,
                    "estimator": estimator,
                    "estimate": estimate,
                    "standard_error": standard_error,
                    "p_value": main_models.normal_p_value(z_value),
                    "ci_low": estimate - 1.96 * standard_error,
                    "ci_high": estimate + 1.96 * standard_error,
                    "effect_10pp_percent": 100 * (math.exp(0.1 * estimate) - 1),
                    "n": len(calibrated_frame),
                    "r2": calibrated_fit["diagnostics"]["r2"],
                    "adjusted_r2": calibrated_fit["diagnostics"]["adjusted_r2"],
                    "rmse": calibrated_fit["diagnostics"]["rmse"],
                    "partial_r2": (
                        control_sse - float(np.sum(np.square(calibrated_fit["residuals"])))
                    )
                    / control_sse,
                    "mean_actor_residual_ar1": calibrated_fit["diagnostics"][
                        "mean_actor_residual_ar1"
                    ],
                }
            )

    raw_binary = [spec.raw_column for spec in SPECS]
    joint = main_models.fit_model(
        sample,
        f"{window_id}_{threshold_rule}_joint_binary_raw",
        raw_binary,
        hac_lags,
    )
    context_terms = [
        "share_context_three_violent_raw",
        "share_context_three_pol_raw",
    ]
    context = main_models.fit_model(
        sample,
        f"{window_id}_{threshold_rule}_context_three_raw",
        context_terms,
        hac_lags,
    )
    secondary_rows: list[dict[str, object]] = []
    for model_type, fit in [("joint_binary", joint), ("event_context", context)]:
        for _, coefficient in fit["coefficients"].loc[fit["coefficients"]["is_signal"]].iterrows():
            secondary_rows.append(
                {
                    "window": window_id,
                    "window_label": WINDOWS[window_id]["label"],
                    "threshold_rule": threshold_rule,
                    "minimum_articles": minimum_articles,
                    "hac_lags": hac_lags,
                    "model": model_type,
                    "term": coefficient["term"],
                    "estimate": coefficient["estimate"],
                    "standard_error": coefficient["hac_se"],
                    "p_value": coefficient["p_value"],
                    "ci_low": coefficient["ci_low"],
                    "ci_high": coefficient["ci_high"],
                    "n": len(fit["frame"]),
                    "r2": fit["diagnostics"]["r2"],
                    "rmse": fit["diagnostics"]["rmse"],
                    "joint_wald_p": fit["diagnostics"]["signal_wald_p"],
                }
            )

    sample_summary = (
        sample.groupby(["actor_key", "actor_name"], as_index=False)
        .agg(
            observations=("period_start", "size"),
            median_articles=("n_articles", "median"),
            minimum_articles=("n_articles", "min"),
            maximum_articles=("n_articles", "max"),
        )
    )
    sample_summary.insert(0, "window", window_id)
    sample_summary.insert(1, "window_label", WINDOWS[window_id]["label"])
    sample_summary.insert(2, "threshold_rule", threshold_rule)
    sample_summary.insert(3, "required_articles", minimum_articles)
    return primary_rows, secondary_rows, sample_summary


def add_fdr(results: pd.DataFrame) -> pd.DataFrame:
    result = results.copy()
    result["fdr_p"] = np.nan
    result["fdr_across_windows"] = np.nan
    raw = result["estimator"].eq("raw")
    for _, index in result.loc[raw].groupby(["window", "threshold_rule"]).groups.items():
        result.loc[index, "fdr_p"] = main_models.fdr_bh(result.loc[index, "p_value"])
    for _, index in result.loc[raw].groupby("threshold_rule").groups.items():
        result.loc[index, "fdr_across_windows"] = main_models.fdr_bh(
            result.loc[index, "p_value"]
        )
    return result


def coefficient_plot(results: pd.DataFrame) -> None:
    data = results.loc[
        results["threshold_rule"].eq("common_min10") & results["estimator"].eq("raw")
    ].copy()
    variables = [spec.key for spec in SPECS]
    labels = {
        "direct_conflict_relevance": "Direct conflict relevance",
        "physical_violence": "Main-story physical violence",
        "fatalities_present": "Main-story fatalities",
    }
    order = list(WINDOWS)
    colors = ["#277da1", "#4f7d44", "#b78b2e", "#b33a3a"]
    figure, axes = plt.subplots(1, 3, figsize=(12.5, 4.2), sharey=True)
    for ax, variable in zip(axes, variables):
        subset = data.loc[data["variable"].eq(variable)].set_index("window").loc[order]
        positions = np.arange(len(order))[::-1]
        for position, color, (_, row) in zip(positions, colors, subset.iterrows()):
            ax.errorbar(
                row["estimate"],
                position,
                xerr=[[row["estimate"] - row["ci_low"]], [row["ci_high"] - row["estimate"]]],
                fmt="o",
                color=color,
                capsize=3,
            )
        ax.axvline(0, color="#333333", linewidth=0.8)
        ax.set_title(labels[variable])
        ax.set_xlabel("Coefficient (95% HAC CI)")
        ax.grid(axis="x", color="#dddddd", linewidth=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_yticks(positions)
        ax.set_yticklabels([WINDOWS[item]["label"] for item in order])
    figure.tight_layout()
    figure.savefig(OUTPUT / "temporal_aggregation_coefficients.pdf", bbox_inches="tight")
    figure.savefig(OUTPUT / "temporal_aggregation_coefficients.png", dpi=300, bbox_inches="tight")
    plt.close(figure)


def write_summary(
    results: pd.DataFrame,
    samples: pd.DataFrame,
    rolling: pd.DataFrame,
    rolling_uncertainty: pd.DataFrame,
) -> None:
    raw = results.loc[
        results["threshold_rule"].eq("common_min10")
        & results["estimator"].eq("raw")
        & results["variable"].eq("physical_violence")
    ].set_index("window")
    lines = [
        "# Temporal aggregation comparison",
        "",
        "All primary comparisons use the final actor-linked UCDP outcome, the same generated-share definitions and controls, a common minimum of ten articles, and HAC bandwidths spanning approximately three months.",
        "",
        "| Window | N | Coefficient | HAC SE | p | Within-window FDR | Across-window FDR | Partial R2 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for window in WINDOWS:
        row = raw.loc[window]
        lines.append(
            f"| {WINDOWS[window]['label']} | {int(row['n'])} | {row['estimate']:.3f} | "
            f"{row['standard_error']:.3f} | {row['p_value']:.3f} | {row['fdr_p']:.3f} | "
            f"{row['fdr_across_windows']:.3f} | {row['partial_r2']:.4f} |"
        )
    lines.extend(
        [
            "",
            "The coefficients refer to different forecast horizons, so their magnitudes are not directly comparable as a common causal or predictive estimand. The comparison is a temporal-aggregation sensitivity analysis.",
            "",
            "## Expanding-window prediction: physical violence",
            "",
            "| Window | Forecast N | RMSE change | Percent change | Bootstrap 95% interval |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    rolling_physical = rolling.loc[
        rolling["threshold_rule"].eq("common_min10")
        & rolling["model_id"].eq("single_physical_violence_raw")
    ].set_index("window")
    rolling_physical_uncertainty = rolling_uncertainty.loc[
        rolling_uncertainty["threshold_rule"].eq("common_min10")
        & rolling_uncertainty["model_id"].eq("single_physical_violence_raw")
    ].set_index("window")
    for window in WINDOWS:
        row = rolling_physical.loc[window]
        uncertainty = rolling_physical_uncertainty.loc[window]
        lines.append(
            f"| {WINDOWS[window]['label']} | {int(row['n_predictions'])} | "
            f"{row['rmse_change_vs_controls']:.4f} | {row['rmse_change_percent']:.2f}% | "
            f"[{uncertainty['bootstrap_ci_low']:.4f}, {uncertainty['bootstrap_ci_high']:.4f}] |"
        )
    lines.extend(
        [
            "",
            "Negative RMSE changes indicate improvement relative to the controls-only model for the same temporal window. Only the two-week bootstrap interval excludes zero. The calendar-month model has the largest proportional RMSE improvement and the largest in-sample partial R2, while the weekly model adds almost no predictive information.",
            "",
            "## Sample by actor",
            "",
            "| Window | Actor | Observations | Median articles |",
            "|---|---|---:|---:|",
        ]
    )
    common = samples.loc[samples["threshold_rule"].eq("common_min10")]
    for window in WINDOWS:
        for _, row in common.loc[common["window"].eq(window)].iterrows():
            lines.append(
                f"| {row['window_label']} | {row['actor_name']} | {int(row['observations'])} | "
                f"{row['median_articles']:.1f} |"
            )
    (OUTPUT / "temporal_aggregation_comparison.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    articles = load_articles()
    _, constituents, _ = actor_builder.load_actor_groups(GROUPS)
    events = load_linked_events(set(constituents["group:farc_bloc"]))
    validation = pd.read_csv(VALIDATION)
    rates = {spec.key: main_models.estimate_binary_calibration(validation, spec) for spec in SPECS}

    primary_rows: list[dict[str, object]] = []
    secondary_rows: list[dict[str, object]] = []
    sample_frames: list[pd.DataFrame] = []
    rolling_frames: list[pd.DataFrame] = []
    rolling_uncertainty_frames: list[pd.DataFrame] = []
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
    panels: dict[str, pd.DataFrame] = {}
    for window_id, settings in WINDOWS.items():
        panel = build_panel(articles, events, settings["weeks"])
        panel = apply_binary_calibration(panel, rates)
        panels[window_id] = panel
        panel.to_csv(OUTPUT / f"panel_{window_id}.csv", index=False)
        for threshold_rule, thresholds in threshold_schemes.items():
            minimum_articles = thresholds[window_id]
            primary, secondary, sample = fit_rows(
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
                {f"single_{spec.key}_raw": [spec.raw_column] for spec in SPECS}
            )
            rolling_predictions, rolling_summary = main_models.rolling_predictions(
                regression_sample,
                rolling_specs,
                start_test="2022-01-01",
            )
            rolling_summary.insert(0, "window", window_id)
            rolling_summary.insert(1, "window_label", settings["label"])
            rolling_summary.insert(2, "threshold_rule", threshold_rule)
            rolling_summary.insert(3, "minimum_articles", minimum_articles)
            rolling_frames.append(rolling_summary)
            rolling_uncertainty = main_models.rolling_prediction_uncertainty(
                rolling_predictions,
                repetitions=1000,
                seed=8731,
            )
            rolling_uncertainty.insert(0, "window", window_id)
            rolling_uncertainty.insert(1, "window_label", settings["label"])
            rolling_uncertainty.insert(2, "threshold_rule", threshold_rule)
            rolling_uncertainty.insert(3, "minimum_articles", minimum_articles)
            rolling_uncertainty_frames.append(rolling_uncertainty)

    primary_results = add_fdr(pd.DataFrame(primary_rows))
    secondary_results = pd.DataFrame(secondary_rows)
    samples = pd.concat(sample_frames, ignore_index=True)
    rolling_results = pd.concat(rolling_frames, ignore_index=True)
    rolling_uncertainty_results = pd.concat(rolling_uncertainty_frames, ignore_index=True)
    primary_results.to_csv(OUTPUT / "temporal_aggregation_primary_results.csv", index=False)
    secondary_results.to_csv(OUTPUT / "temporal_aggregation_secondary_results.csv", index=False)
    samples.to_csv(OUTPUT / "temporal_aggregation_sample_summary.csv", index=False)
    rolling_results.to_csv(OUTPUT / "temporal_aggregation_rolling_prediction.csv", index=False)
    rolling_uncertainty_results.to_csv(
        OUTPUT / "temporal_aggregation_rolling_prediction_uncertainty.csv",
        index=False,
    )
    coefficient_plot(primary_results)
    write_summary(
        primary_results,
        samples,
        rolling_results,
        rolling_uncertainty_results,
    )

    print((OUTPUT / "temporal_aggregation_comparison.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
