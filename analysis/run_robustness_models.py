#!/usr/bin/env python3
"""Reproduce robustness models, extensions, and their principal figure."""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parent))

import joint_model  # noqa: E402
import model_utils as models  # noqa: E402
import temporal_aggregation_utils as aggregation  # noqa: E402
import run_calibration_models as calibration  # noqa: E402
import run_viirs_thermal_models as viirs_fire  # noqa: E402
import run_viirs_nightlight_models as viirs_light  # noqa: E402


FINAL_PANEL = ROOT / "results" / "calibration" / "uniform_joint" / "panel_2_weeks_calibrated.csv"
FINAL_RATES = ROOT / "results" / "calibration" / "uniform_joint" / "validation_rates.csv"
BETA_DRAWS = ROOT / "results" / "calibration" / "joint_beta" / "beta_global_posterior_draws.csv"
BETA_PARAMETERS = ROOT / "results" / "calibration" / "joint_beta" / "beta_parameter_summary.csv"
AGGREGATION = ROOT / "data" / "processed" / "temporal_windows"
TIMING = ROOT / "results" / "robustness" / "timing"
MAIN_MONTHLY_PANEL = ROOT / "data" / "processed" / "monthly_model_panel.csv"
VIIRS_LIGHT_PANEL = ROOT / "data" / "processed" / "viirs_nightlight_panel.csv"
VIIRS_FIRE_PANEL = ROOT / "data" / "processed" / "viirs_thermal_panel.csv"
ARTICLE_LABELS = ROOT / "data" / "processed" / "article_labels_compact.csv"

REPORT = ROOT / "results" / "robustness"
TABLES = REPORT / "tables"
FIGURES = ROOT / "figures" / "robustness"

VARIABLE_LABELS = {
    "direct_conflict_relevance": "Direct conflict relevance",
    "physical_violence": "Main-story physical violence",
    "fatalities_present": "Main-story fatalities",
}
ESTIMATOR_ORDER = ["naive", "direct", "bca", "bcm", "joint_mle"]

EXPLORATORY_SPECS = [
    {
        "key": "event_modality_realized",
        "label": "Realized main-story event",
        "field": "event_modality",
        "positive": {"realized"},
    },
    {
        "key": "actor_role_perpetrator",
        "label": "Matched actor portrayed as perpetrator",
        "field": "matched_actor_role",
        "positive": {"perpetrator"},
    },
    {
        "key": "current_or_recent",
        "label": "Current or recent main story",
        "field": "event_time_relation",
        "positive": {"current_or_recent"},
    },
    {
        "key": "main_story_injuries",
        "label": "Main-story injuries",
        "field": "injuries_present",
        "positive": {"yes"},
    },
    {
        "key": "violence_anywhere",
        "label": "Physical violence mentioned anywhere",
        "field": "any_physical_violence_mentioned",
        "positive": {"yes"},
    },
    {
        "key": "fatalities_anywhere",
        "label": "Fatalities mentioned anywhere",
        "field": "any_fatalities_mentioned",
        "positive": {"yes"},
    },
    {
        "key": "injuries_anywhere",
        "label": "Injuries mentioned anywhere",
        "field": "any_injuries_mentioned",
        "positive": {"yes"},
    },
    {
        "key": "weapons_anywhere",
        "label": "Weapon mentioned anywhere",
        "field": "any_weapon_mentioned",
        "positive": {"yes"},
    },
    {
        "key": "main_story_weapon_use",
        "label": "Main-story weapon use",
        "field": "weapon_use_class",
        "positive": {"salw", "heavy", "mixed"},
    },
    {
        "key": "multiple_events",
        "label": "Multiple events mentioned",
        "field": "multiple_events_mentioned",
        "positive": {"yes"},
    },
]


def write_text(path: Path, text: str) -> None:
    path.write_text(text.strip() + "\n", encoding="utf-8")


def p_text(value: float) -> str:
    if not math.isfinite(value):
        return "--"
    if value < 0.001:
        return "$<0.001$"
    return f"{value:.3f}"


def rate_lookup() -> dict[str, dict[str, float | int]]:
    data = pd.read_csv(FINAL_RATES).set_index("variable")
    result: dict[str, dict[str, float | int]] = {}
    for variable, row in data.iterrows():
        result[variable] = {
            "sensitivity": float(row["sensitivity"]),
            "false_positive_rate": float(row["false_positive_rate"]),
            "denominator": float(row["separation"]),
            "validation_manual_positive": int(
                row["false_negative"] + row["true_positive"]
            ),
            "validation_true_positive": int(row["true_positive"]),
            "validation_manual_negative": int(
                row["true_negative"] + row["false_positive"]
            ),
            "validation_false_positive": int(row["false_positive"]),
        }
    return result


def calibrate_panel(
    panel: pd.DataFrame,
    rates: dict[str, dict[str, float | int]],
) -> pd.DataFrame:
    result = panel.copy()
    for spec in models.BINARY_SPECS:
        rate = rates[spec.key]
        result[spec.calibrated_column] = (
            (
                pd.to_numeric(result[spec.raw_column], errors="coerce")
                - float(rate["false_positive_rate"])
            )
            / float(rate["denominator"])
        ).clip(0, 1)
    return result


def joint_start_values(variable: str) -> dict[str, np.ndarray]:
    draws = pd.read_csv(BETA_DRAWS)
    current = draws.loc[draws["variable"].eq(variable)].copy()
    parameters = pd.read_csv(BETA_PARAMETERS)
    alpha_rows = parameters.loc[
        parameters["variable"].eq(variable)
        & parameters["parameter"].str.startswith("alpha[", na=False)
    ].copy()
    alpha_rows["alpha_index"] = alpha_rows["parameter"].str.extract(r"\[(\d+)\]").astype(int)
    alpha = alpha_rows.sort_values("alpha_index")["mean"].to_numpy(float)
    return {
        "gamma": current["gamma"].to_numpy(float),
        "beta0": current["beta0"].to_numpy(float),
        "beta1": current["beta1"].to_numpy(float),
        "sigma_y": current["sigma_y"].to_numpy(float),
        "latent_mean": current["latent_mean"].to_numpy(float),
        "latent_concentration": current["latent_concentration"].to_numpy(float),
        "alpha": np.repeat(alpha[None, :], len(current), axis=0),
    }


def coefficient_result(
    fit: dict[str, object],
    term: str,
    estimator: str,
) -> dict[str, float | str]:
    row = fit["coefficients"].set_index("term").loc[term]
    return {
        "estimator": estimator,
        "estimate": float(row["estimate"]),
        "standard_error": float(row["hac_se"]),
        "p_value": float(row["p_value"]),
        "ci_low": float(row["ci_low"]),
        "ci_high": float(row["ci_high"]),
        "uncertainty": "conditional actor-specific HAC",
    }


def corrected_result(
    fit: dict[str, object],
    term: str,
    beta: np.ndarray,
    estimator: str,
) -> dict[str, float | str]:
    position = list(fit["names"]).index(term)
    estimate = float(beta[position])
    standard_error = float(
        np.sqrt(max(float(fit["covariance"][position, position]), 0.0))
    )
    p_value = models.normal_p_value(estimate / standard_error)
    return {
        "estimator": estimator,
        "estimate": estimate,
        "standard_error": standard_error,
        "p_value": p_value,
        "ci_low": estimate - 1.96 * standard_error,
        "ci_high": estimate + 1.96 * standard_error,
        "uncertainty": "conditional actor-specific HAC",
    }


def estimate_panel(
    panel: pd.DataFrame,
    rates: dict[str, dict[str, float | int]],
    group: str,
    specification: str,
    minimum_articles: int,
    hac_lags: int,
    history: str,
    include_joint: bool = True,
) -> list[dict[str, object]]:
    calibrated = calibrate_panel(panel, rates)
    sample = models.base_sample(calibrated, minimum_articles)
    actor_counts = sample.groupby("actor_name").size().to_dict()
    rows: list[dict[str, object]] = []

    for spec in models.BINARY_SPECS:
        raw_fit = models.fit_model(
            sample,
            f"{group}_{spec.key}_raw",
            [spec.raw_column],
            hac_lags,
        )
        calibrated_fit = models.fit_model(
            sample,
            f"{group}_{spec.key}_calibrated",
            [spec.calibrated_column],
            hac_lags,
        )
        results = [
            coefficient_result(raw_fit, spec.raw_column, "naive"),
            coefficient_result(calibrated_fit, spec.calibrated_column, "direct"),
        ]
        rate = rates[spec.key]
        omega = models.binary_index_omega(
            calibrated_fit["frame"][spec.calibrated_column].to_numpy(float),
            float(rate["sensitivity"]),
            float(rate["false_positive_rate"]),
        )
        correction = models.corrected_estimator(
            calibrated_fit,
            omega,
            [spec.calibrated_column],
            calibrated_fit["frame"]["n_articles"].to_numpy(float),
        )
        results.append(
            corrected_result(
                calibrated_fit,
                spec.calibrated_column,
                np.asarray(correction["additive_beta"]),
                "bca",
            )
        )
        if correction["multiplicative_beta"] is not None:
            results.append(
                corrected_result(
                    calibrated_fit,
                    spec.calibrated_column,
                    np.asarray(correction["multiplicative_beta"]),
                    "bcm",
                )
            )

        if include_joint:
            joint = calibration.prepare_joint_data(sample, spec)
            mle = joint_model.fit_integrated_beta_latent_mle(
                joint,
                rate,
                joint_start_values(spec.key),
                quadrature_nodes=128,
            )
            results.append(
                {
                    "estimator": "joint_mle",
                    "estimate": float(mle["gamma"]),
                    "standard_error": float(mle["gamma_se"]),
                    "p_value": models.normal_p_value(
                        float(mle["gamma"]) / float(mle["gamma_se"])
                    ),
                    "ci_low": float(mle["gamma"] - 1.96 * mle["gamma_se"]),
                    "ci_high": float(mle["gamma"] + 1.96 * mle["gamma_se"]),
                    "uncertainty": "inverse integrated-likelihood information",
                    "joint_mle_success": bool(mle["success"]),
                }
            )

        for result in results:
            rows.append(
                {
                    "group": group,
                    "specification": specification,
                    "history": history,
                    "minimum_articles": minimum_articles,
                    "hac_lags": hac_lags,
                    "variable": spec.key,
                    "n": int(len(sample)),
                    "n_hamas": int(actor_counts.get("Hamas", 0)),
                    "n_taliban": int(actor_counts.get("Taliban", 0)),
                    "n_farc": int(actor_counts.get("FARC family", 0)),
                    **result,
                }
            )
    return rows


def run_robustness_models() -> pd.DataFrame:
    rates = rate_lookup()
    final_panel = pd.read_csv(FINAL_PANEL, dtype={"actor_key": str})
    rows: list[dict[str, object]] = []

    for threshold in (10, 20, 40):
        label = f"Minimum {threshold} articles"
        if threshold == 20:
            label += " (baseline)"
        rows.extend(
            estimate_panel(
                final_panel,
                rates,
                "Article threshold",
                label,
                threshold,
                6,
                "three two-week fatality terms",
            )
        )

    aggregation_settings = [
        ("1_week", "One week", 10, 13),
        ("2_weeks", "Two weeks", 20, 6),
        ("4_weeks", "Four weeks", 40, 3),
        ("calendar_month", "Calendar month", 40, 3),
    ]
    for file_key, label, threshold, hac_lags in aggregation_settings:
        panel = pd.read_csv(
            AGGREGATION / f"panel_{file_key}.csv",
            dtype={"actor_key": str},
        )
        rows.extend(
            estimate_panel(
                panel,
                rates,
                "Temporal aggregation",
                label,
                threshold,
                hac_lags,
                "approximately 12 weeks",
            )
        )

    rows.extend(
        estimate_panel(
            final_panel,
            rates,
            "Inference",
            "Twelve-lag HAC bandwidth",
            20,
            12,
            "three two-week fatality terms",
            include_joint=False,
        )
    )
    result = pd.DataFrame(rows)
    result["fdr_p"] = np.nan
    for (_, _, estimator), block in result.groupby(
        ["group", "specification", "estimator"],
        sort=False,
    ):
        result.loc[block.index, "fdr_p"] = models.fdr_bh(block["p_value"])
    result.to_csv(REPORT / "robustness_model_results.csv", index=False)
    return result


def viirs_primary_results() -> pd.DataFrame:
    monthly = pd.read_csv(MAIN_MONTHLY_PANEL, dtype={"actor_key": str})
    monthly["month"] = monthly["month"].astype(str)
    shares = {
        "direct_conflict_relevance": "share_direct_conflict_relevance_raw",
        "physical_violence": "share_physical_violence_raw",
        "fatalities_present": "share_fatalities_present_raw",
    }
    merge_columns = ["actor_key", "month", *shares.values()]
    rows: list[dict[str, object]] = []

    outcome_specs = [
        (
            "Monthly nighttime-radiance change",
            viirs_light,
            VIIRS_LIGHT_PANEL,
        ),
        (
            "Next-month thermal-anomaly rate",
            viirs_fire,
            VIIRS_FIRE_PANEL,
        ),
    ]
    for outcome_label, module, panel_path in outcome_specs:
        panel = module.prepare_panel(panel_path)
        panel["month"] = panel["predictor_month"].astype(str)
        panel = panel.drop(columns=list(shares.values()), errors="ignore").merge(
            monthly[merge_columns],
            on=["actor_key", "month"],
            how="left",
            validate="one_to_one",
        )
        controls, diagnostics, _ = module.fit_model(
            panel,
            "controls_only",
            None,
            3,
        )
        article = controls.loc[controls["term"].eq("log1p_n_articles")].iloc[0]
        rows.append(
            {
                "outcome": outcome_label,
                "variable": "article_volume",
                "estimate": float(article["estimate"]),
                "standard_error": float(article["hac_se"]),
                "p_value": float(article["p_value"]),
                "fdr_p": np.nan,
                "n": int(diagnostics["n"]),
            }
        )
        signal_rows = []
        for variable, signal in shares.items():
            coefficients, diagnostics, _ = module.fit_model(
                panel,
                variable,
                signal,
                3,
            )
            coefficient = coefficients.loc[coefficients["is_llm_signal"]].iloc[0]
            signal_rows.append(
                {
                    "outcome": outcome_label,
                    "variable": variable,
                    "estimate": float(coefficient["estimate"]),
                    "standard_error": float(coefficient["hac_se"]),
                    "p_value": float(coefficient["p_value"]),
                    "n": int(diagnostics["n"]),
                }
            )
        signals = pd.DataFrame(signal_rows)
        signals["fdr_p"] = models.fdr_bh(signals["p_value"])
        rows.extend(signals.to_dict("records"))

    result = pd.DataFrame(rows)
    result.to_csv(REPORT / "viirs_primary_results.csv", index=False)
    return result


def exploratory_llm_results() -> pd.DataFrame:
    """Estimate the remaining substantive LLM shares in the baseline model."""
    articles = aggregation.load_articles()
    articles["GlobalEventID"] = articles["GlobalEventID"].astype(str)
    fields = sorted({str(spec["field"]) for spec in EXPLORATORY_SPECS})
    labels = pd.read_csv(
        ARTICLE_LABELS,
        usecols=["GlobalEventID", *fields],
        low_memory=False,
    )
    labels["GlobalEventID"] = labels["GlobalEventID"].astype(str)
    labels = labels.drop_duplicates("GlobalEventID")
    articles = articles.drop(columns=fields, errors="ignore").merge(
        labels,
        on="GlobalEventID",
        how="left",
        validate="one_to_one",
    )
    articles["period_start"] = aggregation.block_start(articles["date"], 2)
    articles["article_actor_weight"] = pd.to_numeric(
        articles["article_actor_weight"], errors="coerce"
    ).fillna(0)
    keys = ["actor_key", "period_start"]

    share_columns: list[tuple[dict[str, object], str]] = []
    aggregated: pd.DataFrame | None = None
    for spec in EXPLORATORY_SPECS:
        field = str(spec["field"])
        positive = set(spec["positive"])
        normalized = (
            articles[field].fillna("").astype(str).str.strip().str.lower()
        )
        values = articles[keys].copy()
        values["positive_weight"] = (
            normalized.isin(positive).astype(float)
            * articles["article_actor_weight"]
        )
        current = (
            values.groupby(keys, as_index=False)["positive_weight"]
            .sum()
            .rename(
                columns={
                    "positive_weight": f"n_exploratory_{spec['key']}"
                }
            )
        )
        aggregated = (
            current
            if aggregated is None
            else aggregated.merge(current, on=keys, how="outer", validate="one_to_one")
        )
        share_columns.append((spec, f"share_exploratory_{spec['key']}"))

    if aggregated is None:
        raise RuntimeError("No exploratory LLM specifications were configured")

    panel = pd.read_csv(FINAL_PANEL, dtype={"actor_key": str})
    panel["period_start"] = pd.to_datetime(panel["period_start"], errors="coerce")
    panel = panel.merge(
        aggregated,
        on=keys,
        how="left",
        validate="one_to_one",
    )
    for spec, share_column in share_columns:
        count_column = f"n_exploratory_{spec['key']}"
        panel[share_column] = (
            pd.to_numeric(panel[count_column], errors="coerce").fillna(0)
            / pd.to_numeric(panel["n_articles"], errors="coerce")
        )

    sample = models.base_sample(panel, 20)
    controls = models.fit_model(sample, "exploratory_controls", [], 6)
    control_sse = float(np.asarray(controls["residuals"]) @ np.asarray(controls["residuals"]))
    rows: list[dict[str, object]] = []
    for spec, share_column in share_columns:
        fit = models.fit_model(
            sample,
            f"exploratory_{spec['key']}",
            [share_column],
            6,
        )
        coefficient = fit["coefficients"].set_index("term").loc[share_column]
        model_sse = float(np.asarray(fit["residuals"]) @ np.asarray(fit["residuals"]))
        rows.append(
            {
                "variable": spec["key"],
                "label": spec["label"],
                "estimate": float(coefficient["estimate"]),
                "standard_error": float(coefficient["hac_se"]),
                "p_value": float(coefficient["p_value"]),
                "ci_low": float(coefficient["ci_low"]),
                "ci_high": float(coefficient["ci_high"]),
                "partial_r2": (control_sse - model_sse) / control_sse,
                "n": int(fit["diagnostics"]["n"]),
            }
        )
    result = pd.DataFrame(rows)
    result["fdr_p"] = models.fdr_bh(result["p_value"])
    result.to_csv(REPORT / "additional_llm_measure_results.csv", index=False)
    return result


def make_exploratory_llm_table(results: pd.DataFrame) -> None:
    lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\caption{Exploratory models for additional LLM-generated measures}",
        r"\label{tab:additional_llm_measure_models}",
        r"\small",
        r"\begin{tabularx}{\textwidth}{@{}Xrrrrr@{}}",
        r"\toprule",
        r"Generated measure & Estimate & HAC SE & $p$ & FDR $p$ & Partial $R^2$ \\",
        r"\midrule",
    ]
    for _, row in results.iterrows():
        lines.append(
            f"{row['label']} & {row['estimate']:.3f} & "
            f"{row['standard_error']:.3f} & {p_text(float(row['p_value']))} & "
            f"{p_text(float(row['fdr_p']))} & {row['partial_r2']:.3f} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabularx}",
            "",
            r"\begin{minipage}{0.96\textwidth}",
            r"\footnotesize",
            r"\textit{Notes:} Each row adds one raw LLM-generated share to the same 360-observation two-week baseline model. All retained articles enter the share denominator, and the named category is coded as one. The FDR column applies the Benjamini--Hochberg adjustment across the ten exploratory measures in this table. Partial $R^2$ is calculated relative to the controls-only model. These tests are secondary and do not replace the prespecified three-measure primary family.",
            r"\end{minipage}",
            r"\end{table}",
        ]
    )
    write_text(TABLES / "additional_llm_measure_models.tex", "\n".join(lines))


def table_cell(
    result: pd.DataFrame,
    estimator: str,
    include_p: bool = False,
) -> str:
    match = result.loc[result["estimator"].eq(estimator)]
    if match.empty:
        return "--"
    row = match.iloc[0]
    if include_p:
        return f"{row['estimate']:.3f} ({p_text(float(row['p_value']))})"
    return f"{row['estimate']:.3f} ({row['standard_error']:.3f})"


def make_physical_table(results: pd.DataFrame) -> None:
    physical = results.loc[results["variable"].eq("physical_violence")].copy()
    groups = ["Article threshold", "Temporal aggregation", "Inference"]
    lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\caption{Robustness of the main-story physical-violence coefficient}",
        r"\label{tab:physical_violence_robustness}",
        r"\scriptsize",
        r"\begin{tabularx}{\textwidth}{@{}Xrrrrrr@{}}",
        r"\toprule",
        r"Specification & $N$ & Naive & Direct & BCA & BCM & Joint MLE \\",
        r"\midrule",
    ]
    for group_position, group in enumerate(groups):
        lines.append(rf"\multicolumn{{7}}{{l}}{{\textit{{{group}}}}} \\")
        block = physical.loc[physical["group"].eq(group)]
        for specification in block["specification"].drop_duplicates():
            current = block.loc[block["specification"].eq(specification)]
            n = int(current["n"].iloc[0])
            lines.append(
                f"{specification} & {n} & "
                + " & ".join(
                    table_cell(current, estimator)
                    for estimator in ESTIMATOR_ORDER
                )
                + r" \\"
            )
        if group_position < len(groups) - 1:
            lines.append(r"\addlinespace")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabularx}",
            "",
            r"\begin{minipage}{0.97\textwidth}",
            r"\footnotesize",
            r"\textit{Notes:} Entries are coefficient estimates with standard errors in parentheses. Naive, direct, BCA, and BCM uncertainty is conditional on the pooled validation-estimated classification rates and uses actor-specific Newey--West HAC standard errors. Joint estimates maximize the integrated beta latent-share likelihood; their standard errors use the inverse observed-information matrix. The article-threshold panel retains the baseline three-period history. The temporal-aggregation panel instead holds the conflict-history span at approximately 12 weeks and uses thresholds of 10, 20, 40, and 40 articles for weekly, two-week, four-week, and calendar-month cells, respectively. The HAC-bandwidth row changes inference only, so the joint estimate is not repeated.",
            r"\end{minipage}",
            r"\end{table}",
        ]
    )
    write_text(TABLES / "physical_violence_robustness.tex", "\n".join(lines))


def make_full_tables(results: pd.DataFrame) -> None:
    table_specs = [
        (
            "Article threshold",
            "Complete generated-share robustness to the article threshold",
            "tab:robustness_all_thresholds",
            "robustness_all_thresholds.tex",
        ),
        (
            "Temporal aggregation",
            "Complete generated-share robustness across aggregation intervals",
            "tab:robustness_all_aggregation",
            "robustness_all_aggregation.tex",
        ),
    ]
    for group, caption, label, filename in table_specs:
        data = results.loc[results["group"].eq(group)]
        lines = [
            r"\begin{table}[!htbp]",
            r"\centering",
            rf"\caption{{{caption}}}",
            rf"\label{{{label}}}",
            r"\scriptsize",
            r"\setlength{\tabcolsep}{4pt}",
            r"\begin{tabularx}{\textwidth}{@{}Xrrrrr@{}}",
            r"\toprule",
            r"Generated measure & Naive & Direct & BCA & BCM & Joint MLE \\",
            r"\midrule",
        ]
        specifications = data["specification"].drop_duplicates().tolist()
        for spec_position, specification in enumerate(specifications):
            block = data.loc[data["specification"].eq(specification)]
            n = int(block["n"].iloc[0])
            lines.append(
                rf"\multicolumn{{6}}{{@{{}}l}}{{\textit{{{specification} ($N={n}$)}}}} \\"
            )
            for variable_position, variable in enumerate(VARIABLE_LABELS):
                current = block.loc[block["variable"].eq(variable)]
                lines.append(
                    f"{VARIABLE_LABELS[variable]} & "
                    + " & ".join(
                        table_cell(current, estimator, include_p=True)
                        for estimator in ESTIMATOR_ORDER
                    )
                    + r" \\"
                )
            if spec_position < len(specifications) - 1:
                lines.append(r"\addlinespace")
        lines.extend(
            [
                r"\bottomrule",
                r"\end{tabularx}",
                "",
                r"\begin{minipage}{0.97\textwidth}",
                r"\footnotesize",
                r"\textit{Notes:} Panel headings report the estimation-sample size. Entries are coefficient estimates with unadjusted two-sided $p$-values in parentheses. Naive and analytical-correction uncertainty is conditional on the pooled validation-estimated classification rates. Joint rows use the integrated beta latent-share likelihood and inverse observed-information uncertainty. Each generated measure enters a separate regression with the relevant baseline controls.",
                r"\end{minipage}",
                r"\end{table}",
            ]
        )
        write_text(TABLES / filename, "\n".join(lines))


def make_viirs_table(results: pd.DataFrame) -> None:
    variable_labels = {
        "article_volume": "Article volume",
        **VARIABLE_LABELS,
    }
    lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\caption{Exploratory models with satellite-derived outcomes}",
        r"\label{tab:viirs_alternative_outcomes}",
        r"\small",
        r"\begin{tabularx}{\textwidth}{@{}XXrrrr@{}}",
        r"\toprule",
        r"Outcome & Predictor & Estimate & HAC SE & $p$ & FDR $p$ \\",
        r"\midrule",
    ]
    outcomes = results["outcome"].drop_duplicates().tolist()
    for outcome_position, outcome in enumerate(outcomes):
        block = results.loc[results["outcome"].eq(outcome)]
        for row_position, (_, row) in enumerate(block.iterrows()):
            outcome_text = outcome if row_position == 0 else ""
            lines.append(
                f"{outcome_text} & {variable_labels[row['variable']]} & "
                f"{row['estimate']:.3f} & {row['standard_error']:.3f} & "
                f"{p_text(float(row['p_value']))} & "
                f"{p_text(float(row['fdr_p']))} \\\\"
            )
        if outcome_position < len(outcomes) - 1:
            lines.append(r"\addlinespace")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabularx}",
            "",
            r"\begin{minipage}{0.96\textwidth}",
            r"\footnotesize",
            r"\textit{Notes:} The satellite models use 201 monthly actor observations. Each LLM-generated share enters a separate regression. Nighttime-radiance models control for current radiance, two lagged changes, article volume, risk-area size, trend, actor effects, and month-of-year effects. Thermal-anomaly models use analogous outcome-history and risk-area controls with actor-specific month-of-year effects. Standard errors are actor-specific three-lag Newey--West HAC estimates. FDR adjustment is applied across the three LLM-generated measures separately for each outcome; it is not applied to article volume.",
            r"\end{minipage}",
            r"\end{table}",
        ]
    )
    write_text(TABLES / "viirs_alternative_outcomes.tex", "\n".join(lines))


def make_timing_figure() -> None:
    results = pd.read_csv(TIMING / "lead_lag_coefficients.csv")
    colors = {
        "Raw LLM share": "#277da1",
        "Directly calibrated share": "#b33a3a",
    }
    offsets = {"Raw LLM share": -0.06, "Directly calibrated share": 0.06}
    figure, axis = plt.subplots(figsize=(8.2, 4.8))
    for specification in colors:
        group = results.loc[
            results["share_specification"].eq(specification)
        ].sort_values("horizon")
        estimates = group["estimate"].to_numpy(float)
        axis.errorbar(
            group["horizon"].to_numpy(float) + offsets[specification],
            estimates,
            yerr=np.vstack(
                [
                    estimates - group["comparison_ci_low"].to_numpy(float),
                    group["comparison_ci_high"].to_numpy(float) - estimates,
                ]
            ),
            marker="o",
            capsize=3,
            linewidth=1.5,
            color=colors[specification],
            label=specification,
        )
    axis.axhline(0, color="#555555", linewidth=0.9)
    axis.axvline(0, color="#999999", linestyle="--", linewidth=0.9)
    axis.set_xticks([-2, -1, 0, 1, 2])
    axis.set_xticklabels(["$-2$", "$-1$", "$0$", "$+1$", "$+2$"])
    axis.set_xlabel("Outcome horizon in two-week periods")
    axis.set_ylabel("Physical-violence share coefficient")
    axis.grid(axis="y", color="#dddddd", linewidth=0.6)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.legend(frameon=False, ncol=2, loc="upper left")
    figure.tight_layout()
    figure.savefig(
        FIGURES / "physical_violence_lead_lag_profile.pdf",
        bbox_inches="tight",
    )
    figure.savefig(
        FIGURES / "physical_violence_lead_lag_profile.png",
        dpi=250,
        bbox_inches="tight",
    )
    plt.close(figure)


def write_section() -> None:
    section = r"""
\section{Robustness Checks and Extensions}
\label{sec:robustness_extensions}

This section is primarily motivated by three concerns:

\begin{enumerate}
    \item[(i)] \textbf{Shared reporting environment.} As discussed in
    Section~\ref{subsec:pipeline_limitations}, the GDELT-based covariates and
    the UCDP outcome are constructed separately but draw partly on the same
    news environment. The temporal placebo examines whether this shared basis
    could contribute to the observed timing pattern. An alternative outcome
    constructed independently of news reporting is also examined briefly using
    satellite data.

    \item[(ii)] \textbf{Dependence on the outcome window.} The positive
    association between main-story physical violence and subsequent
    fatalities may depend on the exact two-week aggregation and forecast
    window. This possibility is examined by varying the aggregation interval
    and by shifting the fatality outcome backward and forward relative to the
    article period.

    \item[(iii)] \textbf{Minimum article requirement.} Actor-period shares
    based on few articles are more sensitive to individual classification
    errors. A higher threshold produces more stable shares but removes
    observations, particularly for the FARC family. The analysis therefore
    compares minimum requirements of ten, twenty, and forty articles.
\end{enumerate}

The HAC bandwidth is varied as an additional inference check.
The robustness analysis focuses on main-story physical violence because it is
the only primary generated measure that produces a consistent positive result
in the baseline analysis. Robustness estimates for the other two primary
measures, direct conflict relevance and main-story fatalities, are reported
alongside the physical-violence estimates in Appendix
Tables~\ref{tab:robustness_all_thresholds} and
\ref{tab:robustness_all_aggregation}. Results for the remaining substantive
LLM-generated measures are reported separately in Appendix
Table~\ref{tab:additional_llm_measure_models}.

\subsection{Aggregation Interval and Article Threshold}
\label{subsec:robustness_aggregation_threshold}

Panel A of Table~\ref{tab:physical_violence_robustness} varies the minimum
number of articles while retaining the baseline two-week specification. At a
minimum of ten articles, the naive and corrected physical-violence
coefficients remain positive and precisely estimated. The baseline threshold
of twenty articles produces a similar conclusion. Raising the threshold to
forty reduces the sample from 360 to 277 observations and leaves only nine
FARC-family periods. The coefficient remains positive, but its standard error
increases and its interval includes zero. This loss of precision therefore
coincides with a substantial change in sample composition rather than a
reversal of the estimated relationship.

Panel B compares weekly, two-week, four-week, and calendar-month aggregation.
To make the dynamic controls more comparable, these models use approximately
twelve weeks of conflict history. The article thresholds are scaled with the
length of the aggregation window, except that both four-week and
calendar-month cells require forty articles. The physical-violence coefficient
is positive under every interval and every estimator. The two-week model
provides the clearest statistical evidence, while the wider intervals contain
fewer observations and produce wider intervals. Because changing the window
also changes the forecast horizon and the scale of the outcome, coefficient
magnitudes should not be compared as if they represented the same estimand.
The two-week estimate in this comparison also differs from the baseline row in
Panel A because the standardized comparison controls for twelve rather than
six weeks of prior fatalities.

\input{tables/physical_violence_robustness}

\subsection{Additional LLM-Generated Measures}
\label{subsec:additional_llm_measures}

The baseline and calibration analyses focus on three primary binary measures:
direct conflict relevance, main-story physical violence, and main-story
fatalities. To examine whether this choice overlooks a stronger signal,
Appendix Table~\ref{tab:additional_llm_measure_models} estimates the same
two-week specification separately for ten other substantive LLM-generated
measures. These cover event modality, actor role, event timing, injuries,
weapons, article-wide mentions, and the presence of multiple events. The
collapsed event-context model is reported separately and is not duplicated in
this table.

None of the ten additional coefficients is statistically significant at the
5\% level before multiple-testing adjustment, and none survives the
exploratory FDR correction. The largest partial $R^2$ is 0.018, indicating
that these measures explain little additional outcome variation after the
baseline controls are included. They are therefore reported as exploratory
checks and are not carried forward into the primary calibration comparison.

\subsection{Temporal Placebo Test}
\label{subsec:temporal_placebo}

A remaining concern is that the physical-violence share and UCDP fatalities
may respond to the same reported episode but record it in adjacent periods.
If this timing mechanism explained the result, the news share in period $t$
should be associated similarly with fatalities immediately before and after
that period. The diagnostic therefore holds the period-$t$ news share fixed
and shifts the fatality outcome from $t-2$ through $t+2$.

The test uses the same 352 actor-period observations at every horizon and a
lean set of controls consisting of article volume, actor fixed effects, the
time trend, and the predefined regime indicators. Lagged fatalities are
omitted because including them would mechanically alter the backward-looking
coefficients. Because this specification omits lagged fatalities, the
resulting raw and calibrated coefficients serve as timing diagnostics and are
not directly comparable with the conditional baseline estimates.

Figure~\ref{fig:physical_violence_lead_lag_profile} shows little association
with fatalities in the two preceding periods. For the raw share, the
coefficient is 0.344 at $h=-1$ compared with 1.778 at $h=+1$. The
corresponding directly calibrated estimates are 0.218 and 1.260. The
forward-minus-backward difference is statistically distinguishable from zero
for both the raw estimate ($p=0.015$) and the calibrated estimate
($p=0.017$).

The horizons have different implications for the timing concern. A positive
coefficient at $h=0$ is not itself problematic: it indicates that the
generated share captures violence reported during the same period. The
negative horizons provide the actual placebo. If the coefficients at $h=-1$
and $h=+1$ were similar, the association could plausibly arise because the
same violent episode was reported and dated on opposite sides of a two-week
boundary. Instead, the forward coefficient is significantly larger than the
backward coefficient. This provides evidence that the positive association is
not primarily explained by such a symmetric reporting or event-dating
artifact. The test cannot rule out other forms of dependence arising from the
broader news environment.

\begin{figure}[!htbp]
\centering
\includegraphics[width=0.88\textwidth]
{figures/physical_violence_lead_lag_profile.pdf}
\caption{Lead-lag profile of the main-story physical-violence coefficient.
The predictor is fixed in period $t$, while the transformed actor-linked UCDP
fatality outcome is shifted from two periods before to two periods after it.
Lines report 95\% intervals based on six-lag actor-specific Newey--West HAC
uncertainty; calibrated intervals additionally include validation-bootstrap
uncertainty. The diagnostic uses identical observations and controls at every
horizon but omits lagged fatalities.}
\label{fig:physical_violence_lead_lag_profile}
\end{figure}

\subsection{Additional Inference Check}
\label{subsec:robustness_inference}

The baseline HAC estimator allows residual dependence over six two-week
periods. Extending the bandwidth to twelve periods leaves the physical-violence
point estimates unchanged and slightly reduces rather than increases their
estimated standard errors. The naive estimate remains positive with a HAC
standard error of 0.402 and $p=0.010$. The baseline inference is therefore not
dependent on restricting residual autocorrelation to six periods.

\subsection{Alternative Satellite-Derived Outcomes}
\label{subsec:alternative_satellite_outcomes}

As discussed in Chapter~\ref{sec:literature}, no consistently measured
official actor-level fatality series is available across all three conflict
settings. Alternative event datasets such as Armed Conflict Location \& Event
Data (ACLED) are also constructed partly from media and other reported sources
and therefore remain exposed to the same underlying reporting environment as
UCDP. Identifying an alternative outcome that is independent of news reporting,
available consistently across all three conflict settings, and observable at
short intervals is therefore difficult.

Satellite observations provide one possible source of independently measured
physical signals. The exploratory premise is that conflict activity may leave
observable traces: explosions or conflict-related fires may generate thermal
anomalies, while infrastructure damage, power outages, and reduced activity may
change nighttime radiance. These signals do not identify individual battles or
fatalities directly. They instead measure possible physical consequences of
conflict. On this basis, two exploratory outcomes were constructed from VIIRS
observations: monthly changes in nighttime radiance
from the stray-light-corrected NOAA monthly composite\footnote{Google Earth
Engine Data Catalog, \texttt{NOAA/VIIRS/DNB/MONTHLY\_V1/VCMSLCFG},
\url{https://developers.google.com/earth-engine/datasets/catalog/NOAA_VIIRS_DNB_MONTHLY_V1_VCMSLCFG}.}
and next-month thermal-anomaly rates from the NASA VNP14A1.002 daily
product.\footnote{NASA VIIRS Land Science Investigator Processing System,
\url{https://doi.org/10.5067/VIIRS/VNP14A1.002}.} Both are measured within the
actors' rolling UCDP-defined activity areas. The satellite observations are
independent of news reports, although UCDP records still determine their
geographic footprint.

Neither article volume nor any of the three primary LLM-generated shares has a
statistically detectable relationship with either satellite outcome
(Appendix Table~\ref{tab:viirs_alternative_outcomes}). Thermal anomalies are
particularly difficult to interpret because the detector also captures
wildfires and agricultural burning; these sources dominate the recurring
variation in the FARC-family regions. Nighttime-light changes and thermal
anomalies also measure economic disruption and heat rather than fatalities.
The satellite results are therefore retained as an exploratory extension and
not used as replacements for the actor-linked UCDP outcome.

\subsection{Robustness Summary}
\label{subsec:robustness_summary}

The physical-violence coefficient remains positive across alternative
aggregation intervals and analytical correction methods. Evidence becomes
less precise when the article threshold is raised to forty because that rule
removes most FARC-family periods. The temporal placebo provides additional
support for a forward-looking component, although it cannot fully separate
news-based measurement from the broader reporting environment. Finally, the
absence of corresponding results for the exploratory satellite outcomes shows
that the main finding does not generalize automatically to other dimensions
of conflict-related disruption.
"""
    write_text(REPORT / "robustness_extension_section.tex", section)


def write_appendix_inputs() -> None:
    appendix = r"""
\subsection{Complete Robustness Results}
\label{app:robustness_full_results}

\input{tables/robustness_all_thresholds}

\input{tables/robustness_all_aggregation}

\input{tables/additional_llm_measure_models}

\input{tables/viirs_alternative_outcomes}
"""
    write_text(REPORT / "robustness_appendix_tables.tex", appendix)


def main() -> None:
    REPORT.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    run_robustness_models()
    viirs_primary_results()
    exploratory_llm_results()
    make_timing_figure()
    print(f"Wrote robustness artifacts to {REPORT}")


if __name__ == "__main__":
    main()
