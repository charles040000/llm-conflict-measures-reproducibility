#!/usr/bin/env python3
"""Rebuild the thesis figures from the included data and model results."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MPL_CACHE = ROOT / ".cache" / "matplotlib"
MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))

import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402


ACTOR_ORDER = ["Hamas", "Taliban", "FARC family"]
ACTOR_NAMES = {
    "209": "Hamas",
    "303": "Taliban",
    "group:farc_bloc": "FARC family",
}
ACTOR_COLORS = {
    "Hamas": "#2B6F8A",
    "Taliban": "#C05A3C",
    "FARC family": "#4D7C57",
}
VARIABLE_ORDER = [
    "direct_conflict_relevance",
    "physical_violence",
    "fatalities_present",
]
VARIABLE_LABELS = {
    "direct_conflict_relevance": "Direct conflict relevance",
    "physical_violence": "Main-story physical violence",
    "fatalities_present": "Main-story fatalities",
}
RAW_COLUMNS = {
    "direct_conflict_relevance": "share_direct_conflict_relevance_raw",
    "physical_violence": "share_physical_violence_raw",
    "fatalities_present": "share_fatalities_present_raw",
}
CALIBRATED_COLUMNS = {
    "direct_conflict_relevance": "share_direct_conflict_relevance_calibrated",
    "physical_violence": "share_physical_violence_calibrated",
    "fatalities_present": "share_fatalities_present_calibrated",
}
OUTCOME = "y_ucdp_actor_fatalities_next"
REQUIRED = [
    "log1p_y_ucdp_actor_fatalities_next",
    "log1p_y_ucdp_actor_fatalities",
    "log1p_y_ucdp_actor_fatalities_lag1",
    "log1p_y_ucdp_actor_fatalities_lag2",
]


def set_style() -> None:
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update(
        {
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "font.size": 9,
            "legend.fontsize": 8,
            "figure.titlesize": 12,
        }
    )


def save(fig: plt.Figure, group: str, stem: str) -> None:
    destination = ROOT / "figures" / group
    destination.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(destination / f"{stem}.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def load_period_panel() -> pd.DataFrame:
    panel = pd.read_csv(
        ROOT / "data" / "processed" / "actor_period_panel.csv",
        dtype={"actor_key": str},
        low_memory=False,
    )
    panel["period_start"] = pd.to_datetime(panel["period_start"])
    panel["outcome_period_start"] = pd.to_datetime(panel["outcome_period_start"])
    panel["display_actor"] = panel["actor_key"].map(ACTOR_NAMES)
    return panel.sort_values(["display_actor", "period_start"])


def baseline_sample(panel: pd.DataFrame) -> pd.DataFrame:
    return panel.loc[panel["n_articles"].ge(20)].dropna(subset=REQUIRED).copy()


def descriptive_figures() -> None:
    period_panel = load_period_panel()
    sample = baseline_sample(period_panel)
    monthly = pd.read_csv(
        ROOT / "data" / "processed" / "actor_month_panel.csv",
        dtype={"actor_key": str},
        low_memory=False,
    )
    articles = pd.read_csv(
        ROOT / "data" / "processed" / "article_labels_compact.csv",
        dtype={"GlobalEventID": str, "actor_key": str},
        low_memory=False,
    )
    articles["display_actor"] = articles["actor_key"].map(ACTOR_NAMES)

    totals = (
        monthly.groupby(["actor_key", "actor_name"], as_index=False)["n_articles"]
        .sum()
        .sort_values("n_articles", ascending=False)
        .head(12)
    )
    totals["display"] = totals["actor_key"].map(ACTOR_NAMES).fillna(totals["actor_name"])
    totals = totals.sort_values("n_articles")
    colors = [ACTOR_COLORS.get(name, "#AAB2B8") for name in totals["display"]]
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    bars = ax.barh(totals["display"], totals["n_articles"], color=colors, height=0.72)
    ax.set_xscale("log")
    ax.set_xlabel("Retained articles (log scale)")
    ax.set_title("Article coverage is concentrated in three actor groups", loc="left", fontweight="bold")
    ax.grid(axis="y", visible=False)
    for bar, value in zip(bars, totals["n_articles"]):
        ax.text(value * 1.08, bar.get_y() + bar.get_height() / 2, f"{value:,.0f}", va="center", fontsize=8)
    ax.set_xlim(5, totals["n_articles"].max() * 2.1)
    sns.despine(ax=ax)
    save(fig, "descriptive", "actor_distribution")

    fig, axes = plt.subplots(3, 1, figsize=(8.2, 7.2), sharex=True)
    for ax, actor in zip(axes, ACTOR_ORDER):
        frame = period_panel.loc[period_panel["display_actor"].eq(actor)]
        color = ACTOR_COLORS[actor]
        ax.plot(frame["period_start"], frame["n_articles"], color=color, lw=1, alpha=0.45, marker="o", ms=2)
        ax.plot(
            frame["period_start"],
            frame["n_articles"].rolling(3, center=True, min_periods=1).mean(),
            color=color,
            lw=2.1,
            label="Centered six-week mean",
        )
        ax.axhline(20, color="#70777C", lw=0.9, ls=":", label="20-article threshold")
        ax.set_ylabel("Articles")
        ax.set_title(actor, loc="left", fontweight="bold")
        ax.grid(axis="x", visible=False)
        ax.legend(loc="upper left")
    axes[-1].xaxis.set_major_locator(mdates.YearLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[-1].set_xlabel("Two-week predictor period")
    fig.suptitle("Two-week article volume, 2020-2025", x=0.08, ha="left", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save(fig, "descriptive", "article_volume_over_time")

    label_specs = {
        "Direct conflict relevance": ("conflict_relevance", "direct"),
        "Main-story physical violence": ("physical_violence_occurred", "yes"),
        "Main-story fatalities": ("fatalities_present", "yes"),
    }
    rows = []
    for actor in ACTOR_ORDER:
        frame = articles.loc[articles["display_actor"].eq(actor)]
        for label, (column, positive) in label_specs.items():
            rows.append(
                {
                    "actor": actor,
                    "label": label,
                    "share": 100 * frame[column].fillna("").astype(str).str.lower().eq(positive).mean(),
                }
            )
    label_shares = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    sns.barplot(data=label_shares, x="label", y="share", hue="actor", palette=ACTOR_COLORS, ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("Positive classifications (%)")
    ax.set_ylim(0, 100)
    ax.tick_params(axis="x", rotation=12)
    ax.legend(title="")
    ax.set_title("Primary article-level LLM labels", loc="left", fontweight="bold")
    fig.tight_layout()
    save(fig, "descriptive", "empirical_binary_label_distributions")

    context = (
        articles.assign(category=articles["event_context"].fillna("missing").astype(str).str.lower())
        .groupby(["display_actor", "category"], as_index=False)
        .size()
    )
    context["share"] = 100 * context["size"] / context.groupby("display_actor")["size"].transform("sum")
    context = context.loc[context["share"].ge(0.5)].copy()
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    sns.barplot(data=context, x="category", y="share", hue="display_actor", palette=ACTOR_COLORS, ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("Articles (%)")
    ax.tick_params(axis="x", rotation=25)
    ax.legend(title="")
    ax.set_title("Event-context classifications", loc="left", fontweight="bold")
    fig.tight_layout()
    save(fig, "descriptive", "empirical_categorical_label_distributions")

    fig, axes = plt.subplots(3, 1, figsize=(8.2, 6.8), sharex=True, sharey=True)
    for ax, actor in zip(axes, ACTOR_ORDER):
        frame = period_panel.loc[period_panel["display_actor"].eq(actor)].copy()
        included = frame["n_articles"].ge(20) & frame[REQUIRED].notna().all(axis=1)
        plotted_share = frame["share_physical_violence_raw"].where(included)
        ax.plot(frame["period_start"], 100 * plotted_share, color=ACTOR_COLORS[actor], lw=1.25)
        ax.axhline(100 * plotted_share.median(), color="#353A3E", ls=":", lw=1)
        ax.set_ylabel("Share (%)")
        ax.set_ylim(0, 100)
        ax.set_title(actor, loc="left", fontweight="bold")
    axes[-1].xaxis.set_major_locator(mdates.YearLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[-1].set_xlabel("Two-week predictor period")
    fig.suptitle("Main-story physical-violence share", x=0.08, ha="left", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save(fig, "descriptive", "physical_violence_actor_period_shares")

    fig, axes = plt.subplots(3, 1, figsize=(8.2, 7.2), sharex=True)
    ticks = [0, 1, 10, 100, 1000, 10000]
    for ax, actor in zip(axes, ACTOR_ORDER):
        frame = period_panel.loc[period_panel["display_actor"].eq(actor)].copy()
        included = frame["n_articles"].ge(20) & frame[REQUIRED].notna().all(axis=1)
        y = np.log1p(frame[OUTCOME].where(included))
        ax.fill_between(frame["outcome_period_start"], 0, y, color=ACTOR_COLORS[actor], alpha=0.22)
        ax.plot(frame["outcome_period_start"], y, color=ACTOR_COLORS[actor], lw=1.5)
        ax.set_yticks(np.log1p(ticks), [f"{value:,}" for value in ticks])
        ax.set_ylim(0, np.log1p(10000))
        ax.set_ylabel("Fatalities")
        ax.set_title(actor, loc="left", fontweight="bold")
    axes[-1].xaxis.set_major_locator(mdates.YearLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[-1].set_xlabel("Following two-week outcome period")
    fig.suptitle("UCDP fatalities in the following two-week period", x=0.08, ha="left", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save(fig, "descriptive", "fatalities_over_time")

    sample["log_outcome_plot"] = np.log1p(sample[OUTCOME])
    zero = sample.groupby("display_actor")[OUTCOME].apply(lambda values: 100 * values.eq(0).mean()).reindex(ACTOR_ORDER)
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.9), gridspec_kw={"width_ratios": [0.85, 1.45]})
    axes[0].bar(ACTOR_ORDER, zero, color=[ACTOR_COLORS[a] for a in ACTOR_ORDER], width=0.65)
    axes[0].set_ylabel("Periods with zero fatalities (%)")
    axes[0].set_title("Zero outcomes", loc="left", fontweight="bold")
    axes[0].tick_params(axis="x", rotation=20)
    for index, value in enumerate(zero):
        axes[0].text(index, value + 0.5, f"{value:.1f}%", ha="center")
    sns.boxplot(data=sample, x="display_actor", y="log_outcome_plot", order=ACTOR_ORDER, hue="display_actor", palette=ACTOR_COLORS, legend=False, showfliers=False, ax=axes[1])
    sns.stripplot(data=sample, x="display_actor", y="log_outcome_plot", order=ACTOR_ORDER, color="#283138", alpha=0.3, size=2.4, jitter=0.18, ax=axes[1])
    axes[1].set_yticks(np.log1p(ticks), [f"{value:,}" for value in ticks])
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Following-period fatalities")
    axes[1].set_title("Distribution across two-week periods", loc="left", fontweight="bold")
    axes[1].tick_params(axis="x", rotation=20)
    fig.tight_layout()
    save(fig, "descriptive", "outcome_distribution")

    sample["log_articles_plot"] = np.log1p(sample["n_articles"])
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.35), sharey=True)
    for ax, actor in zip(axes, ACTOR_ORDER):
        frame = sample.loc[sample["display_actor"].eq(actor)]
        rho, _ = spearmanr(frame["n_articles"], frame[OUTCOME])
        sns.regplot(data=frame, x="log_articles_plot", y="log_outcome_plot", ci=None, scatter_kws={"s": 18, "alpha": 0.5, "color": ACTOR_COLORS[actor]}, line_kws={"lw": 1.7, "color": "#252A2E"}, ax=ax)
        ax.set_title(actor, fontweight="bold")
        ax.set_xlabel("log(1 + articles at t)")
        ax.text(0.04, 0.94, f"Spearman rho = {rho:.2f}", transform=ax.transAxes, va="top")
    axes[0].set_ylabel("log(1 + fatalities at t+1)")
    axes[1].set_ylabel("")
    axes[2].set_ylabel("")
    fig.suptitle("Two-week article volume and following-period fatalities", x=0.06, ha="left", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, "descriptive", "article_volume_vs_fatalities")


def baseline_figures() -> None:
    coefficients = pd.read_csv(ROOT / "results" / "baseline" / "naive_binary_coefficients.csv")
    context = pd.read_csv(ROOT / "results" / "baseline" / "naive_context_coefficients.csv")
    separate = coefficients.loc[coefficients["model_id"].str.startswith("single_") & coefficients["is_signal"]].copy()
    joint = coefficients.loc[coefficients["model_id"].eq("joint_binary_raw") & coefficients["is_signal"]].copy()
    context = context.loc[context["is_signal"]].copy()
    plot = pd.concat(
        [
            separate.assign(group="Separate binary-share models"),
            joint.assign(group="Multivariable binary-share model"),
            context.assign(group="Event-context model"),
        ],
        ignore_index=True,
    )
    term_labels = {
        "share_direct_conflict_relevance_raw": "Direct conflict relevance",
        "share_physical_violence_raw": "Main-story physical violence",
        "share_fatalities_present_raw": "Main-story fatalities",
        "share_context_three_violent_raw": "Violent context",
        "share_context_three_pol_raw": "Political context",
    }
    plot["label"] = plot["term"].map(term_labels)
    colors = {
        "Separate binary-share models": "#2B7B9F",
        "Multivariable binary-share model": "#B54343",
        "Event-context model": "#4F7D45",
    }
    fig, ax = plt.subplots(figsize=(8.3, 5.0))
    for index, row in plot.reset_index(drop=True).iterrows():
        ax.errorbar(row["estimate"], index, xerr=[[row["estimate"] - row["ci_low"]], [row["ci_high"] - row["estimate"]]], fmt="o", color=colors[row["group"]], capsize=3)
    ax.axvline(0, color="#333333", lw=0.9)
    ax.set_yticks(np.arange(len(plot)), plot["label"])
    ax.invert_yaxis()
    ax.set_xlabel("Coefficient with 95% HAC interval")
    handles = [plt.Line2D([0], [0], marker="o", linestyle="", color=color, label=label) for label, color in colors.items()]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.14), frameon=False, ncol=1)
    fig.tight_layout()
    save(fig, "baseline", "baseline_naive_coefficients")

    residuals = pd.read_csv(ROOT / "results" / "baseline" / "model_residuals.csv", parse_dates=["period_start"])
    residuals = residuals.loc[residuals["model_id"].eq("single_physical_violence_raw")].copy()
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.5))
    sns.scatterplot(data=residuals, x="fitted", y="residual", hue="actor_name", palette=ACTOR_COLORS, s=22, alpha=0.65, ax=axes[0, 0], legend=False)
    axes[0, 0].axhline(0, color="#333333", lw=0.8)
    axes[0, 0].set_title("Residuals vs. fitted")
    sns.histplot(residuals["residual"], stat="density", bins=24, color="#6D8EA0", ax=axes[0, 1])
    sns.kdeplot(residuals["residual"], color="#B54343", lw=1.5, ax=axes[0, 1])
    axes[0, 1].set_title("Residual distribution")
    sns.scatterplot(data=residuals, x="observed", y="fitted", hue="actor_name", palette=ACTOR_COLORS, s=22, alpha=0.65, ax=axes[1, 0], legend=False)
    limit = max(residuals["observed"].max(), residuals["fitted"].max())
    axes[1, 0].plot([0, limit], [0, limit], color="#333333", lw=0.8)
    axes[1, 0].set_title("Observed vs. fitted")
    for actor in ACTOR_ORDER:
        actor_residuals = residuals.loc[
            residuals["actor_name"].eq(actor), ["period_start", "residual"]
        ].sort_values("period_start")
        grid = pd.date_range(
            actor_residuals["period_start"].min(),
            actor_residuals["period_start"].max(),
            freq="14D",
        )
        values = actor_residuals.set_index("period_start")["residual"].reindex(grid)
        correlations = [values.corr(values.shift(lag)) for lag in range(1, 13)]
        axes[1, 1].plot(range(1, 13), correlations, marker="o", ms=2.5, lw=1, color=ACTOR_COLORS[actor], label=actor)
    axes[1, 1].axhline(0, color="#333333", lw=0.8)
    axes[1, 1].set_xlabel("Two-week lag")
    axes[1, 1].set_ylabel("Residual autocorrelation")
    axes[1, 1].set_title("Residual autocorrelation")
    handles = [plt.Line2D([0], [0], marker="o", linestyle="", color=ACTOR_COLORS[a], label=a) for a in ACTOR_ORDER]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, "baseline", "baseline_diagnostics")

    appendix = residuals.iloc[0:0].copy()
    all_residuals = pd.read_csv(ROOT / "results" / "baseline" / "model_residuals.csv", parse_dates=["period_start"])
    appendix = all_residuals.loc[
        all_residuals["model_id"].isin(["controls_only", "joint_binary_raw"])
    ].copy()
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.4))
    for row_index, (model_id, title) in enumerate(
        [("controls_only", "Controls only"), ("joint_binary_raw", "Multivariable binary shares")]
    ):
        frame = appendix.loc[appendix["model_id"].eq(model_id)]
        sns.scatterplot(data=frame, x="fitted", y="residual", hue="actor_name", palette=ACTOR_COLORS, s=18, alpha=0.55, legend=False, ax=axes[row_index, 0])
        axes[row_index, 0].axhline(0, color="#333333", lw=0.8)
        axes[row_index, 0].set_title(f"{title}: residuals vs. fitted")
        sns.scatterplot(data=frame, x="observed", y="fitted", hue="actor_name", palette=ACTOR_COLORS, s=18, alpha=0.55, legend=False, ax=axes[row_index, 1])
        limit = max(frame["observed"].max(), frame["fitted"].max())
        axes[row_index, 1].plot([0, limit], [0, limit], color="#333333", lw=0.8)
        axes[row_index, 1].set_title(f"{title}: observed vs. fitted")
    fig.tight_layout()
    save(fig, "baseline", "baseline_diagnostics_appendix")


def calibration_figures() -> None:
    comparison = pd.read_csv(ROOT / "results" / "calibration" / "estimator_comparison.csv")
    estimators = ["naive", "two_step_calibrated", "bca", "bcm", "joint_hmc"]
    estimator_labels = {
        "naive": "Naive raw share",
        "two_step_calibrated": "Direct calibration",
        "bca": "BCA",
        "bcm": "BCM",
        "joint_hmc": "Joint beta model",
    }
    estimator_colors = dict(zip(estimators, ["#666666", "#2B6F8A", "#4D7C57", "#C05A3C", "#7A5A92"]))
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.6), sharey=True)
    positions = np.arange(len(estimators))[::-1]
    for ax, variable in zip(axes, VARIABLE_ORDER):
        frame = comparison.loc[comparison["variable"].eq(variable) & comparison["estimator"].isin(estimators)].set_index("estimator").loc[estimators]
        for position, (estimator, row) in zip(positions, frame.iterrows()):
            low = row["comparison_ci_low"] if pd.notna(row["comparison_ci_low"]) else row["ci_low"]
            high = row["comparison_ci_high"] if pd.notna(row["comparison_ci_high"]) else row["ci_high"]
            ax.errorbar(row["estimate"], position, xerr=[[row["estimate"] - low], [high - row["estimate"]]], fmt="o", color=estimator_colors[estimator], capsize=3)
        ax.axvline(0, color="#333333", lw=0.9)
        ax.set_title(VARIABLE_LABELS[variable])
        ax.set_xlabel("Coefficient")
        ax.set_yticks(positions, [estimator_labels[value] for value in estimators])
    fig.tight_layout()
    save(fig, "calibration", "calibration_estimator_comparison")

    panel = pd.read_csv(ROOT / "results" / "calibration" / "uniform_joint" / "panel_2_weeks_calibrated.csv", dtype={"actor_key": str})
    panel["display_actor"] = panel["actor_key"].map(ACTOR_NAMES)
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.1), sharex=True, sharey=True)
    for ax, variable in zip(axes, VARIABLE_ORDER):
        for actor in ACTOR_ORDER:
            frame = panel.loc[panel["display_actor"].eq(actor)]
            ax.scatter(frame[RAW_COLUMNS[variable]], frame[CALIBRATED_COLUMNS[variable]], s=15, alpha=0.5, color=ACTOR_COLORS[actor], label=actor)
        ax.plot([0, 1], [0, 1], color="#555555", lw=0.8, ls="--")
        ax.set_title(VARIABLE_LABELS[variable])
        ax.set_xlabel("Raw LLM share")
        ax.set_xlim(-0.03, 1.03)
        ax.set_ylim(-0.03, 1.03)
    axes[0].set_ylabel("Directly calibrated share")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    save(fig, "calibration", "share_calibration_comparison")

    rates = pd.read_csv(ROOT / "results" / "calibration" / "uniform_joint" / "validation_rates.csv")
    parameters = pd.read_csv(ROOT / "results" / "calibration" / "joint_beta" / "beta_parameter_summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.4), sharey=True)
    positions = np.arange(len(VARIABLE_ORDER))[::-1]
    for ax, parameter, rate, successes, failures, title in [
        (axes[0], "beta0", "false_positive_rate", "false_positive", "true_negative", "False-positive rate"),
        (axes[1], "beta1", "sensitivity", "true_positive", "false_negative", "True-positive rate"),
    ]:
        for position, variable in zip(positions, VARIABLE_ORDER):
            validation = rates.loc[rates["variable"].eq(variable)].iloc[0]
            trials = int(validation[successes] + validation[failures])
            proportion = float(validation[rate])
            z = 1.96
            denominator = 1 + z**2 / trials
            center = (proportion + z**2 / (2 * trials)) / denominator
            half = z * math.sqrt(proportion * (1 - proportion) / trials + z**2 / (4 * trials**2)) / denominator
            ax.errorbar(proportion, position + 0.10, xerr=[[proportion - (center - half)], [(center + half) - proportion]], fmt="o", color="#303030", capsize=3, label="Validation" if position == positions[0] else None)
            joint = parameters.loc[parameters["variable"].eq(variable) & parameters["parameter"].eq(parameter)].iloc[0]
            ax.errorbar(joint["mean"], position - 0.10, xerr=[[joint["mean"] - joint["ci_low"]], [joint["ci_high"] - joint["mean"]]], fmt="D", color="#B33A3A", capsize=3, label="Joint model" if position == positions[0] else None)
        ax.set_title(title)
        ax.set_xlabel("Probability")
        ax.set_xlim(0, 1)
        ax.set_yticks(positions, [VARIABLE_LABELS[value] for value in VARIABLE_ORDER])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    save(fig, "calibration", "joint_measurement_rate_comparison")


def robustness_figures() -> None:
    timing = pd.read_csv(ROOT / "results" / "robustness" / "timing" / "lead_lag_coefficients.csv")
    fig, ax = plt.subplots(figsize=(7.7, 4.4))
    colors = {"Raw LLM share": "#6B6B6B", "Directly calibrated share": "#2B6F8A"}
    offsets = {"Raw LLM share": -0.08, "Directly calibrated share": 0.08}
    for specification, frame in timing.groupby("share_specification", sort=False):
        x = frame["weeks_relative_to_predictor"] + offsets[specification]
        ax.errorbar(x, frame["estimate"], yerr=[frame["estimate"] - frame["comparison_ci_low"], frame["comparison_ci_high"] - frame["estimate"]], fmt="o-", color=colors[specification], capsize=3, label=specification)
    ax.axhline(0, color="#333333", lw=0.8)
    ax.axvline(0, color="#777777", lw=0.8, ls=":")
    ax.set_xticks([-4, -2, 0, 2, 4])
    ax.set_xlabel("Outcome timing relative to the predictor period (weeks)")
    ax.set_ylabel("Physical-violence coefficient")
    ax.set_title("The association is stronger for contemporaneous and forward outcomes", loc="left", fontweight="bold")
    ax.legend(frameon=False)
    fig.tight_layout()
    save(fig, "robustness", "physical_violence_lead_lag_profile")

    viirs_path = ROOT / "results" / "robustness" / "viirs_primary_results.csv"
    if viirs_path.exists():
        viirs = pd.read_csv(viirs_path)
        viirs = viirs.loc[viirs["variable"].ne("article_volume")].copy()
        viirs["ci_low"] = viirs["estimate"] - 1.96 * viirs["standard_error"]
        viirs["ci_high"] = viirs["estimate"] + 1.96 * viirs["standard_error"]
        viirs["label"] = viirs["variable"].map(VARIABLE_LABELS)
        fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.1), sharey=True)
        for ax, (outcome, frame) in zip(axes, viirs.groupby("outcome", sort=False)):
            frame = frame.set_index("variable").reindex(VARIABLE_ORDER).reset_index()
            positions = np.arange(len(frame))[::-1]
            ax.errorbar(frame["estimate"], positions, xerr=[frame["estimate"] - frame["ci_low"], frame["ci_high"] - frame["estimate"]], fmt="o", color="#2B6F8A", capsize=3)
            ax.axvline(0, color="#333333", lw=0.8)
            ax.set_title(outcome)
            ax.set_xlabel("Coefficient with 95% interval")
            ax.set_yticks(positions, frame["label"])
        fig.tight_layout()
        save(fig, "robustness", "viirs_llm_coefficients")

        night = pd.read_csv(ROOT / "data" / "processed" / "viirs_nightlight_panel.csv", dtype={"actor_key": str})
        night["display_actor"] = night["actor_key"].map(ACTOR_NAMES)
        fig, ax = plt.subplots(figsize=(7.8, 4.2))
        sns.boxplot(data=night, x="display_actor", y="ntl_log_mean_change", order=ACTOR_ORDER, hue="display_actor", palette=ACTOR_COLORS, legend=False, showfliers=False, ax=ax)
        sns.stripplot(data=night, x="display_actor", y="ntl_log_mean_change", order=ACTOR_ORDER, color="#283138", alpha=0.3, size=2.5, jitter=0.18, ax=ax)
        ax.axhline(0, color="#333333", lw=0.8)
        ax.set_xlabel("")
        ax.set_ylabel("Monthly change in log mean radiance")
        ax.set_title("VIIRS nighttime-radiance changes by actor", loc="left", fontweight="bold")
        fig.tight_layout()
        save(fig, "robustness", "viirs_log_mean_change")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--groups",
        nargs="+",
        choices=["descriptive", "baseline", "calibration", "robustness"],
        default=["descriptive", "baseline", "calibration", "robustness"],
    )
    args = parser.parse_args()
    set_style()
    functions = {
        "descriptive": descriptive_figures,
        "baseline": baseline_figures,
        "calibration": calibration_figures,
        "robustness": robustness_figures,
    }
    for group in args.groups:
        functions[group]()
        print(f"Generated {group} figures")


if __name__ == "__main__":
    main()
