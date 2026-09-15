#!/usr/bin/env python3
"""Compare Taliban news measures before and after the August 2021 takeover."""

from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import norm


ROOT = Path(__file__).resolve().parents[1]
ARTICLE_LABELS = ROOT / "data" / "processed" / "article_labels_compact.csv"
MODEL_PANEL = ROOT / "data" / "processed" / "monthly_model_panel.csv"
OUTPUT = ROOT / "results" / "descriptive" / "taliban"
FIGURES = ROOT / "figures" / "descriptive"
TABLES = OUTPUT
MPL_CACHE = ROOT / ".cache" / "matplotlib"
for directory in (OUTPUT, FIGURES, TABLES, MPL_CACHE):
    directory.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))

import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

TAKEOVER_DATE = pd.Timestamp("2021-08-15")
TRANSITION_MONTH = "2021-08"
OUTCOME = "y_ucdp_actor_fatalities_next"

INDICATORS = {
    "Direct conflict relevance": ("conflict_relevance", "direct", {"direct", "background_or_context", "irrelevant"}),
    "Combat context": ("event_context", "cbt", {"cbt", "pol", "civ", "org", "opr", "crm", "pro"}),
    "Realized event": ("event_modality", "realized", {"realized", "threatened", "future_or_planned"}),
    "Actor as perpetrator": ("matched_actor_role", "perpetrator", {"perpetrator", "target", "both", "mentioned_only"}),
    "Current or recent event": (
        "event_time_relation",
        "current_or_recent",
        {"current_or_recent", "future_or_planned", "historical_background"},
    ),
    "Physical violence occurred": ("physical_violence_occurred", "yes", {"yes", "no"}),
    "Main-story fatalities": ("fatalities_present", "yes", {"yes", "no"}),
    "Main-story injuries": ("injuries_present", "yes", {"yes", "no"}),
    "Actor mentioned anywhere": ("any_matched_actor_mentioned", "yes", {"yes", "no"}),
    "Violence mentioned anywhere": ("any_physical_violence_mentioned", "yes", {"yes", "no"}),
    "Fatalities mentioned anywhere": ("any_fatalities_mentioned", "yes", {"yes", "no"}),
    "Injuries mentioned anywhere": ("any_injuries_mentioned", "yes", {"yes", "no"}),
    "Weapon mentioned anywhere": ("any_weapon_mentioned", "yes", {"yes", "no"}),
}

CATEGORICAL_VARIABLES = [
    "conflict_relevance",
    "event_context",
    "event_modality",
    "matched_actor_role",
    "event_time_relation",
]


def normalize(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().str.lower()


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    articles = pd.read_csv(ARTICLE_LABELS, dtype={"GlobalEventID": str}, low_memory=False)
    articles = articles.loc[articles["actor_key"].astype(str).eq("303")].copy()
    articles["article_date"] = pd.to_datetime(articles["date"], errors="coerce")
    if articles["article_date"].isna().any():
        raise ValueError(f"Missing dates for {articles['article_date'].isna().sum()} Taliban articles")
    articles["regime"] = np.where(articles["article_date"].lt(TAKEOVER_DATE), "Pre-takeover", "Post-takeover")
    for _, (column, _, _) in INDICATORS.items():
        articles[column] = normalize(articles[column])

    panel = pd.read_csv(MODEL_PANEL, dtype={"actor_key": str}, low_memory=False)
    panel = panel.loc[panel["actor_key"].eq("303")].copy()
    panel["period_start"] = pd.to_datetime(panel["period_start"], errors="coerce")
    panel["next_month_outcome_available"] = panel[OUTCOME].notna()
    panel["month"] = panel["month"].astype(str)
    panel["regime"] = np.select(
        [panel["month"].lt(TRANSITION_MONTH), panel["month"].gt(TRANSITION_MONTH)],
        ["Pre-takeover", "Post-takeover"],
        default="Transition month",
    )
    return articles, panel


def article_indicator_summary(articles: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for label, (column, positive, clear_values) in INDICATORS.items():
        for regime, group in articles.groupby("regime", sort=False):
            values = group[column]
            clear = values.isin(clear_values)
            rows.append(
                {
                    "label": label,
                    "variable": column,
                    "positive_value": positive,
                    "regime": regime,
                    "n_articles": int(len(group)),
                    "positive_articles": int(values.eq(positive).sum()),
                    "positive_share_all_articles": float(values.eq(positive).mean()),
                    "clear_articles": int(clear.sum()),
                    "positive_share_clear_cases": float(values[clear].eq(positive).mean()) if clear.any() else np.nan,
                    "unresolved_share": float((~clear).mean()),
                }
            )
    result = pd.DataFrame(rows)
    wide = result.pivot(index=["label", "variable", "positive_value"], columns="regime")
    wide.columns = [f"{metric}_{regime.lower().replace('-', '_')}" for metric, regime in wide.columns]
    wide = wide.reset_index()
    for metric in ["positive_share_all_articles", "positive_share_clear_cases", "unresolved_share"]:
        pre = f"{metric}_pre_takeover"
        post = f"{metric}_post_takeover"
        wide[f"{metric}_change_pp"] = 100 * (wide[post] - wide[pre])
    return wide.sort_values("label")


def categorical_summary(articles: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    long_rows: list[dict[str, object]] = []
    divergence_rows: list[dict[str, object]] = []
    for variable in CATEGORICAL_VARIABLES:
        categories = sorted(set(articles[variable].unique()) - {""})
        distributions: dict[str, np.ndarray] = {}
        for regime in ["Pre-takeover", "Post-takeover"]:
            values = articles.loc[articles["regime"].eq(regime), variable]
            shares = values.value_counts(normalize=True).reindex(categories, fill_value=0.0)
            distributions[regime] = shares.to_numpy(float)
            for category, share in shares.items():
                long_rows.append(
                    {
                        "variable": variable,
                        "category": category,
                        "regime": regime,
                        "share": float(share),
                        "n_articles": int(values.size),
                    }
                )
        pre = distributions["Pre-takeover"]
        post = distributions["Post-takeover"]
        divergence_rows.append(
            {
                "variable": variable,
                "jensen_shannon_distance": float(jensenshannon(pre, post, base=2)),
                "largest_category_change_pp": float(100 * np.max(np.abs(post - pre))),
            }
        )
    return pd.DataFrame(long_rows), pd.DataFrame(divergence_rows).sort_values(
        "jensen_shannon_distance", ascending=False
    )


def build_monthly(articles: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for month, group in articles.groupby("month", sort=True):
        row: dict[str, object] = {"month": str(month), "n_articles": int(len(group))}
        for label, (column, positive, clear_values) in INDICATORS.items():
            key = column if positive == "yes" else f"{column}_{positive}"
            values = group[column]
            clear = values.isin(clear_values)
            row[f"share_{key}_all"] = float(values.eq(positive).mean())
            row[f"share_{key}_clear"] = float(values[clear].eq(positive).mean()) if clear.any() else np.nan
            row[f"share_{key}_unresolved"] = float((~clear).mean())
        rows.append(row)
    monthly = pd.DataFrame(rows)
    monthly = panel[["month", "period_start", "regime", OUTCOME, "next_month_outcome_available"]].merge(
        monthly,
        on="month",
        how="left",
        validate="one_to_one",
    )
    monthly["n_articles"] = monthly["n_articles"].fillna(0)
    monthly["log1p_n_articles"] = np.log1p(monthly["n_articles"])
    monthly["log1p_actor_linked_fatalities_next"] = np.log1p(monthly[OUTCOME])
    return monthly


def hac_difference(data: pd.DataFrame, metric: str, adjusted: bool) -> tuple[float, float, float]:
    frame = data.loc[data["regime"].isin(["Pre-takeover", "Post-takeover"]), [metric, "regime", "period_start"]].dropna()
    frame = frame.sort_values("period_start")
    if len(frame) < 10:
        return np.nan, np.nan, np.nan
    post = frame["regime"].eq("Post-takeover").astype(float).to_numpy()
    columns = [np.ones(len(frame)), post]
    if adjusted:
        trend = np.arange(len(frame), dtype=float)
        month = frame["period_start"].dt.month.to_numpy(float)
        angle = 2 * math.pi * month / 12.0
        columns.extend([trend, np.sin(angle), np.cos(angle)])
    x = np.column_stack(columns)
    y = frame[metric].to_numpy(float)
    xtx_inv = np.linalg.pinv(x.T @ x)
    beta = xtx_inv @ x.T @ y
    residuals = y - x @ beta
    scores = x * residuals[:, None]
    meat = scores.T @ scores
    max_lag = min(3, len(frame) - 1)
    for lag in range(1, max_lag + 1):
        weight = 1.0 - lag / (max_lag + 1.0)
        gamma = scores[lag:].T @ scores[:-lag]
        meat += weight * (gamma + gamma.T)
    covariance = xtx_inv @ meat @ xtx_inv
    covariance *= len(frame) / max(len(frame) - x.shape[1], 1)
    standard_error = float(np.sqrt(max(covariance[1, 1], 0.0)))
    z_value = float(beta[1] / standard_error) if standard_error > 0 else np.nan
    p_value = float(2 * norm.sf(abs(z_value))) if np.isfinite(z_value) else np.nan
    return float(beta[1]), standard_error, p_value


def monthly_summary(monthly: pd.DataFrame) -> pd.DataFrame:
    indicator_metrics = [
        col for col in monthly.columns if col.startswith("share_") and col.endswith("_all")
    ]
    metrics = ["log1p_n_articles", "log1p_actor_linked_fatalities_next", *indicator_metrics]
    rows: list[dict[str, object]] = []
    data = monthly.loc[monthly["regime"].isin(["Pre-takeover", "Post-takeover"])].copy()
    for metric in metrics:
        pre = pd.to_numeric(data.loc[data["regime"].eq("Pre-takeover"), metric], errors="coerce").dropna()
        post = pd.to_numeric(data.loc[data["regime"].eq("Post-takeover"), metric], errors="coerce").dropna()
        raw_coef, raw_se, raw_p = hac_difference(data, metric, adjusted=False)
        adj_coef, adj_se, adj_p = hac_difference(data, metric, adjusted=True)
        rows.append(
            {
                "metric": metric,
                "pre_months": int(len(pre)),
                "post_months": int(len(post)),
                "pre_mean": float(pre.mean()),
                "post_mean": float(post.mean()),
                "mean_change": float(post.mean() - pre.mean()),
                "pre_median": float(pre.median()),
                "post_median": float(post.median()),
                "hac_change": raw_coef,
                "hac_se": raw_se,
                "hac_p": raw_p,
                "trend_season_adjusted_hac_change": adj_coef,
                "trend_season_adjusted_hac_se": adj_se,
                "trend_season_adjusted_hac_p": adj_p,
            }
        )
    return pd.DataFrame(rows)


def save_figures(monthly: pd.DataFrame, indicators: pd.DataFrame, categorical: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid", context="talk")
    colors = {"Pre-takeover": "#4C78A8", "Post-takeover": "#C45A3C"}

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(monthly["period_start"], monthly["n_articles"], color="#343A40", linewidth=2)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Articles (log scale)")
    axes[0].set_title("Monthly Taliban article volume", loc="left", fontweight="bold")
    axes[1].plot(monthly["period_start"], monthly[OUTCOME], color="#A3422C", linewidth=2)
    axes[1].set_yscale("symlog", linthresh=1)
    axes[1].set_ylabel("Next-month fatalities")
    axes[1].set_title("UCDP fatalities in events involving the Taliban or Taliban-led government", loc="left", fontweight="bold")
    for ax in axes:
        ax.axvline(TAKEOVER_DATE, color="#111111", linestyle="--", linewidth=1.5)
        ax.text(TAKEOVER_DATE, 0.96, " Takeover", transform=ax.get_xaxis_transform(), va="top", fontsize=10)
        ax.grid(axis="x", visible=False)
    axes[1].xaxis.set_major_locator(mdates.YearLocator())
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[1].set_xlabel("Predictor month")
    fig.tight_layout()
    fig.savefig(FIGURES / "taliban_volume_and_outcome_pre_post.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "taliban_volume_and_outcome_pre_post.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    plot = indicators[[
        "label",
        "positive_share_all_articles_pre_takeover",
        "positive_share_all_articles_post_takeover",
    ]].copy()
    plot = plot.melt("label", var_name="regime", value_name="share")
    plot["regime"] = plot["regime"].map(
        {
            "positive_share_all_articles_pre_takeover": "Pre-takeover",
            "positive_share_all_articles_post_takeover": "Post-takeover",
        }
    )
    order = indicators.sort_values("positive_share_all_articles_change_pp")["label"].tolist()
    fig, ax = plt.subplots(figsize=(11, 8))
    sns.pointplot(
        data=plot,
        y="label",
        x="share",
        hue="regime",
        order=order,
        palette=colors,
        dodge=0.35,
        markers="o",
        linestyles="none",
        ax=ax,
    )
    ax.set_xlabel("Positive share of all articles")
    ax.set_ylabel("")
    ax.set_xlim(0, 1)
    ax.set_title("Taliban LLM-label shares before and after the takeover", loc="left", fontweight="bold")
    ax.legend(title="")
    fig.tight_layout()
    fig.savefig(FIGURES / "taliban_indicator_shares_pre_post.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "taliban_indicator_shares_pre_post.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    key_variables = ["conflict_relevance", "event_context", "matched_actor_role"]
    fig, axes = plt.subplots(len(key_variables), 1, figsize=(12, 12))
    for ax, variable in zip(axes, key_variables):
        frame = categorical.loc[categorical["variable"].eq(variable)].copy()
        order = (
            frame.groupby("category")["share"].max().sort_values(ascending=False).index.tolist()
        )
        sns.barplot(data=frame, x="category", y="share", hue="regime", order=order, palette=colors, ax=ax)
        ax.set_title(variable.replace("_", " ").title(), loc="left", fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("Article share")
        ax.tick_params(axis="x", rotation=25, labelsize=9)
        ax.legend(title="", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES / "taliban_multiclass_distributions_pre_post.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "taliban_multiclass_distributions_pre_post.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_summary(
    articles: pd.DataFrame,
    panel: pd.DataFrame,
    indicators: pd.DataFrame,
    divergence: pd.DataFrame,
    monthly_stats: pd.DataFrame,
) -> None:
    counts = articles["regime"].value_counts()
    months = panel[panel["regime"].isin(["Pre-takeover", "Post-takeover"])]["regime"].value_counts()
    largest = indicators.reindex(indicators["positive_share_all_articles_change_pp"].abs().sort_values(ascending=False).index).head(8)
    volume = monthly_stats.loc[monthly_stats["metric"].eq("log1p_n_articles")].iloc[0]
    outcome = monthly_stats.loc[monthly_stats["metric"].eq("log1p_actor_linked_fatalities_next")].iloc[0]
    lines = [
        "# Taliban pre/post-takeover diagnostic",
        "",
        f"- Exact article cutoff: {TAKEOVER_DATE.date()}",
        f"- Pre-takeover articles: {int(counts.get('Pre-takeover', 0)):,}",
        f"- Post-takeover articles: {int(counts.get('Post-takeover', 0)):,}",
        f"- Monthly comparison: {int(months.get('Pre-takeover', 0))} pre months and {int(months.get('Post-takeover', 0))} post months; August 2021 excluded.",
        "",
        "## Monthly level shifts",
        "",
        f"- log(1 + articles): pre mean {volume.pre_mean:.3f}, post mean {volume.post_mean:.3f}; HAC difference {volume.hac_change:.3f} (p={volume.hac_p:.4g}); trend/season-adjusted difference {volume.trend_season_adjusted_hac_change:.3f} (p={volume.trend_season_adjusted_hac_p:.4g}).",
        f"- log(1 + next-month fatalities): pre mean {outcome.pre_mean:.3f}, post mean {outcome.post_mean:.3f}; HAC difference {outcome.hac_change:.3f} (p={outcome.hac_p:.4g}); trend/season-adjusted difference {outcome.trend_season_adjusted_hac_change:.3f} (p={outcome.trend_season_adjusted_hac_p:.4g}).",
        "",
        "## Largest article-level positive-share changes",
        "",
    ]
    for row in largest.itertuples(index=False):
        lines.append(
            f"- {row.label}: {100 * row.positive_share_all_articles_pre_takeover:.1f}% to "
            f"{100 * row.positive_share_all_articles_post_takeover:.1f}% "
            f"({row.positive_share_all_articles_change_pp:+.1f} percentage points)."
        )
    lines.extend(["", "## Multiclass distribution shifts", ""])
    for row in divergence.itertuples(index=False):
        lines.append(
            f"- {row.variable}: Jensen-Shannon distance {row.jensen_shannon_distance:.3f}; "
            f"largest category change {row.largest_category_change_pp:.1f} percentage points."
        )
    lines.extend(
        [
            "",
            "HAC p-values describe a known pre/post level break and are not causal tests. The adjusted specification includes a linear trend and annual sine/cosine terms. Article-level significance tests are intentionally omitted because the very large article count would make substantively small changes appear highly significant.",
        ]
    )
    (OUTPUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    articles, panel = load_data()
    indicators = article_indicator_summary(articles)
    categorical, divergence = categorical_summary(articles)
    monthly = build_monthly(articles, panel)
    monthly_stats = monthly_summary(monthly)
    monthly_stats.insert(0, "sample", "all_nontransition_months")
    dense_monthly_stats = monthly_summary(monthly.loc[monthly["n_articles"].ge(40)].copy())
    dense_monthly_stats.insert(0, "sample", "months_with_at_least_40_articles")
    monthly_stats = pd.concat([monthly_stats, dense_monthly_stats], ignore_index=True)

    indicators.to_csv(TABLES / "taliban_article_indicator_pre_post.csv", index=False)
    categorical.to_csv(TABLES / "taliban_multiclass_category_pre_post.csv", index=False)
    divergence.to_csv(TABLES / "taliban_multiclass_divergence.csv", index=False)
    monthly.to_csv(TABLES / "taliban_monthly_series.csv", index=False)
    monthly_stats.to_csv(TABLES / "taliban_monthly_pre_post_tests.csv", index=False)
    save_figures(monthly, indicators, categorical)
    write_summary(
        articles,
        panel,
        indicators,
        divergence,
        monthly_stats.loc[monthly_stats["sample"].eq("all_nontransition_months")].copy(),
    )
    print(f"Wrote {OUTPUT}")
    print((OUTPUT / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
