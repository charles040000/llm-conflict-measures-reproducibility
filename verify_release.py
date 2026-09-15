#!/usr/bin/env python3
"""Validate the released data structure and headline sample invariants."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
PRIMARY = {
    "conflict_relevance": ("direct", 0.8902439024, 0.2337662338),
    "physical_violence_occurred": ("yes", 0.8863636364, 0.1913043478),
    "fatalities_present": ("yes", 0.9210526316, 0.1487603306),
}
ARTICLE_COUNTS = {
    "209": 88_554,
    "303": 23_641,
    "group:farc_bloc": 2_947,
}
PHYSICAL_VIOLENCE_ESTIMATES = {
    "naive": 1.0296796043,
    "two_step_calibrated": 0.7304621792,
    "bca": 0.8284028731,
    "bcm": 0.8435681753,
    "joint_hmc": 0.8229780903,
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    validation_dir = ROOT / "data" / "validation"
    source_csvs = sorted(validation_dir.glob("*.csv"))
    require(len(source_csvs) == 2, "data/validation must contain exactly two CSV files")

    manual = pd.read_csv(validation_dir / "manual_validation.csv", dtype={"GlobalEventID": str})
    inter = pd.read_csv(validation_dir / "inter_annotator.csv", dtype={"GlobalEventID": str})
    require(len(manual) == 173, "manual validation file must contain 173 rows")
    require(manual["GlobalEventID"].is_unique, "manual validation IDs must be unique")
    require(len(inter) == 50, "inter-annotator file must contain 50 rows")
    require(inter["GlobalEventID"].is_unique, "inter-annotator IDs must be unique")

    calibration = manual.loc[manual["in_three_actor_calibration"].astype(bool)].copy()
    require(len(calibration) == 159, "pooled calibration sample must contain 159 rows")
    expected_actors = {"Hamas": 78, "Taliban": 52, "FARC family": 29}
    require(
        calibration["analysis_actor_name"].value_counts().to_dict() == expected_actors,
        "calibration actor counts changed",
    )
    for field, (positive, expected_tpr, expected_fpr) in PRIMARY.items():
        actual = calibration[f"{field}_manual"].astype(str).str.strip().str.lower().eq(positive)
        predicted = calibration[f"{field}_llm"].astype(str).str.strip().str.lower().eq(positive)
        require(calibration[f"{field}_manual"].notna().all(), f"missing manual values for {field}")
        tpr = predicted[actual].mean()
        fpr = predicted[~actual].mean()
        require(np.isclose(tpr, expected_tpr), f"TPR changed for {field}")
        require(np.isclose(fpr, expected_fpr), f"FPR changed for {field}")

    panel = pd.read_csv(ROOT / "data" / "processed" / "actor_period_panel.csv")
    required = [
        "log1p_y_ucdp_actor_fatalities_next",
        "log1p_y_ucdp_actor_fatalities",
        "log1p_y_ucdp_actor_fatalities_lag1",
        "log1p_y_ucdp_actor_fatalities_lag2",
    ]
    sample = panel.loc[panel["n_articles"].ge(20)].dropna(subset=required)
    require(len(sample) == 360, "baseline sample must contain 360 observations")
    expected_periods = {"Hamas": 144, "Taliban": 152, "FARC family": 64}
    require(sample["actor_name"].value_counts().to_dict() == expected_periods, "baseline actor counts changed")

    articles_path = ROOT / "data" / "processed" / "article_labels_compact.csv"
    article_header = pd.read_csv(articles_path, nrows=1)
    forbidden = {"text", "text_cleaned", "text_en", "source_url", "url"}
    require(
        not forbidden.intersection(article_header.columns),
        "compact article file contains source text or URLs",
    )
    articles = pd.read_csv(
        articles_path,
        usecols=["GlobalEventID", "actor_key"],
        dtype={"GlobalEventID": str, "actor_key": str},
    )
    require(len(articles) == 115_142, "three-actor article count changed")
    require(articles["GlobalEventID"].is_unique, "compact article IDs must be unique")
    require(
        articles["actor_key"].value_counts().to_dict() == ARTICLE_COUNTS,
        "per-actor article counts changed",
    )

    broad_panel = pd.read_csv(
        ROOT / "data" / "processed" / "actor_month_panel.csv",
        usecols=["n_articles"],
    )
    require(
        np.isclose(broad_panel["n_articles"].sum(), 116_900),
        "final single-actor analytical corpus count changed",
    )

    comparison = pd.read_csv(ROOT / "results" / "calibration" / "estimator_comparison.csv")
    physical = comparison.loc[comparison["variable"].eq("physical_violence")].set_index("estimator")
    for estimator, expected in PHYSICAL_VIOLENCE_ESTIMATES.items():
        require(estimator in physical.index, f"missing physical-violence estimator: {estimator}")
        require(
            np.isclose(float(physical.loc[estimator, "estimate"]), expected),
            f"physical-violence estimate changed for {estimator}",
        )

    robustness = pd.read_csv(ROOT / "results" / "robustness" / "robustness_model_results.csv")
    robustness_baseline = robustness.loc[
        robustness["group"].eq("Article threshold")
        & robustness["minimum_articles"].eq(20)
        & robustness["variable"].eq("physical_violence")
    ].set_index("estimator")
    robustness_names = {
        "naive": "naive",
        "direct": "two_step_calibrated",
        "bca": "bca",
        "bcm": "bcm",
    }
    for robustness_name, comparison_name in robustness_names.items():
        require(robustness_name in robustness_baseline.index, f"missing robustness estimator: {robustness_name}")
        require(
            np.isclose(
                float(robustness_baseline.loc[robustness_name, "estimate"]),
                float(physical.loc[comparison_name, "estimate"]),
            ),
            f"baseline robustness estimate disagrees for {robustness_name}",
        )

    text_extensions = {".py", ".md", ".json", ".txt", ".env", ".csv"}
    private_home_prefix = "/" + "Users/"
    for path in ROOT.rglob("*"):
        if ".cache" in path.parts or "__pycache__" in path.parts or ".mplconfig" in path.parts:
            continue
        if not path.is_file() or path.suffix.lower() not in text_extensions:
            continue
        content = path.read_text(encoding="utf-8", errors="ignore")
        require(private_home_prefix not in content, f"private absolute path found in {path.relative_to(ROOT)}")

    print("Release verification passed")
    print("  manual validation: 173 rows (159 in pooled calibration sample)")
    print("  inter-annotator data: 50 rows")
    print("  baseline panel: 360 observations (144 Hamas, 152 Taliban, 64 FARC family)")
    print("  final analytical corpus: 116,900 single-actor articles")
    print("  three focal actors: 115,142 articles (88,554 Hamas, 23,641 Taliban, 2,947 FARC)")
    print("  baseline calibration and robustness coefficients agree")
    print("  compact article file: no article text or source URLs")
    print("  source and output metadata: no private absolute paths")


if __name__ == "__main__":
    main()
