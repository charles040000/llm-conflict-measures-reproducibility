#!/usr/bin/env python3
"""Combine two-step and preferred joint-model results in one tidy file."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CALIBRATION = ROOT / "results" / "calibration"
VARIABLE_ORDER = [
    "direct_conflict_relevance",
    "physical_violence",
    "fatalities_present",
]


def normal_p_value(estimate: float, standard_error: float) -> float:
    return math.erfc(abs(estimate / standard_error) / math.sqrt(2.0))


def bh_adjust(values: pd.Series) -> pd.Series:
    p = values.to_numpy(float)
    order = np.argsort(p)
    adjusted = p[order] * len(p) / np.arange(1, len(p) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.minimum(adjusted, 1.0)
    return pd.Series(result, index=values.index)


def main() -> None:
    uniform = CALIBRATION / "uniform_joint"
    beta = CALIBRATION / "joint_beta"
    point = pd.read_csv(uniform / "estimator_comparison_point.csv")
    bootstrap = pd.read_csv(uniform / "validation_bootstrap_summary.csv")
    draws = pd.read_csv(beta / "beta_global_posterior_draws.csv")

    result = point.copy()
    result["comparison_se"] = result["standard_error"]
    result["comparison_p"] = result["p_value"]
    result["comparison_ci_low"] = result["ci_low"]
    result["comparison_ci_high"] = result["ci_high"]
    result["validation_se"] = np.nan

    for index, row in result.iterrows():
        if row["estimator"] not in {"two_step_calibrated", "bca", "bcm"}:
            continue
        match = bootstrap.loc[
            bootstrap["variable"].eq(row["variable"])
            & bootstrap["estimator"].eq(row["estimator"])
        ]
        validation_se = float(match["validation_se"].iloc[0])
        combined_se = math.sqrt(float(row["standard_error"]) ** 2 + validation_se**2)
        result.loc[index, "validation_se"] = validation_se
        result.loc[index, "comparison_se"] = combined_se
        result.loc[index, "comparison_p"] = normal_p_value(float(row["estimate"]), combined_se)
        result.loc[index, "comparison_ci_low"] = float(row["estimate"]) - 1.96 * combined_se
        result.loc[index, "comparison_ci_high"] = float(row["estimate"]) + 1.96 * combined_se

    result = result.loc[~result["estimator"].eq("joint_hmc")].copy()
    joint_rows = []
    for variable in VARIABLE_ORDER:
        gamma = draws.loc[draws["variable"].eq(variable), "gamma"].to_numpy(float)
        posterior_p = min(1.0, 2 * min(float(np.mean(gamma > 0)), float(np.mean(gamma < 0))))
        joint_rows.append(
            {
                "variable": variable,
                "estimator": "joint_hmc",
                "estimate": float(gamma.mean()),
                "standard_error": float(gamma.std(ddof=1)),
                "uncertainty": "posterior standard deviation",
                "p_value": posterior_p,
                "ci_low": float(np.quantile(gamma, 0.025)),
                "ci_high": float(np.quantile(gamma, 0.975)),
                "n": 360,
                "r2": np.nan,
                "rmse": np.nan,
                "fdr_p_within_estimator": np.nan,
                "comparison_se": float(gamma.std(ddof=1)),
                "comparison_p": posterior_p,
                "comparison_ci_low": float(np.quantile(gamma, 0.025)),
                "comparison_ci_high": float(np.quantile(gamma, 0.975)),
                "validation_se": np.nan,
            }
        )
    result = pd.concat([result, pd.DataFrame(joint_rows)], ignore_index=True)
    result["comparison_fdr"] = np.nan
    for _, block in result.groupby("estimator", sort=False):
        result.loc[block.index, "comparison_fdr"] = bh_adjust(block["comparison_p"])
    result.to_csv(CALIBRATION / "estimator_comparison.csv", index=False)
    print(result[["variable", "estimator", "estimate", "comparison_se", "comparison_p"]].to_string(index=False))


if __name__ == "__main__":
    main()
