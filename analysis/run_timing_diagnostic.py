#!/usr/bin/env python3
"""Estimate lead-lag profiles for the physical-violence article share."""

from __future__ import annotations

import argparse
import json
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

import model_utils as models  # noqa: E402


DEFAULT_PANEL = ROOT / "results" / "calibration" / "uniform_joint" / "panel_2_weeks_calibrated.csv"
DEFAULT_VALIDATION = ROOT / "data" / "validation" / "manual_validation.csv"
DEFAULT_OUTPUT = ROOT / "results" / "robustness" / "timing"
CURRENT_OUTCOME = "log1p_y_ucdp_actor_fatalities"
SIGNALS = {
    "Raw LLM share": "share_physical_violence_raw",
    "Directly calibrated share": "share_physical_violence_calibrated",
}
LEAN_CONTROLS = [
    "log1p_n_articles",
    "time_years",
    "taliban_outcome_transition_aug2021",
    "taliban_outcome_post_takeover",
    "hamas_outcome_post_oct2023",
]
HORIZONS = range(-2, 3)
MANUAL_FIELD = "physical_violence_occurred_manual"
LLM_FIELD = "physical_violence_occurred_llm"


def resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else ROOT / value


def add_horizon_outcomes(panel: pd.DataFrame) -> pd.DataFrame:
    result = panel.copy()
    lookup = {
        (str(actor), int(period)): float(outcome)
        for actor, period, outcome in result[
            ["actor_key", "period_index", CURRENT_OUTCOME]
        ].itertuples(index=False, name=None)
        if pd.notna(outcome)
    }
    for horizon in HORIZONS:
        result[f"outcome_h{horizon:+d}"] = [
            lookup.get((str(actor), int(period) + horizon), np.nan)
            for actor, period in result[["actor_key", "period_index"]].itertuples(
                index=False, name=None
            )
        ]
    return result


def design_matrix(frame: pd.DataFrame, signal: str) -> tuple[np.ndarray, list[str]]:
    columns: list[tuple[str, np.ndarray]] = [
        (signal, frame[signal].to_numpy(float)),
        *((name, frame[name].to_numpy(float)) for name in LEAN_CONTROLS),
        ("Intercept", np.ones(len(frame))),
        ("actor_Taliban", frame["actor_key"].eq("303").to_numpy(float)),
        (
            "actor_FARC_family",
            frame["actor_key"].eq("group:farc_bloc").to_numpy(float),
        ),
    ]
    names = [name for name, _ in columns]
    matrix = np.column_stack([values for _, values in columns])
    keep = np.nanstd(matrix, axis=0) > 1e-12
    keep[names.index("Intercept")] = True
    return matrix[:, keep], [name for name, use in zip(names, keep) if use]


def fit(
    frame: pd.DataFrame,
    signal: str,
    outcome: np.ndarray,
    hac_lags: int,
) -> dict[str, float]:
    x, names = design_matrix(frame, signal)
    beta = np.linalg.pinv(x.T @ x) @ x.T @ outcome
    residuals = outcome - x @ beta
    covariance = models.panel_hac(
        x,
        residuals,
        frame["actor_key"].to_numpy(),
        frame["period_index"].to_numpy(int),
        hac_lags,
    )
    position = names.index(signal)
    estimate = float(beta[position])
    standard_error = float(math.sqrt(max(covariance[position, position], 0)))
    statistic = estimate / standard_error if standard_error > 0 else np.nan
    return {
        "estimate": estimate,
        "hac_se": standard_error,
        "p_value": models.normal_p_value(statistic),
        "ci_low": estimate - 1.96 * standard_error,
        "ci_high": estimate + 1.96 * standard_error,
    }


def physical_violence_rates(validation: pd.DataFrame) -> tuple[float, float]:
    manual = models.norm(validation[MANUAL_FIELD])
    llm = models.norm(validation[LLM_FIELD])
    comparable = manual.ne("") & llm.ne("")
    manual_positive = manual.loc[comparable].eq("yes")
    llm_positive = llm.loc[comparable].eq("yes")
    if not manual_positive.any() or manual_positive.all():
        raise ValueError("Bootstrap draw does not contain both manual classes")
    sensitivity = float(llm_positive.loc[manual_positive].mean())
    false_positive_rate = float(llm_positive.loc[~manual_positive].mean())
    if sensitivity <= false_positive_rate:
        raise ValueError("Bootstrap draw has non-positive classifier separation")
    return sensitivity, false_positive_rate


def validation_bootstrap(
    sample: pd.DataFrame,
    validation: pd.DataFrame,
    repetitions: int,
    seed: int,
    hac_lags: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    horizon_rows: list[dict[str, float | int]] = []
    contrast_rows: list[dict[str, float | int]] = []
    for draw in range(repetitions):
        resampled = validation.iloc[
            rng.integers(0, len(validation), size=len(validation))
        ]
        try:
            sensitivity, false_positive_rate = physical_violence_rates(resampled)
        except ValueError:
            continue
        frame = sample.copy()
        frame[SIGNALS["Directly calibrated share"]] = (
            (frame[SIGNALS["Raw LLM share"]] - false_positive_rate)
            / (sensitivity - false_positive_rate)
        ).clip(0, 1)
        for horizon in HORIZONS:
            result = fit(
                frame,
                SIGNALS["Directly calibrated share"],
                frame[f"outcome_h{horizon:+d}"].to_numpy(float),
                hac_lags,
            )
            horizon_rows.append(
                {
                    "draw": draw,
                    "horizon": horizon,
                    "estimate": result["estimate"],
                }
            )
        difference = (
            frame["outcome_h+1"].to_numpy(float)
            - frame["outcome_h-1"].to_numpy(float)
        )
        result = fit(
            frame,
            SIGNALS["Directly calibrated share"],
            difference,
            hac_lags,
        )
        contrast_rows.append(
            {"draw": draw, "estimate": result["estimate"]}
        )
    return pd.DataFrame(horizon_rows), pd.DataFrame(contrast_rows)


def make_plot(results: pd.DataFrame, output: Path) -> None:
    colors = {
        "Raw LLM share": "#2A6F97",
        "Directly calibrated share": "#B84A3A",
    }
    figure, axis = plt.subplots(figsize=(8.2, 4.8))
    offsets = {"Raw LLM share": -0.06, "Directly calibrated share": 0.06}
    for specification, group in results.groupby("share_specification", sort=False):
        group = group.sort_values("horizon")
        x = group["horizon"].to_numpy(float) + offsets[specification]
        estimate = group["estimate"].to_numpy(float)
        errors = np.vstack(
            [
                estimate - group["comparison_ci_low"].to_numpy(float),
                group["comparison_ci_high"].to_numpy(float) - estimate,
            ]
        )
        axis.errorbar(
            x,
            estimate,
            yerr=errors,
            marker="o",
            capsize=3,
            linewidth=1.6,
            color=colors[specification],
            label=specification,
        )
    axis.axhline(0, color="#666666", linewidth=1)
    axis.axvline(0, color="#AAAAAA", linewidth=1, linestyle="--")
    axis.set_xticks(list(HORIZONS))
    axis.set_xticklabels(["-2", "-1", "0", "+1", "+2"])
    axis.set_xlabel("Outcome horizon in two-week periods")
    axis.set_ylabel("Physical-violence share coefficient")
    axis.set_title("Lead-lag profile of the physical-violence association")
    axis.legend(frameon=False)
    axis.grid(axis="y", color="#E5E5E5", linewidth=0.8)
    figure.tight_layout()
    figure.savefig(output / "physical_violence_lead_lag_profile.pdf", bbox_inches="tight")
    figure.savefig(
        output / "physical_violence_lead_lag_profile.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--validation", type=Path, default=DEFAULT_VALIDATION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--minimum-articles", type=int, default=20)
    parser.add_argument("--hac-lags", type=int, default=6)
    parser.add_argument("--bootstrap-repetitions", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260911)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = resolve(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    panel = pd.read_csv(resolve(args.panel), dtype={"actor_key": str})
    validation = pd.read_csv(resolve(args.validation))
    panel["period_start"] = pd.to_datetime(panel["period_start"])
    panel = add_horizon_outcomes(panel)
    required = [
        *SIGNALS.values(),
        *LEAN_CONTROLS,
        *(f"outcome_h{horizon:+d}" for horizon in HORIZONS),
    ]
    for column in required:
        panel[column] = pd.to_numeric(panel[column], errors="coerce")
    sample = panel.loc[panel["n_articles"].ge(args.minimum_articles)].dropna(
        subset=required
    )
    sample = sample.sort_values(["actor_key", "period_index"]).copy()

    rows: list[dict[str, object]] = []
    contrasts: list[dict[str, object]] = []
    for specification, signal in SIGNALS.items():
        for horizon in HORIZONS:
            outcome_column = f"outcome_h{horizon:+d}"
            result = fit(
                sample,
                signal,
                sample[outcome_column].to_numpy(float),
                args.hac_lags,
            )
            rows.append(
                {
                    "share_specification": specification,
                    "signal": signal,
                    "horizon": horizon,
                    "weeks_relative_to_predictor": 2 * horizon,
                    "n": len(sample),
                    **result,
                }
            )

        difference = (
            sample["outcome_h+1"].to_numpy(float)
            - sample["outcome_h-1"].to_numpy(float)
        )
        contrast = fit(sample, signal, difference, args.hac_lags)
        contrasts.append(
            {
                "share_specification": specification,
                "contrast": "beta_h+1_minus_beta_h-1",
                "n": len(sample),
                **contrast,
            }
        )

    results = pd.DataFrame(rows)
    contrast_results = pd.DataFrame(contrasts)
    results["validation_se"] = 0.0
    results["comparison_se"] = results["hac_se"]
    results["comparison_p_value"] = results["p_value"]
    results["comparison_ci_low"] = results["ci_low"]
    results["comparison_ci_high"] = results["ci_high"]
    contrast_results["validation_se"] = 0.0
    contrast_results["comparison_se"] = contrast_results["hac_se"]
    contrast_results["comparison_p_value"] = contrast_results["p_value"]
    contrast_results["comparison_ci_low"] = contrast_results["ci_low"]
    contrast_results["comparison_ci_high"] = contrast_results["ci_high"]

    bootstrap_horizons, bootstrap_contrast = validation_bootstrap(
        sample,
        validation,
        args.bootstrap_repetitions,
        args.seed,
        args.hac_lags,
    )
    calibrated = results["share_specification"].eq("Directly calibrated share")
    for index, row in results.loc[calibrated].iterrows():
        estimates = bootstrap_horizons.loc[
            bootstrap_horizons["horizon"].eq(row["horizon"]), "estimate"
        ]
        validation_se = float(estimates.std(ddof=1))
        combined_se = math.sqrt(float(row["hac_se"]) ** 2 + validation_se**2)
        results.loc[index, "validation_se"] = validation_se
        results.loc[index, "comparison_se"] = combined_se
        results.loc[index, "comparison_p_value"] = models.normal_p_value(
            float(row["estimate"]) / combined_se
        )
        results.loc[index, "comparison_ci_low"] = (
            float(row["estimate"]) - 1.96 * combined_se
        )
        results.loc[index, "comparison_ci_high"] = (
            float(row["estimate"]) + 1.96 * combined_se
        )
    contrast_index = contrast_results.index[
        contrast_results["share_specification"].eq("Directly calibrated share")
    ][0]
    validation_se = float(bootstrap_contrast["estimate"].std(ddof=1))
    hac_se = float(contrast_results.loc[contrast_index, "hac_se"])
    combined_se = math.sqrt(hac_se**2 + validation_se**2)
    estimate = float(contrast_results.loc[contrast_index, "estimate"])
    contrast_results.loc[contrast_index, "validation_se"] = validation_se
    contrast_results.loc[contrast_index, "comparison_se"] = combined_se
    contrast_results.loc[contrast_index, "comparison_p_value"] = (
        models.normal_p_value(estimate / combined_se)
    )
    contrast_results.loc[contrast_index, "comparison_ci_low"] = estimate - 1.96 * combined_se
    contrast_results.loc[contrast_index, "comparison_ci_high"] = estimate + 1.96 * combined_se

    results.to_csv(output / "lead_lag_coefficients.csv", index=False)
    contrast_results.to_csv(output / "forward_backward_contrast.csv", index=False)
    bootstrap_horizons.to_csv(output / "validation_bootstrap_horizon_draws.csv", index=False)
    bootstrap_contrast.to_csv(output / "validation_bootstrap_contrast_draws.csv", index=False)
    sample[
        ["actor_key", "actor_name", "period_start", "period_index", "n_articles"]
    ].to_csv(output / "balanced_sample.csv", index=False)
    make_plot(results, output)

    metadata = {
        "outcome": "log(1 + actor-linked UCDP fatalities) at horizons -2 to +2",
        "period_length": "two weeks",
        "minimum_articles": args.minimum_articles,
        "hac_lags": args.hac_lags,
        "validation_bootstrap_repetitions": args.bootstrap_repetitions,
        "controls": LEAN_CONTROLS,
        "actor_fixed_effects": True,
        "lagged_fatality_controls": False,
        "balanced_sample_n": int(len(sample)),
        "actor_counts": sample.groupby("actor_name").size().to_dict(),
    }
    (output / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    print(results.to_string(index=False))
    print("\nForward-minus-backward contrast")
    print(contrast_results.to_string(index=False))


if __name__ == "__main__":
    main()
