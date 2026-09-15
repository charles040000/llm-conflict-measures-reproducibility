#!/usr/bin/env python3
"""Estimate final two-week calibration and joint latent-share models."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import joint_model  # noqa: E402
import model_utils as models  # noqa: E402


DEFAULT_PANEL = ROOT / "data" / "processed" / "actor_period_panel.csv"
DEFAULT_VALIDATION = ROOT / "data" / "validation" / "manual_validation.csv"
ANALYSIS_ACTOR_NAMES = {
    "209": "Hamas",
    "303": "Taliban",
    "group:farc_bloc": "FARC family",
}
DEFAULT_OUTPUT = ROOT / "results" / "calibration" / "uniform_joint"


def restrict_validation_to_analysis_actors(
    validation: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    if "in_three_actor_calibration" in validation:
        flag = validation["in_three_actor_calibration"].astype(str).str.lower()
        restricted = validation.loc[flag.isin({"true", "1"})].copy()
    elif "analysis_actor_key" in validation:
        restricted = validation.loc[
            validation["analysis_actor_key"].astype(str).isin(models.ACTOR_KEYS)
        ].copy()
    else:
        raise ValueError(
            "Validation data must include in_three_actor_calibration or analysis_actor_key"
        )
    diagnostics = {
        "validation_rows_before_actor_restriction": int(len(validation)),
        "validation_rows_after_actor_restriction": int(len(restricted)),
        "validation_rows_excluded": int(len(validation) - len(restricted)),
        "validation_actor_counts": restricted["analysis_actor_name"].value_counts().to_dict(),
    }
    return restricted, diagnostics


def calibration_rate(validation: pd.DataFrame, spec: models.BinarySpec) -> dict[str, object]:
    manual = models.norm(validation[spec.manual_field])
    llm = models.norm(validation[spec.llm_field])
    comparable = manual.ne("") & llm.ne("")
    rate = models.estimate_binary_calibration(validation.loc[comparable].copy(), spec)
    counts = rate["counts"]
    rate.update(
        {
            "validation_N_used": int(counts.to_numpy().sum()),
            "validation_manual_positive": int(counts.loc[1].sum()),
            "validation_true_positive": int(counts.loc[1, 1]),
            "validation_manual_negative": int(counts.loc[0].sum()),
            "validation_false_positive": int(counts.loc[0, 1]),
        }
    )
    return rate


def apply_binary_calibration(
    panel: pd.DataFrame,
    rates: dict[str, dict[str, object]],
) -> pd.DataFrame:
    result = panel.copy()
    for spec in models.BINARY_SPECS:
        rate = rates[spec.key]
        result[spec.calibrated_column] = (
            (result[spec.raw_column] - float(rate["false_positive_rate"]))
            / float(rate["denominator"])
        ).clip(0, 1)
    return result


def coefficient_row(
    fit: dict[str, object],
    term: str,
    variable: str,
    estimator: str,
) -> dict[str, object]:
    row = fit["coefficients"].set_index("term").loc[term]
    return {
        "variable": variable,
        "estimator": estimator,
        "estimate": float(row["estimate"]),
        "standard_error": float(row["hac_se"]),
        "uncertainty": "actor-specific HAC",
        "p_value": float(row["p_value"]),
        "ci_low": float(row["ci_low"]),
        "ci_high": float(row["ci_high"]),
        "n": int(fit["diagnostics"]["n"]),
        "r2": float(fit["diagnostics"]["r2"]),
        "rmse": float(fit["diagnostics"]["rmse"]),
    }


def corrected_row(
    fit: dict[str, object],
    beta: np.ndarray,
    term: str,
    variable: str,
    estimator: str,
) -> dict[str, object]:
    position = list(fit["names"]).index(term)
    estimate = float(beta[position])
    standard_error = float(np.sqrt(max(fit["covariance"][position, position], 0)))
    return {
        "variable": variable,
        "estimator": estimator,
        "estimate": estimate,
        "standard_error": standard_error,
        "uncertainty": "actor-specific HAC; validation uncertainty added separately",
        "p_value": models.normal_p_value(estimate / standard_error),
        "ci_low": estimate - 1.96 * standard_error,
        "ci_high": estimate + 1.96 * standard_error,
        "n": int(fit["diagnostics"]["n"]),
        "r2": float(fit["diagnostics"]["r2"]),
        "rmse": float(fit["diagnostics"]["rmse"]),
    }


def prepare_joint_data(
    frame: pd.DataFrame,
    spec: models.BinarySpec,
) -> dict[str, object]:
    y, x, names = models.design_matrix(frame, [spec.raw_column])
    signal_position = names.index(spec.raw_column)
    q = np.delete(x, signal_position, axis=1)
    q_names = [name for index, name in enumerate(names) if index != signal_position]
    c_index = np.rint(frame["n_articles"].to_numpy(float)).astype(np.int64)
    n_index = np.rint(frame[f"n_{spec.key}"].to_numpy(float)).astype(np.int64)
    if not np.allclose(c_index, frame["n_articles"].to_numpy(float)):
        raise ValueError("Joint model requires integer article counts")
    if not np.allclose(n_index, frame[f"n_{spec.key}"].to_numpy(float)):
        raise ValueError("Joint model requires integer positive counts")
    if np.any(n_index > c_index):
        raise ValueError("Positive counts exceed total article counts")
    return {
        "y": y,
        "q": q,
        "q_names": q_names,
        "data": frame,
        "c_index": c_index,
        "n_index": n_index,
    }


def validation_bootstrap(
    panel: pd.DataFrame,
    validation: pd.DataFrame,
    repetitions: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    draw_rows: list[dict[str, object]] = []
    rate_draw_rows: list[dict[str, object]] = []
    for draw in range(repetitions):
        sampled = validation.iloc[
            rng.integers(0, len(validation), size=len(validation))
        ].reset_index(drop=True)
        for spec in models.BINARY_SPECS:
            try:
                rate = calibration_rate(sampled, spec)
                rate_draw_rows.append(
                    {
                        "draw": draw,
                        "variable": spec.key,
                        "sensitivity": float(rate["sensitivity"]),
                        "false_positive_rate": float(rate["false_positive_rate"]),
                        "separation": float(rate["denominator"]),
                    }
                )
                calibrated = panel.copy()
                calibrated[spec.calibrated_column] = (
                    (calibrated[spec.raw_column] - float(rate["false_positive_rate"]))
                    / float(rate["denominator"])
                ).clip(0, 1)
                sample = models.base_sample(calibrated, 20)
                fit = models.fit_model(
                    sample,
                    f"single_{spec.key}_calibrated",
                    [spec.calibrated_column],
                    6,
                )
                position = list(fit["names"]).index(spec.calibrated_column)
                point_se = float(np.sqrt(max(fit["covariance"][position, position], 0)))
                calibrated_estimate = float(fit["beta"][position])
                draw_rows.append(
                    {
                        "draw": draw,
                        "variable": spec.key,
                        "estimator": "two_step_calibrated",
                        "estimate": calibrated_estimate,
                        "point_hac_se": point_se,
                    }
                )
                omega = models.binary_index_omega(
                    fit["frame"][spec.calibrated_column].to_numpy(float),
                    float(rate["sensitivity"]),
                    float(rate["false_positive_rate"]),
                )
                correction = models.corrected_estimator(
                    fit,
                    omega,
                    [spec.calibrated_column],
                    fit["frame"]["n_articles"].to_numpy(float),
                )
                for estimator, beta in (
                    ("bca", correction["additive_beta"]),
                    ("bcm", correction["multiplicative_beta"]),
                ):
                    if beta is not None:
                        draw_rows.append(
                            {
                                "draw": draw,
                                "variable": spec.key,
                                "estimator": estimator,
                                "estimate": float(np.asarray(beta)[position]),
                                "point_hac_se": point_se,
                            }
                        )
            except (ValueError, np.linalg.LinAlgError):
                continue
    draws = pd.DataFrame(draw_rows)
    summaries = []
    for (variable, estimator), group in draws.groupby(["variable", "estimator"]):
        estimate = float(group["estimate"].median())
        validation_se = float(group["estimate"].std(ddof=1))
        point_se = float(group["point_hac_se"].median())
        combined_se = math.sqrt(point_se**2 + validation_se**2)
        summaries.append(
            {
                "variable": variable,
                "estimator": estimator,
                "successful_draws": int(group["draw"].nunique()),
                "median_estimate": estimate,
                "validation_se": validation_se,
                "point_hac_se": point_se,
                "combined_se": combined_se,
                "combined_p_value": models.normal_p_value(estimate / combined_se),
                "combined_ci_low": estimate - 1.96 * combined_se,
                "combined_ci_high": estimate + 1.96 * combined_se,
                "validation_percentile_low": float(group["estimate"].quantile(0.025)),
                "validation_percentile_high": float(group["estimate"].quantile(0.975)),
            }
        )
    rate_draws = pd.DataFrame(rate_draw_rows)
    rate_summaries: list[dict[str, object]] = []
    for variable, group in rate_draws.groupby("variable"):
        row: dict[str, object] = {
            "variable": variable,
            "successful_draws": int(group["draw"].nunique()),
        }
        for column in ("sensitivity", "false_positive_rate", "separation"):
            row[f"{column}_bootstrap_low"] = float(group[column].quantile(0.025))
            row[f"{column}_bootstrap_high"] = float(group[column].quantile(0.975))
        rate_summaries.append(row)
    return draws, pd.DataFrame(summaries), rate_draws, pd.DataFrame(rate_summaries)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--validation", type=Path, default=DEFAULT_VALIDATION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--target-accept", type=float, default=0.90)
    parser.add_argument("--bootstrap-repetitions", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--progress", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    panel = pd.read_csv(args.panel.expanduser().resolve(), dtype={"actor_key": str})
    panel["period_start"] = pd.to_datetime(panel["period_start"])
    validation_all = pd.read_csv(
        args.validation.expanduser().resolve(), dtype={"GlobalEventID": str}
    )
    validation, validation_diagnostics = restrict_validation_to_analysis_actors(validation_all)
    validation.to_csv(output / "validation_three_actor_sample.csv", index=False)
    rates = {spec.key: calibration_rate(validation, spec) for spec in models.BINARY_SPECS}
    calibrated_panel = apply_binary_calibration(panel, rates)
    calibrated_panel.to_csv(output / "panel_2_weeks_calibrated.csv", index=False)
    sample = models.base_sample(calibrated_panel, 20)

    point_rows: list[dict[str, object]] = []
    rate_rows: list[dict[str, object]] = []
    clipping_rows: list[dict[str, object]] = []
    joint_parameter_frames: list[pd.DataFrame] = []
    joint_draw_frames: list[pd.DataFrame] = []
    convergence_rows: list[dict[str, object]] = []

    hmc_args = SimpleNamespace(
        target_accept=args.target_accept,
        warmup=args.warmup,
        samples=args.samples,
        chains=args.chains,
        progress=args.progress,
        seed=args.seed,
    )

    for position, spec in enumerate(models.BINARY_SPECS):
        rate = rates[spec.key]
        counts = rate["counts"]
        rate_rows.append(
            {
                "variable": spec.key,
                "validation_n": int(rate["validation_N_used"]),
                "true_negative": int(counts.loc[0, 0]),
                "false_positive": int(counts.loc[0, 1]),
                "false_negative": int(counts.loc[1, 0]),
                "true_positive": int(counts.loc[1, 1]),
                "sensitivity": float(rate["sensitivity"]),
                "specificity": float(rate["specificity"]),
                "false_positive_rate": float(rate["false_positive_rate"]),
                "separation": float(rate["denominator"]),
            }
        )
        unbounded = (
            panel[spec.raw_column] - float(rate["false_positive_rate"])
        ) / float(rate["denominator"])
        baseline_mask = panel.index.isin(sample.index)
        clipping_rows.append(
            {
                "variable": spec.key,
                "n": int(baseline_mask.sum()),
                "clipped_low": int((unbounded[baseline_mask] < 0).sum()),
                "clipped_high": int((unbounded[baseline_mask] > 1).sum()),
                "share_clipped": float(
                    ((unbounded[baseline_mask] < 0) | (unbounded[baseline_mask] > 1)).mean()
                ),
            }
        )

        raw_fit = models.fit_model(
            sample, f"single_{spec.key}_raw", [spec.raw_column], 6
        )
        calibrated_fit = models.fit_model(
            sample,
            f"single_{spec.key}_calibrated",
            [spec.calibrated_column],
            6,
        )
        point_rows.append(coefficient_row(raw_fit, spec.raw_column, spec.key, "naive"))
        point_rows.append(
            coefficient_row(
                calibrated_fit,
                spec.calibrated_column,
                spec.key,
                "two_step_calibrated",
            )
        )
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
        point_rows.append(
            corrected_row(
                calibrated_fit,
                np.asarray(correction["additive_beta"]),
                spec.calibrated_column,
                spec.key,
                "bca",
            )
        )
        if correction["multiplicative_beta"] is not None:
            point_rows.append(
                corrected_row(
                    calibrated_fit,
                    np.asarray(correction["multiplicative_beta"]),
                    spec.calibrated_column,
                    spec.key,
                    "bcm",
                )
            )

        joint = prepare_joint_data(sample, spec)
        samples_by_chain, fit_info = joint_model.fit_joint_model(
            joint,
            rate,
            hmc_args,
            seed_offset=100 * position,
            sigma_y_prior="gamma_1_10",
        )
        samples = fit_info["flat"]
        parameter_summary = joint_model.flatten_summary(samples_by_chain)
        parameter_summary.insert(0, "variable", spec.key)
        joint_parameter_frames.append(parameter_summary)
        draws = pd.DataFrame(
            {
                "variable": spec.key,
                "draw": np.arange(len(samples["gamma"])),
                "gamma": samples["gamma"],
                "beta0": samples["beta0"],
                "beta1": samples["beta1"],
                "sigma_y": samples["sigma_y"],
            }
        )
        joint_draw_frames.append(draws)
        gamma = samples["gamma"]
        point_rows.append(
            {
                "variable": spec.key,
                "estimator": "joint_hmc",
                "estimate": float(gamma.mean()),
                "standard_error": float(gamma.std(ddof=1)),
                "uncertainty": "posterior standard deviation",
                "p_value": joint_model.posterior_tail_probability(gamma),
                "ci_low": float(np.quantile(gamma, 0.025)),
                "ci_high": float(np.quantile(gamma, 0.975)),
                "n": len(sample),
                "r2": np.nan,
                "rmse": np.nan,
            }
        )
        mle = joint_model.fit_integrated_mle(joint, rate, samples)
        point_rows.append(
            {
                "variable": spec.key,
                "estimator": "joint_integrated_mle",
                "estimate": float(mle["gamma"]),
                "standard_error": float(mle["gamma_se"]),
                "uncertainty": "inverse observed information",
                "p_value": models.normal_p_value(float(mle["gamma"] / mle["gamma_se"])),
                "ci_low": float(mle["gamma"] - 1.96 * mle["gamma_se"]),
                "ci_high": float(mle["gamma"] + 1.96 * mle["gamma_se"]),
                "n": len(sample),
                "r2": np.nan,
                "rmse": np.nan,
            }
        )
        focus = parameter_summary.loc[
            parameter_summary["parameter"].isin(["gamma", "beta0", "beta1", "sigma_y"])
        ]
        convergence_rows.append(
            {
                "variable": spec.key,
                "divergences": int(fit_info["divergences"]),
                "max_r_hat_global": float(focus["r_hat"].max()),
                "min_effective_sample_size_global": float(focus["n_eff"].min()),
                "mle_success": bool(mle["success"]),
                "mle_iterations": int(mle["iterations"]),
                "mle_minimum_hessian_eigenvalue": float(mle["minimum_hessian_eigenvalue"]),
                "maximum_battaglia_adjustment_eigenvalue": float(
                    correction["maximum_eigenvalue"]
                ),
                "bcm_valid": bool(correction["bcm_valid"]),
            }
        )

    point = pd.DataFrame(point_rows)
    point["fdr_p_within_estimator"] = point.groupby("estimator")["p_value"].transform(
        models.fdr_bh
    )
    point.to_csv(output / "estimator_comparison_point.csv", index=False)
    pd.DataFrame(rate_rows).to_csv(output / "validation_rates.csv", index=False)
    pd.DataFrame(clipping_rows).to_csv(output / "calibration_clipping.csv", index=False)
    pd.concat(joint_parameter_frames, ignore_index=True).to_csv(
        output / "joint_parameter_summary.csv", index=False
    )
    pd.concat(joint_draw_frames, ignore_index=True).to_csv(
        output / "joint_global_posterior_draws.csv", index=False
    )
    pd.DataFrame(convergence_rows).to_csv(output / "convergence_diagnostics.csv", index=False)

    (
        bootstrap_draws,
        bootstrap_summary,
        rate_bootstrap_draws,
        rate_bootstrap_summary,
    ) = validation_bootstrap(
        panel,
        validation,
        repetitions=args.bootstrap_repetitions,
        seed=args.seed + 1,
    )
    bootstrap_draws.to_csv(output / "validation_bootstrap_draws.csv", index=False)
    bootstrap_summary.to_csv(output / "validation_bootstrap_summary.csv", index=False)
    rate_bootstrap_draws.to_csv(
        output / "validation_rate_bootstrap_draws.csv", index=False
    )
    rate_bootstrap_summary.to_csv(
        output / "validation_rate_bootstrap_summary.csv", index=False
    )

    metadata = {
        "sample": "non-overlapping two-week actor periods with at least 20 articles",
        "n": int(len(sample)),
        "actor_counts": sample.groupby("actor_name").size().to_dict(),
        "history_controls": [
            "fatalities in period t",
            "fatalities in period t-1",
            "fatalities in period t-2",
        ],
        "hac_lags": 6,
        "validation_n": int(len(validation)),
        "validation_restriction": validation_diagnostics,
        "joint_model": {
            "latent_share": "Uniform(0,1)",
            "measurement": "Binomial(C, beta0 + (beta1-beta0)*theta)",
            "outcome": "Gaussian log1p fatalities with baseline controls",
            "beta0_prior": "Beta(2,5)",
            "beta1_prior": "Beta(5,2)",
            "coefficient_prior": "Normal(0,10 SD)",
            "sigma_prior": "Gamma(1,10 rate; retained for the uniform benchmark)",
            "orientation": "beta1 > beta0",
            "warmup": args.warmup,
            "samples_per_chain": args.samples,
            "chains": args.chains,
            "target_accept": args.target_accept,
        },
    }
    (output / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(point.to_string(index=False))
    print("\nValidation bootstrap")
    print(bootstrap_summary.to_string(index=False))
    print("\nConvergence")
    print(pd.DataFrame(convergence_rows).to_string(index=False))


if __name__ == "__main__":
    main()
