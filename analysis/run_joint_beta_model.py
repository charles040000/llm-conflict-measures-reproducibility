#!/usr/bin/env python3
"""Estimate the preferred beta latent-share joint model on the final sample."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy.stats import chi2


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import joint_model  # noqa: E402
import model_utils as models  # noqa: E402
import run_calibration_models as calibration  # noqa: E402


DEFAULT_OUTPUT = ROOT / "results" / "calibration" / "joint_beta"
UNIFORM_RESULTS = ROOT / "results" / "calibration" / "uniform_joint"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, default=calibration.DEFAULT_PANEL)
    parser.add_argument("--validation", type=Path, default=calibration.DEFAULT_VALIDATION)
    parser.add_argument("--uniform-results", type=Path, default=UNIFORM_RESULTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--target-accept", type=float, default=0.92)
    parser.add_argument("--quadrature-nodes", type=int, default=128)
    parser.add_argument(
        "--sigma-y-prior",
        choices=["gamma_1_10", "halfnormal_1", "gamma_2_2"],
        default="gamma_1_10",
        help="Prior for the residual standard deviation; the thesis benchmark is Gamma(1,10).",
    )
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--progress", action="store_true")
    return parser.parse_args()


def interval(values: np.ndarray) -> tuple[float, float]:
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def main() -> None:
    args = parse_args()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    panel = pd.read_csv(args.panel.expanduser().resolve(), dtype={"actor_key": str})
    panel["period_start"] = pd.to_datetime(panel["period_start"])
    validation_all = pd.read_csv(
        args.validation.expanduser().resolve(), dtype={"GlobalEventID": str}
    )
    validation, validation_diagnostics = calibration.restrict_validation_to_analysis_actors(
        validation_all
    )
    rates = {
        spec.key: calibration.calibration_rate(validation, spec)
        for spec in models.BINARY_SPECS
    }
    calibrated_panel = calibration.apply_binary_calibration(panel, rates)
    sample = models.base_sample(calibrated_panel, 20)

    uniform_draws = pd.read_csv(
        args.uniform_results.expanduser().resolve() / "joint_global_posterior_draws.csv"
    )
    hmc_args = SimpleNamespace(
        target_accept=args.target_accept,
        warmup=args.warmup,
        samples=args.samples,
        chains=args.chains,
        progress=args.progress,
        seed=args.seed,
    )

    comparison_rows: list[dict[str, object]] = []
    rate_rows: list[dict[str, object]] = []
    convergence_rows: list[dict[str, object]] = []
    parameter_frames: list[pd.DataFrame] = []
    beta_draw_frames: list[pd.DataFrame] = []

    for position, spec in enumerate(models.BINARY_SPECS):
        print(f"Beta latent-share HMC: {spec.key}", flush=True)
        rate = rates[spec.key]
        joint = calibration.prepare_joint_data(sample, spec)
        current_uniform = uniform_draws.loc[
            uniform_draws["variable"].eq(spec.key)
        ].copy()
        uniform_samples = {
            name: current_uniform[name].to_numpy(float)
            for name in ("gamma", "beta0", "beta1", "sigma_y")
        }
        uniform_parameter_summary = pd.read_csv(
            args.uniform_results.expanduser().resolve() / "joint_parameter_summary.csv"
        )
        alpha_rows = uniform_parameter_summary.loc[
            uniform_parameter_summary["variable"].eq(spec.key)
            & uniform_parameter_summary["parameter"].str.startswith("alpha[")
        ].copy()
        alpha_rows["alpha_index"] = alpha_rows["parameter"].str.extract(r"\[(\d+)\]").astype(int)
        alpha_rows = alpha_rows.sort_values("alpha_index")
        alpha_means = alpha_rows["mean"].to_numpy(float)
        uniform_samples["alpha"] = np.repeat(
            alpha_means[None, :], len(current_uniform), axis=0
        )

        uniform_mle = joint_model.fit_integrated_mle(
            joint,
            rate,
            uniform_samples,
            quadrature_nodes=args.quadrature_nodes,
        )
        uniform_parameter_count = int(joint["q"].shape[1] + 4)
        uniform_aic = float(
            2.0 * uniform_parameter_count
            + 2.0 * uniform_mle["negative_log_likelihood"]
        )

        samples_by_chain, fit_info = joint_model.fit_joint_beta_latent_marginalized(
            joint,
            rate,
            hmc_args,
            seed_offset=100 * position,
            quadrature_nodes=args.quadrature_nodes,
            sigma_y_prior=args.sigma_y_prior,
        )
        beta_samples = fit_info["flat"]
        parameter_summary = joint_model.flatten_summary(samples_by_chain)
        parameter_summary.insert(0, "variable", spec.key)
        parameter_frames.append(parameter_summary)

        beta_mle = joint_model.fit_integrated_beta_latent_mle(
            joint,
            rate,
            beta_samples,
            quadrature_nodes=args.quadrature_nodes,
        )

        likelihood_ratio = max(
            0.0,
            2.0
            * (
                float(uniform_mle["negative_log_likelihood"])
                - float(beta_mle["negative_log_likelihood"])
            ),
        )
        gamma = np.asarray(beta_samples["gamma"], dtype=float)
        gamma_low, gamma_high = interval(gamma)
        beta0 = np.asarray(beta_samples["beta0"], dtype=float)
        beta1 = np.asarray(beta_samples["beta1"], dtype=float)
        latent_mean = np.asarray(beta_samples["latent_mean"], dtype=float)
        latent_concentration = np.asarray(
            beta_samples["latent_concentration"], dtype=float
        )

        comparison_rows.extend(
            [
                {
                    "variable": spec.key,
                    "model": "uniform_hmc",
                    "gamma": float(current_uniform["gamma"].mean()),
                    "gamma_se": float(current_uniform["gamma"].std(ddof=1)),
                    "gamma_ci_low": float(current_uniform["gamma"].quantile(0.025)),
                    "gamma_ci_high": float(current_uniform["gamma"].quantile(0.975)),
                    "fpr": float(current_uniform["beta0"].mean()),
                    "tpr": float(current_uniform["beta1"].mean()),
                    "latent_mean": 0.5,
                    "latent_concentration": 2.0,
                    "negative_log_likelihood": float(
                        uniform_mle["negative_log_likelihood"]
                    ),
                    "aic": uniform_aic,
                },
                {
                    "variable": spec.key,
                    "model": "beta_hmc",
                    "gamma": float(gamma.mean()),
                    "gamma_se": float(gamma.std(ddof=1)),
                    "gamma_ci_low": gamma_low,
                    "gamma_ci_high": gamma_high,
                    "fpr": float(beta0.mean()),
                    "tpr": float(beta1.mean()),
                    "latent_mean": float(latent_mean.mean()),
                    "latent_concentration": float(latent_concentration.mean()),
                    "negative_log_likelihood": float(
                        beta_mle["negative_log_likelihood"]
                    ),
                    "aic": float(beta_mle["aic"]),
                },
            ]
        )
        rate_rows.extend(
            [
                {
                    "variable": spec.key,
                    "source": "validation_only",
                    "fpr": float(rate["false_positive_rate"]),
                    "tpr": float(rate["sensitivity"]),
                },
                {
                    "variable": spec.key,
                    "source": "uniform_hmc",
                    "fpr": float(current_uniform["beta0"].mean()),
                    "tpr": float(current_uniform["beta1"].mean()),
                },
                {
                    "variable": spec.key,
                    "source": "beta_hmc",
                    "fpr": float(beta0.mean()),
                    "tpr": float(beta1.mean()),
                },
            ]
        )
        focus = parameter_summary.loc[
            parameter_summary["parameter"].isin(
                [
                    "gamma",
                    "beta0",
                    "beta1",
                    "latent_mean",
                    "latent_concentration",
                    "sigma_y",
                ]
            )
        ]
        convergence_rows.append(
            {
                "variable": spec.key,
                "divergences": int(fit_info["divergences"]),
                "max_r_hat_global": float(focus["r_hat"].max()),
                "min_effective_sample_size_global": float(focus["n_eff"].min()),
                "beta_mle_success": bool(beta_mle["success"]),
                "beta_mle_iterations": int(beta_mle["iterations"]),
                "beta_mle_minimum_hessian_eigenvalue": float(
                    beta_mle["minimum_hessian_eigenvalue"]
                ),
                "likelihood_ratio_vs_uniform": likelihood_ratio,
                "likelihood_ratio_p_value": float(chi2.sf(likelihood_ratio, df=2)),
                "delta_aic_beta_minus_uniform": float(beta_mle["aic"] - uniform_aic),
            }
        )
        beta_draw_frames.append(
            pd.DataFrame(
                {
                    "variable": spec.key,
                    "draw": np.arange(len(gamma)),
                    "gamma": gamma,
                    "beta0": beta0,
                    "beta1": beta1,
                    "latent_mean": latent_mean,
                    "latent_concentration": latent_concentration,
                    "sigma_y": beta_samples["sigma_y"],
                }
            )
        )

    comparison = pd.DataFrame(comparison_rows)
    rates_frame = pd.DataFrame(rate_rows)
    convergence = pd.DataFrame(convergence_rows)
    comparison.to_csv(output / "uniform_beta_comparison.csv", index=False)
    rates_frame.to_csv(output / "classification_rate_comparison.csv", index=False)
    convergence.to_csv(output / "beta_convergence_and_fit.csv", index=False)
    pd.concat(parameter_frames, ignore_index=True).to_csv(
        output / "beta_parameter_summary.csv", index=False
    )
    pd.concat(beta_draw_frames, ignore_index=True).to_csv(
        output / "beta_global_posterior_draws.csv", index=False
    )
    metadata = {
        "sample_n": int(len(sample)),
        "validation": validation_diagnostics,
        "beta_latent_prior": {
            "mean": "Beta(1,1)",
            "concentration": "LogNormal(log(2),1)",
        },
        "sigma_y_prior": args.sigma_y_prior,
        "quadrature_nodes": int(args.quadrature_nodes),
        "hmc": {
            "warmup": int(args.warmup),
            "samples_per_chain": int(args.samples),
            "chains": int(args.chains),
            "target_accept": float(args.target_accept),
        },
    }
    (output / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    print("\nUniform versus Beta latent-share comparison")
    print(comparison.to_string(index=False))
    print("\nClassification rates")
    print(rates_frame.to_string(index=False))
    print("\nConvergence and fit")
    print(convergence.to_string(index=False))


if __name__ == "__main__":
    main()
