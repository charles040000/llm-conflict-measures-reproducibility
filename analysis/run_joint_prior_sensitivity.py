#!/usr/bin/env python3
"""Check sensitivity of the preferred joint model to the sigma_y prior."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import joint_model  # noqa: E402
import model_utils as models  # noqa: E402
import run_calibration_models as calibration  # noqa: E402


BASELINE_OUTPUT = ROOT / "results" / "calibration" / "joint_beta"
DEFAULT_OUTPUT = ROOT / "results" / "calibration" / "sigma_prior_sensitivity"
PRIOR_LABELS = {
    "gamma_1_10": "Gamma(1,10), mean 0.1",
    "halfnormal_1": "Half-Normal(0,1), mean 0.798",
    "gamma_2_2": "Gamma(2,2), mean 1.0",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, default=calibration.DEFAULT_PANEL)
    parser.add_argument("--validation", type=Path, default=calibration.DEFAULT_VALIDATION)
    parser.add_argument("--baseline-output", type=Path, default=BASELINE_OUTPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--priors",
        nargs="+",
        default=["halfnormal_1", "gamma_2_2"],
        choices=sorted(PRIOR_LABELS),
    )
    parser.add_argument(
        "--variables",
        nargs="+",
        default=[spec.key for spec in models.BINARY_SPECS],
        choices=[spec.key for spec in models.BINARY_SPECS],
    )
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--target-accept", type=float, default=0.92)
    parser.add_argument("--quadrature-nodes", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--progress", action="store_true")
    return parser.parse_args()


def summarize_draws(
    variable: str,
    prior: str,
    draws: pd.DataFrame,
    divergences: int,
    max_r_hat: float,
    min_n_eff: float,
) -> dict[str, object]:
    gamma = draws["gamma"].to_numpy(float)
    sigma_y = draws["sigma_y"].to_numpy(float)
    return {
        "variable": variable,
        "sigma_prior": prior,
        "sigma_prior_description": PRIOR_LABELS[prior],
        "gamma_mean": float(gamma.mean()),
        "gamma_sd": float(gamma.std(ddof=1)),
        "gamma_ci_low": float(np.quantile(gamma, 0.025)),
        "gamma_ci_high": float(np.quantile(gamma, 0.975)),
        "sigma_y_mean": float(sigma_y.mean()),
        "sigma_y_sd": float(sigma_y.std(ddof=1)),
        "sigma_y_ci_low": float(np.quantile(sigma_y, 0.025)),
        "sigma_y_ci_high": float(np.quantile(sigma_y, 0.975)),
        "fpr_mean": float(draws["beta0"].mean()),
        "tpr_mean": float(draws["beta1"].mean()),
        "latent_mean": float(draws["latent_mean"].mean()),
        "latent_concentration": float(draws["latent_concentration"].mean()),
        "divergences": int(divergences),
        "max_r_hat_global": float(max_r_hat),
        "min_effective_sample_size_global": float(min_n_eff),
    }


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
    sample = models.base_sample(calibration.apply_binary_calibration(panel, rates), 20)

    baseline_dir = args.baseline_output.expanduser().resolve()
    baseline_draws = pd.read_csv(baseline_dir / "beta_global_posterior_draws.csv")
    baseline_diagnostics = pd.read_csv(baseline_dir / "beta_convergence_and_fit.csv")

    hmc_args = SimpleNamespace(
        target_accept=args.target_accept,
        warmup=args.warmup,
        samples=args.samples,
        chains=args.chains,
        progress=args.progress,
        seed=args.seed,
    )
    summary_rows: list[dict[str, object]] = []
    draw_frames: list[pd.DataFrame] = []

    selected_specs = [
        spec for spec in models.BINARY_SPECS if spec.key in set(args.variables)
    ]
    for position, spec in enumerate(selected_specs):
        variable = spec.key
        baseline = baseline_draws.loc[baseline_draws["variable"].eq(variable)].copy()
        baseline_diag = baseline_diagnostics.loc[
            baseline_diagnostics["variable"].eq(variable)
        ].iloc[0]
        summary_rows.append(
            summarize_draws(
                variable,
                "gamma_1_10",
                baseline,
                int(baseline_diag["divergences"]),
                float(baseline_diag["max_r_hat_global"]),
                float(baseline_diag["min_effective_sample_size_global"]),
            )
        )
        baseline.insert(1, "sigma_prior", "gamma_1_10")
        draw_frames.append(baseline)

        joint = calibration.prepare_joint_data(sample, spec)
        for prior_position, prior in enumerate(args.priors):
            if prior == "gamma_1_10":
                continue
            print(f"{variable}: {PRIOR_LABELS[prior]}", flush=True)
            samples_by_chain, fit_info = joint_model.fit_joint_beta_latent_marginalized(
                joint,
                rates[variable],
                hmc_args,
                seed_offset=100 * position + 10 * prior_position,
                quadrature_nodes=args.quadrature_nodes,
                sigma_y_prior=prior,
            )
            samples = fit_info["flat"]
            parameter_summary = joint_model.flatten_summary(samples_by_chain)
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
            draws = pd.DataFrame(
                {
                    "variable": variable,
                    "sigma_prior": prior,
                    "draw": np.arange(len(samples["gamma"])),
                    "gamma": samples["gamma"],
                    "beta0": samples["beta0"],
                    "beta1": samples["beta1"],
                    "latent_mean": samples["latent_mean"],
                    "latent_concentration": samples["latent_concentration"],
                    "sigma_y": samples["sigma_y"],
                }
            )
            draw_frames.append(draws)
            summary_rows.append(
                summarize_draws(
                    variable,
                    prior,
                    draws,
                    int(fit_info["divergences"]),
                    float(focus["r_hat"].max()),
                    float(focus["n_eff"].min()),
                )
            )

    summary = pd.DataFrame(summary_rows)
    baseline_gamma = summary.loc[
        summary["sigma_prior"].eq("gamma_1_10"), ["variable", "gamma_mean"]
    ].rename(columns={"gamma_mean": "baseline_gamma_mean"})
    summary = summary.merge(baseline_gamma, on="variable", how="left")
    summary["gamma_change_from_baseline"] = (
        summary["gamma_mean"] - summary["baseline_gamma_mean"]
    )
    summary.to_csv(output / "sigma_prior_sensitivity_summary.csv", index=False)
    pd.concat(draw_frames, ignore_index=True).to_csv(
        output / "sigma_prior_sensitivity_draws.csv", index=False
    )
    metadata = {
        "sample_n": int(len(sample)),
        "variables": args.variables,
        "validation": validation_diagnostics,
        "preferred_sigma_y_prior": "gamma_1_10",
        "priors": PRIOR_LABELS,
        "quadrature_nodes": int(args.quadrature_nodes),
        "hmc": {
            "warmup": int(args.warmup),
            "samples_per_chain": int(args.samples),
            "chains": int(args.chains),
            "target_accept": float(args.target_accept),
        },
    }
    (output / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print("\nSigma_y prior sensitivity")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
