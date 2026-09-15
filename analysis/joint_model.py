#!/usr/bin/env python3
"""Joint generated-index likelihoods and numerical estimation utilities.

This script reproduces the central-bank communication specification in
Battaglia et al. (2025), equations (15)-(16), for the thesis actor-block data:

    theta_j ~ Uniform(0, 1)
    N_j | theta_j, C_j ~ Binomial(C_j, (1-theta_j) beta0 + theta_j beta1)
    Y_j | theta_j, q_j ~ Normal(gamma theta_j + q_j' alpha, sigma_y)

The validation confusion counts enter the same likelihood as binomial
observations for beta0 (false-positive rate) and beta1 (sensitivity). Following
the paper's empirical index application, estimation uses NumPyro NUTS with
Beta(2,5), Beta(5,2), Normal(0,10), and HalfNormal(1) priors.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd
from numpyro.diagnostics import summary
from numpyro.infer import MCMC, NUTS
from numpyro.infer.initialization import init_to_value
from scipy.optimize import minimize


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_ARTICLES = ROOT_DIR / "data" / "processed" / "article_labels_compact.csv"
DEFAULT_VALIDATION = ROOT_DIR / "data" / "validation" / "manual_validation.csv"
DEFAULT_GED = ROOT_DIR / "data" / "external" / "GEDEvent_v26_1.csv"
DEFAULT_ACTORS = ROOT_DIR / "data" / "reference" / "Actor_v26_1.csv"
DEFAULT_GROUPS = ROOT_DIR / "data" / "reference" / "actor_group_crosswalk.csv"
DEFAULT_OUTPUT_DIR = (
    ROOT_DIR / "results" / "calibration" / "joint_legacy"
)

BLOCKS = {2: 20, 4: 40}
DEFAULT_VARIABLES = ("any_fatalities_mentioned_yes",)


def sigma_y_distribution(prior: str) -> dist.Distribution:
    """Return the configured prior for the positive outcome residual scale."""
    if prior == "gamma_1_10":
        return dist.Gamma(1.0, 10.0)
    if prior == "halfnormal_1":
        return dist.HalfNormal(1.0)
    if prior == "gamma_2_2":
        return dist.Gamma(2.0, 2.0)
    raise ValueError(f"Unknown sigma_y prior: {prior}")


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def joint_index_model(
    c_index: jnp.ndarray,
    n_index: jnp.ndarray,
    q: jnp.ndarray,
    y: jnp.ndarray,
    validation_positive_n: int,
    validation_true_positive: int,
    validation_negative_n: int,
    validation_false_positive: int,
    sigma_y_prior: str = "gamma_1_10",
) -> None:
    """Battaglia et al. equations (15)-(16), including their stated priors."""
    beta0 = numpyro.sample("beta0", dist.Beta(2.0, 5.0))
    beta1 = numpyro.sample("beta1", dist.Beta(5.0, 2.0))
    # The paper uses asymmetric priors to orient the two latent classes. The
    # thesis sample still admits a remote swapped HMC mode, so impose the same
    # substantive identification directly: TPR must exceed FPR.
    numpyro.factor(
        "classification_orientation",
        jnp.where(beta1 > beta0, 0.0, -jnp.inf),
    )
    gamma = numpyro.sample("gamma", dist.Normal(0.0, 10.0))
    alpha = numpyro.sample("alpha", dist.Normal(0.0, 10.0).expand([q.shape[1]]))
    sigma_y = numpyro.sample("sigma_y", sigma_y_distribution(sigma_y_prior))

    with numpyro.plate("cells", y.shape[0]):
        theta = numpyro.sample("theta", dist.Uniform(0.0, 1.0))
        classified_probability = (1.0 - theta) * beta0 + theta * beta1
        numpyro.sample(
            "classified_positive",
            dist.Binomial(total_count=c_index, probs=classified_probability),
            obs=n_index,
        )
        outcome_mean = gamma * theta + jnp.matmul(q, alpha)
        numpyro.sample("outcome", dist.Normal(outcome_mean, sigma_y), obs=y)

    numpyro.sample(
        "validation_true_positive",
        dist.Binomial(total_count=validation_positive_n, probs=beta1),
        obs=jnp.asarray(validation_true_positive, dtype=jnp.float64),
    )
    numpyro.sample(
        "validation_false_positive",
        dist.Binomial(total_count=validation_negative_n, probs=beta0),
        obs=jnp.asarray(validation_false_positive, dtype=jnp.float64),
    )


def joint_index_model_beta_latent(
    c_index: jnp.ndarray,
    n_index: jnp.ndarray,
    q: jnp.ndarray,
    y: jnp.ndarray,
    validation_positive_n: int,
    validation_true_positive: int,
    validation_negative_n: int,
    validation_false_positive: int,
    sigma_y_prior: str = "gamma_1_10",
) -> None:
    """Joint index model with an estimated Beta latent-share distribution."""
    beta0 = numpyro.sample("beta0", dist.Beta(2.0, 5.0))
    beta1 = numpyro.sample("beta1", dist.Beta(5.0, 2.0))
    numpyro.factor(
        "classification_orientation",
        jnp.where(beta1 > beta0, 0.0, -jnp.inf),
    )
    latent_mean = numpyro.sample("latent_mean", dist.Beta(1.0, 1.0))
    latent_concentration = numpyro.sample(
        "latent_concentration", dist.LogNormal(jnp.log(2.0), 1.0)
    )
    latent_alpha = latent_mean * latent_concentration
    latent_beta = (1.0 - latent_mean) * latent_concentration

    gamma = numpyro.sample("gamma", dist.Normal(0.0, 10.0))
    alpha = numpyro.sample("alpha", dist.Normal(0.0, 10.0).expand([q.shape[1]]))
    sigma_y = numpyro.sample("sigma_y", sigma_y_distribution(sigma_y_prior))

    with numpyro.plate("cells", y.shape[0]):
        theta = numpyro.sample("theta", dist.Beta(latent_alpha, latent_beta))
        classified_probability = beta0 + (beta1 - beta0) * theta
        numpyro.sample(
            "classified_positive",
            dist.Binomial(total_count=c_index, probs=classified_probability),
            obs=n_index,
        )
        outcome_mean = gamma * theta + jnp.matmul(q, alpha)
        numpyro.sample("outcome", dist.Normal(outcome_mean, sigma_y), obs=y)

    numpyro.sample(
        "validation_true_positive",
        dist.Binomial(total_count=validation_positive_n, probs=beta1),
        obs=jnp.asarray(validation_true_positive, dtype=jnp.float64),
    )
    numpyro.sample(
        "validation_false_positive",
        dist.Binomial(total_count=validation_negative_n, probs=beta0),
        obs=jnp.asarray(validation_false_positive, dtype=jnp.float64),
    )


def joint_index_model_beta_latent_marginalized(
    c_index: jnp.ndarray,
    n_index: jnp.ndarray,
    q: jnp.ndarray,
    y: jnp.ndarray,
    theta_nodes: jnp.ndarray,
    log_node_weights: jnp.ndarray,
    validation_positive_n: int,
    validation_true_positive: int,
    validation_negative_n: int,
    validation_false_positive: int,
    sigma_y_prior: str = "gamma_1_10",
) -> None:
    """Beta latent-share model with theta integrated by numerical quadrature."""
    beta0 = numpyro.sample("beta0", dist.Beta(2.0, 5.0))
    beta1 = numpyro.sample("beta1", dist.Beta(5.0, 2.0))
    numpyro.factor(
        "classification_orientation",
        jnp.where(beta1 > beta0, 0.0, -jnp.inf),
    )
    latent_mean = numpyro.sample("latent_mean", dist.Beta(1.0, 1.0))
    latent_concentration = numpyro.sample(
        "latent_concentration", dist.LogNormal(jnp.log(2.0), 1.0)
    )
    latent_alpha = latent_mean * latent_concentration
    latent_beta = (1.0 - latent_mean) * latent_concentration
    gamma = numpyro.sample("gamma", dist.Normal(0.0, 10.0))
    alpha = numpyro.sample("alpha", dist.Normal(0.0, 10.0).expand([q.shape[1]]))
    sigma_y = numpyro.sample("sigma_y", sigma_y_distribution(sigma_y_prior))

    probability = beta0 + (beta1 - beta0) * theta_nodes[None, :]
    c = c_index[:, None]
    n = n_index[:, None]
    log_classified = (
        jax.scipy.special.gammaln(c + 1.0)
        - jax.scipy.special.gammaln(n + 1.0)
        - jax.scipy.special.gammaln(c - n + 1.0)
        + n * jnp.log(probability)
        + (c - n) * jnp.log1p(-probability)
    )
    outcome_mean = jnp.matmul(q, alpha)[:, None] + gamma * theta_nodes[None, :]
    log_outcome = dist.Normal(outcome_mean, sigma_y).log_prob(y[:, None])
    log_latent_density = (
        (latent_alpha - 1.0) * jnp.log(theta_nodes)
        + (latent_beta - 1.0) * jnp.log1p(-theta_nodes)
        - jax.scipy.special.betaln(latent_alpha, latent_beta)
    )
    cell_log_likelihood = jax.scipy.special.logsumexp(
        log_node_weights[None, :]
        + log_latent_density[None, :]
        + log_classified
        + log_outcome,
        axis=1,
    )
    numpyro.factor("integrated_panel_likelihood", cell_log_likelihood.sum())

    numpyro.sample(
        "validation_true_positive",
        dist.Binomial(total_count=validation_positive_n, probs=beta1),
        obs=jnp.asarray(validation_true_positive, dtype=jnp.float64),
    )
    numpyro.sample(
        "validation_false_positive",
        dist.Binomial(total_count=validation_negative_n, probs=beta0),
        obs=jnp.asarray(validation_false_positive, dtype=jnp.float64),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--articles", type=Path, default=DEFAULT_ARTICLES)
    parser.add_argument("--validation", type=Path, default=DEFAULT_VALIDATION)
    parser.add_argument("--ged", type=Path, default=DEFAULT_GED)
    parser.add_argument("--actors", type=Path, default=DEFAULT_ACTORS)
    parser.add_argument("--actor-groups", type=Path, default=DEFAULT_GROUPS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--variables",
        nargs="+",
        default=list(DEFAULT_VARIABLES),
        help="Keys from screen_pooled_weekly_llm_regressors.SPECS.",
    )
    parser.add_argument("--block-weeks", nargs="+", type=int, default=[2, 4], choices=[2, 4])
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--target-accept", type=float, default=0.90)
    parser.add_argument("--progress", action="store_true", help="Show NumPyro progress bars.")
    parser.add_argument("--seed", type=int, default=20260826)
    return parser.parse_args()


def verify_integer_counts(frame: pd.DataFrame) -> None:
    for column in ("C_index", "Nhat_index"):
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(float)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{column} contains non-finite values")
        if not np.allclose(values, np.round(values), atol=1e-9):
            bad = values[~np.isclose(values, np.round(values), atol=1e-9)][:5]
            raise ValueError(f"{column} must contain integer article counts; examples: {bad}")
    if (frame["Nhat_index"] > frame["C_index"]).any():
        raise ValueError("Positive classified counts exceed total classified article counts")


def prepare_joint_data(frame: pd.DataFrame, pooled: Any) -> dict[str, Any]:
    y, x, names, data = pooled.design_matrix(frame, "raw_share")
    signal_position = names.index("raw_share")
    q = np.delete(x, signal_position, axis=1)
    q_names = [name for index, name in enumerate(names) if index != signal_position]
    c_index = np.rint(frame.loc[data.index, "C_index"].to_numpy(float)).astype(np.int64)
    n_index = np.rint(frame.loc[data.index, "Nhat_index"].to_numpy(float)).astype(np.int64)
    return {
        "y": y,
        "q": q,
        "q_names": q_names,
        "data": data,
        "c_index": c_index,
        "n_index": n_index,
    }


def flatten_summary(samples: dict[str, np.ndarray]) -> pd.DataFrame:
    stats = summary(samples, prob=0.95)
    rows = []
    for parameter, values in stats.items():
        means = np.asarray(values["mean"])
        indices = [()] if means.ndim == 0 else list(np.ndindex(means.shape))
        for index in indices:
            suffix = "" if index == () else "[" + ",".join(str(item) for item in index) + "]"
            rows.append(
                {
                    "parameter": f"{parameter}{suffix}",
                    "mean": float(np.asarray(values["mean"])[index]),
                    "sd": float(np.asarray(values["std"])[index]),
                    "median": float(np.asarray(values["median"])[index]),
                    "ci_low": float(np.asarray(values["2.5%"])[index]),
                    "ci_high": float(np.asarray(values["97.5%"])[index]),
                    "n_eff": float(np.asarray(values["n_eff"])[index]),
                    "r_hat": float(np.asarray(values["r_hat"])[index]),
                }
            )
    return pd.DataFrame(rows)


def residual_diagnostics(data: pd.DataFrame, residuals: np.ndarray) -> tuple[float, float, float]:
    residual_frame = data[["actor_key", "week_ord"]].copy()
    residual_frame["residual"] = residuals
    actor_acf = []
    for _, group in residual_frame.groupby("actor_key"):
        group = group.sort_values("week_ord")
        pairs = group.merge(
            group[["week_ord", "residual"]].assign(week_ord=lambda value: value["week_ord"] + 1),
            on="week_ord",
            suffixes=("_current", "_previous"),
        )
        if len(pairs) >= 3:
            actor_acf.append(float(pairs["residual_current"].corr(pairs["residual_previous"])))
    y = data["log1p_y_next"].to_numpy(float)
    sse = float(residuals @ residuals)
    sst = float(np.sum((y - y.mean()) ** 2))
    return (
        1.0 - sse / sst if sst > 0 else np.nan,
        math.sqrt(sse / len(y)),
        float(np.nanmean(actor_acf)) if actor_acf else np.nan,
    )


def posterior_tail_probability(draws: np.ndarray) -> float:
    probability_positive = float(np.mean(draws > 0))
    probability_negative = float(np.mean(draws < 0))
    return min(1.0, 2.0 * min(probability_positive, probability_negative))


def logit(value: float) -> float:
    clipped = float(np.clip(value, 1e-8, 1.0 - 1e-8))
    return math.log(clipped / (1.0 - clipped))


def classification_only_theta_mean(
    c_index: np.ndarray,
    n_index: np.ndarray,
    beta0: float,
    beta1: float,
    quadrature_nodes: int = 64,
) -> np.ndarray:
    """Compute E[theta | classified counts] under the Uniform(0,1) index prior."""
    raw_nodes, raw_weights = np.polynomial.legendre.leggauss(quadrature_nodes)
    nodes = (raw_nodes + 1.0) / 2.0
    log_weights = np.log(raw_weights / 2.0)
    probability = beta0 + (beta1 - beta0) * nodes[None, :]
    c = np.asarray(c_index, dtype=float)[:, None]
    n = np.asarray(n_index, dtype=float)[:, None]
    log_classified = (
        np.asarray(jax.scipy.special.gammaln(c + 1.0))
        - np.asarray(jax.scipy.special.gammaln(n + 1.0))
        - np.asarray(jax.scipy.special.gammaln(c - n + 1.0))
        + n * np.log(probability)
        + (c - n) * np.log1p(-probability)
    )
    log_posterior = log_weights[None, :] + log_classified
    normalized = np.exp(
        log_posterior
        - np.asarray(jax.scipy.special.logsumexp(log_posterior, axis=1))[:, None]
    )
    return normalized @ nodes


def beta_classification_only_theta_mean(
    c_index: np.ndarray,
    n_index: np.ndarray,
    beta0: float,
    beta1: float,
    latent_mean: float,
    latent_concentration: float,
    quadrature_nodes: int,
) -> np.ndarray:
    """Compute E[theta | classified counts] under a fitted Beta index prior."""
    raw_nodes, raw_weights = np.polynomial.legendre.leggauss(quadrature_nodes)
    nodes = (raw_nodes + 1.0) / 2.0
    log_weights = np.log(raw_weights / 2.0)
    latent_alpha = latent_mean * latent_concentration
    latent_beta = (1.0 - latent_mean) * latent_concentration
    probability = beta0 + (beta1 - beta0) * nodes[None, :]
    c = np.asarray(c_index, dtype=float)[:, None]
    n = np.asarray(n_index, dtype=float)[:, None]
    log_classified = (
        np.asarray(jax.scipy.special.gammaln(c + 1.0))
        - np.asarray(jax.scipy.special.gammaln(n + 1.0))
        - np.asarray(jax.scipy.special.gammaln(c - n + 1.0))
        + n * np.log(probability)
        + (c - n) * np.log1p(-probability)
    )
    log_latent_density = (
        (latent_alpha - 1.0) * np.log(nodes)
        + (latent_beta - 1.0) * np.log1p(-nodes)
        - float(jax.scipy.special.betaln(latent_alpha, latent_beta))
    )
    log_posterior = (
        log_weights[None, :] + log_latent_density[None, :] + log_classified
    )
    normalized = np.exp(
        log_posterior
        - np.asarray(jax.scipy.special.logsumexp(log_posterior, axis=1))[:, None]
    )
    return normalized @ nodes


def integrated_mle_objective(
    parameters: jnp.ndarray,
    c_index: jnp.ndarray,
    n_index: jnp.ndarray,
    q: jnp.ndarray,
    y: jnp.ndarray,
    theta_nodes: jnp.ndarray,
    log_node_weights: jnp.ndarray,
    validation_positive_n: int,
    validation_true_positive: int,
    validation_negative_n: int,
    validation_false_positive: int,
) -> jnp.ndarray:
    """Negative integrated log likelihood for equations (15)-(16)."""
    q_width = q.shape[1]
    gamma = parameters[0]
    alpha = parameters[1 : 1 + q_width]
    sigma_y = jnp.exp(parameters[1 + q_width])
    beta0 = jax.nn.sigmoid(parameters[2 + q_width])
    beta1_fraction = jax.nn.sigmoid(parameters[3 + q_width])
    beta1 = beta0 + (1.0 - beta0) * beta1_fraction

    probability = beta0 + (beta1 - beta0) * theta_nodes[None, :]
    c = c_index[:, None]
    n = n_index[:, None]
    log_binomial_coefficient = (
        jax.scipy.special.gammaln(c + 1.0)
        - jax.scipy.special.gammaln(n + 1.0)
        - jax.scipy.special.gammaln(c - n + 1.0)
    )
    log_classified = (
        log_binomial_coefficient
        + n * jnp.log(probability)
        + (c - n) * jnp.log1p(-probability)
    )
    outcome_mean = jnp.matmul(q, alpha)[:, None] + gamma * theta_nodes[None, :]
    log_outcome = dist.Normal(outcome_mean, sigma_y).log_prob(y[:, None])
    cell_log_likelihood = jax.scipy.special.logsumexp(
        log_node_weights[None, :] + log_classified + log_outcome,
        axis=1,
    )
    validation_log_likelihood = (
        dist.Binomial(total_count=validation_positive_n, probs=beta1).log_prob(
            jnp.asarray(validation_true_positive, dtype=jnp.float64)
        )
        + dist.Binomial(total_count=validation_negative_n, probs=beta0).log_prob(
            jnp.asarray(validation_false_positive, dtype=jnp.float64)
        )
    )
    return -(cell_log_likelihood.sum() + validation_log_likelihood)


def integrated_beta_latent_mle_objective(
    parameters: jnp.ndarray,
    c_index: jnp.ndarray,
    n_index: jnp.ndarray,
    q: jnp.ndarray,
    y: jnp.ndarray,
    theta_nodes: jnp.ndarray,
    log_node_weights: jnp.ndarray,
    validation_positive_n: int,
    validation_true_positive: int,
    validation_negative_n: int,
    validation_false_positive: int,
    validation_weight: float,
) -> jnp.ndarray:
    """Negative integrated log likelihood with a Beta latent-share density."""
    q_width = q.shape[1]
    gamma = parameters[0]
    alpha = parameters[1 : 1 + q_width]
    sigma_y = jnp.exp(parameters[1 + q_width])
    beta0 = jax.nn.sigmoid(parameters[2 + q_width])
    beta1_fraction = jax.nn.sigmoid(parameters[3 + q_width])
    beta1 = beta0 + (1.0 - beta0) * beta1_fraction
    latent_mean = jax.nn.sigmoid(parameters[4 + q_width])
    latent_concentration = jnp.exp(parameters[5 + q_width])
    latent_alpha = latent_mean * latent_concentration
    latent_beta = (1.0 - latent_mean) * latent_concentration

    probability = beta0 + (beta1 - beta0) * theta_nodes[None, :]
    c = c_index[:, None]
    n = n_index[:, None]
    log_binomial_coefficient = (
        jax.scipy.special.gammaln(c + 1.0)
        - jax.scipy.special.gammaln(n + 1.0)
        - jax.scipy.special.gammaln(c - n + 1.0)
    )
    log_classified = (
        log_binomial_coefficient
        + n * jnp.log(probability)
        + (c - n) * jnp.log1p(-probability)
    )
    outcome_mean = jnp.matmul(q, alpha)[:, None] + gamma * theta_nodes[None, :]
    log_outcome = dist.Normal(outcome_mean, sigma_y).log_prob(y[:, None])
    log_latent_density = (
        (latent_alpha - 1.0) * jnp.log(theta_nodes)
        + (latent_beta - 1.0) * jnp.log1p(-theta_nodes)
        - jax.scipy.special.betaln(latent_alpha, latent_beta)
    )
    cell_log_likelihood = jax.scipy.special.logsumexp(
        log_node_weights[None, :]
        + log_latent_density[None, :]
        + log_classified
        + log_outcome,
        axis=1,
    )
    validation_log_likelihood = (
        dist.Binomial(total_count=validation_positive_n, probs=beta1).log_prob(
            jnp.asarray(validation_true_positive, dtype=jnp.float64)
        )
        + dist.Binomial(total_count=validation_negative_n, probs=beta0).log_prob(
            jnp.asarray(validation_false_positive, dtype=jnp.float64)
        )
    )
    return -(
        cell_log_likelihood.sum()
        + validation_weight * validation_log_likelihood
    )


def fit_integrated_mle(
    joint: dict[str, Any],
    rate: dict[str, float],
    hmc_samples: dict[str, np.ndarray],
    quadrature_nodes: int = 64,
) -> dict[str, Any]:
    """Directly maximize the integrated likelihood with Gauss-Legendre quadrature."""
    raw_nodes, raw_weights = np.polynomial.legendre.leggauss(quadrature_nodes)
    theta_nodes = jnp.asarray((raw_nodes + 1.0) / 2.0, dtype=jnp.float64)
    log_node_weights = jnp.log(jnp.asarray(raw_weights / 2.0, dtype=jnp.float64))
    c_index = jnp.asarray(joint["c_index"], dtype=jnp.float64)
    n_index = jnp.asarray(joint["n_index"], dtype=jnp.float64)
    q = jnp.asarray(joint["q"], dtype=jnp.float64)
    y = jnp.asarray(joint["y"], dtype=jnp.float64)
    q_width = joint["q"].shape[1]

    beta0_start = float(hmc_samples["beta0"].mean())
    beta1_start = float(hmc_samples["beta1"].mean())
    fraction_start = (beta1_start - beta0_start) / max(1.0 - beta0_start, 1e-8)
    initial = np.concatenate(
        [
            [float(hmc_samples["gamma"].mean())],
            hmc_samples["alpha"].mean(axis=0),
            [
                math.log(float(hmc_samples["sigma_y"].mean())),
                logit(beta0_start),
                logit(fraction_start),
            ],
        ]
    ).astype(float)
    objective_args = (
        c_index,
        n_index,
        q,
        y,
        theta_nodes,
        log_node_weights,
        int(rate["validation_manual_positive"]),
        int(rate["validation_true_positive"]),
        int(rate["validation_manual_negative"]),
        int(rate["validation_false_positive"]),
    )
    value_and_gradient = jax.jit(jax.value_and_grad(integrated_mle_objective))

    def scipy_objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        value, gradient = value_and_gradient(jnp.asarray(parameters), *objective_args)
        return float(value), np.asarray(gradient, dtype=float)

    result = minimize(
        scipy_objective,
        initial,
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": 3000, "ftol": 1e-11, "gtol": 1e-7, "maxls": 50},
    )
    estimate = np.asarray(result.x, dtype=float)
    hessian = np.asarray(
        jax.hessian(integrated_mle_objective)(jnp.asarray(estimate), *objective_args),
        dtype=float,
    )
    covariance = np.linalg.pinv(hessian)
    gamma_se = math.sqrt(max(float(covariance[0, 0]), 0.0))
    beta0 = 1.0 / (1.0 + math.exp(-estimate[2 + q_width]))
    fraction = 1.0 / (1.0 + math.exp(-estimate[3 + q_width]))
    beta1 = beta0 + (1.0 - beta0) * fraction
    sigma_y = math.exp(estimate[1 + q_width])

    probability = beta0 + (beta1 - beta0) * np.asarray(theta_nodes)[None, :]
    c = joint["c_index"][:, None]
    n = joint["n_index"][:, None]
    log_classified = (
        np.asarray(jax.scipy.special.gammaln(c + 1.0))
        - np.asarray(jax.scipy.special.gammaln(n + 1.0))
        - np.asarray(jax.scipy.special.gammaln(c - n + 1.0))
        + n * np.log(probability)
        + (c - n) * np.log1p(-probability)
    )
    alpha = estimate[1 : 1 + q_width]
    outcome_mean = joint["q"] @ alpha[:, None] + estimate[0] * np.asarray(theta_nodes)[None, :]
    log_outcome = np.asarray(
        dist.Normal(jnp.asarray(outcome_mean), sigma_y).log_prob(jnp.asarray(joint["y"][:, None]))
    )
    log_weights = np.asarray(log_node_weights)[None, :] + log_classified + log_outcome
    normalized_weights = np.exp(log_weights - np.asarray(jax.scipy.special.logsumexp(log_weights, axis=1))[:, None])
    theta_mean = normalized_weights @ np.asarray(theta_nodes)
    fitted = estimate[0] * theta_mean + joint["q"] @ alpha
    residuals = joint["y"] - fitted
    predictive_theta_mean = classification_only_theta_mean(
        joint["c_index"], joint["n_index"], beta0, beta1, quadrature_nodes
    )
    predictive_fitted = estimate[0] * predictive_theta_mean + joint["q"] @ alpha
    predictive_residuals = joint["y"] - predictive_fitted
    return {
        "success": bool(result.success),
        "message": str(result.message),
        "iterations": int(result.nit),
        "negative_log_likelihood": float(result.fun),
        "gamma": float(estimate[0]),
        "gamma_se": gamma_se,
        "beta0": beta0,
        "beta1": beta1,
        "sigma_y": sigma_y,
        "theta_mean": theta_mean,
        "fitted": fitted,
        "residuals": residuals,
        "predictive_theta_mean": predictive_theta_mean,
        "predictive_fitted": predictive_fitted,
        "predictive_residuals": predictive_residuals,
        "minimum_hessian_eigenvalue": float(np.linalg.eigvalsh(hessian).min()),
    }


def fit_integrated_beta_latent_mle(
    joint: dict[str, Any],
    rate: dict[str, float],
    hmc_samples: dict[str, np.ndarray],
    quadrature_nodes: int = 64,
    validation_weight: float = 1.0,
) -> dict[str, Any]:
    """Maximize the integrated likelihood under a Beta latent-share density."""
    raw_nodes, raw_weights = np.polynomial.legendre.leggauss(quadrature_nodes)
    theta_nodes = jnp.asarray((raw_nodes + 1.0) / 2.0, dtype=jnp.float64)
    log_node_weights = jnp.log(jnp.asarray(raw_weights / 2.0, dtype=jnp.float64))
    c_index = jnp.asarray(joint["c_index"], dtype=jnp.float64)
    n_index = jnp.asarray(joint["n_index"], dtype=jnp.float64)
    q = jnp.asarray(joint["q"], dtype=jnp.float64)
    y = jnp.asarray(joint["y"], dtype=jnp.float64)
    q_width = joint["q"].shape[1]

    beta0_start = float(hmc_samples["beta0"].mean())
    beta1_start = float(hmc_samples["beta1"].mean())
    fraction_start = (beta1_start - beta0_start) / max(1.0 - beta0_start, 1e-8)
    initial = np.concatenate(
        [
            [float(hmc_samples["gamma"].mean())],
            hmc_samples["alpha"].mean(axis=0),
            [
                math.log(float(hmc_samples["sigma_y"].mean())),
                logit(beta0_start),
                logit(fraction_start),
                logit(float(hmc_samples["latent_mean"].mean())),
                math.log(float(hmc_samples["latent_concentration"].mean())),
            ],
        ]
    ).astype(float)
    objective_args = (
        c_index,
        n_index,
        q,
        y,
        theta_nodes,
        log_node_weights,
        int(rate["validation_manual_positive"]),
        int(rate["validation_true_positive"]),
        int(rate["validation_manual_negative"]),
        int(rate["validation_false_positive"]),
        float(validation_weight),
    )
    value_and_gradient = jax.jit(
        jax.value_and_grad(integrated_beta_latent_mle_objective)
    )

    def scipy_objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        value, gradient = value_and_gradient(jnp.asarray(parameters), *objective_args)
        return float(value), np.asarray(gradient, dtype=float)

    result = minimize(
        scipy_objective,
        initial,
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": 3000, "ftol": 1e-11, "gtol": 1e-7, "maxls": 50},
    )
    estimate = np.asarray(result.x, dtype=float)
    hessian = np.asarray(
        jax.hessian(integrated_beta_latent_mle_objective)(
            jnp.asarray(estimate), *objective_args
        ),
        dtype=float,
    )
    covariance = np.linalg.pinv(hessian)
    gamma_se = math.sqrt(max(float(covariance[0, 0]), 0.0))
    beta0 = 1.0 / (1.0 + math.exp(-estimate[2 + q_width]))
    fraction = 1.0 / (1.0 + math.exp(-estimate[3 + q_width]))
    beta1 = beta0 + (1.0 - beta0) * fraction
    latent_mean = 1.0 / (1.0 + math.exp(-estimate[4 + q_width]))
    latent_concentration = math.exp(estimate[5 + q_width])
    sigma_y = math.exp(estimate[1 + q_width])
    parameter_count = int(len(estimate))
    alpha = estimate[1 : 1 + q_width]
    predictive_theta_mean = beta_classification_only_theta_mean(
        joint["c_index"],
        joint["n_index"],
        beta0,
        beta1,
        latent_mean,
        latent_concentration,
        quadrature_nodes,
    )
    predictive_fitted = estimate[0] * predictive_theta_mean + joint["q"] @ alpha
    predictive_residuals = joint["y"] - predictive_fitted

    return {
        "success": bool(result.success),
        "message": str(result.message),
        "iterations": int(result.nit),
        "negative_log_likelihood": float(result.fun),
        "aic": float(2.0 * parameter_count + 2.0 * result.fun),
        "parameter_count": parameter_count,
        "gamma": float(estimate[0]),
        "gamma_se": gamma_se,
        "beta0": beta0,
        "beta1": beta1,
        "latent_mean": latent_mean,
        "latent_concentration": latent_concentration,
        "latent_alpha": latent_mean * latent_concentration,
        "latent_beta": (1.0 - latent_mean) * latent_concentration,
        "sigma_y": sigma_y,
        "alpha": alpha,
        "predictive_theta_mean": predictive_theta_mean,
        "predictive_fitted": predictive_fitted,
        "predictive_residuals": predictive_residuals,
        "predictive_rmse": float(np.sqrt(np.mean(predictive_residuals**2))),
        "iqr_effect": float(
            estimate[0]
            * (
                np.quantile(predictive_theta_mean, 0.75)
                - np.quantile(predictive_theta_mean, 0.25)
            )
        ),
        "validation_weight": float(validation_weight),
        "minimum_hessian_eigenvalue": float(np.linalg.eigvalsh(hessian).min()),
    }


def estimator_comparison_rows(
    pooled: Any,
    dense: Any,
    frame: pd.DataFrame,
    rate: dict[str, float],
    variable: str,
    block_weeks: int,
) -> list[dict[str, object]]:
    specification = str(frame["specification"].iloc[0])
    rows, _ = pooled.fit_all_estimators(dense, frame, specification, rate)
    return [
        {
            "variable": variable,
            "block_weeks": block_weeks,
            **row,
        }
        for row in rows
        if row["se_type"] == "panel_hac4"
    ]


def fit_joint_model(
    joint: dict[str, Any],
    rate: dict[str, float],
    args: argparse.Namespace,
    seed_offset: int,
    sigma_y_prior: str = "gamma_1_10",
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    init_strategy = init_to_value(
        values={
            "beta0": jnp.asarray(
                np.clip(rate["false_positive_rate"], 1e-4, 1.0 - 1e-4),
                dtype=jnp.float64,
            ),
            "beta1": jnp.asarray(
                np.clip(rate["sensitivity"], 1e-4, 1.0 - 1e-4),
                dtype=jnp.float64,
            ),
        }
    )
    kernel = NUTS(
        joint_index_model,
        target_accept_prob=args.target_accept,
        init_strategy=init_strategy,
    )
    mcmc = MCMC(
        kernel,
        num_warmup=args.warmup,
        num_samples=args.samples,
        num_chains=args.chains,
        chain_method="sequential",
        progress_bar=args.progress,
    )
    model_args = {
        "c_index": jnp.asarray(joint["c_index"], dtype=jnp.float64),
        "n_index": jnp.asarray(joint["n_index"], dtype=jnp.float64),
        "q": jnp.asarray(joint["q"], dtype=jnp.float64),
        "y": jnp.asarray(joint["y"], dtype=jnp.float64),
        "validation_positive_n": int(rate["validation_manual_positive"]),
        "validation_true_positive": int(rate["validation_true_positive"]),
        "validation_negative_n": int(rate["validation_manual_negative"]),
        "validation_false_positive": int(rate["validation_false_positive"]),
        "sigma_y_prior": sigma_y_prior,
    }
    mcmc.run(jax.random.PRNGKey(args.seed + seed_offset), **model_args)
    samples_by_chain = mcmc.get_samples(group_by_chain=True)
    samples_flat = mcmc.get_samples(group_by_chain=False)
    extra = mcmc.get_extra_fields(group_by_chain=False)
    diagnostics = {
        "divergences": int(np.asarray(extra.get("diverging", np.array([]))).sum()),
        "total_post_warmup_draws": int(args.chains * args.samples),
    }
    return {key: np.asarray(value) for key, value in samples_by_chain.items()}, {
        "flat": {key: np.asarray(value) for key, value in samples_flat.items()},
        **diagnostics,
    }


def fit_joint_beta_latent_model(
    joint: dict[str, Any],
    rate: dict[str, float],
    args: argparse.Namespace,
    seed_offset: int,
    sigma_y_prior: str = "gamma_1_10",
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Fit the joint model with an estimated Beta latent-share distribution."""
    calibrated = (
        np.asarray(joint["n_index"], dtype=float)
        / np.asarray(joint["c_index"], dtype=float)
        - float(rate["false_positive_rate"])
    ) / max(
        float(rate["sensitivity"]) - float(rate["false_positive_rate"]), 1e-6
    )
    latent_mean_start = float(np.clip(np.mean(np.clip(calibrated, 0.01, 0.99)), 0.05, 0.95))
    init_strategy = init_to_value(
        values={
            "beta0": jnp.asarray(
                np.clip(rate["false_positive_rate"], 1e-4, 1.0 - 1e-4),
                dtype=jnp.float64,
            ),
            "beta1": jnp.asarray(
                np.clip(rate["sensitivity"], 1e-4, 1.0 - 1e-4),
                dtype=jnp.float64,
            ),
            "latent_mean": jnp.asarray(latent_mean_start, dtype=jnp.float64),
            "latent_concentration": jnp.asarray(2.0, dtype=jnp.float64),
        }
    )
    kernel = NUTS(
        joint_index_model_beta_latent,
        target_accept_prob=args.target_accept,
        init_strategy=init_strategy,
    )
    mcmc = MCMC(
        kernel,
        num_warmup=args.warmup,
        num_samples=args.samples,
        num_chains=args.chains,
        chain_method="sequential",
        progress_bar=args.progress,
    )
    model_args = {
        "c_index": jnp.asarray(joint["c_index"], dtype=jnp.float64),
        "n_index": jnp.asarray(joint["n_index"], dtype=jnp.float64),
        "q": jnp.asarray(joint["q"], dtype=jnp.float64),
        "y": jnp.asarray(joint["y"], dtype=jnp.float64),
        "validation_positive_n": int(rate["validation_manual_positive"]),
        "validation_true_positive": int(rate["validation_true_positive"]),
        "validation_negative_n": int(rate["validation_manual_negative"]),
        "validation_false_positive": int(rate["validation_false_positive"]),
        "sigma_y_prior": sigma_y_prior,
    }
    mcmc.run(jax.random.PRNGKey(args.seed + seed_offset), **model_args)
    samples_by_chain = mcmc.get_samples(group_by_chain=True)
    samples_flat = mcmc.get_samples(group_by_chain=False)
    extra = mcmc.get_extra_fields(group_by_chain=False)
    diagnostics = {
        "divergences": int(np.asarray(extra.get("diverging", np.array([]))).sum()),
        "total_post_warmup_draws": int(args.chains * args.samples),
    }
    return {key: np.asarray(value) for key, value in samples_by_chain.items()}, {
        "flat": {key: np.asarray(value) for key, value in samples_flat.items()},
        **diagnostics,
    }


def fit_joint_beta_latent_marginalized(
    joint: dict[str, Any],
    rate: dict[str, float],
    args: argparse.Namespace,
    seed_offset: int,
    quadrature_nodes: int = 128,
    sigma_y_prior: str = "gamma_1_10",
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Fit the Beta model after integrating out actor-period latent shares."""
    calibrated = (
        np.asarray(joint["n_index"], dtype=float)
        / np.asarray(joint["c_index"], dtype=float)
        - float(rate["false_positive_rate"])
    ) / max(
        float(rate["sensitivity"]) - float(rate["false_positive_rate"]), 1e-6
    )
    latent_mean_start = float(
        np.clip(np.mean(np.clip(calibrated, 0.01, 0.99)), 0.05, 0.95)
    )
    init_strategy = init_to_value(
        values={
            "beta0": jnp.asarray(
                np.clip(rate["false_positive_rate"], 1e-4, 1.0 - 1e-4),
                dtype=jnp.float64,
            ),
            "beta1": jnp.asarray(
                np.clip(rate["sensitivity"], 1e-4, 1.0 - 1e-4),
                dtype=jnp.float64,
            ),
            "latent_mean": jnp.asarray(latent_mean_start, dtype=jnp.float64),
            "latent_concentration": jnp.asarray(2.0, dtype=jnp.float64),
        }
    )
    kernel = NUTS(
        joint_index_model_beta_latent_marginalized,
        target_accept_prob=args.target_accept,
        init_strategy=init_strategy,
    )
    mcmc = MCMC(
        kernel,
        num_warmup=args.warmup,
        num_samples=args.samples,
        num_chains=args.chains,
        chain_method="sequential",
        progress_bar=args.progress,
    )
    raw_nodes, raw_weights = np.polynomial.legendre.leggauss(quadrature_nodes)
    model_args = {
        "c_index": jnp.asarray(joint["c_index"], dtype=jnp.float64),
        "n_index": jnp.asarray(joint["n_index"], dtype=jnp.float64),
        "q": jnp.asarray(joint["q"], dtype=jnp.float64),
        "y": jnp.asarray(joint["y"], dtype=jnp.float64),
        "theta_nodes": jnp.asarray((raw_nodes + 1.0) / 2.0, dtype=jnp.float64),
        "log_node_weights": jnp.log(
            jnp.asarray(raw_weights / 2.0, dtype=jnp.float64)
        ),
        "validation_positive_n": int(rate["validation_manual_positive"]),
        "validation_true_positive": int(rate["validation_true_positive"]),
        "validation_negative_n": int(rate["validation_manual_negative"]),
        "validation_false_positive": int(rate["validation_false_positive"]),
        "sigma_y_prior": sigma_y_prior,
    }
    mcmc.run(jax.random.PRNGKey(args.seed + seed_offset), **model_args)
    samples_by_chain = mcmc.get_samples(group_by_chain=True)
    samples_flat = mcmc.get_samples(group_by_chain=False)
    extra = mcmc.get_extra_fields(group_by_chain=False)
    diagnostics = {
        "divergences": int(np.asarray(extra.get("diverging", np.array([]))).sum()),
        "total_post_warmup_draws": int(args.chains * args.samples),
    }
    return {key: np.asarray(value) for key, value in samples_by_chain.items()}, {
        "flat": {key: np.asarray(value) for key, value in samples_flat.items()},
        **diagnostics,
    }


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    pooled = load_module(ROOT_DIR / "scripts" / "run_pooled_weekly_index_models.py", "joint_pooled")
    screen = load_module(ROOT_DIR / "scripts" / "screen_pooled_weekly_llm_regressors.py", "joint_screen")
    blocks = load_module(ROOT_DIR / "scripts" / "run_pooled_block_horizon_screen.py", "joint_blocks")
    builder = load_module(ROOT_DIR / "scripts" / "build_actor_month_rolling_risk_panel.py", "joint_builder")
    dense = load_module(ROOT_DIR / "scripts" / "run_dense_actor_index_models.py", "joint_dense")

    spec_lookup = {spec.key: spec for spec in screen.SPECS}
    unknown = sorted(set(args.variables) - set(spec_lookup))
    if unknown:
        raise SystemExit(f"Unknown variable key(s): {', '.join(unknown)}")
    selected_specs = [spec_lookup[key] for key in args.variables]
    fields = sorted({spec.field for spec in selected_specs})
    builder.LABEL_FIELDS = fields
    builder.ARTICLE_USECOLS = [
        "GlobalEventID", "date", "SQLDATE", "month", "actor_key", "ucdp_actor_id_norm",
        "matched_actor_id_norm", "matched_actor_id", "actor_name", "matched_actor_name",
        "article_actor_weight", *fields,
    ]
    group_map, _, group_notes = builder.load_actor_groups(args.actor_groups)
    article_actor, actor_dim, article_diagnostics = builder.load_article_actor_labels(
        args.articles,
        "2020-01",
        "2025-12",
        actor_group_map=group_map,
        exclude_multi_actor_articles=True,
    )
    source_dates = pd.read_csv(
        args.articles,
        usecols=["GlobalEventID", "date"],
        dtype={"GlobalEventID": str},
        low_memory=False,
    ).drop_duplicates("GlobalEventID")
    article_actor = article_actor.merge(source_dates, on="GlobalEventID", how="left")
    article_actor["actor_key"] = article_actor["actor_key"].astype(str)
    article_actor = article_actor.loc[article_actor["actor_key"].isin(pooled.TARGET_ACTORS)].copy()
    article_actor["week_start"] = pooled.week_start(article_actor["date"])

    # After actor-family collapse and multi-actor exclusion, every retained
    # observation must be one unique article assigned to one analysis actor.
    duplicate_count = int(article_actor.duplicated(["GlobalEventID", "actor_key"]).sum())
    if duplicate_count:
        raise ValueError(f"Found {duplicate_count} duplicate article-actor rows")
    weights = pd.to_numeric(article_actor["article_actor_weight"], errors="coerce")
    if not np.allclose(weights.to_numpy(float), 1.0, atol=1e-9):
        raise ValueError("Retained article-actor rows are not unit weighted")
    article_actor["article_actor_weight"] = 1.0

    actor_dim = actor_dim.loc[actor_dim["actor_key"].astype(str).isin(pooled.TARGET_ACTORS)].copy()
    actor_dim["actor_key"] = actor_dim["actor_key"].astype(str)
    history, fatality_lookup, ged_max_week = pooled.load_weekly_ged(builder, args.ged)
    validation = pd.read_csv(args.validation, dtype={"GlobalEventID": str}, low_memory=False)

    comparison_rows: list[dict[str, object]] = []
    parameter_frames: list[pd.DataFrame] = []
    cell_frames: list[pd.DataFrame] = []
    convergence_rows: list[dict[str, object]] = []
    global_draw_frames: list[pd.DataFrame] = []

    for spec_index, spec in enumerate(selected_specs):
        rate, _ = screen.estimate_rate(validation, spec)
        manual_col = spec.manual_col
        llm_col = spec.llm_col
        pairs = validation[[manual_col, llm_col]].copy()
        pairs.columns = ["manual", "llm"]
        pairs["manual"] = pairs["manual"].map(screen.normalize)
        pairs["llm"] = pairs["llm"].map(screen.normalize)
        valid_classes = set(spec.valid_classes)
        pairs = pairs.loc[
            pairs["manual"].isin(valid_classes) & pairs["llm"].isin(valid_classes)
        ]
        rate["validation_true_positive"] = float(
            (pairs["manual"].eq(spec.positive) & pairs["llm"].eq(spec.positive)).sum()
        )
        rate["validation_false_positive"] = float(
            (~pairs["manual"].eq(spec.positive) & pairs["llm"].eq(spec.positive)).sum()
        )

        weekly = screen.aggregate_weekly(article_actor, pooled, spec)
        weekly_panel = pooled.build_weekly_panel(
            builder, weekly, actor_dim, args.actors, history, fatality_lookup, ged_max_week
        )
        for block_index, block_weeks in enumerate(args.block_weeks):
            minimum_articles = BLOCKS[block_weeks]
            frame = blocks.build_block_panel(
                weekly_panel,
                fatality_lookup,
                ged_max_week,
                pooled,
                block_weeks,
                minimum_articles,
            )
            verify_integer_counts(frame)
            joint = prepare_joint_data(frame, pooled)
            print(
                f"\nJoint HMC: {spec.key}, {block_weeks}-week blocks, "
                f"n={len(joint['y'])}, validation n={int(rate['validation_N_used'])}",
                flush=True,
            )
            samples_by_chain, fit_info = fit_joint_model(
                joint,
                rate,
                args,
                seed_offset=100 * spec_index + 10 * block_index,
            )
            samples = fit_info["flat"]

            parameter_summary = flatten_summary(samples_by_chain)
            parameter_summary.insert(0, "block_weeks", block_weeks)
            parameter_summary.insert(0, "variable", spec.key)
            parameter_frames.append(parameter_summary)

            global_draws = pd.DataFrame(
                {
                    "gamma": samples["gamma"],
                    "beta0": samples["beta0"],
                    "beta1": samples["beta1"],
                    "sigma_y": samples["sigma_y"],
                }
            )
            global_draws.insert(0, "draw", np.arange(len(global_draws)))
            global_draws.insert(0, "block_weeks", block_weeks)
            global_draws.insert(0, "variable", spec.key)
            global_draw_frames.append(global_draws)

            theta_mean = samples["theta"].mean(axis=0)
            gamma_mean = float(samples["gamma"].mean())
            alpha_mean = samples["alpha"].mean(axis=0)
            fitted = gamma_mean * theta_mean + joint["q"] @ alpha_mean
            residuals = joint["y"] - fitted
            reconstruction_r2, reconstruction_rmse, reconstruction_acf1 = residual_diagnostics(
                joint["data"], residuals
            )
            predictive_theta_mean = classification_only_theta_mean(
                joint["c_index"],
                joint["n_index"],
                float(samples["beta0"].mean()),
                float(samples["beta1"].mean()),
            )
            predictive_fitted = gamma_mean * predictive_theta_mean + joint["q"] @ alpha_mean
            predictive_residuals = joint["y"] - predictive_fitted
            r2, rmse, acf1 = residual_diagnostics(joint["data"], predictive_residuals)
            mle = fit_integrated_mle(joint, rate, samples)
            mle_reconstruction_r2, mle_reconstruction_rmse, mle_reconstruction_acf1 = residual_diagnostics(
                joint["data"], mle["residuals"]
            )
            mle_r2, mle_rmse, mle_acf1 = residual_diagnostics(
                joint["data"], mle["predictive_residuals"]
            )

            cells = joint["data"][["actor_key", "week_ord", "week_start", "log1p_y_next"]].copy()
            cells["variable"] = spec.key
            cells["block_weeks"] = block_weeks
            cells["C_index"] = joint["c_index"]
            cells["Nhat_index"] = joint["n_index"]
            cells["raw_share"] = joint["n_index"] / joint["c_index"]
            cells["posterior_theta_mean"] = theta_mean
            cells["classification_only_theta_mean"] = predictive_theta_mean
            cells["joint_reconstruction_fitted"] = fitted
            cells["joint_reconstruction_residual"] = residuals
            cells["joint_predictive_fitted"] = predictive_fitted
            cells["joint_predictive_residual"] = predictive_residuals
            cells["mle_theta_mean"] = mle["theta_mean"]
            cells["mle_classification_only_theta_mean"] = mle["predictive_theta_mean"]
            cells["mle_reconstruction_fitted"] = mle["fitted"]
            cells["mle_reconstruction_residual"] = mle["residuals"]
            cells["mle_predictive_fitted"] = mle["predictive_fitted"]
            cells["mle_predictive_residual"] = mle["predictive_residuals"]
            cell_frames.append(cells)

            gamma_draws = samples["gamma"]
            joint_row = {
                "variable": spec.key,
                "block_weeks": block_weeks,
                "specification": str(frame["specification"].iloc[0]),
                "model": "joint_hmc_battaglia",
                "se_type": "posterior_sd",
                "coefficient": gamma_mean,
                "std_error": float(gamma_draws.std(ddof=1)),
                "z_value": np.nan,
                "p_value_normal": np.nan,
                "posterior_tail_probability": posterior_tail_probability(gamma_draws),
                "ci_low": float(np.quantile(gamma_draws, 0.025)),
                "ci_high": float(np.quantile(gamma_draws, 0.975)),
                "effect_10pp_log_points": gamma_mean * 0.1,
                "effect_10pp_percent_1plus_y": 100.0 * (math.exp(gamma_mean * 0.1) - 1.0),
                "n": len(joint["y"]),
                "n_actors": joint["data"]["actor_key"].nunique(),
                "n_weeks": joint["data"]["week_ord"].nunique(),
                "k": 1 + joint["q"].shape[1],
                "r2": r2,
                "rmse": rmse,
                "mean_actor_acf1": acf1,
                "posterior_reconstruction_r2": reconstruction_r2,
                "posterior_reconstruction_rmse": reconstruction_rmse,
                "posterior_reconstruction_mean_actor_acf1": reconstruction_acf1,
                "beta0_mean": float(samples["beta0"].mean()),
                "beta1_mean": float(samples["beta1"].mean()),
                "sigma_y_mean": float(samples["sigma_y"].mean()),
                "validation_N_used": rate["validation_N_used"],
                "validation_manual_positive": rate["validation_manual_positive"],
                "validation_manual_negative": rate["validation_manual_negative"],
                "divergences": fit_info["divergences"],
                "posterior_probability_beta1_gt_beta0": float(
                    np.mean(samples["beta1"] > samples["beta0"])
                ),
            }
            comparison_rows.extend(
                estimator_comparison_rows(pooled, dense, frame, rate, spec.key, block_weeks)
            )
            comparison_rows.append(joint_row)
            mle_z = mle["gamma"] / mle["gamma_se"] if mle["gamma_se"] > 0 else np.nan
            comparison_rows.append(
                {
                    "variable": spec.key,
                    "block_weeks": block_weeks,
                    "specification": str(frame["specification"].iloc[0]),
                    "model": "joint_integrated_mle",
                    "se_type": "inverse_observed_information",
                    "coefficient": mle["gamma"],
                    "std_error": mle["gamma_se"],
                    "z_value": mle_z,
                    "p_value_normal": pooled.normal_p_value(mle_z),
                    "posterior_tail_probability": np.nan,
                    "ci_low": mle["gamma"] - 1.96 * mle["gamma_se"],
                    "ci_high": mle["gamma"] + 1.96 * mle["gamma_se"],
                    "effect_10pp_log_points": mle["gamma"] * 0.1,
                    "effect_10pp_percent_1plus_y": 100.0 * (math.exp(mle["gamma"] * 0.1) - 1.0),
                    "n": len(joint["y"]),
                    "n_actors": joint["data"]["actor_key"].nunique(),
                    "n_weeks": joint["data"]["week_ord"].nunique(),
                    "k": 1 + joint["q"].shape[1],
                    "r2": mle_r2,
                    "rmse": mle_rmse,
                    "mean_actor_acf1": mle_acf1,
                    "posterior_reconstruction_r2": mle_reconstruction_r2,
                    "posterior_reconstruction_rmse": mle_reconstruction_rmse,
                    "posterior_reconstruction_mean_actor_acf1": mle_reconstruction_acf1,
                    "beta0_mean": mle["beta0"],
                    "beta1_mean": mle["beta1"],
                    "sigma_y_mean": mle["sigma_y"],
                    "validation_N_used": rate["validation_N_used"],
                    "validation_manual_positive": rate["validation_manual_positive"],
                    "validation_manual_negative": rate["validation_manual_negative"],
                    "mle_success": mle["success"],
                    "mle_message": mle["message"],
                    "mle_iterations": mle["iterations"],
                    "mle_negative_log_likelihood": mle["negative_log_likelihood"],
                    "minimum_hessian_eigenvalue": mle["minimum_hessian_eigenvalue"],
                }
            )

            focus = parameter_summary.loc[
                parameter_summary["parameter"].isin(["gamma", "beta0", "beta1", "sigma_y"])
            ]
            convergence_rows.append(
                {
                    "variable": spec.key,
                    "block_weeks": block_weeks,
                    "n": len(joint["y"]),
                    "divergences": fit_info["divergences"],
                    "max_r_hat_global": float(focus["r_hat"].max()),
                    "min_n_eff_global": float(focus["n_eff"].min()),
                    "max_r_hat_all": float(parameter_summary["r_hat"].max()),
                    "min_n_eff_all": float(parameter_summary["n_eff"].min()),
                    "integer_counts_verified": True,
                    "posterior_probability_beta1_gt_beta0": float(
                        np.mean(samples["beta1"] > samples["beta0"])
                    ),
                    "mle_success": mle["success"],
                    "mle_iterations": mle["iterations"],
                    "mle_minimum_hessian_eigenvalue": mle["minimum_hessian_eigenvalue"],
                }
            )

            # Preserve completed fits if a later variable fails to initialize
            # or the long HMC run is interrupted.
            pd.DataFrame(comparison_rows).to_csv(
                output_dir / "joint_vs_two_step_estimators_checkpoint.csv", index=False
            )
            pd.concat(parameter_frames, ignore_index=True).to_csv(
                output_dir / "joint_parameter_summary_checkpoint.csv", index=False
            )
            pd.DataFrame(convergence_rows).to_csv(
                output_dir / "joint_convergence_diagnostics_checkpoint.csv", index=False
            )

    comparison = pd.DataFrame(comparison_rows)
    comparison["fdr_bh_across_selected_tests"] = np.nan
    for model, model_rows in comparison.groupby("model"):
        if model == "joint_hmc_battaglia":
            continue
        comparison.loc[model_rows.index, "fdr_bh_across_selected_tests"] = screen.bh_adjust(
            model_rows["p_value_normal"]
        )
    comparison.to_csv(output_dir / "joint_vs_two_step_estimators.csv", index=False)
    pd.concat(parameter_frames, ignore_index=True).to_csv(
        output_dir / "joint_parameter_summary.csv", index=False
    )
    pd.concat(cell_frames, ignore_index=True).to_csv(
        output_dir / "joint_cell_posterior_means.csv", index=False
    )
    pd.concat(global_draw_frames, ignore_index=True).to_csv(
        output_dir / "joint_global_posterior_draws.csv", index=False
    )
    convergence = pd.DataFrame(convergence_rows)
    convergence.to_csv(output_dir / "joint_convergence_diagnostics.csv", index=False)

    metadata = {
        "method": "Battaglia et al. (2025) central-bank communication joint index model, equations (15)-(16)",
        "article_assignment": "one unique retained article assigned to one collapsed analysis actor",
        "target_actors": list(pooled.TARGET_ACTORS),
        "variables": args.variables,
        "block_weeks": args.block_weeks,
        "minimum_articles": {str(key): value for key, value in BLOCKS.items()},
        "warmup": args.warmup,
        "samples_per_chain": args.samples,
        "chains": args.chains,
        "target_accept": args.target_accept,
        "seed": args.seed,
        "priors": {
            "theta": "Uniform(0,1)",
            "beta0": "Beta(2,5)",
            "beta1": "Beta(5,2)",
            "gamma_and_alpha": "Normal(0,10), equivalent to Normal(0,100) under variance notation",
            "sigma_y": "HalfNormal(1)",
        },
        "identification_restriction": "beta1 > beta0; strengthens the asymmetric-prior orientation used by Battaglia et al.",
        "article_diagnostics": article_diagnostics,
        "actor_group_notes": group_notes,
        "integer_counts_verified": True,
        "software": {
            "jax": jax.__version__,
            "numpyro": numpyro.__version__,
        },
    }
    (output_dir / "joint_run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("\nJoint-estimation comparison:")
    focus_columns = [
        "variable", "block_weeks", "model", "coefficient", "std_error",
        "p_value_normal", "posterior_tail_probability", "ci_low", "ci_high", "r2", "rmse",
    ]
    print(comparison[focus_columns].to_string(index=False))
    print("\nConvergence diagnostics:")
    print(convergence.to_string(index=False))
    print(f"\nSaved outputs to {output_dir}")


if __name__ == "__main__":
    raise SystemExit(
        "joint_model.py is a library. Run run_calibration_models.py and "
        "run_joint_beta_model.py instead."
    )
