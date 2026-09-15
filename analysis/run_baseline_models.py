#!/usr/bin/env python3
"""Reproduce the final naive two-week baseline regressions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import model_utils as models


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PANEL = ROOT / "data/processed/actor_period_panel.csv"
DEFAULT_OUTPUT = ROOT / "results/baseline"


def residual_rows(model_id: str, fit: dict[str, object]) -> pd.DataFrame:
    frame = fit["frame"]
    return pd.DataFrame(
        {
            "model_id": model_id,
            "actor_key": frame["actor_key"].astype(str).to_numpy(),
            "actor_name": frame["actor_name"].to_numpy(),
            "period_start": frame["period_start"].to_numpy(),
            "observed": fit["y"],
            "fitted": fit["fitted"],
            "residual": fit["residuals"],
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--minimum-articles", type=int, default=20)
    parser.add_argument("--hac-lags", type=int, default=6)
    args = parser.parse_args()

    panel = pd.read_csv(args.panel.expanduser().resolve(), dtype={"actor_key": str})
    panel["period_start"] = pd.to_datetime(panel["period_start"])
    sample = models.base_sample(panel, args.minimum_articles)
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    raw = [spec.raw_column for spec in models.BINARY_SPECS]
    specifications = {
        "controls_only": [],
        **{f"single_{spec.key}_raw": [spec.raw_column] for spec in models.BINARY_SPECS},
        "joint_binary_raw": raw,
        "context_three_raw": [
            "share_context_three_violent_raw",
            "share_context_three_pol_raw",
        ],
    }

    fits = {
        model_id: models.fit_model(sample, model_id, signals, args.hac_lags)
        for model_id, signals in specifications.items()
    }
    coefficients = pd.concat(
        [fit["coefficients"] for fit in fits.values()], ignore_index=True
    )
    diagnostics = pd.DataFrame([fit["diagnostics"] for fit in fits.values()])
    residuals = pd.concat(
        [residual_rows(model_id, fit) for model_id, fit in fits.items()],
        ignore_index=True,
    )

    separate_ids = {f"single_{spec.key}_raw" for spec in models.BINARY_SPECS}
    primary = coefficients.loc[
        coefficients["model_id"].isin(separate_ids) & coefficients["is_signal"]
    ].copy()
    primary["fdr_p"] = models.fdr_bh(primary["p_value"])

    binary_ids = {*separate_ids, "joint_binary_raw"}
    coefficients.loc[coefficients["model_id"].eq("controls_only")].to_csv(
        output / "controls_only_coefficients.csv", index=False
    )
    diagnostics.loc[diagnostics["model_id"].eq("controls_only")].to_csv(
        output / "controls_only_diagnostics.csv", index=False
    )
    coefficients.loc[coefficients["model_id"].isin(binary_ids)].to_csv(
        output / "naive_binary_coefficients.csv", index=False
    )
    diagnostics.loc[diagnostics["model_id"].isin(binary_ids)].to_csv(
        output / "naive_binary_diagnostics.csv", index=False
    )
    primary.to_csv(output / "naive_binary_separate_signals.csv", index=False)
    coefficients.loc[coefficients["model_id"].eq("context_three_raw")].to_csv(
        output / "naive_context_coefficients.csv", index=False
    )
    diagnostics.loc[diagnostics["model_id"].eq("context_three_raw")].to_csv(
        output / "naive_context_diagnostics.csv", index=False
    )
    models.signal_vifs(sample, raw).assign(model_id="joint_binary_raw").to_csv(
        output / "naive_binary_vifs.csv", index=False
    )
    models.signal_vifs(
        sample,
        ["share_context_three_violent_raw", "share_context_three_pol_raw"],
    ).to_csv(output / "naive_context_vifs.csv", index=False)
    residuals.to_csv(output / "model_residuals.csv", index=False)

    metadata = {
        "panel": str(args.panel.resolve().relative_to(ROOT)),
        "observations": int(len(sample)),
        "actor_observations": sample.groupby("actor_name").size().to_dict(),
        "minimum_articles": args.minimum_articles,
        "hac_lags": args.hac_lags,
        "outcome": models.OUTCOME,
        "controls": models.BASE_CONTROLS,
    }
    (output / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(primary[["term", "estimate", "hac_se", "p_value", "fdr_p"]].to_string(index=False))


if __name__ == "__main__":
    main()
