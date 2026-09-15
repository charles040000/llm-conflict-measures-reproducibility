#!/usr/bin/env python3
"""Run the reproducibility workflow from the repository root."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def run(script: str, *arguments: str) -> None:
    command = [sys.executable, str(ROOT / script), *arguments]
    print("\n$ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def quick() -> None:
    run("analysis/run_validation_analysis.py")
    run("analysis/run_baseline_models.py")
    run("analysis/run_taliban_descriptive.py")
    run("analysis/generate_figures.py")


def full() -> None:
    run("analysis/run_validation_analysis.py")
    run("analysis/run_baseline_models.py")
    run("analysis/run_calibration_models.py")
    run("analysis/run_joint_beta_model.py")
    run("analysis/build_estimator_comparison.py")
    run("analysis/run_timing_diagnostic.py")
    run("analysis/run_robustness_models.py")
    run("analysis/run_taliban_descriptive.py")
    run("analysis/generate_figures.py")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "workflow",
        choices=["quick", "full"],
        nargs="?",
        default="quick",
        help="quick rebuilds deterministic results and figures; full also reruns HMC and robustness models",
    )
    args = parser.parse_args()
    quick() if args.workflow == "quick" else full()


if __name__ == "__main__":
    main()
