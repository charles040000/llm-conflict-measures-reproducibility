#!/usr/bin/env python3
"""Reproduce manual-validation and inter-annotator agreement statistics."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANUAL = ROOT / "data/validation/manual_validation.csv"
DEFAULT_INTER = ROOT / "data/validation/inter_annotator.csv"
DEFAULT_OUTPUT = ROOT / "results/validation"

PRIMARY = [
    ("Direct conflict relevance", "conflict_relevance", "positive", "direct"),
    ("Main-story physical violence", "physical_violence_occurred", "positive", "yes"),
    ("Main-story fatalities", "fatalities_present", "positive", "yes"),
]

FULL = [
    ("Conflict relevance", "conflict_relevance", "exact", None),
    ("Event context", "event_context", "exact", None),
    ("Event modality", "event_modality", "exact", None),
    ("Matched-actor role", "matched_actor_role", "exact", None),
    ("Event-time relation", "event_time_relation", "exact", None),
    *PRIMARY[1:],
    ("Main-story injuries", "injuries_present", "positive", "yes"),
    ("Main-story weapon-use class", "weapon_use_class", "exact", None),
    ("Physical violence mentioned anywhere", "any_physical_violence_mentioned", "positive", "yes"),
    ("Fatalities mentioned anywhere", "any_fatalities_mentioned", "positive", "yes"),
    ("Injuries mentioned anywhere", "any_injuries_mentioned", "positive", "yes"),
    ("Weapon mentioned anywhere", "any_weapon_mentioned", "positive", "yes"),
    ("Main-story confidence", "main_story_confidence", "exact", None),
]


def normalize(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().str.lower()


def paired(
    frame: pd.DataFrame,
    left: str,
    right: str,
    mode: str,
    positive: str | None,
) -> tuple[pd.Series, pd.Series]:
    a, b = normalize(frame[left]), normalize(frame[right])
    keep = a.ne("") & b.ne("")
    a, b = a.loc[keep], b.loc[keep]
    if mode == "positive":
        a, b = a.eq(positive).astype(int), b.eq(positive).astype(int)
    return a.reset_index(drop=True), b.reset_index(drop=True)


def kappa(a: pd.Series, b: pd.Series) -> float:
    observed = float(a.eq(b).mean())
    categories = set(a) | set(b)
    expected = sum(float(a.eq(value).mean() * b.eq(value).mean()) for value in categories)
    return float("nan") if np.isclose(expected, 1) else (observed - expected) / (1 - expected)


def validation_metrics(reference: pd.Series, predicted: pd.Series) -> dict[str, float | int]:
    if set(reference.unique()).issubset({0, 1}) and set(predicted.unique()).issubset({0, 1}):
        tp = int(((reference == 1) & (predicted == 1)).sum())
        tn = int(((reference == 0) & (predicted == 0)).sum())
        fp = int(((reference == 0) & (predicted == 1)).sum())
        fn = int(((reference == 1) & (predicted == 0)).sum())
        precision = tp / (tp + fp) if tp + fp else np.nan
        recall = tp / (tp + fn) if tp + fn else np.nan
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else np.nan
    else:
        tp = tn = fp = fn = 0
        precision = recall = f1 = np.nan
    return {
        "n": len(reference),
        "agreement": float(reference.eq(predicted).mean()),
        "accuracy": float(reference.eq(predicted).mean()),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "kappa": kappa(reference, predicted),
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manual", type=Path, default=DEFAULT_MANUAL)
    parser.add_argument("--inter-annotator", type=Path, default=DEFAULT_INTER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    manual = pd.read_csv(args.manual.expanduser().resolve(), dtype={"GlobalEventID": str})
    flag = manual["in_three_actor_calibration"].astype(str).str.lower()
    manual = manual.loc[flag.isin({"true", "1"})].copy()
    inter = pd.read_csv(args.inter_annotator.expanduser().resolve(), dtype={"GlobalEventID": str})

    validation_rows = []
    for label, field, mode, positive in [PRIMARY[0], *FULL]:
        reference, prediction = paired(
            manual, f"{field}_manual", f"{field}_llm", mode, positive
        )
        validation_rows.append(
            {
                "measure": label,
                "operationalization": "Target versus rest" if mode == "positive" else "Exact categories",
                **validation_metrics(reference, prediction),
            }
        )
    validation = pd.DataFrame(validation_rows)
    validation.to_csv(output / "manual_validation_all_labels.csv", index=False)
    validation.loc[validation["measure"].isin([row[0] for row in PRIMARY])].to_csv(
        output / "manual_validation_primary.csv", index=False
    )

    inter_rows = []
    inter_specs = [PRIMARY[0], *FULL]
    seen = set()
    for label, field, mode, positive in inter_specs:
        key = (label, field, mode, positive)
        if key in seen:
            continue
        seen.add(key)
        coder_1, coder_2 = paired(
            inter, f"{field}_coder_1", f"{field}_coder_2", mode, positive
        )
        inter_rows.append(
            {
                "measure": label,
                "operationalization": "Target versus rest" if mode == "positive" else "Exact categories",
                "n": len(coder_1),
                "agreement": float(coder_1.eq(coder_2).mean()),
                "kappa": kappa(coder_1, coder_2),
                "coder_1_positive": int(coder_1.sum()) if mode == "positive" else np.nan,
                "coder_2_positive": int(coder_2.sum()) if mode == "positive" else np.nan,
            }
        )
    agreement = pd.DataFrame(inter_rows)
    agreement.to_csv(output / "inter_annotator_all_labels.csv", index=False)
    agreement.loc[agreement["measure"].isin([row[0] for row in PRIMARY])].to_csv(
        output / "inter_annotator_primary.csv", index=False
    )
    print(validation.loc[validation["measure"].isin([row[0] for row in PRIMARY])].to_string(index=False))
    print(agreement.loc[agreement["measure"].isin([row[0] for row in PRIMARY])].to_string(index=False))


if __name__ == "__main__":
    main()
