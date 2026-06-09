from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.stats import qmc

import cta_geometry as cta


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_CSV = (
    REPO_ROOT
    / "outputs"
    / "CTA_case"
    / "datasets"
    / "campaign_001_exploration"
    / "samples"
    / "cta_dust_vlm_samples.csv"
)


def _unit_samples(method: str, n_samples: int, n_variables: int, seed: int) -> np.ndarray:
    if method == "sobol":
        sampler = qmc.Sobol(d=n_variables, scramble=True, seed=seed)
        return sampler.random(n_samples)
    if method == "halton":
        sampler = qmc.Halton(d=n_variables, scramble=True, seed=seed)
        return sampler.random(n_samples)
    if method == "lhs":
        sampler = qmc.LatinHypercube(d=n_variables, seed=seed)
        return sampler.random(n_samples)
    if method == "random":
        rng = np.random.default_rng(seed)
        return rng.random((n_samples, n_variables))
    msg = f"Unsupported sampling method: {method}"
    raise ValueError(msg)


def generate_samples(method: str, n_samples: int, seed: int) -> list[dict[str, float | int]]:
    variables = cta.CTA_CFD_JUNE_14_DESIGN_VARIABLES
    lower = np.asarray([float(variable.lb) for variable in variables], dtype=float)
    upper = np.asarray([float(variable.ub) for variable in variables], dtype=float)
    unit = _unit_samples(method, n_samples, len(variables), seed)
    values = qmc.scale(unit, lower, upper)

    rows: list[dict[str, float | int]] = []
    for index, sample in enumerate(values):
        row: dict[str, float | int] = {"case_id": index}
        for variable, value in zip(variables, sample):
            row[variable.name] = float(value)
        rows.append(row)
    return rows


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate fixed CTA 14-variable DOE samples for DUST-VLM campaigns.",
    )
    parser.add_argument("--n-samples", type=int, required=True)
    parser.add_argument(
        "--method",
        choices=["sobol", "lhs", "halton", "random"],
        default="sobol",
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    rows = generate_samples(args.method, int(args.n_samples), int(args.seed))
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["case_id", *(variable.name for variable in cta.CTA_CFD_JUNE_14_DESIGN_VARIABLES)]
    with args.output_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("CTA DOE samples generated")
    print(f"  method = {args.method}")
    print(f"  n_samples = {args.n_samples}")
    print(f"  seed = {args.seed}")
    print(f"  output_csv = {args.output_csv.resolve()}")


if __name__ == "__main__":
    main()
