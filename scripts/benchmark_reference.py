"""Compare CYGNET numerical output and runtime with the frozen reference code."""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cygnet  # noqa: E402


def load_reference():
    path = ROOT / "tests" / "reference_lmm.py"
    spec = importlib.util.spec_from_file_location("cygnet_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def median_runtime(function, repeats):
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        timings.append(time.perf_counter() - start)
    return statistics.median(timings)


def max_abs_difference(actual, expected):
    differences = []
    for actual_item, expected_item in zip(actual, expected):
        if actual_item is None and expected_item is None:
            continue
        differences.append(
            float(
                np.max(
                    np.abs(
                        np.asarray(actual_item, dtype=float)
                        - np.asarray(expected_item, dtype=float)
                    )
                )
            )
        )
    return max(differences, default=0.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    reference = load_reference()
    rng = np.random.default_rng(20260802)

    n = 80
    y = rng.normal(size=n)
    X = np.column_stack([np.ones(n), rng.normal(size=n)])
    x = rng.normal(size=n)
    E = rng.normal(size=(n, 4))
    S = rng.normal(size=(n, 3))
    interaction = x[:, None] * E
    null_kernels = [E, S]
    full_kernels = [interaction, E, S]

    def run_public():
        return cygnet.run_cygnet(
            y.copy(), X.copy(), x.copy(), E.copy(), null_kernels, full_kernels, maxiter=8
        )

    def run_reference():
        return reference.run_reference_lmm(
            y.copy(), X.copy(), x.copy(), E.copy(), null_kernels, full_kernels, maxiter=8
        )

    public_result = run_public()
    reference_result = run_reference()
    numerical_difference = max_abs_difference(public_result, reference_result)
    reference_seconds = median_runtime(run_reference, args.repeats)
    public_seconds = median_runtime(run_public, args.repeats)

    locations = rng.normal(size=(50_000, 2))

    def fourier_public():
        return cygnet.get_fourier(
            locations, gamma_in=0.8, n_component=100, random_state=42
        )

    def fourier_reference():
        return reference.get_fourier(
            locations, gamma_in=0.8, n_component=100, random_state=42
        )

    public_fourier = fourier_public()
    reference_fourier = fourier_reference()
    fourier_difference = float(np.max(np.abs(public_fourier - reference_fourier)))
    reference_fourier_seconds = median_runtime(fourier_reference, args.repeats)
    public_fourier_seconds = median_runtime(fourier_public, args.repeats)

    report = {
        "score_test": {
            "max_abs_difference": numerical_difference,
            "reference_median_seconds": reference_seconds,
            "cygnet_median_seconds": public_seconds,
            "cygnet_to_reference_ratio": public_seconds / reference_seconds,
        },
        "fourier_features_50000x2_to_100": {
            "max_abs_difference": fourier_difference,
            "reference_median_seconds": reference_fourier_seconds,
            "cygnet_median_seconds": public_fourier_seconds,
            "cygnet_to_reference_ratio": public_fourier_seconds
            / reference_fourier_seconds,
        },
    }
    text = json.dumps(report, indent=2)
    print(text)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n", encoding="utf-8")

    if numerical_difference > 1e-12 or fourier_difference > 1e-12:
        raise SystemExit("Numerical compatibility threshold exceeded")
    score_ratio = report["score_test"]["cygnet_to_reference_ratio"]
    fourier_ratio = report["fourier_features_50000x2_to_100"][
        "cygnet_to_reference_ratio"
    ]
    if score_ratio > 2.5 or fourier_ratio > 2.5:
        raise SystemExit("Runtime compatibility threshold exceeded")


if __name__ == "__main__":
    main()
