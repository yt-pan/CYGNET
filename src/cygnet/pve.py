"""Trace-aware proportion of variance explained utilities."""

from __future__ import annotations

import ast

import numpy as np
import pandas as pd


def coerce_varcom(varcom):
    """Convert CYGNET variance-component output to a numeric vector."""
    if isinstance(varcom, str):
        text = varcom.strip()
        try:
            parsed = ast.literal_eval(text)
            return np.asarray(parsed, dtype=float).reshape(-1)
        except (ValueError, SyntaxError):
            return np.fromstring(text.strip("[]"), sep=" ", dtype=float)
    return np.asarray(varcom, dtype=float).reshape(-1)


def component_trace(component, component_type="low_rank"):
    """Return the trace contribution for one random-effect component matrix."""
    matrix = np.asarray(component, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("Each component matrix must be two-dimensional.")
    if component_type == "low_rank":
        return float(np.sum(matrix * matrix))
    if component_type == "covariance":
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError("Covariance component matrices must be square.")
        return float(np.trace(matrix))
    raise ValueError("component_type must be 'low_rank' or 'covariance'")


def calculate_pve(
    varcom,
    random_effects,
    effect_names=None,
    component_type="low_rank",
    residual_name="Residual",
    residual_trace=None,
):
    """Calculate trace-aware PVE for CYGNET variance components.

    For CYGNET low-rank random-effect design matrices ``Z_k``, each component
    contribution is ``sigma_k * trace(Z_k @ Z_k.T)``, equivalently
    ``sigma_k * sum(Z_k ** 2)``. The residual contribution is
    ``sigma_e * n`` unless ``residual_trace`` is supplied.
    """
    varcom = coerce_varcom(varcom)
    random_effects = list(random_effects)
    if len(varcom) != len(random_effects) + 1:
        raise ValueError("varcom must contain one value per random effect plus the residual variance.")
    if effect_names is None:
        effect_names = [f"Component {i + 1}" for i in range(len(random_effects))]
    if len(effect_names) != len(random_effects):
        raise ValueError("effect_names must match random_effects length.")

    traces = [component_trace(component, component_type=component_type) for component in random_effects]
    if residual_trace is None:
        first = np.asarray(random_effects[0], dtype=float)
        residual_trace = first.shape[0]
    traces.append(float(residual_trace))
    names = list(effect_names) + [residual_name]
    contributions = varcom * np.asarray(traces, dtype=float)
    total = float(np.sum(contributions))
    if not np.isfinite(total) or total <= 0:
        raise ValueError("Total variance contribution must be positive.")
    return pd.DataFrame(
        {
            "Component": names,
            "Variance": varcom,
            "Trace": traces,
            "Contribution": contributions,
            "PVE": contributions / total,
        }
    )
