"""Regression tests against an independent CYGNET reference."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

import cygnet


def _load_reference_module():
    """Load the independent numerical reference implementation."""
    path = Path(__file__).with_name("reference_lmm.py")
    spec = importlib.util.spec_from_file_location("reference_cygnet", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def small_inputs():
    """Create deterministic full-rank inputs for compatibility checks."""
    rng = np.random.default_rng(8675309)
    n = 18
    y = rng.normal(size=n)
    X = np.column_stack([np.ones(n), rng.normal(size=n)])
    x = rng.normal(size=n)
    E = rng.normal(size=(n, 4))
    S = rng.normal(size=(n, 3))
    inter = x[:, None] * E
    return y, X, x, E, S, inter


def test_null_model_matches_reference(small_inputs):
    """The CYGNET null variance solver should match the standalone implementation."""
    reference = _load_reference_module()
    y, X, x, E, S, _ = small_inputs
    reference_result = reference.reference_lmm_null_multiK(
        y.reshape(-1, 1), np.c_[X, x], [E, S], maxiter=5
    )
    new_result = cygnet.cygnet_null_multiK(y.reshape(-1, 1), np.c_[X, x], [E, S], maxiter=5)
    np.testing.assert_allclose(new_result[0], reference_result[0], rtol=0, atol=0)
    assert new_result[1] == reference_result[1]


def test_davies_score_test_matches_reference(small_inputs):
    """The default Davies score test should produce identical outputs."""
    reference = _load_reference_module()
    y, X, x, E, S, _ = small_inputs
    reference_result = reference.reference_lmm_davies(
        y, X, x, E, [E, S], maxiter=5
    )
    new_result = cygnet.cygnet_davies(y, X, x, E, [E, S], maxiter=5)
    np.testing.assert_allclose(new_result[0], reference_result[0], rtol=0, atol=0)
    assert new_result[1] == reference_result[1]
    assert new_result[2] == reference_result[2]
    assert new_result[3] == reference_result[3]


def test_runner_matches_reference(small_inputs):
    """The high-level CYGNET runner should preserve documented tuple output."""
    reference = _load_reference_module()
    y, X, x, E, S, inter = small_inputs
    reference_result = reference.run_reference_lmm(
        y, X, x, E, [E, S], [inter, E, S], maxiter=5
    )
    new_result = cygnet.run_cygnet(y, X, x, E, [E, S], [inter, E, S], maxiter=5)
    np.testing.assert_allclose(new_result[0], reference_result[0], rtol=0, atol=0)
    assert new_result[1] == reference_result[1]
    assert new_result[2] == reference_result[2]
    np.testing.assert_allclose(new_result[3], reference_result[3], rtol=0, atol=0)
    assert new_result[4] == reference_result[4]


def test_kernel_helpers_match_reference(small_inputs):
    """Kernel construction helpers should match the reference."""
    reference = _load_reference_module()
    _, _, x, E, _, _ = small_inputs
    reference_fourier = reference.get_fourier(
        E, gamma_in=0.5, n_component=8, random_state=7
    )
    new_fourier = cygnet.get_fourier(E, gamma_in=0.5, n_component=8, random_state=7)
    # The BLAS-free Windows-safe projection differs from matrix multiplication
    # only at floating-point rounding scale after the cosine transform.
    np.testing.assert_allclose(new_fourier, reference_fourier, rtol=0, atol=1e-15)

    reference_inter = reference.construct_inter_kernels(E, x)
    new_inter = cygnet.construct_inter_kernels(E, x)
    np.testing.assert_allclose(new_inter, reference_inter, rtol=0, atol=0)


def test_public_functions_have_docstrings():
    """Every public callable exported by cygnet should have documentation."""
    for name in cygnet.__all__:
        obj = getattr(cygnet, name)
        if callable(obj):
            assert obj.__doc__, name
