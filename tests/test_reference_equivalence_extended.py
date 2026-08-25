"""Extended regression coverage against the frozen pre-package implementation."""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.spatial.distance import pdist

import cygnet


REFERENCE_TO_PUBLIC = {
    "reference_lmm_null_multiK": "cygnet_null_multiK",
    "reference_lmm_null_multiK_checkiter": "cygnet_null_multiK_checkiter",
    "reference_lmm": "cygnet",
    "reference_lmm_davies": "cygnet_davies",
    "reference_lmm_davies_iter": "cygnet_davies_iter",
    "run_reference_lmm": "run_cygnet",
    "run_reference_lmm_permu": "run_cygnet_permu",
    "run_reference_lmm_iter": "run_cygnet_iter",
    "reference_lmm_davies_testing_E": "cygnet_davies_testing_E",
    "reference_lmm_wald_testing_x": "cygnet_wald_testing_x",
    "run_reference_lmm_testing_E": "run_cygnet_testing_E",
    "run_reference_lmm_testing_E_permu": "run_cygnet_testing_E_permu",
    "reference_lmm_davies_testing_E_celltype_specific": (
        "cygnet_davies_testing_E_celltype_specific"
    ),
    "run_reference_lmm_testing_E_celltype_specific": (
        "run_cygnet_testing_E_celltype_specific"
    ),
    "run_reference_lmm_testing_E_celltype_specific_permu": (
        "run_cygnet_testing_E_celltype_specific_permu"
    ),
}


def _load_reference():
    path = Path(__file__).with_name("reference_lmm.py")
    spec = importlib.util.spec_from_file_location("cygnet_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def reference():
    return _load_reference()


@pytest.fixture()
def inputs():
    rng = np.random.default_rng(20260802)
    n = 20
    y = rng.normal(size=n)
    X = np.column_stack([np.ones(n), rng.normal(size=n)])
    x = rng.normal(size=n)
    E = rng.normal(size=(n, 4))
    S = rng.normal(size=(n, 3))
    inter = x[:, None] * E
    return y, X, x, E, S, inter


def _assert_equivalent(actual, expected):
    if actual is None or expected is None:
        assert actual is expected
    elif isinstance(actual, BaseException) or isinstance(expected, BaseException):
        assert type(actual) is type(expected)
        assert str(actual) == str(expected)
    elif isinstance(actual, (tuple, list)):
        assert type(actual) is type(expected)
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected):
            _assert_equivalent(actual_item, expected_item)
    elif isinstance(actual, (str, bool)):
        assert actual == expected
    else:
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize(
    ("public_name", "reference_name", "kernel_kind"),
    [
        ("cygnet", "reference_lmm", "null"),
        ("cygnet_davies", "reference_lmm_davies", "null"),
        ("cygnet_davies_iter", "reference_lmm_davies_iter", "null"),
        (
            "cygnet_davies_testing_E",
            "reference_lmm_davies_testing_E",
            "null",
        ),
        (
            "cygnet_davies_testing_E_celltype_specific",
            "reference_lmm_davies_testing_E_celltype_specific",
            "spatial",
        ),
        ("cygnet_wald_testing_x", "reference_lmm_wald_testing_x", "null"),
    ],
)
def test_low_level_numerical_families_match_reference(
    reference, inputs, public_name, reference_name, kernel_kind
):
    y, X, x, E, S, _ = inputs
    kernels = [E, S] if kernel_kind == "null" else [S]
    actual = getattr(cygnet, public_name)(y, X, x, E, kernels, maxiter=5)
    expected = getattr(reference, reference_name)(y, X, x, E, kernels, maxiter=5)
    _assert_equivalent(actual, expected)


@pytest.mark.parametrize(
    ("public_name", "reference_name", "kernel_kind", "full_kind"),
    [
        ("run_cygnet", "run_reference_lmm", "null", "interaction"),
        ("run_cygnet_permu", "run_reference_lmm_permu", "null", "interaction"),
        ("run_cygnet_iter", "run_reference_lmm_iter", "null", "interaction"),
        (
            "run_cygnet_testing_E",
            "run_reference_lmm_testing_E",
            "spatial",
            "environment",
        ),
        (
            "run_cygnet_testing_E_permu",
            "run_reference_lmm_testing_E_permu",
            "spatial",
            "environment",
        ),
        (
            "run_cygnet_testing_E_celltype_specific",
            "run_reference_lmm_testing_E_celltype_specific",
            "spatial",
            "environment",
        ),
        (
            "run_cygnet_testing_E_celltype_specific_permu",
            "run_reference_lmm_testing_E_celltype_specific_permu",
            "spatial",
            "environment",
        ),
    ],
)
def test_high_level_runners_match_reference(
    reference, inputs, public_name, reference_name, kernel_kind, full_kind
):
    y, X, x, E, S, inter = inputs
    null_kernels = [E, S] if kernel_kind == "null" else [S]
    full_kernels = [inter, E, S] if full_kind == "interaction" else [E, S]
    actual = getattr(cygnet, public_name)(
        y.copy(), X.copy(), x.copy(), E.copy(), null_kernels, full_kernels, maxiter=5
    )
    expected = getattr(reference, reference_name)(
        y.copy(), X.copy(), x.copy(), E.copy(), null_kernels, full_kernels, maxiter=5
    )
    _assert_equivalent(actual, expected)


def test_fourier_and_rbf_features_match_reference(reference, inputs):
    _, _, _, E, S, _ = inputs
    actual_fourier = cygnet.get_fourier(E, gamma_in=0.7, n_component=12, random_state=9)
    expected_fourier = reference.get_fourier(
        E, gamma_in=0.7, n_component=12, random_state=9
    )
    np.testing.assert_allclose(actual_fourier, expected_fourier, rtol=0, atol=1e-15)

    actual_rbf, actual_gamma = cygnet.construct_rbf_kernel(
        S,
        gamma=None,
        n=12,
        random_state=9,
        gamma_method="manuscript_range",
    )
    article_gamma = 1.0 / np.median(np.max(S, axis=0) - np.min(S, axis=0))
    expected_rbf = reference.get_fourier(
        S, gamma_in=article_gamma, n_component=12, random_state=9
    )
    assert actual_gamma == article_gamma
    np.testing.assert_allclose(actual_rbf, expected_rbf, rtol=0, atol=1e-15)


def test_default_kernel_construction_uses_standard_median_heuristic(reference, inputs):
    """The public default is inverse median squared pairwise distance."""
    _, _, _, E, S, _ = inputs
    expected_gamma = 1.0 / np.median(pdist(S, metric="sqeuclidean"))
    actual, gamma = cygnet.construct_rbf_kernel(S, n=12)
    expected = reference.get_fourier(
        S, gamma_in=expected_gamma, n_component=12, random_state=1
    )
    assert gamma == expected_gamma
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-15)

    actual_environment = cygnet.get_fourier(E, gamma_in=0.7, n_component=12)
    expected_environment = reference.get_fourier(
        E, gamma_in=0.7, n_component=12, random_state=1
    )
    np.testing.assert_allclose(
        actual_environment, expected_environment, rtol=0, atol=1e-15
    )


def test_explicit_gamma_methods_cover_named_formulas(reference, inputs):
    """Named options retain each documented automatic-gamma formula."""
    _, _, _, _, S, _ = inputs
    manuscript_gamma = 1.0 / np.median(np.max(S, axis=0) - np.min(S, axis=0))
    _, actual_manuscript = cygnet.construct_rbf_kernel(
        S, n=12, gamma_method="manuscript_range"
    )
    assert actual_manuscript == manuscript_gamma

    median_squared_distance = np.median(pdist(S, metric="sqeuclidean"))
    _, actual_half = cygnet.construct_rbf_kernel(
        S, n=12, gamma_method="median_half"
    )
    assert actual_half == 1.0 / (2.0 * median_squared_distance)

    with pytest.raises(ValueError, match="gamma_method"):
        cygnet.construct_rbf_kernel(S, gamma_method="unknown")


def test_default_clr_transform_matches_manuscript_analysis(reference):
    frame = pd.DataFrame(
        [[0.7, 0.2, 0.1], [0.1, 0.0, 0.9], [2.0, 3.0, 5.0]],
        columns=["Ct1", "Ct2", "Ct3"],
    )
    with pytest.warns(UserWarning):
        actual = cygnet.celltype_clr_transform_from_df(frame)
    with pytest.warns(UserWarning):
        expected = reference.celltype_clr_transform_from_df(frame)
    pd.testing.assert_frame_equal(actual, expected)


def test_windows_safe_rank_matches_numpy_rank_decisions():
    from cygnet.core import _matrix_rank

    rng = np.random.default_rng(90210)
    full = rng.normal(size=(250, 12))
    deficient = np.column_stack([full, full[:, 3] - 2.0 * full[:, 7]])
    assert _matrix_rank(full) == np.linalg.matrix_rank(full)
    assert _matrix_rank(deficient) == np.linalg.matrix_rank(deficient)


class _NormalizeReferenceNames(ast.NodeTransformer):
    def visit_FunctionDef(self, node):
        node.name = REFERENCE_TO_PUBLIC.get(node.name, node.name)
        node.decorator_list = []
        if (
            node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)
        ):
            node.body = node.body[1:]
        return self.generic_visit(node)

    def visit_Name(self, node):
        node.id = REFERENCE_TO_PUBLIC.get(node.id, node.id)
        return node


def _normalized_function(node):
    clean = ast.parse(ast.unparse(node)).body[0]
    clean = _NormalizeReferenceNames().visit(clean)
    ast.fix_missing_locations(clean)
    return ast.dump(clean, include_attributes=False)


def test_reference_function_bodies_remain_identical():
    tests_dir = Path(__file__).parent
    package_dir = tests_dir.parent
    reference_tree = ast.parse((tests_dir / "reference_lmm.py").read_text(encoding="utf-8"))
    public_tree = ast.parse(
        (package_dir / "src" / "cygnet" / "core.py").read_text(encoding="utf-8")
    )
    reference_functions = {
        node.name: node for node in reference_tree.body if isinstance(node, ast.FunctionDef)
    }
    public_functions = {
        node.name: node for node in public_tree.body if isinstance(node, ast.FunctionDef)
    }

    approved_differences = {
        "reference_lmm",
        "reference_lmm_davies",
        "reference_lmm_davies_iter",
        "reference_lmm_davies_testing_E",
        "reference_lmm_davies_testing_E_celltype_specific",
        "reference_lmm_wald_testing_x",
        "get_fourier",
        "construct_rbf_kernel",
        "load_simulation_data",
        "celltype_clr_transform_from_df",
    }
    compared = []
    for reference_name, reference_node in reference_functions.items():
        public_name = REFERENCE_TO_PUBLIC.get(reference_name, reference_name)
        if reference_name in approved_differences or public_name not in public_functions:
            continue
        assert _normalized_function(reference_node) == _normalized_function(
            public_functions[public_name]
        ), public_name
        compared.append(public_name)

    assert len(compared) == 28
